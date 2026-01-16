//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// forwardDIT384MixedComplex64 computes a 384-point forward FFT using the
// 128×3 decomposition (radix-3 first, then 128-point FFTs).
func forwardDIT384MixedComplex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 384
	const stride = 128 // Distance between elements in a column

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	// We need scratch space for:
	// 1. work buffer (384 elements) - passed as scratch
	// 2. sub-transform twiddles (128 elements)
	// 3. sub-transform scratch (128 elements)
	// The provided scratch is usually size N (384).
	// We might need more if we want to store twiddles there.
	// However, we can reuse dst as the work buffer for Step 1 & 2 if we are careful,
	// or assume the caller provides enough scratch.
	// For now, we will assume scratch is at least N.
	// To avoid allocating twiddles, we can try to use a part of scratch if it's large enough,
	// or we accept a small allocation for twiddles/bitrev which is much smaller than full N.
	// Ideally, the Plan should provide ample scratch.
	// Let's use the provided scratch for the main work.

	work := scratch // Size 384

	// Step 1: Compute 128 radix-3 column DFTs
	// Input viewed as 128×3 matrix: x[n1, n2] = src[n1 + n2*128]
	for n1 := range stride {
		a0 := src[n1]
		a1 := src[n1+stride]
		a2 := src[n1+2*stride]
		y0, y1, y2 := butterfly3ForwardComplex64(a0, a1, a2)
		work[n1] = y0
		work[n1+stride] = y1
		work[n1+2*stride] = y2
	}

	// Step 2: Apply twiddle factors W_384^(n1*k2)
	for n1 := range stride {
		work[stride+n1] *= twiddle[n1]
	}
	for n1 := range stride {
		work[2*stride+n1] *= twiddle[2*n1]
	}

	// Step 3: Compute 3 independent 128-point FFTs
	// We need 128-point twiddles. These are a subset of W_384.
	// W_128^k = W_384^(3k).
	// We can gather them into a temp buffer.

	// Fast allocation (small stack-allocated-like arrays would be better, but slices work)
	// We need 128 complex64s for twiddles.
	// We reuse 'dst' as temporary storage for twiddles and bitrev since we overwrite dst in Step 4.
	// BUT Step 3 writes to 'dst' (or we need an intermediate).
	// Let's use a small local buffer for twiddles to avoid 'mathpkg' recompute overhead.
	// Note: We really should have these precomputed, but gathering from 'twiddle' is fast O(N).

	subTwiddle := make([]complex64, stride)
	for k := range stride {
		subTwiddle[k] = twiddle[k*3]
	}

	// We need scratch for the sub-transform.
	subScratch := make([]complex64, stride)

	// Step 3: Compute 3 independent 128-point FFTs
	// We process rows from 'work' and write to 'dst' temporarily (interleaved later?)
	// The original code used 'fftOut' temp buffer.
	// We can use 'dst' to store the FFT results, but the order in 'dst' will be [row0, row1, row2].
	// Step 4 expects to read from that and write to 'dst' interleaved.
	// This implies we cannot write directly to final 'dst' positions in Step 3 easily
	// because Step 4 reads randomly.
	// So we keep results in 'work' (in-place FFT?) or write to 'dst' then copy back?
	// The AVX2 kernel is out-of-place (dst, src).
	// Let's write result back to 'work' region? No, AVX2 inputs/outputs shouldn't overlap if not safe.
	// We can write to 'dst[0:128]', 'dst[128:256]', 'dst[256:384]'.

	for k2 := range 3 {
		rowStart := k2 * stride
		// Input from work, Output to dst (temporarily stored as rows)
		if !amd64.ForwardAVX2Size128Mixed24Complex64Asm(
			dst[rowStart:rowStart+stride],
			work[rowStart:rowStart+stride],
			subTwiddle, subScratch,
		) {
			return false
		}
	}

	// Step 4: Interleave output
	// Current 'dst' contains [FFT(row0), FFT(row1), FFT(row2)]
	// We need to permute this into final 'dst' order.
	// X[k1*3 + k2] = FFT_result[k2][k1]
	// We can do this in-place by copying 'dst' back to 'work' first, or swapping?
	// Copying 'dst' to 'work' is safe.
	copy(work, dst)

	for k1 := range stride {
		for k2 := range 3 {
			dst[k1*3+k2] = work[k2*stride+k1]
		}
	}

	return true
}

// inverseDIT384MixedComplex64 computes a 384-point inverse FFT.
func inverseDIT384MixedComplex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 384
	const stride = 128

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := scratch // Size 384

	// Step 1: De-interleave input
	// src[k1*3 + k2] → work[k2*128 + k1] (using work as temp buffer)
	for k1 := range stride {
		for k2 := range 3 {
			work[k2*stride+k1] = src[k1*3+k2]
		}
	}

	// Prepare for 128-point sub-IFFTs
	// Gather twiddles (stride 3)
	subTwiddle := make([]complex64, stride)
	for k := range stride {
		subTwiddle[k] = twiddle[k*3]
	}

	subScratch := make([]complex64, stride)

	// Step 2: Compute 3 independent 128-point IFFTs
	// Input from work, Output to dst (temporarily as rows)
	for k2 := range 3 {
		rowStart := k2 * stride
		if !amd64.InverseAVX2Size128Mixed24Complex64Asm(
			dst[rowStart:rowStart+stride],
			work[rowStart:rowStart+stride],
			subTwiddle, subScratch,
		) {
			return false
		}
	}

	// Step 3 & 4: Apply conjugate twiddles and Compute 128 radix-3 inverse column butterflies
	// We read from 'dst' (rows) and write to 'dst' (final)
	// But Step 4 output indices: dst[n1], dst[n1+stride], dst[n1+2*stride]
	// Step 3 inputs are at same locations.
	// So we can do this in-place in 'dst' IF we are careful.
	// Actually, let's copy 'dst' back to 'work' to be safe and clean.
	copy(work, dst)

	// Apply conjugate twiddle factors to 'work'
	for n1 := range stride {
		work[stride+n1] *= mathpkg.Conj(twiddle[n1])
	}
	for n1 := range stride {
		work[2*stride+n1] *= mathpkg.Conj(twiddle[2*n1])
	}

	scale := complex64(complex(1.0/3.0, 0)) // Additional scaling (128-pt IFFT did 1/128)
	for n1 := range stride {
		a0 := work[n1]
		a1 := work[n1+stride]
		a2 := work[n1+2*stride]
		y0, y1, y2 := butterfly3InverseComplex64(a0, a1, a2)
		dst[n1] = y0 * scale
		dst[n1+stride] = y1 * scale
		dst[n1+2*stride] = y2 * scale
	}

	return true
}

// forwardDIT384MixedComplex128 computes a 384-point forward FFT (complex128).
func forwardDIT384MixedComplex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 384
	const stride = 128

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	// Step 1: Compute 128 radix-3 column DFTs
	copy(scratch, src)
	amd64.Radix3Butterflies384ForwardComplex128Asm(scratch)

	// Step 2: Apply twiddle factors
	amd64.ApplyTwiddle384Complex128Asm(scratch, twiddle)

	// Prepare for 128-point sub-FFTs
	twiddle128 := mathpkg.ComputeTwiddleFactors[complex128](stride)
	subScratch := make([]complex128, stride)
	fftOut := make([]complex128, n)

	// Step 3: Compute 3 independent 128-point FFTs
	for k2 := range 3 {
		rowStart := k2 * stride
		if !amd64.ForwardAVX2Size128Radix2Complex128Asm(
			fftOut[rowStart:rowStart+stride],
			scratch[rowStart:rowStart+stride],
			twiddle128, subScratch,
		) {
			return false
		}
	}

	// Step 4: Interleave output
	for k1 := range stride {
		for k2 := range 3 {
			dst[k1*3+k2] = fftOut[k2*stride+k1]
		}
	}

	return true
}

// inverseDIT384MixedComplex128 computes a 384-point inverse FFT (complex128).
func inverseDIT384MixedComplex128(dst, src, twiddle, scratch []complex128) bool {
	const n = 384
	const stride = 128

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := scratch
	ifftIn := make([]complex128, n)

	// Step 1: De-interleave input
	for k1 := range stride {
		for k2 := range 3 {
			ifftIn[k2*stride+k1] = src[k1*3+k2]
		}
	}

	// Prepare for 128-point sub-IFFTs
	twiddle128 := mathpkg.ComputeTwiddleFactors[complex128](stride)
	subScratch := make([]complex128, stride)

	// Step 2: Compute 3 independent 128-point IFFTs
	for k2 := range 3 {
		rowStart := k2 * stride
		if !amd64.InverseAVX2Size128Radix2Complex128Asm(
			work[rowStart:rowStart+stride],
			ifftIn[rowStart:rowStart+stride],
			twiddle128, subScratch,
		) {
			return false
		}
	}

	// Step 3: Apply conjugate twiddle factors
	for n1 := range stride {
		work[stride+n1] *= mathpkg.Conj(twiddle[n1])
	}
	for n1 := range stride {
		work[2*stride+n1] *= mathpkg.Conj(twiddle[2*n1])
	}

	// Step 4: Compute 128 radix-3 inverse column butterflies
	amd64.Radix3Butterflies384InverseComplex128Asm(work)

	// Scale and copy to dst
	scale := complex128(complex(1.0/3.0, 0))
	for i := range n {
		dst[i] = work[i] * scale
	}

	return true
}
