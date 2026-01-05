//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// forwardDIT384MixedComplex64 computes a 384-point forward FFT using the
// 128×3 decomposition (radix-3 first, then 128-point FFTs).
//
// Algorithm derivation for N = 384 = 128 × 3:
//
//	Index mapping: n = n1 + n2*128, k = k1*3 + k2
//	where n1,k1 ∈ [0,127] and n2,k2 ∈ [0,2]
//
//	X[k1*3 + k2] = Σ_{n1} (Y[n1,k2] * W_384^(n1*k2)) * W_128^(n1*k1)
//	where Y[n1,k2] = Σ_{n2} x[n1 + n2*128] * W_3^(n2*k2)
//
// Steps:
//  1. Compute 128 radix-3 DFTs on columns of 128×3 view (stride-128 access)
//  2. Apply twiddle factors: Y[n1,k2] *= W_384^(n1*k2)
//  3. Compute 3 independent 128-point FFTs
//  4. Interleave output: dst[k1*3+k2] = FFT_result[k2][k1]
func forwardDIT384MixedComplex64(dst, src, twiddle, scratch []complex64, _ []int) bool {
	const n = 384
	const stride = 128 // Distance between elements in a column

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := scratch

	// Step 1: Compute 128 radix-3 column DFTs
	// Input viewed as 128×3 matrix: x[n1, n2] = src[n1 + n2*128]
	// For each column n1: DFT3(src[n1], src[n1+128], src[n1+256])
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
	// k2=0: no twiddle (W^0 = 1)
	// k2=1: work[n1+128] *= W_384^n1 = twiddle[n1]
	// k2=2: work[n1+256] *= W_384^(2*n1) = twiddle[2*n1]
	for n1 := range stride {
		work[stride+n1] *= twiddle[n1]
	}
	for n1 := range stride {
		work[2*stride+n1] *= twiddle[2*n1]
	}

	// Prepare for 128-point sub-FFTs
	twiddle128 := mathpkg.ComputeTwiddleFactors[complex64](stride)
	bitrev128 := mathpkg.ComputeBitReversalIndicesMixed24(stride)
	subScratch := make([]complex64, stride)
	fftOut := make([]complex64, n) // Temporary for FFT outputs

	// Step 3: Compute 3 independent 128-point FFTs
	// Each row k2 of work (work[k2*128:(k2+1)*128]) gets FFT'd
	for k2 := range 3 {
		rowStart := k2 * stride
		if !amd64.ForwardAVX2Size128Mixed24Complex64Asm(
			fftOut[rowStart:rowStart+stride],
			work[rowStart:rowStart+stride],
			twiddle128, subScratch, bitrev128,
		) {
			return false
		}
	}

	// Step 4: Interleave output
	// fftOut[k2*128 + k1] contains X[k1*3 + k2]
	// Copy to dst in natural order
	for k1 := range stride {
		for k2 := range 3 {
			dst[k1*3+k2] = fftOut[k2*stride+k1]
		}
	}

	return true
}

// inverseDIT384MixedComplex64 computes a 384-point inverse FFT.
//
// Algorithm (reverse of forward):
//  1. De-interleave input: work[k2*128+k1] = src[k1*3+k2]
//  2. Compute 3 independent 128-point IFFTs
//  3. Apply conjugate twiddle factors: work[n1+k2*128] *= conj(W_384^(n1*k2))
//  4. Compute 128 radix-3 inverse column butterflies
//
// Note: 128-point IFFT includes 1/128 scaling, so we only need additional 1/3 scaling.
func inverseDIT384MixedComplex64(dst, src, twiddle, scratch []complex64, _ []int) bool {
	const n = 384
	const stride = 128

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := scratch
	ifftIn := make([]complex64, n)

	// Step 1: De-interleave input
	// src[k1*3 + k2] → ifftIn[k2*128 + k1]
	for k1 := range stride {
		for k2 := range 3 {
			ifftIn[k2*stride+k1] = src[k1*3+k2]
		}
	}

	// Prepare for 128-point sub-IFFTs
	twiddle128 := mathpkg.ComputeTwiddleFactors[complex64](stride)
	bitrev128 := mathpkg.ComputeBitReversalIndicesMixed24(stride)
	subScratch := make([]complex64, stride)

	// Step 2: Compute 3 independent 128-point IFFTs
	// Note: The 128-point IFFT includes 1/128 scaling internally
	for k2 := range 3 {
		rowStart := k2 * stride
		if !amd64.InverseAVX2Size128Mixed24Complex64Asm(
			work[rowStart:rowStart+stride],
			ifftIn[rowStart:rowStart+stride],
			twiddle128, subScratch, bitrev128,
		) {
			return false
		}
	}

	// Step 3: Apply conjugate twiddle factors
	// k2=0: no twiddle
	// k2=1: work[n1+128] *= conj(W_384^n1)
	// k2=2: work[n1+256] *= conj(W_384^(2*n1))
	for n1 := range stride {
		work[stride+n1] *= mathpkg.Conj(twiddle[n1])
	}
	for n1 := range stride {
		work[2*stride+n1] *= mathpkg.Conj(twiddle[2*n1])
	}

	// Step 4: Compute 128 radix-3 inverse column butterflies
	// Output directly to dst in natural order
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
