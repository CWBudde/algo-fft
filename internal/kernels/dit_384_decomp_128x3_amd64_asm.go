//go:build amd64 && !purego

package kernels

import (
	"sync"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// The 384-point codelets decompose into three 128-point sub-FFTs. The sub-FFT
// twiddle table depends only on the fixed sub-size (128), so it is computed once
// at package load rather than on every transform (it is read-only and shared
// across all calls); W_128^k == W_384^(3k), so it is exactly the stride-3 gather
// from the 384-point table it replaces. The per-call output and scratch buffers
// are pooled: the codelet runs as a synchronous leaf (its 128-point sub-FFTs are
// assembly, never re-entering Go codelets), so recycling the buffers keeps the
// codelet allocation-free after warm-up while staying safe for concurrent use.
//
//nolint:gochecknoglobals
var (
	dit384Sub128TwiddleC64  = mathpkg.ComputeTwiddleFactors[complex64](128)
	dit384OutPoolC64        = sync.Pool{New: func() any { return new([384]complex64) }}
	dit384SubScratchPoolC64 = sync.Pool{New: func() any { return new([128]complex64) }}

	dit384Sub128TwiddleC128  = mathpkg.ComputeTwiddleFactors[complex128](128)
	dit384OutPoolC128        = sync.Pool{New: func() any { return new([384]complex128) }}
	dit384SubScratchPoolC128 = sync.Pool{New: func() any { return new([128]complex128) }}
)

// forwardDIT384MixedComplex64 computes a 384-point forward FFT using the
// 128×3 decomposition (radix-3 first, then 128-point FFTs).
func forwardDIT384MixedComplex64(dst, src, twiddle, scratch []complex64) bool {
	const n = 384
	const stride = 128 // Distance between elements in a column

	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	work := scratch[:n]

	// Step 1: Compute 128 radix-3 column DFTs.
	// Input viewed as a 128×3 matrix: x[n1, n2] = src[n1 + n2*128].
	copy(work, src[:n])
	amd64.Radix3Butterflies384ForwardComplex64Asm(work)

	// Step 2: Apply twiddle factors W_384^(n1*k2).
	amd64.ApplyTwiddle384Complex64Asm(work, twiddle)

	// Prepare for 128-point sub-FFTs (twiddle precomputed, buffers pooled)
	twiddle128 := dit384Sub128TwiddleC64

	subPtr := dit384SubScratchPoolC64.Get().(*[128]complex64) //nolint:forcetypeassert
	defer dit384SubScratchPoolC64.Put(subPtr)

	subScratch := subPtr[:]

	outPtr := dit384OutPoolC64.Get().(*[384]complex64) //nolint:forcetypeassert
	defer dit384OutPoolC64.Put(outPtr)

	fftOut := outPtr[:]

	// Step 3: Compute 3 independent 128-point FFTs.
	for k2 := range 3 {
		rowStart := k2 * stride
		if !amd64.ForwardAVX2Size128Radix4Then2Complex64Asm(
			fftOut[rowStart:rowStart+stride],
			work[rowStart:rowStart+stride],
			twiddle128, subScratch,
		) {
			return false
		}
	}

	// Step 4: Interleave output — X[k1*3 + k2] = FFT_result[k2][k1].
	for k1 := range stride {
		for k2 := range 3 {
			dst[k1*3+k2] = fftOut[k2*stride+k1]
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

	work := scratch[:n]

	inPtr := dit384OutPoolC64.Get().(*[384]complex64) //nolint:forcetypeassert
	defer dit384OutPoolC64.Put(inPtr)

	ifftIn := inPtr[:]

	// Step 1: De-interleave input — src[k1*3 + k2] → ifftIn[k2*128 + k1].
	for k1 := range stride {
		for k2 := range 3 {
			ifftIn[k2*stride+k1] = src[k1*3+k2]
		}
	}

	// Prepare for 128-point sub-IFFTs (twiddle precomputed, buffers pooled)
	twiddle128 := dit384Sub128TwiddleC64

	subPtr := dit384SubScratchPoolC64.Get().(*[128]complex64) //nolint:forcetypeassert
	defer dit384SubScratchPoolC64.Put(subPtr)

	subScratch := subPtr[:]

	// Step 2: Compute 3 independent 128-point IFFTs.
	for k2 := range 3 {
		rowStart := k2 * stride
		if !amd64.InverseAVX2Size128Radix4Then2Complex64Asm(
			work[rowStart:rowStart+stride],
			ifftIn[rowStart:rowStart+stride],
			twiddle128, subScratch,
		) {
			return false
		}
	}

	// Step 3: Apply conjugate twiddle factors.
	amd64.ApplyConjTwiddle384Complex64Asm(work, twiddle)

	// Step 4: Compute 128 radix-3 inverse column butterflies.
	amd64.Radix3Butterflies384InverseComplex64Asm(work)

	// Scale and copy to dst. Additional scaling (the 128-point IFFT did 1/128).
	// 1/3 is a *real* factor, so it is applied component-wise: routing it through
	// the complex-multiply helper spends two products against a zero imaginary
	// part plus an add and a subtract per output, and the compiler does not fold
	// them away even though scale is a constant.
	const scale = float32(1.0 / 3.0)

	for i := range n {
		dst[i] = complex(real(work[i])*scale, imag(work[i])*scale)
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

	// Prepare for 128-point sub-FFTs (twiddle precomputed, buffers pooled)
	twiddle128 := dit384Sub128TwiddleC128

	subPtr := dit384SubScratchPoolC128.Get().(*[128]complex128) //nolint:forcetypeassert
	defer dit384SubScratchPoolC128.Put(subPtr)

	subScratch := subPtr[:]

	outPtr := dit384OutPoolC128.Get().(*[384]complex128) //nolint:forcetypeassert
	defer dit384OutPoolC128.Put(outPtr)

	fftOut := outPtr[:]

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

	inPtr := dit384OutPoolC128.Get().(*[384]complex128) //nolint:forcetypeassert
	defer dit384OutPoolC128.Put(inPtr)

	ifftIn := inPtr[:]

	// Step 1: De-interleave input
	for k1 := range stride {
		for k2 := range 3 {
			ifftIn[k2*stride+k1] = src[k1*3+k2]
		}
	}

	// Prepare for 128-point sub-IFFTs (twiddle precomputed, buffers pooled)
	twiddle128 := dit384Sub128TwiddleC128

	subPtr := dit384SubScratchPoolC128.Get().(*[128]complex128) //nolint:forcetypeassert
	defer dit384SubScratchPoolC128.Put(subPtr)

	subScratch := subPtr[:]

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
	amd64.ApplyConjTwiddle384Complex128Asm(work, twiddle)

	// Step 4: Compute 128 radix-3 inverse column butterflies
	amd64.Radix3Butterflies384InverseComplex128Asm(work)

	// Scale and copy to dst. 1/3 is a *real* factor, so it is applied
	// component-wise rather than as a complex multiply by complex(1/3, 0): the
	// latter spends two products against a zero imaginary part plus an add and
	// a subtract per output, and the compiler does not fold them away even
	// though scale is a constant.
	const scale = float64(1.0 / 3.0)

	for i := range n {
		dst[i] = complex(real(work[i])*scale, imag(work[i])*scale)
	}

	return true
}
