//go:build amd64 && !purego

package kernels

import (
	"sync"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// The 384-point codelets decompose into three 128-point sub-FFTs, which run on
// the size-generic 256-bit radix-4 kernel: 128 = 2*4^3, so that kernel runs its
// radix-4 stages to 64 and combines the two halves with a radix-2 tail — a
// radix-4-then-2 at this size, by construction rather than by having a file
// named for it.
//
// That kernel takes a *prepared* twiddle table (packed per-stage planes, length
// n+4) rather than the plain length-n DIT table, and it conjugates at prepare
// time, so forward and inverse need separate tables. All four depend only on the
// fixed sub-size, so they are built once at package load rather than per
// transform; they are read-only and shared across all calls.
//
// The per-call output and scratch buffers are pooled: the codelet runs as a
// synchronous leaf (its 128-point sub-FFTs go straight to assembly, never
// re-entering the codelet registry), so recycling the buffers keeps the codelet
// allocation-free after warm-up while staying safe for concurrent use.
//
//nolint:gochecknoglobals
var (
	dit384Sub128FwdTwiddleC64 = newDIT384Sub128TwiddleC64(false)
	dit384Sub128InvTwiddleC64 = newDIT384Sub128TwiddleC64(true)
	dit384OutPoolC64          = sync.Pool{New: func() any { return new([384]complex64) }}
	dit384SubScratchPoolC64   = sync.Pool{New: func() any { return new([128]complex64) }}

	dit384Sub128FwdTwiddleC128 = newDIT384Sub128TwiddleC128(false)
	dit384Sub128InvTwiddleC128 = newDIT384Sub128TwiddleC128(true)
	dit384OutPoolC128          = sync.Pool{New: func() any { return new([384]complex128) }}
	dit384SubScratchPoolC128   = sync.Pool{New: func() any { return new([128]complex128) }}
)

// dit384SubSize is the sub-FFT length of the 128x3 decomposition. It is also the
// stride between the three sub-FFT rows.
const dit384SubSize = 128

// newDIT384Sub128TwiddleC64 builds the packed radix-4 twiddle table for one
// direction of the 128-point sub-FFT.
func newDIT384Sub128TwiddleC64(inverse bool) []complex64 {
	table := make([]complex64, twiddleSizeRadix4AVX2(dit384SubSize))
	prepareTwiddleRadix4AVX2(dit384SubSize, inverse, table)

	return table
}

// newDIT384Sub128TwiddleC128 is the complex128 twin of
// newDIT384Sub128TwiddleC64.
func newDIT384Sub128TwiddleC128(inverse bool) []complex128 {
	table := make([]complex128, twiddleSizeRadix4AVX2Complex128(dit384SubSize))
	prepareTwiddleRadix4AVX2Complex128(dit384SubSize, inverse, table)

	return table
}

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
	twiddle128 := dit384Sub128FwdTwiddleC64

	subPtr := dit384SubScratchPoolC64.Get().(*[128]complex64) //nolint:forcetypeassert
	defer dit384SubScratchPoolC64.Put(subPtr)

	subScratch := subPtr[:]

	outPtr := dit384OutPoolC64.Get().(*[384]complex64) //nolint:forcetypeassert
	defer dit384OutPoolC64.Put(outPtr)

	fftOut := outPtr[:]

	// Step 3: Compute 3 independent 128-point FFTs.
	for k2 := range 3 {
		rowStart := k2 * stride
		if !forwardRadix4AVX2Complex64(
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
	twiddle128 := dit384Sub128InvTwiddleC64

	subPtr := dit384SubScratchPoolC64.Get().(*[128]complex64) //nolint:forcetypeassert
	defer dit384SubScratchPoolC64.Put(subPtr)

	subScratch := subPtr[:]

	// Step 2: Compute 3 independent 128-point IFFTs.
	for k2 := range 3 {
		rowStart := k2 * stride
		if !inverseRadix4AVX2Complex64(
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
	twiddle128 := dit384Sub128FwdTwiddleC128

	subPtr := dit384SubScratchPoolC128.Get().(*[128]complex128) //nolint:forcetypeassert
	defer dit384SubScratchPoolC128.Put(subPtr)

	subScratch := subPtr[:]

	outPtr := dit384OutPoolC128.Get().(*[384]complex128) //nolint:forcetypeassert
	defer dit384OutPoolC128.Put(outPtr)

	fftOut := outPtr[:]

	// Step 3: Compute 3 independent 128-point FFTs
	for k2 := range 3 {
		rowStart := k2 * stride
		if !forwardRadix4AVX2Complex128(
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
	twiddle128 := dit384Sub128InvTwiddleC128

	subPtr := dit384SubScratchPoolC128.Get().(*[128]complex128) //nolint:forcetypeassert
	defer dit384SubScratchPoolC128.Put(subPtr)

	subScratch := subPtr[:]

	// Step 2: Compute 3 independent 128-point IFFTs
	for k2 := range 3 {
		rowStart := k2 * stride
		if !inverseRadix4AVX2Complex128(
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
