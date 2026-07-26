package fft

import mathpkg "github.com/cwbudde/algo-fft/internal/math"

// RecombineForwardComplex64 performs the forward real-FFT recombination step.
// Given the FFT Y of the packed even/odd input in src (length half) and the
// precomputed weights U (length >= half), it writes
//
//	dst[k] = Y[k] - U[k]*(Y[k] - conj(Y[half-k]))  for k = 1..half-1.
//
// Only bins 1..half-1 are written, so dst must have length >= half and must
// not alias src. Callers pass the half+1-length spectrum buffer and fill the
// DC and Nyquist bins (dst[0] and dst[half]) themselves; this function never
// touches them.
func RecombineForwardComplex64(dst, src, weight []complex64) {
	half := len(src)
	if half < 2 {
		return
	}

	start := max(recombineForwardComplex64SIMD(dst, src, weight), 1)

	recombineForwardComplex64Generic(dst, src, weight, start)
}

// RecombineForwardComplex128 performs the forward real-FFT recombination step.
// See RecombineForwardComplex64 for the contract; dst must not alias src.
func RecombineForwardComplex128(dst, src, weight []complex128) {
	half := len(src)
	if half < 2 {
		return
	}

	start := max(recombineForwardComplex128SIMD(dst, src, weight), 1)

	recombineForwardComplex128Generic(dst, src, weight, start)
}

func recombineForwardComplex64Generic(dst, src, weight []complex64, start int) {
	half := len(src)
	for k := start; k < half; k++ {
		a := src[k]
		bSrc := src[half-k]
		b := complex(real(bSrc), -imag(bSrc)) // conj(Y[half-k])

		// mathpkg.MulComplex64 rather than `*`: the operator widens both
		// operands to complex128 and rounds back (see math.MulComplex64), which
		// is also a different rounding from the AVX2/SSE3 kernels that handle
		// the leading bins of this same loop.
		c := mathpkg.MulComplex64(weight[k], a-b)
		dst[k] = a - c
	}
}

func recombineForwardComplex128Generic(dst, src, weight []complex128, start int) {
	half := len(src)
	for k := start; k < half; k++ {
		a := src[k]
		bSrc := src[half-k]
		b := complex(real(bSrc), -imag(bSrc)) // conj(Y[half-k])

		// MulComplex128 is the plain operator; it keeps this loop
		// line-for-line comparable with its complex64 twin above.
		c := mathpkg.MulComplex128(weight[k], a-b)
		dst[k] = a - c
	}
}
