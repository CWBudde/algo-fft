package kernels

import (
	"math"

	m "github.com/cwbudde/algo-fft/internal/math"
)

// Helper wrappers from internal/math.
func complexFromFloat64[T Complex](re, im float64) T {
	return m.ComplexFromFloat64[T](re, im)
}

func conj[T Complex](val T) T {
	return m.Conj[T](val)
}

// ComputeChirpSequence computes the chirp sequence W_n^(k^2/2) = exp(-j * pi * k^2 / n)
// The sequence is of length n.
func ComputeChirpSequence[T Complex](n int) []T {
	chirp := make([]T, n)

	invN := 1.0 / float64(n)
	for k := range n {
		angle := -math.Pi * float64(k*k) * invN
		re := math.Cos(angle)
		im := math.Sin(angle)
		chirp[k] = complexFromFloat64[T](re, im)
	}

	return chirp
}

// BuildBluesteinSequence constructs the time-domain filter sequence b of
// length m for Bluestein's algorithm:
//
//	b[k]   = conj(chirp[k]) for 0 <= k < n
//	b[m-k] = conj(chirp[k]) for 1 <= k < n
//
// with the middle zero-padded. The caller FFTs it to obtain the
// frequency-domain filter (see ComputeBluesteinFilter for the power-of-two
// path; internal/fft handles 5-smooth padded sizes via mixed-radix).
func BuildBluesteinSequence[T Complex](n, m int, chirp []T) []T {
	b := make([]T, m)

	b[0] = conj(chirp[0])
	for k := 1; k < n; k++ {
		val := conj(chirp[k])
		b[k] = val
		b[m-k] = val
	}

	return b
}

// ComputeBluesteinFilter computes the frequency-domain filter for Bluestein's algorithm.
// n is the original size, m is the padded size (power of 2 >= 2n-1).
// chirp is the sequence of length n computed by ComputeChirpSequence.
// twiddles are for size m.
// scratch is a pre-allocated buffer of size m for intermediate computations.
func ComputeBluesteinFilter[T Complex](n, m int, chirp []T, twiddles []T, scratch []T) []T {
	b := BuildBluesteinSequence(n, m, chirp)

	// Perform FFT using provided scratch buffer
	bluesteinSubForward(b, b, twiddles, scratch, nil)

	return b
}

// bluesteinSubFFTDispatchMax bounds the padded sizes routed through the
// size-dispatched DIT kernels (forwardDITComplex64 and friends). Above this
// the dispatcher's remaining fallbacks recompute index tables per call
// (allocating), so the cached-bitrev generic radix-2 path stays both faster
// and allocation-free there.
const bluesteinSubFFTDispatchMax = 4096

// bluesteinSubForward runs the padded power-of-two sub-FFT of a Bluestein
// transform. It prefers the size-dispatched DIT kernels (radix-4 and
// size-specific codelets, SIMD where available), which measure 1.7–2.7x
// faster than the generic radix-2 path this replaced; sizes the dispatcher
// cannot serve allocation-free fall back to generic radix-2 with the caller's
// cached bit-reversal table.
func bluesteinSubForward[T Complex](dst, src, twiddle, scratch []T, bitrev []int) {
	if len(src) <= bluesteinSubFFTDispatchMax {
		switch d := any(dst).(type) {
		case []complex64:
			if forwardDITComplex64(d, any(src).([]complex64), any(twiddle).([]complex64), any(scratch).([]complex64)) { //nolint:forcetypeassert
				return
			}
		case []complex128:
			if forwardDITComplex128(d, any(src).([]complex128), any(twiddle).([]complex128), any(scratch).([]complex128)) { //nolint:forcetypeassert
				return
			}
		}
	}

	ditForwardBitrev(dst, src, twiddle, scratch, bitrev)
}

// bluesteinSubInverse is the inverse counterpart of bluesteinSubForward.
// Both paths scale the output by 1/m, per the library convention.
func bluesteinSubInverse[T Complex](dst, src, twiddle, scratch []T, bitrev []int) {
	if len(src) <= bluesteinSubFFTDispatchMax {
		switch d := any(dst).(type) {
		case []complex64:
			if inverseDITComplex64(d, any(src).([]complex64), any(twiddle).([]complex64), any(scratch).([]complex64)) { //nolint:forcetypeassert
				return
			}
		case []complex128:
			if inverseDITComplex128(d, any(src).([]complex128), any(twiddle).([]complex128), any(scratch).([]complex128)) { //nolint:forcetypeassert
				return
			}
		}
	}

	ditInverseBitrev(dst, src, twiddle, scratch, bitrev)
}

// BluesteinConvolution performs the convolution y = x * b via FFT.
// dst is the output buffer of size m.
// x is the input sequence of size m (padded with zeros).
// filter is the frequency-domain filter (FFT of b) of size m.
// twiddles are for size m.
// scratch is a scratch buffer of size m.
// bitrev is the precomputed bit-reversal table for size m; nil recomputes it
// per call, which allocates — plans should pass their cached table.
func BluesteinConvolution[T Complex](dst, x, filter, twiddles, scratch []T, bitrev []int) {
	// 1. FFT of x
	// We use dst as the work buffer. If dst != x, the sub-FFT handles the copy/transform.
	bluesteinSubForward(dst, x, twiddles, scratch, bitrev)

	// 2. Multiply by filter
	for i := range dst {
		dst[i] *= filter[i]
	}

	// 3. IFFT
	bluesteinSubInverse(dst, dst, twiddles, scratch, bitrev)
}
