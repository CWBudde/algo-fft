package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
)

// raderExecutor runs prime-length transforms via Rader's algorithm (see
// internal/fft/rader.go for the table construction and the algorithm
// derivation). Both directions share the same structure:
//
//  1. Gather the nonzero-index inputs through the generator permutation while
//     accumulating the total sum (which is bin 0 of the result).
//  2. Run the length-(N-1) cyclic convolution against the precomputed
//     frequency-domain filter.
//  3. Scatter x[0] plus the convolution back through the inverse permutation.
//
// The sub-FFT runs at exactly N-1 (5-smooth by eligibility), never padded.
type raderExecutor[T Complex] struct {
	n int // Prime transform length

	permIn  []int // Input gather permutation: g^(-q) mod N
	permOut []int // Output scatter permutation: g^m mod N

	filter    []T // Size N-1
	filterInv []T // Size N-1
	twiddle   []T // Size N-1 (sub-FFT twiddles)

	// bitrev feeds only the power-of-two DIT sub-FFT path; nil when N-1 runs
	// through the mixed-radix engine.
	bitrev []int
}

func (e *raderExecutor[T]) forward(dst, src, scratch, sub []T) {
	l := e.n - 1
	x0 := src[0]
	sum := x0

	for q, j := range e.permIn {
		v := src[j]
		scratch[q] = v
		sum += v
	}

	fft.BluesteinConvolution(scratch[:l], scratch[:l], e.filter, e.twiddle, sub[:l], e.bitrev)

	dst[0] = sum
	for i, k := range e.permOut {
		dst[k] = x0 + scratch[i]
	}
}

func (e *raderExecutor[T]) inverse(dst, src, scratch, sub []T) {
	l := e.n - 1
	x0 := src[0]
	sum := x0

	for q, j := range e.permIn {
		v := src[j]
		scratch[q] = v
		sum += v
	}

	fft.BluesteinConvolution(scratch[:l], scratch[:l], e.filterInv, e.twiddle, sub[:l], e.bitrev)

	// Scatter unscaled, then apply the real 1/N factor over the whole output
	// through the SIMD scale helper. Folding the factor into the scatter loop
	// would need a complex multiply per element, and Go compiles scalar
	// complex64 multiplication by widening to complex128 and rounding back
	// (see math.MulComplex64) — one of the reasons complex64 measured slower
	// than complex128 on the arbitrary-length paths (PLAN.md P5.0).
	dst[0] = sum
	for i, k := range e.permOut {
		dst[k] = x0 + scratch[i]
	}

	fft.ScaleInPlace(dst[:e.n], 1.0/float64(e.n))
}
