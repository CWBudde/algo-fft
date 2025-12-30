package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/fftypes"
	m "github.com/MeKo-Christian/algo-fft/internal/math"
)

// Complex is a type alias for the complex number constraint.
// The canonical definition is in internal/fftypes.
type Complex = fftypes.Complex

// ComputeTwiddleFactors returns the precomputed twiddle factors (roots of unity).
// Re-exported from internal/math.
func ComputeTwiddleFactors[T Complex](n int) []T {
	return m.ComputeTwiddleFactors[T](n)
}

// ComputeBitReversalIndices returns the bit-reversal permutation indices.
// Re-exported from internal/math.
var ComputeBitReversalIndices = m.ComputeBitReversalIndices

// ConjugateOf returns the complex conjugate of val.
// Re-exported from internal/math.
func ConjugateOf[T Complex](val T) T {
	return m.ConjugateOf[T](val)
}

// conj returns the complex conjugate (private wrapper).
func conj[T Complex](val T) T {
	return m.Conj[T](val)
}

// complexFromFloat64 creates a complex number of type T from float64 components (private wrapper).
func complexFromFloat64[T Complex](re, im float64) T {
	return m.ComplexFromFloat64[T](re, im)
}

// log2 returns the base-2 logarithm (private wrapper).
var log2 = m.Log2

// reverseBits reverses bits (private wrapper).
var reverseBits = m.ReverseBits
