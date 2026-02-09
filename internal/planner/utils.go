package planner

import (
	m "github.com/cwbudde/algo-fft/internal/math"
)

// ComputeTwiddleFactors returns the precomputed twiddle factors (roots of unity).
// Re-exported from internal/math.
func ComputeTwiddleFactors[T Complex](n int) []T {
	return m.ComputeTwiddleFactors[T](n)
}

// ComputeBitReversalIndices returns the bit-reversal permutation indices.
// Re-exported from internal/math.
var ComputeBitReversalIndices = m.ComputeBitReversalIndices

// IsPowerOf2 checks if n is a power of 2.
// Re-exported from internal/math.
var IsPowerOf2 = m.IsPowerOf2

// IsHighlyComposite checks if n can be efficiently factored for mixed-radix FFT.
// Re-exported from internal/math.
var IsHighlyComposite = m.IsHighlyComposite

// complexFromFloat64 creates a complex number of type T from float64 components.
func complexFromFloat64[T Complex](re, im float64) T {
	return m.ComplexFromFloat64[T](re, im)
}

// CPUFeatureMask returns a bitmask of CPU features relevant for planning.
func CPUFeatureMask(hasSSE2, hasAVX2, hasAVX512, hasNEON bool) uint64 {
	var mask uint64

	if hasSSE2 {
		mask |= 1 << 0
	}

	if hasAVX2 {
		mask |= 1 << 1
	}

	if hasAVX512 {
		mask |= 1 << 2
	}

	if hasNEON {
		mask |= 1 << 3
	}

	return mask
}

// StrategyToAlgorithmName converts a kernel strategy to an algorithm name.
// This is used for wisdom cache entries and debugging output.
func StrategyToAlgorithmName(strategy KernelStrategy) string {
	switch strategy {
	case KernelDIT:
		return "dit_fallback"
	case KernelStockham:
		return "stockham"
	case KernelSixStep:
		return "sixstep"
	case KernelEightStep:
		return "eightstep"
	case KernelBluestein:
		return "bluestein"
	default:
		return "unknown"
	}
}
