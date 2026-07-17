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

// CPU-feature bit positions used by the wisdom cache key. The layout is part of
// the persisted wisdom format (version 2); changing it requires a format bump.
const (
	featSSE2   uint64 = 1 << 0
	featSSE3   uint64 = 1 << 1
	featAVX2   uint64 = 1 << 2
	featAVX512 uint64 = 1 << 3
	featNEON   uint64 = 1 << 4

	// featMaskAll is the union of all defined feature bits. Any wisdom entry whose
	// feature mask has bits outside this set comes from an incompatible format.
	featMaskAll = featSSE2 | featSSE3 | featAVX2 | featAVX512 | featNEON
)

// CPUFeatureMask returns a bitmask of CPU features relevant for planning.
// SSE3 is tracked separately from SSE2 so wisdom tuned on an SSE3 machine is not
// reused on an SSE2-only one.
func CPUFeatureMask(hasSSE2, hasSSE3, hasAVX2, hasAVX512, hasNEON bool) uint64 {
	var mask uint64

	if hasSSE2 {
		mask |= featSSE2
	}

	if hasSSE3 {
		mask |= featSSE3
	}

	if hasAVX2 {
		mask |= featAVX2
	}

	if hasAVX512 {
		mask |= featAVX512
	}

	if hasNEON {
		mask |= featNEON
	}

	return mask
}

// StrategyToAlgorithmName converts a kernel strategy to an algorithm name.
// This is used for wisdom cache entries and debugging output.
// Algorithm names shared between wisdom entries and strategy mapping.
const (
	algoDITFallback = "dit_fallback"
	algoStockham    = "stockham"
	algoSixStep     = "sixstep"
	algoEightStep   = "eightstep"
	algoBluestein   = "bluestein"
	algoSplitRadix  = "splitradix"
)

func StrategyToAlgorithmName(strategy KernelStrategy) string {
	switch strategy {
	case KernelDIT:
		return algoDITFallback
	case KernelStockham:
		return algoStockham
	case KernelSixStep:
		return algoSixStep
	case KernelEightStep:
		return algoEightStep
	case KernelBluestein:
		return algoBluestein
	case KernelSplitRadix:
		return algoSplitRadix
	default:
		return "unknown"
	}
}
