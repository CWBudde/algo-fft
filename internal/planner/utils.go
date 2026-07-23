package planner

import (
	"math/bits"

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

// IsHighlyComposite checks if n only contains 2, 3, or 5 factors.
// Re-exported from internal/math.
var IsHighlyComposite = m.IsHighlyComposite

// IsMixedRadixSmooth checks if n can be executed exactly by the mixed-radix
// engine (factors 2, 3, 5, 7, 11). Re-exported from internal/math.
var IsMixedRadixSmooth = m.IsMixedRadixSmooth

// maxExactLength mirrors the root package's maxBluesteinLength: beyond it the
// Bluestein pad size 2n-1 is not representable, so the mixed-radix engine is
// the only executable path for a smooth length and the win gate must not
// exclude it (its 2n-1 arithmetic would also wrap).
const maxExactLength = 1 << (bits.UintSize - 3)

// MixedRadixEligible reports whether a non-power-of-two length should run on
// the mixed-radix engine instead of Bluestein. All 5-smooth lengths qualify
// (the engine has been the incumbent for them since mixed-radix landed).
// Lengths with factors 7/11 qualify only where the engine measured faster
// than Bluestein (see mixedRadix7And11Wins); the rest keep the Bluestein
// routing they had before radix-7/11 existed.
func MixedRadixEligible(n int) bool {
	if !m.IsMixedRadixSmooth(n) {
		return false
	}

	if m.IsHighlyComposite(n) {
		return true
	}

	return mixedRadix7And11Wins(n)
}

// mixedRadix7And11Wins is the measured win gate for lengths containing
// factors 7/11 (BenchmarkMixedRadix7And11VsBluestein, AVX2 amd64, both
// precisions; the purego build measured mixed-radix ahead at every tested
// shape, so this gate only forgoes small purego wins at the excluded shapes):
//
//   - power-of-two part >= 8 wins 1.3-6x at every size (56 ... 14080): the
//     schedule strips the odd factors first and lands the pow2 part in
//     radix-8 passes or a tuned codelet leaf;
//   - power-of-two part 2 or 4 measured as losses (14, 22, 28, 44, 308, 462,
//     924): the strided radix-2/4 tail stages dominate, as they did for the
//     Rader gate (see raderConvolutionWins);
//   - odd lengths win when Bluestein's padded power-of-two sub-FFT is
//     >= ~2.5n (11, 33, 35, 49, 77, 165, 385, 539, 693, 1155, 2401 at
//     1.2-3.4x) and wash or lose below that (7, 55, 63, 105, 121, 231, 847),
//     where the ~2x pad lands on an unusually effective codelet.
func mixedRadix7And11Wins(n int) bool {
	if n > maxExactLength {
		return true
	}

	pow2 := n & -n
	if pow2 >= 8 {
		return true
	}

	if pow2 > 1 {
		return false
	}

	// Odd: Bluestein pads to the next power of two >= 2n-1; require
	// pad >= 2.5n. With n odd this is exactly pad-2n >= (n+1)/2, phrased so
	// no intermediate exceeds 2^(UintSize-2): the direct 2*pad and 5*n forms
	// overflow 32-bit int for n near maxExactLength and could flip the
	// comparison. pad-2n >= -1 always (pad >= 2n-1), so the left side is
	// safe too.
	pad := m.NextPowerOfTwo(2*n - 1)

	return pad-2*n >= (n+1)/2
}

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

// Algorithm names shared between wisdom entries and strategy mapping.
const (
	algoDITFallback = "dit_fallback"
	algoStockham    = "stockham"
	algoSixStep     = "sixstep"
	algoEightStep   = "eightstep"
	algoBluestein   = "bluestein"
	algoSplitRadix  = "splitradix"
	algoRecursive   = "recursive"

	// algoUnknown is returned for strategies with no table entry (KernelAuto).
	algoUnknown = "unknown"
)

// strategyAlgorithmNames is the single strategy↔algorithm-name table. Both
// StrategyToAlgorithmName and AlgorithmNameToStrategy derive from it, so the
// two directions cannot drift apart. The names are part of the persisted
// wisdom format — do not rename existing entries.
//
//nolint:gochecknoglobals // static lookup table
var strategyAlgorithmNames = map[KernelStrategy]string{
	KernelDIT:        algoDITFallback,
	KernelStockham:   algoStockham,
	KernelSixStep:    algoSixStep,
	KernelEightStep:  algoEightStep,
	KernelBluestein:  algoBluestein,
	KernelSplitRadix: algoSplitRadix,
	KernelRecursive:  algoRecursive,
}

// algorithmNameStrategies is the reverse of strategyAlgorithmNames, built at
// init from the same table.
//
//nolint:gochecknoglobals // static lookup table
var algorithmNameStrategies = func() map[string]KernelStrategy {
	rev := make(map[string]KernelStrategy, len(strategyAlgorithmNames))
	for strategy, name := range strategyAlgorithmNames {
		rev[name] = strategy
	}

	return rev
}()

// StrategyToAlgorithmName converts a kernel strategy to the algorithm name
// used in wisdom cache entries and debugging output. Unmapped strategies
// (KernelAuto) return "unknown".
func StrategyToAlgorithmName(strategy KernelStrategy) string {
	name, ok := strategyAlgorithmNames[strategy]
	if !ok {
		return algoUnknown
	}

	return name
}

// AlgorithmNameToStrategy converts a wisdom algorithm name back to its kernel
// strategy. Names that do not correspond to a strategy (codelet signatures,
// "unknown") return ok=false.
func AlgorithmNameToStrategy(name string) (KernelStrategy, bool) {
	strategy, ok := algorithmNameStrategies[name]

	return strategy, ok
}
