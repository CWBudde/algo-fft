package fftypes

// KernelStrategy controls how plans choose between DIT, Stockham, and step kernels.
type KernelStrategy uint32

// The constants below use explicit numeric values (not iota) so that the
// removal of KernelEightStep (value 4) does not silently renumber
// KernelBluestein and everything after it. See strategy_numbering_test.go
// in the root package.
const (
	KernelAuto     KernelStrategy = 0
	KernelDIT      KernelStrategy = 1
	KernelStockham KernelStrategy = 2
	KernelSixStep  KernelStrategy = 3
	// 4 was KernelEightStep, removed 2026-08-02 as a duplicate implementation
	// of KernelSixStep (same algorithm, different name). The value is
	// retired and must not be reused.
	KernelBluestein  KernelStrategy = 5
	KernelRecursive  KernelStrategy = 6 // Recursive decomposition with codelet leaves
	KernelSplitRadix KernelStrategy = 7 // Split-radix (2/4) DIT, power-of-two sizes only
	KernelFourStep   KernelStrategy = 8 // Four-step (cache-blocked rectangular six-step), power-of-two sizes
	// KernelMixedRadix is the mixed-radix engine (factors 2/3/5/7/11), the
	// route every non-power-of-two length outside Bluestein actually takes.
	// Plans report it instead of a power-of-two strategy that cannot execute
	// their length; it is resolved by the planner rather than forced.
	KernelMixedRadix KernelStrategy = 9
)

// SIMDLevel describes the minimum required CPU features for a codelet.
type SIMDLevel uint8

const (
	SIMDNone   SIMDLevel = iota // Pure Go implementation
	SIMDSSE2                    // Requires SSE2 (x86_64 baseline)
	SIMDSSE3                    // Requires SSE3
	SIMDAVX2                    // Requires AVX2
	SIMDAVX512                  // Requires AVX-512
	SIMDNEON                    // Requires ARM NEON
)

// String returns a human-readable name for the SIMD level.
func (s SIMDLevel) String() string {
	switch s {
	case SIMDNone:
		return "generic"
	case SIMDSSE2:
		return "sse2"
	case SIMDSSE3:
		return "sse3"
	case SIMDAVX2:
		return "avx2"
	case SIMDAVX512:
		return "avx512"
	case SIMDNEON:
		return "neon"
	default:
		return "unknown"
	}
}
