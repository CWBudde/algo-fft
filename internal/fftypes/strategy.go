package fftypes

// KernelStrategy controls how plans choose between DIT, Stockham, and step kernels.
type KernelStrategy uint32

const (
	KernelAuto KernelStrategy = iota
	KernelDIT
	KernelStockham
	KernelSixStep
	KernelEightStep
	KernelBluestein
	KernelRecursive // Recursive decomposition with codelet leaves
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
