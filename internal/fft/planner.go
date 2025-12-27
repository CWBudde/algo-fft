package fft

import (
	"github.com/MeKo-Christian/algoforge/internal/cpu"
)

// WisdomStore interface for dependency injection from root package.
// This is a minimal interface that doesn't require importing the root package.
type WisdomStore interface {
	// LookupWisdom returns the algorithm name for a given FFT configuration.
	// Returns empty string if no wisdom is available.
	LookupWisdom(size int, precision uint8, cpuFeatures uint64) (algorithm string, found bool)
}

// PlanEstimate holds the result of estimating which kernel/codelet to use.
type PlanEstimate[T Complex] struct {
	// ForwardCodelet is the directly-bound forward codelet (nil if none)
	ForwardCodelet CodeletFunc[T]

	// InverseCodelet is the directly-bound inverse codelet (nil if none)
	InverseCodelet CodeletFunc[T]

	// Algorithm is the human-readable name of the chosen implementation
	Algorithm string

	// Strategy is the kernel strategy (DIT, Stockham, etc.)
	Strategy KernelStrategy
}

// EstimatePlan determines the best kernel/codelet for the given size.
// It checks in order:
//  1. Codelet registry (highest priority - zero dispatch)
//  2. Wisdom cache (if provided)
//  3. Heuristic strategy selection (fallback)
//
// The returned PlanEstimate contains either:
//   - Direct codelet bindings (zero dispatch) if a codelet is registered for the size
//   - Empty codelet fields and just Strategy if no codelet (caller uses fallback kernels)
func EstimatePlan[T Complex](n int, features cpu.Features, wisdom WisdomStore) PlanEstimate[T] {
	strategy := ResolveKernelStrategy(n)

	// For Bluestein, there are no codelets
	if !IsPowerOfTwo(n) && !IsHighlyComposite(n) {
		return PlanEstimate[T]{
			Strategy:  KernelBluestein,
			Algorithm: "bluestein",
		}
	}

	// 1. Try codelet registry first (highest priority - zero dispatch)
	registry := GetRegistry[T]()
	if registry != nil {
		entry := registry.Lookup(n, features)
		if entry != nil {
			return PlanEstimate[T]{
				ForwardCodelet: entry.Forward,
				InverseCodelet: entry.Inverse,
				Algorithm:      entry.Signature,
				Strategy:       entry.Algorithm,
			}
		}
	}

	// 2. Try wisdom cache (if provided)
	if wisdom != nil {
		var precision uint8
		var zero T
		switch any(zero).(type) {
		case complex64:
			precision = 0
		case complex128:
			precision = 1
		}

		cpuFeatures := CPUFeatureMask(features.HasSSE2, features.HasAVX2, features.HasAVX512, features.HasNEON)

		if algorithm, found := wisdom.LookupWisdom(n, precision, cpuFeatures); found {
			// Wisdom provides algorithm name, try to bind specific codelet by signature
			if registry != nil {
				if codelet := registry.LookupBySignature(n, algorithm); codelet != nil {
					return PlanEstimate[T]{
						ForwardCodelet: codelet.Forward,
						InverseCodelet: codelet.Inverse,
						Algorithm:      codelet.Signature,
						Strategy:       codelet.Algorithm,
					}
				}
			}

			// Wisdom algorithm doesn't match a codelet, apply as kernel strategy
			switch algorithm {
			case "dit_fallback":
				strategy = KernelDIT
			case "stockham":
				strategy = KernelStockham
			case "sixstep":
				strategy = KernelSixStep
			case "eightstep":
				strategy = KernelEightStep
			case "bluestein":
				strategy = KernelBluestein
			}
		}
	}

	// 3. Fall back to heuristic kernel selection
	algorithmName := strategyToAlgorithmName(strategy)

	return PlanEstimate[T]{
		Strategy:  strategy,
		Algorithm: algorithmName,
	}
}

// HasCodelet returns true if a codelet is available for the given size.
func HasCodelet[T Complex](n int, features cpu.Features) bool {
	registry := GetRegistry[T]()
	if registry == nil {
		return false
	}

	return registry.Lookup(n, features) != nil
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

// strategyToAlgorithmName converts a kernel strategy to an algorithm name.
func strategyToAlgorithmName(strategy KernelStrategy) string {
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
