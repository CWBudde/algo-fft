package fft

import (
	"github.com/MeKo-Christian/algoforge/internal/cpu"
)

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
// It checks the codelet registry first for direct binding, then falls back
// to the kernel selection logic.
//
// The returned PlanEstimate contains either:
//   - Direct codelet bindings (zero dispatch) if a codelet is registered for the size
//   - Empty codelet fields and just Strategy if no codelet (caller uses fallback kernels)
func EstimatePlan[T Complex](n int, features cpu.Features) PlanEstimate[T] {
	strategy := ResolveKernelStrategy(n)

	// For Bluestein, there are no codelets
	if !IsPowerOfTwo(n) && !IsHighlyComposite(n) {
		return PlanEstimate[T]{
			Strategy:  KernelBluestein,
			Algorithm: "bluestein",
		}
	}

	// Try to find a codelet for this size
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

	// No codelet found, fall back to kernel selection
	// Caller will use SelectKernelsWithStrategy
	algorithmName := ""
	switch strategy {
	case KernelDIT:
		algorithmName = "dit_fallback"
	case KernelStockham:
		algorithmName = "stockham"
	case KernelSixStep:
		algorithmName = "sixstep"
	case KernelEightStep:
		algorithmName = "eightstep"
	}

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
