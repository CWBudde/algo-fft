package planner

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
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

	// Codelet twiddle preparation callbacks (nil if codelet uses standard twiddles)
	TwiddleSize    TwiddleSizeFunc       // Returns element count for codelet twiddles
	PrepareTwiddle PrepareTwiddleFunc[T] // Prepares twiddle layout for the codelet
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
func EstimatePlan[T Complex](n int, features cpu.Features, wisdom WisdomStore, forcedStrategy KernelStrategy) PlanEstimate[T] {
	strategy := ResolveKernelStrategy(n)
	if forcedStrategy != KernelAuto {
		strategy = forcedStrategy
	}

	// For Bluestein, there are no codelets
	if !IsPowerOf2(n) && !IsHighlyComposite(n) {
		return PlanEstimate[T]{
			Strategy:  KernelBluestein,
			Algorithm: "bluestein",
		}
	}

	// 1. Try codelet registry first (highest priority - zero dispatch)
	if est := tryRegistry[T](n, features, forcedStrategy); est != nil {
		return *est
	}

	// 2. Try wisdom cache (if provided)
	if est, wisStrat, found := resolveWisdom[T](n, features, wisdom, forcedStrategy); found {
		if est != nil {
			return *est
		}

		strategy = wisStrat
	}

	// 3. Fall back to heuristic kernel selection
	algorithmName := StrategyToAlgorithmName(strategy)

	return PlanEstimate[T]{
		Strategy:  strategy,
		Algorithm: algorithmName,
	}
}

func tryRegistry[T Complex](n int, features cpu.Features, forcedStrategy KernelStrategy) *PlanEstimate[T] {
	registry := GetRegistry[T]()
	if registry == nil {
		return nil
	}

	entry := registry.Lookup(n, features)
	if entry == nil {
		return nil
	}

	if forcedStrategy != KernelAuto && entry.Algorithm != forcedStrategy {
		return nil
	}

	return &PlanEstimate[T]{
		ForwardCodelet: entry.Forward,
		InverseCodelet: entry.Inverse,
		Algorithm:      entry.Signature,
		Strategy:       entry.Algorithm,
		TwiddleSize:    entry.TwiddleSize,
		PrepareTwiddle: entry.PrepareTwiddle,
	}
}

func resolveWisdom[T Complex](n int, features cpu.Features, wisdom WisdomStore, forcedStrategy KernelStrategy) (*PlanEstimate[T], KernelStrategy, bool) {
	if wisdom == nil {
		return nil, KernelAuto, false
	}

	var (
		precision uint8
		zero      T
	)

	switch any(zero).(type) {
	case complex64:
		precision = 0
	case complex128:
		precision = 1
	}

	cpuFeatures := CPUFeatureMask(features.HasSSE2, features.HasAVX2, features.HasAVX512, features.HasNEON)

	algorithm, found := wisdom.LookupWisdom(n, precision, cpuFeatures)
	if !found {
		return nil, KernelAuto, false
	}

	// Wisdom provides algorithm name, try to bind specific codelet by signature
	registry := GetRegistry[T]()
	if registry != nil {
		if codelet := registry.LookupBySignature(n, algorithm); codelet != nil {
			if forcedStrategy != KernelAuto && codelet.Algorithm != forcedStrategy {
				return nil, KernelAuto, false
			}

			return &PlanEstimate[T]{
				ForwardCodelet: codelet.Forward,
				InverseCodelet: codelet.Inverse,
				Algorithm:      codelet.Signature,
				Strategy:       codelet.Algorithm,
				TwiddleSize:    codelet.TwiddleSize,
				PrepareTwiddle: codelet.PrepareTwiddle,
			}, KernelAuto, true
		}
	}

	// Wisdom algorithm doesn't match a codelet, apply as kernel strategy
	var strategy KernelStrategy

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
	default:
		return nil, KernelAuto, false
	}

	if forcedStrategy != KernelAuto && strategy != forcedStrategy {
		return nil, KernelAuto, false
	}

	return nil, strategy, true
}

// HasCodelet returns true if a codelet is available for the given size.
func HasCodelet[T Complex](n int, features cpu.Features) bool {
	registry := GetRegistry[T]()
	if registry == nil {
		return false
	}

	return registry.Lookup(n, features) != nil
}
