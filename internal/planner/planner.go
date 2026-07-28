package planner

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
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
	ForwardCodelet fftypes.CodeletFunc[T]

	// InverseCodelet is the directly-bound inverse codelet (nil if none)
	InverseCodelet fftypes.CodeletFunc[T]

	// Algorithm is the human-readable name of the chosen implementation
	Algorithm string

	// Strategy is the kernel strategy (DIT, Stockham, etc.)
	Strategy KernelStrategy

	// Codelet twiddle preparation callbacks (nil if codelet uses standard twiddles)
	TwiddleSize    registry.TwiddleSizeFunc       // Returns element count for codelet twiddles
	PrepareTwiddle registry.PrepareTwiddleFunc[T] // Prepares twiddle layout for the codelet
}

// EstimatePlan determines the best kernel/codelet for the given size.
// It checks in order:
//  1. Wisdom naming a concrete codelet signature (zero dispatch)
//  2. Codelet registry (zero dispatch)
//  3. Wisdom naming a kernel strategy
//  4. Heuristic strategy selection (fallback)
//
// Wisdom deliberately straddles the registry rather than sitting wholly above
// or below it, because its two kinds of entry carry very different evidence:
//
//   - A signature entry ("dit64_radix4_sse2") names the same kind of thing the
//     registry does — one specific codelet for this size — but was measured on
//     this machine, whereas the registry's order is a compile-time priority
//     constant. The measurement wins. Ranking wisdom below the registry instead
//     made signature entries unreachable for every size that has a codelet
//     (all powers of two from 4 to 4096), which is exactly the set of sizes
//     where a signature can exist at all — it also made
//     registry.LookupBySignature's stale-entry guard dead code.
//   - A strategy entry ("stockham") is not comparable with a codelet. The
//     measurement behind it (internal/fft.benchmarkStrategy) times only the
//     kernel path and never the codelet, so letting it displace a codelet would
//     act on a comparison that was never made. It therefore stays a fallback
//     for sizes the registry does not cover.
//
// The returned PlanEstimate contains either:
//   - Direct codelet bindings (zero dispatch) if a codelet is registered for the size
//   - Empty codelet fields and just Strategy if no codelet (caller uses fallback kernels)
func EstimatePlan[T Complex](
	n int, features cpu.Features, wisdom WisdomStore, forcedStrategy KernelStrategy,
) PlanEstimate[T] {
	strategy := ResolveKernelStrategy(n)
	if forcedStrategy != KernelAuto {
		strategy = forcedStrategy
	}

	// For Bluestein, there are no codelets
	if !IsPowerOf2(n) && !MixedRadixEligible(n) {
		return PlanEstimate[T]{
			Strategy:  KernelBluestein,
			Algorithm: algoBluestein,
		}
	}

	algorithm, haveWisdom := wisdomAlgorithm[T](n, features, wisdom)

	// 1. A wisdom entry naming a codelet outranks the registry's static order.
	if haveWisdom {
		if est := bindWisdomCodelet[T](n, features, algorithm, forcedStrategy); est != nil {
			return *est
		}
	}

	// 2. Try the codelet registry (zero dispatch)
	if est := tryRegistry[T](n, features, forcedStrategy); est != nil {
		return *est
	}

	// 3. A wisdom entry naming a strategy applies only where no codelet ran.
	if haveWisdom {
		if wisStrat, ok := wisdomStrategy(algorithm, forcedStrategy); ok {
			strategy = wisStrat
		}
	}

	// 4. Fall back to heuristic kernel selection
	algorithmName := StrategyToAlgorithmName(strategy)

	return PlanEstimate[T]{
		Strategy:  strategy,
		Algorithm: algorithmName,
	}
}

func tryRegistry[T Complex](n int, features cpu.Features, forcedStrategy KernelStrategy) *PlanEstimate[T] {
	reg := registry.GetRegistry[T]()
	if reg == nil {
		return nil
	}

	entry := reg.Lookup(n, features)
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

// wisdomAlgorithm looks up the algorithm name wisdom recorded for this size,
// precision and CPU feature set. The name is either a codelet signature or a
// kernel strategy name; the two are resolved separately by bindWisdomCodelet
// and wisdomStrategy.
func wisdomAlgorithm[T Complex](n int, features cpu.Features, wisdom WisdomStore) (string, bool) {
	if wisdom == nil {
		return "", false
	}

	var (
		precision uint8
		zero      T
	)

	switch any(zero).(type) {
	case complex64:
		precision = PrecisionComplex64
	case complex128:
		precision = PrecisionComplex128
	}

	cpuFeatures := CPUFeatureMask(
		features.HasSSE2, features.HasSSE3, features.HasAVX2, features.HasAVX512, features.HasNEON,
	)

	return wisdom.LookupWisdom(n, precision, cpuFeatures)
}

// bindWisdomCodelet binds the codelet whose signature wisdom named, or returns
// nil if the name is not a registered codelet for this size, the codelet is
// disabled, the CPU cannot run it, or it conflicts with a forced strategy.
//
// Wisdom entries can originate from machines with different CPU features
// (e.g. imported wisdom, or FMA masked off under a VM while the feature mask
// still matches on AVX2), so re-check CPUSupports before binding —
// LookupBySignature itself does not filter by CPU features.
func bindWisdomCodelet[T Complex](
	n int, features cpu.Features, algorithm string, forcedStrategy KernelStrategy,
) *PlanEstimate[T] {
	reg := registry.GetRegistry[T]()
	if reg == nil {
		return nil
	}

	codelet := reg.LookupBySignature(n, algorithm)
	if codelet == nil || !registry.CPUSupports(features, codelet.SIMDLevel) {
		return nil
	}

	if forcedStrategy != KernelAuto && codelet.Algorithm != forcedStrategy {
		return nil
	}

	return &PlanEstimate[T]{
		ForwardCodelet: codelet.Forward,
		InverseCodelet: codelet.Inverse,
		Algorithm:      codelet.Signature,
		Strategy:       codelet.Algorithm,
		TwiddleSize:    codelet.TwiddleSize,
		PrepareTwiddle: codelet.PrepareTwiddle,
	}
}

// wisdomStrategy resolves a wisdom algorithm name to a kernel strategy. It
// reports false for codelet signatures (which bindWisdomCodelet handles) and
// for names that conflict with a forced strategy.
func wisdomStrategy(algorithm string, forcedStrategy KernelStrategy) (KernelStrategy, bool) {
	strategy, ok := AlgorithmNameToStrategy(algorithm)
	if !ok {
		return KernelAuto, false
	}

	if forcedStrategy != KernelAuto && strategy != forcedStrategy {
		return KernelAuto, false
	}

	return strategy, true
}

// HasCodelet returns true if a codelet is available for the given size.
func HasCodelet[T Complex](n int, features cpu.Features) bool {
	reg := registry.GetRegistry[T]()
	if reg == nil {
		return false
	}

	return reg.Lookup(n, features) != nil
}
