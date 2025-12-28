package fft

import (
	"runtime"
	"sort"
	"time"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

// PlannerMode controls how much work the planner does to choose kernels.
// This mirrors the public PlannerMode type from the root package.
type PlannerMode uint8

const (
	PlannerEstimate   PlannerMode = iota // Use heuristics only (fast, no benchmarking)
	PlannerMeasure                       // Quick benchmark: test DIT and Stockham
	PlannerPatient                       // Moderate benchmark: test common strategies
	PlannerExhaustive                    // Thorough benchmark: test all strategies
)

// WisdomRecorder extends WisdomStore with the ability to record new entries.
// This interface allows the planner to save benchmark results.
type WisdomRecorder interface {
	WisdomStore
	Store(entry WisdomEntry)
}

// MeasureResult holds the benchmark result for a single strategy.
type MeasureResult struct {
	Strategy  KernelStrategy
	Algorithm string
	NsPerOp   float64
}

// measureConfig holds configuration for benchmarking based on planner mode.
type measureConfig struct {
	warmup int // Number of warmup iterations
	iters  int // Number of benchmark iterations
}

// getMeasureConfig returns the benchmarking configuration for a given mode.
func getMeasureConfig(mode PlannerMode) measureConfig {
	switch mode {
	case PlannerEstimate:
		// Estimate mode uses heuristics, not benchmarking.
		// Return minimal config as fallback if called.
		return measureConfig{warmup: 3, iters: 10}
	case PlannerMeasure:
		return measureConfig{warmup: 3, iters: 10}
	case PlannerPatient:
		return measureConfig{warmup: 5, iters: 50}
	case PlannerExhaustive:
		return measureConfig{warmup: 10, iters: 100}
	}

	return measureConfig{warmup: 3, iters: 10}
}

// selectStrategiesToTest returns the strategies to benchmark based on planner mode.
func selectStrategiesToTest(mode PlannerMode, n int) []KernelStrategy {
	// For non-power-of-two sizes, only Bluestein is available
	if !IsPowerOf2(n) && !IsHighlyComposite(n) {
		return []KernelStrategy{KernelBluestein}
	}

	switch mode {
	case PlannerEstimate:
		// Estimate mode doesn't benchmark, but return default if called
		return []KernelStrategy{KernelDIT, KernelStockham}
	case PlannerMeasure:
		// Quick: test the two most common strategies
		return []KernelStrategy{KernelDIT, KernelStockham}
	case PlannerPatient:
		// Moderate: add SixStep for larger sizes
		return []KernelStrategy{KernelDIT, KernelStockham, KernelSixStep}
	case PlannerExhaustive:
		// Thorough: test all power-of-two strategies
		return []KernelStrategy{KernelDIT, KernelStockham, KernelSixStep, KernelEightStep}
	}

	return []KernelStrategy{KernelDIT, KernelStockham}
}

// MeasureAndSelect benchmarks multiple strategies and returns the best one.
// It optionally records the result to the provided wisdom recorder.
func MeasureAndSelect[T Complex](
	n int,
	features cpu.Features,
	mode PlannerMode,
	wisdom WisdomRecorder,
	forcedStrategy KernelStrategy,
) PlanEstimate[T] {
	// If a specific strategy is forced, skip benchmarking
	if forcedStrategy != KernelAuto {
		return estimateWithStrategy[T](n, features, forcedStrategy)
	}

	strategies := selectStrategiesToTest(mode, n)
	if len(strategies) == 0 {
		return estimateWithStrategy[T](n, features, KernelAuto)
	}

	// Single strategy? Just use it directly
	if len(strategies) == 1 {
		return estimateWithStrategy[T](n, features, strategies[0])
	}

	config := getMeasureConfig(mode)
	results := make([]MeasureResult, 0, len(strategies))

	for _, strategy := range strategies {
		elapsed := benchmarkStrategy[T](n, features, strategy, config)
		if elapsed > 0 {
			results = append(results, MeasureResult{
				Strategy:  strategy,
				Algorithm: strategyToAlgorithmName(strategy),
				NsPerOp:   float64(elapsed.Nanoseconds()) / float64(config.iters),
			})
		}
	}

	// If no strategy succeeded, fall back to heuristics
	if len(results) == 0 {
		return EstimatePlan[T](n, features, nil, KernelAuto)
	}

	// Sort by performance (fastest first)
	sort.Slice(results, func(i, j int) bool {
		return results[i].NsPerOp < results[j].NsPerOp
	})

	best := results[0]

	// Record to wisdom if recorder is provided
	if wisdom != nil {
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

		cpuMask := CPUFeatureMask(
			features.HasSSE2,
			features.HasAVX2,
			features.HasAVX512,
			features.HasNEON,
		)

		entry := WisdomEntry{
			Key: WisdomKey{
				Size:        n,
				Precision:   precision,
				CPUFeatures: cpuMask,
			},
			Algorithm: best.Algorithm,
			Timestamp: time.Now(),
		}
		wisdom.Store(entry)
	}

	return estimateWithStrategy[T](n, features, best.Strategy)
}

// benchmarkStrategy runs a micro-benchmark for a single strategy.
// Returns the total elapsed time for config.iters iterations, or 0 if the strategy failed.
func benchmarkStrategy[T Complex](
	n int,
	features cpu.Features,
	strategy KernelStrategy,
	config measureConfig,
) time.Duration {
	// Prepare data buffers
	src := make([]T, n)
	dst := make([]T, n)
	twiddle := ComputeTwiddleFactors[T](n)
	scratch := make([]T, n)
	bitrev := ComputeBitReversalIndices(n)

	// Initialize source with simple pattern (avoids random number generation)
	for i := range src {
		src[i] = complexFromFloat64[T](float64(i%16)/16.0, float64((i+1)%16)/16.0)
	}

	// Get kernel for this strategy
	kernels := SelectKernelsWithStrategy[T](features, strategy)

	// Warmup: verify the kernel works and warm up CPU caches
	for range config.warmup {
		ok := kernels.Forward(dst, src, twiddle, scratch, bitrev)
		if !ok {
			return 0 // Strategy not implemented
		}
	}

	// Force GC before timing to reduce noise
	runtime.GC()

	// Benchmark
	start := time.Now()

	for range config.iters {
		kernels.Forward(dst, src, twiddle, scratch, bitrev)
	}

	return time.Since(start)
}

// estimateWithStrategy creates a PlanEstimate for a specific strategy.
func estimateWithStrategy[T Complex](
	n int,
	features cpu.Features,
	strategy KernelStrategy,
) PlanEstimate[T] {
	// Check for codelets first
	registry := GetRegistry[T]()
	if registry != nil {
		entry := registry.Lookup(n, features)
		if entry != nil && (strategy == KernelAuto || entry.Algorithm == strategy) {
			return PlanEstimate[T]{
				ForwardCodelet: entry.Forward,
				InverseCodelet: entry.Inverse,
				Algorithm:      entry.Signature,
				Strategy:       entry.Algorithm,
			}
		}
	}

	// Fall back to kernel-based estimate
	if strategy == KernelAuto {
		strategy = ResolveKernelStrategy(n)
	}

	return PlanEstimate[T]{
		ForwardCodelet: nil,
		InverseCodelet: nil,
		Strategy:       strategy,
		Algorithm:      strategyToAlgorithmName(strategy),
	}
}
