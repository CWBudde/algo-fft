package fft

import (
	"runtime"
	"slices"
	"sort"
	"time"

	"github.com/cwbudde/algo-fft/internal/cpu"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
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
	iters  int // Number of benchmark iterations per trial
	trials int // Number of trials; the median trial is used to reject outliers
}

// getMeasureConfig returns the benchmarking configuration for a given mode.
// Each strategy is timed over `trials` independent trials of `iters` iterations
// and the median trial time is kept, so a single noisy trial (GC pause, migration)
// cannot skew the decision — important at small sizes where a transform is only a
// few hundred nanoseconds.
func getMeasureConfig(mode PlannerMode) measureConfig {
	switch mode {
	case PlannerEstimate:
		// Estimate mode uses heuristics, not benchmarking.
		// Return minimal config as fallback if called.
		return measureConfig{warmup: 3, iters: 30, trials: 5}
	case PlannerMeasure:
		return measureConfig{warmup: 5, iters: 30, trials: 5}
	case PlannerPatient:
		return measureConfig{warmup: 5, iters: 50, trials: 7}
	case PlannerExhaustive:
		return measureConfig{warmup: 10, iters: 100, trials: 9}
	}

	return measureConfig{warmup: 3, iters: 30, trials: 5}
}

// selectStrategiesToTest returns the strategies to benchmark based on planner mode.
func selectStrategiesToTest(mode PlannerMode, n int) []KernelStrategy {
	// For non-power-of-two sizes, only Bluestein is available
	if !m.IsPowerOf2(n) && !m.IsHighlyComposite(n) {
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
		// Moderate: add SixStep and SplitRadix for larger sizes
		return []KernelStrategy{KernelDIT, KernelStockham, KernelSixStep, KernelSplitRadix}
	case PlannerExhaustive:
		// Thorough: test all power-of-two strategies
		return []KernelStrategy{KernelDIT, KernelStockham, KernelSixStep, KernelEightStep, KernelSplitRadix}
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
				Algorithm: planner.StrategyToAlgorithmName(strategy),
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
	recordToWisdom[T](n, features, wisdom, best.Algorithm)

	return estimateWithStrategy[T](n, features, best.Strategy)
}

func recordToWisdom[T Complex](n int, features cpu.Features, wisdom WisdomRecorder, algorithm string) {
	if wisdom == nil {
		return
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

	cpuMask := CPUFeatureMask(
		features.HasSSE2,
		features.HasSSE3,
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
		Algorithm: algorithm,
		Timestamp: time.Now(),
	}
	wisdom.Store(entry)
}

// benchmarkStrategy runs a micro-benchmark for a single strategy.
// It runs config.trials independent trials of config.iters iterations each and
// returns the median trial's elapsed time, rejecting outlier trials. Returns 0 if
// the strategy is not implemented for this size.
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

	// Initialize source with simple pattern (avoids random number generation)
	for i := range src {
		src[i] = complexFromFloat64[T](float64(i%16)/16.0, float64((i+1)%16)/16.0)
	}

	// Get kernel for this strategy
	kernels := SelectKernelsWithStrategy[T](features, strategy)

	// Warmup: verify the kernel works and warm up CPU caches
	for range config.warmup {
		ok := kernels.Forward(dst, src, twiddle, scratch)
		if !ok {
			return 0 // Strategy not implemented
		}
	}

	trials := max(config.trials, 1)

	samples := make([]int64, 0, trials)

	for range trials {
		// Force GC before each trial to reduce noise from unrelated allocations.
		runtime.GC()

		startCycles := cpu.ReadCycleCounter()

		for range config.iters {
			kernels.Forward(dst, src, twiddle, scratch)
		}

		elapsedNanos := cpu.CyclesToNanoseconds(cpu.CyclesSince(startCycles))
		if elapsedNanos <= 0 {
			// Should never happen with cycle counters, but handle gracefully.
			elapsedNanos = 1
		}

		samples = append(samples, elapsedNanos)
	}

	return time.Duration(medianInt64(samples))
}

// medianInt64 returns the median of a non-empty slice. For an even count it
// returns the lower of the two middle values, which is sufficient for outlier
// rejection here and avoids overflow from averaging.
func medianInt64(samples []int64) int64 {
	slices.Sort(samples)

	return samples[(len(samples)-1)/2]
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
		Algorithm:      planner.StrategyToAlgorithmName(strategy),
	}
}
