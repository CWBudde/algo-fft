package fft

import (
	"runtime"
	"slices"
	"time"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	mem "github.com/cwbudde/algo-fft/internal/memory"
	"github.com/cwbudde/algo-fft/internal/planner"
	"github.com/cwbudde/algo-fft/internal/registry"
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

// WisdomRecorder extends planner.WisdomStore with the ability to record new entries.
// This interface allows the planner to save benchmark results.
type WisdomRecorder interface {
	planner.WisdomStore
	Store(entry planner.WisdomEntry)
}

// MeasureResult holds the benchmark result for a single strategy.
type MeasureResult struct {
	Strategy  fftypes.KernelStrategy
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
func selectStrategiesToTest(mode PlannerMode, n int) []fftypes.KernelStrategy {
	// Non-power-of-two sizes have exactly one route each: the mixed-radix
	// engine when it can execute the length, Bluestein otherwise. Timing the
	// power-of-two strategies here would time the same mixed-radix transform
	// once per candidate and record the winner under a name that never runs.
	if !mathpkg.IsPowerOf2(n) {
		if planner.MixedRadixEligible(n) {
			return []fftypes.KernelStrategy{fftypes.KernelMixedRadix}
		}

		return []fftypes.KernelStrategy{fftypes.KernelBluestein}
	}

	switch mode {
	case PlannerEstimate:
		// Estimate mode doesn't benchmark, but return default if called
		return []fftypes.KernelStrategy{fftypes.KernelDIT, fftypes.KernelStockham}
	case PlannerMeasure:
		// Quick: test the two most common strategies
		return []fftypes.KernelStrategy{fftypes.KernelDIT, fftypes.KernelStockham}
	case PlannerPatient:
		// Moderate: add SixStep, SplitRadix, and FourStep for larger sizes
		return []fftypes.KernelStrategy{
			fftypes.KernelDIT, fftypes.KernelStockham, fftypes.KernelSixStep,
			fftypes.KernelSplitRadix, fftypes.KernelFourStep,
		}
	case PlannerExhaustive:
		// Thorough: test all power-of-two strategies
		return []fftypes.KernelStrategy{
			fftypes.KernelDIT, fftypes.KernelStockham, fftypes.KernelSixStep,
			fftypes.KernelSplitRadix, fftypes.KernelFourStep,
		}
	}

	return []fftypes.KernelStrategy{fftypes.KernelDIT, fftypes.KernelStockham}
}

// measureCandidate is one implementation the planner can time: either a kernel
// strategy, or a specific codelet from the registry (entry non-nil).
type measureCandidate[T Complex] struct {
	entry     *registry.CodeletEntry[T]
	algorithm string
	strategy  fftypes.KernelStrategy
}

// MeasureAndSelect benchmarks candidate implementations in both directions.
// It binds separate winners when they belong to the same strategy family;
// otherwise it preserves the forward winner for both directions so the plan's
// singular strategy remains truthful. It optionally records the result to the
// provided wisdom recorder.
//
// Codelets are timed alongside the kernel strategies, not assumed. Timing only
// the strategies used to let measure mode return a plan worse than
// PlannerEstimate's: the strategy winner was applied through
// estimateWithStrategy, which substitutes a codelet only when the winning
// strategy happens to match the top-ranked codelet's own algorithm, so at
// complex64 n = 1024 (Stockham kernel beats DIT kernel) measurement discarded
// the dit1024_radix4_avx2 codelet that the unmeasured path would have used.
//
// Because codelets now take part, the winner fully determines the returned
// estimate — a strategy that wins is honored as a kernel plan even where a
// codelet exists, since the codelet was measured and lost.
func MeasureAndSelect[T Complex](
	n int,
	features cpu.Features,
	mode PlannerMode,
	wisdom WisdomRecorder,
	forcedStrategy fftypes.KernelStrategy,
) planner.PlanEstimate[T] {
	// If a specific strategy is forced, skip benchmarking
	if forcedStrategy != fftypes.KernelAuto {
		return estimateWithStrategy[T](n, features, forcedStrategy)
	}

	strategies := selectStrategiesToTest(mode, n)
	if len(strategies) == 0 {
		return estimateWithStrategy[T](n, features, fftypes.KernelAuto)
	}

	candidates := measureCandidates[T](mode, n, features, strategies)

	// Nothing to compare (a single strategy and no codelet, e.g. Bluestein):
	// applying it directly avoids the benchmarking cost.
	if len(candidates) == 1 {
		return estimateWithStrategy[T](n, features, strategies[0])
	}

	config := getMeasureConfig(mode)

	bestForward := -1
	bestForwardNsPerOp := 0.0
	bestInverse := -1
	bestInverseNsPerOp := 0.0

	for i := range candidates {
		forwardElapsed := benchmarkCandidateDirection(n, features, candidates[i], config, false)
		if forwardElapsed > 0 {
			nsPerOp := float64(forwardElapsed.Nanoseconds()) / float64(config.iters)
			if bestForward < 0 || nsPerOp < bestForwardNsPerOp {
				bestForward, bestForwardNsPerOp = i, nsPerOp
			}
		}

		inverseElapsed := benchmarkCandidateDirection(n, features, candidates[i], config, true)
		if inverseElapsed > 0 {
			nsPerOp := float64(inverseElapsed.Nanoseconds()) / float64(config.iters)
			if bestInverse < 0 || nsPerOp < bestInverseNsPerOp {
				bestInverse, bestInverseNsPerOp = i, nsPerOp
			}
		}
	}

	// If no candidate succeeded, fall back to heuristics
	if bestForward < 0 {
		return planner.EstimatePlan[T](n, features, nil, fftypes.KernelAuto)
	}

	forwardWinner := candidates[bestForward]
	inverseWinner := forwardWinner
	if bestInverse >= 0 && candidates[bestInverse].strategy == forwardWinner.strategy {
		inverseWinner = candidates[bestInverse]
	}

	algorithm := planner.DirectionalAlgorithm(forwardWinner.algorithm, inverseWinner.algorithm)

	// Record to wisdom if recorder is provided. The recorded name is the
	// codelet signature (or direction pair) when a codelet won, which
	// planner.EstimatePlan binds directly on replay.
	recordToWisdom[T](n, features, wisdom, algorithm)

	return directionalEstimate(forwardWinner, inverseWinner)
}

// measureCandidates builds the candidate list: one entry per kernel strategy
// plus the codelets worth timing for this size.
func measureCandidates[T Complex](
	mode PlannerMode, n int, features cpu.Features, strategies []fftypes.KernelStrategy,
) []measureCandidate[T] {
	codelets := codeletCandidates[T](mode, n, features)
	candidates := make([]measureCandidate[T], 0, len(strategies)+len(codelets))

	for _, strategy := range strategies {
		candidates = append(candidates, measureCandidate[T]{
			entry:     nil,
			algorithm: planner.StrategyToAlgorithmName(strategy),
			strategy:  strategy,
		})
	}

	for i := range codelets {
		candidates = append(candidates, measureCandidate[T]{
			entry:     &codelets[i],
			algorithm: codelets[i].Signature,
			strategy:  codelets[i].Algorithm,
		})
	}

	return candidates
}

// codeletCandidates returns the codelets to time for this size. The quick mode
// times only the registry's own winner — the one the plan would otherwise take
// unmeasured — while the deeper modes time every enabled codelet the CPU can
// run, which is what lets measurement disagree with the registry's static
// priority order and record the disagreement as wisdom.
func codeletCandidates[T Complex](mode PlannerMode, n int, features cpu.Features) []registry.CodeletEntry[T] {
	reg := registry.GetRegistry[T]()
	if reg == nil {
		return nil
	}

	if mode != PlannerPatient && mode != PlannerExhaustive {
		entry := reg.Lookup(n, features)
		if entry == nil {
			return nil
		}

		return []registry.CodeletEntry[T]{*entry}
	}

	all := reg.GetAllForSize(n)
	out := make([]registry.CodeletEntry[T], 0, len(all))

	for i := range all {
		// Mirror Lookup: disabled codelets stay disabled, and a codelet the CPU
		// cannot run would fault rather than report a time.
		if all[i].Priority < 0 || !registry.CPUSupports(features, all[i].SIMDLevel) {
			continue
		}

		out = append(out, all[i])
	}

	return out
}

// candidateForward returns the forward transform to time for a candidate, plus
// the twiddle table to feed it. A codelet that wants a packed twiddle layout
// gets one prepared here exactly as the plan would; handed the plain table it
// would fail its own length check and report "not implemented" instead of a
// time.
func candidateForward[T Complex](
	n int, features cpu.Features, cand measureCandidate[T], twiddle []T,
) (kernels.Kernel[T], []T) {
	return candidateTransform(n, features, cand, twiddle, false)
}

func candidateTransform[T Complex](
	n int, features cpu.Features, cand measureCandidate[T], twiddle []T, inverse bool,
) (kernels.Kernel[T], []T) {
	if cand.entry == nil {
		selected := SelectKernelsWithStrategy[T](features, cand.strategy)
		if inverse {
			return selected.Inverse, twiddle
		}

		return selected.Forward, twiddle
	}

	if cand.entry.TwiddleSize != nil && cand.entry.PrepareTwiddle != nil {
		if size := cand.entry.TwiddleSize(n); size > 0 {
			packed, _ := mem.AllocAligned[T](size)
			cand.entry.PrepareTwiddle(n, inverse, packed)
			twiddle = packed
		}
	}

	if inverse {
		return kernels.Kernel[T](cand.entry.Inverse), twiddle
	}

	return kernels.Kernel[T](cand.entry.Forward), twiddle
}

// codeletEstimate builds a plan estimate bound directly to one codelet.
func codeletEstimate[T Complex](entry registry.CodeletEntry[T]) planner.PlanEstimate[T] {
	return planner.PlanEstimate[T]{
		ForwardCodelet:        entry.Forward,
		InverseCodelet:        entry.Inverse,
		Algorithm:             entry.Signature,
		ForwardAlgorithm:      entry.Signature,
		InverseAlgorithm:      entry.Signature,
		Strategy:              entry.Algorithm,
		ForwardTwiddleSize:    entry.TwiddleSize,
		ForwardPrepareTwiddle: entry.PrepareTwiddle,
		InverseTwiddleSize:    entry.TwiddleSize,
		InversePrepareTwiddle: entry.PrepareTwiddle,
	}
}

func directionalEstimate[T Complex](
	forward, inverse measureCandidate[T],
) planner.PlanEstimate[T] {
	estimate := planner.PlanEstimate[T]{
		Algorithm:        planner.DirectionalAlgorithm(forward.algorithm, inverse.algorithm),
		ForwardAlgorithm: forward.algorithm,
		InverseAlgorithm: inverse.algorithm,
		Strategy:         forward.strategy,
	}

	if forward.entry != nil {
		estimate.ForwardCodelet = forward.entry.Forward
		estimate.ForwardTwiddleSize = forward.entry.TwiddleSize
		estimate.ForwardPrepareTwiddle = forward.entry.PrepareTwiddle
	}

	if inverse.entry != nil {
		estimate.InverseCodelet = inverse.entry.Inverse
		estimate.InverseTwiddleSize = inverse.entry.TwiddleSize
		estimate.InversePrepareTwiddle = inverse.entry.PrepareTwiddle
	}

	return estimate
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
		precision = planner.PrecisionComplex64
	case complex128:
		precision = planner.PrecisionComplex128
	}

	cpuMask := planner.CPUFeatureMask(
		features.HasSSE2,
		features.HasSSE3,
		features.HasAVX2,
		features.HasAVX512,
		features.HasNEON,
	)

	cpuIdentifier := ""
	if _, ok := wisdom.(planner.MicroarchitectureWisdomStore); ok {
		cpuIdentifier = cpu.WisdomCPUIdentifier(features)
	}

	entry := planner.WisdomEntry{
		Key: planner.WisdomKey{
			Size:          n,
			Precision:     precision,
			CPUFeatures:   cpuMask,
			CPUIdentifier: cpuIdentifier,
		},
		Algorithm: algorithm,
		Timestamp: time.Now(),
	}
	wisdom.Store(entry)
}

// benchmarkCandidate runs a micro-benchmark for a single candidate (kernel
// strategy or codelet). It runs config.trials independent trials of
// config.iters iterations each and returns the median trial's elapsed time,
// rejecting outlier trials. Returns 0 if the candidate declines this size.
func benchmarkCandidate[T Complex](
	n int,
	features cpu.Features,
	cand measureCandidate[T],
	config measureConfig,
) time.Duration {
	return benchmarkCandidateDirection(n, features, cand, config, false)
}

func benchmarkCandidateDirection[T Complex](
	n int,
	features cpu.Features,
	cand measureCandidate[T],
	config measureConfig,
	inverse bool,
) time.Duration {
	// Prepare data buffers
	src := make([]T, n)
	dst := make([]T, n)
	twiddle := mathpkg.ComputeTwiddleFactors[T](n)
	scratch := make([]T, n)

	// Initialize source with simple pattern (avoids random number generation)
	for i := range src {
		src[i] = complexFromFloat64[T](float64(i%16)/16.0, float64((i+1)%16)/16.0)
	}

	transform, twiddle := candidateTransform(n, features, cand, twiddle, inverse)

	// Warmup: verify the candidate works and warm up CPU caches
	for range config.warmup {
		ok := transform(dst, src, twiddle, scratch)
		if !ok {
			return 0 // Not implemented for this size
		}
	}

	trials := max(config.trials, 1)

	samples := make([]int64, 0, trials)

	for range trials {
		// Force GC before each trial to reduce noise from unrelated allocations.
		runtime.GC()

		startCycles := cpu.ReadCycleCounter()

		for range config.iters {
			transform(dst, src, twiddle, scratch)
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

// estimateWithStrategy creates a planner.PlanEstimate for a specific strategy.
func estimateWithStrategy[T Complex](
	n int,
	features cpu.Features,
	strategy fftypes.KernelStrategy,
) planner.PlanEstimate[T] {
	// Check for codelets first
	registry := registry.GetRegistry[T]()
	if registry != nil {
		entry := registry.Lookup(n, features)
		if entry != nil && (strategy == fftypes.KernelAuto || entry.Algorithm == strategy) {
			// The twiddle callbacks must be carried over: a codelet that wants a
			// packed layout (n = 256/1024/8192) is handed the plain twiddle table
			// without them, bails on its length check, and silently runs the
			// fallback kernel while the plan still reports the codelet signature.
			return planner.PlanEstimate[T]{
				ForwardCodelet:        entry.Forward,
				InverseCodelet:        entry.Inverse,
				Algorithm:             entry.Signature,
				ForwardAlgorithm:      entry.Signature,
				InverseAlgorithm:      entry.Signature,
				Strategy:              entry.Algorithm,
				ForwardTwiddleSize:    entry.TwiddleSize,
				ForwardPrepareTwiddle: entry.PrepareTwiddle,
				InverseTwiddleSize:    entry.TwiddleSize,
				InversePrepareTwiddle: entry.PrepareTwiddle,
			}
		}
	}

	// Fall back to kernel-based estimate. Resolving (rather than adopting the
	// forced value verbatim) keeps the reported strategy equal to the route the
	// dispatch takes — a forced power-of-two strategy at a mixed-radix length
	// resolves to KernelMixedRadix, which is what actually runs.
	strategy = planner.ResolveKernelStrategyWithDefault(n, strategy)

	return planner.PlanEstimate[T]{
		ForwardCodelet:   nil,
		InverseCodelet:   nil,
		Strategy:         strategy,
		Algorithm:        planner.StrategyToAlgorithmName(strategy),
		ForwardAlgorithm: planner.StrategyToAlgorithmName(strategy),
		InverseAlgorithm: planner.StrategyToAlgorithmName(strategy),
	}
}
