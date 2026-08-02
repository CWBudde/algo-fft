package algofft

import "time"

// PlannerMode controls how much work the planner does to choose kernels.
//
// The planner modes form a hierarchy of increasing thoroughness:
//   - PlannerEstimate: Use heuristics only (fast, no benchmarking)
//   - PlannerMeasure: Quick benchmark testing DIT and Stockham strategies
//   - PlannerPatient: Moderate benchmark including SixStep
//   - PlannerExhaustive: Thorough benchmark testing all strategies
//
// The measuring modes benchmark size-specific codelets alongside the kernel
// strategies, so the plan they choose is never worse than PlannerEstimate's
// without having measured it to be better. PlannerMeasure times the codelet
// the estimate would have used; the deeper modes time every codelet available
// for the size, which is what lets them disagree with the built-in preference
// order — at roughly double the planning time of strategies alone.
//
// When using PlannerMeasure or higher with a WisdomStore, the planner
// automatically records the winner for future plan creations.
type PlannerMode uint8

const (
	// PlannerEstimate uses heuristics to select the kernel strategy.
	// This is the fastest mode and suitable for most use cases.
	PlannerEstimate PlannerMode = iota

	// PlannerMeasure runs quick micro-benchmarks (warmup=3, iters=10)
	// testing DIT and Stockham strategies to find the faster one.
	PlannerMeasure

	// PlannerPatient runs moderate micro-benchmarks (warmup=5, iters=50)
	// testing DIT, Stockham, and SixStep strategies.
	PlannerPatient

	// PlannerExhaustive runs thorough micro-benchmarks (warmup=10, iters=100)
	// testing all available power-of-two strategies.
	PlannerExhaustive
)

// PlanOptions controls planning decisions. All options are plan-time
// concerns; per-call execution layout (batching, striding) is expressed at
// the call site via ForwardBatch/InverseBatch and ForwardStrided/
// InverseStrided instead.
type PlanOptions struct {
	// Planner controls how much work the planner does to choose kernels.
	// Default is PlannerEstimate (heuristics only, no benchmarking).
	Planner PlannerMode

	// Strategy forces a specific kernel strategy. Use KernelAuto (default)
	// to let the planner choose based on size and benchmarks.
	Strategy KernelStrategy

	// Wisdom provides a cache for storing and retrieving optimal kernel choices.
	// When using PlannerMeasure or higher, benchmark results are automatically
	// stored to this cache. When creating plans, cached decisions are used
	// to skip benchmarking for previously-measured sizes.
	//
	// A wisdom entry overrides the built-in preference order for its size,
	// precision and CPU feature set — naming a codelet signature (e.g.
	// "dit64_radix4_sse2") is the way to pin one codelet against another, and
	// naming a kernel strategy (e.g. "stockham") selects that strategy even
	// where a codelet exists. An entry is ignored when Strategy forces a
	// conflicting strategy, or when it names a codelet that has since been
	// disabled or that this CPU cannot run.
	Wisdom WisdomStore
}

// WisdomStore persists planner decisions for reuse.
// This interface allows saving and reusing optimal kernel choices across program runs.
type WisdomStore interface {
	// LookupWisdom returns the algorithm name for a given FFT configuration.
	// Returns empty string and false if no wisdom is available.
	LookupWisdom(size int, precision uint8, cpuFeatures uint64) (algorithm string, found bool)

	// Lookup returns the full wisdom entry for a given key (for advanced usage).
	Lookup(key WisdomKey) (WisdomEntry, bool)

	// Store saves a planning decision to the wisdom cache.
	Store(entry WisdomEntry)
}

// WisdomKey identifies a planning context for wisdom lookup.
type WisdomKey struct {
	Size        int    // FFT size
	Precision   uint8  // 0 = complex64, 1 = complex128
	CPUFeatures uint64 // Bitmask of CPU features
}

// WisdomEntry stores a planning decision.
type WisdomEntry struct {
	Key       WisdomKey
	Algorithm string    // e.g., "dit64_generic", "stockham"
	Timestamp time.Time // When this entry was recorded
}

// PrecisionKind describes the precision for a plan.
type PrecisionKind uint8

const (
	PrecisionComplex64  PrecisionKind = iota // complex64 (float32 parts)
	PrecisionComplex128                      // complex128 (float64 parts)
)

func normalizePlanOptions(opts PlanOptions) PlanOptions {
	// Default planner mode when unset
	if opts.Planner == 0 {
		opts.Planner = PlannerEstimate
	}

	return opts
}
