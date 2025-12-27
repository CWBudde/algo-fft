package algoforge

import "time"

// PlannerMode controls how much work the planner does to choose kernels.
//
// The planner modes form a hierarchy of increasing thoroughness:
//   - PlannerEstimate: Use heuristics only (fast, no benchmarking)
//   - PlannerMeasure: Quick benchmark testing DIT and Stockham strategies
//   - PlannerPatient: Moderate benchmark including SixStep
//   - PlannerExhaustive: Thorough benchmark testing all strategies
//
// When using PlannerMeasure or higher with a WisdomStore, the planner
// automatically records benchmark results for future plan creations.
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
	// testing all available strategies including EightStep.
	PlannerExhaustive
)

// WorkspacePolicy controls how executors manage scratch space.
// Note: This feature is not yet implemented and will be added in a future release.
type WorkspacePolicy uint8

const (
	WorkspaceAuto     WorkspacePolicy = iota // Not yet implemented
	WorkspacePooled                          // Not yet implemented
	WorkspaceExternal                        // Not yet implemented
)

// PlanOptions controls planning decisions and execution layout.
type PlanOptions struct {
	// Planner controls how much work the planner does to choose kernels.
	// Default is PlannerEstimate (heuristics only, no benchmarking).
	Planner PlannerMode

	// Strategy forces a specific kernel strategy. Use KernelAuto (default)
	// to let the planner choose based on size and benchmarks.
	Strategy KernelStrategy

	// Radices hints at which radices to prefer for mixed-radix FFT.
	Radices []int

	// Batch specifies the number of transforms to execute in a batch.
	Batch int

	// Stride specifies the stride between consecutive elements.
	Stride int

	// InPlace enables in-place transforms when possible.
	InPlace bool

	// Wisdom provides a cache for storing and retrieving optimal kernel choices.
	// When using PlannerMeasure or higher, benchmark results are automatically
	// stored to this cache. When creating plans, cached decisions are used
	// to skip benchmarking for previously-measured sizes.
	Wisdom WisdomStore

	// Workspace controls how executors manage scratch space.
	// Note: This feature is not yet implemented.
	Workspace WorkspacePolicy
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
	PrecisionComplex64 PrecisionKind = iota
	PrecisionComplex128
)

func normalizePlanOptions(opts PlanOptions) PlanOptions {
	// Default planner mode when unset
	if opts.Planner == 0 {
		opts.Planner = PlannerEstimate
	}

	// Ensure batch count is sensible: default to a single transform
	if opts.Batch < 0 {
		opts.Batch = 0 // 0 is treated as 1 in resolveBatchStride
	}

	// Normalize stride: negative values are treated as default (contiguous)
	if opts.Stride < 0 {
		opts.Stride = 0 // 0 means use default stride
	}

	// Normalize radices: drop invalid entries (<= 1)
	// If none remain, fall back to planner defaults by clearing the slice
	if len(opts.Radices) > 0 {
		validRadices := opts.Radices[:0]
		for _, r := range opts.Radices {
			if r > 1 {
				validRadices = append(validRadices, r)
			}
		}

		if len(validRadices) == 0 {
			opts.Radices = nil
		} else {
			opts.Radices = validRadices
		}
	}

	return opts
}
