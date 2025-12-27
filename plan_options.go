package algoforge

import "time"

// PlannerMode controls how much work the planner does to choose kernels.
type PlannerMode uint8

const (
	PlannerEstimate PlannerMode = iota
	PlannerMeasure
	PlannerPatient
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
	Planner   PlannerMode
	Strategy  KernelStrategy
	Radices   []int
	Batch     int
	Stride    int
	InPlace   bool
	Wisdom    WisdomStore     // Not yet implemented - reserved for future wisdom/caching feature
	Workspace WorkspacePolicy // Not yet implemented - reserved for future workspace management
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
