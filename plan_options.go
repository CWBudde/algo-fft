package algoforge

// PlannerMode controls how much work the planner does to choose kernels.
type PlannerMode uint8

const (
	PlannerEstimate PlannerMode = iota
	PlannerMeasure
	PlannerPatient
	PlannerExhaustive
)

// WorkspacePolicy controls how executors manage scratch space.
type WorkspacePolicy uint8

const (
	WorkspaceAuto WorkspacePolicy = iota
	WorkspacePooled
	WorkspaceExternal
)

// PlanOptions controls planning decisions and execution layout.
type PlanOptions struct {
	Planner   PlannerMode
	Strategy  KernelStrategy
	Radices   []int
	Batch     int
	Stride    int
	InPlace   bool
	Wisdom    WisdomStore
	Workspace WorkspacePolicy
}

// WisdomStore persists planner decisions for reuse.
type WisdomStore interface {
	Lookup(key WisdomKey) (KernelStrategy, bool)
	Record(key WisdomKey, strategy KernelStrategy)
}

// WisdomKey identifies a planning context for wisdom lookup.
type WisdomKey struct {
	N         int
	Precision PrecisionKind
}

// PrecisionKind describes the precision for a plan.
type PrecisionKind uint8

const (
	PrecisionComplex64 PrecisionKind = iota
	PrecisionComplex128
)

func normalizePlanOptions(opts PlanOptions) PlanOptions {
	if opts.Planner == 0 {
		opts.Planner = PlannerEstimate
	}

	return opts
}
