package algofft

// PlanMeta captures planner decisions for introspection.
type PlanMeta struct {
	Planner  PlannerMode
	Strategy KernelStrategy
	Batch    int
	Stride   int
	InPlace  bool
}

// Meta returns metadata about how the plan was constructed.
func (p *Plan[T]) Meta() PlanMeta {
	return p.meta
}
