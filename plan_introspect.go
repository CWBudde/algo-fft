package algofft

// This file provides introspection accessors for the plan types beyond the 1D
// Plan[T] (whose Meta/KernelStrategy/Algorithm live in plan.go and
// plan_meta.go). Multi-dimensional plans run an independent 1D plan per axis,
// so they expose per-axis KernelStrategies/Algorithms instead of the singular
// accessors; real plans delegate to their underlying half-size complex plan.

// metaFromOptions builds a PlanMeta view of the options a composite plan was
// constructed with. Unlike the 1D Plan's Meta (whose Strategy is the resolved
// strategy of its single kernel), Strategy here is the requested
// PlanOptions.Strategy — composite plans resolve one strategy per axis, which
// KernelStrategies reports.
func metaFromOptions(opts PlanOptions) PlanMeta {
	return PlanMeta{
		Planner:  opts.Planner,
		Strategy: opts.Strategy,
		Batch:    opts.Batch,
		Stride:   opts.Stride,
		InPlace:  opts.InPlace,
	}
}

// Meta returns metadata about how the plan was constructed. Strategy reflects
// the requested PlanOptions.Strategy; the per-axis resolved strategies are
// available via KernelStrategies.
func (p *Plan2D[T]) Meta() PlanMeta {
	return metaFromOptions(p.options)
}

// KernelStrategies returns the resolved kernel strategy for each axis in
// dimension order: index 0 describes the length-rows transforms applied along
// columns, index 1 the length-cols transforms applied along rows.
func (p *Plan2D[T]) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{p.colPlan.KernelStrategy(), p.rowPlan.KernelStrategy()}
}

// Algorithms returns the human-readable algorithm name for each axis, in the
// same dimension order as KernelStrategies.
func (p *Plan2D[T]) Algorithms() []string {
	return []string{p.colPlan.Algorithm(), p.rowPlan.Algorithm()}
}

// Meta returns metadata about how the plan was constructed. Strategy reflects
// the requested PlanOptions.Strategy; the per-axis resolved strategies are
// available via KernelStrategies.
func (p *Plan3D[T]) Meta() PlanMeta {
	return metaFromOptions(p.options)
}

// KernelStrategies returns the resolved kernel strategy for each axis in
// dimension order (depth, height, width): entry i describes the 1D transforms
// whose length is that dimension's size.
func (p *Plan3D[T]) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{
		p.depthPlan.KernelStrategy(),
		p.heightPlan.KernelStrategy(),
		p.widthPlan.KernelStrategy(),
	}
}

// Algorithms returns the human-readable algorithm name for each axis, in the
// same dimension order as KernelStrategies.
func (p *Plan3D[T]) Algorithms() []string {
	return []string{p.depthPlan.Algorithm(), p.heightPlan.Algorithm(), p.widthPlan.Algorithm()}
}

// Meta returns metadata about how the plan was constructed. Strategy reflects
// the requested PlanOptions.Strategy; the per-axis resolved strategies are
// available via KernelStrategies.
func (p *PlanND[T]) Meta() PlanMeta {
	return metaFromOptions(p.options)
}

// KernelStrategies returns the resolved kernel strategy for each dimension:
// entry i describes the 1D transforms along dimension i (length Dims()[i]).
func (p *PlanND[T]) KernelStrategies() []KernelStrategy {
	strategies := make([]KernelStrategy, len(p.plans))
	for i, plan := range p.plans {
		strategies[i] = plan.KernelStrategy()
	}

	return strategies
}

// Algorithms returns the human-readable algorithm name for each dimension, in
// the same order as KernelStrategies.
func (p *PlanND[T]) Algorithms() []string {
	algorithms := make([]string, len(p.plans))
	for i, plan := range p.plans {
		algorithms[i] = plan.Algorithm()
	}

	return algorithms
}

// Meta returns metadata about the underlying half-size complex plan that
// performs the packed real transform.
func (p *PlanRealT[F, C]) Meta() PlanMeta {
	return p.plan.Meta()
}

// KernelStrategy returns the resolved kernel strategy of the underlying
// half-size complex plan.
func (p *PlanRealT[F, C]) KernelStrategy() KernelStrategy {
	return p.plan.KernelStrategy()
}

// Algorithm returns the human-readable algorithm name of the underlying
// half-size complex plan.
func (p *PlanRealT[F, C]) Algorithm() string {
	return p.plan.Algorithm()
}

// Meta returns metadata about how the plan was constructed. Strategy reflects
// the requested PlanOptions.Strategy; the per-axis resolved strategies are
// available via KernelStrategies.
func (p *PlanReal2D) Meta() PlanMeta {
	return metaFromOptions(p.options)
}

// KernelStrategies returns the resolved kernel strategy for each axis in
// dimension order: index 0 describes the length-rows complex transforms
// applied along columns, index 1 the length-cols real transforms applied
// along rows (reported via their half-size complex plan).
func (p *PlanReal2D) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{p.colPlans[0].KernelStrategy(), p.rowPlan.KernelStrategy()}
}

// Algorithms returns the human-readable algorithm name for each axis, in the
// same dimension order as KernelStrategies.
func (p *PlanReal2D) Algorithms() []string {
	return []string{p.colPlans[0].Algorithm(), p.rowPlan.Algorithm()}
}

// Meta returns metadata about how the plan was constructed. PlanReal3D takes
// no options, so this reports the defaults; the per-axis resolved strategies
// are available via KernelStrategies.
func (p *PlanReal3D) Meta() PlanMeta {
	return metaFromOptions(normalizePlanOptions(PlanOptions{}))
}

// KernelStrategies returns the resolved kernel strategy for each axis in
// dimension order (depth, height, width): the depth and height entries
// describe complex transforms, the width entry the real transform (reported
// via its half-size complex plan).
func (p *PlanReal3D) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{
		p.depthPlans[0].KernelStrategy(),
		p.heightPlans[0].KernelStrategy(),
		p.widthPlan.KernelStrategy(),
	}
}

// Algorithms returns the human-readable algorithm name for each axis, in the
// same dimension order as KernelStrategies.
func (p *PlanReal3D) Algorithms() []string {
	return []string{
		p.depthPlans[0].Algorithm(),
		p.heightPlans[0].Algorithm(),
		p.widthPlan.Algorithm(),
	}
}

// Meta returns metadata about how the plan was constructed. FastPlan always
// uses heuristic planning (PlannerEstimate) and direct codelet bindings.
func (fp *FastPlan[T]) Meta() PlanMeta {
	return PlanMeta{Planner: PlannerEstimate, Strategy: fp.strategy}
}

// KernelStrategy returns the kernel strategy resolved at creation time.
func (fp *FastPlan[T]) KernelStrategy() KernelStrategy {
	return fp.strategy
}

// Algorithm returns the human-readable name of the bound codelet.
func (fp *FastPlan[T]) Algorithm() string {
	return fp.algorithm
}

// Close releases the plan's buffers. FastPlan buffers are not pooled, so Close
// only drops references so the GC can reclaim them promptly. After Close the
// plan must not be used for transforms; calling Close multiple times is safe.
func (fp *FastPlan[T]) Close() {
	fp.twiddle = nil
	fp.codeletTwiddleForward = nil
	fp.codeletTwiddleInverse = nil
	fp.scratch = nil
	fp.twiddleBacking = nil
	fp.codeletTwiddleForwardBacking = nil
	fp.codeletTwiddleInverseBacking = nil
	fp.scratchBacking = nil
	fp.forwardFunc = nil
	fp.inverseFunc = nil
}

// Meta returns metadata about the underlying half-size complex FastPlan.
func (fp *FastPlanReal32) Meta() PlanMeta {
	return fp.inner.Meta()
}

// KernelStrategy returns the kernel strategy of the underlying half-size
// complex FastPlan.
func (fp *FastPlanReal32) KernelStrategy() KernelStrategy {
	return fp.inner.KernelStrategy()
}

// Algorithm returns the codelet name of the underlying half-size complex
// FastPlan.
func (fp *FastPlanReal32) Algorithm() string {
	return fp.inner.Algorithm()
}

// Close releases the plan's buffers, including the underlying complex
// FastPlan's. After Close the plan must not be used for transforms; calling
// Close multiple times is safe.
func (fp *FastPlanReal32) Close() {
	fp.inner.Close()
	fp.weight = nil
	fp.buf = nil
}

// Meta returns metadata about the underlying half-size complex FastPlan.
func (fp *FastPlanReal64) Meta() PlanMeta {
	return fp.inner.Meta()
}

// KernelStrategy returns the kernel strategy of the underlying half-size
// complex FastPlan.
func (fp *FastPlanReal64) KernelStrategy() KernelStrategy {
	return fp.inner.KernelStrategy()
}

// Algorithm returns the codelet name of the underlying half-size complex
// FastPlan.
func (fp *FastPlanReal64) Algorithm() string {
	return fp.inner.Algorithm()
}

// Close releases the plan's buffers, including the underlying complex
// FastPlan's. After Close the plan must not be used for transforms; calling
// Close multiple times is safe.
func (fp *FastPlanReal64) Close() {
	fp.inner.Close()
	fp.weight = nil
	fp.buf = nil
}
