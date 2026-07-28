package algofft

import (
	"strconv"

	"github.com/cwbudde/algo-fft/internal/fftypes"
)

// This file provides the introspection and lifecycle accessors shared by all
// plan types. Every plan implements the PlanInfo interface (see
// plan_interface.go): plural KernelStrategies/Algorithms report the resolved
// kernel per axis (single-kernel plans return one-element slices), and Close
// releases plan resources. Single-kernel plan types additionally keep the
// singular KernelStrategy/Algorithm accessors as convenience.

// Precision names used by the String() methods of all plan types.
const (
	precisionNameComplex64  = "complex64"
	precisionNameComplex128 = "complex128"
)

// Strategy names used by the String() methods and the introspection tests.
const (
	strategyNameAuto       = "auto"
	strategyNameDIT        = "DIT"
	strategyNameStockham   = "Stockham"
	strategyNameSixStep    = "SixStep"
	strategyNameEightStep  = "EightStep"
	strategyNameBluestein  = "Bluestein"
	strategyNameRecursive  = "Recursive"
	strategyNameSplitRadix = "SplitRadix"
	strategyNameFourStep   = "FourStep"
	strategyNameMixedRadix = "MixedRadix"
)

// Len returns the FFT length (number of complex samples) for this Plan.
func (p *Plan[T]) Len() int {
	return p.n
}

// KernelStrategy reports the strategy chosen when the plan was created.
func (p *Plan[T]) KernelStrategy() KernelStrategy {
	return kernelStrategyFromInternal(p.kernelStrategy)
}

// Algorithm returns the name of the bound kernel or codelet (e.g., "dit8_generic").
// Returns empty string if no specific algorithm is bound.
func (p *Plan[T]) Algorithm() string {
	return p.algorithm
}

// String returns a human-readable description of the Plan for debugging.
// The format is: "Plan[type](size, strategy)" where type is "complex64" or "complex128".
func (p *Plan[T]) String() string {
	typeName := complexTypeName[T]()

	strategyName := strategyNameAuto

	switch p.kernelStrategy {
	case fftypes.KernelDIT:
		strategyName = strategyNameDIT
	case fftypes.KernelStockham:
		strategyName = strategyNameStockham
	case fftypes.KernelSixStep:
		strategyName = strategyNameSixStep
	case fftypes.KernelEightStep:
		strategyName = strategyNameEightStep
	case fftypes.KernelBluestein:
		strategyName = strategyNameBluestein
	case fftypes.KernelRecursive:
		strategyName = strategyNameRecursive
	case fftypes.KernelSplitRadix:
		strategyName = strategyNameSplitRadix
	case fftypes.KernelFourStep:
		strategyName = strategyNameFourStep
	case fftypes.KernelMixedRadix:
		strategyName = strategyNameMixedRadix
	case fftypes.KernelAuto:
		// Resolved plans never carry KernelAuto; keep the default name.
	}

	pooled := ""
	if p.pool != nil {
		pooled = ", pooled"
	}

	return "Plan[" + typeName + "](" + strconv.Itoa(p.n) + ", " + strategyName + pooled + ")"
}

// KernelStrategies returns the resolved kernel strategy as a one-element
// slice (1D plans have a single kernel). See KernelStrategy for the singular
// accessor.
func (p *Plan[T]) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{p.KernelStrategy()}
}

// Algorithms returns the bound algorithm name as a one-element slice (1D
// plans have a single kernel). See Algorithm for the singular accessor.
func (p *Plan[T]) Algorithms() []string {
	return []string{p.Algorithm()}
}

// KernelStrategies returns the resolved kernel strategy for each axis in
// dimension order: index 0 describes the length-rows transforms applied along
// columns, index 1 the length-cols transforms applied along rows.
func (p *Plan2D[T]) KernelStrategies() []KernelStrategy {
	return p.nd.KernelStrategies()
}

// Algorithms returns the human-readable algorithm name for each axis, in the
// same dimension order as KernelStrategies.
func (p *Plan2D[T]) Algorithms() []string {
	return p.nd.Algorithms()
}

// Close releases the plan's scratch cache and child-plan references. After
// Close the plan must not be used for transforms; calling Close multiple
// times is safe. Clones are unaffected (they hold their own references).
func (p *Plan2D[T]) Close() {
	if p.nd != nil {
		p.nd.Close()
		p.nd = nil
	}
}

// KernelStrategies returns the resolved kernel strategy for each axis in
// dimension order (depth, height, width): entry i describes the 1D transforms
// whose length is that dimension's size.
func (p *Plan3D[T]) KernelStrategies() []KernelStrategy {
	return p.nd.KernelStrategies()
}

// Algorithms returns the human-readable algorithm name for each axis, in the
// same dimension order as KernelStrategies.
func (p *Plan3D[T]) Algorithms() []string {
	return p.nd.Algorithms()
}

// Close releases the plan's scratch cache and child-plan references. After
// Close the plan must not be used for transforms; calling Close multiple
// times is safe. Clones are unaffected (they hold their own references).
func (p *Plan3D[T]) Close() {
	if p.nd != nil {
		p.nd.Close()
		p.nd = nil
	}
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

// Close releases the plan's scratch cache and child-plan references. After
// Close the plan must not be used for transforms; calling Close multiple
// times is safe. Clones are unaffected (they hold their own references).
func (p *PlanND[T]) Close() {
	p.plans = nil
	p.scratch = nil
}

// KernelStrategy returns the resolved kernel strategy of the underlying
// complex plan (half-size for even lengths, full-size for the odd-length
// fallback).
func (p *PlanReal[F, C]) KernelStrategy() KernelStrategy {
	return p.plan.KernelStrategy()
}

// Algorithm returns the human-readable algorithm name of the underlying
// complex plan (half-size for even lengths, full-size for the odd-length
// fallback).
func (p *PlanReal[F, C]) Algorithm() string {
	return p.plan.Algorithm()
}

// KernelStrategies returns the resolved kernel strategy of the underlying
// complex plan as a one-element slice.
func (p *PlanReal[F, C]) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{p.KernelStrategy()}
}

// Algorithms returns the algorithm name of the underlying complex plan as a
// one-element slice.
func (p *PlanReal[F, C]) Algorithms() []string {
	return []string{p.Algorithm()}
}

// Close releases the plan's buffers and child-plan reference. After Close the
// plan must not be used for transforms; calling Close multiple times is safe.
// Clones are unaffected (they hold their own references).
func (p *PlanReal[F, C]) Close() {
	p.plan = nil
	p.weight = nil
	p.buf = nil
}

// KernelStrategies returns the resolved kernel strategy for each axis in
// dimension order: index 0 describes the length-rows complex transforms
// applied along columns, index 1 the length-cols real transforms applied
// along rows (reported via their half-size complex plan).
func (p *PlanReal2D[F, C]) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{p.colPlans[0].KernelStrategy(), p.rowPlan.KernelStrategy()}
}

// Algorithms returns the human-readable algorithm name for each axis, in the
// same dimension order as KernelStrategies.
func (p *PlanReal2D[F, C]) Algorithms() []string {
	return []string{p.colPlans[0].Algorithm(), p.rowPlan.Algorithm()}
}

// Close releases the plan's scratch cache and child-plan references. After
// Close the plan must not be used for transforms; calling Close multiple
// times is safe. Clones are unaffected (they hold their own references).
func (p *PlanReal2D[F, C]) Close() {
	p.rowPlan = nil
	p.colPlans = nil
	p.scratch = nil
}

// KernelStrategies returns the resolved kernel strategy for each axis in
// dimension order (depth, height, width): the depth and height entries
// describe complex transforms, the width entry the real transform (reported
// via its half-size complex plan).
func (p *PlanReal3D[F, C]) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{
		p.depthPlans[0].KernelStrategy(),
		p.heightPlans[0].KernelStrategy(),
		p.widthPlan.KernelStrategy(),
	}
}

// Algorithms returns the human-readable algorithm name for each axis, in the
// same dimension order as KernelStrategies.
func (p *PlanReal3D[F, C]) Algorithms() []string {
	return []string{
		p.depthPlans[0].Algorithm(),
		p.heightPlans[0].Algorithm(),
		p.widthPlan.Algorithm(),
	}
}

// Close releases the plan's scratch cache and child-plan references. After
// Close the plan must not be used for transforms; calling Close multiple
// times is safe. Clones are unaffected (they hold their own references).
func (p *PlanReal3D[F, C]) Close() {
	p.widthPlan = nil
	p.heightPlans = nil
	p.depthPlans = nil
	p.scratch = nil
}

// KernelStrategy returns the kernel strategy resolved at creation time.
func (fp *FastPlan[T]) KernelStrategy() KernelStrategy {
	return kernelStrategyFromInternal(fp.strategy)
}

// Algorithm returns the human-readable name of the bound codelet.
func (fp *FastPlan[T]) Algorithm() string {
	return fp.algorithm
}

// KernelStrategies returns the resolved kernel strategy as a one-element
// slice (a FastPlan binds exactly one codelet).
func (fp *FastPlan[T]) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{fp.KernelStrategy()}
}

// Algorithms returns the bound codelet name as a one-element slice.
func (fp *FastPlan[T]) Algorithms() []string {
	return []string{fp.algorithm}
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

// KernelStrategy returns the kernel strategy of the underlying half-size
// complex FastPlan.
func (fp *FastPlanReal[F, C]) KernelStrategy() KernelStrategy {
	return fp.inner.KernelStrategy()
}

// Algorithm returns the codelet name of the underlying half-size complex
// FastPlan.
func (fp *FastPlanReal[F, C]) Algorithm() string {
	return fp.inner.Algorithm()
}

// KernelStrategies returns the kernel strategy of the underlying half-size
// complex FastPlan as a one-element slice.
func (fp *FastPlanReal[F, C]) KernelStrategies() []KernelStrategy {
	return []KernelStrategy{fp.inner.KernelStrategy()}
}

// Algorithms returns the codelet name of the underlying half-size complex
// FastPlan as a one-element slice.
func (fp *FastPlanReal[F, C]) Algorithms() []string {
	return []string{fp.inner.Algorithm()}
}

// Close releases the plan's buffers, including the underlying complex
// FastPlan's. After Close the plan must not be used for transforms; calling
// Close multiple times is safe.
func (fp *FastPlanReal[F, C]) Close() {
	fp.inner.Close()
	fp.weight = nil
	fp.buf = nil
}
