package algofft

import "github.com/cwbudde/algo-fft/internal/cpu"

// Planner builds plans with explicit options.
type Planner struct {
	opts     PlanOptions
	features cpu.Features
}

// NewPlanner creates a planner with the provided options.
func NewPlanner(opts PlanOptions) *Planner {
	return &Planner{
		opts:     normalizePlanOptions(opts),
		features: cpu.DetectFeatures(),
	}
}

// Plan1D builds a 1D plan for size n using the planner's options.
// Note: This is a standalone function (not a method) because Go does not allow
// type parameters on methods. Use Plan1D32 or Plan1D64 for method-style calls.
func Plan1D[T Complex](p *Planner, n int) (*Plan[T], error) {
	return newPlanWithFeatures[T](n, p.features, p.opts)
}

// Plan1D32 builds a 1D complex64 plan using the planner's options.
func (p *Planner) Plan1D32(n int) (*Plan[complex64], error) {
	return newPlanWithFeatures[complex64](n, p.features, p.opts)
}

// Plan1D64 builds a 1D complex128 plan using the planner's options.
func (p *Planner) Plan1D64(n int) (*Plan[complex128], error) {
	return newPlanWithFeatures[complex128](n, p.features, p.opts)
}

// Plan2D32 builds a 2D complex64 plan using the planner's options.
func (p *Planner) Plan2D32(rows, cols int) (*Plan2D[complex64], error) {
	return NewPlan2DWithOptions[complex64](rows, cols, p.opts)
}

// Plan2D64 builds a 2D complex128 plan using the planner's options.
func (p *Planner) Plan2D64(rows, cols int) (*Plan2D[complex128], error) {
	return NewPlan2DWithOptions[complex128](rows, cols, p.opts)
}

// Plan3D32 builds a 3D complex64 plan using the planner's options.
func (p *Planner) Plan3D32(depth, rows, cols int) (*Plan3D[complex64], error) {
	return NewPlan3DWithOptions[complex64](depth, rows, cols, p.opts)
}

// Plan3D64 builds a 3D complex128 plan using the planner's options.
func (p *Planner) Plan3D64(depth, rows, cols int) (*Plan3D[complex128], error) {
	return NewPlan3DWithOptions[complex128](depth, rows, cols, p.opts)
}

// PlanND32 builds an N-D complex64 plan using the planner's options.
func (p *Planner) PlanND32(dims []int) (*PlanND[complex64], error) {
	return NewPlanNDWithOptions[complex64](dims, p.opts)
}

// PlanND64 builds an N-D complex128 plan using the planner's options.
func (p *Planner) PlanND64(dims []int) (*PlanND[complex128], error) {
	return NewPlanNDWithOptions[complex128](dims, p.opts)
}

// PlanReal builds a real FFT plan using the planner's options.
func (p *Planner) PlanReal(n int) (*PlanReal, error) {
	return NewPlanRealWithOptions(n, p.opts)
}

// PlanReal2D builds a 2D real FFT plan using the planner's options.
func (p *Planner) PlanReal2D(rows, cols int) (*PlanReal2D, error) {
	return NewPlanReal2DWithOptions(rows, cols, p.opts)
}
