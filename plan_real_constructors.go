package algofft

// NewPlanReal32 creates a new single-precision (float32) real FFT plan.
// This is equivalent to NewPlanRealT[float32, complex64](n).
//
// The forward transform accepts float32 input and produces complex64 output.
// This is the same as the existing NewPlanReal for backward compatibility.
func NewPlanReal32(n int) (*PlanRealT[float32, complex64], error) {
	return NewPlanRealT[float32, complex64](n)
}

// NewPlanReal32WithOptions creates a new single-precision real FFT plan with planner options.
func NewPlanReal32WithOptions(n int, opts PlanOptions) (*PlanRealT[float32, complex64], error) {
	return NewPlanRealTWithOptions[float32, complex64](n, opts)
}

// NewPlanReal64 creates a new double-precision (float64) real FFT plan.
// This is equivalent to NewPlanRealT[float64, complex128](n).
//
// The forward transform accepts float64 input and produces complex128 output.
// Use this when you need higher numerical precision (< 1e-12 round-trip error vs < 1e-6 for float32).
//
// Example:
//
//	// High-precision audio or scientific computing
//	plan, err := algofft.NewPlanReal64(4096)
//	if err != nil {
//	    panic(err)
//	}
//
//	input := make([]float64, 4096)
//	// ... fill input with high-precision data
//
//	output := make([]complex128, 2049)  // N/2+1 bins
//	err = plan.Forward(output, input)
func NewPlanReal64(n int) (*PlanRealT[float64, complex128], error) {
	return NewPlanRealT[float64, complex128](n)
}

// NewPlanReal64WithOptions creates a new double-precision real FFT plan with planner options.
func NewPlanReal64WithOptions(n int, opts PlanOptions) (*PlanRealT[float64, complex128], error) {
	return NewPlanRealTWithOptions[float64, complex128](n, opts)
}
