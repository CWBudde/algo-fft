package algofft

import "github.com/cwbudde/algo-fft/internal/cpu"

// PlanReal is a pre-computed real FFT plan for float32 input. It is an alias for
// the float32/complex64 instantiation of the generic PlanRealT and is retained
// for backward compatibility.
//
// The forward transform returns the non-redundant half-spectrum with length
// N/2+1. Plans are reusable and safe for concurrent use during transforms: the
// pack/unpack buffer is borrowed per call from an internal cache, so multiple
// goroutines may share one instance.
//
// Output bins obey conjugate symmetry for real inputs:
//
//	X[k] = conj(X[N-k]) for k = 1..N/2-1
//
// Index 0 is DC and index N/2 is Nyquist (purely real for even N; odd
// lengths have no Nyquist bin).
type PlanReal = PlanRealT[float32, complex64]

// NewPlanReal creates a new real FFT plan for length n (n >= 2).
// Even lengths use the packed half-size method; odd lengths are supported
// via an internal full-size complex FFT fallback.
func NewPlanReal(n int) (*PlanReal, error) {
	return NewPlanRealT[float32, complex64](n)
}

// NewPlanRealWithOptions creates a new real FFT plan with explicit planner options.
func NewPlanRealWithOptions(n int, opts PlanOptions) (*PlanReal, error) {
	return NewPlanRealTWithOptions[float32, complex64](n, opts)
}

func newPlanRealWithFeatures(n int, features cpu.Features, opts PlanOptions) (*PlanReal, error) {
	return newPlanRealTWithFeatures[float32, complex64](n, features, opts)
}
