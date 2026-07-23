package algofft

// PlanInfo is the introspection and lifecycle interface implemented by every
// plan type in the package. Transform methods are excluded because their
// signatures depend on the plan's element types; use the concrete plan types
// (or the generic Plan[T]/PlanReal[F, C]) for transforms.
//
// KernelStrategies and Algorithms report the resolved kernel per axis in
// dimension order; single-kernel plans (1D, real 1D, fast plans) return
// one-element slices.
type PlanInfo interface {
	// Len returns the total number of input elements the plan transforms.
	Len() int

	// KernelStrategies returns the resolved kernel strategy per axis.
	KernelStrategies() []KernelStrategy

	// Algorithms returns the bound algorithm name per axis.
	Algorithms() []string

	// String returns a human-readable description of the plan.
	String() string

	// Close releases the plan's resources. After Close the plan must not be
	// used for transforms; calling Close multiple times is safe.
	Close()
}

// Compile-time interface conformance for every plan type.
var (
	_ PlanInfo = (*Plan[complex64])(nil)
	_ PlanInfo = (*Plan2D[complex64])(nil)
	_ PlanInfo = (*Plan3D[complex64])(nil)
	_ PlanInfo = (*PlanND[complex64])(nil)
	_ PlanInfo = (*PlanReal[float32, complex64])(nil)
	_ PlanInfo = (*PlanReal2D[float32, complex64])(nil)
	_ PlanInfo = (*PlanReal3D[float32, complex64])(nil)
	_ PlanInfo = (*FastPlan[complex64])(nil)
	_ PlanInfo = (*FastPlanReal[float32, complex64])(nil)
)
