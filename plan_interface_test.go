package algofft

import (
	"testing"
)

// Compile-time checks: every plan type implements the common PlanInfo
// interface (introspection + lifecycle). This is the API contract from
// PLAN.md A1 — a missing method here is a build failure, not a runtime
// surprise.
var (
	_ PlanInfo = (*Plan[complex64])(nil)
	_ PlanInfo = (*Plan[complex128])(nil)
	_ PlanInfo = (*Plan2D[complex64])(nil)
	_ PlanInfo = (*Plan2D[complex128])(nil)
	_ PlanInfo = (*Plan3D[complex64])(nil)
	_ PlanInfo = (*Plan3D[complex128])(nil)
	_ PlanInfo = (*PlanND[complex64])(nil)
	_ PlanInfo = (*PlanND[complex128])(nil)
	_ PlanInfo = (*PlanReal[float32, complex64])(nil)
	_ PlanInfo = (*PlanReal[float64, complex128])(nil)
	_ PlanInfo = (*PlanReal2D[float32, complex64])(nil)
	_ PlanInfo = (*PlanReal2D[float64, complex128])(nil)
	_ PlanInfo = (*PlanReal3D[float32, complex64])(nil)
	_ PlanInfo = (*PlanReal3D[float64, complex128])(nil)
	_ PlanInfo = (*FastPlan[complex64])(nil)
	_ PlanInfo = (*FastPlan[complex128])(nil)
	_ PlanInfo = (*FastPlanReal[float32, complex64])(nil)
	_ PlanInfo = (*FastPlanReal[float64, complex128])(nil)
)

// TestIntrospection_PluralSingularConsistency verifies the 1D plan types
// report one-element plural introspection slices consistent with their
// singular accessors.
func TestIntrospection_PluralSingularConsistency(t *testing.T) {
	t.Parallel()

	p, err := NewPlan[complex64](64)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	if got := p.Algorithms(); len(got) != 1 || got[0] != p.Algorithm() {
		t.Errorf("Plan.Algorithms() = %v, want [%q]", got, p.Algorithm())
	}

	if got := p.KernelStrategies(); len(got) != 1 || got[0] != p.KernelStrategy() {
		t.Errorf("Plan.KernelStrategies() = %v, want [%v]", got, p.KernelStrategy())
	}

	pr, err := NewPlanReal[float32, complex64](64)
	if err != nil {
		t.Fatalf("NewPlanReal failed: %v", err)
	}

	if got := pr.Algorithms(); len(got) != 1 || got[0] != pr.Algorithm() {
		t.Errorf("PlanReal.Algorithms() = %v, want [%q]", got, pr.Algorithm())
	}

	if got := pr.KernelStrategies(); len(got) != 1 || got[0] != pr.KernelStrategy() {
		t.Errorf("PlanReal.KernelStrategies() = %v, want [%v]", got, pr.KernelStrategy())
	}

	fp, err := NewFastPlan[complex64](256)
	if err != nil {
		t.Skipf("no codelet for size 256 on this build: %v", err)
	}

	if got := fp.Algorithms(); len(got) != 1 || got[0] != fp.Algorithm() {
		t.Errorf("FastPlan.Algorithms() = %v, want [%q]", got, fp.Algorithm())
	}

	if got := fp.KernelStrategies(); len(got) != 1 || got[0] != fp.KernelStrategy() {
		t.Errorf("FastPlan.KernelStrategies() = %v, want [%v]", got, fp.KernelStrategy())
	}
}

// TestPlanInfo_CloseIdempotent verifies Close is callable and idempotent on
// every plan type reachable without codelet support.
func TestPlanInfo_CloseIdempotent(t *testing.T) {
	t.Parallel()

	plans := make([]PlanInfo, 0, 8)

	p1, err := NewPlan[complex64](16)
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}

	plans = append(plans, p1)

	p2, err := NewPlan2D[complex64](4, 8)
	if err != nil {
		t.Fatalf("NewPlan2D: %v", err)
	}

	plans = append(plans, p2)

	p3, err := NewPlan3D[complex64](2, 4, 8)
	if err != nil {
		t.Fatalf("NewPlan3D: %v", err)
	}

	plans = append(plans, p3)

	pn, err := NewPlanND[complex64]([]int{4, 4})
	if err != nil {
		t.Fatalf("NewPlanND: %v", err)
	}

	plans = append(plans, pn)

	pr, err := NewPlanReal[float64, complex128](16)
	if err != nil {
		t.Fatalf("NewPlanReal: %v", err)
	}

	plans = append(plans, pr)

	pr2, err := NewPlanReal2D[float64, complex128](4, 8)
	if err != nil {
		t.Fatalf("NewPlanReal2D: %v", err)
	}

	plans = append(plans, pr2)

	pr3, err := NewPlanReal3D[float64, complex128](2, 4, 8)
	if err != nil {
		t.Fatalf("NewPlanReal3D: %v", err)
	}

	plans = append(plans, pr3)

	for _, p := range plans {
		desc := p.String()

		if len(p.Algorithms()) == 0 {
			t.Errorf("%s: Algorithms() is empty", desc)
		}

		if len(p.KernelStrategies()) != len(p.Algorithms()) {
			t.Errorf("%s: KernelStrategies()/Algorithms() length mismatch", desc)
		}

		p.Close()
		p.Close() // must be idempotent
	}
}

// TestFastPlan_Clone verifies FastPlan clones share immutable tables but own
// their scratch, making concurrent use of the clones safe.
func TestFastPlan_Clone(t *testing.T) {
	t.Parallel()

	fp, err := NewFastPlan[complex64](64)
	if err != nil {
		t.Skipf("no codelet for size 64 on this build: %v", err)
	}

	clone := fp.Clone()

	if clone.Len() != fp.Len() {
		t.Fatalf("clone.Len() = %d, want %d", clone.Len(), fp.Len())
	}

	src := make([]complex64, 64)
	src[1] = 1

	dstOrig := make([]complex64, 64)
	dstClone := make([]complex64, 64)

	fp.Forward(dstOrig, src)
	clone.Forward(dstClone, src)

	for i := range dstOrig {
		if dstOrig[i] != dstClone[i] {
			t.Fatalf("clone output diverges at %d: %v vs %v", i, dstOrig[i], dstClone[i])
		}
	}

	// Closing the clone must not disturb the original.
	clone.Close()

	fp.Forward(dstOrig, src)
}

// TestFastPlanReal_Generic verifies the generic FastPlanReal round-trips in
// both precisions.
func TestFastPlanReal_Generic(t *testing.T) {
	t.Parallel()

	fp32, err := NewFastPlanReal[float32, complex64](128)
	if err != nil {
		t.Skipf("no complex64 codelet for size 64 on this build: %v", err)
	}

	src32 := make([]float32, 128)
	for i := range src32 {
		src32[i] = float32(i%7)*0.25 - 0.5
	}

	spec32 := make([]complex64, fp32.SpectrumLen())
	back32 := make([]float32, 128)

	fp32.Forward(spec32, src32)
	fp32.Inverse(back32, spec32)

	for i := range src32 {
		if diff := back32[i] - src32[i]; diff > 1e-4 || diff < -1e-4 {
			t.Fatalf("float32 round-trip mismatch at %d: got %v, want %v", i, back32[i], src32[i])
		}
	}

	fp64, err := NewFastPlanReal[float64, complex128](128)
	if err != nil {
		t.Skipf("no complex128 codelet for size 64 on this build: %v", err)
	}

	src64 := make([]float64, 128)
	for i := range src64 {
		src64[i] = float64(i%7)*0.25 - 0.5
	}

	spec64 := make([]complex128, fp64.SpectrumLen())
	back64 := make([]float64, 128)

	fp64.Forward(spec64, src64)
	fp64.Inverse(back64, spec64)

	for i := range src64 {
		if diff := back64[i] - src64[i]; diff > 1e-10 || diff < -1e-10 {
			t.Fatalf("float64 round-trip mismatch at %d: got %v, want %v", i, back64[i], src64[i])
		}
	}
}
