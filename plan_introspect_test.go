package algofft

import (
	"testing"
)

// TestIntrospection_MultiDim verifies that 2D/3D/ND plans report per-axis
// strategies and algorithms consistent with equivalent 1D plans, and that
// Meta reflects the requested options.
func TestIntrospection_MultiDim(t *testing.T) {
	t.Parallel()

	ref16, err := NewPlanT[complex64](16)
	if err != nil {
		t.Fatalf("NewPlanT(16) failed: %v", err)
	}

	ref32, err := NewPlanT[complex64](32)
	if err != nil {
		t.Fatalf("NewPlanT(32) failed: %v", err)
	}

	p2d, err := NewPlan2D[complex64](16, 32)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	wantAlgos := []string{ref16.Algorithm(), ref32.Algorithm()}
	wantStrategies := []KernelStrategy{ref16.KernelStrategy(), ref32.KernelStrategy()}

	if got := p2d.Algorithms(); got[0] != wantAlgos[0] || got[1] != wantAlgos[1] {
		t.Errorf("Plan2D.Algorithms() = %v, want %v", got, wantAlgos)
	}

	if got := p2d.KernelStrategies(); got[0] != wantStrategies[0] || got[1] != wantStrategies[1] {
		t.Errorf("Plan2D.KernelStrategies() = %v, want %v", got, wantStrategies)
	}

	if meta := p2d.Meta(); meta.Planner != PlannerEstimate || meta.Strategy != KernelAuto {
		t.Errorf("Plan2D.Meta() = %+v, want default planner/strategy", meta)
	}

	p3d, err := NewPlan3D[complex64](8, 16, 32)
	if err != nil {
		t.Fatalf("NewPlan3D failed: %v", err)
	}

	if got := p3d.Algorithms(); len(got) != 3 || got[1] != ref16.Algorithm() || got[2] != ref32.Algorithm() {
		t.Errorf("Plan3D.Algorithms() = %v, want [_, %q, %q]", got, ref16.Algorithm(), ref32.Algorithm())
	}

	if got := p3d.KernelStrategies(); len(got) != 3 {
		t.Errorf("Plan3D.KernelStrategies() = %v, want 3 entries", got)
	}

	pnd, err := NewPlanND[complex64]([]int{16, 32})
	if err != nil {
		t.Fatalf("NewPlanND failed: %v", err)
	}

	if got := pnd.Algorithms(); got[0] != ref16.Algorithm() || got[1] != ref32.Algorithm() {
		t.Errorf("PlanND.Algorithms() = %v, want %v", got, wantAlgos)
	}

	if got := pnd.KernelStrategies(); got[0] != wantStrategies[0] || got[1] != wantStrategies[1] {
		t.Errorf("PlanND.KernelStrategies() = %v, want %v", got, wantStrategies)
	}
}

// TestIntrospection_MultiDimForcedStrategy verifies the requested strategy is
// visible in Meta and resolved per axis.
func TestIntrospection_MultiDimForcedStrategy(t *testing.T) {
	t.Parallel()

	p2d, err := NewPlan2DWithOptions[complex64](16, 16, PlanOptions{Strategy: KernelStockham})
	if err != nil {
		t.Fatalf("NewPlan2DWithOptions failed: %v", err)
	}

	if meta := p2d.Meta(); meta.Strategy != KernelStockham {
		t.Errorf("Meta().Strategy = %v, want %v", meta.Strategy, KernelStockham)
	}

	for i, s := range p2d.KernelStrategies() {
		if s != KernelStockham {
			t.Errorf("KernelStrategies()[%d] = %v, want %v", i, s, KernelStockham)
		}
	}
}

// TestIntrospection_Real verifies the real plans delegate introspection to
// their underlying complex plans.
func TestIntrospection_Real(t *testing.T) {
	t.Parallel()

	pr, err := NewPlanRealT[float32, complex64](64)
	if err != nil {
		t.Fatalf("NewPlanRealT failed: %v", err)
	}

	ref32, err := NewPlanT[complex64](32)
	if err != nil {
		t.Fatalf("NewPlanT(32) failed: %v", err)
	}

	if got := pr.Algorithm(); got != ref32.Algorithm() {
		t.Errorf("PlanRealT.Algorithm() = %q, want %q (half-size complex plan)", got, ref32.Algorithm())
	}

	if got := pr.KernelStrategy(); got != ref32.KernelStrategy() {
		t.Errorf("PlanRealT.KernelStrategy() = %v, want %v", got, ref32.KernelStrategy())
	}

	pr2d, err := NewPlanReal2D(16, 32)
	if err != nil {
		t.Fatalf("NewPlanReal2D failed: %v", err)
	}

	if got := pr2d.Algorithms(); len(got) != 2 || got[0] == "" || got[1] == "" {
		t.Errorf("PlanReal2D.Algorithms() = %v, want 2 non-empty entries", got)
	}

	if got := pr2d.KernelStrategies(); len(got) != 2 {
		t.Errorf("PlanReal2D.KernelStrategies() = %v, want 2 entries", got)
	}

	pr3d, err := NewPlanReal3D(8, 16, 32)
	if err != nil {
		t.Fatalf("NewPlanReal3D failed: %v", err)
	}

	if got := pr3d.Algorithms(); len(got) != 3 || got[0] == "" || got[1] == "" || got[2] == "" {
		t.Errorf("PlanReal3D.Algorithms() = %v, want 3 non-empty entries", got)
	}

	if meta := pr3d.Meta(); meta.Planner != PlannerEstimate {
		t.Errorf("PlanReal3D.Meta().Planner = %v, want %v", meta.Planner, PlannerEstimate)
	}
}

// TestIntrospection_FastPlan verifies FastPlan reports its bound codelet and
// that Close is idempotent and leaves introspection working.
func TestIntrospection_FastPlan(t *testing.T) {
	t.Parallel()

	fp, err := NewFastPlan[complex64](256)
	if err != nil {
		t.Skipf("no codelet for size 256 on this build: %v", err)
	}

	if fp.Algorithm() == "" {
		t.Error("FastPlan.Algorithm() is empty, want codelet name")
	}

	if meta := fp.Meta(); meta.Planner != PlannerEstimate || meta.Strategy != fp.KernelStrategy() {
		t.Errorf("FastPlan.Meta() = %+v, inconsistent with KernelStrategy()", meta)
	}

	algo := fp.Algorithm()

	fp.Close()
	fp.Close() // Close must be idempotent

	if got := fp.Algorithm(); got != algo {
		t.Errorf("Algorithm() after Close = %q, want %q", got, algo)
	}
}

// TestIntrospection_FastPlanReal verifies the real fast plans delegate to
// their inner complex FastPlan and support Close.
func TestIntrospection_FastPlanReal(t *testing.T) {
	t.Parallel()

	fpr, err := NewFastPlanReal32(128)
	if err != nil {
		t.Skipf("no codelet for size 64 on this build: %v", err)
	}

	inner, err := NewFastPlan[complex64](64)
	if err != nil {
		t.Fatalf("NewFastPlan(64) failed: %v", err)
	}

	if got := fpr.Algorithm(); got != inner.Algorithm() {
		t.Errorf("FastPlanReal32.Algorithm() = %q, want %q", got, inner.Algorithm())
	}

	if got := fpr.KernelStrategy(); got != inner.KernelStrategy() {
		t.Errorf("FastPlanReal32.KernelStrategy() = %v, want %v", got, inner.KernelStrategy())
	}

	fpr.Close()
	fpr.Close() // Close must be idempotent

	fpr64, err := NewFastPlanReal64(128)
	if err != nil {
		t.Skipf("no complex128 codelet for size 64 on this build: %v", err)
	}

	if fpr64.Algorithm() == "" {
		t.Error("FastPlanReal64.Algorithm() is empty, want codelet name")
	}

	fpr64.Close()
	fpr64.Close() // Close must be idempotent
}
