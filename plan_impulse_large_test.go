package algofft

import "testing"

func TestForward_Impulse_2048(t *testing.T) {
	t.Parallel()

	const n = 2048

	plan, err := NewPlanT[complex64](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) returned error: %v", n, err)
	}

	t.Logf("strategy=%v algorithm=%q", plan.KernelStrategy(), plan.Algorithm())

	src := make([]complex64, n)
	src[0] = 1
	dst := make([]complex64, n)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	for i := range dst {
		assertApproxComplex64f(t, dst[i], 1, 1e-4, "dst[%d]", i)
	}
}
