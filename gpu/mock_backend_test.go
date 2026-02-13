package gpu

import "testing"

func TestMockBackendForwardInverse(t *testing.T) {
	RegisterMockBackend()
	plan, err := NewPlan[complex64](8, PlanOptions{})
	if err != nil {
		t.Fatalf("NewPlan: %v", err)
	}
	defer func() { _ = plan.Close() }()

	src := []complex64{1, 0, 0, 0, 0, 0, 0, 0}
	dst := make([]complex64, 8)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	out := make([]complex64, 8)
	if err := plan.Inverse(out, dst); err != nil {
		t.Fatalf("Inverse: %v", err)
	}
}
