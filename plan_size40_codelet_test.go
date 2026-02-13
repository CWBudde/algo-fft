package algofft

import (
	"testing"
)

// TestPlanSize40Codelet checks if a codelet is being used for size 40.
func TestPlanSize40Codelet(t *testing.T) {
	const size = 40

	plan, err := NewPlanT[complex128](size)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	t.Logf("Plan for size %d:", size)
	t.Logf("  Algorithm: %s", plan.Algorithm())
	t.Logf("  Strategy: %v", plan.kernelStrategy)
	t.Logf("  Has forwardCodelet: %v", plan.forwardCodelet != nil)
	t.Logf("  Has inverseCodelet: %v", plan.inverseCodelet != nil)
	t.Logf("  Has forwardKernel: %v", plan.forwardKernel != nil)
	t.Logf("  Has inverseKernel: %v", plan.inverseKernel != nil)
}
