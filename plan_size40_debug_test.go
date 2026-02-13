package algofft

import (
	"testing"
)

// TestPlanSize40Algorithm identifies which algorithm is being used for size 40.
func TestPlanSize40Algorithm(t *testing.T) {
	const size = 40

	plan, err := NewPlanT[complex128](size)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	t.Logf("Plan for size %d:", size)
	t.Logf("  Algorithm: %s", plan.Algorithm())
	t.Logf("  N: %d", plan.n)
	t.Logf("  Strategy: %v", plan.kernelStrategy)
}
