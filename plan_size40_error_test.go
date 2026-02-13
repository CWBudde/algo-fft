package algofft

import (
	"testing"
)

// TestPlanSize40Errors checks if errors are being returned.
func TestPlanSize40Errors(t *testing.T) {
	const size = 40

	data := make([]complex128, size)
	for i := range data {
		data[i] = complex(float64(i), 0)
	}

	plan, err := NewPlanT[complex128](size)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	t.Logf("Calling Forward...")
	err = plan.Forward(data, data)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}
	t.Logf("Forward succeeded")

	t.Logf("Calling Inverse...")
	err = plan.Inverse(data, data)
	if err != nil {
		t.Fatalf("Inverse returned error: %v", err)
	}
	t.Logf("Inverse succeeded")
}
