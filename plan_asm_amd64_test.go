//go:build amd64 && asm

package algofft

import "testing"

func TestForwardInverse_Size2_AsmRequired(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](2)
	if err != nil {
		t.Fatalf("NewPlan(2) returned error: %v", err)
	}

	src := []complex64{1 + 2i, 3 + 4i}
	dst := make([]complex64, 2)

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	roundTrip := make([]complex64, 2)
	if err := plan.Inverse(roundTrip, dst); err != nil {
		t.Fatalf("Inverse() returned error: %v", err)
	}

	for i := range src {
		if roundTrip[i] != src[i] {
			t.Fatalf("roundTrip[%d] = %v, want %v", i, roundTrip[i], src[i])
		}
	}
}
