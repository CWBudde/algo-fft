//go:build amd64 && !purego

package algofft

import "testing"

func TestForwardInverse_Size2_AsmRequired(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](2)
	if err != nil {
		t.Fatalf("NewPlan[complex64](2) returned error: %v", err)
	}

	src := []complex64{1 + 2i, 3 + 4i}
	dst := make([]complex64, 2)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	roundTrip := make([]complex64, 2)
	err = plan.Inverse(roundTrip, dst)
	if err != nil {
		t.Fatalf("Inverse() returned error: %v", err)
	}

	for i := range src {
		if roundTrip[i] != src[i] {
			t.Fatalf("roundTrip[%d] = %v, want %v", i, roundTrip[i], src[i])
		}
	}
}
