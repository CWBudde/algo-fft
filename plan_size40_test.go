package algofft

import (
	"math/cmplx"
	"testing"
)

// TestPlanSize40RoundTrip tests the specific case that's failing in algo-pde.
// Size 40 = 2^3 * 5, which should use Bluestein algorithm.
func TestPlanSize40RoundTrip(t *testing.T) {
	const size = 40
	const tol = 1e-10

	// Create simple test data (0, 1, 2, ..., 39)
	data := make([]complex128, size)
	for i := range data {
		data[i] = complex(float64(i), 0)
	}

	// Save original
	orig := make([]complex128, size)
	copy(orig, data)

	// Create plan and do round-trip
	plan, err := NewPlanT[complex128](size)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	err = plan.Forward(data, data)
	if err != nil {
		t.Fatalf("Forward transform failed: %v", err)
	}

	err = plan.Inverse(data, data)
	if err != nil {
		t.Fatalf("Inverse transform failed: %v", err)
	}

	// Check round-trip error
	maxErr := 0.0
	maxErrIdx := 0
	for i := range data {
		err := cmplx.Abs(data[i] - orig[i])
		if err > maxErr {
			maxErr = err
			maxErrIdx = i
		}
	}

	if maxErr > tol {
		t.Errorf("Round-trip error too large: %e > %e", maxErr, tol)
		t.Errorf("Max error at index %d: orig=%v, result=%v", maxErrIdx, orig[maxErrIdx], data[maxErrIdx])
		t.Logf("First few values:")
		for i := 0; i < 5; i++ {
			t.Logf("  [%d] orig=%v, result=%v, diff=%v", i, orig[i], data[i], data[i]-orig[i])
		}
	}
}

func TestSize40ForwardDCOnly(t *testing.T) {
	const size = 40

	data := make([]complex128, size)
	for i := range data {
		data[i] = complex(float64(i), 0)
	}

	plan, _ := NewPlanT[complex128](size)
	plan.Forward(data, data)

	// DC component should be sum of inputs = 0+1+2+...+39 = 780
	expectedDC := complex(780.0, 0)
	dcError := cmplx.Abs(data[0] - expectedDC)

	t.Logf("DC component: got %v, expected %v, error %e", data[0], expectedDC, dcError)
	if dcError > 1e-10 {
		t.Errorf("DC component is wrong!")
	}
}
