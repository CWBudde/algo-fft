package algofft

import (
	"fmt"
	"math/cmplx"
	"testing"
)

// TestPlanSize40Trace traces the intermediate values during transform.
func TestPlanSize40Trace(t *testing.T) {
	const size = 40

	// Create simple test data (just 0,1,2,3,4,... easier to track)
	data := make([]complex128, size)
	for i := range data {
		data[i] = complex(float64(i), 0)
	}

	orig := make([]complex128, size)
	copy(orig, data)

	plan, err := NewPlanT[complex128](size)
	if err != nil {
		t.Fatalf("Failed to create plan: %v", err)
	}

	t.Logf("Before forward transform:")
	for i := 0; i < 5; i++ {
		t.Logf("  data[%d] = %v", i, data[i])
	}

	err = plan.Forward(data, data)
	if err != nil {
		t.Fatalf("Forward transform failed: %v", err)
	}

	t.Logf("After forward transform:")
	for i := 0; i < 5; i++ {
		t.Logf("  data[%d] = %v", i, data[i])
	}

	// Check if DC component (data[0]) is correct
	// Should be sum of all inputs = 0+1+2+...+39 = 39*40/2 = 780
	expectedDC := complex(float64(size-1)*float64(size)/2, 0)
	dcError := cmplx.Abs(data[0] - expectedDC)
	t.Logf("DC component: got %v, expected %v, error %e", data[0], expectedDC, dcError)

	err = plan.Inverse(data, data)
	if err != nil {
		t.Fatalf("Inverse transform failed: %v", err)
	}

	t.Logf("After inverse transform:")
	for i := 0; i < 5; i++ {
		t.Logf("  data[%d] = %v (orig %v, diff %v)", i, data[i], orig[i], data[i]-orig[i])
	}

	// Check the error
	maxErr := 0.0
	for i := range data {
		err := cmplx.Abs(data[i] - orig[i])
		if err > maxErr {
			maxErr = err
		}
	}
	t.Logf("Max round-trip error: %e", maxErr)
}

// TestSize40ForwardOnly just tests the forward transform to see if DC is correct
func TestSize40ForwardOnly(t *testing.T) {
	const size = 40

	// Test with simple delta function at position 0
	delta := make([]complex128, size)
	delta[0] = 1

	plan, _ := NewPlanT[complex128](size)
	plan.Forward(delta, delta)

	t.Logf("FFT of delta function:")
	for i := 0; i < 10; i++ {
		t.Logf("  [%d] = %v", i, delta[i])
	}

	// All values should be 1.0 for delta function FFT
	for i := 0; i < size; i++ {
		expected := complex(1.0, 0)
		if cmplx.Abs(delta[i]-expected) > 1e-10 {
			t.Errorf("FFT[%d] = %v, expected %v", i, delta[i], expected)
		}
	}
}

// TestSize40CompareWith39 compares size 40 with nearby size 39 (prime)
func TestSize40CompareWith39(t *testing.T) {
	sizes := []int{39, 40, 41}

	for _, size := range sizes {
		data := make([]complex128, size)
		for i := range data {
			data[i] = complex(float64(i), 0)
		}

		orig := make([]complex128, size)
		copy(orig, data)

		plan, err := NewPlanT[complex128](size)
		if err != nil {
			t.Logf("Size %d: failed to create plan: %v", size, err)
			continue
		}

		t.Logf("Size %d: algorithm=%s", size, plan.Algorithm())

		plan.Forward(data, data)
		plan.Inverse(data, data)

		maxErr := 0.0
		for i := range data {
			err := cmplx.Abs(data[i] - orig[i])
			if err > maxErr {
				maxErr = err
			}
		}

		status := "PASS"
		if maxErr > 1e-10 {
			status = fmt.Sprintf("FAIL (error %e)", maxErr)
		}
		t.Logf("Size %d: %s", size, status)
	}
}
