package algofft

import (
	"math/cmplx"
	"testing"
)

func TestSize5RoundTrip(t *testing.T) {
	const size = 5
	const tol = 1e-10

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

	t.Logf("Size %d algorithm: %s", size, plan.Algorithm())

	err = plan.Forward(data, data)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	err = plan.Inverse(data, data)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	maxErr := 0.0
	for i := range data {
		err := cmplx.Abs(data[i] - orig[i])
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > tol {
		t.Errorf("Round-trip error: %e", maxErr)
	} else {
		t.Logf("Round-trip error: %e (PASS)", maxErr)
	}
}

func TestSize8RoundTrip(t *testing.T) {
	const size = 8
	const tol = 1e-10

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

	t.Logf("Size %d algorithm: %s", size, plan.Algorithm())

	err = plan.Forward(data, data)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	err = plan.Inverse(data, data)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	maxErr := 0.0
	for i := range data {
		err := cmplx.Abs(data[i] - orig[i])
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > tol {
		t.Errorf("Round-trip error: %e", maxErr)
	} else {
		t.Logf("Round-trip error: %e (PASS)", maxErr)
	}
}

func TestSize10RoundTrip(t *testing.T) {
	const size = 10
	const tol = 1e-10

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

	t.Logf("Size %d algorithm: %s", size, plan.Algorithm())

	err = plan.Forward(data, data)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	err = plan.Inverse(data, data)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	maxErr := 0.0
	for i := range data {
		err := cmplx.Abs(data[i] - orig[i])
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > tol {
		t.Errorf("Round-trip error: %e", maxErr)
	} else {
		t.Logf("Round-trip error: %e (PASS)", maxErr)
	}
}

func TestSize20RoundTrip(t *testing.T) {
	const size = 20
	const tol = 1e-10

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

	t.Logf("Size %d algorithm: %s", size, plan.Algorithm())

	err = plan.Forward(data, data)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	err = plan.Inverse(data, data)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	maxErr := 0.0
	for i := range data {
		err := cmplx.Abs(data[i] - orig[i])
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > tol {
		t.Errorf("Round-trip error: %e", maxErr)
		for i := 0; i < 5; i++ {
			t.Logf("  [%d] orig=%v result=%v", i, orig[i], data[i])
		}
	} else {
		t.Logf("Round-trip error: %e (PASS)", maxErr)
	}
}
