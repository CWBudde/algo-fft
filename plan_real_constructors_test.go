package algofft

import (
	"testing"
)

// TestNewPlanReal32WithOptions tests the constructor with options.
func TestNewPlanReal32WithOptions(t *testing.T) {
	t.Parallel()

	n := 64
	opts := PlanOptions{
		Planner: PlannerPatient,
	}

	plan, err := NewPlanReal32WithOptions(n, opts)
	if err != nil {
		t.Fatalf("NewPlanReal32WithOptions(%d, %+v) failed: %v", n, opts, err)
	}

	if plan == nil {
		t.Fatal("plan is nil")
	}

	if plan.Len() != n {
		t.Errorf("plan.Len() = %d, want %d", plan.Len(), n)
	}

	if plan.SpectrumLen() != n/2+1 {
		t.Errorf("plan.SpectrumLen() = %d, want %d", plan.SpectrumLen(), n/2+1)
	}
}

// TestNewPlanReal64WithOptions tests the constructor with options.
func TestNewPlanReal64WithOptions(t *testing.T) {
	t.Parallel()

	n := 64
	opts := PlanOptions{
		Planner: PlannerPatient,
	}

	plan, err := NewPlanReal64WithOptions(n, opts)
	if err != nil {
		t.Fatalf("NewPlanReal64WithOptions(%d, %+v) failed: %v", n, opts, err)
	}

	if plan == nil {
		t.Fatal("plan is nil")
	}

	if plan.Len() != n {
		t.Errorf("plan.Len() = %d, want %d", plan.Len(), n)
	}

	if plan.SpectrumLen() != n/2+1 {
		t.Errorf("plan.SpectrumLen() = %d, want %d", plan.SpectrumLen(), n/2+1)
	}
}

// TestNewPlanReal32WithOptions_InvalidLength tests error handling.
func TestNewPlanReal32WithOptions_InvalidLength(t *testing.T) {
	t.Parallel()

	_, err := NewPlanReal32WithOptions(0, PlanOptions{})
	if err == nil {
		t.Error("expected error for n=0, got nil")
	}

	_, err = NewPlanReal32WithOptions(7, PlanOptions{})
	if err == nil {
		t.Error("expected error for odd n, got nil")
	}
}

// TestNewPlanReal64WithOptions_InvalidLength tests error handling.
func TestNewPlanReal64WithOptions_InvalidLength(t *testing.T) {
	t.Parallel()

	_, err := NewPlanReal64WithOptions(0, PlanOptions{})
	if err == nil {
		t.Error("expected error for n=0, got nil")
	}

	_, err = NewPlanReal64WithOptions(7, PlanOptions{})
	if err == nil {
		t.Error("expected error for odd n, got nil")
	}
}

// TestNewPlanReal32WithOptions_RoundTrip tests basic functionality.
func TestNewPlanReal32WithOptions_RoundTrip(t *testing.T) {
	t.Parallel()

	n := 128
	opts := PlanOptions{
		Planner: PlannerEstimate,
	}

	plan, err := NewPlanReal32WithOptions(n, opts)
	if err != nil {
		t.Fatalf("NewPlanReal32WithOptions failed: %v", err)
	}

	// Create test input
	input := make([]float32, n)
	for i := range input {
		input[i] = float32(i)
	}

	// Forward
	spectrum := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(spectrum, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Inverse
	recovered := make([]float32, n)

	err = plan.Inverse(recovered, spectrum)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	// Verify round-trip
	for i := range input {
		diff := recovered[i] - input[i]
		if diff < 0 {
			diff = -diff
		}

		if diff > 1e-3 {
			t.Errorf("round-trip error at [%d]: got %v, want %v (diff=%v)", i, recovered[i], input[i], diff)
			break
		}
	}
}

// TestNewPlanReal64WithOptions_RoundTrip tests basic functionality.
func TestNewPlanReal64WithOptions_RoundTrip(t *testing.T) {
	t.Parallel()

	n := 128
	opts := PlanOptions{
		Planner: PlannerEstimate,
	}

	plan, err := NewPlanReal64WithOptions(n, opts)
	if err != nil {
		t.Fatalf("NewPlanReal64WithOptions failed: %v", err)
	}

	// Create test input
	input := make([]float64, n)
	for i := range input {
		input[i] = float64(i)
	}

	// Forward
	spectrum := make([]complex128, plan.SpectrumLen())

	err = plan.Forward(spectrum, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Inverse
	recovered := make([]float64, n)

	err = plan.Inverse(recovered, spectrum)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	// Verify round-trip
	for i := range input {
		diff := recovered[i] - input[i]
		if diff < 0 {
			diff = -diff
		}

		if diff > 1e-11 {
			t.Errorf("round-trip error at [%d]: got %v, want %v (diff=%v)", i, recovered[i], input[i], diff)
			break
		}
	}
}
