package algofft

import (
	"testing"
)

// TestPlan_Meta tests the Meta() accessor method.
func TestPlan_Meta(t *testing.T) {
	t.Parallel()

	n := 64
	opts := PlanOptions{
		Planner:  PlannerPatient,
		Strategy: KernelDIT,
		Batch:    2,
		Stride:   128,
		InPlace:  true,
	}

	plan, err := NewPlanWithOptions[complex64](n, opts)
	if err != nil {
		t.Fatalf("NewPlanWithOptions failed: %v", err)
	}

	meta := plan.Meta()

	if meta.Planner != opts.Planner {
		t.Errorf("Meta().Planner = %v, want %v", meta.Planner, opts.Planner)
	}

	if meta.Strategy != opts.Strategy {
		t.Errorf("Meta().Strategy = %v, want %v", meta.Strategy, opts.Strategy)
	}

	if meta.Batch != opts.Batch {
		t.Errorf("Meta().Batch = %d, want %d", meta.Batch, opts.Batch)
	}

	if meta.Stride != opts.Stride {
		t.Errorf("Meta().Stride = %d, want %d", meta.Stride, opts.Stride)
	}

	if meta.InPlace != opts.InPlace {
		t.Errorf("Meta().InPlace = %v, want %v", meta.InPlace, opts.InPlace)
	}
}

// TestPlan32_Meta tests Meta() for float32 plan.
func TestPlan32_Meta(t *testing.T) {
	t.Parallel()

	n := 128
	opts := PlanOptions{
		Planner:  PlannerEstimate,
		Strategy: KernelStockham,
	}

	plan, err := NewPlanWithOptions[complex64](n, opts)
	if err != nil {
		t.Fatalf("NewPlanWithOptions failed: %v", err)
	}

	meta := plan.Meta()

	if meta.Planner != opts.Planner {
		t.Errorf("Meta().Planner = %v, want %v", meta.Planner, opts.Planner)
	}

	if meta.Strategy != opts.Strategy {
		t.Errorf("Meta().Strategy = %v, want %v", meta.Strategy, opts.Strategy)
	}
}

// TestPlan64_Meta tests Meta() for float64 plan.
func TestPlan64_Meta(t *testing.T) {
	t.Parallel()

	n := 256
	opts := PlanOptions{
		Planner:  PlannerMeasure,
		Strategy: KernelStockham, // Use explicit strategy instead of Auto
	}

	plan, err := NewPlanWithOptions[complex128](n, opts)
	if err != nil {
		t.Fatalf("NewPlanWithOptions failed: %v", err)
	}

	meta := plan.Meta()

	if meta.Planner != opts.Planner {
		t.Errorf("Meta().Planner = %v, want %v", meta.Planner, opts.Planner)
	}

	if meta.Strategy != opts.Strategy {
		t.Errorf("Meta().Strategy = %v, want %v", meta.Strategy, opts.Strategy)
	}
}

// TestPlan_Meta_DefaultOptions tests Meta() with default options.
func TestPlan_Meta_DefaultOptions(t *testing.T) {
	t.Parallel()

	n := 32

	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	meta := plan.Meta()

	// Default planner should be PlannerEstimate
	if meta.Planner != PlannerEstimate {
		t.Errorf("Meta().Planner = %v, want %v", meta.Planner, PlannerEstimate)
	}

	// Default batch should be 0 or 1
	if meta.Batch < 0 {
		t.Errorf("Meta().Batch = %d, should be >= 0", meta.Batch)
	}

	// InPlace should be false by default
	if meta.InPlace != false {
		t.Errorf("Meta().InPlace = %v, want false", meta.InPlace)
	}
}
