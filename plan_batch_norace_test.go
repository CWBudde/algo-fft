//go:build !race

package algofft

import "testing"

// TestPlanBatch_ZeroAllocations verifies no allocations during batch transforms.
//
// This test is excluded from race builds because the race detector adds
// instrumentation that causes allocations in defer statements that don't
// exist in normal builds.
//
//nolint:paralleltest
func TestPlanBatch_ZeroAllocations(t *testing.T) {
	// Note: t.Parallel() cannot be used here because testing.AllocsPerRun
	// panics when called during a parallel test.
	n := 64
	count := 10

	plan, err := NewPlan(n)
	if err != nil {
		t.Fatal(err)
	}

	src := make([]complex64, n*count)
	dst := make([]complex64, n*count)

	// Warm up
	for range 5 {
		_ = plan.ForwardBatch(dst, src, count)
	}

	// Measure allocations
	allocs := testing.AllocsPerRun(100, func() {
		_ = plan.ForwardBatch(dst, src, count)
	})

	if allocs > 0 {
		t.Errorf("ForwardBatch allocated %f times per run, want 0", allocs)
	}

	// Same for inverse
	allocs = testing.AllocsPerRun(100, func() {
		_ = plan.InverseBatch(dst, src, count)
	})

	if allocs > 0 {
		t.Errorf("InverseBatch allocated %f times per run, want 0", allocs)
	}
}
