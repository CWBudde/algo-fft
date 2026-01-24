package algofft

import (
	"testing"
)

func TestNewPlanner(t *testing.T) {
	t.Parallel()

	opts := PlanOptions{
		Planner:  PlannerEstimate,
		Strategy: KernelAuto,
	}

	planner := NewPlanner(opts)
	if planner == nil {
		t.Fatal("NewPlanner() returned nil")
	}

	if planner.opts.Planner != PlannerEstimate {
		t.Errorf("Planner mode = %v, want %v", planner.opts.Planner, PlannerEstimate)
	}
}

func TestNewPlanner_DefaultOptions(t *testing.T) {
	t.Parallel()

	// Empty options should be normalized
	planner := NewPlanner(PlanOptions{})
	if planner == nil {
		t.Fatal("NewPlanner() returned nil")
	}

	// Default planner mode should be PlannerEstimate
	if planner.opts.Planner != PlannerEstimate {
		t.Errorf("Default planner mode = %v, want %v", planner.opts.Planner, PlannerEstimate)
	}
}

func TestPlanner_Plan1D32(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	plan, err := planner.Plan1D32(128)
	if err != nil {
		t.Fatalf("Plan1D32(128) failed: %v", err)
	}

	if plan.Len() != 128 {
		t.Errorf("plan.Len() = %d, want 128", plan.Len())
	}

	// Test the plan works
	src := make([]complex64, 128)
	src[0] = 1
	dst := make([]complex64, 128)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}

	// Impulse should produce all ones in frequency domain
	for i, v := range dst {
		expected := complex(float32(1), 0)
		if absComplex64(v-expected) > 1e-5 {
			t.Errorf("dst[%d] = %v, want %v", i, v, expected)
		}
	}
}

func TestPlanner_Plan1D64(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	plan, err := planner.Plan1D64(256)
	if err != nil {
		t.Fatalf("Plan1D64(256) failed: %v", err)
	}

	if plan.Len() != 256 {
		t.Errorf("plan.Len() = %d, want 256", plan.Len())
	}

	// Test the plan works
	src := make([]complex128, 256)
	src[0] = 1
	dst := make([]complex128, 256)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}

	// Impulse should produce all ones in frequency domain
	for i, v := range dst {
		expected := complex(1.0, 0.0)
		if absComplex128(v-expected) > 1e-10 {
			t.Errorf("dst[%d] = %v, want %v", i, v, expected)
		}
	}
}

func TestPlanner_Plan1D_GenericFunction(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	// Test generic Plan1D function with complex64
	plan64, err := Plan1D[complex64](planner, 64)
	if err != nil {
		t.Fatalf("Plan1D[complex64](64) failed: %v", err)
	}

	if plan64.Len() != 64 {
		t.Errorf("plan64.Len() = %d, want 64", plan64.Len())
	}

	// Test generic Plan1D function with complex128
	plan128, err := Plan1D[complex128](planner, 64)
	if err != nil {
		t.Fatalf("Plan1D[complex128](64) failed: %v", err)
	}

	if plan128.Len() != 64 {
		t.Errorf("plan128.Len() = %d, want 64", plan128.Len())
	}
}

func TestPlanner_Plan2D32(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	plan, err := planner.Plan2D32(16, 32)
	if err != nil {
		t.Fatalf("Plan2D32(16, 32) failed: %v", err)
	}

	if plan.Rows() != 16 {
		t.Errorf("plan.Rows() = %d, want 16", plan.Rows())
	}

	if plan.Cols() != 32 {
		t.Errorf("plan.Cols() = %d, want 32", plan.Cols())
	}

	if plan.Len() != 16*32 {
		t.Errorf("plan.Len() = %d, want %d", plan.Len(), 16*32)
	}

	// Test the plan works
	src := make([]complex64, 16*32)
	src[0] = 1
	dst := make([]complex64, 16*32)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}
}

func TestPlanner_Plan2D64(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	plan, err := planner.Plan2D64(8, 16)
	if err != nil {
		t.Fatalf("Plan2D64(8, 16) failed: %v", err)
	}

	if plan.Rows() != 8 {
		t.Errorf("plan.Rows() = %d, want 8", plan.Rows())
	}

	if plan.Cols() != 16 {
		t.Errorf("plan.Cols() = %d, want 16", plan.Cols())
	}

	// Test the plan works
	src := make([]complex128, 8*16)
	src[0] = 1
	dst := make([]complex128, 8*16)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}
}

func TestPlanner_Plan3D32(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	plan, err := planner.Plan3D32(4, 8, 16)
	if err != nil {
		t.Fatalf("Plan3D32(4, 8, 16) failed: %v", err)
	}

	if plan.Depth() != 4 {
		t.Errorf("plan.Depth() = %d, want 4", plan.Depth())
	}

	if plan.Height() != 8 {
		t.Errorf("plan.Height() = %d, want 8", plan.Height())
	}

	if plan.Width() != 16 {
		t.Errorf("plan.Width() = %d, want 16", plan.Width())
	}

	if plan.Len() != 4*8*16 {
		t.Errorf("plan.Len() = %d, want %d", plan.Len(), 4*8*16)
	}

	// Test the plan works
	src := make([]complex64, 4*8*16)
	src[0] = 1
	dst := make([]complex64, 4*8*16)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}
}

func TestPlanner_Plan3D64(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	plan, err := planner.Plan3D64(8, 8, 8)
	if err != nil {
		t.Fatalf("Plan3D64(8, 8, 8) failed: %v", err)
	}

	if plan.Depth() != 8 {
		t.Errorf("plan.Depth() = %d, want 8", plan.Depth())
	}

	if plan.Height() != 8 {
		t.Errorf("plan.Height() = %d, want 8", plan.Height())
	}

	if plan.Width() != 8 {
		t.Errorf("plan.Width() = %d, want 8", plan.Width())
	}

	// Test the plan works
	src := make([]complex128, 8*8*8)
	src[0] = 1
	dst := make([]complex128, 8*8*8)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}
}

func TestPlanner_PlanND32(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	dims := []int{4, 8, 16}

	plan, err := planner.PlanND32(dims)
	if err != nil {
		t.Fatalf("PlanND32(%v) failed: %v", dims, err)
	}

	if plan.NDims() != 3 {
		t.Errorf("plan.NDims() = %d, want 3", plan.NDims())
	}

	expectedLen := 4 * 8 * 16
	if plan.Len() != expectedLen {
		t.Errorf("plan.Len() = %d, want %d", plan.Len(), expectedLen)
	}

	// Verify dimensions
	planDims := plan.Dims()
	for i, want := range dims {
		if planDims[i] != want {
			t.Errorf("plan.Dims()[%d] = %d, want %d", i, planDims[i], want)
		}
	}

	// Test the plan works
	src := make([]complex64, expectedLen)
	src[0] = 1
	dst := make([]complex64, expectedLen)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}
}

func TestPlanner_PlanND64(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	dims := []int{8, 8}

	plan, err := planner.PlanND64(dims)
	if err != nil {
		t.Fatalf("PlanND64(%v) failed: %v", dims, err)
	}

	if plan.NDims() != 2 {
		t.Errorf("plan.NDims() = %d, want 2", plan.NDims())
	}

	expectedLen := 8 * 8
	if plan.Len() != expectedLen {
		t.Errorf("plan.Len() = %d, want %d", plan.Len(), expectedLen)
	}

	// Test the plan works
	src := make([]complex128, expectedLen)
	src[0] = 1
	dst := make([]complex128, expectedLen)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}
}

func TestPlanner_PlanReal(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	plan, err := planner.PlanReal(128)
	if err != nil {
		t.Fatalf("PlanReal(128) failed: %v", err)
	}

	if plan.Len() != 128 {
		t.Errorf("plan.Len() = %d, want 128", plan.Len())
	}

	expectedSpectrumLen := 128/2 + 1
	if plan.SpectrumLen() != expectedSpectrumLen {
		t.Errorf("plan.SpectrumLen() = %d, want %d", plan.SpectrumLen(), expectedSpectrumLen)
	}

	// Test the plan works
	src := make([]float32, 128)
	src[0] = 1
	spectrum := make([]complex64, expectedSpectrumLen)

	err = plan.Forward(spectrum, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}
}

func TestPlanner_PlanReal2D(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	plan, err := planner.PlanReal2D(16, 32)
	if err != nil {
		t.Fatalf("PlanReal2D(16, 32) failed: %v", err)
	}

	expectedLen := 16 * 32
	if plan.Len() != expectedLen {
		t.Errorf("plan.Len() = %d, want %d", plan.Len(), expectedLen)
	}

	expectedSpectrumLen := 16 * (32/2 + 1)
	if plan.SpectrumLen() != expectedSpectrumLen {
		t.Errorf("plan.SpectrumLen() = %d, want %d", plan.SpectrumLen(), expectedSpectrumLen)
	}

	// Test the plan works
	src := make([]float32, expectedLen)
	src[0] = 1
	spectrum := make([]complex64, expectedSpectrumLen)

	err = plan.Forward(spectrum, src)
	if err != nil {
		t.Errorf("Forward() failed: %v", err)
	}
}

func TestPlanner_WithStrategy(t *testing.T) {
	t.Parallel()

	// Test with different strategies
	strategies := []KernelStrategy{
		KernelAuto,
		KernelDIT,
		KernelStockham,
	}

	for _, strategy := range strategies {
		opts := PlanOptions{
			Strategy: strategy,
		}

		planner := NewPlanner(opts)

		plan, err := planner.Plan1D32(64)
		if err != nil {
			t.Errorf("Plan1D32(64) with strategy %v failed: %v", strategy, err)
			continue
		}

		if plan.Len() != 64 {
			t.Errorf("plan.Len() = %d, want 64 for strategy %v", plan.Len(), strategy)
		}
	}
}

func TestPlanner_WithPlannerModes(t *testing.T) {
	t.Parallel()

	modes := []PlannerMode{
		PlannerEstimate,
		PlannerMeasure,
		PlannerPatient,
		PlannerExhaustive,
	}

	for _, mode := range modes {
		opts := PlanOptions{
			Planner: mode,
		}

		planner := NewPlanner(opts)
		if planner.opts.Planner != mode {
			t.Errorf("Planner mode = %v, want %v", planner.opts.Planner, mode)
		}

		// Test that plans can still be created (though some modes are slow)
		// We'll use a small size to keep tests fast
		if mode == PlannerEstimate || mode == PlannerMeasure {
			plan, err := planner.Plan1D32(32)
			if err != nil {
				t.Errorf("Plan1D32(32) with mode %v failed: %v", mode, err)
			} else if plan.Len() != 32 {
				t.Errorf("plan.Len() = %d, want 32 for mode %v", plan.Len(), mode)
			}
		}
	}
}

func TestPlanner_InvalidSizes(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	// Test invalid 1D size
	_, err := planner.Plan1D32(0)
	if err == nil {
		t.Error("Plan1D32(0) should return error")
	}

	// Test invalid 2D sizes
	_, err = planner.Plan2D32(0, 16)
	if err == nil {
		t.Error("Plan2D32(0, 16) should return error")
	}

	_, err = planner.Plan2D32(16, 0)
	if err == nil {
		t.Error("Plan2D32(16, 0) should return error")
	}

	// Test invalid 3D sizes
	_, err = planner.Plan3D32(0, 8, 8)
	if err == nil {
		t.Error("Plan3D32(0, 8, 8) should return error")
	}

	// Test invalid ND sizes
	_, err = planner.PlanND32([]int{8, 0, 8})
	if err == nil {
		t.Error("PlanND32 with zero dimension should return error")
	}

	// Test invalid real FFT size
	_, err = planner.PlanReal(-1)
	if err == nil {
		t.Error("PlanReal(-1) should return error")
	}
}

func TestPlanner_OptionsNormalization(t *testing.T) {
	t.Parallel()

	// Test that invalid options are normalized
	opts := PlanOptions{
		Batch:   -5,                    // Invalid: should be normalized to 0
		Stride:  -10,                   // Invalid: should be normalized to 0
		Radices: []int{2, -1, 4, 0, 8}, // Invalid entries should be removed
	}

	planner := NewPlanner(opts)

	// Batch should be normalized
	if planner.opts.Batch < 0 {
		t.Errorf("Batch = %d, should be >= 0 after normalization", planner.opts.Batch)
	}

	// Stride should be normalized
	if planner.opts.Stride < 0 {
		t.Errorf("Stride = %d, should be >= 0 after normalization", planner.opts.Stride)
	}

	// Radices should only contain valid values (> 1)
	for _, r := range planner.opts.Radices {
		if r <= 1 {
			t.Errorf("Radices contains invalid value %d, should be > 1", r)
		}
	}

	// Should only have 2, 4, 8 from the original list
	if len(planner.opts.Radices) != 3 {
		t.Errorf("Radices length = %d, want 3 (valid entries)", len(planner.opts.Radices))
	}
}

func TestPlanner_AllInvalidRadices(t *testing.T) {
	t.Parallel()

	// Test that all invalid radices results in nil slice
	opts := PlanOptions{
		Radices: []int{-1, 0, 1}, // All invalid
	}

	planner := NewPlanner(opts)

	// Should be cleared to nil
	if planner.opts.Radices != nil {
		t.Errorf("Radices = %v, want nil after removing all invalid entries", planner.opts.Radices)
	}
}

func TestPlanner_ConcurrentPlanCreation(t *testing.T) {
	t.Parallel()

	planner := NewPlanner(PlanOptions{})

	// Create multiple plans concurrently
	done := make(chan bool, 3)

	go func() {
		plan, err := planner.Plan1D32(64)
		if err != nil || plan.Len() != 64 {
			t.Errorf("Concurrent Plan1D32 failed")
		}

		done <- true
	}()

	go func() {
		plan, err := planner.Plan2D64(8, 16)
		if err != nil || plan.Len() != 8*16 {
			t.Errorf("Concurrent Plan2D64 failed")
		}

		done <- true
	}()

	go func() {
		plan, err := planner.PlanReal(128)
		if err != nil || plan.Len() != 128 {
			t.Errorf("Concurrent PlanReal failed")
		}

		done <- true
	}()

	<-done
	<-done
	<-done
}
