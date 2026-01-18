package algofft

import (
	"errors"
	"testing"
)

func TestNewPlanPooled_Complex64(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanPooled[complex64](1024)
	if err != nil {
		t.Fatalf("NewPlanPooled failed: %v", err)
	}
	defer plan.Close()

	if plan.Len() != 1024 {
		t.Errorf("expected length 1024, got %d", plan.Len())
	}

	// Verify transform works
	src := make([]complex64, 1024)
	dst := make([]complex64, 1024)
	src[0] = 1 // impulse

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
}

func TestNewPlanPooled_Complex128(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanPooled[complex128](1024)
	if err != nil {
		t.Fatalf("NewPlanPooled failed: %v", err)
	}
	defer plan.Close()

	if plan.Len() != 1024 {
		t.Errorf("expected length 1024, got %d", plan.Len())
	}

	// Verify transform works
	src := make([]complex128, 1024)
	dst := make([]complex128, 1024)
	src[0] = 1 // impulse

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
}

func TestPlanPooled_BufferReuse(t *testing.T) {
	t.Parallel()

	const size = 256

	// Create and close multiple plans to populate the pool
	for range 5 {
		plan, err := NewPlanPooled[complex64](size)
		if err != nil {
			t.Fatalf("NewPlanPooled failed: %v", err)
		}

		plan.Close()
	}

	// Create a new plan - should reuse pooled buffers
	plan, err := NewPlanPooled[complex64](size)
	if err != nil {
		t.Fatalf("NewPlanPooled failed: %v", err)
	}
	defer plan.Close()

	// Verify it works correctly
	src := make([]complex64, size)
	dst := make([]complex64, size)
	src[0] = 1

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
}

func TestPlan_Reset(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](64)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	// Perform a transform to populate scratch
	src := make([]complex64, 64)
	dst := make([]complex64, 64)
	src[0] = 1

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Reset should not cause any errors
	plan.Reset()

	// Plan should still work after reset
	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward after Reset failed: %v", err)
	}
}

func TestPlan_Close_NonPooled(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](64)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	// Close on non-pooled plan should be a no-op
	plan.Close()

	// Multiple closes should be safe
	plan.Close()
}

func TestPlan_Close_Pooled(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanPooled[complex64](64)
	if err != nil {
		t.Fatalf("NewPlanPooled failed: %v", err)
	}

	// Close should return buffers to pool
	plan.Close()

	// Multiple closes should be safe
	plan.Close()
}

func TestPlanPooled_InvalidLength(t *testing.T) {
	t.Parallel()

	_, err := NewPlanPooled[complex64](98) // Includes unsupported prime factor
	if !errors.Is(err, ErrInvalidLength) {
		t.Errorf("expected ErrInvalidLength, got %v", err)
	}
}

func TestPlan_String_Complex64(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](256)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	s := plan.String()
	if s == "" {
		t.Error("String() returned empty string")
	}

	// Should contain type and size
	if !contains(s, "complex64") {
		t.Errorf("String() should contain 'complex64', got: %s", s)
	}

	if !contains(s, "256") {
		t.Errorf("String() should contain '256', got: %s", s)
	}
}

func TestPlan_String_Complex128(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex128](512)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	s := plan.String()

	if !contains(s, "complex128") {
		t.Errorf("String() should contain 'complex128', got: %s", s)
	}

	if !contains(s, "512") {
		t.Errorf("String() should contain '512', got: %s", s)
	}
}

func TestPlan_String_Pooled(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanPooled[complex64](128)
	if err != nil {
		t.Fatalf("NewPlanPooled failed: %v", err)
	}
	defer plan.Close()

	s := plan.String()

	if !contains(s, "pooled") {
		t.Errorf("String() for pooled plan should contain 'pooled', got: %s", s)
	}
}

func TestPlan_Clone(t *testing.T) {
	t.Parallel()

	original, err := NewPlanT[complex64](256)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	clone := original.Clone()

	// Verify clone has same properties
	if clone.Len() != original.Len() {
		t.Errorf("Clone Len mismatch: got %d, want %d", clone.Len(), original.Len())
	}

	// Verify clone produces same results
	src := make([]complex64, 256)
	src[0] = 1 // impulse

	dstOriginal := make([]complex64, 256)
	dstClone := make([]complex64, 256)

	err = original.Forward(dstOriginal, src)
	if err != nil {
		t.Fatalf("original.Forward failed: %v", err)
	}

	err = clone.Forward(dstClone, src)
	if err != nil {
		t.Fatalf("clone.Forward failed: %v", err)
	}

	for i := range dstOriginal {
		if dstOriginal[i] != dstClone[i] {
			t.Errorf("output mismatch at %d: got %v, want %v", i, dstClone[i], dstOriginal[i])
			break
		}
	}
}

func TestPlan_Clone_Independent(t *testing.T) {
	t.Parallel()

	original, err := NewPlanT[complex64](64)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	clone := original.Clone()

	// Close on clone should be no-op (not pooled)
	clone.Close()

	// Original should still work
	src := make([]complex64, 64)
	dst := make([]complex64, 64)
	src[0] = 1

	err = original.Forward(dst, src)
	if err != nil {
		t.Fatalf("original.Forward after clone.Close failed: %v", err)
	}
}

func TestPlan_Clone_Complex128(t *testing.T) {
	t.Parallel()

	original, err := NewPlanT[complex128](128)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	clone := original.Clone()

	if clone.Len() != original.Len() {
		t.Errorf("Clone Len mismatch: got %d, want %d", clone.Len(), original.Len())
	}

	// Verify transform works
	src := make([]complex128, 128)
	dst := make([]complex128, 128)
	src[0] = 1

	err = clone.Forward(dst, src)
	if err != nil {
		t.Fatalf("clone.Forward failed: %v", err)
	}
}

// contains checks if substr is in s (simple implementation to avoid strings import).
func contains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}

	return false
}

func BenchmarkPooledVsRegular(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		// Warm up pool for this size
		for range 5 {
			plan, _ := NewPlanPooled[complex64](size)
			plan.Close()
		}

		b.Run("Pooled/"+itoa(size), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				plan, _ := NewPlanPooled[complex64](size)
				plan.Close()
			}
		})

		b.Run("Regular/"+itoa(size), func(b *testing.B) {
			b.ReportAllocs()

			for b.Loop() {
				_, _ = NewPlanT[complex64](size)
			}
		})
	}
}
