package algoforge

import (
	"testing"
)

func TestPlanAlgorithm(t *testing.T) {
	// Test that plans for codelet sizes report the correct algorithm
	sizes := []int{8, 16, 32, 64, 128}

	for _, size := range sizes {
		plan, err := NewPlan32(size)
		if err != nil {
			t.Fatalf("NewPlan32(%d) failed: %v", size, err)
		}

		algo := plan.Algorithm()
		if algo == "" {
			t.Errorf("size %d: expected non-empty algorithm", size)
		}

		// Should be a DIT codelet
		expected := "dit" + itoa(size) + "_generic"
		if algo != expected {
			t.Errorf("size %d: expected algorithm %q, got %q", size, expected, algo)
		}
	}
}

func TestPlanWithFlagsWisdom(t *testing.T) {
	// Clear wisdom first
	ClearWisdom()

	// Create plan with wisdom saving
	plan, err := NewPlanWithFlags[complex64](64, FlagSaveWisdom)
	if err != nil {
		t.Fatalf("NewPlanWithFlags failed: %v", err)
	}

	if plan.Algorithm() == "" {
		t.Error("expected non-empty algorithm")
	}

	// Check wisdom was saved
	if WisdomLen() != 1 {
		t.Errorf("expected 1 wisdom entry, got %d", WisdomLen())
	}

	// Create another plan with wisdom lookup
	plan2, err := NewPlanWithFlags[complex64](64, FlagUseWisdom)
	if err != nil {
		t.Fatalf("NewPlanWithFlags with wisdom failed: %v", err)
	}

	// Should have same algorithm
	if plan.Algorithm() != plan2.Algorithm() {
		t.Errorf("expected same algorithm from wisdom, got %q vs %q",
			plan.Algorithm(), plan2.Algorithm())
	}

	// Clean up
	ClearWisdom()
}

func TestPlanCodeletPath(t *testing.T) {
	// Verify the codelet path is used for supported sizes
	sizes := []int{8, 16, 32, 64, 128}

	for _, size := range sizes {
		plan, err := NewPlan32(size)
		if err != nil {
			t.Fatalf("NewPlan32(%d) failed: %v", size, err)
		}

		// Test forward transform
		src := make([]complex64, size)
		dst := make([]complex64, size)
		src[0] = 1 // Impulse

		if err := plan.Forward(dst, src); err != nil {
			t.Errorf("size %d: Forward failed: %v", size, err)
		}

		// FFT of impulse should be all ones (1+0i)
		for i, v := range dst {
			if real(v) < 0.99 || real(v) > 1.01 {
				t.Errorf("size %d: dst[%d] = %v, expected ~1+0i", size, i, v)
			}
			if imag(v) < -0.01 || imag(v) > 0.01 {
				t.Errorf("size %d: dst[%d] = %v, imaginary part should be ~0", size, i, v)
			}
		}

		// Test inverse transform
		for i := range dst {
			src[i] = dst[i]
		}

		if err := plan.Inverse(dst, src); err != nil {
			t.Errorf("size %d: Inverse failed: %v", size, err)
		}

		// IFFT should recover impulse (1+0i at index 0, ~0 elsewhere)
		if real(dst[0]) < 0.99 || real(dst[0]) > 1.01 {
			t.Errorf("size %d: dst[0] = %v, expected ~1+0i", size, dst[0])
		}
		if imag(dst[0]) < -0.01 || imag(dst[0]) > 0.01 {
			t.Errorf("size %d: dst[0] = %v, imaginary part should be ~0", size, dst[0])
		}

		for i := 1; i < size; i++ {
			if real(dst[i]) > 0.01 || real(dst[i]) < -0.01 {
				t.Errorf("size %d: dst[%d] = %v, expected ~0", size, i, dst[i])
			}
			if imag(dst[i]) > 0.01 || imag(dst[i]) < -0.01 {
				t.Errorf("size %d: dst[%d] = %v, imaginary part should be ~0", size, i, dst[i])
			}
		}
	}
}

func TestPlanFallbackPath(t *testing.T) {
	// Sizes without codelets should use fallback kernel
	sizes := []int{256, 512, 1024}

	for _, size := range sizes {
		plan, err := NewPlan32(size)
		if err != nil {
			t.Fatalf("NewPlan32(%d) failed: %v", size, err)
		}

		algo := plan.Algorithm()
		if algo == "" {
			t.Errorf("size %d: expected non-empty algorithm for fallback", size)
		}

		// Should not be a codelet (those are dit8, dit16, etc.)
		// For 256, 512, 1024 it should use DIT fallback
		if algo != "dit_fallback" && algo != "stockham" {
			t.Logf("size %d: algorithm = %q", size, algo)
		}
	}
}
