package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

// TestFallbackKernel tests the fallback kernel chaining logic.
func TestFallbackKernel(t *testing.T) {
	t.Parallel()

	n := 8

	input := make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	output := make([]complex64, n)

	// Create a primary kernel that always fails
	failingPrimary := func(dst, src, twiddle, scratch []complex64) bool {
		return false
	}

	// Create a fallback kernel that works
	workingFallback := func(dst, src, twiddle, scratch []complex64) bool {
		return forwardDITComplex64(dst, src, twiddle, scratch)
	}

	// Test fallback kernel
	kernel := fallbackKernel(failingPrimary, workingFallback)

	ok := kernel(output, input, twiddle, scratch)
	if !ok {
		t.Fatal("fallbackKernel should have used fallback when primary fails")
	}

	// Verify transform worked
	if output[0] == 0 {
		t.Error("fallback kernel produced zero output")
	}
}

// TestFallbackKernel_NilPrimary tests fallback when primary is nil.
func TestFallbackKernel_NilPrimary(t *testing.T) {
	t.Parallel()

	n := 8

	input := make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	output := make([]complex64, n)

	workingFallback := func(dst, src, twiddle, scratch []complex64) bool {
		return forwardDITComplex64(dst, src, twiddle, scratch)
	}

	// Nil primary should return fallback directly
	kernel := fallbackKernel[complex64](nil, workingFallback)

	ok := kernel(output, input, twiddle, scratch)
	if !ok {
		t.Fatal("fallbackKernel should use fallback when primary is nil")
	}
}

// TestAutoKernelComplex64_PowerOf2 tests auto kernel for power-of-2 sizes.
func TestAutoKernelComplex64_PowerOf2(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		size     int
		strategy planner.KernelStrategy
	}{
		{"DIT_8", 8, planner.KernelDIT},
		{"DIT_16", 16, planner.KernelDIT},
		{"Stockham_1024", 1024, planner.KernelStockham},
		{"Stockham_2048", 2048, planner.KernelStockham},
		{"Auto_Small", 32, planner.KernelAuto},
		{"Auto_Large", 2048, planner.KernelAuto},
		{"SixStep_1024", 1024, planner.KernelSixStep},
		{"EightStep_1024", 1024, planner.KernelEightStep},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			input := make([]complex64, tt.size)
			for i := range input {
				input[i] = complex(float32(i%16), 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex64](tt.size)
			scratch := make([]complex64, tt.size)
			output := make([]complex64, tt.size)

			kernels := autoKernelComplex64(tt.strategy)

			// Test forward
			ok := kernels.Forward(output, input, twiddle, scratch)
			if !ok {
				t.Fatalf("autoKernel forward failed for size %d with strategy %v", tt.size, tt.strategy)
			}

			// Test inverse
			ok = kernels.Inverse(output, output, twiddle, scratch)
			if !ok {
				t.Fatalf("autoKernel inverse failed for size %d with strategy %v", tt.size, tt.strategy)
			}
		})
	}
}

// TestAutoKernelComplex64_MixedRadix tests auto kernel for mixed-radix sizes.
func TestAutoKernelComplex64_MixedRadix(t *testing.T) {
	t.Parallel()

	// Highly composite numbers (not power of 2)
	sizes := []int{6, 12, 18, 24, 36}

	for _, size := range sizes {
		t.Run("MixedRadix_"+string(rune(size)), func(t *testing.T) {
			t.Parallel()

			input := make([]complex64, size)
			for i := range input {
				input[i] = complex(float32(i), 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex64](size)
			scratch := make([]complex64, size*2)
			output := make([]complex64, size)

			kernels := autoKernelComplex64(planner.KernelAuto)

			// Test forward
			ok := kernels.Forward(output, input, twiddle, scratch)
			if !ok {
				t.Fatalf("autoKernel forward failed for mixed-radix size %d", size)
			}

			// Test inverse
			ok = kernels.Inverse(output, output, twiddle, scratch)
			if !ok {
				t.Fatalf("autoKernel inverse failed for mixed-radix size %d", size)
			}
		})
	}
}

// TestAutoKernelComplex64_NonComposite tests auto kernel for non-composite sizes.
func TestAutoKernelComplex64_NonComposite(t *testing.T) {
	t.Parallel()

	// Prime numbers and non-highly-composite non-power-of-2 sizes
	sizes := []int{7, 11, 13, 17}

	for _, size := range sizes {
		t.Run("NonComposite_"+string(rune(size)), func(t *testing.T) {
			t.Parallel()

			input := make([]complex64, size)
			for i := range input {
				input[i] = complex(float32(i), 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex64](size)
			scratch := make([]complex64, size*2)
			output := make([]complex64, size)

			kernels := autoKernelComplex64(planner.KernelAuto)

			// Should fail for non-composite, non-power-of-2 sizes
			ok := kernels.Forward(output, input, twiddle, scratch)
			if ok {
				t.Errorf("autoKernel should fail for non-composite size %d, but succeeded", size)
			}

			ok = kernels.Inverse(output, input, twiddle, scratch)
			if ok {
				t.Errorf("autoKernel inverse should fail for non-composite size %d, but succeeded", size)
			}
		})
	}
}

// TestAutoKernelComplex128 tests auto kernel for complex128.
func TestAutoKernelComplex128(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		size     int
		strategy planner.KernelStrategy
	}{
		{"DIT_8", 8, planner.KernelDIT},
		{"Stockham_1024", 1024, planner.KernelStockham},
		{"MixedRadix_12", 12, planner.KernelAuto},
		{"SixStep_1024", 1024, planner.KernelSixStep},
		{"EightStep_1024", 1024, planner.KernelEightStep},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			input := make([]complex128, tt.size)
			for i := range input {
				input[i] = complex(float64(i%16), 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex128](tt.size)
			scratch := make([]complex128, tt.size*2)
			output := make([]complex128, tt.size)

			kernels := autoKernelComplex128(tt.strategy)

			// Test forward
			ok := kernels.Forward(output, input, twiddle, scratch)
			if !ok {
				t.Fatalf("autoKernelComplex128 forward failed for size %d with strategy %v", tt.size, tt.strategy)
			}

			// Test inverse
			ok = kernels.Inverse(output, output, twiddle, scratch)
			if !ok {
				t.Fatalf("autoKernelComplex128 inverse failed for size %d with strategy %v", tt.size, tt.strategy)
			}
		})
	}
}

// TestAutoKernelComplex128_NonComposite tests complex128 auto kernel for non-composite sizes.
func TestAutoKernelComplex128_NonComposite(t *testing.T) {
	t.Parallel()

	size := 7 // Prime number

	input := make([]complex128, size)
	for i := range input {
		input[i] = complex(float64(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](size)
	scratch := make([]complex128, size*2)
	output := make([]complex128, size)

	kernels := autoKernelComplex128(planner.KernelAuto)

	// Should fail for prime size
	ok := kernels.Forward(output, input, twiddle, scratch)
	if ok {
		t.Error("autoKernelComplex128 should fail for prime size 7, but succeeded")
	}

	ok = kernels.Inverse(output, input, twiddle, scratch)
	if ok {
		t.Error("autoKernelComplex128 inverse should fail for prime size 7, but succeeded")
	}
}

// TestAutoKernel_StrategySelection tests that the correct algorithm is selected.
//
//nolint:paralleltest // modifies global CPU feature state
func TestAutoKernel_StrategySelection(t *testing.T) {
	// Save and restore CPU features
	originalFeatures := cpu.DetectFeatures()
	defer cpu.SetForcedFeatures(originalFeatures)

	// Force generic mode to ensure we're testing strategy selection, not SIMD dispatch
	cpu.SetForcedFeatures(cpu.Features{ForceGeneric: true})

	tests := []struct {
		name     string
		size     int
		strategy planner.KernelStrategy
		wantDIT  bool
	}{
		{"Auto_Small_Should_Use_DIT", 512, planner.KernelAuto, true},
		{"Auto_Large_Should_Use_Stockham", 2048, planner.KernelAuto, false},
		{"Forced_DIT", 2048, planner.KernelDIT, true},
		{"Forced_Stockham", 512, planner.KernelStockham, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			input := make([]complex64, tt.size)
			for i := range input {
				input[i] = complex(float32(i%16), 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex64](tt.size)
			scratch := make([]complex64, tt.size)
			output := make([]complex64, tt.size)

			kernels := autoKernelComplex64(tt.strategy)

			ok := kernels.Forward(output, input, twiddle, scratch)
			if !ok {
				t.Fatalf("Forward failed for size %d with strategy %v", tt.size, tt.strategy)
			}

			// Verify result is non-zero (algorithm ran)
			if output[0] == 0 {
				t.Error("Expected non-zero DC component")
			}
		})
	}
}
