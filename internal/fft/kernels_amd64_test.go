//go:build amd64

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

// TestSelectKernelsComplex64_SSE2Only tests SSE2 path without AVX2.
//
//nolint:paralleltest // modifies global CPU feature state
func TestSelectKernelsComplex64_SSE2Only(t *testing.T) {
	// Save and restore CPU features
	originalFeatures := cpu.DetectFeatures()
	defer cpu.SetForcedFeatures(originalFeatures)

	// Force SSE2 only (no AVX2)
	cpu.SetForcedFeatures(cpu.Features{HasSSE2: true, HasAVX2: false})

	kernels := selectKernelsComplex64(cpu.DetectFeatures())

	// Test forward
	n := 16

	input := make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	output := make([]complex64, n)

	ok := kernels.Forward(output, input, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("SSE2 forward kernel failed")
	}

	// Test inverse
	ok = kernels.Inverse(output, output, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("SSE2 inverse kernel failed")
	}
}

// TestSelectKernelsComplex128_SSE2Only tests SSE2 path without AVX2 for complex128.
//
//nolint:paralleltest // modifies global CPU feature state
func TestSelectKernelsComplex128_SSE2Only(t *testing.T) {
	originalFeatures := cpu.DetectFeatures()
	defer cpu.SetForcedFeatures(originalFeatures)

	cpu.SetForcedFeatures(cpu.Features{HasSSE2: true, HasAVX2: false})

	kernels := selectKernelsComplex128(cpu.DetectFeatures())

	n := 16

	input := make([]complex128, n)
	for i := range input {
		input[i] = complex(float64(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	output := make([]complex128, n)

	ok := kernels.Forward(output, input, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("SSE2 complex128 forward kernel failed")
	}

	ok = kernels.Inverse(output, output, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("SSE2 complex128 inverse kernel failed")
	}
}

// TestSelectKernelsWithStrategy_SSE2 tests strategy selection with SSE2.
//
//nolint:paralleltest // modifies global CPU feature state
func TestSelectKernelsWithStrategy_SSE2(t *testing.T) {
	originalFeatures := cpu.DetectFeatures()
	defer cpu.SetForcedFeatures(originalFeatures)

	cpu.SetForcedFeatures(cpu.Features{HasSSE2: true, HasAVX2: false})

	tests := []struct {
		name     string
		strategy planner.KernelStrategy
	}{
		{"DIT_Strategy", planner.KernelDIT},
		{"Stockham_Strategy", planner.KernelStockham},
		{"Auto_Strategy", planner.KernelAuto},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			kernels := selectKernelsComplex64WithStrategy(cpu.DetectFeatures(), tt.strategy)

			n := 32

			input := make([]complex64, n)
			for i := range input {
				input[i] = complex(float32(i%8), 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
			bitrev := mathpkg.ComputeBitReversalIndices(n)
			scratch := make([]complex64, n)
			output := make([]complex64, n)

			ok := kernels.Forward(output, input, twiddle, scratch, bitrev)
			if !ok {
				t.Fatalf("SSE2 forward failed with strategy %v", tt.strategy)
			}

			ok = kernels.Inverse(output, output, twiddle, scratch, bitrev)
			if !ok {
				t.Fatalf("SSE2 inverse failed with strategy %v", tt.strategy)
			}
		})
	}
}

// TestForwardSSE2Complex64 tests SSE2 complex64 forward kernel directly.
func TestForwardSSE2Complex64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		size int
	}{
		{"Small_DIT", 16},        // Should use DIT
		{"Large_Stockham", 2048}, // Should use Stockham
		{"Medium", 512},          // Around threshold
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			input := make([]complex64, tt.size)
			for i := range input {
				input[i] = complex(float32(i%16), 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex64](tt.size)
			bitrev := mathpkg.ComputeBitReversalIndices(tt.size)
			scratch := make([]complex64, tt.size)
			output := make([]complex64, tt.size)

			ok := forwardSSE2Complex64(output, input, twiddle, scratch, bitrev)
			if !ok {
				t.Fatalf("forwardSSE2Complex64 failed for size %d", tt.size)
			}

			// Verify non-zero output
			if output[0] == 0 {
				t.Error("Expected non-zero DC component")
			}
		})
	}
}

// TestInverseSSE2Complex64 tests SSE2 complex64 inverse kernel directly.
func TestInverseSSE2Complex64(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		size int
	}{
		{"Small_DIT", 16},
		{"Large_Stockham", 2048},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			freq := make([]complex64, tt.size)
			freq[0] = complex(float32(tt.size), 0)

			twiddle := mathpkg.ComputeTwiddleFactors[complex64](tt.size)
			bitrev := mathpkg.ComputeBitReversalIndices(tt.size)
			scratch := make([]complex64, tt.size)
			output := make([]complex64, tt.size)

			ok := inverseSSE2Complex64(output, freq, twiddle, scratch, bitrev)
			if !ok {
				t.Fatalf("inverseSSE2Complex64 failed for size %d", tt.size)
			}

			// First element should be approximately 1 (size/size)
			if real(output[0]) < 0.9 || real(output[0]) > 1.1 {
				t.Errorf("Expected output[0] ≈ 1, got %v", output[0])
			}
		})
	}
}

// TestForwardSSE2Complex128 tests SSE2 complex128 forward kernel.
func TestForwardSSE2Complex128(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 512, 2048}

	for _, size := range sizes {
		t.Run("Size_"+string(rune(size)), func(t *testing.T) {
			t.Parallel()

			input := make([]complex128, size)
			for i := range input {
				input[i] = complex(float64(i%16), 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex128](size)
			bitrev := mathpkg.ComputeBitReversalIndices(size)
			scratch := make([]complex128, size)
			output := make([]complex128, size)

			ok := forwardSSE2Complex128(output, input, twiddle, scratch, bitrev)
			if !ok {
				t.Fatalf("forwardSSE2Complex128 failed for size %d", size)
			}
		})
	}
}

// TestInverseSSE2Complex128 tests SSE2 complex128 inverse kernel.
func TestInverseSSE2Complex128(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 512, 2048}

	for _, size := range sizes {
		t.Run("Size_"+string(rune(size)), func(t *testing.T) {
			t.Parallel()

			freq := make([]complex128, size)
			freq[0] = complex(float64(size), 0)

			twiddle := mathpkg.ComputeTwiddleFactors[complex128](size)
			bitrev := mathpkg.ComputeBitReversalIndices(size)
			scratch := make([]complex128, size)
			output := make([]complex128, size)

			ok := inverseSSE2Complex128(output, freq, twiddle, scratch, bitrev)
			if !ok {
				t.Fatalf("inverseSSE2Complex128 failed for size %d", size)
			}
		})
	}
}

// TestSSE2Kernels_NonPowerOf2 tests SSE2 kernels reject non-power-of-2 sizes.
func TestSSE2Kernels_NonPowerOf2(t *testing.T) {
	t.Parallel()

	size := 7 // Prime number

	input := make([]complex64, size)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](size)
	bitrev := mathpkg.ComputeBitReversalIndices(size)
	scratch := make([]complex64, size)
	output := make([]complex64, size)

	// All SSE2 kernels should reject non-power-of-2
	if ok := forwardSSE2Complex64(output, input, twiddle, scratch, bitrev); ok {
		t.Error("forwardSSE2Complex64 should reject non-power-of-2 size")
	}

	if ok := inverseSSE2Complex64(output, input, twiddle, scratch, bitrev); ok {
		t.Error("inverseSSE2Complex64 should reject non-power-of-2 size")
	}

	input128 := make([]complex128, size)
	output128 := make([]complex128, size)
	twiddle128 := mathpkg.ComputeTwiddleFactors[complex128](size)
	scratch128 := make([]complex128, size)

	if ok := forwardSSE2Complex128(output128, input128, twiddle128, scratch128, bitrev); ok {
		t.Error("forwardSSE2Complex128 should reject non-power-of-2 size")
	}

	if ok := inverseSSE2Complex128(output128, input128, twiddle128, scratch128, bitrev); ok {
		t.Error("inverseSSE2Complex128 should reject non-power-of-2 size")
	}
}

// TestAVX2Kernels_NonPowerOf2 tests AVX2 kernels reject non-power-of-2 sizes.
func TestAVX2Kernels_NonPowerOf2(t *testing.T) {
	t.Parallel()

	size := 7

	input := make([]complex64, size)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](size)
	bitrev := mathpkg.ComputeBitReversalIndices(size)
	scratch := make([]complex64, size)
	output := make([]complex64, size)

	// AVX2 kernels should also reject non-power-of-2
	if ok := forwardAVX2Complex64(output, input, twiddle, scratch, bitrev); ok {
		t.Error("forwardAVX2Complex64 should reject non-power-of-2 size")
	}

	if ok := inverseAVX2Complex64(output, input, twiddle, scratch, bitrev); ok {
		t.Error("inverseAVX2Complex64 should reject non-power-of-2 size")
	}

	if ok := forwardAVX2StockhamComplex64(output, input, twiddle, scratch, bitrev); ok {
		t.Error("forwardAVX2StockhamComplex64 should reject non-power-of-2 size")
	}

	if ok := inverseAVX2StockhamComplex64(output, input, twiddle, scratch, bitrev); ok {
		t.Error("inverseAVX2StockhamComplex64 should reject non-power-of-2 size")
	}

	// Complex128 variants
	input128 := make([]complex128, size)
	output128 := make([]complex128, size)
	twiddle128 := mathpkg.ComputeTwiddleFactors[complex128](size)
	scratch128 := make([]complex128, size)

	if ok := forwardAVX2Complex128(output128, input128, twiddle128, scratch128, bitrev); ok {
		t.Error("forwardAVX2Complex128 should reject non-power-of-2 size")
	}

	if ok := inverseAVX2Complex128(output128, input128, twiddle128, scratch128, bitrev); ok {
		t.Error("inverseAVX2Complex128 should reject non-power-of-2 size")
	}

	if ok := forwardAVX2StockhamComplex128(output128, input128, twiddle128, scratch128, bitrev); ok {
		t.Error("forwardAVX2StockhamComplex128 should reject non-power-of-2 size")
	}

	if ok := inverseAVX2StockhamComplex128(output128, input128, twiddle128, scratch128, bitrev); ok {
		t.Error("inverseAVX2StockhamComplex128 should reject non-power-of-2 size")
	}
}

// TestSelectKernelsComplex128WithStrategy_SSE2 tests complex128 strategy selection with SSE2.
//
//nolint:paralleltest // modifies global CPU feature state
func TestSelectKernelsComplex128WithStrategy_SSE2(t *testing.T) {
	originalFeatures := cpu.DetectFeatures()
	defer cpu.SetForcedFeatures(originalFeatures)

	cpu.SetForcedFeatures(cpu.Features{HasSSE2: true, HasAVX2: false})

	strategies := []planner.KernelStrategy{
		planner.KernelDIT,
		planner.KernelStockham,
		planner.KernelAuto,
	}

	for _, strategy := range strategies {
		t.Run("Strategy_"+string(rune(strategy)), func(t *testing.T) {
			kernels := selectKernelsComplex128WithStrategy(cpu.DetectFeatures(), strategy)

			n := 32

			input := make([]complex128, n)
			for i := range input {
				input[i] = complex(float64(i%8), 0)
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
			bitrev := mathpkg.ComputeBitReversalIndices(n)
			scratch := make([]complex128, n)
			output := make([]complex128, n)

			ok := kernels.Forward(output, input, twiddle, scratch, bitrev)
			if !ok {
				t.Fatalf("Forward failed with strategy %v", strategy)
			}

			ok = kernels.Inverse(output, output, twiddle, scratch, bitrev)
			if !ok {
				t.Fatalf("Inverse failed with strategy %v", strategy)
			}
		})
	}
}
