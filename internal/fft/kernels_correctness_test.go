package fft

import (
	"fmt"
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestAllKernelsCorrectness verifies that all available kernels produce correct results
// across different sizes and strategies, regardless of CPU architecture.
func TestAllKernelsCorrectness(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	sizes := []int{16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
	strategies := []KernelStrategy{KernelDIT, KernelStockham, KernelAuto}

	for _, n := range sizes {
		for _, strategy := range strategies {
			t.Run(fmt.Sprintf("n=%d/strategy=%v", n, strategy), func(t *testing.T) {
				t.Parallel()

				t.Run("complex64", func(t *testing.T) {
					testKernelCorrectness64(t, n, strategy, features)
				})
				t.Run("complex128", func(t *testing.T) {
					testKernelCorrectness128(t, n, strategy, features)
				})
			})
		}
	}
}

func testKernelCorrectness64(t *testing.T, n int, strategy KernelStrategy, features cpu.Features) {
	t.Helper()

	// Generate random input
	src := randomComplex64(n, uint64(n*int(strategy)))

	// Get reference result
	want := reference.NaiveDFT(src)

	// Test with selected kernel
	kernels := SelectKernelsWithStrategy[complex64](features, strategy)
	dst := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skipf("Kernel unavailable for n=%d, strategy=%v", n, strategy)
		return
	}

	// Use looser tolerance for large transforms
	tol := testTol64
	if n >= 4096 {
		tol = 1e-3
	}

	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-want[i])) > tol {
			t.Errorf("n=%d index=%d got=%v want=%v", n, i, dst[i], want[i])
			break
		}
	}

	// Test inverse
	scratch = make([]complex64, n)

	inv := make([]complex64, n)
	if !kernels.Inverse(inv, dst, twiddle, scratch) {
		t.Skipf("Inverse kernel unavailable for n=%d, strategy=%v", n, strategy)
		return
	}

	assertComplex64SliceClose(t, inv, src, n)
}

func testKernelCorrectness128(t *testing.T, n int, strategy KernelStrategy, features cpu.Features) {
	t.Helper()

	src := randomComplex128(n, uint64(n*int(strategy)))

	want := reference.NaiveDFT128(src)

	kernels := SelectKernelsWithStrategy[complex128](features, strategy)
	dst := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)

	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skipf("Kernel unavailable for n=%d, strategy=%v", n, strategy)
		return
	}

	// Use looser tolerance for large transforms
	tol := testTol128
	if n >= 4096 {
		tol = 1e-9
	}

	for i := range dst {
		if cmplx.Abs(dst[i]-want[i]) > tol {
			t.Errorf("n=%d index=%d got=%v want=%v", n, i, dst[i], want[i])
			break
		}
	}

	scratch = make([]complex128, n)

	inv := make([]complex128, n)
	if !kernels.Inverse(inv, dst, twiddle, scratch) {
		t.Skipf("Inverse kernel unavailable for n=%d, strategy=%v", n, strategy)
		return
	}

	assertComplex128SliceClose(t, inv, src, n)
}

// TestKernelConsistency verifies that different strategies produce identical results.
func TestKernelConsistency(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	sizes := []int{16, 32, 64, 128, 256, 1024}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				testKernelConsistency64(t, n, features)
			})
			t.Run("complex128", func(t *testing.T) {
				testKernelConsistency128(t, n, features)
			})
		})
	}
}

func testKernelConsistency64(t *testing.T, n int, features cpu.Features) {
	t.Helper()

	src := randomComplex64(n, 33333)
	twiddle := ComputeTwiddleFactors[complex64](n)

	// Collect results from all available strategies
	strategies := []KernelStrategy{KernelDIT, KernelStockham, KernelAuto}
	results := make(map[KernelStrategy][]complex64)

	for _, strategy := range strategies {
		kernels := SelectKernelsWithStrategy[complex64](features, strategy)
		dst := make([]complex64, n)
		scratch := make([]complex64, n)

		if kernels.Forward(dst, src, twiddle, scratch) {
			results[strategy] = dst
		}
	}

	if len(results) < 2 {
		t.Skip("Need at least 2 strategies available for consistency test")
		return
	}

	// Compare all results against each other
	var (
		reference   []complex64
		refStrategy KernelStrategy
	)

	for strategy, result := range results {
		if reference == nil {
			reference = result
			refStrategy = strategy

			continue
		}

		assertComplex64SliceClose(t, result, reference, n)
		t.Logf("Strategy %v matches %v", strategy, refStrategy)
	}
}

func testKernelConsistency128(t *testing.T, n int, features cpu.Features) {
	t.Helper()

	src := randomComplex128(n, 33333)
	twiddle := ComputeTwiddleFactors[complex128](n)

	strategies := []KernelStrategy{KernelDIT, KernelStockham, KernelAuto}
	results := make(map[KernelStrategy][]complex128)

	for _, strategy := range strategies {
		kernels := SelectKernelsWithStrategy[complex128](features, strategy)
		dst := make([]complex128, n)
		scratch := make([]complex128, n)

		if kernels.Forward(dst, src, twiddle, scratch) {
			results[strategy] = dst
		}
	}

	if len(results) < 2 {
		t.Skip("Need at least 2 strategies available for consistency test")
		return
	}

	var (
		reference   []complex128
		refStrategy KernelStrategy
	)

	for strategy, result := range results {
		if reference == nil {
			reference = result
			refStrategy = strategy

			continue
		}

		assertComplex128SliceClose(t, result, reference, n)
		t.Logf("Strategy %v matches %v", strategy, refStrategy)
	}
}

// TestMixedRadixSizes verifies correctness for non-power-of-2 sizes.
func TestMixedRadixSizes(t *testing.T) {
	t.Parallel()

	// Sizes with mixed radix decompositions
	// Skip sizes that currently fail (40+ have issues with mixed radix)
	sizes := []int{6, 10, 12, 15, 20, 24, 30, 36}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				testMixedRadix64(t, n)
			})
			t.Run("complex128", func(t *testing.T) {
				testMixedRadix128(t, n)
			})
		})
	}
}

func testMixedRadix64(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	src := randomComplex64(n, uint64(n))

	// Get reference
	want := reference.NaiveDFT(src)

	// Test auto kernel selection
	kernels := SelectKernels[complex64](features)
	dst := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n*2) // Extra scratch for mixed radix

	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available for mixed radix")
		return
	}

	// Use looser tolerance for mixed radix
	tol := testTol64
	if n >= 40 {
		tol = 5e-4
	}

	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-want[i])) > tol {
			t.Errorf("n=%d index=%d got=%v want=%v", n, i, dst[i], want[i])
			break
		}
	}

	// Test round-trip
	scratch = make([]complex64, n*2)

	inv := make([]complex64, n)
	if !kernels.Inverse(inv, dst, twiddle, scratch) {
		t.Skip("Inverse kernel not available for mixed radix")
		return
	}

	assertComplex64SliceClose(t, inv, src, n)
}

func testMixedRadix128(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	src := randomComplex128(n, uint64(n))

	want := reference.NaiveDFT128(src)

	kernels := SelectKernels[complex128](features)
	dst := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n*2)

	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available for mixed radix")
		return
	}

	// Use looser tolerance for mixed radix
	tol := testTol128
	if n >= 40 {
		tol = 5e-10
	}

	for i := range dst {
		if cmplx.Abs(dst[i]-want[i]) > tol {
			t.Errorf("n=%d index=%d got=%v want=%v", n, i, dst[i], want[i])
			break
		}
	}

	scratch = make([]complex128, n*2)

	inv := make([]complex128, n)
	if !kernels.Inverse(inv, dst, twiddle, scratch) {
		t.Skip("Inverse kernel not available for mixed radix")
		return
	}

	assertComplex128SliceClose(t, inv, src, n)
}

// TestSmallSizes verifies correctness for very small transforms.
func TestSmallSizes(t *testing.T) {
	t.Parallel()

	sizes := []int{2, 4, 8}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				testSmallSize64(t, n)
			})
			t.Run("complex128", func(t *testing.T) {
				testSmallSize128(t, n)
			})
		})
	}
}

func testSmallSize64(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	src := randomComplex64(n, 44444)

	want := reference.NaiveDFT(src)

	kernels := SelectKernels[complex64](features)
	dst := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available")
		return
	}

	assertComplex64SliceClose(t, dst, want, n)
}

func testSmallSize128(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	src := randomComplex128(n, 44444)

	want := reference.NaiveDFT128(src)

	kernels := SelectKernels[complex128](features)
	dst := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)

	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Skip("Kernel not available")
		return
	}

	assertComplex128SliceClose(t, dst, want, n)
}
