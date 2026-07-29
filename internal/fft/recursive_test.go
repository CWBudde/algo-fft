package fft

import (
	"math/cmplx"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
	"github.com/cwbudde/algo-fft/internal/registry"
	"github.com/cwbudde/algo-fft/internal/transform"
)

// TestRecursiveTransform tests the recursive FFT decomposition.
func TestRecursiveTransform(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 32, 64, 128}

	for _, n := range sizes {
		t.Run("complex64", func(t *testing.T) {
			testRecursive64(t, n)
		})
		t.Run("complex128", func(t *testing.T) {
			testRecursive128(t, n)
		})
	}
}

func testRecursive64(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	src := randomComplex64(n, uint64(n))
	want := reference.NaiveDFT(src)

	// Create decomposition strategy
	codeletSizes := []int{4, 8, 16, 32}
	cacheSize := 32768 // 32KB L1 cache

	strategy := transform.PlanDecomposition(n, codeletSizes, cacheSize)
	if strategy == nil {
		t.Skip("Cannot decompose size")
		return
	}

	// Get twiddle factors for recursive transform
	twiddle := transform.TwiddleFactorsRecursive[complex64](strategy)
	if len(twiddle) == 0 {
		t.Skip("No twiddle factors generated")
		return
	}

	// Allocate scratch space
	scratchSize := transform.ScratchSizeRecursive(strategy)
	scratch := make([]complex64, scratchSize)

	// Get registry
	registry := registry.GetRegistry[complex64]()

	// Test forward
	dst := make([]complex64, n)
	transform.RecursiveForward(dst, src, strategy, twiddle, scratch, registry, features)

	// Verify against reference
	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-want[i])) > testTol64 {
			t.Errorf("Forward: index %d: got %v, want %v", i, dst[i], want[i])
			break
		}
	}

	// Test inverse
	inv := make([]complex64, n)
	transform.RecursiveInverse(inv, dst, strategy, twiddle, scratch, registry, features)

	// Verify round-trip
	for i := range inv {
		if cmplx.Abs(complex128(inv[i]-src[i])) > testTol64 {
			t.Errorf("Inverse: index %d: got %v, want %v", i, inv[i], src[i])
			break
		}
	}
}

func testRecursive128(t *testing.T, n int) {
	t.Helper()

	features := cpu.DetectFeatures()
	src := randomComplex128(n, uint64(n))
	want := reference.NaiveDFT128(src)

	codeletSizes := []int{4, 8, 16, 32}
	cacheSize := 32768

	strategy := transform.PlanDecomposition(n, codeletSizes, cacheSize)
	if strategy == nil {
		t.Skip("Cannot decompose size")
		return
	}

	twiddle := transform.TwiddleFactorsRecursive[complex128](strategy)
	if len(twiddle) == 0 {
		t.Skip("No twiddle factors generated")
		return
	}

	scratchSize := transform.ScratchSizeRecursive(strategy)
	scratch := make([]complex128, scratchSize)

	registry := registry.GetRegistry[complex128]()

	dst := make([]complex128, n)
	transform.RecursiveForward(dst, src, strategy, twiddle, scratch, registry, features)

	for i := range dst {
		if cmplx.Abs(dst[i]-want[i]) > testTol128 {
			t.Errorf("Forward: index %d: got %v, want %v", i, dst[i], want[i])
			break
		}
	}

	inv := make([]complex128, n)
	transform.RecursiveInverse(inv, dst, strategy, twiddle, scratch, registry, features)

	for i := range inv {
		if cmplx.Abs(inv[i]-src[i]) > testTol128 {
			t.Errorf("Inverse: index %d: got %v, want %v", i, inv[i], src[i])
			break
		}
	}
}

// TestStockhamPacked tests the packed Stockham FFT variant.
func TestStockhamPacked(t *testing.T) {
	t.Parallel()

	// Test power-of-4 sizes which work well with radix-4
	sizes := []int{16, 64, 256}

	for _, n := range sizes {
		t.Run("complex64", func(t *testing.T) {
			testStockhamPacked64(t, n)
		})
		t.Run("complex128", func(t *testing.T) {
			testStockhamPacked128(t, n)
		})
	}
}

func testStockhamPacked64(t *testing.T, n int) {
	t.Helper()

	// No availability check: the packed engine is compiled in on every build.
	// This used to skip whenever the old dispatch toggle was off, which meant
	// it never ran on a SIMD build at all.
	src := randomComplex64(n, uint64(n+123))
	want := reference.NaiveDFT(src)

	// Prepare packed twiddles
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)

	packed := transform.ComputePackedTwiddles[complex64](n, 4, twiddle)
	if packed == nil {
		t.Skip("Failed to compute packed twiddles")
		return
	}

	// Test forward
	dst := make([]complex64, n)

	scratch := make([]complex64, n)
	if !transform.ForwardStockhamPacked(dst, src, twiddle, scratch, packed) {
		t.Skip("transform.ForwardStockhamPacked not implemented")
		return
	}

	// Verify against reference
	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-want[i])) > testTol64 {
			t.Errorf("Forward: index %d: got %v, want %v", i, dst[i], want[i])
			break
		}
	}

	// Test inverse with the shared forward table (conjugated on load).
	inv := make([]complex64, n)

	scratch = make([]complex64, n)
	if !transform.InverseStockhamPacked(inv, dst, twiddle, scratch, packed) {
		t.Skip("transform.InverseStockhamPacked not implemented")
		return
	}

	// Verify round-trip
	for i := range inv {
		if cmplx.Abs(complex128(inv[i]-src[i])) > testTol64 {
			t.Errorf("Inverse: index %d: got %v, want %v", i, inv[i], src[i])
			break
		}
	}
}

func testStockhamPacked128(t *testing.T, n int) {
	t.Helper()

	src := randomComplex128(n, uint64(n+456))
	want := reference.NaiveDFT128(src)

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)

	packed := transform.ComputePackedTwiddles[complex128](n, 4, twiddle)
	if packed == nil {
		t.Skip("Failed to compute packed twiddles")
		return
	}

	dst := make([]complex128, n)

	scratch := make([]complex128, n)
	if !transform.ForwardStockhamPacked(dst, src, twiddle, scratch, packed) {
		t.Skip("transform.ForwardStockhamPacked not implemented")
		return
	}

	for i := range dst {
		if cmplx.Abs(dst[i]-want[i]) > testTol128 {
			t.Errorf("Forward: index %d: got %v, want %v", i, dst[i], want[i])
			break
		}
	}

	inv := make([]complex128, n)

	scratch = make([]complex128, n)
	if !transform.InverseStockhamPacked(inv, dst, twiddle, scratch, packed) {
		t.Skip("transform.InverseStockhamPacked not implemented")
		return
	}

	for i := range inv {
		if cmplx.Abs(inv[i]-src[i]) > testTol128 {
			t.Errorf("Inverse: index %d: got %v, want %v", i, inv[i], src[i])
			break
		}
	}
}

// TestTwiddleFactorsRecursive tests the twiddle factor generation for recursive decomposition.
func TestTwiddleFactorsRecursive(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 32, 64}

	for _, n := range sizes {
		codeletSizes := []int{4, 8, 16, 32}
		cacheSize := 32768

		strategy := transform.PlanDecomposition(n, codeletSizes, cacheSize)
		if strategy == nil {
			continue
		}

		t.Run("complex64", func(t *testing.T) {
			twiddle := transform.TwiddleFactorsRecursive[complex64](strategy)
			if len(twiddle) == 0 {
				t.Skip("No twiddles generated")
			}

			// Verify no NaN or Inf
			for i, tw := range twiddle {
				re := real(tw)

				im := imag(tw)
				if re != re || im != im { // NaN check
					t.Errorf("index %d: NaN twiddle factor", i)
				}

				if re > 1e10 || re < -1e10 || im > 1e10 || im < -1e10 {
					t.Errorf("index %d: potentially infinite twiddle factor", i)
				}
			}
		})

		t.Run("complex128", func(t *testing.T) {
			twiddle := transform.TwiddleFactorsRecursive[complex128](strategy)
			if len(twiddle) == 0 {
				t.Skip("No twiddles generated")
			}

			for i, tw := range twiddle {
				if cmplx.IsNaN(tw) {
					t.Errorf("index %d: NaN twiddle factor", i)
				}

				if cmplx.IsInf(tw) {
					t.Errorf("index %d: Inf twiddle factor", i)
				}
			}
		})
	}
}
