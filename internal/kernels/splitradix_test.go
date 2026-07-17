package kernels

import (
	"math/cmplx"
	"math/rand"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

var splitRadixTestSizes = []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}

func TestSplitRadix_MatchesReference_Complex128(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(7)) //nolint:gosec // deterministic test data

	for _, n := range splitRadixTestSizes {
		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
		}

		twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
		dst := make([]complex128, n)
		scratch := make([]complex128, n)

		if !ForwardSplitRadixComplex128(dst, src, twiddle, scratch) {
			t.Fatalf("n=%d: forward returned false", n)
		}

		want := reference.NaiveDFT128(src)
		for i := range dst {
			if diff := cmplx.Abs(dst[i] - want[i]); diff > 1e-9 {
				t.Fatalf("n=%d bin %d: got %v, want %v (diff %v)", n, i, dst[i], want[i], diff)
			}
		}

		inv := make([]complex128, n)
		if !InverseSplitRadixComplex128(inv, dst, twiddle, scratch) {
			t.Fatalf("n=%d: inverse returned false", n)
		}

		for i := range inv {
			if diff := cmplx.Abs(inv[i] - src[i]); diff > 1e-11 {
				t.Fatalf("n=%d round-trip at %d: got %v, want %v", n, i, inv[i], src[i])
			}
		}
	}
}

func TestSplitRadix_MatchesReference_Complex64(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(11)) //nolint:gosec // deterministic test data

	for _, n := range splitRadixTestSizes {
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(rng.Float64()*2-1), float32(rng.Float64()*2-1))
		}

		twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
		dst := make([]complex64, n)
		scratch := make([]complex64, n)

		if !ForwardSplitRadixComplex64(dst, src, twiddle, scratch) {
			t.Fatalf("n=%d: forward returned false", n)
		}

		want := reference.NaiveDFT(src)
		for i := range dst {
			if diff := cmplx.Abs(complex128(dst[i] - want[i])); diff > 1e-2 {
				t.Fatalf("n=%d bin %d: got %v, want %v (diff %v)", n, i, dst[i], want[i], diff)
			}
		}

		inv := make([]complex64, n)
		if !InverseSplitRadixComplex64(inv, dst, twiddle, scratch) {
			t.Fatalf("n=%d: inverse returned false", n)
		}

		for i := range inv {
			if diff := cmplx.Abs(complex128(inv[i] - src[i])); diff > 1e-4 {
				t.Fatalf("n=%d round-trip at %d: got %v, want %v", n, i, inv[i], src[i])
			}
		}
	}
}

func TestSplitRadix_InPlaceAndRejects(t *testing.T) {
	t.Parallel()

	n := 64
	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i), float64(-i))
	}

	want := make([]complex128, n)
	if !ForwardSplitRadixComplex128(want, src, twiddle, scratch) {
		t.Fatal("out-of-place forward returned false")
	}

	data := make([]complex128, n)
	copy(data, src)

	if !ForwardSplitRadixComplex128(data, data, twiddle, scratch) {
		t.Fatal("in-place forward returned false")
	}

	for i := range want {
		if data[i] != want[i] {
			t.Fatalf("in-place mismatch at %d: got %v, want %v", i, data[i], want[i])
		}
	}

	// Non-power-of-two and short slices must be rejected.
	if ForwardSplitRadixComplex128(make([]complex128, 12), make([]complex128, 12),
		mathpkg.ComputeTwiddleFactors[complex128](12), make([]complex128, 12)) {
		t.Fatal("n=12 accepted, want reject")
	}

	if ForwardSplitRadixComplex128(make([]complex128, 8), make([]complex128, 16),
		mathpkg.ComputeTwiddleFactors[complex128](16), make([]complex128, 16)) {
		t.Fatal("short dst accepted, want reject")
	}
}
