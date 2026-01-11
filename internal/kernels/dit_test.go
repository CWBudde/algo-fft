package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	ditTol64  = 1e-4
	ditTol128 = 1e-10
)

// TestDITForwardComplex64 tests the generic DIT forward kernel.
func TestDITForwardComplex64(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0xDEADBEEF+uint64(n))
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !ditForward(dst, src, twiddle, scratch) {
				t.Fatalf("ditForward failed for n=%d", n)
			}

			want := reference.NaiveDFT(src)
			assertComplex64Close(t, dst, want, ditTol64)
		})
	}
}

// TestDITInverseComplex64 tests the generic DIT inverse kernel.
func TestDITInverseComplex64(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	for _, n := range sizes {
		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0xCAFEBABE+uint64(n))
			fwd := make([]complex64, n)
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !ditForward(fwd, src, twiddle, scratch) {
				t.Fatalf("ditForward failed for n=%d", n)
			}

			if !ditInverse(dst, fwd, twiddle, scratch) {
				t.Fatalf("ditInverse failed for n=%d", n)
			}

			want := reference.NaiveIDFT(fwd)
			assertComplex64Close(t, dst, want, ditTol64)
		})
	}
}

// TestDITForwardComplex128 tests the generic DIT forward kernel (complex128).
func TestDITForwardComplex128(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0xBEEFCAFE+uint64(n))
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			if !ditForward(dst, src, twiddle, scratch) {
				t.Fatalf("ditForward failed for n=%d", n)
			}

			want := reference.NaiveDFT128(src)
			assertComplex128Close(t, dst, want, ditTol128)
		})
	}
}

// TestDITInverseComplex128 tests the generic DIT inverse kernel (complex128).
func TestDITInverseComplex128(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	for _, n := range sizes {
		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0xFEEDFACE+uint64(n))
			fwd := make([]complex128, n)
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			if !ditForward(fwd, src, twiddle, scratch) {
				t.Fatalf("ditForward failed for n=%d", n)
			}

			if !ditInverse(dst, fwd, twiddle, scratch) {
				t.Fatalf("ditInverse failed for n=%d", n)
			}

			want := reference.NaiveIDFT128(fwd)
			assertComplex128Close(t, dst, want, ditTol128)
		})
	}
}
