package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestStockhamComplex64(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x12345678+uint64(n))
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !stockhamForward(dst, src, twiddle, scratch) {
				t.Fatalf("stockhamForward failed for n=%d", n)
			}

			want := reference.NaiveDFT(src)
			assertComplex64Close(t, dst, want, 1e-4)
		})

		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x87654321+uint64(n))
			fwd := make([]complex64, n)
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !stockhamForward(fwd, src, twiddle, scratch) {
				t.Fatalf("stockhamForward failed for n=%d", n)
			}

			if !stockhamInverse(dst, fwd, twiddle, scratch) {
				t.Fatalf("stockhamInverse failed for n=%d", n)
			}

			want := reference.NaiveIDFT(fwd)
			assertComplex64Close(t, dst, want, 1e-4)
		})
	}
}

func TestStockhamComplex128(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}

	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x12345678+uint64(n))
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			if !stockhamForward(dst, src, twiddle, scratch) {
				t.Fatalf("stockhamForward failed for n=%d", n)
			}

			want := reference.NaiveDFT128(src)
			assertComplex128Close(t, dst, want, 1e-10)
		})

		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x87654321+uint64(n))
			fwd := make([]complex128, n)
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			if !stockhamForward(fwd, src, twiddle, scratch) {
				t.Fatalf("stockhamForward failed for n=%d", n)
			}

			if !stockhamInverse(dst, fwd, twiddle, scratch) {
				t.Fatalf("stockhamInverse failed for n=%d", n)
			}

			want := reference.NaiveIDFT128(fwd)
			assertComplex128Close(t, dst, want, 1e-10)
		})
	}
}
