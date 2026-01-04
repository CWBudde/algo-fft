package kernels

import (
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	stockhamTol64  = 1e-4
	stockhamTol128 = 1e-10
)

// TestStockhamForwardComplex64 tests the Stockham forward kernel.
func TestStockhamForwardComplex64(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0xABCD+uint64(n))
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)
			bitrev := mathpkg.ComputeBitReversalIndices(n)

			if !stockhamForward(dst, src, twiddle, scratch, bitrev) {
				t.Fatalf("stockhamForward failed for n=%d", n)
			}

			want := reference.NaiveDFT(src)
			assertComplex64Close(t, dst, want, stockhamTol64)
		})
	}
}

// TestStockhamInverseComplex64 tests the Stockham inverse kernel.
func TestStockhamInverseComplex64(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	for _, n := range sizes {
		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x1357+uint64(n))
			fwd := make([]complex64, n)
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)
			bitrev := mathpkg.ComputeBitReversalIndices(n)

			if !stockhamForward(fwd, src, twiddle, scratch, bitrev) {
				t.Fatalf("stockhamForward failed for n=%d", n)
			}

			if !stockhamInverse(dst, fwd, twiddle, scratch, bitrev) {
				t.Fatalf("stockhamInverse failed for n=%d", n)
			}

			want := reference.NaiveIDFT(fwd)
			assertComplex64Close(t, dst, want, stockhamTol64)
		})
	}
}

// TestStockhamForwardComplex128 tests the Stockham forward kernel (complex128).
func TestStockhamForwardComplex128(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x2468+uint64(n))
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := mathpkg.ComputeBitReversalIndices(n)

			if !stockhamForward(dst, src, twiddle, scratch, bitrev) {
				t.Fatalf("stockhamForward failed for n=%d", n)
			}

			want := reference.NaiveDFT128(src)
			assertComplex128Close(t, dst, want, stockhamTol128)
		})
	}
}

// TestStockhamInverseComplex128 tests the Stockham inverse kernel (complex128).
func TestStockhamInverseComplex128(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	for _, n := range sizes {
		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x9753+uint64(n))
			fwd := make([]complex128, n)
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := mathpkg.ComputeBitReversalIndices(n)

			if !stockhamForward(fwd, src, twiddle, scratch, bitrev) {
				t.Fatalf("stockhamForward failed for n=%d", n)
			}

			if !stockhamInverse(dst, fwd, twiddle, scratch, bitrev) {
				t.Fatalf("stockhamInverse failed for n=%d", n)
			}

			want := reference.NaiveIDFT128(fwd)
			assertComplex128Close(t, dst, want, stockhamTol128)
		})
	}
}
