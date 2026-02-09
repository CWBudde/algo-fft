package fft

import (
	"math/cmplx"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestDITWrappers(t *testing.T) {
	t.Parallel()

	n := 16

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	dst := make([]complex64, n)

	// Forward
	if !ditForward(dst, src, twiddle, scratch) {
		t.Fatal("ditForward failed")
	}

	ref := reference.NaiveDFT(src)
	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-ref[i])) > 1e-5 {
			t.Errorf("ditForward mismatch at %d", i)
		}
	}

	// Inverse
	fwd := make([]complex64, n)
	copy(fwd, dst)

	if !ditInverse(dst, fwd, twiddle, scratch) {
		t.Fatal("ditInverse failed")
	}

	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-src[i])) > 1e-5 {
			t.Errorf("ditInverse mismatch at %d", i)
		}
	}
}

func TestStockhamWrappers(t *testing.T) {
	t.Parallel()

	n := 16

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	dst := make([]complex64, n)

	// Forward
	if !stockhamForward(dst, src, twiddle, scratch) {
		t.Fatal("stockhamForward failed")
	}

	ref := reference.NaiveDFT(src)
	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-ref[i])) > 1e-5 {
			t.Errorf("stockhamForward mismatch at %d", i)
		}
	}

	// Inverse
	fwd := make([]complex64, n)
	copy(fwd, dst)

	if !stockhamInverse(dst, fwd, twiddle, scratch) {
		t.Fatal("stockhamInverse failed")
	}

	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-src[i])) > 1e-5 {
			t.Errorf("stockhamInverse mismatch at %d", i)
		}
	}
}

func TestBluesteinWrappers(t *testing.T) {
	t.Parallel()

	// Use generic wrapper
	n := 3
	// ComputeBluesteinFilter uses m >= 2n-1.
	// 2*3-1 = 5. next power of 2 >= 5 is 8.
	m := 8

	chirp := ComputeChirpSequence[complex64](n)
	twiddles := mathpkg.ComputeTwiddleFactors[complex64](m)
	scratch := make([]complex64, m)

	// Filter
	filter := ComputeBluesteinFilter[complex64](n, m, chirp, twiddles, scratch)
	if len(filter) != m {
		t.Errorf("ComputeBluesteinFilter len=%d want %d", len(filter), m)
	}

	// Convolution
	x := make([]complex64, m)
	dst := make([]complex64, m)
	BluesteinConvolution(dst, x, filter, twiddles, scratch)
}

func TestBluesteinWrappers128(t *testing.T) {
	t.Parallel()

	n := 3
	m := 8

	chirp := ComputeChirpSequence[complex128](n)
	twiddles := mathpkg.ComputeTwiddleFactors[complex128](m)
	scratch := make([]complex128, m)

	filter := ComputeBluesteinFilter[complex128](n, m, chirp, twiddles, scratch)
	if len(filter) != m {
		t.Errorf("ComputeBluesteinFilter len=%d want %d", len(filter), m)
	}

	x := make([]complex128, m)
	dst := make([]complex128, m)
	BluesteinConvolution(dst, x, filter, twiddles, scratch)
}
