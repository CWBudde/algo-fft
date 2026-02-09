package fft

import (
	"math/cmplx"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestMixedRadixComplex64(t *testing.T) {
	t.Parallel()

	// 12 = 4 * 3 (mixed radix)
	n := 12

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n*2) // Extra scratch for recursive
	dst := make([]complex64, n)

	// Forward
	if !forwardMixedRadixComplex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardMixedRadixComplex64 failed")
	}

	ref := reference.NaiveDFT(src)
	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-ref[i])) > 1e-5 {
			t.Errorf("forwardMixedRadixComplex64 mismatch at %d: got %v want %v", i, dst[i], ref[i])
		}
	}

	// Inverse
	fwd := make([]complex64, n)
	copy(fwd, dst)

	if !inverseMixedRadixComplex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseMixedRadixComplex64 failed")
	}

	for i := range dst {
		if cmplx.Abs(complex128(dst[i]-src[i])) > 1e-5 {
			t.Errorf("inverseMixedRadixComplex64 mismatch at %d", i)
		}
	}
}

func TestMixedRadixComplex128(t *testing.T) {
	t.Parallel()

	n := 12

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i), 0)
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n*2)
	dst := make([]complex128, n)

	// Forward
	if !forwardMixedRadixComplex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardMixedRadixComplex128 failed")
	}

	ref := reference.NaiveDFT128(src)
	for i := range dst {
		if cmplx.Abs(dst[i]-ref[i]) > 1e-10 {
			t.Errorf("forwardMixedRadixComplex128 mismatch at %d", i)
		}
	}

	// Inverse
	fwd := make([]complex128, n)
	copy(fwd, dst)

	if !inverseMixedRadixComplex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseMixedRadixComplex128 failed")
	}

	for i := range dst {
		if cmplx.Abs(dst[i]-src[i]) > 1e-10 {
			t.Errorf("inverseMixedRadixComplex128 mismatch at %d", i)
		}
	}
}
