//go:build arm64 && fft_asm && !purego

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestNEONSize4Radix4Complex64(t *testing.T) {
	const n = 4
	src := []complex64{
		complex(1, -2),
		complex(3, 4),
		complex(-5, 6),
		complex(7, -8),
	}

	dst := make([]complex64, n)
	inv := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)

	if !forwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardNEONSize4Radix4Complex64Asm failed")
	}

	ref := reference.NaiveDFT(src)
	assertComplex64MaxError(t, dst, ref, 1e-5, "reference")

	if !inverseNEONSize4Radix4Complex64Asm(inv, dst, twiddle, scratch, bitrev) {
		t.Fatal("inverseNEONSize4Radix4Complex64Asm failed")
	}

	assertComplex64MaxError(t, inv, src, 1e-5, "round-trip")
}

func TestNEONSize8Radix8Complex64(t *testing.T) {
	const n = 8
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i*3-7), float32(11-i*2))
	}

	dst := make([]complex64, n)
	inv := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)

	if !forwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardNEONSize8Radix8Complex64Asm failed")
	}

	ref := reference.NaiveDFT(src)
	assertComplex64MaxError(t, dst, ref, 1e-5, "reference")

	if !inverseNEONSize8Radix8Complex64Asm(inv, dst, twiddle, scratch, bitrev) {
		t.Fatal("inverseNEONSize8Radix8Complex64Asm failed")
	}

	assertComplex64MaxError(t, inv, src, 1e-5, "round-trip")
}
