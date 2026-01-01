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
	assertComplex64MaxError(t, dst, ref, 1e-4, "reference")

	if !inverseNEONSize4Radix4Complex64Asm(inv, dst, twiddle, scratch, bitrev) {
		t.Fatal("inverseNEONSize4Radix4Complex64Asm failed")
	}

	assertComplex64MaxError(t, inv, src, 1e-4, "round-trip")
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
	assertComplex64MaxError(t, dst, ref, 1e-4, "reference")

	if !inverseNEONSize8Radix8Complex64Asm(inv, dst, twiddle, scratch, bitrev) {
		t.Fatal("inverseNEONSize8Radix8Complex64Asm failed")
	}

	assertComplex64MaxError(t, inv, src, 1e-4, "round-trip")
}

func TestNEONSize32MixedRadix24Complex64(t *testing.T) {
	const n = 32
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i*2-9), float32(5-i))
	}

	dst := make([]complex64, n)
	inv := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	if !forwardNEONSize32MixedRadix24Complex64Asm(dst, src, twiddle, scratch, bitrevSize32Mixed24) {
		t.Fatal("forwardNEONSize32MixedRadix24Complex64Asm failed")
	}

	ref := reference.NaiveDFT(src)
	assertComplex64MaxError(t, dst, ref, 1e-4, "reference")

	if !inverseNEONSize32MixedRadix24Complex64Asm(inv, dst, twiddle, scratch, bitrevSize32Mixed24) {
		t.Fatal("inverseNEONSize32MixedRadix24Complex64Asm failed")
	}

	assertComplex64MaxError(t, inv, src, 1e-4, "round-trip")
}

func TestNEONSize128MixedRadix24Complex64(t *testing.T) {
	const n = 128
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i*3-11), float32(7-i*2))
	}

	dst := make([]complex64, n)
	inv := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	if !forwardNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch, bitrevSize128Mixed24) {
		t.Fatal("forwardNEONSize128MixedRadix24Complex64Asm failed")
	}

	ref := reference.NaiveDFT(src)
	assertComplex64MaxError(t, dst, ref, 1e-3, "reference")

	if !inverseNEONSize128MixedRadix24Complex64Asm(inv, dst, twiddle, scratch, bitrevSize128Mixed24) {
		t.Fatal("inverseNEONSize128MixedRadix24Complex64Asm failed")
	}

	assertComplex64MaxError(t, inv, src, 1e-3, "round-trip")
}

func TestNEONSize256Radix4Complex64(t *testing.T) {
	const n = 256
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i*5-17), float32(13-i*3))
	}

	dst := make([]complex64, n)
	inv := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	if !forwardNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize256Radix4) {
		t.Fatal("forwardNEONSize256Radix4Complex64Asm failed")
	}

	ref := reference.NaiveDFT(src)
	assertComplex64MaxError(t, dst, ref, 5e-3, "reference")

	if !inverseNEONSize256Radix4Complex64Asm(inv, dst, twiddle, scratch, bitrevSize256Radix4) {
		t.Fatal("inverseNEONSize256Radix4Complex64Asm failed")
	}

	assertComplex64MaxError(t, inv, src, 5e-3, "round-trip")
}
