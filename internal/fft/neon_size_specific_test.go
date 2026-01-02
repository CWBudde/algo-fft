//go:build arm64 && asm && !purego

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

type neonKernel64 func(dst, src, twiddle, scratch []complex64, bitrev []int) bool

func runNEONRoundTripComplex64(t *testing.T, n int, forward, inverse neonKernel64, bitrev []int, tol float32) {
	t.Helper()

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i*3-7), float32(11-i*2))
	}

	dst := make([]complex64, n)
	inv := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	if bitrev == nil {
		bitrev = ComputeBitReversalIndices(n)
	}
	scratch := make([]complex64, n)

	if !forward(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forward kernel failed")
	}

	ref := reference.NaiveDFT(src)
	assertComplex64MaxError(t, dst, ref, tol, "reference")

	if !inverse(inv, dst, twiddle, scratch, bitrev) {
		t.Fatal("inverse kernel failed")
	}

	assertComplex64MaxError(t, inv, src, tol, "round-trip")
}

func TestNEONSize4Radix4Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 4, forwardNEONSize4Radix4Complex64Asm, inverseNEONSize4Radix4Complex64Asm, nil, 1e-4)
}

func TestNEONSize8Radix8Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 8, forwardNEONSize8Radix8Complex64Asm, inverseNEONSize8Radix8Complex64Asm, nil, 1e-4)
}

func TestNEONSize8Radix2Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 8, forwardNEONSize8Radix2Complex64Asm, inverseNEONSize8Radix2Complex64Asm, nil, 1e-4)
}

func TestNEONSize8Radix4Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 8, forwardNEONSize8Radix4Complex64Asm, inverseNEONSize8Radix4Complex64Asm, nil, 1e-4)
}

func TestNEONSize16Radix4Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 16, forwardNEONSize16Radix4Complex64Asm, inverseNEONSize16Radix4Complex64Asm, bitrevSize16Radix4, 1e-4)
}

func TestNEONSize16Radix2Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 16, forwardNEONSize16Complex64Asm, inverseNEONSize16Complex64Asm, nil, 1e-4)
}

func TestNEONSize32Radix2Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 32, forwardNEONSize32Complex64Asm, inverseNEONSize32Complex64Asm, nil, 1e-4)
}

func TestNEONSize32MixedRadix24Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 32, forwardNEONSize32MixedRadix24Complex64Asm, inverseNEONSize32MixedRadix24Complex64Asm, bitrevSize32Mixed24, 1e-4)
}

func TestNEONSize64Radix2Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 64, forwardNEONSize64Complex64Asm, inverseNEONSize64Complex64Asm, nil, 1e-3)
}

func TestNEONSize64Radix4Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 64, forwardNEONSize64Radix4Complex64Asm, inverseNEONSize64Radix4Complex64Asm, bitrevSize64Radix4, 1e-3)
}

func TestNEONSize128Radix2Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 128, forwardNEONSize128Complex64Asm, inverseNEONSize128Complex64Asm, nil, 1e-3)
}

func TestNEONSize128MixedRadix24Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 128, forwardNEONSize128MixedRadix24Complex64Asm, inverseNEONSize128MixedRadix24Complex64Asm, bitrevSize128Mixed24, 1e-3)
}

func TestNEONSize256Radix2Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 256, forwardNEONSize256Radix2Complex64Asm, inverseNEONSize256Radix2Complex64Asm, nil, 5e-3)
}

func TestNEONSize256Radix4Complex64(t *testing.T) {
	runNEONRoundTripComplex64(t, 256, forwardNEONSize256Radix4Complex64Asm, inverseNEONSize256Radix4Complex64Asm, bitrevSize256Radix4, 5e-3)
}
