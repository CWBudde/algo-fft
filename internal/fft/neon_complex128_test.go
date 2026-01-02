//go:build arm64 && asm && !purego

package fft

import (
	"math"
	"strconv"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestNEONComplex128_AsmPath(t *testing.T) {
	sizes := []int{2, 4, 8, 16}

	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i+1), float64(-i)*0.5)
			}

			dst := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex128, n)

			if !forwardNEONComplex128Asm(dst, src, twiddle, scratch, bitrev) {
				t.Fatalf("forwardNEONComplex128Asm returned false for n=%d", n)
			}

			ref := reference.NaiveDFT128(src)
			assertComplex128MaxError(t, dst, ref, 1e-12, "forward")

			roundTrip := make([]complex128, n)
			if !inverseNEONComplex128Asm(roundTrip, dst, twiddle, scratch, bitrev) {
				t.Fatalf("inverseNEONComplex128Asm returned false for n=%d", n)
			}

			assertComplex128MaxError(t, roundTrip, src, 1e-12, "inverse")
		})
	}
}

func TestNEONComplex128_CorrectnessVsReference(t *testing.T) {
	sizes := []int{16, 32, 64, 128, 256}

	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i%10), float64((i*3)%7))
			}

			dst := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex128, n)

			if !forwardNEONComplex128Asm(dst, src, twiddle, scratch, bitrev) {
				t.Fatalf("forwardNEONComplex128Asm returned false for n=%d", n)
			}

			ref := reference.NaiveDFT128(src)
			assertComplex128MaxError(t, dst, ref, 2e-11, "reference")
		})
	}
}

func TestNEONComplex128_RoundTrip(t *testing.T) {
	sizes := []int{16, 32, 64, 128, 256}

	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			original := make([]complex128, n)
			for i := range original {
				original[i] = complex(float64(i*7%13), float64((i*11)%17))
			}

			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex128, n)
			freq := make([]complex128, n)
			recovered := make([]complex128, n)

			if !forwardNEONComplex128Asm(freq, original, twiddle, scratch, bitrev) {
				t.Fatalf("forwardNEONComplex128Asm returned false for n=%d", n)
			}

			if !inverseNEONComplex128Asm(recovered, freq, twiddle, scratch, bitrev) {
				t.Fatalf("inverseNEONComplex128Asm returned false for n=%d", n)
			}

			assertComplex128MaxError(t, recovered, original, 1e-12, "round-trip")
		})
	}
}

func TestNEONComplex128_VsGoDIT(t *testing.T) {
	sizes := []int{16, 32, 64, 128, 256}

	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i%10), float64((i*3)%7))
			}

			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex128, n)
			neonResult := make([]complex128, n)
			goResult := make([]complex128, n)

			if !forwardNEONComplex128Asm(neonResult, src, twiddle, scratch, bitrev) {
				t.Fatalf("forwardNEONComplex128Asm returned false for n=%d", n)
			}

			if !forwardDITComplex128(goResult, src, twiddle, scratch, bitrev) {
				t.Fatalf("forwardDITComplex128(%d) failed", n)
			}

			assertComplex128MaxError(t, neonResult, goResult, 1e-12, "go-dit")
		})
	}
}

func assertComplex128MaxError(t *testing.T, got, want []complex128, tol float64, label string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got %d want %d", label, len(got), len(want))
	}

	maxErr := float64(0)
	for i := range got {
		diff := got[i] - want[i]
		err := math.Sqrt(real(diff)*real(diff) + imag(diff)*imag(diff))
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > tol {
		t.Fatalf("%s: max error %e exceeds %e", label, maxErr, tol)
	}
}
