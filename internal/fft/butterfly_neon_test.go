//go:build arm64 && asm && !purego

package fft

import (
	"math"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestNEONComplex64_CorrectnessVsReference(t *testing.T) {
	sizes := []int{16, 32, 64, 128, 256}
	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i%10), float32((i*3)%7))
			}

			kernels := SelectKernels[complex64](cpu.DetectFeatures())
			neonResult := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)
			scratch := make([]complex64, n)

			if !kernels.Forward(neonResult, src, twiddle, scratch) {
				t.Fatalf("Forward kernel returned false for n=%d", n)
			}

			refResult := reference.NaiveDFT(src)
			assertComplex64MaxError(t, neonResult, refResult, 1e-4, "reference")
		})
	}
}

func TestNEONComplex64_RoundTrip(t *testing.T) {
	sizes := []int{16, 32, 64, 128, 256}
	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			original := make([]complex64, n)
			for i := range original {
				original[i] = complex(float32(i*7%13), float32((i*11)%17))
			}

			freq := make([]complex64, n)
			recovered := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)
			scratch := make([]complex64, n)
			kernels := SelectKernels[complex64](cpu.DetectFeatures())

			if !kernels.Forward(freq, original, twiddle, scratch) {
				t.Fatalf("Forward kernel returned false for n=%d", n)
			}

			if !kernels.Inverse(recovered, freq, twiddle, scratch) {
				t.Fatalf("Inverse kernel returned false for n=%d", n)
			}

			assertComplex64MaxError(t, recovered, original, 1e-5, "round-trip")
		})
	}
}

func TestNEONComplex64_VsGoDIT(t *testing.T) {
	sizes := []int{16, 32, 64, 128, 256}
	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i%10), float32((i*3)%7))
			}

			neonResult := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)
			scratch := make([]complex64, n)
			kernels := SelectKernels[complex64](cpu.DetectFeatures())

			if !kernels.Forward(neonResult, src, twiddle, scratch) {
				t.Fatalf("Forward kernel returned false for n=%d", n)
			}

			goResult := make([]complex64, n)

			if !forwardDITComplex64(goResult, src, twiddle, scratch) {
				t.Fatalf("forwardDITComplex64(%d) failed", n)
			}

			assertComplex64MaxError(t, neonResult, goResult, 5e-5, "go-dit")
		})
	}
}

func TestNEONComplex64_Strided1024(t *testing.T) {
	const n = 1024
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i%17), float32((i*5)%11))
	}

	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	neonResult := make([]complex64, n)
	goResult := make([]complex64, n)

	kernels := SelectKernels[complex64](cpu.DetectFeatures())
	if !kernels.Forward(neonResult, src, twiddle, scratch) {
		t.Fatalf("Forward kernel returned false for n=%d", n)
	}

	if !forwardDITComplex64(goResult, src, twiddle, scratch) {
		t.Fatalf("forwardDITComplex64(%d) failed", n)
	}

	assertComplex64MaxError(t, neonResult, goResult, 3e-4, "strided-1024")
}

func assertComplex64MaxError(t *testing.T, got, want []complex64, tol float32, label string) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got %d want %d", label, len(got), len(want))
	}

	maxErr := float32(0)
	for i := range got {
		diff := got[i] - want[i]
		err := float32(math.Sqrt(float64(real(diff)*real(diff) + imag(diff)*imag(diff))))
		if err > maxErr {
			maxErr = err
		}
	}

	if maxErr > tol {
		t.Fatalf("%s: max error %e exceeds %e", label, maxErr, tol)
	}
}
