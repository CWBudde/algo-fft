//go:build arm64 && asm && !purego

package fft

import (
	"math"
	"strconv"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestNEONSizeSpecificComplex64 validates size-specific NEON kernels against reference implementation.
func TestNEONSizeSpecificComplex64(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		forward func([]complex64, []complex64, []complex64, []complex64) bool
		inverse func([]complex64, []complex64, []complex64, []complex64) bool
		tol     float32
	}{
		{
			name:    "Size4_Radix4",
			size:    4,
			forward: forwardNEONSize4Radix4Complex64Asm,
			inverse: inverseNEONSize4Radix4Complex64Asm,
			tol:     1e-4,
		},
		{
			name:    "Size8_Radix8",
			size:    8,
			forward: forwardNEONSize8Radix8Complex64Asm,
			inverse: inverseNEONSize8Radix8Complex64Asm,
			tol:     1e-4,
		},
		{
			name:    "Size8_Radix2",
			size:    8,
			forward: forwardNEONSize8Radix2Complex64Asm,
			inverse: inverseNEONSize8Radix2Complex64Asm,
			tol:     1e-4,
		},
		{
			name:    "Size8_Radix4",
			size:    8,
			forward: forwardNEONSize8Radix4Complex64Asm,
			inverse: inverseNEONSize8Radix4Complex64Asm,
			tol:     1e-4,
		},
		{
			name:    "Size16_Radix4",
			size:    16,
			forward: forwardNEONSize16Radix4Complex64Asm,
			inverse: inverseNEONSize16Radix4Complex64Asm,
			tol:     1e-4,
		},
		{
			name:    "Size16_Radix2",
			size:    16,
			forward: forwardNEONSize16Radix2Complex64Asm,
			inverse: inverseNEONSize16Radix2Complex64Asm,
			tol:     1e-4,
		},
		{
			name:    "Size32_Radix2",
			size:    32,
			forward: forwardNEONSize32Radix2Complex64Asm,
			inverse: inverseNEONSize32Radix2Complex64Asm,
			tol:     1e-4,
		},
		{
			name:    "Size32_MixedRadix24",
			size:    32,
			forward: forwardNEONSize32MixedRadix24Complex64Asm,
			inverse: inverseNEONSize32MixedRadix24Complex64Asm,
			tol:     1e-4,
		},
		{
			name:    "Size64_Radix2",
			size:    64,
			forward: forwardNEONSize64Radix2Complex64Asm,
			inverse: inverseNEONSize64Radix2Complex64Asm,
			tol:     1e-3,
		},
		{
			name:    "Size64_Radix4",
			size:    64,
			forward: forwardNEONSize64Radix4Complex64Asm,
			inverse: inverseNEONSize64Radix4Complex64Asm,
			tol:     1e-3,
		},
		{
			name:    "Size128_Radix2",
			size:    128,
			forward: forwardNEONSize128Radix2Complex64Asm,
			inverse: inverseNEONSize128Radix2Complex64Asm,
			tol:     1e-3,
		},
		{
			name:    "Size128_MixedRadix24",
			size:    128,
			forward: forwardNEONSize128MixedRadix24Complex64Asm,
			inverse: inverseNEONSize128MixedRadix24Complex64Asm,
			tol:     1e-3,
		},
		{
			name:    "Size256_Radix2",
			size:    256,
			forward: forwardNEONSize256Radix2Complex64Asm,
			inverse: inverseNEONSize256Radix2Complex64Asm,
			tol:     5e-3,
		},
		{
			name:    "Size256_Radix4",
			size:    256,
			forward: forwardNEONSize256Radix4Complex64Asm,
			inverse: inverseNEONSize256Radix4Complex64Asm,
			tol:     5e-3,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := make([]complex64, tc.size)
			for i := range src {
				src[i] = complex(float32(i*3-7), float32(11-i*2))
			}

			dst := make([]complex64, tc.size)
			inv := make([]complex64, tc.size)
			twiddle := ComputeTwiddleFactors[complex64](tc.size)
			scratch := make([]complex64, tc.size)

			// Test forward transform vs reference
			if !tc.forward(dst, src, twiddle, scratch) {
				t.Fatal("forward kernel failed")
			}

			ref := reference.NaiveDFT(src)
			assertComplex64MaxError(t, dst, ref, tc.tol, "reference")

			// Test inverse transform and round-trip
			if !tc.inverse(inv, dst, twiddle, scratch) {
				t.Fatal("inverse kernel failed")
			}

			assertComplex64MaxError(t, inv, src, tc.tol, "round-trip")
		})
	}
}

// TestNEONSizeSpecificComplex128 validates size-specific NEON kernels for complex128.
func TestNEONSizeSpecificComplex128(t *testing.T) {
	tests := []struct {
		name    string
		size    int
		forward func([]complex128, []complex128, []complex128, []complex128) bool
		inverse func([]complex128, []complex128, []complex128, []complex128) bool
		tol     float64
	}{
		{
			name:    "Size4_Radix4",
			size:    4,
			forward: forwardNEONSize4Radix4Complex128Asm,
			inverse: inverseNEONSize4Radix4Complex128Asm,
			tol:     1e-12,
		},
		{
			name:    "Size8_Radix2",
			size:    8,
			forward: forwardNEONSize8Radix2Complex128Asm,
			inverse: inverseNEONSize8Radix2Complex128Asm,
			tol:     1e-12,
		},
		{
			name:    "Size16_Radix4",
			size:    16,
			forward: forwardNEONSize16Radix4Complex128Asm,
			inverse: inverseNEONSize16Radix4Complex128Asm,
			tol:     2e-12,
		},
		{
			name:    "Size16_Radix2",
			size:    16,
			forward: forwardNEONSize16Complex128Asm,
			inverse: inverseNEONSize16Complex128Asm,
			tol:     2e-12,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := make([]complex128, tc.size)
			for i := range src {
				src[i] = complex(float64(i*3-7), float64(11-i*2))
			}

			dst := make([]complex128, tc.size)
			inv := make([]complex128, tc.size)
			twiddle := ComputeTwiddleFactors[complex128](tc.size)
			scratch := make([]complex128, tc.size)

			// Test forward transform vs reference
			if !tc.forward(dst, src, twiddle, scratch) {
				t.Fatal("forward kernel failed")
			}

			ref := reference.NaiveDFT128(src)
			assertComplex128MaxError(t, dst, ref, tc.tol, "reference")

			// Test inverse transform and round-trip
			if !tc.inverse(inv, dst, twiddle, scratch) {
				t.Fatal("inverse kernel failed")
			}

			assertComplex128MaxError(t, inv, src, tc.tol, "round-trip")
		})
	}
}

// TestNEONComplex128_AsmPath tests basic asm functionality for complex128.
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
			scratch := make([]complex128, n)

			if !forwardNEONComplex128Asm(dst, src, twiddle, scratch) {
				t.Fatalf("forwardNEONComplex128Asm returned false for n=%d", n)
			}

			ref := reference.NaiveDFT128(src)
			assertComplex128MaxError(t, dst, ref, 1e-12, "forward")

			roundTrip := make([]complex128, n)
			if !inverseNEONComplex128Asm(roundTrip, dst, twiddle, scratch) {
				t.Fatalf("inverseNEONComplex128Asm returned false for n=%d", n)
			}

			assertComplex128MaxError(t, roundTrip, src, 1e-12, "inverse")
		})
	}
}

// TestNEONComplex128_CorrectnessVsReference validates against naive DFT.
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
			scratch := make([]complex128, n)

			if !forwardNEONComplex128Asm(dst, src, twiddle, scratch) {
				t.Fatalf("forwardNEONComplex128Asm returned false for n=%d", n)
			}

			ref := reference.NaiveDFT128(src)
			assertComplex128MaxError(t, dst, ref, 2e-11, "reference")
		})
	}
}

// TestNEONComplex128_RoundTrip validates forward/inverse round-trip accuracy.
func TestNEONComplex128_RoundTrip(t *testing.T) {
	sizes := []int{16, 32, 64, 128, 256}

	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			original := make([]complex128, n)
			for i := range original {
				original[i] = complex(float64(i*7%13), float64((i*11)%17))
			}

			twiddle := ComputeTwiddleFactors[complex128](n)
			scratch := make([]complex128, n)
			freq := make([]complex128, n)
			recovered := make([]complex128, n)

			if !forwardNEONComplex128Asm(freq, original, twiddle, scratch) {
				t.Fatalf("forwardNEONComplex128Asm returned false for n=%d", n)
			}

			if !inverseNEONComplex128Asm(recovered, freq, twiddle, scratch) {
				t.Fatalf("inverseNEONComplex128Asm returned false for n=%d", n)
			}

			assertComplex128MaxError(t, recovered, original, 1e-12, "round-trip")
		})
	}
}

// TestNEONComplex128_VsGoDIT compares NEON results with pure-Go DIT implementation.
func TestNEONComplex128_VsGoDIT(t *testing.T) {
	sizes := []int{16, 32, 64, 128, 256}

	for _, n := range sizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i%10), float64((i*3)%7))
			}

			twiddle := ComputeTwiddleFactors[complex128](n)
			scratch := make([]complex128, n)
			neonResult := make([]complex128, n)
			goResult := make([]complex128, n)

			if !forwardNEONComplex128Asm(neonResult, src, twiddle, scratch) {
				t.Fatalf("forwardNEONComplex128Asm returned false for n=%d", n)
			}

			// forwardDITComplex128 now takes 4 args!
			if !forwardDITComplex128(goResult, src, twiddle, scratch) {
				t.Fatalf("forwardDITComplex128(%d) failed", n)
			}

			assertComplex128MaxError(t, neonResult, goResult, 1e-12, "go-dit")
		})
	}
}

// assertComplex128MaxError validates complex128 slices within tolerance.
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