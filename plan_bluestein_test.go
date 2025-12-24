package algoforge

import (
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algoforge/internal/reference"
)

func TestNewPlan_Bluestein(t *testing.T) {
	t.Parallel()

	// Prime lengths trigger Bluestein
	primes := []int{7, 11, 13, 17}
	for _, n := range primes {
		t.Run("complex64_"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex64](n)
			if err != nil {
				t.Errorf("NewPlan(%d) error: %v", n, err)
				return
			}

			if plan.Len() != n {
				t.Errorf("Len() = %d, want %d", plan.Len(), n)
			}

			if plan.KernelStrategy() != KernelBluestein {
				t.Errorf("Strategy = %v, want KernelBluestein", plan.KernelStrategy())
			}
		})
		t.Run("complex128_"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex128](n)
			if err != nil {
				t.Errorf("NewPlan(%d) error: %v", n, err)
				return
			}

			if plan.Len() != n {
				t.Errorf("Len() = %d, want %d", plan.Len(), n)
			}

			if plan.KernelStrategy() != KernelBluestein {
				t.Errorf("Strategy = %v, want KernelBluestein", plan.KernelStrategy())
			}
		})
	}
}

func TestBluestein_RoundTrip(t *testing.T) {
	t.Parallel()

	// Simple round trip test for a prime size
	n := 13

	t.Run("complex64", func(t *testing.T) {
		t.Parallel()

		plan, err := NewPlan[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) failed: %v", n, err)
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i), 0)
		}

		dst := make([]complex64, n)

		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		back := make([]complex64, n)
		if err := plan.Inverse(back, dst); err != nil {
			t.Fatalf("Inverse failed: %v", err)
		}

		for i := range src {
			if cmplx.Abs(complex128(src[i]-back[i])) > 1e-4 {
				t.Errorf("Mismatch at %d: got %v, want %v", i, back[i], src[i])
			}
		}
	})

	t.Run("complex128", func(t *testing.T) {
		t.Parallel()

		plan, err := NewPlan[complex128](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) failed: %v", n, err)
		}

		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(float64(i), 0)
		}

		dst := make([]complex128, n)

		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		back := make([]complex128, n)
		if err := plan.Inverse(back, dst); err != nil {
			t.Fatalf("Inverse failed: %v", err)
		}

		for i := range src {
			if cmplx.Abs(src[i]-back[i]) > 1e-10 {
				t.Errorf("Mismatch at %d: got %v, want %v", i, back[i], src[i])
			}
		}
	})
}

func TestBluestein_LargePrimes(t *testing.T) {
	t.Parallel()

	primes := []int{251, 509, 1021}

	for _, n := range primes {
		t.Run(itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i), float32(-i))
			}

			dst := make([]complex64, n)

			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			back := make([]complex64, n)
			if err := plan.Inverse(back, dst); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			for i := range src {
				if cmplx.Abs(complex128(src[i]-back[i])) > 1e-3 {
					t.Fatalf("Mismatch at %d: got %v, want %v", i, back[i], src[i])
				}
			}
		})
	}
}

// TestBluestein_MatchesReference validates Bluestein FFT against naive DFT.
// This is the critical correctness test - it proves the FFT computes the right answer,
// not just that it's invertible.
func TestBluestein_MatchesReference(t *testing.T) {
	t.Parallel()

	// Test various prime sizes
	primes := []int{7, 11, 13, 17, 19, 23, 31}

	for _, n := range primes {
		t.Run("complex64_"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			// Create a non-trivial input signal
			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i*i), float32(i))
			}

			// Compute FFT with Bluestein
			fftResult := make([]complex64, n)
			if err := plan.Forward(fftResult, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Compute reference naive DFT
			naiveResult := reference.NaiveDFT(src)

			// Compare results
			// Note: complex64 has limited precision, especially for larger sizes
			// Tolerance is relaxed compared to complex128
			for i := range fftResult {
				diff := cmplx.Abs(complex128(fftResult[i] - naiveResult[i]))
				if diff > 1e-3 {
					t.Errorf("Bin %d: FFT=%v, Naive=%v, diff=%v", i, fftResult[i], naiveResult[i], diff)
				}
			}
		})

		t.Run("complex128_"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			// Create a non-trivial input signal
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i*i), float64(i))
			}

			// Compute FFT with Bluestein
			fftResult := make([]complex128, n)
			if err := plan.Forward(fftResult, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Compute reference naive DFT
			naiveResult := reference.NaiveDFT128(src)

			// Compare results
			for i := range fftResult {
				diff := cmplx.Abs(fftResult[i] - naiveResult[i])
				if diff > 1e-10 {
					t.Errorf("Bin %d: FFT=%v, Naive=%v, diff=%v", i, fftResult[i], naiveResult[i], diff)
				}
			}
		})
	}
}

// TestBluestein_InverseMatchesReference validates inverse Bluestein FFT against naive IDFT.
func TestBluestein_InverseMatchesReference(t *testing.T) {
	t.Parallel()

	primes := []int{7, 11, 13}

	for _, n := range primes {
		t.Run("complex128_"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			// Create frequency domain input
			freq := make([]complex128, n)
			for i := range freq {
				freq[i] = complex(float64(i), float64(-i*i))
			}

			// Compute IFFT with Bluestein
			ifftResult := make([]complex128, n)
			if err := plan.Inverse(ifftResult, freq); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			// Compute reference naive IDFT
			naiveResult := reference.NaiveIDFT128(freq)

			// Compare results
			for i := range ifftResult {
				diff := cmplx.Abs(ifftResult[i] - naiveResult[i])
				if diff > 1e-10 {
					t.Errorf("Bin %d: IFFT=%v, Naive=%v, diff=%v", i, ifftResult[i], naiveResult[i], diff)
				}
			}
		})
	}
}
