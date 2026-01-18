package algofft

import (
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestNewPlan_Bluestein_EdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("N=1", func(t *testing.T) {
		t.Parallel()

		plan, err := NewPlanT[complex64](1)
		if err != nil {
			t.Fatalf("NewPlan(1) failed: %v", err)
		}

		src := []complex64{complex(42, 0)}
		dst := make([]complex64, 1)

		err = plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		// For N=1, DFT is identity: X[0] = x[0]
		if dst[0] != src[0] {
			t.Errorf("Forward: got %v, want %v", dst[0], src[0])
		}

		back := make([]complex64, 1)

		err = plan.Inverse(back, dst)
		if err != nil {
			t.Fatalf("Inverse failed: %v", err)
		}

		if cmplx.Abs(complex128(back[0]-src[0])) > 1e-5 {
			t.Errorf("Inverse: got %v, want %v", back[0], src[0])
		}
	})

	t.Run("N=2_prime", func(t *testing.T) {
		t.Parallel()

		// N=2 is power of 2, but test it doesn't break if forced to Bluestein
		plan, err := NewPlanT[complex64](2)
		if err != nil {
			t.Fatalf("NewPlan(2) failed: %v", err)
		}

		src := []complex64{complex(1, 0), complex(2, 0)}
		dst := make([]complex64, 2)

		err = plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		back := make([]complex64, 2)

		err = plan.Inverse(back, dst)
		if err != nil {
			t.Fatalf("Inverse failed: %v", err)
		}

		for i := range src {
			if cmplx.Abs(complex128(back[i]-src[i])) > 1e-5 {
				t.Errorf("Round-trip mismatch at %d: got %v, want %v", i, back[i], src[i])
			}
		}
	})
}

func TestNewPlan_Bluestein(t *testing.T) {
	t.Parallel()

	// Prime lengths trigger Bluestein
	primes := []int{7, 11, 13, 17}
	for _, n := range primes {
		t.Run("complex64_"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanT[complex64](n)
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

			plan, err := NewPlanT[complex128](n)
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

		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) failed: %v", n, err)
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i), 0)
		}

		dst := make([]complex64, n)

		err = plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		back := make([]complex64, n)

		err = plan.Inverse(back, dst)
		if err != nil {
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

		plan, err := NewPlanT[complex128](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) failed: %v", n, err)
		}

		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(float64(i), 0)
		}

		dst := make([]complex128, n)

		err = plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		back := make([]complex128, n)

		err = plan.Inverse(back, dst)
		if err != nil {
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

			plan, err := NewPlanT[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i), float32(-i))
			}

			dst := make([]complex64, n)

			err = plan.Forward(dst, src)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			back := make([]complex64, n)

			err = plan.Inverse(back, dst)
			if err != nil {
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
//
//nolint:gocognit
func TestBluestein_MatchesReference(t *testing.T) {
	t.Parallel()

	// Test various prime sizes
	primes := []int{7, 11, 13, 17, 19, 23, 31}

	for _, n := range primes {
		t.Run("complex64_"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanT[complex64](n)
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

			err = plan.Forward(fftResult, src)
			if err != nil {
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

			plan, err := NewPlanT[complex128](n)
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

			err = plan.Forward(fftResult, src)
			if err != nil {
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

			plan, err := NewPlanT[complex128](n)
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

			err = plan.Inverse(ifftResult, freq)
			if err != nil {
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

// BenchmarkBluesteinVsNaive compares Bluestein FFT performance against naive O(N²) DFT.
// This demonstrates that Bluestein achieves O(N log N) complexity for prime sizes.
func BenchmarkBluesteinVsNaive(b *testing.B) {
	primes := []int{7, 13, 31, 127}

	for _, n := range primes {
		b.Run("Bluestein_"+itoa(n), func(b *testing.B) {
			plan, err := NewPlanT[complex64](n)
			if err != nil {
				b.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i), 0)
			}

			dst := make([]complex64, n)

			b.ReportAllocs()
			b.SetBytes(int64(n * 8)) // complex64 = 8 bytes

			b.ResetTimer()

			for range b.N {
				_ = plan.Forward(dst, src)
			}
		})

		b.Run("Naive_"+itoa(n), func(b *testing.B) {
			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i), 0)
			}

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))

			b.ResetTimer()

			for range b.N {
				_ = reference.NaiveDFT(src)
			}
		})
	}
}

// BenchmarkBluesteinForward benchmarks Bluestein forward transform for various prime sizes.
func BenchmarkBluesteinForward(b *testing.B) {
	primes := []int{7, 13, 31, 127, 509}

	for _, n := range primes {
		b.Run("complex64_"+itoa(n), func(b *testing.B) {
			plan, err := NewPlanT[complex64](n)
			if err != nil {
				b.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i), float32(-i))
			}

			dst := make([]complex64, n)

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))

			b.ResetTimer()

			for range b.N {
				_ = plan.Forward(dst, src)
			}
		})

		b.Run("complex128_"+itoa(n), func(b *testing.B) {
			plan, err := NewPlanT[complex128](n)
			if err != nil {
				b.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i), float64(-i))
			}

			dst := make([]complex128, n)

			b.ReportAllocs()
			b.SetBytes(int64(n * 16))

			b.ResetTimer()

			for range b.N {
				_ = plan.Forward(dst, src)
			}
		})
	}
}

// BenchmarkBluesteinInverse benchmarks Bluestein inverse transform.
func BenchmarkBluesteinInverse(b *testing.B) {
	primes := []int{7, 13, 31, 127, 509}

	for _, n := range primes {
		b.Run("complex64_"+itoa(n), func(b *testing.B) {
			plan, err := NewPlanT[complex64](n)
			if err != nil {
				b.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			freq := make([]complex64, n)
			for i := range freq {
				freq[i] = complex(float32(i), float32(-i))
			}

			dst := make([]complex64, n)

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))

			b.ResetTimer()

			for range b.N {
				_ = plan.Inverse(dst, freq)
			}
		})

		b.Run("complex128_"+itoa(n), func(b *testing.B) {
			plan, err := NewPlanT[complex128](n)
			if err != nil {
				b.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			freq := make([]complex128, n)
			for i := range freq {
				freq[i] = complex(float64(i), float64(-i))
			}

			dst := make([]complex128, n)

			b.ReportAllocs()
			b.SetBytes(int64(n * 16))

			b.ResetTimer()

			for range b.N {
				_ = plan.Inverse(dst, freq)
			}
		})
	}
}

// BenchmarkBluesteinRoundTrip benchmarks a complete round-trip transform.
func BenchmarkBluesteinRoundTrip(b *testing.B) {
	primes := []int{13, 31, 127}

	for _, n := range primes {
		b.Run(itoa(n), func(b *testing.B) {
			plan, err := NewPlanT[complex64](n)
			if err != nil {
				b.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i), 0)
			}

			freq := make([]complex64, n)
			dst := make([]complex64, n)

			b.ReportAllocs()
			b.SetBytes(int64(n * 8 * 2)) // 2 transforms

			b.ResetTimer()

			for range b.N {
				_ = plan.Forward(freq, src)
				_ = plan.Inverse(dst, freq)
			}
		})
	}
}
