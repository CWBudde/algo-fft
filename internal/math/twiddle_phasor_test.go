package math

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestComputeTwiddleFactorsPhasor(t *testing.T) {
	t.Parallel()
	t.Run("edge cases", func(t *testing.T) {
		t.Parallel()
		// Zero size
		result := ComputeTwiddleFactorsPhasor[complex64](0)
		if result != nil {
			t.Errorf("ComputeTwiddleFactorsPhasor(0) = %v, want nil", result)
		}

		// Negative size
		result = ComputeTwiddleFactorsPhasor[complex64](-1)
		if result != nil {
			t.Errorf("ComputeTwiddleFactorsPhasor(-1) = %v, want nil", result)
		}

		// Size 1
		result = ComputeTwiddleFactorsPhasor[complex64](1)
		if len(result) != 1 {
			t.Fatalf("len = %d, want 1", len(result))
		}

		if result[0] != 1 {
			t.Errorf("twiddle[0] = %v, want 1", result[0])
		}
	})

	t.Run("small sizes use direct", func(t *testing.T) {
		t.Parallel()
		// Below threshold, should produce same results as direct
		for n := 2; n < PhasorThreshold; n++ {
			direct := ComputeTwiddleFactors[complex128](n)
			phasor := ComputeTwiddleFactorsPhasor[complex128](n)

			if len(direct) != len(phasor) {
				t.Errorf("n=%d: len(direct)=%d, len(phasor)=%d", n, len(direct), len(phasor))
				continue
			}

			for k := range direct {
				if direct[k] != phasor[k] {
					t.Errorf("n=%d, k=%d: direct=%v, phasor=%v", n, k, direct[k], phasor[k])
				}
			}
		}
	})
}

func TestPhasorExactValues(t *testing.T) {
	t.Parallel()

	sizes := []int{32, 64, 128, 256, 512, 1024, 2048, 4096}

	for _, n := range sizes {
		t.Run(formatSizePhasor(n), func(t *testing.T) {
			t.Parallel()

			twiddle := ComputeTwiddleFactorsPhasor[complex128](n)

			// W_0 = 1 (exact)
			if twiddle[0] != 1+0i {
				t.Errorf("W_0 = %v, want 1+0i", twiddle[0])
			}

			// W_{n/2} = -1 (exact)
			if twiddle[n/2] != -1+0i {
				t.Errorf("W_%d = %v, want -1+0i", n/2, twiddle[n/2])
			}

			// W_{n/4} = -i (exact, for n divisible by 4)
			if n%4 == 0 {
				if twiddle[n/4] != 0-1i {
					t.Errorf("W_%d = %v, want 0-1i", n/4, twiddle[n/4])
				}
			}

			// W_{3n/4} = i (exact, for n divisible by 4)
			if n%4 == 0 {
				if twiddle[3*n/4] != 0+1i {
					t.Errorf("W_%d = %v, want 0+1i", 3*n/4, twiddle[3*n/4])
				}
			}
		})
	}
}

func TestPhasorMagnitude(t *testing.T) {
	t.Parallel()

	sizes := []int{32, 64, 128, 256, 512, 1024, 4096, 8192}

	for _, n := range sizes {
		t.Run(formatSizePhasor(n), func(t *testing.T) {
			t.Parallel()

			twiddle64 := ComputeTwiddleFactorsPhasor[complex64](n)
			twiddle128 := ComputeTwiddleFactorsPhasor[complex128](n)

			const (
				eps64  = 1e-5
				eps128 = 1e-13
			)

			// All twiddle factors should have magnitude 1 (roots of unity)

			for k, w := range twiddle64 {
				mag := cmplx.Abs(complex128(w))
				if math.Abs(mag-1.0) > eps64 {
					t.Errorf("complex64[%d] magnitude = %v, want 1.0 (error = %v)", k, mag, math.Abs(mag-1.0))
				}
			}

			for k, w := range twiddle128 {
				mag := cmplx.Abs(w)
				if math.Abs(mag-1.0) > eps128 {
					t.Errorf("complex128[%d] magnitude = %v, want 1.0 (error = %v)", k, mag, math.Abs(mag-1.0))
				}
			}
		})
	}
}

func TestPhasorSymmetry(t *testing.T) {
	t.Parallel()
	// Property: W_n^k and W_n^(n-k) are complex conjugates
	sizes := []int{32, 64, 128, 256, 512, 1024}

	for _, n := range sizes {
		t.Run(formatSizePhasor(n), func(t *testing.T) {
			t.Parallel()

			twiddle := ComputeTwiddleFactorsPhasor[complex128](n)

			const eps = 1e-13

			for k := 1; k < n/2; k++ {
				wk := twiddle[k]
				wnk := twiddle[n-k]
				expectedConj := Conj(wk)

				if !approxEqualPhasor128(wnk, expectedConj, eps) {
					t.Errorf("W_%d^%d = %v, W_%d^%d = %v, expected conjugates (diff = %v)",
						n, k, wk, n, n-k, wnk, cmplx.Abs(wnk-expectedConj))
				}
			}
		})
	}
}

func TestPhasorVsDirect(t *testing.T) {
	t.Parallel()
	// Compare phasor results against direct computation
	sizes := []int{32, 64, 128, 256, 512, 1024, 2048, 4096}

	for _, n := range sizes {
		t.Run(formatSizePhasor(n), func(t *testing.T) {
			t.Parallel()

			direct := ComputeTwiddleFactors[complex128](n)
			phasor := ComputeTwiddleFactorsPhasor[complex128](n)

			if len(direct) != len(phasor) {
				t.Fatalf("len(direct)=%d, len(phasor)=%d", len(direct), len(phasor))
			}

			// complex128 tolerance: block size 1024 should keep error < 1e-12
			const eps = 1e-12

			var (
				maxError    float64
				maxErrorIdx int
			)

			for k := range direct {
				diff := cmplx.Abs(direct[k] - phasor[k])
				if diff > maxError {
					maxError = diff
					maxErrorIdx = k
				}
			}

			if maxError > eps {
				t.Errorf("max error = %v at index %d (tolerance = %v)", maxError, maxErrorIdx, eps)
			}
		})
	}
}

func TestPhasorVsDirectComplex64(t *testing.T) {
	t.Parallel()
	// Test complex64 separately with looser tolerance
	sizes := []int{32, 64, 128, 256, 512, 1024, 2048, 4096}

	for _, n := range sizes {
		t.Run(formatSizePhasor(n), func(t *testing.T) {
			t.Parallel()

			direct := ComputeTwiddleFactors[complex64](n)
			phasor := ComputeTwiddleFactorsPhasor[complex64](n)

			if len(direct) != len(phasor) {
				t.Fatalf("len(direct)=%d, len(phasor)=%d", len(direct), len(phasor))
			}

			// complex64 tolerance: block size 64 should keep error < 1e-5
			const eps = 1e-5

			var (
				maxError    float64
				maxErrorIdx int
			)

			for k := range direct {
				diff := cmplx.Abs(complex128(direct[k]) - complex128(phasor[k]))
				if diff > maxError {
					maxError = diff
					maxErrorIdx = k
				}
			}

			if maxError > eps {
				t.Errorf("max error = %v at index %d (tolerance = %v)", maxError, maxErrorIdx, eps)
			}
		})
	}
}

func TestPhasorLargeSizes(t *testing.T) {
	t.Parallel()
	// Test with very large sizes to stress error accumulation
	if testing.Short() {
		t.Skip("skipping large size test in short mode")
	}

	sizes := []int{8192, 16384, 32768, 65536}

	for _, n := range sizes {
		t.Run(formatSizePhasor(n), func(t *testing.T) {
			t.Parallel()

			twiddle128 := ComputeTwiddleFactorsPhasor[complex128](n)

			// Check magnitude for all elements
			const eps = 1e-12

			var maxMagError float64

			for k, w := range twiddle128 {
				mag := cmplx.Abs(w)

				magError := math.Abs(mag - 1.0)
				if magError > maxMagError {
					maxMagError = magError
				}

				if magError > eps {
					t.Errorf("complex128[%d] magnitude error = %v (>%v)", k, magError, eps)
					break
				}
			}

			// Check exact values
			if twiddle128[0] != 1+0i {
				t.Errorf("W_0 = %v, want 1+0i", twiddle128[0])
			}

			if twiddle128[n/2] != -1+0i {
				t.Errorf("W_%d = %v, want -1+0i", n/2, twiddle128[n/2])
			}
		})
	}
}

// Helper functions

func approxEqualPhasor128(a, b complex128, eps float64) bool {
	return math.Abs(real(a)-real(b)) < eps &&
		math.Abs(imag(a)-imag(b)) < eps
}

func formatSizePhasor(n int) string {
	if n >= 1000 {
		return formatIntPhasor(n/1000) + "k"
	}

	return formatIntPhasor(n)
}

func formatIntPhasor(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}

	if n < 100 {
		return string(rune('0'+n/10)) + string(rune('0'+n%10))
	}

	hundreds := n / 100
	tens := (n % 100) / 10
	ones := n % 10

	return string(rune('0'+hundreds)) + string(rune('0'+tens)) + string(rune('0'+ones))
}

// Benchmarks

func BenchmarkComputeTwiddleFactorsPhasor(b *testing.B) {
	sizes := []int{32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}

	for _, size := range sizes {
		b.Run("complex64/"+formatSizePhasor(size), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_ = ComputeTwiddleFactorsPhasor[complex64](size)
			}
		})

		b.Run("complex128/"+formatSizePhasor(size), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_ = ComputeTwiddleFactorsPhasor[complex128](size)
			}
		})
	}
}

func BenchmarkTwiddleDirectVsPhasor(b *testing.B) {
	sizes := []int{256, 1024, 4096, 16384, 65536}

	for _, size := range sizes {
		b.Run("Direct/complex128/"+formatSizePhasor(size), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_ = ComputeTwiddleFactors[complex128](size)
			}
		})

		b.Run("Phasor/complex128/"+formatSizePhasor(size), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_ = ComputeTwiddleFactorsPhasor[complex128](size)
			}
		})
	}
}
