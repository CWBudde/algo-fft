package algofft

import (
	"errors"
	"math"
	"math/cmplx"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// Odd sizes exercised across the tests: small odds, odd primes (Bluestein/
// Rader complex fallbacks), 5-smooth composites, and prime powers.
var oddRealSizes = []int{3, 5, 7, 9, 15, 21, 25, 27, 45, 63, 81, 101, 105, 225, 343}

func testSignalF64(n int) []float64 {
	src := make([]float64, n)
	for i := range src {
		src[i] = math.Sin(0.2*float64(i)) + 0.5*math.Cos(0.7*float64(i))
	}

	return src
}

func TestPlanRealOddForwardMatchesReference32(t *testing.T) {
	t.Parallel()

	for _, n := range oddRealSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanReal32(n)
			if err != nil {
				t.Fatalf("NewPlanReal32(%d) returned error: %v", n, err)
			}

			if plan.SpectrumLen() != n/2+1 {
				t.Fatalf("SpectrumLen() = %d, want %d", plan.SpectrumLen(), n/2+1)
			}

			sig := testSignalF64(n)
			src := make([]float32, n)
			complexSrc := make([]complex64, n)

			for i, v := range sig {
				src[i] = float32(v)
				complexSrc[i] = complex(float32(v), 0)
			}

			dst := make([]complex64, plan.SpectrumLen())

			err = plan.Forward(dst, src)
			if err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			ref := reference.NaiveDFT(complexSrc)
			for k := range dst {
				assertApproxComplex64Tolf(t, dst[k], ref[k], 1e-3, "dst[%d]", k)
			}
		})
	}
}

func TestPlanRealOddForwardMatchesReference64(t *testing.T) {
	t.Parallel()

	for _, n := range oddRealSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanReal64(n)
			if err != nil {
				t.Fatalf("NewPlanReal64(%d) returned error: %v", n, err)
			}

			src := testSignalF64(n)

			complexSrc := make([]complex128, n)
			for i, v := range src {
				complexSrc[i] = complex(v, 0)
			}

			dst := make([]complex128, plan.SpectrumLen())

			err = plan.Forward(dst, src)
			if err != nil {
				t.Fatalf("Forward returned error: %v", err)
			}

			ref := reference.NaiveDFT128(complexSrc)
			for k := range dst {
				if cmplx.Abs(dst[k]-ref[k]) > 1e-9 {
					t.Errorf("dst[%d] = %v, want %v", k, dst[k], ref[k])
				}
			}
		})
	}
}

func TestPlanRealOddRoundTrip(t *testing.T) {
	t.Parallel()

	for _, n := range oddRealSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			t.Run("float32", func(t *testing.T) {
				t.Parallel()

				plan, err := NewPlanReal32(n)
				if err != nil {
					t.Fatalf("NewPlanReal32(%d) returned error: %v", n, err)
				}

				sig := testSignalF64(n)
				src := make([]float32, n)

				for i, v := range sig {
					src[i] = float32(v)
				}

				freq := make([]complex64, plan.SpectrumLen())
				out := make([]float32, n)

				if err := plan.Forward(freq, src); err != nil {
					t.Fatalf("Forward returned error: %v", err)
				}

				if err := plan.Inverse(out, freq); err != nil {
					t.Fatalf("Inverse returned error: %v", err)
				}

				for i := range src {
					if math.Abs(float64(out[i]-src[i])) > 1e-4 {
						t.Errorf("out[%d] = %v, want %v", i, out[i], src[i])
					}
				}
			})

			t.Run("float64", func(t *testing.T) {
				t.Parallel()

				plan, err := NewPlanReal64(n)
				if err != nil {
					t.Fatalf("NewPlanReal64(%d) returned error: %v", n, err)
				}

				src := testSignalF64(n)
				freq := make([]complex128, plan.SpectrumLen())
				out := make([]float64, n)

				if err := plan.Forward(freq, src); err != nil {
					t.Fatalf("Forward returned error: %v", err)
				}

				if err := plan.Inverse(out, freq); err != nil {
					t.Fatalf("Inverse returned error: %v", err)
				}

				for i := range src {
					if math.Abs(out[i]-src[i]) > 1e-10 {
						t.Errorf("out[%d] = %v, want %v", i, out[i], src[i])
					}
				}
			})
		})
	}
}

func TestPlanRealOddNormalizedAndUnitary(t *testing.T) {
	t.Parallel()

	const n = 15

	plan, err := NewPlanReal64(n)
	if err != nil {
		t.Fatalf("NewPlanReal64(%d) returned error: %v", n, err)
	}

	// Constant signal: DC = n (plain), 1 (normalized), sqrt(n) (unitary).
	src := make([]float64, n)
	for i := range src {
		src[i] = 1
	}

	dst := make([]complex128, plan.SpectrumLen())

	if err := plan.ForwardNormalized(dst, src); err != nil {
		t.Fatalf("ForwardNormalized returned error: %v", err)
	}

	if cmplx.Abs(dst[0]-1) > 1e-12 {
		t.Errorf("normalized DC = %v, want 1", dst[0])
	}

	if err := plan.ForwardUnitary(dst, src); err != nil {
		t.Fatalf("ForwardUnitary returned error: %v", err)
	}

	if cmplx.Abs(dst[0]-complex(math.Sqrt(n), 0)) > 1e-12 {
		t.Errorf("unitary DC = %v, want sqrt(%d)", dst[0], n)
	}
}

func TestPlanRealOddInverseValidation(t *testing.T) {
	t.Parallel()

	const n = 9

	plan, err := NewPlanReal64(n)
	if err != nil {
		t.Fatalf("NewPlanReal64(%d) returned error: %v", n, err)
	}

	out := make([]float64, n)
	freq := make([]complex128, plan.SpectrumLen())

	// Non-real DC must be rejected.
	freq[0] = complex(1, 1)

	err = plan.Inverse(out, freq)
	if !errors.Is(err, ErrInvalidSpectrum) {
		t.Errorf("expected ErrInvalidSpectrum for complex DC, got %v", err)
	}

	// The last bin is a regular bin for odd n (no Nyquist constraint): a
	// complex value there must be accepted.
	freq[0] = complex(1, 0)
	freq[plan.SpectrumLen()-1] = complex(0.5, 0.5)

	err = plan.Inverse(out, freq)
	if err != nil {
		t.Errorf("Inverse rejected a valid odd-length spectrum: %v", err)
	}
}

func TestPlanRealOddLengthMismatch(t *testing.T) {
	t.Parallel()

	const n = 21

	plan, err := NewPlanReal32(n)
	if err != nil {
		t.Fatalf("NewPlanReal32(%d) returned error: %v", n, err)
	}

	err = plan.Forward(make([]complex64, plan.SpectrumLen()-1), make([]float32, n))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("expected ErrLengthMismatch for short dst, got %v", err)
	}

	err = plan.Forward(make([]complex64, plan.SpectrumLen()), make([]float32, n+1))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("expected ErrLengthMismatch for long src, got %v", err)
	}

	err = plan.Inverse(make([]float32, n-1), make([]complex64, plan.SpectrumLen()))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("expected ErrLengthMismatch for short dst, got %v", err)
	}
}

func TestPlanRealOddClone(t *testing.T) {
	t.Parallel()

	const n = 27

	plan, err := NewPlanReal64(n)
	if err != nil {
		t.Fatalf("NewPlanReal64(%d) returned error: %v", n, err)
	}

	clone := plan.Clone()
	src := testSignalF64(n)
	freq := make([]complex128, clone.SpectrumLen())
	out := make([]float64, n)

	if err := clone.Forward(freq, src); err != nil {
		t.Fatalf("clone Forward returned error: %v", err)
	}

	if err := clone.Inverse(out, freq); err != nil {
		t.Fatalf("clone Inverse returned error: %v", err)
	}

	for i := range src {
		if math.Abs(out[i]-src[i]) > 1e-10 {
			t.Errorf("out[%d] = %v, want %v", i, out[i], src[i])
		}
	}
}

// BenchmarkPlanRealForwardOdd measures the odd-length complex-fallback real
// FFT against the plain complex plan of the same size (the previous only
// option for odd lengths, minus the caller-side widen/narrow copies).
func BenchmarkPlanRealForwardOdd(b *testing.B) {
	sizes := []int{135, 1215, 3645, 10935} // 5·27, 5·243, 5·729, 5·2187

	for _, n := range sizes {
		b.Run("Real_N="+strconv.Itoa(n), func(b *testing.B) {
			plan, err := NewPlanReal[float32, complex64](n)
			if err != nil {
				b.Fatalf("NewPlanReal[float32, complex64](%d) returned error: %v", n, err)
			}

			src := make([]float32, n)
			for i := range src {
				src[i] = float32(i)
			}

			dst := make([]complex64, plan.SpectrumLen())

			b.ReportAllocs()
			b.SetBytes(int64(n * 4)) // float32 = 4 bytes

			b.ResetTimer()

			for range b.N {
				_ = plan.Forward(dst, src)
			}
		})

		b.Run("Complex_N="+strconv.Itoa(n), func(b *testing.B) {
			plan, err := NewPlan[complex64](n)
			if err != nil {
				b.Fatalf("NewPlan[complex64](%d) returned error: %v", n, err)
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
	}
}
