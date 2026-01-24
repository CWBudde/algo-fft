package algofft

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestPlanRealForwardImpulse(t *testing.T) {
	t.Parallel()

	const n = 16

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
	}

	src := make([]float32, n)
	src[0] = 1
	dst := make([]complex64, n/2+1)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	for k := range dst {
		assertApproxComplex64f(t, dst[k], complex(1, 0), 1e-5, "dst[%d]", k)
	}
}

func TestPlanReal_BatchStrideRoundTrip(t *testing.T) {
	t.Parallel()

	const (
		n      = 16
		batch  = 2
		stride = n + 5
	)

	plan, err := NewPlanRealWithOptions(n, PlanOptions{
		Batch:  batch,
		Stride: stride,
	})
	if err != nil {
		t.Fatalf("NewPlanRealWithOptions failed: %v", err)
	}

	src := make([]float32, batch*stride)
	dst := make([]complex64, batch*stride)
	roundTrip := make([]float32, batch*stride)

	rng := rand.New(rand.NewSource(42))

	for b := range batch {
		base := b * stride
		for i := range n {
			src[base+i] = float32(rng.Float64()*2 - 1)
		}
	}

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	err = plan.Inverse(roundTrip, dst)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	const tol = 1e-3

	for b := range batch {
		base := b * stride
		for i := range n {
			if math.Abs(float64(roundTrip[base+i]-src[base+i])) > tol {
				t.Fatalf("batch %d idx %d mismatch: got %v want %v", b, i, roundTrip[base+i], src[base+i])
			}
		}
	}
}

func TestPlanRealForwardConstant(t *testing.T) {
	t.Parallel()

	const (
		n     = 32
		value = 2.0
	)

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
	}

	src := make([]float32, n)
	for i := range src {
		src[i] = value
	}

	dst := make([]complex64, n/2+1)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	assertApproxComplex64f(t, dst[0], complex(float32(n*value), 0), 1e-4, "dst[0]")

	for k := 1; k < len(dst); k++ {
		assertApproxComplex64f(t, dst[k], 0, 1e-4, "dst[%d]", k)
	}
}

func TestPlanRealForwardCosine(t *testing.T) {
	t.Parallel()

	const (
		n = 32
		k = 3
	)

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
	}

	src := make([]float32, n)
	for i := range src {
		angle := 2 * math.Pi * float64(k) * float64(i) / float64(n)
		src[i] = float32(math.Cos(angle))
	}

	dst := make([]complex64, n/2+1)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	target := float32(n / 2)

	for i := range dst {
		got := dst[i]
		if i == k {
			assertApproxComplex64f(t, got, complex(target, 0), 1e-4, "dst[%d]", i)
			continue
		}

		if cmplx.Abs(complex128(got)) > 1e-3 {
			t.Fatalf("dst[%d] = %v want ~0", i, got)
		}
	}
}

func TestPlanRealForwardMatchesReference(t *testing.T) {
	t.Parallel()

	const n = 32

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
	}

	src := make([]float32, n)
	for i := range src {
		src[i] = float32(math.Sin(0.2*float64(i)) + 0.5*math.Cos(0.7*float64(i)))
	}

	dst := make([]complex64, n/2+1)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	complexSrc := make([]complex64, n)
	for i := range src {
		complexSrc[i] = complex(src[i], 0)
	}

	ref := reference.NaiveDFT(complexSrc)

	for k := range dst {
		assertApproxComplex64f(t, dst[k], ref[k], 1e-4, "dst[%d]", k)
	}
}

func TestPlanRealForwardConjugateSymmetry(t *testing.T) {
	t.Parallel()

	const n = 64

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
	}

	// Use a mixed signal to get non-trivial spectrum
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(math.Sin(0.3*float64(i)) + 0.7*math.Cos(0.5*float64(i)))
	}

	dst := make([]complex64, n/2+1)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	// Compute full spectrum via reference to verify conjugate symmetry
	complexSrc := make([]complex64, n)
	for i := range src {
		complexSrc[i] = complex(src[i], 0)
	}

	ref := reference.NaiveDFT(complexSrc)

	// Verify X[k] = conj(X[N-k]) for k = 1..N/2-1
	const tol = 1e-4

	for k := 1; k < n/2; k++ {
		spectrumValueAtK := ref[k]
		spectrumValueAtNMinusK := ref[n-k]
		conjugateOfSpectrumValueAtNMinusK := complex(real(spectrumValueAtNMinusK), -imag(spectrumValueAtNMinusK))

		if cmplx.Abs(complex128(spectrumValueAtK-conjugateOfSpectrumValueAtNMinusK)) > tol {
			t.Errorf("conjugate symmetry violated: X[%d]=%v != conj(X[%d])=%v",
				k, spectrumValueAtK, n-k, conjugateOfSpectrumValueAtNMinusK)
		}
	}

	// Verify DC and Nyquist are purely real
	if imag(dst[0]) != 0 {
		t.Errorf("DC bin should be purely real, got imag=%v", imag(dst[0]))
	}

	if imag(dst[n/2]) != 0 {
		t.Errorf("Nyquist bin should be purely real, got imag=%v", imag(dst[n/2]))
	}
}

//nolint:funlen
func TestPlanRealRoundTripSignals(t *testing.T) {
	t.Parallel()

	sizes := []int{32, 128}
	for _, n := range sizes {
		t.Run(fmt.Sprintf("N=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("noise", func(t *testing.T) {
				t.Parallel()

				plan, err := NewPlanReal(n)
				if err != nil {
					t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
				}

				rng := rand.New(rand.NewSource(1))

				src := make([]float32, n)
				for i := range src {
					src[i] = float32(rng.Float64()*2 - 1)
				}

				assertPlanRealRoundTrip(t, plan, src, 1e-3)
			})

			t.Run("tones", func(t *testing.T) {
				t.Parallel()

				plan, err := NewPlanReal(n)
				if err != nil {
					t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
				}

				src := make([]float32, n)
				for i := range src {
					angle1 := 2 * math.Pi * 3 * float64(i) / float64(n)
					angle2 := 2 * math.Pi * 5 * float64(i) / float64(n)
					src[i] = float32(math.Cos(angle1) + 0.5*math.Sin(angle2))
				}

				assertPlanRealRoundTrip(t, plan, src, 1e-3)
			})

			t.Run("chirp", func(t *testing.T) {
				t.Parallel()

				plan, err := NewPlanReal(n)
				if err != nil {
					t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
				}

				const (
					baseFreq = 1.0
					addFreq  = 8.0
				)

				src := make([]float32, n)
				phase := 0.0

				for i := range src {
					tp := float64(i) / float64(n-1)
					freq := baseFreq + (addFreq-baseFreq)*tp
					phase += 2 * math.Pi * freq / float64(n)
					src[i] = float32(math.Sin(phase))
				}

				assertPlanRealRoundTrip(t, plan, src, 1e-3)
			})
		})
	}
}

func TestPlanRealEdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("n=2", func(t *testing.T) {
		t.Parallel()

		plan, err := NewPlanReal(2)
		if err != nil {
			t.Fatalf("NewPlanReal(2) returned error: %v", err)
		}

		src := []float32{1, 2}
		dst := make([]complex64, 2)

		err = plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward returned error: %v", err)
		}

		// DC = sum = 3, Nyquist = difference = -1
		assertApproxComplex64f(t, dst[0], complex(float32(3), 0), 1e-5, "dst[0]")
		assertApproxComplex64f(t, dst[1], complex(float32(-1), 0), 1e-5, "dst[1]")
	})

	t.Run("n=4096", func(t *testing.T) {
		t.Parallel()

		const n = 4096

		plan, err := NewPlanReal(n)
		if err != nil {
			t.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
		}

		src := make([]float32, n)
		src[0] = 1 // impulse

		dst := make([]complex64, n/2+1)

		err = plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward returned error: %v", err)
		}

		// Impulse should give flat spectrum
		for k := range dst {
			assertApproxComplex64f(t, dst[k], complex(float32(1), 0), 1e-4, "dst[%d]", k)
		}
	})
}

func TestPlanRealErrors(t *testing.T) {
	t.Parallel()

	t.Run("invalid length 0", func(t *testing.T) {
		t.Parallel()

		_, err := NewPlanReal(0)
		if !errors.Is(err, ErrInvalidLength) {
			t.Errorf("expected ErrInvalidLength, got %v", err)
		}
	})

	t.Run("invalid length 1", func(t *testing.T) {
		t.Parallel()

		_, err := NewPlanReal(1)
		if !errors.Is(err, ErrInvalidLength) {
			t.Errorf("expected ErrInvalidLength, got %v", err)
		}
	})

	t.Run("odd length", func(t *testing.T) {
		t.Parallel()

		_, err := NewPlanReal(7)
		if !errors.Is(err, ErrInvalidLength) {
			t.Errorf("expected ErrInvalidLength, got %v", err)
		}
	})

	t.Run("nil slices", func(t *testing.T) {
		t.Parallel()

		plan, _ := NewPlanReal(16)

		err := plan.Forward(nil, make([]float32, 16))
		if !errors.Is(err, ErrNilSlice) {
			t.Errorf("expected ErrNilSlice for nil dst, got %v", err)
		}

		err = plan.Forward(make([]complex64, 9), nil)
		if !errors.Is(err, ErrNilSlice) {
			t.Errorf("expected ErrNilSlice for nil src, got %v", err)
		}
	})

	t.Run("length mismatch", func(t *testing.T) {
		t.Parallel()

		plan, _ := NewPlanReal(16)

		err := plan.Forward(make([]complex64, 8), make([]float32, 16))
		if !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("expected ErrLengthMismatch for wrong dst len, got %v", err)
		}

		err = plan.Forward(make([]complex64, 9), make([]float32, 8))
		if !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("expected ErrLengthMismatch for wrong src len, got %v", err)
		}
	})
}

func assertPlanRealRoundTrip(t *testing.T, plan *PlanReal, src []float32, tol float64) {
	t.Helper()

	freq := make([]complex64, plan.SpectrumLen())

	err := plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	out := make([]float32, plan.Len())

	err = plan.Inverse(out, freq)
	if err != nil {
		t.Fatalf("Inverse returned error: %v", err)
	}

	for i := range src {
		if math.Abs(float64(out[i]-src[i])) > tol {
			t.Fatalf("out[%d] = %v want %v", i, out[i], src[i])
		}
	}
}

func BenchmarkPlanRealForward(b *testing.B) {
	sizes := []int{256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("Real_N=%d", n), func(b *testing.B) {
			plan, err := NewPlanReal(n)
			if err != nil {
				b.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
			}

			src := make([]float32, n)
			for i := range src {
				src[i] = float32(i)
			}

			dst := make([]complex64, n/2+1)

			b.ReportAllocs()
			b.SetBytes(int64(n * 4)) // float32 = 4 bytes

			b.ResetTimer()

			for range b.N {
				_ = plan.Forward(dst, src)
			}
		})

		b.Run(fmt.Sprintf("Complex_N=%d", n), func(b *testing.B) {
			plan, err := NewPlanT[complex64](n)
			if err != nil {
				b.Fatalf("NewPlan(%d) returned error: %v", n, err)
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

func BenchmarkPlanRealInverse(b *testing.B) {
	sizes := []int{256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("Real_N=%d", n), func(b *testing.B) {
			plan, err := NewPlanReal(n)
			if err != nil {
				b.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
			}

			src := make([]float32, n)
			for i := range src {
				src[i] = float32(i)
			}

			freq := make([]complex64, n/2+1)

			err = plan.Forward(freq, src)
			if err != nil {
				b.Fatalf("Forward returned error: %v", err)
			}

			dst := make([]float32, n)

			b.ReportAllocs()
			b.SetBytes(int64(n * 4)) // float32 = 4 bytes

			b.ResetTimer()

			for range b.N {
				_ = plan.Inverse(dst, freq)
			}
		})
	}
}

// TestPlanReal_ForwardNormalized tests normalized forward transform.
func TestPlanReal_ForwardNormalized(t *testing.T) {
	t.Parallel()

	n := 64

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) failed: %v", n, err)
	}

	// Create constant input (all ones)
	input := make([]float32, n)
	for i := range input {
		input[i] = 1.0
	}

	// Get normal forward output
	outputNormal := make([]complex64, plan.SpectrumLen())
	err = plan.Forward(outputNormal, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Get normalized forward output
	outputNormalized := make([]complex64, plan.SpectrumLen())
	err = plan.ForwardNormalized(outputNormalized, input)
	if err != nil {
		t.Fatalf("ForwardNormalized failed: %v", err)
	}

	// Check that normalized output equals normal output scaled by 1/N
	scale := float32(1.0 / float64(n))
	for k := range outputNormalized {
		expected := complex(real(outputNormal[k])*scale, imag(outputNormal[k])*scale)
		if cmplx.Abs(complex128(outputNormalized[k]-expected)) > 1e-5 {
			t.Errorf("bin[%d]: got %v, want %v", k, outputNormalized[k], expected)
		}
	}
}

// TestPlanReal_ForwardUnitary tests unitary forward transform.
func TestPlanReal_ForwardUnitary(t *testing.T) {
	t.Parallel()

	n := 64

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) failed: %v", n, err)
	}

	// Create test signal
	input := make([]float32, n)
	for i := range input {
		input[i] = 1.0
	}

	// Get normal forward output
	outputNormal := make([]complex64, plan.SpectrumLen())
	err = plan.Forward(outputNormal, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Get unitary forward output
	outputUnitary := make([]complex64, plan.SpectrumLen())
	err = plan.ForwardUnitary(outputUnitary, input)
	if err != nil {
		t.Fatalf("ForwardUnitary failed: %v", err)
	}

	// Check that unitary output equals normal output scaled by 1/sqrt(N)
	scale := float32(1.0 / math.Sqrt(float64(n)))
	for k := range outputUnitary {
		expected := complex(real(outputNormal[k])*scale, imag(outputNormal[k])*scale)
		if cmplx.Abs(complex128(outputUnitary[k]-expected)) > 1e-5 {
			t.Errorf("bin[%d]: got %v, want %v", k, outputUnitary[k], expected)
		}
	}
}

// TestPlanReal_NormalizedWithRealSignal tests normalized transform with mixed signal.
func TestPlanReal_NormalizedWithRealSignal(t *testing.T) {
	t.Parallel()

	n := 128

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) failed: %v", n, err)
	}

	// Create a mix of frequencies
	input := make([]float32, n)
	for i := range input {
		t := float64(i) / float64(n)
		input[i] = float32(math.Sin(2*math.Pi*3*t) + 0.5*math.Cos(2*math.Pi*7*t))
	}

	// Forward and normalized
	spectrum := make([]complex64, plan.SpectrumLen())
	err = plan.ForwardNormalized(spectrum, input)
	if err != nil {
		t.Fatalf("ForwardNormalized failed: %v", err)
	}

	// Verify non-zero values at expected frequencies
	// Should have peaks at bins 3 and 7
	if cmplx.Abs(complex128(spectrum[3])) < 0.1 {
		t.Errorf("Expected significant energy at bin 3, got %v", spectrum[3])
	}
	if cmplx.Abs(complex128(spectrum[7])) < 0.1 {
		t.Errorf("Expected significant energy at bin 7, got %v", spectrum[7])
	}
}

// TestPlanReal_UnitaryWithRealSignal tests unitary transform with mixed signal.
func TestPlanReal_UnitaryWithRealSignal(t *testing.T) {
	t.Parallel()

	n := 128

	plan, err := NewPlanReal(n)
	if err != nil {
		t.Fatalf("NewPlanReal(%d) failed: %v", n, err)
	}

	// Create a mix of frequencies
	input := make([]float32, n)
	for i := range input {
		t := float64(i) / float64(n)
		input[i] = float32(math.Sin(2*math.Pi*3*t) + 0.5*math.Cos(2*math.Pi*7*t))
	}

	// Forward unitary
	spectrum := make([]complex64, plan.SpectrumLen())
	err = plan.ForwardUnitary(spectrum, input)
	if err != nil {
		t.Fatalf("ForwardUnitary failed: %v", err)
	}

	// Verify non-zero values at expected frequencies
	// Should have peaks at bins 3 and 7
	if cmplx.Abs(complex128(spectrum[3])) < 0.1 {
		t.Errorf("Expected significant energy at bin 3, got %v", spectrum[3])
	}
	if cmplx.Abs(complex128(spectrum[7])) < 0.1 {
		t.Errorf("Expected significant energy at bin 7, got %v", spectrum[7])
	}
}

// TestPlanReal_NormalizedErrors tests error handling in normalized transforms.
func TestPlanReal_NormalizedErrors(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanReal(16)
	if err != nil {
		t.Fatalf("NewPlanReal failed: %v", err)
	}

	t.Run("ForwardNormalized_NilInput", func(t *testing.T) {
		t.Parallel()

		dst := make([]complex64, plan.SpectrumLen())
		err := plan.ForwardNormalized(dst, nil)
		if !errors.Is(err, ErrNilSlice) {
			t.Errorf("expected ErrNilSlice, got %v", err)
		}
	})

	t.Run("ForwardNormalized_WrongLength", func(t *testing.T) {
		t.Parallel()

		dst := make([]complex64, plan.SpectrumLen())
		src := make([]float32, 8) // wrong length
		err := plan.ForwardNormalized(dst, src)
		if !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("expected ErrLengthMismatch, got %v", err)
		}
	})

	t.Run("ForwardUnitary_NilInput", func(t *testing.T) {
		t.Parallel()

		dst := make([]complex64, plan.SpectrumLen())
		err := plan.ForwardUnitary(dst, nil)
		if !errors.Is(err, ErrNilSlice) {
			t.Errorf("expected ErrNilSlice, got %v", err)
		}
	})

	t.Run("ForwardUnitary_WrongLength", func(t *testing.T) {
		t.Parallel()

		dst := make([]complex64, plan.SpectrumLen())
		src := make([]float32, 8) // wrong length
		err := plan.ForwardUnitary(dst, src)
		if !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("expected ErrLengthMismatch, got %v", err)
		}
	})
}

func BenchmarkPlanRealForwardNormalized(b *testing.B) {
	sizes := []int{256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("Real_N=%d", n), func(b *testing.B) {
			plan, err := NewPlanReal(n)
			if err != nil {
				b.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
			}

			src := make([]float32, n)
			for i := range src {
				src[i] = float32(i)
			}

			dst := make([]complex64, n/2+1)

			b.ReportAllocs()
			b.SetBytes(int64(n * 4)) // float32 = 4 bytes

			b.ResetTimer()

			for range b.N {
				_ = plan.ForwardNormalized(dst, src)
			}
		})
	}
}

func BenchmarkPlanRealForwardUnitary(b *testing.B) {
	sizes := []int{256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("Real_N=%d", n), func(b *testing.B) {
			plan, err := NewPlanReal(n)
			if err != nil {
				b.Fatalf("NewPlanReal(%d) returned error: %v", n, err)
			}

			src := make([]float32, n)
			for i := range src {
				src[i] = float32(i)
			}

			dst := make([]complex64, n/2+1)

			b.ReportAllocs()
			b.SetBytes(int64(n * 4)) // float32 = 4 bytes

			b.ResetTimer()

			for range b.N {
				_ = plan.ForwardUnitary(dst, src)
			}
		})
	}
}
