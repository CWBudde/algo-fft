package algoforge

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algoforge/internal/reference"
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

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	for k := range dst {
		assertApproxComplex64(t, dst[k], complex(1, 0), 1e-5, "dst[%d]", k)
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

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	assertApproxComplex64(t, dst[0], complex(float32(n*value), 0), 1e-4, "dst[0]")

	for k := 1; k < len(dst); k++ {
		assertApproxComplex64(t, dst[k], 0, 1e-4, "dst[%d]", k)
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

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	target := float32(n / 2)

	for i := range dst {
		got := dst[i]
		if i == k {
			assertApproxComplex64(t, got, complex(target, 0), 1e-4, "dst[%d]", i)
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

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	complexSrc := make([]complex64, n)
	for i := range src {
		complexSrc[i] = complex(src[i], 0)
	}

	ref := reference.NaiveDFT(complexSrc)

	for k := range dst {
		assertApproxComplex64(t, dst[k], ref[k], 1e-4, "dst[%d]", k)
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

	if err := plan.Forward(dst, src); err != nil {
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
		xk := ref[k]
		xnk := ref[n-k]
		conjXnk := complex(real(xnk), -imag(xnk))

		if cmplx.Abs(complex128(xk-conjXnk)) > tol {
			t.Errorf("conjugate symmetry violated: X[%d]=%v != conj(X[%d])=%v",
				k, xk, n-k, conjXnk)
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

		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("Forward returned error: %v", err)
		}

		// DC = sum = 3, Nyquist = difference = -1
		assertApproxComplex64(t, dst[0], complex(float32(3), 0), 1e-5, "dst[0]")
		assertApproxComplex64(t, dst[1], complex(float32(-1), 0), 1e-5, "dst[1]")
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

		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("Forward returned error: %v", err)
		}

		// Impulse should give flat spectrum
		for k := range dst {
			assertApproxComplex64(t, dst[k], complex(float32(1), 0), 1e-4, "dst[%d]", k)
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

		if err := plan.Forward(nil, make([]float32, 16)); !errors.Is(err, ErrNilSlice) {
			t.Errorf("expected ErrNilSlice for nil dst, got %v", err)
		}

		if err := plan.Forward(make([]complex64, 9), nil); !errors.Is(err, ErrNilSlice) {
			t.Errorf("expected ErrNilSlice for nil src, got %v", err)
		}
	})

	t.Run("length mismatch", func(t *testing.T) {
		t.Parallel()

		plan, _ := NewPlanReal(16)

		if err := plan.Forward(make([]complex64, 8), make([]float32, 16)); !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("expected ErrLengthMismatch for wrong dst len, got %v", err)
		}

		if err := plan.Forward(make([]complex64, 9), make([]float32, 8)); !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("expected ErrLengthMismatch for wrong src len, got %v", err)
		}
	})
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
			plan, err := NewPlan[complex64](n)
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
