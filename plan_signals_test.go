package algofft

import (
	"math"
	"testing"
)

func TestFFTConstantSignal(t *testing.T) {
	t.Parallel()

	n := 16

	plan, err := NewPlanT[complex64](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) returned error: %v", n, err)
	}

	src := make([]complex64, n)
	for i := range src {
		src[i] = 1 + 0i
	}

	dst := make([]complex64, n)
	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	expectDC := complex(float32(n), 0)
	if !complexSliceEqual([]complex64{dst[0]}, []complex64{expectDC}, 1e-3) {
		t.Fatalf("DC bin = %v, want %v", dst[0], expectDC)
	}

	for i := 1; i < n; i++ {
		if absComplex64(dst[i]) > 1e-3 {
			t.Fatalf("bin[%d] = %v, want near 0", i, dst[i])
		}
	}
}

func TestFFTPureSinusoid(t *testing.T) {
	t.Parallel()

	n := 32
	k := 3

	plan, err := NewPlanT[complex64](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) returned error: %v", n, err)
	}

	src := make([]complex64, n)
	for i := range src {
		angle := 2 * math.Pi * float64(k) * float64(i) / float64(n)
		src[i] = complex64(complex(math.Cos(angle), math.Sin(angle)))
	}

	dst := make([]complex64, n)
	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	for i := range dst {
		if i == k {
			if absComplex64(dst[i]-complex(float32(n), 0)) > 1e-2 {
				t.Fatalf("bin[%d] = %v, want %v", i, dst[i], complex(float32(n), 0))
			}

			continue
		}

		if absComplex64(dst[i]) > 1e-2 {
			t.Fatalf("bin[%d] = %v, want near 0", i, dst[i])
		}
	}
}

func TestFFTNyquistFrequency(t *testing.T) {
	t.Parallel()

	n := 32

	plan, err := NewPlanT[complex64](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) returned error: %v", n, err)
	}

	src := make([]complex64, n)
	for i := range src {
		if i%2 == 0 {
			src[i] = 1
		} else {
			src[i] = -1
		}
	}

	dst := make([]complex64, n)
	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward() returned error: %v", err)
	}

	nyquist := n / 2
	for i := range dst {
		if i == nyquist {
			if absComplex64(dst[i]-complex(float32(n), 0)) > 1e-2 {
				t.Fatalf("bin[%d] = %v, want %v", i, dst[i], complex(float32(n), 0))
			}

			continue
		}

		if absComplex64(dst[i]) > 1e-2 {
			t.Fatalf("bin[%d] = %v, want near 0", i, dst[i])
		}
	}
}

func TestFFTEdgeCases(t *testing.T) {
	t.Parallel()

	for _, n := range []int{1, 2} {
		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) returned error: %v", n, err)
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i+1), float32(-i))
		}

		dst := make([]complex64, n)
		err = plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward(%d) returned error: %v", n, err)
		}

		out := make([]complex64, n)
		err = plan.Inverse(out, dst)
		if err != nil {
			t.Fatalf("Inverse(%d) returned error: %v", n, err)
		}

		for i := range src {
			if absComplex64(out[i]-src[i]) > 1e-3 {
				t.Fatalf("n=%d out[%d] = %v, want %v", n, i, out[i], src[i])
			}
		}
	}
}

func complexSliceEqual(a, b []complex64, tol float32) bool {
	if len(a) != len(b) {
		return false
	}

	limit := float64(tol)
	for i := range a {
		if absComplex64(a[i]-b[i]) > limit {
			return false
		}
	}

	return true
}

func absComplex64(v complex64) float64 {
	return math.Hypot(float64(real(v)), float64(imag(v)))
}
