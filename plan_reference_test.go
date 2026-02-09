package algofft

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestForwardMatchesReferenceSmall(t *testing.T) {
	t.Parallel()

	sizes := []int{2, 4, 8, 16, 32, 64, 128, 256}
	for _, n := range sizes {
		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) returned error: %v", n, err)
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i+1), float32(-i)*0.5)
		}

		got := make([]complex64, n)

		err = plan.Forward(got, src)
		if err != nil {
			t.Fatalf("Forward(%d) returned error: %v", n, err)
		}

		want := reference.NaiveDFT(src)
		for i := range got {
			assertApproxComplex64Tolf(t, got[i], want[i], 5e-3, "n=%d idx=%d", n, i)
		}
	}
}

func TestInverseMatchesReferenceSmall(t *testing.T) {
	t.Parallel()

	sizes := []int{2, 4, 8, 16}
	for _, n := range sizes {
		plan, err := NewPlanT[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d) returned error: %v", n, err)
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i+1), float32(-i)*0.5)
		}

		freq := reference.NaiveDFT(src)

		got := make([]complex64, n)

		err = plan.Inverse(got, freq)
		if err != nil {
			t.Fatalf("Inverse(%d) returned error: %v", n, err)
		}

		want := reference.NaiveIDFT(freq)
		for i := range got {
			assertApproxComplex64Tolf(t, got[i], want[i], 1e-4, "n=%d idx=%d", n, i)
		}
	}
}

func TestForwardMatchesReferenceSmall128(t *testing.T) {
	t.Parallel()

	sizes := []int{2, 4, 8, 16}
	for _, n := range sizes {
		plan, err := NewPlan64(n)
		if err != nil {
			t.Fatalf("NewPlan64(%d) returned error: %v", n, err)
		}

		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(float64(i+1), float64(-i)*0.5)
		}

		got := make([]complex128, n)

		err = plan.Forward(got, src)
		if err != nil {
			t.Fatalf("Forward(%d) returned error: %v", n, err)
		}

		want := reference.NaiveDFT128(src)
		for i := range got {
			assertApproxComplex128Tolf(t, got[i], want[i], 1e-10, "n=%d idx=%d", n, i)
		}
	}
}

func TestInverseMatchesReferenceSmall128(t *testing.T) {
	t.Parallel()

	sizes := []int{2, 4, 8, 16}
	for _, n := range sizes {
		plan, err := NewPlan64(n)
		if err != nil {
			t.Fatalf("NewPlan64(%d) returned error: %v", n, err)
		}

		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(float64(i+1), float64(-i)*0.5)
		}

		freq := reference.NaiveDFT128(src)

		got := make([]complex128, n)

		err = plan.Inverse(got, freq)
		if err != nil {
			t.Fatalf("Inverse(%d) returned error: %v", n, err)
		}

		want := reference.NaiveIDFT128(freq)
		for i := range got {
			assertApproxComplex128Tolf(t, got[i], want[i], 1e-10, "n=%d idx=%d", n, i)
		}
	}
}

func assertApproxComplex64Tolf(t *testing.T, got, want complex64, tol float64, format string, args ...any) {
	t.Helper()

	diff := complex128(got - want)
	if real(diff)*real(diff)+imag(diff)*imag(diff) > tol*tol {
		t.Fatalf(format+": got %v want %v", append(args, got, want)...)
	}
}
