package algofft

import (
	"math"
	"math/rand"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// These tests cover the double-precision real 2D/3D plans introduced by the
// generic PlanReal2D[F, C]/PlanReal3D[F, C] (PLAN.md A1): before the
// generics unification, float64 real multi-dimensional transforms did not
// exist at all.

func TestPlanReal2D_Float64_ForwardMatchesReference(t *testing.T) {
	t.Parallel()

	const rows, cols = 8, 16

	plan, err := NewPlanReal2D[float64, complex128](rows, cols)
	if err != nil {
		t.Fatalf("NewPlanReal2D[float64, complex128] failed: %v", err)
	}

	rng := rand.New(rand.NewSource(7))

	src := make([]float64, rows*cols)
	for i := range src {
		src[i] = rng.Float64()*2 - 1
	}

	halfCols := cols/2 + 1
	dst := make([]complex128, rows*halfCols)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Reference: full complex 2D DFT of the widened input.
	widened := make([]complex128, rows*cols)
	for i, v := range src {
		widened[i] = complex(v, 0)
	}

	want := reference.NaiveDFT2D128(widened, rows, cols)

	const tol = 1e-10

	for r := range rows {
		for c := range halfCols {
			got := dst[r*halfCols+c]

			ref := want[r*cols+c]
			if math.Abs(real(got-ref)) > tol || math.Abs(imag(got-ref)) > tol {
				t.Errorf("[%d,%d] = %v, want %v", r, c, got, ref)
			}
		}
	}
}

func TestPlanReal2D_Float64_RoundTrip(t *testing.T) {
	t.Parallel()

	const rows, cols = 6, 10

	plan, err := NewPlanReal2D[float64, complex128](rows, cols)
	if err != nil {
		t.Fatalf("NewPlanReal2D[float64, complex128] failed: %v", err)
	}

	rng := rand.New(rand.NewSource(11))

	src := make([]float64, rows*cols)
	for i := range src {
		src[i] = rng.Float64()*2 - 1
	}

	spectrum := make([]complex128, plan.SpectrumLen())
	back := make([]float64, rows*cols)

	err = plan.Forward(spectrum, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	err = plan.Inverse(back, spectrum)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	for i := range src {
		if math.Abs(back[i]-src[i]) > 1e-10 {
			t.Errorf("round-trip mismatch at %d: got %v, want %v", i, back[i], src[i])
		}
	}
}

func TestPlanReal3D_Float64_ForwardMatchesReference(t *testing.T) {
	t.Parallel()

	const depth, height, width = 4, 6, 8

	plan, err := NewPlanReal3D[float64, complex128](depth, height, width)
	if err != nil {
		t.Fatalf("NewPlanReal3D[float64, complex128] failed: %v", err)
	}

	rng := rand.New(rand.NewSource(13))

	src := make([]float64, depth*height*width)
	for i := range src {
		src[i] = rng.Float64()*2 - 1
	}

	halfWidth := width/2 + 1
	dst := make([]complex128, depth*height*halfWidth)

	err = plan.Forward(dst, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	widened := make([]complex128, len(src))
	for i, v := range src {
		widened[i] = complex(v, 0)
	}

	want := reference.NaiveDFT3D128(widened, depth, height, width)

	const tol = 1e-10

	for d := range depth {
		for h := range height {
			for w := range halfWidth {
				got := dst[d*height*halfWidth+h*halfWidth+w]

				ref := want[d*height*width+h*width+w]
				if math.Abs(real(got-ref)) > tol || math.Abs(imag(got-ref)) > tol {
					t.Errorf("[%d,%d,%d] = %v, want %v", d, h, w, got, ref)
				}
			}
		}
	}
}

func TestPlanReal3D_Float64_RoundTrip(t *testing.T) {
	t.Parallel()

	const depth, height, width = 3, 5, 8

	plan, err := NewPlanReal3D[float64, complex128](depth, height, width)
	if err != nil {
		t.Fatalf("NewPlanReal3D[float64, complex128] failed: %v", err)
	}

	rng := rand.New(rand.NewSource(17))

	src := make([]float64, depth*height*width)
	for i := range src {
		src[i] = rng.Float64()*2 - 1
	}

	spectrum := make([]complex128, plan.SpectrumLen())
	back := make([]float64, len(src))

	err = plan.Forward(spectrum, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	err = plan.Inverse(back, spectrum)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	for i := range src {
		if math.Abs(back[i]-src[i]) > 1e-10 {
			t.Errorf("round-trip mismatch at %d: got %v, want %v", i, back[i], src[i])
		}
	}
}
