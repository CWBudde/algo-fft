package algofft

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func TestConvolveReal64Basic(t *testing.T) {
	t.Parallel()

	a := []float64{1, 2, 3}
	b := []float64{4, 5}
	want := []float64{4, 13, 22, 15}

	got := make([]float64, len(a)+len(b)-1)

	err := ConvolveReal64(got, a, b)
	if err != nil {
		t.Fatalf("ConvolveReal64() returned error: %v", err)
	}

	for i := range want {
		if diff := math.Abs(got[i] - want[i]); diff > 1e-9 {
			t.Fatalf("got[%d]=%v want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

func TestConvolveReal64RandomMatchesNaive(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(7))
	a := make([]float64, 9)
	b := make([]float64, 6)

	for i := range a {
		a[i] = rng.Float64()*2 - 1
	}

	for i := range b {
		b[i] = rng.Float64()*2 - 1
	}

	want := naiveConvolveReal64(a, b)
	got := make([]float64, len(want))

	err := ConvolveReal64(got, a, b)
	if err != nil {
		t.Fatalf("ConvolveReal64() returned error: %v", err)
	}

	for i := range want {
		if diff := math.Abs(got[i] - want[i]); diff > 1e-9 {
			t.Fatalf("got[%d]=%v want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

func TestConvolveReal64Errors(t *testing.T) {
	t.Parallel()

	err := ConvolveReal64(nil, []float64{1}, []float64{1})
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("ConvolveReal64(nil, a, b) = %v, want ErrNilSlice", err)
	}

	err = ConvolveReal64([]float64{}, []float64{}, []float64{1})
	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("ConvolveReal64(dst, empty, b) = %v, want ErrInvalidLength", err)
	}

	err = ConvolveReal64([]float64{0}, []float64{1, 2}, []float64{3, 4})
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("ConvolveReal64(dst, a, b) = %v, want ErrLengthMismatch", err)
	}
}

func naiveConvolveReal64(a, b []float64) []float64 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}

	out := make([]float64, len(a)+len(b)-1)
	for i := range a {
		for j := range b {
			out[i+j] += a[i] * b[j]
		}
	}

	return out
}
