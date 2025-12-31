package algofft

import (
	"errors"
	"math/rand"
	"testing"
)

func TestConvolve128Basic(t *testing.T) {
	t.Parallel()

	a := []complex128{1 + 0i, 2 + 0i, 3 + 0i}
	b := []complex128{4 + 0i, 5 + 0i}
	want := []complex128{4 + 0i, 13 + 0i, 22 + 0i, 15 + 0i}

	got := make([]complex128, len(a)+len(b)-1)

	err := Convolve128(got, a, b)
	if err != nil {
		t.Fatalf("Convolve128() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex128Tolf(t, got[i], want[i], 1e-12, "got[%d]", i)
	}
}

func TestConvolve128RandomMatchesNaive(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(1))
	a := make([]complex128, 7)
	b := make([]complex128, 5)

	for i := range a {
		a[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	for i := range b {
		b[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	want := naiveConvolveComplex128(a, b)
	got := make([]complex128, len(want))

	err := Convolve128(got, a, b)
	if err != nil {
		t.Fatalf("Convolve128() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex128Tolf(t, got[i], want[i], 1e-11, "got[%d]", i)
	}
}

func TestConvolve128Errors(t *testing.T) {
	t.Parallel()

	err := Convolve128(nil, []complex128{1}, []complex128{1})
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("Convolve128(nil, a, b) = %v, want ErrNilSlice", err)
	}

	err = Convolve128([]complex128{1}, nil, []complex128{1})
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("Convolve128(dst, nil, b) = %v, want ErrNilSlice", err)
	}

	err = Convolve128([]complex128{1}, []complex128{1}, nil)
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("Convolve128(dst, a, nil) = %v, want ErrNilSlice", err)
	}

	err = Convolve128([]complex128{}, []complex128{}, []complex128{1})
	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("Convolve128(dst, empty, b) = %v, want ErrInvalidLength", err)
	}

	err = Convolve128([]complex128{}, []complex128{1}, []complex128{})
	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("Convolve128(dst, a, empty) = %v, want ErrInvalidLength", err)
	}

	err = Convolve128([]complex128{0}, []complex128{1, 2}, []complex128{3, 4})
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("Convolve128(dst, a, b) = %v, want ErrLengthMismatch", err)
	}
}

func naiveConvolveComplex128(a, b []complex128) []complex128 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}

	out := make([]complex128, len(a)+len(b)-1)
	for i := range a {
		for j := range b {
			out[i+j] += a[i] * b[j]
		}
	}

	return out
}
