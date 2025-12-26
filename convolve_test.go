package algoforge

import (
	"math/rand"
	"testing"
)

func TestConvolveBasic(t *testing.T) {
	a := []complex64{1 + 0i, 2 + 0i, 3 + 0i}
	b := []complex64{4 + 0i, 5 + 0i}
	want := []complex64{4 + 0i, 13 + 0i, 22 + 0i, 15 + 0i}

	got := make([]complex64, len(a)+len(b)-1)
	if err := Convolve(got, a, b); err != nil {
		t.Fatalf("Convolve() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex64(t, got[i], want[i], 1e-4, "got[%d]", i)
	}
}

func TestConvolveRandomMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	a := make([]complex64, 7)
	b := make([]complex64, 5)

	for i := range a {
		a[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}
	for i := range b {
		b[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	want := naiveConvolveComplex64(a, b)
	got := make([]complex64, len(want))

	if err := Convolve(got, a, b); err != nil {
		t.Fatalf("Convolve() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex64(t, got[i], want[i], 1e-3, "got[%d]", i)
	}
}

func TestConvolveErrors(t *testing.T) {
	if err := Convolve(nil, []complex64{1}, []complex64{1}); err != ErrNilSlice {
		t.Fatalf("Convolve(nil, a, b) = %v, want ErrNilSlice", err)
	}
	if err := Convolve([]complex64{1}, nil, []complex64{1}); err != ErrNilSlice {
		t.Fatalf("Convolve(dst, nil, b) = %v, want ErrNilSlice", err)
	}
	if err := Convolve([]complex64{1}, []complex64{1}, nil); err != ErrNilSlice {
		t.Fatalf("Convolve(dst, a, nil) = %v, want ErrNilSlice", err)
	}
	if err := Convolve([]complex64{}, []complex64{}, []complex64{1}); err != ErrInvalidLength {
		t.Fatalf("Convolve(dst, empty, b) = %v, want ErrInvalidLength", err)
	}
	if err := Convolve([]complex64{}, []complex64{1}, []complex64{}); err != ErrInvalidLength {
		t.Fatalf("Convolve(dst, a, empty) = %v, want ErrInvalidLength", err)
	}
	if err := Convolve([]complex64{0}, []complex64{1, 2}, []complex64{3, 4}); err != ErrLengthMismatch {
		t.Fatalf("Convolve(dst, a, b) = %v, want ErrLengthMismatch", err)
	}
}

func naiveConvolveComplex64(a, b []complex64) []complex64 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}

	out := make([]complex64, len(a)+len(b)-1)
	for i := range a {
		for j := range b {
			out[i+j] += a[i] * b[j]
		}
	}

	return out
}
