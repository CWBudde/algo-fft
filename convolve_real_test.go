package algoforge

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func TestConvolveRealBasic(t *testing.T) {
	t.Parallel()

	a := []float32{1, 2, 3}
	b := []float32{4, 5}
	want := []float32{4, 13, 22, 15}

	got := make([]float32, len(a)+len(b)-1)

	err := ConvolveReal(got, a, b)
	if err != nil {
		t.Fatalf("ConvolveReal() returned error: %v", err)
	}

	for i := range want {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > 1e-3 {
			t.Fatalf("got[%d]=%v want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

func TestConvolveRealGaussianKernel(t *testing.T) {
	t.Parallel()

	signal := make([]float32, 32)
	for i := range signal {
		if i >= 8 && i < 24 {
			signal[i] = 1
		}
	}

	kernel := gaussianKernel1D(2, 1.0)
	want := naiveConvolveReal(signal, kernel)

	got := make([]float32, len(want))

	err := ConvolveReal(got, signal, kernel)
	if err != nil {
		t.Fatalf("ConvolveReal() returned error: %v", err)
	}

	for i := range want {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > 1e-3 {
			t.Fatalf("got[%d]=%v want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

func TestConvolveRealRandomMatchesNaive(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(2))
	a := make([]float32, 9)
	b := make([]float32, 6)

	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}

	for i := range b {
		b[i] = rng.Float32()*2 - 1
	}

	want := naiveConvolveReal(a, b)
	got := make([]float32, len(want))

	err := ConvolveReal(got, a, b)
	if err != nil {
		t.Fatalf("ConvolveReal() returned error: %v", err)
	}

	for i := range want {
		if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > 1e-3 {
			t.Fatalf("got[%d]=%v want %v (diff=%v)", i, got[i], want[i], diff)
		}
	}
}

func TestConvolveRealErrors(t *testing.T) {
	t.Parallel()

	err := ConvolveReal(nil, []float32{1}, []float32{1})
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("ConvolveReal(nil, a, b) = %v, want ErrNilSlice", err)
	}

	err = ConvolveReal([]float32{1}, nil, []float32{1})
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("ConvolveReal(dst, nil, b) = %v, want ErrNilSlice", err)
	}

	err = ConvolveReal([]float32{1}, []float32{1}, nil)
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("ConvolveReal(dst, a, nil) = %v, want ErrNilSlice", err)
	}

	err = ConvolveReal([]float32{}, []float32{}, []float32{1})
	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("ConvolveReal(dst, empty, b) = %v, want ErrInvalidLength", err)
	}

	err = ConvolveReal([]float32{}, []float32{1}, []float32{})
	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("ConvolveReal(dst, a, empty) = %v, want ErrInvalidLength", err)
	}

	err = ConvolveReal([]float32{0}, []float32{1, 2}, []float32{3, 4})
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("ConvolveReal(dst, a, b) = %v, want ErrLengthMismatch", err)
	}
}

func gaussianKernel1D(radius int, sigma float64) []float32 {
	size := radius*2 + 1
	kernel := make([]float32, size)

	var sum float64
	for i := -radius; i <= radius; i++ {
		x := float64(i)
		v := math.Exp(-(x * x) / (2 * sigma * sigma))
		kernel[i+radius] = float32(v)
		sum += v
	}

	if sum == 0 {
		return kernel
	}

	inv := float32(1.0 / sum)
	for i := range kernel {
		kernel[i] *= inv
	}

	return kernel
}

func naiveConvolveReal(a, b []float32) []float32 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}

	out := make([]float32, len(a)+len(b)-1)
	for i := range a {
		for j := range b {
			out[i+j] += a[i] * b[j]
		}
	}

	return out
}
