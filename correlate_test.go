package algofft

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func TestCrossCorrelateMatchesNaive(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(3))
	a := make([]complex64, 6)
	b := make([]complex64, 4)

	for i := range a {
		a[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	for i := range b {
		b[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	want := naiveCrossCorrelate(a, b)
	got := make([]complex64, len(want))

	err := CrossCorrelate(got, a, b)
	if err != nil {
		t.Fatalf("CrossCorrelate() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex64f(t, got[i], want[i], 1e-3, "got[%d]", i)
	}
}

func TestAutoCorrelateZeroLagEnergy(t *testing.T) {
	t.Parallel()

	a := []complex64{1 + 2i, -3 + 0.5i, 2 - 1i}
	dst := make([]complex64, 2*len(a)-1)

	err := AutoCorrelate(dst, a)
	if err != nil {
		t.Fatalf("AutoCorrelate() returned error: %v", err)
	}

	var energy float64
	for _, v := range a {
		vr := float64(real(v))
		vi := float64(imag(v))
		energy += vr*vr + vi*vi
	}

	zeroLag := dst[len(a)-1]

	diff := math.Abs(float64(real(zeroLag)) - energy)
	if diff > 1e-3 || math.Abs(float64(imag(zeroLag))) > 1e-3 {
		t.Fatalf("zero lag=%v want %v (diff=%v)", zeroLag, energy, diff)
	}
}

func TestCrossCorrelateErrors(t *testing.T) {
	t.Parallel()

	err := CrossCorrelate(nil, []complex64{1}, []complex64{1})
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("CrossCorrelate(nil, a, b) = %v, want ErrNilSlice", err)
	}

	err = CrossCorrelate([]complex64{1}, nil, []complex64{1})

	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("CrossCorrelate(dst, nil, b) = %v, want ErrNilSlice", err)
	}

	err = CrossCorrelate([]complex64{1}, []complex64{1}, nil)

	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("CrossCorrelate(dst, a, nil) = %v, want ErrNilSlice", err)
	}

	err = CrossCorrelate([]complex64{}, []complex64{}, []complex64{1})

	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("CrossCorrelate(dst, empty, b) = %v, want ErrInvalidLength", err)
	}

	err = CrossCorrelate([]complex64{}, []complex64{1}, []complex64{})

	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("CrossCorrelate(dst, a, empty) = %v, want ErrInvalidLength", err)
	}

	err = CrossCorrelate([]complex64{0}, []complex64{1, 2}, []complex64{3, 4})

	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("CrossCorrelate(dst, a, b) = %v, want ErrLengthMismatch", err)
	}
}

func TestCorrelateAlias(t *testing.T) {
	t.Parallel()

	a := []complex64{1 + 0i, 2 + 1i}
	b := []complex64{3 - 1i}

	want := naiveCrossCorrelate(a, b)
	got := make([]complex64, len(want))

	err := Correlate(got, a, b)
	if err != nil {
		t.Fatalf("Correlate() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex64f(t, got[i], want[i], 1e-4, "got[%d]", i)
	}
}

func naiveCrossCorrelate(a, b []complex64) []complex64 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}

	out := make([]complex64, len(a)+len(b)-1)
	for lag := -(len(b) - 1); lag <= len(a)-1; lag++ {
		var sum complex64

		for n := range a {
			m := n - lag
			if m < 0 || m >= len(b) {
				continue
			}

			bv := b[m]
			sum += a[n] * complex(real(bv), -imag(bv))
		}

		out[lag+len(b)-1] = sum
	}

	return out
}
