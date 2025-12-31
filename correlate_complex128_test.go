package algofft

import (
	"errors"
	"math"
	"math/rand"
	"testing"
)

func TestCrossCorrelate128MatchesNaive(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(3))
	a := make([]complex128, 6)
	b := make([]complex128, 4)

	for i := range a {
		a[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	for i := range b {
		b[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	want := naiveCrossCorrelate128(a, b)
	got := make([]complex128, len(want))

	err := CrossCorrelate128(got, a, b)
	if err != nil {
		t.Fatalf("CrossCorrelate128() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex128Tolf(t, got[i], want[i], 1e-11, "got[%d]", i)
	}
}

func TestAutoCorrelate128ZeroLagEnergy(t *testing.T) {
	t.Parallel()

	a := []complex128{1 + 2i, -3 + 0.5i, 2 - 1i}
	dst := make([]complex128, 2*len(a)-1)

	err := AutoCorrelate128(dst, a)
	if err != nil {
		t.Fatalf("AutoCorrelate128() returned error: %v", err)
	}

	var energy float64
	for _, v := range a {
		vr := real(v)
		vi := imag(v)
		energy += vr*vr + vi*vi
	}

	zeroLag := dst[len(a)-1]
	diff := math.Abs(real(zeroLag) - energy)
	if diff > 1e-11 || math.Abs(imag(zeroLag)) > 1e-11 {
		t.Fatalf("zero lag=%v want %v (diff=%v)", zeroLag, energy, diff)
	}
}

func TestCrossCorrelate128Errors(t *testing.T) {
	t.Parallel()

	err := CrossCorrelate128(nil, []complex128{1}, []complex128{1})
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("CrossCorrelate128(nil, a, b) = %v, want ErrNilSlice", err)
	}

	err = CrossCorrelate128([]complex128{1}, nil, []complex128{1})
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("CrossCorrelate128(dst, nil, b) = %v, want ErrNilSlice", err)
	}

	err = CrossCorrelate128([]complex128{1}, []complex128{1}, nil)
	if !errors.Is(err, ErrNilSlice) {
		t.Fatalf("CrossCorrelate128(dst, a, nil) = %v, want ErrNilSlice", err)
	}

	err = CrossCorrelate128([]complex128{}, []complex128{}, []complex128{1})
	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("CrossCorrelate128(dst, empty, b) = %v, want ErrInvalidLength", err)
	}

	err = CrossCorrelate128([]complex128{}, []complex128{1}, []complex128{})
	if !errors.Is(err, ErrInvalidLength) {
		t.Fatalf("CrossCorrelate128(dst, a, empty) = %v, want ErrInvalidLength", err)
	}

	err = CrossCorrelate128([]complex128{0}, []complex128{1, 2}, []complex128{3, 4})
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("CrossCorrelate128(dst, a, b) = %v, want ErrLengthMismatch", err)
	}
}

func TestCorrelate128Alias(t *testing.T) {
	t.Parallel()

	a := []complex128{1 + 0i, 2 + 1i}
	b := []complex128{3 - 1i}

	want := naiveCrossCorrelate128(a, b)
	got := make([]complex128, len(want))

	err := Correlate128(got, a, b)
	if err != nil {
		t.Fatalf("Correlate128() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex128Tolf(t, got[i], want[i], 1e-12, "got[%d]", i)
	}
}

func naiveCrossCorrelate128(a, b []complex128) []complex128 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}

	out := make([]complex128, len(a)+len(b)-1)
	for lag := -(len(b) - 1); lag <= len(a)-1; lag++ {
		var sum complex128

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
