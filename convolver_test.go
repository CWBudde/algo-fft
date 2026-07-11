package algofft

import (
	"errors"
	"math/cmplx"
	"testing"
)

func TestConvolver_MatchesConvolve(t *testing.T) {
	t.Parallel()

	const lenA, lenB = 37, 12

	a := make([]complex64, lenA)
	b := make([]complex64, lenB)

	for i := range a {
		a[i] = complex(float32(i%5)-2, float32(i%3)-1)
	}

	for i := range b {
		b[i] = complex(float32(i%4)-1, float32(i%7)-3)
	}

	want := make([]complex64, lenA+lenB-1)
	if err := Convolve(want, a, b); err != nil {
		t.Fatalf("Convolve failed: %v", err)
	}

	conv, err := NewConvolver[complex64](lenA, lenB)
	if err != nil {
		t.Fatalf("NewConvolver failed: %v", err)
	}

	if conv.Len() != lenA+lenB-1 {
		t.Fatalf("Len() = %d, want %d", conv.Len(), lenA+lenB-1)
	}

	got := make([]complex64, conv.Len())

	// Run twice to exercise buffer reuse.
	for run := range 2 {
		if err := conv.Convolve(got, a, b); err != nil {
			t.Fatalf("run %d: Convolver.Convolve failed: %v", run, err)
		}

		for i := range want {
			if cmplx.Abs(complex128(got[i]-want[i])) > 1e-4 {
				t.Fatalf("run %d: mismatch at %d: got %v, want %v", run, i, got[i], want[i])
			}
		}
	}
}

func TestConvolver_MatchesConvolve128(t *testing.T) {
	t.Parallel()

	const lenA, lenB = 16, 16

	a := make([]complex128, lenA)
	b := make([]complex128, lenB)

	for i := range a {
		a[i] = complex(float64(i%5)-2, float64(i%3)-1)
		b[i] = complex(float64(i%4)-1, float64(i%7)-3)
	}

	want := make([]complex128, lenA+lenB-1)
	if err := Convolve128(want, a, b); err != nil {
		t.Fatalf("Convolve128 failed: %v", err)
	}

	conv, err := NewConvolver[complex128](lenA, lenB)
	if err != nil {
		t.Fatalf("NewConvolver failed: %v", err)
	}

	got := make([]complex128, conv.Len())
	if err := conv.Convolve(got, a, b); err != nil {
		t.Fatalf("Convolver.Convolve failed: %v", err)
	}

	for i := range want {
		if cmplx.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("mismatch at %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestConvolver_Errors(t *testing.T) {
	t.Parallel()

	if _, err := NewConvolver[complex64](0, 4); !errors.Is(err, ErrInvalidLength) {
		t.Errorf("NewConvolver(0, 4): got %v, want ErrInvalidLength", err)
	}

	conv, err := NewConvolver[complex64](8, 4)
	if err != nil {
		t.Fatalf("NewConvolver failed: %v", err)
	}

	dst := make([]complex64, conv.Len())
	a := make([]complex64, 8)
	b := make([]complex64, 4)

	if err := conv.Convolve(nil, a, b); !errors.Is(err, ErrNilSlice) {
		t.Errorf("nil dst: got %v, want ErrNilSlice", err)
	}

	if err := conv.Convolve(dst, a[:7], b); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("short a: got %v, want ErrLengthMismatch", err)
	}

	if err := conv.Convolve(dst[:3], a, b); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("short dst: got %v, want ErrLengthMismatch", err)
	}
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestConvolver_ZeroAllocSteadyState(t *testing.T) {
	const lenA, lenB = 24, 9 // convLen 32, power of two

	conv, err := NewConvolver[complex64](lenA, lenB)
	if err != nil {
		t.Fatalf("NewConvolver failed: %v", err)
	}

	a := make([]complex64, lenA)
	b := make([]complex64, lenB)
	dst := make([]complex64, conv.Len())

	a[0], b[0] = 1, 1

	// Warm up the scratch cache.
	if err := conv.Convolve(dst, a, b); err != nil {
		t.Fatalf("warm-up Convolve failed: %v", err)
	}

	allocs := testing.AllocsPerRun(10, func() {
		if err := conv.Convolve(dst, a, b); err != nil {
			t.Errorf("Convolve failed: %v", err)
		}
	})

	if allocs != 0 {
		t.Errorf("Convolver.Convolve allocates %.1f per call, want 0", allocs)
	}
}

func TestCorrelator_MatchesCrossCorrelate(t *testing.T) {
	t.Parallel()

	const lenA, lenB = 21, 13

	a := make([]complex64, lenA)
	b := make([]complex64, lenB)

	for i := range a {
		a[i] = complex(float32(i%6)-3, float32(i%2))
	}

	for i := range b {
		b[i] = complex(float32(i%3)-1, float32(i%5)-2)
	}

	want := make([]complex64, lenA+lenB-1)
	if err := CrossCorrelate(want, a, b); err != nil {
		t.Fatalf("CrossCorrelate failed: %v", err)
	}

	corr, err := NewCorrelator[complex64](lenA, lenB)
	if err != nil {
		t.Fatalf("NewCorrelator failed: %v", err)
	}

	if corr.Len() != lenA+lenB-1 {
		t.Fatalf("Len() = %d, want %d", corr.Len(), lenA+lenB-1)
	}

	got := make([]complex64, corr.Len())

	for run := range 2 {
		if err := corr.CrossCorrelate(got, a, b); err != nil {
			t.Fatalf("run %d: Correlator.CrossCorrelate failed: %v", run, err)
		}

		for i := range want {
			if cmplx.Abs(complex128(got[i]-want[i])) > 1e-4 {
				t.Fatalf("run %d: mismatch at %d: got %v, want %v", run, i, got[i], want[i])
			}
		}
	}
}

func TestCorrelator_Errors(t *testing.T) {
	t.Parallel()

	corr, err := NewCorrelator[complex64](8, 4)
	if err != nil {
		t.Fatalf("NewCorrelator failed: %v", err)
	}

	dst := make([]complex64, corr.Len())
	a := make([]complex64, 8)
	b := make([]complex64, 4)

	if err := corr.CrossCorrelate(dst, nil, b); !errors.Is(err, ErrNilSlice) {
		t.Errorf("nil a: got %v, want ErrNilSlice", err)
	}

	if err := corr.CrossCorrelate(dst, a, b[:3]); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("short b: got %v, want ErrLengthMismatch", err)
	}

	if err := corr.CrossCorrelate(dst, a[:7], b); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("short a: got %v, want ErrLengthMismatch", err)
	}

	if err := corr.CrossCorrelate(dst[:5], a, b); !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("short dst: got %v, want ErrLengthMismatch", err)
	}
}

func TestRealConvolver_MatchesConvolveReal(t *testing.T) {
	t.Parallel()

	const lenA, lenB = 40, 17

	a := make([]float32, lenA)
	b := make([]float32, lenB)

	for i := range a {
		a[i] = float32(i%9) - 4
	}

	for i := range b {
		b[i] = float32(i%5) - 2
	}

	want := make([]float32, lenA+lenB-1)
	if err := ConvolveReal(want, a, b); err != nil {
		t.Fatalf("ConvolveReal failed: %v", err)
	}

	conv, err := NewRealConvolver[float32, complex64](lenA, lenB)
	if err != nil {
		t.Fatalf("NewRealConvolver failed: %v", err)
	}

	got := make([]float32, conv.Len())

	for run := range 2 {
		if err := conv.Convolve(got, a, b); err != nil {
			t.Fatalf("run %d: RealConvolver.Convolve failed: %v", run, err)
		}

		for i := range want {
			diff := got[i] - want[i]
			if diff < 0 {
				diff = -diff
			}

			if diff > 1e-3 {
				t.Fatalf("run %d: mismatch at %d: got %v, want %v", run, i, got[i], want[i])
			}
		}
	}
}

func TestRealConvolver_Float64(t *testing.T) {
	t.Parallel()

	const lenA, lenB = 15, 15

	a := make([]float64, lenA)
	b := make([]float64, lenB)

	for i := range a {
		a[i] = float64(i%7) - 3
		b[i] = float64(i%4) - 1
	}

	want := make([]float64, lenA+lenB-1)
	if err := ConvolveReal64(want, a, b); err != nil {
		t.Fatalf("ConvolveReal64 failed: %v", err)
	}

	conv, err := NewRealConvolver[float64, complex128](lenA, lenB)
	if err != nil {
		t.Fatalf("NewRealConvolver failed: %v", err)
	}

	got := make([]float64, conv.Len())
	if err := conv.Convolve(got, a, b); err != nil {
		t.Fatalf("RealConvolver.Convolve failed: %v", err)
	}

	for i := range want {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}

		if diff > 1e-9 {
			t.Fatalf("mismatch at %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestRealConvolver_ZeroAllocSteadyState(t *testing.T) {
	const lenA, lenB = 24, 9

	conv, err := NewRealConvolver[float32, complex64](lenA, lenB)
	if err != nil {
		t.Fatalf("NewRealConvolver failed: %v", err)
	}

	a := make([]float32, lenA)
	b := make([]float32, lenB)
	dst := make([]float32, conv.Len())

	a[0], b[0] = 1, 1

	if err := conv.Convolve(dst, a, b); err != nil {
		t.Fatalf("warm-up Convolve failed: %v", err)
	}

	allocs := testing.AllocsPerRun(10, func() {
		if err := conv.Convolve(dst, a, b); err != nil {
			t.Errorf("Convolve failed: %v", err)
		}
	})

	if allocs != 0 {
		t.Errorf("RealConvolver.Convolve allocates %.1f per call, want 0", allocs)
	}
}
