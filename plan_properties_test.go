package algofft

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// planPropertySizes exercises every dispatch family of the public 1D plan:
// power-of-two (DIT/Stockham/codelets: 8..1024), mixed radix 2/3/5
// (12, 60, 96, 384, 1000), and Bluestein (primes 17, 31).
var planPropertySizes = []int{8, 12, 16, 17, 31, 60, 64, 96, 128, 256, 384, 1000, 1024}

// Property tolerances are relative to the largest output magnitude.
const (
	planPropTol64  = 5e-3
	planPropTol128 = 1e-9
)

func randomPlanPropInput[T Complex](n int, seed int64) []T {
	rng := rand.New(rand.NewSource(seed))

	out := make([]T, n)
	for i := range out {
		out[i] = T(complex(rng.Float64()*2-1, rng.Float64()*2-1))
	}

	return out
}

func assertPlanPropClose[T Complex](t *testing.T, got, want []T, tol float64) {
	t.Helper()

	var maxMag float64

	for _, w := range want {
		if m := cmplx.Abs(complex128(w)); m > maxMag {
			maxMag = m
		}
	}

	if maxMag == 0 {
		maxMag = 1
	}

	for i := range want {
		if diff := cmplx.Abs(complex128(got[i] - want[i])); diff > tol*maxMag {
			t.Fatalf("index %d: got %v, want %v (diff %e > tol %e)",
				i, got[i], want[i], diff, tol*maxMag)
		}
	}
}

// TestPlanLinearity verifies FFT(a*x + b*y) = a*FFT(x) + b*FFT(y) through the
// public Plan API (the full dispatch path, unlike the raw-kernel variant in
// internal/fft).
func TestPlanLinearity(t *testing.T) {
	t.Parallel()

	for _, n := range planPropertySizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				t.Parallel()
				planPropLinearity[complex64](t, n, planPropTol64)
			})
			t.Run("complex128", func(t *testing.T) {
				t.Parallel()
				planPropLinearity[complex128](t, n, planPropTol128)
			})
		})
	}
}

func planPropLinearity[T Complex](t *testing.T, n int, tol float64) {
	t.Helper()

	plan, err := NewPlanT[T](n)
	if err != nil {
		t.Fatalf("NewPlanT(%d) returned error: %v", n, err)
	}

	x := randomPlanPropInput[T](n, 12345)
	y := randomPlanPropInput[T](n, 67890)
	a := T(complex(2.5, 1.3))
	b := T(complex(-1.7, 0.8))

	combined := make([]T, n)
	for i := range n {
		combined[i] = a*x[i] + b*y[i]
	}

	fftCombined := make([]T, n)
	if err := plan.Forward(fftCombined, combined); err != nil {
		t.Fatalf("Forward(combined) returned error: %v", err)
	}

	fftX := make([]T, n)
	if err := plan.Forward(fftX, x); err != nil {
		t.Fatalf("Forward(x) returned error: %v", err)
	}

	fftY := make([]T, n)
	if err := plan.Forward(fftY, y); err != nil {
		t.Fatalf("Forward(y) returned error: %v", err)
	}

	expected := make([]T, n)
	for i := range n {
		expected[i] = a*fftX[i] + b*fftY[i]
	}

	assertPlanPropClose(t, fftCombined, expected, tol)
}

// TestPlanShiftTheorem verifies the shift theorem through the public Plan API:
// if y[k] = x[(k-m) mod n] then FFT(y)[k] = FFT(x)[k] * exp(-2πikm/n).
func TestPlanShiftTheorem(t *testing.T) {
	t.Parallel()

	shifts := []int{1, 3}

	for _, n := range planPropertySizes {
		for _, m := range shifts {
			t.Run(fmt.Sprintf("n=%d/m=%d", n, m), func(t *testing.T) {
				t.Parallel()

				t.Run("complex64", func(t *testing.T) {
					t.Parallel()
					planPropShiftTheorem[complex64](t, n, m, planPropTol64)
				})
				t.Run("complex128", func(t *testing.T) {
					t.Parallel()
					planPropShiftTheorem[complex128](t, n, m, planPropTol128)
				})
			})
		}
	}
}

func planPropShiftTheorem[T Complex](t *testing.T, n, m int, tol float64) {
	t.Helper()

	plan, err := NewPlanT[T](n)
	if err != nil {
		t.Fatalf("NewPlanT(%d) returned error: %v", n, err)
	}

	x := randomPlanPropInput[T](n, 77777)

	y := make([]T, n)
	for k := range n {
		y[k] = x[(k-m+n)%n]
	}

	fftX := make([]T, n)
	if err := plan.Forward(fftX, x); err != nil {
		t.Fatalf("Forward(x) returned error: %v", err)
	}

	fftY := make([]T, n)
	if err := plan.Forward(fftY, y); err != nil {
		t.Fatalf("Forward(y) returned error: %v", err)
	}

	expected := make([]T, n)

	for k := range n {
		phase := -2 * math.Pi * float64(k*m) / float64(n)
		shift := T(complex(math.Cos(phase), math.Sin(phase)))
		expected[k] = fftX[k] * shift
	}

	assertPlanPropClose(t, fftY, expected, tol)
}

// TestPlanParseval verifies Parseval's theorem through the public Plan API for
// mixed-radix and Bluestein sizes as well as powers of two (the power-of-two
// sweep in precision_test.go covers larger sizes).
func TestPlanParseval(t *testing.T) {
	t.Parallel()

	for _, n := range planPropertySizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			t.Parallel()

			t.Run("complex64", func(t *testing.T) {
				t.Parallel()
				planPropParseval[complex64](t, n, planPropTol64)
			})
			t.Run("complex128", func(t *testing.T) {
				t.Parallel()
				planPropParseval[complex128](t, n, planPropTol128)
			})
		})
	}
}

func planPropParseval[T Complex](t *testing.T, n int, tol float64) {
	t.Helper()

	plan, err := NewPlanT[T](n)
	if err != nil {
		t.Fatalf("NewPlanT(%d) returned error: %v", n, err)
	}

	src := randomPlanPropInput[T](n, 11111)

	var timeEnergy float64

	for _, v := range src {
		c := complex128(v)
		timeEnergy += real(c)*real(c) + imag(c)*imag(c)
	}

	dst := make([]T, n)
	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward returned error: %v", err)
	}

	var freqEnergy float64

	for _, v := range dst {
		c := complex128(v)
		freqEnergy += real(c)*real(c) + imag(c)*imag(c)
	}

	freqEnergy /= float64(n)

	relError := math.Abs(timeEnergy-freqEnergy) / math.Max(timeEnergy, freqEnergy)
	if relError > tol {
		t.Errorf("Parseval's theorem violated: time=%v, freq=%v, relError=%e",
			timeEnergy, freqEnergy, relError)
	}
}
