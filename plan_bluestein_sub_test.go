package algofft

import (
	"math"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// Bluestein lengths whose pad is a power of two, i.e. the lengths that take the
// bound sub-FFT. 1009 pads to 2048 and 2003 to 4096 — both sizes with
// registered codelets that the unbound route could not reach, which is why
// these two measured *slower* on the default build than under -tags purego
// (PLAN.md P3). 9973 pads to 24576 and is the mixed-radix control: it must
// stay on the unbound (engine) route.
var bluesteinSubLengths = []struct {
	n     int
	pad   int
	bound bool
}{
	{n: 1009, pad: 2048, bound: true},
	{n: 2003, pad: 4096, bound: true},
	{n: 9973, pad: 24576, bound: false},
}

// TestBluesteinSubFFTIsBound pins that the plan-time binding actually happens.
// Without this the rest of the suite would pass identically whether the bound
// path runs or silently falls back — the failure mode PLAN.md 1.12 records as
// "declared-but-uncalled assembly is untested assembly".
func TestBluesteinSubFFTIsBound(t *testing.T) {
	t.Parallel()

	for _, tc := range bluesteinSubLengths {
		t.Run(strconv.Itoa(tc.n), func(t *testing.T) {
			t.Parallel()

			if got := bluesteinPadSize(tc.n); got != tc.pad {
				t.Fatalf("test premise broken: bluesteinPadSize(%d) = %d, want %d", tc.n, got, tc.pad)
			}

			assertBluesteinBound[complex64](t, tc.n, tc.bound)
			assertBluesteinBound[complex128](t, tc.n, tc.bound)
		})
	}
}

func assertBluesteinBound[T Complex](t *testing.T, n int, want bool) {
	t.Helper()

	plan, err := NewPlan[T](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) failed: %v", n, err)
	}

	exec, ok := plan.exec.(*bluesteinExecutor[T])
	if !ok {
		t.Fatalf("n=%d: executor is %T, want *bluesteinExecutor", n, plan.exec)
	}

	if got := exec.sub != nil; got != want {
		t.Fatalf("n=%d %T: sub-FFT bound = %v, want %v", n, *new(T), got, want)
	}
}

// TestBluesteinSubFFTMatchesReference compares the bound route bin-by-bin
// against the naive DFT. A round-trip test would not catch it: the sub-FFT runs
// forward and inverse, so a consistently wrong sub-FFT cancels itself out and
// the round trip still returns the input.
//
// The input is broadband, never an impulse: an impulse's spectrum is all-ones,
// so every twiddle multiplies a zero and every reordering of the output is
// still all-ones (PLAN.md 2.4).
func TestBluesteinSubFFTMatchesReference(t *testing.T) {
	t.Parallel()

	// 9973 is excluded: its O(n^2) reference costs ~1e8 complex operations.
	// Its route is pinned by TestBluesteinSubFFTIsBound and its correctness by
	// the round-trip sweep below.
	for _, n := range []int{1009, 2003} {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			src := broadbandComplex64(n)

			plan, err := NewPlan[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			dst := make([]complex64, n)
			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// NaiveDFTWide accumulates in float64, so the comparison is not
			// limited by the reference's own rounding (PLAN.md 1.6).
			ref := reference.NaiveDFTWide(src)

			var peak float64
			for _, v := range ref {
				peak = math.Max(peak, cabs128(v))
			}

			// Relative to the spectrum peak. The measured peak-normalized error
			// is ~1.2e-7 (about one float32 ulp) on both the bound AVX2 route
			// and the unbound pure-Go one, so this leaves ~80x headroom for
			// other codelet tiers while still being tight enough to fail on a
			// subtly wrong kernel. A wrong bin ordering or twiddle table is
			// O(1) and fails by orders of magnitude.
			const tol = 1e-5

			for i := range dst {
				if diff := cabs128(complex128(dst[i]) - ref[i]); diff > tol*peak {
					t.Fatalf("bin %d: got %v, want %v (diff %g, peak %g)", i, dst[i], ref[i], diff, peak)
				}
			}
		})
	}
}

// TestBluesteinSubFFTRoundTrip covers both precisions at every length,
// including the mixed-radix control, where the O(n^2) reference is too slow.
func TestBluesteinSubFFTRoundTrip(t *testing.T) {
	t.Parallel()

	for _, tc := range bluesteinSubLengths {
		t.Run(strconv.Itoa(tc.n), func(t *testing.T) {
			t.Parallel()

			bluesteinRoundTrip[complex64](t, tc.n, 2e-4)
			bluesteinRoundTrip[complex128](t, tc.n, 1e-11)
		})
	}
}

func bluesteinRoundTrip[T Complex](t *testing.T, n int, tol float64) {
	t.Helper()

	plan, err := NewPlan[T](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) failed: %v", n, err)
	}

	src := make([]T, n)
	for i, v := range broadbandComplex64(n) {
		src[i] = T(complex128(v))
	}

	freq := make([]T, n)
	if err := plan.Forward(freq, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	got := make([]T, n)
	if err := plan.Inverse(got, freq); err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	for i := range got {
		if diff := cabs128(complex128(got[i]) - complex128(src[i])); diff > tol {
			t.Fatalf("n=%d %T bin %d: round trip got %v, want %v (diff %g)",
				n, *new(T), i, got[i], src[i], diff)
		}
	}
}

// TestBluesteinSubFFTZeroAlloc pins that binding the sub-FFT kept the transform
// hot path allocation-free. The bound kernel may carry a prepared twiddle
// layout, but preparing it is plan-time work; nothing may be allocated per
// transform.
//
// 9973 is excluded: its mixed-radix route is covered by the norace file, whose
// pooled scratch does not survive race instrumentation.
func TestBluesteinSubFFTZeroAlloc(t *testing.T) {
	for _, n := range []int{1009, 2003} {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			bluesteinZeroAlloc[complex64](t, n)
			bluesteinZeroAlloc[complex128](t, n)
		})
	}
}

func bluesteinZeroAlloc[T Complex](t *testing.T, n int) {
	t.Helper()

	plan, err := NewPlan[T](n)
	if err != nil {
		t.Fatalf("NewPlan(%d) failed: %v", n, err)
	}

	exec, ok := plan.exec.(*bluesteinExecutor[T])
	if !ok || exec.sub == nil {
		t.Fatalf("n=%d %T: sub-FFT not bound; this test would measure the wrong path", n, *new(T))
	}

	src := make([]T, n)
	for i, v := range broadbandComplex64(n) {
		src[i] = T(complex128(v))
	}

	dst := make([]T, n)

	// Warm the pooled scratch before counting.
	_ = plan.Forward(dst, src)
	_ = plan.Inverse(dst, src)

	if allocs := testing.AllocsPerRun(50, func() {
		_ = plan.Forward(dst, src)
		_ = plan.Inverse(dst, src)
	}); allocs != 0 {
		t.Errorf("n=%d %T: transforms allocate %v per run, want 0", n, *new(T), allocs)
	}
}

// broadbandComplex64 builds an input with energy in every bin, so a wrong
// twiddle or a wrong output ordering shows up as a wrong spectrum.
func broadbandComplex64(n int) []complex64 {
	src := make([]complex64, n)
	for i := range src {
		// Deterministic, aperiodic, and not a lattice frequency: a real sine at
		// an integer bin would leave n-2 bins at zero and measure nothing.
		phase := 2 * math.Pi * float64(i) * 0.31830988618 // 1/pi
		src[i] = complex64(complex(
			math.Cos(phase)+0.5*math.Cos(3.7*phase),
			math.Sin(1.3*phase)-0.25*math.Sin(0.9*phase),
		))
	}

	return src
}

func cabs128(v complex128) float64 {
	return math.Hypot(real(v), imag(v))
}
