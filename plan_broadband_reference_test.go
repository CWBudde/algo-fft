package algofft

import (
	"math"
	"math/cmplx"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// The plan-level reference tests split unevenly by precision:
// TestPlannerModesMatchReference sweeps complex64 across every planner mode up
// to 16384, but complex128 was only compared against the naive DFT at n <= 16
// (plan_reference_test.go) and at non-power-of-two lengths (Bluestein, Rader,
// mixed-radix). Everything above that relied on round-trip and impulse tests,
// both of which are blind to a wrong twiddle table and a wrong bin order — the
// combination that hid a wrong-spectrum bug in the recursive complex128 path
// at every size >= 1024.
//
// These tests close that gap: a broadband signal, every strategy the dispatch
// can route a power of two to, compared bin-by-bin against internal/reference.

// broadbandStrategies is every strategy a power-of-two length can be forced
// to. Forcing one the dispatch cannot honour at a given size resolves to the
// route that runs (see internal/planner.ResolveKernelStrategy), so each entry
// exercises whatever actually executes rather than failing the plan.
//
//nolint:gochecknoglobals // shared table for the broadband reference tests
var broadbandStrategies = []struct {
	name     string
	strategy KernelStrategy
}{
	{"Auto", KernelAuto},
	{"DIT", KernelDIT},
	{"Stockham", KernelStockham},
	{"SplitRadix", KernelSplitRadix},
	{"Recursive", KernelRecursive},
	{"SixStep", KernelSixStep},
	{"FourStep", KernelFourStep},
	{"Bluestein", KernelBluestein},
}

// broadbandReferenceSizes stop at 4096: the naive complex128 DFT costs ~0.27 s
// there and ~4.3 s at 16384, which is not worth paying per strategy.
//
//nolint:gochecknoglobals // shared table for the broadband reference tests
var broadbandReferenceSizes = []int{256, 1024, 4096}

// broadbandSrc128 has energy in every bin and no symmetry a permutation could
// preserve: two incommensurate tones plus a ramp, with a different phase in
// the imaginary part.
func broadbandSrc128(n int) []complex128 {
	src := make([]complex128, n)
	for i := range src {
		f := float64(i)
		src[i] = complex(
			math.Cos(0.7*f)+0.3*math.Sin(2.9*f)+0.05*math.Sqrt(f),
			math.Sin(1.3*f)-0.4*math.Cos(0.11*f),
		)
	}

	return src
}

// TestForwardBroadbandMatchesReference128 pins the forward complex128 spectrum
// of every power-of-two route against the naive DFT.
func TestForwardBroadbandMatchesReference128(t *testing.T) {
	t.Parallel()

	for _, n := range broadbandReferenceSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			src := broadbandSrc128(n)
			want := reference.NaiveDFT128(src)

			var peak float64
			for _, v := range want {
				peak = math.Max(peak, cmplx.Abs(v))
			}

			// Relative to the spectrum peak: bounding each bin against its own
			// magnitude sets an unreachable target for the near-zero ones.
			tol := 1e-11 * peak

			for _, s := range broadbandStrategies {
				t.Run(s.name, func(t *testing.T) {
					t.Parallel()

					plan, err := NewPlanWithOptions[complex128](n, PlanOptions{Strategy: s.strategy})
					if err != nil {
						t.Fatalf("NewPlanWithOptions(%d, %v): %v", n, s.strategy, err)
					}

					got := make([]complex128, n)
					if err := plan.Forward(got, src); err != nil {
						t.Fatalf("Forward: %v", err)
					}

					for i := range got {
						if diff := cmplx.Abs(got[i] - want[i]); diff > tol {
							t.Fatalf("n=%d algo=%s bin %d: got %v, want %v (diff %.3e > %.3e)",
								n, plan.Algorithm(), i, got[i], want[i], diff, tol)
						}
					}
				})
			}
		})
	}
}

// TestInverseBroadbandMatchesReference128 is the inverse counterpart: a
// forward-only check cannot see an inverse-side twiddle or ordering error that
// the round-trip tests cancel out.
func TestInverseBroadbandMatchesReference128(t *testing.T) {
	t.Parallel()

	for _, n := range broadbandReferenceSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			freq := broadbandSrc128(n)
			want := reference.NaiveIDFT128(freq)

			var peak float64
			for _, v := range want {
				peak = math.Max(peak, cmplx.Abs(v))
			}

			tol := 1e-11 * peak

			for _, s := range broadbandStrategies {
				t.Run(s.name, func(t *testing.T) {
					t.Parallel()

					plan, err := NewPlanWithOptions[complex128](n, PlanOptions{Strategy: s.strategy})
					if err != nil {
						t.Fatalf("NewPlanWithOptions(%d, %v): %v", n, s.strategy, err)
					}

					got := make([]complex128, n)
					if err := plan.Inverse(got, freq); err != nil {
						t.Fatalf("Inverse: %v", err)
					}

					for i := range got {
						if diff := cmplx.Abs(got[i] - want[i]); diff > tol {
							t.Fatalf("n=%d algo=%s bin %d: got %v, want %v (diff %.3e > %.3e)",
								n, plan.Algorithm(), i, got[i], want[i], diff, tol)
						}
					}
				})
			}
		})
	}
}

// TestForwardBroadbandMatchesReference64 gives the complex64 side the same
// per-strategy sweep. TestPlannerModesMatchReference covers the planner modes
// but only the route each mode picks; forcing a strategy reaches kernels no
// mode would select at that size.
func TestForwardBroadbandMatchesReference64(t *testing.T) {
	t.Parallel()

	for _, n := range broadbandReferenceSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			wide := broadbandSrc128(n)
			src := make([]complex64, n)

			for i, v := range wide {
				src[i] = complex64(v)
			}

			// Accumulated in float64 so the reference is not itself limited by
			// float32 rounding.
			want := reference.NaiveDFTWide(src)

			var peak float64
			for _, v := range want {
				peak = math.Max(peak, cmplx.Abs(v))
			}

			tol := 1e-5 * peak

			for _, s := range broadbandStrategies {
				t.Run(s.name, func(t *testing.T) {
					t.Parallel()

					plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: s.strategy})
					if err != nil {
						t.Fatalf("NewPlanWithOptions(%d, %v): %v", n, s.strategy, err)
					}

					got := make([]complex64, n)
					if err := plan.Forward(got, src); err != nil {
						t.Fatalf("Forward: %v", err)
					}

					for i := range got {
						if diff := cmplx.Abs(complex128(got[i]) - want[i]); diff > tol {
							t.Fatalf("n=%d algo=%s bin %d: got %v, want %v (diff %.3e > %.3e)",
								n, plan.Algorithm(), i, got[i], want[i], diff, tol)
						}
					}
				})
			}
		})
	}
}
