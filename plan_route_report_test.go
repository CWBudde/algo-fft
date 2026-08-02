package algofft

import (
	"math/cmplx"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// Lengths that reach the mixed-radix engine: the smooth non-powers of two
// named in the benchmark logs (1000 was reported as dit_fallback, the rest as
// stockham, and none of those routes ever executed), plus a 5-smooth square
// large enough that the old auto heuristic would have claimed six-step.
//
//nolint:gochecknoglobals // shared by the tests below
var mixedRadixRoutedLengths = []int{1000, 2205, 3600, 12000, 44100, 900 * 900}

// TestMixedRadixLengthsReportTheirRoute pins the reported strategy and
// algorithm of every non-power-of-two length the mixed-radix engine executes.
// The kernel dispatch takes that engine before it looks at the strategy, so a
// power-of-two strategy name here would describe a route that never runs.
func TestMixedRadixLengthsReportTheirRoute(t *testing.T) {
	t.Parallel()

	for _, n := range mixedRadixRoutedLengths {
		plan64, err := NewPlan32(n)
		if err != nil {
			t.Fatalf("NewPlan32(%d): %v", n, err)
		}

		plan128, err := NewPlan64(n)
		if err != nil {
			t.Fatalf("NewPlan64(%d): %v", n, err)
		}

		for _, p := range []PlanInfo{plan64, plan128} {
			if got := p.KernelStrategies()[0]; got != KernelMixedRadix {
				t.Errorf("n=%d: KernelStrategy() = %v, want MixedRadix", n, got)
			}

			if got := p.Algorithms()[0]; got != "mixedradix" {
				t.Errorf("n=%d: Algorithm() = %q, want \"mixedradix\"", n, got)
			}
		}
	}
}

// TestForcedStrategyAtMixedRadixLengthReportsRoute verifies that forcing a
// power-of-two strategy at a mixed-radix length does not buy a label: the
// engine still runs, so the plan still says so.
func TestForcedStrategyAtMixedRadixLengthReportsRoute(t *testing.T) {
	t.Parallel()

	forced := []KernelStrategy{
		KernelDIT, KernelStockham, KernelSixStep,
		KernelSplitRadix, KernelFourStep,
	}

	for _, strategy := range forced {
		plan, err := NewPlanWithOptions[complex64](1000, PlanOptions{Strategy: strategy})
		if err != nil {
			t.Fatalf("NewPlanWithOptions(1000, %v): %v", strategy, err)
		}

		if got := plan.KernelStrategy(); got != KernelMixedRadix {
			t.Errorf("forced %v at n=1000: KernelStrategy() = %v, want MixedRadix", strategy, got)
		}
	}
}

// TestForcedBluesteinAtMixedRadixLengthIsHonored guards the one strategy that
// really does replace the mixed-radix route at a smooth length: Bluestein is
// executed as asked, so it is reported as asked.
func TestForcedBluesteinAtMixedRadixLengthIsHonored(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanWithOptions[complex64](1000, PlanOptions{Strategy: KernelBluestein})
	if err != nil {
		t.Fatalf("NewPlanWithOptions(1000, Bluestein): %v", err)
	}

	if got := plan.KernelStrategy(); got != KernelBluestein {
		t.Errorf("forced Bluestein at n=1000: KernelStrategy() = %v, want Bluestein", got)
	}

	if got := plan.Algorithm(); got != "bluestein" {
		t.Errorf("forced Bluestein at n=1000: Algorithm() = %q, want \"bluestein\"", got)
	}
}

// TestForcedMixedRadixAtPowerOfTwoFallsBack checks the reverse direction: the
// engine is not the route for a power of two, so forcing it must not mislabel
// the plan either.
func TestForcedMixedRadixAtPowerOfTwoFallsBack(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanWithOptions[complex64](4096, PlanOptions{Strategy: KernelMixedRadix})
	if err != nil {
		t.Fatalf("NewPlanWithOptions(4096, MixedRadix): %v", err)
	}

	if got := plan.KernelStrategy(); got == KernelMixedRadix {
		t.Errorf("forced MixedRadix at n=4096: KernelStrategy() = %v, want a power-of-two strategy", got)
	}
}

// TestMixedRadixRoutedLengthsStayCorrect guards the relabeling itself: the
// strategy value feeds kernel dispatch and scratch sizing, so a spectrum check
// against the reference DFT confirms the route did not move with its name.
// Driven by a broadband signal, not an impulse (see the permutation-blind
// test-vector finding in PLAN.md §3).
func TestMixedRadixRoutedLengthsStayCorrect(t *testing.T) {
	t.Parallel()

	for _, n := range []int{1000, 2205, 3600} {
		plan, err := NewPlan64(n)
		if err != nil {
			t.Fatalf("NewPlan64(%d): %v", n, err)
		}

		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(float64((i*37)%101)/101.0-0.5, float64((i*53)%97)/97.0-0.5)
		}

		got := make([]complex128, n)
		if err := plan.Forward(got, src); err != nil {
			t.Fatalf("Forward(%d): %v", n, err)
		}

		want := reference.NaiveDFT128(src)

		for i := range want {
			if diff := cmplx.Abs(got[i] - want[i]); diff > 1e-8*float64(n) {
				t.Fatalf("n=%d bin %d: got %v, want %v (diff %g)", n, i, got[i], want[i], diff)
			}
		}
	}
}
