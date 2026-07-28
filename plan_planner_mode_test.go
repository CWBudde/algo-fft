package algofft

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// plannerModes lists every planner mode, named for subtests.
//
//nolint:gochecknoglobals // shared table for the planner-mode tests
var plannerModes = []struct {
	name string
	mode PlannerMode
}{
	{"Estimate", PlannerEstimate},
	{"Measure", PlannerMeasure},
	{"Patient", PlannerPatient},
	{"Exhaustive", PlannerExhaustive},
}

// plannerModeSizes covers the codelet-backed power-of-two sizes plus one size
// past the codelet range, where the measured winner is a kernel strategy.
//
//nolint:gochecknoglobals // shared table for the planner-mode tests
var plannerModeSizes = []int{8, 64, 256, 1024, 4096, 16384}

// TestPlannerModesMatchReference checks every planner mode against the naive
// DFT. The measuring modes benchmark codelets as candidates and bind whichever
// one wins, so any codelet in the registry can end up in a plan — including
// ones that need a prepared twiddle layout. A binding that drops that layout
// silently falls back to a generic kernel, and one that mismatches it would
// produce wrong output; this pins both.
func TestPlannerModesMatchReference(t *testing.T) {
	t.Parallel()

	for _, mode := range plannerModes {
		t.Run(mode.name, func(t *testing.T) {
			t.Parallel()

			for _, n := range plannerModeSizes {
				src := make([]complex64, n)
				for i := range src {
					src[i] = complex(float32(math.Sin(float64(i)*0.37)), float32(math.Cos(float64(i)*0.11)))
				}

				want := reference.NaiveDFTWide(src)

				plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Planner: mode.mode})
				if err != nil {
					t.Fatalf("n=%d: NewPlanWithOptions: %v", n, err)
				}

				got := make([]complex64, n)

				err = plan.Forward(got, src)
				if err != nil {
					t.Fatalf("n=%d: Forward: %v", n, err)
				}

				// Relative to the transform magnitude: the naive DFT sums n terms.
				tol := 1e-4 * float64(n)

				for k := range got {
					diff := cmplx.Abs(complex(float64(real(got[k])), float64(imag(got[k]))) - want[k])
					if diff > tol {
						t.Fatalf("n=%d algo=%s: bin %d = %v, want %v (diff %g > %g)",
							n, plan.Algorithm(), k, got[k], want[k], diff, tol)
					}
				}

				roundTrip := make([]complex64, n)

				err = plan.Inverse(roundTrip, got)
				if err != nil {
					t.Fatalf("n=%d: Inverse: %v", n, err)
				}

				for i := range roundTrip {
					diff := cmplx.Abs(complex128(roundTrip[i]) - complex128(src[i]))
					if diff > 1e-3 {
						t.Fatalf("n=%d algo=%s: round trip at %d = %v, want %v",
							n, plan.Algorithm(), i, roundTrip[i], src[i])
					}
				}
			}
		})
	}
}

// TestPlannerMeasureRecordsWinnerToWisdom verifies the measure/wisdom round
// trip: a measuring mode records the implementation it actually chose, and a
// later PlannerEstimate plan with that wisdom reproduces it instead of
// re-deriving the choice from the registry's static priority order.
func TestPlannerMeasureRecordsWinnerToWisdom(t *testing.T) {
	t.Parallel()

	const n = 1024

	wisdom := NewWisdom()

	measured, err := NewPlanWithOptions[complex64](n, PlanOptions{Planner: PlannerMeasure, Wisdom: wisdom})
	if err != nil {
		t.Fatalf("measure plan: %v", err)
	}

	if wisdom.Len() != 1 {
		t.Fatalf("wisdom has %d entries, want 1", wisdom.Len())
	}

	replayed, err := NewPlanWithOptions[complex64](n, PlanOptions{Planner: PlannerEstimate, Wisdom: wisdom})
	if err != nil {
		t.Fatalf("replay plan: %v", err)
	}

	if replayed.Algorithm() != measured.Algorithm() {
		t.Errorf("replayed algorithm = %q, want the measured %q", replayed.Algorithm(), measured.Algorithm())
	}
}
