package algofft

import (
	"math"
	"math/cmplx"
	"strconv"
	"testing"
)

// recursiveLargeSizes are the power-of-two lengths at which the recursive
// strategy must decompose (the largest codelet leaf is 512), so they exercise
// the split/combine path rather than a single codelet call.
var recursiveLargeSizes = []int{1024, 2048, 4096, 8192, 16384}

// TestRecursiveMatchesDefaultPlan cross-checks the recursive strategy against
// the default (auto-selected) plan at the sizes that force decomposition.
// Plan-level coverage of KernelRecursive previously stopped at n=64, so the
// twiddle layout and scratch sizing for multi-level trees went unverified.
func TestRecursiveMatchesDefaultPlan(t *testing.T) {
	t.Parallel()

	for _, n := range recursiveLargeSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			src := make([]complex128, n)
			for i := range src {
				phase := float64(i) * 0.37
				src[i] = complex(math.Sin(phase)+0.25*float64(i%7), math.Cos(phase*1.7))
			}

			want := make([]complex128, n)

			ref, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan(%d): %v", n, err)
			}

			if err := ref.Forward(want, src); err != nil {
				t.Fatalf("reference Forward: %v", err)
			}

			rec, err := NewPlanWithOptions[complex128](n, PlanOptions{Strategy: KernelRecursive})
			if err != nil {
				t.Fatalf("NewPlanWithOptions(%d, recursive): %v", n, err)
			}

			got := make([]complex128, n)
			if err := rec.Forward(got, src); err != nil {
				t.Fatalf("recursive Forward: %v", err)
			}

			assertSpectrumClose(t, "Forward", got, want, 1e-9)

			// Round-trip: Inverse(Forward(x)) == x.
			back := make([]complex128, n)
			if err := rec.Inverse(back, got); err != nil {
				t.Fatalf("recursive Inverse: %v", err)
			}

			assertSpectrumClose(t, "round-trip", back, src, 1e-9)
		})
	}
}

// TestRecursiveMatchesDefaultPlanComplex64 repeats the cross-check in single
// precision, where the codelet leaves differ from the complex128 registry.
func TestRecursiveMatchesDefaultPlanComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range recursiveLargeSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			src := make([]complex64, n)
			for i := range src {
				phase := float64(i) * 0.37
				src[i] = complex(float32(math.Sin(phase)), float32(math.Cos(phase*1.7)))
			}

			ref, err := NewPlan[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan(%d): %v", n, err)
			}

			want := make([]complex64, n)
			if err := ref.Forward(want, src); err != nil {
				t.Fatalf("reference Forward: %v", err)
			}

			rec, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: KernelRecursive})
			if err != nil {
				t.Fatalf("NewPlanWithOptions(%d, recursive): %v", n, err)
			}

			got := make([]complex64, n)
			if err := rec.Forward(got, src); err != nil {
				t.Fatalf("recursive Forward: %v", err)
			}

			// float32 accumulation over a length-n transform: scale the bound
			// with n and with the spectrum magnitude.
			tol := 2e-4 * float64(n)
			want128 := make([]complex128, n)
			got128 := make([]complex128, n)

			for i := range want {
				want128[i] = complex128(want[i])
				got128[i] = complex128(got[i])
			}

			assertSpectrumClose(t, "Forward", got128, want128, tol)
		})
	}
}

func assertSpectrumClose(t *testing.T, label string, got, want []complex128, tol float64) {
	t.Helper()

	var (
		worst    float64
		worstIdx int
	)

	for i := range want {
		d := cmplx.Abs(got[i] - want[i])
		if d > worst {
			worst, worstIdx = d, i
		}
	}

	if worst > tol {
		t.Errorf("%s: max abs error %.6g at bin %d (tolerance %.6g)\n got=%v\nwant=%v",
			label, worst, worstIdx, tol, got[worstIdx], want[worstIdx])
	}
}
