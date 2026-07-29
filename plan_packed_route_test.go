package algofft

import (
	"math"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
	"github.com/cwbudde/algo-fft/internal/transform"
)

// These tests exercise the packed radix-4 Stockham engine *through a Plan*.
// Until the build toggle became a runtime policy that was impossible on a SIMD
// build — the route was compiled out, so nothing above internal/transform ever
// ran it. The ForceOn override is what makes it reachable here, and at small
// sizes where the naive DFT is still affordable as a reference.
//
// The override is process-global and read at plan construction, so none of
// these may call t.Parallel().

func withPackedOverride(t *testing.T, mode transform.PackedOverride) {
	t.Helper()
	transform.SetPackedStockhamOverride(mode)
	t.Cleanup(func() { transform.SetPackedStockhamOverride(transform.PackedOverrideDefault) })
}

// packedOf returns the plan's packed table, failing if the plan is not a
// kernel-executor plan at all.
func packedOf[T Complex](t *testing.T, plan *Plan[T]) *transform.PackedTwiddles[T] {
	t.Helper()

	exec, ok := plan.exec.(*kernelExecutor[T])
	if !ok {
		t.Fatalf("executor is %T, want *kernelExecutor", plan.exec)
	}

	return exec.packed
}

// TestPackedRouteFollowsOverride pins that the override actually reaches the
// plan. Without this, a mis-wired knob would make the measurement compare the
// same code against itself and report "no difference".
func TestPackedRouteFollowsOverride(t *testing.T) {
	// Forced Stockham, so tryRegistry declines the codelet at every size and
	// the estimate really does resolve to KernelStockham (see PLAN.md 3).
	opts := PlanOptions{Strategy: KernelStockham}

	for _, n := range []int{1024, 4096, 1 << 17} {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			for _, tc := range []struct {
				mode transform.PackedOverride
				want bool
			}{
				{transform.PackedOverrideForceOn, true},
				{transform.PackedOverrideForceOff, false},
			} {
				withPackedOverride(t, tc.mode)

				plan, err := NewPlanWithOptions[complex64](n, opts)
				if err != nil {
					t.Fatalf("NewPlan(%d) failed: %v", n, err)
				}

				if got := packedOf(t, plan) != nil; got != tc.want {
					t.Errorf("n=%d mode=%v: packed bound = %v, want %v", n, tc.mode, got, tc.want)
				}
			}
		})
	}
}

// TestPackedRouteNeverBehindCodelet pins the invariant the ForwardCodelet guard
// in newKernelExecutor encodes: an auto plan at a codelet-covered size must not
// allocate the packed table under *any* override. Every registered codelet is
// Algorithm: KernelDIT, so the estimate never reports Stockham there — this
// test is what would catch that changing.
func TestPackedRouteNeverBehindCodelet(t *testing.T) {
	for _, mode := range []transform.PackedOverride{
		transform.PackedOverrideDefault,
		transform.PackedOverrideForceOn,
		transform.PackedOverrideForceOff,
	} {
		withPackedOverride(t, mode)

		for _, n := range []int{256, 1024, 4096} {
			plan, err := NewPlan[complex64](n)
			if err != nil {
				t.Fatalf("NewPlan(%d) failed: %v", n, err)
			}

			if plan.forwardCodelet == nil {
				continue // no codelet at this size on this build; nothing to pin
			}

			if packedOf(t, plan) != nil {
				t.Errorf("n=%d mode=%v: packed table allocated behind a bound codelet", n, mode)
			}
		}
	}
}

// TestPackedRouteMatchesReference compares the packed route bin-by-bin against
// the naive DFT and element-wise against the route it would replace. Sizes
// include odd log2 (2048, 512), which take the extra leading radix-2 stage.
func TestPackedRouteMatchesReference(t *testing.T) {
	for _, n := range []int{256, 512, 1024, 2048, 4096} {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			packedRouteMatches[complex64](t, n, 2e-5)
			packedRouteMatches[complex128](t, n, 1e-12)
		})
	}
}

func packedRouteMatches[T Complex](t *testing.T, n int, tol float64) {
	t.Helper()

	opts := PlanOptions{Strategy: KernelStockham}

	src := make([]T, n)
	for i, v := range packedBroadband(n) {
		src[i] = T(complex128(v))
	}

	run := func(mode transform.PackedOverride) ([]T, []T) {
		withPackedOverride(t, mode)

		plan, err := NewPlanWithOptions[T](n, opts)
		if err != nil {
			t.Fatalf("NewPlan(%d) failed: %v", n, err)
		}

		if bound := packedOf(t, plan) != nil; bound != (mode == transform.PackedOverrideForceOn) {
			t.Fatalf("n=%d mode=%v: packed bound = %v; test would measure the wrong path", n, mode, bound)
		}

		freq := make([]T, n)
		if err := plan.Forward(freq, src); err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		back := make([]T, n)
		if err := plan.Inverse(back, freq); err != nil {
			t.Fatalf("Inverse failed: %v", err)
		}

		return freq, back
	}

	packedFreq, packedBack := run(transform.PackedOverrideForceOn)
	kernelFreq, _ := run(transform.PackedOverrideForceOff)

	want := packedReference(src)

	var peak float64
	for _, v := range want {
		peak = math.Max(peak, math.Hypot(real(v), imag(v)))
	}

	for i := range packedFreq {
		if d := absDiff(packedFreq[i], want[i]); d > tol*peak {
			t.Fatalf("n=%d %T bin %d: packed forward %v, reference %v (diff %g)",
				n, *new(T), i, packedFreq[i], want[i], d)
		}

		if d := absDiff(packedFreq[i], complex128(kernelFreq[i])); d > tol*peak {
			t.Fatalf("n=%d %T bin %d: packed %v vs kernel %v (diff %g)",
				n, *new(T), i, packedFreq[i], kernelFreq[i], d)
		}
	}

	for i := range packedBack {
		if d := absDiff(packedBack[i], complex128(src[i])); d > tol {
			t.Fatalf("n=%d %T bin %d: round trip %v, want %v (diff %g)",
				n, *new(T), i, packedBack[i], src[i], d)
		}
	}
}

// packedReference computes the naive DFT in float64 regardless of T, so the
// comparison is not limited by the reference's own rounding (PLAN.md 1.6).
func packedReference[T Complex](src []T) []complex128 {
	switch s := any(src).(type) {
	case []complex64:
		return reference.NaiveDFTWide(s)
	case []complex128:
		return reference.NaiveDFT128(s)
	default:
		return nil
	}
}

func absDiff[T Complex](got T, want complex128) float64 {
	d := complex128(got) - want

	return math.Hypot(real(d), imag(d))
}

// packedBroadband builds an input with energy in every bin. An impulse would
// not do: its spectrum is all-ones, so a wrong twiddle or a wrong output
// ordering both survive it (PLAN.md 2.4).
func packedBroadband(n int) []complex64 {
	src := make([]complex64, n)
	for i := range src {
		phase := 2 * math.Pi * float64(i) * 0.31830988618 // 1/pi: aperiodic on any lattice
		src[i] = complex64(complex(
			math.Cos(phase)+0.5*math.Cos(3.7*phase),
			math.Sin(1.3*phase)-0.25*math.Sin(0.9*phase),
		))
	}

	return src
}
