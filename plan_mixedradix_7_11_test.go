package algofft

import (
	"math/cmplx"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// mixedRadix7And11PlanSizes are representative lengths with factors 7 and/or
// 11 that previously required Bluestein and now run exactly on the
// mixed-radix engine (they pass the planner.MixedRadixEligible win gate).
//
//nolint:gochecknoglobals
var mixedRadix7And11PlanSizes = []int{11, 21, 33, 35, 49, 77, 385, 448, 704, 1344}

// TestNewPlan_MixedRadix7And11 locks in the routing: sizes whose factors are
// all in {2,3,5,7,11} are served by the mixed-radix engine (not Bluestein),
// and the transform matches the naive DFT with a clean round-trip.
func TestNewPlan_MixedRadix7And11(t *testing.T) {
	t.Parallel()

	for _, n := range mixedRadix7And11PlanSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
			}

			if plan.KernelStrategy() == KernelBluestein {
				t.Fatalf("NewPlan[complex64](%d) resolved to Bluestein, want mixed-radix", n)
			}

			src := randomComplex128(n, int64(n))
			dst := make([]complex128, n)

			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			ref := reference.NaiveDFT128(src)

			var maxRef float64
			for i := range ref {
				if m := cmplx.Abs(ref[i]); m > maxRef {
					maxRef = m
				}
			}

			for i := range dst {
				if cmplx.Abs(dst[i]-ref[i])/maxRef > 1e-11 {
					t.Fatalf("n=%d: forward mismatch at %d: got %v want %v", n, i, dst[i], ref[i])
				}
			}

			back := make([]complex128, n)
			if err := plan.Inverse(back, dst); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			for i := range back {
				if cmplx.Abs(back[i]-src[i]) > 1e-11*float64(n) {
					t.Fatalf("n=%d: round-trip mismatch at %d: got %v want %v", n, i, back[i], src[i])
				}
			}
		})
	}
}

// TestNewPlan_7And11GateKeepsBluestein locks in the other side of the win
// gate: 7/11-smooth shapes that measured as losses against Bluestein's padded
// sub-FFT (tiny odd sizes with a pad under ~2.5n, and power-of-two parts of
// 2 or 4) keep their previous Bluestein routing.
func TestNewPlan_7And11GateKeepsBluestein(t *testing.T) {
	t.Parallel()

	for _, n := range []int{7, 14, 22, 28, 63, 121, 231, 308, 462, 847, 924} {
		plan, err := NewPlan[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		if got := plan.KernelStrategy(); got != KernelBluestein {
			t.Errorf("NewPlan[complex64](%d): KernelStrategy() = %v, want KernelBluestein", n, got)
		}
	}
}

// TestNewPlan_ForcedBluestein7And11 checks that explicitly forcing
// KernelBluestein still works for the newly mixed-radix-capable lengths.
func TestNewPlan_ForcedBluestein7And11(t *testing.T) {
	t.Parallel()

	for _, n := range []int{7, 77, 448} {
		plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: KernelBluestein})
		if err != nil {
			t.Fatalf("NewPlanWithOptions(%d, Bluestein) failed: %v", n, err)
		}

		if got := plan.KernelStrategy(); got != KernelBluestein {
			t.Fatalf("n=%d: KernelStrategy() = %v, want KernelBluestein", n, got)
		}

		src := randomComplex64(n, int64(n))
		freq := make([]complex64, n)
		back := make([]complex64, n)

		if err := plan.Forward(freq, src); err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		if err := plan.Inverse(back, freq); err != nil {
			t.Fatalf("Inverse failed: %v", err)
		}

		for i := range back {
			if cmplx.Abs(complex128(back[i]-src[i])) > 1e-4 {
				t.Fatalf("n=%d: round-trip mismatch at %d: got %v want %v", n, i, back[i], src[i])
			}
		}
	}
}

// TestPlanReal_Radix7And11 covers the real-FFT path over the new lengths:
// even widths use the packed half-size method (whose child complex plan may
// now be mixed-radix, e.g. 224 for n=448), odd widths use the complex
// fallback.
func TestPlanReal_Radix7And11(t *testing.T) {
	t.Parallel()

	for _, n := range []int{14, 63, 448, 704, 1344} {
		plan, err := NewPlanReal64(n)
		if err != nil {
			t.Fatalf("NewPlanReal64(%d) failed: %v", n, err)
		}

		src := make([]float64, n)
		for i := range src {
			src[i] = float64(i%13) - 6
		}

		complexSrc := make([]complex128, n)
		for i, v := range src {
			complexSrc[i] = complex(v, 0)
		}

		spectrum := make([]complex128, plan.SpectrumLen())
		if err := plan.Forward(spectrum, src); err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		ref := reference.NaiveDFT128(complexSrc)
		for i := range spectrum {
			if cmplx.Abs(spectrum[i]-ref[i]) > 1e-9*float64(n) {
				t.Fatalf("n=%d: spectrum mismatch at %d: got %v want %v", n, i, spectrum[i], ref[i])
			}
		}

		back := make([]float64, n)
		if err := plan.Inverse(back, spectrum); err != nil {
			t.Fatalf("Inverse failed: %v", err)
		}

		for i := range back {
			if diff := back[i] - src[i]; diff > 1e-9*float64(n) || diff < -1e-9*float64(n) {
				t.Fatalf("n=%d: round-trip mismatch at %d: got %v want %v", n, i, back[i], src[i])
			}
		}
	}
}

// BenchmarkMixedRadix7And11VsBluestein compares the mixed-radix path (now the
// default for gate-passing lengths with factors 7/11) against the previous
// Bluestein routing, which stays reachable via PlanOptions.Strategy. This is
// the measurement behind the planner.MixedRadixEligible win gate; to
// re-evaluate the gated-out shapes, temporarily add them here (e.g. 7, 14,
// 28, 63, 121, 231, 308, 462, 847, 924) and relax the gate.
func BenchmarkMixedRadix7And11VsBluestein(b *testing.B) {
	sizes := []int{11, 21, 33, 35, 49, 77, 385, 448, 693, 704, 1155, 1344, 2401, 3584, 7168, 11264, 14080}

	run64 := func(b *testing.B, n int, opts PlanOptions) {
		b.Helper()

		plan, err := NewPlanWithOptions[complex64](n, opts)
		if err != nil {
			b.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		src := randomComplex64(n, int64(n))
		dst := make([]complex64, n)

		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()

		for range b.N {
			_ = plan.Forward(dst, src)
		}
	}

	run128 := func(b *testing.B, n int, opts PlanOptions) {
		b.Helper()

		plan, err := NewPlanWithOptions[complex128](n, opts)
		if err != nil {
			b.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		src := randomComplex128(n, int64(n))
		dst := make([]complex128, n)

		b.ReportAllocs()
		b.SetBytes(int64(n * 16))
		b.ResetTimer()

		for range b.N {
			_ = plan.Forward(dst, src)
		}
	}

	for _, n := range sizes {
		b.Run("MixedRadix_"+strconv.Itoa(n), func(b *testing.B) {
			run64(b, n, PlanOptions{})
		})
		b.Run("Bluestein_"+strconv.Itoa(n), func(b *testing.B) {
			run64(b, n, PlanOptions{Strategy: KernelBluestein})
		})
		b.Run("MixedRadix128_"+strconv.Itoa(n), func(b *testing.B) {
			run128(b, n, PlanOptions{})
		})
		b.Run("Bluestein128_"+strconv.Itoa(n), func(b *testing.B) {
			run128(b, n, PlanOptions{Strategy: KernelBluestein})
		})
	}
}
