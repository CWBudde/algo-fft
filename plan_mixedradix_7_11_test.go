package algofft

import (
	"math"
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
var mixedRadix7And11PlanSizes = []int{11, 21, 33, 35, 44, 49, 77, 308, 385, 448, 704, 1100, 1344, 2156}

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
// gate: 7/11-smooth shapes whose Bluestein pad stays under ~2.5n, so the
// padded power-of-two sub-FFT lands on an unusually effective codelet and
// measured at or ahead of the mixed-radix engine. 22, 44 and 308 used to be
// listed here on the power-of-two-part rule; they now pad to >= 2.5n and
// route to mixed-radix (see TestNewPlan_MixedRadix7And11).
func TestNewPlan_7And11GateKeepsBluestein(t *testing.T) {
	t.Parallel()

	for _, n := range []int{7, 14, 28, 63, 121, 231, 462, 847, 924} {
		plan, err := NewPlan[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		if got := plan.KernelStrategy(); got != KernelBluestein {
			t.Errorf("NewPlan[complex64](%d): KernelStrategy() = %v, want KernelBluestein", n, got)
		}
	}
}

// TestNewPlan_MixedRadixAudioRates covers the large 7-smooth lengths the win
// gate started routing to the mixed-radix engine in 2026-07 — 44100 above all,
// the canonical audio sample rate and the worst result in the v0.7.0 external
// sweep. They are too large for reference.NaiveDFT (44100 would be 1.9e9
// operations), so each is cross-checked bin-by-bin against the Bluestein route
// it used to take, which is itself reference-validated at smaller sizes.
//
// The input is broadband on purpose. An impulse would pass over a wrong
// twiddle table or a wrong output ordering, which is exactly how a silent
// wrong-answer defect survived in the recursive decomposition (PLAN.md P5.0).
func TestNewPlan_MixedRadixAudioRates(t *testing.T) {
	t.Parallel()

	for _, n := range []int{4900, 6300, 8820, 22050, 44100} {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			mixed, err := NewPlan[complex128](n)
			if err != nil {
				t.Fatalf("NewPlan[complex128](%d) failed: %v", n, err)
			}

			if got := mixed.KernelStrategy(); got == KernelBluestein {
				t.Fatalf("NewPlan[complex128](%d) resolved to Bluestein, want mixed-radix", n)
			}

			blue, err := NewPlanWithOptions[complex128](n, PlanOptions{Strategy: KernelBluestein})
			if err != nil {
				t.Fatalf("NewPlanWithOptions(%d, Bluestein) failed: %v", n, err)
			}

			src := randomComplex128(n, int64(n))
			got := make([]complex128, n)
			want := make([]complex128, n)

			if err := mixed.Forward(got, src); err != nil {
				t.Fatalf("mixed-radix Forward failed: %v", err)
			}

			if err := blue.Forward(want, src); err != nil {
				t.Fatalf("Bluestein Forward failed: %v", err)
			}

			var peak float64
			for _, v := range want {
				peak = math.Max(peak, cmplx.Abs(v))
			}

			// Both routes accumulate float64 rounding over ~log(n) stages; the
			// bound is relative to the spectrum peak, not to each bin, so
			// near-zero bins do not set an unreachable target.
			tol := 1e-9 * peak
			for i := range got {
				if diff := cmplx.Abs(got[i] - want[i]); diff > tol {
					t.Fatalf("n=%d bin %d: mixed-radix %v, Bluestein %v (diff %.3e > %.3e)",
						n, i, got[i], want[i], diff, tol)
				}
			}

			back := make([]complex128, n)
			if err := mixed.Inverse(back, got); err != nil {
				t.Fatalf("mixed-radix Inverse failed: %v", err)
			}

			for i := range back {
				if diff := cmplx.Abs(back[i] - src[i]); diff > 1e-11*float64(n) {
					t.Fatalf("n=%d: round-trip mismatch at %d: got %v want %v", n, i, back[i], src[i])
				}
			}
		})
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

// BenchmarkMixedRadixInverse covers the mixed-radix inverse at non-power-of-two
// lengths. BenchmarkMixedRadix7And11VsBluestein above measures only the forward
// direction, which leaves the inverse-only tail of the mixed-radix driver — the
// loop that undoes a codelet's built-in 1/n scaling
// (mixedRadixRecursivePingPong*AVX2 in internal/fft) — with no benchmark at all.
func BenchmarkMixedRadixInverse(b *testing.B) {
	// Mixed-radix-eligible lengths spanning one codelet-sized sub-transform up
	// to several: 385 = 5·7·11, 1155 = 3·5·7·11, 3584 = 2^9·7, 7168 = 2^10·7.
	sizes := []int{385, 1155, 3584, 7168}

	run64 := func(b *testing.B, n int) {
		b.Helper()

		plan, err := NewPlan[complex64](n)
		if err != nil {
			b.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		src := randomComplex64(n, int64(n))
		dst := make([]complex64, n)

		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()

		for range b.N {
			_ = plan.Inverse(dst, src)
		}
	}

	run128 := func(b *testing.B, n int) {
		b.Helper()

		plan, err := NewPlan[complex128](n)
		if err != nil {
			b.Fatalf("NewPlan[complex128](%d) failed: %v", n, err)
		}

		src := randomComplex128(n, int64(n))
		dst := make([]complex128, n)

		b.ReportAllocs()
		b.SetBytes(int64(n * 16))
		b.ResetTimer()

		for range b.N {
			_ = plan.Inverse(dst, src)
		}
	}

	for _, n := range sizes {
		b.Run("Complex64_"+strconv.Itoa(n), func(b *testing.B) { run64(b, n) })
		b.Run("Complex128_"+strconv.Itoa(n), func(b *testing.B) { run128(b, n) })
	}
}
