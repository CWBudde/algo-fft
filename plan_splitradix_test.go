package algofft

import (
	"math/cmplx"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestSplitRadix_PlanForced validates plans with an explicitly forced
// split-radix strategy against the naive reference for both precisions.
func TestSplitRadix_PlanForced(t *testing.T) {
	t.Parallel()

	for _, n := range []int{8, 64, 256, 1024} {
		t.Run("complex128_"+strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanWithOptions[complex128](n, PlanOptions{Strategy: KernelSplitRadix})
			if err != nil {
				t.Fatalf("NewPlanWithOptions(%d) failed: %v", n, err)
			}

			if plan.KernelStrategy() != KernelSplitRadix {
				t.Fatalf("strategy = %v, want KernelSplitRadix", plan.KernelStrategy())
			}

			src := randomComplex128(n, int64(n))
			dst := make([]complex128, n)

			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			want := reference.NaiveDFT128(src)
			for i := range dst {
				if diff := cmplx.Abs(dst[i] - want[i]); diff > 1e-9 {
					t.Fatalf("bin %d: got %v, want %v (diff %v)", i, dst[i], want[i], diff)
				}
			}

			back := make([]complex128, n)
			if err := plan.Inverse(back, dst); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			for i := range back {
				if diff := cmplx.Abs(back[i] - src[i]); diff > 1e-11 {
					t.Fatalf("round-trip at %d: got %v, want %v", i, back[i], src[i])
				}
			}
		})

		t.Run("complex64_"+strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: KernelSplitRadix})
			if err != nil {
				t.Fatalf("NewPlanWithOptions(%d) failed: %v", n, err)
			}

			src := randomComplex64(n, int64(n))
			dst := make([]complex64, n)

			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			want := reference.NaiveDFT(src)
			for i := range dst {
				if diff := cmplx.Abs(complex128(dst[i] - want[i])); diff > 1e-2 {
					t.Fatalf("bin %d: got %v, want %v (diff %v)", i, dst[i], want[i], diff)
				}
			}
		})
	}
}

// TestSplitRadix_ZeroAllocAndString covers the plan hot path and String().
func TestSplitRadix_ZeroAllocAndString(t *testing.T) {
	n := 512

	plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: KernelSplitRadix})
	if err != nil {
		t.Fatalf("NewPlanWithOptions(%d) failed: %v", n, err)
	}

	if got := plan.String(); got != "Plan[complex64](512, SplitRadix)" {
		t.Errorf("String() = %q", got)
	}

	src := randomComplex64(n, 3)
	dst := make([]complex64, n)

	_ = plan.Forward(dst, src)
	_ = plan.Inverse(dst, src)

	if allocs := testing.AllocsPerRun(100, func() {
		_ = plan.Forward(dst, src)
		_ = plan.Inverse(dst, src)
	}); allocs != 0 {
		t.Errorf("transforms allocate %v times per run, want 0", allocs)
	}
}

// BenchmarkSplitRadixVsIncumbents compares the forced split-radix strategy
// against forced DIT/Stockham and the auto-selected plan (codelets on SIMD
// builds). Run with -tags purego for the scalar comparison that motivated
// the kernel (PLAN.md P4.1).
func BenchmarkSplitRadixVsIncumbents(b *testing.B) {
	run := func(b *testing.B, n int, opts PlanOptions) {
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

	for _, n := range []int{64, 256, 1024, 4096, 16384, 65536, 262144} {
		b.Run("Auto_"+strconv.Itoa(n), func(b *testing.B) { run(b, n, PlanOptions{}) })
		b.Run("SplitRadix_"+strconv.Itoa(n), func(b *testing.B) { run(b, n, PlanOptions{Strategy: KernelSplitRadix}) })
		b.Run("DIT_"+strconv.Itoa(n), func(b *testing.B) { run(b, n, PlanOptions{Strategy: KernelDIT}) })
		b.Run("Stockham_"+strconv.Itoa(n), func(b *testing.B) { run(b, n, PlanOptions{Strategy: KernelStockham}) })
	}
}
