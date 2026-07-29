package algofft

import (
	"math"
	"math/cmplx"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestFourStepStrategyPlan verifies a forced four-step plan resolves to the
// fourstep algorithm and produces the same spectrum as an auto plan,
// including at non-square power-of-two sizes six-step declines.
func TestFourStepStrategyPlan(t *testing.T) {
	t.Parallel()

	for _, n := range []int{1024, 8192} {
		plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: KernelFourStep})
		if err != nil {
			t.Fatalf("NewPlanWithOptions(%d, FourStep): %v", n, err)
		}

		if got := plan.KernelStrategy(); got != KernelFourStep {
			t.Errorf("n=%d: KernelStrategy() = %v, want KernelFourStep", n, got)
		}

		if got := plan.Algorithm(); got != "fourstep" {
			t.Errorf("n=%d: Algorithm() = %q, want \"fourstep\"", n, got)
		}

		auto, err := NewPlan[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan(%d): %v", n, err)
		}

		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i%31)/31, float32(i%17)/17)
		}

		got := make([]complex64, n)
		want := make([]complex64, n)

		if err := plan.Forward(got, src); err != nil {
			t.Fatalf("n=%d: fourstep Forward: %v", n, err)
		}

		if err := auto.Forward(want, src); err != nil {
			t.Fatalf("n=%d: auto Forward: %v", n, err)
		}

		for i := range want {
			diff := cmplx.Abs(complex128(got[i]) - complex128(want[i]))
			if diff > 1e-2 {
				t.Fatalf("n=%d: bin %d differs by %g: fourstep %v, auto %v",
					n, i, diff, got[i], want[i])
			}
		}
	}
}

// TestFourStepStrategyComplex128 covers the generated complex128 twin through
// the public API.
func TestFourStepStrategyComplex128(t *testing.T) {
	t.Parallel()

	const n = 4096

	plan, err := NewPlanWithOptions[complex128](n, PlanOptions{Strategy: KernelFourStep})
	if err != nil {
		t.Fatalf("NewPlanWithOptions: %v", err)
	}

	if got := plan.Algorithm(); got != "fourstep" {
		t.Errorf("Algorithm() = %q, want \"fourstep\"", got)
	}

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i%29)/29, float64(i%13)/13)
	}

	fwd := make([]complex128, n)
	back := make([]complex128, n)

	if err := plan.Forward(fwd, src); err != nil {
		t.Fatalf("Forward: %v", err)
	}

	if err := plan.Inverse(back, fwd); err != nil {
		t.Fatalf("Inverse: %v", err)
	}

	for i := range src {
		if diff := cmplx.Abs(back[i] - src[i]); diff > 1e-9 {
			t.Fatalf("round trip bin %d differs by %g", i, diff)
		}
	}
}

// TestFourStepStrategyFallsBack verifies unsupported lengths still produce a
// working plan: like forced six-step on non-square sizes, the dispatch layer
// reroutes to a supported kernel at transform time.
func TestFourStepStrategyFallsBack(t *testing.T) {
	t.Parallel()

	for _, n := range []int{2, 1000} {
		plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: KernelFourStep})
		if err != nil {
			t.Fatalf("NewPlanWithOptions(%d): %v", n, err)
		}

		// Broadband, not an impulse: an impulse transforms to an all-ones
		// spectrum regardless of kernel, so it cannot tell a correct fallback
		// route from one with wrong twiddles or a wrong bin order.
		src := make([]complex64, n)
		for i := range src {
			f := float64(i)
			src[i] = complex(
				float32(math.Cos(0.7*f)+0.3*math.Sin(2.9*f)),
				float32(math.Sin(1.3*f)-0.4),
			)
		}

		dst := make([]complex64, n)
		if err := plan.Forward(dst, src); err != nil {
			t.Fatalf("n=%d: Forward after fallback: %v", n, err)
		}

		want := reference.NaiveDFTWide(src)

		var peak float64
		for _, v := range want {
			peak = math.Max(peak, cmplx.Abs(v))
		}

		tol := 1e-5 * peak
		for i, v := range dst {
			if diff := cmplx.Abs(complex128(v) - want[i]); diff > tol {
				t.Fatalf("n=%d algo=%s: bin %d = %v, want %v (diff %.3e > %.3e)",
					n, plan.Algorithm(), i, v, want[i], diff, tol)
			}
		}
	}
}

// BenchmarkFourStepVsIncumbents compares four-step against the strategies
// currently winning the above-L2 range (split-radix for power-of-two sizes,
// six-step for squares, Stockham as baseline). Sizes 2^18 ... 2^23 are where
// the cache-blocked split targets; odd exponents have no six-step entry.
func BenchmarkFourStepVsIncumbents(b *testing.B) {
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

	for _, n := range []int{1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22, 1 << 23} {
		b.Run("FourStep_"+strconv.Itoa(n), func(b *testing.B) { run(b, n, PlanOptions{Strategy: KernelFourStep}) })
		b.Run("SplitRadix_"+strconv.Itoa(n), func(b *testing.B) { run(b, n, PlanOptions{Strategy: KernelSplitRadix}) })
		b.Run("Stockham_"+strconv.Itoa(n), func(b *testing.B) { run(b, n, PlanOptions{Strategy: KernelStockham}) })

		if m := intSqrtPlanTest(n); m*m == n {
			b.Run("SixStep_"+strconv.Itoa(n), func(b *testing.B) { run(b, n, PlanOptions{Strategy: KernelSixStep}) })
		}
	}
}

// intSqrtPlanTest is a minimal integer square root for benchmark gating.
func intSqrtPlanTest(n int) int {
	root := 0
	for (root+1)*(root+1) <= n {
		root++
	}

	return root
}

func TestFourStepStrategyString(t *testing.T) {
	t.Parallel()

	if got := KernelFourStep.String(); got != "FourStep" {
		t.Errorf("KernelFourStep.String() = %q, want \"FourStep\"", got)
	}
}
