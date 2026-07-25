package algofft

import (
	"fmt"
	"math/rand"
	"testing"
)

// autoSquareSizes are the power-of-two square sizes the auto rule in
// internal/planner.resolveKernelStrategy treats specially, plus the eight-step
// boundary above them. 2^18 and 2^20 are the only power-of-two squares in
// [2^18, 2^22), i.e. the only sizes the split-radix branch of the rule can
// ever reach; 2^22 is the first size the eight-step branch takes.
//
//nolint:gochecknoglobals // benchmark input table
var autoSquareSizes = []struct {
	n     int
	arms  []KernelStrategy
	wide  bool // also measure complex128
	label string
}{
	{
		n:     1 << 18,
		arms:  []KernelStrategy{KernelSplitRadix, KernelStockham, KernelSixStep, KernelFourStep},
		wide:  true,
		label: "512x512",
	},
	{
		n:     1 << 20,
		arms:  []KernelStrategy{KernelSplitRadix, KernelStockham, KernelSixStep, KernelFourStep},
		wide:  true,
		label: "1024x1024",
	},
	{
		n:     1 << 22,
		arms:  []KernelStrategy{KernelEightStep, KernelStockham, KernelSixStep, KernelFourStep, KernelSplitRadix},
		wide:  false, // 2^22 complex128 needs 64 MiB per buffer; complex64 settles the ordering
		label: "2048x2048",
	},
}

// BenchmarkSquareAutoRule measures every strategy the auto rule could pick for
// a power-of-two square, so the rule can be re-derived from numbers rather
// than inherited. PLAN.md P4.3 recorded that plain Stockham had overtaken
// split-radix at these sizes after the cache-blocked transpose landed, which
// contradicts the live rule (2^18/2^20 -> split-radix).
//
// The arms for one size run adjacent in one process so thermal drift biases
// them equally — on this laptop an all-A-then-all-B comparison does not
// reproduce (see the pad-model benchmark for the same reasoning). Run with
// -count=5 and take medians; run again with -tags purego, since the rule must
// hold on both builds.
func BenchmarkSquareAutoRule(b *testing.B) {
	for _, tc := range autoSquareSizes {
		for _, dir := range []string{"fwd", "inv"} {
			for _, arm := range tc.arms {
				name := tc.label + "/" + dir + "/" + arm.String()

				b.Run("complex64/"+name, func(b *testing.B) {
					benchSquareArm[complex64](b, tc.n, arm, dir == "fwd")
				})

				if !tc.wide {
					continue
				}

				b.Run("complex128/"+name, func(b *testing.B) {
					benchSquareArm[complex128](b, tc.n, arm, dir == "fwd")
				})
			}
		}
	}
}

// squareArmCache holds one plan and one input/output buffer pair per
// (precision, size, strategy). Go re-invokes a benchmark function once per
// N-attempt, and plan construction at these sizes costs far more than the
// transform being timed — building per invocation made plan setup ~95% of the
// wall time and left the run thermally soaked before it finished. Benchmarks
// run sequentially, so a plain map needs no locking.
//
//nolint:gochecknoglobals // benchmark fixture cache
var squareArmCache = map[string]any{}

type squareArmFixture[T Complex] struct {
	plan     *Plan[T]
	src, dst []T
}

func squareArm[T Complex](b *testing.B, n int, arm KernelStrategy) *squareArmFixture[T] {
	b.Helper()

	var zero T

	key := fmt.Sprintf("%T/%d/%v", zero, n, arm)
	if hit, ok := squareArmCache[key]; ok {
		fixture, ok := hit.(*squareArmFixture[T])
		if !ok {
			b.Fatalf("cache key %q holds %T", key, hit)
		}

		return fixture
	}

	plan, err := NewPlanWithOptions[T](n, PlanOptions{Strategy: arm})
	if err != nil {
		b.Fatalf("NewPlan(%d, %v) failed: %v", n, arm, err)
	}

	if got := plan.KernelStrategy(); got != arm {
		b.Fatalf("n=%d: strategy = %v, want %v (forcing did not take)", n, got, arm)
	}

	// Random input, not a short repeating pattern: a periodic input has an
	// almost entirely zero spectrum, so the arms end up timing denormal
	// arithmetic instead of the transform, which differs per strategy and
	// per direction. Seeding by size keeps the arms comparable.
	rng := rand.New(rand.NewSource(int64(n))) //nolint:gosec // deterministic benchmark data

	src := make([]T, n)
	for i := range src {
		src[i] = T(complex(rng.Float64()*2-1, rng.Float64()*2-1))
	}

	fixture := &squareArmFixture[T]{plan: plan, src: src, dst: make([]T, n)}
	squareArmCache[key] = fixture

	return fixture
}

func benchSquareArm[T Complex](b *testing.B, n int, arm KernelStrategy, forward bool) {
	b.Helper()

	fixture := squareArm[T](b, n, arm)
	plan, src, dst := fixture.plan, fixture.src, fixture.dst

	run := plan.Forward
	if !forward {
		run = plan.Inverse
	}

	if err := run(dst, src); err != nil {
		b.Fatalf("warm-up failed: %v", err)
	}

	b.ReportAllocs()
	b.SetBytes(int64(n) * int64(complexWidth[T]()))
	b.ResetTimer()

	for range b.N {
		_ = run(dst, src)
	}
}
