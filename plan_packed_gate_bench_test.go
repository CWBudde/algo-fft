package algofft

import (
	"fmt"
	"math/rand"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/transform"
)

// Crossover sweep for the packed radix-4 Stockham gate (PLAN.md 3).
//
// Both arms run in ONE binary, selected by the runtime override, rather than
// comparing a default build against -tags purego. That comparison would
// confound the routing change with every other difference between the builds,
// and code layout alone has moved results here by more than the effect being
// chased (PLAN.md 2.2).
//
// Cells are named prec/size/arm/dir to match BenchmarkCodeletCandidates, so the
// existing gated-sweep analyzer parses them unchanged.
//
// Arms:
//
//	packed  - override ForceOn: the candidate route
//	kernel  - override ForceOff: the incumbent (today's SIMD behaviour)
//	nullsix - forced KernelSixStep, which the override cannot reach at all
//
// nullsix is the control. A codelet-covered size under *forced* Stockham would
// NOT be one: forcing a strategy makes tryRegistry decline the codelet, so the
// packed table is allocated there too and the change does reach it.
//
// Memory: plans and buffers are cached per (precision, size, arm) because plan
// construction dominates at these sizes. At 2^22 complex128 one arm holds
// src+dst+twiddle+scratch+packed ~= 320 MiB, so run one size per invocation
// (and one precision at 2^21/2^22):
//
//	go test -run='^$' -bench='BenchmarkPackedGate64/size1048576' -benchtime=0.3s .
//
// Interleave rounds and rotate arm order per round; take medians of the
// within-round packed/kernel ratio. Reject the run if nullsix moves >3%.

//nolint:gochecknoglobals // benchmark size table
var packedGateSizes = []int{1 << 16, 1 << 17, 1 << 18, 1 << 19, 1 << 20, 1 << 21, 1 << 22}

func BenchmarkPackedGate64(b *testing.B) { benchmarkPackedGate[complex64](b) }

func BenchmarkPackedGate128(b *testing.B) { benchmarkPackedGate[complex128](b) }

func benchmarkPackedGate[T Complex](b *testing.B) {
	b.Helper()

	for _, n := range packedGateSizes {
		label := "size" + strconv.Itoa(n)

		for _, arm := range []string{"packed", "kernel", "nullsix"} {
			for _, dir := range []string{"forward", "inverse"} {
				b.Run(label+"/"+arm+"/"+dir, func(b *testing.B) {
					benchPackedGateArm[T](b, n, arm, dir == "forward")
				})
			}
		}
	}
}

// packedGateCache holds one plan and buffer pair per (precision, size, arm).
// Go re-invokes a benchmark function once per N-attempt and plan construction
// at these sizes costs far more than the transform being timed. The arms cannot
// share a plan: the override is read at construction.
//
//nolint:gochecknoglobals // benchmark fixture cache
var packedGateCache = map[string]any{}

type packedGateFixture[T Complex] struct {
	plan     *Plan[T]
	src, dst []T
}

func packedGateArm[T Complex](b *testing.B, n int, arm string) *packedGateFixture[T] {
	b.Helper()

	var zero T

	key := fmt.Sprintf("%T/%d/%s", zero, n, arm)
	if hit, ok := packedGateCache[key]; ok {
		fixture, ok := hit.(*packedGateFixture[T])
		if !ok {
			b.Fatalf("cache key %q holds %T", key, hit)
		}

		return fixture
	}

	opts := PlanOptions{Strategy: KernelStockham}
	mode := transform.PackedOverrideForceOff

	switch arm {
	case "packed":
		mode = transform.PackedOverrideForceOn
	case "kernel":
	case "nullsix":
		opts = PlanOptions{Strategy: KernelSixStep}
	default:
		b.Fatalf("unknown arm %q", arm)
	}

	transform.SetPackedStockhamOverride(mode)
	defer transform.SetPackedStockhamOverride(transform.PackedOverrideDefault)

	plan, err := NewPlanWithOptions[T](n, opts)
	if err != nil {
		b.Fatalf("NewPlan(%d, %v) failed: %v", n, opts.Strategy, err)
	}

	// Two assertions, because either one alone can pass while the benchmark
	// measures the wrong thing. The strategy check catches a silent
	// re-resolution; the packed check catches a mis-wired override, which would
	// otherwise time identical code twice and report "no difference".
	if arm != "nullsix" {
		if got := plan.KernelStrategy(); got != KernelStockham {
			b.Fatalf("n=%d: strategy = %v, want Stockham (forcing did not take)", n, got)
		}

		if bound := packedBoundForBench(b, plan); bound != (arm == "packed") {
			b.Fatalf("n=%d arm=%s: packed bound = %v, want %v", n, arm, bound, arm == "packed")
		}
	}

	// Random input, not a repeating pattern: a periodic input has an almost
	// entirely zero spectrum, so the arms would time denormal arithmetic
	// instead of the transform.
	rng := rand.New(rand.NewSource(int64(n))) //nolint:gosec // deterministic benchmark data

	src := make([]T, n)
	for i := range src {
		src[i] = T(complex(rng.Float64()*2-1, rng.Float64()*2-1))
	}

	fixture := &packedGateFixture[T]{plan: plan, src: src, dst: make([]T, n)}
	packedGateCache[key] = fixture

	return fixture
}

func packedBoundForBench[T Complex](b *testing.B, plan *Plan[T]) bool {
	b.Helper()

	exec, ok := plan.exec.(*kernelExecutor[T])
	if !ok {
		b.Fatalf("executor is %T, want *kernelExecutor", plan.exec)
	}

	return exec.packed != nil
}

func benchPackedGateArm[T Complex](b *testing.B, n int, arm string, forward bool) {
	b.Helper()

	fixture := packedGateArm[T](b, n, arm)
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
