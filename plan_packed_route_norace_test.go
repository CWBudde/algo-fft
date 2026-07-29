//go:build !race

package algofft

import (
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/transform"
)

// TestPackedRouteZeroAlloc extends the zero-allocation guarantee (PLAN.md 2.1
// rule 3) to the packed route. No existing guard in plan_alloc_test.go reaches
// a Stockham-resolved size at all, so without this the newly-reachable path
// would be unguarded.
//
// Excluded from the race build for the same reason as the other _norace
// guards: race instrumentation defeats the pooled scratch and AllocsPerRun then
// reports allocations the normal build does not make.
func TestPackedRouteZeroAlloc(t *testing.T) {
	withPackedOverride(t, transform.PackedOverrideForceOn)

	// 2^17 is the first size an auto plan resolves to Stockham on AVX2 (the
	// codelet ladder now reaches 65536); 8192 covers the forced-Stockham case
	// at a codelet-covered size, which the packed route is also reachable at.
	for _, n := range []int{8192, 1 << 17} {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			packedZeroAlloc[complex64](t, n)
			packedZeroAlloc[complex128](t, n)
		})
	}
}

func packedZeroAlloc[T Complex](t *testing.T, n int) {
	t.Helper()

	plan, err := NewPlanWithOptions[T](n, PlanOptions{Strategy: KernelStockham})
	if err != nil {
		t.Fatalf("NewPlan(%d) failed: %v", n, err)
	}

	if packedOf(t, plan) == nil {
		t.Fatalf("n=%d %T: packed route not bound; this would guard the wrong path", n, *new(T))
	}

	src := make([]T, n)
	for i, v := range packedBroadband(n) {
		src[i] = T(complex128(v))
	}

	dst := make([]T, n)

	// Warm the pooled scratch before counting.
	_ = plan.Forward(dst, src)
	_ = plan.Inverse(dst, src)

	if allocs := testing.AllocsPerRun(20, func() {
		_ = plan.Forward(dst, src)
		_ = plan.Inverse(dst, src)
	}); allocs != 0 {
		t.Errorf("n=%d %T: transforms allocate %v per run, want 0", n, *new(T), allocs)
	}
}
