//go:build !race

package algofft

import "testing"

// TestMixedRadix7And11_ZeroAlloc checks the transform hot path stays
// allocation-free for lengths with factors 7 and 11 (radix-7/11 stages plus
// pow2/3/5 stages, both precisions).
func TestMixedRadix7And11_ZeroAlloc(t *testing.T) {
	for _, n := range []int{77, 448, 704, 1344} {
		plan, err := NewPlan[complex64](n)
		if err != nil {
			t.Fatalf("NewPlan[complex64](%d) failed: %v", n, err)
		}

		src := randomComplex64(n, 3)
		dst := make([]complex64, n)

		// Warm up pooled scratch and the mixed-radix schedule pool.
		_ = plan.Forward(dst, src)
		_ = plan.Inverse(dst, src)

		if allocs := testing.AllocsPerRun(100, func() {
			_ = plan.Forward(dst, src)
			_ = plan.Inverse(dst, src)
		}); allocs != 0 {
			t.Errorf("n=%d complex64: transforms allocate %v times per run, want 0", n, allocs)
		}

		plan128, err := NewPlan[complex128](n)
		if err != nil {
			t.Fatalf("NewPlan128(%d) failed: %v", n, err)
		}

		src128 := randomComplex128(n, 5)
		dst128 := make([]complex128, n)

		_ = plan128.Forward(dst128, src128)
		_ = plan128.Inverse(dst128, src128)

		if allocs := testing.AllocsPerRun(100, func() {
			_ = plan128.Forward(dst128, src128)
			_ = plan128.Inverse(dst128, src128)
		}); allocs != 0 {
			t.Errorf("n=%d complex128: transforms allocate %v times per run, want 0", n, allocs)
		}
	}
}
