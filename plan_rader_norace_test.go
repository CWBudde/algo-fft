//go:build !race

package algofft

import "testing"

// TestRader_ZeroAlloc checks the transform hot path stays allocation-free for
// the power-of-two (257), 5-smooth mixed-radix (641), and radix-7/11
// mixed-radix (353 -> [11, 32], 2269 -> [7, 4, 3, 3, 3, 3]) sub-FFT variants.
func TestRader_ZeroAlloc(t *testing.T) {
	for _, n := range []int{257, 641, 353, 2269} {
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
			t.Errorf("n=%d: transforms allocate %v times per run, want 0", n, allocs)
		}
	}
}
