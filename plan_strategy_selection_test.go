package algofft

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/fft"
)

// TestKernelSelectionStrategy verifies that the auto/forced distinction
// survives estimate pre-resolution: heuristic-derived strategies map back to
// KernelAuto (so the AVX-512 dispatch tier may substitute faster kernels),
// while explicit forces and wisdom-style overrides pass through unchanged.
func TestKernelSelectionStrategy(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name      string
		n         int
		requested KernelStrategy
		estimated KernelStrategy
		want      KernelStrategy
	}{
		// Auto plans whose estimate equals the size heuristic keep KernelAuto.
		{"AutoSmallDIT", 512, KernelAuto, fft.ResolveKernelStrategy(512), KernelAuto},
		{"AutoLargeStockham", 4096, KernelAuto, fft.ResolveKernelStrategy(4096), KernelAuto},
		// Explicit forces are always passed through.
		{"ForcedStockham", 4096, KernelStockham, KernelStockham, KernelStockham},
		{"ForcedDIT", 4096, KernelDIT, KernelDIT, KernelDIT},
		{"ForcedStockhamSmall", 512, KernelStockham, KernelStockham, KernelStockham},
		// Auto plans with a wisdom/measurement override (estimate deviates
		// from the heuristic) keep the override.
		{"WisdomDITAtLargeSize", 4096, KernelAuto, KernelDIT, KernelDIT},
		{"WisdomStockhamAtSmallSize", 512, KernelAuto, KernelStockham, KernelStockham},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if got := kernelSelectionStrategy(tc.n, tc.requested, tc.estimated); got != tc.want {
				t.Errorf("kernelSelectionStrategy(%d, %v, %v) = %v, want %v",
					tc.n, tc.requested, tc.estimated, got, tc.want)
			}
		})
	}
}
