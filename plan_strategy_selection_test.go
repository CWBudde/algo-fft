package algofft

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/planner"
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
		requested fftypes.KernelStrategy
		estimated fftypes.KernelStrategy
		want      fftypes.KernelStrategy
	}{
		// Auto plans whose estimate equals the size heuristic keep KernelAuto.
		{"AutoSmallDIT", 512, fftypes.KernelAuto, planner.ResolveKernelStrategy(512), fftypes.KernelAuto},
		{"AutoLargeStockham", 4096, fftypes.KernelAuto, planner.ResolveKernelStrategy(4096), fftypes.KernelAuto},
		// Explicit forces are always passed through.
		{"ForcedStockham", 4096, fftypes.KernelStockham, fftypes.KernelStockham, fftypes.KernelStockham},
		{"ForcedDIT", 4096, fftypes.KernelDIT, fftypes.KernelDIT, fftypes.KernelDIT},
		{"ForcedStockhamSmall", 512, fftypes.KernelStockham, fftypes.KernelStockham, fftypes.KernelStockham},
		// Auto plans with a wisdom/measurement override (estimate deviates
		// from the heuristic) keep the override.
		{"WisdomDITAtLargeSize", 4096, fftypes.KernelAuto, fftypes.KernelDIT, fftypes.KernelDIT},
		{"WisdomStockhamAtSmallSize", 512, fftypes.KernelAuto, fftypes.KernelStockham, fftypes.KernelStockham},
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
