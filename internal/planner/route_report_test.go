package planner

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// fixedWisdom answers every lookup with the same algorithm name, standing in
// for a wisdom file recorded before non-power-of-two lengths were named
// honestly.
type fixedWisdom struct{ algorithm string }

func (w fixedWisdom) LookupWisdom(int, uint8, uint64) (string, bool) {
	return w.algorithm, true
}

// TestEstimateReportsMixedRadixRoute pins the estimate for the smooth
// non-powers of two: the mixed-radix engine is the only route the kernel
// dispatch can take for them, so it is the only strategy an estimate may name.
func TestEstimateReportsMixedRadixRoute(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	for _, n := range []int{1000, 2205, 3600, 12000, 44100} {
		est := EstimatePlan[complex64](n, features, nil, KernelAuto)
		if est.Strategy != KernelMixedRadix {
			t.Errorf("EstimatePlan(%d, auto).Strategy = %v, want MixedRadix", n, est.Strategy)
		}

		if est.Algorithm != algoMixedRadix {
			t.Errorf("EstimatePlan(%d, auto).Algorithm = %q, want %q", n, est.Algorithm, algoMixedRadix)
		}
	}
}

// TestStaleStrategyWisdomDoesNotRelabel checks that a wisdom entry recorded
// under the old naming ("stockham" / "dit_fallback" at a mixed-radix length)
// resolves to the route that runs rather than resurrecting the label. Those
// entries are not wrong about a measurement — every candidate they compared
// executed the same mixed-radix transform — only about its name.
func TestStaleStrategyWisdomDoesNotRelabel(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	for _, name := range []string{algoStockham, algoDITFallback, algoSixStep} {
		est := EstimatePlan[complex64](1000, features, fixedWisdom{name}, KernelAuto)
		if est.Strategy != KernelMixedRadix {
			t.Errorf("wisdom %q at n=1000: Strategy = %v, want MixedRadix", name, est.Strategy)
		}
	}
}

// TestForcedMixedRadixAtPowerOfTwoFallsBack is the reverse guard: the engine
// is not the route for a power of two, so forcing it must not name it.
func TestForcedMixedRadixAtPowerOfTwoFallsBack(t *testing.T) {
	t.Parallel()

	for _, n := range []int{512, 4096} {
		if got := ResolveKernelStrategyWithDefault(n, KernelMixedRadix); got == KernelMixedRadix {
			t.Errorf("ResolveKernelStrategyWithDefault(%d, MixedRadix) = %v, want a power-of-two strategy", n, got)
		}
	}
}
