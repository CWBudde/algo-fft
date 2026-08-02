package planner

import "testing"

// TestStrategyAlgorithmNameRoundTrip pins the single strategy↔algorithm-name
// table: every concrete strategy maps to a distinct name, and the reverse
// lookup returns the original strategy.
func TestStrategyAlgorithmNameRoundTrip(t *testing.T) {
	t.Parallel()

	strategies := []KernelStrategy{
		KernelDIT, KernelStockham, KernelSixStep,
		KernelBluestein, KernelSplitRadix, KernelRecursive,
		KernelFourStep, KernelMixedRadix,
	}

	seen := make(map[string]KernelStrategy)

	for _, s := range strategies {
		name := StrategyToAlgorithmName(s)
		if name == "unknown" || name == "" {
			t.Errorf("StrategyToAlgorithmName(%v) = %q, want a concrete name", s, name)
			continue
		}

		if prev, dup := seen[name]; dup {
			t.Errorf("name %q maps to both %v and %v", name, prev, s)
		}

		seen[name] = s

		back, ok := AlgorithmNameToStrategy(name)
		if !ok || back != s {
			t.Errorf("AlgorithmNameToStrategy(%q) = %v, %v; want %v, true", name, back, ok, s)
		}
	}
}

// TestAlgorithmNameToStrategy_Unknown checks that unknown names are rejected
// rather than mapped to a strategy.
func TestAlgorithmNameToStrategy_Unknown(t *testing.T) {
	t.Parallel()

	for _, name := range []string{"unknown", "", "dit8_avx2", "nonsense"} {
		if s, ok := AlgorithmNameToStrategy(name); ok {
			t.Errorf("AlgorithmNameToStrategy(%q) = %v, true; want ok=false", name, s)
		}
	}
}

// TestStrategyToAlgorithmName_Recursive pins the explicit recursive entry
// (previously fell through to "unknown").
func TestStrategyToAlgorithmName_Recursive(t *testing.T) {
	t.Parallel()

	if got := StrategyToAlgorithmName(KernelRecursive); got != "recursive" {
		t.Errorf("StrategyToAlgorithmName(KernelRecursive) = %q, want %q", got, "recursive")
	}
}
