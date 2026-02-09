package planner

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// TestEstimatePlanPowerOf2 tests EstimatePlan with power-of-2 sizes.
// Note: Avoiding hard expectations on specific strategies due to EstimatePlan's
// complex logic with codelet registry and wisdom cache. Instead, test observable behavior.
func TestEstimatePlanPowerOf2(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		size int
	}{
		{"Size 8", 8},
		{"Size 16", 16},
		{"Size 1024", 1024},
		{"Size 2048", 2048},
		{"Size 4096", 4096},
		{"Size 65536", 65536},
	}

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			estimate := EstimatePlan[complex64](tt.size, features, nil, KernelAuto)

			// Verify the estimate is non-zero and has valid strategy
			if estimate.Strategy == 0 {
				t.Errorf("EstimatePlan(%d) returned zero strategy", tt.size)
			}

			if estimate.Algorithm == "" {
				t.Errorf("EstimatePlan(%d) returned empty algorithm name", tt.size)
			}

			// For forced strategy, ensure it respects the force
			forcedEstimate := EstimatePlan[complex64](tt.size, features, nil, KernelDIT)
			if forcedEstimate.Strategy != KernelDIT {
				t.Errorf("EstimatePlan(%d, forced=DIT) strategy = %v, want KernelDIT", tt.size, forcedEstimate.Strategy)
			}
		})
	}
}

// TestEstimatePlanWithForcedStrategy tests EstimatePlan with forced kernel strategy.
func TestEstimatePlanWithForcedStrategy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		size          int
		forcedStrat   KernelStrategy
		expectedStrat KernelStrategy
	}{
		{"Force DIT on 2048", 2048, KernelDIT, KernelDIT},
		{"Force Stockham on 256", 256, KernelStockham, KernelStockham},
		{"Force Stockham on 2048", 2048, KernelStockham, KernelStockham},
	}

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			estimate := EstimatePlan[complex64](tt.size, features, nil, tt.forcedStrat)

			if estimate.Strategy != tt.expectedStrat {
				t.Errorf("EstimatePlan(%d, forced=%v) strategy = %v, want %v",
					tt.size, tt.forcedStrat, estimate.Strategy, tt.expectedStrat)
			}
		})
	}
}

// TestEstimatePlanNonPowerOf2 tests EstimatePlan with non-power-of-2 sizes.
func TestEstimatePlanNonPowerOf2(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name            string
		size            int
		expectBluestein bool
	}{
		{"Size 1000 (highly composite)", 1000, false}, // 2³ × 5³ - not bluestein
		{"Size 1500 (highly composite)", 1500, false}, // 2² × 3 × 5³ - not bluestein
		{"Size 3072 (highly composite)", 3072, false}, // 2¹⁰ × 3 - not bluestein
		{"Size 1001 (not composite)", 1001, true},     // 7 × 11 × 13 - bluestein required
	}

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			estimate := EstimatePlan[complex64](tt.size, features, nil, KernelAuto)

			if tt.expectBluestein {
				if estimate.Algorithm != "bluestein" {
					t.Errorf("EstimatePlan(%d) algorithm = %q, want \"bluestein\" for non-5-smooth number",
						tt.size, estimate.Algorithm)
				}
			} else {
				// Highly composite numbers use fallback strategies, not bluestein
				if estimate.Algorithm == "bluestein" {
					t.Errorf("EstimatePlan(%d) algorithm = \"bluestein\", but %d is 5-smooth",
						tt.size, tt.size)
				}
			}
		})
	}
}

// TestEstimatePlanComplex128 tests EstimatePlan with complex128 precision.
func TestEstimatePlanComplex128(t *testing.T) {
	t.Parallel()

	features := cpu.Features{
		Architecture: "amd64",
		HasAVX2:      true,
	}

	estimate := EstimatePlan[complex128](1024, features, nil, KernelAuto)

	if estimate.Algorithm != "dit_fallback" {
		t.Errorf("EstimatePlan[complex128](1024) algorithm = %q, want \"dit_fallback\"", estimate.Algorithm)
	}
}

// TestEstimatePlanWithWisdom tests EstimatePlan with wisdom cache fallback.
func TestEstimatePlanWithWisdom(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()
	wisdom.Store(WisdomEntry{
		Key: WisdomKey{
			Size:        512,
			Precision:   0,
			CPUFeatures: CPUFeatureMask(true, true, false, false),
		},
		Algorithm: "stockham",
	})

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
		HasAVX2:      true,
	}

	estimate := EstimatePlan[complex64](512, features, wisdom, KernelAuto)

	// Wisdom recommends stockham for size 512
	if estimate.Strategy != KernelStockham {
		t.Errorf("EstimatePlan with wisdom: strategy = %v, want KernelStockham", estimate.Strategy)
	}
}

// TestEstimatePlanWisdomOverriddenByForce tests that forced strategy overrides wisdom.
func TestEstimatePlanWisdomOverriddenByForce(t *testing.T) {
	t.Parallel()

	wisdom := NewWisdom()
	wisdom.Store(WisdomEntry{
		Key: WisdomKey{
			Size:        512,
			Precision:   0,
			CPUFeatures: CPUFeatureMask(true, true, false, false),
		},
		Algorithm: "stockham",
	})

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
		HasAVX2:      true,
	}

	estimate := EstimatePlan[complex64](512, features, wisdom, KernelDIT)

	// Forced strategy should override wisdom
	if estimate.Strategy != KernelDIT {
		t.Errorf("EstimatePlan forced override: strategy = %v, want KernelDIT", estimate.Strategy)
	}
}

// TestHasCodelet tests the HasCodelet function.
func TestHasCodelet(t *testing.T) {
	t.Parallel()

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	// Initially no codelets registered
	has := HasCodelet[complex64](256, features)
	if has {
		t.Error("HasCodelet should return false when no codelets registered")
	}
}
