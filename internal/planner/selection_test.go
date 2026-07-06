package planner

import (
	"testing"
)

// TestResolveKernelStrategyAuto tests the heuristic threshold behavior when no
// strategy is forced (KernelAuto).
func TestResolveKernelStrategyAuto(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		size int
		want KernelStrategy
	}{
		{"below threshold", 256, KernelDIT},
		{"at threshold", 1024, KernelDIT},
		{"above threshold", 2048, KernelStockham},
	}

	for _, tt := range tests {
		got := ResolveKernelStrategy(tt.size)
		if got != tt.want {
			t.Errorf("ResolveKernelStrategy(%d) = %v, want %v", tt.size, got, tt.want)
		}
	}
}

// TestResolveKernelStrategyForced tests resolution with a forced strategy passed
// as the default (mirrors PlanOptions.Strategy threading).
func TestResolveKernelStrategyForced(t *testing.T) {
	t.Parallel()

	strategies := []KernelStrategy{KernelDIT, KernelStockham}
	for _, strategy := range strategies {
		got := ResolveKernelStrategyWithDefault(512, strategy)
		if got != strategy {
			t.Errorf("ResolveKernelStrategyWithDefault(512, %v) = %v, want %v",
				strategy, got, strategy)
		}
	}
}

// TestResolveKernelStrategyWithDefault tests that a concrete default is honored and
// KernelAuto falls through to heuristics.
func TestResolveKernelStrategyWithDefault(t *testing.T) {
	t.Parallel()

	got := ResolveKernelStrategyWithDefault(512, KernelStockham)
	if got != KernelStockham {
		t.Errorf("ResolveKernelStrategyWithDefault(512, Stockham) = %v, want Stockham", got)
	}

	// KernelAuto default falls back to the size heuristic (512 <= threshold -> DIT).
	got = ResolveKernelStrategyWithDefault(512, KernelAuto)
	if got != KernelDIT {
		t.Errorf("ResolveKernelStrategyWithDefault(512, Auto) = %v, want DIT", got)
	}
}

// TestIntSqrt tests the integer square root function.
func TestIntSqrt(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want int
	}{
		{0, 0},
		{1, 1},
		{4, 2},
		{9, 3},
		{16, 4},
		{25, 5},
		{100, 10},
		{256, 16},
		{65536, 256},
		{10, 3},
		{15, 3},
		{17, 4},
	}

	for _, tt := range tests {
		got := intSqrt(tt.n)
		if got != tt.want {
			t.Errorf("intSqrt(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

// TestIsSquareSize tests the square size detection.
func TestIsSquareSize(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want bool
	}{
		{0, false},
		{1, true},
		{4, true},
		{9, true},
		{16, true},
		{100, true},
		{256, true},
		{65536, true},
		{10, false},
		{15, false},
		{17, false},
		{1024, true},    // 32 * 32
		{1048576, true}, // 1024 * 1024
	}

	for _, tt := range tests {
		got := isSquareSize(tt.n)
		if got != tt.want {
			t.Errorf("isSquareSize(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}

// TestFallbackKernelStrategy tests fallback strategy selection.
func TestFallbackKernelStrategy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		size int
		want KernelStrategy
	}{
		{"Size 512", 512, KernelDIT},
		{"Size 1024", 1024, KernelDIT},
		{"Size 2048", 2048, KernelStockham},
		{"Size 4096", 4096, KernelStockham},
	}

	for _, tt := range tests {
		got := fallbackKernelStrategy(tt.size)
		if got != tt.want {
			t.Errorf("fallbackKernelStrategy(%d) = %v, want %v", tt.size, got, tt.want)
		}
	}
}

// TestSixStepEightStepSquareSizes tests strategy selection for square sizes under
// the auto heuristic.
func TestSixStepEightStepSquareSizes(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		size int
		want KernelStrategy
	}{
		{"2048x2048", 2048 * 2048, KernelEightStep}, // 4194304 >= 1<<22 (4194304)? Yes
		{"512x512", 512 * 512, KernelSixStep},       // 262144 >= 1<<18 (262144)? Yes
		{"256x256", 256 * 256, KernelStockham},      // 65536 is square but < 1<<18
		{"32x32", 32 * 32, KernelDIT},               // 1024 is square but <= ditAutoThreshold
	}

	for _, tt := range tests {
		got := ResolveKernelStrategy(tt.size)
		if got != tt.want {
			t.Errorf("ResolveKernelStrategy(%d, square) = %v, want %v",
				tt.size, got, tt.want)
		}
	}
}

// TestForcedSixStepOnNonSquare tests that six/eight-step forced on a non-square size
// falls back to a size-appropriate strategy.
func TestForcedSixStepOnNonSquare(t *testing.T) {
	t.Parallel()

	// Non-square size forced to sixstep should fall back.
	got := ResolveKernelStrategyWithDefault(1000, KernelSixStep)
	if got == KernelSixStep || got == KernelEightStep {
		t.Errorf("ResolveKernelStrategyWithDefault(1000, SixStep) = %v, should fall back for non-square", got)
	}

	// Size <= ditAutoThreshold should fall back to DIT.
	got = ResolveKernelStrategyWithDefault(512, KernelSixStep)
	if got != KernelDIT {
		t.Errorf("ResolveKernelStrategyWithDefault(512, SixStep) = %v, want fallback DIT", got)
	}
}
