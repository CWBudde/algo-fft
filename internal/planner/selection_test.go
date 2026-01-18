package planner

import (
	"sync"
	"testing"
)

// TestSetGetKernelStrategy tests setting and getting kernel strategy.
//
//nolint:paralleltest
func TestSetGetKernelStrategy(t *testing.T) {
	originalStrategy := GetKernelStrategy()
	defer SetKernelStrategy(originalStrategy)

	tests := []KernelStrategy{
		KernelAuto,
		KernelDIT,
		KernelStockham,
		KernelSixStep,
		KernelEightStep,
	}

	for _, strategy := range tests {
		t.Run("strategy", func(t *testing.T) {
			SetKernelStrategy(strategy)

			got := GetKernelStrategy()
			if got != strategy {
				t.Errorf("SetKernelStrategy(%v); GetKernelStrategy() = %v, want %v",
					strategy, got, strategy)
			}
		})
	}
}

// TestResolveKernelStrategyAutoThreshold tests the threshold behavior for auto strategy.
//
//nolint:paralleltest
func TestResolveKernelStrategyAutoThreshold(t *testing.T) {
	originalStrategy := GetKernelStrategy()
	defer SetKernelStrategy(originalStrategy)

	SetKernelStrategy(KernelDIT)

	// When DIT is forced, sizes at threshold should return DIT
	ditResult := ResolveKernelStrategy(1024)
	if ditResult != KernelDIT {
		t.Errorf("ResolveKernelStrategy(1024) with forced DIT = %v, want KernelDIT", ditResult)
	}

	SetKernelStrategy(KernelStockham)

	// When Stockham is forced, all sizes should return Stockham
	stockhamResult := ResolveKernelStrategy(256)
	if stockhamResult != KernelStockham {
		t.Errorf("ResolveKernelStrategy(256) with forced Stockham = %v, want KernelStockham", stockhamResult)
	}
}

// TestResolveKernelStrategyForced tests resolution with forced strategy.
//
//nolint:paralleltest
func TestResolveKernelStrategyForced(t *testing.T) {
	originalStrategy := GetKernelStrategy()
	defer SetKernelStrategy(originalStrategy)

	strategies := []KernelStrategy{KernelDIT, KernelStockham}
	for _, strategy := range strategies {
		t.Run("forced_strategy", func(t *testing.T) {
			SetKernelStrategy(strategy)

			got := ResolveKernelStrategy(512)
			if got != strategy {
				t.Errorf("ResolveKernelStrategy(512) with forced %v = %v, want %v",
					strategy, got, strategy)
			}
		})
	}
}

// TestResolveKernelStrategyWithDefault tests resolution with default strategy.
//
//nolint:paralleltest
func TestResolveKernelStrategyWithDefault(t *testing.T) {
	originalStrategy := GetKernelStrategy()
	defer SetKernelStrategy(originalStrategy)

	SetKernelStrategy(KernelAuto)

	got := ResolveKernelStrategyWithDefault(512, KernelStockham)
	if got != KernelStockham {
		t.Errorf("ResolveKernelStrategyWithDefault(512, Stockham) = %v, want Stockham", got)
	}
}

// TestRecordBenchmarkDecision tests recording and retrieving benchmark decisions.
//
//nolint:paralleltest
func TestRecordBenchmarkDecision(t *testing.T) {
	originalDecisions := benchDecisions

	defer func() {
		benchMu.Lock()

		benchDecisions = originalDecisions

		benchMu.Unlock()
	}()

	benchMu.Lock()

	benchDecisions = make(map[int]KernelStrategy)

	benchMu.Unlock()

	originalStrategy := GetKernelStrategy()
	defer SetKernelStrategy(originalStrategy)

	SetKernelStrategy(KernelAuto)

	// Record a decision
	RecordBenchmarkDecision(512, KernelStockham)

	// Verify it's used
	got := ResolveKernelStrategy(512)
	if got != KernelStockham {
		t.Errorf("ResolveKernelStrategy(512) after recording decision = %v, want KernelStockham",
			got)
	}
}

// TestRecordBenchmarkDecisionInvalid tests that invalid decisions are ignored.
//
//nolint:paralleltest
func TestRecordBenchmarkDecisionInvalid(t *testing.T) {
	originalDecisions := benchDecisions

	defer func() {
		benchMu.Lock()

		benchDecisions = originalDecisions

		benchMu.Unlock()
	}()

	benchMu.Lock()

	benchDecisions = make(map[int]KernelStrategy)

	benchMu.Unlock()

	originalStrategy := GetKernelStrategy()
	defer SetKernelStrategy(originalStrategy)

	SetKernelStrategy(KernelAuto)

	// Record invalid decisions (should be ignored)
	RecordBenchmarkDecision(-1, KernelDIT)        // Negative size
	RecordBenchmarkDecision(512, KernelBluestein) // Invalid strategy

	benchMu.RLock()

	count := len(benchDecisions)

	benchMu.RUnlock()

	if count != 0 {
		t.Errorf("Expected no decisions recorded, got %d", count)
	}
}

// TestRecordBenchmarkDecisionZeroSize tests that zero size is ignored.
//
//nolint:paralleltest
func TestRecordBenchmarkDecisionZeroSize(t *testing.T) {
	originalDecisions := benchDecisions

	defer func() {
		benchMu.Lock()

		benchDecisions = originalDecisions

		benchMu.Unlock()
	}()

	benchMu.Lock()

	benchDecisions = make(map[int]KernelStrategy)

	benchMu.Unlock()

	RecordBenchmarkDecision(0, KernelDIT)

	benchMu.RLock()

	count := len(benchDecisions)

	benchMu.RUnlock()

	if count != 0 {
		t.Errorf("Expected no decisions for size 0, got %d", count)
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

// TestSixStepEightStepSquareSizes tests strategy selection for square sizes.
//
//nolint:paralleltest
func TestSixStepEightStepSquareSizes(t *testing.T) {
	originalStrategy := GetKernelStrategy()
	originalDecisions := benchDecisions

	defer func() {
		SetKernelStrategy(originalStrategy)
		benchMu.Lock()

		benchDecisions = originalDecisions

		benchMu.Unlock()
	}()

	SetKernelStrategy(KernelAuto)
	benchMu.Lock()

	benchDecisions = make(map[int]KernelStrategy)

	benchMu.Unlock()

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

// TestForcedSixStepOnNonSquare tests that six/eight-step forced on non-square falls back.
//
//nolint:paralleltest
func TestForcedSixStepOnNonSquare(t *testing.T) {
	originalStrategy := GetKernelStrategy()
	defer SetKernelStrategy(originalStrategy)

	SetKernelStrategy(KernelSixStep)

	// Non-square size forced to sixstep should fall back
	got := ResolveKernelStrategy(1000)
	if got == KernelSixStep || got == KernelEightStep {
		t.Errorf("ResolveKernelStrategy(1000, forced SixStep) = %v, should not be SixStep/EightStep for non-square",
			got)
	}

	// Size <= ditAutoThreshold should fall back to DIT
	got = ResolveKernelStrategy(512)
	if got != KernelDIT {
		t.Errorf("ResolveKernelStrategy(512, forced SixStep) = %v, want fallback DIT", got)
	}
}

// TestConcurrentBenchmarkDecisions tests concurrent recording of benchmark decisions.
//
//nolint:paralleltest
func TestConcurrentBenchmarkDecisions(t *testing.T) {
	originalDecisions := benchDecisions

	defer func() {
		benchMu.Lock()

		benchDecisions = originalDecisions

		benchMu.Unlock()
	}()

	benchMu.Lock()

	benchDecisions = make(map[int]KernelStrategy)

	benchMu.Unlock()

	originalStrategy := GetKernelStrategy()
	defer SetKernelStrategy(originalStrategy)

	SetKernelStrategy(KernelAuto)

	var waitGroup sync.WaitGroup

	strategies := []KernelStrategy{KernelDIT, KernelStockham}

	for i := range 100 {
		waitGroup.Add(1)

		go func(size int, strategy KernelStrategy) {
			defer waitGroup.Done()

			RecordBenchmarkDecision(size, strategy)
		}(256+i, strategies[i%2])
	}

	waitGroup.Wait()

	// Verify all decisions were recorded
	benchMu.RLock()

	if len(benchDecisions) != 100 {
		t.Errorf("Expected 100 decisions recorded, got %d", len(benchDecisions))
	}

	benchMu.RUnlock()
}

// TestConcurrentKernelStrategy tests concurrent access to kernel strategy.
//
//nolint:paralleltest
func TestConcurrentKernelStrategy(t *testing.T) {
	originalStrategy := GetKernelStrategy()
	defer SetKernelStrategy(originalStrategy)

	var waitGroup sync.WaitGroup

	const goroutines = 100

	for i := range goroutines {
		waitGroup.Add(1)

		go func(idx int) {
			defer waitGroup.Done()

			strategies := []KernelStrategy{KernelDIT, KernelStockham, KernelAuto}
			strategy := strategies[idx%len(strategies)]
			SetKernelStrategy(strategy)

			_ = GetKernelStrategy()
		}(i)
	}

	waitGroup.Wait()
}
