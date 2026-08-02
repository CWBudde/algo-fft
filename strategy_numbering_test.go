package algofft

import "testing"

// TestKernelStrategyNumbering pins the exact numeric value of every public
// KernelStrategy constant. The constants are declared with explicit values
// (not iota) specifically so that this can be a gate rather than an
// accident: a caller may have persisted a KernelStrategy value, and
// renumbering it silently would change what that persisted value means.
//
// Value 4 is a deliberate gap: it used to be KernelEightStep, removed
// 2026-08-02 because internal/kernels/eightstep.go was a byte-for-byte
// duplicate of internal/kernels/sixstep.go with the names changed (no
// eighth step anywhere in it). The value is retired rather than reused so
// every other constant keeps the number it already had.
func TestKernelStrategyNumbering(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		strategy KernelStrategy
		want     uint8
	}{
		{"KernelAuto", KernelAuto, 0},
		{"KernelDIT", KernelDIT, 1},
		{"KernelStockham", KernelStockham, 2},
		{"KernelSixStep", KernelSixStep, 3},
		// 4 is deliberately absent: it was KernelEightStep.
		{"KernelBluestein", KernelBluestein, 5},
		{"KernelRecursive", KernelRecursive, 6},
		{"KernelSplitRadix", KernelSplitRadix, 7},
		{"KernelFourStep", KernelFourStep, 8},
		{"KernelMixedRadix", KernelMixedRadix, 9},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := uint8(tt.strategy); got != tt.want {
				t.Errorf("%s = %d, want %d", tt.name, got, tt.want)
			}
		})
	}
}
