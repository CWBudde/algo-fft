//nolint:paralleltest
package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

// TestBenchmarkStrategyNeverReturnsZero verifies that benchmarkStrategy
// always returns a non-zero duration, even for very small FFT sizes.
// This test was added to ensure the cycle counter implementation fixes
// the Windows timer resolution issue that could cause zero durations.
func TestBenchmarkStrategyNeverReturnsZero(t *testing.T) {
	// Force generic implementation to avoid CPU feature dependencies
	cpu.SetForcedFeatures(cpu.Features{ForceGeneric: true})
	defer cpu.ResetDetection()

	features := cpu.DetectFeatures()
	config := measureConfig{warmup: 2, iters: 5}

	// Test very small sizes that previously could return zero on Windows
	sizes := []int{4, 8, 16, 32, 64}

	for _, n := range sizes {
		t.Run("Complex64", func(t *testing.T) {
			elapsed := benchmarkStrategy[complex64](n, features, KernelDIT, config)
			if elapsed == 0 {
				t.Errorf("benchmarkStrategy returned zero duration for size %d (complex64)", n)
			}
		})

		t.Run("Complex128", func(t *testing.T) {
			elapsed := benchmarkStrategy[complex128](n, features, KernelDIT, config)
			if elapsed == 0 {
				t.Errorf("benchmarkStrategy returned zero duration for size %d (complex128)", n)
			}
		})
	}
}
