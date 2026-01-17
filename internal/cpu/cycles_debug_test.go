package cpu

import (
	"runtime"
	"testing"
)

// TestCycleCounterDebug provides diagnostic information about the cycle counter.
func TestCycleCounterDebug(t *testing.T) {
	t.Logf("Platform: %s/%s", runtime.GOOS, runtime.GOARCH)
	t.Logf("High precision platform: %v", isHighPrecisionPlatform())

	if runtime.GOARCH == "arm64" {
		freq := getCounterFrequencyHz()
		t.Logf("ARM64 counter frequency (CNTFRQ_EL0): %d Hz", freq)

		if freq > 0 {
			t.Logf("Counter frequency: %.2f MHz", float64(freq)/1_000_000)
		}
	}

	t.Logf("counterFrequencyHz: %d", counterFrequencyHz)
	t.Logf("cyclesPerNanosecond: %d", cyclesPerNanosecond)

	// Read some sample values
	c1 := ReadCycleCounter()
	c2 := ReadCycleCounter()
	c3 := ReadCycleCounter()

	t.Logf("Sample readings: c1=%d, c2=%d, c3=%d", c1, c2, c3)
	t.Logf("Deltas: c2-c1=%d, c3-c2=%d", c2-c1, c3-c2)
}
