package cpu

import "testing"

// TestHasSSEAccessor verifies the HasSSE accessor agrees with DetectFeatures.
func TestHasSSEAccessor(t *testing.T) {
	t.Parallel()

	if HasSSE() != DetectFeatures().HasSSE {
		t.Errorf("HasSSE() = %v, want %v", HasSSE(), DetectFeatures().HasSSE)
	}
}

// TestForceSSEOnlyForTests verifies the SSE-only override exposes SSE and
// nothing above it. Deliberately not parallel: it mutates the forced-feature
// global (restored via ResetDetection).
func TestForceSSEOnlyForTests(t *testing.T) { //nolint:paralleltest // mutates global detection state
	defer ResetDetection()

	ForceSSEOnlyForTests()

	features := DetectFeatures()
	if !features.HasSSE {
		t.Error("HasSSE = false after ForceSSEOnlyForTests, want true")
	}

	for name, got := range map[string]bool{
		"HasSSE2":   features.HasSSE2,
		"HasSSE3":   features.HasSSE3,
		"HasSSSE3":  features.HasSSSE3,
		"HasSSE41":  features.HasSSE41,
		"HasAVX":    features.HasAVX,
		"HasAVX2":   features.HasAVX2,
		"HasFMA":    features.HasFMA,
		"HasAVX512": features.HasAVX512,
	} {
		if got {
			t.Errorf("%s = true after ForceSSEOnlyForTests, want false", name)
		}
	}
}

// TestCyclesToNanosecondsBranches pins the three conversion paths (fixed
// counter frequency, calibrated TSC, time.Now fallback). Deliberately not
// parallel: it swaps the package-level calibration values (restored on exit).
func TestCyclesToNanosecondsBranches(t *testing.T) { //nolint:paralleltest // mutates calibration globals
	savedFreq := counterFrequencyHz
	savedCPN := cyclesPerNanosecond

	defer func() {
		counterFrequencyHz = savedFreq
		cyclesPerNanosecond = savedCPN
	}()

	// Fixed-frequency path (ARM64 CNTFRQ): 1 GHz counter → 1 cycle = 1 ns.
	counterFrequencyHz = 1_000_000_000
	cyclesPerNanosecond = 0

	if got := CyclesToNanoseconds(500); got != 500 {
		t.Errorf("fixed-frequency conversion = %d, want 500", got)
	}

	// Calibrated TSC path (AMD64): 3 cycles per nanosecond.
	counterFrequencyHz = 0
	cyclesPerNanosecond = 3

	if got := CyclesToNanoseconds(300); got != 100 {
		t.Errorf("calibrated conversion = %d, want 100", got)
	}

	// time.Now() fallback: cycles already are nanoseconds.
	counterFrequencyHz = 0
	cyclesPerNanosecond = 0

	if got := CyclesToNanoseconds(1234); got != 1234 {
		t.Errorf("fallback conversion = %d, want 1234", got)
	}
}
