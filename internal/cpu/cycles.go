package cpu

import "time"

// ReadCycleCounter reads the CPU's cycle counter (TSC on x86, CNTVCT on ARM).
// This provides high-precision timing for micro-benchmarking.
// On platforms without assembly support, falls back to time.Now().
func ReadCycleCounter() int64 {
	return readCycleCounter()
}

// CyclesSince returns the number of cycles elapsed since the given start cycle count.
func CyclesSince(start int64) int64 {
	return ReadCycleCounter() - start
}

// CyclesToNanoseconds converts cycle count to approximate nanoseconds.
// This uses a calibrated conversion factor determined at initialization.
// The conversion is approximate and should only be used for reporting purposes.
func CyclesToNanoseconds(cycles int64) int64 {
	if counterFrequencyHz != 0 {
		// ARM64 path: counter runs at a fixed frequency (CNTFRQ_EL0)
		// nanoseconds = cycles * (1e9 / freqHz) = (cycles * 1e9) / freqHz
		return (cycles * 1_000_000_000) / counterFrequencyHz
	}

	if cyclesPerNanosecond == 0 {
		// Fallback when using time.Now() - cycles are already in nanoseconds
		return cycles
	}

	// AMD64 path: TSC runs at CPU frequency, calibrated at startup
	return cycles / cyclesPerNanosecond
}

// cyclesPerNanosecond is the calibrated CPU frequency in cycles/ns.
// Used on AMD64. Initialized at package load time.
var cyclesPerNanosecond int64

// counterFrequencyHz is the counter frequency in Hz.
// Used on ARM64 where the counter runs at a fixed, known frequency.
var counterFrequencyHz int64

func init() {
	initCycleCounter()
}

// initCycleCounter initializes the cycle counter calibration.
// On ARM64, reads the hardware frequency register.
// On AMD64, calibrates by measuring cycles over a known time period.
func initCycleCounter() {
	// Try to get architecture-specific frequency first
	counterFrequencyHz = getCounterFrequencyHz()

	// If not available, calibrate
	if counterFrequencyHz == 0 {
		calibrateCycleCounter()
	}
}

// calibrateCycleCounter determines the relationship between cycle counts and wall time.
// This runs a brief calibration to estimate the CPU frequency.
// Used on AMD64 and other platforms without a frequency register.
func calibrateCycleCounter() {
	// Run a quick calibration: measure cycles over a known time period
	const calibrationDuration = 10 * time.Millisecond

	start := time.Now()
	startCycles := ReadCycleCounter()

	// Busy-wait for the calibration duration
	for time.Since(start) < calibrationDuration {
		// Spin
	}

	endCycles := ReadCycleCounter()
	elapsed := time.Since(start)

	cycles := endCycles - startCycles
	nanoseconds := elapsed.Nanoseconds()

	if nanoseconds > 0 && cycles > 0 {
		cyclesPerNanosecond = cycles / nanoseconds
	}

	// If calibration failed or we're using the time.Now() fallback,
	// cyclesPerNanosecond remains 0 and conversion is a no-op
}
