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
	if cyclesPerNanosecond == 0 {
		// Fallback when using time.Now() - cycles are already in nanoseconds
		return cycles
	}
	return cycles / cyclesPerNanosecond
}

// cyclesPerNanosecond is the calibrated CPU frequency in cycles/ns.
// Initialized at package load time.
var cyclesPerNanosecond int64

func init() {
	calibrateCycleCounter()
}

// calibrateCycleCounter determines the relationship between cycle counts and wall time.
// This runs a brief calibration to estimate the CPU frequency.
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
