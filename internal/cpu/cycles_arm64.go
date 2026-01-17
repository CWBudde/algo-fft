//go:build arm64

package cpu

// readCycleCounter reads the virtual counter (CNTVCT_EL0).
// Implemented in cycles_arm64.s
//
//go:noescape
func readCycleCounter() int64

// readCounterFrequency reads the counter frequency register (CNTFRQ_EL0).
// Returns the frequency in Hz at which CNTVCT_EL0 increments.
// Implemented in cycles_arm64.s
//
//go:noescape
func readCounterFrequency() int64

// getCounterFrequencyHz returns the counter frequency in Hz for ARM64.
// This reads the CNTFRQ_EL0 register which provides the exact frequency.
func getCounterFrequencyHz() int64 {
	return readCounterFrequency()
}
