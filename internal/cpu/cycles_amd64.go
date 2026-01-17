//go:build amd64

package cpu

// readCycleCounter reads the CPU timestamp counter using RDTSC.
// Implemented in cycles_amd64.s
//
//go:noescape
func readCycleCounter() int64

// getCounterFrequencyHz returns 0 for AMD64 since TSC frequency varies.
// AMD64 uses calibration instead.
func getCounterFrequencyHz() int64 {
	return 0
}
