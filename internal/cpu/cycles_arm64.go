//go:build arm64

package cpu

// readCycleCounter reads the virtual counter (CNTVCT_EL0).
// Implemented in cycles_arm64.s
//
//go:noescape
func readCycleCounter() int64
