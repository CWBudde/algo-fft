//go:build amd64

package cpu

// readCycleCounter reads the CPU timestamp counter using RDTSC.
// Implemented in cycles_amd64.s
//
//go:noescape
func readCycleCounter() int64
