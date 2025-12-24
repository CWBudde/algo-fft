//go:build arm64

package cpu

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// detectFeaturesImpl performs CPU feature detection on arm64 systems.
//
// This implementation uses golang.org/x/sys/cpu to detect ARM Advanced SIMD
// (NEON) support. On ARMv8 (arm64), NEON is mandatory, so HasNEON should
// always be true on conforming implementations.
func detectFeaturesImpl() Features {
	return Features{
		HasNEON:      cpu.ARM64.HasASIMD,
		Architecture: runtime.GOARCH,
	}
}
