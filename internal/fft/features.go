package fft

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// DetectFeatures reports the available CPU features for the current process.
// TODO: Integrate real feature detection (x/sys/cpu or custom CPUID).
func DetectFeatures() Features {
	return Features{
		HasAVX2:      cpu.X86.HasAVX2,
		HasAVX512:    cpu.X86.HasAVX512,
		HasSSE2:      cpu.X86.HasSSE2,
		HasNEON:      cpu.ARM64.HasASIMD,
		Architecture: runtime.GOARCH,
	}
}
