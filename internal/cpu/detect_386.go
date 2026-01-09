//go:build 386

package cpu

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// detectFeaturesImpl performs CPU feature detection on 386 systems.
//
// This implementation uses golang.org/x/sys/cpu which exposes CPUID flags
// for 32-bit x86 builds, including SSE/SSE2 support.
func detectFeaturesImpl() Features {
	// Detect SSE using manual CPUID since x/sys/cpu doesn't expose it for x86.
	// CPUID function 1, EDX bit 25 is SSE.
	hasSSE := false
	maxEAX, _, _, _ := cpuid(0, 0)
	if maxEAX >= 1 {
		_, _, _, edx := cpuid(1, 0)
		hasSSE = (edx & (1 << 25)) != 0
	}

	return Features{
		HasSSE:       hasSSE,
		HasSSE2:      cpu.X86.HasSSE2,
		HasSSE3:      cpu.X86.HasSSE3,
		HasSSSE3:     cpu.X86.HasSSSE3,
		HasSSE41:     cpu.X86.HasSSE41,
		HasAVX:       cpu.X86.HasAVX,
		HasAVX2:      cpu.X86.HasAVX2,
		HasAVX512:    cpu.X86.HasAVX512,
		Architecture: runtime.GOARCH,
	}
}
