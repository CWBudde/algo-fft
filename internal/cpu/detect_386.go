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
	return Features{
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
