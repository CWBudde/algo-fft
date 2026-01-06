//go:build amd64

package cpu

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// detectFeaturesImpl performs CPU feature detection on amd64 systems.
//
// This implementation uses golang.org/x/sys/cpu which provides a portable
// interface to CPUID instruction results. All modern amd64 CPUs support
// SSE2 as it's part of the x86-64 baseline, so HasSSE2 is always true.
func detectFeaturesImpl() Features {
	return Features{
		HasSSE:       true,
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
