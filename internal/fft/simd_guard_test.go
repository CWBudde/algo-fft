//go:build amd64 && !purego

package fft

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// requireAVX2 skips tests and benchmarks that execute AVX2 kernels directly
// when the host CPU lacks AVX2. Library dispatch checks CPU features at plan
// time, but these tests bypass dispatch and would SIGILL otherwise.
func requireAVX2(tb testing.TB) {
	tb.Helper()

	if !cpu.DetectFeatures().HasAVX2 {
		tb.Skip("host CPU lacks AVX2")
	}
}

// requireSSE3 is the SSE3 analogue of requireAVX2 (SSE2 is part of the amd64
// baseline, SSE3 is not).
func requireSSE3(tb testing.TB) {
	tb.Helper()

	if !cpu.DetectFeatures().HasSSE3 {
		tb.Skip("host CPU lacks SSE3")
	}
}
