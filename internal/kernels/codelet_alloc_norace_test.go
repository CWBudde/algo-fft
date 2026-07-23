//go:build !race

package kernels

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// These sweeps assert that every runnable registered codelet performs zero
// heap allocations. Transforms are zero-allocation by design; the classic
// regression is a stage buffer that fits on the stack for complex64 but not
// for complex128, where it silently moves to the heap. The compiler
// stack-allocates explicit declarations (`var x [n]T`) up to 128 KiB and
// implicit ones (`make`, `new`, `&T{}`) up to only 64 KiB, so a
// `make([]T, 8192)` scratch buffer is stack-friendly as complex64 (64 KiB)
// but heap-allocates as complex128 (128 KiB) — this sweep caught exactly
// that in the 8192-point codelets when it was introduced.

func TestCodeletsZeroAlloc64(t *testing.T) {
	features := cpu.DetectFeatures()

	checked := 0

	for _, size := range registry.Registry64.Sizes() {
		for _, entry := range registry.Registry64.GetAllForSize(size) {
			if entry.Priority < 0 || !registry.CPUSupports(features, entry.SIMDLevel) {
				continue
			}

			checked++

			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				checkCodeletZeroAlloc64(t, &entry)
			})
		}
	}

	if checked == 0 {
		t.Fatal("no runnable codelets found in registry.Registry64 — alloc sweep is vacuous")
	}
}

func TestCodeletsZeroAlloc128(t *testing.T) {
	features := cpu.DetectFeatures()

	checked := 0

	for _, size := range registry.Registry128.Sizes() {
		for _, entry := range registry.Registry128.GetAllForSize(size) {
			if entry.Priority < 0 || !registry.CPUSupports(features, entry.SIMDLevel) {
				continue
			}

			checked++

			t.Run(fmt.Sprintf("size%d/%s", size, entry.Signature), func(t *testing.T) {
				checkCodeletZeroAlloc128(t, &entry)
			})
		}
	}

	if checked == 0 {
		t.Fatal("no runnable codelets found in registry.Registry128 — alloc sweep is vacuous")
	}
}

func checkCodeletZeroAlloc64(t *testing.T, entry *registry.CodeletEntry[complex64]) {
	t.Helper()

	size := entry.Size
	twiddle := ComputeTwiddleFactors[complex64](size)
	twiddleForward, twiddleInverse, forwardBacking, inverseBacking := prepareCodeletTwiddles64(size, twiddle, entry)

	dst := make([]complex64, size)
	src := randomComplex64(size, 1)
	scratch := make([]complex64, size)

	if entry.Forward != nil {
		allocs := testing.AllocsPerRun(3, func() {
			entry.Forward(dst, src, twiddleForward, scratch)
		})
		if allocs != 0 {
			t.Errorf("forward allocates %.1f times per run, want 0", allocs)
		}
	}

	if entry.Inverse != nil {
		allocs := testing.AllocsPerRun(3, func() {
			entry.Inverse(dst, src, twiddleInverse, scratch)
		})
		if allocs != 0 {
			t.Errorf("inverse allocates %.1f times per run, want 0", allocs)
		}
	}

	runtime.KeepAlive(forwardBacking)
	runtime.KeepAlive(inverseBacking)
}

func checkCodeletZeroAlloc128(t *testing.T, entry *registry.CodeletEntry[complex128]) {
	t.Helper()

	size := entry.Size
	twiddle := ComputeTwiddleFactors[complex128](size)
	twiddleForward, twiddleInverse, forwardBacking, inverseBacking := prepareCodeletTwiddles128(size, twiddle, entry)

	dst := make([]complex128, size)
	src := randomComplex128(size, 1)
	scratch := make([]complex128, size)

	if entry.Forward != nil {
		allocs := testing.AllocsPerRun(3, func() {
			entry.Forward(dst, src, twiddleForward, scratch)
		})
		if allocs != 0 {
			t.Errorf("forward allocates %.1f times per run, want 0", allocs)
		}
	}

	if entry.Inverse != nil {
		allocs := testing.AllocsPerRun(3, func() {
			entry.Inverse(dst, src, twiddleInverse, scratch)
		})
		if allocs != 0 {
			t.Errorf("inverse allocates %.1f times per run, want 0", allocs)
		}
	}

	runtime.KeepAlive(forwardBacking)
	runtime.KeepAlive(inverseBacking)
}
