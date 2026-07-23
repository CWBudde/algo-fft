//go:build amd64 && !purego

package fft

import (
	"fmt"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestSelectKernelsComplex64_SSE3Only verifies the SSE3 dispatch tier (below
// AVX2, above SSE2) against the reference DFT across every size with a
// size-specific SSE3 kernel, plus a large size served by the generic SSE3
// kernel. Machines with AVX2 never take this tier in normal runs, so it is
// forced here (mirroring the SSE2-only tests).
//
//nolint:paralleltest // modifies global CPU feature state
func TestSelectKernelsComplex64_SSE3Only(t *testing.T) {
	requireSSE3(t) // forcing HasSSE3 past dispatch would SIGILL on a non-SSE3 host

	originalFeatures := cpu.DetectFeatures()
	defer cpu.SetForcedFeatures(originalFeatures)

	cpu.SetForcedFeatures(cpu.Features{HasSSE: true, HasSSE2: true, HasSSE3: true})

	kern := selectKernelsComplex64(cpu.DetectFeatures())

	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512, 2048}
	for _, n := range sizes {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			src := randomComplex64(n, 0x55E3+uint64(n))
			twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
			scratch := make([]complex64, n)

			dst := make([]complex64, n)
			if !kern.Forward(dst, src, twiddle, scratch) {
				t.Fatalf("SSE3 forward kernel failed for n=%d", n)
			}

			want := reference.NaiveDFT(src)
			assertComplex64SliceClose(t, dst, want, n)

			inv := make([]complex64, n)
			if !kern.Inverse(inv, dst, twiddle, scratch) {
				t.Fatalf("SSE3 inverse kernel failed for n=%d", n)
			}

			assertComplex64SliceClose(t, inv, src, n)
		})
	}
}

// TestSelectKernelsWithStrategy_SSE3 exercises both branches of the SSE3
// size-specific selector: forced DIT (size-specific kernels) and forced
// Stockham (generic SSE3 kernel).
//
//nolint:paralleltest // modifies global CPU feature state
func TestSelectKernelsWithStrategy_SSE3(t *testing.T) {
	requireSSE3(t) // forcing HasSSE3 past dispatch would SIGILL on a non-SSE3 host

	originalFeatures := cpu.DetectFeatures()
	defer cpu.SetForcedFeatures(originalFeatures)

	cpu.SetForcedFeatures(cpu.Features{HasSSE: true, HasSSE2: true, HasSSE3: true})

	strategies := []struct {
		name     string
		strategy planner.KernelStrategy
	}{
		{"DIT", planner.KernelDIT},
		{"Stockham", planner.KernelStockham},
		{"Auto", planner.KernelAuto},
	}

	for _, tt := range strategies {
		t.Run(tt.name, func(t *testing.T) {
			kern := selectKernelsComplex64WithStrategy(cpu.DetectFeatures(), tt.strategy)

			for _, n := range []int{32, 128} {
				src := randomComplex64(n, 0x57A7+uint64(n))
				twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
				scratch := make([]complex64, n)

				dst := make([]complex64, n)
				if !kern.Forward(dst, src, twiddle, scratch) {
					t.Fatalf("SSE3 forward failed with strategy %v, n=%d", tt.strategy, n)
				}

				want := reference.NaiveDFT(src)
				assertComplex64SliceClose(t, dst, want, n)

				inv := make([]complex64, n)
				if !kern.Inverse(inv, dst, twiddle, scratch) {
					t.Fatalf("SSE3 inverse failed with strategy %v, n=%d", tt.strategy, n)
				}

				assertComplex64SliceClose(t, inv, src, n)
			}
		})
	}
}
