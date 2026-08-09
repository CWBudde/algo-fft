//go:build arm64 && !purego && fftprobe

package kernels

// Probe for the existing size-generic NEON radix-2 DIT kernel. The corrected
// implementation lost every size from 4 through 65536 on an Apple M5; the
// nearest result was complex128 at 32768 (1.45x slower forward, 1.69x inverse).
// It remains here so another ARM microarchitecture can reproduce the sweep.
//
// Take the number on native arm64 with:
//
//	GOFLAGS=-tags=fftprobe go test -run '^$' \
//	  -bench 'BenchmarkCodeletCandidates(64|128)/.*/dit[0-9]+_radix2ladder_neon' \
//	  -benchmem -benchtime=100ms -count=3 ./internal/kernels

import (
	"github.com/cwbudde/algo-fft/internal/asm/arm64"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

//nolint:gochecknoglobals // probe-only size table
var radix2NEONProbeSizes = []int{
	4, 8, 16, 32, 64, 128, 256, 512,
	1024, 2048, 4096, 8192, 16384, 32768, 65536,
}

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	for _, size := range radix2NEONProbeSizes {
		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: size, Forward: arm64.ForwardNEONComplex64Asm, Inverse: arm64.InverseNEONComplex64Asm,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNEON,
			Signature:        "dit" + itoa(size) + "_radix2ladder_neon",
			Priority:         1,
			KernelType:       fftypes.KernelTypeDIT,
			RankBelowGeneric: true,
		})

		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: size, Forward: arm64.ForwardNEONComplex128Asm, Inverse: arm64.InverseNEONComplex128Asm,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNEON,
			Signature:        "dit" + itoa(size) + "_radix2ladder_neon",
			Priority:         1,
			KernelType:       fftypes.KernelTypeDIT,
			RankBelowGeneric: true,
		})
	}
}
