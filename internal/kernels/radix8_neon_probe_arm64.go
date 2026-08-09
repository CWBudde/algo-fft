//go:build arm64 && !purego && fftprobe

package kernels

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// Size 65536 exceeded the repository's 1.5x retention cutoff on the Apple M5
// in both precisions. Smaller rows live in the generated production registry;
// these two cells remain available only for cross-microarchitecture retesting.
//
//nolint:gochecknoglobals // probe-only size table
var radix8NEONProbeSizes64 = []int{65536}

//nolint:gochecknoglobals // probe-only size table
var radix8NEONProbeSizes128 = []int{65536}

//nolint:gochecknoinits // matches the generated codelet registration files
func init() {
	for _, size := range radix8NEONProbeSizes64 {
		registry.Registry64.Register(registry.CodeletEntry[complex64]{
			Size: size, Forward: forwardRadix8NEONComplex64, Inverse: inverseRadix8NEONComplex64,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNEON,
			Signature:   "dit" + itoa(size) + "_radix8ladder_neon",
			Priority:    20,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex64,
			RankBelowGeneric: true,
		})
	}

	for _, size := range radix8NEONProbeSizes128 {
		registry.Registry128.Register(registry.CodeletEntry[complex128]{
			Size: size, Forward: forwardRadix8NEONComplex128, Inverse: inverseRadix8NEONComplex128,
			Algorithm: fftypes.KernelDIT, SIMDLevel: fftypes.SIMDNEON,
			Signature:   "dit" + itoa(size) + "_radix8ladder_neon",
			Priority:    20,
			KernelType:  fftypes.KernelTypeDIT,
			TwiddleSize: twiddleSizeRadix8, PrepareTwiddle: prepareTwiddleRadix8Complex128,
			RankBelowGeneric: true,
		})
	}
}
