//go:build arm64 && fft_asm && !purego

package fft

import (
	"github.com/MeKo-Christian/algofft/internal/cpu"
)

func selectKernelsComplex64(features cpu.Features) Kernels[complex64] {
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: forwardNEONComplex64Asm,
			Inverse: inverseNEONComplex64Asm,
		}
	}

	return Kernels[complex64]{
		Forward: stubKernel[complex64],
		Inverse: stubKernel[complex64],
	}
}

func selectKernelsComplex128(features cpu.Features) Kernels[complex128] {
	auto := autoKernelComplex128(KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardNEONComplex128Asm, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex128Asm, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex64] {
	// For NEON assembly, ignore strategy for now and use same logic as selectKernelsComplex64
	// Strategy selection (DIT vs Stockham) will be handled in pure-Go fallback
	return selectKernelsComplex64(features)
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardNEONComplex128Asm, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex128Asm, auto.Inverse),
		}
	}

	return auto
}
