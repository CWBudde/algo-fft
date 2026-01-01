//go:build arm64 && fft_asm && !purego && asm_dispatch

package arm64

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func selectKernelsComplex64(features cpu.Features) Kernels[complex64] {
	auto := autoKernelComplex64(KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		sizeSpecific := neonSizeSpecificOrGenericComplex64(KernelAuto)
		return Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	return auto
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
	auto := autoKernelComplex64(strategy)
	if features.HasNEON && !features.ForceGeneric {
		sizeSpecific := neonSizeSpecificOrGenericComplex64(strategy)
		return Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	return auto
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
