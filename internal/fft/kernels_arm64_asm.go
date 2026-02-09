//go:build arm64 && asm && !purego

package fft

import "github.com/cwbudde/algo-fft/internal/cpu"

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
		sizeSpecific := neonSizeSpecificOrGenericComplex128(KernelAuto)
		return Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
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
		sizeSpecific := neonSizeSpecificOrGenericComplex128(strategy)
		return Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	return auto
}
