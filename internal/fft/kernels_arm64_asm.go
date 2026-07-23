//go:build arm64 && !purego

package fft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
)

func selectKernelsComplex64(features cpu.Features) kernels.Kernels[complex64] {
	auto := autoKernelComplex64(fftypes.KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		sizeSpecific := neonSizeSpecificOrGenericComplex64(fftypes.KernelAuto)
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128(features cpu.Features) kernels.Kernels[complex128] {
	auto := autoKernelComplex128(fftypes.KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		sizeSpecific := neonSizeSpecificOrGenericComplex128(fftypes.KernelAuto)
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasNEON && !features.ForceGeneric {
		sizeSpecific := neonSizeSpecificOrGenericComplex64(strategy)
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasNEON && !features.ForceGeneric {
		sizeSpecific := neonSizeSpecificOrGenericComplex128(strategy)
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	return auto
}
