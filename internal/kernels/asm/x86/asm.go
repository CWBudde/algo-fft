//go:build 386 && fft_asm && !purego && asm_dispatch

package x86

import "github.com/MeKo-Christian/algo-fft/internal/cpu"

func selectKernelsComplex64(features cpu.Features) Kernels[complex64] {
	auto := autoKernelComplex64(KernelAuto)
	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128(features cpu.Features) Kernels[complex128] {
	_ = features
	return autoKernelComplex128(KernelAuto)
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex128] {
	_ = features
	return autoKernelComplex128(strategy)
}
