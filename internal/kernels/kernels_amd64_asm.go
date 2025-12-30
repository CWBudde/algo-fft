//go:build amd64 && fft_asm && !purego

package kernels

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func selectKernelsComplex64(features cpu.Features) Kernels[complex64] {
	auto := autoKernelComplex64(KernelAuto)
	if features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx2SizeSpecificOrGenericComplex64(KernelAuto)
		return Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128(features cpu.Features) Kernels[complex128] {
	auto := autoKernelComplex128(KernelAuto)
	if features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx2SizeSpecificOrGenericComplex128(KernelAuto)
		return Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx2SizeSpecificOrGenericComplex64(strategy)
		return Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx2SizeSpecificOrGenericComplex128(strategy)
		return Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128, auto.Inverse),
		}
	}

	return auto
}
