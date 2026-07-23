//go:build amd64 && !purego

package fft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
)

func selectKernelsComplex64(features cpu.Features) kernels.Kernels[complex64] {
	auto := autoKernelComplex64(fftypes.KernelAuto)
	// AVX-512 implies AVX2 on every real CPU; the explicit HasAVX2 check keeps
	// the chain safe under forced test feature sets.
	if features.HasAVX512 && features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx512SizeSpecificOrGenericComplex64(fftypes.KernelAuto)
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx2SizeSpecificOrGenericComplex64(fftypes.KernelAuto)
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasSSE3 && !features.ForceGeneric {
		sizeSpecific := sse3SizeSpecificOrGenericComplex64(fftypes.KernelAuto)
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}
	if features.HasSSE2 && !features.ForceGeneric {
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128(features cpu.Features) kernels.Kernels[complex128] {
	auto := autoKernelComplex128(fftypes.KernelAuto)
	if features.HasAVX512 && features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx512SizeSpecificOrGenericComplex128(fftypes.KernelAuto)
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx2SizeSpecificOrGenericComplex128(fftypes.KernelAuto)
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128Asm, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128Asm, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasAVX512 && features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx512SizeSpecificOrGenericComplex64(strategy)
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx2SizeSpecificOrGenericComplex64(strategy)
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasSSE3 && !features.ForceGeneric {
		sizeSpecific := sse3SizeSpecificOrGenericComplex64(strategy)
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}
	if features.HasSSE2 && !features.ForceGeneric && simdTierServesStrategy(strategy) {
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasAVX512 && features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx512SizeSpecificOrGenericComplex128(strategy)
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasAVX2 && !features.ForceGeneric {
		sizeSpecific := avx2SizeSpecificOrGenericComplex128(strategy)
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(sizeSpecific.Forward, auto.Forward),
			Inverse: fallbackKernel(sizeSpecific.Inverse, auto.Inverse),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric && simdTierServesStrategy(strategy) {
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128Asm, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128Asm, auto.Inverse),
		}
	}

	return auto
}
