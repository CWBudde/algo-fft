//go:build amd64 && fft_asm

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	auto := autoKernelComplex64(KernelAuto)
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardAVX2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseAVX2Complex64, auto.Inverse),
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

func selectKernelsComplex128(features Features) Kernels[complex128] {
	auto := autoKernelComplex128(KernelAuto)
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardAVX2Complex128Asm, auto.Forward),
			Inverse: fallbackKernel(inverseAVX2Complex128Asm, auto.Inverse),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128Asm, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128Asm, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex64WithStrategy(features Features, strategy KernelStrategy) Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardAVX2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseAVX2Complex64, auto.Inverse),
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

func selectKernelsComplex128WithStrategy(features Features, strategy KernelStrategy) Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardAVX2Complex128Asm, auto.Forward),
			Inverse: fallbackKernel(inverseAVX2Complex128Asm, auto.Inverse),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128Asm, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128Asm, auto.Inverse),
		}
	}

	return auto
}
