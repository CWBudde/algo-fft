//go:build amd64 && fft_asm

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: forwardAVX2Complex64,
			Inverse: inverseAVX2Complex64,
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: forwardSSE2Complex64,
			Inverse: inverseSSE2Complex64,
		}
	}

	return Kernels[complex64]{
		Forward: stubKernel[complex64],
		Inverse: stubKernel[complex64],
	}
}

func selectKernelsComplex128(features Features) Kernels[complex128] {
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: forwardAVX2Complex128Asm,
			Inverse: inverseAVX2Complex128Asm,
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: forwardSSE2Complex128Asm,
			Inverse: inverseSSE2Complex128Asm,
		}
	}

	return Kernels[complex128]{
		Forward: stubKernel[complex128],
		Inverse: stubKernel[complex128],
	}
}
