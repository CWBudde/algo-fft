//go:build arm64 && fft_asm

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
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

func selectKernelsComplex128(features Features) Kernels[complex128] {
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: forwardNEONComplex128Asm,
			Inverse: inverseNEONComplex128Asm,
		}
	}

	return Kernels[complex128]{
		Forward: stubKernel[complex128],
		Inverse: stubKernel[complex128],
	}
}
