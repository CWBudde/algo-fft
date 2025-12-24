//go:build amd64 && fft_asm

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardAVX2Complex64, forwardStockhamComplex64),
			Inverse: fallbackKernel(inverseAVX2Complex64, inverseStockhamComplex64),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, forwardStockhamComplex64),
			Inverse: fallbackKernel(inverseSSE2Complex64, inverseStockhamComplex64),
		}
	}

	return Kernels[complex64]{
		Forward: forwardStockhamComplex64,
		Inverse: inverseStockhamComplex64,
	}
}

func selectKernelsComplex128(features Features) Kernels[complex128] {
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardAVX2Complex128Asm, forwardStockhamComplex128),
			Inverse: fallbackKernel(inverseAVX2Complex128Asm, inverseStockhamComplex128),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128Asm, forwardStockhamComplex128),
			Inverse: fallbackKernel(inverseSSE2Complex128Asm, inverseStockhamComplex128),
		}
	}

	return Kernels[complex128]{
		Forward: forwardStockhamComplex128,
		Inverse: inverseStockhamComplex128,
	}
}
