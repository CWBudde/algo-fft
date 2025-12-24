//go:build arm64 && !fft_asm

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardNEONComplex64, forwardStockhamComplex64),
			Inverse: fallbackKernel(inverseNEONComplex64, inverseStockhamComplex64),
		}
	}

	return Kernels[complex64]{
		Forward: forwardStockhamComplex64,
		Inverse: inverseStockhamComplex64,
	}
}

func selectKernelsComplex128(features Features) Kernels[complex128] {
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardNEONComplex128, forwardStockhamComplex128),
			Inverse: fallbackKernel(inverseNEONComplex128, inverseStockhamComplex128),
		}
	}

	return Kernels[complex128]{
		Forward: forwardStockhamComplex128,
		Inverse: inverseStockhamComplex128,
	}
}

// TODO: Replace these with assembly-backed kernels.
func forwardNEONComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev
	return false
}

func inverseNEONComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev
	return false
}

func forwardNEONComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev
	return false
}

func inverseNEONComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev
	return false
}
