//go:build arm64 && !fft_asm

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: forwardNEONComplex64,
			Inverse: inverseNEONComplex64,
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
			Forward: forwardNEONComplex128,
			Inverse: inverseNEONComplex128,
		}
	}

	return Kernels[complex128]{
		Forward: stubKernel[complex128],
		Inverse: stubKernel[complex128],
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
