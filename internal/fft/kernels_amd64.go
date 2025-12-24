//go:build amd64 && !fft_asm

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
			Forward: forwardAVX2Complex128,
			Inverse: inverseAVX2Complex128,
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: forwardSSE2Complex128,
			Inverse: inverseSSE2Complex128,
		}
	}

	return Kernels[complex128]{
		Forward: stubKernel[complex128],
		Inverse: stubKernel[complex128],
	}
}

// TODO: Replace these with assembly-backed kernels.
func forwardAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev

	return false
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev

	return false
}

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev

	return false
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev

	return false
}

func forwardAVX2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev

	return false
}

func inverseAVX2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev

	return false
}

func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev

	return false
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev

	return false
}
