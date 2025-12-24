//go:build amd64 && !fft_asm

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
			Forward: fallbackKernel(forwardAVX2Complex128, auto.Forward),
			Inverse: fallbackKernel(inverseAVX2Complex128, auto.Inverse),
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
			Forward: fallbackKernel(forwardAVX2Complex128, auto.Forward),
			Inverse: fallbackKernel(inverseAVX2Complex128, auto.Inverse),
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
