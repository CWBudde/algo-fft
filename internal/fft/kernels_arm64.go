//go:build arm64 && !fft_asm

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	auto := autoKernelComplex64(KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardNEONComplex64, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128(features Features) Kernels[complex128] {
	auto := autoKernelComplex128(KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardNEONComplex128, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex128, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex64WithStrategy(features Features, strategy KernelStrategy) Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardNEONComplex64, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128WithStrategy(features Features, strategy KernelStrategy) Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardNEONComplex128, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex128, auto.Inverse),
		}
	}

	return auto
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
