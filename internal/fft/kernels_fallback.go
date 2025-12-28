package fft

const ditAutoThreshold = 1024

func fallbackKernel[T Complex](primary, fallback Kernel[T]) Kernel[T] {
	if primary == nil {
		return fallback
	}

	return func(dst, src, twiddle, scratch []T, bitrev []int) bool {
		if primary != nil && primary(dst, src, twiddle, scratch, bitrev) {
			return true
		}

		return fallback(dst, src, twiddle, scratch, bitrev)
	}
}

func autoKernelComplex64(strategy KernelStrategy) Kernels[complex64] {
	return Kernels[complex64]{
		Forward: func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
			if !IsPowerOf2(len(src)) {
				if isHighlyComposite(len(src)) {
					return forwardMixedRadixComplex64(dst, src, twiddle, scratch, bitrev)
				}

				return false
			}

			switch resolveKernelStrategy(len(src), strategy) {
			case KernelDIT:
				return forwardDITComplex64(dst, src, twiddle, scratch, bitrev)
			case KernelStockham:
				return forwardStockhamComplex64(dst, src, twiddle, scratch, bitrev)
			case KernelSixStep:
				return forwardSixStepComplex64(dst, src, twiddle, scratch, bitrev)
			case KernelEightStep:
				return forwardEightStepComplex64(dst, src, twiddle, scratch, bitrev)
			default:
				return forwardStockhamComplex64(dst, src, twiddle, scratch, bitrev)
			}
		},
		Inverse: func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
			if !IsPowerOf2(len(src)) {
				if isHighlyComposite(len(src)) {
					return inverseMixedRadixComplex64(dst, src, twiddle, scratch, bitrev)
				}

				return false
			}

			switch resolveKernelStrategy(len(src), strategy) {
			case KernelDIT:
				return inverseDITComplex64(dst, src, twiddle, scratch, bitrev)
			case KernelStockham:
				return inverseStockhamComplex64(dst, src, twiddle, scratch, bitrev)
			case KernelSixStep:
				return inverseSixStepComplex64(dst, src, twiddle, scratch, bitrev)
			case KernelEightStep:
				return inverseEightStepComplex64(dst, src, twiddle, scratch, bitrev)
			default:
				return inverseStockhamComplex64(dst, src, twiddle, scratch, bitrev)
			}
		},
	}
}

func autoKernelComplex128(strategy KernelStrategy) Kernels[complex128] {
	return Kernels[complex128]{
		Forward: func(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
			if !IsPowerOf2(len(src)) {
				if isHighlyComposite(len(src)) {
					return forwardMixedRadixComplex128(dst, src, twiddle, scratch, bitrev)
				}

				return false
			}

			switch resolveKernelStrategy(len(src), strategy) {
			case KernelDIT:
				return forwardDITComplex128(dst, src, twiddle, scratch, bitrev)
			case KernelStockham:
				return forwardStockhamComplex128(dst, src, twiddle, scratch, bitrev)
			case KernelSixStep:
				return forwardSixStepComplex128(dst, src, twiddle, scratch, bitrev)
			case KernelEightStep:
				return forwardEightStepComplex128(dst, src, twiddle, scratch, bitrev)
			default:
				return forwardStockhamComplex128(dst, src, twiddle, scratch, bitrev)
			}
		},
		Inverse: func(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
			if !IsPowerOf2(len(src)) {
				if isHighlyComposite(len(src)) {
					return inverseMixedRadixComplex128(dst, src, twiddle, scratch, bitrev)
				}

				return false
			}

			switch resolveKernelStrategy(len(src), strategy) {
			case KernelDIT:
				return inverseDITComplex128(dst, src, twiddle, scratch, bitrev)
			case KernelStockham:
				return inverseStockhamComplex128(dst, src, twiddle, scratch, bitrev)
			case KernelSixStep:
				return inverseSixStepComplex128(dst, src, twiddle, scratch, bitrev)
			case KernelEightStep:
				return inverseEightStepComplex128(dst, src, twiddle, scratch, bitrev)
			default:
				return inverseStockhamComplex128(dst, src, twiddle, scratch, bitrev)
			}
		},
	}
}
