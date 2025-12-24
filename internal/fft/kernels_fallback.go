package fft

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
