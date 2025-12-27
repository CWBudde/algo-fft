//go:build amd64 && fft_asm && !purego

package fft

// avx2SizeSpecificOrGenericDITComplex64 returns a kernel that tries size-specific
// AVX2 implementations for common sizes (16, 32, 64, 128), falling back to the
// generic AVX2 kernel for other sizes or if the size-specific kernel fails.
func avx2SizeSpecificOrGenericDITComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
		n := len(src)

		// Determine which algorithm (DIT vs Stockham) based on strategy
		resolved := resolveKernelStrategy(n, strategy)
		if resolved != KernelDIT {
			// For non-DIT strategies, use the existing strategy-based dispatch
			return avx2KernelComplex64(strategy, forwardAVX2Complex64, forwardAVX2StockhamComplex64)(
				dst, src, twiddle, scratch, bitrev,
			)
		}

		// DIT strategy: try size-specific, fall back to generic AVX2
		switch n {
		case 16:
			if forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 32:
			if forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 64:
			if forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 128:
			if forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		default:
			// For other sizes, use generic AVX2
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
		}
	}
}

// avx2SizeSpecificOrGenericDITInverseComplex64 returns a kernel that tries size-specific
// AVX2 implementations for inverse transforms.
func avx2SizeSpecificOrGenericDITInverseComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
		n := len(src)

		// Determine which algorithm (DIT vs Stockham) based on strategy
		resolved := resolveKernelStrategy(n, strategy)
		if resolved != KernelDIT {
			// For non-DIT strategies, use the existing strategy-based dispatch
			return avx2KernelComplex64(strategy, inverseAVX2Complex64, inverseAVX2StockhamComplex64)(
				dst, src, twiddle, scratch, bitrev,
			)
		}

		// DIT strategy: try size-specific, fall back to generic AVX2
		switch n {
		case 16:
			if inverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 32:
			if inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 64:
			if inverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 128:
			if inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		default:
			// For other sizes, use generic AVX2
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)
		}
	}
}

// avx2SizeSpecificOrGenericComplex64 wraps both forward and inverse size-specific kernels
// for convenience, matching the pattern in selectKernelsComplex64.
func avx2SizeSpecificOrGenericComplex64(strategy KernelStrategy) Kernels[complex64] {
	return Kernels[complex64]{
		Forward: avx2SizeSpecificOrGenericDITComplex64(strategy),
		Inverse: avx2SizeSpecificOrGenericDITInverseComplex64(strategy),
	}
}
