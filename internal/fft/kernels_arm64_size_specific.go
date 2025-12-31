//go:build arm64 && fft_asm && !purego

package fft

// neonSizeSpecificOrGenericDITComplex64 returns a kernel that tries size-specific
// NEON implementations for common sizes (16, 32, 64, 128), falling back to the
// generic NEON kernel for other sizes or if the size-specific kernel fails.
func neonSizeSpecificOrGenericDITComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
		n := len(src)

		// Determine which algorithm (DIT vs Stockham) based on strategy
		resolved := resolveKernelStrategy(n, strategy)
		if resolved != KernelDIT {
			// For non-DIT strategies, use generic NEON
			// (NEON Stockham not yet implemented for ARM64)
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
		}

	// DIT strategy: try size-specific, fall back to generic NEON
	switch n {
	case 8:
		// Default to radix-2; mixed-radix (radix-4) can be selected via benchmarks.
		if forwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			return true
		}
		if forwardNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			return true
		}
		return forwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

	case 16:
		if forwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize16Radix4) {
			return true
		}
		if forwardNEONSize16Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			return true
		}
		return forwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

		case 32:
			if forwardNEONSize32Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

	case 64:
		if forwardNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize64Radix4) {
			return true
		}
		if forwardNEONSize64Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			return true
		}
		return forwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

		case 128:
			if forwardNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

		default:
			// For other sizes, use generic NEON
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
		}
	}
}

// neonSizeSpecificOrGenericDITInverseComplex64 returns a kernel that tries size-specific
// NEON implementations for inverse transforms.
func neonSizeSpecificOrGenericDITInverseComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
		n := len(src)

		// Determine which algorithm (DIT vs Stockham) based on strategy
		resolved := resolveKernelStrategy(n, strategy)
		if resolved != KernelDIT {
			// For non-DIT strategies, use generic NEON
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
		}

	// DIT strategy: try size-specific, fall back to generic NEON
	switch n {
	case 8:
		// Default to radix-2; mixed-radix (radix-4) can be selected via benchmarks.
		if inverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			return true
		}
		if inverseNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			return true
		}
		return inverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

	case 16:
		if inverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize16Radix4) {
			return true
		}
		if inverseNEONSize16Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			return true
		}
		return inverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

		case 32:
			if inverseNEONSize32Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

	case 64:
		if inverseNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize64Radix4) {
			return true
		}
		if inverseNEONSize64Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			return true
		}
		return inverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

		case 128:
			if inverseNEONSize128Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)

		default:
			// For other sizes, use generic NEON
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch, bitrev)
		}
	}
}

// neonSizeSpecificOrGenericComplex64 wraps both forward and inverse size-specific kernels
// for convenience, matching the pattern in selectKernelsComplex64.
func neonSizeSpecificOrGenericComplex64(strategy KernelStrategy) Kernels[complex64] {
	return Kernels[complex64]{
		Forward: neonSizeSpecificOrGenericDITComplex64(strategy),
		Inverse: neonSizeSpecificOrGenericDITInverseComplex64(strategy),
	}
}
