//go:build arm64 && asm && !purego

package fft

import (
	"github.com/cwbudde/algo-fft/internal/planner"
)

// neonSizeSpecificOrGenericDITComplex64 returns a kernel that tries size-specific
// NEON implementations for common sizes (4, 8, 16, 32, 64, 128), falling back to the
// generic NEON kernel for other sizes or if the size-specific kernel fails.
func neonSizeSpecificOrGenericDITComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64) bool {
		n := len(src)

		// Determine which algorithm (DIT vs Stockham) based on strategy
		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			// For non-DIT strategies, use generic NEON
			// (NEON Stockham not yet implemented for ARM64)
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch)
		}

		// DIT strategy: try size-specific, fall back to generic NEON
		switch n {
		case 4:
			if forwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch)

		case 8:
			// Prefer radix-8; fall back to radix-2/mixed-radix.
			if forwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch)

		case 16:
			if forwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch)

		case 32:
			if forwardNEONSize32Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch)

		case 64:
			if forwardNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch)

		case 128:
			if forwardNEONSize128Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardNEONSize128Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch)

		case 256:
			if forwardNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch)

		default:
			// For other sizes, use generic NEON
			return forwardNEONComplex64Asm(dst, src, twiddle, scratch)
		}
	}
}

// neonSizeSpecificOrGenericDITInverseComplex64 returns a kernel that tries size-specific
// NEON implementations for inverse transforms.
func neonSizeSpecificOrGenericDITInverseComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64) bool {
		n := len(src)

		// Determine which algorithm (DIT vs Stockham) based on strategy
		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			// For non-DIT strategies, use generic NEON
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch)
		}

		// DIT strategy: try size-specific, fall back to generic NEON
		switch n {
		case 4:
			if inverseNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch)

		case 8:
			// Prefer radix-8; fall back to radix-2/mixed-radix.
			if inverseNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch)

		case 16:
			if inverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseNEONSize16Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch)

		case 32:
			if inverseNEONSize32Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseNEONSize32Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch)

		case 64:
			if inverseNEONSize64Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch)

		case 128:
			if inverseNEONSize128Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseNEONSize128Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch)

		case 256:
			if inverseNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch)

		default:
			// For other sizes, use generic NEON
			return inverseNEONComplex64Asm(dst, src, twiddle, scratch)
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

// neonSizeSpecificOrGenericDITComplex128 returns a kernel that tries size-specific
// NEON implementations for sizes where we have asm complex128 code, falling back to
// the generic NEON kernel (which currently delegates to pure Go) otherwise.
func neonSizeSpecificOrGenericDITComplex128(strategy KernelStrategy) Kernel[complex128] {
	return func(dst, src, twiddle, scratch []complex128) bool {
		n := len(src)

		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			return forwardNEONComplex128Asm(dst, src, twiddle, scratch)
		}

		switch n {
		case 4:
			if forwardNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex128Asm(dst, src, twiddle, scratch)
		case 8:
			if forwardNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex128Asm(dst, src, twiddle, scratch)
		case 16:
			if forwardNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardNEONSize16Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex128Asm(dst, src, twiddle, scratch)
		case 32:
			if forwardNEONSize32Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardNEONComplex128Asm(dst, src, twiddle, scratch)
		default:
			return forwardNEONComplex128Asm(dst, src, twiddle, scratch)
		}
	}
}

func neonSizeSpecificOrGenericDITInverseComplex128(strategy KernelStrategy) Kernel[complex128] {
	return func(dst, src, twiddle, scratch []complex128) bool {
		n := len(src)

		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			return inverseNEONComplex128Asm(dst, src, twiddle, scratch)
		}

		switch n {
		case 4:
			if inverseNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex128Asm(dst, src, twiddle, scratch)
		case 8:
			if inverseNEONSize8Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex128Asm(dst, src, twiddle, scratch)
		case 16:
			if inverseNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseNEONSize16Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex128Asm(dst, src, twiddle, scratch)
		case 32:
			if inverseNEONSize32Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseNEONComplex128Asm(dst, src, twiddle, scratch)
		default:
			return inverseNEONComplex128Asm(dst, src, twiddle, scratch)
		}
	}
}

func neonSizeSpecificOrGenericComplex128(strategy KernelStrategy) Kernels[complex128] {
	return Kernels[complex128]{
		Forward: neonSizeSpecificOrGenericDITComplex128(strategy),
		Inverse: neonSizeSpecificOrGenericDITInverseComplex128(strategy),
	}
}
