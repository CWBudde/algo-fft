//go:build amd64 && asm && !purego

package fft

import m "github.com/MeKo-Christian/algo-fft/internal/math"

// Precomputed mixed-radix bit-reversal indices for size 8 (2 * 4^1).
//
//nolint:gochecknoglobals
var bitrevSize8Mixed24 = ComputeBitReversalIndicesMixed24(8)
var bitrevSize512Mixed24 = ComputeBitReversalIndicesMixed24(512)

// avx2SizeSpecificOrGenericDITComplex64 returns a kernel that tries size-specific
// AVX2 implementations for common sizes (8, 16, 32, 64, 128), falling back to the
// generic AVX2 kernel for other sizes or if the size-specific kernel fails.
func avx2SizeSpecificOrGenericDITComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

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
		case 8:
			if forwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize8Mixed24) {
				return true
			}
			if forwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 16:
			if forwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize16Radix4) {
				return true
			}
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
			if forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize64Radix4) {
				return true
			}
			if forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 128:
			if forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 256:
			if forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 512:
			if forwardAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch, bitrevSize512Mixed24) {
				return true
			}
			if forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
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
		if !m.IsPowerOf2(n) {
			return false
		}

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
		case 8:
			if inverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize8Mixed24) {
				return true
			}
			if inverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 16:
			if inverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize16Radix4) {
				return true
			}
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
			if inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevSize64Radix4) {
				return true
			}
			if inverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 128:
			if inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 256:
			if inverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch, bitrev)

		case 512:
			if inverseAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch, bitrevSize512Mixed24) {
				return true
			}
			if inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
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

// avx2SizeSpecificOrGenericDITComplex128 returns a kernel that tries size-specific
// AVX2 implementations for sizes where we have asm complex128 code, falling back to
// the generic AVX2 kernel otherwise.
func avx2SizeSpecificOrGenericDITComplex128(strategy KernelStrategy) Kernel[complex128] {
	return func(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

		resolved := resolveKernelStrategy(n, strategy)
		if resolved != KernelDIT {
			return avx2KernelComplex128(strategy, forwardAVX2Complex128, forwardAVX2StockhamComplex128)(
				dst, src, twiddle, scratch, bitrev,
			)
		}

		switch n {
		case 4:
			if forwardAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		case 8:
			if forwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch, bitrevSize8Mixed24) {
				return true
			}
			if forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			if forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		case 16:
			if forwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrevSize16Radix4) {
				return true
			}
			if forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		case 32:
			if forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		case 512:
			if forwardAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch, bitrevSize512Mixed24) {
				return true
			}
			if forwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		default:
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		}
	}
}

func avx2SizeSpecificOrGenericDITInverseComplex128(strategy KernelStrategy) Kernel[complex128] {
	return func(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

		resolved := resolveKernelStrategy(n, strategy)
		if resolved != KernelDIT {
			return avx2KernelComplex128(strategy, inverseAVX2Complex128, inverseAVX2StockhamComplex128)(
				dst, src, twiddle, scratch, bitrev,
			)
		}

		switch n {
		case 4:
			if inverseAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		case 8:
			if inverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch, bitrevSize8Mixed24) {
				return true
			}
			if inverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			if inverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		case 16:
			if inverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrevSize16Radix4) {
				return true
			}
			if inverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		case 32:
			if inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		case 512:
			if inverseAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch, bitrevSize512Mixed24) {
				return true
			}
			if inverseAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		default:
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch, bitrev)
		}
	}
}

func avx2SizeSpecificOrGenericComplex128(strategy KernelStrategy) Kernels[complex128] {
	return Kernels[complex128]{
		Forward: avx2SizeSpecificOrGenericDITComplex128(strategy),
		Inverse: avx2SizeSpecificOrGenericDITInverseComplex128(strategy),
	}
}
