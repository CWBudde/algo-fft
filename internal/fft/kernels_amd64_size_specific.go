//go:build amd64 && asm && !purego

package fft

import (
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
)

// avx2SizeSpecificOrGenericDITComplex64 returns a kernel that tries size-specific
// AVX2 implementations for common sizes (8, 16, 32, 64, 128), falling back to the
// generic AVX2 kernel for other sizes or if the size-specific kernel fails.
func avx2SizeSpecificOrGenericDITComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

		// Determine which algorithm (DIT vs Stockham) based on strategy
		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			// For non-DIT strategies, use the existing strategy-based dispatch
			return avx2KernelComplex64(strategy, forwardAVX2Complex64, forwardAVX2StockhamComplex64)(
				dst, src, twiddle, scratch,
			)
		}

		// DIT strategy: try size-specific, fall back to generic AVX2
		switch n {
		case 8:
			if forwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 16:
			if forwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 32:
			if forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 64:
			if forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size64Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 128:
			if forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 256:
			if forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 512:
			if forwardAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 2048:
			if forwardAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 8192:
			if forwardAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)

		default:
			// For other sizes, use generic AVX2
			return forwardAVX2Complex64Asm(dst, src, twiddle, scratch)
		}
	}
}

// avx2SizeSpecificOrGenericDITInverseComplex64 returns a kernel that tries size-specific
// AVX2 implementations for inverse transforms.
func avx2SizeSpecificOrGenericDITInverseComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

		// Determine which algorithm (DIT vs Stockham) based on strategy
		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			// For non-DIT strategies, use the existing strategy-based dispatch
			return avx2KernelComplex64(strategy, inverseAVX2Complex64, inverseAVX2StockhamComplex64)(
				dst, src, twiddle, scratch,
			)
		}

		// DIT strategy: try size-specific, fall back to generic AVX2
		switch n {
		case 8:
			if inverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 16:
			if inverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 32:
			if inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 64:
			if inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size64Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 128:
			if inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 256:
			if inverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 512:
			if inverseAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 2048:
			if inverseAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)

		case 8192:
			if inverseAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)

		default:
			// For other sizes, use generic AVX2
			return inverseAVX2Complex64Asm(dst, src, twiddle, scratch)
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

// sse3SizeSpecificOrGenericDITComplex64 returns a kernel that tries size-specific
// SSE3 implementations for common sizes, falling back to the generic SSE3 kernel.
func sse3SizeSpecificOrGenericDITComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)
		}

		switch n {
		case 4:
			if forwardSSE3Size4Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 8:
			if forwardSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 16:
			if forwardSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 32:
			if forwardSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 64:
			if forwardSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 128:
			if forwardSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 256:
			if forwardSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 512:
			if forwardSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)

		default:
			return forwardSSE3Complex64Asm(dst, src, twiddle, scratch)
		}
	}
}

func sse3SizeSpecificOrGenericDITInverseComplex64(strategy KernelStrategy) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)
		}

		switch n {
		case 4:
			if inverseSSE3Size4Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 8:
			if inverseSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 16:
			if inverseSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 32:
			if inverseSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 64:
			if inverseSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 128:
			if inverseSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 256:
			if inverseSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)

		case 512:
			if inverseSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)

		default:
			return inverseSSE3Complex64Asm(dst, src, twiddle, scratch)
		}
	}
}

func sse3SizeSpecificOrGenericComplex64(strategy KernelStrategy) Kernels[complex64] {
	return Kernels[complex64]{
		Forward: sse3SizeSpecificOrGenericDITComplex64(strategy),
		Inverse: sse3SizeSpecificOrGenericDITInverseComplex64(strategy),
	}
}

// avx2SizeSpecificOrGenericDITComplex128 returns a kernel that tries size-specific
// AVX2 implementations for sizes where we have asm complex128 code, falling back to
// the generic AVX2 kernel otherwise.
func avx2SizeSpecificOrGenericDITComplex128(strategy KernelStrategy) Kernel[complex128] {
	return func(dst, src, twiddle, scratch []complex128) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			return avx2KernelComplex128(strategy, forwardAVX2Complex128, forwardAVX2StockhamComplex128)(
				dst, src, twiddle, scratch,
			)
		}

		switch n {
		case 4:
			if forwardAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 8:
			if forwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 16:
			if forwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size16Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 32:
			if forwardAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 64:
			if forwardAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 512:
			if forwardAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if forwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch)
		default:
			return forwardAVX2Complex128Asm(dst, src, twiddle, scratch)
		}
	}
}

func avx2SizeSpecificOrGenericDITInverseComplex128(strategy KernelStrategy) Kernel[complex128] {
	return func(dst, src, twiddle, scratch []complex128) bool {
		n := len(src)
		if !m.IsPowerOf2(n) {
			return false
		}

		resolved := planner.ResolveKernelStrategyWithDefault(n, strategy)
		if resolved != KernelDIT {
			return avx2KernelComplex128(strategy, inverseAVX2Complex128, inverseAVX2StockhamComplex128)(
				dst, src, twiddle, scratch,
			)
		}

		switch n {
		case 4:
			if inverseAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 8:
			if inverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 16:
			if inverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size16Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 32:
			if inverseAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 64:
			if inverseAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch)
		case 512:
			if inverseAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			if inverseAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch) {
				return true
			}
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch)
		default:
			return inverseAVX2Complex128Asm(dst, src, twiddle, scratch)
		}
	}
}

func avx2SizeSpecificOrGenericComplex128(strategy KernelStrategy) Kernels[complex128] {
	return Kernels[complex128]{
		Forward: avx2SizeSpecificOrGenericDITComplex128(strategy),
		Inverse: avx2SizeSpecificOrGenericDITInverseComplex128(strategy),
	}
}
