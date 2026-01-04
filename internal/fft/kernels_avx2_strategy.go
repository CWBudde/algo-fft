package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

func avx2KernelComplex64(strategy KernelStrategy, dit, stockham Kernel[complex64]) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
		switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
		case KernelDIT:
			return dit(dst, src, twiddle, scratch, bitrev)
		case KernelStockham:
			return stockham(dst, src, twiddle, scratch, bitrev)
		default:
			return false
		}
	}
}

func avx2KernelComplex128(strategy KernelStrategy, dit, stockham Kernel[complex128]) Kernel[complex128] {
	return func(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
		switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
		case KernelDIT:
			return dit(dst, src, twiddle, scratch, bitrev)
		case KernelStockham:
			return stockham(dst, src, twiddle, scratch, bitrev)
		default:
			return false
		}
	}
}
