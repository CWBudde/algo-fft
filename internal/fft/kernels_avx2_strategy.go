package fft

import (
	"github.com/cwbudde/algo-fft/internal/planner"
)

func avx2KernelComplex64(strategy KernelStrategy, dit, stockham Kernel[complex64]) Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64) bool {
		switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
		case KernelDIT:
			return dit(dst, src, twiddle, scratch)
		case KernelStockham:
			return stockham(dst, src, twiddle, scratch)
		default:
			return false
		}
	}
}

func avx2KernelComplex128(strategy KernelStrategy, dit, stockham Kernel[complex128]) Kernel[complex128] {
	return func(dst, src, twiddle, scratch []complex128) bool {
		switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
		case KernelDIT:
			return dit(dst, src, twiddle, scratch)
		case KernelStockham:
			return stockham(dst, src, twiddle, scratch)
		default:
			return false
		}
	}
}
