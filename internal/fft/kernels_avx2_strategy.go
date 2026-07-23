package fft

import (
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
	"github.com/cwbudde/algo-fft/internal/planner"
)

func avx2KernelComplex64(strategy fftypes.KernelStrategy, dit, stockham kernels.Kernel[complex64]) kernels.Kernel[complex64] {
	return func(dst, src, twiddle, scratch []complex64) bool {
		switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
		case fftypes.KernelDIT:
			return dit(dst, src, twiddle, scratch)
		case fftypes.KernelStockham:
			return stockham(dst, src, twiddle, scratch)
		default:
			return false
		}
	}
}

func avx2KernelComplex128(strategy fftypes.KernelStrategy, dit, stockham kernels.Kernel[complex128]) kernels.Kernel[complex128] {
	return func(dst, src, twiddle, scratch []complex128) bool {
		switch planner.ResolveKernelStrategyWithDefault(len(src), strategy) {
		case fftypes.KernelDIT:
			return dit(dst, src, twiddle, scratch)
		case fftypes.KernelStockham:
			return stockham(dst, src, twiddle, scratch)
		default:
			return false
		}
	}
}
