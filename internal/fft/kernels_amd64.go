//go:build amd64 && purego

package fft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
)

func selectKernelsComplex64(features cpu.Features) kernels.Kernels[complex64] {
	return autoKernelComplex64(fftypes.KernelAuto)
}

func selectKernelsComplex128(features cpu.Features) kernels.Kernels[complex128] {
	return autoKernelComplex128(fftypes.KernelAuto)
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex64] {
	return autoKernelComplex64(strategy)
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex128] {
	return autoKernelComplex128(strategy)
}

// Fallback wrappers for tests when asm is disabled.
func forwardAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	return kernels.ForwardDITComplex64(dst, src, twiddle, scratch)
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	return kernels.InverseDITComplex64(dst, src, twiddle, scratch)
}

func forwardAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64) bool {
	return kernels.ForwardStockhamComplex64(dst, src, twiddle, scratch)
}

func inverseAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64) bool {
	return kernels.InverseStockhamComplex64(dst, src, twiddle, scratch)
}

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64) bool {
	return kernels.ForwardDITComplex64(dst, src, twiddle, scratch)
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64) bool {
	return kernels.InverseDITComplex64(dst, src, twiddle, scratch)
}

func forwardAVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return kernels.ForwardDITComplex128(dst, src, twiddle, scratch)
}

func inverseAVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return kernels.InverseDITComplex128(dst, src, twiddle, scratch)
}

func forwardAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128) bool {
	return kernels.ForwardStockhamComplex128(dst, src, twiddle, scratch)
}

func inverseAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128) bool {
	return kernels.InverseStockhamComplex128(dst, src, twiddle, scratch)
}

func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return kernels.ForwardDITComplex128(dst, src, twiddle, scratch)
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return kernels.InverseDITComplex128(dst, src, twiddle, scratch)
}
