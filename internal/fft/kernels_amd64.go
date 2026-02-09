//go:build amd64 && (!asm || purego)

package fft

import "github.com/cwbudde/algo-fft/internal/cpu"

func selectKernelsComplex64(features cpu.Features) Kernels[complex64] {
	return autoKernelComplex64(KernelAuto)
}

func selectKernelsComplex128(features cpu.Features) Kernels[complex128] {
	return autoKernelComplex128(KernelAuto)
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex64] {
	return autoKernelComplex64(strategy)
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex128] {
	return autoKernelComplex128(strategy)
}

// Fallback wrappers for tests when asm is disabled.
func forwardAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	return forwardDITComplex64(dst, src, twiddle, scratch)
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	return inverseDITComplex64(dst, src, twiddle, scratch)
}

func forwardAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64) bool {
	return forwardStockhamComplex64(dst, src, twiddle, scratch)
}

func inverseAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64) bool {
	return inverseStockhamComplex64(dst, src, twiddle, scratch)
}

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64) bool {
	return forwardDITComplex64(dst, src, twiddle, scratch)
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64) bool {
	return inverseDITComplex64(dst, src, twiddle, scratch)
}

func forwardAVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return forwardDITComplex128(dst, src, twiddle, scratch)
}

func inverseAVX2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return inverseDITComplex128(dst, src, twiddle, scratch)
}

func forwardAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128) bool {
	return forwardStockhamComplex128(dst, src, twiddle, scratch)
}

func inverseAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128) bool {
	return inverseStockhamComplex128(dst, src, twiddle, scratch)
}

func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return forwardDITComplex128(dst, src, twiddle, scratch)
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128) bool {
	return inverseDITComplex128(dst, src, twiddle, scratch)
}
