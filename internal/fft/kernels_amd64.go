//go:build amd64 && (!asm || purego)

package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	m "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/planner"
)

func selectKernelsComplex64(features cpu.Features) Kernels[complex64] {
	auto := autoKernelComplex64(KernelAuto)
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(
				avx2KernelComplex64(KernelAuto, forwardAVX2Complex64, forwardAVX2StockhamComplex64),
				auto.Forward,
			),
			Inverse: fallbackKernel(
				avx2KernelComplex64(KernelAuto, inverseAVX2Complex64, inverseAVX2StockhamComplex64),
				auto.Inverse,
			),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128(features cpu.Features) Kernels[complex128] {
	auto := autoKernelComplex128(KernelAuto)
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(
				avx2KernelComplex128(KernelAuto, forwardAVX2Complex128, forwardAVX2StockhamComplex128),
				auto.Forward,
			),
			Inverse: fallbackKernel(
				avx2KernelComplex128(KernelAuto, inverseAVX2Complex128, inverseAVX2StockhamComplex128),
				auto.Inverse,
			),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(
				avx2KernelComplex64(strategy, forwardAVX2Complex64, forwardAVX2StockhamComplex64),
				auto.Forward,
			),
			Inverse: fallbackKernel(
				avx2KernelComplex64(strategy, inverseAVX2Complex64, inverseAVX2StockhamComplex64),
				auto.Inverse,
			),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardSSE2Complex64, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasAVX2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(
				avx2KernelComplex128(strategy, forwardAVX2Complex128, forwardAVX2StockhamComplex128),
				auto.Forward,
			),
			Inverse: fallbackKernel(
				avx2KernelComplex128(strategy, inverseAVX2Complex128, inverseAVX2StockhamComplex128),
				auto.Inverse,
			),
		}
	}

	if features.HasSSE2 && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardSSE2Complex128, auto.Forward),
			Inverse: fallbackKernel(inverseSSE2Complex128, auto.Inverse),
		}
	}

	return auto
}

func forwardAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return forwardDITComplex64(dst, src, twiddle, scratch)
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return inverseDITComplex64(dst, src, twiddle, scratch)
}

func forwardAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return forwardStockhamComplex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2StockhamComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return inverseStockhamComplex64(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch planner.ResolveKernelStrategyWithDefault(len(src), KernelAuto) {
	case KernelDIT:
		return forwardDITComplex64(dst, src, twiddle, scratch)
	case KernelStockham:
		return forwardStockhamComplex64(dst, src, twiddle, scratch, bitrev)
	default:
		return false
	}
}

func inverseSSE2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch planner.ResolveKernelStrategyWithDefault(len(src), KernelAuto) {
	case KernelDIT:
		return inverseDITComplex64(dst, src, twiddle, scratch)
	case KernelStockham:
		return inverseStockhamComplex64(dst, src, twiddle, scratch, bitrev)
	default:
		return false
	}
}

func forwardAVX2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return forwardDITComplex128(dst, src, twiddle, scratch)
}

func inverseAVX2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return inverseDITComplex128(dst, src, twiddle, scratch)
}

func forwardAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return forwardStockhamComplex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2StockhamComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return inverseStockhamComplex128(dst, src, twiddle, scratch, bitrev)
}

func forwardSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch planner.ResolveKernelStrategyWithDefault(len(src), KernelAuto) {
	case KernelDIT:
		return forwardDITComplex128(dst, src, twiddle, scratch)
	case KernelStockham:
		return forwardStockhamComplex128(dst, src, twiddle, scratch, bitrev)
	default:
		return false
	}
}

func inverseSSE2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	switch planner.ResolveKernelStrategyWithDefault(len(src), KernelAuto) {
	case KernelDIT:
		return inverseDITComplex128(dst, src, twiddle, scratch)
	case KernelStockham:
		return inverseStockhamComplex128(dst, src, twiddle, scratch, bitrev)
	default:
		return false
	}
}
