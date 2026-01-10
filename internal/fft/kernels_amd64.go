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
	switch len(src) {
	case 4:
		return forwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch)
	case 8:
		return forwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch)
	case 16:
		return forwardAVX2Size16Radix16Complex64Asm(dst, src, twiddle, scratch)
	case 32:
		return forwardAVX2Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
	case 64:
		return forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch)
	case 128:
		return forwardAVX2Size128Mixed24Complex64Asm(dst, src, twiddle, scratch)
	case 256:
		// Attempt to use highest priority codelet from registry for better performance
		if entry, ok := Registry64.GetBest(len(src), SIMDAVX2, KernelDIT); ok {
			return entry.Forward(dst, src, twiddle, scratch, bitrev)
		}
		return forwardAVX2Size256Radix4Complex64Safe(dst, src, twiddle, scratch)
	case 512:
		return forwardAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 1024:
		return forwardAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 2048:
		return forwardAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 4096:
		return forwardAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 8192:
		return forwardAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 16384:
		return forwardAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}

	return forwardDITComplex64(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Complex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	switch len(src) {
	case 4:
		return inverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 8:
		return inverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 16:
		return inverseAVX2Size16Radix16Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 32:
		return inverseAVX2Size32Radix32Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 64:
		return inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 128:
		return inverseAVX2Size128Mixed24Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 256:
		if entry, ok := Registry64.GetBest(len(src), SIMDAVX2, KernelDIT); ok {
			return entry.Inverse(dst, src, twiddle, scratch, bitrev)
		}
		return inverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 512:
		return inverseAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 1024:
		return inverseAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 2048:
		return inverseAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 4096:
		return inverseAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 8192:
		return inverseAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch, bitrev)
	case 16384:
		return inverseAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}

	return inverseDITComplex64(dst, src, twiddle, scratch, bitrev)
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
		return forwardDITComplex64(dst, src, twiddle, scratch, bitrev)
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
		return inverseDITComplex64(dst, src, twiddle, scratch, bitrev)
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

	return forwardDITComplex128(dst, src, twiddle, scratch, bitrev)
}

func inverseAVX2Complex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return inverseDITComplex128(dst, src, twiddle, scratch, bitrev)
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
		return forwardDITComplex128(dst, src, twiddle, scratch, bitrev)
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
		return inverseDITComplex128(dst, src, twiddle, scratch, bitrev)
	case KernelStockham:
		return inverseStockhamComplex128(dst, src, twiddle, scratch, bitrev)
	default:
		return false
	}
}
