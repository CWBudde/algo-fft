//go:build arm64 && (!asm || purego)

package fft

import (
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	m "github.com/MeKo-Christian/algo-fft/internal/math"
)

func selectKernelsComplex64(features cpu.Features) Kernels[complex64] {
	auto := autoKernelComplex64(KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardNEONComplex64, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128(features cpu.Features) Kernels[complex128] {
	auto := autoKernelComplex128(KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardNEONComplex128, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex128, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex64]{
			Forward: fallbackKernel(forwardNEONComplex64, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasNEON && !features.ForceGeneric {
		return Kernels[complex128]{
			Forward: fallbackKernel(forwardNEONComplex128, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex128, auto.Inverse),
		}
	}

	return auto
}

func forwardNEONComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return forwardDITComplex64(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONComplex64(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return inverseDITComplex64(dst, src, twiddle, scratch, bitrev)
}

func forwardNEONComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return forwardDITComplex128(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONComplex128(dst, src, twiddle, scratch []complex128, bitrev []int) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return inverseDITComplex128(dst, src, twiddle, scratch, bitrev)
}
