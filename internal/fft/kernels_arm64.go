//go:build arm64 && purego

package fft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
	m "github.com/cwbudde/algo-fft/internal/math"
)

func selectKernelsComplex64(features cpu.Features) kernels.Kernels[complex64] {
	auto := autoKernelComplex64(fftypes.KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(forwardNEONComplex64, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128(features cpu.Features) kernels.Kernels[complex128] {
	auto := autoKernelComplex128(fftypes.KernelAuto)
	if features.HasNEON && !features.ForceGeneric {
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(forwardNEONComplex128, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex128, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex64] {
	auto := autoKernelComplex64(strategy)
	if features.HasNEON && !features.ForceGeneric {
		return kernels.Kernels[complex64]{
			Forward: fallbackKernel(forwardNEONComplex64, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex64, auto.Inverse),
		}
	}

	return auto
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex128] {
	auto := autoKernelComplex128(strategy)
	if features.HasNEON && !features.ForceGeneric {
		return kernels.Kernels[complex128]{
			Forward: fallbackKernel(forwardNEONComplex128, auto.Forward),
			Inverse: fallbackKernel(inverseNEONComplex128, auto.Inverse),
		}
	}

	return auto
}

func forwardNEONComplex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return kernels.ForwardDITComplex64(dst, src, twiddle, scratch)
}

func inverseNEONComplex64(dst, src, twiddle, scratch []complex64) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return kernels.InverseDITComplex64(dst, src, twiddle, scratch)
}

func forwardNEONComplex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return kernels.ForwardDITComplex128(dst, src, twiddle, scratch)
}

func inverseNEONComplex128(dst, src, twiddle, scratch []complex128) bool {
	if !m.IsPowerOf2(len(src)) {
		return false
	}

	return kernels.InverseDITComplex128(dst, src, twiddle, scratch)
}
