//go:build !amd64 && !arm64 && !386

package fft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/kernels"
)

func selectKernelsComplex64(features cpu.Features) kernels.Kernels[complex64] {
	_ = features
	return autoKernelComplex64(fftypes.KernelAuto)
}

func selectKernelsComplex128(features cpu.Features) kernels.Kernels[complex128] {
	_ = features
	return autoKernelComplex128(fftypes.KernelAuto)
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex64] {
	_ = features
	return autoKernelComplex64(strategy)
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy fftypes.KernelStrategy) kernels.Kernels[complex128] {
	_ = features
	return autoKernelComplex128(strategy)
}
