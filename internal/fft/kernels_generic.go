//go:build !amd64 && !arm64 && !386

package fft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
)

func selectKernelsComplex64(features cpu.Features) Kernels[complex64] {
	_ = features
	return autoKernelComplex64(KernelAuto)
}

func selectKernelsComplex128(features cpu.Features) Kernels[complex128] {
	_ = features
	return autoKernelComplex128(KernelAuto)
}

func selectKernelsComplex64WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex64] {
	_ = features
	return autoKernelComplex64(strategy)
}

func selectKernelsComplex128WithStrategy(features cpu.Features, strategy KernelStrategy) Kernels[complex128] {
	_ = features
	return autoKernelComplex128(strategy)
}
