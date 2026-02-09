//go:build 386 && (!asm || purego)

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
