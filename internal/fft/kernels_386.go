//go:build 386 && (!fft_asm || purego)

package fft

import "github.com/MeKo-Christian/algofft/internal/cpu"

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
