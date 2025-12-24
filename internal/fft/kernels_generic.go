//go:build !amd64 && !arm64

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	_ = features
	return autoKernelComplex64(KernelAuto)
}

func selectKernelsComplex128(features Features) Kernels[complex128] {
	_ = features
	return autoKernelComplex128(KernelAuto)
}

func selectKernelsComplex64WithStrategy(features Features, strategy KernelStrategy) Kernels[complex64] {
	_ = features
	return autoKernelComplex64(strategy)
}

func selectKernelsComplex128WithStrategy(features Features, strategy KernelStrategy) Kernels[complex128] {
	_ = features
	return autoKernelComplex128(strategy)
}
