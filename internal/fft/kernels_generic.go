//go:build !amd64 && !arm64

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	_ = features
	return autoKernelComplex64()
}

func selectKernelsComplex128(features Features) Kernels[complex128] {
	_ = features
	return autoKernelComplex128()
}
