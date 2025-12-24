//go:build !amd64 && !arm64

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	_ = features
	return Kernels[complex64]{
		Forward: stubKernel[complex64],
		Inverse: stubKernel[complex64],
	}
}

func selectKernelsComplex128(features Features) Kernels[complex128] {
	_ = features
	return Kernels[complex128]{
		Forward: stubKernel[complex128],
		Inverse: stubKernel[complex128],
	}
}
