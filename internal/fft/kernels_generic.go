//go:build !amd64 && !arm64

package fft

func selectKernelsComplex64(features Features) Kernels[complex64] {
	_ = features
	return Kernels[complex64]{
		Forward: forwardStockhamComplex64,
		Inverse: inverseStockhamComplex64,
	}
}

func selectKernelsComplex128(features Features) Kernels[complex128] {
	_ = features
	return Kernels[complex128]{
		Forward: forwardStockhamComplex128,
		Inverse: inverseStockhamComplex128,
	}
}
