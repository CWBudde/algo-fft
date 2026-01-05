//go:build !amd64 || !asm || purego

package kernels

func radix3AVX2Available() bool {
	return false
}

func butterfly3ForwardAVX2Complex64Slices(y0, y1, y2, a0, a1, a2 []complex64) {
}

func butterfly3InverseAVX2Complex64Slices(y0, y1, y2, a0, a1, a2 []complex64) {
}
