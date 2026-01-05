//go:build !amd64 || !asm || purego

package kernels

func radix5AVX2Available() bool {
	return false
}

func butterfly5ForwardAVX2Complex64Slices(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4 []complex64) {
}

func butterfly5InverseAVX2Complex64Slices(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4 []complex64) {
}
