//go:build amd64 && asm && !purego

package amd64

// ForwardAVX2Size128Radix2Complex64Asm runs a size-128 radix-2 FFT using the
// generic AVX2 complex64 kernel with a fixed transform length.
func ForwardAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	const n = 128
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	return ForwardAVX2Complex64Asm(dst[:n], src[:n], twiddle[:n], scratch[:n], nil)
}

// InverseAVX2Size128Radix2Complex64Asm runs a size-128 radix-2 inverse FFT using
// the generic AVX2 complex64 kernel with a fixed transform length.
func InverseAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	const n = 128
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	return InverseAVX2Complex64Asm(dst[:n], src[:n], twiddle[:n], scratch[:n], nil)
}
