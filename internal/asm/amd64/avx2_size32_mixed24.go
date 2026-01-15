//go:build amd64 && asm && !purego

package amd64

// ForwardAVX2Size32Mixed24Complex64Asm wraps the size-32 radix-2 AVX2 kernel.
// This keeps the mixed-2/4 entry available even without a dedicated AVX2 path.
func ForwardAVX2Size32Mixed24Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	return ForwardAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
}

// InverseAVX2Size32Mixed24Complex64Asm wraps the size-32 radix-2 AVX2 kernel.
func InverseAVX2Size32Mixed24Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	return InverseAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch)
}

// ForwardAVX2Size32Mixed24Complex128Asm wraps the size-32 AVX2 kernel for complex128.
func ForwardAVX2Size32Mixed24Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	return ForwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch)
}

// InverseAVX2Size32Mixed24Complex128Asm wraps the size-32 AVX2 kernel for complex128.
func InverseAVX2Size32Mixed24Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	const n = 32
	if len(dst) < n || len(src) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	return InverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch)
}
