//go:build amd64 && asm && !purego

package amd64

// Aliases for radix-4-then-2 kernels (backed by mixed-radix asm symbols).

func ForwardSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardSSE3Size32Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func InverseSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseSSE3Size32Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func ForwardSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardSSE3Size128Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func InverseSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseSSE3Size128Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func ForwardAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardAVX2Size32Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func InverseAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseAVX2Size32Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func ForwardAVX2Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardAVX2Size128Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func InverseAVX2Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseAVX2Size128Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func ForwardAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func InverseAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func ForwardAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func InverseAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func ForwardAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func InverseAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch)
}

func ForwardAVX2Size8192Radix4Then2ParamsComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return ForwardAVX2Size8192Mixed24ParamsComplex64Asm(dst, src, twiddle, scratch)
}

func InverseAVX2Size8192Radix4Then2ParamsComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return InverseAVX2Size8192Mixed24ParamsComplex64Asm(dst, src, twiddle, scratch)
}

func ForwardAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return ForwardAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch)
}

func InverseAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return InverseAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch)
}

func ForwardSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return ForwardSSE2Size32Mixed24Complex128Asm(dst, src, twiddle, scratch)
}

func InverseSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return InverseSSE2Size32Mixed24Complex128Asm(dst, src, twiddle, scratch)
}

func ForwardSSE2Size128Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return ForwardSSE2Size128Mixed24Complex128Asm(dst, src, twiddle, scratch)
}

func InverseSSE2Size128Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return InverseSSE2Size128Mixed24Complex128Asm(dst, src, twiddle, scratch)
}

func ForwardAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return ForwardAVX2Size32Mixed24Complex128Asm(dst, src, twiddle, scratch)
}

func InverseAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return InverseAVX2Size32Mixed24Complex128Asm(dst, src, twiddle, scratch)
}
