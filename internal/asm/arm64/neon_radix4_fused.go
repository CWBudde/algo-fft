//go:build arm64 && !purego

package arm64

func forwardNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch []complex64, n int) bool {
	return neonRadix4ForwardC64(dst, src, twiddle, scratch, -n)
}

func inverseNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch []complex64, n int) bool {
	return neonRadix4InverseC64(dst, src, twiddle, scratch, -n, 1/float32(n))
}

func forwardNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch []complex128, n int) bool {
	return neonRadix4ForwardC128(dst, src, twiddle, scratch, -n)
}

func inverseNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch []complex128, n int) bool {
	return neonRadix4InverseC128(dst, src, twiddle, scratch, -n, 1/float64(n))
}

// ForwardNEONSize32Radix4FusedComplex64Asm computes a fused-tail size-32 FFT.
func ForwardNEONSize32Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return forwardNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 32)
}

// InverseNEONSize32Radix4FusedComplex64Asm computes a normalized fused-tail size-32 inverse FFT.
func InverseNEONSize32Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return inverseNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 32)
}

// ForwardNEONSize128Radix4FusedComplex64Asm computes a fused-tail size-128 FFT.
func ForwardNEONSize128Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return forwardNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 128)
}

// InverseNEONSize128Radix4FusedComplex64Asm computes a normalized fused-tail size-128 inverse FFT.
func InverseNEONSize128Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return inverseNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 128)
}

// ForwardNEONSize512Radix4FusedComplex64Asm computes a fused-tail size-512 FFT.
func ForwardNEONSize512Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return forwardNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 512)
}

// InverseNEONSize512Radix4FusedComplex64Asm computes a normalized fused-tail size-512 inverse FFT.
func InverseNEONSize512Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return inverseNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 512)
}

// ForwardNEONSize2048Radix4FusedComplex64Asm computes a fused-tail size-2048 FFT.
func ForwardNEONSize2048Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return forwardNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 2048)
}

// InverseNEONSize2048Radix4FusedComplex64Asm computes a normalized fused-tail size-2048 inverse FFT.
func InverseNEONSize2048Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return inverseNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 2048)
}

// ForwardNEONSize8192Radix4FusedComplex64Asm computes a fused-tail size-8192 FFT.
func ForwardNEONSize8192Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return forwardNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 8192)
}

// InverseNEONSize8192Radix4FusedComplex64Asm computes a normalized fused-tail size-8192 inverse FFT.
func InverseNEONSize8192Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return inverseNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 8192)
}

// ForwardNEONSize32768Radix4FusedComplex64Asm computes a fused-tail size-32768 FFT.
func ForwardNEONSize32768Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return forwardNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 32768)
}

// InverseNEONSize32768Radix4FusedComplex64Asm computes a normalized fused-tail size-32768 inverse FFT.
func InverseNEONSize32768Radix4FusedComplex64Asm(dst, src, twiddle, scratch []complex64) bool {
	return inverseNEONRadix4FusedTailComplex64(dst, src, twiddle, scratch, 32768)
}

// ForwardNEONSize32Radix4FusedComplex128Asm computes a fused-tail size-32 FFT.
func ForwardNEONSize32Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return forwardNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 32)
}

// InverseNEONSize32Radix4FusedComplex128Asm computes a normalized fused-tail size-32 inverse FFT.
func InverseNEONSize32Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return inverseNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 32)
}

// ForwardNEONSize128Radix4FusedComplex128Asm computes a fused-tail size-128 FFT.
func ForwardNEONSize128Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return forwardNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 128)
}

// InverseNEONSize128Radix4FusedComplex128Asm computes a normalized fused-tail size-128 inverse FFT.
func InverseNEONSize128Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return inverseNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 128)
}

// ForwardNEONSize512Radix4FusedComplex128Asm computes a fused-tail size-512 FFT.
func ForwardNEONSize512Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return forwardNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 512)
}

// InverseNEONSize512Radix4FusedComplex128Asm computes a normalized fused-tail size-512 inverse FFT.
func InverseNEONSize512Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return inverseNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 512)
}

// ForwardNEONSize2048Radix4FusedComplex128Asm computes a fused-tail size-2048 FFT.
func ForwardNEONSize2048Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return forwardNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 2048)
}

// InverseNEONSize2048Radix4FusedComplex128Asm computes a normalized fused-tail size-2048 inverse FFT.
func InverseNEONSize2048Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return inverseNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 2048)
}

// ForwardNEONSize8192Radix4FusedComplex128Asm computes a fused-tail size-8192 FFT.
func ForwardNEONSize8192Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return forwardNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 8192)
}

// InverseNEONSize8192Radix4FusedComplex128Asm computes a normalized fused-tail size-8192 inverse FFT.
func InverseNEONSize8192Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return inverseNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 8192)
}

// ForwardNEONSize32768Radix4FusedComplex128Asm computes a fused-tail size-32768 FFT.
func ForwardNEONSize32768Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return forwardNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 32768)
}

// InverseNEONSize32768Radix4FusedComplex128Asm computes a normalized fused-tail size-32768 inverse FFT.
func InverseNEONSize32768Radix4FusedComplex128Asm(dst, src, twiddle, scratch []complex128) bool {
	return inverseNEONRadix4FusedTailComplex128(dst, src, twiddle, scratch, 32768)
}
