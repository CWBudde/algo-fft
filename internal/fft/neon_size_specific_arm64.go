//go:build arm64 && fft_asm && !purego

package fft

// Size-specific NEON kernels implemented in Go to mirror AVX2 size coverage.
// These provide radix-2, radix-4, and mixed-radix variants for ARM64 paths.

// Radix-2, size 16.
func forwardNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT16Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT16Complex64(dst, src, twiddle, scratch, bitrev)
}

// Radix-2, size 32.
func forwardNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT32Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT32Complex64(dst, src, twiddle, scratch, bitrev)
}

// Radix-2, size 64.
func forwardNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT64Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT64Complex64(dst, src, twiddle, scratch, bitrev)
}

// Radix-2, size 128.
func forwardNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return forwardDIT128Complex64(dst, src, twiddle, scratch, bitrev)
}

func inverseNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return inverseDIT128Complex64(dst, src, twiddle, scratch, bitrev)
}
