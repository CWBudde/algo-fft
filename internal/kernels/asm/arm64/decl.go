//go:build arm64 && fft_asm && !purego

package arm64

// NOTE: These are Go declarations for ARM64 assembly routines implemented in the *.s files in this directory.

//go:noescape
func forwardNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func neonComplexMul2Asm(dst, a, b *complex64)

//go:noescape
func forwardNEONComplex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseNEONComplex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// Size-specific complex64 NEON kernels.

//go:noescape
func forwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize32MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize32MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize128MixedRadix24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
