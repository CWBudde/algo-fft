//go:build arm64 && fft_asm && !purego

package fft

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

// Size-specific NEON kernels (forward, complex64)
//
//go:noescape
func forwardNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// Size-specific NEON kernels (inverse, complex64)
//
//go:noescape
func inverseNEONSize16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONSize128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
