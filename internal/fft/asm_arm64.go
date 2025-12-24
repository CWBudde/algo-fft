//go:build arm64 && fft_asm

package fft

//go:noescape
func forwardNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseNEONComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardNEONComplex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseNEONComplex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
