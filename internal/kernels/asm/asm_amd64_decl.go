//go:build amd64 && fft_asm && !purego

package asm

//go:noescape
func forwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// Size-specific SSE2 kernels (forward, complex64)
//
//go:noescape
func forwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// Size-specific SSE2 kernels (inverse, complex64)
//
//go:noescape
func inverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// Size-specific SSE2 kernels (forward, complex64, size 64)
//
//go:noescape
func forwardSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// Size-specific SSE2 kernels (inverse, complex64, size 64)
//
//go:noescape
func inverseSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// Size-specific AVX2 kernels (forward, complex64)
//
//go:noescape
func forwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// Size-specific AVX2 kernels (inverse, complex64)
//
//go:noescape
func inverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// Size-specific AVX2 kernels (forward, complex128)
//
//go:noescape
func forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// Size-specific AVX2 kernels (inverse, complex128)
//
//go:noescape
func inverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
