//go:build amd64 && fft_asm && !purego

package amd64

// AVX2 and SSE2 FFT kernels for complex64 and complex128 data types.

// ============================================================================
// Generic FFT Kernels (Variable Size)
// ============================================================================

//go:noescape
func forwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// ============================================================================
// Stockham FFT Kernels
// ============================================================================

//go:noescape
func forwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// ============================================================================
// Size-Specific FFT Kernels (Complex64)
// ============================================================================

// --- SSE2 Kernels (Complex64) ---

//go:noescape
func forwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardSSE2Size128Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseSSE2Size128Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// --- AVX2 Kernels (Complex64) ---

//go:noescape
func forwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// ============================================================================
// Size-Specific FFT Kernels (Complex128)
// ============================================================================

// --- SSE2 Kernels (Complex128) ---

//go:noescape
func forwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// --- AVX2 Kernels (Complex128) ---

//go:noescape
func forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func forwardAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func inverseAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// ============================================================================
// Element-wise Operations
// ============================================================================

// Complex array multiplication (element-wise) - AVX2 optimized.
// These functions are available on any amd64 platform, with runtime
// CPU feature detection for optimal path selection.

//go:noescape
func ComplexMulArrayComplex64AVX2Asm(dst, a, b []complex64)

//go:noescape
func ComplexMulArrayInPlaceComplex64AVX2Asm(dst, src []complex64)

//go:noescape
func ComplexMulArrayComplex128AVX2Asm(dst, a, b []complex128)

//go:noescape
func ComplexMulArrayInPlaceComplex128AVX2Asm(dst, src []complex128)
