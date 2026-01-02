//go:build amd64 && asm && !purego

package amd64

// AVX2 and SSE2 FFT kernels for complex64 and complex128 data types.

// ============================================================================
// Generic FFT Kernels (Variable Size)
// ============================================================================

//go:noescape
func ForwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseSSE2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// ============================================================================
// Stockham FFT Kernels
// ============================================================================

//go:noescape
func ForwardAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2StockhamComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// ============================================================================
// Size-Specific FFT Kernels (Complex64)
// ============================================================================

// --- SSE2 Kernels (Complex64) ---

//go:noescape
func ForwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Size16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Size16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Size128Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Size128Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// --- AVX2 Kernels (Complex64) ---

//go:noescape
func ForwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// ============================================================================
// Size-Specific FFT Kernels (Complex128)
// ============================================================================

// --- SSE2 Kernels (Complex128) ---

//go:noescape
func ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// --- AVX2 Kernels (Complex128) ---

//go:noescape
func ForwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

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
