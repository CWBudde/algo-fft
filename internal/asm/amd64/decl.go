//go:build amd64 && asm && !purego

package amd64

// AVX2 and SSE2/SSE3 FFT kernels for complex64 and complex128 data types.

// ============================================================================
// Generic FFT Kernels (Variable Size)
// ============================================================================

//go:noescape
func ForwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Complex64Radix4Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Complex64Radix4Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Complex128Radix4Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func ForwardSSE2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

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

// --- SSE2/SSE3 Kernels (Complex64) ---

//go:noescape
func ForwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size32Radix32Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size256Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseSSE3Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardSSE2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

// --- AVX2 Kernels (Complex64) ---

//go:noescape
func ForwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size8Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size16Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size32Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size32Radix32Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size32Radix32Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size128Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size128Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size256Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size256Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size512Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size512Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size512Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size8192Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

// Params-enabled variants for size 8192: use pre-broadcast twiddle data for improved SIMD efficiency.
// The twiddle slice contains pre-broadcast factors for stages 2-7, eliminating runtime
// index computation and scalar-to-vector broadcasts in the hot loop.

//go:noescape
func ForwardAVX2Size8192Radix4Then2ParamsComplex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size8192Radix4Then2ParamsComplex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size1024Radix32x32Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size1024Radix32x32Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

// ============================================================================
// Size-Specific FFT Kernels (Complex128)
// ============================================================================

//go:noescape
func ForwardAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size512Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size2048Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size2048Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size8192Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size8192Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size16384Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size16384Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

// ============================================================================
// Matrix Transpose Operations (for Six-Step FFT)
// ============================================================================

// Transpose64x64Complex64AVX2Asm transposes a 64×64 matrix of complex64 values.
// Uses AVX2 8×8 block processing for efficient cache utilization.
//
//go:noescape
func Transpose64x64Complex64AVX2Asm(dst, src []complex64) bool

// TransposeTwiddle64x64Complex64AVX2Asm performs fused transpose + twiddle multiply:
//
//	dst[i,j] = src[j,i] * twiddle[(i*j) % 4096]
//
// Used in forward six-step FFT (steps 3-4 fused).
//
//go:noescape
func TransposeTwiddle64x64Complex64AVX2Asm(dst, src, twiddle []complex64) bool

// TransposeTwiddleConj64x64Complex64AVX2Asm performs fused transpose + conjugate twiddle:
//
//	dst[i,j] = src[j,i] * conj(twiddle[(i*j) % 4096])
//
// Used in inverse six-step FFT (steps 3-4 fused).
//
//go:noescape
func TransposeTwiddleConj64x64Complex64AVX2Asm(dst, src, twiddle []complex64) bool

// Transpose128x128Complex64AVX2Asm transposes a 128×128 matrix of complex64 values.
// Uses AVX2 4×4 block processing for efficient cache utilization.
// Used in size-16384 six-step FFT.
//
//go:noescape
func Transpose128x128Complex64AVX2Asm(dst, src []complex64) bool

// TransposeTwiddle128x128Complex64AVX2Asm performs fused transpose + twiddle multiply:
//
//	dst[i,j] = src[j,i] * twiddle[(i*j) % 16384]
//
// Used in forward six-step FFT for size 16384 (steps 3-4 fused).
//
//go:noescape
func TransposeTwiddle128x128Complex64AVX2Asm(dst, src, twiddle []complex64) bool

// TransposeTwiddleConj128x128Complex64AVX2Asm performs fused transpose + conjugate twiddle:
//
//	dst[i,j] = src[j,i] * conj(twiddle[(i*j) % 16384])
//
// Used in inverse six-step FFT for size 16384 (steps 3-4 fused).
//
//go:noescape
func TransposeTwiddleConj128x128Complex64AVX2Asm(dst, src, twiddle []complex64) bool

// ============================================================================
// Size-Specific FFT Kernels (Complex128)
// ============================================================================

// --- SSE2 Kernels (Complex128) ---

//go:noescape
func ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size16Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size16Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size32Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size32Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size128Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size128Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size128Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size128Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardSSE2Size256Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseSSE2Size256Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

// --- AVX2 Kernels (Complex128) ---

//go:noescape
func ForwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size16Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size16Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size64Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size128Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func ForwardAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

//go:noescape
func InverseAVX2Size256Radix2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

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

// Complex array scaling (element-wise) with scalar factors.

//go:noescape
func ScaleComplex64AVX2Asm(dst []complex64, scale float32)

//go:noescape
func ScaleComplex64SSE2Asm(dst []complex64, scale float32)

//go:noescape
func ScaleComplex128AVX2Asm(dst []complex128, scale float64)

//go:noescape
func ScaleComplex128SSE2Asm(dst []complex128, scale float64)

// Inverse real FFT repack helpers (complex64 only).

//go:noescape
func InverseRepackComplex64AVX2Asm(dst, src, weight []complex64, kStartMax int)

//go:noescape
func InverseRepackComplex64SSE2Asm(dst, src, weight []complex64, kStartMax int)

// ============================================================================
// Radix-3 FFT Butterfly Operations
// ============================================================================

// Butterfly3ForwardAVX2Complex64 processes 4 radix-3 forward butterflies in parallel.
// Each input slice must have length >= 4 (representing 4 complex64 values).
// The function computes:
//
//	for i in 0..3:
//	  t1 = a1[i] + a2[i]
//	  t2 = a1[i] - a2[i]
//	  y0[i] = a0[i] + t1
//	  base = a0[i] + (-0.5)*t1
//	  y1[i] = base + (0 - i*sqrt(3)/2)*t2
//	  y2[i] = base - (0 - i*sqrt(3)/2)*t2
//
//go:noescape
func Butterfly3ForwardAVX2Complex64(y0, y1, y2, a0, a1, a2 []complex64)

// Butterfly3InverseAVX2Complex64 processes 4 radix-3 inverse butterflies in parallel.
// Each input slice must have length >= 4.
// Uses the conjugate coefficient: 0 + i*sqrt(3)/2
//
//go:noescape
func Butterfly3InverseAVX2Complex64(y0, y1, y2, a0, a1, a2 []complex64)

// ============================================================================
// Radix-5 FFT Butterfly Operations
// ============================================================================

// Butterfly5ForwardAVX2Complex64 processes 2 radix-5 forward butterflies in parallel.
// Each input slice must have length >= 2.
//
//go:noescape
func Butterfly5ForwardAVX2Complex64(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4 []complex64)

// Butterfly5InverseAVX2Complex64 processes 2 radix-5 inverse butterflies in parallel.
// Each input slice must have length >= 2.
//
//go:noescape
func Butterfly5InverseAVX2Complex64(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4 []complex64)

// ============================================================================
// Size-384 Mixed-Radix (128×3) FFT Operations
// ============================================================================

// ForwardAVX2Size384MixedComplex64Asm validates parameters for size-384 FFT.
// Returns true if the kernel can handle this size, false otherwise.
//
//go:noescape
func ForwardAVX2Size384MixedComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// InverseAVX2Size384MixedComplex64Asm validates parameters for size-384 IFFT.
//
//go:noescape
func InverseAVX2Size384MixedComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// ApplyTwiddle384Complex64Asm applies twiddle factors for 384-point mixed-radix FFT.
// Multiplies data[128+k] by twiddle[k] and data[256+k] by twiddle[2*k] for k=0..127.
//
//go:noescape
func ApplyTwiddle384Complex64Asm(data, twiddle []complex64)

// Radix3Butterflies384ForwardComplex64Asm performs 128 radix-3 forward butterflies
// across the three 128-element sub-arrays of a 384-element array.
//
//go:noescape
func Radix3Butterflies384ForwardComplex64Asm(data []complex64)

// Radix3Butterflies384InverseComplex64Asm performs 128 radix-3 inverse butterflies
// across the three 128-element sub-arrays of a 384-element array.
//
//go:noescape
func Radix3Butterflies384InverseComplex64Asm(data []complex64)

// ForwardAVX2Size384MixedComplex128Asm validates parameters for size-384 FFT.
// Returns true if the kernel can handle this size, false otherwise.
//
//go:noescape
func ForwardAVX2Size384MixedComplex128Asm(dst, src, twiddle, scratch []complex128) bool

// InverseAVX2Size384MixedComplex128Asm validates parameters for size-384 IFFT.
//
//go:noescape
func InverseAVX2Size384MixedComplex128Asm(dst, src, twiddle, scratch []complex128) bool

// ApplyTwiddle384Complex128Asm applies twiddle factors for 384-point mixed-radix FFT.
// Multiplies data[128+k] by twiddle[k] and data[256+k] by twiddle[2*k] for k=0..127.
//
//go:noescape
func ApplyTwiddle384Complex128Asm(data, twiddle []complex128)

// Radix3Butterflies384ForwardComplex128Asm performs 128 radix-3 forward butterflies
// across the three 128-element sub-arrays of a 384-element array.
//
//go:noescape
func Radix3Butterflies384ForwardComplex128Asm(data []complex128)

// Radix3Butterflies384InverseComplex128Asm performs 128 radix-3 inverse butterflies
// across the three 128-element sub-arrays of a 384-element array.
//
//go:noescape
func Radix3Butterflies384InverseComplex128Asm(data []complex128)
