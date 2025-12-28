//go:build amd64 && fft_asm && !purego

package fft

// Assembly constants (defined in asm_amd64.s)
var (
	half32 float32 // 0.5f for scaling
	one32  float32 // 1.0f for scaling
)

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

//go:noescape
func asmCopyComplex64(dst, src *complex64)

//go:noescape
func asmForward2Complex64(dst, src *complex64) bool

//go:noescape
func asmInverse2Complex64(dst, src *complex64) bool

// Size-specific AVX2 kernels (forward, complex64)
//
//go:noescape
func forwardAVX2Size8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func forwardAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// Size-specific AVX2 kernels (inverse, complex64)
//
//go:noescape
func inverseAVX2Size8Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size32Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size64Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseAVX2Size128Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
