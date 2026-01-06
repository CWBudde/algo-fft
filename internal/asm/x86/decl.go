//go:build 386 && asm && !purego

package x86

// ============================================================================
// Generic FFT Kernels (Variable Size)
// ============================================================================

//go:noescape

func ForwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func InverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func ForwardSSEComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func InverseSSEComplex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func ForwardAVX2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// --- Complex64 Kernels ---

//go:noescape

func ForwardSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func InverseSSE2Size2Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func ForwardSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func InverseSSE2Size4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func ForwardSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func InverseSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func ForwardSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func InverseSSE2Size8Radix2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func ForwardSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func InverseSSE2Size16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func ForwardSSE2Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape

func InverseSSE2Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// --- Complex128 Kernels ---

//go:noescape

func ForwardSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape

func InverseSSE2Size2Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape

func ForwardSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape

func InverseSSE2Size4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape

func ForwardSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape

func InverseSSE2Size8Radix2Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape

func ForwardSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape

func InverseSSE2Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
