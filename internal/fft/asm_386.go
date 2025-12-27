//go:build 386 && fft_asm && !purego

package fft

// Assembly constants (defined in asm_386.s)
var (
	half32 float32 // 0.5f for scaling
	one32  float32 // 1.0f for scaling
)

//go:noescape
func forwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func inverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
