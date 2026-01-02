//go:build 386 && asm && !purego

package fft

import kasm "github.com/MeKo-Christian/algo-fft/internal/asm/x86"

// Wrapper functions that call the x86 assembly implementations
func forwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.ForwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}

func inverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool {
	return kasm.InverseSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
}
