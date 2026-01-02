//go:build 386 && asm && !purego

package x86

// NOTE: These are Go declarations for 386 (x86) assembly routines implemented in the *.s files in this directory.

//go:noescape
func ForwardSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseSSE2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
