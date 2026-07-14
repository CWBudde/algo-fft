//go:build amd64 && !purego

package amd64

// AVX-512 generic radix-2 DIT FFT kernels. All require only AVX512F;
// callers must gate on cpu.Features.HasAVX512. The bitrev slice must hold a
// radix-2 bit-reversal index table of at least len(src) entries. Sizes below
// 16 (or non-powers of two) return false so callers can fall back.

//go:noescape
func ForwardAVX512Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func InverseAVX512Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool

//go:noescape
func ForwardAVX512Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool

//go:noescape
func InverseAVX512Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
