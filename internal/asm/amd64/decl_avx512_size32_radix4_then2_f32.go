//go:build amd64 && !purego

package amd64

// AVX-512 size-32 radix-4-then-2 codelets (complex64). They require AVX512F
// only; callers gate on cpu.Features.HasAVX512. Both directions return false
// for any length other than 32. All 32 points stay in four ZMM registers, so
// the transform never touches scratch and works in place.

//go:noescape
func ForwardAVX512Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX512Size32Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool
