//go:build amd64 && !purego

package amd64

// AVX-512 size-16 complex64 codelets. They require only AVX512F; callers gate
// on cpu.Features.HasAVX512. Both return false for any length other than 16 (or
// when dst/twiddle/scratch are shorter than 16) so the caller can fall back.
//
// The transform is register resident (16 complex64 = two ZMM registers), so
// in-place calls (dst == src) work without touching scratch.

//go:noescape
func ForwardAVX512Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX512Size16Radix16Complex64Asm(dst, src, twiddle, scratch []complex64) bool
