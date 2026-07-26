//go:build amd64 && !purego

package amd64

// AVX-512 size-64 codelets (complex64), two variants of the same 8x8 four-step
// structure: the whole transform stays in eight ZMM registers between the
// initial loads and the final stores, so there is no bit-reversal pass and no
// working buffer (dst == src is safe, scratch is never touched). Radix4
// decomposes each vertical 8-point sub-FFT as radix-4 then radix-2, Radix2 as
// three radix-2 stages. See internal/asm/amd64/avx512_f32_size64_radix4.s.
//
// All four require AVX512F only; callers gate on cpu.Features.HasAVX512. Any
// length other than 64 returns false so the caller can fall back.

//go:noescape
func ForwardAVX512Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX512Size64Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX512Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX512Size64Radix2Complex64Asm(dst, src, twiddle, scratch []complex64) bool
