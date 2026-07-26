//go:build amd64 && !purego

package amd64

// AVX-512 size-16 and size-32 codelets (complex128). A ZMM register holds four
// complex128, so both transforms are fully register-resident: the kernels load
// once, run every stage in registers and store once, never touching the
// `scratch` slice. They require AVX512F only; callers gate on
// cpu.Features.HasAVX512. Each returns false for any length other than its own.

// ForwardAVX512Size16Radix4Complex128Asm computes a 16-point forward FFT using
// two radix-4 AVX-512 stages.
//
//go:noescape
func ForwardAVX512Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

// InverseAVX512Size16Radix4Complex128Asm computes a 16-point inverse FFT using
// two radix-4 AVX-512 stages, including the 1/16 scaling.
//
//go:noescape
func InverseAVX512Size16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool

// ForwardAVX512Size32Radix4Then2Complex128Asm computes a 32-point forward FFT
// using two radix-4 AVX-512 stages followed by one radix-2 stage.
//
//go:noescape
func ForwardAVX512Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool

// InverseAVX512Size32Radix4Then2Complex128Asm computes a 32-point inverse FFT
// using two radix-4 AVX-512 stages followed by one radix-2 stage, including the
// 1/32 scaling.
//
//go:noescape
func InverseAVX512Size32Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool
