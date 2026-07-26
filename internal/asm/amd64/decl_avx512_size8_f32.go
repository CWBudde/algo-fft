//go:build amd64 && !purego

package amd64

// AVX-512 size-8 codelets (complex64). All eight complex64 values live in a
// single ZMM register, so the transforms are register resident: one load, one
// store, no scratch traffic. They require only AVX512F; callers gate on
// cpu.Features.HasAVX512. Any length other than 8 returns false.

//go:noescape
func ForwardAVX512Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX512Size8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool
