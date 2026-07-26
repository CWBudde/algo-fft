//go:build amd64 && !purego

package amd64

// AVX-512 size-128 and size-256 radix-8-then-2 codelets (complex64).
//
// Both sizes use a radix-2 DIT vectorised so that one ZMM holds 8 consecutive
// complex64 of the bit-reversed working array; the first three stages are fused
// into an in-register 8-point DFT and the remaining stages are elementwise
// between whole ZMM registers. Size 128 is fully register resident (16 ZMM) and
// never touches memory between its load and its store; size 256 runs as two
// such sub-transforms plus one final radix-2 stage.
//
// These require AVX512F only; callers gate on cpu.Features.HasAVX512. Each
// returns false for any length other than the one it handles.

//go:noescape
func ForwardAVX512Size128Radix8Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX512Size128Radix8Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func ForwardAVX512Size256Radix8Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool

//go:noescape
func InverseAVX512Size256Radix8Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool
