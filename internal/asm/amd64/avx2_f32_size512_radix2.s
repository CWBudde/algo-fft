//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-512 Radix-2 FFT Kernels for AMD64 (wrapper)
// ===========================================================================
//
// These wrappers validate the expected size (512) and delegate to the generic
// AVX2 radix-2 implementation. This keeps the size-specific dispatch and
// codelet registry consistent while we work toward a fully-unrolled kernel.
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 512, complex64
TEXT ·ForwardAVX2Size512Radix2Complex64Asm(SB), NOSPLIT, $0-121
	MOVQ src+32(FP), AX
	CMPQ AX, $512
	JNE  size512_r2_forward_return_false
	JMP  ·ForwardAVX2Complex64Asm(SB)

size512_r2_forward_return_false:
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 512, complex64
TEXT ·InverseAVX2Size512Radix2Complex64Asm(SB), NOSPLIT, $0-121
	MOVQ src+32(FP), AX
	CMPQ AX, $512
	JNE  size512_r2_inverse_return_false
	JMP  ·InverseAVX2Complex64Asm(SB)

size512_r2_inverse_return_false:
	MOVB $0, ret+120(FP)
	RET
