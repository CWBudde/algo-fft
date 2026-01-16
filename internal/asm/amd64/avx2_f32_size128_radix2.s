//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-128 Radix-2 FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Size 128 = 2^7, 7 radix-2 stages
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 128, complex64, radix-2
TEXT ·ForwardAVX2Size128Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size128Radix2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $128
	JNE  size128_avx2_r2_fwd_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $128
	JL   size128_avx2_r2_fwd_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $128
	JL   size128_avx2_r2_fwd_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $128
	JL   size128_avx2_r2_fwd_return_false

	CMPQ R8, R9
	JNE  size128_avx2_r2_fwd_use_dst
	MOVQ R11, R8

size128_avx2_r2_fwd_use_dst:
	// Stage 1 & 2 (Combined) - 32 blocks of 4
	MOVQ R8, SI
	MOVQ $32, CX
	VMOVUPS ·maskNegHiPS(SB), X15

size128_avx2_r2_fwd_stage12_loop:
	MOVQ (R12), DX
	VMOVSD (R9)(DX*8), X0
	MOVQ 8(R12), DX
	VMOVSD (R9)(DX*8), X1
	MOVQ 16(R12), DX
	VMOVSD (R9)(DX*8), X2
	MOVQ 24(R12), DX
	VMOVSD (R9)(DX*8), X3
	ADDQ $32, R12
	// Stage 1
	VMOVAPS X0, X8
	VADDPS  X1, X8, X8
	VMOVAPS X0, X9
	VSUBPS  X1, X9, X9
	VMOVAPS X8, X0
	VMOVAPS X9, X1
	VMOVAPS X2, X8
	VADDPS  X3, X8, X8
	VMOVAPS X2, X9
	VSUBPS  X3, X9, X9
	VMOVAPS X8, X2
	VMOVAPS X9, X3
	// Stage 2
	VMOVAPS X0, X8
	VADDPS  X2, X8, X8
	VMOVAPS X0, X9
	VSUBPS  X2, X9, X9
	VMOVAPS X8, X0
	VMOVAPS X9, X2
	VMOVAPS X3, X10
	VSHUFPS $0xB1, X10, X10, X10
	VXORPS  X15, X10, X10          // t = X3 * -i
	VMOVAPS X1, X8
	VADDPS  X10, X8, X8
	VMOVAPS X1, X9
	VSUBPS  X10, X9, X9
	VMOVAPS X8, X1
	VMOVAPS X9, X3
	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  size128_avx2_r2_fwd_stage12_loop

	// Stage 3 (dist 4) - 16 blocks of 8
	MOVQ R8, SI
	MOVQ $16, CX
size128_avx2_r2_fwd_s3_loop:
	MOVQ $4, DX
size128_avx2_r2_fwd_s3_inner:
	VMOVSD (SI), X0
	VMOVSD 32(SI), X1
	MOVQ $4, AX
	SUBQ DX, AX              // k = 0..3
	SHLQ $4, AX              // k * 128/8 = k * 16
	VMOVSD (R10)(AX*8), X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 32(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_fwd_s3_inner
	ADDQ $32, SI
	DECQ CX
	JNZ size128_avx2_r2_fwd_s3_loop

	// Stage 4 (dist 8) - 8 blocks of 16
	MOVQ R8, SI
	MOVQ $8, CX
size128_avx2_r2_fwd_s4_loop:
	MOVQ $8, DX
size128_avx2_r2_fwd_s4_inner:
	VMOVSD (SI), X0
	VMOVSD 64(SI), X1
	MOVQ $8, AX
	SUBQ DX, AX              // k = 0..7
	SHLQ $3, AX              // k * 128/16 = k * 8
	VMOVSD (R10)(AX*8), X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 64(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_fwd_s4_inner
	ADDQ $64, SI
	DECQ CX
	JNZ size128_avx2_r2_fwd_s4_loop

	// Stage 5 (dist 16) - 4 blocks of 32
	MOVQ R8, SI
	MOVQ $4, CX
size128_avx2_r2_fwd_s5_loop:
	MOVQ $16, DX
size128_avx2_r2_fwd_s5_inner:
	VMOVSD (SI), X0
	VMOVSD 128(SI), X1
	MOVQ $16, AX
	SUBQ DX, AX              // k = 0..15
	SHLQ $2, AX              // k * 128/32 = k * 4
	VMOVSD (R10)(AX*8), X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 128(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_fwd_s5_inner
	ADDQ $128, SI
	DECQ CX
	JNZ size128_avx2_r2_fwd_s5_loop

	// Stage 6 (dist 32) - 2 blocks of 64
	MOVQ R8, SI
	MOVQ $2, CX
size128_avx2_r2_fwd_s6_loop:
	MOVQ $32, DX
size128_avx2_r2_fwd_s6_inner:
	VMOVSD (SI), X0
	VMOVSD 256(SI), X1
	MOVQ $32, AX
	SUBQ DX, AX              // k = 0..31
	SHLQ $1, AX              // k * 128/64 = k * 2
	VMOVSD (R10)(AX*8), X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 256(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_fwd_s6_inner
	ADDQ $256, SI
	DECQ CX
	JNZ size128_avx2_r2_fwd_s6_loop

	// Stage 7 (dist 64) - 1 block of 128
	MOVQ R8, SI
	MOVQ $64, DX
size128_avx2_r2_fwd_s7_inner:
	VMOVSD (SI), X0
	VMOVSD 512(SI), X1
	MOVQ $64, AX
	SUBQ DX, AX              // k = 0..63
	VMOVSD (R10)(AX*8), X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 512(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_fwd_s7_inner

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   size128_avx2_r2_fwd_done
	MOVQ $64, CX
	MOVQ R8, SI
size128_avx2_r2_fwd_copy_loop:
	VMOVUPS (SI), X0
	VMOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ size128_avx2_r2_fwd_copy_loop

size128_avx2_r2_fwd_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size128_avx2_r2_fwd_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 128, complex64, radix-2
TEXT ·InverseAVX2Size128Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size128Radix2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $128
	JNE  size128_avx2_r2_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $128
	JL   size128_avx2_r2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $128
	JL   size128_avx2_r2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $128
	JL   size128_avx2_r2_inv_return_false

	CMPQ R8, R9
	JNE  size128_avx2_r2_inv_use_dst
	MOVQ R11, R8

size128_avx2_r2_inv_use_dst:
	// Stage 1 & 2
	MOVQ R8, SI
	MOVQ $32, CX
	VMOVUPS ·maskNegLoPS(SB), X15 // for i
	VMOVUPS ·maskNegHiPS(SB), X14 // for conjugation

size128_avx2_r2_inv_stage12_loop:
	MOVQ (R12), DX
	VMOVSD (R9)(DX*8), X0
	MOVQ 8(R12), DX
	VMOVSD (R9)(DX*8), X1
	MOVQ 16(R12), DX
	VMOVSD (R9)(DX*8), X2
	MOVQ 24(R12), DX
	VMOVSD (R9)(DX*8), X3
	ADDQ $32, R12
	// Stage 1
	VMOVAPS X0, X8
	VADDPS  X1, X8, X8
	VMOVAPS X0, X9
	VSUBPS  X1, X9, X9
	VMOVAPS X8, X0
	VMOVAPS X9, X1
	VMOVAPS X2, X8
	VADDPS  X3, X8, X8
	VMOVAPS X2, X9
	VSUBPS  X3, X9, X9
	VMOVAPS X8, X2
	VMOVAPS X9, X3
	// Stage 2 (w=[1, i])
	VMOVAPS X0, X8
	VADDPS  X2, X8, X8
	VMOVAPS X0, X9
	VSUBPS  X2, X9, X9
	VMOVAPS X8, X0
	VMOVAPS X9, X2
	VMOVAPS X3, X10
	VSHUFPS $0xB1, X10, X10, X10
	VXORPS  X15, X10, X10          // t = X3 * i
	VMOVAPS X1, X8
	VADDPS  X10, X8, X8
	VMOVAPS X1, X9
	VSUBPS  X10, X9, X9
	VMOVAPS X8, X1
	VMOVAPS X9, X3
	VMOVSD X0, (SI)
	VMOVSD X1, 8(SI)
	VMOVSD X2, 16(SI)
	VMOVSD X3, 24(SI)
	ADDQ $32, SI
	DECQ CX
	JNZ  size128_avx2_r2_inv_stage12_loop

	// Stage 3 (dist 4)
	MOVQ R8, SI
	MOVQ $16, CX
size128_avx2_r2_inv_s3_loop:
	MOVQ $4, DX
size128_avx2_r2_inv_s3_inner:
	VMOVSD (SI), X0
	VMOVSD 32(SI), X1
	MOVQ $4, AX
	SUBQ DX, AX
	SHLQ $4, AX
	VMOVSD (R10)(AX*8), X10
	VXORPS X14, X10, X10           // Conjugate
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 32(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_inv_s3_inner
	ADDQ $32, SI
	DECQ CX
	JNZ size128_avx2_r2_inv_s3_loop

	// Stage 4 (dist 8)
	MOVQ R8, SI
	MOVQ $8, CX
size128_avx2_r2_inv_s4_loop:
	MOVQ $8, DX
size128_avx2_r2_inv_s4_inner:
	VMOVSD (SI), X0
	VMOVSD 64(SI), X1
	MOVQ $8, AX
	SUBQ DX, AX
	SHLQ $3, AX
	VMOVSD (R10)(AX*8), X10
	VXORPS X14, X10, X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 64(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_inv_s4_inner
	ADDQ $64, SI
	DECQ CX
	JNZ size128_avx2_r2_inv_s4_loop

	// Stage 5 (dist 16)
	MOVQ R8, SI
	MOVQ $4, CX
size128_avx2_r2_inv_s5_loop:
	MOVQ $16, DX
size128_avx2_r2_inv_s5_inner:
	VMOVSD (SI), X0
	VMOVSD 128(SI), X1
	MOVQ $16, AX
	SUBQ DX, AX
	SHLQ $2, AX
	VMOVSD (R10)(AX*8), X10
	VXORPS X14, X10, X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 128(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_inv_s5_inner
	ADDQ $128, SI
	DECQ CX
	JNZ size128_avx2_r2_inv_s5_loop

	// Stage 6 (dist 32)
	MOVQ R8, SI
	MOVQ $2, CX
size128_avx2_r2_inv_s6_loop:
	MOVQ $32, DX
size128_avx2_r2_inv_s6_inner:
	VMOVSD (SI), X0
	VMOVSD 256(SI), X1
	MOVQ $32, AX
	SUBQ DX, AX
	SHLQ $1, AX
	VMOVSD (R10)(AX*8), X10
	VXORPS X14, X10, X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 256(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_inv_s6_inner
	ADDQ $256, SI
	DECQ CX
	JNZ size128_avx2_r2_inv_s6_loop

	// Stage 7 (dist 64)
	MOVQ R8, SI
	MOVQ $64, DX
size128_avx2_r2_inv_s7_inner:
	VMOVSD (SI), X0
	VMOVSD 512(SI), X1
	MOVQ $64, AX
	SUBQ DX, AX
	VMOVSD (R10)(AX*8), X10
	VXORPS X14, X10, X10
	VMOVAPS X10, X11
	VSHUFPS $0x00, X11, X11, X11
	VMOVAPS X10, X12
	VSHUFPS $0x55, X12, X12, X12
	VMOVAPS X1, X13
	VSHUFPS $0xB1, X13, X13, X13
	VMULPS X11, X1, X1
	VMULPS X12, X13, X13
	VADDSUBPS X13, X1, X1
	VMOVAPS X0, X2
	VADDPS X1, X0, X0
	VSUBPS X1, X2, X2
	VMOVSD X0, (SI)
	VMOVSD X2, 512(SI)
	ADDQ $8, SI
	DECQ DX
	JNZ size128_avx2_r2_inv_s7_inner

	// Scale by 1/128
	VMOVSS ·oneTwentyEighth32(SB), X15
	VSHUFPS $0x00, X15, X15, X15
	MOVQ $64, CX
	MOVQ R8, SI
size128_avx2_r2_inv_scale_loop:
	VMOVUPS (SI), X0
	VMULPS X15, X0, X0
	VMOVUPS X0, (SI)
	ADDQ $16, SI
	DECQ CX
	JNZ size128_avx2_r2_inv_scale_loop

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   size128_avx2_r2_inv_done
	MOVQ $64, CX
	MOVQ R8, SI
size128_avx2_r2_inv_copy_loop:
	VMOVUPS (SI), X0
	VMOVUPS X0, (R14)
	ADDQ $16, SI
	ADDQ $16, R14
	DECQ CX
	JNZ size128_avx2_r2_inv_copy_loop

size128_avx2_r2_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size128_avx2_r2_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET
