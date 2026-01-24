//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-64 Radix-4 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 64, complex128, radix-4
TEXT ·ForwardSSE2Size64Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size64Radix4(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $64
	JNE  size64_r4_err

	MOVQ R8, R14 // save original dst
	CMPQ R8, R9
	JNE  fwd_r4_use_dst
	MOVQ R11, R8

fwd_r4_use_dst:
	// Bit-reversal (Radix-4)
	XORQ CX, CX
fwd_r4_bitrev_loop:
	MOVQ (R12)(CX*8), DX
  SHLQ $4, DX
  MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
  SHLQ $4, AX
  MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $64
	JL   fwd_r4_bitrev_loop

	// Stage 1: 16 Radix-4 butterflies (stride 16) - No twiddles
	MOVQ R8, SI
	MOVQ $16, CX
	MOVUPS ·maskNegHiPS(SB), X15 // Still used for -i swap logic

fwd_r4_s1_loop:
	MOVUPD (SI), X0
  MOVUPD 16(SI), X1
  MOVUPD 32(SI), X2
  MOVUPD 48(SI), X3
	MOVAPD X0, X4
  ADDPD X2, X4
  MOVAPD X0, X5
  SUBPD X2, X5 // t0, t1
	MOVAPD X1, X6
  ADDPD X3, X6
  MOVAPD X1, X7
  SUBPD X3, X7 // t2, t3
	MOVAPD X7, X8
  SHUFPD $1, X8, X8
  XORPD ·maskNegHiPD(SB), X8 // t3 * -i
	MOVAPD X4, X0
  ADDPD X6, X0 // a0
	MOVAPD X5, X1
  ADDPD X8, X1 // a1
	MOVAPD X4, X2
  SUBPD X6, X2 // a2
	MOVAPD X5, X3
  SUBPD X8, X3 // a3
	MOVUPD X0, (SI)
  MOVUPD X1, 16(SI)
  MOVUPD X2, 32(SI)
  MOVUPD X3, 48(SI)
	ADDQ $64, SI
	DECQ CX
	JNZ  fwd_r4_s1_loop

	// Stage 2: 4 groups, 4 butterflies (stride 4)
	// Stage 3: 1 group, 16 butterflies (stride 1)
	// I'll use a generic radix-4 DIT stage implementation for these
	
	// Stage 2: distance 4, step 16 (scaled for complex128)
	// Butterflies: (i, i+4, i+8, i+12)
	MOVQ R8, SI
	XORQ BX, BX // group
fwd_r4_s2_outer:
	XORQ DX, DX // butterfly
fwd_r4_s2_inner:
	// twiddle[DX*4], twiddle[DX*8], twiddle[DX*12]
	MOVQ DX, AX
  SHLQ $2, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X8 // w1
	MOVQ DX, AX
  SHLQ $3, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X9 // w2
	MOVQ DX, AX
  IMULQ $12, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // w3

	// Load a0..a3
	MOVQ BX, SI
  IMULQ $16, SI
  ADDQ DX, SI
  SHLQ $4, SI
  LEAQ (R8)(SI*1), SI
	MOVUPD (SI), X0
  MOVUPD 64(SI), X1
  MOVUPD 128(SI), X2
  MOVUPD 192(SI), X3

	// Complex mul
	// a1*w1
	MOVAPD X1, X4
  MOVDDUP X4, X4
  MULPD X8, X4
  MOVAPD X1, X5
  UNPCKHPD X5, X5
  MOVAPD X8, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X1
	// a2*w2
	MOVAPD X2, X4
  MOVDDUP X4, X4
  MULPD X9, X4
  MOVAPD X2, X5
  UNPCKHPD X5, X5
  MOVAPD X9, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X2
	// a3*w3
	MOVAPD X3, X4
  MOVDDUP X4, X4
  MULPD X10, X4
  MOVAPD X3, X5
  UNPCKHPD X5, X5
  MOVAPD X10, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X3

	// Radix-4 butterfly
	MOVAPD X0, X4
  ADDPD X2, X4
  MOVAPD X0, X5
  SUBPD X2, X5 // t0, t1
	MOVAPD X1, X6
  ADDPD X3, X6
  MOVAPD X1, X7
  SUBPD X3, X7 // t2, t3
	MOVAPD X7, X8
  SHUFPD $1, X8, X8
  XORPD ·maskNegHiPD(SB), X8 // t3 * -i
	MOVAPD X4, X0
  ADDPD X6, X0 // y0
	MOVAPD X5, X1
  ADDPD X8, X1 // y1
	MOVAPD X4, X2
  SUBPD X6, X2 // y2
	MOVAPD X5, X3
  SUBPD X8, X3 // y3

	MOVUPD X0, (SI)
  MOVUPD X1, 64(SI)
  MOVUPD X2, 128(SI)
  MOVUPD X3, 192(SI)

	INCQ DX
	CMPQ DX, $4
	JL fwd_r4_s2_inner
	INCQ BX
	CMPQ BX, $4
	JL fwd_r4_s2_outer

	// Stage 3: distance 16, step 1 (scaled for complex128)
	XORQ DX, DX
fwd_r4_s3_loop:
	// twiddle[DX], twiddle[2*DX], twiddle[3*DX]
	MOVQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X8
	MOVQ DX, AX
  SHLQ $1, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X9
	MOVQ DX, AX
  IMULQ $3, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10

	MOVQ DX, SI
  SHLQ $4, SI
  LEAQ (R8)(SI*1), SI
	MOVUPD (SI), X0
  MOVUPD 256(SI), X1
  MOVUPD 512(SI), X2
  MOVUPD 768(SI), X3

	// Complex mul
	MOVAPD X1, X4
  MOVDDUP X4, X4
  MULPD X8, X4
  MOVAPD X1, X5
  UNPCKHPD X5, X5
  MOVAPD X8, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X1
	MOVAPD X2, X4
  MOVDDUP X4, X4
  MULPD X9, X4
  MOVAPD X2, X5
  UNPCKHPD X5, X5
  MOVAPD X9, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X2
	MOVAPD X3, X4
  MOVDDUP X4, X4
  MULPD X10, X4
  MOVAPD X3, X5
  UNPCKHPD X5, X5
  MOVAPD X10, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X3

	// Radix-4 butterfly
	MOVAPD X0, X4
  ADDPD X2, X4
  MOVAPD X0, X5
  SUBPD X2, X5 // t0, t1
	MOVAPD X1, X6
  ADDPD X3, X6
  MOVAPD X1, X7
  SUBPD X3, X7 // t2, t3
	MOVAPD X7, X8
  SHUFPD $1, X8, X8
  XORPD ·maskNegHiPD(SB), X8 // t3 * -i
	MOVAPD X4, X0
  ADDPD X6, X0 // y0
	MOVAPD X5, X1
  ADDPD X8, X1 // y1
	MOVAPD X4, X2
  SUBPD X6, X2 // y2
	MOVAPD X5, X3
  SUBPD X8, X3 // y3

	MOVUPD X0, (SI)
  MOVUPD X1, 256(SI)
  MOVUPD X2, 512(SI)
  MOVUPD X3, 768(SI)

	INCQ DX
	CMPQ DX, $16
	JL fwd_r4_s3_loop

	// Copy to R14 if needed
	CMPQ R8, R14
	JE size64_r4_fwd_done
	MOVQ $32, CX
  MOVQ R8, SI
  MOVQ R14, DI
size64_r4_fwd_copy:
	MOVUPD (SI), X0
  MOVUPD 16(SI), X1
  MOVUPD X0, (DI)
  MOVUPD X1, 16(DI)
  ADDQ $32, SI
  ADDQ $32, DI
  DECQ CX
  JNZ size64_r4_fwd_copy

size64_r4_fwd_done:
	MOVB $1, ret+96(FP)
	RET
size64_r4_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 64, complex128, radix-4
TEXT ·InverseSSE2Size64Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size64Radix4(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $64
	JNE  size64_r4_inv_err

	MOVQ R8, R14
	CMPQ R8, R9
	JNE  inv_r4_use_dst
	MOVQ R11, R8

inv_r4_use_dst:
	XORQ CX, CX
inv_r4_bitrev_loop:
	MOVQ (R12)(CX*8), DX
  SHLQ $4, DX
  MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
  SHLQ $4, AX
  MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $64
	JL   inv_r4_bitrev_loop

	// Stage 1 (+i)
	MOVQ R8, SI
  MOVQ $16, CX
inv_r4_s1_loop:
	MOVUPD (SI), X0
  MOVUPD 16(SI), X1
  MOVUPD 32(SI), X2
  MOVUPD 48(SI), X3
	MOVAPD X0, X4
  ADDPD X2, X4
  MOVAPD X0, X5
  SUBPD X2, X5
	MOVAPD X1, X6
  ADDPD X3, X6
  MOVAPD X1, X7
  SUBPD X3, X7
	MOVAPD X7, X8
  SHUFPD $1, X8, X8
  XORPD ·maskNegLoPD(SB), X8 // t3 * i
	MOVAPD X4, X0
  ADDPD X6, X0
	MOVAPD X5, X1
  ADDPD X8, X1
	MOVAPD X4, X2
  SUBPD X6, X2
	MOVAPD X5, X3
  SUBPD X8, X3
	MOVUPD X0, (SI)
  MOVUPD X1, 16(SI)
  MOVUPD X2, 32(SI)
  MOVUPD X3, 48(SI)
	ADDQ $64, SI
	DECQ CX
	JNZ  inv_r4_s1_loop

	MOVUPS ·maskNegHiPD(SB), X15

	// Stage 2
	XORQ BX, BX
inv_r4_s2_outer:
	XORQ DX, DX
inv_r4_s2_inner:
	MOVQ DX, AX
  SHLQ $2, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X8
  XORPD X15, X8
	MOVQ DX, AX
  SHLQ $3, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X9
  XORPD X15, X9
	MOVQ DX, AX
  IMULQ $12, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10
  XORPD X15, X10

	MOVQ BX, SI
  IMULQ $16, SI
  ADDQ DX, SI
  SHLQ $4, SI
  LEAQ (R8)(SI*1), SI
	MOVUPD (SI), X0
  MOVUPD 64(SI), X1
  MOVUPD 128(SI), X2
  MOVUPD 192(SI), X3

	// Complex mul
	MOVAPD X1, X4
  MOVDDUP X4, X4
  MULPD X8, X4
  MOVAPD X1, X5
  UNPCKHPD X5, X5
  MOVAPD X8, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X1
	MOVAPD X2, X4
  MOVDDUP X4, X4
  MULPD X9, X4
  MOVAPD X2, X5
  UNPCKHPD X5, X5
  MOVAPD X9, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X2
	MOVAPD X3, X4
  MOVDDUP X4, X4
  MULPD X10, X4
  MOVAPD X3, X5
  UNPCKHPD X5, X5
  MOVAPD X10, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X3

	// Radix-4 (+i)
	MOVAPD X0, X4
  ADDPD X2, X4
  MOVAPD X0, X5
  SUBPD X2, X5
	MOVAPD X1, X6
  ADDPD X3, X6
  MOVAPD X1, X7
  SUBPD X3, X7
	MOVAPD X7, X8
  SHUFPD $1, X8, X8
  XORPD ·maskNegLoPD(SB), X8
	MOVAPD X4, X0
  ADDPD X6, X0
	MOVAPD X5, X1
  ADDPD X8, X1
	MOVAPD X4, X2
  SUBPD X6, X2
	MOVAPD X5, X3
  SUBPD X8, X3

	MOVUPD X0, (SI)
  MOVUPD X1, 64(SI)
  MOVUPD X2, 128(SI)
  MOVUPD X3, 192(SI)

	INCQ DX
	CMPQ DX, $4
	JL inv_r4_s2_inner
	INCQ BX
	CMPQ BX, $4
	JL inv_r4_s2_outer

	// Stage 3
	XORQ DX, DX
inv_r4_s3_loop:
	MOVQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X8
  XORPD X15, X8
	MOVQ DX, AX
  SHLQ $1, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X9
  XORPD X15, X9
	MOVQ DX, AX
  IMULQ $3, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10
  XORPD X15, X10

	MOVQ DX, SI
  SHLQ $4, SI
  LEAQ (R8)(SI*1), SI
	MOVUPD (SI), X0
  MOVUPD 256(SI), X1
  MOVUPD 512(SI), X2
  MOVUPD 768(SI), X3

	// Complex mul
	MOVAPD X1, X4
  MOVDDUP X4, X4
  MULPD X8, X4
  MOVAPD X1, X5
  UNPCKHPD X5, X5
  MOVAPD X8, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X1
	MOVAPD X2, X4
  MOVDDUP X4, X4
  MULPD X9, X4
  MOVAPD X2, X5
  UNPCKHPD X5, X5
  MOVAPD X9, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X2
	MOVAPD X3, X4
  MOVDDUP X4, X4
  MULPD X10, X4
  MOVAPD X3, X5
  UNPCKHPD X5, X5
  MOVAPD X10, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
  XORPD ·maskNegLoPD(SB), X6
  ADDPD X6, X4
  MOVAPD X4, X3

	// Radix-4 (+i)
	MOVAPD X0, X4
  ADDPD X2, X4
  MOVAPD X0, X5
  SUBPD X2, X5
	MOVAPD X1, X6
  ADDPD X3, X6
  MOVAPD X1, X7
  SUBPD X3, X7
	MOVAPD X7, X8
  SHUFPD $1, X8, X8
  XORPD ·maskNegLoPD(SB), X8
	MOVAPD X4, X0
  ADDPD X6, X0
	MOVAPD X5, X1
  ADDPD X8, X1
	MOVAPD X4, X2
  SUBPD X6, X2
	MOVAPD X5, X3
  SUBPD X8, X3

	MOVUPD X0, (SI)
  MOVUPD X1, 256(SI)
  MOVUPD X2, 512(SI)
  MOVUPD X3, 768(SI)

	INCQ DX
	CMPQ DX, $16
	JL inv_r4_s3_loop

	// Scale by 1/64 and Copy
	MOVSD ·sixtyFourth64(SB), X15
  SHUFPD $0, X15, X15
	MOVQ $32, CX
  MOVQ R8, SI
  MOVQ R14, DI
inv_r4_scale_copy:
	MOVUPD (SI), X0
  MOVUPD 16(SI), X1
  MULPD X15, X0
  MULPD X15, X1
  MOVUPD X0, (DI)
  MOVUPD X1, 16(DI)
  ADDQ $32, SI
  ADDQ $32, DI
  DECQ CX
  JNZ inv_r4_scale_copy

	MOVB $1, ret+96(FP)
	RET
size64_r4_inv_err:
	MOVB $0, ret+96(FP)
	RET
