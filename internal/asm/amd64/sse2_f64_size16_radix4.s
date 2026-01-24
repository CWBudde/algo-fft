//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-16 Radix-4 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 16, complex128, radix-4
TEXT ·ForwardSSE2Size16Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size16Radix4(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $16
	JNE  size16_r4_fwd_err

	CMPQ R8, R9
	JNE  size16_r4_fwd_use_dst
	MOVQ R11, R8

size16_r4_fwd_use_dst:
	// Bit-reversal (Radix-4 indices)
	XORQ CX, CX
size16_r4_fwd_bitrev_loop:
	MOVQ (R12)(CX*8), DX
  SHLQ $4, DX
  MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
  SHLQ $4, AX
  MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $16
	JL   size16_r4_fwd_bitrev_loop

	// Stage 1: 4 Radix-4 butterflies (stride 4)
	MOVQ R8, SI
	MOVQ $4, CX
	MOVUPS ·maskNegHiPD(SB), X15

size16_r4_fwd_stage1_loop:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD 32(SI), X2
	MOVUPD 48(SI), X3
	MOVAPD X0, X8
  ADDPD X2, X8
  MOVAPD X0, X9
  SUBPD X2, X9 // t0, t1
	MOVAPD X1, X10
  ADDPD X3, X10
  MOVAPD X1, X11
  SUBPD X3, X11 // t2, t3
	MOVAPD X11, X12
  SHUFPD $1, X12, X12
  XORPD X15, X12 // t3 * -i
	MOVAPD X8, X0
  ADDPD X10, X0 // a0
	MOVAPD X9, X1
  ADDPD X12, X1 // a1
	MOVAPD X8, X2
  SUBPD X10, X2 // a2
	MOVAPD X9, X3
  SUBPD X12, X3 // a3
	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)
	ADDQ $64, SI
	DECQ CX
	JNZ  size16_r4_fwd_stage1_loop

	// Stage 2: 1 group, 4 butterflies (stride 1)
	MOVQ R8, SI
	XORQ DX, DX
	MOVUPS ·maskNegLoPD(SB), X14

size16_r4_fwd_stage2_loop:
	// Twiddle indices: j, 2j, 3j
	// For size 16, n=16. j=0..3. twiddle[j]
	// w1 = tw[DX], w2 = tw[2*DX], w3 = tw[3*DX]
	MOVQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X8 // w1
	MOVQ DX, AX
  SHLQ $1, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X9 // w2
	MOVQ DX, AX
  SHLQ $1, AX
  ADDQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10 // w3

	// Load a0..a3
	MOVQ DX, AX
  SHLQ $4, AX
  MOVUPD (R8)(AX*1), X0
	ADDQ $64, AX
  MOVUPD (R8)(AX*1), X1
	ADDQ $64, AX
  MOVUPD (R8)(AX*1), X2
	ADDQ $64, AX
  MOVUPD (R8)(AX*1), X3

	// Complex mul a1*w1, a2*w2, a3*w3
	// a1 * w1
	MOVAPD X1, X4
  UNPCKLPD X4, X4
  MULPD X8, X4
	MOVAPD X1, X5
  UNPCKHPD X5, X5
  MOVAPD X8, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
	XORPD X14, X6
  ADDPD X6, X4
  MOVAPD X4, X1
	// a2 * w2
	MOVAPD X2, X4
  UNPCKLPD X4, X4
  MULPD X9, X4
	MOVAPD X2, X5
  UNPCKHPD X5, X5
  MOVAPD X9, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
	XORPD X14, X6
  ADDPD X6, X4
  MOVAPD X4, X2
	// a3 * w3
	MOVAPD X3, X4
  UNPCKLPD X4, X4
  MULPD X10, X4
	MOVAPD X3, X5
  UNPCKHPD X5, X5
  MOVAPD X10, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
	XORPD X14, X6
  ADDPD X6, X4
  MOVAPD X4, X3

	// Radix-4 butterfly
	MOVAPD X0, X8
  ADDPD X2, X8
  MOVAPD X0, X9
  SUBPD X2, X9 // t0, t1
	MOVAPD X1, X10
  ADDPD X3, X10
  MOVAPD X1, X11
  SUBPD X3, X11 // t2, t3
	MOVAPD X11, X12
  SHUFPD $1, X12, X12
  XORPD X15, X12 // t3 * -i
	MOVAPD X8, X0
  ADDPD X10, X0 // y0
	MOVAPD X9, X1
  ADDPD X12, X1 // y1
	MOVAPD X8, X2
  SUBPD X10, X2 // y2
	MOVAPD X9, X3
  SUBPD X12, X3 // y3

	// Store
	MOVQ DX, AX
  SHLQ $4, AX
  MOVUPD X0, (R8)(AX*1)
	ADDQ $64, AX
  MOVUPD X1, (R8)(AX*1)
	ADDQ $64, AX
  MOVUPD X2, (R8)(AX*1)
	ADDQ $64, AX
  MOVUPD X3, (R8)(AX*1)

	INCQ DX
	CMPQ DX, $4
	JL size16_r4_fwd_stage2_loop

	// Copy to R14 if needed
	CMPQ R8, R14
	JE size16_r4_fwd_done
	MOVQ $16, CX
  MOVQ R8, SI
  MOVQ R14, DI
size16_r4_fwd_copy:
	MOVUPD (SI), X0
  MOVUPD X0, (DI)
  ADDQ $16, SI
  ADDQ $16, DI
  DECQ CX
  JNZ size16_r4_fwd_copy

size16_r4_fwd_done:
	MOVB $1, ret+96(FP)
	RET
size16_r4_fwd_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 16, complex128, radix-4
TEXT ·InverseSSE2Size16Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size16Radix4(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $16
	JNE  size16_r4_inv_err

	CMPQ R8, R9
	JNE  size16_r4_inv_use_dst
	MOVQ R11, R8

size16_r4_inv_use_dst:
	// Bit-reversal
	XORQ CX, CX
size16_r4_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
  SHLQ $4, DX
  MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
  SHLQ $4, AX
  MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $16
	JL   size16_r4_inv_bitrev_loop

	// Stage 1: Radix-4 butterflies (+i)
	MOVQ R8, SI
	MOVQ $4, CX
	MOVUPS ·maskNegLoPD(SB), X14

size16_r4_inv_stage1_loop:
	MOVUPD (SI), X0
  MOVUPD 16(SI), X1
  MOVUPD 32(SI), X2
  MOVUPD 48(SI), X3
	MOVAPD X0, X8
  ADDPD X2, X8
  MOVAPD X0, X9
  SUBPD X2, X9 // t0, t1
	MOVAPD X1, X10
  ADDPD X3, X10
  MOVAPD X1, X11
  SUBPD X3, X11 // t2, t3
	MOVAPD X11, X12
  SHUFPD $1, X12, X12
  XORPD X14, X12 // t3 * i
	MOVAPD X8, X0
  ADDPD X10, X0 // a0
	MOVAPD X9, X1
  ADDPD X12, X1 // a1
	MOVAPD X8, X2
  SUBPD X10, X2 // a2
	MOVAPD X9, X3
  SUBPD X12, X3 // a3
	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)
	ADDQ $64, SI
	DECQ CX
	JNZ  size16_r4_inv_stage1_loop

	// Stage 2: Radix-4 butterflies with conjugated twiddles
	MOVQ R8, SI
	XORQ DX, DX
	MOVUPS ·maskNegHiPD(SB), X15

size16_r4_inv_stage2_loop:
	MOVQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X8
  XORPD X15, X8 // conj(w1)
	MOVQ DX, AX
  SHLQ $1, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X9
  XORPD X15, X9 // conj(w2)
	MOVQ DX, AX
  SHLQ $1, AX
  ADDQ DX, AX
  SHLQ $4, AX
  MOVUPD (R10)(AX*1), X10
  XORPD X15, X10 // conj(w3)

	MOVQ DX, AX
  SHLQ $4, AX
  MOVUPD (R8)(AX*1), X0
	ADDQ $64, AX
  MOVUPD (R8)(AX*1), X1
	ADDQ $64, AX
  MOVUPD (R8)(AX*1), X2
	ADDQ $64, AX
  MOVUPD (R8)(AX*1), X3

	// Complex mul
	MOVAPD X1, X4
  UNPCKLPD X4, X4
  MULPD X8, X4
	MOVAPD X1, X5
  UNPCKHPD X5, X5
  MOVAPD X8, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
	XORPD X14, X6
  ADDPD X6, X4
  MOVAPD X4, X1
	MOVAPD X2, X4
  UNPCKLPD X4, X4
  MULPD X9, X4
	MOVAPD X2, X5
  UNPCKHPD X5, X5
  MOVAPD X9, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
	XORPD X14, X6
  ADDPD X6, X4
  MOVAPD X4, X2
	MOVAPD X3, X4
  UNPCKLPD X4, X4
  MULPD X10, X4
	MOVAPD X3, X5
  UNPCKHPD X5, X5
  MOVAPD X10, X6
  SHUFPD $1, X6, X6
  MULPD X5, X6
	XORPD X14, X6
  ADDPD X6, X4
  MOVAPD X4, X3

	// Radix-4 butterfly (+i)
	MOVAPD X0, X8
  ADDPD X2, X8
  MOVAPD X0, X9
  SUBPD X2, X9 // t0, t1
	MOVAPD X1, X10
  ADDPD X3, X10
  MOVAPD X1, X11
  SUBPD X3, X11 // t2, t3
	MOVAPD X11, X12
  SHUFPD $1, X12, X12
  XORPD X14, X12 // t3 * i
	MOVAPD X8, X0
  ADDPD X10, X0
	MOVAPD X9, X1
  ADDPD X12, X1
	MOVAPD X8, X2
  SUBPD X10, X2
	MOVAPD X9, X3
  SUBPD X12, X3

	MOVQ DX, AX
  SHLQ $4, AX
  MOVUPD X0, (R8)(AX*1)
	ADDQ $64, AX
  MOVUPD X1, (R8)(AX*1)
	ADDQ $64, AX
  MOVUPD X2, (R8)(AX*1)
	ADDQ $64, AX
  MOVUPD X3, (R8)(AX*1)

	INCQ DX
	CMPQ DX, $4
	JL size16_r4_inv_stage2_loop

	// Scale by 1/16 and Copy
	MOVSD ·sixteenth64(SB), X15
  SHUFPD $0, X15, X15
	MOVQ $16, CX
  MOVQ R8, SI
  MOVQ R14, DI
size16_r4_inv_scale_copy:
	MOVUPD (SI), X0
  MULPD X15, X0
  MOVUPD X0, (DI)
  ADDQ $16, SI
  ADDQ $16, DI
  DECQ CX
  JNZ size16_r4_inv_scale_copy

	MOVB $1, ret+96(FP)
	RET
size16_r4_inv_err:
	MOVB $0, ret+96(FP)
	RET
