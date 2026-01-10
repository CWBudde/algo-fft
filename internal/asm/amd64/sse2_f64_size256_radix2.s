//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2 Size-256 Radix-2 FFT Kernels for AMD64 (complex128)
// ===========================================================================

#include "textflag.h"

// Forward transform, size 256, complex128, radix-2
TEXT ·ForwardSSE2Size256Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size256Radix2(SB), R12
	MOVQ src+32(FP), R13

	// Check size n == 256
	CMPQ R13, $256
	JNE  fwd_err

	// Check if dst == src (in-place)
	CMPQ R8, R9
	JNE  fwd_use_dst
	MOVQ R11, R8       // Use scratch as temp dst if in-place

fwd_use_dst:
	// Bit-reversal permutation
	XORQ CX, CX        // i = 0

fwd_bitrev_loop:
	MOVQ (R12)(CX*8), DX    // DX = bitrev[i]
	SHLQ $4, DX             // DX *= 16 (sizeof complex128)
	MOVUPD (R9)(DX*1), X0   // X0 = src[bitrev[i]]
	MOVQ CX, AX             // AX = i
	SHLQ $4, AX             // AX *= 16
	MOVUPD X0, (R8)(AX*1)   // dst[i] = X0
	INCQ CX                 // i++
	CMPQ CX, $256           // i < 256
	JL   fwd_bitrev_loop

	// Stage 1 & 2 (Combined) - 64 blocks of 4
	MOVQ R8, SI             // SI = dst
	MOVQ $64, CX            // Loop counter = 64
	MOVUPS ·maskNegHiPD(SB), X15 // Mask for mul by -i (negate imaginary part)

fwd_stage12_loop:
	// Load 4 complex numbers
	MOVUPD (SI), X0         // A
	MOVUPD 16(SI), X1       // B
	MOVUPD 32(SI), X2       // C
	MOVUPD 48(SI), X3       // D

	// Stage 1 butterflies
	MOVAPD X0, X8           // Copy A
	ADDPD X1, X0            // X0 = A + B
	SUBPD X1, X8            // X8 = A - B

	MOVAPD X2, X9           // Copy C
	ADDPD X3, X2            // X2 = C + D
	SUBPD X3, X9            // X9 = C - D

	// Stage 2 butterflies
	MOVAPD X0, X10          // Copy (A+B)
	ADDPD X2, X0            // X0 = (A+B) + (C+D)
	SUBPD X2, X10           // X10 = (A+B) - (C+D)

	// Prepare (C-D) * -i
	MOVAPD X9, X11          // Copy (C-D)
	SHUFPD $1, X11, X11     // Swap Re/Im
	XORPD X15, X11          // Negate Re (implies * -i for complex128 layout)

	MOVAPD X8, X12          // Copy (A-B)
	ADDPD X11, X8           // X8 = (A-B) + (C-D)*-i
	SUBPD X11, X12          // X12 = (A-B) - (C-D)*-i

	// Store results
	MOVUPD X0, (SI)
	MOVUPD X8, 16(SI)
	MOVUPD X10, 32(SI)
	MOVUPD X12, 48(SI)

	ADDQ $64, SI            // Advance pointer
	DECQ CX
	JNZ  fwd_stage12_loop

	// Prepare mask for next stages (negate Real part for standard Complex Mul)
	MOVUPS ·maskNegLoPD(SB), X14

	// Stage 3: dist 4
	MOVQ R8, SI             // Reset pointer
	MOVQ $32, CX            // Loop counter (256 / 8 = 32)

fwd_s3_loop:
	MOVQ $4, DX             // Inner loop counter

fwd_s3_inner:
	MOVUPD (SI), X0         // Load A
	MOVUPD 64(SI), X1       // Load B (at dist 4*16 = 64 bytes)

	// Twiddle factor calculation
	MOVQ $4, AX             // Base
	SUBQ DX, AX             // k = 4 - DX
	SHLQ $5, AX             // k * 32 (stride for stage 3)
	SHLQ $4, AX             // Convert to bytes (*16)
	MOVUPD (R10)(AX*1), X10 // Load Twiddle W

	// Complex Multiply B * W
	MOVAPD X1, X2
	UNPCKLPD X2, X2         // (Re(B), Re(B))
	MULPD X10, X2           // Re(B)*Re(W), Re(B)*Im(W)

	MOVAPD X1, X3
	UNPCKHPD X3, X3         // (Im(B), Im(B))
	MOVAPD X10, X4
	SHUFPD $1, X4, X4       // (Im(W), Re(W))
	MULPD X3, X4            // Im(B)*Im(W), Im(B)*Re(W)
	XORPD X14, X4           // -Im(B)*Im(W), Im(B)*Re(W)
	ADDPD X4, X2            // X2 = B * W

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0            // A + B*W
	SUBPD X2, X3            // A - B*W

	MOVUPD X0, (SI)
	MOVUPD X3, 64(SI)

	ADDQ $16, SI            // Next pair
	DECQ DX
	JNZ fwd_s3_inner

	ADDQ $64, SI            // Skip next block
	DECQ CX
	JNZ fwd_s3_loop

	// Stage 4: dist 8
	MOVQ R8, SI
	MOVQ $16, CX            // 256 / 16 = 16

fwd_s4_loop:
	MOVQ $8, DX

fwd_s4_inner:
	MOVUPD (SI), X0
	MOVUPD 128(SI), X1      // dist 8*16 = 128

	MOVQ $8, AX
	SUBQ DX, AX
	SHLQ $4, AX             // k * 16 (stride)
	SHLQ $4, AX             // bytes
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 128(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s4_inner

	ADDQ $128, SI
	DECQ CX
	JNZ fwd_s4_loop

	// Stage 5: dist 16
	MOVQ R8, SI
	MOVQ $8, CX             // 256 / 32 = 8

fwd_s5_loop:
	MOVQ $16, DX

fwd_s5_inner:
	MOVUPD (SI), X0
	MOVUPD 256(SI), X1      // dist 16*16 = 256

	MOVQ $16, AX
	SUBQ DX, AX
	SHLQ $3, AX             // k * 8 (stride)
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 256(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s5_inner

	ADDQ $256, SI
	DECQ CX
	JNZ fwd_s5_loop

	// Stage 6: dist 32
	MOVQ R8, SI
	MOVQ $4, CX             // 256 / 64 = 4

fwd_s6_loop:
	MOVQ $32, DX

fwd_s6_inner:
	MOVUPD (SI), X0
	MOVUPD 512(SI), X1      // dist 32*16 = 512

	MOVQ $32, AX
	SUBQ DX, AX
	SHLQ $2, AX             // k * 4 (stride)
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 512(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s6_inner

	ADDQ $512, SI
	DECQ CX
	JNZ fwd_s6_loop

	// Stage 7: dist 64
	MOVQ R8, SI
	MOVQ $2, CX             // 256 / 128 = 2

fwd_s7_loop:
	MOVQ $64, DX

fwd_s7_inner:
	MOVUPD (SI), X0
	MOVUPD 1024(SI), X1     // dist 64*16 = 1024

	MOVQ $64, AX
	SUBQ DX, AX
	SHLQ $1, AX             // k * 2 (stride)
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 1024(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s7_inner

	ADDQ $1024, SI
	DECQ CX
	JNZ fwd_s7_loop

	// Stage 8: dist 128
	MOVQ R8, SI
	MOVQ $128, DX           // Single loop

fwd_s8_inner:
	MOVUPD (SI), X0
	MOVUPD 2048(SI), X1     // dist 128*16 = 2048

	MOVQ $128, AX
	SUBQ DX, AX
	                        // k * 1 (stride) -> no shift needed
	SHLQ $4, AX             // bytes
	MOVUPD (R10)(AX*1), X10

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2
	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X14, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 2048(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ fwd_s8_inner

	// Copy to dst if needed
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   fwd_done

	MOVQ $128, CX           // 256 / 2 = 128 iterations of 2xComplex128 (32 bytes)
	MOVQ R8, SI
	MOVQ R14, DI

fwd_copy_loop:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD X0, (DI)
	MOVUPD X1, 16(DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ fwd_copy_loop

fwd_done:
	MOVB $1, ret+96(FP)
	RET

fwd_err:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 256, complex128, radix-2
TEXT ·InverseSSE2Size256Radix2Complex128Asm(SB), NOSPLIT, $0-97
	MOVQ dst+0(FP), R8
	MOVQ R8, R14
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrevSSE2Size256Radix2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $256
	JNE  inv_err

	CMPQ R8, R9
	JNE  inv_use_dst
	MOVQ R11, R8

inv_use_dst:
	// Bit-reversal
	XORQ CX, CX

inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)
	INCQ CX
	CMPQ CX, $256
	JL   inv_bitrev_loop

	// Stage 1 & 2
	MOVQ R8, SI
	MOVQ $64, CX
	MOVUPS ·maskNegLoPD(SB), X15 // Mask for * i (negate Real part of (Im, Re)) -> (-Im, Re)?
	                             // Wait. maskNegLoPD negates low double.
	                             // i = (0, 1). (a+bi)*i = -b + ai.
	                             // Input: (a, b). SHUFPD $1 -> (b, a).
	                             // (b, a) XOR maskNegLoPD -> (-b, a).
	                             // Result (-b, a). Correct.

inv_stage12_loop:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD 32(SI), X2
	MOVUPD 48(SI), X3

	// Stage 1
	MOVAPD X0, X8
	ADDPD X1, X0
	SUBPD X1, X8

	MOVAPD X2, X9
	ADDPD X3, X2
	SUBPD X3, X9

	// Stage 2
	MOVAPD X0, X10
	ADDPD X2, X0
	SUBPD X2, X10

	// Multiply (C-D) by i
	MOVAPD X9, X11
	SHUFPD $1, X11, X11
	XORPD X15, X11

	MOVAPD X8, X12
	ADDPD X11, X8
	SUBPD X11, X12

	MOVUPD X0, (SI)
	MOVUPD X8, 16(SI)
	MOVUPD X10, 32(SI)
	MOVUPD X12, 48(SI)

	ADDQ $64, SI
	DECQ CX
	JNZ  inv_stage12_loop

	// Prepare masks for next stages
	MOVUPS ·maskNegHiPD(SB), X14 // Conjugate W: (wr, wi) -> (wr, -wi)
	MOVUPS ·maskNegLoPD(SB), X13 // Negate term in complex mul

	// Stage 3
	MOVQ R8, SI
	MOVQ $32, CX

inv_s3_loop:
	MOVQ $4, DX

inv_s3_inner:
	MOVUPD (SI), X0
	MOVUPD 64(SI), X1

	MOVQ $4, AX
	SUBQ DX, AX
	SHLQ $5, AX             // k * 32
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 64(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s3_inner

	ADDQ $64, SI
	DECQ CX
	JNZ inv_s3_loop

	// Stage 4
	MOVQ R8, SI
	MOVQ $16, CX

inv_s4_loop:
	MOVQ $8, DX

inv_s4_inner:
	MOVUPD (SI), X0
	MOVUPD 128(SI), X1

	MOVQ $8, AX
	SUBQ DX, AX
	SHLQ $4, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 128(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s4_inner

	ADDQ $128, SI
	DECQ CX
	JNZ inv_s4_loop

	// Stage 5
	MOVQ R8, SI
	MOVQ $8, CX

inv_s5_loop:
	MOVQ $16, DX

inv_s5_inner:
	MOVUPD (SI), X0
	MOVUPD 256(SI), X1

	MOVQ $16, AX
	SUBQ DX, AX
	SHLQ $3, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 256(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s5_inner

	ADDQ $256, SI
	DECQ CX
	JNZ inv_s5_loop

	// Stage 6
	MOVQ R8, SI
	MOVQ $4, CX

inv_s6_loop:
	MOVQ $32, DX

inv_s6_inner:
	MOVUPD (SI), X0
	MOVUPD 512(SI), X1

	MOVQ $32, AX
	SUBQ DX, AX
	SHLQ $2, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 512(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s6_inner

	ADDQ $512, SI
	DECQ CX
	JNZ inv_s6_loop

	// Stage 7
	MOVQ R8, SI
	MOVQ $2, CX

inv_s7_loop:
	MOVQ $64, DX

inv_s7_inner:
	MOVUPD (SI), X0
	MOVUPD 1024(SI), X1

	MOVQ $64, AX
	SUBQ DX, AX
	SHLQ $1, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 1024(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s7_inner

	ADDQ $1024, SI
	DECQ CX
	JNZ inv_s7_loop

	// Stage 8
	MOVQ R8, SI
	MOVQ $128, DX

inv_s8_inner:
	MOVUPD (SI), X0
	MOVUPD 2048(SI), X1

	MOVQ $128, AX
	SUBQ DX, AX
	                        // k * 1
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X10
	XORPD X14, X10          // Conj(W)

	// Complex Mul
	MOVAPD X1, X2
	UNPCKLPD X2, X2
	MULPD X10, X2

	MOVAPD X1, X3
	UNPCKHPD X3, X3
	MOVAPD X10, X4
	SHUFPD $1, X4, X4
	MULPD X3, X4
	XORPD X13, X4
	ADDPD X4, X2

	// Butterfly
	MOVAPD X0, X3
	ADDPD X2, X0
	SUBPD X2, X3

	MOVUPD X0, (SI)
	MOVUPD X3, 2048(SI)

	ADDQ $16, SI
	DECQ DX
	JNZ inv_s8_inner

	// Scale by 1/256
	MOVSD ·twoFiftySixth64(SB), X15
	SHUFPD $0, X15, X15     // Broadcast scaler
	MOVQ $128, CX
	MOVQ R8, SI

inv_scale:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MULPD X15, X0
	MULPD X15, X1
	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	ADDQ $32, SI
	DECQ CX
	JNZ inv_scale

	// Copy to dst
	MOVQ dst+0(FP), R14
	CMPQ R8, R14
	JE   inv_done

	MOVQ $128, CX
	MOVQ R8, SI
	MOVQ R14, DI

inv_copy:
	MOVUPD (SI), X0
	MOVUPD 16(SI), X1
	MOVUPD X0, (DI)
	MOVUPD X1, 16(DI)
	ADDQ $32, SI
	ADDQ $32, DI
	DECQ CX
	JNZ inv_copy

inv_done:
	MOVB $1, ret+96(FP)
	RET

inv_err:
	MOVB $0, ret+96(FP)
	RET
