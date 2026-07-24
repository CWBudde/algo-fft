//go:build amd64 && !purego

// ===========================================================================
// AVX2 Size-1024 Radix-4 FFT Kernel for AMD64 (complex128)
// ===========================================================================
//
// Algorithm: Radix-4 Decimation-in-Time (DIT) FFT
// Stages: 5 (log4(1024) = 5)
//
// Stage structure:
//   Stage 1: 256 groups x 1 butterfly,  stride=4,    no twiddle (W^0 = 1)
//   Stage 2: 64 groups  x 4 butterflies, stride=16,   twiddle step=64
//   Stage 3: 16 groups  x 16 butterflies, stride=64,  twiddle step=16
//   Stage 4: 4 groups   x 64 butterflies, stride=256, twiddle step=4
//   Stage 5: 1 group    x 256 butterflies, stride=1024, twiddle step=1
//
// Uses byte-pointer addressing (SSE2-style) to avoid a shift on every memory
// access, with the AVX2 3-operand FMA (VFMADDSUB/VFMSUBADD) complex-multiply.
// The bit-reversal table (·bitrev1024_r4) is shared with the complex64 kernel
// in avx2_f32_size1024_radix4.s (entries are element indices).
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardAVX2Size1024Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src_len+32(FP), R13 // R13 = n (should be 1024)
	LEAQ ·bitrev1024_r4(SB), R12

	// Verify n == 1024
	CMPQ R13, $1024
	JNE  r4_1024f64_return_false

	// Validate slice lengths
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $1024
	JL   r4_1024f64_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $1024
	JL   r4_1024f64_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $1024
	JL   r4_1024f64_return_false

	// Select working buffer (in-place: use scratch)
	CMPQ R8, R9
	JNE  r4_1024f64_use_dst
	MOVQ R11, R8

r4_1024f64_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 digit reversal)
	// ==================================================================
	XORQ CX, CX

r4_1024f64_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0    // load src[bitrev[i]]
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)    // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $1024
	JL   r4_1024f64_bitrev_loop

r4_1024f64_stage1:
	// ==================================================================
	// Stage 1: 256 groups x 1 butterfly, stride=4, no twiddle (W^0 = 1)
	// ==================================================================
	MOVQ R8, SI
	MOVQ $256, CX

r4_1024f64_stage1_loop:
	MOVUPD (SI), X0          // a0
	MOVUPD 16(SI), X1        // a1
	MOVUPD 32(SI), X2        // a2
	MOVUPD 48(SI), X3        // a3

	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3

	VSHUFPD $0x1, X7, X7, X8
	VXORPD ·maskNegHiPD(SB), X8, X8 // (-i)*t3

	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X8, X1        // y1 = t1 + (-i)*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X8, X5, X3        // y3 = t1 - (-i)*t3

	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)

	ADDQ $64, SI
	DECQ CX
	JNZ  r4_1024f64_stage1_loop

r4_1024f64_stage2:
	// ==================================================================
	// Stage 2: 64 groups x 4 butterflies, span=16, twiddle step=64
	// ==================================================================
	XORQ BX, BX              // BX = group index

r4_1024f64_stage2_outer:
	CMPQ BX, $64
	JGE  r4_1024f64_stage3

	XORQ DX, DX              // DX = butterfly index in group

r4_1024f64_stage2_inner:
	CMPQ DX, $4
	JGE  r4_1024f64_stage2_next

	// Twiddles: w[j*64], w[2*j*64], w[3*j*64]
	MOVQ DX, AX
	IMULQ $1024, AX
	MOVUPD (R10)(AX*1), X8
	MOVQ DX, AX
	IMULQ $2048, AX
	MOVUPD (R10)(AX*1), X9
	MOVQ DX, AX
	IMULQ $3072, AX
	MOVUPD (R10)(AX*1), X10

	// Data pointers (byte addressing)
	MOVQ BX, R13
	IMULQ $256, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 64(SI), DI
	LEAQ 128(SI), R14
	LEAQ 192(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// complex multiply X1 *= X8
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1
	// complex multiply X2 *= X9
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2
	// complex multiply X3 *= X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3
	// radix-4 butterfly
	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14 // (-i)*t3
	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X14, X1       // y1 = t1 + (-i)*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X14, X5, X3       // y3 = t1 - (-i)*t3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  r4_1024f64_stage2_inner

r4_1024f64_stage2_next:
	INCQ BX
	JMP  r4_1024f64_stage2_outer

r4_1024f64_stage3:
	// ==================================================================
	// Stage 3: 16 groups x 16 butterflies, span=64, twiddle step=16
	// ==================================================================
	XORQ BX, BX              // BX = group index

r4_1024f64_stage3_outer:
	CMPQ BX, $16
	JGE  r4_1024f64_stage4

	XORQ DX, DX              // DX = butterfly index in group

r4_1024f64_stage3_inner:
	CMPQ DX, $16
	JGE  r4_1024f64_stage3_next

	// Twiddles: w[j*16], w[2*j*16], w[3*j*16]
	MOVQ DX, AX
	IMULQ $256, AX
	MOVUPD (R10)(AX*1), X8
	MOVQ DX, AX
	IMULQ $512, AX
	MOVUPD (R10)(AX*1), X9
	MOVQ DX, AX
	IMULQ $768, AX
	MOVUPD (R10)(AX*1), X10

	// Data pointers (byte addressing)
	MOVQ BX, R13
	IMULQ $1024, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), R14
	LEAQ 768(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// complex multiply X1 *= X8
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1
	// complex multiply X2 *= X9
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2
	// complex multiply X3 *= X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3
	// radix-4 butterfly
	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14 // (-i)*t3
	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X14, X1       // y1 = t1 + (-i)*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X14, X5, X3       // y3 = t1 - (-i)*t3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  r4_1024f64_stage3_inner

r4_1024f64_stage3_next:
	INCQ BX
	JMP  r4_1024f64_stage3_outer

r4_1024f64_stage4:
	// ==================================================================
	// Stage 4: 4 groups x 64 butterflies, span=256, twiddle step=4
	// ==================================================================
	XORQ BX, BX              // BX = group index

r4_1024f64_stage4_outer:
	CMPQ BX, $4
	JGE  r4_1024f64_stage5

	XORQ DX, DX              // DX = butterfly index in group

r4_1024f64_stage4_inner:
	CMPQ DX, $64
	JGE  r4_1024f64_stage4_next

	// Twiddles: w[j*4], w[2*j*4], w[3*j*4]
	MOVQ DX, AX
	IMULQ $64, AX
	MOVUPD (R10)(AX*1), X8
	MOVQ DX, AX
	IMULQ $128, AX
	MOVUPD (R10)(AX*1), X9
	MOVQ DX, AX
	IMULQ $192, AX
	MOVUPD (R10)(AX*1), X10

	// Data pointers (byte addressing)
	MOVQ BX, R13
	IMULQ $4096, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 1024(SI), DI
	LEAQ 2048(SI), R14
	LEAQ 3072(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// complex multiply X1 *= X8
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1
	// complex multiply X2 *= X9
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2
	// complex multiply X3 *= X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3
	// radix-4 butterfly
	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14 // (-i)*t3
	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X14, X1       // y1 = t1 + (-i)*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X14, X5, X3       // y3 = t1 - (-i)*t3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  r4_1024f64_stage4_inner

r4_1024f64_stage4_next:
	INCQ BX
	JMP  r4_1024f64_stage4_outer

r4_1024f64_stage5:
	// ==================================================================
	// Stage 5: 1 group x 256 butterflies, span=1024, twiddle step=1 (last)
	// ==================================================================
	XORQ DX, DX

r4_1024f64_stage5_loop:
	CMPQ DX, $256
	JGE  r4_1024f64_done

	// Twiddles: w[j], w[2*j], w[3*j]
	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	MOVQ DX, AX
	SHLQ $5, AX              // 2*j*16
	MOVUPD (R10)(AX*1), X9
	MOVQ DX, AX
	IMULQ $48, AX            // 3*j*16
	MOVUPD (R10)(AX*1), X10

	// Data pointers (byte addressing)
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R8)(SI*1), SI
	LEAQ 4096(SI), DI
	LEAQ 8192(SI), R14
	LEAQ 12288(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// complex multiply X1 *= X8
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X1, X13
	VMOVAPD X13, X1
	// complex multiply X2 *= X9
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X2, X13
	VMOVAPD X13, X2
	// complex multiply X3 *= X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMADDSUB231PD X11, X3, X13
	VMOVAPD X13, X3
	// radix-4 butterfly
	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegHiPD(SB), X14, X14 // (-i)*t3
	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X14, X1       // y1 = t1 + (-i)*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X14, X5, X3       // y3 = t1 - (-i)*t3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  r4_1024f64_stage5_loop

r4_1024f64_done:
	// Copy results to dst if working buffer is scratch
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_1024f64_ret

	XORQ CX, CX

r4_1024f64_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $16384         // 1024 * 16 bytes
	JL   r4_1024f64_copy_loop

r4_1024f64_ret:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

r4_1024f64_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform
// ===========================================================================
TEXT ·InverseAVX2Size1024Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src_len+32(FP), R13 // R13 = n (should be 1024)
	LEAQ ·bitrev1024_r4(SB), R12

	// Verify n == 1024
	CMPQ R13, $1024
	JNE  r4_1024f64_inv_return_false

	// Validate slice lengths
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $1024
	JL   r4_1024f64_inv_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $1024
	JL   r4_1024f64_inv_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $1024
	JL   r4_1024f64_inv_return_false

	// Select working buffer (in-place: use scratch)
	CMPQ R8, R9
	JNE  r4_1024f64_inv_use_dst
	MOVQ R11, R8

r4_1024f64_inv_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 digit reversal)
	// ==================================================================
	XORQ CX, CX

r4_1024f64_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	SHLQ $4, DX
	MOVUPD (R9)(DX*1), X0    // load src[bitrev[i]]
	MOVQ CX, AX
	SHLQ $4, AX
	MOVUPD X0, (R8)(AX*1)    // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $1024
	JL   r4_1024f64_inv_bitrev_loop

r4_1024f64_inv_stage1:
	// ==================================================================
	// Stage 1: 256 groups x 1 butterfly, stride=4, no twiddle (W^0 = 1)
	// ==================================================================
	MOVQ R8, SI
	MOVQ $256, CX

r4_1024f64_inv_stage1_loop:
	MOVUPD (SI), X0          // a0
	MOVUPD 16(SI), X1        // a1
	MOVUPD 32(SI), X2        // a2
	MOVUPD 48(SI), X3        // a3

	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3

	VSHUFPD $0x1, X7, X7, X8
	VXORPD ·maskNegLoPD(SB), X8, X8 // i*t3

	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X8, X1        // y1 = t1 + i*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X8, X5, X3        // y3 = t1 - i*t3

	MOVUPD X0, (SI)
	MOVUPD X1, 16(SI)
	MOVUPD X2, 32(SI)
	MOVUPD X3, 48(SI)

	ADDQ $64, SI
	DECQ CX
	JNZ  r4_1024f64_inv_stage1_loop

r4_1024f64_inv_stage2:
	// ==================================================================
	// Stage 2: 64 groups x 4 butterflies, span=16, twiddle step=64
	// ==================================================================
	XORQ BX, BX              // BX = group index

r4_1024f64_inv_stage2_outer:
	CMPQ BX, $64
	JGE  r4_1024f64_inv_stage3

	XORQ DX, DX              // DX = butterfly index in group

r4_1024f64_inv_stage2_inner:
	CMPQ DX, $4
	JGE  r4_1024f64_inv_stage2_next

	// Twiddles: w[j*64], w[2*j*64], w[3*j*64]
	MOVQ DX, AX
	IMULQ $1024, AX
	MOVUPD (R10)(AX*1), X8
	MOVQ DX, AX
	IMULQ $2048, AX
	MOVUPD (R10)(AX*1), X9
	MOVQ DX, AX
	IMULQ $3072, AX
	MOVUPD (R10)(AX*1), X10

	// Data pointers (byte addressing)
	MOVQ BX, R13
	IMULQ $256, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 64(SI), DI
	LEAQ 128(SI), R14
	LEAQ 192(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// complex multiply X1 *= X8
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1
	// complex multiply X2 *= X9
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2
	// complex multiply X3 *= X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3
	// radix-4 butterfly
	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14 // i*t3
	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X14, X1       // y1 = t1 + i*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X14, X5, X3       // y3 = t1 - i*t3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  r4_1024f64_inv_stage2_inner

r4_1024f64_inv_stage2_next:
	INCQ BX
	JMP  r4_1024f64_inv_stage2_outer

r4_1024f64_inv_stage3:
	// ==================================================================
	// Stage 3: 16 groups x 16 butterflies, span=64, twiddle step=16
	// ==================================================================
	XORQ BX, BX              // BX = group index

r4_1024f64_inv_stage3_outer:
	CMPQ BX, $16
	JGE  r4_1024f64_inv_stage4

	XORQ DX, DX              // DX = butterfly index in group

r4_1024f64_inv_stage3_inner:
	CMPQ DX, $16
	JGE  r4_1024f64_inv_stage3_next

	// Twiddles: w[j*16], w[2*j*16], w[3*j*16]
	MOVQ DX, AX
	IMULQ $256, AX
	MOVUPD (R10)(AX*1), X8
	MOVQ DX, AX
	IMULQ $512, AX
	MOVUPD (R10)(AX*1), X9
	MOVQ DX, AX
	IMULQ $768, AX
	MOVUPD (R10)(AX*1), X10

	// Data pointers (byte addressing)
	MOVQ BX, R13
	IMULQ $1024, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), R14
	LEAQ 768(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// complex multiply X1 *= X8
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1
	// complex multiply X2 *= X9
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2
	// complex multiply X3 *= X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3
	// radix-4 butterfly
	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14 // i*t3
	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X14, X1       // y1 = t1 + i*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X14, X5, X3       // y3 = t1 - i*t3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  r4_1024f64_inv_stage3_inner

r4_1024f64_inv_stage3_next:
	INCQ BX
	JMP  r4_1024f64_inv_stage3_outer

r4_1024f64_inv_stage4:
	// ==================================================================
	// Stage 4: 4 groups x 64 butterflies, span=256, twiddle step=4
	// ==================================================================
	XORQ BX, BX              // BX = group index

r4_1024f64_inv_stage4_outer:
	CMPQ BX, $4
	JGE  r4_1024f64_inv_stage5

	XORQ DX, DX              // DX = butterfly index in group

r4_1024f64_inv_stage4_inner:
	CMPQ DX, $64
	JGE  r4_1024f64_inv_stage4_next

	// Twiddles: w[j*4], w[2*j*4], w[3*j*4]
	MOVQ DX, AX
	IMULQ $64, AX
	MOVUPD (R10)(AX*1), X8
	MOVQ DX, AX
	IMULQ $128, AX
	MOVUPD (R10)(AX*1), X9
	MOVQ DX, AX
	IMULQ $192, AX
	MOVUPD (R10)(AX*1), X10

	// Data pointers (byte addressing)
	MOVQ BX, R13
	IMULQ $4096, R13
	LEAQ (R8)(R13*1), R13
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R13)(SI*1), SI
	LEAQ 1024(SI), DI
	LEAQ 2048(SI), R14
	LEAQ 3072(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// complex multiply X1 *= X8
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1
	// complex multiply X2 *= X9
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2
	// complex multiply X3 *= X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3
	// radix-4 butterfly
	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14 // i*t3
	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X14, X1       // y1 = t1 + i*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X14, X5, X3       // y3 = t1 - i*t3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  r4_1024f64_inv_stage4_inner

r4_1024f64_inv_stage4_next:
	INCQ BX
	JMP  r4_1024f64_inv_stage4_outer

r4_1024f64_inv_stage5:
	// ==================================================================
	// Stage 5: 1 group x 256 butterflies, span=1024, twiddle step=1 (last)
	// ==================================================================
	XORQ DX, DX

r4_1024f64_inv_stage5_loop:
	CMPQ DX, $256
	JGE  r4_1024f64_inv_scale

	// Twiddles: w[j], w[2*j], w[3*j]
	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X8
	MOVQ DX, AX
	SHLQ $5, AX              // 2*j*16
	MOVUPD (R10)(AX*1), X9
	MOVQ DX, AX
	IMULQ $48, AX            // 3*j*16
	MOVUPD (R10)(AX*1), X10

	// Data pointers (byte addressing)
	MOVQ DX, SI
	SHLQ $4, SI
	LEAQ (R8)(SI*1), SI
	LEAQ 4096(SI), DI
	LEAQ 8192(SI), R14
	LEAQ 12288(SI), R15

	MOVUPD (SI), X0
	MOVUPD (DI), X1
	MOVUPD (R14), X2
	MOVUPD (R15), X3

	// complex multiply X1 *= X8
	VMOVDDUP X8, X11
	VPERMILPD $1, X8, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13
	VMOVAPD X13, X1
	// complex multiply X2 *= X9
	VMOVDDUP X9, X11
	VPERMILPD $1, X9, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13
	VMOVAPD X13, X2
	// complex multiply X3 *= X10
	VMOVDDUP X10, X11
	VPERMILPD $1, X10, X12
	VMOVDDUP X12, X12
	VPERMILPD $1, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13
	VMOVAPD X13, X3
	// radix-4 butterfly
	VADDPD X0, X2, X4        // t0 = a0 + a2
	VSUBPD X2, X0, X5        // t1 = a0 - a2
	VADDPD X1, X3, X6        // t2 = a1 + a3
	VSUBPD X3, X1, X7        // t3 = a1 - a3
	VSHUFPD $0x1, X7, X7, X14
	VXORPD ·maskNegLoPD(SB), X14, X14 // i*t3
	VADDPD X4, X6, X0        // y0 = t0 + t2
	VADDPD X5, X14, X1       // y1 = t1 + i*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VSUBPD X14, X5, X3       // y3 = t1 - i*t3

	MOVUPD X0, (SI)
	MOVUPD X1, (DI)
	MOVUPD X2, (R14)
	MOVUPD X3, (R15)

	INCQ DX
	JMP  r4_1024f64_inv_stage5_loop

r4_1024f64_inv_scale:
	// 1/1024 scaling
	MOVSD ·oneThousandTwentyFourth64(SB), X8
	VBROADCASTSD X8, Y8

	XORQ CX, CX

r4_1024f64_inv_scale_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMULPD Y8, Y0, Y0
	VMULPD Y8, Y1, Y1
	VMOVUPD Y0, (R8)(CX*1)
	VMOVUPD Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $16384         // 1024 * 16 bytes
	JL   r4_1024f64_inv_scale_loop

	// Copy to dst if working buffer is scratch
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_1024f64_inv_done

	XORQ CX, CX

r4_1024f64_inv_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $16384
	JL   r4_1024f64_inv_copy_loop

r4_1024f64_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

r4_1024f64_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET
