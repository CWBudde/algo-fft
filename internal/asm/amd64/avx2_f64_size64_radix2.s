//go:build amd64 && !purego

// ===========================================================================
// AVX2 Size-64 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// Size-specific entrypoints for n==64 that use XMM operations for
// correctness and a fixed-size DIT schedule.
//
// Radix-2: 6 stages (64 = 2^6)
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 64, complex128, radix-2
// ===========================================================================
TEXT ·ForwardAVX2Size64Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // dst pointer
	MOVQ src+24(FP), R9      // src pointer
	MOVQ twiddle+48(FP), R10 // twiddle pointer
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ src_len+32(FP), R13     // n (should be 64)

	CMPQ R13, $64
	JNE  size64_128_r2_return_false

	// Validate all slice lengths >= 64
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $64
	JL   size64_128_r2_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $64
	JL   size64_128_r2_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $64
	JL   size64_128_r2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size64_128_r2_use_dst
	MOVQ R11, R8             // In-place: use scratch as work

size64_128_r2_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation + Stage 1 (identity twiddles)
	// -----------------------------------------------------------------------
	// Bitrev pattern: [0,32,16,48, 8,40,24,56, 4,36,20,52, 12,44,28,60,
	//                  2,34,18,50, 10,42,26,58, 6,38,22,54, 14,46,30,62,
	//                  1,33,17,49, 9,41,25,57, 5,37,21,53, 13,45,29,61,
	//                  3,35,19,51, 11,43,27,59, 7,39,23,55, 15,47,31,63]
	// Stage 1 butterflies: a' = a + b, b' = a - b (twiddle[0] = 1).
	//
	// TODO: Check if a smaller loop odd/even is more efficient here.

	// (0,32) -> work[0], work[1]
	VMOVUPD 0(R9), X0
	VMOVUPD 512(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 0(R8)
	VMOVUPD X3, 16(R8)

	// (16,48) -> work[2], work[3]
	VMOVUPD 256(R9), X0
	VMOVUPD 768(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 32(R8)
	VMOVUPD X3, 48(R8)

	// (8,40) -> work[4], work[5]
	VMOVUPD 128(R9), X0
	VMOVUPD 640(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 64(R8)
	VMOVUPD X3, 80(R8)

	// (24,56) -> work[6], work[7]
	VMOVUPD 384(R9), X0
	VMOVUPD 896(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 96(R8)
	VMOVUPD X3, 112(R8)

	// (4,36) -> work[8], work[9]
	VMOVUPD 64(R9), X0
	VMOVUPD 576(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 128(R8)
	VMOVUPD X3, 144(R8)

	// (20,52) -> work[10], work[11]
	VMOVUPD 320(R9), X0
	VMOVUPD 832(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 160(R8)
	VMOVUPD X3, 176(R8)

	// (12,44) -> work[12], work[13]
	VMOVUPD 192(R9), X0
	VMOVUPD 704(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 192(R8)
	VMOVUPD X3, 208(R8)

	// (28,60) -> work[14], work[15]
	VMOVUPD 448(R9), X0
	VMOVUPD 960(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 224(R8)
	VMOVUPD X3, 240(R8)

	// (2,34) -> work[16], work[17]
	VMOVUPD 32(R9), X0
	VMOVUPD 544(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 256(R8)
	VMOVUPD X3, 272(R8)

	// (18,50) -> work[18], work[19]
	VMOVUPD 288(R9), X0
	VMOVUPD 800(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 288(R8)
	VMOVUPD X3, 304(R8)

	// (10,42) -> work[20], work[21]
	VMOVUPD 160(R9), X0
	VMOVUPD 672(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 320(R8)
	VMOVUPD X3, 336(R8)

	// (26,58) -> work[22], work[23]
	VMOVUPD 416(R9), X0
	VMOVUPD 928(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 352(R8)
	VMOVUPD X3, 368(R8)

	// (6,38) -> work[24], work[25]
	VMOVUPD 96(R9), X0
	VMOVUPD 608(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 384(R8)
	VMOVUPD X3, 400(R8)

	// (22,54) -> work[26], work[27]
	VMOVUPD 352(R9), X0
	VMOVUPD 864(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 416(R8)
	VMOVUPD X3, 432(R8)

	// (14,46) -> work[28], work[29]
	VMOVUPD 224(R9), X0
	VMOVUPD 736(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 448(R8)
	VMOVUPD X3, 464(R8)

	// (30,62) -> work[30], work[31]
	VMOVUPD 480(R9), X0
	VMOVUPD 992(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 480(R8)
	VMOVUPD X3, 496(R8)

	// (1,33) -> work[32], work[33]
	VMOVUPD 16(R9), X0
	VMOVUPD 528(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 512(R8)
	VMOVUPD X3, 528(R8)

	// (17,49) -> work[34], work[35]
	VMOVUPD 272(R9), X0
	VMOVUPD 784(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 544(R8)
	VMOVUPD X3, 560(R8)

	// (9,41) -> work[36], work[37]
	VMOVUPD 144(R9), X0
	VMOVUPD 656(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 576(R8)
	VMOVUPD X3, 592(R8)

	// (25,57) -> work[38], work[39]
	VMOVUPD 400(R9), X0
	VMOVUPD 912(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 608(R8)
	VMOVUPD X3, 624(R8)

	// (5,37) -> work[40], work[41]
	VMOVUPD 80(R9), X0
	VMOVUPD 592(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 640(R8)
	VMOVUPD X3, 656(R8)

	// (21,53) -> work[42], work[43]
	VMOVUPD 336(R9), X0
	VMOVUPD 848(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 672(R8)
	VMOVUPD X3, 688(R8)

	// (13,45) -> work[44], work[45]
	VMOVUPD 208(R9), X0
	VMOVUPD 720(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 704(R8)
	VMOVUPD X3, 720(R8)

	// (29,61) -> work[46], work[47]
	VMOVUPD 464(R9), X0
	VMOVUPD 976(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 736(R8)
	VMOVUPD X3, 752(R8)

	// (3,35) -> work[48], work[49]
	VMOVUPD 48(R9), X0
	VMOVUPD 560(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 768(R8)
	VMOVUPD X3, 784(R8)

	// (19,51) -> work[50], work[51]
	VMOVUPD 304(R9), X0
	VMOVUPD 816(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 800(R8)
	VMOVUPD X3, 816(R8)

	// (11,43) -> work[52], work[53]
	VMOVUPD 176(R9), X0
	VMOVUPD 688(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 832(R8)
	VMOVUPD X3, 848(R8)

	// (27,59) -> work[54], work[55]
	VMOVUPD 432(R9), X0
	VMOVUPD 944(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 864(R8)
	VMOVUPD X3, 880(R8)

	// (7,39) -> work[56], work[57]
	VMOVUPD 112(R9), X0
	VMOVUPD 624(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 896(R8)
	VMOVUPD X3, 912(R8)

	// (23,55) -> work[58], work[59]
	VMOVUPD 368(R9), X0
	VMOVUPD 880(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 928(R8)
	VMOVUPD X3, 944(R8)

	// (15,47) -> work[60], work[61]
	VMOVUPD 240(R9), X0
	VMOVUPD 752(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 960(R8)
	VMOVUPD X3, 976(R8)

	// (31,63) -> work[62], work[63]
	VMOVUPD 496(R9), X0
	VMOVUPD 1008(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 992(R8)
	VMOVUPD X3, 1008(R8)

size64_128_r2_stage2:
	// -----------------------------------------------------------------------
	// Stage 2: size=4, half=2, step=16
	// -----------------------------------------------------------------------
	MOVQ $16, BX             // step
	XORQ CX, CX              // base

size64_128_r2_stage2_base:
	CMPQ CX, $64
	JGE  size64_128_r2_stage3

	XORQ DX, DX              // j

size64_128_r2_stage2_j:
	CMPQ DX, $2
	JGE  size64_128_r2_stage2_next

	// Offsets (bytes)
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $2, DI              // +half
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0    // a
	VMOVUPD (R8)(DI*1), X1    // b

	// Load twiddle w = twiddle[j*step]
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	// t = w * b
	VMOVDDUP X2, X3          // [w.r, w.r]
	VPERMILPD $1, X2, X4     // [w.i, w.r]
	VMOVDDUP X4, X4          // [w.i, w.i]
	VPERMILPD $1, X1, X6     // [b.i, b.r]
	VMULPD X4, X6, X6        // [w.i*b.i, w.i*b.r]
	VFMADDSUB231PD X3, X1, X6  // X6 = w*b

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_128_r2_stage2_j

size64_128_r2_stage2_next:
	ADDQ $4, CX
	JMP  size64_128_r2_stage2_base

size64_128_r2_stage3:
	// -----------------------------------------------------------------------
	// Stage 3: size=8, half=4, step=8
	// -----------------------------------------------------------------------
	MOVQ $8, BX              // step
	XORQ CX, CX              // base

size64_128_r2_stage3_base:
	CMPQ CX, $64
	JGE  size64_128_r2_stage4

	XORQ DX, DX

size64_128_r2_stage3_j:
	CMPQ DX, $4
	JGE  size64_128_r2_stage3_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $4, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0
	VMOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_128_r2_stage3_j

size64_128_r2_stage3_next:
	ADDQ $8, CX
	JMP  size64_128_r2_stage3_base

size64_128_r2_stage4:
	// -----------------------------------------------------------------------
	// Stage 4: size=16, half=8, step=4
	// -----------------------------------------------------------------------
	MOVQ $4, BX              // step
	XORQ CX, CX              // base

size64_128_r2_stage4_base:
	CMPQ CX, $64
	JGE  size64_128_r2_stage5

	XORQ DX, DX

size64_128_r2_stage4_j:
	CMPQ DX, $8
	JGE  size64_128_r2_stage4_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $8, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0
	VMOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_128_r2_stage4_j

size64_128_r2_stage4_next:
	ADDQ $16, CX
	JMP  size64_128_r2_stage4_base

size64_128_r2_stage5:
	// -----------------------------------------------------------------------
	// Stage 5: size=32, half=16, step=2
	// -----------------------------------------------------------------------
	MOVQ $2, BX              // step
	XORQ CX, CX              // base

size64_128_r2_stage5_base:
	CMPQ CX, $64
	JGE  size64_128_r2_stage6

	XORQ DX, DX

size64_128_r2_stage5_j:
	CMPQ DX, $16
	JGE  size64_128_r2_stage5_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $16, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0
	VMOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_128_r2_stage5_j

size64_128_r2_stage5_next:
	ADDQ $32, CX
	JMP  size64_128_r2_stage5_base

size64_128_r2_stage6:
	// -----------------------------------------------------------------------
	// Stage 6: size=64, half=32, step=1
	// -----------------------------------------------------------------------
	MOVQ $1, BX              // step
	XORQ CX, CX              // base=0 only
	XORQ DX, DX              // j

size64_128_r2_stage6_j:
	CMPQ DX, $32
	JGE  size64_128_r2_finalize

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $32, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0
	VMOVUPD (R8)(DI*1), X1

	MOVQ DX, AX              // j*step (step=1)
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_128_r2_stage6_j

size64_128_r2_finalize:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size64_128_r2_done

	XORQ CX, CX
size64_128_r2_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $1024           // 64 * 16 bytes
	JL   size64_128_r2_copy_loop

size64_128_r2_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size64_128_r2_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 64, complex128, radix-2
// ===========================================================================
TEXT ·InverseAVX2Size64Radix2Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src_len+32(FP), R13

	CMPQ R13, $64
	JNE  size64_inv_128_r2_return_false

	// Validate all slice lengths >= 64
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $64
	JL   size64_inv_128_r2_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $64
	JL   size64_inv_128_r2_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $64
	JL   size64_inv_128_r2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size64_inv_128_r2_use_dst
	MOVQ R11, R8

size64_inv_128_r2_use_dst:
	// -----------------------------------------------------------------------
	// FUSED: Bit-reversal permutation + Stage 1 (identity twiddles)
	// -----------------------------------------------------------------------
	// Bitrev pattern: [0,32,16,48, 8,40,24,56, 4,36,20,52, 12,44,28,60,
	//                  2,34,18,50, 10,42,26,58, 6,38,22,54, 14,46,30,62,
	//                  1,33,17,49, 9,41,25,57, 5,37,21,53, 13,45,29,61,
	//                  3,35,19,51, 11,43,27,59, 7,39,23,55, 15,47,31,63]
	// Stage 1 butterflies: a' = a + b, b' = a - b (twiddle[0] = 1).

	// (0,32) -> work[0], work[1]
	VMOVUPD 0(R9), X0
	VMOVUPD 512(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 0(R8)
	VMOVUPD X3, 16(R8)

	// (16,48) -> work[2], work[3]
	VMOVUPD 256(R9), X0
	VMOVUPD 768(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 32(R8)
	VMOVUPD X3, 48(R8)

	// (8,40) -> work[4], work[5]
	VMOVUPD 128(R9), X0
	VMOVUPD 640(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 64(R8)
	VMOVUPD X3, 80(R8)

	// (24,56) -> work[6], work[7]
	VMOVUPD 384(R9), X0
	VMOVUPD 896(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 96(R8)
	VMOVUPD X3, 112(R8)

	// (4,36) -> work[8], work[9]
	VMOVUPD 64(R9), X0
	VMOVUPD 576(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 128(R8)
	VMOVUPD X3, 144(R8)

	// (20,52) -> work[10], work[11]
	VMOVUPD 320(R9), X0
	VMOVUPD 832(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 160(R8)
	VMOVUPD X3, 176(R8)

	// (12,44) -> work[12], work[13]
	VMOVUPD 192(R9), X0
	VMOVUPD 704(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 192(R8)
	VMOVUPD X3, 208(R8)

	// (28,60) -> work[14], work[15]
	VMOVUPD 448(R9), X0
	VMOVUPD 960(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 224(R8)
	VMOVUPD X3, 240(R8)

	// (2,34) -> work[16], work[17]
	VMOVUPD 32(R9), X0
	VMOVUPD 544(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 256(R8)
	VMOVUPD X3, 272(R8)

	// (18,50) -> work[18], work[19]
	VMOVUPD 288(R9), X0
	VMOVUPD 800(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 288(R8)
	VMOVUPD X3, 304(R8)

	// (10,42) -> work[20], work[21]
	VMOVUPD 160(R9), X0
	VMOVUPD 672(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 320(R8)
	VMOVUPD X3, 336(R8)

	// (26,58) -> work[22], work[23]
	VMOVUPD 416(R9), X0
	VMOVUPD 928(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 352(R8)
	VMOVUPD X3, 368(R8)

	// (6,38) -> work[24], work[25]
	VMOVUPD 96(R9), X0
	VMOVUPD 608(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 384(R8)
	VMOVUPD X3, 400(R8)

	// (22,54) -> work[26], work[27]
	VMOVUPD 352(R9), X0
	VMOVUPD 864(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 416(R8)
	VMOVUPD X3, 432(R8)

	// (14,46) -> work[28], work[29]
	VMOVUPD 224(R9), X0
	VMOVUPD 736(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 448(R8)
	VMOVUPD X3, 464(R8)

	// (30,62) -> work[30], work[31]
	VMOVUPD 480(R9), X0
	VMOVUPD 992(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 480(R8)
	VMOVUPD X3, 496(R8)

	// (1,33) -> work[32], work[33]
	VMOVUPD 16(R9), X0
	VMOVUPD 528(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 512(R8)
	VMOVUPD X3, 528(R8)

	// (17,49) -> work[34], work[35]
	VMOVUPD 272(R9), X0
	VMOVUPD 784(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 544(R8)
	VMOVUPD X3, 560(R8)

	// (9,41) -> work[36], work[37]
	VMOVUPD 144(R9), X0
	VMOVUPD 656(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 576(R8)
	VMOVUPD X3, 592(R8)

	// (25,57) -> work[38], work[39]
	VMOVUPD 400(R9), X0
	VMOVUPD 912(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 608(R8)
	VMOVUPD X3, 624(R8)

	// (5,37) -> work[40], work[41]
	VMOVUPD 80(R9), X0
	VMOVUPD 592(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 640(R8)
	VMOVUPD X3, 656(R8)

	// (21,53) -> work[42], work[43]
	VMOVUPD 336(R9), X0
	VMOVUPD 848(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 672(R8)
	VMOVUPD X3, 688(R8)

	// (13,45) -> work[44], work[45]
	VMOVUPD 208(R9), X0
	VMOVUPD 720(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 704(R8)
	VMOVUPD X3, 720(R8)

	// (29,61) -> work[46], work[47]
	VMOVUPD 464(R9), X0
	VMOVUPD 976(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 736(R8)
	VMOVUPD X3, 752(R8)

	// (3,35) -> work[48], work[49]
	VMOVUPD 48(R9), X0
	VMOVUPD 560(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 768(R8)
	VMOVUPD X3, 784(R8)

	// (19,51) -> work[50], work[51]
	VMOVUPD 304(R9), X0
	VMOVUPD 816(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 800(R8)
	VMOVUPD X3, 816(R8)

	// (11,43) -> work[52], work[53]
	VMOVUPD 176(R9), X0
	VMOVUPD 688(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 832(R8)
	VMOVUPD X3, 848(R8)

	// (27,59) -> work[54], work[55]
	VMOVUPD 432(R9), X0
	VMOVUPD 944(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 864(R8)
	VMOVUPD X3, 880(R8)

	// (7,39) -> work[56], work[57]
	VMOVUPD 112(R9), X0
	VMOVUPD 624(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 896(R8)
	VMOVUPD X3, 912(R8)

	// (23,55) -> work[58], work[59]
	VMOVUPD 368(R9), X0
	VMOVUPD 880(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 928(R8)
	VMOVUPD X3, 944(R8)

	// (15,47) -> work[60], work[61]
	VMOVUPD 240(R9), X0
	VMOVUPD 752(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 960(R8)
	VMOVUPD X3, 976(R8)

	// (31,63) -> work[62], work[63]
	VMOVUPD 496(R9), X0
	VMOVUPD 1008(R9), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	VMOVUPD X2, 992(R8)
	VMOVUPD X3, 1008(R8)

size64_inv_128_r2_stage2:
	MOVQ $16, BX
	XORQ CX, CX

size64_inv_128_r2_stage2_base:
	CMPQ CX, $64
	JGE  size64_inv_128_r2_stage3
	XORQ DX, DX

size64_inv_128_r2_stage2_j:
	CMPQ DX, $2
	JGE  size64_inv_128_r2_stage2_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $2, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0
	VMOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6  // conj(w) * b

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_inv_128_r2_stage2_j

size64_inv_128_r2_stage2_next:
	ADDQ $4, CX
	JMP  size64_inv_128_r2_stage2_base

size64_inv_128_r2_stage3:
	MOVQ $8, BX
	XORQ CX, CX

size64_inv_128_r2_stage3_base:
	CMPQ CX, $64
	JGE  size64_inv_128_r2_stage4
	XORQ DX, DX

size64_inv_128_r2_stage3_j:
	CMPQ DX, $4
	JGE  size64_inv_128_r2_stage3_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $4, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0
	VMOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_inv_128_r2_stage3_j

size64_inv_128_r2_stage3_next:
	ADDQ $8, CX
	JMP  size64_inv_128_r2_stage3_base

size64_inv_128_r2_stage4:
	MOVQ $4, BX
	XORQ CX, CX

size64_inv_128_r2_stage4_base:
	CMPQ CX, $64
	JGE  size64_inv_128_r2_stage5
	XORQ DX, DX

size64_inv_128_r2_stage4_j:
	CMPQ DX, $8
	JGE  size64_inv_128_r2_stage4_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $8, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0
	VMOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_inv_128_r2_stage4_j

size64_inv_128_r2_stage4_next:
	ADDQ $16, CX
	JMP  size64_inv_128_r2_stage4_base

size64_inv_128_r2_stage5:
	MOVQ $2, BX
	XORQ CX, CX

size64_inv_128_r2_stage5_base:
	CMPQ CX, $64
	JGE  size64_inv_128_r2_stage6
	XORQ DX, DX

size64_inv_128_r2_stage5_j:
	CMPQ DX, $16
	JGE  size64_inv_128_r2_stage5_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $16, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0
	VMOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_inv_128_r2_stage5_j

size64_inv_128_r2_stage5_next:
	ADDQ $32, CX
	JMP  size64_inv_128_r2_stage5_base

size64_inv_128_r2_stage6:
	MOVQ $1, BX
	XORQ CX, CX
	XORQ DX, DX

size64_inv_128_r2_stage6_j:
	CMPQ DX, $32
	JGE  size64_inv_128_r2_scale

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $32, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0
	VMOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	VMOVUPD X7, (R8)(SI*1)
	VMOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size64_inv_128_r2_stage6_j

size64_inv_128_r2_scale:
	// Apply 1/n scaling (1/64 = 0.015625)
	MOVQ ·sixtyFourth64(SB), AX  // 1/64 in float64
	VMOVQ AX, X9
	VMOVDDUP X9, X9

	XORQ CX, CX
size64_inv_128_r2_scale_loop:
	CMPQ CX, $64
	JGE  size64_inv_128_r2_finalize
	MOVQ CX, SI
	SHLQ $4, SI
	VMOVUPD (R8)(SI*1), X0
	VMULPD X9, X0, X0
	VMOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size64_inv_128_r2_scale_loop

size64_inv_128_r2_finalize:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size64_inv_128_r2_done

	XORQ CX, CX
size64_inv_128_r2_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $1024
	JL   size64_inv_128_r2_copy_loop

size64_inv_128_r2_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size64_inv_128_r2_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET
