//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-32 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// Size-specific entrypoints for n==32 that use scalar XMM operations for
// correctness and a fixed-size DIT schedule.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 32, complex128
// ===========================================================================
TEXT ·ForwardAVX2Size32Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // dst pointer
	MOVQ src+24(FP), R9      // src pointer
	MOVQ twiddle+48(FP), R10 // twiddle pointer
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ src+32(FP), R13     // n (should be 32)

	CMPQ R13, $32
	JNE  size32_128_return_false

	// Validate all slice lengths >= 32
	MOVQ dst+8(FP), AX
	CMPQ AX, $32
	JL   size32_128_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $32
	JL   size32_128_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $32
	JL   size32_128_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size32_128_use_dst
	MOVQ R11, R8             // In-place: use scratch as work

size32_128_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// -----------------------------------------------------------------------
	// For size 32, bitrev = [0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,
	//                        1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31]
	// complex128 is 16 bytes
	// Group 0: work[0..3] = src[0,16,8,24]
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD X0, 0(R8)
	MOVUPD 256(R9), X0       // src[16]
	MOVUPD X0, 16(R8)
	MOVUPD 128(R9), X0       // src[8]
	MOVUPD X0, 32(R8)
	MOVUPD 384(R9), X0       // src[24]
	MOVUPD X0, 48(R8)

	// Group 1: work[4..7] = src[4,20,12,28]
	MOVUPD 64(R9), X0        // src[4]
	MOVUPD X0, 64(R8)
	MOVUPD 320(R9), X0       // src[20]
	MOVUPD X0, 80(R8)
	MOVUPD 192(R9), X0       // src[12]
	MOVUPD X0, 96(R8)
	MOVUPD 448(R9), X0       // src[28]
	MOVUPD X0, 112(R8)

	// Group 2: work[8..11] = src[2,18,10,26]
	MOVUPD 32(R9), X0        // src[2]
	MOVUPD X0, 128(R8)
	MOVUPD 288(R9), X0       // src[18]
	MOVUPD X0, 144(R8)
	MOVUPD 160(R9), X0       // src[10]
	MOVUPD X0, 160(R8)
	MOVUPD 416(R9), X0       // src[26]
	MOVUPD X0, 176(R8)

	// Group 3: work[12..15] = src[6,22,14,30]
	MOVUPD 96(R9), X0        // src[6]
	MOVUPD X0, 192(R8)
	MOVUPD 352(R9), X0       // src[22]
	MOVUPD X0, 208(R8)
	MOVUPD 224(R9), X0       // src[14]
	MOVUPD X0, 224(R8)
	MOVUPD 480(R9), X0       // src[30]
	MOVUPD X0, 240(R8)

	// Group 4: work[16..19] = src[1,17,9,25]
	MOVUPD 16(R9), X0        // src[1]
	MOVUPD X0, 256(R8)
	MOVUPD 272(R9), X0       // src[17]
	MOVUPD X0, 272(R8)
	MOVUPD 144(R9), X0       // src[9]
	MOVUPD X0, 288(R8)
	MOVUPD 400(R9), X0       // src[25]
	MOVUPD X0, 304(R8)

	// Group 5: work[20..23] = src[5,21,13,29]
	MOVUPD 80(R9), X0        // src[5]
	MOVUPD X0, 320(R8)
	MOVUPD 336(R9), X0       // src[21]
	MOVUPD X0, 336(R8)
	MOVUPD 208(R9), X0       // src[13]
	MOVUPD X0, 352(R8)
	MOVUPD 464(R9), X0       // src[29]
	MOVUPD X0, 368(R8)

	// Group 6: work[24..27] = src[3,19,11,27]
	MOVUPD 48(R9), X0        // src[3]
	MOVUPD X0, 384(R8)
	MOVUPD 304(R9), X0       // src[19]
	MOVUPD X0, 400(R8)
	MOVUPD 176(R9), X0       // src[11]
	MOVUPD X0, 416(R8)
	MOVUPD 432(R9), X0       // src[27]
	MOVUPD X0, 432(R8)

	// Group 7: work[28..31] = src[7,23,15,31]
	MOVUPD 112(R9), X0       // src[7]
	MOVUPD X0, 448(R8)
	MOVUPD 368(R9), X0       // src[23]
	MOVUPD X0, 464(R8)
	MOVUPD 240(R9), X0       // src[15]
	MOVUPD X0, 480(R8)
	MOVUPD 496(R9), X0       // src[31]
	MOVUPD X0, 496(R8)

size32_128_stage1:
	// -----------------------------------------------------------------------
	// Stage 1: size=2, half=1, step=16, twiddle[0]=1 => t=b
	// -----------------------------------------------------------------------
	XORQ CX, CX              // base

size32_128_stage1_base:
	CMPQ CX, $32
	JGE  size32_128_stage2

	// a index = base, b index = base+1
	MOVQ CX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	INCQ DI
	SHLQ $4, DI
	MOVUPD (R8)(SI*1), X0    // a
	MOVUPD (R8)(DI*1), X1    // b
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, (R8)(SI*1)
	MOVUPD X3, (R8)(DI*1)

	ADDQ $2, CX
	JMP  size32_128_stage1_base

size32_128_stage2:
	// -----------------------------------------------------------------------
	// Stage 2: size=4, half=2, step=8
	// -----------------------------------------------------------------------
	MOVQ $8, BX              // step
	XORQ CX, CX              // base

size32_128_stage2_base:
	CMPQ CX, $32
	JGE  size32_128_stage3

	XORQ DX, DX              // j

size32_128_stage2_j:
	CMPQ DX, $2
	JGE  size32_128_stage2_next

	// Offsets (bytes)
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $2, DI              // +half
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0    // a
	MOVUPD (R8)(DI*1), X1    // b

	// Load twiddle w = twiddle[j*step]
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	// t = w * b
	VMOVDDUP X2, X3          // [w.r, w.r]
	VPERMILPD $1, X2, X4     // [w.i, w.r]
	VMOVDDUP X4, X4          // [w.i, w.i]
	VPERMILPD $1, X1, X6     // [b.i, b.r]
	VMULPD X4, X6, X6        // [w.i*b.i, w.i*b.r]
	VFMADDSUB231PD X3, X1, X6  // X6 = w*b

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size32_128_stage2_j

size32_128_stage2_next:
	ADDQ $4, CX
	JMP  size32_128_stage2_base

size32_128_stage3:
	// -----------------------------------------------------------------------
	// Stage 3: size=8, half=4, step=4
	// -----------------------------------------------------------------------
	MOVQ $4, BX              // step
	XORQ CX, CX              // base

size32_128_stage3_base:
	CMPQ CX, $32
	JGE  size32_128_stage4

	XORQ DX, DX

size32_128_stage3_j:
	CMPQ DX, $4
	JGE  size32_128_stage3_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $4, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size32_128_stage3_j

size32_128_stage3_next:
	ADDQ $8, CX
	JMP  size32_128_stage3_base

size32_128_stage4:
	// -----------------------------------------------------------------------
	// Stage 4: size=16, half=8, step=2
	// -----------------------------------------------------------------------
	MOVQ $2, BX              // step
	XORQ CX, CX              // base

size32_128_stage4_base:
	CMPQ CX, $32
	JGE  size32_128_stage5

	XORQ DX, DX

size32_128_stage4_j:
	CMPQ DX, $8
	JGE  size32_128_stage4_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $8, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size32_128_stage4_j

size32_128_stage4_next:
	ADDQ $16, CX
	JMP  size32_128_stage4_base

size32_128_stage5:
	// -----------------------------------------------------------------------
	// Stage 5: size=32, half=16, step=1
	// -----------------------------------------------------------------------
	MOVQ $1, BX              // step
	XORQ CX, CX              // base=0 only
	XORQ DX, DX              // j

size32_128_stage5_j:
	CMPQ DX, $16
	JGE  size32_128_finalize

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $16, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX              // j*step (step=1)
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size32_128_stage5_j

size32_128_finalize:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size32_128_done

	XORQ CX, CX
size32_128_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $512
	JL   size32_128_copy_loop

size32_128_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size32_128_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 32, complex128
// ===========================================================================
TEXT ·InverseAVX2Size32Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	CMPQ R13, $32
	JNE  size32_inv_128_return_false

	// Validate all slice lengths >= 32
	MOVQ dst+8(FP), AX
	CMPQ AX, $32
	JL   size32_inv_128_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $32
	JL   size32_inv_128_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $32
	JL   size32_inv_128_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size32_inv_128_use_dst
	MOVQ R11, R8

size32_inv_128_use_dst:
	// Bit-reversal permutation
	// For size 32, bitrev = [0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,
	//                        1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31]
	// complex128 is 16 bytes
	// Group 0: work[0..3] = src[0,16,8,24]
	MOVUPD 0(R9), X0         // src[0]
	MOVUPD X0, 0(R8)
	MOVUPD 256(R9), X0       // src[16]
	MOVUPD X0, 16(R8)
	MOVUPD 128(R9), X0       // src[8]
	MOVUPD X0, 32(R8)
	MOVUPD 384(R9), X0       // src[24]
	MOVUPD X0, 48(R8)

	// Group 1: work[4..7] = src[4,20,12,28]
	MOVUPD 64(R9), X0        // src[4]
	MOVUPD X0, 64(R8)
	MOVUPD 320(R9), X0       // src[20]
	MOVUPD X0, 80(R8)
	MOVUPD 192(R9), X0       // src[12]
	MOVUPD X0, 96(R8)
	MOVUPD 448(R9), X0       // src[28]
	MOVUPD X0, 112(R8)

	// Group 2: work[8..11] = src[2,18,10,26]
	MOVUPD 32(R9), X0        // src[2]
	MOVUPD X0, 128(R8)
	MOVUPD 288(R9), X0       // src[18]
	MOVUPD X0, 144(R8)
	MOVUPD 160(R9), X0       // src[10]
	MOVUPD X0, 160(R8)
	MOVUPD 416(R9), X0       // src[26]
	MOVUPD X0, 176(R8)

	// Group 3: work[12..15] = src[6,22,14,30]
	MOVUPD 96(R9), X0        // src[6]
	MOVUPD X0, 192(R8)
	MOVUPD 352(R9), X0       // src[22]
	MOVUPD X0, 208(R8)
	MOVUPD 224(R9), X0       // src[14]
	MOVUPD X0, 224(R8)
	MOVUPD 480(R9), X0       // src[30]
	MOVUPD X0, 240(R8)

	// Group 4: work[16..19] = src[1,17,9,25]
	MOVUPD 16(R9), X0        // src[1]
	MOVUPD X0, 256(R8)
	MOVUPD 272(R9), X0       // src[17]
	MOVUPD X0, 272(R8)
	MOVUPD 144(R9), X0       // src[9]
	MOVUPD X0, 288(R8)
	MOVUPD 400(R9), X0       // src[25]
	MOVUPD X0, 304(R8)

	// Group 5: work[20..23] = src[5,21,13,29]
	MOVUPD 80(R9), X0        // src[5]
	MOVUPD X0, 320(R8)
	MOVUPD 336(R9), X0       // src[21]
	MOVUPD X0, 336(R8)
	MOVUPD 208(R9), X0       // src[13]
	MOVUPD X0, 352(R8)
	MOVUPD 464(R9), X0       // src[29]
	MOVUPD X0, 368(R8)

	// Group 6: work[24..27] = src[3,19,11,27]
	MOVUPD 48(R9), X0        // src[3]
	MOVUPD X0, 384(R8)
	MOVUPD 304(R9), X0       // src[19]
	MOVUPD X0, 400(R8)
	MOVUPD 176(R9), X0       // src[11]
	MOVUPD X0, 416(R8)
	MOVUPD 432(R9), X0       // src[27]
	MOVUPD X0, 432(R8)

	// Group 7: work[28..31] = src[7,23,15,31]
	MOVUPD 112(R9), X0       // src[7]
	MOVUPD X0, 448(R8)
	MOVUPD 368(R9), X0       // src[23]
	MOVUPD X0, 464(R8)
	MOVUPD 240(R9), X0       // src[15]
	MOVUPD X0, 480(R8)
	MOVUPD 496(R9), X0       // src[31]
	MOVUPD X0, 496(R8)

size32_inv_128_stage1:
	// Stage 1: size=2, half=1, step=16 (twiddle[0]=1)
	XORQ CX, CX

size32_inv_128_stage1_base:
	CMPQ CX, $32
	JGE  size32_inv_128_stage2
	MOVQ CX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	INCQ DI
	SHLQ $4, DI
	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1
	VADDPD X1, X0, X2
	VSUBPD X1, X0, X3
	MOVUPD X2, (R8)(SI*1)
	MOVUPD X3, (R8)(DI*1)
	ADDQ $2, CX
	JMP  size32_inv_128_stage1_base

size32_inv_128_stage2:
	MOVQ $8, BX
	XORQ CX, CX

size32_inv_128_stage2_base:
	CMPQ CX, $32
	JGE  size32_inv_128_stage3
	XORQ DX, DX

size32_inv_128_stage2_j:
	CMPQ DX, $2
	JGE  size32_inv_128_stage2_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $2, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6  // conj(w) * b

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size32_inv_128_stage2_j

size32_inv_128_stage2_next:
	ADDQ $4, CX
	JMP  size32_inv_128_stage2_base

size32_inv_128_stage3:
	MOVQ $4, BX
	XORQ CX, CX

size32_inv_128_stage3_base:
	CMPQ CX, $32
	JGE  size32_inv_128_stage4
	XORQ DX, DX

size32_inv_128_stage3_j:
	CMPQ DX, $4
	JGE  size32_inv_128_stage3_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $4, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size32_inv_128_stage3_j

size32_inv_128_stage3_next:
	ADDQ $8, CX
	JMP  size32_inv_128_stage3_base

size32_inv_128_stage4:
	MOVQ $2, BX
	XORQ CX, CX

size32_inv_128_stage4_base:
	CMPQ CX, $32
	JGE  size32_inv_128_stage5
	XORQ DX, DX

size32_inv_128_stage4_j:
	CMPQ DX, $8
	JGE  size32_inv_128_stage4_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $8, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size32_inv_128_stage4_j

size32_inv_128_stage4_next:
	ADDQ $16, CX
	JMP  size32_inv_128_stage4_base

size32_inv_128_stage5:
	MOVQ $1, BX
	XORQ CX, CX
	XORQ DX, DX

size32_inv_128_stage5_j:
	CMPQ DX, $16
	JGE  size32_inv_128_scale

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $16, DI
	SHLQ $4, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VPERMILPD $1, X2, X4
	VMOVDDUP X4, X4
	VPERMILPD $1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size32_inv_128_stage5_j

size32_inv_128_scale:
	// Apply 1/n scaling (1/32)
	MOVQ ·thirtySecond64(SB), AX
	VMOVQ AX, X9
	VMOVDDUP X9, X9

	XORQ CX, CX
size32_inv_128_scale_loop:
	CMPQ CX, $32
	JGE  size32_inv_128_finalize
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0
	VMULPD X9, X0, X0
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size32_inv_128_scale_loop

size32_inv_128_finalize:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size32_inv_128_done

	XORQ CX, CX
size32_inv_128_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $512
	JL   size32_inv_128_copy_loop

size32_inv_128_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size32_inv_128_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET
