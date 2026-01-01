//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-256 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// Size-specific entrypoints for n==256 that use XMM operations for
// correctness and a fixed-size DIT schedule.
//
// Radix-2: 8 stages (256 = 2^8)
// Radix-4: 4 stages (256 = 4^4)
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 256, complex128, radix-2
// ===========================================================================
TEXT ·forwardAVX2Size256Radix2Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // dst pointer
	MOVQ src+24(FP), R9      // src pointer
	MOVQ twiddle+48(FP), R10 // twiddle pointer
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ bitrev+96(FP), R12  // bitrev pointer
	MOVQ src+32(FP), R13     // n (should be 256)

	CMPQ R13, $256
	JNE  size256_128_r2_return_false

	// Validate all slice lengths >= 256
	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   size256_128_r2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   size256_128_r2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   size256_128_r2_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $256
	JL   size256_128_r2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size256_128_r2_use_dst
	MOVQ R11, R8

size256_128_r2_use_dst:
	// Bit-reversal permutation
	XORQ CX, CX

size256_128_r2_bitrev_loop:
	CMPQ CX, $256
	JGE  size256_128_r2_stage1

	MOVQ (R12)(CX*8), DX
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD (R9)(SI*1), X0
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size256_128_r2_bitrev_loop

size256_128_r2_stage1:
	// Stage 1: size=2, half=1
	XORQ CX, CX

size256_128_r2_stage1_base:
	CMPQ CX, $256
	JGE  size256_128_r2_stage2

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
	JMP  size256_128_r2_stage1_base

size256_128_r2_stage2:
	// Stage 2: size=4, half=2, step=64
	MOVQ $64, BX
	XORQ CX, CX

size256_128_r2_stage2_base:
	CMPQ CX, $256
	JGE  size256_128_r2_stage3

	XORQ DX, DX

size256_128_r2_stage2_j:
	CMPQ DX, $2
	JGE  size256_128_r2_stage2_next

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
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size256_128_r2_stage2_j

size256_128_r2_stage2_next:
	ADDQ $4, CX
	JMP  size256_128_r2_stage2_base

size256_128_r2_stage3:
	// Stage 3: size=8, half=4, step=32
	MOVQ $32, BX
	XORQ CX, CX

size256_128_r2_stage3_base:
	CMPQ CX, $256
	JGE  size256_128_r2_stage4

	XORQ DX, DX

size256_128_r2_stage3_j:
	CMPQ DX, $4
	JGE  size256_128_r2_stage3_next

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
	JMP  size256_128_r2_stage3_j

size256_128_r2_stage3_next:
	ADDQ $8, CX
	JMP  size256_128_r2_stage3_base

size256_128_r2_stage4:
	// Stage 4: size=16, half=8, step=16
	MOVQ $16, BX
	XORQ CX, CX

size256_128_r2_stage4_base:
	CMPQ CX, $256
	JGE  size256_128_r2_stage5

	XORQ DX, DX

size256_128_r2_stage4_j:
	CMPQ DX, $8
	JGE  size256_128_r2_stage4_next

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
	JMP  size256_128_r2_stage4_j

size256_128_r2_stage4_next:
	ADDQ $16, CX
	JMP  size256_128_r2_stage4_base

size256_128_r2_stage5:
	// Stage 5: size=32, half=16, step=8
	MOVQ $8, BX
	XORQ CX, CX

size256_128_r2_stage5_base:
	CMPQ CX, $256
	JGE  size256_128_r2_stage6

	XORQ DX, DX

size256_128_r2_stage5_j:
	CMPQ DX, $16
	JGE  size256_128_r2_stage5_next

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
	JMP  size256_128_r2_stage5_j

size256_128_r2_stage5_next:
	ADDQ $32, CX
	JMP  size256_128_r2_stage5_base

size256_128_r2_stage6:
	// Stage 6: size=64, half=32, step=4
	MOVQ $4, BX
	XORQ CX, CX

size256_128_r2_stage6_base:
	CMPQ CX, $256
	JGE  size256_128_r2_stage7

	XORQ DX, DX

size256_128_r2_stage6_j:
	CMPQ DX, $32
	JGE  size256_128_r2_stage6_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $32, DI
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
	JMP  size256_128_r2_stage6_j

size256_128_r2_stage6_next:
	ADDQ $64, CX
	JMP  size256_128_r2_stage6_base

size256_128_r2_stage7:
	// Stage 7: size=128, half=64, step=2
	MOVQ $2, BX
	XORQ CX, CX

size256_128_r2_stage7_base:
	CMPQ CX, $256
	JGE  size256_128_r2_stage8

	XORQ DX, DX

size256_128_r2_stage7_j:
	CMPQ DX, $64
	JGE  size256_128_r2_stage7_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $64, DI
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
	JMP  size256_128_r2_stage7_j

size256_128_r2_stage7_next:
	ADDQ $128, CX
	JMP  size256_128_r2_stage7_base

size256_128_r2_stage8:
	// Stage 8: size=256, half=128, step=1
	MOVQ $1, BX
	XORQ CX, CX
	XORQ DX, DX

size256_128_r2_stage8_j:
	CMPQ DX, $128
	JGE  size256_128_r2_finalize

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $128, DI
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
	VFMADDSUB231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size256_128_r2_stage8_j

size256_128_r2_finalize:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size256_128_r2_done

	XORQ CX, CX
size256_128_r2_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $4096           // 256 * 16 bytes
	JL   size256_128_r2_copy_loop

size256_128_r2_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size256_128_r2_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 256, complex128, radix-2
// ===========================================================================
TEXT ·inverseAVX2Size256Radix2Complex128Asm(SB), NOSPLIT, $0-121
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $256
	JNE  size256_inv_128_r2_return_false

	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   size256_inv_128_r2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   size256_inv_128_r2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   size256_inv_128_r2_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $256
	JL   size256_inv_128_r2_return_false

	CMPQ R8, R9
	JNE  size256_inv_128_r2_use_dst
	MOVQ R11, R8

size256_inv_128_r2_use_dst:
	// Bit-reversal permutation
	XORQ CX, CX

size256_inv_128_r2_bitrev_loop:
	CMPQ CX, $256
	JGE  size256_inv_128_r2_stage1
	MOVQ (R12)(CX*8), DX
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD (R9)(SI*1), X0
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size256_inv_128_r2_bitrev_loop

size256_inv_128_r2_stage1:
	XORQ CX, CX

size256_inv_128_r2_stage1_base:
	CMPQ CX, $256
	JGE  size256_inv_128_r2_stage2
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
	JMP  size256_inv_128_r2_stage1_base

size256_inv_128_r2_stage2:
	MOVQ $64, BX
	XORQ CX, CX

size256_inv_128_r2_stage2_base:
	CMPQ CX, $256
	JGE  size256_inv_128_r2_stage3
	XORQ DX, DX

size256_inv_128_r2_stage2_j:
	CMPQ DX, $2
	JGE  size256_inv_128_r2_stage2_next

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
	VFMSUBADD231PD X3, X1, X6

	VADDPD X6, X0, X7
	VSUBPD X6, X0, X8
	MOVUPD X7, (R8)(SI*1)
	MOVUPD X8, (R8)(DI*1)

	INCQ DX
	JMP  size256_inv_128_r2_stage2_j

size256_inv_128_r2_stage2_next:
	ADDQ $4, CX
	JMP  size256_inv_128_r2_stage2_base

size256_inv_128_r2_stage3:
	MOVQ $32, BX
	XORQ CX, CX

size256_inv_128_r2_stage3_base:
	CMPQ CX, $256
	JGE  size256_inv_128_r2_stage4
	XORQ DX, DX

size256_inv_128_r2_stage3_j:
	CMPQ DX, $4
	JGE  size256_inv_128_r2_stage3_next

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
	JMP  size256_inv_128_r2_stage3_j

size256_inv_128_r2_stage3_next:
	ADDQ $8, CX
	JMP  size256_inv_128_r2_stage3_base

size256_inv_128_r2_stage4:
	MOVQ $16, BX
	XORQ CX, CX

size256_inv_128_r2_stage4_base:
	CMPQ CX, $256
	JGE  size256_inv_128_r2_stage5
	XORQ DX, DX

size256_inv_128_r2_stage4_j:
	CMPQ DX, $8
	JGE  size256_inv_128_r2_stage4_next

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
	JMP  size256_inv_128_r2_stage4_j

size256_inv_128_r2_stage4_next:
	ADDQ $16, CX
	JMP  size256_inv_128_r2_stage4_base

size256_inv_128_r2_stage5:
	MOVQ $8, BX
	XORQ CX, CX

size256_inv_128_r2_stage5_base:
	CMPQ CX, $256
	JGE  size256_inv_128_r2_stage6
	XORQ DX, DX

size256_inv_128_r2_stage5_j:
	CMPQ DX, $16
	JGE  size256_inv_128_r2_stage5_next

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
	JMP  size256_inv_128_r2_stage5_j

size256_inv_128_r2_stage5_next:
	ADDQ $32, CX
	JMP  size256_inv_128_r2_stage5_base

size256_inv_128_r2_stage6:
	MOVQ $4, BX
	XORQ CX, CX

size256_inv_128_r2_stage6_base:
	CMPQ CX, $256
	JGE  size256_inv_128_r2_stage7
	XORQ DX, DX

size256_inv_128_r2_stage6_j:
	CMPQ DX, $32
	JGE  size256_inv_128_r2_stage6_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $32, DI
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
	JMP  size256_inv_128_r2_stage6_j

size256_inv_128_r2_stage6_next:
	ADDQ $64, CX
	JMP  size256_inv_128_r2_stage6_base

size256_inv_128_r2_stage7:
	MOVQ $2, BX
	XORQ CX, CX

size256_inv_128_r2_stage7_base:
	CMPQ CX, $256
	JGE  size256_inv_128_r2_stage8
	XORQ DX, DX

size256_inv_128_r2_stage7_j:
	CMPQ DX, $64
	JGE  size256_inv_128_r2_stage7_next

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $64, DI
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
	JMP  size256_inv_128_r2_stage7_j

size256_inv_128_r2_stage7_next:
	ADDQ $128, CX
	JMP  size256_inv_128_r2_stage7_base

size256_inv_128_r2_stage8:
	MOVQ $1, BX
	XORQ CX, CX
	XORQ DX, DX

size256_inv_128_r2_stage8_j:
	CMPQ DX, $128
	JGE  size256_inv_128_r2_scale

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVQ CX, DI
	ADDQ DX, DI
	ADDQ $128, DI
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
	JMP  size256_inv_128_r2_stage8_j

size256_inv_128_r2_scale:
	// Apply 1/n scaling (1/256 = 0.00390625)
	MOVQ $0x3f70000000000000, AX  // 1/256 in float64
	VMOVQ AX, X9
	VMOVDDUP X9, X9

	XORQ CX, CX
size256_inv_128_r2_scale_loop:
	CMPQ CX, $256
	JGE  size256_inv_128_r2_finalize
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0
	VMULPD X9, X0, X0
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size256_inv_128_r2_scale_loop

size256_inv_128_r2_finalize:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size256_inv_128_r2_done

	XORQ CX, CX
size256_inv_128_r2_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $4096
	JL   size256_inv_128_r2_copy_loop

size256_inv_128_r2_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size256_inv_128_r2_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET
