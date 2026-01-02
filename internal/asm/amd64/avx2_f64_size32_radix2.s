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
TEXT ·ForwardAVX2Size32Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // dst pointer
	MOVQ src+24(FP), R9      // src pointer
	MOVQ twiddle+48(FP), R10 // twiddle pointer
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ bitrev+96(FP), R12  // bitrev pointer
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

	MOVQ bitrev+104(FP), AX
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
	XORQ CX, CX

size32_128_bitrev_loop:
	CMPQ CX, $32
	JGE  size32_128_stage1

	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ DX, SI
	SHLQ $4, SI              // SI = DX * 16 (bytes)
	MOVUPD (R9)(SI*1), X0
	MOVQ CX, SI
	SHLQ $4, SI              // SI = i * 16
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size32_128_bitrev_loop

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
	MOVB $1, ret+120(FP)
	RET

size32_128_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 32, complex128
// ===========================================================================
TEXT ·InverseAVX2Size32Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
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

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $32
	JL   size32_inv_128_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size32_inv_128_use_dst
	MOVQ R11, R8

size32_inv_128_use_dst:
	// Bit-reversal permutation
	XORQ CX, CX

size32_inv_128_bitrev_loop:
	CMPQ CX, $32
	JGE  size32_inv_128_stage1
	MOVQ (R12)(CX*8), DX
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD (R9)(SI*1), X0
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size32_inv_128_bitrev_loop

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
	MOVB $1, ret+120(FP)
	RET

size32_inv_128_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET
