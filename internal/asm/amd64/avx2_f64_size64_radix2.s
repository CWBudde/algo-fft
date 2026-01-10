//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-64 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// Size-specific entrypoints for n==64 that use XMM operations for
// correctness and a fixed-size DIT schedule.
//
// Radix-2: 6 stages (64 = 2^6)
// Radix-4: 3 stages (64 = 4^3)
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
	LEAQ ·bitrev64_r2(SB), R12
	MOVQ src+32(FP), R13     // n (should be 64)

	CMPQ R13, $64
	JNE  size64_128_r2_return_false

	// Validate all slice lengths >= 64
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   size64_128_r2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   size64_128_r2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   size64_128_r2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size64_128_r2_use_dst
	MOVQ R11, R8             // In-place: use scratch as work

size64_128_r2_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// -----------------------------------------------------------------------
	XORQ CX, CX

size64_128_r2_bitrev_loop:
	CMPQ CX, $64
	JGE  size64_128_r2_stage1

	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ DX, SI
	SHLQ $4, SI              // SI = DX * 16 (bytes)
	MOVUPD (R9)(SI*1), X0
	MOVQ CX, SI
	SHLQ $4, SI              // SI = i * 16
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size64_128_r2_bitrev_loop

size64_128_r2_stage1:
	// -----------------------------------------------------------------------
	// Stage 1: size=2, half=1, step=32, twiddle[0]=1 => t=b
	// -----------------------------------------------------------------------
	XORQ CX, CX              // base

size64_128_r2_stage1_base:
	CMPQ CX, $64
	JGE  size64_128_r2_stage2

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
	JMP  size64_128_r2_stage1_base

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
	LEAQ ·bitrev64_r2(SB), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $64
	JNE  size64_inv_128_r2_return_false

	// Validate all slice lengths >= 64
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   size64_inv_128_r2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   size64_inv_128_r2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   size64_inv_128_r2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size64_inv_128_r2_use_dst
	MOVQ R11, R8

size64_inv_128_r2_use_dst:
	// Bit-reversal permutation
	XORQ CX, CX

size64_inv_128_r2_bitrev_loop:
	CMPQ CX, $64
	JGE  size64_inv_128_r2_stage1
	MOVQ (R12)(CX*8), DX
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD (R9)(SI*1), X0
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  size64_inv_128_r2_bitrev_loop

size64_inv_128_r2_stage1:
	// Stage 1: size=2, half=1, step=32 (twiddle[0]=1)
	XORQ CX, CX

size64_inv_128_r2_stage1_base:
	CMPQ CX, $64
	JGE  size64_inv_128_r2_stage2
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
	JMP  size64_inv_128_r2_stage1_base

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
	MOVUPD (R8)(SI*1), X0
	VMULPD X9, X0, X0
	MOVUPD X0, (R8)(SI*1)
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

// ===========================================================================
// Forward transform, size 64, complex128, radix-4
// ===========================================================================
//
// Size 64 = 4^3, so the radix-4 algorithm uses 3 stages:
//   Stage 1: 16 butterflies, stride=4
//   Stage 2: 4 groups x 4 butterflies, stride=16
//   Stage 3: 1 group x 16 butterflies
//
// ===========================================================================
TEXT ·ForwardAVX2Size64Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 64)

	// Verify n == 64
	CMPQ R13, $64
	JNE  r4_64_128_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   r4_64_128_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   r4_64_128_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   r4_64_128_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $64
	JL   r4_64_128_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  r4_64_128_use_dst
	MOVQ R11, R8             // In-place: use scratch

r4_64_128_use_dst:
	// ==================================================================
	// Bit-reversal permutation (base-4 bit-reversal)
	// ==================================================================
	XORQ CX, CX

r4_64_128_bitrev_loop:
	CMPQ CX, $64
	JGE  r4_64_128_stage1

	MOVQ (R12)(CX*8), DX
	MOVQ DX, SI
	SHLQ $4, SI              // DX * 16
	MOVUPD (R9)(SI*1), X0
	MOVQ CX, SI
	SHLQ $4, SI              // CX * 16
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  r4_64_128_bitrev_loop

r4_64_128_stage1:
	// ==================================================================
	// Stage 1: 16 radix-4 butterflies, stride=4
	// No twiddle factors needed (all w=1)
	// ==================================================================
	XORQ CX, CX              // base index

r4_64_128_stage1_loop:
	CMPQ CX, $64
	JGE  r4_64_128_stage2

	// Load 4 elements at indices CX, CX+1, CX+2, CX+3
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0    // x0

	MOVQ CX, SI
	INCQ SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X1    // x1

	MOVQ CX, SI
	ADDQ $2, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X2    // x2

	MOVQ CX, SI
	ADDQ $3, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X3    // x3

	// Radix-4 butterfly:
	// t0 = x0 + x2, t1 = x0 - x2
	// t2 = x1 + x3, t3 = x1 - x3
	// y0 = t0 + t2, y1 = t1 - i*t3, y2 = t0 - t2, y3 = t1 + i*t3
	VADDPD X2, X0, X4        // t0 = x0 + x2
	VSUBPD X2, X0, X5        // t1 = x0 - x2
	VADDPD X3, X1, X6        // t2 = x1 + x3
	VSUBPD X3, X1, X7        // t3 = x1 - x3

	// Compute -i*t3: swap re/im, negate new imaginary
	// For forward: -i*z = (z.im, -z.re)
	VPERMILPD $1, X7, X8     // [t3.im, t3.re]
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X10
	VXORPD X11, X11, X11
	VUNPCKLPD X10, X11, X10  // [0, signbit] for -i multiplication
	VXORPD X10, X8, X8       // [t3.im, -t3.re] = -i*t3

	// Compute i*t3: swap re/im, negate new real
	// For forward: i*z = (-z.im, z.re)
	VPERMILPD $1, X7, X9     // [t3.im, t3.re]
	VXORPD X12, X12, X12
	VMOVQ AX, X10
	VUNPCKLPD X11, X10, X12  // [signbit, 0] for i multiplication
	VXORPD X12, X9, X9       // [-t3.im, t3.re] = i*t3

	VADDPD X6, X4, X0        // y0 = t0 + t2
	VADDPD X8, X5, X1        // y1 = t1 + (-i*t3) = t1 - i*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VADDPD X9, X5, X3        // y3 = t1 + i*t3

	// Store results
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)

	MOVQ CX, SI
	INCQ SI
	SHLQ $4, SI
	MOVUPD X1, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ $2, SI
	SHLQ $4, SI
	MOVUPD X2, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ $3, SI
	SHLQ $4, SI
	MOVUPD X3, (R8)(SI*1)

	ADDQ $4, CX
	JMP  r4_64_128_stage1_loop

r4_64_128_stage2:
	// ==================================================================
	// Stage 2: 4 groups x 4 butterflies each, twiddle step=4
	// ==================================================================
	XORQ CX, CX              // group index (0, 16, 32, 48)

r4_64_128_stage2_outer:
	CMPQ CX, $64
	JGE  r4_64_128_stage3

	XORQ DX, DX              // j within group (0, 1, 2, 3)

r4_64_128_stage2_inner:
	CMPQ DX, $4
	JGE  r4_64_128_stage2_next

	// Indices: base=CX, j=DX
	// x0 at CX+DX, x1 at CX+DX+4, x2 at CX+DX+8, x3 at CX+DX+12
	MOVQ CX, BX
	ADDQ DX, BX              // idx0 = CX + DX

	MOVQ BX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0    // x0

	MOVQ BX, SI
	ADDQ $4, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X1    // x1

	MOVQ BX, SI
	ADDQ $8, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X2    // x2

	MOVQ BX, SI
	ADDQ $12, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X3    // x3

	// Load twiddle factors: w1=twiddle[j*4], w2=twiddle[2*j*4], w3=twiddle[3*j*4]
	MOVQ DX, R15
	SHLQ $2, R15             // j*4
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X13  // w1

	MOVQ R15, R14
	SHLQ $1, R15             // 2*j*4
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X14  // w2

	ADDQ R14, R15            // 3*j*4
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X15  // w3

	// Complex multiply x1 * w1
	VMOVDDUP X13, X8
	VPERMILPD $1, X13, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X1, X10
	VMULPD X9, X10, X10
	VFMADDSUB231PD X8, X1, X10
	VMOVAPD X10, X1

	// Complex multiply x2 * w2
	VMOVDDUP X14, X8
	VPERMILPD $1, X14, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X2, X10
	VMULPD X9, X10, X10
	VFMADDSUB231PD X8, X2, X10
	VMOVAPD X10, X2

	// Complex multiply x3 * w3
	VMOVDDUP X15, X8
	VPERMILPD $1, X15, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X3, X10
	VMULPD X9, X10, X10
	VFMADDSUB231PD X8, X3, X10
	VMOVAPD X10, X3

	// Radix-4 butterfly
	VADDPD X2, X0, X4        // t0 = x0 + x2
	VSUBPD X2, X0, X5        // t1 = x0 - x2
	VADDPD X3, X1, X6        // t2 = x1 + x3
	VSUBPD X3, X1, X7        // t3 = x1 - x3

	// -i*t3
	VPERMILPD $1, X7, X8
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X10
	VXORPD X11, X11, X11
	VUNPCKLPD X10, X11, X10
	VXORPD X10, X8, X8

	// i*t3
	VPERMILPD $1, X7, X9
	VXORPD X12, X12, X12
	VMOVQ AX, X10
	VUNPCKLPD X11, X10, X12
	VXORPD X12, X9, X9

	VADDPD X6, X4, X0        // y0
	VADDPD X8, X5, X1        // y1
	VSUBPD X6, X4, X2        // y2
	VADDPD X9, X5, X3        // y3

	// Store
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ DX, SI
	ADDQ $4, SI
	SHLQ $4, SI
	MOVUPD X1, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ DX, SI
	ADDQ $8, SI
	SHLQ $4, SI
	MOVUPD X2, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ DX, SI
	ADDQ $12, SI
	SHLQ $4, SI
	MOVUPD X3, (R8)(SI*1)

	INCQ DX
	JMP  r4_64_128_stage2_inner

r4_64_128_stage2_next:
	ADDQ $16, CX
	JMP  r4_64_128_stage2_outer

r4_64_128_stage3:
	// ==================================================================
	// Stage 3: 1 group x 16 butterflies, twiddle step=1
	// ==================================================================
	XORQ DX, DX              // j = 0..15

r4_64_128_stage3_loop:
	CMPQ DX, $16
	JGE  r4_64_128_done

	// x0 at DX, x1 at DX+16, x2 at DX+32, x3 at DX+48
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0

	MOVQ DX, SI
	ADDQ $16, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X1

	MOVQ DX, SI
	ADDQ $32, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X2

	MOVQ DX, SI
	ADDQ $48, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X3

	// Load twiddles: w1=twiddle[j], w2=twiddle[2*j], w3=twiddle[3*j]
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X13

	MOVQ DX, R15
	SHLQ $1, R15
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X14

	ADDQ DX, R15             // 3*j
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X15

	// Complex multiply x1 * w1
	VMOVDDUP X13, X8
	VPERMILPD $1, X13, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X1, X10
	VMULPD X9, X10, X10
	VFMADDSUB231PD X8, X1, X10
	VMOVAPD X10, X1

	// Complex multiply x2 * w2
	VMOVDDUP X14, X8
	VPERMILPD $1, X14, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X2, X10
	VMULPD X9, X10, X10
	VFMADDSUB231PD X8, X2, X10
	VMOVAPD X10, X2

	// Complex multiply x3 * w3
	VMOVDDUP X15, X8
	VPERMILPD $1, X15, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X3, X10
	VMULPD X9, X10, X10
	VFMADDSUB231PD X8, X3, X10
	VMOVAPD X10, X3

	// Radix-4 butterfly
	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// -i*t3
	VPERMILPD $1, X7, X8
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X10
	VXORPD X11, X11, X11
	VUNPCKLPD X10, X11, X10
	VXORPD X10, X8, X8

	// i*t3
	VPERMILPD $1, X7, X9
	VXORPD X12, X12, X12
	VMOVQ AX, X10
	VUNPCKLPD X11, X10, X12
	VXORPD X12, X9, X9

	VADDPD X6, X4, X0
	VADDPD X8, X5, X1
	VSUBPD X6, X4, X2
	VADDPD X9, X5, X3

	// Store (order: idx0=j, idx2=j+32, idx1=j+16, idx3=j+48)
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)      // work[j] = y0

	MOVQ DX, SI
	ADDQ $32, SI
	SHLQ $4, SI
	MOVUPD X2, (R8)(SI*1)      // work[j+32] = y2

	MOVQ DX, SI
	ADDQ $16, SI
	SHLQ $4, SI
	MOVUPD X1, (R8)(SI*1)      // work[j+16] = y1

	MOVQ DX, SI
	ADDQ $48, SI
	SHLQ $4, SI
	MOVUPD X3, (R8)(SI*1)      // work[j+48] = y3

	INCQ DX
	JMP  r4_64_128_stage3_loop

r4_64_128_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_64_128_done_direct

	XORQ CX, CX

r4_64_128_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $1024
	JL   r4_64_128_copy_loop

r4_64_128_done_direct:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

r4_64_128_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 64, complex128, radix-4
// ===========================================================================
TEXT ·InverseAVX2Size64Radix4Complex128Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ bitrev+96(FP), R12
	MOVQ src+32(FP), R13

	CMPQ R13, $64
	JNE  r4_64_128_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   r4_64_128_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   r4_64_128_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   r4_64_128_inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $64
	JL   r4_64_128_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  r4_64_128_inv_use_dst
	MOVQ R11, R8

r4_64_128_inv_use_dst:
	// Bit-reversal permutation
	XORQ CX, CX

r4_64_128_inv_bitrev_loop:
	CMPQ CX, $64
	JGE  r4_64_128_inv_stage1

	MOVQ (R12)(CX*8), DX
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD (R9)(SI*1), X0
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  r4_64_128_inv_bitrev_loop

r4_64_128_inv_stage1:
	// Stage 1: 16 radix-4 butterflies, stride=4 (no twiddles)
	// For inverse, use +i instead of -i for the butterfly
	XORQ CX, CX

r4_64_128_inv_stage1_loop:
	CMPQ CX, $64
	JGE  r4_64_128_inv_stage2

	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0

	MOVQ CX, SI
	INCQ SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X1

	MOVQ CX, SI
	ADDQ $2, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X2

	MOVQ CX, SI
	ADDQ $3, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X3

	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// For inverse: i*t3 for y1, -i*t3 for y3 (opposite of forward)
	// i*t3: swap and negate real
	VPERMILPD $1, X7, X8
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X10
	VXORPD X11, X11, X11
	VUNPCKLPD X11, X10, X12  // [signbit, 0] for i multiplication
	VXORPD X12, X8, X8       // i*t3

	// -i*t3: swap and negate imaginary
	VPERMILPD $1, X7, X9
	VUNPCKLPD X10, X11, X10  // [0, signbit] for -i multiplication
	VXORPD X10, X9, X9       // -i*t3

	VADDPD X6, X4, X0        // y0 = t0 + t2
	VADDPD X8, X5, X1        // y1 = t1 + i*t3
	VSUBPD X6, X4, X2        // y2 = t0 - t2
	VADDPD X9, X5, X3        // y3 = t1 - i*t3

	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)

	MOVQ CX, SI
	INCQ SI
	SHLQ $4, SI
	MOVUPD X1, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ $2, SI
	SHLQ $4, SI
	MOVUPD X2, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ $3, SI
	SHLQ $4, SI
	MOVUPD X3, (R8)(SI*1)

	ADDQ $4, CX
	JMP  r4_64_128_inv_stage1_loop

r4_64_128_inv_stage2:
	// Stage 2: 4 groups x 4 butterflies, twiddle step=4 (conjugated)
	XORQ CX, CX

r4_64_128_inv_stage2_outer:
	CMPQ CX, $64
	JGE  r4_64_128_inv_stage3

	XORQ DX, DX

r4_64_128_inv_stage2_inner:
	CMPQ DX, $4
	JGE  r4_64_128_inv_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX

	MOVQ BX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0

	MOVQ BX, SI
	ADDQ $4, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X1

	MOVQ BX, SI
	ADDQ $8, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X2

	MOVQ BX, SI
	ADDQ $12, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X3

	// Load twiddles
	MOVQ DX, R15
	SHLQ $2, R15
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X13

	MOVQ R15, R14
	SHLQ $1, R15
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X14

	ADDQ R14, R15
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X15

	// Conjugate complex multiply x1 * conj(w1)
	VMOVDDUP X13, X8
	VPERMILPD $1, X13, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X1, X10
	VMULPD X9, X10, X10
	VFMSUBADD231PD X8, X1, X10
	VMOVAPD X10, X1

	// Conjugate complex multiply x2 * conj(w2)
	VMOVDDUP X14, X8
	VPERMILPD $1, X14, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X2, X10
	VMULPD X9, X10, X10
	VFMSUBADD231PD X8, X2, X10
	VMOVAPD X10, X2

	// Conjugate complex multiply x3 * conj(w3)
	VMOVDDUP X15, X8
	VPERMILPD $1, X15, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X3, X10
	VMULPD X9, X10, X10
	VFMSUBADD231PD X8, X3, X10
	VMOVAPD X10, X3

	// Radix-4 butterfly (inverse)
	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// i*t3
	VPERMILPD $1, X7, X8
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X10
	VXORPD X11, X11, X11
	VUNPCKLPD X11, X10, X12
	VXORPD X12, X8, X8

	// -i*t3
	VPERMILPD $1, X7, X9
	VUNPCKLPD X10, X11, X10
	VXORPD X10, X9, X9

	VADDPD X6, X4, X0
	VADDPD X8, X5, X1
	VSUBPD X6, X4, X2
	VADDPD X9, X5, X3

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ DX, SI
	ADDQ $4, SI
	SHLQ $4, SI
	MOVUPD X1, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ DX, SI
	ADDQ $8, SI
	SHLQ $4, SI
	MOVUPD X2, (R8)(SI*1)

	MOVQ CX, SI
	ADDQ DX, SI
	ADDQ $12, SI
	SHLQ $4, SI
	MOVUPD X3, (R8)(SI*1)

	INCQ DX
	JMP  r4_64_128_inv_stage2_inner

r4_64_128_inv_stage2_next:
	ADDQ $16, CX
	JMP  r4_64_128_inv_stage2_outer

r4_64_128_inv_stage3:
	// Stage 3: 1 group x 16 butterflies, twiddle step=1 (conjugated)
	XORQ DX, DX

r4_64_128_inv_stage3_loop:
	CMPQ DX, $16
	JGE  r4_64_128_inv_scale

	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0

	MOVQ DX, SI
	ADDQ $16, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X1

	MOVQ DX, SI
	ADDQ $32, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X2

	MOVQ DX, SI
	ADDQ $48, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X3

	// Load twiddles
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X13

	MOVQ DX, R15
	SHLQ $1, R15
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X14

	ADDQ DX, R15
	MOVQ R15, SI
	SHLQ $4, SI
	MOVUPD (R10)(SI*1), X15

	// Conjugate complex multiply x1 * conj(w1)
	VMOVDDUP X13, X8
	VPERMILPD $1, X13, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X1, X10
	VMULPD X9, X10, X10
	VFMSUBADD231PD X8, X1, X10
	VMOVAPD X10, X1

	// Conjugate complex multiply x2 * conj(w2)
	VMOVDDUP X14, X8
	VPERMILPD $1, X14, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X2, X10
	VMULPD X9, X10, X10
	VFMSUBADD231PD X8, X2, X10
	VMOVAPD X10, X2

	// Conjugate complex multiply x3 * conj(w3)
	VMOVDDUP X15, X8
	VPERMILPD $1, X15, X9
	VMOVDDUP X9, X9
	VPERMILPD $1, X3, X10
	VMULPD X9, X10, X10
	VFMSUBADD231PD X8, X3, X10
	VMOVAPD X10, X3

	// Radix-4 butterfly (inverse)
	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// i*t3
	VPERMILPD $1, X7, X8
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X10
	VXORPD X11, X11, X11
	VUNPCKLPD X11, X10, X12
	VXORPD X12, X8, X8

	// -i*t3
	VPERMILPD $1, X7, X9
	VUNPCKLPD X10, X11, X10
	VXORPD X10, X9, X9

	VADDPD X6, X4, X0
	VADDPD X8, X5, X1
	VSUBPD X6, X4, X2
	VADDPD X9, X5, X3

	// Store (order: idx0=j, idx2=j+32, idx1=j+16, idx3=j+48)
	MOVQ DX, SI
	SHLQ $4, SI
	MOVUPD X0, (R8)(SI*1)      // work[j] = y0

	MOVQ DX, SI
	ADDQ $32, SI
	SHLQ $4, SI
	MOVUPD X2, (R8)(SI*1)      // work[j+32] = y2

	MOVQ DX, SI
	ADDQ $16, SI
	SHLQ $4, SI
	MOVUPD X1, (R8)(SI*1)      // work[j+16] = y1

	MOVQ DX, SI
	ADDQ $48, SI
	SHLQ $4, SI
	MOVUPD X3, (R8)(SI*1)      // work[j+48] = y3

	INCQ DX
	JMP  r4_64_128_inv_stage3_loop

r4_64_128_inv_scale:
	// Apply 1/64 scaling
	MOVQ ·sixtyFourth64(SB), AX
	VMOVQ AX, X9
	VMOVDDUP X9, X9

	XORQ CX, CX
r4_64_128_inv_scale_loop:
	CMPQ CX, $64
	JGE  r4_64_128_inv_finalize
	MOVQ CX, SI
	SHLQ $4, SI
	MOVUPD (R8)(SI*1), X0
	VMULPD X9, X0, X0
	MOVUPD X0, (R8)(SI*1)
	INCQ CX
	JMP  r4_64_128_inv_scale_loop

r4_64_128_inv_finalize:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_64_128_inv_done

	XORQ CX, CX
r4_64_128_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $1024
	JL   r4_64_128_inv_copy_loop

r4_64_128_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

r4_64_128_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET
