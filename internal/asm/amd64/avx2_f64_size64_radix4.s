//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-64 FFT Kernels for AMD64 (complex128)
// ===========================================================================
//
// Size-specific entrypoints for n==64 that use XMM operations for
// correctness and a fixed-size DIT schedule.
//
// Radix-4: 3 stages (64 = 4^3)
//
// ===========================================================================

#include "textflag.h"

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
TEXT ·ForwardAVX2Size64Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	LEAQ ·bitrev64_r4_c128(SB), R12 // R12 = internal bitrev table
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
	MOVB $1, ret+96(FP)
	RET

r4_64_128_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 64, complex128, radix-4
// ===========================================================================
TEXT ·InverseAVX2Size64Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	LEAQ ·bitrev64_r4_c128(SB), R12 // R12 = internal bitrev table
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
	MOVB $1, ret+96(FP)
	RET

r4_64_128_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// DATA SECTION: Bit-reversal indices for size 64 radix-4 (complex128)
// ===========================================================================

DATA ·bitrev64_r4_c128+0(SB)/8, $0
DATA ·bitrev64_r4_c128+8(SB)/8, $16
DATA ·bitrev64_r4_c128+16(SB)/8, $32
DATA ·bitrev64_r4_c128+24(SB)/8, $48
DATA ·bitrev64_r4_c128+32(SB)/8, $4
DATA ·bitrev64_r4_c128+40(SB)/8, $20
DATA ·bitrev64_r4_c128+48(SB)/8, $36
DATA ·bitrev64_r4_c128+56(SB)/8, $52
DATA ·bitrev64_r4_c128+64(SB)/8, $8
DATA ·bitrev64_r4_c128+72(SB)/8, $24
DATA ·bitrev64_r4_c128+80(SB)/8, $40
DATA ·bitrev64_r4_c128+88(SB)/8, $56
DATA ·bitrev64_r4_c128+96(SB)/8, $12
DATA ·bitrev64_r4_c128+104(SB)/8, $28
DATA ·bitrev64_r4_c128+112(SB)/8, $44
DATA ·bitrev64_r4_c128+120(SB)/8, $60
DATA ·bitrev64_r4_c128+128(SB)/8, $1
DATA ·bitrev64_r4_c128+136(SB)/8, $17
DATA ·bitrev64_r4_c128+144(SB)/8, $33
DATA ·bitrev64_r4_c128+152(SB)/8, $49
DATA ·bitrev64_r4_c128+160(SB)/8, $5
DATA ·bitrev64_r4_c128+168(SB)/8, $21
DATA ·bitrev64_r4_c128+176(SB)/8, $37
DATA ·bitrev64_r4_c128+184(SB)/8, $53
DATA ·bitrev64_r4_c128+192(SB)/8, $9
DATA ·bitrev64_r4_c128+200(SB)/8, $25
DATA ·bitrev64_r4_c128+208(SB)/8, $41
DATA ·bitrev64_r4_c128+216(SB)/8, $57
DATA ·bitrev64_r4_c128+224(SB)/8, $13
DATA ·bitrev64_r4_c128+232(SB)/8, $29
DATA ·bitrev64_r4_c128+240(SB)/8, $45
DATA ·bitrev64_r4_c128+248(SB)/8, $61
DATA ·bitrev64_r4_c128+256(SB)/8, $2
DATA ·bitrev64_r4_c128+264(SB)/8, $18
DATA ·bitrev64_r4_c128+272(SB)/8, $34
DATA ·bitrev64_r4_c128+280(SB)/8, $50
DATA ·bitrev64_r4_c128+288(SB)/8, $6
DATA ·bitrev64_r4_c128+296(SB)/8, $22
DATA ·bitrev64_r4_c128+304(SB)/8, $38
DATA ·bitrev64_r4_c128+312(SB)/8, $54
DATA ·bitrev64_r4_c128+320(SB)/8, $10
DATA ·bitrev64_r4_c128+328(SB)/8, $26
DATA ·bitrev64_r4_c128+336(SB)/8, $42
DATA ·bitrev64_r4_c128+344(SB)/8, $58
DATA ·bitrev64_r4_c128+352(SB)/8, $14
DATA ·bitrev64_r4_c128+360(SB)/8, $30
DATA ·bitrev64_r4_c128+368(SB)/8, $46
DATA ·bitrev64_r4_c128+376(SB)/8, $62
DATA ·bitrev64_r4_c128+384(SB)/8, $3
DATA ·bitrev64_r4_c128+392(SB)/8, $19
DATA ·bitrev64_r4_c128+400(SB)/8, $35
DATA ·bitrev64_r4_c128+408(SB)/8, $51
DATA ·bitrev64_r4_c128+416(SB)/8, $7
DATA ·bitrev64_r4_c128+424(SB)/8, $23
DATA ·bitrev64_r4_c128+432(SB)/8, $39
DATA ·bitrev64_r4_c128+440(SB)/8, $55
DATA ·bitrev64_r4_c128+448(SB)/8, $11
DATA ·bitrev64_r4_c128+456(SB)/8, $27
DATA ·bitrev64_r4_c128+464(SB)/8, $43
DATA ·bitrev64_r4_c128+472(SB)/8, $59
DATA ·bitrev64_r4_c128+480(SB)/8, $15
DATA ·bitrev64_r4_c128+488(SB)/8, $31
DATA ·bitrev64_r4_c128+496(SB)/8, $47
DATA ·bitrev64_r4_c128+504(SB)/8, $63
GLOBL ·bitrev64_r4_c128(SB), RODATA, $512
