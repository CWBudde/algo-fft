//go:build amd64 && !purego

// ===========================================================================
// AVX2 radix-8 DIT FFT, 256-bit wide, complex128
// ===========================================================================
//
// The complex128 twin of avx2_f32_radix8.s. Same algorithm, same twiddle-plane
// layout, same group index table -- the permutation is precision-independent,
// so both kernels share internal/kernels' radix8GroupIndices. Read the f32
// file's header for why the ladder is shaped the way it is and for the
// derivation of the eighth-root rotations.
//
// The differences are the ones the width forces:
//
//   - Stage 1 processes two groups per iteration instead of four. There is no
//     128-bit-element gather, so each group's eight inputs are XMM loads folded
//     together with VINSERTF128. Eight streams would need eight base pointers
//     and the register file has no room for them next to the loop state, so
//     the two element offsets are stepped by q*16 between digits instead --
//     fourteen ADDQ per iteration, which the load ports absorb.
//   - The output transpose is a 2x2 block of 128-bit lanes, so eight
//     VPERM2F128 replace the f32 kernel's two 4x4 VUNPCK/VPERM2F128 sequences.
//   - The stage bound is `AX <= 2*limit` rather than `AX <= limit`: AX is the
//     byte stride m*16, and the span it must be compared against is 8*m.
//
// All instructions are VEX-encoded; no legacy-SSE forms appear.
// ===========================================================================

#include "textflag.h"

// r8dNegOdd flips the sign of the high float64 of each 128-bit lane (the
// imaginary part), so xor(swap(v), r8dNegOdd) = -i*v.
GLOBL ·r8dNegOdd<>(SB), RODATA|NOPTR, $32
DATA ·r8dNegOdd<>+0(SB)/8,  $0x0000000000000000
DATA ·r8dNegOdd<>+8(SB)/8,  $0x8000000000000000
DATA ·r8dNegOdd<>+16(SB)/8, $0x0000000000000000
DATA ·r8dNegOdd<>+24(SB)/8, $0x8000000000000000

// r8dNegEven flips the sign of the low float64 of each 128-bit lane (the real
// part), so xor(swap(v), r8dNegEven) = +i*v.
GLOBL ·r8dNegEven<>(SB), RODATA|NOPTR, $32
DATA ·r8dNegEven<>+0(SB)/8,  $0x8000000000000000
DATA ·r8dNegEven<>+8(SB)/8,  $0x0000000000000000
DATA ·r8dNegEven<>+16(SB)/8, $0x8000000000000000
DATA ·r8dNegEven<>+24(SB)/8, $0x0000000000000000

// r8Root2f64 is sqrt(2)/2 as a float64, broadcast from memory for the reason
// the f32 twin gives.
GLOBL ·r8Root2f64<>(SB), RODATA|NOPTR, $8
DATA ·r8Root2f64<>+0(SB)/8, $0x3FE6A09E667F3BCD

// func Radix8Complex128Asm(dst, src, twiddle, scratch []complex128, idx []int32, limit int, inverse bool, scale float64) bool
TEXT ·Radix8Complex128Asm(SB), NOSPLIT, $0-145
	MOVQ dst+0(FP), R8       // R8  = working buffer (dst, or scratch when in-place)
	MOVQ src+24(FP), R9      // R9  = src
	MOVQ scratch+72(FP), R11 // R11 = scratch
	MOVQ idx+96(FP), R12     // R12 = stage-1 group index table
	MOVQ src_len+32(FP), R13 // R13 = n

	CMPQ R13, $32
	JL   r8d_false

	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   r8d_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   r8d_false

	// The packed table is n+8 elements (see twiddleSizeRadix8); requiring more
	// than n is what makes a caller that passes the plain length-n twiddle
	// table decline here rather than transform against it. The padding also
	// covers the 8-byte-offset VMOVDDUP the imaginary broadcasts use.
	MOVQ twiddle_len+56(FP), AX
	LEAQ 8(R13), BX
	CMPQ AX, BX
	JL   r8d_false

	MOVQ idx_len+104(FP), AX // needs n/8 entries
	MOVQ R13, BX
	SHRQ $3, BX
	CMPQ AX, BX
	JL   r8d_false

	// Rotation masks. Forward: the first rotation is -i (negOdd) and the
	// second +i (negEven); the inverse conjugates the butterfly, so they swap.
	MOVBLZX inverse+128(FP), AX
	TESTL   AX, AX
	JNZ     r8d_inverse_masks

	VMOVUPD ·r8dNegOdd<>(SB), Y13
	VMOVUPD ·r8dNegEven<>(SB), Y14
	JMP     r8d_masks_done

r8d_inverse_masks:
	VMOVUPD ·r8dNegEven<>(SB), Y13
	VMOVUPD ·r8dNegOdd<>(SB), Y14

r8d_masks_done:
	VBROADCASTSD ·r8Root2f64<>(SB), Y15 // sqrt(2)/2
	VBROADCASTSD scale+136(FP), Y12     // 1/n for inverse, 1.0 for forward

	// In-place transforms work out of scratch and copy back at the end.
	CMPQ R8, R9
	JNE  r8d_permute
	MOVQ R11, R8

r8d_permute:
	// =====================================================================
	// Permutation and stage 1, fused. The eight inputs of group g are
	// src[idx[g] + d*q] for d = 0..7, q = n/8; two groups at a time give each
	// x_d vector directly, so the separate permutation pass and the input
	// transpose both disappear.
	// =====================================================================
	MOVQ R13, R10
	SHRQ $3, R10 // R10 = q = n/8 = the group count
	MOVQ R10, SI
	SHLQ $4, SI  // SI = q*16 bytes, the distance between streams

	XORQ CX, CX // group index
	XORQ DX, DX // byte offset into work

r8d_stage1_loop:
	MOVLQZX (R12)(CX*4), AX  // idx[g]
	MOVLQZX 4(R12)(CX*4), BX // idx[g+1]
	SHLQ    $4, AX           // element index -> byte offset
	SHLQ    $4, BX

	VMOVUPD     (R9)(AX*1), X0
	VINSERTF128 $1, (R9)(BX*1), Y0, Y0 // x0 = [src[i0], src[i1]]
	ADDQ        SI, AX
	ADDQ        SI, BX
	VMOVUPD     (R9)(AX*1), X1
	VINSERTF128 $1, (R9)(BX*1), Y1, Y1 // x1
	ADDQ        SI, AX
	ADDQ        SI, BX
	VMOVUPD     (R9)(AX*1), X2
	VINSERTF128 $1, (R9)(BX*1), Y2, Y2 // x2
	ADDQ        SI, AX
	ADDQ        SI, BX
	VMOVUPD     (R9)(AX*1), X3
	VINSERTF128 $1, (R9)(BX*1), Y3, Y3 // x3
	ADDQ        SI, AX
	ADDQ        SI, BX
	VMOVUPD     (R9)(AX*1), X4
	VINSERTF128 $1, (R9)(BX*1), Y4, Y4 // x4
	ADDQ        SI, AX
	ADDQ        SI, BX
	VMOVUPD     (R9)(AX*1), X5
	VINSERTF128 $1, (R9)(BX*1), Y5, Y5 // x5
	ADDQ        SI, AX
	ADDQ        SI, BX
	VMOVUPD     (R9)(AX*1), X6
	VINSERTF128 $1, (R9)(BX*1), Y6, Y6 // x6
	ADDQ        SI, AX
	ADDQ        SI, BX
	VMOVUPD     (R9)(AX*1), X7
	VINSERTF128 $1, (R9)(BX*1), Y7, Y7 // x7

	// ---- the eight-point butterfly, x0..x7 in Y0..Y7 -------------------
	VADDPD Y4, Y0, Y8  // a0 = x0 + x4
	VSUBPD Y4, Y0, Y9  // a1 = x0 - x4
	VADDPD Y6, Y2, Y10 // a2 = x2 + x6
	VSUBPD Y6, Y2, Y11 // a3 = x2 - x6
	VADDPD Y5, Y1, Y0  // a4 = x1 + x5
	VSUBPD Y5, Y1, Y2  // a5 = x1 - x5
	VADDPD Y7, Y3, Y4  // a6 = x3 + x7
	VSUBPD Y7, Y3, Y6  // a7 = x3 - x7

	VADDPD    Y10, Y8, Y1 // e0 = a0 + a2
	VSUBPD    Y10, Y8, Y3 // e2 = a0 - a2
	VPERMILPD $0x5, Y11, Y5
	VXORPD    Y13, Y5, Y7 // rot1(a3)
	VXORPD    Y14, Y5, Y5 // rot2(a3)
	VADDPD    Y9, Y7, Y7  // e1
	VADDPD    Y9, Y5, Y5  // e3

	VADDPD    Y4, Y0, Y8  // o0 = a4 + a6
	VSUBPD    Y4, Y0, Y10 // o2 = a4 - a6
	VPERMILPD $0x5, Y6, Y9
	VXORPD    Y13, Y9, Y11 // rot1(a7)
	VXORPD    Y14, Y9, Y9  // rot2(a7)
	VADDPD    Y2, Y11, Y11 // o1
	VADDPD    Y2, Y9, Y9   // o3

	VPERMILPD $0x5, Y11, Y0
	VXORPD    Y13, Y0, Y0
	VADDPD    Y11, Y0, Y0 // o1 + rot1(o1)
	VMULPD    Y15, Y0, Y0 // t1 = W_8^1 * o1

	VPERMILPD $0x5, Y10, Y2
	VXORPD    Y13, Y2, Y2 // t2 = W_8^2 * o2

	VPERMILPD $0x5, Y9, Y4
	VXORPD    Y13, Y4, Y4
	VSUBPD    Y9, Y4, Y4  // rot1(o3) - o3
	VMULPD    Y15, Y4, Y4 // t3 = W_8^3 * o3

	VADDPD Y8, Y1, Y6  // y0
	VSUBPD Y8, Y1, Y1  // y4
	VADDPD Y0, Y7, Y9  // y1
	VSUBPD Y0, Y7, Y7  // y5
	VADDPD Y2, Y3, Y10 // y2
	VSUBPD Y2, Y3, Y3  // y6
	VADDPD Y4, Y5, Y11 // y3
	VSUBPD Y4, Y5, Y5  // y7

	// Fold the inverse 1/n here: exact for a power of two, and it saves a
	// separate streaming pass over the whole buffer.
	VMULPD Y12, Y6, Y6
	VMULPD Y12, Y9, Y9
	VMULPD Y12, Y10, Y10
	VMULPD Y12, Y11, Y11
	VMULPD Y12, Y1, Y1
	VMULPD Y12, Y7, Y7
	VMULPD Y12, Y3, Y3
	VMULPD Y12, Y5, Y5

	// Transpose the 2x2 block of 128-bit lanes back into group-major order:
	// y_d holds [A_d, B_d] for the two groups A and B, and the store needs
	// [A0..A7][B0..B7].
	VPERM2F128 $0x20, Y9, Y6, Y0   // [A0, A1]
	VPERM2F128 $0x20, Y11, Y10, Y2 // [A2, A3]
	VPERM2F128 $0x20, Y7, Y1, Y4   // [A4, A5]
	VPERM2F128 $0x20, Y5, Y3, Y8   // [A6, A7]
	VMOVUPD    Y0, (R8)(DX*1)
	VMOVUPD    Y2, 32(R8)(DX*1)
	VMOVUPD    Y4, 64(R8)(DX*1)
	VMOVUPD    Y8, 96(R8)(DX*1)

	VPERM2F128 $0x31, Y9, Y6, Y0   // [B0, B1]
	VPERM2F128 $0x31, Y11, Y10, Y2 // [B2, B3]
	VPERM2F128 $0x31, Y7, Y1, Y4   // [B4, B5]
	VPERM2F128 $0x31, Y5, Y3, Y8   // [B6, B7]
	VMOVUPD    Y0, 128(R8)(DX*1)
	VMOVUPD    Y2, 160(R8)(DX*1)
	VMOVUPD    Y4, 192(R8)(DX*1)
	VMOVUPD    Y8, 224(R8)(DX*1)

	ADDQ $2, CX
	ADDQ $256, DX
	CMPQ CX, R10
	JL   r8d_stage1_loop

	// =====================================================================
	// Remaining radix-8 stages, m = 8, 64, 512, ...
	// =====================================================================
	MOVQ twiddle+48(FP), R10 // R10 = &w1[0] of the current stage
	MOVQ R13, R11
	SHLQ $4, R11
	ADDQ R8, R11 // R11 = end of work

	MOVQ $128, AX            // AX = m*16 with m = 8
	MOVQ limit+120(FP), DX
	SHLQ $1, DX              // the span 8*m is AX/2, so compare against 2*limit
	CMPQ AX, DX
	JG   r8d_tail

r8d_stage_setup:
	MOVQ AX, R12
	SHLQ $3, R12 // R12 = 8*m*16, the byte stride between groups
	MOVQ R8, R9  // R9 = group base (the src pointer is dead now)

r8d_group_loop:
	MOVQ R9, SI          // SI = &a0, a1 at (SI)(AX*1)
	LEAQ (R9)(AX*2), R14 // R14 = &a2, a3 at (R14)(AX*1)
	LEAQ (R9)(AX*4), DI  // DI = &a4, a5 at (DI)(AX*1)
	LEAQ (DI)(AX*2), BP  // BP = &a6, a7 at (BP)(AX*1)

	MOVQ R10, CX          // CX = &w1, w2 at (CX)(AX*1), w3 at (CX)(AX*2)
	LEAQ (R10)(AX*2), R15
	ADDQ AX, R15          // R15 = &w4, w5 at (R15)(AX*1), w6 at (R15)(AX*2)
	LEAQ (R15)(AX*2), BX
	ADDQ AX, BX           // BX = &w7

	MOVQ AX, DX
	SHRQ $4, DX // DX = m, butterflies remaining in this group

r8d_inner_loop:
	// ---- load and twiddle: x_d = w_d * a_d, x0 untouched ---------------
	VMOVUPD (SI), Y0

	VMOVUPD        (SI)(AX*1), Y1
	VMOVDDUP       (CX), Y8
	VMOVDDUP       8(CX), Y9
	VPERMILPD      $0x5, Y1, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y1

	VMOVUPD        (R14), Y2
	VMOVDDUP       (CX)(AX*1), Y8
	VMOVDDUP       8(CX)(AX*1), Y9
	VPERMILPD      $0x5, Y2, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y2

	VMOVUPD        (R14)(AX*1), Y3
	VMOVDDUP       (CX)(AX*2), Y8
	VMOVDDUP       8(CX)(AX*2), Y9
	VPERMILPD      $0x5, Y3, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y3

	VMOVUPD        (DI), Y4
	VMOVDDUP       (R15), Y8
	VMOVDDUP       8(R15), Y9
	VPERMILPD      $0x5, Y4, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y4

	VMOVUPD        (DI)(AX*1), Y5
	VMOVDDUP       (R15)(AX*1), Y8
	VMOVDDUP       8(R15)(AX*1), Y9
	VPERMILPD      $0x5, Y5, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y5

	VMOVUPD        (BP), Y6
	VMOVDDUP       (R15)(AX*2), Y8
	VMOVDDUP       8(R15)(AX*2), Y9
	VPERMILPD      $0x5, Y6, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y6

	VMOVUPD        (BP)(AX*1), Y7
	VMOVDDUP       (BX), Y8
	VMOVDDUP       8(BX), Y9
	VPERMILPD      $0x5, Y7, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y7

	// ---- the eight-point butterfly, identical to stage 1's -------------
	VADDPD Y4, Y0, Y8  // a0
	VSUBPD Y4, Y0, Y9  // a1
	VADDPD Y6, Y2, Y10 // a2
	VSUBPD Y6, Y2, Y11 // a3
	VADDPD Y5, Y1, Y0  // a4
	VSUBPD Y5, Y1, Y2  // a5
	VADDPD Y7, Y3, Y4  // a6
	VSUBPD Y7, Y3, Y6  // a7

	VADDPD    Y10, Y8, Y1 // e0
	VSUBPD    Y10, Y8, Y3 // e2
	VPERMILPD $0x5, Y11, Y5
	VXORPD    Y13, Y5, Y7
	VXORPD    Y14, Y5, Y5
	VADDPD    Y9, Y7, Y7 // e1
	VADDPD    Y9, Y5, Y5 // e3

	VADDPD    Y4, Y0, Y8  // o0
	VSUBPD    Y4, Y0, Y10 // o2
	VPERMILPD $0x5, Y6, Y9
	VXORPD    Y13, Y9, Y11
	VXORPD    Y14, Y9, Y9
	VADDPD    Y2, Y11, Y11 // o1
	VADDPD    Y2, Y9, Y9   // o3

	VPERMILPD $0x5, Y11, Y0
	VXORPD    Y13, Y0, Y0
	VADDPD    Y11, Y0, Y0
	VMULPD    Y15, Y0, Y0 // t1

	VPERMILPD $0x5, Y10, Y2
	VXORPD    Y13, Y2, Y2 // t2

	VPERMILPD $0x5, Y9, Y4
	VXORPD    Y13, Y4, Y4
	VSUBPD    Y9, Y4, Y4
	VMULPD    Y15, Y4, Y4 // t3

	VADDPD Y8, Y1, Y6  // y0
	VSUBPD Y8, Y1, Y1  // y4
	VADDPD Y0, Y7, Y9  // y1
	VSUBPD Y0, Y7, Y7  // y5
	VADDPD Y2, Y3, Y10 // y2
	VSUBPD Y2, Y3, Y3  // y6
	VADDPD Y4, Y5, Y11 // y3
	VSUBPD Y4, Y5, Y5  // y7

	VMOVUPD Y6, (SI)
	VMOVUPD Y9, (SI)(AX*1)
	VMOVUPD Y10, (R14)
	VMOVUPD Y11, (R14)(AX*1)
	VMOVUPD Y1, (DI)
	VMOVUPD Y7, (DI)(AX*1)
	VMOVUPD Y3, (BP)
	VMOVUPD Y5, (BP)(AX*1)

	ADDQ $32, SI
	ADDQ $32, R14
	ADDQ $32, DI
	ADDQ $32, BP
	ADDQ $32, CX
	ADDQ $32, R15
	ADDQ $32, BX
	SUBQ $2, DX
	JNZ  r8d_inner_loop

	ADDQ R12, R9
	CMPQ R9, R11
	JL   r8d_group_loop

	// Next stage: advance past this stage's seven m-element planes (7*AX
	// bytes) and multiply m by eight.
	LEAQ (R10)(AX*8), R10
	SUBQ AX, R10
	SHLQ $3, AX
	MOVQ limit+120(FP), DX
	SHLQ $1, DX
	CMPQ AX, DX
	JLE  r8d_stage_setup

r8d_tail:
	// =====================================================================
	// Tail stage: radix-2 for n = 2*8^k, radix-4 for n = 4*8^k, none for 8^k.
	// R10 already points at its planes.
	// =====================================================================
	MOVQ limit+120(FP), AX
	CMPQ AX, R13
	JGE  r8d_copy_out

	SHLQ $1, AX
	CMPQ AX, R13
	JNE  r8d_tail4

	// ---- radix-2 tail ---------------------------------------------------
	MOVQ R13, DX
	SHRQ $1, DX          // DX = n/2
	MOVQ R8, SI          // SI = &a0
	LEAQ (R8)(DX*8), DI
	LEAQ (DI)(DX*8), DI  // DI = &a1 = work + (n/2)*16
	SHLQ $4, DX          // DX = byte length of one half

	XORQ CX, CX

r8d_tail2_loop:
	VMOVDDUP (R10)(CX*1), Y8
	VMOVDDUP 8(R10)(CX*1), Y9
	VMOVUPD  (SI)(CX*1), Y0
	VMOVUPD  (DI)(CX*1), Y1

	VPERMILPD      $0x5, Y1, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y1

	VADDPD Y0, Y1, Y2
	VSUBPD Y1, Y0, Y3

	VMOVUPD Y2, (SI)(CX*1)
	VMOVUPD Y3, (DI)(CX*1)

	ADDQ $32, CX
	CMPQ CX, DX
	JL   r8d_tail2_loop

	JMP r8d_copy_out

r8d_tail4:
	// ---- radix-4 tail, m = n/4 ------------------------------------------
	MOVQ R13, AX
	SHRQ $2, AX
	SHLQ $4, AX         // AX = (n/4)*16, the byte stride between quarters
	MOVQ R8, SI         // SI = &a0, a1 at (SI)(AX*1)
	LEAQ (R8)(AX*2), DI // DI = &a2, a3 at (DI)(AX*1)
	MOVQ R10, CX        // CX = &w1, w2 at (CX)(AX*1), w3 at (CX)(AX*2)
	MOVQ AX, DX         // DX = bytes remaining in a quarter

r8d_tail4_loop:
	VMOVUPD (SI), Y0

	VMOVUPD        (SI)(AX*1), Y1
	VMOVDDUP       (CX), Y8
	VMOVDDUP       8(CX), Y9
	VPERMILPD      $0x5, Y1, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y1

	VMOVUPD        (DI), Y2
	VMOVDDUP       (CX)(AX*1), Y8
	VMOVDDUP       8(CX)(AX*1), Y9
	VPERMILPD      $0x5, Y2, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y2

	VMOVUPD        (DI)(AX*1), Y3
	VMOVDDUP       (CX)(AX*2), Y8
	VMOVDDUP       8(CX)(AX*2), Y9
	VPERMILPD      $0x5, Y3, Y10
	VMULPD         Y9, Y10, Y10
	VFMADDSUB213PD Y10, Y8, Y3

	VADDPD Y2, Y0, Y4 // t0 = a0 + a2
	VSUBPD Y2, Y0, Y5 // t1 = a0 - a2
	VADDPD Y3, Y1, Y6 // t2 = a1 + a3
	VSUBPD Y3, Y1, Y7 // t3 = a1 - a3

	VPERMILPD $0x5, Y7, Y10
	VXORPD    Y13, Y10, Y11 // rot1(t3)
	VXORPD    Y14, Y10, Y10 // rot2(t3)

	VADDPD Y6, Y4, Y0
	VADDPD Y11, Y5, Y1
	VSUBPD Y6, Y4, Y2
	VADDPD Y10, Y5, Y3

	VMOVUPD Y0, (SI)
	VMOVUPD Y1, (SI)(AX*1)
	VMOVUPD Y2, (DI)
	VMOVUPD Y3, (DI)(AX*1)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, CX
	SUBQ $32, DX
	JNZ  r8d_tail4_loop

	// =====================================================================
	// Copy out if the transform ran in scratch.
	// =====================================================================
r8d_copy_out:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r8d_true

	MOVQ R13, AX
	SHLQ $4, AX
	XORQ CX, CX

r8d_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ    $64, CX
	CMPQ    CX, AX
	JL      r8d_copy_loop

r8d_true:
	VZEROUPPER
	MOVB $1, ret+144(FP)
	RET

r8d_false:
	VZEROUPPER
	MOVB $0, ret+144(FP)
	RET
