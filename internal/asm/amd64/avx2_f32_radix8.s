//go:build amd64 && !purego

// ===========================================================================
// AVX2 radix-8 DIT FFT, 256-bit wide, size-generic (n = 8^k, 2*8^k or 4*8^k)
// ===========================================================================
//
// The radix-8 sibling of avx2_f32_radix4.s, and deliberately its shape: same
// fused permutation/stage-1 gather, same packed twiddle planes, same
// permute+xor rotations, same direction-blind loop body.
//
// Why radix-8. A radix-8 ladder makes log2(n)/3 passes over the buffer where
// radix-4 makes log2(n)/2. Counting the uops of the two inner loops (per point
// per stage: 0.5 load/store, 1.09 twiddle, 1.25 butterfly here against 0.5,
// 0.94, 0.69 there) the radix-8 stage is 1.33x the work of a radix-4 stage for
// the same points -- but there are only 0.67x as many stages, so the ladder
// retires about 11% fewer uops end to end and touches the buffer a third less
// often. The pure-Go prototype in internal/kernels/radix8_generic.go measured
// that as a 0.87 forward geomean over 512..32768, which is what justified
// writing this.
//
//   Permute: work[8g+d] = src[idx[g] + d*(n/8)], d = 0..7
//   Stage 1: n/8 groups x 1 butterfly, no twiddles. Four groups are gathered
//            into eight YMM registers and transposed into group-major order.
//   Stage s: n/(8m) groups x m butterflies, m = 8^(s-1), while 8m <= limit.
//            The m butterflies of a group are contiguous, so a0..a7 are plain
//            256-bit loads at stride m and the seven twiddles come from seven
//            contiguous planes (see prepareTwiddleRadix8Complex64).
//   Tail:    one radix-2 stage when n = 2*8^k, one radix-4 stage when
//            n = 4*8^k, nothing when n = 8^k.
//
// Rotations. rot(v) = xor(swap(v), mask): negOdd gives -i*v, negEven gives
// +i*v. The eighth-root multiplies reduce to the same two ops plus an add and
// a scale, because
//
//	W_8^1 * p = c*(1-i)*p = c*(p + (-i)p)     (forward)
//	W_8^3 * q = c*(-1-i)*q = c*((-i)q - q)    (forward)
//
// and conjugating both -- which is all the inverse does -- turns every -i into
// +i. So both use the *same* mask the butterfly's first rotation uses, and the
// only difference between the directions is which mask that is. One loop body
// serves both; the caller passes `inverse` to pick the masks and `scale` (1/n,
// exact for a power of two) which stage 1 folds in for free.
//
// Register budget. Eight live streams plus the two rotation masks and the
// sqrt(2)/2 broadcast leave five scratch YMM, which is exactly enough for the
// butterfly and for a twiddle multiply's broadcast pair. That is why the
// twiddle planes are re-broadcast from memory per iteration rather than held:
// VMOVSLDUP/VMOVSHDUP with a 256-bit memory source are pure load uops, and the
// loop is short of registers, not of load slots.
//
// All instructions are VEX-encoded; no legacy-SSE forms appear.
// ===========================================================================

#include "textflag.h"

// r8NegOdd flips the sign of every odd float32 lane (the imaginary slot), so
// xor(swap(v), r8NegOdd) = -i*v.
GLOBL ·r8NegOdd<>(SB), RODATA|NOPTR, $32
DATA ·r8NegOdd<>+0(SB)/8,  $0x8000000000000000
DATA ·r8NegOdd<>+8(SB)/8,  $0x8000000000000000
DATA ·r8NegOdd<>+16(SB)/8, $0x8000000000000000
DATA ·r8NegOdd<>+24(SB)/8, $0x8000000000000000

// r8NegEven flips the sign of every even float32 lane (the real slot), so
// xor(swap(v), r8NegEven) = +i*v.
GLOBL ·r8NegEven<>(SB), RODATA|NOPTR, $32
DATA ·r8NegEven<>+0(SB)/8,  $0x0000000080000000
DATA ·r8NegEven<>+8(SB)/8,  $0x0000000080000000
DATA ·r8NegEven<>+16(SB)/8, $0x0000000080000000
DATA ·r8NegEven<>+24(SB)/8, $0x0000000080000000

// r8Root2f32 is sqrt(2)/2 as a float32, broadcast from memory rather than
// materialised through a GP register: the MOVL/VMOVQ/VBROADCASTSS route costs
// a fixed ~100ns per call, which is invisible at n = 32768 and mis-ranks the
// kernel at n = 32.
GLOBL ·r8Root2f32<>(SB), RODATA|NOPTR, $4
DATA ·r8Root2f32<>+0(SB)/4, $0x3F3504F3

// func Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, idx []int32, limit int, inverse bool, scale float32) bool
TEXT ·Radix8Complex64Asm(SB), NOSPLIT, $0-137
	MOVQ dst+0(FP), R8       // R8  = working buffer (dst, or scratch when in-place)
	MOVQ src+24(FP), R9      // R9  = src
	MOVQ scratch+72(FP), R11 // R11 = scratch
	MOVQ idx+96(FP), R12     // R12 = stage-1 group index table
	MOVQ src_len+32(FP), R13 // R13 = n

	// n >= 32 and a power of two is guaranteed by the Go caller; here we only
	// check that every slice is long enough to be safe to write. Stage 1
	// retires four groups per iteration, so n/8 must be a multiple of four --
	// which every power of two from 32 up is.
	CMPQ R13, $32
	JL   r8_false

	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   r8_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   r8_false

	// The packed table is n+8 elements (see twiddleSizeRadix8); requiring more
	// than n is what makes a caller that passes the plain length-n twiddle
	// table decline here rather than transform against it.
	MOVQ twiddle_len+56(FP), AX
	LEAQ 8(R13), BX
	CMPQ AX, BX
	JL   r8_false

	MOVQ idx_len+104(FP), AX // needs n/8 entries
	MOVQ R13, BX
	SHRQ $3, BX
	CMPQ AX, BX
	JL   r8_false

	// Rotation masks. Forward: the first rotation is -i (negOdd) and the
	// second +i (negEven). The inverse conjugates the whole butterfly, so the
	// two swap and nothing else changes.
	MOVBLZX inverse+128(FP), AX
	TESTL   AX, AX
	JNZ     r8_inverse_masks

	VMOVUPS ·r8NegOdd<>(SB), Y13
	VMOVUPS ·r8NegEven<>(SB), Y14
	JMP     r8_masks_done

r8_inverse_masks:
	VMOVUPS ·r8NegEven<>(SB), Y13
	VMOVUPS ·r8NegOdd<>(SB), Y14

r8_masks_done:
	VBROADCASTSS ·r8Root2f32<>(SB), Y15 // sqrt(2)/2
	VBROADCASTSS scale+132(FP), Y12     // 1/n for inverse, 1.0 for forward

	// In-place transforms work out of scratch and copy back at the end.
	CMPQ R8, R9
	JNE  r8_permute
	MOVQ R11, R8

r8_permute:
	// =====================================================================
	// Permutation and stage 1, fused.
	//
	// The eight inputs of stage-1 group g are src[idx[g] + d*q] for d = 0..7,
	// q = n/8, so for four consecutive groups each x_d vector is one
	// VPGATHERDQ: the index vector idx[g..g+3] is shared and only the base
	// pointer differs. That removes the separate permutation pass and the
	// input transpose both; only the output transpose remains, to store the
	// four groups' 32 outputs contiguously.
	// =====================================================================
	MOVQ R13, R10
	SHRQ $3, R10 // R10 = q = n/8 = the group count
	MOVQ R10, AX
	SHLQ $3, AX  // AX = q*8 bytes (= n bytes), the distance between streams

	LEAQ (R9)(AX*1), SI   // src + 1*q
	LEAQ (SI)(AX*1), DI   // src + 2*q
	LEAQ (DI)(AX*1), BP   // src + 3*q
	LEAQ (BP)(AX*1), R11  // src + 4*q (the scratch pointer is dead now)
	LEAQ (R11)(AX*1), R14 // src + 5*q
	LEAQ (R14)(AX*1), R15 // src + 6*q
	LEAQ (R15)(AX*1), BX  // src + 7*q

	XORQ CX, CX // group index
	XORQ DX, DX // byte offset into work

r8_stage1_loop:
	VMOVDQU (R12)(CX*4), X8 // idx[g..g+3]

	VPCMPEQD   Y9, Y9, Y9 // the gather consumes its mask, so rebuild each time
	VPGATHERDQ Y9, (R9)(X8*8), Y0
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (SI)(X8*8), Y1
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (DI)(X8*8), Y2
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (BP)(X8*8), Y3
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (R11)(X8*8), Y4
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (R14)(X8*8), Y5
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (R15)(X8*8), Y6
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (BX)(X8*8), Y7

	// ---- the eight-point butterfly, x0..x7 in Y0..Y7 -------------------
	VADDPS Y4, Y0, Y8  // a0 = x0 + x4
	VSUBPS Y4, Y0, Y9  // a1 = x0 - x4
	VADDPS Y6, Y2, Y10 // a2 = x2 + x6
	VSUBPS Y6, Y2, Y11 // a3 = x2 - x6
	VADDPS Y5, Y1, Y0  // a4 = x1 + x5
	VSUBPS Y5, Y1, Y2  // a5 = x1 - x5
	VADDPS Y7, Y3, Y4  // a6 = x3 + x7
	VSUBPS Y7, Y3, Y6  // a7 = x3 - x7

	VADDPS    Y10, Y8, Y1 // e0 = a0 + a2
	VSUBPS    Y10, Y8, Y3 // e2 = a0 - a2
	VPERMILPS $0xB1, Y11, Y5
	VXORPS    Y13, Y5, Y7 // rot1(a3)
	VXORPS    Y14, Y5, Y5 // rot2(a3)
	VADDPS    Y9, Y7, Y7  // e1 = a1 + rot1(a3)
	VADDPS    Y9, Y5, Y5  // e3 = a1 + rot2(a3)

	VADDPS    Y4, Y0, Y8   // o0 = a4 + a6
	VSUBPS    Y4, Y0, Y10  // o2 = a4 - a6
	VPERMILPS $0xB1, Y6, Y9
	VXORPS    Y13, Y9, Y11 // rot1(a7)
	VXORPS    Y14, Y9, Y9  // rot2(a7)
	VADDPS    Y2, Y11, Y11 // o1 = a5 + rot1(a7)
	VADDPS    Y2, Y9, Y9   // o3 = a5 + rot2(a7)

	VPERMILPS $0xB1, Y11, Y0
	VXORPS    Y13, Y0, Y0
	VADDPS    Y11, Y0, Y0 // o1 + rot1(o1)
	VMULPS    Y15, Y0, Y0 // t1 = W_8^1 * o1

	VPERMILPS $0xB1, Y10, Y2
	VXORPS    Y13, Y2, Y2 // t2 = W_8^2 * o2 = rot1(o2)

	VPERMILPS $0xB1, Y9, Y4
	VXORPS    Y13, Y4, Y4
	VSUBPS    Y9, Y4, Y4 // rot1(o3) - o3
	VMULPS    Y15, Y4, Y4 // t3 = W_8^3 * o3

	VADDPS Y8, Y1, Y6  // y0 = e0 + o0
	VSUBPS Y8, Y1, Y1  // y4 = e0 - o0
	VADDPS Y0, Y7, Y9  // y1 = e1 + t1
	VSUBPS Y0, Y7, Y7  // y5 = e1 - t1
	VADDPS Y2, Y3, Y10 // y2 = e2 + t2
	VSUBPS Y2, Y3, Y3  // y6 = e2 - t2
	VADDPS Y4, Y5, Y11 // y3 = e3 + t3
	VSUBPS Y4, Y5, Y5  // y7 = e3 - t3

	// Fold the inverse 1/n here: exact for a power of two, and it saves a
	// separate streaming pass over the whole buffer.
	VMULPS Y12, Y6, Y6
	VMULPS Y12, Y9, Y9
	VMULPS Y12, Y10, Y10
	VMULPS Y12, Y11, Y11
	VMULPS Y12, Y1, Y1
	VMULPS Y12, Y7, Y7
	VMULPS Y12, Y3, Y3
	VMULPS Y12, Y5, Y5

	// Transpose back into group-major order. y_d holds digit d for the four
	// groups, and the store wants [g0: d0..d7][g1: d0..d7]... -- which is two
	// independent 4x4 transposes of 64-bit lanes, one for d = 0..3 and one for
	// d = 4..7, interleaved 32 bytes apart at the store.
	VUNPCKLPD  Y9, Y6, Y0
	VUNPCKHPD  Y9, Y6, Y2
	VUNPCKLPD  Y11, Y10, Y4
	VUNPCKHPD  Y11, Y10, Y8
	VPERM2F128 $0x20, Y4, Y0, Y6  // g0: d0..d3
	VPERM2F128 $0x20, Y8, Y2, Y9  // g1: d0..d3
	VPERM2F128 $0x31, Y4, Y0, Y10 // g2: d0..d3
	VPERM2F128 $0x31, Y8, Y2, Y11 // g3: d0..d3

	VUNPCKLPD  Y7, Y1, Y0
	VUNPCKHPD  Y7, Y1, Y2
	VUNPCKLPD  Y5, Y3, Y4
	VUNPCKHPD  Y5, Y3, Y8
	VPERM2F128 $0x20, Y4, Y0, Y1 // g0: d4..d7
	VPERM2F128 $0x20, Y8, Y2, Y7 // g1: d4..d7
	VPERM2F128 $0x31, Y4, Y0, Y3 // g2: d4..d7
	VPERM2F128 $0x31, Y8, Y2, Y5 // g3: d4..d7

	VMOVUPS Y6, (R8)(DX*1)
	VMOVUPS Y1, 32(R8)(DX*1)
	VMOVUPS Y9, 64(R8)(DX*1)
	VMOVUPS Y7, 96(R8)(DX*1)
	VMOVUPS Y10, 128(R8)(DX*1)
	VMOVUPS Y3, 160(R8)(DX*1)
	VMOVUPS Y11, 192(R8)(DX*1)
	VMOVUPS Y5, 224(R8)(DX*1)

	ADDQ $4, CX
	ADDQ $256, DX
	CMPQ CX, R10
	JL   r8_stage1_loop

	// =====================================================================
	// Remaining radix-8 stages, m = 8, 64, 512, ...
	// =====================================================================
	MOVQ twiddle+48(FP), R10 // R10 = &w1[0] of the current stage
	MOVQ R13, R11
	SHLQ $3, R11
	ADDQ R8, R11 // R11 = end of work

	// AX is the byte stride between adjacent streams, m*8 -- which for
	// complex64 is numerically the stage span 8*m, so `AX <= limit` is exactly
	// the "another radix-8 stage fits" test.
	MOVQ $64, AX // m = 8
	CMPQ AX, limit+120(FP)
	JG   r8_tail

r8_stage_setup:
	MOVQ AX, R12
	SHLQ $3, R12 // R12 = 8*m*8, the byte stride between groups
	MOVQ R8, R9  // R9 = group base (the src pointer is dead now)

r8_group_loop:
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
	SHRQ $3, DX // DX = m, butterflies remaining in this group

r8_inner_loop:
	// ---- load and twiddle: x_d = w_d * a_d, x0 untouched ---------------
	VMOVUPS (SI), Y0

	VMOVUPS        (SI)(AX*1), Y1
	VMOVSLDUP      (CX), Y8
	VMOVSHDUP      (CX), Y9
	VSHUFPS        $0xB1, Y1, Y1, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y1

	VMOVUPS        (R14), Y2
	VMOVSLDUP      (CX)(AX*1), Y8
	VMOVSHDUP      (CX)(AX*1), Y9
	VSHUFPS        $0xB1, Y2, Y2, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y2

	VMOVUPS        (R14)(AX*1), Y3
	VMOVSLDUP      (CX)(AX*2), Y8
	VMOVSHDUP      (CX)(AX*2), Y9
	VSHUFPS        $0xB1, Y3, Y3, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y3

	VMOVUPS        (DI), Y4
	VMOVSLDUP      (R15), Y8
	VMOVSHDUP      (R15), Y9
	VSHUFPS        $0xB1, Y4, Y4, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y4

	VMOVUPS        (DI)(AX*1), Y5
	VMOVSLDUP      (R15)(AX*1), Y8
	VMOVSHDUP      (R15)(AX*1), Y9
	VSHUFPS        $0xB1, Y5, Y5, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y5

	VMOVUPS        (BP), Y6
	VMOVSLDUP      (R15)(AX*2), Y8
	VMOVSHDUP      (R15)(AX*2), Y9
	VSHUFPS        $0xB1, Y6, Y6, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y6

	VMOVUPS        (BP)(AX*1), Y7
	VMOVSLDUP      (BX), Y8
	VMOVSHDUP      (BX), Y9
	VSHUFPS        $0xB1, Y7, Y7, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y7

	// ---- the eight-point butterfly, identical to stage 1's -------------
	VADDPS Y4, Y0, Y8  // a0 = x0 + x4
	VSUBPS Y4, Y0, Y9  // a1 = x0 - x4
	VADDPS Y6, Y2, Y10 // a2 = x2 + x6
	VSUBPS Y6, Y2, Y11 // a3 = x2 - x6
	VADDPS Y5, Y1, Y0  // a4 = x1 + x5
	VSUBPS Y5, Y1, Y2  // a5 = x1 - x5
	VADDPS Y7, Y3, Y4  // a6 = x3 + x7
	VSUBPS Y7, Y3, Y6  // a7 = x3 - x7

	VADDPS    Y10, Y8, Y1 // e0
	VSUBPS    Y10, Y8, Y3 // e2
	VPERMILPS $0xB1, Y11, Y5
	VXORPS    Y13, Y5, Y7
	VXORPS    Y14, Y5, Y5
	VADDPS    Y9, Y7, Y7 // e1
	VADDPS    Y9, Y5, Y5 // e3

	VADDPS    Y4, Y0, Y8  // o0
	VSUBPS    Y4, Y0, Y10 // o2
	VPERMILPS $0xB1, Y6, Y9
	VXORPS    Y13, Y9, Y11
	VXORPS    Y14, Y9, Y9
	VADDPS    Y2, Y11, Y11 // o1
	VADDPS    Y2, Y9, Y9   // o3

	VPERMILPS $0xB1, Y11, Y0
	VXORPS    Y13, Y0, Y0
	VADDPS    Y11, Y0, Y0
	VMULPS    Y15, Y0, Y0 // t1

	VPERMILPS $0xB1, Y10, Y2
	VXORPS    Y13, Y2, Y2 // t2

	VPERMILPS $0xB1, Y9, Y4
	VXORPS    Y13, Y4, Y4
	VSUBPS    Y9, Y4, Y4
	VMULPS    Y15, Y4, Y4 // t3

	VADDPS Y8, Y1, Y6  // y0
	VSUBPS Y8, Y1, Y1  // y4
	VADDPS Y0, Y7, Y9  // y1
	VSUBPS Y0, Y7, Y7  // y5
	VADDPS Y2, Y3, Y10 // y2
	VSUBPS Y2, Y3, Y3  // y6
	VADDPS Y4, Y5, Y11 // y3
	VSUBPS Y4, Y5, Y5  // y7

	VMOVUPS Y6, (SI)
	VMOVUPS Y9, (SI)(AX*1)
	VMOVUPS Y10, (R14)
	VMOVUPS Y11, (R14)(AX*1)
	VMOVUPS Y1, (DI)
	VMOVUPS Y7, (DI)(AX*1)
	VMOVUPS Y3, (BP)
	VMOVUPS Y5, (BP)(AX*1)

	ADDQ $32, SI
	ADDQ $32, R14
	ADDQ $32, DI
	ADDQ $32, BP
	ADDQ $32, CX
	ADDQ $32, R15
	ADDQ $32, BX
	SUBQ $4, DX
	JNZ  r8_inner_loop

	ADDQ R12, R9
	CMPQ R9, R11
	JL   r8_group_loop

	// Next stage: advance past this stage's seven m-element planes (7*m*8
	// bytes = 7*AX) and multiply m by eight.
	LEAQ (R10)(AX*8), R10
	SUBQ AX, R10
	SHLQ $3, AX
	CMPQ AX, limit+120(FP)
	JLE  r8_stage_setup

r8_tail:
	// =====================================================================
	// Tail stage. The radix-8 stages above transformed `tail` interleaved
	// sub-sequences independently; one wide stage of radix tail = n/limit
	// combines them. R10 already points at its planes.
	// =====================================================================
	MOVQ limit+120(FP), AX
	CMPQ AX, R13
	JGE  r8_copy_out // n = 8^k: nothing left to do

	SHLQ $1, AX
	CMPQ AX, R13
	JNE  r8_tail4 // limit*2 != n, so tail = 4

	// ---- radix-2 tail: y0[j] = a0 + w[j]*a1, y1[j] = a0 - w[j]*a1 ------
	MOVQ R13, DX
	SHRQ $1, DX         // DX = n/2
	MOVQ R8, SI         // SI = &a0
	LEAQ (R8)(DX*8), DI // DI = &a1
	SHLQ $3, DX         // DX = byte length of one half

	XORQ CX, CX

r8_tail2_loop:
	VMOVSLDUP (R10)(CX*1), Y8
	VMOVSHDUP (R10)(CX*1), Y9
	VMOVUPS   (SI)(CX*1), Y0
	VMOVUPS   (DI)(CX*1), Y1

	VSHUFPS        $0xB1, Y1, Y1, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y1

	VADDPS Y0, Y1, Y2
	VSUBPS Y1, Y0, Y3

	VMOVUPS Y2, (SI)(CX*1)
	VMOVUPS Y3, (DI)(CX*1)

	ADDQ $32, CX
	CMPQ CX, DX
	JL   r8_tail2_loop

	JMP r8_copy_out

r8_tail4:
	// ---- radix-4 tail, m = n/4 -----------------------------------------
	MOVQ R13, AX
	SHRQ $2, AX
	SHLQ $3, AX          // AX = (n/4)*8, the byte stride between quarters
	MOVQ R8, SI          // SI = &a0, a1 at (SI)(AX*1)
	LEAQ (R8)(AX*2), DI  // DI = &a2, a3 at (DI)(AX*1)
	MOVQ R10, CX         // CX = &w1, w2 at (CX)(AX*1), w3 at (CX)(AX*2)
	MOVQ AX, DX          // DX = bytes remaining in a quarter

r8_tail4_loop:
	VMOVUPS (SI), Y0

	VMOVUPS        (SI)(AX*1), Y1
	VMOVSLDUP      (CX), Y8
	VMOVSHDUP      (CX), Y9
	VSHUFPS        $0xB1, Y1, Y1, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y1

	VMOVUPS        (DI), Y2
	VMOVSLDUP      (CX)(AX*1), Y8
	VMOVSHDUP      (CX)(AX*1), Y9
	VSHUFPS        $0xB1, Y2, Y2, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y2

	VMOVUPS        (DI)(AX*1), Y3
	VMOVSLDUP      (CX)(AX*2), Y8
	VMOVSHDUP      (CX)(AX*2), Y9
	VSHUFPS        $0xB1, Y3, Y3, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y3

	VADDPS Y2, Y0, Y4 // t0 = a0 + a2
	VSUBPS Y2, Y0, Y5 // t1 = a0 - a2
	VADDPS Y3, Y1, Y6 // t2 = a1 + a3
	VSUBPS Y3, Y1, Y7 // t3 = a1 - a3

	VPERMILPS $0xB1, Y7, Y10
	VXORPS    Y13, Y10, Y11 // rot1(t3)
	VXORPS    Y14, Y10, Y10 // rot2(t3)

	VADDPS Y6, Y4, Y0
	VADDPS Y11, Y5, Y1
	VSUBPS Y6, Y4, Y2
	VADDPS Y10, Y5, Y3

	VMOVUPS Y0, (SI)
	VMOVUPS Y1, (SI)(AX*1)
	VMOVUPS Y2, (DI)
	VMOVUPS Y3, (DI)(AX*1)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, CX
	SUBQ $32, DX
	JNZ  r8_tail4_loop

	// =====================================================================
	// Copy out if the transform ran in scratch.
	// =====================================================================
r8_copy_out:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r8_true

	MOVQ R13, AX
	SHLQ $3, AX
	XORQ CX, CX

r8_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ    $64, CX
	CMPQ    CX, AX
	JL      r8_copy_loop

r8_true:
	VZEROUPPER
	MOVB $1, ret+136(FP)
	RET

r8_false:
	VZEROUPPER
	MOVB $0, ret+136(FP)
	RET
