//go:build amd64 && !purego

// ===========================================================================
// AVX-512 radix-8 DIT FFT, 512-bit wide, size-generic (n = 8^k, 2*8^k or 4*8^k)
// ===========================================================================
//
// The 512-bit sibling of avx2_f32_radix8.s: same ladder, same packed twiddle
// planes (prepareTwiddleRadix8Complex64), same stage-1 group index table
// (radix8GroupIndices), same direction-blind loop body. Read that file's header
// for the derivation of the eighth-root rotations and of the ladder shape; it
// is not repeated here. Only what the width changes is documented below.
//
//   Permute: work[8g+d] = src[idx[g] + d*(n/8)], d = 0..7
//   Stage 1: n/8 groups x 1 butterfly, no twiddles. EIGHT groups are gathered
//            into eight ZMM registers and transposed into group-major order.
//   Stage s: n/(8m) groups x m butterflies, m = 8^(s-1), while 8m <= limit.
//   Tail:    one radix-2 stage when n = 2*8^k, one radix-4 stage when
//            n = 4*8^k, nothing when n = 8^k.
//
// WHAT THE WIDTH CHANGES
// ----------------------
// A ZMM holds eight complex64 where a YMM holds four, so every count doubles:
// stage 1 retires eight groups per iteration (index vector idx[g..g+7] is one
// YMM of int32, each of the eight streams one VPGATHERDQ with a ZMM
// destination, stores at 0,64,...,448 bytes); the inner loop steps 64 bytes and
// SUBQ $8 off the butterfly counter; the tail loops step 64 bytes and the
// copy-out 128. Every trip count still divides exactly: n/8 is a power of two
// >= 8 at n >= 64, m is 8/64/512/..., the radix-2 tail exists only from n = 128
// (n/2 = 64 points) and the radix-4 tail only from n = 256 (n/4 = 64 points).
//
// The stage-1 output transpose becomes one 8x8 transpose of 64-bit elements
// instead of two interleaved 4x4s: eight VUNPCKLPD/VUNPCKHPD then two levels of
// VSHUFF64X2, 24 instructions. It was validated by simulating the exact
// instruction semantics on symbolic lanes before it was written down.
//
// The floor rises to n >= 64: stage 1 needs n/8 >= 8. Below that the per-size
// AVX-512 codelets own the range.
//
// REGISTER BUDGET -- the reason this kernel exists
// ------------------------------------------------
// The 256-bit radix-8 stage measured 1.24-1.56x a radix-4 stage per pass at
// n = 512..2048, all of it inside L1 and with a $0 frame, so neither memory nor
// spill explains it. What does is that eight live streams plus two rotation
// masks and the sqrt(2)/2 broadcast leave five scratch YMM of sixteen -- one
// butterfly's worth, and not one register more.
//
// Here the four constants sit in Z28..Z31 and twenty scratch remain, spent on:
//
//   * All eight data loads issued at the top of the inner body, into eight
//     dedicated registers, before any arithmetic. The AVX2 loop cannot do this;
//     it interleaves each load with the twiddle multiply that consumes it,
//     which is the only shape five scratch registers allow. Eight independent
//     load chains in flight at once is the single largest thing the wider file
//     buys inside a stage.
//   * Seven independent twiddle broadcast pairs (Z8..Z21) instead of one shared
//     pair reused seven times, and six rotating swap temporaries (Z22..Z27)
//     instead of one. The seven complex multiplies are then seven
//     architecturally independent three-instruction chains.
//   * A butterfly that writes into fresh registers throughout (a0..a7 into
//     Z8..Z15, e/o into Z0..Z7, y0..y7 into Z11..Z18) rather than recycling its
//     own inputs in place as the AVX2 body must.
//
// What was deliberately NOT done, and why: a rotating two-bank software
// pipeline that keeps the loads of iteration i+1 in flight across the back edge
// of the inner loop. Two reasons. First, m starts at 8 and the counter steps by
// 8, so the first radix-8 stage of every size runs the inner loop exactly once;
// a 2x-unrolled pipeline would need a peeled single-trip path beside it,
// doubling a file that cannot be executed on the development host. Second, the
// iterations are already fully independent (different j, disjoint addresses,
// no loop-carried value at all) and the body is roughly 60 uops, so eight of
// them fit inside an Ice Lake-SP reorder buffer: the overlap a hand pipeline
// would create is overlap the machine creates anyway. The register pressure the
// AVX2 file suffers is a scheduling constraint on the *emitted* code, which the
// three bullets above remove; it is not a spill, and it is not a dependency
// that survives renaming.
//
// AVX-512 SPELLINGS
// -----------------
// VXORPS/VXORPD on ZMM are AVX512DQ and the feature gate here is AVX512F, so
// the rotations use VPXORD (bit-identical on float payloads). Everything else
// used below -- VPERMILPS imm, VSHUFPS, VMOVSLDUP/VMOVSHDUP, VFMADDSUB213PS,
// VUNPCKLPD/VUNPCKHPD, VSHUFF64X2, VPGATHERDQ, VBROADCASTSS, KXNORW -- is
// AVX512F. VPGATHERDQ consumes its mask, so K1 is rebuilt before every gather.
//
// All instructions are VEX- or EVEX-encoded; no legacy-SSE form appears.
// ===========================================================================

#include "textflag.h"

// r8zNegOdd flips the sign of every odd float32 lane (the imaginary slot), so
// xor(swap(v), r8zNegOdd) = -i*v. File-scoped, so it does not collide with the
// 32-byte twin in avx2_f32_radix8.s.
GLOBL ·r8zNegOdd<>(SB), RODATA|NOPTR, $64
DATA ·r8zNegOdd<>+0(SB)/8, $0x8000000000000000
DATA ·r8zNegOdd<>+8(SB)/8, $0x8000000000000000
DATA ·r8zNegOdd<>+16(SB)/8, $0x8000000000000000
DATA ·r8zNegOdd<>+24(SB)/8, $0x8000000000000000
DATA ·r8zNegOdd<>+32(SB)/8, $0x8000000000000000
DATA ·r8zNegOdd<>+40(SB)/8, $0x8000000000000000
DATA ·r8zNegOdd<>+48(SB)/8, $0x8000000000000000
DATA ·r8zNegOdd<>+56(SB)/8, $0x8000000000000000

// r8zNegEven flips the sign of every even float32 lane (the real slot), so
// xor(swap(v), r8zNegEven) = +i*v.
GLOBL ·r8zNegEven<>(SB), RODATA|NOPTR, $64
DATA ·r8zNegEven<>+0(SB)/8, $0x0000000080000000
DATA ·r8zNegEven<>+8(SB)/8, $0x0000000080000000
DATA ·r8zNegEven<>+16(SB)/8, $0x0000000080000000
DATA ·r8zNegEven<>+24(SB)/8, $0x0000000080000000
DATA ·r8zNegEven<>+32(SB)/8, $0x0000000080000000
DATA ·r8zNegEven<>+40(SB)/8, $0x0000000080000000
DATA ·r8zNegEven<>+48(SB)/8, $0x0000000080000000
DATA ·r8zNegEven<>+56(SB)/8, $0x0000000080000000

// r8zRoot2f32 is sqrt(2)/2 as a float32, broadcast from memory rather than
// materialised through a GP register: the MOVL/VMOVQ/VBROADCASTSS route costs a
// fixed ~100ns per call, which mis-ranks the kernel at small n.
GLOBL ·r8zRoot2f32<>(SB), RODATA|NOPTR, $4
DATA ·r8zRoot2f32<>+0(SB)/4, $0x3F3504F3

// func Radix8AVX512Complex64Asm(dst, src, twiddle, scratch []complex64, idx []int32, limit int, inverse bool, scale float32) bool
TEXT ·Radix8AVX512Complex64Asm(SB), NOSPLIT, $0-137
	MOVQ dst+0(FP), R8       // R8  = working buffer (dst, or scratch when in-place)
	MOVQ src+24(FP), R9      // R9  = src
	MOVQ scratch+72(FP), R11 // R11 = scratch
	MOVQ idx+96(FP), R12     // R12 = stage-1 group index table
	MOVQ src_len+32(FP), R13 // R13 = n

	// n >= 64 and a power of two is guaranteed by the Go caller; here we only
	// check that every slice is long enough to be safe to write. Stage 1
	// retires eight groups per iteration, so n/8 must be a multiple of eight --
	// which every power of two from 64 up is.
	CMPQ R13, $64
	JL   r8z_false

	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   r8z_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   r8z_false

	// The packed table is n+8 elements (see twiddleSizeRadix8); requiring more
	// than n is what makes a caller that passes the plain length-n twiddle
	// table decline here rather than transform against it.
	MOVQ twiddle_len+56(FP), AX
	LEAQ 8(R13), BX
	CMPQ AX, BX
	JL   r8z_false

	MOVQ idx_len+104(FP), AX // needs n/8 entries
	MOVQ R13, BX
	SHRQ $3, BX
	CMPQ AX, BX
	JL   r8z_false

	// Rotation masks. Forward: the first rotation is -i (negOdd) and the
	// second +i (negEven). The inverse conjugates the whole butterfly, so the
	// two swap and nothing else changes.
	MOVBLZX inverse+128(FP), AX
	TESTL   AX, AX
	JNZ     r8z_inverse_masks

	VMOVUPS ·r8zNegOdd<>(SB), Z28
	VMOVUPS ·r8zNegEven<>(SB), Z29
	JMP     r8z_masks_done

r8z_inverse_masks:
	VMOVUPS ·r8zNegEven<>(SB), Z28
	VMOVUPS ·r8zNegOdd<>(SB), Z29

r8z_masks_done:
	VBROADCASTSS ·r8zRoot2f32<>(SB), Z30 // sqrt(2)/2
	VBROADCASTSS scale+132(FP), Z31      // 1/n for inverse, 1.0 for forward

	// In-place transforms work out of scratch and copy back at the end.
	CMPQ R8, R9
	JNE  r8z_permute
	MOVQ R11, R8

r8z_permute:
	// =====================================================================
	// Permutation and stage 1, fused.
	//
	// The eight inputs of stage-1 group g are src[idx[g] + d*q] for d = 0..7,
	// q = n/8, so for EIGHT consecutive groups each x_d vector is one
	// VPGATHERDQ: the index vector idx[g..g+7] (8 int32 = one YMM) is shared
	// and only the base pointer differs. That removes the separate permutation
	// pass and the input transpose both; only the output transpose remains, to
	// store the eight groups' 64 outputs contiguously.
	// =====================================================================
	MOVQ R13, R10
	SHRQ $3, R10  // R10 = q = n/8 = the group count
	MOVQ R10, AX
	SHLQ $3, AX   // AX = q*8 bytes (= n bytes), the distance between streams

	LEAQ (R9)(AX*1), SI   // src + 1*q
	LEAQ (SI)(AX*1), DI   // src + 2*q
	LEAQ (DI)(AX*1), BP   // src + 3*q
	LEAQ (BP)(AX*1), R11  // src + 4*q (the scratch pointer is dead now)
	LEAQ (R11)(AX*1), R14 // src + 5*q
	LEAQ (R14)(AX*1), R15 // src + 6*q
	LEAQ (R15)(AX*1), BX  // src + 7*q

	XORQ CX, CX // group index
	XORQ DX, DX // byte offset into work

r8z_stage1_loop:
	VMOVDQU (R12)(CX*4), Y8 // idx[g..g+7]

	KXNORW     K1, K1, K1          // the gather consumes its mask, so rebuild each time
	VPGATHERDQ (R9)(Y8*8), K1, Z0
	KXNORW     K1, K1, K1
	VPGATHERDQ (SI)(Y8*8), K1, Z1
	KXNORW     K1, K1, K1
	VPGATHERDQ (DI)(Y8*8), K1, Z2
	KXNORW     K1, K1, K1
	VPGATHERDQ (BP)(Y8*8), K1, Z3
	KXNORW     K1, K1, K1
	VPGATHERDQ (R11)(Y8*8), K1, Z4
	KXNORW     K1, K1, K1
	VPGATHERDQ (R14)(Y8*8), K1, Z5
	KXNORW     K1, K1, K1
	VPGATHERDQ (R15)(Y8*8), K1, Z6
	KXNORW     K1, K1, K1
	VPGATHERDQ (BX)(Y8*8), K1, Z7

	// ---- the eight-point butterfly, x0..x7 in Z0..Z7 -------------------
	VADDPS Z4, Z0, Z8  // a0 = x0 + x4
	VSUBPS Z4, Z0, Z9  // a1 = x0 - x4
	VADDPS Z6, Z2, Z10 // a2 = x2 + x6
	VSUBPS Z6, Z2, Z11 // a3 = x2 - x6
	VADDPS Z5, Z1, Z12 // a4 = x1 + x5
	VSUBPS Z5, Z1, Z13 // a5 = x1 - x5
	VADDPS Z7, Z3, Z14 // a6 = x3 + x7
	VSUBPS Z7, Z3, Z15 // a7 = x3 - x7

	VADDPS    Z10, Z8, Z0     // e0 = a0 + a2
	VSUBPS    Z10, Z8, Z1     // e2 = a0 - a2
	VPERMILPS $0xB1, Z11, Z16
	VPXORD    Z28, Z16, Z17   // rot1(a3)
	VPXORD    Z29, Z16, Z18   // rot2(a3)
	VADDPS    Z9, Z17, Z2     // e1 = a1 + rot1(a3)
	VADDPS    Z9, Z18, Z3     // e3 = a1 + rot2(a3)

	VADDPS    Z14, Z12, Z4    // o0 = a4 + a6
	VSUBPS    Z14, Z12, Z5    // o2 = a4 - a6
	VPERMILPS $0xB1, Z15, Z16
	VPXORD    Z28, Z16, Z17   // rot1(a7)
	VPXORD    Z29, Z16, Z18   // rot2(a7)
	VADDPS    Z13, Z17, Z6    // o1 = a5 + rot1(a7)
	VADDPS    Z13, Z18, Z7    // o3 = a5 + rot2(a7)

	VPERMILPS $0xB1, Z6, Z8
	VPXORD    Z28, Z8, Z8
	VADDPS    Z6, Z8, Z8    // o1 + rot1(o1)
	VMULPS    Z30, Z8, Z8   // t1 = W_8^1 * o1

	VPERMILPS $0xB1, Z5, Z9
	VPXORD    Z28, Z9, Z9   // t2 = W_8^2 * o2 = rot1(o2)

	VPERMILPS $0xB1, Z7, Z10
	VPXORD    Z28, Z10, Z10
	VSUBPS    Z7, Z10, Z10   // rot1(o3) - o3
	VMULPS    Z30, Z10, Z10  // t3 = W_8^3 * o3

	VADDPS Z4, Z0, Z11  // y0 = e0 + o0
	VADDPS Z8, Z2, Z12  // y1 = e1 + t1
	VADDPS Z9, Z1, Z13  // y2 = e2 + t2
	VADDPS Z10, Z3, Z14 // y3 = e3 + t3
	VSUBPS Z4, Z0, Z15  // y4 = e0 - o0
	VSUBPS Z8, Z2, Z16  // y5 = e1 - t1
	VSUBPS Z9, Z1, Z17  // y6 = e2 - t2
	VSUBPS Z10, Z3, Z18 // y7 = e3 - t3

	// Fold the inverse 1/n here: exact for a power of two, and it saves a
	// separate streaming pass over the whole buffer.
	VMULPS Z31, Z11, Z11
	VMULPS Z31, Z12, Z12
	VMULPS Z31, Z13, Z13
	VMULPS Z31, Z14, Z14
	VMULPS Z31, Z15, Z15
	VMULPS Z31, Z16, Z16
	VMULPS Z31, Z17, Z17
	VMULPS Z31, Z18, Z18

	// ---- 8x8 transpose of 64-bit elements ------------------------------
	// y_d holds digit d for the eight groups; the store wants
	// [g0: d0..d7][g1: d0..d7]... which is exactly the transpose. Level 1
	// interleaves adjacent digit rows within each 128-bit lane; levels 2 and 3
	// shuffle 128-bit lanes. Output register j holds group j's d0..d7 in order
	// -- asserted by simulating these 24 instructions on symbolic lanes before
	// they were written.
	VUNPCKLPD Z12, Z11, Z0 // L1
	VUNPCKHPD Z12, Z11, Z1
	VUNPCKLPD Z14, Z13, Z2
	VUNPCKHPD Z14, Z13, Z3
	VUNPCKLPD Z16, Z15, Z4
	VUNPCKHPD Z16, Z15, Z5
	VUNPCKLPD Z18, Z17, Z6
	VUNPCKHPD Z18, Z17, Z7

	VSHUFF64X2 $0x88, Z2, Z0, Z8  // L2
	VSHUFF64X2 $0xDD, Z2, Z0, Z9
	VSHUFF64X2 $0x88, Z3, Z1, Z10
	VSHUFF64X2 $0xDD, Z3, Z1, Z11
	VSHUFF64X2 $0x88, Z6, Z4, Z12
	VSHUFF64X2 $0xDD, Z6, Z4, Z13
	VSHUFF64X2 $0x88, Z7, Z5, Z14
	VSHUFF64X2 $0xDD, Z7, Z5, Z15

	VSHUFF64X2 $0x88, Z12, Z8, Z0  // L3: group 0
	VSHUFF64X2 $0xDD, Z12, Z8, Z4  // group 4
	VSHUFF64X2 $0x88, Z13, Z9, Z2  // group 2
	VSHUFF64X2 $0xDD, Z13, Z9, Z6  // group 6
	VSHUFF64X2 $0x88, Z14, Z10, Z1 // group 1
	VSHUFF64X2 $0xDD, Z14, Z10, Z5 // group 5
	VSHUFF64X2 $0x88, Z15, Z11, Z3 // group 3
	VSHUFF64X2 $0xDD, Z15, Z11, Z7 // group 7

	VMOVUPS Z0, (R8)(DX*1)
	VMOVUPS Z1, 64(R8)(DX*1)
	VMOVUPS Z2, 128(R8)(DX*1)
	VMOVUPS Z3, 192(R8)(DX*1)
	VMOVUPS Z4, 256(R8)(DX*1)
	VMOVUPS Z5, 320(R8)(DX*1)
	VMOVUPS Z6, 384(R8)(DX*1)
	VMOVUPS Z7, 448(R8)(DX*1)

	ADDQ $8, CX
	ADDQ $512, DX
	CMPQ CX, R10
	JL   r8z_stage1_loop

	// =====================================================================
	// Remaining radix-8 stages, m = 8, 64, 512, ...
	// =====================================================================
	MOVQ twiddle+48(FP), R10 // R10 = &w1[0] of the current stage
	MOVQ R13, R11
	SHLQ $3, R11
	ADDQ R8, R11             // R11 = end of work

	// AX is the byte stride between adjacent streams, m*8 -- which for
	// complex64 is numerically the stage span 8*m, so `AX <= limit` is exactly
	// the "another radix-8 stage fits" test.
	MOVQ $64, AX           // m = 8
	CMPQ AX, limit+120(FP)
	JG   r8z_tail

r8z_stage_setup:
	MOVQ AX, R12
	SHLQ $3, R12 // R12 = 8*m*8, the byte stride between groups
	MOVQ R8, R9  // R9 = group base (the src pointer is dead now)

r8z_group_loop:
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

r8z_inner_loop:
	// ---- all eight loads first, into eight dedicated registers ---------
	// Five scratch YMM force the AVX2 body to interleave each load with the
	// multiply that consumes it; twenty scratch ZMM do not, so the eight
	// independent load chains all issue before any arithmetic.
	VMOVUPS (SI), Z0        // a0 (untwiddled)
	VMOVUPS (SI)(AX*1), Z1  // a1
	VMOVUPS (R14), Z2       // a2
	VMOVUPS (R14)(AX*1), Z3 // a3
	VMOVUPS (DI), Z4        // a4
	VMOVUPS (DI)(AX*1), Z5  // a5
	VMOVUPS (BP), Z6        // a6
	VMOVUPS (BP)(AX*1), Z7  // a7

	// ---- seven independent twiddle broadcast pairs ---------------------
	VMOVSLDUP (CX), Z8         // Re(w1) duplicated
	VMOVSHDUP (CX), Z9         // Im(w1) duplicated
	VMOVSLDUP (CX)(AX*1), Z10  // w2
	VMOVSHDUP (CX)(AX*1), Z11
	VMOVSLDUP (CX)(AX*2), Z12  // w3
	VMOVSHDUP (CX)(AX*2), Z13
	VMOVSLDUP (R15), Z14       // w4
	VMOVSHDUP (R15), Z15
	VMOVSLDUP (R15)(AX*1), Z16 // w5
	VMOVSHDUP (R15)(AX*1), Z17
	VMOVSLDUP (R15)(AX*2), Z18 // w6
	VMOVSHDUP (R15)(AX*2), Z19
	VMOVSLDUP (BX), Z20        // w7
	VMOVSHDUP (BX), Z21

	// ---- x_d = w_d * a_d, x0 untouched; seven independent chains -------
	VSHUFPS        $0xB1, Z1, Z1, Z22
	VMULPS         Z9, Z22, Z22
	VFMADDSUB213PS Z22, Z8, Z1        // x1 = w1 * a1

	VSHUFPS        $0xB1, Z2, Z2, Z23
	VMULPS         Z11, Z23, Z23
	VFMADDSUB213PS Z23, Z10, Z2       // x2 = w2 * a2

	VSHUFPS        $0xB1, Z3, Z3, Z24
	VMULPS         Z13, Z24, Z24
	VFMADDSUB213PS Z24, Z12, Z3       // x3 = w3 * a3

	VSHUFPS        $0xB1, Z4, Z4, Z25
	VMULPS         Z15, Z25, Z25
	VFMADDSUB213PS Z25, Z14, Z4       // x4 = w4 * a4

	VSHUFPS        $0xB1, Z5, Z5, Z26
	VMULPS         Z17, Z26, Z26
	VFMADDSUB213PS Z26, Z16, Z5       // x5 = w5 * a5

	VSHUFPS        $0xB1, Z6, Z6, Z27
	VMULPS         Z19, Z27, Z27
	VFMADDSUB213PS Z27, Z18, Z6       // x6 = w6 * a6

	VSHUFPS        $0xB1, Z7, Z7, Z22
	VMULPS         Z21, Z22, Z22
	VFMADDSUB213PS Z22, Z20, Z7       // x7 = w7 * a7

	// ---- the eight-point butterfly, identical to stage 1's -------------
	VADDPS Z4, Z0, Z8  // a0 = x0 + x4
	VSUBPS Z4, Z0, Z9  // a1 = x0 - x4
	VADDPS Z6, Z2, Z10 // a2 = x2 + x6
	VSUBPS Z6, Z2, Z11 // a3 = x2 - x6
	VADDPS Z5, Z1, Z12 // a4 = x1 + x5
	VSUBPS Z5, Z1, Z13 // a5 = x1 - x5
	VADDPS Z7, Z3, Z14 // a6 = x3 + x7
	VSUBPS Z7, Z3, Z15 // a7 = x3 - x7

	VADDPS    Z10, Z8, Z0     // e0
	VSUBPS    Z10, Z8, Z1     // e2
	VPERMILPS $0xB1, Z11, Z16
	VPXORD    Z28, Z16, Z17
	VPXORD    Z29, Z16, Z18
	VADDPS    Z9, Z17, Z2     // e1
	VADDPS    Z9, Z18, Z3     // e3

	VADDPS    Z14, Z12, Z4    // o0
	VSUBPS    Z14, Z12, Z5    // o2
	VPERMILPS $0xB1, Z15, Z16
	VPXORD    Z28, Z16, Z17
	VPXORD    Z29, Z16, Z18
	VADDPS    Z13, Z17, Z6    // o1
	VADDPS    Z13, Z18, Z7    // o3

	VPERMILPS $0xB1, Z6, Z8
	VPXORD    Z28, Z8, Z8
	VADDPS    Z6, Z8, Z8
	VMULPS    Z30, Z8, Z8   // t1

	VPERMILPS $0xB1, Z5, Z9
	VPXORD    Z28, Z9, Z9   // t2

	VPERMILPS $0xB1, Z7, Z10
	VPXORD    Z28, Z10, Z10
	VSUBPS    Z7, Z10, Z10
	VMULPS    Z30, Z10, Z10  // t3

	VADDPS Z4, Z0, Z11  // y0
	VADDPS Z8, Z2, Z12  // y1
	VADDPS Z9, Z1, Z13  // y2
	VADDPS Z10, Z3, Z14 // y3
	VSUBPS Z4, Z0, Z15  // y4
	VSUBPS Z8, Z2, Z16  // y5
	VSUBPS Z9, Z1, Z17  // y6
	VSUBPS Z10, Z3, Z18 // y7

	VMOVUPS Z11, (SI)
	VMOVUPS Z12, (SI)(AX*1)
	VMOVUPS Z13, (R14)
	VMOVUPS Z14, (R14)(AX*1)
	VMOVUPS Z15, (DI)
	VMOVUPS Z16, (DI)(AX*1)
	VMOVUPS Z17, (BP)
	VMOVUPS Z18, (BP)(AX*1)

	ADDQ $64, SI
	ADDQ $64, R14
	ADDQ $64, DI
	ADDQ $64, BP
	ADDQ $64, CX
	ADDQ $64, R15
	ADDQ $64, BX
	SUBQ $8, DX
	JNZ  r8z_inner_loop

	ADDQ R12, R9
	CMPQ R9, R11
	JL   r8z_group_loop

	// Next stage: advance past this stage's seven m-element planes (7*m*8
	// bytes = 7*AX) and multiply m by eight.
	LEAQ (R10)(AX*8), R10
	SUBQ AX, R10
	SHLQ $3, AX
	CMPQ AX, limit+120(FP)
	JLE  r8z_stage_setup

r8z_tail:
	// =====================================================================
	// Tail stage. The radix-8 stages above transformed `tail` interleaved
	// sub-sequences independently; one wide stage of radix tail = n/limit
	// combines them. R10 already points at its planes.
	// =====================================================================
	MOVQ limit+120(FP), AX
	CMPQ AX, R13
	JGE  r8z_copy_out      // n = 8^k: nothing left to do

	SHLQ $1, AX
	CMPQ AX, R13
	JNE  r8z_tail4 // limit*2 != n, so tail = 4

	// ---- radix-2 tail: y0[j] = a0 + w[j]*a1, y1[j] = a0 - w[j]*a1 ------
	// Reached only from n = 128 up, so n/2 is a multiple of eight complex64
	// and the 64-byte step divides the half exactly.
	MOVQ R13, DX
	SHRQ $1, DX         // DX = n/2
	MOVQ R8, SI         // SI = &a0
	LEAQ (R8)(DX*8), DI // DI = &a1
	SHLQ $3, DX         // DX = byte length of one half

	XORQ CX, CX

r8z_tail2_loop:
	VMOVSLDUP (R10)(CX*1), Z8
	VMOVSHDUP (R10)(CX*1), Z9
	VMOVUPS   (SI)(CX*1), Z0
	VMOVUPS   (DI)(CX*1), Z1

	VSHUFPS        $0xB1, Z1, Z1, Z10
	VMULPS         Z9, Z10, Z10
	VFMADDSUB213PS Z10, Z8, Z1

	VADDPS Z0, Z1, Z2
	VSUBPS Z1, Z0, Z3

	VMOVUPS Z2, (SI)(CX*1)
	VMOVUPS Z3, (DI)(CX*1)

	ADDQ $64, CX
	CMPQ CX, DX
	JL   r8z_tail2_loop

	JMP r8z_copy_out

r8z_tail4:
	// ---- radix-4 tail, m = n/4 -----------------------------------------
	// Reached only from n = 256 up, so n/4 is a multiple of eight complex64.
	MOVQ R13, AX
	SHRQ $2, AX         // AX = n/4
	SHLQ $3, AX         // AX = (n/4)*8, the byte stride between quarters
	MOVQ R8, SI         // SI = &a0, a1 at (SI)(AX*1)
	LEAQ (R8)(AX*2), DI // DI = &a2, a3 at (DI)(AX*1)
	MOVQ R10, CX        // CX = &w1, w2 at (CX)(AX*1), w3 at (CX)(AX*2)
	MOVQ AX, DX         // DX = bytes remaining in a quarter

r8z_tail4_loop:
	VMOVUPS (SI), Z0
	VMOVUPS (SI)(AX*1), Z1
	VMOVUPS (DI), Z2
	VMOVUPS (DI)(AX*1), Z3

	VMOVSLDUP (CX), Z8
	VMOVSHDUP (CX), Z9
	VMOVSLDUP (CX)(AX*1), Z10
	VMOVSHDUP (CX)(AX*1), Z11
	VMOVSLDUP (CX)(AX*2), Z12
	VMOVSHDUP (CX)(AX*2), Z13

	VSHUFPS        $0xB1, Z1, Z1, Z14
	VMULPS         Z9, Z14, Z14
	VFMADDSUB213PS Z14, Z8, Z1        // x1 = w1 * a1

	VSHUFPS        $0xB1, Z2, Z2, Z15
	VMULPS         Z11, Z15, Z15
	VFMADDSUB213PS Z15, Z10, Z2       // x2 = w2 * a2

	VSHUFPS        $0xB1, Z3, Z3, Z16
	VMULPS         Z13, Z16, Z16
	VFMADDSUB213PS Z16, Z12, Z3       // x3 = w3 * a3

	VADDPS Z2, Z0, Z4 // t0 = x0 + x2
	VSUBPS Z2, Z0, Z5 // t1 = x0 - x2
	VADDPS Z3, Z1, Z6 // t2 = x1 + x3
	VSUBPS Z3, Z1, Z7 // t3 = x1 - x3

	VPERMILPS $0xB1, Z7, Z17
	VPXORD    Z28, Z17, Z18  // rot1(t3)
	VPXORD    Z29, Z17, Z17  // rot2(t3)

	VADDPS Z6, Z4, Z0
	VADDPS Z18, Z5, Z1
	VSUBPS Z6, Z4, Z2
	VADDPS Z17, Z5, Z3

	VMOVUPS Z0, (SI)
	VMOVUPS Z1, (SI)(AX*1)
	VMOVUPS Z2, (DI)
	VMOVUPS Z3, (DI)(AX*1)

	ADDQ $64, SI
	ADDQ $64, DI
	ADDQ $64, CX
	SUBQ $64, DX
	JNZ  r8z_tail4_loop

	// =====================================================================
	// Copy out if the transform ran in scratch.
	// =====================================================================
r8z_copy_out:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r8z_true

	MOVQ R13, AX
	SHLQ $3, AX
	XORQ CX, CX

r8z_copy_loop:
	VMOVUPS (R8)(CX*1), Z0
	VMOVUPS 64(R8)(CX*1), Z1
	VMOVUPS Z0, (R9)(CX*1)
	VMOVUPS Z1, 64(R9)(CX*1)
	ADDQ    $128, CX
	CMPQ    CX, AX
	JL      r8z_copy_loop

r8z_true:
	VZEROUPPER
	MOVB $1, ret+136(FP)
	RET

r8z_false:
	VZEROUPPER
	MOVB $0, ret+136(FP)
	RET
