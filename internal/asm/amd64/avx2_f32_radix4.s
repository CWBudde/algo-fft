//go:build amd64 && !purego

// ===========================================================================
// AVX2 radix-4 DIT FFT, 256-bit wide, size-generic (n = 4^k, k >= 2)
// ===========================================================================
//
// Unlike the per-size avx2_f32_size*_radix4*.s kernels -- which are VEX-encoded
// but load one complex64 at a time with VMOVSD and do all arithmetic in X
// registers -- this kernel keeps four butterflies in flight per instruction in
// Y registers.
//
//   Permute: work[4g+d] = src[idx[g] + d*(n/4)], d = 0..3
//   Stage 1: n/4 groups x 1 butterfly, no twiddles. Four groups are loaded as
//            four YMM registers and transposed into a0..a3 vectors.
//   Stage s: n/(4m) groups x m butterflies, m = 4^(s-1), s = 2..k. The m
//            butterflies of a group are contiguous, so a0..a3 are plain
//            256-bit loads at stride m, and the three twiddles come from three
//            contiguous planes (see prepareTwiddleRadix4AVX2).
//
// The +-i rotations use permute+xor against a sign mask (2 ops) rather than the
// permute/xor-zero/sub/blend sequence (4 ops) of the older kernels. Forward and
// inverse differ only in which mask goes to which output, so both directions
// share one loop body: the caller passes `inverse` to pick the masks and
// `scale` (1/n, exact for a power of four) which stage 1 folds in for free.
//
// All instructions are VEX-encoded; no legacy-SSE forms appear.
// ===========================================================================

#include "textflag.h"

// r4NegOdd flips the sign of every odd float32 lane (the imaginary slot).
GLOBL ·r4NegOdd<>(SB), RODATA|NOPTR, $32
DATA ·r4NegOdd<>+0(SB)/8,  $0x8000000000000000
DATA ·r4NegOdd<>+8(SB)/8,  $0x8000000000000000
DATA ·r4NegOdd<>+16(SB)/8, $0x8000000000000000
DATA ·r4NegOdd<>+24(SB)/8, $0x8000000000000000

// r4NegEven flips the sign of every even float32 lane (the real slot).
GLOBL ·r4NegEven<>(SB), RODATA|NOPTR, $32
DATA ·r4NegEven<>+0(SB)/8,  $0x0000000080000000
DATA ·r4NegEven<>+8(SB)/8,  $0x0000000080000000
DATA ·r4NegEven<>+16(SB)/8, $0x0000000080000000
DATA ·r4NegEven<>+24(SB)/8, $0x0000000080000000

// func Radix4Complex64Asm(dst, src, twiddle, scratch []complex64, idx []int32, r4End int, inverse, fuse bool, scale float32) bool
TEXT ·Radix4Complex64Asm(SB), NOSPLIT, $0-137
	MOVQ dst+0(FP), R8       // R8  = working buffer (dst, or scratch when in-place)
	MOVQ src+24(FP), R9      // R9  = src
	MOVQ twiddle+48(FP), R10 // R10 = packed twiddle planes
	MOVQ scratch+72(FP), R11 // R11 = scratch
	MOVQ idx+96(FP), R12     // R12 = stage-1 group index table
	MOVQ src_len+32(FP), R13 // R13 = n

	// n >= 16 and a power of four is guaranteed by the Go caller; here we only
	// check that every slice is long enough to be safe to write.
	CMPQ R13, $16
	JL   r4_false

	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   r4_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   r4_false

	// The packed table is n+4 elements (see twiddleSizeRadix4AVX2); requiring
	// more than n is what makes a caller that passes the plain length-n
	// twiddle table decline here rather than transform against it.
	MOVQ twiddle_len+56(FP), AX
	LEAQ 4(R13), BX
	CMPQ AX, BX
	JL   r4_false

	MOVQ idx_len+104(FP), AX // needs n/4 entries
	MOVQ R13, BX
	SHRQ $2, BX
	CMPQ AX, BX
	JL   r4_false

	// Rotation masks. Forward: y1 uses -i (negOdd), y3 uses +i (negEven).
	// Inverse conjugates the whole butterfly, so the two swap.
	MOVBLZX inverse+128(FP), AX
	TESTL   AX, AX
	JNZ     r4_inverse_masks

	VMOVUPS ·r4NegOdd<>(SB), Y14
	VMOVUPS ·r4NegEven<>(SB), Y15
	JMP     r4_masks_done

r4_inverse_masks:
	VMOVUPS ·r4NegEven<>(SB), Y14
	VMOVUPS ·r4NegOdd<>(SB), Y15

r4_masks_done:
	// In-place transforms work out of scratch and copy back at the end.
	CMPQ R8, R9
	JNE  r4_permute
	MOVQ R11, R8

r4_permute:
	// =====================================================================
	// Permutation and stage 1, fused.
	//
	// The four inputs of stage-1 group g are src[idx[g] + d*q] for d = 0..3,
	// q = n/4, so for four consecutive groups each a_d vector is one
	// VPGATHERDQ: the index vector idx[g..g+3] is shared and only the base
	// pointer differs. That removes the separate permutation pass (a full
	// store-then-load of the whole buffer) and the input transpose as well,
	// because the gather already delivers a0..a3 in separate registers. Only
	// the output transpose remains, to store the four groups contiguously.
	//
	// Measured at n = 16384 on Alder Lake: 13.9 us for the separate passes
	// against 9.7 us fused, with stage 1's arithmetic entirely hidden behind
	// the gather. Gather throughput varies a lot between microarchitectures,
	// so this balance is worth re-checking on AMD parts.
	// =====================================================================
	MOVQ R13, BX
	SHRQ $2, BX  // BX = q = n/4 = the group count
	MOVQ BX, R14
	SHLQ $3, R14 // R14 = q*8 bytes

	LEAQ (R9)(R14*1), SI  // src + 1*q
	LEAQ (SI)(R14*1), DI  // src + 2*q
	LEAQ (DI)(R14*1), R15 // src + 3*q

	VBROADCASTSS scale+132(FP), Y10 // 1/n for inverse, 1.0 for forward

	XORQ CX, CX // group index
	XORQ DX, DX // byte offset into work

r4_stage1_loop:
	VMOVDQU (R12)(CX*4), X8 // idx[g..g+3]

	VPCMPEQD   Y9, Y9, Y9 // the gather consumes its mask, so rebuild each time
	VPGATHERDQ Y9, (R9)(X8*8), Y0
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (SI)(X8*8), Y1
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (DI)(X8*8), Y2
	VPCMPEQD   Y9, Y9, Y9
	VPGATHERDQ Y9, (R15)(X8*8), Y3

	VADDPS Y0, Y2, Y4 // t0 = a0 + a2
	VSUBPS Y2, Y0, Y5 // t1 = a0 - a2
	VADDPS Y1, Y3, Y6 // t2 = a1 + a3
	VSUBPS Y3, Y1, Y7 // t3 = a1 - a3

	VPERMILPS $0xB1, Y7, Y11 // swap re/im, shared by both rotations
	VXORPS    Y14, Y11, Y12  // -i*t3 (forward)
	VXORPS    Y15, Y11, Y11  // +i*t3 (forward)

	VADDPS Y4, Y6, Y0  // y0 = t0 + t2
	VADDPS Y5, Y12, Y1 // y1 = t1 - i*t3
	VSUBPS Y6, Y4, Y2  // y2 = t0 - t2
	VADDPS Y5, Y11, Y3 // y3 = t1 + i*t3

	// Fold the inverse 1/n here: exact for a power of two, and it saves a
	// separate streaming pass over the whole buffer.
	VMULPS Y10, Y0, Y0
	VMULPS Y10, Y1, Y1
	VMULPS Y10, Y2, Y2
	VMULPS Y10, Y3, Y3

	// Transpose the 4x4 block of complex64 (64-bit lanes) back into
	// group-major order and store four contiguous vectors.
	VUNPCKLPD  Y1, Y0, Y4
	VUNPCKHPD  Y1, Y0, Y5
	VUNPCKLPD  Y3, Y2, Y6
	VUNPCKHPD  Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3

	VMOVUPS Y0, (R8)(DX*1)
	VMOVUPS Y1, 32(R8)(DX*1)
	VMOVUPS Y2, 64(R8)(DX*1)
	VMOVUPS Y3, 96(R8)(DX*1)

	ADDQ $4, CX
	ADDQ $128, DX
	CMPQ CX, BX
	JL   r4_stage1_loop

	// =====================================================================
	// Stage m = 4.
	//
	// This stage exists for every supported n, and its three twiddle planes
	// are four elements each and identical for every group -- so both the real
	// and imaginary broadcasts hoist out of the loop, saving three loads and
	// six shuffles per four butterflies. Measured 2.54 -> 1.91 us at
	// n = 16384. Larger m cannot do this: the planes no longer fit in
	// registers, which is why only this stage is special-cased.
	// =====================================================================
	VMOVUPS   (R10), Y0
	VMOVSLDUP Y0, Y8 // w1 real
	VMOVSHDUP Y0, Y9 // w1 imag
	VMOVUPS   32(R10), Y0
	VMOVSLDUP Y0, Y10 // w2 real
	VMOVSHDUP Y0, Y11 // w2 imag
	VMOVUPS   64(R10), Y0
	VMOVSLDUP Y0, Y12 // w3 real
	VMOVSHDUP Y0, Y13 // w3 imag

	MOVQ R8, SI
	LEAQ (R8)(R13*8), R11 // end of work (the scratch pointer is dead now)

r4_m4_loop:
	VMOVUPS (SI), Y0
	VMOVUPS 32(SI), Y1
	VMOVUPS 64(SI), Y2
	VMOVUPS 96(SI), Y3

	VSHUFPS        $0xB1, Y1, Y1, Y4
	VMULPS         Y9, Y4, Y4
	VFMADDSUB213PS Y4, Y8, Y1 // a1 *= w1

	VSHUFPS        $0xB1, Y2, Y2, Y5
	VMULPS         Y11, Y5, Y5
	VFMADDSUB213PS Y5, Y10, Y2 // a2 *= w2

	VSHUFPS        $0xB1, Y3, Y3, Y6
	VMULPS         Y13, Y6, Y6
	VFMADDSUB213PS Y6, Y12, Y3 // a3 *= w3

	VADDPS Y0, Y2, Y4 // t0
	VSUBPS Y2, Y0, Y5 // t1
	VADDPS Y1, Y3, Y6 // t2
	VSUBPS Y3, Y1, Y7 // t3

	VPERMILPS $0xB1, Y7, Y2 // shared by both rotations
	VXORPS    Y15, Y2, Y3  // +i*t3
	VXORPS    Y14, Y2, Y2  // -i*t3

	VADDPS Y4, Y6, Y0 // y0
	VADDPS Y5, Y2, Y1 // y1
	VSUBPS Y6, Y4, Y2 // y2
	VADDPS Y5, Y3, Y3 // y3

	VMOVUPS Y0, (SI)
	VMOVUPS Y1, 32(SI)
	VMOVUPS Y2, 64(SI)
	VMOVUPS Y3, 96(SI)

	ADDQ $128, SI
	CMPQ SI, R11
	JL   r4_m4_loop

	// Advance past this stage's three 4-element planes and start the general
	// loop at m = 16, if another radix-4 stage fits.
	ADDQ $96, R10
	MOVQ $16, BX
	MOVQ BX, DX
	SHLQ $2, DX
	CMPQ DX, r4End+120(FP)
	JG   r4_radix2_tail

	// =====================================================================
	// Remaining radix-4 stages, m = 16, 64, ... The m butterflies of a group
	// are contiguous, so a0..a3 are plain 256-bit loads at stride m and the
	// three twiddles come from three contiguous planes.
	// =====================================================================
r4_stage_setup:
	MOVQ BX, AX
	SHLQ $3, AX  // AX = m*8, the byte stride between a0/a1, a2/a3 and planes
	MOVQ AX, R15
	SHLQ $2, R15 // R15 = 4*m*8, the byte stride between groups
	MOVQ R8, R14 // R14 = group base

	// Take the fused path when this is the last radix-4 stage of a shape that
	// has a tail. Both conditions are read off the loop bounds, not off n:
	// a tail exists iff r4End < n, and this stage is the last iff the next one
	// (span 16m) would overrun r4End. n = 32 is the one shape that reaches the
	// tail without entering this loop at all -- there the hoisted m = 4 stage
	// is the last one -- and it keeps the separate tail below.
	MOVBLZX fuse+129(FP), DX
	TESTL   DX, DX
	JZ      r4_group_loop
	MOVQ    r4End+120(FP), DX
	CMPQ    DX, R13
	JGE     r4_group_loop // r4End == n: power of four, no tail
	MOVQ    BX, DX
	SHLQ    $4, DX
	CMPQ    DX, r4End+120(FP)
	JG      r4_fused_last

r4_group_loop:
	MOVQ R14, SI          // SI = &a0
	LEAQ (R14)(AX*2), DI  // DI = &a2
	MOVQ R10, CX          // CX = &w1[0] for this stage
	MOVQ BX, DX           // DX = butterflies remaining in this group

r4_inner_loop:
	// The twiddle broadcasts come straight from memory: VMOVSLDUP/VMOVSHDUP
	// with a 256-bit memory source are pure load uops, where the register
	// forms are port-5 shuffles. That takes six shuffles per iteration off
	// the critical port and costs three extra loads, which have spare slots.
	VMOVSLDUP (CX), Y8         // [w1.re, w1.re, ...]
	VMOVSHDUP (CX), Y9         // [w1.im, w1.im, ...]
	VMOVSLDUP (CX)(AX*1), Y10  // w2.re
	VMOVSHDUP (CX)(AX*1), Y11  // w2.im

	VMOVUPS (SI), Y0        // a0
	VMOVUPS (SI)(AX*1), Y1  // a1
	VMOVUPS (DI), Y2        // a2
	VMOVUPS (DI)(AX*1), Y3  // a3

	// a1 *= w1. VFMADDSUB213PS gives dst = Y8*dst -/+ Y13, i.e.
	// re = a.re*w.re - a.im*w.im, im = a.im*w.re + a.re*w.im.
	VSHUFPS        $0xB1, Y1, Y1, Y13
	VMULPS         Y9, Y13, Y13
	VFMADDSUB213PS Y13, Y8, Y1

	// a2 *= w2
	VSHUFPS        $0xB1, Y2, Y2, Y13
	VMULPS         Y11, Y13, Y13
	VFMADDSUB213PS Y13, Y10, Y2

	// a3 *= w3
	VMOVSLDUP      (CX)(AX*2), Y8
	VMOVSHDUP      (CX)(AX*2), Y9
	VSHUFPS        $0xB1, Y3, Y3, Y13
	VMULPS         Y9, Y13, Y13
	VFMADDSUB213PS Y13, Y8, Y3

	VADDPS Y0, Y2, Y4 // t0 = a0 + a2
	VSUBPS Y2, Y0, Y5 // t1 = a0 - a2
	VADDPS Y1, Y3, Y6 // t2 = a1 + a3
	VSUBPS Y3, Y1, Y7 // t3 = a1 - a3

	// Both rotations permute the same t3, so permute once and branch on the
	// mask: two XORs can issue away from port 5, a second permute cannot.
	VPERMILPS $0xB1, Y7, Y11
	VXORPS    Y14, Y11, Y12 // -i*t3 (forward)
	VXORPS    Y15, Y11, Y11 // +i*t3 (forward)

	VADDPS Y4, Y6, Y0  // y0
	VADDPS Y5, Y12, Y1 // y1
	VSUBPS Y6, Y4, Y2  // y2
	VADDPS Y5, Y11, Y3 // y3

	VMOVUPS Y0, (SI)
	VMOVUPS Y1, (SI)(AX*1)
	VMOVUPS Y2, (DI)
	VMOVUPS Y3, (DI)(AX*1)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, CX
	SUBQ $4, DX
	JNZ  r4_inner_loop

	ADDQ R15, R14
	CMPQ R14, R11
	JL   r4_group_loop

	// Next stage: advance past this stage's three planes (3*m elements) and
	// quadruple m. Continue while another radix-4 stage fits within r4End.
	LEAQ (R10)(AX*2), R10
	ADDQ AX, R10
	SHLQ $2, BX
	MOVQ BX, DX
	SHLQ $2, DX
	CMPQ DX, r4End+120(FP)
	JLE  r4_stage_setup

	JMP r4_radix2_tail // the fused block below is only reached by its guard

r4_fused_last:
	// =====================================================================
	// Last radix-4 stage with the radix-2 tail fused in. See the f64 twin for
	// the full argument; in short, the last stage always has 4m = r4End = n/2
	// and therefore exactly two groups -- the even half and the odd half --
	// and the tail pairs one output of each at the same position. Running the
	// groups in lockstep leaves both operands in registers, so the tail's
	// separate pass over the whole buffer disappears.
	//
	// Y0..Y3 hold group 0's outputs across group 1's whole computation and
	// Y4..Y7 group 1's, which leaves only Y8..Y13 as scratch (Y14/Y15 are the
	// rotation masks). Group 1 therefore re-loads the three twiddle
	// broadcasts rather than keeping them: they are L1-hot pure load uops on a
	// loop bound by port 5.
	// =====================================================================
	MOVQ R8, SI          // SI = &a0, group 0
	LEAQ (R8)(AX*2), DI  // DI = &a2, group 0
	LEAQ (SI)(R15*1), R9 // R9 = &a0, group 1 (one half further on)
	LEAQ (DI)(R15*1), R11
	MOVQ R10, CX         // CX = &w1[0] for this stage
	LEAQ (R10)(AX*2), R12
	ADDQ AX, R12         // R12 = the tail's n/2 twiddles, just past 3*m planes
	MOVQ BX, DX          // DX = butterflies remaining in a group

r4_fused_loop:
	// ---- group 0: the even half -------------------------------------
	VMOVSLDUP (CX), Y8
	VMOVSHDUP (CX), Y9
	VMOVSLDUP (CX)(AX*1), Y10
	VMOVSHDUP (CX)(AX*1), Y11

	VMOVUPS (SI), Y0       // a0
	VMOVUPS (SI)(AX*1), Y1 // a1
	VMOVUPS (DI), Y2       // a2
	VMOVUPS (DI)(AX*1), Y3 // a3

	VSHUFPS        $0xB1, Y1, Y1, Y13
	VMULPS         Y9, Y13, Y13
	VFMADDSUB213PS Y13, Y8, Y1 // a1 *= w1

	VSHUFPS        $0xB1, Y2, Y2, Y13
	VMULPS         Y11, Y13, Y13
	VFMADDSUB213PS Y13, Y10, Y2 // a2 *= w2

	VMOVSLDUP      (CX)(AX*2), Y8
	VMOVSHDUP      (CX)(AX*2), Y9
	VSHUFPS        $0xB1, Y3, Y3, Y13
	VMULPS         Y9, Y13, Y13
	VFMADDSUB213PS Y13, Y8, Y3 // a3 *= w3

	VADDPS Y0, Y2, Y4 // t0 = a0 + a2
	VSUBPS Y2, Y0, Y5 // t1 = a0 - a2
	VADDPS Y1, Y3, Y6 // t2 = a1 + a3
	VSUBPS Y3, Y1, Y7 // t3 = a1 - a3

	VPERMILPS $0xB1, Y7, Y11
	VXORPS    Y14, Y11, Y12 // -i*t3 (forward)
	VXORPS    Y15, Y11, Y11 // +i*t3 (forward)

	VADDPS Y4, Y6, Y0  // y0
	VADDPS Y5, Y12, Y1 // y1
	VSUBPS Y6, Y4, Y2  // y2
	VADDPS Y5, Y11, Y3 // y3

	// ---- group 1: the odd half, same twiddle planes -----------------
	VMOVSLDUP (CX), Y8
	VMOVSHDUP (CX), Y9
	VMOVSLDUP (CX)(AX*1), Y10
	VMOVSHDUP (CX)(AX*1), Y11

	VMOVUPS (R9), Y4        // a0
	VMOVUPS (R9)(AX*1), Y5  // a1
	VMOVUPS (R11), Y6       // a2
	VMOVUPS (R11)(AX*1), Y7 // a3

	VSHUFPS        $0xB1, Y5, Y5, Y13
	VMULPS         Y9, Y13, Y13
	VFMADDSUB213PS Y13, Y8, Y5 // a1 *= w1

	VSHUFPS        $0xB1, Y6, Y6, Y13
	VMULPS         Y11, Y13, Y13
	VFMADDSUB213PS Y13, Y10, Y6 // a2 *= w2

	VMOVSLDUP      (CX)(AX*2), Y8
	VMOVSHDUP      (CX)(AX*2), Y9
	VSHUFPS        $0xB1, Y7, Y7, Y13
	VMULPS         Y9, Y13, Y13
	VFMADDSUB213PS Y13, Y8, Y7 // a3 *= w3

	// t0..t3 land in Y8..Y11: Y0..Y3 are group 0's outputs and Y4..Y7 become
	// group 1's, so the scratch bank is the only place left.
	VADDPS Y4, Y6, Y8  // t0 = a0 + a2
	VSUBPS Y6, Y4, Y9  // t1 = a0 - a2
	VADDPS Y5, Y7, Y10 // t2 = a1 + a3
	VSUBPS Y7, Y5, Y11 // t3 = a1 - a3

	VPERMILPS $0xB1, Y11, Y12
	VXORPS    Y14, Y12, Y13 // -i*t3 (forward)
	VXORPS    Y15, Y12, Y12 // +i*t3 (forward)

	VADDPS Y8, Y10, Y4 // z0
	VADDPS Y9, Y13, Y5 // z1
	VSUBPS Y10, Y8, Y6 // z2
	VADDPS Y9, Y12, Y7 // z3

	// ---- the fused radix-2 tail: four butterflies (y_d, z_d) --------
	// The tail twiddle for the output at offset j + d*m is W_n^(j+d*m), so
	// the four are at the same d*m stride the stage already addresses with AX.
	LEAQ (R12)(AX*2), R14 // &w[j + 2m], so d=3 is (R14)(AX*1)

	// d = 0
	VMOVSLDUP      (R12), Y8
	VMOVSHDUP      (R12), Y9
	VSHUFPS        $0xB1, Y4, Y4, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y4 // z0 *= w
	VADDPS         Y4, Y0, Y11
	VSUBPS         Y4, Y0, Y12
	VMOVUPS        Y11, (SI)
	VMOVUPS        Y12, (R9)

	// d = 1
	VMOVSLDUP      (R12)(AX*1), Y8
	VMOVSHDUP      (R12)(AX*1), Y9
	VSHUFPS        $0xB1, Y5, Y5, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y5
	VADDPS         Y5, Y1, Y11
	VSUBPS         Y5, Y1, Y12
	VMOVUPS        Y11, (SI)(AX*1)
	VMOVUPS        Y12, (R9)(AX*1)

	// d = 2
	VMOVSLDUP      (R12)(AX*2), Y8
	VMOVSHDUP      (R12)(AX*2), Y9
	VSHUFPS        $0xB1, Y6, Y6, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y6
	VADDPS         Y6, Y2, Y11
	VSUBPS         Y6, Y2, Y12
	VMOVUPS        Y11, (DI)
	VMOVUPS        Y12, (R11)

	// d = 3
	VMOVSLDUP      (R14)(AX*1), Y8
	VMOVSHDUP      (R14)(AX*1), Y9
	VSHUFPS        $0xB1, Y7, Y7, Y10
	VMULPS         Y9, Y10, Y10
	VFMADDSUB213PS Y10, Y8, Y7
	VADDPS         Y7, Y3, Y11
	VSUBPS         Y7, Y3, Y12
	VMOVUPS        Y11, (DI)(AX*1)
	VMOVUPS        Y12, (R11)(AX*1)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, R9
	ADDQ $32, R11
	ADDQ $32, CX
	ADDQ $32, R12
	SUBQ $4, DX
	JNZ  r4_fused_loop

	JMP r4_copy_out

r4_radix2_tail:
	// =====================================================================
	// Radix-2 tail, for n = 2*4^k only: the radix-4 stages above transformed
	// the two halves independently, so combine them at distance n/2.
	//   y0[j] = a0 + w[j]*a1,  y1[j] = a0 - w[j]*a1
	// R10 already points at the tail's n/2 twiddles.
	// =====================================================================
	MOVQ r4End+120(FP), AX
	CMPQ AX, R13
	JGE  r4_copy_out

	MOVQ R13, DX
	SHRQ $1, DX  // DX = half = n/2
	MOVQ R8, SI  // SI = &a0
	LEAQ (R8)(DX*8), DI // DI = &a1
	SHLQ $3, DX  // DX = half*8, byte length of one half

	XORQ CX, CX

r4_tail_loop:
	VMOVSLDUP (R10)(CX*1), Y11 // w.re, broadcast by the load itself
	VMOVSHDUP (R10)(CX*1), Y12 // w.im
	VMOVUPS (SI)(CX*1), Y0  // a0
	VMOVUPS (DI)(CX*1), Y1  // a1

	VSHUFPS        $0xB1, Y1, Y1, Y13
	VMULPS         Y12, Y13, Y13
	VFMADDSUB213PS Y13, Y11, Y1 // a1 *= w

	VADDPS Y0, Y1, Y2 // y0 = a0 + a1
	VSUBPS Y1, Y0, Y3 // y1 = a0 - a1

	VMOVUPS Y2, (SI)(CX*1)
	VMOVUPS Y3, (DI)(CX*1)

	ADDQ $32, CX
	CMPQ CX, DX
	JL   r4_tail_loop

	// =====================================================================
	// Copy out if the transform ran in scratch.
	// =====================================================================
r4_copy_out:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4_true

	MOVQ R13, AX
	SHLQ $3, AX
	XORQ CX, CX

r4_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ    $64, CX
	CMPQ    CX, AX
	JL      r4_copy_loop

r4_true:
	VZEROUPPER
	MOVB $1, ret+136(FP)
	RET

r4_false:
	VZEROUPPER
	MOVB $0, ret+136(FP)
	RET
