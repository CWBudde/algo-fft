//go:build amd64 && !purego

// ===========================================================================
// AVX2 radix-4 DIT FFT, 256-bit wide, size-generic (n = 4^k or 2*4^k)
// ===========================================================================
//
// The complex128 twin of avx2_f32_radix4.s. Same algorithm, same twiddle-plane
// layout, same group index table -- the permutation is precision-independent,
// so both kernels share internal/kernels' radix4GroupIndices.
//
// The one structural difference is the vector width: a YMM register holds two
// complex128 rather than four complex64, so every loop here retires half as
// many butterflies per iteration. Two consequences:
//
//   - Stage 1 processes two groups per iteration instead of four. The f32
//     kernel gathers its four inputs with VPGATHERDQ (one qword per
//     complex64); there is no 128-bit-element gather, so this kernel loads
//     each group's four inputs as XMM and folds the second group in with
//     VINSERTF128. That is 8 instructions per iteration against the f32 path's
//     4 gathers, but a gather of 4 elements costs far more than 4 loads --
//     and it keeps the permutation fused into stage 1 either way, which is
//     where the win actually came from.
//   - The output transpose is a 2x2 block of 128-bit lanes, so four
//     VPERM2F128 replace the f32 kernel's 4x4 VUNPCK/VPERM2F128 sequence.
//
// There is no hoisted m = 4 stage: at this width the stage's three twiddle
// planes need two YMM each, and six broadcast registers plus the two rotation
// masks leave too few for the butterfly. The general stage loop starts at
// m = 4 instead.
//
// The +-i rotations use permute+xor against a sign mask. Forward and inverse
// differ only in which mask goes to which output, so both directions share one
// loop body: the caller passes `inverse` to pick the masks and `scale` (1/n,
// exact for a power of two) which stage 1 folds in for free.
//
// All instructions are VEX-encoded; no legacy-SSE forms appear.
// ===========================================================================

#include "textflag.h"

// r4dNegOdd flips the sign of the high float64 of each 128-bit lane (the
// imaginary part).
GLOBL ·r4dNegOdd<>(SB), RODATA|NOPTR, $32
DATA ·r4dNegOdd<>+0(SB)/8,  $0x0000000000000000
DATA ·r4dNegOdd<>+8(SB)/8,  $0x8000000000000000
DATA ·r4dNegOdd<>+16(SB)/8, $0x0000000000000000
DATA ·r4dNegOdd<>+24(SB)/8, $0x8000000000000000

// r4dNegEven flips the sign of the low float64 of each 128-bit lane (the real
// part).
GLOBL ·r4dNegEven<>(SB), RODATA|NOPTR, $32
DATA ·r4dNegEven<>+0(SB)/8,  $0x8000000000000000
DATA ·r4dNegEven<>+8(SB)/8,  $0x0000000000000000
DATA ·r4dNegEven<>+16(SB)/8, $0x8000000000000000
DATA ·r4dNegEven<>+24(SB)/8, $0x0000000000000000

// func Radix4Complex128Asm(dst, src, twiddle, scratch []complex128, idx []int32, r4End int, inverse bool, scale float64) bool
TEXT ·Radix4Complex128Asm(SB), NOSPLIT, $0-145
	MOVQ dst+0(FP), R8       // R8  = working buffer (dst, or scratch when in-place)
	MOVQ src+24(FP), R9      // R9  = src
	MOVQ twiddle+48(FP), R10 // R10 = packed twiddle planes
	MOVQ scratch+72(FP), R11 // R11 = scratch
	MOVQ idx+96(FP), R12     // R12 = stage-1 group index table
	MOVQ src_len+32(FP), R13 // R13 = n

	// n >= 16 and a power of two is guaranteed by the Go caller; here we only
	// check that every slice is long enough to be safe to write.
	CMPQ R13, $16
	JL   r4d_false

	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   r4d_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   r4d_false

	// The packed table is n+4 elements (see twiddleSizeRadix4AVX2Complex128);
	// requiring more than n is what makes a caller that passes the plain
	// length-n twiddle table decline here rather than transform against it.
	MOVQ twiddle_len+56(FP), AX
	LEAQ 4(R13), BX
	CMPQ AX, BX
	JL   r4d_false

	MOVQ idx_len+104(FP), AX // needs n/4 entries
	MOVQ R13, BX
	SHRQ $2, BX
	CMPQ AX, BX
	JL   r4d_false

	// Rotation masks. Forward: y1 uses -i (negOdd), y3 uses +i (negEven).
	// Inverse conjugates the whole butterfly, so the two swap.
	MOVBLZX inverse+128(FP), AX
	TESTL   AX, AX
	JNZ     r4d_inverse_masks

	VMOVUPD ·r4dNegOdd<>(SB), Y14
	VMOVUPD ·r4dNegEven<>(SB), Y15
	JMP     r4d_masks_done

r4d_inverse_masks:
	VMOVUPD ·r4dNegEven<>(SB), Y14
	VMOVUPD ·r4dNegOdd<>(SB), Y15

r4d_masks_done:
	// In-place transforms work out of scratch and copy back at the end.
	CMPQ R8, R9
	JNE  r4d_permute
	MOVQ R11, R8

r4d_permute:
	// =====================================================================
	// Permutation and stage 1, fused.
	//
	// The four inputs of stage-1 group g are src[idx[g] + d*q] for d = 0..3,
	// q = n/4. Loading two consecutive groups gives each a_d vector directly,
	// so the separate permutation pass (a full store-then-load of the whole
	// buffer) and the input transpose both disappear; only the output
	// transpose remains, to store the two groups contiguously.
	// =====================================================================
	MOVQ R13, R11
	SHRQ $2, R11 // R11 = q = n/4 = the group count
	MOVQ R11, R14
	SHLQ $4, R14 // R14 = q*16 bytes

	LEAQ (R9)(R14*1), SI  // src + 1*q
	LEAQ (SI)(R14*1), DI  // src + 2*q
	LEAQ (DI)(R14*1), R15 // src + 3*q

	VBROADCASTSD scale+136(FP), Y10 // 1/n for inverse, 1.0 for forward

	XORQ CX, CX // group index
	XORQ DX, DX // byte offset into work

r4d_stage1_loop:
	MOVLQZX (R12)(CX*4), AX  // idx[g]
	MOVLQZX 4(R12)(CX*4), BX // idx[g+1]
	SHLQ    $4, AX           // element index -> byte offset
	SHLQ    $4, BX

	VMOVUPD     (R9)(AX*1), X0
	VINSERTF128 $1, (R9)(BX*1), Y0, Y0 // a0 = [src[i0], src[i1]]
	VMOVUPD     (SI)(AX*1), X1
	VINSERTF128 $1, (SI)(BX*1), Y1, Y1 // a1
	VMOVUPD     (DI)(AX*1), X2
	VINSERTF128 $1, (DI)(BX*1), Y2, Y2 // a2
	VMOVUPD     (R15)(AX*1), X3
	VINSERTF128 $1, (R15)(BX*1), Y3, Y3 // a3

	VADDPD Y0, Y2, Y4 // t0 = a0 + a2
	VSUBPD Y2, Y0, Y5 // t1 = a0 - a2
	VADDPD Y1, Y3, Y6 // t2 = a1 + a3
	VSUBPD Y3, Y1, Y7 // t3 = a1 - a3

	VPERMILPD $0x5, Y7, Y11 // swap re/im, shared by both rotations
	VXORPD    Y14, Y11, Y12 // -i*t3 (forward)
	VXORPD    Y15, Y11, Y11 // +i*t3 (forward)

	VADDPD Y4, Y6, Y0  // y0 = t0 + t2
	VADDPD Y5, Y12, Y1 // y1 = t1 - i*t3
	VSUBPD Y6, Y4, Y2  // y2 = t0 - t2
	VADDPD Y5, Y11, Y3 // y3 = t1 + i*t3

	// Fold the inverse 1/n here: exact for a power of two, and it saves a
	// separate streaming pass over the whole buffer.
	VMULPD Y10, Y0, Y0
	VMULPD Y10, Y1, Y1
	VMULPD Y10, Y2, Y2
	VMULPD Y10, Y3, Y3

	// Transpose the 2x2 block of 128-bit lanes back into group-major order:
	// y0..y3 hold [A_d, B_d] for the two groups A and B, and the store needs
	// [A0 A1 A2 A3][B0 B1 B2 B3].
	VPERM2F128 $0x20, Y1, Y0, Y4 // [A0, A1]
	VPERM2F128 $0x20, Y3, Y2, Y5 // [A2, A3]
	VPERM2F128 $0x31, Y1, Y0, Y6 // [B0, B1]
	VPERM2F128 $0x31, Y3, Y2, Y7 // [B2, B3]

	VMOVUPD Y4, (R8)(DX*1)
	VMOVUPD Y5, 32(R8)(DX*1)
	VMOVUPD Y6, 64(R8)(DX*1)
	VMOVUPD Y7, 96(R8)(DX*1)

	ADDQ $2, CX
	ADDQ $128, DX
	CMPQ CX, R11
	JL   r4d_stage1_loop

	// =====================================================================
	// Remaining radix-4 stages, m = 4, 16, 64, ... The m butterflies of a
	// group are contiguous, so a0..a3 are plain 256-bit loads at stride m and
	// the three twiddles come from three contiguous planes.
	// =====================================================================
	MOVQ R13, R11
	SHLQ $4, R11
	ADDQ R8, R11 // R11 = end of work (the scratch pointer is dead now)

	MOVQ $4, BX // m
	MOVQ BX, DX
	SHLQ $2, DX
	CMPQ DX, r4End+120(FP)
	JG   r4d_radix2_tail

r4d_stage_setup:
	MOVQ BX, AX
	SHLQ $4, AX  // AX = m*16, the byte stride between a0/a1, a2/a3 and planes
	MOVQ AX, R15
	SHLQ $2, R15 // R15 = 4*m*16, the byte stride between groups
	MOVQ R8, R14 // R14 = group base

r4d_group_loop:
	MOVQ R14, SI         // SI = &a0
	LEAQ (R14)(AX*2), DI // DI = &a2
	MOVQ R10, CX         // CX = &w1[0] for this stage
	MOVQ BX, DX          // DX = butterflies remaining in this group

r4d_inner_loop:
	// The twiddle broadcasts come straight from memory: VMOVDDUP with a
	// 256-bit memory source duplicates the low float64 of each 128-bit lane
	// and is a pure load uop, where the register form is a port-5 shuffle.
	// Offsetting the address by 8 bytes duplicates the high float64 instead,
	// so the imaginary broadcast is also free of port 5. That reads 8 bytes
	// past the last plane, which is why twiddleSizeRadix4AVX2Complex128 pads
	// the table -- and why the length check above insists on n+4.
	VMOVDDUP (CX), Y8         // [w1.re, w1.re] per lane
	VMOVDDUP 8(CX), Y9        // [w1.im, w1.im] per lane
	VMOVDDUP (CX)(AX*1), Y10  // w2.re
	VMOVDDUP 8(CX)(AX*1), Y11 // w2.im

	VMOVUPD (SI), Y0       // a0
	VMOVUPD (SI)(AX*1), Y1 // a1
	VMOVUPD (DI), Y2       // a2
	VMOVUPD (DI)(AX*1), Y3 // a3

	// a1 *= w1. VFMADDSUB213PD gives dst = Y8*dst -/+ Y13, i.e.
	// re = a.re*w.re - a.im*w.im, im = a.im*w.re + a.re*w.im.
	VPERMILPD      $0x5, Y1, Y13 // swap re/im
	VMULPD         Y9, Y13, Y13
	VFMADDSUB213PD Y13, Y8, Y1

	// a2 *= w2
	VPERMILPD      $0x5, Y2, Y13
	VMULPD         Y11, Y13, Y13
	VFMADDSUB213PD Y13, Y10, Y2

	// a3 *= w3
	VMOVDDUP       (CX)(AX*2), Y8
	VMOVDDUP       8(CX)(AX*2), Y9
	VPERMILPD      $0x5, Y3, Y13
	VMULPD         Y9, Y13, Y13
	VFMADDSUB213PD Y13, Y8, Y3

	VADDPD Y0, Y2, Y4 // t0 = a0 + a2
	VSUBPD Y2, Y0, Y5 // t1 = a0 - a2
	VADDPD Y1, Y3, Y6 // t2 = a1 + a3
	VSUBPD Y3, Y1, Y7 // t3 = a1 - a3

	// Both rotations permute the same t3, so permute once and branch on the
	// mask: two XORs can issue away from port 5, a second permute cannot.
	VPERMILPD $0x5, Y7, Y11
	VXORPD    Y14, Y11, Y12 // -i*t3 (forward)
	VXORPD    Y15, Y11, Y11 // +i*t3 (forward)

	VADDPD Y4, Y6, Y0  // y0
	VADDPD Y5, Y12, Y1 // y1
	VSUBPD Y6, Y4, Y2  // y2
	VADDPD Y5, Y11, Y3 // y3

	VMOVUPD Y0, (SI)
	VMOVUPD Y1, (SI)(AX*1)
	VMOVUPD Y2, (DI)
	VMOVUPD Y3, (DI)(AX*1)

	ADDQ $32, SI
	ADDQ $32, DI
	ADDQ $32, CX
	SUBQ $2, DX
	JNZ  r4d_inner_loop

	ADDQ R15, R14
	CMPQ R14, R11
	JL   r4d_group_loop

	// Next stage: advance past this stage's three planes (3*m elements) and
	// quadruple m. Continue while another radix-4 stage fits within r4End.
	LEAQ (R10)(AX*2), R10
	ADDQ AX, R10
	SHLQ $2, BX
	MOVQ BX, DX
	SHLQ $2, DX
	CMPQ DX, r4End+120(FP)
	JLE  r4d_stage_setup

r4d_radix2_tail:
	// =====================================================================
	// Radix-2 tail, for n = 2*4^k only: the radix-4 stages above transformed
	// the two halves independently, so combine them at distance n/2.
	//   y0[j] = a0 + w[j]*a1,  y1[j] = a0 - w[j]*a1
	// R10 already points at the tail's n/2 twiddles.
	// =====================================================================
	MOVQ r4End+120(FP), AX
	CMPQ AX, R13
	JGE  r4d_copy_out

	MOVQ R13, DX
	SHRQ $1, DX          // DX = half = n/2
	SHLQ $4, DX          // DX = half*16, byte length of one half
	MOVQ R8, SI          // SI = &a0
	LEAQ (R8)(DX*1), DI  // DI = &a1

	XORQ CX, CX

r4d_tail_loop:
	VMOVDDUP  (R10)(CX*1), Y11 // w.re, broadcast by the load itself
	VMOVDDUP  8(R10)(CX*1), Y12 // w.im
	VMOVUPD (SI)(CX*1), Y0  // a0
	VMOVUPD (DI)(CX*1), Y1  // a1

	VPERMILPD      $0x5, Y1, Y13
	VMULPD         Y12, Y13, Y13
	VFMADDSUB213PD Y13, Y11, Y1 // a1 *= w

	VADDPD Y0, Y1, Y2 // y0 = a0 + a1
	VSUBPD Y1, Y0, Y3 // y1 = a0 - a1

	VMOVUPD Y2, (SI)(CX*1)
	VMOVUPD Y3, (DI)(CX*1)

	ADDQ $32, CX
	CMPQ CX, DX
	JL   r4d_tail_loop

	// =====================================================================
	// Copy out if the transform ran in scratch.
	// =====================================================================
r4d_copy_out:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   r4d_true

	MOVQ R13, AX
	SHLQ $4, AX
	XORQ CX, CX

r4d_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ    $64, CX
	CMPQ    CX, AX
	JL      r4d_copy_loop

r4d_true:
	VZEROUPPER
	MOVB $1, ret+144(FP)
	RET

r4d_false:
	VZEROUPPER
	MOVB $0, ret+144(FP)
	RET
