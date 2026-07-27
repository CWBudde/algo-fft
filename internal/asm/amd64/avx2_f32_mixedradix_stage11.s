//go:build amd64 && !purego

// ===========================================================================
// AVX2 fused radix-11 mixed-radix stage (complex64) for AMD64
// ===========================================================================
//
// The radix-11 sibling of avx2_f32_mixedradix_stage7.s. See that file for the
// stage contract, the layout of `table`, and why k is the vector axis. Two
// things differ, and both follow from the radix no longer fitting the register
// file.
//
// ===========================================================================
//
// RADIX-11 BUTTERFLY
// ------------------
// With W = exp(-2*pi*i/11), cr = cos(2*pi*r/11) and sr = sin(2*pi*r/11), the
// five conjugate pairs W^(11-r) = conj(W^r) collapse the direct form the same
// way radix 7's three pairs do:
//
//	tk = ak + a(11-k)   uk = ak - a(11-k)     k = 1..5
//	y0 = a0 + t1 + t2 + t3 + t4 + t5
//	mj = a0 + sum_k c[jk mod 11] * tk
//	qj = -i *  sum_k s[jk mod 11] * uk
//	yj = mj + qj    y(11-j) = mj - qj         j = 1..5
//
// Both index maps are r = j*k mod 11 folded into 1..5: cos is even about 11,
// so c[11-r] = c[r] and the cosine rows need no signs; sin is odd, so
// s[11-r] = -s[r] and the folding shows up as the FNMADD rows below. Written
// out (verified against kernels.Butterfly11*Complex128, the 11x11 matrix form):
//
//	m1 = a0 + c1*t1 + c2*t2 + c3*t3 + c4*t4 + c5*t5
//	q1 = -i*( s1*u1 + s2*u2 + s3*u3 + s4*u4 + s5*u5)
//	m2 = a0 + c2*t1 + c4*t2 + c5*t3 + c3*t4 + c1*t5
//	q2 = -i*( s2*u1 + s4*u2 - s5*u3 - s3*u4 - s1*u5)
//	m3 = a0 + c3*t1 + c5*t2 + c2*t3 + c1*t4 + c4*t5
//	q3 = -i*( s3*u1 - s5*u2 - s2*u3 + s1*u4 + s4*u5)
//	m4 = a0 + c4*t1 + c3*t2 + c1*t3 + c5*t4 + c2*t5
//	q4 = -i*( s4*u1 - s3*u2 + s1*u3 + s5*u4 - s2*u5)
//	m5 = a0 + c5*t1 + c1*t2 + c4*t3 + c2*t4 + c3*t5
//	q5 = -i*( s5*u1 - s1*u2 + s4*u3 - s2*u4 + s3*u5)
//
// As in the radix-5 and radix-7 kernels the inverse stage replaces every -i
// with +i, which is the same pair swap against a different XOR mask.
//
// ===========================================================================
//
// REGISTER BUDGET: WHY THE CONSTANTS LIVE IN MEMORY
// -------------------------------------------------
// t1..t5 and u1..u5 are all live across the whole output half — mj reads every
// t and qj reads every u — so ten YMM registers are spoken for before any
// constant. Ten broadcast constants plus the sign mask would need eleven more,
// and there are sixteen.
//
// So the constants are held pre-broadcast in RODATA and read as FMA memory
// operands instead. Each is 320 bytes of table in total, hot in L1 after the
// first block, and an FMA with a memory source stays a single fused uop, so
// this costs issue slots only where a broadcast would have cost registers.
// That leaves Y1..Y5 for t1..t5, Y10..Y14 for u1..u5, Y15 for the sign mask
// and Y0/Y6..Y9 as scratch.
//
// Row offsets are the same squeeze one level down: ten rows would need ten
// index registers, but the SIB scale covers the even multiples for free.
// Holding 1x, 3x, 5x, 7x and 9x the row stride reaches all ten rows —
// row 2 = 1x*2, row 4 = 1x*4, row 6 = 3x*2, row 8 = 1x*8, row 10 = 5x*2 —
// which also shortens the prologue, and at these spans the prologue is where
// the break-even sits (see mixedRadixStageMinMuls).
//
// The store order is the radix-7 one for the radix-7 reason: a0 is never
// resident, so dst row 0 is written last, after the final read of input row 0.
// Rows 1..10 of dst are written only once every input row has been consumed
// into a t/u pair, so the documented dst == input aliasing still holds.
//
// ===========================================================================

#include "textflag.h"

// Sign masks for the multiply-by-i step. Selected by the `inverse` argument.
DATA stage11_negodd<>+0x00(SB)/4, $0x00000000
DATA stage11_negodd<>+0x04(SB)/4, $0x80000000
DATA stage11_negodd<>+0x08(SB)/4, $0x00000000
DATA stage11_negodd<>+0x0C(SB)/4, $0x80000000
GLOBL stage11_negodd<>(SB), RODATA|NOPTR, $16

DATA stage11_negeven<>+0x00(SB)/4, $0x80000000
DATA stage11_negeven<>+0x04(SB)/4, $0x00000000
DATA stage11_negeven<>+0x08(SB)/4, $0x80000000
DATA stage11_negeven<>+0x0C(SB)/4, $0x00000000
GLOBL stage11_negeven<>(SB), RODATA|NOPTR, $16

// Butterfly constants, stored pre-broadcast so they can be FMA memory
// operands. cr = cos(2*pi*r/11), sr = sin(2*pi*r/11).
#define BCAST8(sym, bits) \
	DATA sym+0x00(SB)/4, $bits \
	DATA sym+0x04(SB)/4, $bits \
	DATA sym+0x08(SB)/4, $bits \
	DATA sym+0x0C(SB)/4, $bits \
	DATA sym+0x10(SB)/4, $bits \
	DATA sym+0x14(SB)/4, $bits \
	DATA sym+0x18(SB)/4, $bits \
	DATA sym+0x1C(SB)/4, $bits \
	GLOBL sym(SB), RODATA|NOPTR, $32

BCAST8(stage11_c1<>, 0x3F575C64) // c1 =  0.84125353
BCAST8(stage11_c2<>, 0x3ED4B147) // c2 =  0.41541501
BCAST8(stage11_c3<>, 0xBE11BAFB) // c3 = -0.14231484
BCAST8(stage11_c4<>, 0xBF27A4F4) // c4 = -0.65486073
BCAST8(stage11_c5<>, 0xBF75A155) // c5 = -0.95949297
BCAST8(stage11_s1<>, 0x3F0A6770) // s1 =  0.54064082
BCAST8(stage11_s2<>, 0x3F68DDA4) // s2 =  0.90963200
BCAST8(stage11_s3<>, 0x3F7D64F0) // s3 =  0.98982144
BCAST8(stage11_s4<>, 0x3F4178CE) // s4 =  0.75574957
BCAST8(stage11_s5<>, 0x3E903F40) // s5 =  0.28173256

// ===========================================================================
// func MixedRadixStage11Complex64AVX2Asm(dst, input, table []complex64, span int, inverse bool)
//
// Stack frame (offsets from FP):
//   dst:     0(FP) ptr,   8(FP) len,  16(FP) cap
//   input:  24(FP) ptr,  32(FP) len,  40(FP) cap
//   table:  48(FP) ptr,  56(FP) len,  64(FP) cap
//   span:   72(FP)
//   inverse:80(FP)
// ===========================================================================
TEXT ·MixedRadixStage11Complex64AVX2Asm(SB), NOSPLIT, $0-81
	MOVQ dst+0(FP), DI       // DI = &dst[k],   advances
	MOVQ input+24(FP), SI    // SI = &input[k], advances
	MOVQ table+48(FP), DX    // DX = &table[k], advances
	MOVQ span+72(FP), R8     // R8 = span (elements per row)

	// Vector bound in bytes: (span &^ 3) * 8. Nothing to do below 4.
	MOVQ R8, CX              // CX = span
	ANDQ $-4, CX             // CX = span &^ 3 (whole 4-element blocks)
	SHLQ $3, CX              // CX = vector bound in bytes
	TESTQ CX, CX             // any full block at all?
	JZ   stage11_c64_done    // no: caller's Go tail does everything

	LEAQ (SI)(CX*1), BX      // BX = end of the input's vector range

	// Odd multiples of the row stride; the even ones come from the SIB scale.
	SHLQ $3, R8              // R8  = span*8   (1x row stride)
	LEAQ (R8)(R8*2), R9      // R9  = 3x
	LEAQ (R8)(R8*4), R10     // R10 = 5x
	LEAQ (R9)(R8*4), R11     // R11 = 7x
	LEAQ (R10)(R8*4), R12    // R12 = 9x

	// Direction: forward multiplies by -i (negate imag), inverse by +i
	// (negate real). Broadcast the 16-byte mask to both 128-bit halves.
	MOVBLZX inverse+80(FP), AX
	TESTL AX, AX
	JNZ   stage11_c64_inv_mask

	VBROADCASTF128 stage11_negodd<>(SB), Y15
	JMP  stage11_c64_mask_done

stage11_c64_inv_mask:
	VBROADCASTF128 stage11_negeven<>(SB), Y15

stage11_c64_mask_done:

stage11_c64_loop:
	// ---- Pairs: aj = input[j*span+k] * table[j*span+k], folded straight into
	// tk = ak + a(11-k) and uk = ak - a(11-k) so only two a values are ever
	// live at once. Row 0 carries no twiddle (table row 0 is all ones, never
	// read) and is not loaded here at all. ----

	// pair 1: rows 1 and 10
	VMOVUPS (SI)(R8*1), Y7   // Y7 = x = input row 1
	VMOVUPS (DX)(R8*1), Y8   // Y8 = t = table row 1
	VMOVSLDUP Y7, Y0         // Y0 = [x.r, x.r, ...]
	VMOVSHDUP Y7, Y7         // Y7 = [x.i, x.i, ...]
	VSHUFPS $0xB1, Y8, Y8, Y6 // Y6 = [t.i, t.r, ...]
	VMULPS Y7, Y6, Y6        // Y6 = x.i * [t.i, t.r]
	VFMADDSUB231PS Y0, Y8, Y6 // Y6 = x.r*t -/+ Y6 = a1
	VMOVUPS (SI)(R10*2), Y7
	VMOVUPS (DX)(R10*2), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y9
	VMULPS Y7, Y9, Y9
	VFMADDSUB231PS Y0, Y8, Y9 // Y9 = a10
	VADDPS Y9, Y6, Y1        // Y1  = t1
	VSUBPS Y9, Y6, Y10       // Y10 = u1

	// pair 2: rows 2 and 9
	VMOVUPS (SI)(R8*2), Y7
	VMOVUPS (DX)(R8*2), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y6
	VMULPS Y7, Y6, Y6
	VFMADDSUB231PS Y0, Y8, Y6 // Y6 = a2
	VMOVUPS (SI)(R12*1), Y7
	VMOVUPS (DX)(R12*1), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y9
	VMULPS Y7, Y9, Y9
	VFMADDSUB231PS Y0, Y8, Y9 // Y9 = a9
	VADDPS Y9, Y6, Y2        // Y2  = t2
	VSUBPS Y9, Y6, Y11       // Y11 = u2

	// pair 3: rows 3 and 8
	VMOVUPS (SI)(R9*1), Y7
	VMOVUPS (DX)(R9*1), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y6
	VMULPS Y7, Y6, Y6
	VFMADDSUB231PS Y0, Y8, Y6 // Y6 = a3
	VMOVUPS (SI)(R8*8), Y7
	VMOVUPS (DX)(R8*8), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y9
	VMULPS Y7, Y9, Y9
	VFMADDSUB231PS Y0, Y8, Y9 // Y9 = a8
	VADDPS Y9, Y6, Y3        // Y3  = t3
	VSUBPS Y9, Y6, Y12       // Y12 = u3

	// pair 4: rows 4 and 7
	VMOVUPS (SI)(R8*4), Y7
	VMOVUPS (DX)(R8*4), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y6
	VMULPS Y7, Y6, Y6
	VFMADDSUB231PS Y0, Y8, Y6 // Y6 = a4
	VMOVUPS (SI)(R11*1), Y7
	VMOVUPS (DX)(R11*1), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y9
	VMULPS Y7, Y9, Y9
	VFMADDSUB231PS Y0, Y8, Y9 // Y9 = a7
	VADDPS Y9, Y6, Y4        // Y4  = t4
	VSUBPS Y9, Y6, Y13       // Y13 = u4

	// pair 5: rows 5 and 6
	VMOVUPS (SI)(R10*1), Y7
	VMOVUPS (DX)(R10*1), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y6
	VMULPS Y7, Y6, Y6
	VFMADDSUB231PS Y0, Y8, Y6 // Y6 = a5
	VMOVUPS (SI)(R9*2), Y7
	VMOVUPS (DX)(R9*2), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y9
	VMULPS Y7, Y9, Y9
	VFMADDSUB231PS Y0, Y8, Y9 // Y9 = a6
	VADDPS Y9, Y6, Y5        // Y5  = t5
	VSUBPS Y9, Y6, Y14       // Y14 = u5

	// ---- j = 1: rows 1 and 10 ----
	VMOVUPS (SI), Y7                     // Y7 = a0
	VFMADD231PS stage11_c1<>(SB), Y1, Y7  // += c1*t1
	VFMADD231PS stage11_c2<>(SB), Y2, Y7  // += c2*t2
	VFMADD231PS stage11_c3<>(SB), Y3, Y7  // += c3*t3
	VFMADD231PS stage11_c4<>(SB), Y4, Y7  // += c4*t4
	VFMADD231PS stage11_c5<>(SB), Y5, Y7  // Y7 = m1
	VMULPS stage11_s1<>(SB), Y10, Y8      // Y8 = s1*u1
	VFMADD231PS stage11_s2<>(SB), Y11, Y8 // += s2*u2
	VFMADD231PS stage11_s3<>(SB), Y12, Y8 // += s3*u3
	VFMADD231PS stage11_s4<>(SB), Y13, Y8 // += s4*u4
	VFMADD231PS stage11_s5<>(SB), Y14, Y8 // += s5*u5
	VSHUFPS $0xB1, Y8, Y8, Y8             // swap re/im
	VXORPS Y15, Y8, Y8                    // Y8 = q1
	VADDPS Y8, Y7, Y0                     // y1  = m1 + q1
	VMOVUPS Y0, (DI)(R8*1)
	VSUBPS Y8, Y7, Y0                     // y10 = m1 - q1
	VMOVUPS Y0, (DI)(R10*2)

	// ---- j = 2: rows 2 and 9 ----
	VMOVUPS (SI), Y7
	VFMADD231PS stage11_c2<>(SB), Y1, Y7
	VFMADD231PS stage11_c4<>(SB), Y2, Y7
	VFMADD231PS stage11_c5<>(SB), Y3, Y7
	VFMADD231PS stage11_c3<>(SB), Y4, Y7
	VFMADD231PS stage11_c1<>(SB), Y5, Y7   // Y7 = m2
	VMULPS stage11_s2<>(SB), Y10, Y8       // Y8 = s2*u1
	VFMADD231PS stage11_s4<>(SB), Y11, Y8  // += s4*u2
	VFNMADD231PS stage11_s5<>(SB), Y12, Y8 // -= s5*u3
	VFNMADD231PS stage11_s3<>(SB), Y13, Y8 // -= s3*u4
	VFNMADD231PS stage11_s1<>(SB), Y14, Y8 // -= s1*u5
	VSHUFPS $0xB1, Y8, Y8, Y8
	VXORPS Y15, Y8, Y8                     // Y8 = q2
	VADDPS Y8, Y7, Y0                      // y2 = m2 + q2
	VMOVUPS Y0, (DI)(R8*2)
	VSUBPS Y8, Y7, Y0                      // y9 = m2 - q2
	VMOVUPS Y0, (DI)(R12*1)

	// ---- j = 3: rows 3 and 8 ----
	VMOVUPS (SI), Y7
	VFMADD231PS stage11_c3<>(SB), Y1, Y7
	VFMADD231PS stage11_c5<>(SB), Y2, Y7
	VFMADD231PS stage11_c2<>(SB), Y3, Y7
	VFMADD231PS stage11_c1<>(SB), Y4, Y7
	VFMADD231PS stage11_c4<>(SB), Y5, Y7   // Y7 = m3
	VMULPS stage11_s3<>(SB), Y10, Y8       // Y8 = s3*u1
	VFNMADD231PS stage11_s5<>(SB), Y11, Y8 // -= s5*u2
	VFNMADD231PS stage11_s2<>(SB), Y12, Y8 // -= s2*u3
	VFMADD231PS stage11_s1<>(SB), Y13, Y8  // += s1*u4
	VFMADD231PS stage11_s4<>(SB), Y14, Y8  // += s4*u5
	VSHUFPS $0xB1, Y8, Y8, Y8
	VXORPS Y15, Y8, Y8                     // Y8 = q3
	VADDPS Y8, Y7, Y0                      // y3 = m3 + q3
	VMOVUPS Y0, (DI)(R9*1)
	VSUBPS Y8, Y7, Y0                      // y8 = m3 - q3
	VMOVUPS Y0, (DI)(R8*8)

	// ---- j = 4: rows 4 and 7 ----
	VMOVUPS (SI), Y7
	VFMADD231PS stage11_c4<>(SB), Y1, Y7
	VFMADD231PS stage11_c3<>(SB), Y2, Y7
	VFMADD231PS stage11_c1<>(SB), Y3, Y7
	VFMADD231PS stage11_c5<>(SB), Y4, Y7
	VFMADD231PS stage11_c2<>(SB), Y5, Y7   // Y7 = m4
	VMULPS stage11_s4<>(SB), Y10, Y8       // Y8 = s4*u1
	VFNMADD231PS stage11_s3<>(SB), Y11, Y8 // -= s3*u2
	VFMADD231PS stage11_s1<>(SB), Y12, Y8  // += s1*u3
	VFMADD231PS stage11_s5<>(SB), Y13, Y8  // += s5*u4
	VFNMADD231PS stage11_s2<>(SB), Y14, Y8 // -= s2*u5
	VSHUFPS $0xB1, Y8, Y8, Y8
	VXORPS Y15, Y8, Y8                     // Y8 = q4
	VADDPS Y8, Y7, Y0                      // y4 = m4 + q4
	VMOVUPS Y0, (DI)(R8*4)
	VSUBPS Y8, Y7, Y0                      // y7 = m4 - q4
	VMOVUPS Y0, (DI)(R11*1)

	// ---- j = 5: rows 5 and 6 ----
	VMOVUPS (SI), Y7
	VFMADD231PS stage11_c5<>(SB), Y1, Y7
	VFMADD231PS stage11_c1<>(SB), Y2, Y7
	VFMADD231PS stage11_c4<>(SB), Y3, Y7
	VFMADD231PS stage11_c2<>(SB), Y4, Y7
	VFMADD231PS stage11_c3<>(SB), Y5, Y7   // Y7 = m5
	VMULPS stage11_s5<>(SB), Y10, Y8       // Y8 = s5*u1
	VFNMADD231PS stage11_s1<>(SB), Y11, Y8 // -= s1*u2
	VFMADD231PS stage11_s4<>(SB), Y12, Y8  // += s4*u3
	VFNMADD231PS stage11_s2<>(SB), Y13, Y8 // -= s2*u4
	VFMADD231PS stage11_s3<>(SB), Y14, Y8  // += s3*u5
	VSHUFPS $0xB1, Y8, Y8, Y8
	VXORPS Y15, Y8, Y8                     // Y8 = q5
	VADDPS Y8, Y7, Y0                      // y5 = m5 + q5
	VMOVUPS Y0, (DI)(R10*1)
	VSUBPS Y8, Y7, Y0                      // y6 = m5 - q5
	VMOVUPS Y0, (DI)(R9*2)

	// ---- y0 = a0 + t1 + ... + t5. Last, because it is the only store that
	// can clobber input row 0 when dst aliases input. ----
	VMOVUPS (SI), Y7         // Y7 = a0
	VADDPS Y1, Y7, Y7
	VADDPS Y2, Y7, Y7
	VADDPS Y3, Y7, Y7
	VADDPS Y4, Y7, Y7
	VADDPS Y5, Y7, Y7        // Y7 = y0
	VMOVUPS Y7, (DI)

	ADDQ $32, SI             // advance one 4-element block
	ADDQ $32, DX
	ADDQ $32, DI
	CMPQ SI, BX
	JL   stage11_c64_loop

	VZEROUPPER

stage11_c64_done:
	RET
