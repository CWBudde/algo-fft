//go:build amd64 && !purego

// ===========================================================================
// AVX2 fused radix-7 mixed-radix stage (complex64) for AMD64
// ===========================================================================
//
// The radix-7 sibling of avx2_f32_mixedradix_stage5.s. See that file for the
// stage contract, the layout of `table`, and why k is the vector axis; only
// the butterfly and the register budget differ.
//
// ===========================================================================
//
// RADIX-7 BUTTERFLY
// -----------------
// With w = exp(-2*pi*i/7), ck = cos(2*pi*k/7) and sk = sin(2*pi*k/7), and
// using w6 = conj(w1), w5 = conj(w2), w4 = conj(w3), the direct form
// collapses to three conjugate output pairs:
//
//	t1 = a1 + a6    u1 = a1 - a6
//	t2 = a2 + a5    u2 = a2 - a5
//	t3 = a3 + a4    u3 = a3 - a4
//	y0 = a0 + t1 + t2 + t3
//	m1 = a0 + c1*t1 + c2*t2 + c3*t3   q1 = -i * ( s1*u1 + s2*u2 + s3*u3)
//	m2 = a0 + c2*t1 + c3*t2 + c1*t3   q2 = -i * ( s2*u1 - s3*u2 - s1*u3)
//	m3 = a0 + c3*t1 + c1*t2 + c2*t3   q3 = -i * ( s3*u1 - s1*u2 + s2*u3)
//	y1 = m1 + q1   y6 = m1 - q1
//	y2 = m2 + q2   y5 = m2 - q2
//	y3 = m3 + q3   y4 = m3 - q3
//
// The cosine rows are the index map c[j*m mod 7] written out; the sine rows
// are the same map with the folding s[7-k] = -s[k] already applied. As in the
// radix-5 kernel the inverse stage replaces every -i with +i, which is the
// same pair swap against a different XOR mask.
//
// ===========================================================================
//
// REGISTER BUDGET
// ---------------
// Six constants plus the sign mask occupy Y9..Y15, leaving Y0..Y8 for the
// nine live values a1..a6 (later t1..t3, u1..u3) plus three scratch. That is
// one short of also holding a0, so a0 stays in memory and is re-read for each
// of m1..m3 and for y0 — four L1 hits against a line the row-0 load already
// pulled in.
//
// Keeping a0 in memory is what forces the store order: dst row 0 is written
// last, after the final read of input row 0. Rows 1..6 of dst are written
// only once every input row has been consumed into a register, so the
// documented dst == input aliasing still holds.
//
// ===========================================================================

#include "textflag.h"

// Sign masks for the multiply-by-i step. Selected by the `inverse` argument.
DATA stage7_negodd<>+0x00(SB)/4, $0x00000000
DATA stage7_negodd<>+0x04(SB)/4, $0x80000000
DATA stage7_negodd<>+0x08(SB)/4, $0x00000000
DATA stage7_negodd<>+0x0C(SB)/4, $0x80000000
GLOBL stage7_negodd<>(SB), RODATA|NOPTR, $16

DATA stage7_negeven<>+0x00(SB)/4, $0x80000000
DATA stage7_negeven<>+0x04(SB)/4, $0x00000000
DATA stage7_negeven<>+0x08(SB)/4, $0x80000000
DATA stage7_negeven<>+0x0C(SB)/4, $0x00000000
GLOBL stage7_negeven<>(SB), RODATA|NOPTR, $16

// ===========================================================================
// func MixedRadixStage7Complex64AVX2Asm(dst, input, table []complex64, span int, inverse bool)
//
// Stack frame (offsets from FP):
//   dst:     0(FP) ptr,   8(FP) len,  16(FP) cap
//   input:  24(FP) ptr,  32(FP) len,  40(FP) cap
//   table:  48(FP) ptr,  56(FP) len,  64(FP) cap
//   span:   72(FP)
//   inverse:80(FP)
//
// Seven row offsets would not fit alongside three block bases, so the three
// base pointers advance with the loop instead of being re-derived from a byte
// offset; BX holds the end of the input's vector range.
// ===========================================================================
TEXT ·MixedRadixStage7Complex64AVX2Asm(SB), NOSPLIT, $0-81
	MOVQ dst+0(FP), DI       // DI = &dst[k],   advances
	MOVQ input+24(FP), SI    // SI = &input[k], advances
	MOVQ table+48(FP), DX    // DX = &table[k], advances
	MOVQ span+72(FP), R8     // R8 = span (elements per row)

	// Vector bound in bytes: (span &^ 3) * 8. Nothing to do below 4.
	MOVQ R8, CX              // CX = span
	ANDQ $-4, CX             // CX = span &^ 3 (whole 4-element blocks)
	SHLQ $3, CX              // CX = vector bound in bytes
	TESTQ CX, CX             // any full block at all?
	JZ   stage7_c64_done     // no: caller's Go tail does everything

	LEAQ (SI)(CX*1), BX      // BX = end of the input's vector range

	// Row stride in bytes and its multiples, used as index registers so each
	// row load is a single addressed move off the block base.
	SHLQ $3, R8              // R8 = span*8   (row 1 offset)
	MOVQ R8, R9
	ADDQ R8, R9              // R9 = span*16  (row 2 offset)
	MOVQ R9, R10
	ADDQ R8, R10             // R10 = span*24 (row 3 offset)
	MOVQ R10, R11
	ADDQ R8, R11             // R11 = span*32 (row 4 offset)
	MOVQ R11, R12
	ADDQ R8, R12             // R12 = span*40 (row 5 offset)
	MOVQ R12, R13
	ADDQ R8, R13             // R13 = span*48 (row 6 offset)

	// Butterfly constants, broadcast once.
	MOVL $0x3F1F9D07, AX     // c1 =  0.62348980 (cos 2pi/7)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y9
	MOVL $0xBE63DC87, AX     // c2 = -0.22252093 (cos 4pi/7)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y10
	MOVL $0xBF66A5E5, AX     // c3 = -0.90096887 (cos 6pi/7)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y11
	MOVL $0x3F48261C, AX     // s1 =  0.78183150 (sin 2pi/7)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y12
	MOVL $0x3F7994E0, AX     // s2 =  0.97492790 (sin 4pi/7)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y13
	MOVL $0x3EDE2602, AX     // s3 =  0.43388373 (sin 6pi/7)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y14

	// Direction: forward multiplies by -i (negate imag), inverse by +i
	// (negate real). Broadcast the 16-byte mask to both 128-bit halves.
	MOVBLZX inverse+80(FP), AX
	TESTL AX, AX
	JNZ   stage7_c64_inv_mask

	VBROADCASTF128 stage7_negodd<>(SB), Y15
	JMP  stage7_c64_mask_done

stage7_c64_inv_mask:
	VBROADCASTF128 stage7_negeven<>(SB), Y15

stage7_c64_mask_done:

stage7_c64_loop:
	// ---- Rows 1..6: aj = input[j*span+k] * table[j*span+k]. ----
	// Complex multiply with one register fewer than the radix-5 kernel uses:
	// the destination doubles as the third scratch, so only Y0, Y7 and Y8 are
	// consumed. Row 0 carries no twiddle (table row 0 is all ones, never
	// read) and is not loaded here at all.
	VMOVUPS (SI)(R8*1), Y7   // Y7 = x = input row 1
	VMOVUPS (DX)(R8*1), Y8   // Y8 = t = table row 1
	VMOVSLDUP Y7, Y0         // Y0 = [x.r, x.r, ...]
	VMOVSHDUP Y7, Y7         // Y7 = [x.i, x.i, ...]
	VSHUFPS $0xB1, Y8, Y8, Y1 // Y1 = [t.i, t.r, ...]
	VMULPS Y7, Y1, Y1        // Y1 = x.i * [t.i, t.r]
	VFMADDSUB231PS Y0, Y8, Y1 // Y1 = x.r*t -/+ Y1 = a1

	VMOVUPS (SI)(R9*1), Y7
	VMOVUPS (DX)(R9*1), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y2
	VMULPS Y7, Y2, Y2
	VFMADDSUB231PS Y0, Y8, Y2 // Y2 = a2

	VMOVUPS (SI)(R10*1), Y7
	VMOVUPS (DX)(R10*1), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y3
	VMULPS Y7, Y3, Y3
	VFMADDSUB231PS Y0, Y8, Y3 // Y3 = a3

	VMOVUPS (SI)(R11*1), Y7
	VMOVUPS (DX)(R11*1), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y4
	VMULPS Y7, Y4, Y4
	VFMADDSUB231PS Y0, Y8, Y4 // Y4 = a4

	VMOVUPS (SI)(R12*1), Y7
	VMOVUPS (DX)(R12*1), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y5
	VMULPS Y7, Y5, Y5
	VFMADDSUB231PS Y0, Y8, Y5 // Y5 = a5

	VMOVUPS (SI)(R13*1), Y7
	VMOVUPS (DX)(R13*1), Y8
	VMOVSLDUP Y7, Y0
	VMOVSHDUP Y7, Y7
	VSHUFPS $0xB1, Y8, Y8, Y6
	VMULPS Y7, Y6, Y6
	VFMADDSUB231PS Y0, Y8, Y6 // Y6 = a6

	// ---- Conjugate pairs. Each sum lands on the lower row's register and
	// each difference on the upper one's, so no value needs a home of its
	// own: Y1..Y3 = t1..t3, Y6, Y5, Y4 = u1, u2, u3. ----
	VADDPS Y6, Y1, Y0        // Y0 = t1 = a1 + a6
	VSUBPS Y6, Y1, Y6        // Y6 = u1 = a1 - a6
	VMOVAPS Y0, Y1           // Y1 = t1

	VADDPS Y5, Y2, Y0        // Y0 = t2 = a2 + a5
	VSUBPS Y5, Y2, Y5        // Y5 = u2 = a2 - a5
	VMOVAPS Y0, Y2           // Y2 = t2

	VADDPS Y4, Y3, Y0        // Y0 = t3 = a3 + a4
	VSUBPS Y4, Y3, Y4        // Y4 = u3 = a3 - a4
	VMOVAPS Y0, Y3           // Y3 = t3

	// ---- m1 = a0 + c1*t1 + c2*t2 + c3*t3, q1 = -/+i*(s1*u1+s2*u2+s3*u3) ----
	VMOVUPS (SI), Y7         // Y7 = a0
	VFMADD231PS Y9, Y1, Y7   // += c1*t1
	VFMADD231PS Y10, Y2, Y7  // += c2*t2
	VFMADD231PS Y11, Y3, Y7  // Y7 = m1
	VMULPS Y12, Y6, Y8       // Y8 = s1*u1
	VFMADD231PS Y13, Y5, Y8  // += s2*u2
	VFMADD231PS Y14, Y4, Y8  // += s3*u3
	VSHUFPS $0xB1, Y8, Y8, Y8 // swap re/im
	VXORPS Y15, Y8, Y8       // Y8 = q1
	VADDPS Y8, Y7, Y0        // y1 = m1 + q1
	VMOVUPS Y0, (DI)(R8*1)
	VSUBPS Y8, Y7, Y0        // y6 = m1 - q1
	VMOVUPS Y0, (DI)(R13*1)

	// ---- m2 = a0 + c2*t1 + c3*t2 + c1*t3, q2 = -/+i*(s2*u1-s3*u2-s1*u3) ----
	VMOVUPS (SI), Y7         // Y7 = a0
	VFMADD231PS Y10, Y1, Y7  // += c2*t1
	VFMADD231PS Y11, Y2, Y7  // += c3*t2
	VFMADD231PS Y9, Y3, Y7   // Y7 = m2
	VMULPS Y13, Y6, Y8       // Y8 = s2*u1
	VFNMADD231PS Y14, Y5, Y8 // -= s3*u2
	VFNMADD231PS Y12, Y4, Y8 // -= s1*u3
	VSHUFPS $0xB1, Y8, Y8, Y8
	VXORPS Y15, Y8, Y8       // Y8 = q2
	VADDPS Y8, Y7, Y0        // y2 = m2 + q2
	VMOVUPS Y0, (DI)(R9*1)
	VSUBPS Y8, Y7, Y0        // y5 = m2 - q2
	VMOVUPS Y0, (DI)(R12*1)

	// ---- m3 = a0 + c3*t1 + c1*t2 + c2*t3, q3 = -/+i*(s3*u1-s1*u2+s2*u3) ----
	VMOVUPS (SI), Y7         // Y7 = a0
	VFMADD231PS Y11, Y1, Y7  // += c3*t1
	VFMADD231PS Y9, Y2, Y7   // += c1*t2
	VFMADD231PS Y10, Y3, Y7  // Y7 = m3
	VMULPS Y14, Y6, Y8       // Y8 = s3*u1
	VFNMADD231PS Y12, Y5, Y8 // -= s1*u2
	VFMADD231PS Y13, Y4, Y8  // += s2*u3
	VSHUFPS $0xB1, Y8, Y8, Y8
	VXORPS Y15, Y8, Y8       // Y8 = q3
	VADDPS Y8, Y7, Y0        // y3 = m3 + q3
	VMOVUPS Y0, (DI)(R10*1)
	VSUBPS Y8, Y7, Y0        // y4 = m3 - q3
	VMOVUPS Y0, (DI)(R11*1)

	// ---- y0 = a0 + t1 + t2 + t3. Last, because it is the only store that
	// can clobber input row 0 when dst aliases input. ----
	VMOVUPS (SI), Y7         // Y7 = a0
	VADDPS Y1, Y7, Y7
	VADDPS Y2, Y7, Y7
	VADDPS Y3, Y7, Y7        // Y7 = y0
	VMOVUPS Y7, (DI)

	ADDQ $32, SI             // advance one 4-element block
	ADDQ $32, DX
	ADDQ $32, DI
	CMPQ SI, BX
	JL   stage7_c64_loop

	VZEROUPPER

stage7_c64_done:
	RET
