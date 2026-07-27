//go:build amd64 && !purego

// ===========================================================================
// AVX2 fused radix-5 mixed-radix stage (complex64) for AMD64
// ===========================================================================
//
// One stage of the mixed-radix recursion, twiddle multiply and butterfly
// fused into a single pass over the data:
//
//	for k := range span {
//	    a0 := input[k]                                  // row 0 twiddle is 1
//	    aj := input[j*span+k] * table[j*span+k]         // j = 1..4
//	    y0..y4 := butterfly5(a0..a4)
//	    dst[j*span+k] = yj                              // j = 0..4
//	}
//
// The Go stage it replaces runs the same arithmetic as two passes: an
// in-place array multiply over rows 1..4 and then a twiddle-free butterfly
// loop. Fusing them halves the traffic — rows 1..4 are read once instead of
// read/written/read — and keeps a0..a4 in registers across the butterfly.
//
// The k index is the vector axis. Every lane of a YMM is a different k
// running the *same* butterfly, so the butterfly needs no cross-lane data
// movement at all: the only shuffles in the loop are the ones inside the
// complex multiply and the two multiply-by-i steps.
//
// Layout: complex64 [re, im] pairs, 4 complex per YMM (32 bytes).
// Rows are span elements apart; `table` is laid out exactly like the data
// (entry j*span+k holds W_n^(j*k)), which is what makes both operands of
// the twiddle multiply contiguous.
//
// Only floor(span/4)*4 elements are processed. The caller handles the
// 0-3 element tail in Go; see mixedradix_stage_asm_amd64.go.
//
// ===========================================================================
//
// RADIX-5 BUTTERFLY
// -----------------
// With w = exp(-2*pi*i/5), c1 = cos(2*pi/5), s1 = sin(2*pi/5),
// c2 = cos(4*pi/5), s2 = sin(4*pi/5), and using w4 = conj(w1),
// w3 = conj(w2), the direct form collapses to:
//
//	t1 = a1 + a4    t2 = a2 + a3
//	t3 = a1 - a4    t4 = a2 - a3
//	y0 = a0 + t1 + t2
//	m1 = a0 + c1*t1 + c2*t2      q1 = -i * (s1*t3 + s2*t4)
//	m2 = a0 + c2*t1 + c1*t2      q2 = -i * (s2*t3 - s1*t4)
//	y1 = m1 + q1   y4 = m1 - q1
//	y2 = m2 + q2   y3 = m2 - q2
//
// The inverse stage conjugates every w, which flips the sign of s1 and s2
// and so replaces both -i factors with +i. Multiplying a packed [re, im]
// pair by -i is a pair swap followed by negating the imaginary lane; by +i
// it is the same swap followed by negating the real lane. Direction
// therefore costs exactly one register: the XOR mask picked at entry.
//
// ===========================================================================

#include "textflag.h"

// Sign masks for the multiply-by-i step. Selected by the `inverse` argument.
DATA stage5_negodd<>+0x00(SB)/4, $0x00000000
DATA stage5_negodd<>+0x04(SB)/4, $0x80000000
DATA stage5_negodd<>+0x08(SB)/4, $0x00000000
DATA stage5_negodd<>+0x0C(SB)/4, $0x80000000
GLOBL stage5_negodd<>(SB), RODATA|NOPTR, $16

DATA stage5_negeven<>+0x00(SB)/4, $0x80000000
DATA stage5_negeven<>+0x04(SB)/4, $0x00000000
DATA stage5_negeven<>+0x08(SB)/4, $0x80000000
DATA stage5_negeven<>+0x0C(SB)/4, $0x00000000
GLOBL stage5_negeven<>(SB), RODATA|NOPTR, $16

// ===========================================================================
// func MixedRadixStage5Complex64AVX2Asm(dst, input, table []complex64, span int, inverse bool)
//
// Stack frame (offsets from FP):
//   dst:     0(FP) ptr,   8(FP) len,  16(FP) cap
//   input:  24(FP) ptr,  32(FP) len,  40(FP) cap
//   table:  48(FP) ptr,  56(FP) len,  64(FP) cap
//   span:   72(FP)
//   inverse:80(FP)
// ===========================================================================
TEXT ·MixedRadixStage5Complex64AVX2Asm(SB), NOSPLIT, $0-81
	MOVQ dst+0(FP), DI       // DI = dst base
	MOVQ input+24(FP), SI    // SI = input base
	MOVQ table+48(FP), DX    // DX = table base
	MOVQ span+72(FP), R8     // R8 = span (elements per row)

	// Vector bound in bytes: (span &^ 3) * 8. Nothing to do below 4.
	MOVQ R8, CX              // CX = span
	ANDQ $-4, CX             // CX = span &^ 3 (whole 4-element blocks)
	SHLQ $3, CX              // CX = vector bound in bytes
	TESTQ CX, CX             // any full block at all?
	JZ   stage5_c64_done     // no: caller's Go tail does everything

	// Row stride in bytes and its multiples, used as index registers so each
	// row load is a single addressed move off the block base.
	SHLQ $3, R8              // R8 = span*8   (row 1 offset)
	MOVQ R8, R9
	ADDQ R8, R9              // R9 = span*16  (row 2 offset)
	MOVQ R9, R10
	ADDQ R8, R10             // R10 = span*24 (row 3 offset)
	MOVQ R10, R11
	ADDQ R8, R11             // R11 = span*32 (row 4 offset)

	// Butterfly constants, broadcast once.
	MOVL $0x3E9E377A, AX     // c1 =  0.30901699 (cos 2pi/5)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y11
	MOVL $0xBF4F1BBD, AX     // c2 = -0.80901699 (cos 4pi/5)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y12
	MOVL $0x3F737871, AX     // s1 =  0.95105654 (sin 2pi/5)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y13
	MOVL $0x3F167918, AX     // s2 =  0.58778524 (sin 4pi/5)
	VMOVQ AX, X0
	VBROADCASTSS X0, Y14

	// Direction: forward multiplies by -i (negate imag), inverse by +i
	// (negate real). Broadcast the 16-byte mask to both 128-bit halves.
	MOVBLZX inverse+80(FP), AX
	TESTL AX, AX
	JNZ   stage5_c64_inv_mask

	VBROADCASTF128 stage5_negodd<>(SB), Y15
	JMP  stage5_c64_mask_done

stage5_c64_inv_mask:
	VBROADCASTF128 stage5_negeven<>(SB), Y15

stage5_c64_mask_done:
	XORQ AX, AX              // AX = byte offset of the current k block

stage5_c64_loop:
	LEAQ (SI)(AX*1), R12     // R12 = &input[k]
	LEAQ (DX)(AX*1), R13     // R13 = &table[k]
	LEAQ (DI)(AX*1), R14     // R14 = &dst[k]

	// ---- Row 0: no twiddle (table row 0 is all ones, never read). ----
	VMOVUPS (R12), Y0        // Y0 = a0

	// ---- Rows 1..4: aj = input[j*span+k] * table[j*span+k]. ----
	// Complex multiply, per the avx2_f32_complex_mul.s idiom:
	//   Y5 = dup real of x, Y6 = dup imag of x, Y7 = swapped t
	//   result = x.r*t -/+ x.i*t_swapped   (VFMADDSUB231PS)
	VMOVUPS (R12)(R8*1), Y1  // Y1 = input row 1
	VMOVUPS (R13)(R8*1), Y8  // Y8 = table row 1
	VMOVSLDUP Y1, Y5         // Y5 = [x.r, x.r, ...]
	VMOVSHDUP Y1, Y6         // Y6 = [x.i, x.i, ...]
	VSHUFPS $0xB1, Y8, Y8, Y7 // Y7 = [t.i, t.r, ...]
	VMULPS Y6, Y7, Y7        // Y7 = x.i * [t.i, t.r]
	VFMADDSUB231PS Y5, Y8, Y7 // Y7 = x.r*t -/+ Y7 = x*t
	VMOVAPS Y7, Y1           // Y1 = a1

	VMOVUPS (R12)(R9*1), Y2  // Y2 = input row 2
	VMOVUPS (R13)(R9*1), Y8  // Y8 = table row 2
	VMOVSLDUP Y2, Y5
	VMOVSHDUP Y2, Y6
	VSHUFPS $0xB1, Y8, Y8, Y7
	VMULPS Y6, Y7, Y7
	VFMADDSUB231PS Y5, Y8, Y7
	VMOVAPS Y7, Y2           // Y2 = a2

	VMOVUPS (R12)(R10*1), Y3 // Y3 = input row 3
	VMOVUPS (R13)(R10*1), Y8 // Y8 = table row 3
	VMOVSLDUP Y3, Y5
	VMOVSHDUP Y3, Y6
	VSHUFPS $0xB1, Y8, Y8, Y7
	VMULPS Y6, Y7, Y7
	VFMADDSUB231PS Y5, Y8, Y7
	VMOVAPS Y7, Y3           // Y3 = a3

	VMOVUPS (R12)(R11*1), Y4 // Y4 = input row 4
	VMOVUPS (R13)(R11*1), Y8 // Y8 = table row 4
	VMOVSLDUP Y4, Y5
	VMOVSHDUP Y4, Y6
	VSHUFPS $0xB1, Y8, Y8, Y7
	VMULPS Y6, Y7, Y7
	VFMADDSUB231PS Y5, Y8, Y7
	VMOVAPS Y7, Y4           // Y4 = a4

	// ---- Butterfly. Y0..Y4 = a0..a4, all lanes independent in k. ----
	VADDPS Y4, Y1, Y5        // Y5 = t1 = a1 + a4
	VADDPS Y3, Y2, Y6        // Y6 = t2 = a2 + a3
	VSUBPS Y4, Y1, Y7        // Y7 = t3 = a1 - a4
	VSUBPS Y3, Y2, Y8        // Y8 = t4 = a2 - a3
	                         // Y1..Y4 are now free.

	// y0 = a0 + t1 + t2
	VADDPS Y6, Y5, Y1        // Y1 = t1 + t2
	VADDPS Y0, Y1, Y1        // Y1 = y0
	VMOVUPS Y1, (R14)        // dst[k] = y0

	// m1 = a0 + c1*t1 + c2*t2
	VMOVAPS Y0, Y1
	VFMADD231PS Y11, Y5, Y1  // Y1 += c1*t1
	VFMADD231PS Y12, Y6, Y1  // Y1 = m1

	// m2 = a0 + c2*t1 + c1*t2
	VMOVAPS Y0, Y2
	VFMADD231PS Y12, Y5, Y2  // Y2 += c2*t1
	VFMADD231PS Y11, Y6, Y2  // Y2 = m2

	// q1 = -/+i * (s1*t3 + s2*t4)
	VMULPS Y13, Y7, Y3       // Y3 = s1*t3
	VFMADD231PS Y14, Y8, Y3  // Y3 += s2*t4
	VSHUFPS $0xB1, Y3, Y3, Y3 // swap re/im
	VXORPS Y15, Y3, Y3       // Y3 = q1

	// q2 = -/+i * (s2*t3 - s1*t4)
	VMULPS Y14, Y7, Y4       // Y4 = s2*t3
	VFNMADD231PS Y13, Y8, Y4 // Y4 -= s1*t4
	VSHUFPS $0xB1, Y4, Y4, Y4 // swap re/im
	VXORPS Y15, Y4, Y4       // Y4 = q2

	VADDPS Y3, Y1, Y5        // y1 = m1 + q1
	VMOVUPS Y5, (R14)(R8*1)
	VADDPS Y4, Y2, Y6        // y2 = m2 + q2
	VMOVUPS Y6, (R14)(R9*1)
	VSUBPS Y4, Y2, Y7        // y3 = m2 - q2
	VMOVUPS Y7, (R14)(R10*1)
	VSUBPS Y3, Y1, Y8        // y4 = m1 - q1
	VMOVUPS Y8, (R14)(R11*1)

	ADDQ $32, AX             // advance one 4-element block
	CMPQ AX, CX
	JL   stage5_c64_loop

	VZEROUPPER

stage5_c64_done:
	RET
