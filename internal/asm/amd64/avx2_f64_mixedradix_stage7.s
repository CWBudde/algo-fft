//go:build amd64 && !purego

// ===========================================================================
// AVX2 fused radix-7 mixed-radix stage (complex128) for AMD64
// ===========================================================================
//
// The complex128 twin of avx2_f32_mixedradix_stage7.s. See that file for the
// radix-7 butterfly derivation, the register budget and the store order that
// keeps dst == input aliasing safe; only the packing differs.
//
// Layout: complex128 [re, im] pairs, 2 complex per YMM (32 bytes). The
// complex multiply uses the f64 idiom of avx2_f64_complex_mul.s: VMOVDDUP
// for the real broadcast, VPERMILPD $0x0F for the imaginary broadcast and
// VPERMILPD $0x05 for the pair swap.
//
// Only floor(span/2)*2 elements are processed; the caller handles the tail.
//
// ===========================================================================

#include "textflag.h"

// Sign masks for the multiply-by-i step (one 128-bit lane, broadcast).
DATA stage7d_negodd<>+0x00(SB)/8, $0x0000000000000000
DATA stage7d_negodd<>+0x08(SB)/8, $0x8000000000000000
GLOBL stage7d_negodd<>(SB), RODATA|NOPTR, $16

DATA stage7d_negeven<>+0x00(SB)/8, $0x8000000000000000
DATA stage7d_negeven<>+0x08(SB)/8, $0x0000000000000000
GLOBL stage7d_negeven<>(SB), RODATA|NOPTR, $16

// ===========================================================================
// func MixedRadixStage7Complex128AVX2Asm(dst, input, table []complex128, span int, inverse bool)
// ===========================================================================
TEXT ·MixedRadixStage7Complex128AVX2Asm(SB), NOSPLIT, $0-81
	MOVQ dst+0(FP), DI       // DI = &dst[k],   advances
	MOVQ input+24(FP), SI    // SI = &input[k], advances
	MOVQ table+48(FP), DX    // DX = &table[k], advances
	MOVQ span+72(FP), R8     // R8 = span

	MOVQ R8, CX
	ANDQ $-2, CX             // whole 2-element blocks only
	SHLQ $4, CX              // CX = vector bound in bytes
	TESTQ CX, CX
	JZ   stage7_c128_done

	LEAQ (SI)(CX*1), BX      // BX = end of the input's vector range

	SHLQ $4, R8              // R8 = span*16  (row 1 offset)
	MOVQ R8, R9
	ADDQ R8, R9              // R9 = span*32  (row 2 offset)
	MOVQ R9, R10
	ADDQ R8, R10             // R10 = span*48 (row 3 offset)
	MOVQ R10, R11
	ADDQ R8, R11             // R11 = span*64 (row 4 offset)
	MOVQ R11, R12
	ADDQ R8, R12             // R12 = span*80 (row 5 offset)
	MOVQ R12, R13
	ADDQ R8, R13             // R13 = span*96 (row 6 offset)

	MOVQ $0x3FE3F3A0E28BEDD2, AX // c1 =  0.62348980185873359
	VMOVQ AX, X0
	VBROADCASTSD X0, Y9
	MOVQ $0xBFCC7B90E3024580, AX // c2 = -0.22252093395631434
	VMOVQ AX, X0
	VBROADCASTSD X0, Y10
	MOVQ $0xBFECD4BCA9CB5C70, AX // c3 = -0.90096886790241903
	VMOVQ AX, X0
	VBROADCASTSD X0, Y11
	MOVQ $0x3FE904C37505DE4B, AX // s1 =  0.78183148246802980
	VMOVQ AX, X0
	VBROADCASTSD X0, Y12
	MOVQ $0x3FEF329C0558E96A, AX // s2 =  0.97492791218182373
	VMOVQ AX, X0
	VBROADCASTSD X0, Y13
	MOVQ $0x3FDBC4C04D71ABC3, AX // s3 =  0.43388373911755823
	VMOVQ AX, X0
	VBROADCASTSD X0, Y14

	MOVBLZX inverse+80(FP), AX
	TESTL AX, AX
	JNZ   stage7_c128_inv_mask

	VBROADCASTF128 stage7d_negodd<>(SB), Y15
	JMP  stage7_c128_mask_done

stage7_c128_inv_mask:
	VBROADCASTF128 stage7d_negeven<>(SB), Y15

stage7_c128_mask_done:

stage7_c128_loop:
	// ---- Rows 1..6: aj = input[j*span+k] * table[j*span+k]. ----
	VMOVUPD (SI)(R8*1), Y7   // Y7 = x = input row 1
	VMOVUPD (DX)(R8*1), Y8   // Y8 = t = table row 1
	VMOVDDUP Y7, Y0          // Y0 = [x.r, x.r, ...]
	VPERMILPD $0x0F, Y7, Y7  // Y7 = [x.i, x.i, ...]
	VPERMILPD $0x05, Y8, Y1  // Y1 = [t.i, t.r, ...]
	VMULPD Y7, Y1, Y1
	VFMADDSUB231PD Y0, Y8, Y1 // Y1 = a1

	VMOVUPD (SI)(R9*1), Y7
	VMOVUPD (DX)(R9*1), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y2
	VMULPD Y7, Y2, Y2
	VFMADDSUB231PD Y0, Y8, Y2 // Y2 = a2

	VMOVUPD (SI)(R10*1), Y7
	VMOVUPD (DX)(R10*1), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y3
	VMULPD Y7, Y3, Y3
	VFMADDSUB231PD Y0, Y8, Y3 // Y3 = a3

	VMOVUPD (SI)(R11*1), Y7
	VMOVUPD (DX)(R11*1), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y4
	VMULPD Y7, Y4, Y4
	VFMADDSUB231PD Y0, Y8, Y4 // Y4 = a4

	VMOVUPD (SI)(R12*1), Y7
	VMOVUPD (DX)(R12*1), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y5
	VMULPD Y7, Y5, Y5
	VFMADDSUB231PD Y0, Y8, Y5 // Y5 = a5

	VMOVUPD (SI)(R13*1), Y7
	VMOVUPD (DX)(R13*1), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y6
	VMULPD Y7, Y6, Y6
	VFMADDSUB231PD Y0, Y8, Y6 // Y6 = a6

	// ---- Conjugate pairs: Y1..Y3 = t1..t3, Y6, Y5, Y4 = u1, u2, u3. ----
	VADDPD Y6, Y1, Y0        // Y0 = t1 = a1 + a6
	VSUBPD Y6, Y1, Y6        // Y6 = u1 = a1 - a6
	VMOVAPD Y0, Y1           // Y1 = t1

	VADDPD Y5, Y2, Y0        // Y0 = t2 = a2 + a5
	VSUBPD Y5, Y2, Y5        // Y5 = u2 = a2 - a5
	VMOVAPD Y0, Y2           // Y2 = t2

	VADDPD Y4, Y3, Y0        // Y0 = t3 = a3 + a4
	VSUBPD Y4, Y3, Y4        // Y4 = u3 = a3 - a4
	VMOVAPD Y0, Y3           // Y3 = t3

	// ---- m1 = a0 + c1*t1 + c2*t2 + c3*t3, q1 = -/+i*(s1*u1+s2*u2+s3*u3) ----
	VMOVUPD (SI), Y7         // Y7 = a0
	VFMADD231PD Y9, Y1, Y7   // += c1*t1
	VFMADD231PD Y10, Y2, Y7  // += c2*t2
	VFMADD231PD Y11, Y3, Y7  // Y7 = m1
	VMULPD Y12, Y6, Y8       // Y8 = s1*u1
	VFMADD231PD Y13, Y5, Y8  // += s2*u2
	VFMADD231PD Y14, Y4, Y8  // += s3*u3
	VPERMILPD $0x05, Y8, Y8  // swap re/im
	VXORPD Y15, Y8, Y8       // Y8 = q1
	VADDPD Y8, Y7, Y0        // y1 = m1 + q1
	VMOVUPD Y0, (DI)(R8*1)
	VSUBPD Y8, Y7, Y0        // y6 = m1 - q1
	VMOVUPD Y0, (DI)(R13*1)

	// ---- m2 = a0 + c2*t1 + c3*t2 + c1*t3, q2 = -/+i*(s2*u1-s3*u2-s1*u3) ----
	VMOVUPD (SI), Y7         // Y7 = a0
	VFMADD231PD Y10, Y1, Y7  // += c2*t1
	VFMADD231PD Y11, Y2, Y7  // += c3*t2
	VFMADD231PD Y9, Y3, Y7   // Y7 = m2
	VMULPD Y13, Y6, Y8       // Y8 = s2*u1
	VFNMADD231PD Y14, Y5, Y8 // -= s3*u2
	VFNMADD231PD Y12, Y4, Y8 // -= s1*u3
	VPERMILPD $0x05, Y8, Y8
	VXORPD Y15, Y8, Y8       // Y8 = q2
	VADDPD Y8, Y7, Y0        // y2 = m2 + q2
	VMOVUPD Y0, (DI)(R9*1)
	VSUBPD Y8, Y7, Y0        // y5 = m2 - q2
	VMOVUPD Y0, (DI)(R12*1)

	// ---- m3 = a0 + c3*t1 + c1*t2 + c2*t3, q3 = -/+i*(s3*u1-s1*u2+s2*u3) ----
	VMOVUPD (SI), Y7         // Y7 = a0
	VFMADD231PD Y11, Y1, Y7  // += c3*t1
	VFMADD231PD Y9, Y2, Y7   // += c1*t2
	VFMADD231PD Y10, Y3, Y7  // Y7 = m3
	VMULPD Y14, Y6, Y8       // Y8 = s3*u1
	VFNMADD231PD Y12, Y5, Y8 // -= s1*u2
	VFMADD231PD Y13, Y4, Y8  // += s2*u3
	VPERMILPD $0x05, Y8, Y8
	VXORPD Y15, Y8, Y8       // Y8 = q3
	VADDPD Y8, Y7, Y0        // y3 = m3 + q3
	VMOVUPD Y0, (DI)(R10*1)
	VSUBPD Y8, Y7, Y0        // y4 = m3 - q3
	VMOVUPD Y0, (DI)(R11*1)

	// ---- y0 = a0 + t1 + t2 + t3, stored last so it cannot clobber input
	// row 0 when dst aliases input. ----
	VMOVUPD (SI), Y7         // Y7 = a0
	VADDPD Y1, Y7, Y7
	VADDPD Y2, Y7, Y7
	VADDPD Y3, Y7, Y7        // Y7 = y0
	VMOVUPD Y7, (DI)

	ADDQ $32, SI             // advance one 2-element block
	ADDQ $32, DX
	ADDQ $32, DI
	CMPQ SI, BX
	JL   stage7_c128_loop

	VZEROUPPER

stage7_c128_done:
	RET
