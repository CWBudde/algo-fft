//go:build amd64 && !purego

// ===========================================================================
// AVX2 fused radix-11 mixed-radix stage (complex128) for AMD64
// ===========================================================================
//
// The complex128 twin of avx2_f32_mixedradix_stage11.s. See that file for the
// radix-11 butterfly derivation, why the ten constants live in RODATA rather
// than in registers, how the five odd stride multiples reach all ten rows, and
// the store order that keeps dst == input aliasing safe; only the packing
// differs.
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
DATA stage11d_negodd<>+0x00(SB)/8, $0x0000000000000000
DATA stage11d_negodd<>+0x08(SB)/8, $0x8000000000000000
GLOBL stage11d_negodd<>(SB), RODATA|NOPTR, $16

DATA stage11d_negeven<>+0x00(SB)/8, $0x8000000000000000
DATA stage11d_negeven<>+0x08(SB)/8, $0x0000000000000000
GLOBL stage11d_negeven<>(SB), RODATA|NOPTR, $16

// Butterfly constants, stored pre-broadcast so they can be FMA memory
// operands. cr = cos(2*pi*r/11), sr = sin(2*pi*r/11).
#define BCAST4(sym, bits) \
	DATA sym+0x00(SB)/8, $bits \
	DATA sym+0x08(SB)/8, $bits \
	DATA sym+0x10(SB)/8, $bits \
	DATA sym+0x18(SB)/8, $bits \
	GLOBL sym(SB), RODATA|NOPTR, $32

BCAST4(stage11d_c1<>, 0x3FEAEB8C8764F0BA) // c1 =  0.84125353283118121
BCAST4(stage11d_c2<>, 0x3FDA9628D9C712B6) // c2 =  0.41541501300188644
BCAST4(stage11d_c3<>, 0xBFC2375F640F44D6) // c3 = -0.14231483827328500
BCAST4(stage11d_c4<>, 0xBFE4F49E7F775886) // c4 = -0.65486073394528499
BCAST4(stage11d_c5<>, 0xBFEEB42A9BCD5057) // c5 = -0.95949297361449737
BCAST4(stage11d_s1<>, 0x3FE14CEDF8BB580B) // s1 =  0.54064081745559756
BCAST4(stage11d_s2<>, 0x3FED1BB48EEE2C13) // s2 =  0.90963199535451833
BCAST4(stage11d_s3<>, 0x3FEFAC9E043842F0) // s3 =  0.98982144188093280
BCAST4(stage11d_s4<>, 0x3FE82F19BB3A28A2) // s4 =  0.75574957435425838
BCAST4(stage11d_s5<>, 0x3FD207E7FD768DBE) // s5 =  0.28173255684142962

// ===========================================================================
// func MixedRadixStage11Complex128AVX2Asm(dst, input, table []complex128, span int, inverse bool)
// ===========================================================================
TEXT ·MixedRadixStage11Complex128AVX2Asm(SB), NOSPLIT, $0-81
	MOVQ dst+0(FP), DI       // DI = &dst[k],   advances
	MOVQ input+24(FP), SI    // SI = &input[k], advances
	MOVQ table+48(FP), DX    // DX = &table[k], advances
	MOVQ span+72(FP), R8     // R8 = span

	MOVQ R8, CX
	ANDQ $-2, CX             // whole 2-element blocks only
	SHLQ $4, CX              // CX = vector bound in bytes
	TESTQ CX, CX
	JZ   stage11_c128_done

	LEAQ (SI)(CX*1), BX      // BX = end of the input's vector range

	// Odd multiples of the row stride; the even ones come from the SIB scale.
	SHLQ $4, R8              // R8  = span*16  (1x row stride)
	LEAQ (R8)(R8*2), R9      // R9  = 3x
	LEAQ (R8)(R8*4), R10     // R10 = 5x
	LEAQ (R9)(R8*4), R11     // R11 = 7x
	LEAQ (R10)(R8*4), R12    // R12 = 9x

	MOVBLZX inverse+80(FP), AX
	TESTL AX, AX
	JNZ   stage11_c128_inv_mask

	VBROADCASTF128 stage11d_negodd<>(SB), Y15
	JMP  stage11_c128_mask_done

stage11_c128_inv_mask:
	VBROADCASTF128 stage11d_negeven<>(SB), Y15

stage11_c128_mask_done:

stage11_c128_loop:
	// ---- Pairs: tk = ak + a(11-k), uk = ak - a(11-k), formed as each pair is
	// twiddled so only two a values are ever live at once. ----

	// pair 1: rows 1 and 10
	VMOVUPD (SI)(R8*1), Y7   // Y7 = x = input row 1
	VMOVUPD (DX)(R8*1), Y8   // Y8 = t = table row 1
	VMOVDDUP Y7, Y0          // Y0 = [x.r, x.r, ...]
	VPERMILPD $0x0F, Y7, Y7  // Y7 = [x.i, x.i, ...]
	VPERMILPD $0x05, Y8, Y6  // Y6 = [t.i, t.r, ...]
	VMULPD Y7, Y6, Y6
	VFMADDSUB231PD Y0, Y8, Y6 // Y6 = a1
	VMOVUPD (SI)(R10*2), Y7
	VMOVUPD (DX)(R10*2), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y9
	VMULPD Y7, Y9, Y9
	VFMADDSUB231PD Y0, Y8, Y9 // Y9 = a10
	VADDPD Y9, Y6, Y1        // Y1  = t1
	VSUBPD Y9, Y6, Y10       // Y10 = u1

	// pair 2: rows 2 and 9
	VMOVUPD (SI)(R8*2), Y7
	VMOVUPD (DX)(R8*2), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y6
	VMULPD Y7, Y6, Y6
	VFMADDSUB231PD Y0, Y8, Y6 // Y6 = a2
	VMOVUPD (SI)(R12*1), Y7
	VMOVUPD (DX)(R12*1), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y9
	VMULPD Y7, Y9, Y9
	VFMADDSUB231PD Y0, Y8, Y9 // Y9 = a9
	VADDPD Y9, Y6, Y2        // Y2  = t2
	VSUBPD Y9, Y6, Y11       // Y11 = u2

	// pair 3: rows 3 and 8
	VMOVUPD (SI)(R9*1), Y7
	VMOVUPD (DX)(R9*1), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y6
	VMULPD Y7, Y6, Y6
	VFMADDSUB231PD Y0, Y8, Y6 // Y6 = a3
	VMOVUPD (SI)(R8*8), Y7
	VMOVUPD (DX)(R8*8), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y9
	VMULPD Y7, Y9, Y9
	VFMADDSUB231PD Y0, Y8, Y9 // Y9 = a8
	VADDPD Y9, Y6, Y3        // Y3  = t3
	VSUBPD Y9, Y6, Y12       // Y12 = u3

	// pair 4: rows 4 and 7
	VMOVUPD (SI)(R8*4), Y7
	VMOVUPD (DX)(R8*4), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y6
	VMULPD Y7, Y6, Y6
	VFMADDSUB231PD Y0, Y8, Y6 // Y6 = a4
	VMOVUPD (SI)(R11*1), Y7
	VMOVUPD (DX)(R11*1), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y9
	VMULPD Y7, Y9, Y9
	VFMADDSUB231PD Y0, Y8, Y9 // Y9 = a7
	VADDPD Y9, Y6, Y4        // Y4  = t4
	VSUBPD Y9, Y6, Y13       // Y13 = u4

	// pair 5: rows 5 and 6
	VMOVUPD (SI)(R10*1), Y7
	VMOVUPD (DX)(R10*1), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y6
	VMULPD Y7, Y6, Y6
	VFMADDSUB231PD Y0, Y8, Y6 // Y6 = a5
	VMOVUPD (SI)(R9*2), Y7
	VMOVUPD (DX)(R9*2), Y8
	VMOVDDUP Y7, Y0
	VPERMILPD $0x0F, Y7, Y7
	VPERMILPD $0x05, Y8, Y9
	VMULPD Y7, Y9, Y9
	VFMADDSUB231PD Y0, Y8, Y9 // Y9 = a6
	VADDPD Y9, Y6, Y5        // Y5  = t5
	VSUBPD Y9, Y6, Y14       // Y14 = u5

	// ---- j = 1: rows 1 and 10 ----
	VMOVUPD (SI), Y7                        // Y7 = a0
	VFMADD231PD stage11d_c1<>(SB), Y1, Y7   // += c1*t1
	VFMADD231PD stage11d_c2<>(SB), Y2, Y7   // += c2*t2
	VFMADD231PD stage11d_c3<>(SB), Y3, Y7   // += c3*t3
	VFMADD231PD stage11d_c4<>(SB), Y4, Y7   // += c4*t4
	VFMADD231PD stage11d_c5<>(SB), Y5, Y7   // Y7 = m1
	VMULPD stage11d_s1<>(SB), Y10, Y8       // Y8 = s1*u1
	VFMADD231PD stage11d_s2<>(SB), Y11, Y8  // += s2*u2
	VFMADD231PD stage11d_s3<>(SB), Y12, Y8  // += s3*u3
	VFMADD231PD stage11d_s4<>(SB), Y13, Y8  // += s4*u4
	VFMADD231PD stage11d_s5<>(SB), Y14, Y8  // += s5*u5
	VPERMILPD $0x05, Y8, Y8                 // swap re/im
	VXORPD Y15, Y8, Y8                      // Y8 = q1
	VADDPD Y8, Y7, Y0                       // y1  = m1 + q1
	VMOVUPD Y0, (DI)(R8*1)
	VSUBPD Y8, Y7, Y0                       // y10 = m1 - q1
	VMOVUPD Y0, (DI)(R10*2)

	// ---- j = 2: rows 2 and 9 ----
	VMOVUPD (SI), Y7
	VFMADD231PD stage11d_c2<>(SB), Y1, Y7
	VFMADD231PD stage11d_c4<>(SB), Y2, Y7
	VFMADD231PD stage11d_c5<>(SB), Y3, Y7
	VFMADD231PD stage11d_c3<>(SB), Y4, Y7
	VFMADD231PD stage11d_c1<>(SB), Y5, Y7   // Y7 = m2
	VMULPD stage11d_s2<>(SB), Y10, Y8       // Y8 = s2*u1
	VFMADD231PD stage11d_s4<>(SB), Y11, Y8  // += s4*u2
	VFNMADD231PD stage11d_s5<>(SB), Y12, Y8 // -= s5*u3
	VFNMADD231PD stage11d_s3<>(SB), Y13, Y8 // -= s3*u4
	VFNMADD231PD stage11d_s1<>(SB), Y14, Y8 // -= s1*u5
	VPERMILPD $0x05, Y8, Y8
	VXORPD Y15, Y8, Y8                      // Y8 = q2
	VADDPD Y8, Y7, Y0                       // y2 = m2 + q2
	VMOVUPD Y0, (DI)(R8*2)
	VSUBPD Y8, Y7, Y0                       // y9 = m2 - q2
	VMOVUPD Y0, (DI)(R12*1)

	// ---- j = 3: rows 3 and 8 ----
	VMOVUPD (SI), Y7
	VFMADD231PD stage11d_c3<>(SB), Y1, Y7
	VFMADD231PD stage11d_c5<>(SB), Y2, Y7
	VFMADD231PD stage11d_c2<>(SB), Y3, Y7
	VFMADD231PD stage11d_c1<>(SB), Y4, Y7
	VFMADD231PD stage11d_c4<>(SB), Y5, Y7   // Y7 = m3
	VMULPD stage11d_s3<>(SB), Y10, Y8       // Y8 = s3*u1
	VFNMADD231PD stage11d_s5<>(SB), Y11, Y8 // -= s5*u2
	VFNMADD231PD stage11d_s2<>(SB), Y12, Y8 // -= s2*u3
	VFMADD231PD stage11d_s1<>(SB), Y13, Y8  // += s1*u4
	VFMADD231PD stage11d_s4<>(SB), Y14, Y8  // += s4*u5
	VPERMILPD $0x05, Y8, Y8
	VXORPD Y15, Y8, Y8                      // Y8 = q3
	VADDPD Y8, Y7, Y0                       // y3 = m3 + q3
	VMOVUPD Y0, (DI)(R9*1)
	VSUBPD Y8, Y7, Y0                       // y8 = m3 - q3
	VMOVUPD Y0, (DI)(R8*8)

	// ---- j = 4: rows 4 and 7 ----
	VMOVUPD (SI), Y7
	VFMADD231PD stage11d_c4<>(SB), Y1, Y7
	VFMADD231PD stage11d_c3<>(SB), Y2, Y7
	VFMADD231PD stage11d_c1<>(SB), Y3, Y7
	VFMADD231PD stage11d_c5<>(SB), Y4, Y7
	VFMADD231PD stage11d_c2<>(SB), Y5, Y7   // Y7 = m4
	VMULPD stage11d_s4<>(SB), Y10, Y8       // Y8 = s4*u1
	VFNMADD231PD stage11d_s3<>(SB), Y11, Y8 // -= s3*u2
	VFMADD231PD stage11d_s1<>(SB), Y12, Y8  // += s1*u3
	VFMADD231PD stage11d_s5<>(SB), Y13, Y8  // += s5*u4
	VFNMADD231PD stage11d_s2<>(SB), Y14, Y8 // -= s2*u5
	VPERMILPD $0x05, Y8, Y8
	VXORPD Y15, Y8, Y8                      // Y8 = q4
	VADDPD Y8, Y7, Y0                       // y4 = m4 + q4
	VMOVUPD Y0, (DI)(R8*4)
	VSUBPD Y8, Y7, Y0                       // y7 = m4 - q4
	VMOVUPD Y0, (DI)(R11*1)

	// ---- j = 5: rows 5 and 6 ----
	VMOVUPD (SI), Y7
	VFMADD231PD stage11d_c5<>(SB), Y1, Y7
	VFMADD231PD stage11d_c1<>(SB), Y2, Y7
	VFMADD231PD stage11d_c4<>(SB), Y3, Y7
	VFMADD231PD stage11d_c2<>(SB), Y4, Y7
	VFMADD231PD stage11d_c3<>(SB), Y5, Y7   // Y7 = m5
	VMULPD stage11d_s5<>(SB), Y10, Y8       // Y8 = s5*u1
	VFNMADD231PD stage11d_s1<>(SB), Y11, Y8 // -= s1*u2
	VFMADD231PD stage11d_s4<>(SB), Y12, Y8  // += s4*u3
	VFNMADD231PD stage11d_s2<>(SB), Y13, Y8 // -= s2*u4
	VFMADD231PD stage11d_s3<>(SB), Y14, Y8  // += s3*u5
	VPERMILPD $0x05, Y8, Y8
	VXORPD Y15, Y8, Y8                      // Y8 = q5
	VADDPD Y8, Y7, Y0                       // y5 = m5 + q5
	VMOVUPD Y0, (DI)(R10*1)
	VSUBPD Y8, Y7, Y0                       // y6 = m5 - q5
	VMOVUPD Y0, (DI)(R9*2)

	// ---- y0 = a0 + t1 + ... + t5, stored last so it cannot clobber input
	// row 0 when dst aliases input. ----
	VMOVUPD (SI), Y7         // Y7 = a0
	VADDPD Y1, Y7, Y7
	VADDPD Y2, Y7, Y7
	VADDPD Y3, Y7, Y7
	VADDPD Y4, Y7, Y7
	VADDPD Y5, Y7, Y7        // Y7 = y0
	VMOVUPD Y7, (DI)

	ADDQ $32, SI             // advance one 2-element block
	ADDQ $32, DX
	ADDQ $32, DI
	CMPQ SI, BX
	JL   stage11_c128_loop

	VZEROUPPER

stage11_c128_done:
	RET
