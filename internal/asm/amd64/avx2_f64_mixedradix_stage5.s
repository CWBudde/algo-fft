//go:build amd64 && !purego

// ===========================================================================
// AVX2 fused radix-5 mixed-radix stage (complex128) for AMD64
// ===========================================================================
//
// The complex128 twin of avx2_f32_mixedradix_stage5.s. See that file for the
// stage contract and the radix-5 butterfly derivation; only the packing
// differs.
//
// Layout: complex128 [re, im] pairs, 2 complex per YMM (32 bytes). The
// complex multiply uses the f64 idiom of avx2_f64_complex_mul.s:
// VMOVDDUP for the real broadcast, VPERMILPD $0x0F for the imaginary
// broadcast and VPERMILPD $0x05 for the pair swap.
//
// Only floor(span/2)*2 elements are processed; the caller handles the tail.
//
// ===========================================================================

#include "textflag.h"

// Sign masks for the multiply-by-i step (one 128-bit lane, broadcast).
DATA stage5d_negodd<>+0x00(SB)/8, $0x0000000000000000
DATA stage5d_negodd<>+0x08(SB)/8, $0x8000000000000000
GLOBL stage5d_negodd<>(SB), RODATA|NOPTR, $16

DATA stage5d_negeven<>+0x00(SB)/8, $0x8000000000000000
DATA stage5d_negeven<>+0x08(SB)/8, $0x0000000000000000
GLOBL stage5d_negeven<>(SB), RODATA|NOPTR, $16

// ===========================================================================
// func MixedRadixStage5Complex128AVX2Asm(dst, input, table []complex128, span int, inverse bool)
// ===========================================================================
TEXT ·MixedRadixStage5Complex128AVX2Asm(SB), NOSPLIT, $0-81
	MOVQ dst+0(FP), DI       // DI = dst base
	MOVQ input+24(FP), SI    // SI = input base
	MOVQ table+48(FP), DX    // DX = table base
	MOVQ span+72(FP), R8     // R8 = span

	MOVQ R8, CX
	ANDQ $-2, CX             // whole 2-element blocks only
	SHLQ $4, CX              // CX = vector bound in bytes
	TESTQ CX, CX
	JZ   stage5_c128_done

	SHLQ $4, R8              // R8 = span*16  (row 1 offset)
	MOVQ R8, R9
	ADDQ R8, R9              // R9 = span*32  (row 2 offset)
	MOVQ R9, R10
	ADDQ R8, R10             // R10 = span*48 (row 3 offset)
	MOVQ R10, R11
	ADDQ R8, R11             // R11 = span*64 (row 4 offset)

	MOVQ $0x3FD3C6EF372FE950, R15 // c1 =  0.30901699437494745
	MOVQ R15, X0
	VBROADCASTSD X0, Y11
	MOVQ $0xBFE9E3779B97F4A8, R15 // c2 = -0.8090169943749475
	MOVQ R15, X0
	VBROADCASTSD X0, Y12
	MOVQ $0x3FEE6F0E134454FF, R15 // s1 =  0.9510565162951535
	MOVQ R15, X0
	VBROADCASTSD X0, Y13
	MOVQ $0x3FE2CF2304755A5F, R15 // s2 =  0.5877852522924732
	MOVQ R15, X0
	VBROADCASTSD X0, Y14

	MOVBLZX inverse+80(FP), AX
	TESTL AX, AX
	JNZ   stage5_c128_inv_mask

	VBROADCASTF128 stage5d_negodd<>(SB), Y15
	JMP  stage5_c128_mask_done

stage5_c128_inv_mask:
	VBROADCASTF128 stage5d_negeven<>(SB), Y15

stage5_c128_mask_done:
	XORQ AX, AX

stage5_c128_loop:
	LEAQ (SI)(AX*1), R12     // R12 = &input[k]
	LEAQ (DX)(AX*1), R13     // R13 = &table[k]
	LEAQ (DI)(AX*1), R14     // R14 = &dst[k]

	VMOVUPD (R12), Y0        // Y0 = a0 (row 0 twiddle is 1)

	VMOVUPD (R12)(R8*1), Y1  // input row 1
	VMOVUPD (R13)(R8*1), Y8  // table row 1
	VMOVDDUP Y1, Y5          // Y5 = [x.r, x.r, ...]
	VPERMILPD $0x0F, Y1, Y6  // Y6 = [x.i, x.i, ...]
	VPERMILPD $0x05, Y8, Y7  // Y7 = [t.i, t.r, ...]
	VMULPD Y6, Y7, Y7
	VFMADDSUB231PD Y5, Y8, Y7
	VMOVAPD Y7, Y1           // Y1 = a1

	VMOVUPD (R12)(R9*1), Y2  // input row 2
	VMOVUPD (R13)(R9*1), Y8  // table row 2
	VMOVDDUP Y2, Y5
	VPERMILPD $0x0F, Y2, Y6
	VPERMILPD $0x05, Y8, Y7
	VMULPD Y6, Y7, Y7
	VFMADDSUB231PD Y5, Y8, Y7
	VMOVAPD Y7, Y2           // Y2 = a2

	VMOVUPD (R12)(R10*1), Y3 // input row 3
	VMOVUPD (R13)(R10*1), Y8 // table row 3
	VMOVDDUP Y3, Y5
	VPERMILPD $0x0F, Y3, Y6
	VPERMILPD $0x05, Y8, Y7
	VMULPD Y6, Y7, Y7
	VFMADDSUB231PD Y5, Y8, Y7
	VMOVAPD Y7, Y3           // Y3 = a3

	VMOVUPD (R12)(R11*1), Y4 // input row 4
	VMOVUPD (R13)(R11*1), Y8 // table row 4
	VMOVDDUP Y4, Y5
	VPERMILPD $0x0F, Y4, Y6
	VPERMILPD $0x05, Y8, Y7
	VMULPD Y6, Y7, Y7
	VFMADDSUB231PD Y5, Y8, Y7
	VMOVAPD Y7, Y4           // Y4 = a4

	VADDPD Y4, Y1, Y5        // Y5 = t1 = a1 + a4
	VADDPD Y3, Y2, Y6        // Y6 = t2 = a2 + a3
	VSUBPD Y4, Y1, Y7        // Y7 = t3 = a1 - a4
	VSUBPD Y3, Y2, Y8        // Y8 = t4 = a2 - a3

	VADDPD Y6, Y5, Y1        // Y1 = t1 + t2
	VADDPD Y0, Y1, Y1        // Y1 = y0
	VMOVUPD Y1, (R14)

	VMOVAPD Y0, Y1
	VFMADD231PD Y11, Y5, Y1  // += c1*t1
	VFMADD231PD Y12, Y6, Y1  // Y1 = m1

	VMOVAPD Y0, Y2
	VFMADD231PD Y12, Y5, Y2  // += c2*t1
	VFMADD231PD Y11, Y6, Y2  // Y2 = m2

	VMULPD Y13, Y7, Y3       // Y3 = s1*t3
	VFMADD231PD Y14, Y8, Y3  // Y3 += s2*t4
	VPERMILPD $0x05, Y3, Y3  // swap re/im
	VXORPD Y15, Y3, Y3       // Y3 = q1

	VMULPD Y14, Y7, Y4       // Y4 = s2*t3
	VFNMADD231PD Y13, Y8, Y4 // Y4 -= s1*t4
	VPERMILPD $0x05, Y4, Y4  // swap re/im
	VXORPD Y15, Y4, Y4       // Y4 = q2

	VADDPD Y3, Y1, Y5        // y1 = m1 + q1
	VMOVUPD Y5, (R14)(R8*1)
	VADDPD Y4, Y2, Y6        // y2 = m2 + q2
	VMOVUPD Y6, (R14)(R9*1)
	VSUBPD Y4, Y2, Y7        // y3 = m2 - q2
	VMOVUPD Y7, (R14)(R10*1)
	VSUBPD Y3, Y1, Y8        // y4 = m1 - q1
	VMOVUPD Y8, (R14)(R11*1)

	ADDQ $32, AX             // advance one 2-element block
	CMPQ AX, CX
	JL   stage5_c128_loop

	VZEROUPPER

stage5_c128_done:
	RET
