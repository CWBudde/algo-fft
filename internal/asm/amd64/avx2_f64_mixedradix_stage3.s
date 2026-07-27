//go:build amd64 && !purego

// ===========================================================================
// AVX2 fused radix-3 mixed-radix stage (complex128) for AMD64
// ===========================================================================
//
// The complex128 twin of avx2_f32_mixedradix_stage3.s; see it for the
// butterfly derivation, and avx2_f64_mixedradix_stage5.s for the f64
// packing and complex-multiply idiom.
//
// Only floor(span/2)*2 elements are processed; the caller handles the tail.
//
// ===========================================================================

#include "textflag.h"

DATA stage3d_negodd<>+0x00(SB)/8, $0x0000000000000000
DATA stage3d_negodd<>+0x08(SB)/8, $0x8000000000000000
GLOBL stage3d_negodd<>(SB), RODATA|NOPTR, $16

DATA stage3d_negeven<>+0x00(SB)/8, $0x8000000000000000
DATA stage3d_negeven<>+0x08(SB)/8, $0x0000000000000000
GLOBL stage3d_negeven<>(SB), RODATA|NOPTR, $16

// ===========================================================================
// func MixedRadixStage3Complex128AVX2Asm(dst, input, table []complex128, span int, inverse bool)
// ===========================================================================
TEXT ·MixedRadixStage3Complex128AVX2Asm(SB), NOSPLIT, $0-81
	MOVQ dst+0(FP), DI       // DI = dst base
	MOVQ input+24(FP), SI    // SI = input base
	MOVQ table+48(FP), DX    // DX = table base
	MOVQ span+72(FP), R8     // R8 = span

	MOVQ R8, CX
	ANDQ $-2, CX             // whole 2-element blocks only
	SHLQ $4, CX              // CX = vector bound in bytes
	TESTQ CX, CX
	JZ   stage3_c128_done

	SHLQ $4, R8              // R8 = span*16 (row 1 offset)
	MOVQ R8, R9
	ADDQ R8, R9              // R9 = span*32 (row 2 offset)

	MOVQ $0xBFE0000000000000, R15 // -0.5
	VMOVQ R15, X0
	VBROADCASTSD X0, Y11
	MOVQ $0x3FEBB67AE8584CAA, R15 // sqrt(3)/2 = 0.8660254037844386
	VMOVQ R15, X0
	VBROADCASTSD X0, Y12

	MOVBLZX inverse+80(FP), AX
	TESTL AX, AX
	JNZ   stage3_c128_inv_mask

	VBROADCASTF128 stage3d_negodd<>(SB), Y15
	JMP  stage3_c128_mask_done

stage3_c128_inv_mask:
	VBROADCASTF128 stage3d_negeven<>(SB), Y15

stage3_c128_mask_done:
	XORQ AX, AX

stage3_c128_loop:
	LEAQ (SI)(AX*1), R12     // R12 = &input[k]
	LEAQ (DX)(AX*1), R13     // R13 = &table[k]
	LEAQ (DI)(AX*1), R14     // R14 = &dst[k]

	VMOVUPD (R12), Y0        // Y0 = a0 (row 0 twiddle is 1)

	VMOVUPD (R12)(R8*1), Y1  // input row 1
	VMOVUPD (R13)(R8*1), Y8  // table row 1
	VMOVDDUP Y1, Y5
	VPERMILPD $0x0F, Y1, Y6
	VPERMILPD $0x05, Y8, Y7
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

	VADDPD Y2, Y1, Y5        // Y5 = t1 = a1 + a2
	VSUBPD Y2, Y1, Y6        // Y6 = t2 = a1 - a2

	VADDPD Y5, Y0, Y3        // y0 = a0 + t1
	VMOVUPD Y3, (R14)

	VMOVAPD Y0, Y4
	VFMADD231PD Y11, Y5, Y4  // Y4 = m = a0 - 0.5*t1

	VMULPD Y12, Y6, Y7       // Y7 = sqrt(3)/2 * t2
	VPERMILPD $0x05, Y7, Y7
	VXORPD Y15, Y7, Y7       // Y7 = q

	VADDPD Y7, Y4, Y1        // y1 = m + q
	VMOVUPD Y1, (R14)(R8*1)
	VSUBPD Y7, Y4, Y2        // y2 = m - q
	VMOVUPD Y2, (R14)(R9*1)

	ADDQ $32, AX
	CMPQ AX, CX
	JL   stage3_c128_loop

	VZEROUPPER

stage3_c128_done:
	RET
