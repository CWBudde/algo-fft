//go:build amd64 && !purego

// ===========================================================================
// AVX2 fused radix-3 mixed-radix stage (complex64) for AMD64
// ===========================================================================
//
// The radix-3 sibling of avx2_f32_mixedradix_stage5.s; see that file for the
// stage contract, the memory layout, the complex-multiply idiom and why the
// k index vectorises without any cross-lane movement.
//
//	for k := range span {
//	    a0 := input[k]
//	    aj := input[j*span+k] * table[j*span+k]     // j = 1, 2
//	    y0..y2 := butterfly3(a0, a1, a2)
//	    dst[j*span+k] = yj
//	}
//
// RADIX-3 BUTTERFLY
// -----------------
// With w = exp(-2*pi*i/3) = -1/2 - i*sqrt(3)/2 and w^2 = conj(w):
//
//	t1 = a1 + a2    t2 = a1 - a2
//	y0 = a0 + t1
//	m  = a0 - 0.5*t1
//	q  = -i * (sqrt(3)/2 * t2)
//	y1 = m + q      y2 = m - q
//
// The inverse conjugates w, which turns the -i into +i — the same one-mask
// choice as the radix-5 stage.
//
// Only floor(span/4)*4 elements are processed; the caller handles the tail.
//
// ===========================================================================

#include "textflag.h"

DATA stage3_negodd<>+0x00(SB)/4, $0x00000000
DATA stage3_negodd<>+0x04(SB)/4, $0x80000000
DATA stage3_negodd<>+0x08(SB)/4, $0x00000000
DATA stage3_negodd<>+0x0C(SB)/4, $0x80000000
GLOBL stage3_negodd<>(SB), RODATA|NOPTR, $16

DATA stage3_negeven<>+0x00(SB)/4, $0x80000000
DATA stage3_negeven<>+0x04(SB)/4, $0x00000000
DATA stage3_negeven<>+0x08(SB)/4, $0x80000000
DATA stage3_negeven<>+0x0C(SB)/4, $0x00000000
GLOBL stage3_negeven<>(SB), RODATA|NOPTR, $16

// ===========================================================================
// func MixedRadixStage3Complex64AVX2Asm(dst, input, table []complex64, span int, inverse bool)
// ===========================================================================
TEXT ·MixedRadixStage3Complex64AVX2Asm(SB), NOSPLIT, $0-81
	MOVQ dst+0(FP), DI       // DI = dst base
	MOVQ input+24(FP), SI    // SI = input base
	MOVQ table+48(FP), DX    // DX = table base
	MOVQ span+72(FP), R8     // R8 = span

	MOVQ R8, CX
	ANDQ $-4, CX             // whole 4-element blocks only
	SHLQ $3, CX              // CX = vector bound in bytes
	TESTQ CX, CX
	JZ   stage3_c64_done

	SHLQ $3, R8              // R8 = span*8  (row 1 offset)
	MOVQ R8, R9
	ADDQ R8, R9              // R9 = span*16 (row 2 offset)

	MOVL $0xBF000000, AX     // -0.5
	MOVD AX, X0
	VBROADCASTSS X0, Y11
	MOVL $0x3F5DB3D7, AX     // sqrt(3)/2 = 0.8660254
	MOVD AX, X0
	VBROADCASTSS X0, Y12

	MOVBLZX inverse+80(FP), AX
	TESTL AX, AX
	JNZ   stage3_c64_inv_mask

	VBROADCASTF128 stage3_negodd<>(SB), Y15
	JMP  stage3_c64_mask_done

stage3_c64_inv_mask:
	VBROADCASTF128 stage3_negeven<>(SB), Y15

stage3_c64_mask_done:
	XORQ AX, AX

stage3_c64_loop:
	LEAQ (SI)(AX*1), R12     // R12 = &input[k]
	LEAQ (DX)(AX*1), R13     // R13 = &table[k]
	LEAQ (DI)(AX*1), R14     // R14 = &dst[k]

	VMOVUPS (R12), Y0        // Y0 = a0 (row 0 twiddle is 1)

	VMOVUPS (R12)(R8*1), Y1  // input row 1
	VMOVUPS (R13)(R8*1), Y8  // table row 1
	VMOVSLDUP Y1, Y5
	VMOVSHDUP Y1, Y6
	VSHUFPS $0xB1, Y8, Y8, Y7
	VMULPS Y6, Y7, Y7
	VFMADDSUB231PS Y5, Y8, Y7
	VMOVAPS Y7, Y1           // Y1 = a1

	VMOVUPS (R12)(R9*1), Y2  // input row 2
	VMOVUPS (R13)(R9*1), Y8  // table row 2
	VMOVSLDUP Y2, Y5
	VMOVSHDUP Y2, Y6
	VSHUFPS $0xB1, Y8, Y8, Y7
	VMULPS Y6, Y7, Y7
	VFMADDSUB231PS Y5, Y8, Y7
	VMOVAPS Y7, Y2           // Y2 = a2

	VADDPS Y2, Y1, Y5        // Y5 = t1 = a1 + a2
	VSUBPS Y2, Y1, Y6        // Y6 = t2 = a1 - a2

	VADDPS Y5, Y0, Y3        // y0 = a0 + t1
	VMOVUPS Y3, (R14)

	VMOVAPS Y0, Y4
	VFMADD231PS Y11, Y5, Y4  // Y4 = m = a0 - 0.5*t1

	VMULPS Y12, Y6, Y7       // Y7 = sqrt(3)/2 * t2
	VSHUFPS $0xB1, Y7, Y7, Y7
	VXORPS Y15, Y7, Y7       // Y7 = q

	VADDPS Y7, Y4, Y1        // y1 = m + q
	VMOVUPS Y1, (R14)(R8*1)
	VSUBPS Y7, Y4, Y2        // y2 = m - q
	VMOVUPS Y2, (R14)(R9*1)

	ADDQ $32, AX
	CMPQ AX, CX
	JL   stage3_c64_loop

	VZEROUPPER

stage3_c64_done:
	RET
