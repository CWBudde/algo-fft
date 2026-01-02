//go:build arm64 && asm && !purego

// =========================================================================== 
// NEON complex multiply helpers for ARM64
// ===========================================================================

#include "textflag.h"

// func ComplexMulArrayComplex64NEONAsm(dst, a, b []complex64)
TEXT ·ComplexMulArrayComplex64NEONAsm(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	MOVD dst+8(FP), R3

	CBZ  R3, cmul64_done
	MOVD R3, R4
	LSR  $1, R4, R4            // pairs = n / 2
	CBZ  R4, cmul64_tail

cmul64_loop:
	VLD1 (R1), [V0.S4]
	VLD1 (R2), [V1.S4]

	VUZP1 V0.S4, V0.S4, V2.S4   // ar
	VUZP2 V0.S4, V0.S4, V3.S4   // ai
	VUZP1 V1.S4, V1.S4, V4.S4   // br
	VUZP2 V1.S4, V1.S4, V5.S4   // bi

	VEOR V6.B16, V6.B16, V6.B16
	VFMLA V2.S4, V4.S4, V6.S4
	VFMLS V3.S4, V5.S4, V6.S4

	VEOR V7.B16, V7.B16, V7.B16
	VFMLA V2.S4, V5.S4, V7.S4
	VFMLA V3.S4, V4.S4, V7.S4

	VZIP1 V7.S4, V6.S4, V8.S4
	VST1 [V8.S4], (R0)

	ADD  $16, R1, R1
	ADD  $16, R2, R2
	ADD  $16, R0, R0
	SUB  $1, R4, R4
	CBNZ R4, cmul64_loop

cmul64_tail:
	AND  $1, R3, R4
	CBZ  R4, cmul64_done

	FMOVS 0(R1), F0
	FMOVS 4(R1), F1
	FMOVS 0(R2), F2
	FMOVS 4(R2), F3

	FMULS F0, F2, F4
	FMULS F1, F3, F5
	FSUBS F5, F4, F6
	FMULS F0, F3, F7
	FMULS F1, F2, F8
	FADDS F8, F7, F9

	FMOVS F6, 0(R0)
	FMOVS F9, 4(R0)

cmul64_done:
	RET

// func ComplexMulArrayInPlaceComplex64NEONAsm(dst, src []complex64)
TEXT ·ComplexMulArrayInPlaceComplex64NEONAsm(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD dst+8(FP), R2

	CBZ  R2, cmul64_inplace_done
	MOVD R2, R3
	LSR  $1, R3, R3            // pairs = n / 2
	CBZ  R3, cmul64_inplace_tail

cmul64_inplace_loop:
	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]

	VUZP1 V0.S4, V0.S4, V2.S4
	VUZP2 V0.S4, V0.S4, V3.S4
	VUZP1 V1.S4, V1.S4, V4.S4
	VUZP2 V1.S4, V1.S4, V5.S4

	VEOR V6.B16, V6.B16, V6.B16
	VFMLA V2.S4, V4.S4, V6.S4
	VFMLS V3.S4, V5.S4, V6.S4

	VEOR V7.B16, V7.B16, V7.B16
	VFMLA V2.S4, V5.S4, V7.S4
	VFMLA V3.S4, V4.S4, V7.S4

	VZIP1 V7.S4, V6.S4, V8.S4
	VST1 [V8.S4], (R0)

	ADD  $16, R0, R0
	ADD  $16, R1, R1
	SUB  $1, R3, R3
	CBNZ R3, cmul64_inplace_loop

cmul64_inplace_tail:
	AND  $1, R2, R3
	CBZ  R3, cmul64_inplace_done

	FMOVS 0(R0), F0
	FMOVS 4(R0), F1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3

	FMULS F0, F2, F4
	FMULS F1, F3, F5
	FSUBS F5, F4, F6
	FMULS F0, F3, F7
	FMULS F1, F2, F8
	FADDS F8, F7, F9

	FMOVS F6, 0(R0)
	FMOVS F9, 4(R0)

cmul64_inplace_done:
	RET

// func ComplexMulArrayComplex128NEONAsm(dst, a, b []complex128)
TEXT ·ComplexMulArrayComplex128NEONAsm(SB), NOSPLIT, $0-72
	MOVD dst+0(FP), R0
	MOVD a+24(FP), R1
	MOVD b+48(FP), R2
	MOVD dst+8(FP), R3

	CBZ  R3, cmul128_done

cmul128_loop:
	FMOVD 0(R1), F0
	FMOVD 8(R1), F1
	FMOVD 0(R2), F2
	FMOVD 8(R2), F3

	FMULD F0, F2, F4
	FMULD F1, F3, F5
	FSUBD F5, F4, F6
	FMULD F0, F3, F7
	FMULD F1, F2, F8
	FADDD F8, F7, F9

	FMOVD F6, 0(R0)
	FMOVD F9, 8(R0)

	ADD  $16, R1, R1
	ADD  $16, R2, R2
	ADD  $16, R0, R0
	SUB  $1, R3, R3
	CBNZ R3, cmul128_loop

cmul128_done:
	RET

// func ComplexMulArrayInPlaceComplex128NEONAsm(dst, src []complex128)
TEXT ·ComplexMulArrayInPlaceComplex128NEONAsm(SB), NOSPLIT, $0-48
	MOVD dst+0(FP), R0
	MOVD src+24(FP), R1
	MOVD dst+8(FP), R2

	CBZ  R2, cmul128_inplace_done

cmul128_inplace_loop:
	FMOVD 0(R0), F0
	FMOVD 8(R0), F1
	FMOVD 0(R1), F2
	FMOVD 8(R1), F3

	FMULD F0, F2, F4
	FMULD F1, F3, F5
	FSUBD F5, F4, F6
	FMULD F0, F3, F7
	FMULD F1, F2, F8
	FADDD F8, F7, F9

	FMOVD F6, 0(R0)
	FMOVD F9, 8(R0)

	ADD  $16, R0, R0
	ADD  $16, R1, R1
	SUB  $1, R2, R2
	CBNZ R2, cmul128_inplace_loop

cmul128_inplace_done:
	RET
