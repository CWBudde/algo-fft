//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Complex Scale Helpers for ARM64
// ===========================================================================

#include "textflag.h"

// func ScaleComplex64NEONAsm(dst []complex64, scale float32)
TEXT ·ScaleComplex64NEONAsm(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD dst+8(FP), R1
	CBZ  R1, scale64_done

	MOVW scale+24(FP), R2
	VMOV R2, V0.S[0]
	VMOV R2, V0.S[1]
	VMOV R2, V0.S[2]
	VMOV R2, V0.S[3]

	MOVD R1, R3
	LSR  $1, R3, R3
	CBZ  R3, scale64_tail

scale64_loop:
	VLD1 (R0), [V1.S4]
	VEOR V2.B16, V2.B16, V2.B16
	VFMLA V0.S4, V1.S4, V2.S4
	VST1 [V2.S4], (R0)

	ADD  $16, R0, R0
	SUB  $1, R3, R3
	CBNZ R3, scale64_loop

scale64_tail:
	AND  $1, R1, R3
	CBZ  R3, scale64_done

	FMOVS scale+24(FP), F0
	FMOVS 0(R0), F1
	FMULS F0, F1, F1
	FMOVS F1, 0(R0)
	FMOVS 4(R0), F2
	FMULS F0, F2, F2
	FMOVS F2, 4(R0)

scale64_done:
	RET

// func ScaleComplex128NEONAsm(dst []complex128, scale float64)
TEXT ·ScaleComplex128NEONAsm(SB), NOSPLIT, $0-32
	MOVD dst+0(FP), R0
	MOVD dst+8(FP), R1
	CBZ  R1, scale128_done

	MOVD scale+24(FP), R2
	VMOV R2, V0.D[0]
	VMOV R2, V0.D[1]

scale128_loop:
	VLD1 (R0), [V1.D2]
	VEOR V2.B16, V2.B16, V2.B16
	VFMLA V0.D2, V1.D2, V2.D2
	VST1 [V2.D2], (R0)

	ADD  $16, R0, R0
	SUB  $1, R1, R1
	CBNZ R1, scale128_loop

scale128_done:
	RET
