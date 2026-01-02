//go:build arm64 && fft_asm && !purego

// ===========================================================================
// NEON Size-128 Mixed-Radix (Radix-4 + Radix-2) FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

// Forward transform, size 128, mixed radix (radix-4, radix-4, radix-4, radix-2).
TEXT ·ForwardNEONSize128MixedRadix24Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $128, R13
	BNE  neon128m24_return_false

	MOVD dst+8(FP), R0
	CMP  $128, R0
	BLT  neon128m24_return_false

	MOVD twiddle+56(FP), R0
	CMP  $128, R0
	BLT  neon128m24_return_false

	MOVD scratch+80(FP), R0
	CMP  $128, R0
	BLT  neon128m24_return_false

	MOVD bitrev+104(FP), R0
	CMP  $128, R0
	BLT  neon128m24_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon128m24_use_dst
	MOVD R11, R8

neon128m24_use_dst:
	// Bit-reversal permutation (mixed-radix 2/4)
	MOVD $0, R0

neon128m24_bitrev_loop:
	CMP  $128, R0
	BGE  neon128m24_stage1

	LSL  $3, R0, R1
	ADD  R12, R1, R1
	MOVD (R1), R2

	LSL  $3, R2, R3
	ADD  R9, R3, R3
	MOVD (R3), R4

	LSL  $3, R0, R3
	ADD  R8, R3, R3
	MOVD R4, (R3)

	ADD  $1, R0, R0
	B    neon128m24_bitrev_loop

neon128m24_stage1:
	// Stage 1: 32 radix-4 butterflies (no twiddles)
	MOVD $0, R14

neon128m24_stage1_loop:
	CMP  $128, R14
	BGE  neon128m24_stage2

	LSL  $3, R14, R1
	ADD  R8, R1, R1

	FMOVS 0(R1), F0
	FMOVS 4(R1), F1
	FMOVS 8(R1), F2
	FMOVS 12(R1), F3
	FMOVS 16(R1), F4
	FMOVS 20(R1), F5
	FMOVS 24(R1), F6
	FMOVS 28(R1), F7

	FADDS F4, F0, F8
	FADDS F5, F1, F9
	FSUBS F4, F0, F10
	FSUBS F5, F1, F11

	FADDS F6, F2, F12
	FADDS F7, F3, F13
	FSUBS F6, F2, F14
	FSUBS F7, F3, F15

	FADDS F12, F8, F16
	FADDS F13, F9, F17
	FSUBS F12, F8, F18
	FSUBS F13, F9, F19

	FMOVS F15, F20
	FNEGS F14, F21
	FADDS F20, F10, F22
	FADDS F21, F11, F23

	FNEGS F15, F24
	FMOVS F14, F25
	FADDS F24, F10, F26
	FADDS F25, F11, F27

	FMOVS F16, 0(R1)
	FMOVS F17, 4(R1)
	FMOVS F22, 8(R1)
	FMOVS F23, 12(R1)
	FMOVS F18, 16(R1)
	FMOVS F19, 20(R1)
	FMOVS F26, 24(R1)
	FMOVS F27, 28(R1)

	ADD  $4, R14, R14
	B    neon128m24_stage1_loop

neon128m24_stage2:
	// Stage 2: radix-4, size=16, step=8
	MOVD $0, R14

neon128m24_stage2_base:
	CMP  $128, R14
	BGE  neon128m24_stage3

	MOVD $0, R15

neon128m24_stage2_j:
	CMP  $4, R15
	BGE  neon128m24_stage2_next

	ADD  R14, R15, R0       // idx0
	ADD  $4, R0, R1         // idx1
	ADD  $8, R0, R2         // idx2
	ADD  $12, R0, R3        // idx3

	LSL  $3, R15, R4        // j*8
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1

	LSL  $4, R15, R4        // j*16
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3

	LSL  $3, R15, R6        // j*8
	LSL  $4, R15, R4        // j*16
	ADD  R4, R6, R6         // j*24
	LSL  $3, R6, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15

	// a2 = w2 * a2
	FMULS F2, F10, F16
	FMULS F3, F11, F17
	FSUBS F17, F16, F16
	FMULS F2, F11, F17
	FMULS F3, F10, F18
	FADDS F18, F17, F17

	// a3 = w3 * a3
	FMULS F4, F12, F18
	FMULS F5, F13, F19
	FSUBS F19, F18, F18
	FMULS F4, F13, F19
	FMULS F5, F12, F20
	FADDS F20, F19, F19

	FADDS F16, F6, F20
	FADDS F17, F7, F21
	FSUBS F16, F6, F22
	FSUBS F17, F7, F23

	FADDS F18, F14, F24
	FADDS F19, F15, F25
	FSUBS F18, F14, F26
	FSUBS F19, F15, F27

	// out0 = t0 + t2
	FADDS F24, F20, F28
	FADDS F25, F21, F29
	// out2 = t0 - t2
	FSUBS F24, F20, F30
	FSUBS F25, F21, F31

	// out1 = t1 + mulNegI(t3)
	FADDS F27, F22, F6
	FSUBS F26, F23, F7
	// out3 = t1 + mulI(t3)
	FSUBS F27, F22, F8
	FADDS F26, F23, F9

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F6, 0(R7)
	FMOVS F7, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F30, 0(R7)
	FMOVS F31, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F8, 0(R7)
	FMOVS F9, 4(R7)

	ADD  $1, R15, R15
	B    neon128m24_stage2_j

neon128m24_stage2_next:
	ADD  $16, R14, R14
	B    neon128m24_stage2_base

neon128m24_stage3:
	// Stage 3: radix-4, size=64, step=2
	MOVD $0, R14

neon128m24_stage3_base:
	CMP  $128, R14
	BGE  neon128m24_stage4

	MOVD $0, R15

neon128m24_stage3_j:
	CMP  $16, R15
	BGE  neon128m24_stage3_next

	ADD  R14, R15, R0       // idx0
	ADD  $16, R0, R1        // idx1
	ADD  $32, R0, R2        // idx2
	ADD  $48, R0, R3        // idx3

	LSL  $1, R15, R4        // j*2
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1

	LSL  $2, R15, R4        // j*4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3

	LSL  $1, R15, R6        // j*2
	LSL  $2, R15, R4        // j*4
	ADD  R4, R6, R6         // j*6
	LSL  $3, R6, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15

	// a2 = w2 * a2
	FMULS F2, F10, F16
	FMULS F3, F11, F17
	FSUBS F17, F16, F16
	FMULS F2, F11, F17
	FMULS F3, F10, F18
	FADDS F18, F17, F17

	// a3 = w3 * a3
	FMULS F4, F12, F18
	FMULS F5, F13, F19
	FSUBS F19, F18, F18
	FMULS F4, F13, F19
	FMULS F5, F12, F20
	FADDS F20, F19, F19

	FADDS F16, F6, F20
	FADDS F17, F7, F21
	FSUBS F16, F6, F22
	FSUBS F17, F7, F23

	FADDS F18, F14, F24
	FADDS F19, F15, F25
	FSUBS F18, F14, F26
	FSUBS F19, F15, F27

	// out0 = t0 + t2
	FADDS F24, F20, F28
	FADDS F25, F21, F29
	// out2 = t0 - t2
	FSUBS F24, F20, F30
	FSUBS F25, F21, F31

	// out1 = t1 + mulNegI(t3)
	FADDS F27, F22, F6
	FSUBS F26, F23, F7
	// out3 = t1 + mulI(t3)
	FSUBS F27, F22, F8
	FADDS F26, F23, F9

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F6, 0(R7)
	FMOVS F7, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F30, 0(R7)
	FMOVS F31, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F8, 0(R7)
	FMOVS F9, 4(R7)

	ADD  $1, R15, R15
	B    neon128m24_stage3_j

neon128m24_stage3_next:
	ADD  $64, R14, R14
	B    neon128m24_stage3_base

neon128m24_stage4:
	// Stage 4: radix-2, size=128, step=1
	MOVD $0, R0

neon128m24_stage4_loop:
	CMP  $64, R0
	BGE  neon128m24_done

	ADD  $64, R0, R1

	LSL  $3, R0, R2
	ADD  R10, R2, R2
	FMOVS 0(R2), F0
	FMOVS 4(R2), F1

	LSL  $3, R0, R2
	ADD  R8, R2, R2
	FMOVS 0(R2), F2
	FMOVS 4(R2), F3

	LSL  $3, R1, R2
	ADD  R8, R2, R2
	FMOVS 0(R2), F4
	FMOVS 4(R2), F5

	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	FADDS F6, F2, F8
	FADDS F7, F3, F9
	FSUBS F6, F2, F10
	FSUBS F7, F3, F11

	LSL  $3, R0, R2
	ADD  R8, R2, R2
	FMOVS F8, 0(R2)
	FMOVS F9, 4(R2)

	LSL  $3, R1, R2
	ADD  R8, R2, R2
	FMOVS F10, 0(R2)
	FMOVS F11, 4(R2)

	ADD  $1, R0, R0
	B    neon128m24_stage4_loop

neon128m24_done:
	CMP  R8, R20
	BEQ  neon128m24_return_true

	MOVD $0, R0
neon128m24_copy_loop:
	CMP  $128, R0
	BGE  neon128m24_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon128m24_copy_loop

neon128m24_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon128m24_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET

// Inverse transform, size 128, mixed radix (radix-4, radix-4, radix-4, radix-2).
TEXT ·InverseNEONSize128MixedRadix24Complex64Asm(SB), NOSPLIT, $0-121
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD bitrev+96(FP), R12
	MOVD src+32(FP), R13

	CMP  $128, R13
	BNE  neon128m24_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $128, R0
	BLT  neon128m24_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $128, R0
	BLT  neon128m24_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $128, R0
	BLT  neon128m24_inv_return_false

	MOVD bitrev+104(FP), R0
	CMP  $128, R0
	BLT  neon128m24_inv_return_false

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon128m24_inv_use_dst
	MOVD R11, R8

neon128m24_inv_use_dst:
	// Bit-reversal permutation (mixed-radix 2/4)
	MOVD $0, R0

neon128m24_inv_bitrev_loop:
	CMP  $128, R0
	BGE  neon128m24_inv_stage1

	LSL  $3, R0, R1
	ADD  R12, R1, R1
	MOVD (R1), R2

	LSL  $3, R2, R3
	ADD  R9, R3, R3
	MOVD (R3), R4

	LSL  $3, R0, R3
	ADD  R8, R3, R3
	MOVD R4, (R3)

	ADD  $1, R0, R0
	B    neon128m24_inv_bitrev_loop

neon128m24_inv_stage1:
	// Stage 1: 32 radix-4 butterflies (no twiddles)
	MOVD $0, R14

neon128m24_inv_stage1_loop:
	CMP  $128, R14
	BGE  neon128m24_inv_stage2

	LSL  $3, R14, R1
	ADD  R8, R1, R1

	FMOVS 0(R1), F0
	FMOVS 4(R1), F1
	FMOVS 8(R1), F2
	FMOVS 12(R1), F3
	FMOVS 16(R1), F4
	FMOVS 20(R1), F5
	FMOVS 24(R1), F6
	FMOVS 28(R1), F7

	FADDS F4, F0, F8
	FADDS F5, F1, F9
	FSUBS F4, F0, F10
	FSUBS F5, F1, F11

	FADDS F6, F2, F12
	FADDS F7, F3, F13
	FSUBS F6, F2, F14
	FSUBS F7, F3, F15

	FADDS F12, F8, F16
	FADDS F13, F9, F17
	FSUBS F12, F8, F18
	FSUBS F13, F9, F19

	FNEGS F15, F20
	FMOVS F14, F21
	FADDS F20, F10, F22
	FADDS F21, F11, F23

	FMOVS F15, F24
	FNEGS F14, F25
	FADDS F24, F10, F26
	FADDS F25, F11, F27

	FMOVS F16, 0(R1)
	FMOVS F17, 4(R1)
	FMOVS F26, 8(R1)
	FMOVS F27, 12(R1)
	FMOVS F18, 16(R1)
	FMOVS F19, 20(R1)
	FMOVS F22, 24(R1)
	FMOVS F23, 28(R1)

	ADD  $4, R14, R14
	B    neon128m24_inv_stage1_loop

neon128m24_inv_stage2:
	// Stage 2: radix-4, size=16, step=8 (conjugated twiddles)
	MOVD $0, R14

neon128m24_inv_stage2_base:
	CMP  $128, R14
	BGE  neon128m24_inv_stage3

	MOVD $0, R15

neon128m24_inv_stage2_j:
	CMP  $4, R15
	BGE  neon128m24_inv_stage2_next

	ADD  R14, R15, R0       // idx0
	ADD  $4, R0, R1         // idx1
	ADD  $8, R0, R2         // idx2
	ADD  $12, R0, R3        // idx3

	LSL  $3, R15, R4        // j*8
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1
	FNEGS F1, F1

	LSL  $4, R15, R4        // j*16
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3
	FNEGS F3, F3

	LSL  $3, R15, R6        // j*8
	LSL  $4, R15, R4        // j*16
	ADD  R4, R6, R6         // j*24
	LSL  $3, R6, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5
	FNEGS F5, F5

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15

	// a2 = w2 * a2
	FMULS F2, F10, F16
	FMULS F3, F11, F17
	FSUBS F17, F16, F16
	FMULS F2, F11, F17
	FMULS F3, F10, F18
	FADDS F18, F17, F17

	// a3 = w3 * a3
	FMULS F4, F12, F18
	FMULS F5, F13, F19
	FSUBS F19, F18, F18
	FMULS F4, F13, F19
	FMULS F5, F12, F20
	FADDS F20, F19, F19

	FADDS F16, F6, F20
	FADDS F17, F7, F21
	FSUBS F16, F6, F22
	FSUBS F17, F7, F23

	FADDS F18, F14, F24
	FADDS F19, F15, F25
	FSUBS F18, F14, F26
	FSUBS F19, F15, F27

	// out0 = t0 + t2
	FADDS F24, F20, F28
	FADDS F25, F21, F29
	// out2 = t0 - t2
	FSUBS F24, F20, F30
	FSUBS F25, F21, F31

	// out1 = t1 + mulI(t3)
	FSUBS F27, F22, F6
	FADDS F26, F23, F7
	// out3 = t1 + mulNegI(t3)
	FADDS F27, F22, F8
	FSUBS F26, F23, F9

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F6, 0(R7)
	FMOVS F7, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F30, 0(R7)
	FMOVS F31, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F8, 0(R7)
	FMOVS F9, 4(R7)

	ADD  $1, R15, R15
	B    neon128m24_inv_stage2_j

neon128m24_inv_stage2_next:
	ADD  $16, R14, R14
	B    neon128m24_inv_stage2_base

neon128m24_inv_stage3:
	// Stage 3: radix-4, size=64, step=2 (conjugated twiddles)
	MOVD $0, R14

neon128m24_inv_stage3_base:
	CMP  $128, R14
	BGE  neon128m24_inv_stage4

	MOVD $0, R15

neon128m24_inv_stage3_j:
	CMP  $16, R15
	BGE  neon128m24_inv_stage3_next

	ADD  R14, R15, R0       // idx0
	ADD  $16, R0, R1        // idx1
	ADD  $32, R0, R2        // idx2
	ADD  $48, R0, R3        // idx3

	LSL  $1, R15, R4        // j*2
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1
	FNEGS F1, F1

	LSL  $2, R15, R4        // j*4
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3
	FNEGS F3, F3

	LSL  $1, R15, R6        // j*2
	LSL  $2, R15, R4        // j*4
	ADD  R4, R6, R6         // j*6
	LSL  $3, R6, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F4
	FMOVS 4(R5), F5
	FNEGS F5, F5

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F6
	FMOVS 4(R7), F7

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F8
	FMOVS 4(R7), F9

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F10
	FMOVS 4(R7), F11

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS 0(R7), F12
	FMOVS 4(R7), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15

	// a2 = w2 * a2
	FMULS F2, F10, F16
	FMULS F3, F11, F17
	FSUBS F17, F16, F16
	FMULS F2, F11, F17
	FMULS F3, F10, F18
	FADDS F18, F17, F17

	// a3 = w3 * a3
	FMULS F4, F12, F18
	FMULS F5, F13, F19
	FSUBS F19, F18, F18
	FMULS F4, F13, F19
	FMULS F5, F12, F20
	FADDS F20, F19, F19

	FADDS F16, F6, F20
	FADDS F17, F7, F21
	FSUBS F16, F6, F22
	FSUBS F17, F7, F23

	FADDS F18, F14, F24
	FADDS F19, F15, F25
	FSUBS F18, F14, F26
	FSUBS F19, F15, F27

	// out0 = t0 + t2
	FADDS F24, F20, F28
	FADDS F25, F21, F29
	// out2 = t0 - t2
	FSUBS F24, F20, F30
	FSUBS F25, F21, F31

	// out1 = t1 + mulI(t3)
	FSUBS F27, F22, F6
	FADDS F26, F23, F7
	// out3 = t1 + mulNegI(t3)
	FADDS F27, F22, F8
	FSUBS F26, F23, F9

	LSL  $3, R0, R7
	ADD  R8, R7, R7
	FMOVS F28, 0(R7)
	FMOVS F29, 4(R7)

	LSL  $3, R1, R7
	ADD  R8, R7, R7
	FMOVS F6, 0(R7)
	FMOVS F7, 4(R7)

	LSL  $3, R2, R7
	ADD  R8, R7, R7
	FMOVS F30, 0(R7)
	FMOVS F31, 4(R7)

	LSL  $3, R3, R7
	ADD  R8, R7, R7
	FMOVS F8, 0(R7)
	FMOVS F9, 4(R7)

	ADD  $1, R15, R15
	B    neon128m24_inv_stage3_j

neon128m24_inv_stage3_next:
	ADD  $64, R14, R14
	B    neon128m24_inv_stage3_base

neon128m24_inv_stage4:
	// Stage 4: radix-2, size=128, step=1 (conjugated twiddles)
	MOVD $0, R0

neon128m24_inv_stage4_loop:
	CMP  $64, R0
	BGE  neon128m24_inv_scale

	ADD  $64, R0, R1

	LSL  $3, R0, R2
	ADD  R10, R2, R2
	FMOVS 0(R2), F0
	FMOVS 4(R2), F1
	FNEGS F1, F1

	LSL  $3, R0, R2
	ADD  R8, R2, R2
	FMOVS 0(R2), F2
	FMOVS 4(R2), F3

	LSL  $3, R1, R2
	ADD  R8, R2, R2
	FMOVS 0(R2), F4
	FMOVS 4(R2), F5

	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	FADDS F6, F2, F8
	FADDS F7, F3, F9
	FSUBS F6, F2, F10
	FSUBS F7, F3, F11

	LSL  $3, R0, R2
	ADD  R8, R2, R2
	FMOVS F8, 0(R2)
	FMOVS F9, 4(R2)

	LSL  $3, R1, R2
	ADD  R8, R2, R2
	FMOVS F10, 0(R2)
	FMOVS F11, 4(R2)

	ADD  $1, R0, R0
	B    neon128m24_inv_stage4_loop

neon128m24_inv_scale:
	CMP  R8, R20
	BEQ  neon128m24_inv_scale_apply

	MOVD $0, R0
neon128m24_inv_copy_loop:
	CMP  $128, R0
	BGE  neon128m24_inv_scale_apply
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon128m24_inv_copy_loop

neon128m24_inv_scale_apply:
	MOVD $·neonInv128(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon128m24_inv_scale_loop:
	CMP  $128, R0
	BGE  neon128m24_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon128m24_inv_scale_loop

neon128m24_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+120(FP)
	RET

neon128m24_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+120(FP)
	RET
