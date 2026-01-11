//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-128 Mixed-Radix (Radix-4 + Radix-2) FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

// Forward transform, size 128, mixed radix (radix-4, radix-4, radix-4, radix-2).
TEXT ·ForwardNEONSize128MixedRadix24Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
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

	MOVD $bitrev_size128_mixed24<>(SB), R12

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
	MOVB R0, ret+96(FP)
	RET

neon128m24_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 128, mixed radix (radix-4, radix-4, radix-4, radix-2).
TEXT ·InverseNEONSize128MixedRadix24Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
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

	MOVD $bitrev_size128_mixed24<>(SB), R12

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
	MOVB R0, ret+96(FP)
	RET

neon128m24_inv_return_false:
	MOVB R0, ret+96(FP)
	RET

// Bit-reversal table for size 128 mixed-radix 2/4
GLOBL bitrev_size128_mixed24<>(SB), RODATA, $1024
DATA bitrev_size128_mixed24<>+0(SB)/8, $0
DATA bitrev_size128_mixed24<>+8(SB)/8, $32
DATA bitrev_size128_mixed24<>+16(SB)/8, $64
DATA bitrev_size128_mixed24<>+24(SB)/8, $96
DATA bitrev_size128_mixed24<>+32(SB)/8, $8
DATA bitrev_size128_mixed24<>+40(SB)/8, $40
DATA bitrev_size128_mixed24<>+48(SB)/8, $72
DATA bitrev_size128_mixed24<>+56(SB)/8, $104
DATA bitrev_size128_mixed24<>+64(SB)/8, $16
DATA bitrev_size128_mixed24<>+72(SB)/8, $48
DATA bitrev_size128_mixed24<>+80(SB)/8, $80
DATA bitrev_size128_mixed24<>+88(SB)/8, $112
DATA bitrev_size128_mixed24<>+96(SB)/8, $24
DATA bitrev_size128_mixed24<>+104(SB)/8, $56
DATA bitrev_size128_mixed24<>+112(SB)/8, $88
DATA bitrev_size128_mixed24<>+120(SB)/8, $120
DATA bitrev_size128_mixed24<>+128(SB)/8, $2
DATA bitrev_size128_mixed24<>+136(SB)/8, $34
DATA bitrev_size128_mixed24<>+144(SB)/8, $66
DATA bitrev_size128_mixed24<>+152(SB)/8, $98
DATA bitrev_size128_mixed24<>+160(SB)/8, $10
DATA bitrev_size128_mixed24<>+168(SB)/8, $42
DATA bitrev_size128_mixed24<>+176(SB)/8, $74
DATA bitrev_size128_mixed24<>+184(SB)/8, $106
DATA bitrev_size128_mixed24<>+192(SB)/8, $18
DATA bitrev_size128_mixed24<>+200(SB)/8, $50
DATA bitrev_size128_mixed24<>+208(SB)/8, $82
DATA bitrev_size128_mixed24<>+216(SB)/8, $114
DATA bitrev_size128_mixed24<>+224(SB)/8, $26
DATA bitrev_size128_mixed24<>+232(SB)/8, $58
DATA bitrev_size128_mixed24<>+240(SB)/8, $90
DATA bitrev_size128_mixed24<>+248(SB)/8, $122
DATA bitrev_size128_mixed24<>+256(SB)/8, $4
DATA bitrev_size128_mixed24<>+264(SB)/8, $36
DATA bitrev_size128_mixed24<>+272(SB)/8, $68
DATA bitrev_size128_mixed24<>+280(SB)/8, $100
DATA bitrev_size128_mixed24<>+288(SB)/8, $12
DATA bitrev_size128_mixed24<>+296(SB)/8, $44
DATA bitrev_size128_mixed24<>+304(SB)/8, $76
DATA bitrev_size128_mixed24<>+312(SB)/8, $108
DATA bitrev_size128_mixed24<>+320(SB)/8, $20
DATA bitrev_size128_mixed24<>+328(SB)/8, $52
DATA bitrev_size128_mixed24<>+336(SB)/8, $84
DATA bitrev_size128_mixed24<>+344(SB)/8, $116
DATA bitrev_size128_mixed24<>+352(SB)/8, $28
DATA bitrev_size128_mixed24<>+360(SB)/8, $60
DATA bitrev_size128_mixed24<>+368(SB)/8, $92
DATA bitrev_size128_mixed24<>+376(SB)/8, $124
DATA bitrev_size128_mixed24<>+384(SB)/8, $6
DATA bitrev_size128_mixed24<>+392(SB)/8, $38
DATA bitrev_size128_mixed24<>+400(SB)/8, $70
DATA bitrev_size128_mixed24<>+408(SB)/8, $102
DATA bitrev_size128_mixed24<>+416(SB)/8, $14
DATA bitrev_size128_mixed24<>+424(SB)/8, $46
DATA bitrev_size128_mixed24<>+432(SB)/8, $78
DATA bitrev_size128_mixed24<>+440(SB)/8, $110
DATA bitrev_size128_mixed24<>+448(SB)/8, $22
DATA bitrev_size128_mixed24<>+456(SB)/8, $54
DATA bitrev_size128_mixed24<>+464(SB)/8, $86
DATA bitrev_size128_mixed24<>+472(SB)/8, $118
DATA bitrev_size128_mixed24<>+480(SB)/8, $30
DATA bitrev_size128_mixed24<>+488(SB)/8, $62
DATA bitrev_size128_mixed24<>+496(SB)/8, $94
DATA bitrev_size128_mixed24<>+504(SB)/8, $126
DATA bitrev_size128_mixed24<>+512(SB)/8, $1
DATA bitrev_size128_mixed24<>+520(SB)/8, $33
DATA bitrev_size128_mixed24<>+528(SB)/8, $65
DATA bitrev_size128_mixed24<>+536(SB)/8, $97
DATA bitrev_size128_mixed24<>+544(SB)/8, $9
DATA bitrev_size128_mixed24<>+552(SB)/8, $41
DATA bitrev_size128_mixed24<>+560(SB)/8, $73
DATA bitrev_size128_mixed24<>+568(SB)/8, $105
DATA bitrev_size128_mixed24<>+576(SB)/8, $17
DATA bitrev_size128_mixed24<>+584(SB)/8, $49
DATA bitrev_size128_mixed24<>+592(SB)/8, $81
DATA bitrev_size128_mixed24<>+600(SB)/8, $113
DATA bitrev_size128_mixed24<>+608(SB)/8, $25
DATA bitrev_size128_mixed24<>+616(SB)/8, $57
DATA bitrev_size128_mixed24<>+624(SB)/8, $89
DATA bitrev_size128_mixed24<>+632(SB)/8, $121
DATA bitrev_size128_mixed24<>+640(SB)/8, $3
DATA bitrev_size128_mixed24<>+648(SB)/8, $35
DATA bitrev_size128_mixed24<>+656(SB)/8, $67
DATA bitrev_size128_mixed24<>+664(SB)/8, $99
DATA bitrev_size128_mixed24<>+672(SB)/8, $11
DATA bitrev_size128_mixed24<>+680(SB)/8, $43
DATA bitrev_size128_mixed24<>+688(SB)/8, $75
DATA bitrev_size128_mixed24<>+696(SB)/8, $107
DATA bitrev_size128_mixed24<>+704(SB)/8, $19
DATA bitrev_size128_mixed24<>+712(SB)/8, $51
DATA bitrev_size128_mixed24<>+720(SB)/8, $83
DATA bitrev_size128_mixed24<>+728(SB)/8, $115
DATA bitrev_size128_mixed24<>+736(SB)/8, $27
DATA bitrev_size128_mixed24<>+744(SB)/8, $59
DATA bitrev_size128_mixed24<>+752(SB)/8, $91
DATA bitrev_size128_mixed24<>+760(SB)/8, $123
DATA bitrev_size128_mixed24<>+768(SB)/8, $5
DATA bitrev_size128_mixed24<>+776(SB)/8, $37
DATA bitrev_size128_mixed24<>+784(SB)/8, $69
DATA bitrev_size128_mixed24<>+792(SB)/8, $101
DATA bitrev_size128_mixed24<>+800(SB)/8, $13
DATA bitrev_size128_mixed24<>+808(SB)/8, $45
DATA bitrev_size128_mixed24<>+816(SB)/8, $77
DATA bitrev_size128_mixed24<>+824(SB)/8, $109
DATA bitrev_size128_mixed24<>+832(SB)/8, $21
DATA bitrev_size128_mixed24<>+840(SB)/8, $53
DATA bitrev_size128_mixed24<>+848(SB)/8, $85
DATA bitrev_size128_mixed24<>+856(SB)/8, $117
DATA bitrev_size128_mixed24<>+864(SB)/8, $29
DATA bitrev_size128_mixed24<>+872(SB)/8, $61
DATA bitrev_size128_mixed24<>+880(SB)/8, $93
DATA bitrev_size128_mixed24<>+888(SB)/8, $125
DATA bitrev_size128_mixed24<>+896(SB)/8, $7
DATA bitrev_size128_mixed24<>+904(SB)/8, $39
DATA bitrev_size128_mixed24<>+912(SB)/8, $71
DATA bitrev_size128_mixed24<>+920(SB)/8, $103
DATA bitrev_size128_mixed24<>+928(SB)/8, $15
DATA bitrev_size128_mixed24<>+936(SB)/8, $47
DATA bitrev_size128_mixed24<>+944(SB)/8, $79
DATA bitrev_size128_mixed24<>+952(SB)/8, $111
DATA bitrev_size128_mixed24<>+960(SB)/8, $23
DATA bitrev_size128_mixed24<>+968(SB)/8, $55
DATA bitrev_size128_mixed24<>+976(SB)/8, $87
DATA bitrev_size128_mixed24<>+984(SB)/8, $119
DATA bitrev_size128_mixed24<>+992(SB)/8, $31
DATA bitrev_size128_mixed24<>+1000(SB)/8, $63
DATA bitrev_size128_mixed24<>+1008(SB)/8, $95
DATA bitrev_size128_mixed24<>+1016(SB)/8, $127
