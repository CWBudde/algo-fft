//go:build arm64 && !purego

// ===========================================================================
// NEON Size-2048 Mixed-Radix (Radix-4 + Radix-2) FFT Kernels for ARM64 (complex64)
// ===========================================================================
//
// Size 2048 = 2 * 4^5, mixed-radix algorithm:
//   Stage 1: 512 radix-4 butterflies (no twiddles), stride=4
//   Stage 2: radix-4 with twiddles, 128 groups of 4, step=128
//   Stage 3: radix-4 with twiddles, 32 groups of 16, step=32
//   Stage 4: radix-4 with twiddles, 8 groups of 64, step=8
//   Stage 5: radix-4 with twiddles, 2 groups of 256, step=2
//   Stage 6: radix-2 with twiddles, 1024 butterflies, step=1
//
// Each complex64 element is 8 bytes (real f32 + imag f32). NEON idioms follow
// neon_f32_size128_mixed24.s (scalar FMOVS/FADDS/FSUBS/FMULS/FNEGS on the
// D/S register file), scaled from n=128=2*4^3 to n=2048=2*4^5.
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 2048, complex64, mixed radix (radix-4 x5, radix-2).
// func ForwardNEONSize2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool
TEXT ·ForwardNEONSize2048Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $2048, R13
	BNE  neon2048m24_return_false

	MOVD dst_len+8(FP), R0
	CMP  $2048, R0
	BLT  neon2048m24_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $2048, R0
	BLT  neon2048m24_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $2048, R0
	BLT  neon2048m24_return_false

	MOVD $bitrev_size2048_mixed24<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon2048m24_use_dst
	MOVD R11, R8

neon2048m24_use_dst:
	// Bit-reversal permutation (mixed-radix 2/4)
	MOVD $0, R0

neon2048m24_bitrev_loop:
	CMP  $2048, R0
	BGE  neon2048m24_stage1

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
	B    neon2048m24_bitrev_loop

neon2048m24_stage1:
	// Stage 1: 512 radix-4 butterflies (no twiddles)
	MOVD $0, R14

neon2048m24_stage1_loop:
	CMP  $2048, R14
	BGE  neon2048m24_stage2

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
	B    neon2048m24_stage1_loop

neon2048m24_stage2:
	// Stage 2: radix-4, size=16, step=128
	MOVD $0, R14

neon2048m24_stage2_base:
	CMP  $2048, R14
	BGE  neon2048m24_stage3

	MOVD $0, R15

neon2048m24_stage2_j:
	CMP  $4, R15
	BGE  neon2048m24_stage2_next

	ADD  R14, R15, R0       // idx0
	ADD  $4, R0, R1        // idx1
	ADD  $8, R0, R2        // idx2
	ADD  $12, R0, R3        // idx3

	LSL  $7, R15, R4        // j*128
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1

	LSL  $8, R15, R4        // j*256
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3

	LSL  $7, R15, R6        // j*128
	LSL  $8, R15, R4        // j*256
	ADD  R4, R6, R6         // j*384
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
	B    neon2048m24_stage2_j

neon2048m24_stage2_next:
	ADD  $16, R14, R14
	B    neon2048m24_stage2_base

neon2048m24_stage3:
	// Stage 3: radix-4, size=64, step=32
	MOVD $0, R14

neon2048m24_stage3_base:
	CMP  $2048, R14
	BGE  neon2048m24_stage4

	MOVD $0, R15

neon2048m24_stage3_j:
	CMP  $16, R15
	BGE  neon2048m24_stage3_next

	ADD  R14, R15, R0       // idx0
	ADD  $16, R0, R1        // idx1
	ADD  $32, R0, R2        // idx2
	ADD  $48, R0, R3        // idx3

	LSL  $5, R15, R4        // j*32
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1

	LSL  $6, R15, R4        // j*64
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3

	LSL  $5, R15, R6        // j*32
	LSL  $6, R15, R4        // j*64
	ADD  R4, R6, R6         // j*96
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
	B    neon2048m24_stage3_j

neon2048m24_stage3_next:
	ADD  $64, R14, R14
	B    neon2048m24_stage3_base

neon2048m24_stage4:
	// Stage 4: radix-4, size=256, step=8
	MOVD $0, R14

neon2048m24_stage4_base:
	CMP  $2048, R14
	BGE  neon2048m24_stage5

	MOVD $0, R15

neon2048m24_stage4_j:
	CMP  $64, R15
	BGE  neon2048m24_stage4_next

	ADD  R14, R15, R0       // idx0
	ADD  $64, R0, R1        // idx1
	ADD  $128, R0, R2        // idx2
	ADD  $192, R0, R3        // idx3

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
	B    neon2048m24_stage4_j

neon2048m24_stage4_next:
	ADD  $256, R14, R14
	B    neon2048m24_stage4_base

neon2048m24_stage5:
	// Stage 5: radix-4, size=1024, step=2
	MOVD $0, R14

neon2048m24_stage5_base:
	CMP  $2048, R14
	BGE  neon2048m24_stage6

	MOVD $0, R15

neon2048m24_stage5_j:
	CMP  $256, R15
	BGE  neon2048m24_stage5_next

	ADD  R14, R15, R0       // idx0
	ADD  $256, R0, R1        // idx1
	ADD  $512, R0, R2        // idx2
	ADD  $768, R0, R3        // idx3

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
	B    neon2048m24_stage5_j

neon2048m24_stage5_next:
	ADD  $1024, R14, R14
	B    neon2048m24_stage5_base

neon2048m24_stage6:
	// Stage 6: radix-2, size=2048, step=1
	MOVD $0, R0

neon2048m24_stage6_loop:
	CMP  $1024, R0
	BGE  neon2048m24_done

	ADD  $1024, R0, R1

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
	B    neon2048m24_stage6_loop

neon2048m24_done:
	CMP  R8, R20
	BEQ  neon2048m24_return_true

	MOVD $0, R0
neon2048m24_copy_loop:
	CMP  $2048, R0
	BGE  neon2048m24_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon2048m24_copy_loop

neon2048m24_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon2048m24_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 2048, complex64, mixed radix (radix-4 x5, radix-2).
// func InverseNEONSize2048Radix4Then2Complex64Asm(dst, src, twiddle, scratch []complex64) bool
TEXT ·InverseNEONSize2048Radix4Then2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $2048, R13
	BNE  neon2048m24_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $2048, R0
	BLT  neon2048m24_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $2048, R0
	BLT  neon2048m24_inv_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $2048, R0
	BLT  neon2048m24_inv_return_false

	MOVD $bitrev_size2048_mixed24<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon2048m24_inv_use_dst
	MOVD R11, R8

neon2048m24_inv_use_dst:
	// Bit-reversal permutation (mixed-radix 2/4)
	MOVD $0, R0

neon2048m24_inv_bitrev_loop:
	CMP  $2048, R0
	BGE  neon2048m24_inv_stage1

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
	B    neon2048m24_inv_bitrev_loop

neon2048m24_inv_stage1:
	// Stage 1: 512 radix-4 butterflies (no twiddles)
	MOVD $0, R14

neon2048m24_inv_stage1_loop:
	CMP  $2048, R14
	BGE  neon2048m24_inv_stage2

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

	// Inverse radix-4 butterfly (conjugated ±i): X1 = s1 + i*s3, X3 = s1 - i*s3.
	FNEGS F15, F20
	FMOVS F14, F21
	FADDS F20, F10, F22          // (F22,F23) = s1 + i*s3 = X1
	FADDS F21, F11, F23

	FMOVS F15, F24
	FNEGS F14, F25
	FADDS F24, F10, F26          // (F26,F27) = s1 - i*s3 = X3
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
	B    neon2048m24_inv_stage1_loop

neon2048m24_inv_stage2:
	// Stage 2: radix-4, size=16, step=128
	MOVD $0, R14

neon2048m24_inv_stage2_base:
	CMP  $2048, R14
	BGE  neon2048m24_inv_stage3

	MOVD $0, R15

neon2048m24_inv_stage2_j:
	CMP  $4, R15
	BGE  neon2048m24_inv_stage2_next

	ADD  R14, R15, R0       // idx0
	ADD  $4, R0, R1        // idx1
	ADD  $8, R0, R2        // idx2
	ADD  $12, R0, R3        // idx3

	LSL  $7, R15, R4        // j*128
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1
	FNEGS F1, F1

	LSL  $8, R15, R4        // j*256
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3
	FNEGS F3, F3

	LSL  $7, R15, R6        // j*128
	LSL  $8, R15, R4        // j*256
	ADD  R4, R6, R6         // j*384
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
	B    neon2048m24_inv_stage2_j

neon2048m24_inv_stage2_next:
	ADD  $16, R14, R14
	B    neon2048m24_inv_stage2_base

neon2048m24_inv_stage3:
	// Stage 3: radix-4, size=64, step=32
	MOVD $0, R14

neon2048m24_inv_stage3_base:
	CMP  $2048, R14
	BGE  neon2048m24_inv_stage4

	MOVD $0, R15

neon2048m24_inv_stage3_j:
	CMP  $16, R15
	BGE  neon2048m24_inv_stage3_next

	ADD  R14, R15, R0       // idx0
	ADD  $16, R0, R1        // idx1
	ADD  $32, R0, R2        // idx2
	ADD  $48, R0, R3        // idx3

	LSL  $5, R15, R4        // j*32
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1
	FNEGS F1, F1

	LSL  $6, R15, R4        // j*64
	LSL  $3, R4, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F2
	FMOVS 4(R5), F3
	FNEGS F3, F3

	LSL  $5, R15, R6        // j*32
	LSL  $6, R15, R4        // j*64
	ADD  R4, R6, R6         // j*96
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
	B    neon2048m24_inv_stage3_j

neon2048m24_inv_stage3_next:
	ADD  $64, R14, R14
	B    neon2048m24_inv_stage3_base

neon2048m24_inv_stage4:
	// Stage 4: radix-4, size=256, step=8
	MOVD $0, R14

neon2048m24_inv_stage4_base:
	CMP  $2048, R14
	BGE  neon2048m24_inv_stage5

	MOVD $0, R15

neon2048m24_inv_stage4_j:
	CMP  $64, R15
	BGE  neon2048m24_inv_stage4_next

	ADD  R14, R15, R0       // idx0
	ADD  $64, R0, R1        // idx1
	ADD  $128, R0, R2        // idx2
	ADD  $192, R0, R3        // idx3

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
	B    neon2048m24_inv_stage4_j

neon2048m24_inv_stage4_next:
	ADD  $256, R14, R14
	B    neon2048m24_inv_stage4_base

neon2048m24_inv_stage5:
	// Stage 5: radix-4, size=1024, step=2
	MOVD $0, R14

neon2048m24_inv_stage5_base:
	CMP  $2048, R14
	BGE  neon2048m24_inv_stage6

	MOVD $0, R15

neon2048m24_inv_stage5_j:
	CMP  $256, R15
	BGE  neon2048m24_inv_stage5_next

	ADD  R14, R15, R0       // idx0
	ADD  $256, R0, R1        // idx1
	ADD  $512, R0, R2        // idx2
	ADD  $768, R0, R3        // idx3

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
	B    neon2048m24_inv_stage5_j

neon2048m24_inv_stage5_next:
	ADD  $1024, R14, R14
	B    neon2048m24_inv_stage5_base

neon2048m24_inv_stage6:
	// Stage 6: radix-2, size=2048, step=1
	MOVD $0, R0

neon2048m24_inv_stage6_loop:
	CMP  $1024, R0
	BGE  neon2048m24_inv_scale

	ADD  $1024, R0, R1

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
	B    neon2048m24_inv_stage6_loop

neon2048m24_inv_scale:
	CMP  R8, R20
	BEQ  neon2048m24_inv_scale_apply

	MOVD $0, R0
neon2048m24_inv_copy_loop:
	CMP  $2048, R0
	BGE  neon2048m24_inv_scale_apply
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon2048m24_inv_copy_loop

neon2048m24_inv_scale_apply:
	MOVD $neonInv2048<>(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon2048m24_inv_scale_loop:
	CMP  $2048, R0
	BGE  neon2048m24_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon2048m24_inv_scale_loop

neon2048m24_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon2048m24_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Bit-reversal table for size 2048 mixed-radix 4,4,4,4,4,2
// ===========================================================================
GLOBL bitrev_size2048_mixed24<>(SB), RODATA, $16384
DATA bitrev_size2048_mixed24<>+0x000(SB)/8, $0
DATA bitrev_size2048_mixed24<>+0x008(SB)/8, $512
DATA bitrev_size2048_mixed24<>+0x010(SB)/8, $1024
DATA bitrev_size2048_mixed24<>+0x018(SB)/8, $1536
DATA bitrev_size2048_mixed24<>+0x020(SB)/8, $128
DATA bitrev_size2048_mixed24<>+0x028(SB)/8, $640
DATA bitrev_size2048_mixed24<>+0x030(SB)/8, $1152
DATA bitrev_size2048_mixed24<>+0x038(SB)/8, $1664
DATA bitrev_size2048_mixed24<>+0x040(SB)/8, $256
DATA bitrev_size2048_mixed24<>+0x048(SB)/8, $768
DATA bitrev_size2048_mixed24<>+0x050(SB)/8, $1280
DATA bitrev_size2048_mixed24<>+0x058(SB)/8, $1792
DATA bitrev_size2048_mixed24<>+0x060(SB)/8, $384
DATA bitrev_size2048_mixed24<>+0x068(SB)/8, $896
DATA bitrev_size2048_mixed24<>+0x070(SB)/8, $1408
DATA bitrev_size2048_mixed24<>+0x078(SB)/8, $1920
DATA bitrev_size2048_mixed24<>+0x080(SB)/8, $32
DATA bitrev_size2048_mixed24<>+0x088(SB)/8, $544
DATA bitrev_size2048_mixed24<>+0x090(SB)/8, $1056
DATA bitrev_size2048_mixed24<>+0x098(SB)/8, $1568
DATA bitrev_size2048_mixed24<>+0x0A0(SB)/8, $160
DATA bitrev_size2048_mixed24<>+0x0A8(SB)/8, $672
DATA bitrev_size2048_mixed24<>+0x0B0(SB)/8, $1184
DATA bitrev_size2048_mixed24<>+0x0B8(SB)/8, $1696
DATA bitrev_size2048_mixed24<>+0x0C0(SB)/8, $288
DATA bitrev_size2048_mixed24<>+0x0C8(SB)/8, $800
DATA bitrev_size2048_mixed24<>+0x0D0(SB)/8, $1312
DATA bitrev_size2048_mixed24<>+0x0D8(SB)/8, $1824
DATA bitrev_size2048_mixed24<>+0x0E0(SB)/8, $416
DATA bitrev_size2048_mixed24<>+0x0E8(SB)/8, $928
DATA bitrev_size2048_mixed24<>+0x0F0(SB)/8, $1440
DATA bitrev_size2048_mixed24<>+0x0F8(SB)/8, $1952
DATA bitrev_size2048_mixed24<>+0x100(SB)/8, $64
DATA bitrev_size2048_mixed24<>+0x108(SB)/8, $576
DATA bitrev_size2048_mixed24<>+0x110(SB)/8, $1088
DATA bitrev_size2048_mixed24<>+0x118(SB)/8, $1600
DATA bitrev_size2048_mixed24<>+0x120(SB)/8, $192
DATA bitrev_size2048_mixed24<>+0x128(SB)/8, $704
DATA bitrev_size2048_mixed24<>+0x130(SB)/8, $1216
DATA bitrev_size2048_mixed24<>+0x138(SB)/8, $1728
DATA bitrev_size2048_mixed24<>+0x140(SB)/8, $320
DATA bitrev_size2048_mixed24<>+0x148(SB)/8, $832
DATA bitrev_size2048_mixed24<>+0x150(SB)/8, $1344
DATA bitrev_size2048_mixed24<>+0x158(SB)/8, $1856
DATA bitrev_size2048_mixed24<>+0x160(SB)/8, $448
DATA bitrev_size2048_mixed24<>+0x168(SB)/8, $960
DATA bitrev_size2048_mixed24<>+0x170(SB)/8, $1472
DATA bitrev_size2048_mixed24<>+0x178(SB)/8, $1984
DATA bitrev_size2048_mixed24<>+0x180(SB)/8, $96
DATA bitrev_size2048_mixed24<>+0x188(SB)/8, $608
DATA bitrev_size2048_mixed24<>+0x190(SB)/8, $1120
DATA bitrev_size2048_mixed24<>+0x198(SB)/8, $1632
DATA bitrev_size2048_mixed24<>+0x1A0(SB)/8, $224
DATA bitrev_size2048_mixed24<>+0x1A8(SB)/8, $736
DATA bitrev_size2048_mixed24<>+0x1B0(SB)/8, $1248
DATA bitrev_size2048_mixed24<>+0x1B8(SB)/8, $1760
DATA bitrev_size2048_mixed24<>+0x1C0(SB)/8, $352
DATA bitrev_size2048_mixed24<>+0x1C8(SB)/8, $864
DATA bitrev_size2048_mixed24<>+0x1D0(SB)/8, $1376
DATA bitrev_size2048_mixed24<>+0x1D8(SB)/8, $1888
DATA bitrev_size2048_mixed24<>+0x1E0(SB)/8, $480
DATA bitrev_size2048_mixed24<>+0x1E8(SB)/8, $992
DATA bitrev_size2048_mixed24<>+0x1F0(SB)/8, $1504
DATA bitrev_size2048_mixed24<>+0x1F8(SB)/8, $2016
DATA bitrev_size2048_mixed24<>+0x200(SB)/8, $8
DATA bitrev_size2048_mixed24<>+0x208(SB)/8, $520
DATA bitrev_size2048_mixed24<>+0x210(SB)/8, $1032
DATA bitrev_size2048_mixed24<>+0x218(SB)/8, $1544
DATA bitrev_size2048_mixed24<>+0x220(SB)/8, $136
DATA bitrev_size2048_mixed24<>+0x228(SB)/8, $648
DATA bitrev_size2048_mixed24<>+0x230(SB)/8, $1160
DATA bitrev_size2048_mixed24<>+0x238(SB)/8, $1672
DATA bitrev_size2048_mixed24<>+0x240(SB)/8, $264
DATA bitrev_size2048_mixed24<>+0x248(SB)/8, $776
DATA bitrev_size2048_mixed24<>+0x250(SB)/8, $1288
DATA bitrev_size2048_mixed24<>+0x258(SB)/8, $1800
DATA bitrev_size2048_mixed24<>+0x260(SB)/8, $392
DATA bitrev_size2048_mixed24<>+0x268(SB)/8, $904
DATA bitrev_size2048_mixed24<>+0x270(SB)/8, $1416
DATA bitrev_size2048_mixed24<>+0x278(SB)/8, $1928
DATA bitrev_size2048_mixed24<>+0x280(SB)/8, $40
DATA bitrev_size2048_mixed24<>+0x288(SB)/8, $552
DATA bitrev_size2048_mixed24<>+0x290(SB)/8, $1064
DATA bitrev_size2048_mixed24<>+0x298(SB)/8, $1576
DATA bitrev_size2048_mixed24<>+0x2A0(SB)/8, $168
DATA bitrev_size2048_mixed24<>+0x2A8(SB)/8, $680
DATA bitrev_size2048_mixed24<>+0x2B0(SB)/8, $1192
DATA bitrev_size2048_mixed24<>+0x2B8(SB)/8, $1704
DATA bitrev_size2048_mixed24<>+0x2C0(SB)/8, $296
DATA bitrev_size2048_mixed24<>+0x2C8(SB)/8, $808
DATA bitrev_size2048_mixed24<>+0x2D0(SB)/8, $1320
DATA bitrev_size2048_mixed24<>+0x2D8(SB)/8, $1832
DATA bitrev_size2048_mixed24<>+0x2E0(SB)/8, $424
DATA bitrev_size2048_mixed24<>+0x2E8(SB)/8, $936
DATA bitrev_size2048_mixed24<>+0x2F0(SB)/8, $1448
DATA bitrev_size2048_mixed24<>+0x2F8(SB)/8, $1960
DATA bitrev_size2048_mixed24<>+0x300(SB)/8, $72
DATA bitrev_size2048_mixed24<>+0x308(SB)/8, $584
DATA bitrev_size2048_mixed24<>+0x310(SB)/8, $1096
DATA bitrev_size2048_mixed24<>+0x318(SB)/8, $1608
DATA bitrev_size2048_mixed24<>+0x320(SB)/8, $200
DATA bitrev_size2048_mixed24<>+0x328(SB)/8, $712
DATA bitrev_size2048_mixed24<>+0x330(SB)/8, $1224
DATA bitrev_size2048_mixed24<>+0x338(SB)/8, $1736
DATA bitrev_size2048_mixed24<>+0x340(SB)/8, $328
DATA bitrev_size2048_mixed24<>+0x348(SB)/8, $840
DATA bitrev_size2048_mixed24<>+0x350(SB)/8, $1352
DATA bitrev_size2048_mixed24<>+0x358(SB)/8, $1864
DATA bitrev_size2048_mixed24<>+0x360(SB)/8, $456
DATA bitrev_size2048_mixed24<>+0x368(SB)/8, $968
DATA bitrev_size2048_mixed24<>+0x370(SB)/8, $1480
DATA bitrev_size2048_mixed24<>+0x378(SB)/8, $1992
DATA bitrev_size2048_mixed24<>+0x380(SB)/8, $104
DATA bitrev_size2048_mixed24<>+0x388(SB)/8, $616
DATA bitrev_size2048_mixed24<>+0x390(SB)/8, $1128
DATA bitrev_size2048_mixed24<>+0x398(SB)/8, $1640
DATA bitrev_size2048_mixed24<>+0x3A0(SB)/8, $232
DATA bitrev_size2048_mixed24<>+0x3A8(SB)/8, $744
DATA bitrev_size2048_mixed24<>+0x3B0(SB)/8, $1256
DATA bitrev_size2048_mixed24<>+0x3B8(SB)/8, $1768
DATA bitrev_size2048_mixed24<>+0x3C0(SB)/8, $360
DATA bitrev_size2048_mixed24<>+0x3C8(SB)/8, $872
DATA bitrev_size2048_mixed24<>+0x3D0(SB)/8, $1384
DATA bitrev_size2048_mixed24<>+0x3D8(SB)/8, $1896
DATA bitrev_size2048_mixed24<>+0x3E0(SB)/8, $488
DATA bitrev_size2048_mixed24<>+0x3E8(SB)/8, $1000
DATA bitrev_size2048_mixed24<>+0x3F0(SB)/8, $1512
DATA bitrev_size2048_mixed24<>+0x3F8(SB)/8, $2024
DATA bitrev_size2048_mixed24<>+0x400(SB)/8, $16
DATA bitrev_size2048_mixed24<>+0x408(SB)/8, $528
DATA bitrev_size2048_mixed24<>+0x410(SB)/8, $1040
DATA bitrev_size2048_mixed24<>+0x418(SB)/8, $1552
DATA bitrev_size2048_mixed24<>+0x420(SB)/8, $144
DATA bitrev_size2048_mixed24<>+0x428(SB)/8, $656
DATA bitrev_size2048_mixed24<>+0x430(SB)/8, $1168
DATA bitrev_size2048_mixed24<>+0x438(SB)/8, $1680
DATA bitrev_size2048_mixed24<>+0x440(SB)/8, $272
DATA bitrev_size2048_mixed24<>+0x448(SB)/8, $784
DATA bitrev_size2048_mixed24<>+0x450(SB)/8, $1296
DATA bitrev_size2048_mixed24<>+0x458(SB)/8, $1808
DATA bitrev_size2048_mixed24<>+0x460(SB)/8, $400
DATA bitrev_size2048_mixed24<>+0x468(SB)/8, $912
DATA bitrev_size2048_mixed24<>+0x470(SB)/8, $1424
DATA bitrev_size2048_mixed24<>+0x478(SB)/8, $1936
DATA bitrev_size2048_mixed24<>+0x480(SB)/8, $48
DATA bitrev_size2048_mixed24<>+0x488(SB)/8, $560
DATA bitrev_size2048_mixed24<>+0x490(SB)/8, $1072
DATA bitrev_size2048_mixed24<>+0x498(SB)/8, $1584
DATA bitrev_size2048_mixed24<>+0x4A0(SB)/8, $176
DATA bitrev_size2048_mixed24<>+0x4A8(SB)/8, $688
DATA bitrev_size2048_mixed24<>+0x4B0(SB)/8, $1200
DATA bitrev_size2048_mixed24<>+0x4B8(SB)/8, $1712
DATA bitrev_size2048_mixed24<>+0x4C0(SB)/8, $304
DATA bitrev_size2048_mixed24<>+0x4C8(SB)/8, $816
DATA bitrev_size2048_mixed24<>+0x4D0(SB)/8, $1328
DATA bitrev_size2048_mixed24<>+0x4D8(SB)/8, $1840
DATA bitrev_size2048_mixed24<>+0x4E0(SB)/8, $432
DATA bitrev_size2048_mixed24<>+0x4E8(SB)/8, $944
DATA bitrev_size2048_mixed24<>+0x4F0(SB)/8, $1456
DATA bitrev_size2048_mixed24<>+0x4F8(SB)/8, $1968
DATA bitrev_size2048_mixed24<>+0x500(SB)/8, $80
DATA bitrev_size2048_mixed24<>+0x508(SB)/8, $592
DATA bitrev_size2048_mixed24<>+0x510(SB)/8, $1104
DATA bitrev_size2048_mixed24<>+0x518(SB)/8, $1616
DATA bitrev_size2048_mixed24<>+0x520(SB)/8, $208
DATA bitrev_size2048_mixed24<>+0x528(SB)/8, $720
DATA bitrev_size2048_mixed24<>+0x530(SB)/8, $1232
DATA bitrev_size2048_mixed24<>+0x538(SB)/8, $1744
DATA bitrev_size2048_mixed24<>+0x540(SB)/8, $336
DATA bitrev_size2048_mixed24<>+0x548(SB)/8, $848
DATA bitrev_size2048_mixed24<>+0x550(SB)/8, $1360
DATA bitrev_size2048_mixed24<>+0x558(SB)/8, $1872
DATA bitrev_size2048_mixed24<>+0x560(SB)/8, $464
DATA bitrev_size2048_mixed24<>+0x568(SB)/8, $976
DATA bitrev_size2048_mixed24<>+0x570(SB)/8, $1488
DATA bitrev_size2048_mixed24<>+0x578(SB)/8, $2000
DATA bitrev_size2048_mixed24<>+0x580(SB)/8, $112
DATA bitrev_size2048_mixed24<>+0x588(SB)/8, $624
DATA bitrev_size2048_mixed24<>+0x590(SB)/8, $1136
DATA bitrev_size2048_mixed24<>+0x598(SB)/8, $1648
DATA bitrev_size2048_mixed24<>+0x5A0(SB)/8, $240
DATA bitrev_size2048_mixed24<>+0x5A8(SB)/8, $752
DATA bitrev_size2048_mixed24<>+0x5B0(SB)/8, $1264
DATA bitrev_size2048_mixed24<>+0x5B8(SB)/8, $1776
DATA bitrev_size2048_mixed24<>+0x5C0(SB)/8, $368
DATA bitrev_size2048_mixed24<>+0x5C8(SB)/8, $880
DATA bitrev_size2048_mixed24<>+0x5D0(SB)/8, $1392
DATA bitrev_size2048_mixed24<>+0x5D8(SB)/8, $1904
DATA bitrev_size2048_mixed24<>+0x5E0(SB)/8, $496
DATA bitrev_size2048_mixed24<>+0x5E8(SB)/8, $1008
DATA bitrev_size2048_mixed24<>+0x5F0(SB)/8, $1520
DATA bitrev_size2048_mixed24<>+0x5F8(SB)/8, $2032
DATA bitrev_size2048_mixed24<>+0x600(SB)/8, $24
DATA bitrev_size2048_mixed24<>+0x608(SB)/8, $536
DATA bitrev_size2048_mixed24<>+0x610(SB)/8, $1048
DATA bitrev_size2048_mixed24<>+0x618(SB)/8, $1560
DATA bitrev_size2048_mixed24<>+0x620(SB)/8, $152
DATA bitrev_size2048_mixed24<>+0x628(SB)/8, $664
DATA bitrev_size2048_mixed24<>+0x630(SB)/8, $1176
DATA bitrev_size2048_mixed24<>+0x638(SB)/8, $1688
DATA bitrev_size2048_mixed24<>+0x640(SB)/8, $280
DATA bitrev_size2048_mixed24<>+0x648(SB)/8, $792
DATA bitrev_size2048_mixed24<>+0x650(SB)/8, $1304
DATA bitrev_size2048_mixed24<>+0x658(SB)/8, $1816
DATA bitrev_size2048_mixed24<>+0x660(SB)/8, $408
DATA bitrev_size2048_mixed24<>+0x668(SB)/8, $920
DATA bitrev_size2048_mixed24<>+0x670(SB)/8, $1432
DATA bitrev_size2048_mixed24<>+0x678(SB)/8, $1944
DATA bitrev_size2048_mixed24<>+0x680(SB)/8, $56
DATA bitrev_size2048_mixed24<>+0x688(SB)/8, $568
DATA bitrev_size2048_mixed24<>+0x690(SB)/8, $1080
DATA bitrev_size2048_mixed24<>+0x698(SB)/8, $1592
DATA bitrev_size2048_mixed24<>+0x6A0(SB)/8, $184
DATA bitrev_size2048_mixed24<>+0x6A8(SB)/8, $696
DATA bitrev_size2048_mixed24<>+0x6B0(SB)/8, $1208
DATA bitrev_size2048_mixed24<>+0x6B8(SB)/8, $1720
DATA bitrev_size2048_mixed24<>+0x6C0(SB)/8, $312
DATA bitrev_size2048_mixed24<>+0x6C8(SB)/8, $824
DATA bitrev_size2048_mixed24<>+0x6D0(SB)/8, $1336
DATA bitrev_size2048_mixed24<>+0x6D8(SB)/8, $1848
DATA bitrev_size2048_mixed24<>+0x6E0(SB)/8, $440
DATA bitrev_size2048_mixed24<>+0x6E8(SB)/8, $952
DATA bitrev_size2048_mixed24<>+0x6F0(SB)/8, $1464
DATA bitrev_size2048_mixed24<>+0x6F8(SB)/8, $1976
DATA bitrev_size2048_mixed24<>+0x700(SB)/8, $88
DATA bitrev_size2048_mixed24<>+0x708(SB)/8, $600
DATA bitrev_size2048_mixed24<>+0x710(SB)/8, $1112
DATA bitrev_size2048_mixed24<>+0x718(SB)/8, $1624
DATA bitrev_size2048_mixed24<>+0x720(SB)/8, $216
DATA bitrev_size2048_mixed24<>+0x728(SB)/8, $728
DATA bitrev_size2048_mixed24<>+0x730(SB)/8, $1240
DATA bitrev_size2048_mixed24<>+0x738(SB)/8, $1752
DATA bitrev_size2048_mixed24<>+0x740(SB)/8, $344
DATA bitrev_size2048_mixed24<>+0x748(SB)/8, $856
DATA bitrev_size2048_mixed24<>+0x750(SB)/8, $1368
DATA bitrev_size2048_mixed24<>+0x758(SB)/8, $1880
DATA bitrev_size2048_mixed24<>+0x760(SB)/8, $472
DATA bitrev_size2048_mixed24<>+0x768(SB)/8, $984
DATA bitrev_size2048_mixed24<>+0x770(SB)/8, $1496
DATA bitrev_size2048_mixed24<>+0x778(SB)/8, $2008
DATA bitrev_size2048_mixed24<>+0x780(SB)/8, $120
DATA bitrev_size2048_mixed24<>+0x788(SB)/8, $632
DATA bitrev_size2048_mixed24<>+0x790(SB)/8, $1144
DATA bitrev_size2048_mixed24<>+0x798(SB)/8, $1656
DATA bitrev_size2048_mixed24<>+0x7A0(SB)/8, $248
DATA bitrev_size2048_mixed24<>+0x7A8(SB)/8, $760
DATA bitrev_size2048_mixed24<>+0x7B0(SB)/8, $1272
DATA bitrev_size2048_mixed24<>+0x7B8(SB)/8, $1784
DATA bitrev_size2048_mixed24<>+0x7C0(SB)/8, $376
DATA bitrev_size2048_mixed24<>+0x7C8(SB)/8, $888
DATA bitrev_size2048_mixed24<>+0x7D0(SB)/8, $1400
DATA bitrev_size2048_mixed24<>+0x7D8(SB)/8, $1912
DATA bitrev_size2048_mixed24<>+0x7E0(SB)/8, $504
DATA bitrev_size2048_mixed24<>+0x7E8(SB)/8, $1016
DATA bitrev_size2048_mixed24<>+0x7F0(SB)/8, $1528
DATA bitrev_size2048_mixed24<>+0x7F8(SB)/8, $2040
DATA bitrev_size2048_mixed24<>+0x800(SB)/8, $2
DATA bitrev_size2048_mixed24<>+0x808(SB)/8, $514
DATA bitrev_size2048_mixed24<>+0x810(SB)/8, $1026
DATA bitrev_size2048_mixed24<>+0x818(SB)/8, $1538
DATA bitrev_size2048_mixed24<>+0x820(SB)/8, $130
DATA bitrev_size2048_mixed24<>+0x828(SB)/8, $642
DATA bitrev_size2048_mixed24<>+0x830(SB)/8, $1154
DATA bitrev_size2048_mixed24<>+0x838(SB)/8, $1666
DATA bitrev_size2048_mixed24<>+0x840(SB)/8, $258
DATA bitrev_size2048_mixed24<>+0x848(SB)/8, $770
DATA bitrev_size2048_mixed24<>+0x850(SB)/8, $1282
DATA bitrev_size2048_mixed24<>+0x858(SB)/8, $1794
DATA bitrev_size2048_mixed24<>+0x860(SB)/8, $386
DATA bitrev_size2048_mixed24<>+0x868(SB)/8, $898
DATA bitrev_size2048_mixed24<>+0x870(SB)/8, $1410
DATA bitrev_size2048_mixed24<>+0x878(SB)/8, $1922
DATA bitrev_size2048_mixed24<>+0x880(SB)/8, $34
DATA bitrev_size2048_mixed24<>+0x888(SB)/8, $546
DATA bitrev_size2048_mixed24<>+0x890(SB)/8, $1058
DATA bitrev_size2048_mixed24<>+0x898(SB)/8, $1570
DATA bitrev_size2048_mixed24<>+0x8A0(SB)/8, $162
DATA bitrev_size2048_mixed24<>+0x8A8(SB)/8, $674
DATA bitrev_size2048_mixed24<>+0x8B0(SB)/8, $1186
DATA bitrev_size2048_mixed24<>+0x8B8(SB)/8, $1698
DATA bitrev_size2048_mixed24<>+0x8C0(SB)/8, $290
DATA bitrev_size2048_mixed24<>+0x8C8(SB)/8, $802
DATA bitrev_size2048_mixed24<>+0x8D0(SB)/8, $1314
DATA bitrev_size2048_mixed24<>+0x8D8(SB)/8, $1826
DATA bitrev_size2048_mixed24<>+0x8E0(SB)/8, $418
DATA bitrev_size2048_mixed24<>+0x8E8(SB)/8, $930
DATA bitrev_size2048_mixed24<>+0x8F0(SB)/8, $1442
DATA bitrev_size2048_mixed24<>+0x8F8(SB)/8, $1954
DATA bitrev_size2048_mixed24<>+0x900(SB)/8, $66
DATA bitrev_size2048_mixed24<>+0x908(SB)/8, $578
DATA bitrev_size2048_mixed24<>+0x910(SB)/8, $1090
DATA bitrev_size2048_mixed24<>+0x918(SB)/8, $1602
DATA bitrev_size2048_mixed24<>+0x920(SB)/8, $194
DATA bitrev_size2048_mixed24<>+0x928(SB)/8, $706
DATA bitrev_size2048_mixed24<>+0x930(SB)/8, $1218
DATA bitrev_size2048_mixed24<>+0x938(SB)/8, $1730
DATA bitrev_size2048_mixed24<>+0x940(SB)/8, $322
DATA bitrev_size2048_mixed24<>+0x948(SB)/8, $834
DATA bitrev_size2048_mixed24<>+0x950(SB)/8, $1346
DATA bitrev_size2048_mixed24<>+0x958(SB)/8, $1858
DATA bitrev_size2048_mixed24<>+0x960(SB)/8, $450
DATA bitrev_size2048_mixed24<>+0x968(SB)/8, $962
DATA bitrev_size2048_mixed24<>+0x970(SB)/8, $1474
DATA bitrev_size2048_mixed24<>+0x978(SB)/8, $1986
DATA bitrev_size2048_mixed24<>+0x980(SB)/8, $98
DATA bitrev_size2048_mixed24<>+0x988(SB)/8, $610
DATA bitrev_size2048_mixed24<>+0x990(SB)/8, $1122
DATA bitrev_size2048_mixed24<>+0x998(SB)/8, $1634
DATA bitrev_size2048_mixed24<>+0x9A0(SB)/8, $226
DATA bitrev_size2048_mixed24<>+0x9A8(SB)/8, $738
DATA bitrev_size2048_mixed24<>+0x9B0(SB)/8, $1250
DATA bitrev_size2048_mixed24<>+0x9B8(SB)/8, $1762
DATA bitrev_size2048_mixed24<>+0x9C0(SB)/8, $354
DATA bitrev_size2048_mixed24<>+0x9C8(SB)/8, $866
DATA bitrev_size2048_mixed24<>+0x9D0(SB)/8, $1378
DATA bitrev_size2048_mixed24<>+0x9D8(SB)/8, $1890
DATA bitrev_size2048_mixed24<>+0x9E0(SB)/8, $482
DATA bitrev_size2048_mixed24<>+0x9E8(SB)/8, $994
DATA bitrev_size2048_mixed24<>+0x9F0(SB)/8, $1506
DATA bitrev_size2048_mixed24<>+0x9F8(SB)/8, $2018
DATA bitrev_size2048_mixed24<>+0xA00(SB)/8, $10
DATA bitrev_size2048_mixed24<>+0xA08(SB)/8, $522
DATA bitrev_size2048_mixed24<>+0xA10(SB)/8, $1034
DATA bitrev_size2048_mixed24<>+0xA18(SB)/8, $1546
DATA bitrev_size2048_mixed24<>+0xA20(SB)/8, $138
DATA bitrev_size2048_mixed24<>+0xA28(SB)/8, $650
DATA bitrev_size2048_mixed24<>+0xA30(SB)/8, $1162
DATA bitrev_size2048_mixed24<>+0xA38(SB)/8, $1674
DATA bitrev_size2048_mixed24<>+0xA40(SB)/8, $266
DATA bitrev_size2048_mixed24<>+0xA48(SB)/8, $778
DATA bitrev_size2048_mixed24<>+0xA50(SB)/8, $1290
DATA bitrev_size2048_mixed24<>+0xA58(SB)/8, $1802
DATA bitrev_size2048_mixed24<>+0xA60(SB)/8, $394
DATA bitrev_size2048_mixed24<>+0xA68(SB)/8, $906
DATA bitrev_size2048_mixed24<>+0xA70(SB)/8, $1418
DATA bitrev_size2048_mixed24<>+0xA78(SB)/8, $1930
DATA bitrev_size2048_mixed24<>+0xA80(SB)/8, $42
DATA bitrev_size2048_mixed24<>+0xA88(SB)/8, $554
DATA bitrev_size2048_mixed24<>+0xA90(SB)/8, $1066
DATA bitrev_size2048_mixed24<>+0xA98(SB)/8, $1578
DATA bitrev_size2048_mixed24<>+0xAA0(SB)/8, $170
DATA bitrev_size2048_mixed24<>+0xAA8(SB)/8, $682
DATA bitrev_size2048_mixed24<>+0xAB0(SB)/8, $1194
DATA bitrev_size2048_mixed24<>+0xAB8(SB)/8, $1706
DATA bitrev_size2048_mixed24<>+0xAC0(SB)/8, $298
DATA bitrev_size2048_mixed24<>+0xAC8(SB)/8, $810
DATA bitrev_size2048_mixed24<>+0xAD0(SB)/8, $1322
DATA bitrev_size2048_mixed24<>+0xAD8(SB)/8, $1834
DATA bitrev_size2048_mixed24<>+0xAE0(SB)/8, $426
DATA bitrev_size2048_mixed24<>+0xAE8(SB)/8, $938
DATA bitrev_size2048_mixed24<>+0xAF0(SB)/8, $1450
DATA bitrev_size2048_mixed24<>+0xAF8(SB)/8, $1962
DATA bitrev_size2048_mixed24<>+0xB00(SB)/8, $74
DATA bitrev_size2048_mixed24<>+0xB08(SB)/8, $586
DATA bitrev_size2048_mixed24<>+0xB10(SB)/8, $1098
DATA bitrev_size2048_mixed24<>+0xB18(SB)/8, $1610
DATA bitrev_size2048_mixed24<>+0xB20(SB)/8, $202
DATA bitrev_size2048_mixed24<>+0xB28(SB)/8, $714
DATA bitrev_size2048_mixed24<>+0xB30(SB)/8, $1226
DATA bitrev_size2048_mixed24<>+0xB38(SB)/8, $1738
DATA bitrev_size2048_mixed24<>+0xB40(SB)/8, $330
DATA bitrev_size2048_mixed24<>+0xB48(SB)/8, $842
DATA bitrev_size2048_mixed24<>+0xB50(SB)/8, $1354
DATA bitrev_size2048_mixed24<>+0xB58(SB)/8, $1866
DATA bitrev_size2048_mixed24<>+0xB60(SB)/8, $458
DATA bitrev_size2048_mixed24<>+0xB68(SB)/8, $970
DATA bitrev_size2048_mixed24<>+0xB70(SB)/8, $1482
DATA bitrev_size2048_mixed24<>+0xB78(SB)/8, $1994
DATA bitrev_size2048_mixed24<>+0xB80(SB)/8, $106
DATA bitrev_size2048_mixed24<>+0xB88(SB)/8, $618
DATA bitrev_size2048_mixed24<>+0xB90(SB)/8, $1130
DATA bitrev_size2048_mixed24<>+0xB98(SB)/8, $1642
DATA bitrev_size2048_mixed24<>+0xBA0(SB)/8, $234
DATA bitrev_size2048_mixed24<>+0xBA8(SB)/8, $746
DATA bitrev_size2048_mixed24<>+0xBB0(SB)/8, $1258
DATA bitrev_size2048_mixed24<>+0xBB8(SB)/8, $1770
DATA bitrev_size2048_mixed24<>+0xBC0(SB)/8, $362
DATA bitrev_size2048_mixed24<>+0xBC8(SB)/8, $874
DATA bitrev_size2048_mixed24<>+0xBD0(SB)/8, $1386
DATA bitrev_size2048_mixed24<>+0xBD8(SB)/8, $1898
DATA bitrev_size2048_mixed24<>+0xBE0(SB)/8, $490
DATA bitrev_size2048_mixed24<>+0xBE8(SB)/8, $1002
DATA bitrev_size2048_mixed24<>+0xBF0(SB)/8, $1514
DATA bitrev_size2048_mixed24<>+0xBF8(SB)/8, $2026
DATA bitrev_size2048_mixed24<>+0xC00(SB)/8, $18
DATA bitrev_size2048_mixed24<>+0xC08(SB)/8, $530
DATA bitrev_size2048_mixed24<>+0xC10(SB)/8, $1042
DATA bitrev_size2048_mixed24<>+0xC18(SB)/8, $1554
DATA bitrev_size2048_mixed24<>+0xC20(SB)/8, $146
DATA bitrev_size2048_mixed24<>+0xC28(SB)/8, $658
DATA bitrev_size2048_mixed24<>+0xC30(SB)/8, $1170
DATA bitrev_size2048_mixed24<>+0xC38(SB)/8, $1682
DATA bitrev_size2048_mixed24<>+0xC40(SB)/8, $274
DATA bitrev_size2048_mixed24<>+0xC48(SB)/8, $786
DATA bitrev_size2048_mixed24<>+0xC50(SB)/8, $1298
DATA bitrev_size2048_mixed24<>+0xC58(SB)/8, $1810
DATA bitrev_size2048_mixed24<>+0xC60(SB)/8, $402
DATA bitrev_size2048_mixed24<>+0xC68(SB)/8, $914
DATA bitrev_size2048_mixed24<>+0xC70(SB)/8, $1426
DATA bitrev_size2048_mixed24<>+0xC78(SB)/8, $1938
DATA bitrev_size2048_mixed24<>+0xC80(SB)/8, $50
DATA bitrev_size2048_mixed24<>+0xC88(SB)/8, $562
DATA bitrev_size2048_mixed24<>+0xC90(SB)/8, $1074
DATA bitrev_size2048_mixed24<>+0xC98(SB)/8, $1586
DATA bitrev_size2048_mixed24<>+0xCA0(SB)/8, $178
DATA bitrev_size2048_mixed24<>+0xCA8(SB)/8, $690
DATA bitrev_size2048_mixed24<>+0xCB0(SB)/8, $1202
DATA bitrev_size2048_mixed24<>+0xCB8(SB)/8, $1714
DATA bitrev_size2048_mixed24<>+0xCC0(SB)/8, $306
DATA bitrev_size2048_mixed24<>+0xCC8(SB)/8, $818
DATA bitrev_size2048_mixed24<>+0xCD0(SB)/8, $1330
DATA bitrev_size2048_mixed24<>+0xCD8(SB)/8, $1842
DATA bitrev_size2048_mixed24<>+0xCE0(SB)/8, $434
DATA bitrev_size2048_mixed24<>+0xCE8(SB)/8, $946
DATA bitrev_size2048_mixed24<>+0xCF0(SB)/8, $1458
DATA bitrev_size2048_mixed24<>+0xCF8(SB)/8, $1970
DATA bitrev_size2048_mixed24<>+0xD00(SB)/8, $82
DATA bitrev_size2048_mixed24<>+0xD08(SB)/8, $594
DATA bitrev_size2048_mixed24<>+0xD10(SB)/8, $1106
DATA bitrev_size2048_mixed24<>+0xD18(SB)/8, $1618
DATA bitrev_size2048_mixed24<>+0xD20(SB)/8, $210
DATA bitrev_size2048_mixed24<>+0xD28(SB)/8, $722
DATA bitrev_size2048_mixed24<>+0xD30(SB)/8, $1234
DATA bitrev_size2048_mixed24<>+0xD38(SB)/8, $1746
DATA bitrev_size2048_mixed24<>+0xD40(SB)/8, $338
DATA bitrev_size2048_mixed24<>+0xD48(SB)/8, $850
DATA bitrev_size2048_mixed24<>+0xD50(SB)/8, $1362
DATA bitrev_size2048_mixed24<>+0xD58(SB)/8, $1874
DATA bitrev_size2048_mixed24<>+0xD60(SB)/8, $466
DATA bitrev_size2048_mixed24<>+0xD68(SB)/8, $978
DATA bitrev_size2048_mixed24<>+0xD70(SB)/8, $1490
DATA bitrev_size2048_mixed24<>+0xD78(SB)/8, $2002
DATA bitrev_size2048_mixed24<>+0xD80(SB)/8, $114
DATA bitrev_size2048_mixed24<>+0xD88(SB)/8, $626
DATA bitrev_size2048_mixed24<>+0xD90(SB)/8, $1138
DATA bitrev_size2048_mixed24<>+0xD98(SB)/8, $1650
DATA bitrev_size2048_mixed24<>+0xDA0(SB)/8, $242
DATA bitrev_size2048_mixed24<>+0xDA8(SB)/8, $754
DATA bitrev_size2048_mixed24<>+0xDB0(SB)/8, $1266
DATA bitrev_size2048_mixed24<>+0xDB8(SB)/8, $1778
DATA bitrev_size2048_mixed24<>+0xDC0(SB)/8, $370
DATA bitrev_size2048_mixed24<>+0xDC8(SB)/8, $882
DATA bitrev_size2048_mixed24<>+0xDD0(SB)/8, $1394
DATA bitrev_size2048_mixed24<>+0xDD8(SB)/8, $1906
DATA bitrev_size2048_mixed24<>+0xDE0(SB)/8, $498
DATA bitrev_size2048_mixed24<>+0xDE8(SB)/8, $1010
DATA bitrev_size2048_mixed24<>+0xDF0(SB)/8, $1522
DATA bitrev_size2048_mixed24<>+0xDF8(SB)/8, $2034
DATA bitrev_size2048_mixed24<>+0xE00(SB)/8, $26
DATA bitrev_size2048_mixed24<>+0xE08(SB)/8, $538
DATA bitrev_size2048_mixed24<>+0xE10(SB)/8, $1050
DATA bitrev_size2048_mixed24<>+0xE18(SB)/8, $1562
DATA bitrev_size2048_mixed24<>+0xE20(SB)/8, $154
DATA bitrev_size2048_mixed24<>+0xE28(SB)/8, $666
DATA bitrev_size2048_mixed24<>+0xE30(SB)/8, $1178
DATA bitrev_size2048_mixed24<>+0xE38(SB)/8, $1690
DATA bitrev_size2048_mixed24<>+0xE40(SB)/8, $282
DATA bitrev_size2048_mixed24<>+0xE48(SB)/8, $794
DATA bitrev_size2048_mixed24<>+0xE50(SB)/8, $1306
DATA bitrev_size2048_mixed24<>+0xE58(SB)/8, $1818
DATA bitrev_size2048_mixed24<>+0xE60(SB)/8, $410
DATA bitrev_size2048_mixed24<>+0xE68(SB)/8, $922
DATA bitrev_size2048_mixed24<>+0xE70(SB)/8, $1434
DATA bitrev_size2048_mixed24<>+0xE78(SB)/8, $1946
DATA bitrev_size2048_mixed24<>+0xE80(SB)/8, $58
DATA bitrev_size2048_mixed24<>+0xE88(SB)/8, $570
DATA bitrev_size2048_mixed24<>+0xE90(SB)/8, $1082
DATA bitrev_size2048_mixed24<>+0xE98(SB)/8, $1594
DATA bitrev_size2048_mixed24<>+0xEA0(SB)/8, $186
DATA bitrev_size2048_mixed24<>+0xEA8(SB)/8, $698
DATA bitrev_size2048_mixed24<>+0xEB0(SB)/8, $1210
DATA bitrev_size2048_mixed24<>+0xEB8(SB)/8, $1722
DATA bitrev_size2048_mixed24<>+0xEC0(SB)/8, $314
DATA bitrev_size2048_mixed24<>+0xEC8(SB)/8, $826
DATA bitrev_size2048_mixed24<>+0xED0(SB)/8, $1338
DATA bitrev_size2048_mixed24<>+0xED8(SB)/8, $1850
DATA bitrev_size2048_mixed24<>+0xEE0(SB)/8, $442
DATA bitrev_size2048_mixed24<>+0xEE8(SB)/8, $954
DATA bitrev_size2048_mixed24<>+0xEF0(SB)/8, $1466
DATA bitrev_size2048_mixed24<>+0xEF8(SB)/8, $1978
DATA bitrev_size2048_mixed24<>+0xF00(SB)/8, $90
DATA bitrev_size2048_mixed24<>+0xF08(SB)/8, $602
DATA bitrev_size2048_mixed24<>+0xF10(SB)/8, $1114
DATA bitrev_size2048_mixed24<>+0xF18(SB)/8, $1626
DATA bitrev_size2048_mixed24<>+0xF20(SB)/8, $218
DATA bitrev_size2048_mixed24<>+0xF28(SB)/8, $730
DATA bitrev_size2048_mixed24<>+0xF30(SB)/8, $1242
DATA bitrev_size2048_mixed24<>+0xF38(SB)/8, $1754
DATA bitrev_size2048_mixed24<>+0xF40(SB)/8, $346
DATA bitrev_size2048_mixed24<>+0xF48(SB)/8, $858
DATA bitrev_size2048_mixed24<>+0xF50(SB)/8, $1370
DATA bitrev_size2048_mixed24<>+0xF58(SB)/8, $1882
DATA bitrev_size2048_mixed24<>+0xF60(SB)/8, $474
DATA bitrev_size2048_mixed24<>+0xF68(SB)/8, $986
DATA bitrev_size2048_mixed24<>+0xF70(SB)/8, $1498
DATA bitrev_size2048_mixed24<>+0xF78(SB)/8, $2010
DATA bitrev_size2048_mixed24<>+0xF80(SB)/8, $122
DATA bitrev_size2048_mixed24<>+0xF88(SB)/8, $634
DATA bitrev_size2048_mixed24<>+0xF90(SB)/8, $1146
DATA bitrev_size2048_mixed24<>+0xF98(SB)/8, $1658
DATA bitrev_size2048_mixed24<>+0xFA0(SB)/8, $250
DATA bitrev_size2048_mixed24<>+0xFA8(SB)/8, $762
DATA bitrev_size2048_mixed24<>+0xFB0(SB)/8, $1274
DATA bitrev_size2048_mixed24<>+0xFB8(SB)/8, $1786
DATA bitrev_size2048_mixed24<>+0xFC0(SB)/8, $378
DATA bitrev_size2048_mixed24<>+0xFC8(SB)/8, $890
DATA bitrev_size2048_mixed24<>+0xFD0(SB)/8, $1402
DATA bitrev_size2048_mixed24<>+0xFD8(SB)/8, $1914
DATA bitrev_size2048_mixed24<>+0xFE0(SB)/8, $506
DATA bitrev_size2048_mixed24<>+0xFE8(SB)/8, $1018
DATA bitrev_size2048_mixed24<>+0xFF0(SB)/8, $1530
DATA bitrev_size2048_mixed24<>+0xFF8(SB)/8, $2042
DATA bitrev_size2048_mixed24<>+0x1000(SB)/8, $4
DATA bitrev_size2048_mixed24<>+0x1008(SB)/8, $516
DATA bitrev_size2048_mixed24<>+0x1010(SB)/8, $1028
DATA bitrev_size2048_mixed24<>+0x1018(SB)/8, $1540
DATA bitrev_size2048_mixed24<>+0x1020(SB)/8, $132
DATA bitrev_size2048_mixed24<>+0x1028(SB)/8, $644
DATA bitrev_size2048_mixed24<>+0x1030(SB)/8, $1156
DATA bitrev_size2048_mixed24<>+0x1038(SB)/8, $1668
DATA bitrev_size2048_mixed24<>+0x1040(SB)/8, $260
DATA bitrev_size2048_mixed24<>+0x1048(SB)/8, $772
DATA bitrev_size2048_mixed24<>+0x1050(SB)/8, $1284
DATA bitrev_size2048_mixed24<>+0x1058(SB)/8, $1796
DATA bitrev_size2048_mixed24<>+0x1060(SB)/8, $388
DATA bitrev_size2048_mixed24<>+0x1068(SB)/8, $900
DATA bitrev_size2048_mixed24<>+0x1070(SB)/8, $1412
DATA bitrev_size2048_mixed24<>+0x1078(SB)/8, $1924
DATA bitrev_size2048_mixed24<>+0x1080(SB)/8, $36
DATA bitrev_size2048_mixed24<>+0x1088(SB)/8, $548
DATA bitrev_size2048_mixed24<>+0x1090(SB)/8, $1060
DATA bitrev_size2048_mixed24<>+0x1098(SB)/8, $1572
DATA bitrev_size2048_mixed24<>+0x10A0(SB)/8, $164
DATA bitrev_size2048_mixed24<>+0x10A8(SB)/8, $676
DATA bitrev_size2048_mixed24<>+0x10B0(SB)/8, $1188
DATA bitrev_size2048_mixed24<>+0x10B8(SB)/8, $1700
DATA bitrev_size2048_mixed24<>+0x10C0(SB)/8, $292
DATA bitrev_size2048_mixed24<>+0x10C8(SB)/8, $804
DATA bitrev_size2048_mixed24<>+0x10D0(SB)/8, $1316
DATA bitrev_size2048_mixed24<>+0x10D8(SB)/8, $1828
DATA bitrev_size2048_mixed24<>+0x10E0(SB)/8, $420
DATA bitrev_size2048_mixed24<>+0x10E8(SB)/8, $932
DATA bitrev_size2048_mixed24<>+0x10F0(SB)/8, $1444
DATA bitrev_size2048_mixed24<>+0x10F8(SB)/8, $1956
DATA bitrev_size2048_mixed24<>+0x1100(SB)/8, $68
DATA bitrev_size2048_mixed24<>+0x1108(SB)/8, $580
DATA bitrev_size2048_mixed24<>+0x1110(SB)/8, $1092
DATA bitrev_size2048_mixed24<>+0x1118(SB)/8, $1604
DATA bitrev_size2048_mixed24<>+0x1120(SB)/8, $196
DATA bitrev_size2048_mixed24<>+0x1128(SB)/8, $708
DATA bitrev_size2048_mixed24<>+0x1130(SB)/8, $1220
DATA bitrev_size2048_mixed24<>+0x1138(SB)/8, $1732
DATA bitrev_size2048_mixed24<>+0x1140(SB)/8, $324
DATA bitrev_size2048_mixed24<>+0x1148(SB)/8, $836
DATA bitrev_size2048_mixed24<>+0x1150(SB)/8, $1348
DATA bitrev_size2048_mixed24<>+0x1158(SB)/8, $1860
DATA bitrev_size2048_mixed24<>+0x1160(SB)/8, $452
DATA bitrev_size2048_mixed24<>+0x1168(SB)/8, $964
DATA bitrev_size2048_mixed24<>+0x1170(SB)/8, $1476
DATA bitrev_size2048_mixed24<>+0x1178(SB)/8, $1988
DATA bitrev_size2048_mixed24<>+0x1180(SB)/8, $100
DATA bitrev_size2048_mixed24<>+0x1188(SB)/8, $612
DATA bitrev_size2048_mixed24<>+0x1190(SB)/8, $1124
DATA bitrev_size2048_mixed24<>+0x1198(SB)/8, $1636
DATA bitrev_size2048_mixed24<>+0x11A0(SB)/8, $228
DATA bitrev_size2048_mixed24<>+0x11A8(SB)/8, $740
DATA bitrev_size2048_mixed24<>+0x11B0(SB)/8, $1252
DATA bitrev_size2048_mixed24<>+0x11B8(SB)/8, $1764
DATA bitrev_size2048_mixed24<>+0x11C0(SB)/8, $356
DATA bitrev_size2048_mixed24<>+0x11C8(SB)/8, $868
DATA bitrev_size2048_mixed24<>+0x11D0(SB)/8, $1380
DATA bitrev_size2048_mixed24<>+0x11D8(SB)/8, $1892
DATA bitrev_size2048_mixed24<>+0x11E0(SB)/8, $484
DATA bitrev_size2048_mixed24<>+0x11E8(SB)/8, $996
DATA bitrev_size2048_mixed24<>+0x11F0(SB)/8, $1508
DATA bitrev_size2048_mixed24<>+0x11F8(SB)/8, $2020
DATA bitrev_size2048_mixed24<>+0x1200(SB)/8, $12
DATA bitrev_size2048_mixed24<>+0x1208(SB)/8, $524
DATA bitrev_size2048_mixed24<>+0x1210(SB)/8, $1036
DATA bitrev_size2048_mixed24<>+0x1218(SB)/8, $1548
DATA bitrev_size2048_mixed24<>+0x1220(SB)/8, $140
DATA bitrev_size2048_mixed24<>+0x1228(SB)/8, $652
DATA bitrev_size2048_mixed24<>+0x1230(SB)/8, $1164
DATA bitrev_size2048_mixed24<>+0x1238(SB)/8, $1676
DATA bitrev_size2048_mixed24<>+0x1240(SB)/8, $268
DATA bitrev_size2048_mixed24<>+0x1248(SB)/8, $780
DATA bitrev_size2048_mixed24<>+0x1250(SB)/8, $1292
DATA bitrev_size2048_mixed24<>+0x1258(SB)/8, $1804
DATA bitrev_size2048_mixed24<>+0x1260(SB)/8, $396
DATA bitrev_size2048_mixed24<>+0x1268(SB)/8, $908
DATA bitrev_size2048_mixed24<>+0x1270(SB)/8, $1420
DATA bitrev_size2048_mixed24<>+0x1278(SB)/8, $1932
DATA bitrev_size2048_mixed24<>+0x1280(SB)/8, $44
DATA bitrev_size2048_mixed24<>+0x1288(SB)/8, $556
DATA bitrev_size2048_mixed24<>+0x1290(SB)/8, $1068
DATA bitrev_size2048_mixed24<>+0x1298(SB)/8, $1580
DATA bitrev_size2048_mixed24<>+0x12A0(SB)/8, $172
DATA bitrev_size2048_mixed24<>+0x12A8(SB)/8, $684
DATA bitrev_size2048_mixed24<>+0x12B0(SB)/8, $1196
DATA bitrev_size2048_mixed24<>+0x12B8(SB)/8, $1708
DATA bitrev_size2048_mixed24<>+0x12C0(SB)/8, $300
DATA bitrev_size2048_mixed24<>+0x12C8(SB)/8, $812
DATA bitrev_size2048_mixed24<>+0x12D0(SB)/8, $1324
DATA bitrev_size2048_mixed24<>+0x12D8(SB)/8, $1836
DATA bitrev_size2048_mixed24<>+0x12E0(SB)/8, $428
DATA bitrev_size2048_mixed24<>+0x12E8(SB)/8, $940
DATA bitrev_size2048_mixed24<>+0x12F0(SB)/8, $1452
DATA bitrev_size2048_mixed24<>+0x12F8(SB)/8, $1964
DATA bitrev_size2048_mixed24<>+0x1300(SB)/8, $76
DATA bitrev_size2048_mixed24<>+0x1308(SB)/8, $588
DATA bitrev_size2048_mixed24<>+0x1310(SB)/8, $1100
DATA bitrev_size2048_mixed24<>+0x1318(SB)/8, $1612
DATA bitrev_size2048_mixed24<>+0x1320(SB)/8, $204
DATA bitrev_size2048_mixed24<>+0x1328(SB)/8, $716
DATA bitrev_size2048_mixed24<>+0x1330(SB)/8, $1228
DATA bitrev_size2048_mixed24<>+0x1338(SB)/8, $1740
DATA bitrev_size2048_mixed24<>+0x1340(SB)/8, $332
DATA bitrev_size2048_mixed24<>+0x1348(SB)/8, $844
DATA bitrev_size2048_mixed24<>+0x1350(SB)/8, $1356
DATA bitrev_size2048_mixed24<>+0x1358(SB)/8, $1868
DATA bitrev_size2048_mixed24<>+0x1360(SB)/8, $460
DATA bitrev_size2048_mixed24<>+0x1368(SB)/8, $972
DATA bitrev_size2048_mixed24<>+0x1370(SB)/8, $1484
DATA bitrev_size2048_mixed24<>+0x1378(SB)/8, $1996
DATA bitrev_size2048_mixed24<>+0x1380(SB)/8, $108
DATA bitrev_size2048_mixed24<>+0x1388(SB)/8, $620
DATA bitrev_size2048_mixed24<>+0x1390(SB)/8, $1132
DATA bitrev_size2048_mixed24<>+0x1398(SB)/8, $1644
DATA bitrev_size2048_mixed24<>+0x13A0(SB)/8, $236
DATA bitrev_size2048_mixed24<>+0x13A8(SB)/8, $748
DATA bitrev_size2048_mixed24<>+0x13B0(SB)/8, $1260
DATA bitrev_size2048_mixed24<>+0x13B8(SB)/8, $1772
DATA bitrev_size2048_mixed24<>+0x13C0(SB)/8, $364
DATA bitrev_size2048_mixed24<>+0x13C8(SB)/8, $876
DATA bitrev_size2048_mixed24<>+0x13D0(SB)/8, $1388
DATA bitrev_size2048_mixed24<>+0x13D8(SB)/8, $1900
DATA bitrev_size2048_mixed24<>+0x13E0(SB)/8, $492
DATA bitrev_size2048_mixed24<>+0x13E8(SB)/8, $1004
DATA bitrev_size2048_mixed24<>+0x13F0(SB)/8, $1516
DATA bitrev_size2048_mixed24<>+0x13F8(SB)/8, $2028
DATA bitrev_size2048_mixed24<>+0x1400(SB)/8, $20
DATA bitrev_size2048_mixed24<>+0x1408(SB)/8, $532
DATA bitrev_size2048_mixed24<>+0x1410(SB)/8, $1044
DATA bitrev_size2048_mixed24<>+0x1418(SB)/8, $1556
DATA bitrev_size2048_mixed24<>+0x1420(SB)/8, $148
DATA bitrev_size2048_mixed24<>+0x1428(SB)/8, $660
DATA bitrev_size2048_mixed24<>+0x1430(SB)/8, $1172
DATA bitrev_size2048_mixed24<>+0x1438(SB)/8, $1684
DATA bitrev_size2048_mixed24<>+0x1440(SB)/8, $276
DATA bitrev_size2048_mixed24<>+0x1448(SB)/8, $788
DATA bitrev_size2048_mixed24<>+0x1450(SB)/8, $1300
DATA bitrev_size2048_mixed24<>+0x1458(SB)/8, $1812
DATA bitrev_size2048_mixed24<>+0x1460(SB)/8, $404
DATA bitrev_size2048_mixed24<>+0x1468(SB)/8, $916
DATA bitrev_size2048_mixed24<>+0x1470(SB)/8, $1428
DATA bitrev_size2048_mixed24<>+0x1478(SB)/8, $1940
DATA bitrev_size2048_mixed24<>+0x1480(SB)/8, $52
DATA bitrev_size2048_mixed24<>+0x1488(SB)/8, $564
DATA bitrev_size2048_mixed24<>+0x1490(SB)/8, $1076
DATA bitrev_size2048_mixed24<>+0x1498(SB)/8, $1588
DATA bitrev_size2048_mixed24<>+0x14A0(SB)/8, $180
DATA bitrev_size2048_mixed24<>+0x14A8(SB)/8, $692
DATA bitrev_size2048_mixed24<>+0x14B0(SB)/8, $1204
DATA bitrev_size2048_mixed24<>+0x14B8(SB)/8, $1716
DATA bitrev_size2048_mixed24<>+0x14C0(SB)/8, $308
DATA bitrev_size2048_mixed24<>+0x14C8(SB)/8, $820
DATA bitrev_size2048_mixed24<>+0x14D0(SB)/8, $1332
DATA bitrev_size2048_mixed24<>+0x14D8(SB)/8, $1844
DATA bitrev_size2048_mixed24<>+0x14E0(SB)/8, $436
DATA bitrev_size2048_mixed24<>+0x14E8(SB)/8, $948
DATA bitrev_size2048_mixed24<>+0x14F0(SB)/8, $1460
DATA bitrev_size2048_mixed24<>+0x14F8(SB)/8, $1972
DATA bitrev_size2048_mixed24<>+0x1500(SB)/8, $84
DATA bitrev_size2048_mixed24<>+0x1508(SB)/8, $596
DATA bitrev_size2048_mixed24<>+0x1510(SB)/8, $1108
DATA bitrev_size2048_mixed24<>+0x1518(SB)/8, $1620
DATA bitrev_size2048_mixed24<>+0x1520(SB)/8, $212
DATA bitrev_size2048_mixed24<>+0x1528(SB)/8, $724
DATA bitrev_size2048_mixed24<>+0x1530(SB)/8, $1236
DATA bitrev_size2048_mixed24<>+0x1538(SB)/8, $1748
DATA bitrev_size2048_mixed24<>+0x1540(SB)/8, $340
DATA bitrev_size2048_mixed24<>+0x1548(SB)/8, $852
DATA bitrev_size2048_mixed24<>+0x1550(SB)/8, $1364
DATA bitrev_size2048_mixed24<>+0x1558(SB)/8, $1876
DATA bitrev_size2048_mixed24<>+0x1560(SB)/8, $468
DATA bitrev_size2048_mixed24<>+0x1568(SB)/8, $980
DATA bitrev_size2048_mixed24<>+0x1570(SB)/8, $1492
DATA bitrev_size2048_mixed24<>+0x1578(SB)/8, $2004
DATA bitrev_size2048_mixed24<>+0x1580(SB)/8, $116
DATA bitrev_size2048_mixed24<>+0x1588(SB)/8, $628
DATA bitrev_size2048_mixed24<>+0x1590(SB)/8, $1140
DATA bitrev_size2048_mixed24<>+0x1598(SB)/8, $1652
DATA bitrev_size2048_mixed24<>+0x15A0(SB)/8, $244
DATA bitrev_size2048_mixed24<>+0x15A8(SB)/8, $756
DATA bitrev_size2048_mixed24<>+0x15B0(SB)/8, $1268
DATA bitrev_size2048_mixed24<>+0x15B8(SB)/8, $1780
DATA bitrev_size2048_mixed24<>+0x15C0(SB)/8, $372
DATA bitrev_size2048_mixed24<>+0x15C8(SB)/8, $884
DATA bitrev_size2048_mixed24<>+0x15D0(SB)/8, $1396
DATA bitrev_size2048_mixed24<>+0x15D8(SB)/8, $1908
DATA bitrev_size2048_mixed24<>+0x15E0(SB)/8, $500
DATA bitrev_size2048_mixed24<>+0x15E8(SB)/8, $1012
DATA bitrev_size2048_mixed24<>+0x15F0(SB)/8, $1524
DATA bitrev_size2048_mixed24<>+0x15F8(SB)/8, $2036
DATA bitrev_size2048_mixed24<>+0x1600(SB)/8, $28
DATA bitrev_size2048_mixed24<>+0x1608(SB)/8, $540
DATA bitrev_size2048_mixed24<>+0x1610(SB)/8, $1052
DATA bitrev_size2048_mixed24<>+0x1618(SB)/8, $1564
DATA bitrev_size2048_mixed24<>+0x1620(SB)/8, $156
DATA bitrev_size2048_mixed24<>+0x1628(SB)/8, $668
DATA bitrev_size2048_mixed24<>+0x1630(SB)/8, $1180
DATA bitrev_size2048_mixed24<>+0x1638(SB)/8, $1692
DATA bitrev_size2048_mixed24<>+0x1640(SB)/8, $284
DATA bitrev_size2048_mixed24<>+0x1648(SB)/8, $796
DATA bitrev_size2048_mixed24<>+0x1650(SB)/8, $1308
DATA bitrev_size2048_mixed24<>+0x1658(SB)/8, $1820
DATA bitrev_size2048_mixed24<>+0x1660(SB)/8, $412
DATA bitrev_size2048_mixed24<>+0x1668(SB)/8, $924
DATA bitrev_size2048_mixed24<>+0x1670(SB)/8, $1436
DATA bitrev_size2048_mixed24<>+0x1678(SB)/8, $1948
DATA bitrev_size2048_mixed24<>+0x1680(SB)/8, $60
DATA bitrev_size2048_mixed24<>+0x1688(SB)/8, $572
DATA bitrev_size2048_mixed24<>+0x1690(SB)/8, $1084
DATA bitrev_size2048_mixed24<>+0x1698(SB)/8, $1596
DATA bitrev_size2048_mixed24<>+0x16A0(SB)/8, $188
DATA bitrev_size2048_mixed24<>+0x16A8(SB)/8, $700
DATA bitrev_size2048_mixed24<>+0x16B0(SB)/8, $1212
DATA bitrev_size2048_mixed24<>+0x16B8(SB)/8, $1724
DATA bitrev_size2048_mixed24<>+0x16C0(SB)/8, $316
DATA bitrev_size2048_mixed24<>+0x16C8(SB)/8, $828
DATA bitrev_size2048_mixed24<>+0x16D0(SB)/8, $1340
DATA bitrev_size2048_mixed24<>+0x16D8(SB)/8, $1852
DATA bitrev_size2048_mixed24<>+0x16E0(SB)/8, $444
DATA bitrev_size2048_mixed24<>+0x16E8(SB)/8, $956
DATA bitrev_size2048_mixed24<>+0x16F0(SB)/8, $1468
DATA bitrev_size2048_mixed24<>+0x16F8(SB)/8, $1980
DATA bitrev_size2048_mixed24<>+0x1700(SB)/8, $92
DATA bitrev_size2048_mixed24<>+0x1708(SB)/8, $604
DATA bitrev_size2048_mixed24<>+0x1710(SB)/8, $1116
DATA bitrev_size2048_mixed24<>+0x1718(SB)/8, $1628
DATA bitrev_size2048_mixed24<>+0x1720(SB)/8, $220
DATA bitrev_size2048_mixed24<>+0x1728(SB)/8, $732
DATA bitrev_size2048_mixed24<>+0x1730(SB)/8, $1244
DATA bitrev_size2048_mixed24<>+0x1738(SB)/8, $1756
DATA bitrev_size2048_mixed24<>+0x1740(SB)/8, $348
DATA bitrev_size2048_mixed24<>+0x1748(SB)/8, $860
DATA bitrev_size2048_mixed24<>+0x1750(SB)/8, $1372
DATA bitrev_size2048_mixed24<>+0x1758(SB)/8, $1884
DATA bitrev_size2048_mixed24<>+0x1760(SB)/8, $476
DATA bitrev_size2048_mixed24<>+0x1768(SB)/8, $988
DATA bitrev_size2048_mixed24<>+0x1770(SB)/8, $1500
DATA bitrev_size2048_mixed24<>+0x1778(SB)/8, $2012
DATA bitrev_size2048_mixed24<>+0x1780(SB)/8, $124
DATA bitrev_size2048_mixed24<>+0x1788(SB)/8, $636
DATA bitrev_size2048_mixed24<>+0x1790(SB)/8, $1148
DATA bitrev_size2048_mixed24<>+0x1798(SB)/8, $1660
DATA bitrev_size2048_mixed24<>+0x17A0(SB)/8, $252
DATA bitrev_size2048_mixed24<>+0x17A8(SB)/8, $764
DATA bitrev_size2048_mixed24<>+0x17B0(SB)/8, $1276
DATA bitrev_size2048_mixed24<>+0x17B8(SB)/8, $1788
DATA bitrev_size2048_mixed24<>+0x17C0(SB)/8, $380
DATA bitrev_size2048_mixed24<>+0x17C8(SB)/8, $892
DATA bitrev_size2048_mixed24<>+0x17D0(SB)/8, $1404
DATA bitrev_size2048_mixed24<>+0x17D8(SB)/8, $1916
DATA bitrev_size2048_mixed24<>+0x17E0(SB)/8, $508
DATA bitrev_size2048_mixed24<>+0x17E8(SB)/8, $1020
DATA bitrev_size2048_mixed24<>+0x17F0(SB)/8, $1532
DATA bitrev_size2048_mixed24<>+0x17F8(SB)/8, $2044
DATA bitrev_size2048_mixed24<>+0x1800(SB)/8, $6
DATA bitrev_size2048_mixed24<>+0x1808(SB)/8, $518
DATA bitrev_size2048_mixed24<>+0x1810(SB)/8, $1030
DATA bitrev_size2048_mixed24<>+0x1818(SB)/8, $1542
DATA bitrev_size2048_mixed24<>+0x1820(SB)/8, $134
DATA bitrev_size2048_mixed24<>+0x1828(SB)/8, $646
DATA bitrev_size2048_mixed24<>+0x1830(SB)/8, $1158
DATA bitrev_size2048_mixed24<>+0x1838(SB)/8, $1670
DATA bitrev_size2048_mixed24<>+0x1840(SB)/8, $262
DATA bitrev_size2048_mixed24<>+0x1848(SB)/8, $774
DATA bitrev_size2048_mixed24<>+0x1850(SB)/8, $1286
DATA bitrev_size2048_mixed24<>+0x1858(SB)/8, $1798
DATA bitrev_size2048_mixed24<>+0x1860(SB)/8, $390
DATA bitrev_size2048_mixed24<>+0x1868(SB)/8, $902
DATA bitrev_size2048_mixed24<>+0x1870(SB)/8, $1414
DATA bitrev_size2048_mixed24<>+0x1878(SB)/8, $1926
DATA bitrev_size2048_mixed24<>+0x1880(SB)/8, $38
DATA bitrev_size2048_mixed24<>+0x1888(SB)/8, $550
DATA bitrev_size2048_mixed24<>+0x1890(SB)/8, $1062
DATA bitrev_size2048_mixed24<>+0x1898(SB)/8, $1574
DATA bitrev_size2048_mixed24<>+0x18A0(SB)/8, $166
DATA bitrev_size2048_mixed24<>+0x18A8(SB)/8, $678
DATA bitrev_size2048_mixed24<>+0x18B0(SB)/8, $1190
DATA bitrev_size2048_mixed24<>+0x18B8(SB)/8, $1702
DATA bitrev_size2048_mixed24<>+0x18C0(SB)/8, $294
DATA bitrev_size2048_mixed24<>+0x18C8(SB)/8, $806
DATA bitrev_size2048_mixed24<>+0x18D0(SB)/8, $1318
DATA bitrev_size2048_mixed24<>+0x18D8(SB)/8, $1830
DATA bitrev_size2048_mixed24<>+0x18E0(SB)/8, $422
DATA bitrev_size2048_mixed24<>+0x18E8(SB)/8, $934
DATA bitrev_size2048_mixed24<>+0x18F0(SB)/8, $1446
DATA bitrev_size2048_mixed24<>+0x18F8(SB)/8, $1958
DATA bitrev_size2048_mixed24<>+0x1900(SB)/8, $70
DATA bitrev_size2048_mixed24<>+0x1908(SB)/8, $582
DATA bitrev_size2048_mixed24<>+0x1910(SB)/8, $1094
DATA bitrev_size2048_mixed24<>+0x1918(SB)/8, $1606
DATA bitrev_size2048_mixed24<>+0x1920(SB)/8, $198
DATA bitrev_size2048_mixed24<>+0x1928(SB)/8, $710
DATA bitrev_size2048_mixed24<>+0x1930(SB)/8, $1222
DATA bitrev_size2048_mixed24<>+0x1938(SB)/8, $1734
DATA bitrev_size2048_mixed24<>+0x1940(SB)/8, $326
DATA bitrev_size2048_mixed24<>+0x1948(SB)/8, $838
DATA bitrev_size2048_mixed24<>+0x1950(SB)/8, $1350
DATA bitrev_size2048_mixed24<>+0x1958(SB)/8, $1862
DATA bitrev_size2048_mixed24<>+0x1960(SB)/8, $454
DATA bitrev_size2048_mixed24<>+0x1968(SB)/8, $966
DATA bitrev_size2048_mixed24<>+0x1970(SB)/8, $1478
DATA bitrev_size2048_mixed24<>+0x1978(SB)/8, $1990
DATA bitrev_size2048_mixed24<>+0x1980(SB)/8, $102
DATA bitrev_size2048_mixed24<>+0x1988(SB)/8, $614
DATA bitrev_size2048_mixed24<>+0x1990(SB)/8, $1126
DATA bitrev_size2048_mixed24<>+0x1998(SB)/8, $1638
DATA bitrev_size2048_mixed24<>+0x19A0(SB)/8, $230
DATA bitrev_size2048_mixed24<>+0x19A8(SB)/8, $742
DATA bitrev_size2048_mixed24<>+0x19B0(SB)/8, $1254
DATA bitrev_size2048_mixed24<>+0x19B8(SB)/8, $1766
DATA bitrev_size2048_mixed24<>+0x19C0(SB)/8, $358
DATA bitrev_size2048_mixed24<>+0x19C8(SB)/8, $870
DATA bitrev_size2048_mixed24<>+0x19D0(SB)/8, $1382
DATA bitrev_size2048_mixed24<>+0x19D8(SB)/8, $1894
DATA bitrev_size2048_mixed24<>+0x19E0(SB)/8, $486
DATA bitrev_size2048_mixed24<>+0x19E8(SB)/8, $998
DATA bitrev_size2048_mixed24<>+0x19F0(SB)/8, $1510
DATA bitrev_size2048_mixed24<>+0x19F8(SB)/8, $2022
DATA bitrev_size2048_mixed24<>+0x1A00(SB)/8, $14
DATA bitrev_size2048_mixed24<>+0x1A08(SB)/8, $526
DATA bitrev_size2048_mixed24<>+0x1A10(SB)/8, $1038
DATA bitrev_size2048_mixed24<>+0x1A18(SB)/8, $1550
DATA bitrev_size2048_mixed24<>+0x1A20(SB)/8, $142
DATA bitrev_size2048_mixed24<>+0x1A28(SB)/8, $654
DATA bitrev_size2048_mixed24<>+0x1A30(SB)/8, $1166
DATA bitrev_size2048_mixed24<>+0x1A38(SB)/8, $1678
DATA bitrev_size2048_mixed24<>+0x1A40(SB)/8, $270
DATA bitrev_size2048_mixed24<>+0x1A48(SB)/8, $782
DATA bitrev_size2048_mixed24<>+0x1A50(SB)/8, $1294
DATA bitrev_size2048_mixed24<>+0x1A58(SB)/8, $1806
DATA bitrev_size2048_mixed24<>+0x1A60(SB)/8, $398
DATA bitrev_size2048_mixed24<>+0x1A68(SB)/8, $910
DATA bitrev_size2048_mixed24<>+0x1A70(SB)/8, $1422
DATA bitrev_size2048_mixed24<>+0x1A78(SB)/8, $1934
DATA bitrev_size2048_mixed24<>+0x1A80(SB)/8, $46
DATA bitrev_size2048_mixed24<>+0x1A88(SB)/8, $558
DATA bitrev_size2048_mixed24<>+0x1A90(SB)/8, $1070
DATA bitrev_size2048_mixed24<>+0x1A98(SB)/8, $1582
DATA bitrev_size2048_mixed24<>+0x1AA0(SB)/8, $174
DATA bitrev_size2048_mixed24<>+0x1AA8(SB)/8, $686
DATA bitrev_size2048_mixed24<>+0x1AB0(SB)/8, $1198
DATA bitrev_size2048_mixed24<>+0x1AB8(SB)/8, $1710
DATA bitrev_size2048_mixed24<>+0x1AC0(SB)/8, $302
DATA bitrev_size2048_mixed24<>+0x1AC8(SB)/8, $814
DATA bitrev_size2048_mixed24<>+0x1AD0(SB)/8, $1326
DATA bitrev_size2048_mixed24<>+0x1AD8(SB)/8, $1838
DATA bitrev_size2048_mixed24<>+0x1AE0(SB)/8, $430
DATA bitrev_size2048_mixed24<>+0x1AE8(SB)/8, $942
DATA bitrev_size2048_mixed24<>+0x1AF0(SB)/8, $1454
DATA bitrev_size2048_mixed24<>+0x1AF8(SB)/8, $1966
DATA bitrev_size2048_mixed24<>+0x1B00(SB)/8, $78
DATA bitrev_size2048_mixed24<>+0x1B08(SB)/8, $590
DATA bitrev_size2048_mixed24<>+0x1B10(SB)/8, $1102
DATA bitrev_size2048_mixed24<>+0x1B18(SB)/8, $1614
DATA bitrev_size2048_mixed24<>+0x1B20(SB)/8, $206
DATA bitrev_size2048_mixed24<>+0x1B28(SB)/8, $718
DATA bitrev_size2048_mixed24<>+0x1B30(SB)/8, $1230
DATA bitrev_size2048_mixed24<>+0x1B38(SB)/8, $1742
DATA bitrev_size2048_mixed24<>+0x1B40(SB)/8, $334
DATA bitrev_size2048_mixed24<>+0x1B48(SB)/8, $846
DATA bitrev_size2048_mixed24<>+0x1B50(SB)/8, $1358
DATA bitrev_size2048_mixed24<>+0x1B58(SB)/8, $1870
DATA bitrev_size2048_mixed24<>+0x1B60(SB)/8, $462
DATA bitrev_size2048_mixed24<>+0x1B68(SB)/8, $974
DATA bitrev_size2048_mixed24<>+0x1B70(SB)/8, $1486
DATA bitrev_size2048_mixed24<>+0x1B78(SB)/8, $1998
DATA bitrev_size2048_mixed24<>+0x1B80(SB)/8, $110
DATA bitrev_size2048_mixed24<>+0x1B88(SB)/8, $622
DATA bitrev_size2048_mixed24<>+0x1B90(SB)/8, $1134
DATA bitrev_size2048_mixed24<>+0x1B98(SB)/8, $1646
DATA bitrev_size2048_mixed24<>+0x1BA0(SB)/8, $238
DATA bitrev_size2048_mixed24<>+0x1BA8(SB)/8, $750
DATA bitrev_size2048_mixed24<>+0x1BB0(SB)/8, $1262
DATA bitrev_size2048_mixed24<>+0x1BB8(SB)/8, $1774
DATA bitrev_size2048_mixed24<>+0x1BC0(SB)/8, $366
DATA bitrev_size2048_mixed24<>+0x1BC8(SB)/8, $878
DATA bitrev_size2048_mixed24<>+0x1BD0(SB)/8, $1390
DATA bitrev_size2048_mixed24<>+0x1BD8(SB)/8, $1902
DATA bitrev_size2048_mixed24<>+0x1BE0(SB)/8, $494
DATA bitrev_size2048_mixed24<>+0x1BE8(SB)/8, $1006
DATA bitrev_size2048_mixed24<>+0x1BF0(SB)/8, $1518
DATA bitrev_size2048_mixed24<>+0x1BF8(SB)/8, $2030
DATA bitrev_size2048_mixed24<>+0x1C00(SB)/8, $22
DATA bitrev_size2048_mixed24<>+0x1C08(SB)/8, $534
DATA bitrev_size2048_mixed24<>+0x1C10(SB)/8, $1046
DATA bitrev_size2048_mixed24<>+0x1C18(SB)/8, $1558
DATA bitrev_size2048_mixed24<>+0x1C20(SB)/8, $150
DATA bitrev_size2048_mixed24<>+0x1C28(SB)/8, $662
DATA bitrev_size2048_mixed24<>+0x1C30(SB)/8, $1174
DATA bitrev_size2048_mixed24<>+0x1C38(SB)/8, $1686
DATA bitrev_size2048_mixed24<>+0x1C40(SB)/8, $278
DATA bitrev_size2048_mixed24<>+0x1C48(SB)/8, $790
DATA bitrev_size2048_mixed24<>+0x1C50(SB)/8, $1302
DATA bitrev_size2048_mixed24<>+0x1C58(SB)/8, $1814
DATA bitrev_size2048_mixed24<>+0x1C60(SB)/8, $406
DATA bitrev_size2048_mixed24<>+0x1C68(SB)/8, $918
DATA bitrev_size2048_mixed24<>+0x1C70(SB)/8, $1430
DATA bitrev_size2048_mixed24<>+0x1C78(SB)/8, $1942
DATA bitrev_size2048_mixed24<>+0x1C80(SB)/8, $54
DATA bitrev_size2048_mixed24<>+0x1C88(SB)/8, $566
DATA bitrev_size2048_mixed24<>+0x1C90(SB)/8, $1078
DATA bitrev_size2048_mixed24<>+0x1C98(SB)/8, $1590
DATA bitrev_size2048_mixed24<>+0x1CA0(SB)/8, $182
DATA bitrev_size2048_mixed24<>+0x1CA8(SB)/8, $694
DATA bitrev_size2048_mixed24<>+0x1CB0(SB)/8, $1206
DATA bitrev_size2048_mixed24<>+0x1CB8(SB)/8, $1718
DATA bitrev_size2048_mixed24<>+0x1CC0(SB)/8, $310
DATA bitrev_size2048_mixed24<>+0x1CC8(SB)/8, $822
DATA bitrev_size2048_mixed24<>+0x1CD0(SB)/8, $1334
DATA bitrev_size2048_mixed24<>+0x1CD8(SB)/8, $1846
DATA bitrev_size2048_mixed24<>+0x1CE0(SB)/8, $438
DATA bitrev_size2048_mixed24<>+0x1CE8(SB)/8, $950
DATA bitrev_size2048_mixed24<>+0x1CF0(SB)/8, $1462
DATA bitrev_size2048_mixed24<>+0x1CF8(SB)/8, $1974
DATA bitrev_size2048_mixed24<>+0x1D00(SB)/8, $86
DATA bitrev_size2048_mixed24<>+0x1D08(SB)/8, $598
DATA bitrev_size2048_mixed24<>+0x1D10(SB)/8, $1110
DATA bitrev_size2048_mixed24<>+0x1D18(SB)/8, $1622
DATA bitrev_size2048_mixed24<>+0x1D20(SB)/8, $214
DATA bitrev_size2048_mixed24<>+0x1D28(SB)/8, $726
DATA bitrev_size2048_mixed24<>+0x1D30(SB)/8, $1238
DATA bitrev_size2048_mixed24<>+0x1D38(SB)/8, $1750
DATA bitrev_size2048_mixed24<>+0x1D40(SB)/8, $342
DATA bitrev_size2048_mixed24<>+0x1D48(SB)/8, $854
DATA bitrev_size2048_mixed24<>+0x1D50(SB)/8, $1366
DATA bitrev_size2048_mixed24<>+0x1D58(SB)/8, $1878
DATA bitrev_size2048_mixed24<>+0x1D60(SB)/8, $470
DATA bitrev_size2048_mixed24<>+0x1D68(SB)/8, $982
DATA bitrev_size2048_mixed24<>+0x1D70(SB)/8, $1494
DATA bitrev_size2048_mixed24<>+0x1D78(SB)/8, $2006
DATA bitrev_size2048_mixed24<>+0x1D80(SB)/8, $118
DATA bitrev_size2048_mixed24<>+0x1D88(SB)/8, $630
DATA bitrev_size2048_mixed24<>+0x1D90(SB)/8, $1142
DATA bitrev_size2048_mixed24<>+0x1D98(SB)/8, $1654
DATA bitrev_size2048_mixed24<>+0x1DA0(SB)/8, $246
DATA bitrev_size2048_mixed24<>+0x1DA8(SB)/8, $758
DATA bitrev_size2048_mixed24<>+0x1DB0(SB)/8, $1270
DATA bitrev_size2048_mixed24<>+0x1DB8(SB)/8, $1782
DATA bitrev_size2048_mixed24<>+0x1DC0(SB)/8, $374
DATA bitrev_size2048_mixed24<>+0x1DC8(SB)/8, $886
DATA bitrev_size2048_mixed24<>+0x1DD0(SB)/8, $1398
DATA bitrev_size2048_mixed24<>+0x1DD8(SB)/8, $1910
DATA bitrev_size2048_mixed24<>+0x1DE0(SB)/8, $502
DATA bitrev_size2048_mixed24<>+0x1DE8(SB)/8, $1014
DATA bitrev_size2048_mixed24<>+0x1DF0(SB)/8, $1526
DATA bitrev_size2048_mixed24<>+0x1DF8(SB)/8, $2038
DATA bitrev_size2048_mixed24<>+0x1E00(SB)/8, $30
DATA bitrev_size2048_mixed24<>+0x1E08(SB)/8, $542
DATA bitrev_size2048_mixed24<>+0x1E10(SB)/8, $1054
DATA bitrev_size2048_mixed24<>+0x1E18(SB)/8, $1566
DATA bitrev_size2048_mixed24<>+0x1E20(SB)/8, $158
DATA bitrev_size2048_mixed24<>+0x1E28(SB)/8, $670
DATA bitrev_size2048_mixed24<>+0x1E30(SB)/8, $1182
DATA bitrev_size2048_mixed24<>+0x1E38(SB)/8, $1694
DATA bitrev_size2048_mixed24<>+0x1E40(SB)/8, $286
DATA bitrev_size2048_mixed24<>+0x1E48(SB)/8, $798
DATA bitrev_size2048_mixed24<>+0x1E50(SB)/8, $1310
DATA bitrev_size2048_mixed24<>+0x1E58(SB)/8, $1822
DATA bitrev_size2048_mixed24<>+0x1E60(SB)/8, $414
DATA bitrev_size2048_mixed24<>+0x1E68(SB)/8, $926
DATA bitrev_size2048_mixed24<>+0x1E70(SB)/8, $1438
DATA bitrev_size2048_mixed24<>+0x1E78(SB)/8, $1950
DATA bitrev_size2048_mixed24<>+0x1E80(SB)/8, $62
DATA bitrev_size2048_mixed24<>+0x1E88(SB)/8, $574
DATA bitrev_size2048_mixed24<>+0x1E90(SB)/8, $1086
DATA bitrev_size2048_mixed24<>+0x1E98(SB)/8, $1598
DATA bitrev_size2048_mixed24<>+0x1EA0(SB)/8, $190
DATA bitrev_size2048_mixed24<>+0x1EA8(SB)/8, $702
DATA bitrev_size2048_mixed24<>+0x1EB0(SB)/8, $1214
DATA bitrev_size2048_mixed24<>+0x1EB8(SB)/8, $1726
DATA bitrev_size2048_mixed24<>+0x1EC0(SB)/8, $318
DATA bitrev_size2048_mixed24<>+0x1EC8(SB)/8, $830
DATA bitrev_size2048_mixed24<>+0x1ED0(SB)/8, $1342
DATA bitrev_size2048_mixed24<>+0x1ED8(SB)/8, $1854
DATA bitrev_size2048_mixed24<>+0x1EE0(SB)/8, $446
DATA bitrev_size2048_mixed24<>+0x1EE8(SB)/8, $958
DATA bitrev_size2048_mixed24<>+0x1EF0(SB)/8, $1470
DATA bitrev_size2048_mixed24<>+0x1EF8(SB)/8, $1982
DATA bitrev_size2048_mixed24<>+0x1F00(SB)/8, $94
DATA bitrev_size2048_mixed24<>+0x1F08(SB)/8, $606
DATA bitrev_size2048_mixed24<>+0x1F10(SB)/8, $1118
DATA bitrev_size2048_mixed24<>+0x1F18(SB)/8, $1630
DATA bitrev_size2048_mixed24<>+0x1F20(SB)/8, $222
DATA bitrev_size2048_mixed24<>+0x1F28(SB)/8, $734
DATA bitrev_size2048_mixed24<>+0x1F30(SB)/8, $1246
DATA bitrev_size2048_mixed24<>+0x1F38(SB)/8, $1758
DATA bitrev_size2048_mixed24<>+0x1F40(SB)/8, $350
DATA bitrev_size2048_mixed24<>+0x1F48(SB)/8, $862
DATA bitrev_size2048_mixed24<>+0x1F50(SB)/8, $1374
DATA bitrev_size2048_mixed24<>+0x1F58(SB)/8, $1886
DATA bitrev_size2048_mixed24<>+0x1F60(SB)/8, $478
DATA bitrev_size2048_mixed24<>+0x1F68(SB)/8, $990
DATA bitrev_size2048_mixed24<>+0x1F70(SB)/8, $1502
DATA bitrev_size2048_mixed24<>+0x1F78(SB)/8, $2014
DATA bitrev_size2048_mixed24<>+0x1F80(SB)/8, $126
DATA bitrev_size2048_mixed24<>+0x1F88(SB)/8, $638
DATA bitrev_size2048_mixed24<>+0x1F90(SB)/8, $1150
DATA bitrev_size2048_mixed24<>+0x1F98(SB)/8, $1662
DATA bitrev_size2048_mixed24<>+0x1FA0(SB)/8, $254
DATA bitrev_size2048_mixed24<>+0x1FA8(SB)/8, $766
DATA bitrev_size2048_mixed24<>+0x1FB0(SB)/8, $1278
DATA bitrev_size2048_mixed24<>+0x1FB8(SB)/8, $1790
DATA bitrev_size2048_mixed24<>+0x1FC0(SB)/8, $382
DATA bitrev_size2048_mixed24<>+0x1FC8(SB)/8, $894
DATA bitrev_size2048_mixed24<>+0x1FD0(SB)/8, $1406
DATA bitrev_size2048_mixed24<>+0x1FD8(SB)/8, $1918
DATA bitrev_size2048_mixed24<>+0x1FE0(SB)/8, $510
DATA bitrev_size2048_mixed24<>+0x1FE8(SB)/8, $1022
DATA bitrev_size2048_mixed24<>+0x1FF0(SB)/8, $1534
DATA bitrev_size2048_mixed24<>+0x1FF8(SB)/8, $2046
DATA bitrev_size2048_mixed24<>+0x2000(SB)/8, $1
DATA bitrev_size2048_mixed24<>+0x2008(SB)/8, $513
DATA bitrev_size2048_mixed24<>+0x2010(SB)/8, $1025
DATA bitrev_size2048_mixed24<>+0x2018(SB)/8, $1537
DATA bitrev_size2048_mixed24<>+0x2020(SB)/8, $129
DATA bitrev_size2048_mixed24<>+0x2028(SB)/8, $641
DATA bitrev_size2048_mixed24<>+0x2030(SB)/8, $1153
DATA bitrev_size2048_mixed24<>+0x2038(SB)/8, $1665
DATA bitrev_size2048_mixed24<>+0x2040(SB)/8, $257
DATA bitrev_size2048_mixed24<>+0x2048(SB)/8, $769
DATA bitrev_size2048_mixed24<>+0x2050(SB)/8, $1281
DATA bitrev_size2048_mixed24<>+0x2058(SB)/8, $1793
DATA bitrev_size2048_mixed24<>+0x2060(SB)/8, $385
DATA bitrev_size2048_mixed24<>+0x2068(SB)/8, $897
DATA bitrev_size2048_mixed24<>+0x2070(SB)/8, $1409
DATA bitrev_size2048_mixed24<>+0x2078(SB)/8, $1921
DATA bitrev_size2048_mixed24<>+0x2080(SB)/8, $33
DATA bitrev_size2048_mixed24<>+0x2088(SB)/8, $545
DATA bitrev_size2048_mixed24<>+0x2090(SB)/8, $1057
DATA bitrev_size2048_mixed24<>+0x2098(SB)/8, $1569
DATA bitrev_size2048_mixed24<>+0x20A0(SB)/8, $161
DATA bitrev_size2048_mixed24<>+0x20A8(SB)/8, $673
DATA bitrev_size2048_mixed24<>+0x20B0(SB)/8, $1185
DATA bitrev_size2048_mixed24<>+0x20B8(SB)/8, $1697
DATA bitrev_size2048_mixed24<>+0x20C0(SB)/8, $289
DATA bitrev_size2048_mixed24<>+0x20C8(SB)/8, $801
DATA bitrev_size2048_mixed24<>+0x20D0(SB)/8, $1313
DATA bitrev_size2048_mixed24<>+0x20D8(SB)/8, $1825
DATA bitrev_size2048_mixed24<>+0x20E0(SB)/8, $417
DATA bitrev_size2048_mixed24<>+0x20E8(SB)/8, $929
DATA bitrev_size2048_mixed24<>+0x20F0(SB)/8, $1441
DATA bitrev_size2048_mixed24<>+0x20F8(SB)/8, $1953
DATA bitrev_size2048_mixed24<>+0x2100(SB)/8, $65
DATA bitrev_size2048_mixed24<>+0x2108(SB)/8, $577
DATA bitrev_size2048_mixed24<>+0x2110(SB)/8, $1089
DATA bitrev_size2048_mixed24<>+0x2118(SB)/8, $1601
DATA bitrev_size2048_mixed24<>+0x2120(SB)/8, $193
DATA bitrev_size2048_mixed24<>+0x2128(SB)/8, $705
DATA bitrev_size2048_mixed24<>+0x2130(SB)/8, $1217
DATA bitrev_size2048_mixed24<>+0x2138(SB)/8, $1729
DATA bitrev_size2048_mixed24<>+0x2140(SB)/8, $321
DATA bitrev_size2048_mixed24<>+0x2148(SB)/8, $833
DATA bitrev_size2048_mixed24<>+0x2150(SB)/8, $1345
DATA bitrev_size2048_mixed24<>+0x2158(SB)/8, $1857
DATA bitrev_size2048_mixed24<>+0x2160(SB)/8, $449
DATA bitrev_size2048_mixed24<>+0x2168(SB)/8, $961
DATA bitrev_size2048_mixed24<>+0x2170(SB)/8, $1473
DATA bitrev_size2048_mixed24<>+0x2178(SB)/8, $1985
DATA bitrev_size2048_mixed24<>+0x2180(SB)/8, $97
DATA bitrev_size2048_mixed24<>+0x2188(SB)/8, $609
DATA bitrev_size2048_mixed24<>+0x2190(SB)/8, $1121
DATA bitrev_size2048_mixed24<>+0x2198(SB)/8, $1633
DATA bitrev_size2048_mixed24<>+0x21A0(SB)/8, $225
DATA bitrev_size2048_mixed24<>+0x21A8(SB)/8, $737
DATA bitrev_size2048_mixed24<>+0x21B0(SB)/8, $1249
DATA bitrev_size2048_mixed24<>+0x21B8(SB)/8, $1761
DATA bitrev_size2048_mixed24<>+0x21C0(SB)/8, $353
DATA bitrev_size2048_mixed24<>+0x21C8(SB)/8, $865
DATA bitrev_size2048_mixed24<>+0x21D0(SB)/8, $1377
DATA bitrev_size2048_mixed24<>+0x21D8(SB)/8, $1889
DATA bitrev_size2048_mixed24<>+0x21E0(SB)/8, $481
DATA bitrev_size2048_mixed24<>+0x21E8(SB)/8, $993
DATA bitrev_size2048_mixed24<>+0x21F0(SB)/8, $1505
DATA bitrev_size2048_mixed24<>+0x21F8(SB)/8, $2017
DATA bitrev_size2048_mixed24<>+0x2200(SB)/8, $9
DATA bitrev_size2048_mixed24<>+0x2208(SB)/8, $521
DATA bitrev_size2048_mixed24<>+0x2210(SB)/8, $1033
DATA bitrev_size2048_mixed24<>+0x2218(SB)/8, $1545
DATA bitrev_size2048_mixed24<>+0x2220(SB)/8, $137
DATA bitrev_size2048_mixed24<>+0x2228(SB)/8, $649
DATA bitrev_size2048_mixed24<>+0x2230(SB)/8, $1161
DATA bitrev_size2048_mixed24<>+0x2238(SB)/8, $1673
DATA bitrev_size2048_mixed24<>+0x2240(SB)/8, $265
DATA bitrev_size2048_mixed24<>+0x2248(SB)/8, $777
DATA bitrev_size2048_mixed24<>+0x2250(SB)/8, $1289
DATA bitrev_size2048_mixed24<>+0x2258(SB)/8, $1801
DATA bitrev_size2048_mixed24<>+0x2260(SB)/8, $393
DATA bitrev_size2048_mixed24<>+0x2268(SB)/8, $905
DATA bitrev_size2048_mixed24<>+0x2270(SB)/8, $1417
DATA bitrev_size2048_mixed24<>+0x2278(SB)/8, $1929
DATA bitrev_size2048_mixed24<>+0x2280(SB)/8, $41
DATA bitrev_size2048_mixed24<>+0x2288(SB)/8, $553
DATA bitrev_size2048_mixed24<>+0x2290(SB)/8, $1065
DATA bitrev_size2048_mixed24<>+0x2298(SB)/8, $1577
DATA bitrev_size2048_mixed24<>+0x22A0(SB)/8, $169
DATA bitrev_size2048_mixed24<>+0x22A8(SB)/8, $681
DATA bitrev_size2048_mixed24<>+0x22B0(SB)/8, $1193
DATA bitrev_size2048_mixed24<>+0x22B8(SB)/8, $1705
DATA bitrev_size2048_mixed24<>+0x22C0(SB)/8, $297
DATA bitrev_size2048_mixed24<>+0x22C8(SB)/8, $809
DATA bitrev_size2048_mixed24<>+0x22D0(SB)/8, $1321
DATA bitrev_size2048_mixed24<>+0x22D8(SB)/8, $1833
DATA bitrev_size2048_mixed24<>+0x22E0(SB)/8, $425
DATA bitrev_size2048_mixed24<>+0x22E8(SB)/8, $937
DATA bitrev_size2048_mixed24<>+0x22F0(SB)/8, $1449
DATA bitrev_size2048_mixed24<>+0x22F8(SB)/8, $1961
DATA bitrev_size2048_mixed24<>+0x2300(SB)/8, $73
DATA bitrev_size2048_mixed24<>+0x2308(SB)/8, $585
DATA bitrev_size2048_mixed24<>+0x2310(SB)/8, $1097
DATA bitrev_size2048_mixed24<>+0x2318(SB)/8, $1609
DATA bitrev_size2048_mixed24<>+0x2320(SB)/8, $201
DATA bitrev_size2048_mixed24<>+0x2328(SB)/8, $713
DATA bitrev_size2048_mixed24<>+0x2330(SB)/8, $1225
DATA bitrev_size2048_mixed24<>+0x2338(SB)/8, $1737
DATA bitrev_size2048_mixed24<>+0x2340(SB)/8, $329
DATA bitrev_size2048_mixed24<>+0x2348(SB)/8, $841
DATA bitrev_size2048_mixed24<>+0x2350(SB)/8, $1353
DATA bitrev_size2048_mixed24<>+0x2358(SB)/8, $1865
DATA bitrev_size2048_mixed24<>+0x2360(SB)/8, $457
DATA bitrev_size2048_mixed24<>+0x2368(SB)/8, $969
DATA bitrev_size2048_mixed24<>+0x2370(SB)/8, $1481
DATA bitrev_size2048_mixed24<>+0x2378(SB)/8, $1993
DATA bitrev_size2048_mixed24<>+0x2380(SB)/8, $105
DATA bitrev_size2048_mixed24<>+0x2388(SB)/8, $617
DATA bitrev_size2048_mixed24<>+0x2390(SB)/8, $1129
DATA bitrev_size2048_mixed24<>+0x2398(SB)/8, $1641
DATA bitrev_size2048_mixed24<>+0x23A0(SB)/8, $233
DATA bitrev_size2048_mixed24<>+0x23A8(SB)/8, $745
DATA bitrev_size2048_mixed24<>+0x23B0(SB)/8, $1257
DATA bitrev_size2048_mixed24<>+0x23B8(SB)/8, $1769
DATA bitrev_size2048_mixed24<>+0x23C0(SB)/8, $361
DATA bitrev_size2048_mixed24<>+0x23C8(SB)/8, $873
DATA bitrev_size2048_mixed24<>+0x23D0(SB)/8, $1385
DATA bitrev_size2048_mixed24<>+0x23D8(SB)/8, $1897
DATA bitrev_size2048_mixed24<>+0x23E0(SB)/8, $489
DATA bitrev_size2048_mixed24<>+0x23E8(SB)/8, $1001
DATA bitrev_size2048_mixed24<>+0x23F0(SB)/8, $1513
DATA bitrev_size2048_mixed24<>+0x23F8(SB)/8, $2025
DATA bitrev_size2048_mixed24<>+0x2400(SB)/8, $17
DATA bitrev_size2048_mixed24<>+0x2408(SB)/8, $529
DATA bitrev_size2048_mixed24<>+0x2410(SB)/8, $1041
DATA bitrev_size2048_mixed24<>+0x2418(SB)/8, $1553
DATA bitrev_size2048_mixed24<>+0x2420(SB)/8, $145
DATA bitrev_size2048_mixed24<>+0x2428(SB)/8, $657
DATA bitrev_size2048_mixed24<>+0x2430(SB)/8, $1169
DATA bitrev_size2048_mixed24<>+0x2438(SB)/8, $1681
DATA bitrev_size2048_mixed24<>+0x2440(SB)/8, $273
DATA bitrev_size2048_mixed24<>+0x2448(SB)/8, $785
DATA bitrev_size2048_mixed24<>+0x2450(SB)/8, $1297
DATA bitrev_size2048_mixed24<>+0x2458(SB)/8, $1809
DATA bitrev_size2048_mixed24<>+0x2460(SB)/8, $401
DATA bitrev_size2048_mixed24<>+0x2468(SB)/8, $913
DATA bitrev_size2048_mixed24<>+0x2470(SB)/8, $1425
DATA bitrev_size2048_mixed24<>+0x2478(SB)/8, $1937
DATA bitrev_size2048_mixed24<>+0x2480(SB)/8, $49
DATA bitrev_size2048_mixed24<>+0x2488(SB)/8, $561
DATA bitrev_size2048_mixed24<>+0x2490(SB)/8, $1073
DATA bitrev_size2048_mixed24<>+0x2498(SB)/8, $1585
DATA bitrev_size2048_mixed24<>+0x24A0(SB)/8, $177
DATA bitrev_size2048_mixed24<>+0x24A8(SB)/8, $689
DATA bitrev_size2048_mixed24<>+0x24B0(SB)/8, $1201
DATA bitrev_size2048_mixed24<>+0x24B8(SB)/8, $1713
DATA bitrev_size2048_mixed24<>+0x24C0(SB)/8, $305
DATA bitrev_size2048_mixed24<>+0x24C8(SB)/8, $817
DATA bitrev_size2048_mixed24<>+0x24D0(SB)/8, $1329
DATA bitrev_size2048_mixed24<>+0x24D8(SB)/8, $1841
DATA bitrev_size2048_mixed24<>+0x24E0(SB)/8, $433
DATA bitrev_size2048_mixed24<>+0x24E8(SB)/8, $945
DATA bitrev_size2048_mixed24<>+0x24F0(SB)/8, $1457
DATA bitrev_size2048_mixed24<>+0x24F8(SB)/8, $1969
DATA bitrev_size2048_mixed24<>+0x2500(SB)/8, $81
DATA bitrev_size2048_mixed24<>+0x2508(SB)/8, $593
DATA bitrev_size2048_mixed24<>+0x2510(SB)/8, $1105
DATA bitrev_size2048_mixed24<>+0x2518(SB)/8, $1617
DATA bitrev_size2048_mixed24<>+0x2520(SB)/8, $209
DATA bitrev_size2048_mixed24<>+0x2528(SB)/8, $721
DATA bitrev_size2048_mixed24<>+0x2530(SB)/8, $1233
DATA bitrev_size2048_mixed24<>+0x2538(SB)/8, $1745
DATA bitrev_size2048_mixed24<>+0x2540(SB)/8, $337
DATA bitrev_size2048_mixed24<>+0x2548(SB)/8, $849
DATA bitrev_size2048_mixed24<>+0x2550(SB)/8, $1361
DATA bitrev_size2048_mixed24<>+0x2558(SB)/8, $1873
DATA bitrev_size2048_mixed24<>+0x2560(SB)/8, $465
DATA bitrev_size2048_mixed24<>+0x2568(SB)/8, $977
DATA bitrev_size2048_mixed24<>+0x2570(SB)/8, $1489
DATA bitrev_size2048_mixed24<>+0x2578(SB)/8, $2001
DATA bitrev_size2048_mixed24<>+0x2580(SB)/8, $113
DATA bitrev_size2048_mixed24<>+0x2588(SB)/8, $625
DATA bitrev_size2048_mixed24<>+0x2590(SB)/8, $1137
DATA bitrev_size2048_mixed24<>+0x2598(SB)/8, $1649
DATA bitrev_size2048_mixed24<>+0x25A0(SB)/8, $241
DATA bitrev_size2048_mixed24<>+0x25A8(SB)/8, $753
DATA bitrev_size2048_mixed24<>+0x25B0(SB)/8, $1265
DATA bitrev_size2048_mixed24<>+0x25B8(SB)/8, $1777
DATA bitrev_size2048_mixed24<>+0x25C0(SB)/8, $369
DATA bitrev_size2048_mixed24<>+0x25C8(SB)/8, $881
DATA bitrev_size2048_mixed24<>+0x25D0(SB)/8, $1393
DATA bitrev_size2048_mixed24<>+0x25D8(SB)/8, $1905
DATA bitrev_size2048_mixed24<>+0x25E0(SB)/8, $497
DATA bitrev_size2048_mixed24<>+0x25E8(SB)/8, $1009
DATA bitrev_size2048_mixed24<>+0x25F0(SB)/8, $1521
DATA bitrev_size2048_mixed24<>+0x25F8(SB)/8, $2033
DATA bitrev_size2048_mixed24<>+0x2600(SB)/8, $25
DATA bitrev_size2048_mixed24<>+0x2608(SB)/8, $537
DATA bitrev_size2048_mixed24<>+0x2610(SB)/8, $1049
DATA bitrev_size2048_mixed24<>+0x2618(SB)/8, $1561
DATA bitrev_size2048_mixed24<>+0x2620(SB)/8, $153
DATA bitrev_size2048_mixed24<>+0x2628(SB)/8, $665
DATA bitrev_size2048_mixed24<>+0x2630(SB)/8, $1177
DATA bitrev_size2048_mixed24<>+0x2638(SB)/8, $1689
DATA bitrev_size2048_mixed24<>+0x2640(SB)/8, $281
DATA bitrev_size2048_mixed24<>+0x2648(SB)/8, $793
DATA bitrev_size2048_mixed24<>+0x2650(SB)/8, $1305
DATA bitrev_size2048_mixed24<>+0x2658(SB)/8, $1817
DATA bitrev_size2048_mixed24<>+0x2660(SB)/8, $409
DATA bitrev_size2048_mixed24<>+0x2668(SB)/8, $921
DATA bitrev_size2048_mixed24<>+0x2670(SB)/8, $1433
DATA bitrev_size2048_mixed24<>+0x2678(SB)/8, $1945
DATA bitrev_size2048_mixed24<>+0x2680(SB)/8, $57
DATA bitrev_size2048_mixed24<>+0x2688(SB)/8, $569
DATA bitrev_size2048_mixed24<>+0x2690(SB)/8, $1081
DATA bitrev_size2048_mixed24<>+0x2698(SB)/8, $1593
DATA bitrev_size2048_mixed24<>+0x26A0(SB)/8, $185
DATA bitrev_size2048_mixed24<>+0x26A8(SB)/8, $697
DATA bitrev_size2048_mixed24<>+0x26B0(SB)/8, $1209
DATA bitrev_size2048_mixed24<>+0x26B8(SB)/8, $1721
DATA bitrev_size2048_mixed24<>+0x26C0(SB)/8, $313
DATA bitrev_size2048_mixed24<>+0x26C8(SB)/8, $825
DATA bitrev_size2048_mixed24<>+0x26D0(SB)/8, $1337
DATA bitrev_size2048_mixed24<>+0x26D8(SB)/8, $1849
DATA bitrev_size2048_mixed24<>+0x26E0(SB)/8, $441
DATA bitrev_size2048_mixed24<>+0x26E8(SB)/8, $953
DATA bitrev_size2048_mixed24<>+0x26F0(SB)/8, $1465
DATA bitrev_size2048_mixed24<>+0x26F8(SB)/8, $1977
DATA bitrev_size2048_mixed24<>+0x2700(SB)/8, $89
DATA bitrev_size2048_mixed24<>+0x2708(SB)/8, $601
DATA bitrev_size2048_mixed24<>+0x2710(SB)/8, $1113
DATA bitrev_size2048_mixed24<>+0x2718(SB)/8, $1625
DATA bitrev_size2048_mixed24<>+0x2720(SB)/8, $217
DATA bitrev_size2048_mixed24<>+0x2728(SB)/8, $729
DATA bitrev_size2048_mixed24<>+0x2730(SB)/8, $1241
DATA bitrev_size2048_mixed24<>+0x2738(SB)/8, $1753
DATA bitrev_size2048_mixed24<>+0x2740(SB)/8, $345
DATA bitrev_size2048_mixed24<>+0x2748(SB)/8, $857
DATA bitrev_size2048_mixed24<>+0x2750(SB)/8, $1369
DATA bitrev_size2048_mixed24<>+0x2758(SB)/8, $1881
DATA bitrev_size2048_mixed24<>+0x2760(SB)/8, $473
DATA bitrev_size2048_mixed24<>+0x2768(SB)/8, $985
DATA bitrev_size2048_mixed24<>+0x2770(SB)/8, $1497
DATA bitrev_size2048_mixed24<>+0x2778(SB)/8, $2009
DATA bitrev_size2048_mixed24<>+0x2780(SB)/8, $121
DATA bitrev_size2048_mixed24<>+0x2788(SB)/8, $633
DATA bitrev_size2048_mixed24<>+0x2790(SB)/8, $1145
DATA bitrev_size2048_mixed24<>+0x2798(SB)/8, $1657
DATA bitrev_size2048_mixed24<>+0x27A0(SB)/8, $249
DATA bitrev_size2048_mixed24<>+0x27A8(SB)/8, $761
DATA bitrev_size2048_mixed24<>+0x27B0(SB)/8, $1273
DATA bitrev_size2048_mixed24<>+0x27B8(SB)/8, $1785
DATA bitrev_size2048_mixed24<>+0x27C0(SB)/8, $377
DATA bitrev_size2048_mixed24<>+0x27C8(SB)/8, $889
DATA bitrev_size2048_mixed24<>+0x27D0(SB)/8, $1401
DATA bitrev_size2048_mixed24<>+0x27D8(SB)/8, $1913
DATA bitrev_size2048_mixed24<>+0x27E0(SB)/8, $505
DATA bitrev_size2048_mixed24<>+0x27E8(SB)/8, $1017
DATA bitrev_size2048_mixed24<>+0x27F0(SB)/8, $1529
DATA bitrev_size2048_mixed24<>+0x27F8(SB)/8, $2041
DATA bitrev_size2048_mixed24<>+0x2800(SB)/8, $3
DATA bitrev_size2048_mixed24<>+0x2808(SB)/8, $515
DATA bitrev_size2048_mixed24<>+0x2810(SB)/8, $1027
DATA bitrev_size2048_mixed24<>+0x2818(SB)/8, $1539
DATA bitrev_size2048_mixed24<>+0x2820(SB)/8, $131
DATA bitrev_size2048_mixed24<>+0x2828(SB)/8, $643
DATA bitrev_size2048_mixed24<>+0x2830(SB)/8, $1155
DATA bitrev_size2048_mixed24<>+0x2838(SB)/8, $1667
DATA bitrev_size2048_mixed24<>+0x2840(SB)/8, $259
DATA bitrev_size2048_mixed24<>+0x2848(SB)/8, $771
DATA bitrev_size2048_mixed24<>+0x2850(SB)/8, $1283
DATA bitrev_size2048_mixed24<>+0x2858(SB)/8, $1795
DATA bitrev_size2048_mixed24<>+0x2860(SB)/8, $387
DATA bitrev_size2048_mixed24<>+0x2868(SB)/8, $899
DATA bitrev_size2048_mixed24<>+0x2870(SB)/8, $1411
DATA bitrev_size2048_mixed24<>+0x2878(SB)/8, $1923
DATA bitrev_size2048_mixed24<>+0x2880(SB)/8, $35
DATA bitrev_size2048_mixed24<>+0x2888(SB)/8, $547
DATA bitrev_size2048_mixed24<>+0x2890(SB)/8, $1059
DATA bitrev_size2048_mixed24<>+0x2898(SB)/8, $1571
DATA bitrev_size2048_mixed24<>+0x28A0(SB)/8, $163
DATA bitrev_size2048_mixed24<>+0x28A8(SB)/8, $675
DATA bitrev_size2048_mixed24<>+0x28B0(SB)/8, $1187
DATA bitrev_size2048_mixed24<>+0x28B8(SB)/8, $1699
DATA bitrev_size2048_mixed24<>+0x28C0(SB)/8, $291
DATA bitrev_size2048_mixed24<>+0x28C8(SB)/8, $803
DATA bitrev_size2048_mixed24<>+0x28D0(SB)/8, $1315
DATA bitrev_size2048_mixed24<>+0x28D8(SB)/8, $1827
DATA bitrev_size2048_mixed24<>+0x28E0(SB)/8, $419
DATA bitrev_size2048_mixed24<>+0x28E8(SB)/8, $931
DATA bitrev_size2048_mixed24<>+0x28F0(SB)/8, $1443
DATA bitrev_size2048_mixed24<>+0x28F8(SB)/8, $1955
DATA bitrev_size2048_mixed24<>+0x2900(SB)/8, $67
DATA bitrev_size2048_mixed24<>+0x2908(SB)/8, $579
DATA bitrev_size2048_mixed24<>+0x2910(SB)/8, $1091
DATA bitrev_size2048_mixed24<>+0x2918(SB)/8, $1603
DATA bitrev_size2048_mixed24<>+0x2920(SB)/8, $195
DATA bitrev_size2048_mixed24<>+0x2928(SB)/8, $707
DATA bitrev_size2048_mixed24<>+0x2930(SB)/8, $1219
DATA bitrev_size2048_mixed24<>+0x2938(SB)/8, $1731
DATA bitrev_size2048_mixed24<>+0x2940(SB)/8, $323
DATA bitrev_size2048_mixed24<>+0x2948(SB)/8, $835
DATA bitrev_size2048_mixed24<>+0x2950(SB)/8, $1347
DATA bitrev_size2048_mixed24<>+0x2958(SB)/8, $1859
DATA bitrev_size2048_mixed24<>+0x2960(SB)/8, $451
DATA bitrev_size2048_mixed24<>+0x2968(SB)/8, $963
DATA bitrev_size2048_mixed24<>+0x2970(SB)/8, $1475
DATA bitrev_size2048_mixed24<>+0x2978(SB)/8, $1987
DATA bitrev_size2048_mixed24<>+0x2980(SB)/8, $99
DATA bitrev_size2048_mixed24<>+0x2988(SB)/8, $611
DATA bitrev_size2048_mixed24<>+0x2990(SB)/8, $1123
DATA bitrev_size2048_mixed24<>+0x2998(SB)/8, $1635
DATA bitrev_size2048_mixed24<>+0x29A0(SB)/8, $227
DATA bitrev_size2048_mixed24<>+0x29A8(SB)/8, $739
DATA bitrev_size2048_mixed24<>+0x29B0(SB)/8, $1251
DATA bitrev_size2048_mixed24<>+0x29B8(SB)/8, $1763
DATA bitrev_size2048_mixed24<>+0x29C0(SB)/8, $355
DATA bitrev_size2048_mixed24<>+0x29C8(SB)/8, $867
DATA bitrev_size2048_mixed24<>+0x29D0(SB)/8, $1379
DATA bitrev_size2048_mixed24<>+0x29D8(SB)/8, $1891
DATA bitrev_size2048_mixed24<>+0x29E0(SB)/8, $483
DATA bitrev_size2048_mixed24<>+0x29E8(SB)/8, $995
DATA bitrev_size2048_mixed24<>+0x29F0(SB)/8, $1507
DATA bitrev_size2048_mixed24<>+0x29F8(SB)/8, $2019
DATA bitrev_size2048_mixed24<>+0x2A00(SB)/8, $11
DATA bitrev_size2048_mixed24<>+0x2A08(SB)/8, $523
DATA bitrev_size2048_mixed24<>+0x2A10(SB)/8, $1035
DATA bitrev_size2048_mixed24<>+0x2A18(SB)/8, $1547
DATA bitrev_size2048_mixed24<>+0x2A20(SB)/8, $139
DATA bitrev_size2048_mixed24<>+0x2A28(SB)/8, $651
DATA bitrev_size2048_mixed24<>+0x2A30(SB)/8, $1163
DATA bitrev_size2048_mixed24<>+0x2A38(SB)/8, $1675
DATA bitrev_size2048_mixed24<>+0x2A40(SB)/8, $267
DATA bitrev_size2048_mixed24<>+0x2A48(SB)/8, $779
DATA bitrev_size2048_mixed24<>+0x2A50(SB)/8, $1291
DATA bitrev_size2048_mixed24<>+0x2A58(SB)/8, $1803
DATA bitrev_size2048_mixed24<>+0x2A60(SB)/8, $395
DATA bitrev_size2048_mixed24<>+0x2A68(SB)/8, $907
DATA bitrev_size2048_mixed24<>+0x2A70(SB)/8, $1419
DATA bitrev_size2048_mixed24<>+0x2A78(SB)/8, $1931
DATA bitrev_size2048_mixed24<>+0x2A80(SB)/8, $43
DATA bitrev_size2048_mixed24<>+0x2A88(SB)/8, $555
DATA bitrev_size2048_mixed24<>+0x2A90(SB)/8, $1067
DATA bitrev_size2048_mixed24<>+0x2A98(SB)/8, $1579
DATA bitrev_size2048_mixed24<>+0x2AA0(SB)/8, $171
DATA bitrev_size2048_mixed24<>+0x2AA8(SB)/8, $683
DATA bitrev_size2048_mixed24<>+0x2AB0(SB)/8, $1195
DATA bitrev_size2048_mixed24<>+0x2AB8(SB)/8, $1707
DATA bitrev_size2048_mixed24<>+0x2AC0(SB)/8, $299
DATA bitrev_size2048_mixed24<>+0x2AC8(SB)/8, $811
DATA bitrev_size2048_mixed24<>+0x2AD0(SB)/8, $1323
DATA bitrev_size2048_mixed24<>+0x2AD8(SB)/8, $1835
DATA bitrev_size2048_mixed24<>+0x2AE0(SB)/8, $427
DATA bitrev_size2048_mixed24<>+0x2AE8(SB)/8, $939
DATA bitrev_size2048_mixed24<>+0x2AF0(SB)/8, $1451
DATA bitrev_size2048_mixed24<>+0x2AF8(SB)/8, $1963
DATA bitrev_size2048_mixed24<>+0x2B00(SB)/8, $75
DATA bitrev_size2048_mixed24<>+0x2B08(SB)/8, $587
DATA bitrev_size2048_mixed24<>+0x2B10(SB)/8, $1099
DATA bitrev_size2048_mixed24<>+0x2B18(SB)/8, $1611
DATA bitrev_size2048_mixed24<>+0x2B20(SB)/8, $203
DATA bitrev_size2048_mixed24<>+0x2B28(SB)/8, $715
DATA bitrev_size2048_mixed24<>+0x2B30(SB)/8, $1227
DATA bitrev_size2048_mixed24<>+0x2B38(SB)/8, $1739
DATA bitrev_size2048_mixed24<>+0x2B40(SB)/8, $331
DATA bitrev_size2048_mixed24<>+0x2B48(SB)/8, $843
DATA bitrev_size2048_mixed24<>+0x2B50(SB)/8, $1355
DATA bitrev_size2048_mixed24<>+0x2B58(SB)/8, $1867
DATA bitrev_size2048_mixed24<>+0x2B60(SB)/8, $459
DATA bitrev_size2048_mixed24<>+0x2B68(SB)/8, $971
DATA bitrev_size2048_mixed24<>+0x2B70(SB)/8, $1483
DATA bitrev_size2048_mixed24<>+0x2B78(SB)/8, $1995
DATA bitrev_size2048_mixed24<>+0x2B80(SB)/8, $107
DATA bitrev_size2048_mixed24<>+0x2B88(SB)/8, $619
DATA bitrev_size2048_mixed24<>+0x2B90(SB)/8, $1131
DATA bitrev_size2048_mixed24<>+0x2B98(SB)/8, $1643
DATA bitrev_size2048_mixed24<>+0x2BA0(SB)/8, $235
DATA bitrev_size2048_mixed24<>+0x2BA8(SB)/8, $747
DATA bitrev_size2048_mixed24<>+0x2BB0(SB)/8, $1259
DATA bitrev_size2048_mixed24<>+0x2BB8(SB)/8, $1771
DATA bitrev_size2048_mixed24<>+0x2BC0(SB)/8, $363
DATA bitrev_size2048_mixed24<>+0x2BC8(SB)/8, $875
DATA bitrev_size2048_mixed24<>+0x2BD0(SB)/8, $1387
DATA bitrev_size2048_mixed24<>+0x2BD8(SB)/8, $1899
DATA bitrev_size2048_mixed24<>+0x2BE0(SB)/8, $491
DATA bitrev_size2048_mixed24<>+0x2BE8(SB)/8, $1003
DATA bitrev_size2048_mixed24<>+0x2BF0(SB)/8, $1515
DATA bitrev_size2048_mixed24<>+0x2BF8(SB)/8, $2027
DATA bitrev_size2048_mixed24<>+0x2C00(SB)/8, $19
DATA bitrev_size2048_mixed24<>+0x2C08(SB)/8, $531
DATA bitrev_size2048_mixed24<>+0x2C10(SB)/8, $1043
DATA bitrev_size2048_mixed24<>+0x2C18(SB)/8, $1555
DATA bitrev_size2048_mixed24<>+0x2C20(SB)/8, $147
DATA bitrev_size2048_mixed24<>+0x2C28(SB)/8, $659
DATA bitrev_size2048_mixed24<>+0x2C30(SB)/8, $1171
DATA bitrev_size2048_mixed24<>+0x2C38(SB)/8, $1683
DATA bitrev_size2048_mixed24<>+0x2C40(SB)/8, $275
DATA bitrev_size2048_mixed24<>+0x2C48(SB)/8, $787
DATA bitrev_size2048_mixed24<>+0x2C50(SB)/8, $1299
DATA bitrev_size2048_mixed24<>+0x2C58(SB)/8, $1811
DATA bitrev_size2048_mixed24<>+0x2C60(SB)/8, $403
DATA bitrev_size2048_mixed24<>+0x2C68(SB)/8, $915
DATA bitrev_size2048_mixed24<>+0x2C70(SB)/8, $1427
DATA bitrev_size2048_mixed24<>+0x2C78(SB)/8, $1939
DATA bitrev_size2048_mixed24<>+0x2C80(SB)/8, $51
DATA bitrev_size2048_mixed24<>+0x2C88(SB)/8, $563
DATA bitrev_size2048_mixed24<>+0x2C90(SB)/8, $1075
DATA bitrev_size2048_mixed24<>+0x2C98(SB)/8, $1587
DATA bitrev_size2048_mixed24<>+0x2CA0(SB)/8, $179
DATA bitrev_size2048_mixed24<>+0x2CA8(SB)/8, $691
DATA bitrev_size2048_mixed24<>+0x2CB0(SB)/8, $1203
DATA bitrev_size2048_mixed24<>+0x2CB8(SB)/8, $1715
DATA bitrev_size2048_mixed24<>+0x2CC0(SB)/8, $307
DATA bitrev_size2048_mixed24<>+0x2CC8(SB)/8, $819
DATA bitrev_size2048_mixed24<>+0x2CD0(SB)/8, $1331
DATA bitrev_size2048_mixed24<>+0x2CD8(SB)/8, $1843
DATA bitrev_size2048_mixed24<>+0x2CE0(SB)/8, $435
DATA bitrev_size2048_mixed24<>+0x2CE8(SB)/8, $947
DATA bitrev_size2048_mixed24<>+0x2CF0(SB)/8, $1459
DATA bitrev_size2048_mixed24<>+0x2CF8(SB)/8, $1971
DATA bitrev_size2048_mixed24<>+0x2D00(SB)/8, $83
DATA bitrev_size2048_mixed24<>+0x2D08(SB)/8, $595
DATA bitrev_size2048_mixed24<>+0x2D10(SB)/8, $1107
DATA bitrev_size2048_mixed24<>+0x2D18(SB)/8, $1619
DATA bitrev_size2048_mixed24<>+0x2D20(SB)/8, $211
DATA bitrev_size2048_mixed24<>+0x2D28(SB)/8, $723
DATA bitrev_size2048_mixed24<>+0x2D30(SB)/8, $1235
DATA bitrev_size2048_mixed24<>+0x2D38(SB)/8, $1747
DATA bitrev_size2048_mixed24<>+0x2D40(SB)/8, $339
DATA bitrev_size2048_mixed24<>+0x2D48(SB)/8, $851
DATA bitrev_size2048_mixed24<>+0x2D50(SB)/8, $1363
DATA bitrev_size2048_mixed24<>+0x2D58(SB)/8, $1875
DATA bitrev_size2048_mixed24<>+0x2D60(SB)/8, $467
DATA bitrev_size2048_mixed24<>+0x2D68(SB)/8, $979
DATA bitrev_size2048_mixed24<>+0x2D70(SB)/8, $1491
DATA bitrev_size2048_mixed24<>+0x2D78(SB)/8, $2003
DATA bitrev_size2048_mixed24<>+0x2D80(SB)/8, $115
DATA bitrev_size2048_mixed24<>+0x2D88(SB)/8, $627
DATA bitrev_size2048_mixed24<>+0x2D90(SB)/8, $1139
DATA bitrev_size2048_mixed24<>+0x2D98(SB)/8, $1651
DATA bitrev_size2048_mixed24<>+0x2DA0(SB)/8, $243
DATA bitrev_size2048_mixed24<>+0x2DA8(SB)/8, $755
DATA bitrev_size2048_mixed24<>+0x2DB0(SB)/8, $1267
DATA bitrev_size2048_mixed24<>+0x2DB8(SB)/8, $1779
DATA bitrev_size2048_mixed24<>+0x2DC0(SB)/8, $371
DATA bitrev_size2048_mixed24<>+0x2DC8(SB)/8, $883
DATA bitrev_size2048_mixed24<>+0x2DD0(SB)/8, $1395
DATA bitrev_size2048_mixed24<>+0x2DD8(SB)/8, $1907
DATA bitrev_size2048_mixed24<>+0x2DE0(SB)/8, $499
DATA bitrev_size2048_mixed24<>+0x2DE8(SB)/8, $1011
DATA bitrev_size2048_mixed24<>+0x2DF0(SB)/8, $1523
DATA bitrev_size2048_mixed24<>+0x2DF8(SB)/8, $2035
DATA bitrev_size2048_mixed24<>+0x2E00(SB)/8, $27
DATA bitrev_size2048_mixed24<>+0x2E08(SB)/8, $539
DATA bitrev_size2048_mixed24<>+0x2E10(SB)/8, $1051
DATA bitrev_size2048_mixed24<>+0x2E18(SB)/8, $1563
DATA bitrev_size2048_mixed24<>+0x2E20(SB)/8, $155
DATA bitrev_size2048_mixed24<>+0x2E28(SB)/8, $667
DATA bitrev_size2048_mixed24<>+0x2E30(SB)/8, $1179
DATA bitrev_size2048_mixed24<>+0x2E38(SB)/8, $1691
DATA bitrev_size2048_mixed24<>+0x2E40(SB)/8, $283
DATA bitrev_size2048_mixed24<>+0x2E48(SB)/8, $795
DATA bitrev_size2048_mixed24<>+0x2E50(SB)/8, $1307
DATA bitrev_size2048_mixed24<>+0x2E58(SB)/8, $1819
DATA bitrev_size2048_mixed24<>+0x2E60(SB)/8, $411
DATA bitrev_size2048_mixed24<>+0x2E68(SB)/8, $923
DATA bitrev_size2048_mixed24<>+0x2E70(SB)/8, $1435
DATA bitrev_size2048_mixed24<>+0x2E78(SB)/8, $1947
DATA bitrev_size2048_mixed24<>+0x2E80(SB)/8, $59
DATA bitrev_size2048_mixed24<>+0x2E88(SB)/8, $571
DATA bitrev_size2048_mixed24<>+0x2E90(SB)/8, $1083
DATA bitrev_size2048_mixed24<>+0x2E98(SB)/8, $1595
DATA bitrev_size2048_mixed24<>+0x2EA0(SB)/8, $187
DATA bitrev_size2048_mixed24<>+0x2EA8(SB)/8, $699
DATA bitrev_size2048_mixed24<>+0x2EB0(SB)/8, $1211
DATA bitrev_size2048_mixed24<>+0x2EB8(SB)/8, $1723
DATA bitrev_size2048_mixed24<>+0x2EC0(SB)/8, $315
DATA bitrev_size2048_mixed24<>+0x2EC8(SB)/8, $827
DATA bitrev_size2048_mixed24<>+0x2ED0(SB)/8, $1339
DATA bitrev_size2048_mixed24<>+0x2ED8(SB)/8, $1851
DATA bitrev_size2048_mixed24<>+0x2EE0(SB)/8, $443
DATA bitrev_size2048_mixed24<>+0x2EE8(SB)/8, $955
DATA bitrev_size2048_mixed24<>+0x2EF0(SB)/8, $1467
DATA bitrev_size2048_mixed24<>+0x2EF8(SB)/8, $1979
DATA bitrev_size2048_mixed24<>+0x2F00(SB)/8, $91
DATA bitrev_size2048_mixed24<>+0x2F08(SB)/8, $603
DATA bitrev_size2048_mixed24<>+0x2F10(SB)/8, $1115
DATA bitrev_size2048_mixed24<>+0x2F18(SB)/8, $1627
DATA bitrev_size2048_mixed24<>+0x2F20(SB)/8, $219
DATA bitrev_size2048_mixed24<>+0x2F28(SB)/8, $731
DATA bitrev_size2048_mixed24<>+0x2F30(SB)/8, $1243
DATA bitrev_size2048_mixed24<>+0x2F38(SB)/8, $1755
DATA bitrev_size2048_mixed24<>+0x2F40(SB)/8, $347
DATA bitrev_size2048_mixed24<>+0x2F48(SB)/8, $859
DATA bitrev_size2048_mixed24<>+0x2F50(SB)/8, $1371
DATA bitrev_size2048_mixed24<>+0x2F58(SB)/8, $1883
DATA bitrev_size2048_mixed24<>+0x2F60(SB)/8, $475
DATA bitrev_size2048_mixed24<>+0x2F68(SB)/8, $987
DATA bitrev_size2048_mixed24<>+0x2F70(SB)/8, $1499
DATA bitrev_size2048_mixed24<>+0x2F78(SB)/8, $2011
DATA bitrev_size2048_mixed24<>+0x2F80(SB)/8, $123
DATA bitrev_size2048_mixed24<>+0x2F88(SB)/8, $635
DATA bitrev_size2048_mixed24<>+0x2F90(SB)/8, $1147
DATA bitrev_size2048_mixed24<>+0x2F98(SB)/8, $1659
DATA bitrev_size2048_mixed24<>+0x2FA0(SB)/8, $251
DATA bitrev_size2048_mixed24<>+0x2FA8(SB)/8, $763
DATA bitrev_size2048_mixed24<>+0x2FB0(SB)/8, $1275
DATA bitrev_size2048_mixed24<>+0x2FB8(SB)/8, $1787
DATA bitrev_size2048_mixed24<>+0x2FC0(SB)/8, $379
DATA bitrev_size2048_mixed24<>+0x2FC8(SB)/8, $891
DATA bitrev_size2048_mixed24<>+0x2FD0(SB)/8, $1403
DATA bitrev_size2048_mixed24<>+0x2FD8(SB)/8, $1915
DATA bitrev_size2048_mixed24<>+0x2FE0(SB)/8, $507
DATA bitrev_size2048_mixed24<>+0x2FE8(SB)/8, $1019
DATA bitrev_size2048_mixed24<>+0x2FF0(SB)/8, $1531
DATA bitrev_size2048_mixed24<>+0x2FF8(SB)/8, $2043
DATA bitrev_size2048_mixed24<>+0x3000(SB)/8, $5
DATA bitrev_size2048_mixed24<>+0x3008(SB)/8, $517
DATA bitrev_size2048_mixed24<>+0x3010(SB)/8, $1029
DATA bitrev_size2048_mixed24<>+0x3018(SB)/8, $1541
DATA bitrev_size2048_mixed24<>+0x3020(SB)/8, $133
DATA bitrev_size2048_mixed24<>+0x3028(SB)/8, $645
DATA bitrev_size2048_mixed24<>+0x3030(SB)/8, $1157
DATA bitrev_size2048_mixed24<>+0x3038(SB)/8, $1669
DATA bitrev_size2048_mixed24<>+0x3040(SB)/8, $261
DATA bitrev_size2048_mixed24<>+0x3048(SB)/8, $773
DATA bitrev_size2048_mixed24<>+0x3050(SB)/8, $1285
DATA bitrev_size2048_mixed24<>+0x3058(SB)/8, $1797
DATA bitrev_size2048_mixed24<>+0x3060(SB)/8, $389
DATA bitrev_size2048_mixed24<>+0x3068(SB)/8, $901
DATA bitrev_size2048_mixed24<>+0x3070(SB)/8, $1413
DATA bitrev_size2048_mixed24<>+0x3078(SB)/8, $1925
DATA bitrev_size2048_mixed24<>+0x3080(SB)/8, $37
DATA bitrev_size2048_mixed24<>+0x3088(SB)/8, $549
DATA bitrev_size2048_mixed24<>+0x3090(SB)/8, $1061
DATA bitrev_size2048_mixed24<>+0x3098(SB)/8, $1573
DATA bitrev_size2048_mixed24<>+0x30A0(SB)/8, $165
DATA bitrev_size2048_mixed24<>+0x30A8(SB)/8, $677
DATA bitrev_size2048_mixed24<>+0x30B0(SB)/8, $1189
DATA bitrev_size2048_mixed24<>+0x30B8(SB)/8, $1701
DATA bitrev_size2048_mixed24<>+0x30C0(SB)/8, $293
DATA bitrev_size2048_mixed24<>+0x30C8(SB)/8, $805
DATA bitrev_size2048_mixed24<>+0x30D0(SB)/8, $1317
DATA bitrev_size2048_mixed24<>+0x30D8(SB)/8, $1829
DATA bitrev_size2048_mixed24<>+0x30E0(SB)/8, $421
DATA bitrev_size2048_mixed24<>+0x30E8(SB)/8, $933
DATA bitrev_size2048_mixed24<>+0x30F0(SB)/8, $1445
DATA bitrev_size2048_mixed24<>+0x30F8(SB)/8, $1957
DATA bitrev_size2048_mixed24<>+0x3100(SB)/8, $69
DATA bitrev_size2048_mixed24<>+0x3108(SB)/8, $581
DATA bitrev_size2048_mixed24<>+0x3110(SB)/8, $1093
DATA bitrev_size2048_mixed24<>+0x3118(SB)/8, $1605
DATA bitrev_size2048_mixed24<>+0x3120(SB)/8, $197
DATA bitrev_size2048_mixed24<>+0x3128(SB)/8, $709
DATA bitrev_size2048_mixed24<>+0x3130(SB)/8, $1221
DATA bitrev_size2048_mixed24<>+0x3138(SB)/8, $1733
DATA bitrev_size2048_mixed24<>+0x3140(SB)/8, $325
DATA bitrev_size2048_mixed24<>+0x3148(SB)/8, $837
DATA bitrev_size2048_mixed24<>+0x3150(SB)/8, $1349
DATA bitrev_size2048_mixed24<>+0x3158(SB)/8, $1861
DATA bitrev_size2048_mixed24<>+0x3160(SB)/8, $453
DATA bitrev_size2048_mixed24<>+0x3168(SB)/8, $965
DATA bitrev_size2048_mixed24<>+0x3170(SB)/8, $1477
DATA bitrev_size2048_mixed24<>+0x3178(SB)/8, $1989
DATA bitrev_size2048_mixed24<>+0x3180(SB)/8, $101
DATA bitrev_size2048_mixed24<>+0x3188(SB)/8, $613
DATA bitrev_size2048_mixed24<>+0x3190(SB)/8, $1125
DATA bitrev_size2048_mixed24<>+0x3198(SB)/8, $1637
DATA bitrev_size2048_mixed24<>+0x31A0(SB)/8, $229
DATA bitrev_size2048_mixed24<>+0x31A8(SB)/8, $741
DATA bitrev_size2048_mixed24<>+0x31B0(SB)/8, $1253
DATA bitrev_size2048_mixed24<>+0x31B8(SB)/8, $1765
DATA bitrev_size2048_mixed24<>+0x31C0(SB)/8, $357
DATA bitrev_size2048_mixed24<>+0x31C8(SB)/8, $869
DATA bitrev_size2048_mixed24<>+0x31D0(SB)/8, $1381
DATA bitrev_size2048_mixed24<>+0x31D8(SB)/8, $1893
DATA bitrev_size2048_mixed24<>+0x31E0(SB)/8, $485
DATA bitrev_size2048_mixed24<>+0x31E8(SB)/8, $997
DATA bitrev_size2048_mixed24<>+0x31F0(SB)/8, $1509
DATA bitrev_size2048_mixed24<>+0x31F8(SB)/8, $2021
DATA bitrev_size2048_mixed24<>+0x3200(SB)/8, $13
DATA bitrev_size2048_mixed24<>+0x3208(SB)/8, $525
DATA bitrev_size2048_mixed24<>+0x3210(SB)/8, $1037
DATA bitrev_size2048_mixed24<>+0x3218(SB)/8, $1549
DATA bitrev_size2048_mixed24<>+0x3220(SB)/8, $141
DATA bitrev_size2048_mixed24<>+0x3228(SB)/8, $653
DATA bitrev_size2048_mixed24<>+0x3230(SB)/8, $1165
DATA bitrev_size2048_mixed24<>+0x3238(SB)/8, $1677
DATA bitrev_size2048_mixed24<>+0x3240(SB)/8, $269
DATA bitrev_size2048_mixed24<>+0x3248(SB)/8, $781
DATA bitrev_size2048_mixed24<>+0x3250(SB)/8, $1293
DATA bitrev_size2048_mixed24<>+0x3258(SB)/8, $1805
DATA bitrev_size2048_mixed24<>+0x3260(SB)/8, $397
DATA bitrev_size2048_mixed24<>+0x3268(SB)/8, $909
DATA bitrev_size2048_mixed24<>+0x3270(SB)/8, $1421
DATA bitrev_size2048_mixed24<>+0x3278(SB)/8, $1933
DATA bitrev_size2048_mixed24<>+0x3280(SB)/8, $45
DATA bitrev_size2048_mixed24<>+0x3288(SB)/8, $557
DATA bitrev_size2048_mixed24<>+0x3290(SB)/8, $1069
DATA bitrev_size2048_mixed24<>+0x3298(SB)/8, $1581
DATA bitrev_size2048_mixed24<>+0x32A0(SB)/8, $173
DATA bitrev_size2048_mixed24<>+0x32A8(SB)/8, $685
DATA bitrev_size2048_mixed24<>+0x32B0(SB)/8, $1197
DATA bitrev_size2048_mixed24<>+0x32B8(SB)/8, $1709
DATA bitrev_size2048_mixed24<>+0x32C0(SB)/8, $301
DATA bitrev_size2048_mixed24<>+0x32C8(SB)/8, $813
DATA bitrev_size2048_mixed24<>+0x32D0(SB)/8, $1325
DATA bitrev_size2048_mixed24<>+0x32D8(SB)/8, $1837
DATA bitrev_size2048_mixed24<>+0x32E0(SB)/8, $429
DATA bitrev_size2048_mixed24<>+0x32E8(SB)/8, $941
DATA bitrev_size2048_mixed24<>+0x32F0(SB)/8, $1453
DATA bitrev_size2048_mixed24<>+0x32F8(SB)/8, $1965
DATA bitrev_size2048_mixed24<>+0x3300(SB)/8, $77
DATA bitrev_size2048_mixed24<>+0x3308(SB)/8, $589
DATA bitrev_size2048_mixed24<>+0x3310(SB)/8, $1101
DATA bitrev_size2048_mixed24<>+0x3318(SB)/8, $1613
DATA bitrev_size2048_mixed24<>+0x3320(SB)/8, $205
DATA bitrev_size2048_mixed24<>+0x3328(SB)/8, $717
DATA bitrev_size2048_mixed24<>+0x3330(SB)/8, $1229
DATA bitrev_size2048_mixed24<>+0x3338(SB)/8, $1741
DATA bitrev_size2048_mixed24<>+0x3340(SB)/8, $333
DATA bitrev_size2048_mixed24<>+0x3348(SB)/8, $845
DATA bitrev_size2048_mixed24<>+0x3350(SB)/8, $1357
DATA bitrev_size2048_mixed24<>+0x3358(SB)/8, $1869
DATA bitrev_size2048_mixed24<>+0x3360(SB)/8, $461
DATA bitrev_size2048_mixed24<>+0x3368(SB)/8, $973
DATA bitrev_size2048_mixed24<>+0x3370(SB)/8, $1485
DATA bitrev_size2048_mixed24<>+0x3378(SB)/8, $1997
DATA bitrev_size2048_mixed24<>+0x3380(SB)/8, $109
DATA bitrev_size2048_mixed24<>+0x3388(SB)/8, $621
DATA bitrev_size2048_mixed24<>+0x3390(SB)/8, $1133
DATA bitrev_size2048_mixed24<>+0x3398(SB)/8, $1645
DATA bitrev_size2048_mixed24<>+0x33A0(SB)/8, $237
DATA bitrev_size2048_mixed24<>+0x33A8(SB)/8, $749
DATA bitrev_size2048_mixed24<>+0x33B0(SB)/8, $1261
DATA bitrev_size2048_mixed24<>+0x33B8(SB)/8, $1773
DATA bitrev_size2048_mixed24<>+0x33C0(SB)/8, $365
DATA bitrev_size2048_mixed24<>+0x33C8(SB)/8, $877
DATA bitrev_size2048_mixed24<>+0x33D0(SB)/8, $1389
DATA bitrev_size2048_mixed24<>+0x33D8(SB)/8, $1901
DATA bitrev_size2048_mixed24<>+0x33E0(SB)/8, $493
DATA bitrev_size2048_mixed24<>+0x33E8(SB)/8, $1005
DATA bitrev_size2048_mixed24<>+0x33F0(SB)/8, $1517
DATA bitrev_size2048_mixed24<>+0x33F8(SB)/8, $2029
DATA bitrev_size2048_mixed24<>+0x3400(SB)/8, $21
DATA bitrev_size2048_mixed24<>+0x3408(SB)/8, $533
DATA bitrev_size2048_mixed24<>+0x3410(SB)/8, $1045
DATA bitrev_size2048_mixed24<>+0x3418(SB)/8, $1557
DATA bitrev_size2048_mixed24<>+0x3420(SB)/8, $149
DATA bitrev_size2048_mixed24<>+0x3428(SB)/8, $661
DATA bitrev_size2048_mixed24<>+0x3430(SB)/8, $1173
DATA bitrev_size2048_mixed24<>+0x3438(SB)/8, $1685
DATA bitrev_size2048_mixed24<>+0x3440(SB)/8, $277
DATA bitrev_size2048_mixed24<>+0x3448(SB)/8, $789
DATA bitrev_size2048_mixed24<>+0x3450(SB)/8, $1301
DATA bitrev_size2048_mixed24<>+0x3458(SB)/8, $1813
DATA bitrev_size2048_mixed24<>+0x3460(SB)/8, $405
DATA bitrev_size2048_mixed24<>+0x3468(SB)/8, $917
DATA bitrev_size2048_mixed24<>+0x3470(SB)/8, $1429
DATA bitrev_size2048_mixed24<>+0x3478(SB)/8, $1941
DATA bitrev_size2048_mixed24<>+0x3480(SB)/8, $53
DATA bitrev_size2048_mixed24<>+0x3488(SB)/8, $565
DATA bitrev_size2048_mixed24<>+0x3490(SB)/8, $1077
DATA bitrev_size2048_mixed24<>+0x3498(SB)/8, $1589
DATA bitrev_size2048_mixed24<>+0x34A0(SB)/8, $181
DATA bitrev_size2048_mixed24<>+0x34A8(SB)/8, $693
DATA bitrev_size2048_mixed24<>+0x34B0(SB)/8, $1205
DATA bitrev_size2048_mixed24<>+0x34B8(SB)/8, $1717
DATA bitrev_size2048_mixed24<>+0x34C0(SB)/8, $309
DATA bitrev_size2048_mixed24<>+0x34C8(SB)/8, $821
DATA bitrev_size2048_mixed24<>+0x34D0(SB)/8, $1333
DATA bitrev_size2048_mixed24<>+0x34D8(SB)/8, $1845
DATA bitrev_size2048_mixed24<>+0x34E0(SB)/8, $437
DATA bitrev_size2048_mixed24<>+0x34E8(SB)/8, $949
DATA bitrev_size2048_mixed24<>+0x34F0(SB)/8, $1461
DATA bitrev_size2048_mixed24<>+0x34F8(SB)/8, $1973
DATA bitrev_size2048_mixed24<>+0x3500(SB)/8, $85
DATA bitrev_size2048_mixed24<>+0x3508(SB)/8, $597
DATA bitrev_size2048_mixed24<>+0x3510(SB)/8, $1109
DATA bitrev_size2048_mixed24<>+0x3518(SB)/8, $1621
DATA bitrev_size2048_mixed24<>+0x3520(SB)/8, $213
DATA bitrev_size2048_mixed24<>+0x3528(SB)/8, $725
DATA bitrev_size2048_mixed24<>+0x3530(SB)/8, $1237
DATA bitrev_size2048_mixed24<>+0x3538(SB)/8, $1749
DATA bitrev_size2048_mixed24<>+0x3540(SB)/8, $341
DATA bitrev_size2048_mixed24<>+0x3548(SB)/8, $853
DATA bitrev_size2048_mixed24<>+0x3550(SB)/8, $1365
DATA bitrev_size2048_mixed24<>+0x3558(SB)/8, $1877
DATA bitrev_size2048_mixed24<>+0x3560(SB)/8, $469
DATA bitrev_size2048_mixed24<>+0x3568(SB)/8, $981
DATA bitrev_size2048_mixed24<>+0x3570(SB)/8, $1493
DATA bitrev_size2048_mixed24<>+0x3578(SB)/8, $2005
DATA bitrev_size2048_mixed24<>+0x3580(SB)/8, $117
DATA bitrev_size2048_mixed24<>+0x3588(SB)/8, $629
DATA bitrev_size2048_mixed24<>+0x3590(SB)/8, $1141
DATA bitrev_size2048_mixed24<>+0x3598(SB)/8, $1653
DATA bitrev_size2048_mixed24<>+0x35A0(SB)/8, $245
DATA bitrev_size2048_mixed24<>+0x35A8(SB)/8, $757
DATA bitrev_size2048_mixed24<>+0x35B0(SB)/8, $1269
DATA bitrev_size2048_mixed24<>+0x35B8(SB)/8, $1781
DATA bitrev_size2048_mixed24<>+0x35C0(SB)/8, $373
DATA bitrev_size2048_mixed24<>+0x35C8(SB)/8, $885
DATA bitrev_size2048_mixed24<>+0x35D0(SB)/8, $1397
DATA bitrev_size2048_mixed24<>+0x35D8(SB)/8, $1909
DATA bitrev_size2048_mixed24<>+0x35E0(SB)/8, $501
DATA bitrev_size2048_mixed24<>+0x35E8(SB)/8, $1013
DATA bitrev_size2048_mixed24<>+0x35F0(SB)/8, $1525
DATA bitrev_size2048_mixed24<>+0x35F8(SB)/8, $2037
DATA bitrev_size2048_mixed24<>+0x3600(SB)/8, $29
DATA bitrev_size2048_mixed24<>+0x3608(SB)/8, $541
DATA bitrev_size2048_mixed24<>+0x3610(SB)/8, $1053
DATA bitrev_size2048_mixed24<>+0x3618(SB)/8, $1565
DATA bitrev_size2048_mixed24<>+0x3620(SB)/8, $157
DATA bitrev_size2048_mixed24<>+0x3628(SB)/8, $669
DATA bitrev_size2048_mixed24<>+0x3630(SB)/8, $1181
DATA bitrev_size2048_mixed24<>+0x3638(SB)/8, $1693
DATA bitrev_size2048_mixed24<>+0x3640(SB)/8, $285
DATA bitrev_size2048_mixed24<>+0x3648(SB)/8, $797
DATA bitrev_size2048_mixed24<>+0x3650(SB)/8, $1309
DATA bitrev_size2048_mixed24<>+0x3658(SB)/8, $1821
DATA bitrev_size2048_mixed24<>+0x3660(SB)/8, $413
DATA bitrev_size2048_mixed24<>+0x3668(SB)/8, $925
DATA bitrev_size2048_mixed24<>+0x3670(SB)/8, $1437
DATA bitrev_size2048_mixed24<>+0x3678(SB)/8, $1949
DATA bitrev_size2048_mixed24<>+0x3680(SB)/8, $61
DATA bitrev_size2048_mixed24<>+0x3688(SB)/8, $573
DATA bitrev_size2048_mixed24<>+0x3690(SB)/8, $1085
DATA bitrev_size2048_mixed24<>+0x3698(SB)/8, $1597
DATA bitrev_size2048_mixed24<>+0x36A0(SB)/8, $189
DATA bitrev_size2048_mixed24<>+0x36A8(SB)/8, $701
DATA bitrev_size2048_mixed24<>+0x36B0(SB)/8, $1213
DATA bitrev_size2048_mixed24<>+0x36B8(SB)/8, $1725
DATA bitrev_size2048_mixed24<>+0x36C0(SB)/8, $317
DATA bitrev_size2048_mixed24<>+0x36C8(SB)/8, $829
DATA bitrev_size2048_mixed24<>+0x36D0(SB)/8, $1341
DATA bitrev_size2048_mixed24<>+0x36D8(SB)/8, $1853
DATA bitrev_size2048_mixed24<>+0x36E0(SB)/8, $445
DATA bitrev_size2048_mixed24<>+0x36E8(SB)/8, $957
DATA bitrev_size2048_mixed24<>+0x36F0(SB)/8, $1469
DATA bitrev_size2048_mixed24<>+0x36F8(SB)/8, $1981
DATA bitrev_size2048_mixed24<>+0x3700(SB)/8, $93
DATA bitrev_size2048_mixed24<>+0x3708(SB)/8, $605
DATA bitrev_size2048_mixed24<>+0x3710(SB)/8, $1117
DATA bitrev_size2048_mixed24<>+0x3718(SB)/8, $1629
DATA bitrev_size2048_mixed24<>+0x3720(SB)/8, $221
DATA bitrev_size2048_mixed24<>+0x3728(SB)/8, $733
DATA bitrev_size2048_mixed24<>+0x3730(SB)/8, $1245
DATA bitrev_size2048_mixed24<>+0x3738(SB)/8, $1757
DATA bitrev_size2048_mixed24<>+0x3740(SB)/8, $349
DATA bitrev_size2048_mixed24<>+0x3748(SB)/8, $861
DATA bitrev_size2048_mixed24<>+0x3750(SB)/8, $1373
DATA bitrev_size2048_mixed24<>+0x3758(SB)/8, $1885
DATA bitrev_size2048_mixed24<>+0x3760(SB)/8, $477
DATA bitrev_size2048_mixed24<>+0x3768(SB)/8, $989
DATA bitrev_size2048_mixed24<>+0x3770(SB)/8, $1501
DATA bitrev_size2048_mixed24<>+0x3778(SB)/8, $2013
DATA bitrev_size2048_mixed24<>+0x3780(SB)/8, $125
DATA bitrev_size2048_mixed24<>+0x3788(SB)/8, $637
DATA bitrev_size2048_mixed24<>+0x3790(SB)/8, $1149
DATA bitrev_size2048_mixed24<>+0x3798(SB)/8, $1661
DATA bitrev_size2048_mixed24<>+0x37A0(SB)/8, $253
DATA bitrev_size2048_mixed24<>+0x37A8(SB)/8, $765
DATA bitrev_size2048_mixed24<>+0x37B0(SB)/8, $1277
DATA bitrev_size2048_mixed24<>+0x37B8(SB)/8, $1789
DATA bitrev_size2048_mixed24<>+0x37C0(SB)/8, $381
DATA bitrev_size2048_mixed24<>+0x37C8(SB)/8, $893
DATA bitrev_size2048_mixed24<>+0x37D0(SB)/8, $1405
DATA bitrev_size2048_mixed24<>+0x37D8(SB)/8, $1917
DATA bitrev_size2048_mixed24<>+0x37E0(SB)/8, $509
DATA bitrev_size2048_mixed24<>+0x37E8(SB)/8, $1021
DATA bitrev_size2048_mixed24<>+0x37F0(SB)/8, $1533
DATA bitrev_size2048_mixed24<>+0x37F8(SB)/8, $2045
DATA bitrev_size2048_mixed24<>+0x3800(SB)/8, $7
DATA bitrev_size2048_mixed24<>+0x3808(SB)/8, $519
DATA bitrev_size2048_mixed24<>+0x3810(SB)/8, $1031
DATA bitrev_size2048_mixed24<>+0x3818(SB)/8, $1543
DATA bitrev_size2048_mixed24<>+0x3820(SB)/8, $135
DATA bitrev_size2048_mixed24<>+0x3828(SB)/8, $647
DATA bitrev_size2048_mixed24<>+0x3830(SB)/8, $1159
DATA bitrev_size2048_mixed24<>+0x3838(SB)/8, $1671
DATA bitrev_size2048_mixed24<>+0x3840(SB)/8, $263
DATA bitrev_size2048_mixed24<>+0x3848(SB)/8, $775
DATA bitrev_size2048_mixed24<>+0x3850(SB)/8, $1287
DATA bitrev_size2048_mixed24<>+0x3858(SB)/8, $1799
DATA bitrev_size2048_mixed24<>+0x3860(SB)/8, $391
DATA bitrev_size2048_mixed24<>+0x3868(SB)/8, $903
DATA bitrev_size2048_mixed24<>+0x3870(SB)/8, $1415
DATA bitrev_size2048_mixed24<>+0x3878(SB)/8, $1927
DATA bitrev_size2048_mixed24<>+0x3880(SB)/8, $39
DATA bitrev_size2048_mixed24<>+0x3888(SB)/8, $551
DATA bitrev_size2048_mixed24<>+0x3890(SB)/8, $1063
DATA bitrev_size2048_mixed24<>+0x3898(SB)/8, $1575
DATA bitrev_size2048_mixed24<>+0x38A0(SB)/8, $167
DATA bitrev_size2048_mixed24<>+0x38A8(SB)/8, $679
DATA bitrev_size2048_mixed24<>+0x38B0(SB)/8, $1191
DATA bitrev_size2048_mixed24<>+0x38B8(SB)/8, $1703
DATA bitrev_size2048_mixed24<>+0x38C0(SB)/8, $295
DATA bitrev_size2048_mixed24<>+0x38C8(SB)/8, $807
DATA bitrev_size2048_mixed24<>+0x38D0(SB)/8, $1319
DATA bitrev_size2048_mixed24<>+0x38D8(SB)/8, $1831
DATA bitrev_size2048_mixed24<>+0x38E0(SB)/8, $423
DATA bitrev_size2048_mixed24<>+0x38E8(SB)/8, $935
DATA bitrev_size2048_mixed24<>+0x38F0(SB)/8, $1447
DATA bitrev_size2048_mixed24<>+0x38F8(SB)/8, $1959
DATA bitrev_size2048_mixed24<>+0x3900(SB)/8, $71
DATA bitrev_size2048_mixed24<>+0x3908(SB)/8, $583
DATA bitrev_size2048_mixed24<>+0x3910(SB)/8, $1095
DATA bitrev_size2048_mixed24<>+0x3918(SB)/8, $1607
DATA bitrev_size2048_mixed24<>+0x3920(SB)/8, $199
DATA bitrev_size2048_mixed24<>+0x3928(SB)/8, $711
DATA bitrev_size2048_mixed24<>+0x3930(SB)/8, $1223
DATA bitrev_size2048_mixed24<>+0x3938(SB)/8, $1735
DATA bitrev_size2048_mixed24<>+0x3940(SB)/8, $327
DATA bitrev_size2048_mixed24<>+0x3948(SB)/8, $839
DATA bitrev_size2048_mixed24<>+0x3950(SB)/8, $1351
DATA bitrev_size2048_mixed24<>+0x3958(SB)/8, $1863
DATA bitrev_size2048_mixed24<>+0x3960(SB)/8, $455
DATA bitrev_size2048_mixed24<>+0x3968(SB)/8, $967
DATA bitrev_size2048_mixed24<>+0x3970(SB)/8, $1479
DATA bitrev_size2048_mixed24<>+0x3978(SB)/8, $1991
DATA bitrev_size2048_mixed24<>+0x3980(SB)/8, $103
DATA bitrev_size2048_mixed24<>+0x3988(SB)/8, $615
DATA bitrev_size2048_mixed24<>+0x3990(SB)/8, $1127
DATA bitrev_size2048_mixed24<>+0x3998(SB)/8, $1639
DATA bitrev_size2048_mixed24<>+0x39A0(SB)/8, $231
DATA bitrev_size2048_mixed24<>+0x39A8(SB)/8, $743
DATA bitrev_size2048_mixed24<>+0x39B0(SB)/8, $1255
DATA bitrev_size2048_mixed24<>+0x39B8(SB)/8, $1767
DATA bitrev_size2048_mixed24<>+0x39C0(SB)/8, $359
DATA bitrev_size2048_mixed24<>+0x39C8(SB)/8, $871
DATA bitrev_size2048_mixed24<>+0x39D0(SB)/8, $1383
DATA bitrev_size2048_mixed24<>+0x39D8(SB)/8, $1895
DATA bitrev_size2048_mixed24<>+0x39E0(SB)/8, $487
DATA bitrev_size2048_mixed24<>+0x39E8(SB)/8, $999
DATA bitrev_size2048_mixed24<>+0x39F0(SB)/8, $1511
DATA bitrev_size2048_mixed24<>+0x39F8(SB)/8, $2023
DATA bitrev_size2048_mixed24<>+0x3A00(SB)/8, $15
DATA bitrev_size2048_mixed24<>+0x3A08(SB)/8, $527
DATA bitrev_size2048_mixed24<>+0x3A10(SB)/8, $1039
DATA bitrev_size2048_mixed24<>+0x3A18(SB)/8, $1551
DATA bitrev_size2048_mixed24<>+0x3A20(SB)/8, $143
DATA bitrev_size2048_mixed24<>+0x3A28(SB)/8, $655
DATA bitrev_size2048_mixed24<>+0x3A30(SB)/8, $1167
DATA bitrev_size2048_mixed24<>+0x3A38(SB)/8, $1679
DATA bitrev_size2048_mixed24<>+0x3A40(SB)/8, $271
DATA bitrev_size2048_mixed24<>+0x3A48(SB)/8, $783
DATA bitrev_size2048_mixed24<>+0x3A50(SB)/8, $1295
DATA bitrev_size2048_mixed24<>+0x3A58(SB)/8, $1807
DATA bitrev_size2048_mixed24<>+0x3A60(SB)/8, $399
DATA bitrev_size2048_mixed24<>+0x3A68(SB)/8, $911
DATA bitrev_size2048_mixed24<>+0x3A70(SB)/8, $1423
DATA bitrev_size2048_mixed24<>+0x3A78(SB)/8, $1935
DATA bitrev_size2048_mixed24<>+0x3A80(SB)/8, $47
DATA bitrev_size2048_mixed24<>+0x3A88(SB)/8, $559
DATA bitrev_size2048_mixed24<>+0x3A90(SB)/8, $1071
DATA bitrev_size2048_mixed24<>+0x3A98(SB)/8, $1583
DATA bitrev_size2048_mixed24<>+0x3AA0(SB)/8, $175
DATA bitrev_size2048_mixed24<>+0x3AA8(SB)/8, $687
DATA bitrev_size2048_mixed24<>+0x3AB0(SB)/8, $1199
DATA bitrev_size2048_mixed24<>+0x3AB8(SB)/8, $1711
DATA bitrev_size2048_mixed24<>+0x3AC0(SB)/8, $303
DATA bitrev_size2048_mixed24<>+0x3AC8(SB)/8, $815
DATA bitrev_size2048_mixed24<>+0x3AD0(SB)/8, $1327
DATA bitrev_size2048_mixed24<>+0x3AD8(SB)/8, $1839
DATA bitrev_size2048_mixed24<>+0x3AE0(SB)/8, $431
DATA bitrev_size2048_mixed24<>+0x3AE8(SB)/8, $943
DATA bitrev_size2048_mixed24<>+0x3AF0(SB)/8, $1455
DATA bitrev_size2048_mixed24<>+0x3AF8(SB)/8, $1967
DATA bitrev_size2048_mixed24<>+0x3B00(SB)/8, $79
DATA bitrev_size2048_mixed24<>+0x3B08(SB)/8, $591
DATA bitrev_size2048_mixed24<>+0x3B10(SB)/8, $1103
DATA bitrev_size2048_mixed24<>+0x3B18(SB)/8, $1615
DATA bitrev_size2048_mixed24<>+0x3B20(SB)/8, $207
DATA bitrev_size2048_mixed24<>+0x3B28(SB)/8, $719
DATA bitrev_size2048_mixed24<>+0x3B30(SB)/8, $1231
DATA bitrev_size2048_mixed24<>+0x3B38(SB)/8, $1743
DATA bitrev_size2048_mixed24<>+0x3B40(SB)/8, $335
DATA bitrev_size2048_mixed24<>+0x3B48(SB)/8, $847
DATA bitrev_size2048_mixed24<>+0x3B50(SB)/8, $1359
DATA bitrev_size2048_mixed24<>+0x3B58(SB)/8, $1871
DATA bitrev_size2048_mixed24<>+0x3B60(SB)/8, $463
DATA bitrev_size2048_mixed24<>+0x3B68(SB)/8, $975
DATA bitrev_size2048_mixed24<>+0x3B70(SB)/8, $1487
DATA bitrev_size2048_mixed24<>+0x3B78(SB)/8, $1999
DATA bitrev_size2048_mixed24<>+0x3B80(SB)/8, $111
DATA bitrev_size2048_mixed24<>+0x3B88(SB)/8, $623
DATA bitrev_size2048_mixed24<>+0x3B90(SB)/8, $1135
DATA bitrev_size2048_mixed24<>+0x3B98(SB)/8, $1647
DATA bitrev_size2048_mixed24<>+0x3BA0(SB)/8, $239
DATA bitrev_size2048_mixed24<>+0x3BA8(SB)/8, $751
DATA bitrev_size2048_mixed24<>+0x3BB0(SB)/8, $1263
DATA bitrev_size2048_mixed24<>+0x3BB8(SB)/8, $1775
DATA bitrev_size2048_mixed24<>+0x3BC0(SB)/8, $367
DATA bitrev_size2048_mixed24<>+0x3BC8(SB)/8, $879
DATA bitrev_size2048_mixed24<>+0x3BD0(SB)/8, $1391
DATA bitrev_size2048_mixed24<>+0x3BD8(SB)/8, $1903
DATA bitrev_size2048_mixed24<>+0x3BE0(SB)/8, $495
DATA bitrev_size2048_mixed24<>+0x3BE8(SB)/8, $1007
DATA bitrev_size2048_mixed24<>+0x3BF0(SB)/8, $1519
DATA bitrev_size2048_mixed24<>+0x3BF8(SB)/8, $2031
DATA bitrev_size2048_mixed24<>+0x3C00(SB)/8, $23
DATA bitrev_size2048_mixed24<>+0x3C08(SB)/8, $535
DATA bitrev_size2048_mixed24<>+0x3C10(SB)/8, $1047
DATA bitrev_size2048_mixed24<>+0x3C18(SB)/8, $1559
DATA bitrev_size2048_mixed24<>+0x3C20(SB)/8, $151
DATA bitrev_size2048_mixed24<>+0x3C28(SB)/8, $663
DATA bitrev_size2048_mixed24<>+0x3C30(SB)/8, $1175
DATA bitrev_size2048_mixed24<>+0x3C38(SB)/8, $1687
DATA bitrev_size2048_mixed24<>+0x3C40(SB)/8, $279
DATA bitrev_size2048_mixed24<>+0x3C48(SB)/8, $791
DATA bitrev_size2048_mixed24<>+0x3C50(SB)/8, $1303
DATA bitrev_size2048_mixed24<>+0x3C58(SB)/8, $1815
DATA bitrev_size2048_mixed24<>+0x3C60(SB)/8, $407
DATA bitrev_size2048_mixed24<>+0x3C68(SB)/8, $919
DATA bitrev_size2048_mixed24<>+0x3C70(SB)/8, $1431
DATA bitrev_size2048_mixed24<>+0x3C78(SB)/8, $1943
DATA bitrev_size2048_mixed24<>+0x3C80(SB)/8, $55
DATA bitrev_size2048_mixed24<>+0x3C88(SB)/8, $567
DATA bitrev_size2048_mixed24<>+0x3C90(SB)/8, $1079
DATA bitrev_size2048_mixed24<>+0x3C98(SB)/8, $1591
DATA bitrev_size2048_mixed24<>+0x3CA0(SB)/8, $183
DATA bitrev_size2048_mixed24<>+0x3CA8(SB)/8, $695
DATA bitrev_size2048_mixed24<>+0x3CB0(SB)/8, $1207
DATA bitrev_size2048_mixed24<>+0x3CB8(SB)/8, $1719
DATA bitrev_size2048_mixed24<>+0x3CC0(SB)/8, $311
DATA bitrev_size2048_mixed24<>+0x3CC8(SB)/8, $823
DATA bitrev_size2048_mixed24<>+0x3CD0(SB)/8, $1335
DATA bitrev_size2048_mixed24<>+0x3CD8(SB)/8, $1847
DATA bitrev_size2048_mixed24<>+0x3CE0(SB)/8, $439
DATA bitrev_size2048_mixed24<>+0x3CE8(SB)/8, $951
DATA bitrev_size2048_mixed24<>+0x3CF0(SB)/8, $1463
DATA bitrev_size2048_mixed24<>+0x3CF8(SB)/8, $1975
DATA bitrev_size2048_mixed24<>+0x3D00(SB)/8, $87
DATA bitrev_size2048_mixed24<>+0x3D08(SB)/8, $599
DATA bitrev_size2048_mixed24<>+0x3D10(SB)/8, $1111
DATA bitrev_size2048_mixed24<>+0x3D18(SB)/8, $1623
DATA bitrev_size2048_mixed24<>+0x3D20(SB)/8, $215
DATA bitrev_size2048_mixed24<>+0x3D28(SB)/8, $727
DATA bitrev_size2048_mixed24<>+0x3D30(SB)/8, $1239
DATA bitrev_size2048_mixed24<>+0x3D38(SB)/8, $1751
DATA bitrev_size2048_mixed24<>+0x3D40(SB)/8, $343
DATA bitrev_size2048_mixed24<>+0x3D48(SB)/8, $855
DATA bitrev_size2048_mixed24<>+0x3D50(SB)/8, $1367
DATA bitrev_size2048_mixed24<>+0x3D58(SB)/8, $1879
DATA bitrev_size2048_mixed24<>+0x3D60(SB)/8, $471
DATA bitrev_size2048_mixed24<>+0x3D68(SB)/8, $983
DATA bitrev_size2048_mixed24<>+0x3D70(SB)/8, $1495
DATA bitrev_size2048_mixed24<>+0x3D78(SB)/8, $2007
DATA bitrev_size2048_mixed24<>+0x3D80(SB)/8, $119
DATA bitrev_size2048_mixed24<>+0x3D88(SB)/8, $631
DATA bitrev_size2048_mixed24<>+0x3D90(SB)/8, $1143
DATA bitrev_size2048_mixed24<>+0x3D98(SB)/8, $1655
DATA bitrev_size2048_mixed24<>+0x3DA0(SB)/8, $247
DATA bitrev_size2048_mixed24<>+0x3DA8(SB)/8, $759
DATA bitrev_size2048_mixed24<>+0x3DB0(SB)/8, $1271
DATA bitrev_size2048_mixed24<>+0x3DB8(SB)/8, $1783
DATA bitrev_size2048_mixed24<>+0x3DC0(SB)/8, $375
DATA bitrev_size2048_mixed24<>+0x3DC8(SB)/8, $887
DATA bitrev_size2048_mixed24<>+0x3DD0(SB)/8, $1399
DATA bitrev_size2048_mixed24<>+0x3DD8(SB)/8, $1911
DATA bitrev_size2048_mixed24<>+0x3DE0(SB)/8, $503
DATA bitrev_size2048_mixed24<>+0x3DE8(SB)/8, $1015
DATA bitrev_size2048_mixed24<>+0x3DF0(SB)/8, $1527
DATA bitrev_size2048_mixed24<>+0x3DF8(SB)/8, $2039
DATA bitrev_size2048_mixed24<>+0x3E00(SB)/8, $31
DATA bitrev_size2048_mixed24<>+0x3E08(SB)/8, $543
DATA bitrev_size2048_mixed24<>+0x3E10(SB)/8, $1055
DATA bitrev_size2048_mixed24<>+0x3E18(SB)/8, $1567
DATA bitrev_size2048_mixed24<>+0x3E20(SB)/8, $159
DATA bitrev_size2048_mixed24<>+0x3E28(SB)/8, $671
DATA bitrev_size2048_mixed24<>+0x3E30(SB)/8, $1183
DATA bitrev_size2048_mixed24<>+0x3E38(SB)/8, $1695
DATA bitrev_size2048_mixed24<>+0x3E40(SB)/8, $287
DATA bitrev_size2048_mixed24<>+0x3E48(SB)/8, $799
DATA bitrev_size2048_mixed24<>+0x3E50(SB)/8, $1311
DATA bitrev_size2048_mixed24<>+0x3E58(SB)/8, $1823
DATA bitrev_size2048_mixed24<>+0x3E60(SB)/8, $415
DATA bitrev_size2048_mixed24<>+0x3E68(SB)/8, $927
DATA bitrev_size2048_mixed24<>+0x3E70(SB)/8, $1439
DATA bitrev_size2048_mixed24<>+0x3E78(SB)/8, $1951
DATA bitrev_size2048_mixed24<>+0x3E80(SB)/8, $63
DATA bitrev_size2048_mixed24<>+0x3E88(SB)/8, $575
DATA bitrev_size2048_mixed24<>+0x3E90(SB)/8, $1087
DATA bitrev_size2048_mixed24<>+0x3E98(SB)/8, $1599
DATA bitrev_size2048_mixed24<>+0x3EA0(SB)/8, $191
DATA bitrev_size2048_mixed24<>+0x3EA8(SB)/8, $703
DATA bitrev_size2048_mixed24<>+0x3EB0(SB)/8, $1215
DATA bitrev_size2048_mixed24<>+0x3EB8(SB)/8, $1727
DATA bitrev_size2048_mixed24<>+0x3EC0(SB)/8, $319
DATA bitrev_size2048_mixed24<>+0x3EC8(SB)/8, $831
DATA bitrev_size2048_mixed24<>+0x3ED0(SB)/8, $1343
DATA bitrev_size2048_mixed24<>+0x3ED8(SB)/8, $1855
DATA bitrev_size2048_mixed24<>+0x3EE0(SB)/8, $447
DATA bitrev_size2048_mixed24<>+0x3EE8(SB)/8, $959
DATA bitrev_size2048_mixed24<>+0x3EF0(SB)/8, $1471
DATA bitrev_size2048_mixed24<>+0x3EF8(SB)/8, $1983
DATA bitrev_size2048_mixed24<>+0x3F00(SB)/8, $95
DATA bitrev_size2048_mixed24<>+0x3F08(SB)/8, $607
DATA bitrev_size2048_mixed24<>+0x3F10(SB)/8, $1119
DATA bitrev_size2048_mixed24<>+0x3F18(SB)/8, $1631
DATA bitrev_size2048_mixed24<>+0x3F20(SB)/8, $223
DATA bitrev_size2048_mixed24<>+0x3F28(SB)/8, $735
DATA bitrev_size2048_mixed24<>+0x3F30(SB)/8, $1247
DATA bitrev_size2048_mixed24<>+0x3F38(SB)/8, $1759
DATA bitrev_size2048_mixed24<>+0x3F40(SB)/8, $351
DATA bitrev_size2048_mixed24<>+0x3F48(SB)/8, $863
DATA bitrev_size2048_mixed24<>+0x3F50(SB)/8, $1375
DATA bitrev_size2048_mixed24<>+0x3F58(SB)/8, $1887
DATA bitrev_size2048_mixed24<>+0x3F60(SB)/8, $479
DATA bitrev_size2048_mixed24<>+0x3F68(SB)/8, $991
DATA bitrev_size2048_mixed24<>+0x3F70(SB)/8, $1503
DATA bitrev_size2048_mixed24<>+0x3F78(SB)/8, $2015
DATA bitrev_size2048_mixed24<>+0x3F80(SB)/8, $127
DATA bitrev_size2048_mixed24<>+0x3F88(SB)/8, $639
DATA bitrev_size2048_mixed24<>+0x3F90(SB)/8, $1151
DATA bitrev_size2048_mixed24<>+0x3F98(SB)/8, $1663
DATA bitrev_size2048_mixed24<>+0x3FA0(SB)/8, $255
DATA bitrev_size2048_mixed24<>+0x3FA8(SB)/8, $767
DATA bitrev_size2048_mixed24<>+0x3FB0(SB)/8, $1279
DATA bitrev_size2048_mixed24<>+0x3FB8(SB)/8, $1791
DATA bitrev_size2048_mixed24<>+0x3FC0(SB)/8, $383
DATA bitrev_size2048_mixed24<>+0x3FC8(SB)/8, $895
DATA bitrev_size2048_mixed24<>+0x3FD0(SB)/8, $1407
DATA bitrev_size2048_mixed24<>+0x3FD8(SB)/8, $1919
DATA bitrev_size2048_mixed24<>+0x3FE0(SB)/8, $511
DATA bitrev_size2048_mixed24<>+0x3FE8(SB)/8, $1023
DATA bitrev_size2048_mixed24<>+0x3FF0(SB)/8, $1535
DATA bitrev_size2048_mixed24<>+0x3FF8(SB)/8, $2047


// 1/2048 as float32 bits (verified: exponent 127-11=116=0x74, mantissa 0).
DATA neonInv2048<>+0(SB)/4, $0x3a000000 // 1/2048
GLOBL neonInv2048<>(SB), RODATA, $4
