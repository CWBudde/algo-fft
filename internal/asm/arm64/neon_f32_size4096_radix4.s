//go:build arm64 && !purego

// ===========================================================================
// NEON Size-4096 Radix-4 FFT Kernels for ARM64
// ===========================================================================
//
// Size 4096 = 4^6, radix-4 algorithm uses 6 stages:
//   Stage 1: 1024 groups x 1 butterfly, stride=4,   no twiddle multiply (W^0 = 1)
//   Stage 2: 256 groups x 4 butterflies, twiddle step=256
//   Stage 3: 64 groups x 16 butterflies, twiddle step=64
//   Stage 4: 16 groups x 64 butterflies, twiddle step=16
//   Stage 5: 4 groups x 256 butterflies, twiddle step=4
//   Stage 6: 1 group x 1024 butterflies, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

DATA ·neonInv4096Radix4+0(SB)/4, $0x39800000 // 1/4096
GLOBL ·neonInv4096Radix4(SB), RODATA, $4

// Forward transform, size 4096, complex64, radix-4 variant
TEXT ·ForwardNEONSize4096Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $4096, R13
	BNE  neon4096r4_return_false

	MOVD dst_len+8(FP), R0
	CMP  $4096, R0
	BLT  neon4096r4_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $4096, R0
	BLT  neon4096r4_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $4096, R0
	BLT  neon4096r4_return_false

	// Load static bit-reversal table
	MOVD $bitrev_size4096_radix4<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon4096r4_use_dst
	MOVD R11, R8

neon4096r4_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon4096r4_bitrev_loop:
	CMP  $4096, R0
	BGE  neon4096r4_stage1

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
	B    neon4096r4_bitrev_loop

neon4096r4_stage1:
	// Stage 1: 1024 radix-4 butterflies
	MOVD $0, R0

neon4096r4_stage1_loop:
	CMP  $4096, R0
	BGE  neon4096r4_stage2

	LSL  $3, R0, R1
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

	ADD  $4, R0, R0
	B    neon4096r4_stage1_loop

neon4096r4_stage2:
	// Stage 2: 256 groups x 4 butterflies, twiddle step=256
	MOVD $0, R0

neon4096r4_stage2_outer:
	CMP  $4096, R0
	BGE  neon4096r4_stage3

	MOVD $0, R1

neon4096r4_stage2_inner:
	CMP  $4, R1
	BGE  neon4096r4_stage2_next

	ADD  R0, R1, R2
	ADD  $4, R2, R3
	ADD  $8, R2, R4
	ADD  $12, R2, R5

	// twiddle indices: j*256, j*512, j*768
	LSL  $8, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1

	LSL  $9, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3

	LSL  $8, R1, R6
	LSL  $9, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FNEGS F21, F26
	FMOVS F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FMOVS F21, F30
	FNEGS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon4096r4_stage2_inner

neon4096r4_stage2_next:
	ADD  $16, R0, R0
	B    neon4096r4_stage2_outer

neon4096r4_stage3:
	// Stage 3: 64 groups x 16 butterflies, twiddle step=64
	MOVD $0, R0

neon4096r4_stage3_outer:
	CMP  $4096, R0
	BGE  neon4096r4_stage4

	MOVD $0, R1

neon4096r4_stage3_inner:
	CMP  $16, R1
	BGE  neon4096r4_stage3_next

	ADD  R0, R1, R2
	ADD  $16, R2, R3
	ADD  $32, R2, R4
	ADD  $48, R2, R5

	// twiddle indices: j*64, j*128, j*192
	LSL  $6, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1

	LSL  $7, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3

	LSL  $6, R1, R6
	LSL  $7, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FNEGS F21, F26
	FMOVS F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FMOVS F21, F30
	FNEGS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon4096r4_stage3_inner

neon4096r4_stage3_next:
	ADD  $64, R0, R0
	B    neon4096r4_stage3_outer

neon4096r4_stage4:
	// Stage 4: 16 groups x 64 butterflies, twiddle step=16
	MOVD $0, R0

neon4096r4_stage4_outer:
	CMP  $4096, R0
	BGE  neon4096r4_stage5

	MOVD $0, R1

neon4096r4_stage4_inner:
	CMP  $64, R1
	BGE  neon4096r4_stage4_next

	ADD  R0, R1, R2
	ADD  $64, R2, R3
	ADD  $128, R2, R4
	ADD  $192, R2, R5

	// twiddle indices: j*16, j*32, j*48
	LSL  $4, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1

	LSL  $5, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3

	LSL  $4, R1, R6
	LSL  $5, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FNEGS F21, F26
	FMOVS F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FMOVS F21, F30
	FNEGS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon4096r4_stage4_inner

neon4096r4_stage4_next:
	ADD  $256, R0, R0
	B    neon4096r4_stage4_outer

neon4096r4_stage5:
	// Stage 5: 4 groups x 256 butterflies, twiddle step=4
	MOVD $0, R0

neon4096r4_stage5_outer:
	CMP  $4096, R0
	BGE  neon4096r4_stage6

	MOVD $0, R1

neon4096r4_stage5_inner:
	CMP  $256, R1
	BGE  neon4096r4_stage5_next

	ADD  R0, R1, R2
	ADD  $256, R2, R3
	ADD  $512, R2, R4
	ADD  $768, R2, R5

	// twiddle indices: j*4, j*8, j*12
	LSL  $2, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1

	LSL  $3, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3

	LSL  $2, R1, R6
	LSL  $3, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FNEGS F21, F26
	FMOVS F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FMOVS F21, F30
	FNEGS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon4096r4_stage5_inner

neon4096r4_stage5_next:
	ADD  $1024, R0, R0
	B    neon4096r4_stage5_outer

neon4096r4_stage6:
	// Stage 6: 1 group x 1024 butterflies, twiddle step=1
	MOVD $0, R0

neon4096r4_stage6_loop:
	CMP  $1024, R0
	BGE  neon4096r4_done

	MOVD R0, R1
	ADD  $1024, R1, R2
	ADD  $2048, R1, R3
	ADD  $3072, R1, R4

	// twiddle indices: j, 2j, 3j
	LSL  $3, R1, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1

	LSL  $1, R1, R6
	ADD  R6, R1, R7
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F2
	FMOVS 4(R6), F3

	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F4
	FMOVS 4(R7), F5

	LSL  $3, R1, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F6
	FMOVS 4(R5), F7

	LSL  $3, R2, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F8
	FMOVS 4(R5), F9

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F10
	FMOVS 4(R5), F11

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F12
	FMOVS 4(R5), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FNEGS F21, F26
	FMOVS F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FMOVS F21, F30
	FNEGS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R1, R5
	ADD  R8, R5, R5
	FMOVS F22, 0(R5)
	FMOVS F23, 4(R5)

	LSL  $3, R2, R5
	ADD  R8, R5, R5
	FMOVS F20, 0(R5)
	FMOVS F21, 4(R5)

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS F24, 0(R5)
	FMOVS F25, 4(R5)

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS F28, 0(R5)
	FMOVS F29, 4(R5)

	ADD  $1, R0, R0
	B    neon4096r4_stage6_loop

neon4096r4_done:
	CMP  R8, R20
	BEQ  neon4096r4_return_true

	MOVD $0, R0
neon4096r4_copy_loop:
	CMP  $4096, R0
	BGE  neon4096r4_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon4096r4_copy_loop

neon4096r4_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon4096r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 4096, complex64, radix-4 variant
TEXT ·InverseNEONSize4096Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $4096, R13
	BNE  neon4096r4_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $4096, R0
	BLT  neon4096r4_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $4096, R0
	BLT  neon4096r4_inv_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $4096, R0
	BLT  neon4096r4_inv_return_false

	// Load static bit-reversal table
	MOVD $bitrev_size4096_radix4<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon4096r4_inv_use_dst
	MOVD R11, R8

neon4096r4_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon4096r4_inv_bitrev_loop:
	CMP  $4096, R0
	BGE  neon4096r4_inv_stage1

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
	B    neon4096r4_inv_bitrev_loop

neon4096r4_inv_stage1:
	// Stage 1: 1024 radix-4 butterflies
	MOVD $0, R0

neon4096r4_inv_stage1_loop:
	CMP  $4096, R0
	BGE  neon4096r4_inv_stage2

	LSL  $3, R0, R1
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
	FNEGS  F14, F21
	FADDS F20, F10, F22
	FADDS F21, F11, F23

	FNEGS  F15, F24
	FMOVS F14, F25
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

	ADD  $4, R0, R0
	B    neon4096r4_inv_stage1_loop

neon4096r4_inv_stage2:
	// Stage 2: 256 groups x 4 butterflies, twiddle step=256
	MOVD $0, R0

neon4096r4_inv_stage2_outer:
	CMP  $4096, R0
	BGE  neon4096r4_inv_stage3

	MOVD $0, R1

neon4096r4_inv_stage2_inner:
	CMP  $4, R1
	BGE  neon4096r4_inv_stage2_next

	ADD  R0, R1, R2
	ADD  $4, R2, R3
	ADD  $8, R2, R4
	ADD  $12, R2, R5

	// twiddle indices: j*256, j*512, j*768 (conjugated)
	LSL  $8, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1
	FNEGS  F1, F1

	LSL  $9, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3
	FNEGS  F3, F3

	LSL  $8, R1, R6
	LSL  $9, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5
	FNEGS  F5, F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FMOVS F21, F26
	FNEGS  F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FNEGS  F21, F30
	FMOVS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon4096r4_inv_stage2_inner

neon4096r4_inv_stage2_next:
	ADD  $16, R0, R0
	B    neon4096r4_inv_stage2_outer

neon4096r4_inv_stage3:
	// Stage 3: 64 groups x 16 butterflies, twiddle step=64
	MOVD $0, R0

neon4096r4_inv_stage3_outer:
	CMP  $4096, R0
	BGE  neon4096r4_inv_stage4

	MOVD $0, R1

neon4096r4_inv_stage3_inner:
	CMP  $16, R1
	BGE  neon4096r4_inv_stage3_next

	ADD  R0, R1, R2
	ADD  $16, R2, R3
	ADD  $32, R2, R4
	ADD  $48, R2, R5

	// twiddle indices: j*64, j*128, j*192 (conjugated)
	LSL  $6, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1
	FNEGS  F1, F1

	LSL  $7, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3
	FNEGS  F3, F3

	LSL  $6, R1, R6
	LSL  $7, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5
	FNEGS  F5, F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FMOVS F21, F26
	FNEGS  F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FNEGS  F21, F30
	FMOVS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon4096r4_inv_stage3_inner

neon4096r4_inv_stage3_next:
	ADD  $64, R0, R0
	B    neon4096r4_inv_stage3_outer

neon4096r4_inv_stage4:
	// Stage 4: 16 groups x 64 butterflies, twiddle step=16
	MOVD $0, R0

neon4096r4_inv_stage4_outer:
	CMP  $4096, R0
	BGE  neon4096r4_inv_stage5

	MOVD $0, R1

neon4096r4_inv_stage4_inner:
	CMP  $64, R1
	BGE  neon4096r4_inv_stage4_next

	ADD  R0, R1, R2
	ADD  $64, R2, R3
	ADD  $128, R2, R4
	ADD  $192, R2, R5

	// twiddle indices: j*16, j*32, j*48 (conjugated)
	LSL  $4, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1
	FNEGS  F1, F1

	LSL  $5, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3
	FNEGS  F3, F3

	LSL  $4, R1, R6
	LSL  $5, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5
	FNEGS  F5, F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FMOVS F21, F26
	FNEGS  F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FNEGS  F21, F30
	FMOVS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon4096r4_inv_stage4_inner

neon4096r4_inv_stage4_next:
	ADD  $256, R0, R0
	B    neon4096r4_inv_stage4_outer

neon4096r4_inv_stage5:
	// Stage 5: 4 groups x 256 butterflies, twiddle step=4
	MOVD $0, R0

neon4096r4_inv_stage5_outer:
	CMP  $4096, R0
	BGE  neon4096r4_inv_stage6

	MOVD $0, R1

neon4096r4_inv_stage5_inner:
	CMP  $256, R1
	BGE  neon4096r4_inv_stage5_next

	ADD  R0, R1, R2
	ADD  $256, R2, R3
	ADD  $512, R2, R4
	ADD  $768, R2, R5

	// twiddle indices: j*4, j*8, j*12 (conjugated)
	LSL  $2, R1, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1
	FNEGS  F1, F1

	LSL  $3, R1, R7
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3
	FNEGS  F3, F3

	LSL  $2, R1, R6
	LSL  $3, R1, R7
	ADD  R6, R7, R6
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5
	FNEGS  F5, F5

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F6
	FMOVS 4(R6), F7

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F8
	FMOVS 4(R6), F9

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F10
	FMOVS 4(R6), F11

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS 0(R6), F12
	FMOVS 4(R6), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FMOVS F21, F26
	FNEGS  F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FNEGS  F21, F30
	FMOVS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	ADD  $1, R1, R1
	B    neon4096r4_inv_stage5_inner

neon4096r4_inv_stage5_next:
	ADD  $1024, R0, R0
	B    neon4096r4_inv_stage5_outer

neon4096r4_inv_stage6:
	// Stage 6: 1 group x 1024 butterflies, twiddle step=1
	MOVD $0, R0

neon4096r4_inv_stage6_loop:
	CMP  $1024, R0
	BGE  neon4096r4_inv_done

	MOVD R0, R1
	ADD  $1024, R1, R2
	ADD  $2048, R1, R3
	ADD  $3072, R1, R4

	// twiddle indices: j, 2j, 3j (conjugated)
	LSL  $3, R1, R5
	ADD  R10, R5, R5
	FMOVS 0(R5), F0
	FMOVS 4(R5), F1
	FNEGS  F1, F1

	LSL  $1, R1, R6
	ADD  R6, R1, R7
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F2
	FMOVS 4(R6), F3
	FNEGS  F3, F3

	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F4
	FMOVS 4(R7), F5
	FNEGS  F5, F5

	LSL  $3, R1, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F6
	FMOVS 4(R5), F7

	LSL  $3, R2, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F8
	FMOVS 4(R5), F9

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F10
	FMOVS 4(R5), F11

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS 0(R5), F12
	FMOVS 4(R5), F13

	// a1 = w1 * a1
	FMULS F0, F8, F14
	FMULS F1, F9, F15
	FSUBS F15, F14, F14
	FMULS F0, F9, F15
	FMULS F1, F8, F16
	FADDS F16, F15, F15
	FMOVS F14, F8
	FMOVS F15, F9

	// a2 = w2 * a2
	FMULS F2, F10, F14
	FMULS F3, F11, F15
	FSUBS F15, F14, F14
	FMULS F2, F11, F15
	FMULS F3, F10, F16
	FADDS F16, F15, F15
	FMOVS F14, F10
	FMOVS F15, F11

	// a3 = w3 * a3
	FMULS F4, F12, F14
	FMULS F5, F13, F15
	FSUBS F15, F14, F14
	FMULS F4, F13, F15
	FMULS F5, F12, F16
	FADDS F16, F15, F15
	FMOVS F14, F12
	FMOVS F15, F13

	FADDS F10, F6, F14
	FADDS F11, F7, F15
	FSUBS F10, F6, F16
	FSUBS F11, F7, F17

	FADDS F12, F8, F18
	FADDS F13, F9, F19
	FSUBS F12, F8, F20
	FSUBS F13, F9, F21

	FADDS F18, F14, F22
	FADDS F19, F15, F23
	FSUBS F18, F14, F24
	FSUBS F19, F15, F25

	FMOVS F21, F26
	FNEGS  F20, F27
	FADDS F26, F16, F28
	FADDS F27, F17, F29

	FNEGS  F21, F30
	FMOVS F20, F31
	FADDS F30, F16, F20
	FADDS F31, F17, F21

	LSL  $3, R1, R5
	ADD  R8, R5, R5
	FMOVS F22, 0(R5)
	FMOVS F23, 4(R5)

	LSL  $3, R2, R5
	ADD  R8, R5, R5
	FMOVS F20, 0(R5)
	FMOVS F21, 4(R5)

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS F24, 0(R5)
	FMOVS F25, 4(R5)

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS F28, 0(R5)
	FMOVS F29, 4(R5)

	ADD  $1, R0, R0
	B    neon4096r4_inv_stage6_loop

neon4096r4_inv_done:
	CMP  R8, R20
	BEQ  neon4096r4_inv_scale

	MOVD $0, R0
neon4096r4_inv_copy_loop:
	CMP  $4096, R0
	BGE  neon4096r4_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon4096r4_inv_copy_loop

neon4096r4_inv_scale:
	MOVD $·neonInv4096Radix4(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon4096r4_inv_scale_loop:
	CMP  $4096, R0
	BGE  neon4096r4_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon4096r4_inv_scale_loop

neon4096r4_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon4096r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Size-4096 Radix-4 bit-reversal table (base-4 digit reversal)
DATA bitrev_size4096_radix4<>+0x000(SB)/8, $0    // bitrev[0] = 0
DATA bitrev_size4096_radix4<>+0x008(SB)/8, $1024 // bitrev[1] = 1024
DATA bitrev_size4096_radix4<>+0x010(SB)/8, $2048 // bitrev[2] = 2048
DATA bitrev_size4096_radix4<>+0x018(SB)/8, $3072 // bitrev[3] = 3072
DATA bitrev_size4096_radix4<>+0x020(SB)/8, $256  // bitrev[4] = 256
DATA bitrev_size4096_radix4<>+0x028(SB)/8, $1280 // bitrev[5] = 1280
DATA bitrev_size4096_radix4<>+0x030(SB)/8, $2304 // bitrev[6] = 2304
DATA bitrev_size4096_radix4<>+0x038(SB)/8, $3328 // bitrev[7] = 3328
DATA bitrev_size4096_radix4<>+0x040(SB)/8, $512  // bitrev[8] = 512
DATA bitrev_size4096_radix4<>+0x048(SB)/8, $1536 // bitrev[9] = 1536
DATA bitrev_size4096_radix4<>+0x050(SB)/8, $2560 // bitrev[10] = 2560
DATA bitrev_size4096_radix4<>+0x058(SB)/8, $3584 // bitrev[11] = 3584
DATA bitrev_size4096_radix4<>+0x060(SB)/8, $768  // bitrev[12] = 768
DATA bitrev_size4096_radix4<>+0x068(SB)/8, $1792 // bitrev[13] = 1792
DATA bitrev_size4096_radix4<>+0x070(SB)/8, $2816 // bitrev[14] = 2816
DATA bitrev_size4096_radix4<>+0x078(SB)/8, $3840 // bitrev[15] = 3840
DATA bitrev_size4096_radix4<>+0x080(SB)/8, $64   // bitrev[16] = 64
DATA bitrev_size4096_radix4<>+0x088(SB)/8, $1088 // bitrev[17] = 1088
DATA bitrev_size4096_radix4<>+0x090(SB)/8, $2112 // bitrev[18] = 2112
DATA bitrev_size4096_radix4<>+0x098(SB)/8, $3136 // bitrev[19] = 3136
DATA bitrev_size4096_radix4<>+0x0a0(SB)/8, $320  // bitrev[20] = 320
DATA bitrev_size4096_radix4<>+0x0a8(SB)/8, $1344 // bitrev[21] = 1344
DATA bitrev_size4096_radix4<>+0x0b0(SB)/8, $2368 // bitrev[22] = 2368
DATA bitrev_size4096_radix4<>+0x0b8(SB)/8, $3392 // bitrev[23] = 3392
DATA bitrev_size4096_radix4<>+0x0c0(SB)/8, $576  // bitrev[24] = 576
DATA bitrev_size4096_radix4<>+0x0c8(SB)/8, $1600 // bitrev[25] = 1600
DATA bitrev_size4096_radix4<>+0x0d0(SB)/8, $2624 // bitrev[26] = 2624
DATA bitrev_size4096_radix4<>+0x0d8(SB)/8, $3648 // bitrev[27] = 3648
DATA bitrev_size4096_radix4<>+0x0e0(SB)/8, $832  // bitrev[28] = 832
DATA bitrev_size4096_radix4<>+0x0e8(SB)/8, $1856 // bitrev[29] = 1856
DATA bitrev_size4096_radix4<>+0x0f0(SB)/8, $2880 // bitrev[30] = 2880
DATA bitrev_size4096_radix4<>+0x0f8(SB)/8, $3904 // bitrev[31] = 3904
DATA bitrev_size4096_radix4<>+0x100(SB)/8, $128  // bitrev[32] = 128
DATA bitrev_size4096_radix4<>+0x108(SB)/8, $1152 // bitrev[33] = 1152
DATA bitrev_size4096_radix4<>+0x110(SB)/8, $2176 // bitrev[34] = 2176
DATA bitrev_size4096_radix4<>+0x118(SB)/8, $3200 // bitrev[35] = 3200
DATA bitrev_size4096_radix4<>+0x120(SB)/8, $384  // bitrev[36] = 384
DATA bitrev_size4096_radix4<>+0x128(SB)/8, $1408 // bitrev[37] = 1408
DATA bitrev_size4096_radix4<>+0x130(SB)/8, $2432 // bitrev[38] = 2432
DATA bitrev_size4096_radix4<>+0x138(SB)/8, $3456 // bitrev[39] = 3456
DATA bitrev_size4096_radix4<>+0x140(SB)/8, $640  // bitrev[40] = 640
DATA bitrev_size4096_radix4<>+0x148(SB)/8, $1664 // bitrev[41] = 1664
DATA bitrev_size4096_radix4<>+0x150(SB)/8, $2688 // bitrev[42] = 2688
DATA bitrev_size4096_radix4<>+0x158(SB)/8, $3712 // bitrev[43] = 3712
DATA bitrev_size4096_radix4<>+0x160(SB)/8, $896  // bitrev[44] = 896
DATA bitrev_size4096_radix4<>+0x168(SB)/8, $1920 // bitrev[45] = 1920
DATA bitrev_size4096_radix4<>+0x170(SB)/8, $2944 // bitrev[46] = 2944
DATA bitrev_size4096_radix4<>+0x178(SB)/8, $3968 // bitrev[47] = 3968
DATA bitrev_size4096_radix4<>+0x180(SB)/8, $192  // bitrev[48] = 192
DATA bitrev_size4096_radix4<>+0x188(SB)/8, $1216 // bitrev[49] = 1216
DATA bitrev_size4096_radix4<>+0x190(SB)/8, $2240 // bitrev[50] = 2240
DATA bitrev_size4096_radix4<>+0x198(SB)/8, $3264 // bitrev[51] = 3264
DATA bitrev_size4096_radix4<>+0x1a0(SB)/8, $448  // bitrev[52] = 448
DATA bitrev_size4096_radix4<>+0x1a8(SB)/8, $1472 // bitrev[53] = 1472
DATA bitrev_size4096_radix4<>+0x1b0(SB)/8, $2496 // bitrev[54] = 2496
DATA bitrev_size4096_radix4<>+0x1b8(SB)/8, $3520 // bitrev[55] = 3520
DATA bitrev_size4096_radix4<>+0x1c0(SB)/8, $704  // bitrev[56] = 704
DATA bitrev_size4096_radix4<>+0x1c8(SB)/8, $1728 // bitrev[57] = 1728
DATA bitrev_size4096_radix4<>+0x1d0(SB)/8, $2752 // bitrev[58] = 2752
DATA bitrev_size4096_radix4<>+0x1d8(SB)/8, $3776 // bitrev[59] = 3776
DATA bitrev_size4096_radix4<>+0x1e0(SB)/8, $960  // bitrev[60] = 960
DATA bitrev_size4096_radix4<>+0x1e8(SB)/8, $1984 // bitrev[61] = 1984
DATA bitrev_size4096_radix4<>+0x1f0(SB)/8, $3008 // bitrev[62] = 3008
DATA bitrev_size4096_radix4<>+0x1f8(SB)/8, $4032 // bitrev[63] = 4032
DATA bitrev_size4096_radix4<>+0x200(SB)/8, $16   // bitrev[64] = 16
DATA bitrev_size4096_radix4<>+0x208(SB)/8, $1040 // bitrev[65] = 1040
DATA bitrev_size4096_radix4<>+0x210(SB)/8, $2064 // bitrev[66] = 2064
DATA bitrev_size4096_radix4<>+0x218(SB)/8, $3088 // bitrev[67] = 3088
DATA bitrev_size4096_radix4<>+0x220(SB)/8, $272  // bitrev[68] = 272
DATA bitrev_size4096_radix4<>+0x228(SB)/8, $1296 // bitrev[69] = 1296
DATA bitrev_size4096_radix4<>+0x230(SB)/8, $2320 // bitrev[70] = 2320
DATA bitrev_size4096_radix4<>+0x238(SB)/8, $3344 // bitrev[71] = 3344
DATA bitrev_size4096_radix4<>+0x240(SB)/8, $528  // bitrev[72] = 528
DATA bitrev_size4096_radix4<>+0x248(SB)/8, $1552 // bitrev[73] = 1552
DATA bitrev_size4096_radix4<>+0x250(SB)/8, $2576 // bitrev[74] = 2576
DATA bitrev_size4096_radix4<>+0x258(SB)/8, $3600 // bitrev[75] = 3600
DATA bitrev_size4096_radix4<>+0x260(SB)/8, $784  // bitrev[76] = 784
DATA bitrev_size4096_radix4<>+0x268(SB)/8, $1808 // bitrev[77] = 1808
DATA bitrev_size4096_radix4<>+0x270(SB)/8, $2832 // bitrev[78] = 2832
DATA bitrev_size4096_radix4<>+0x278(SB)/8, $3856 // bitrev[79] = 3856
DATA bitrev_size4096_radix4<>+0x280(SB)/8, $80   // bitrev[80] = 80
DATA bitrev_size4096_radix4<>+0x288(SB)/8, $1104 // bitrev[81] = 1104
DATA bitrev_size4096_radix4<>+0x290(SB)/8, $2128 // bitrev[82] = 2128
DATA bitrev_size4096_radix4<>+0x298(SB)/8, $3152 // bitrev[83] = 3152
DATA bitrev_size4096_radix4<>+0x2a0(SB)/8, $336  // bitrev[84] = 336
DATA bitrev_size4096_radix4<>+0x2a8(SB)/8, $1360 // bitrev[85] = 1360
DATA bitrev_size4096_radix4<>+0x2b0(SB)/8, $2384 // bitrev[86] = 2384
DATA bitrev_size4096_radix4<>+0x2b8(SB)/8, $3408 // bitrev[87] = 3408
DATA bitrev_size4096_radix4<>+0x2c0(SB)/8, $592  // bitrev[88] = 592
DATA bitrev_size4096_radix4<>+0x2c8(SB)/8, $1616 // bitrev[89] = 1616
DATA bitrev_size4096_radix4<>+0x2d0(SB)/8, $2640 // bitrev[90] = 2640
DATA bitrev_size4096_radix4<>+0x2d8(SB)/8, $3664 // bitrev[91] = 3664
DATA bitrev_size4096_radix4<>+0x2e0(SB)/8, $848  // bitrev[92] = 848
DATA bitrev_size4096_radix4<>+0x2e8(SB)/8, $1872 // bitrev[93] = 1872
DATA bitrev_size4096_radix4<>+0x2f0(SB)/8, $2896 // bitrev[94] = 2896
DATA bitrev_size4096_radix4<>+0x2f8(SB)/8, $3920 // bitrev[95] = 3920
DATA bitrev_size4096_radix4<>+0x300(SB)/8, $144  // bitrev[96] = 144
DATA bitrev_size4096_radix4<>+0x308(SB)/8, $1168 // bitrev[97] = 1168
DATA bitrev_size4096_radix4<>+0x310(SB)/8, $2192 // bitrev[98] = 2192
DATA bitrev_size4096_radix4<>+0x318(SB)/8, $3216 // bitrev[99] = 3216
DATA bitrev_size4096_radix4<>+0x320(SB)/8, $400  // bitrev[100] = 400
DATA bitrev_size4096_radix4<>+0x328(SB)/8, $1424 // bitrev[101] = 1424
DATA bitrev_size4096_radix4<>+0x330(SB)/8, $2448 // bitrev[102] = 2448
DATA bitrev_size4096_radix4<>+0x338(SB)/8, $3472 // bitrev[103] = 3472
DATA bitrev_size4096_radix4<>+0x340(SB)/8, $656  // bitrev[104] = 656
DATA bitrev_size4096_radix4<>+0x348(SB)/8, $1680 // bitrev[105] = 1680
DATA bitrev_size4096_radix4<>+0x350(SB)/8, $2704 // bitrev[106] = 2704
DATA bitrev_size4096_radix4<>+0x358(SB)/8, $3728 // bitrev[107] = 3728
DATA bitrev_size4096_radix4<>+0x360(SB)/8, $912  // bitrev[108] = 912
DATA bitrev_size4096_radix4<>+0x368(SB)/8, $1936 // bitrev[109] = 1936
DATA bitrev_size4096_radix4<>+0x370(SB)/8, $2960 // bitrev[110] = 2960
DATA bitrev_size4096_radix4<>+0x378(SB)/8, $3984 // bitrev[111] = 3984
DATA bitrev_size4096_radix4<>+0x380(SB)/8, $208  // bitrev[112] = 208
DATA bitrev_size4096_radix4<>+0x388(SB)/8, $1232 // bitrev[113] = 1232
DATA bitrev_size4096_radix4<>+0x390(SB)/8, $2256 // bitrev[114] = 2256
DATA bitrev_size4096_radix4<>+0x398(SB)/8, $3280 // bitrev[115] = 3280
DATA bitrev_size4096_radix4<>+0x3a0(SB)/8, $464  // bitrev[116] = 464
DATA bitrev_size4096_radix4<>+0x3a8(SB)/8, $1488 // bitrev[117] = 1488
DATA bitrev_size4096_radix4<>+0x3b0(SB)/8, $2512 // bitrev[118] = 2512
DATA bitrev_size4096_radix4<>+0x3b8(SB)/8, $3536 // bitrev[119] = 3536
DATA bitrev_size4096_radix4<>+0x3c0(SB)/8, $720  // bitrev[120] = 720
DATA bitrev_size4096_radix4<>+0x3c8(SB)/8, $1744 // bitrev[121] = 1744
DATA bitrev_size4096_radix4<>+0x3d0(SB)/8, $2768 // bitrev[122] = 2768
DATA bitrev_size4096_radix4<>+0x3d8(SB)/8, $3792 // bitrev[123] = 3792
DATA bitrev_size4096_radix4<>+0x3e0(SB)/8, $976  // bitrev[124] = 976
DATA bitrev_size4096_radix4<>+0x3e8(SB)/8, $2000 // bitrev[125] = 2000
DATA bitrev_size4096_radix4<>+0x3f0(SB)/8, $3024 // bitrev[126] = 3024
DATA bitrev_size4096_radix4<>+0x3f8(SB)/8, $4048 // bitrev[127] = 4048
DATA bitrev_size4096_radix4<>+0x400(SB)/8, $32   // bitrev[128] = 32
DATA bitrev_size4096_radix4<>+0x408(SB)/8, $1056 // bitrev[129] = 1056
DATA bitrev_size4096_radix4<>+0x410(SB)/8, $2080 // bitrev[130] = 2080
DATA bitrev_size4096_radix4<>+0x418(SB)/8, $3104 // bitrev[131] = 3104
DATA bitrev_size4096_radix4<>+0x420(SB)/8, $288  // bitrev[132] = 288
DATA bitrev_size4096_radix4<>+0x428(SB)/8, $1312 // bitrev[133] = 1312
DATA bitrev_size4096_radix4<>+0x430(SB)/8, $2336 // bitrev[134] = 2336
DATA bitrev_size4096_radix4<>+0x438(SB)/8, $3360 // bitrev[135] = 3360
DATA bitrev_size4096_radix4<>+0x440(SB)/8, $544  // bitrev[136] = 544
DATA bitrev_size4096_radix4<>+0x448(SB)/8, $1568 // bitrev[137] = 1568
DATA bitrev_size4096_radix4<>+0x450(SB)/8, $2592 // bitrev[138] = 2592
DATA bitrev_size4096_radix4<>+0x458(SB)/8, $3616 // bitrev[139] = 3616
DATA bitrev_size4096_radix4<>+0x460(SB)/8, $800  // bitrev[140] = 800
DATA bitrev_size4096_radix4<>+0x468(SB)/8, $1824 // bitrev[141] = 1824
DATA bitrev_size4096_radix4<>+0x470(SB)/8, $2848 // bitrev[142] = 2848
DATA bitrev_size4096_radix4<>+0x478(SB)/8, $3872 // bitrev[143] = 3872
DATA bitrev_size4096_radix4<>+0x480(SB)/8, $96   // bitrev[144] = 96
DATA bitrev_size4096_radix4<>+0x488(SB)/8, $1120 // bitrev[145] = 1120
DATA bitrev_size4096_radix4<>+0x490(SB)/8, $2144 // bitrev[146] = 2144
DATA bitrev_size4096_radix4<>+0x498(SB)/8, $3168 // bitrev[147] = 3168
DATA bitrev_size4096_radix4<>+0x4a0(SB)/8, $352  // bitrev[148] = 352
DATA bitrev_size4096_radix4<>+0x4a8(SB)/8, $1376 // bitrev[149] = 1376
DATA bitrev_size4096_radix4<>+0x4b0(SB)/8, $2400 // bitrev[150] = 2400
DATA bitrev_size4096_radix4<>+0x4b8(SB)/8, $3424 // bitrev[151] = 3424
DATA bitrev_size4096_radix4<>+0x4c0(SB)/8, $608  // bitrev[152] = 608
DATA bitrev_size4096_radix4<>+0x4c8(SB)/8, $1632 // bitrev[153] = 1632
DATA bitrev_size4096_radix4<>+0x4d0(SB)/8, $2656 // bitrev[154] = 2656
DATA bitrev_size4096_radix4<>+0x4d8(SB)/8, $3680 // bitrev[155] = 3680
DATA bitrev_size4096_radix4<>+0x4e0(SB)/8, $864  // bitrev[156] = 864
DATA bitrev_size4096_radix4<>+0x4e8(SB)/8, $1888 // bitrev[157] = 1888
DATA bitrev_size4096_radix4<>+0x4f0(SB)/8, $2912 // bitrev[158] = 2912
DATA bitrev_size4096_radix4<>+0x4f8(SB)/8, $3936 // bitrev[159] = 3936
DATA bitrev_size4096_radix4<>+0x500(SB)/8, $160  // bitrev[160] = 160
DATA bitrev_size4096_radix4<>+0x508(SB)/8, $1184 // bitrev[161] = 1184
DATA bitrev_size4096_radix4<>+0x510(SB)/8, $2208 // bitrev[162] = 2208
DATA bitrev_size4096_radix4<>+0x518(SB)/8, $3232 // bitrev[163] = 3232
DATA bitrev_size4096_radix4<>+0x520(SB)/8, $416  // bitrev[164] = 416
DATA bitrev_size4096_radix4<>+0x528(SB)/8, $1440 // bitrev[165] = 1440
DATA bitrev_size4096_radix4<>+0x530(SB)/8, $2464 // bitrev[166] = 2464
DATA bitrev_size4096_radix4<>+0x538(SB)/8, $3488 // bitrev[167] = 3488
DATA bitrev_size4096_radix4<>+0x540(SB)/8, $672  // bitrev[168] = 672
DATA bitrev_size4096_radix4<>+0x548(SB)/8, $1696 // bitrev[169] = 1696
DATA bitrev_size4096_radix4<>+0x550(SB)/8, $2720 // bitrev[170] = 2720
DATA bitrev_size4096_radix4<>+0x558(SB)/8, $3744 // bitrev[171] = 3744
DATA bitrev_size4096_radix4<>+0x560(SB)/8, $928  // bitrev[172] = 928
DATA bitrev_size4096_radix4<>+0x568(SB)/8, $1952 // bitrev[173] = 1952
DATA bitrev_size4096_radix4<>+0x570(SB)/8, $2976 // bitrev[174] = 2976
DATA bitrev_size4096_radix4<>+0x578(SB)/8, $4000 // bitrev[175] = 4000
DATA bitrev_size4096_radix4<>+0x580(SB)/8, $224  // bitrev[176] = 224
DATA bitrev_size4096_radix4<>+0x588(SB)/8, $1248 // bitrev[177] = 1248
DATA bitrev_size4096_radix4<>+0x590(SB)/8, $2272 // bitrev[178] = 2272
DATA bitrev_size4096_radix4<>+0x598(SB)/8, $3296 // bitrev[179] = 3296
DATA bitrev_size4096_radix4<>+0x5a0(SB)/8, $480  // bitrev[180] = 480
DATA bitrev_size4096_radix4<>+0x5a8(SB)/8, $1504 // bitrev[181] = 1504
DATA bitrev_size4096_radix4<>+0x5b0(SB)/8, $2528 // bitrev[182] = 2528
DATA bitrev_size4096_radix4<>+0x5b8(SB)/8, $3552 // bitrev[183] = 3552
DATA bitrev_size4096_radix4<>+0x5c0(SB)/8, $736  // bitrev[184] = 736
DATA bitrev_size4096_radix4<>+0x5c8(SB)/8, $1760 // bitrev[185] = 1760
DATA bitrev_size4096_radix4<>+0x5d0(SB)/8, $2784 // bitrev[186] = 2784
DATA bitrev_size4096_radix4<>+0x5d8(SB)/8, $3808 // bitrev[187] = 3808
DATA bitrev_size4096_radix4<>+0x5e0(SB)/8, $992  // bitrev[188] = 992
DATA bitrev_size4096_radix4<>+0x5e8(SB)/8, $2016 // bitrev[189] = 2016
DATA bitrev_size4096_radix4<>+0x5f0(SB)/8, $3040 // bitrev[190] = 3040
DATA bitrev_size4096_radix4<>+0x5f8(SB)/8, $4064 // bitrev[191] = 4064
DATA bitrev_size4096_radix4<>+0x600(SB)/8, $48   // bitrev[192] = 48
DATA bitrev_size4096_radix4<>+0x608(SB)/8, $1072 // bitrev[193] = 1072
DATA bitrev_size4096_radix4<>+0x610(SB)/8, $2096 // bitrev[194] = 2096
DATA bitrev_size4096_radix4<>+0x618(SB)/8, $3120 // bitrev[195] = 3120
DATA bitrev_size4096_radix4<>+0x620(SB)/8, $304  // bitrev[196] = 304
DATA bitrev_size4096_radix4<>+0x628(SB)/8, $1328 // bitrev[197] = 1328
DATA bitrev_size4096_radix4<>+0x630(SB)/8, $2352 // bitrev[198] = 2352
DATA bitrev_size4096_radix4<>+0x638(SB)/8, $3376 // bitrev[199] = 3376
DATA bitrev_size4096_radix4<>+0x640(SB)/8, $560  // bitrev[200] = 560
DATA bitrev_size4096_radix4<>+0x648(SB)/8, $1584 // bitrev[201] = 1584
DATA bitrev_size4096_radix4<>+0x650(SB)/8, $2608 // bitrev[202] = 2608
DATA bitrev_size4096_radix4<>+0x658(SB)/8, $3632 // bitrev[203] = 3632
DATA bitrev_size4096_radix4<>+0x660(SB)/8, $816  // bitrev[204] = 816
DATA bitrev_size4096_radix4<>+0x668(SB)/8, $1840 // bitrev[205] = 1840
DATA bitrev_size4096_radix4<>+0x670(SB)/8, $2864 // bitrev[206] = 2864
DATA bitrev_size4096_radix4<>+0x678(SB)/8, $3888 // bitrev[207] = 3888
DATA bitrev_size4096_radix4<>+0x680(SB)/8, $112  // bitrev[208] = 112
DATA bitrev_size4096_radix4<>+0x688(SB)/8, $1136 // bitrev[209] = 1136
DATA bitrev_size4096_radix4<>+0x690(SB)/8, $2160 // bitrev[210] = 2160
DATA bitrev_size4096_radix4<>+0x698(SB)/8, $3184 // bitrev[211] = 3184
DATA bitrev_size4096_radix4<>+0x6a0(SB)/8, $368  // bitrev[212] = 368
DATA bitrev_size4096_radix4<>+0x6a8(SB)/8, $1392 // bitrev[213] = 1392
DATA bitrev_size4096_radix4<>+0x6b0(SB)/8, $2416 // bitrev[214] = 2416
DATA bitrev_size4096_radix4<>+0x6b8(SB)/8, $3440 // bitrev[215] = 3440
DATA bitrev_size4096_radix4<>+0x6c0(SB)/8, $624  // bitrev[216] = 624
DATA bitrev_size4096_radix4<>+0x6c8(SB)/8, $1648 // bitrev[217] = 1648
DATA bitrev_size4096_radix4<>+0x6d0(SB)/8, $2672 // bitrev[218] = 2672
DATA bitrev_size4096_radix4<>+0x6d8(SB)/8, $3696 // bitrev[219] = 3696
DATA bitrev_size4096_radix4<>+0x6e0(SB)/8, $880  // bitrev[220] = 880
DATA bitrev_size4096_radix4<>+0x6e8(SB)/8, $1904 // bitrev[221] = 1904
DATA bitrev_size4096_radix4<>+0x6f0(SB)/8, $2928 // bitrev[222] = 2928
DATA bitrev_size4096_radix4<>+0x6f8(SB)/8, $3952 // bitrev[223] = 3952
DATA bitrev_size4096_radix4<>+0x700(SB)/8, $176  // bitrev[224] = 176
DATA bitrev_size4096_radix4<>+0x708(SB)/8, $1200 // bitrev[225] = 1200
DATA bitrev_size4096_radix4<>+0x710(SB)/8, $2224 // bitrev[226] = 2224
DATA bitrev_size4096_radix4<>+0x718(SB)/8, $3248 // bitrev[227] = 3248
DATA bitrev_size4096_radix4<>+0x720(SB)/8, $432  // bitrev[228] = 432
DATA bitrev_size4096_radix4<>+0x728(SB)/8, $1456 // bitrev[229] = 1456
DATA bitrev_size4096_radix4<>+0x730(SB)/8, $2480 // bitrev[230] = 2480
DATA bitrev_size4096_radix4<>+0x738(SB)/8, $3504 // bitrev[231] = 3504
DATA bitrev_size4096_radix4<>+0x740(SB)/8, $688  // bitrev[232] = 688
DATA bitrev_size4096_radix4<>+0x748(SB)/8, $1712 // bitrev[233] = 1712
DATA bitrev_size4096_radix4<>+0x750(SB)/8, $2736 // bitrev[234] = 2736
DATA bitrev_size4096_radix4<>+0x758(SB)/8, $3760 // bitrev[235] = 3760
DATA bitrev_size4096_radix4<>+0x760(SB)/8, $944  // bitrev[236] = 944
DATA bitrev_size4096_radix4<>+0x768(SB)/8, $1968 // bitrev[237] = 1968
DATA bitrev_size4096_radix4<>+0x770(SB)/8, $2992 // bitrev[238] = 2992
DATA bitrev_size4096_radix4<>+0x778(SB)/8, $4016 // bitrev[239] = 4016
DATA bitrev_size4096_radix4<>+0x780(SB)/8, $240  // bitrev[240] = 240
DATA bitrev_size4096_radix4<>+0x788(SB)/8, $1264 // bitrev[241] = 1264
DATA bitrev_size4096_radix4<>+0x790(SB)/8, $2288 // bitrev[242] = 2288
DATA bitrev_size4096_radix4<>+0x798(SB)/8, $3312 // bitrev[243] = 3312
DATA bitrev_size4096_radix4<>+0x7a0(SB)/8, $496  // bitrev[244] = 496
DATA bitrev_size4096_radix4<>+0x7a8(SB)/8, $1520 // bitrev[245] = 1520
DATA bitrev_size4096_radix4<>+0x7b0(SB)/8, $2544 // bitrev[246] = 2544
DATA bitrev_size4096_radix4<>+0x7b8(SB)/8, $3568 // bitrev[247] = 3568
DATA bitrev_size4096_radix4<>+0x7c0(SB)/8, $752  // bitrev[248] = 752
DATA bitrev_size4096_radix4<>+0x7c8(SB)/8, $1776 // bitrev[249] = 1776
DATA bitrev_size4096_radix4<>+0x7d0(SB)/8, $2800 // bitrev[250] = 2800
DATA bitrev_size4096_radix4<>+0x7d8(SB)/8, $3824 // bitrev[251] = 3824
DATA bitrev_size4096_radix4<>+0x7e0(SB)/8, $1008 // bitrev[252] = 1008
DATA bitrev_size4096_radix4<>+0x7e8(SB)/8, $2032 // bitrev[253] = 2032
DATA bitrev_size4096_radix4<>+0x7f0(SB)/8, $3056 // bitrev[254] = 3056
DATA bitrev_size4096_radix4<>+0x7f8(SB)/8, $4080 // bitrev[255] = 4080
DATA bitrev_size4096_radix4<>+0x800(SB)/8, $4    // bitrev[256] = 4
DATA bitrev_size4096_radix4<>+0x808(SB)/8, $1028 // bitrev[257] = 1028
DATA bitrev_size4096_radix4<>+0x810(SB)/8, $2052 // bitrev[258] = 2052
DATA bitrev_size4096_radix4<>+0x818(SB)/8, $3076 // bitrev[259] = 3076
DATA bitrev_size4096_radix4<>+0x820(SB)/8, $260  // bitrev[260] = 260
DATA bitrev_size4096_radix4<>+0x828(SB)/8, $1284 // bitrev[261] = 1284
DATA bitrev_size4096_radix4<>+0x830(SB)/8, $2308 // bitrev[262] = 2308
DATA bitrev_size4096_radix4<>+0x838(SB)/8, $3332 // bitrev[263] = 3332
DATA bitrev_size4096_radix4<>+0x840(SB)/8, $516  // bitrev[264] = 516
DATA bitrev_size4096_radix4<>+0x848(SB)/8, $1540 // bitrev[265] = 1540
DATA bitrev_size4096_radix4<>+0x850(SB)/8, $2564 // bitrev[266] = 2564
DATA bitrev_size4096_radix4<>+0x858(SB)/8, $3588 // bitrev[267] = 3588
DATA bitrev_size4096_radix4<>+0x860(SB)/8, $772  // bitrev[268] = 772
DATA bitrev_size4096_radix4<>+0x868(SB)/8, $1796 // bitrev[269] = 1796
DATA bitrev_size4096_radix4<>+0x870(SB)/8, $2820 // bitrev[270] = 2820
DATA bitrev_size4096_radix4<>+0x878(SB)/8, $3844 // bitrev[271] = 3844
DATA bitrev_size4096_radix4<>+0x880(SB)/8, $68   // bitrev[272] = 68
DATA bitrev_size4096_radix4<>+0x888(SB)/8, $1092 // bitrev[273] = 1092
DATA bitrev_size4096_radix4<>+0x890(SB)/8, $2116 // bitrev[274] = 2116
DATA bitrev_size4096_radix4<>+0x898(SB)/8, $3140 // bitrev[275] = 3140
DATA bitrev_size4096_radix4<>+0x8a0(SB)/8, $324  // bitrev[276] = 324
DATA bitrev_size4096_radix4<>+0x8a8(SB)/8, $1348 // bitrev[277] = 1348
DATA bitrev_size4096_radix4<>+0x8b0(SB)/8, $2372 // bitrev[278] = 2372
DATA bitrev_size4096_radix4<>+0x8b8(SB)/8, $3396 // bitrev[279] = 3396
DATA bitrev_size4096_radix4<>+0x8c0(SB)/8, $580  // bitrev[280] = 580
DATA bitrev_size4096_radix4<>+0x8c8(SB)/8, $1604 // bitrev[281] = 1604
DATA bitrev_size4096_radix4<>+0x8d0(SB)/8, $2628 // bitrev[282] = 2628
DATA bitrev_size4096_radix4<>+0x8d8(SB)/8, $3652 // bitrev[283] = 3652
DATA bitrev_size4096_radix4<>+0x8e0(SB)/8, $836  // bitrev[284] = 836
DATA bitrev_size4096_radix4<>+0x8e8(SB)/8, $1860 // bitrev[285] = 1860
DATA bitrev_size4096_radix4<>+0x8f0(SB)/8, $2884 // bitrev[286] = 2884
DATA bitrev_size4096_radix4<>+0x8f8(SB)/8, $3908 // bitrev[287] = 3908
DATA bitrev_size4096_radix4<>+0x900(SB)/8, $132  // bitrev[288] = 132
DATA bitrev_size4096_radix4<>+0x908(SB)/8, $1156 // bitrev[289] = 1156
DATA bitrev_size4096_radix4<>+0x910(SB)/8, $2180 // bitrev[290] = 2180
DATA bitrev_size4096_radix4<>+0x918(SB)/8, $3204 // bitrev[291] = 3204
DATA bitrev_size4096_radix4<>+0x920(SB)/8, $388  // bitrev[292] = 388
DATA bitrev_size4096_radix4<>+0x928(SB)/8, $1412 // bitrev[293] = 1412
DATA bitrev_size4096_radix4<>+0x930(SB)/8, $2436 // bitrev[294] = 2436
DATA bitrev_size4096_radix4<>+0x938(SB)/8, $3460 // bitrev[295] = 3460
DATA bitrev_size4096_radix4<>+0x940(SB)/8, $644  // bitrev[296] = 644
DATA bitrev_size4096_radix4<>+0x948(SB)/8, $1668 // bitrev[297] = 1668
DATA bitrev_size4096_radix4<>+0x950(SB)/8, $2692 // bitrev[298] = 2692
DATA bitrev_size4096_radix4<>+0x958(SB)/8, $3716 // bitrev[299] = 3716
DATA bitrev_size4096_radix4<>+0x960(SB)/8, $900  // bitrev[300] = 900
DATA bitrev_size4096_radix4<>+0x968(SB)/8, $1924 // bitrev[301] = 1924
DATA bitrev_size4096_radix4<>+0x970(SB)/8, $2948 // bitrev[302] = 2948
DATA bitrev_size4096_radix4<>+0x978(SB)/8, $3972 // bitrev[303] = 3972
DATA bitrev_size4096_radix4<>+0x980(SB)/8, $196  // bitrev[304] = 196
DATA bitrev_size4096_radix4<>+0x988(SB)/8, $1220 // bitrev[305] = 1220
DATA bitrev_size4096_radix4<>+0x990(SB)/8, $2244 // bitrev[306] = 2244
DATA bitrev_size4096_radix4<>+0x998(SB)/8, $3268 // bitrev[307] = 3268
DATA bitrev_size4096_radix4<>+0x9a0(SB)/8, $452  // bitrev[308] = 452
DATA bitrev_size4096_radix4<>+0x9a8(SB)/8, $1476 // bitrev[309] = 1476
DATA bitrev_size4096_radix4<>+0x9b0(SB)/8, $2500 // bitrev[310] = 2500
DATA bitrev_size4096_radix4<>+0x9b8(SB)/8, $3524 // bitrev[311] = 3524
DATA bitrev_size4096_radix4<>+0x9c0(SB)/8, $708  // bitrev[312] = 708
DATA bitrev_size4096_radix4<>+0x9c8(SB)/8, $1732 // bitrev[313] = 1732
DATA bitrev_size4096_radix4<>+0x9d0(SB)/8, $2756 // bitrev[314] = 2756
DATA bitrev_size4096_radix4<>+0x9d8(SB)/8, $3780 // bitrev[315] = 3780
DATA bitrev_size4096_radix4<>+0x9e0(SB)/8, $964  // bitrev[316] = 964
DATA bitrev_size4096_radix4<>+0x9e8(SB)/8, $1988 // bitrev[317] = 1988
DATA bitrev_size4096_radix4<>+0x9f0(SB)/8, $3012 // bitrev[318] = 3012
DATA bitrev_size4096_radix4<>+0x9f8(SB)/8, $4036 // bitrev[319] = 4036
DATA bitrev_size4096_radix4<>+0xa00(SB)/8, $20   // bitrev[320] = 20
DATA bitrev_size4096_radix4<>+0xa08(SB)/8, $1044 // bitrev[321] = 1044
DATA bitrev_size4096_radix4<>+0xa10(SB)/8, $2068 // bitrev[322] = 2068
DATA bitrev_size4096_radix4<>+0xa18(SB)/8, $3092 // bitrev[323] = 3092
DATA bitrev_size4096_radix4<>+0xa20(SB)/8, $276  // bitrev[324] = 276
DATA bitrev_size4096_radix4<>+0xa28(SB)/8, $1300 // bitrev[325] = 1300
DATA bitrev_size4096_radix4<>+0xa30(SB)/8, $2324 // bitrev[326] = 2324
DATA bitrev_size4096_radix4<>+0xa38(SB)/8, $3348 // bitrev[327] = 3348
DATA bitrev_size4096_radix4<>+0xa40(SB)/8, $532  // bitrev[328] = 532
DATA bitrev_size4096_radix4<>+0xa48(SB)/8, $1556 // bitrev[329] = 1556
DATA bitrev_size4096_radix4<>+0xa50(SB)/8, $2580 // bitrev[330] = 2580
DATA bitrev_size4096_radix4<>+0xa58(SB)/8, $3604 // bitrev[331] = 3604
DATA bitrev_size4096_radix4<>+0xa60(SB)/8, $788  // bitrev[332] = 788
DATA bitrev_size4096_radix4<>+0xa68(SB)/8, $1812 // bitrev[333] = 1812
DATA bitrev_size4096_radix4<>+0xa70(SB)/8, $2836 // bitrev[334] = 2836
DATA bitrev_size4096_radix4<>+0xa78(SB)/8, $3860 // bitrev[335] = 3860
DATA bitrev_size4096_radix4<>+0xa80(SB)/8, $84   // bitrev[336] = 84
DATA bitrev_size4096_radix4<>+0xa88(SB)/8, $1108 // bitrev[337] = 1108
DATA bitrev_size4096_radix4<>+0xa90(SB)/8, $2132 // bitrev[338] = 2132
DATA bitrev_size4096_radix4<>+0xa98(SB)/8, $3156 // bitrev[339] = 3156
DATA bitrev_size4096_radix4<>+0xaa0(SB)/8, $340  // bitrev[340] = 340
DATA bitrev_size4096_radix4<>+0xaa8(SB)/8, $1364 // bitrev[341] = 1364
DATA bitrev_size4096_radix4<>+0xab0(SB)/8, $2388 // bitrev[342] = 2388
DATA bitrev_size4096_radix4<>+0xab8(SB)/8, $3412 // bitrev[343] = 3412
DATA bitrev_size4096_radix4<>+0xac0(SB)/8, $596  // bitrev[344] = 596
DATA bitrev_size4096_radix4<>+0xac8(SB)/8, $1620 // bitrev[345] = 1620
DATA bitrev_size4096_radix4<>+0xad0(SB)/8, $2644 // bitrev[346] = 2644
DATA bitrev_size4096_radix4<>+0xad8(SB)/8, $3668 // bitrev[347] = 3668
DATA bitrev_size4096_radix4<>+0xae0(SB)/8, $852  // bitrev[348] = 852
DATA bitrev_size4096_radix4<>+0xae8(SB)/8, $1876 // bitrev[349] = 1876
DATA bitrev_size4096_radix4<>+0xaf0(SB)/8, $2900 // bitrev[350] = 2900
DATA bitrev_size4096_radix4<>+0xaf8(SB)/8, $3924 // bitrev[351] = 3924
DATA bitrev_size4096_radix4<>+0xb00(SB)/8, $148  // bitrev[352] = 148
DATA bitrev_size4096_radix4<>+0xb08(SB)/8, $1172 // bitrev[353] = 1172
DATA bitrev_size4096_radix4<>+0xb10(SB)/8, $2196 // bitrev[354] = 2196
DATA bitrev_size4096_radix4<>+0xb18(SB)/8, $3220 // bitrev[355] = 3220
DATA bitrev_size4096_radix4<>+0xb20(SB)/8, $404  // bitrev[356] = 404
DATA bitrev_size4096_radix4<>+0xb28(SB)/8, $1428 // bitrev[357] = 1428
DATA bitrev_size4096_radix4<>+0xb30(SB)/8, $2452 // bitrev[358] = 2452
DATA bitrev_size4096_radix4<>+0xb38(SB)/8, $3476 // bitrev[359] = 3476
DATA bitrev_size4096_radix4<>+0xb40(SB)/8, $660  // bitrev[360] = 660
DATA bitrev_size4096_radix4<>+0xb48(SB)/8, $1684 // bitrev[361] = 1684
DATA bitrev_size4096_radix4<>+0xb50(SB)/8, $2708 // bitrev[362] = 2708
DATA bitrev_size4096_radix4<>+0xb58(SB)/8, $3732 // bitrev[363] = 3732
DATA bitrev_size4096_radix4<>+0xb60(SB)/8, $916  // bitrev[364] = 916
DATA bitrev_size4096_radix4<>+0xb68(SB)/8, $1940 // bitrev[365] = 1940
DATA bitrev_size4096_radix4<>+0xb70(SB)/8, $2964 // bitrev[366] = 2964
DATA bitrev_size4096_radix4<>+0xb78(SB)/8, $3988 // bitrev[367] = 3988
DATA bitrev_size4096_radix4<>+0xb80(SB)/8, $212  // bitrev[368] = 212
DATA bitrev_size4096_radix4<>+0xb88(SB)/8, $1236 // bitrev[369] = 1236
DATA bitrev_size4096_radix4<>+0xb90(SB)/8, $2260 // bitrev[370] = 2260
DATA bitrev_size4096_radix4<>+0xb98(SB)/8, $3284 // bitrev[371] = 3284
DATA bitrev_size4096_radix4<>+0xba0(SB)/8, $468  // bitrev[372] = 468
DATA bitrev_size4096_radix4<>+0xba8(SB)/8, $1492 // bitrev[373] = 1492
DATA bitrev_size4096_radix4<>+0xbb0(SB)/8, $2516 // bitrev[374] = 2516
DATA bitrev_size4096_radix4<>+0xbb8(SB)/8, $3540 // bitrev[375] = 3540
DATA bitrev_size4096_radix4<>+0xbc0(SB)/8, $724  // bitrev[376] = 724
DATA bitrev_size4096_radix4<>+0xbc8(SB)/8, $1748 // bitrev[377] = 1748
DATA bitrev_size4096_radix4<>+0xbd0(SB)/8, $2772 // bitrev[378] = 2772
DATA bitrev_size4096_radix4<>+0xbd8(SB)/8, $3796 // bitrev[379] = 3796
DATA bitrev_size4096_radix4<>+0xbe0(SB)/8, $980  // bitrev[380] = 980
DATA bitrev_size4096_radix4<>+0xbe8(SB)/8, $2004 // bitrev[381] = 2004
DATA bitrev_size4096_radix4<>+0xbf0(SB)/8, $3028 // bitrev[382] = 3028
DATA bitrev_size4096_radix4<>+0xbf8(SB)/8, $4052 // bitrev[383] = 4052
DATA bitrev_size4096_radix4<>+0xc00(SB)/8, $36   // bitrev[384] = 36
DATA bitrev_size4096_radix4<>+0xc08(SB)/8, $1060 // bitrev[385] = 1060
DATA bitrev_size4096_radix4<>+0xc10(SB)/8, $2084 // bitrev[386] = 2084
DATA bitrev_size4096_radix4<>+0xc18(SB)/8, $3108 // bitrev[387] = 3108
DATA bitrev_size4096_radix4<>+0xc20(SB)/8, $292  // bitrev[388] = 292
DATA bitrev_size4096_radix4<>+0xc28(SB)/8, $1316 // bitrev[389] = 1316
DATA bitrev_size4096_radix4<>+0xc30(SB)/8, $2340 // bitrev[390] = 2340
DATA bitrev_size4096_radix4<>+0xc38(SB)/8, $3364 // bitrev[391] = 3364
DATA bitrev_size4096_radix4<>+0xc40(SB)/8, $548  // bitrev[392] = 548
DATA bitrev_size4096_radix4<>+0xc48(SB)/8, $1572 // bitrev[393] = 1572
DATA bitrev_size4096_radix4<>+0xc50(SB)/8, $2596 // bitrev[394] = 2596
DATA bitrev_size4096_radix4<>+0xc58(SB)/8, $3620 // bitrev[395] = 3620
DATA bitrev_size4096_radix4<>+0xc60(SB)/8, $804  // bitrev[396] = 804
DATA bitrev_size4096_radix4<>+0xc68(SB)/8, $1828 // bitrev[397] = 1828
DATA bitrev_size4096_radix4<>+0xc70(SB)/8, $2852 // bitrev[398] = 2852
DATA bitrev_size4096_radix4<>+0xc78(SB)/8, $3876 // bitrev[399] = 3876
DATA bitrev_size4096_radix4<>+0xc80(SB)/8, $100  // bitrev[400] = 100
DATA bitrev_size4096_radix4<>+0xc88(SB)/8, $1124 // bitrev[401] = 1124
DATA bitrev_size4096_radix4<>+0xc90(SB)/8, $2148 // bitrev[402] = 2148
DATA bitrev_size4096_radix4<>+0xc98(SB)/8, $3172 // bitrev[403] = 3172
DATA bitrev_size4096_radix4<>+0xca0(SB)/8, $356  // bitrev[404] = 356
DATA bitrev_size4096_radix4<>+0xca8(SB)/8, $1380 // bitrev[405] = 1380
DATA bitrev_size4096_radix4<>+0xcb0(SB)/8, $2404 // bitrev[406] = 2404
DATA bitrev_size4096_radix4<>+0xcb8(SB)/8, $3428 // bitrev[407] = 3428
DATA bitrev_size4096_radix4<>+0xcc0(SB)/8, $612  // bitrev[408] = 612
DATA bitrev_size4096_radix4<>+0xcc8(SB)/8, $1636 // bitrev[409] = 1636
DATA bitrev_size4096_radix4<>+0xcd0(SB)/8, $2660 // bitrev[410] = 2660
DATA bitrev_size4096_radix4<>+0xcd8(SB)/8, $3684 // bitrev[411] = 3684
DATA bitrev_size4096_radix4<>+0xce0(SB)/8, $868  // bitrev[412] = 868
DATA bitrev_size4096_radix4<>+0xce8(SB)/8, $1892 // bitrev[413] = 1892
DATA bitrev_size4096_radix4<>+0xcf0(SB)/8, $2916 // bitrev[414] = 2916
DATA bitrev_size4096_radix4<>+0xcf8(SB)/8, $3940 // bitrev[415] = 3940
DATA bitrev_size4096_radix4<>+0xd00(SB)/8, $164  // bitrev[416] = 164
DATA bitrev_size4096_radix4<>+0xd08(SB)/8, $1188 // bitrev[417] = 1188
DATA bitrev_size4096_radix4<>+0xd10(SB)/8, $2212 // bitrev[418] = 2212
DATA bitrev_size4096_radix4<>+0xd18(SB)/8, $3236 // bitrev[419] = 3236
DATA bitrev_size4096_radix4<>+0xd20(SB)/8, $420  // bitrev[420] = 420
DATA bitrev_size4096_radix4<>+0xd28(SB)/8, $1444 // bitrev[421] = 1444
DATA bitrev_size4096_radix4<>+0xd30(SB)/8, $2468 // bitrev[422] = 2468
DATA bitrev_size4096_radix4<>+0xd38(SB)/8, $3492 // bitrev[423] = 3492
DATA bitrev_size4096_radix4<>+0xd40(SB)/8, $676  // bitrev[424] = 676
DATA bitrev_size4096_radix4<>+0xd48(SB)/8, $1700 // bitrev[425] = 1700
DATA bitrev_size4096_radix4<>+0xd50(SB)/8, $2724 // bitrev[426] = 2724
DATA bitrev_size4096_radix4<>+0xd58(SB)/8, $3748 // bitrev[427] = 3748
DATA bitrev_size4096_radix4<>+0xd60(SB)/8, $932  // bitrev[428] = 932
DATA bitrev_size4096_radix4<>+0xd68(SB)/8, $1956 // bitrev[429] = 1956
DATA bitrev_size4096_radix4<>+0xd70(SB)/8, $2980 // bitrev[430] = 2980
DATA bitrev_size4096_radix4<>+0xd78(SB)/8, $4004 // bitrev[431] = 4004
DATA bitrev_size4096_radix4<>+0xd80(SB)/8, $228  // bitrev[432] = 228
DATA bitrev_size4096_radix4<>+0xd88(SB)/8, $1252 // bitrev[433] = 1252
DATA bitrev_size4096_radix4<>+0xd90(SB)/8, $2276 // bitrev[434] = 2276
DATA bitrev_size4096_radix4<>+0xd98(SB)/8, $3300 // bitrev[435] = 3300
DATA bitrev_size4096_radix4<>+0xda0(SB)/8, $484  // bitrev[436] = 484
DATA bitrev_size4096_radix4<>+0xda8(SB)/8, $1508 // bitrev[437] = 1508
DATA bitrev_size4096_radix4<>+0xdb0(SB)/8, $2532 // bitrev[438] = 2532
DATA bitrev_size4096_radix4<>+0xdb8(SB)/8, $3556 // bitrev[439] = 3556
DATA bitrev_size4096_radix4<>+0xdc0(SB)/8, $740  // bitrev[440] = 740
DATA bitrev_size4096_radix4<>+0xdc8(SB)/8, $1764 // bitrev[441] = 1764
DATA bitrev_size4096_radix4<>+0xdd0(SB)/8, $2788 // bitrev[442] = 2788
DATA bitrev_size4096_radix4<>+0xdd8(SB)/8, $3812 // bitrev[443] = 3812
DATA bitrev_size4096_radix4<>+0xde0(SB)/8, $996  // bitrev[444] = 996
DATA bitrev_size4096_radix4<>+0xde8(SB)/8, $2020 // bitrev[445] = 2020
DATA bitrev_size4096_radix4<>+0xdf0(SB)/8, $3044 // bitrev[446] = 3044
DATA bitrev_size4096_radix4<>+0xdf8(SB)/8, $4068 // bitrev[447] = 4068
DATA bitrev_size4096_radix4<>+0xe00(SB)/8, $52   // bitrev[448] = 52
DATA bitrev_size4096_radix4<>+0xe08(SB)/8, $1076 // bitrev[449] = 1076
DATA bitrev_size4096_radix4<>+0xe10(SB)/8, $2100 // bitrev[450] = 2100
DATA bitrev_size4096_radix4<>+0xe18(SB)/8, $3124 // bitrev[451] = 3124
DATA bitrev_size4096_radix4<>+0xe20(SB)/8, $308  // bitrev[452] = 308
DATA bitrev_size4096_radix4<>+0xe28(SB)/8, $1332 // bitrev[453] = 1332
DATA bitrev_size4096_radix4<>+0xe30(SB)/8, $2356 // bitrev[454] = 2356
DATA bitrev_size4096_radix4<>+0xe38(SB)/8, $3380 // bitrev[455] = 3380
DATA bitrev_size4096_radix4<>+0xe40(SB)/8, $564  // bitrev[456] = 564
DATA bitrev_size4096_radix4<>+0xe48(SB)/8, $1588 // bitrev[457] = 1588
DATA bitrev_size4096_radix4<>+0xe50(SB)/8, $2612 // bitrev[458] = 2612
DATA bitrev_size4096_radix4<>+0xe58(SB)/8, $3636 // bitrev[459] = 3636
DATA bitrev_size4096_radix4<>+0xe60(SB)/8, $820  // bitrev[460] = 820
DATA bitrev_size4096_radix4<>+0xe68(SB)/8, $1844 // bitrev[461] = 1844
DATA bitrev_size4096_radix4<>+0xe70(SB)/8, $2868 // bitrev[462] = 2868
DATA bitrev_size4096_radix4<>+0xe78(SB)/8, $3892 // bitrev[463] = 3892
DATA bitrev_size4096_radix4<>+0xe80(SB)/8, $116  // bitrev[464] = 116
DATA bitrev_size4096_radix4<>+0xe88(SB)/8, $1140 // bitrev[465] = 1140
DATA bitrev_size4096_radix4<>+0xe90(SB)/8, $2164 // bitrev[466] = 2164
DATA bitrev_size4096_radix4<>+0xe98(SB)/8, $3188 // bitrev[467] = 3188
DATA bitrev_size4096_radix4<>+0xea0(SB)/8, $372  // bitrev[468] = 372
DATA bitrev_size4096_radix4<>+0xea8(SB)/8, $1396 // bitrev[469] = 1396
DATA bitrev_size4096_radix4<>+0xeb0(SB)/8, $2420 // bitrev[470] = 2420
DATA bitrev_size4096_radix4<>+0xeb8(SB)/8, $3444 // bitrev[471] = 3444
DATA bitrev_size4096_radix4<>+0xec0(SB)/8, $628  // bitrev[472] = 628
DATA bitrev_size4096_radix4<>+0xec8(SB)/8, $1652 // bitrev[473] = 1652
DATA bitrev_size4096_radix4<>+0xed0(SB)/8, $2676 // bitrev[474] = 2676
DATA bitrev_size4096_radix4<>+0xed8(SB)/8, $3700 // bitrev[475] = 3700
DATA bitrev_size4096_radix4<>+0xee0(SB)/8, $884  // bitrev[476] = 884
DATA bitrev_size4096_radix4<>+0xee8(SB)/8, $1908 // bitrev[477] = 1908
DATA bitrev_size4096_radix4<>+0xef0(SB)/8, $2932 // bitrev[478] = 2932
DATA bitrev_size4096_radix4<>+0xef8(SB)/8, $3956 // bitrev[479] = 3956
DATA bitrev_size4096_radix4<>+0xf00(SB)/8, $180  // bitrev[480] = 180
DATA bitrev_size4096_radix4<>+0xf08(SB)/8, $1204 // bitrev[481] = 1204
DATA bitrev_size4096_radix4<>+0xf10(SB)/8, $2228 // bitrev[482] = 2228
DATA bitrev_size4096_radix4<>+0xf18(SB)/8, $3252 // bitrev[483] = 3252
DATA bitrev_size4096_radix4<>+0xf20(SB)/8, $436  // bitrev[484] = 436
DATA bitrev_size4096_radix4<>+0xf28(SB)/8, $1460 // bitrev[485] = 1460
DATA bitrev_size4096_radix4<>+0xf30(SB)/8, $2484 // bitrev[486] = 2484
DATA bitrev_size4096_radix4<>+0xf38(SB)/8, $3508 // bitrev[487] = 3508
DATA bitrev_size4096_radix4<>+0xf40(SB)/8, $692  // bitrev[488] = 692
DATA bitrev_size4096_radix4<>+0xf48(SB)/8, $1716 // bitrev[489] = 1716
DATA bitrev_size4096_radix4<>+0xf50(SB)/8, $2740 // bitrev[490] = 2740
DATA bitrev_size4096_radix4<>+0xf58(SB)/8, $3764 // bitrev[491] = 3764
DATA bitrev_size4096_radix4<>+0xf60(SB)/8, $948  // bitrev[492] = 948
DATA bitrev_size4096_radix4<>+0xf68(SB)/8, $1972 // bitrev[493] = 1972
DATA bitrev_size4096_radix4<>+0xf70(SB)/8, $2996 // bitrev[494] = 2996
DATA bitrev_size4096_radix4<>+0xf78(SB)/8, $4020 // bitrev[495] = 4020
DATA bitrev_size4096_radix4<>+0xf80(SB)/8, $244  // bitrev[496] = 244
DATA bitrev_size4096_radix4<>+0xf88(SB)/8, $1268 // bitrev[497] = 1268
DATA bitrev_size4096_radix4<>+0xf90(SB)/8, $2292 // bitrev[498] = 2292
DATA bitrev_size4096_radix4<>+0xf98(SB)/8, $3316 // bitrev[499] = 3316
DATA bitrev_size4096_radix4<>+0xfa0(SB)/8, $500  // bitrev[500] = 500
DATA bitrev_size4096_radix4<>+0xfa8(SB)/8, $1524 // bitrev[501] = 1524
DATA bitrev_size4096_radix4<>+0xfb0(SB)/8, $2548 // bitrev[502] = 2548
DATA bitrev_size4096_radix4<>+0xfb8(SB)/8, $3572 // bitrev[503] = 3572
DATA bitrev_size4096_radix4<>+0xfc0(SB)/8, $756  // bitrev[504] = 756
DATA bitrev_size4096_radix4<>+0xfc8(SB)/8, $1780 // bitrev[505] = 1780
DATA bitrev_size4096_radix4<>+0xfd0(SB)/8, $2804 // bitrev[506] = 2804
DATA bitrev_size4096_radix4<>+0xfd8(SB)/8, $3828 // bitrev[507] = 3828
DATA bitrev_size4096_radix4<>+0xfe0(SB)/8, $1012 // bitrev[508] = 1012
DATA bitrev_size4096_radix4<>+0xfe8(SB)/8, $2036 // bitrev[509] = 2036
DATA bitrev_size4096_radix4<>+0xff0(SB)/8, $3060 // bitrev[510] = 3060
DATA bitrev_size4096_radix4<>+0xff8(SB)/8, $4084 // bitrev[511] = 4084
DATA bitrev_size4096_radix4<>+0x1000(SB)/8, $8    // bitrev[512] = 8
DATA bitrev_size4096_radix4<>+0x1008(SB)/8, $1032 // bitrev[513] = 1032
DATA bitrev_size4096_radix4<>+0x1010(SB)/8, $2056 // bitrev[514] = 2056
DATA bitrev_size4096_radix4<>+0x1018(SB)/8, $3080 // bitrev[515] = 3080
DATA bitrev_size4096_radix4<>+0x1020(SB)/8, $264  // bitrev[516] = 264
DATA bitrev_size4096_radix4<>+0x1028(SB)/8, $1288 // bitrev[517] = 1288
DATA bitrev_size4096_radix4<>+0x1030(SB)/8, $2312 // bitrev[518] = 2312
DATA bitrev_size4096_radix4<>+0x1038(SB)/8, $3336 // bitrev[519] = 3336
DATA bitrev_size4096_radix4<>+0x1040(SB)/8, $520  // bitrev[520] = 520
DATA bitrev_size4096_radix4<>+0x1048(SB)/8, $1544 // bitrev[521] = 1544
DATA bitrev_size4096_radix4<>+0x1050(SB)/8, $2568 // bitrev[522] = 2568
DATA bitrev_size4096_radix4<>+0x1058(SB)/8, $3592 // bitrev[523] = 3592
DATA bitrev_size4096_radix4<>+0x1060(SB)/8, $776  // bitrev[524] = 776
DATA bitrev_size4096_radix4<>+0x1068(SB)/8, $1800 // bitrev[525] = 1800
DATA bitrev_size4096_radix4<>+0x1070(SB)/8, $2824 // bitrev[526] = 2824
DATA bitrev_size4096_radix4<>+0x1078(SB)/8, $3848 // bitrev[527] = 3848
DATA bitrev_size4096_radix4<>+0x1080(SB)/8, $72   // bitrev[528] = 72
DATA bitrev_size4096_radix4<>+0x1088(SB)/8, $1096 // bitrev[529] = 1096
DATA bitrev_size4096_radix4<>+0x1090(SB)/8, $2120 // bitrev[530] = 2120
DATA bitrev_size4096_radix4<>+0x1098(SB)/8, $3144 // bitrev[531] = 3144
DATA bitrev_size4096_radix4<>+0x10a0(SB)/8, $328  // bitrev[532] = 328
DATA bitrev_size4096_radix4<>+0x10a8(SB)/8, $1352 // bitrev[533] = 1352
DATA bitrev_size4096_radix4<>+0x10b0(SB)/8, $2376 // bitrev[534] = 2376
DATA bitrev_size4096_radix4<>+0x10b8(SB)/8, $3400 // bitrev[535] = 3400
DATA bitrev_size4096_radix4<>+0x10c0(SB)/8, $584  // bitrev[536] = 584
DATA bitrev_size4096_radix4<>+0x10c8(SB)/8, $1608 // bitrev[537] = 1608
DATA bitrev_size4096_radix4<>+0x10d0(SB)/8, $2632 // bitrev[538] = 2632
DATA bitrev_size4096_radix4<>+0x10d8(SB)/8, $3656 // bitrev[539] = 3656
DATA bitrev_size4096_radix4<>+0x10e0(SB)/8, $840  // bitrev[540] = 840
DATA bitrev_size4096_radix4<>+0x10e8(SB)/8, $1864 // bitrev[541] = 1864
DATA bitrev_size4096_radix4<>+0x10f0(SB)/8, $2888 // bitrev[542] = 2888
DATA bitrev_size4096_radix4<>+0x10f8(SB)/8, $3912 // bitrev[543] = 3912
DATA bitrev_size4096_radix4<>+0x1100(SB)/8, $136  // bitrev[544] = 136
DATA bitrev_size4096_radix4<>+0x1108(SB)/8, $1160 // bitrev[545] = 1160
DATA bitrev_size4096_radix4<>+0x1110(SB)/8, $2184 // bitrev[546] = 2184
DATA bitrev_size4096_radix4<>+0x1118(SB)/8, $3208 // bitrev[547] = 3208
DATA bitrev_size4096_radix4<>+0x1120(SB)/8, $392  // bitrev[548] = 392
DATA bitrev_size4096_radix4<>+0x1128(SB)/8, $1416 // bitrev[549] = 1416
DATA bitrev_size4096_radix4<>+0x1130(SB)/8, $2440 // bitrev[550] = 2440
DATA bitrev_size4096_radix4<>+0x1138(SB)/8, $3464 // bitrev[551] = 3464
DATA bitrev_size4096_radix4<>+0x1140(SB)/8, $648  // bitrev[552] = 648
DATA bitrev_size4096_radix4<>+0x1148(SB)/8, $1672 // bitrev[553] = 1672
DATA bitrev_size4096_radix4<>+0x1150(SB)/8, $2696 // bitrev[554] = 2696
DATA bitrev_size4096_radix4<>+0x1158(SB)/8, $3720 // bitrev[555] = 3720
DATA bitrev_size4096_radix4<>+0x1160(SB)/8, $904  // bitrev[556] = 904
DATA bitrev_size4096_radix4<>+0x1168(SB)/8, $1928 // bitrev[557] = 1928
DATA bitrev_size4096_radix4<>+0x1170(SB)/8, $2952 // bitrev[558] = 2952
DATA bitrev_size4096_radix4<>+0x1178(SB)/8, $3976 // bitrev[559] = 3976
DATA bitrev_size4096_radix4<>+0x1180(SB)/8, $200  // bitrev[560] = 200
DATA bitrev_size4096_radix4<>+0x1188(SB)/8, $1224 // bitrev[561] = 1224
DATA bitrev_size4096_radix4<>+0x1190(SB)/8, $2248 // bitrev[562] = 2248
DATA bitrev_size4096_radix4<>+0x1198(SB)/8, $3272 // bitrev[563] = 3272
DATA bitrev_size4096_radix4<>+0x11a0(SB)/8, $456  // bitrev[564] = 456
DATA bitrev_size4096_radix4<>+0x11a8(SB)/8, $1480 // bitrev[565] = 1480
DATA bitrev_size4096_radix4<>+0x11b0(SB)/8, $2504 // bitrev[566] = 2504
DATA bitrev_size4096_radix4<>+0x11b8(SB)/8, $3528 // bitrev[567] = 3528
DATA bitrev_size4096_radix4<>+0x11c0(SB)/8, $712  // bitrev[568] = 712
DATA bitrev_size4096_radix4<>+0x11c8(SB)/8, $1736 // bitrev[569] = 1736
DATA bitrev_size4096_radix4<>+0x11d0(SB)/8, $2760 // bitrev[570] = 2760
DATA bitrev_size4096_radix4<>+0x11d8(SB)/8, $3784 // bitrev[571] = 3784
DATA bitrev_size4096_radix4<>+0x11e0(SB)/8, $968  // bitrev[572] = 968
DATA bitrev_size4096_radix4<>+0x11e8(SB)/8, $1992 // bitrev[573] = 1992
DATA bitrev_size4096_radix4<>+0x11f0(SB)/8, $3016 // bitrev[574] = 3016
DATA bitrev_size4096_radix4<>+0x11f8(SB)/8, $4040 // bitrev[575] = 4040
DATA bitrev_size4096_radix4<>+0x1200(SB)/8, $24   // bitrev[576] = 24
DATA bitrev_size4096_radix4<>+0x1208(SB)/8, $1048 // bitrev[577] = 1048
DATA bitrev_size4096_radix4<>+0x1210(SB)/8, $2072 // bitrev[578] = 2072
DATA bitrev_size4096_radix4<>+0x1218(SB)/8, $3096 // bitrev[579] = 3096
DATA bitrev_size4096_radix4<>+0x1220(SB)/8, $280  // bitrev[580] = 280
DATA bitrev_size4096_radix4<>+0x1228(SB)/8, $1304 // bitrev[581] = 1304
DATA bitrev_size4096_radix4<>+0x1230(SB)/8, $2328 // bitrev[582] = 2328
DATA bitrev_size4096_radix4<>+0x1238(SB)/8, $3352 // bitrev[583] = 3352
DATA bitrev_size4096_radix4<>+0x1240(SB)/8, $536  // bitrev[584] = 536
DATA bitrev_size4096_radix4<>+0x1248(SB)/8, $1560 // bitrev[585] = 1560
DATA bitrev_size4096_radix4<>+0x1250(SB)/8, $2584 // bitrev[586] = 2584
DATA bitrev_size4096_radix4<>+0x1258(SB)/8, $3608 // bitrev[587] = 3608
DATA bitrev_size4096_radix4<>+0x1260(SB)/8, $792  // bitrev[588] = 792
DATA bitrev_size4096_radix4<>+0x1268(SB)/8, $1816 // bitrev[589] = 1816
DATA bitrev_size4096_radix4<>+0x1270(SB)/8, $2840 // bitrev[590] = 2840
DATA bitrev_size4096_radix4<>+0x1278(SB)/8, $3864 // bitrev[591] = 3864
DATA bitrev_size4096_radix4<>+0x1280(SB)/8, $88   // bitrev[592] = 88
DATA bitrev_size4096_radix4<>+0x1288(SB)/8, $1112 // bitrev[593] = 1112
DATA bitrev_size4096_radix4<>+0x1290(SB)/8, $2136 // bitrev[594] = 2136
DATA bitrev_size4096_radix4<>+0x1298(SB)/8, $3160 // bitrev[595] = 3160
DATA bitrev_size4096_radix4<>+0x12a0(SB)/8, $344  // bitrev[596] = 344
DATA bitrev_size4096_radix4<>+0x12a8(SB)/8, $1368 // bitrev[597] = 1368
DATA bitrev_size4096_radix4<>+0x12b0(SB)/8, $2392 // bitrev[598] = 2392
DATA bitrev_size4096_radix4<>+0x12b8(SB)/8, $3416 // bitrev[599] = 3416
DATA bitrev_size4096_radix4<>+0x12c0(SB)/8, $600  // bitrev[600] = 600
DATA bitrev_size4096_radix4<>+0x12c8(SB)/8, $1624 // bitrev[601] = 1624
DATA bitrev_size4096_radix4<>+0x12d0(SB)/8, $2648 // bitrev[602] = 2648
DATA bitrev_size4096_radix4<>+0x12d8(SB)/8, $3672 // bitrev[603] = 3672
DATA bitrev_size4096_radix4<>+0x12e0(SB)/8, $856  // bitrev[604] = 856
DATA bitrev_size4096_radix4<>+0x12e8(SB)/8, $1880 // bitrev[605] = 1880
DATA bitrev_size4096_radix4<>+0x12f0(SB)/8, $2904 // bitrev[606] = 2904
DATA bitrev_size4096_radix4<>+0x12f8(SB)/8, $3928 // bitrev[607] = 3928
DATA bitrev_size4096_radix4<>+0x1300(SB)/8, $152  // bitrev[608] = 152
DATA bitrev_size4096_radix4<>+0x1308(SB)/8, $1176 // bitrev[609] = 1176
DATA bitrev_size4096_radix4<>+0x1310(SB)/8, $2200 // bitrev[610] = 2200
DATA bitrev_size4096_radix4<>+0x1318(SB)/8, $3224 // bitrev[611] = 3224
DATA bitrev_size4096_radix4<>+0x1320(SB)/8, $408  // bitrev[612] = 408
DATA bitrev_size4096_radix4<>+0x1328(SB)/8, $1432 // bitrev[613] = 1432
DATA bitrev_size4096_radix4<>+0x1330(SB)/8, $2456 // bitrev[614] = 2456
DATA bitrev_size4096_radix4<>+0x1338(SB)/8, $3480 // bitrev[615] = 3480
DATA bitrev_size4096_radix4<>+0x1340(SB)/8, $664  // bitrev[616] = 664
DATA bitrev_size4096_radix4<>+0x1348(SB)/8, $1688 // bitrev[617] = 1688
DATA bitrev_size4096_radix4<>+0x1350(SB)/8, $2712 // bitrev[618] = 2712
DATA bitrev_size4096_radix4<>+0x1358(SB)/8, $3736 // bitrev[619] = 3736
DATA bitrev_size4096_radix4<>+0x1360(SB)/8, $920  // bitrev[620] = 920
DATA bitrev_size4096_radix4<>+0x1368(SB)/8, $1944 // bitrev[621] = 1944
DATA bitrev_size4096_radix4<>+0x1370(SB)/8, $2968 // bitrev[622] = 2968
DATA bitrev_size4096_radix4<>+0x1378(SB)/8, $3992 // bitrev[623] = 3992
DATA bitrev_size4096_radix4<>+0x1380(SB)/8, $216  // bitrev[624] = 216
DATA bitrev_size4096_radix4<>+0x1388(SB)/8, $1240 // bitrev[625] = 1240
DATA bitrev_size4096_radix4<>+0x1390(SB)/8, $2264 // bitrev[626] = 2264
DATA bitrev_size4096_radix4<>+0x1398(SB)/8, $3288 // bitrev[627] = 3288
DATA bitrev_size4096_radix4<>+0x13a0(SB)/8, $472  // bitrev[628] = 472
DATA bitrev_size4096_radix4<>+0x13a8(SB)/8, $1496 // bitrev[629] = 1496
DATA bitrev_size4096_radix4<>+0x13b0(SB)/8, $2520 // bitrev[630] = 2520
DATA bitrev_size4096_radix4<>+0x13b8(SB)/8, $3544 // bitrev[631] = 3544
DATA bitrev_size4096_radix4<>+0x13c0(SB)/8, $728  // bitrev[632] = 728
DATA bitrev_size4096_radix4<>+0x13c8(SB)/8, $1752 // bitrev[633] = 1752
DATA bitrev_size4096_radix4<>+0x13d0(SB)/8, $2776 // bitrev[634] = 2776
DATA bitrev_size4096_radix4<>+0x13d8(SB)/8, $3800 // bitrev[635] = 3800
DATA bitrev_size4096_radix4<>+0x13e0(SB)/8, $984  // bitrev[636] = 984
DATA bitrev_size4096_radix4<>+0x13e8(SB)/8, $2008 // bitrev[637] = 2008
DATA bitrev_size4096_radix4<>+0x13f0(SB)/8, $3032 // bitrev[638] = 3032
DATA bitrev_size4096_radix4<>+0x13f8(SB)/8, $4056 // bitrev[639] = 4056
DATA bitrev_size4096_radix4<>+0x1400(SB)/8, $40   // bitrev[640] = 40
DATA bitrev_size4096_radix4<>+0x1408(SB)/8, $1064 // bitrev[641] = 1064
DATA bitrev_size4096_radix4<>+0x1410(SB)/8, $2088 // bitrev[642] = 2088
DATA bitrev_size4096_radix4<>+0x1418(SB)/8, $3112 // bitrev[643] = 3112
DATA bitrev_size4096_radix4<>+0x1420(SB)/8, $296  // bitrev[644] = 296
DATA bitrev_size4096_radix4<>+0x1428(SB)/8, $1320 // bitrev[645] = 1320
DATA bitrev_size4096_radix4<>+0x1430(SB)/8, $2344 // bitrev[646] = 2344
DATA bitrev_size4096_radix4<>+0x1438(SB)/8, $3368 // bitrev[647] = 3368
DATA bitrev_size4096_radix4<>+0x1440(SB)/8, $552  // bitrev[648] = 552
DATA bitrev_size4096_radix4<>+0x1448(SB)/8, $1576 // bitrev[649] = 1576
DATA bitrev_size4096_radix4<>+0x1450(SB)/8, $2600 // bitrev[650] = 2600
DATA bitrev_size4096_radix4<>+0x1458(SB)/8, $3624 // bitrev[651] = 3624
DATA bitrev_size4096_radix4<>+0x1460(SB)/8, $808  // bitrev[652] = 808
DATA bitrev_size4096_radix4<>+0x1468(SB)/8, $1832 // bitrev[653] = 1832
DATA bitrev_size4096_radix4<>+0x1470(SB)/8, $2856 // bitrev[654] = 2856
DATA bitrev_size4096_radix4<>+0x1478(SB)/8, $3880 // bitrev[655] = 3880
DATA bitrev_size4096_radix4<>+0x1480(SB)/8, $104  // bitrev[656] = 104
DATA bitrev_size4096_radix4<>+0x1488(SB)/8, $1128 // bitrev[657] = 1128
DATA bitrev_size4096_radix4<>+0x1490(SB)/8, $2152 // bitrev[658] = 2152
DATA bitrev_size4096_radix4<>+0x1498(SB)/8, $3176 // bitrev[659] = 3176
DATA bitrev_size4096_radix4<>+0x14a0(SB)/8, $360  // bitrev[660] = 360
DATA bitrev_size4096_radix4<>+0x14a8(SB)/8, $1384 // bitrev[661] = 1384
DATA bitrev_size4096_radix4<>+0x14b0(SB)/8, $2408 // bitrev[662] = 2408
DATA bitrev_size4096_radix4<>+0x14b8(SB)/8, $3432 // bitrev[663] = 3432
DATA bitrev_size4096_radix4<>+0x14c0(SB)/8, $616  // bitrev[664] = 616
DATA bitrev_size4096_radix4<>+0x14c8(SB)/8, $1640 // bitrev[665] = 1640
DATA bitrev_size4096_radix4<>+0x14d0(SB)/8, $2664 // bitrev[666] = 2664
DATA bitrev_size4096_radix4<>+0x14d8(SB)/8, $3688 // bitrev[667] = 3688
DATA bitrev_size4096_radix4<>+0x14e0(SB)/8, $872  // bitrev[668] = 872
DATA bitrev_size4096_radix4<>+0x14e8(SB)/8, $1896 // bitrev[669] = 1896
DATA bitrev_size4096_radix4<>+0x14f0(SB)/8, $2920 // bitrev[670] = 2920
DATA bitrev_size4096_radix4<>+0x14f8(SB)/8, $3944 // bitrev[671] = 3944
DATA bitrev_size4096_radix4<>+0x1500(SB)/8, $168  // bitrev[672] = 168
DATA bitrev_size4096_radix4<>+0x1508(SB)/8, $1192 // bitrev[673] = 1192
DATA bitrev_size4096_radix4<>+0x1510(SB)/8, $2216 // bitrev[674] = 2216
DATA bitrev_size4096_radix4<>+0x1518(SB)/8, $3240 // bitrev[675] = 3240
DATA bitrev_size4096_radix4<>+0x1520(SB)/8, $424  // bitrev[676] = 424
DATA bitrev_size4096_radix4<>+0x1528(SB)/8, $1448 // bitrev[677] = 1448
DATA bitrev_size4096_radix4<>+0x1530(SB)/8, $2472 // bitrev[678] = 2472
DATA bitrev_size4096_radix4<>+0x1538(SB)/8, $3496 // bitrev[679] = 3496
DATA bitrev_size4096_radix4<>+0x1540(SB)/8, $680  // bitrev[680] = 680
DATA bitrev_size4096_radix4<>+0x1548(SB)/8, $1704 // bitrev[681] = 1704
DATA bitrev_size4096_radix4<>+0x1550(SB)/8, $2728 // bitrev[682] = 2728
DATA bitrev_size4096_radix4<>+0x1558(SB)/8, $3752 // bitrev[683] = 3752
DATA bitrev_size4096_radix4<>+0x1560(SB)/8, $936  // bitrev[684] = 936
DATA bitrev_size4096_radix4<>+0x1568(SB)/8, $1960 // bitrev[685] = 1960
DATA bitrev_size4096_radix4<>+0x1570(SB)/8, $2984 // bitrev[686] = 2984
DATA bitrev_size4096_radix4<>+0x1578(SB)/8, $4008 // bitrev[687] = 4008
DATA bitrev_size4096_radix4<>+0x1580(SB)/8, $232  // bitrev[688] = 232
DATA bitrev_size4096_radix4<>+0x1588(SB)/8, $1256 // bitrev[689] = 1256
DATA bitrev_size4096_radix4<>+0x1590(SB)/8, $2280 // bitrev[690] = 2280
DATA bitrev_size4096_radix4<>+0x1598(SB)/8, $3304 // bitrev[691] = 3304
DATA bitrev_size4096_radix4<>+0x15a0(SB)/8, $488  // bitrev[692] = 488
DATA bitrev_size4096_radix4<>+0x15a8(SB)/8, $1512 // bitrev[693] = 1512
DATA bitrev_size4096_radix4<>+0x15b0(SB)/8, $2536 // bitrev[694] = 2536
DATA bitrev_size4096_radix4<>+0x15b8(SB)/8, $3560 // bitrev[695] = 3560
DATA bitrev_size4096_radix4<>+0x15c0(SB)/8, $744  // bitrev[696] = 744
DATA bitrev_size4096_radix4<>+0x15c8(SB)/8, $1768 // bitrev[697] = 1768
DATA bitrev_size4096_radix4<>+0x15d0(SB)/8, $2792 // bitrev[698] = 2792
DATA bitrev_size4096_radix4<>+0x15d8(SB)/8, $3816 // bitrev[699] = 3816
DATA bitrev_size4096_radix4<>+0x15e0(SB)/8, $1000 // bitrev[700] = 1000
DATA bitrev_size4096_radix4<>+0x15e8(SB)/8, $2024 // bitrev[701] = 2024
DATA bitrev_size4096_radix4<>+0x15f0(SB)/8, $3048 // bitrev[702] = 3048
DATA bitrev_size4096_radix4<>+0x15f8(SB)/8, $4072 // bitrev[703] = 4072
DATA bitrev_size4096_radix4<>+0x1600(SB)/8, $56   // bitrev[704] = 56
DATA bitrev_size4096_radix4<>+0x1608(SB)/8, $1080 // bitrev[705] = 1080
DATA bitrev_size4096_radix4<>+0x1610(SB)/8, $2104 // bitrev[706] = 2104
DATA bitrev_size4096_radix4<>+0x1618(SB)/8, $3128 // bitrev[707] = 3128
DATA bitrev_size4096_radix4<>+0x1620(SB)/8, $312  // bitrev[708] = 312
DATA bitrev_size4096_radix4<>+0x1628(SB)/8, $1336 // bitrev[709] = 1336
DATA bitrev_size4096_radix4<>+0x1630(SB)/8, $2360 // bitrev[710] = 2360
DATA bitrev_size4096_radix4<>+0x1638(SB)/8, $3384 // bitrev[711] = 3384
DATA bitrev_size4096_radix4<>+0x1640(SB)/8, $568  // bitrev[712] = 568
DATA bitrev_size4096_radix4<>+0x1648(SB)/8, $1592 // bitrev[713] = 1592
DATA bitrev_size4096_radix4<>+0x1650(SB)/8, $2616 // bitrev[714] = 2616
DATA bitrev_size4096_radix4<>+0x1658(SB)/8, $3640 // bitrev[715] = 3640
DATA bitrev_size4096_radix4<>+0x1660(SB)/8, $824  // bitrev[716] = 824
DATA bitrev_size4096_radix4<>+0x1668(SB)/8, $1848 // bitrev[717] = 1848
DATA bitrev_size4096_radix4<>+0x1670(SB)/8, $2872 // bitrev[718] = 2872
DATA bitrev_size4096_radix4<>+0x1678(SB)/8, $3896 // bitrev[719] = 3896
DATA bitrev_size4096_radix4<>+0x1680(SB)/8, $120  // bitrev[720] = 120
DATA bitrev_size4096_radix4<>+0x1688(SB)/8, $1144 // bitrev[721] = 1144
DATA bitrev_size4096_radix4<>+0x1690(SB)/8, $2168 // bitrev[722] = 2168
DATA bitrev_size4096_radix4<>+0x1698(SB)/8, $3192 // bitrev[723] = 3192
DATA bitrev_size4096_radix4<>+0x16a0(SB)/8, $376  // bitrev[724] = 376
DATA bitrev_size4096_radix4<>+0x16a8(SB)/8, $1400 // bitrev[725] = 1400
DATA bitrev_size4096_radix4<>+0x16b0(SB)/8, $2424 // bitrev[726] = 2424
DATA bitrev_size4096_radix4<>+0x16b8(SB)/8, $3448 // bitrev[727] = 3448
DATA bitrev_size4096_radix4<>+0x16c0(SB)/8, $632  // bitrev[728] = 632
DATA bitrev_size4096_radix4<>+0x16c8(SB)/8, $1656 // bitrev[729] = 1656
DATA bitrev_size4096_radix4<>+0x16d0(SB)/8, $2680 // bitrev[730] = 2680
DATA bitrev_size4096_radix4<>+0x16d8(SB)/8, $3704 // bitrev[731] = 3704
DATA bitrev_size4096_radix4<>+0x16e0(SB)/8, $888  // bitrev[732] = 888
DATA bitrev_size4096_radix4<>+0x16e8(SB)/8, $1912 // bitrev[733] = 1912
DATA bitrev_size4096_radix4<>+0x16f0(SB)/8, $2936 // bitrev[734] = 2936
DATA bitrev_size4096_radix4<>+0x16f8(SB)/8, $3960 // bitrev[735] = 3960
DATA bitrev_size4096_radix4<>+0x1700(SB)/8, $184  // bitrev[736] = 184
DATA bitrev_size4096_radix4<>+0x1708(SB)/8, $1208 // bitrev[737] = 1208
DATA bitrev_size4096_radix4<>+0x1710(SB)/8, $2232 // bitrev[738] = 2232
DATA bitrev_size4096_radix4<>+0x1718(SB)/8, $3256 // bitrev[739] = 3256
DATA bitrev_size4096_radix4<>+0x1720(SB)/8, $440  // bitrev[740] = 440
DATA bitrev_size4096_radix4<>+0x1728(SB)/8, $1464 // bitrev[741] = 1464
DATA bitrev_size4096_radix4<>+0x1730(SB)/8, $2488 // bitrev[742] = 2488
DATA bitrev_size4096_radix4<>+0x1738(SB)/8, $3512 // bitrev[743] = 3512
DATA bitrev_size4096_radix4<>+0x1740(SB)/8, $696  // bitrev[744] = 696
DATA bitrev_size4096_radix4<>+0x1748(SB)/8, $1720 // bitrev[745] = 1720
DATA bitrev_size4096_radix4<>+0x1750(SB)/8, $2744 // bitrev[746] = 2744
DATA bitrev_size4096_radix4<>+0x1758(SB)/8, $3768 // bitrev[747] = 3768
DATA bitrev_size4096_radix4<>+0x1760(SB)/8, $952  // bitrev[748] = 952
DATA bitrev_size4096_radix4<>+0x1768(SB)/8, $1976 // bitrev[749] = 1976
DATA bitrev_size4096_radix4<>+0x1770(SB)/8, $3000 // bitrev[750] = 3000
DATA bitrev_size4096_radix4<>+0x1778(SB)/8, $4024 // bitrev[751] = 4024
DATA bitrev_size4096_radix4<>+0x1780(SB)/8, $248  // bitrev[752] = 248
DATA bitrev_size4096_radix4<>+0x1788(SB)/8, $1272 // bitrev[753] = 1272
DATA bitrev_size4096_radix4<>+0x1790(SB)/8, $2296 // bitrev[754] = 2296
DATA bitrev_size4096_radix4<>+0x1798(SB)/8, $3320 // bitrev[755] = 3320
DATA bitrev_size4096_radix4<>+0x17a0(SB)/8, $504  // bitrev[756] = 504
DATA bitrev_size4096_radix4<>+0x17a8(SB)/8, $1528 // bitrev[757] = 1528
DATA bitrev_size4096_radix4<>+0x17b0(SB)/8, $2552 // bitrev[758] = 2552
DATA bitrev_size4096_radix4<>+0x17b8(SB)/8, $3576 // bitrev[759] = 3576
DATA bitrev_size4096_radix4<>+0x17c0(SB)/8, $760  // bitrev[760] = 760
DATA bitrev_size4096_radix4<>+0x17c8(SB)/8, $1784 // bitrev[761] = 1784
DATA bitrev_size4096_radix4<>+0x17d0(SB)/8, $2808 // bitrev[762] = 2808
DATA bitrev_size4096_radix4<>+0x17d8(SB)/8, $3832 // bitrev[763] = 3832
DATA bitrev_size4096_radix4<>+0x17e0(SB)/8, $1016 // bitrev[764] = 1016
DATA bitrev_size4096_radix4<>+0x17e8(SB)/8, $2040 // bitrev[765] = 2040
DATA bitrev_size4096_radix4<>+0x17f0(SB)/8, $3064 // bitrev[766] = 3064
DATA bitrev_size4096_radix4<>+0x17f8(SB)/8, $4088 // bitrev[767] = 4088
DATA bitrev_size4096_radix4<>+0x1800(SB)/8, $12   // bitrev[768] = 12
DATA bitrev_size4096_radix4<>+0x1808(SB)/8, $1036 // bitrev[769] = 1036
DATA bitrev_size4096_radix4<>+0x1810(SB)/8, $2060 // bitrev[770] = 2060
DATA bitrev_size4096_radix4<>+0x1818(SB)/8, $3084 // bitrev[771] = 3084
DATA bitrev_size4096_radix4<>+0x1820(SB)/8, $268  // bitrev[772] = 268
DATA bitrev_size4096_radix4<>+0x1828(SB)/8, $1292 // bitrev[773] = 1292
DATA bitrev_size4096_radix4<>+0x1830(SB)/8, $2316 // bitrev[774] = 2316
DATA bitrev_size4096_radix4<>+0x1838(SB)/8, $3340 // bitrev[775] = 3340
DATA bitrev_size4096_radix4<>+0x1840(SB)/8, $524  // bitrev[776] = 524
DATA bitrev_size4096_radix4<>+0x1848(SB)/8, $1548 // bitrev[777] = 1548
DATA bitrev_size4096_radix4<>+0x1850(SB)/8, $2572 // bitrev[778] = 2572
DATA bitrev_size4096_radix4<>+0x1858(SB)/8, $3596 // bitrev[779] = 3596
DATA bitrev_size4096_radix4<>+0x1860(SB)/8, $780  // bitrev[780] = 780
DATA bitrev_size4096_radix4<>+0x1868(SB)/8, $1804 // bitrev[781] = 1804
DATA bitrev_size4096_radix4<>+0x1870(SB)/8, $2828 // bitrev[782] = 2828
DATA bitrev_size4096_radix4<>+0x1878(SB)/8, $3852 // bitrev[783] = 3852
DATA bitrev_size4096_radix4<>+0x1880(SB)/8, $76   // bitrev[784] = 76
DATA bitrev_size4096_radix4<>+0x1888(SB)/8, $1100 // bitrev[785] = 1100
DATA bitrev_size4096_radix4<>+0x1890(SB)/8, $2124 // bitrev[786] = 2124
DATA bitrev_size4096_radix4<>+0x1898(SB)/8, $3148 // bitrev[787] = 3148
DATA bitrev_size4096_radix4<>+0x18a0(SB)/8, $332  // bitrev[788] = 332
DATA bitrev_size4096_radix4<>+0x18a8(SB)/8, $1356 // bitrev[789] = 1356
DATA bitrev_size4096_radix4<>+0x18b0(SB)/8, $2380 // bitrev[790] = 2380
DATA bitrev_size4096_radix4<>+0x18b8(SB)/8, $3404 // bitrev[791] = 3404
DATA bitrev_size4096_radix4<>+0x18c0(SB)/8, $588  // bitrev[792] = 588
DATA bitrev_size4096_radix4<>+0x18c8(SB)/8, $1612 // bitrev[793] = 1612
DATA bitrev_size4096_radix4<>+0x18d0(SB)/8, $2636 // bitrev[794] = 2636
DATA bitrev_size4096_radix4<>+0x18d8(SB)/8, $3660 // bitrev[795] = 3660
DATA bitrev_size4096_radix4<>+0x18e0(SB)/8, $844  // bitrev[796] = 844
DATA bitrev_size4096_radix4<>+0x18e8(SB)/8, $1868 // bitrev[797] = 1868
DATA bitrev_size4096_radix4<>+0x18f0(SB)/8, $2892 // bitrev[798] = 2892
DATA bitrev_size4096_radix4<>+0x18f8(SB)/8, $3916 // bitrev[799] = 3916
DATA bitrev_size4096_radix4<>+0x1900(SB)/8, $140  // bitrev[800] = 140
DATA bitrev_size4096_radix4<>+0x1908(SB)/8, $1164 // bitrev[801] = 1164
DATA bitrev_size4096_radix4<>+0x1910(SB)/8, $2188 // bitrev[802] = 2188
DATA bitrev_size4096_radix4<>+0x1918(SB)/8, $3212 // bitrev[803] = 3212
DATA bitrev_size4096_radix4<>+0x1920(SB)/8, $396  // bitrev[804] = 396
DATA bitrev_size4096_radix4<>+0x1928(SB)/8, $1420 // bitrev[805] = 1420
DATA bitrev_size4096_radix4<>+0x1930(SB)/8, $2444 // bitrev[806] = 2444
DATA bitrev_size4096_radix4<>+0x1938(SB)/8, $3468 // bitrev[807] = 3468
DATA bitrev_size4096_radix4<>+0x1940(SB)/8, $652  // bitrev[808] = 652
DATA bitrev_size4096_radix4<>+0x1948(SB)/8, $1676 // bitrev[809] = 1676
DATA bitrev_size4096_radix4<>+0x1950(SB)/8, $2700 // bitrev[810] = 2700
DATA bitrev_size4096_radix4<>+0x1958(SB)/8, $3724 // bitrev[811] = 3724
DATA bitrev_size4096_radix4<>+0x1960(SB)/8, $908  // bitrev[812] = 908
DATA bitrev_size4096_radix4<>+0x1968(SB)/8, $1932 // bitrev[813] = 1932
DATA bitrev_size4096_radix4<>+0x1970(SB)/8, $2956 // bitrev[814] = 2956
DATA bitrev_size4096_radix4<>+0x1978(SB)/8, $3980 // bitrev[815] = 3980
DATA bitrev_size4096_radix4<>+0x1980(SB)/8, $204  // bitrev[816] = 204
DATA bitrev_size4096_radix4<>+0x1988(SB)/8, $1228 // bitrev[817] = 1228
DATA bitrev_size4096_radix4<>+0x1990(SB)/8, $2252 // bitrev[818] = 2252
DATA bitrev_size4096_radix4<>+0x1998(SB)/8, $3276 // bitrev[819] = 3276
DATA bitrev_size4096_radix4<>+0x19a0(SB)/8, $460  // bitrev[820] = 460
DATA bitrev_size4096_radix4<>+0x19a8(SB)/8, $1484 // bitrev[821] = 1484
DATA bitrev_size4096_radix4<>+0x19b0(SB)/8, $2508 // bitrev[822] = 2508
DATA bitrev_size4096_radix4<>+0x19b8(SB)/8, $3532 // bitrev[823] = 3532
DATA bitrev_size4096_radix4<>+0x19c0(SB)/8, $716  // bitrev[824] = 716
DATA bitrev_size4096_radix4<>+0x19c8(SB)/8, $1740 // bitrev[825] = 1740
DATA bitrev_size4096_radix4<>+0x19d0(SB)/8, $2764 // bitrev[826] = 2764
DATA bitrev_size4096_radix4<>+0x19d8(SB)/8, $3788 // bitrev[827] = 3788
DATA bitrev_size4096_radix4<>+0x19e0(SB)/8, $972  // bitrev[828] = 972
DATA bitrev_size4096_radix4<>+0x19e8(SB)/8, $1996 // bitrev[829] = 1996
DATA bitrev_size4096_radix4<>+0x19f0(SB)/8, $3020 // bitrev[830] = 3020
DATA bitrev_size4096_radix4<>+0x19f8(SB)/8, $4044 // bitrev[831] = 4044
DATA bitrev_size4096_radix4<>+0x1a00(SB)/8, $28   // bitrev[832] = 28
DATA bitrev_size4096_radix4<>+0x1a08(SB)/8, $1052 // bitrev[833] = 1052
DATA bitrev_size4096_radix4<>+0x1a10(SB)/8, $2076 // bitrev[834] = 2076
DATA bitrev_size4096_radix4<>+0x1a18(SB)/8, $3100 // bitrev[835] = 3100
DATA bitrev_size4096_radix4<>+0x1a20(SB)/8, $284  // bitrev[836] = 284
DATA bitrev_size4096_radix4<>+0x1a28(SB)/8, $1308 // bitrev[837] = 1308
DATA bitrev_size4096_radix4<>+0x1a30(SB)/8, $2332 // bitrev[838] = 2332
DATA bitrev_size4096_radix4<>+0x1a38(SB)/8, $3356 // bitrev[839] = 3356
DATA bitrev_size4096_radix4<>+0x1a40(SB)/8, $540  // bitrev[840] = 540
DATA bitrev_size4096_radix4<>+0x1a48(SB)/8, $1564 // bitrev[841] = 1564
DATA bitrev_size4096_radix4<>+0x1a50(SB)/8, $2588 // bitrev[842] = 2588
DATA bitrev_size4096_radix4<>+0x1a58(SB)/8, $3612 // bitrev[843] = 3612
DATA bitrev_size4096_radix4<>+0x1a60(SB)/8, $796  // bitrev[844] = 796
DATA bitrev_size4096_radix4<>+0x1a68(SB)/8, $1820 // bitrev[845] = 1820
DATA bitrev_size4096_radix4<>+0x1a70(SB)/8, $2844 // bitrev[846] = 2844
DATA bitrev_size4096_radix4<>+0x1a78(SB)/8, $3868 // bitrev[847] = 3868
DATA bitrev_size4096_radix4<>+0x1a80(SB)/8, $92   // bitrev[848] = 92
DATA bitrev_size4096_radix4<>+0x1a88(SB)/8, $1116 // bitrev[849] = 1116
DATA bitrev_size4096_radix4<>+0x1a90(SB)/8, $2140 // bitrev[850] = 2140
DATA bitrev_size4096_radix4<>+0x1a98(SB)/8, $3164 // bitrev[851] = 3164
DATA bitrev_size4096_radix4<>+0x1aa0(SB)/8, $348  // bitrev[852] = 348
DATA bitrev_size4096_radix4<>+0x1aa8(SB)/8, $1372 // bitrev[853] = 1372
DATA bitrev_size4096_radix4<>+0x1ab0(SB)/8, $2396 // bitrev[854] = 2396
DATA bitrev_size4096_radix4<>+0x1ab8(SB)/8, $3420 // bitrev[855] = 3420
DATA bitrev_size4096_radix4<>+0x1ac0(SB)/8, $604  // bitrev[856] = 604
DATA bitrev_size4096_radix4<>+0x1ac8(SB)/8, $1628 // bitrev[857] = 1628
DATA bitrev_size4096_radix4<>+0x1ad0(SB)/8, $2652 // bitrev[858] = 2652
DATA bitrev_size4096_radix4<>+0x1ad8(SB)/8, $3676 // bitrev[859] = 3676
DATA bitrev_size4096_radix4<>+0x1ae0(SB)/8, $860  // bitrev[860] = 860
DATA bitrev_size4096_radix4<>+0x1ae8(SB)/8, $1884 // bitrev[861] = 1884
DATA bitrev_size4096_radix4<>+0x1af0(SB)/8, $2908 // bitrev[862] = 2908
DATA bitrev_size4096_radix4<>+0x1af8(SB)/8, $3932 // bitrev[863] = 3932
DATA bitrev_size4096_radix4<>+0x1b00(SB)/8, $156  // bitrev[864] = 156
DATA bitrev_size4096_radix4<>+0x1b08(SB)/8, $1180 // bitrev[865] = 1180
DATA bitrev_size4096_radix4<>+0x1b10(SB)/8, $2204 // bitrev[866] = 2204
DATA bitrev_size4096_radix4<>+0x1b18(SB)/8, $3228 // bitrev[867] = 3228
DATA bitrev_size4096_radix4<>+0x1b20(SB)/8, $412  // bitrev[868] = 412
DATA bitrev_size4096_radix4<>+0x1b28(SB)/8, $1436 // bitrev[869] = 1436
DATA bitrev_size4096_radix4<>+0x1b30(SB)/8, $2460 // bitrev[870] = 2460
DATA bitrev_size4096_radix4<>+0x1b38(SB)/8, $3484 // bitrev[871] = 3484
DATA bitrev_size4096_radix4<>+0x1b40(SB)/8, $668  // bitrev[872] = 668
DATA bitrev_size4096_radix4<>+0x1b48(SB)/8, $1692 // bitrev[873] = 1692
DATA bitrev_size4096_radix4<>+0x1b50(SB)/8, $2716 // bitrev[874] = 2716
DATA bitrev_size4096_radix4<>+0x1b58(SB)/8, $3740 // bitrev[875] = 3740
DATA bitrev_size4096_radix4<>+0x1b60(SB)/8, $924  // bitrev[876] = 924
DATA bitrev_size4096_radix4<>+0x1b68(SB)/8, $1948 // bitrev[877] = 1948
DATA bitrev_size4096_radix4<>+0x1b70(SB)/8, $2972 // bitrev[878] = 2972
DATA bitrev_size4096_radix4<>+0x1b78(SB)/8, $3996 // bitrev[879] = 3996
DATA bitrev_size4096_radix4<>+0x1b80(SB)/8, $220  // bitrev[880] = 220
DATA bitrev_size4096_radix4<>+0x1b88(SB)/8, $1244 // bitrev[881] = 1244
DATA bitrev_size4096_radix4<>+0x1b90(SB)/8, $2268 // bitrev[882] = 2268
DATA bitrev_size4096_radix4<>+0x1b98(SB)/8, $3292 // bitrev[883] = 3292
DATA bitrev_size4096_radix4<>+0x1ba0(SB)/8, $476  // bitrev[884] = 476
DATA bitrev_size4096_radix4<>+0x1ba8(SB)/8, $1500 // bitrev[885] = 1500
DATA bitrev_size4096_radix4<>+0x1bb0(SB)/8, $2524 // bitrev[886] = 2524
DATA bitrev_size4096_radix4<>+0x1bb8(SB)/8, $3548 // bitrev[887] = 3548
DATA bitrev_size4096_radix4<>+0x1bc0(SB)/8, $732  // bitrev[888] = 732
DATA bitrev_size4096_radix4<>+0x1bc8(SB)/8, $1756 // bitrev[889] = 1756
DATA bitrev_size4096_radix4<>+0x1bd0(SB)/8, $2780 // bitrev[890] = 2780
DATA bitrev_size4096_radix4<>+0x1bd8(SB)/8, $3804 // bitrev[891] = 3804
DATA bitrev_size4096_radix4<>+0x1be0(SB)/8, $988  // bitrev[892] = 988
DATA bitrev_size4096_radix4<>+0x1be8(SB)/8, $2012 // bitrev[893] = 2012
DATA bitrev_size4096_radix4<>+0x1bf0(SB)/8, $3036 // bitrev[894] = 3036
DATA bitrev_size4096_radix4<>+0x1bf8(SB)/8, $4060 // bitrev[895] = 4060
DATA bitrev_size4096_radix4<>+0x1c00(SB)/8, $44   // bitrev[896] = 44
DATA bitrev_size4096_radix4<>+0x1c08(SB)/8, $1068 // bitrev[897] = 1068
DATA bitrev_size4096_radix4<>+0x1c10(SB)/8, $2092 // bitrev[898] = 2092
DATA bitrev_size4096_radix4<>+0x1c18(SB)/8, $3116 // bitrev[899] = 3116
DATA bitrev_size4096_radix4<>+0x1c20(SB)/8, $300  // bitrev[900] = 300
DATA bitrev_size4096_radix4<>+0x1c28(SB)/8, $1324 // bitrev[901] = 1324
DATA bitrev_size4096_radix4<>+0x1c30(SB)/8, $2348 // bitrev[902] = 2348
DATA bitrev_size4096_radix4<>+0x1c38(SB)/8, $3372 // bitrev[903] = 3372
DATA bitrev_size4096_radix4<>+0x1c40(SB)/8, $556  // bitrev[904] = 556
DATA bitrev_size4096_radix4<>+0x1c48(SB)/8, $1580 // bitrev[905] = 1580
DATA bitrev_size4096_radix4<>+0x1c50(SB)/8, $2604 // bitrev[906] = 2604
DATA bitrev_size4096_radix4<>+0x1c58(SB)/8, $3628 // bitrev[907] = 3628
DATA bitrev_size4096_radix4<>+0x1c60(SB)/8, $812  // bitrev[908] = 812
DATA bitrev_size4096_radix4<>+0x1c68(SB)/8, $1836 // bitrev[909] = 1836
DATA bitrev_size4096_radix4<>+0x1c70(SB)/8, $2860 // bitrev[910] = 2860
DATA bitrev_size4096_radix4<>+0x1c78(SB)/8, $3884 // bitrev[911] = 3884
DATA bitrev_size4096_radix4<>+0x1c80(SB)/8, $108  // bitrev[912] = 108
DATA bitrev_size4096_radix4<>+0x1c88(SB)/8, $1132 // bitrev[913] = 1132
DATA bitrev_size4096_radix4<>+0x1c90(SB)/8, $2156 // bitrev[914] = 2156
DATA bitrev_size4096_radix4<>+0x1c98(SB)/8, $3180 // bitrev[915] = 3180
DATA bitrev_size4096_radix4<>+0x1ca0(SB)/8, $364  // bitrev[916] = 364
DATA bitrev_size4096_radix4<>+0x1ca8(SB)/8, $1388 // bitrev[917] = 1388
DATA bitrev_size4096_radix4<>+0x1cb0(SB)/8, $2412 // bitrev[918] = 2412
DATA bitrev_size4096_radix4<>+0x1cb8(SB)/8, $3436 // bitrev[919] = 3436
DATA bitrev_size4096_radix4<>+0x1cc0(SB)/8, $620  // bitrev[920] = 620
DATA bitrev_size4096_radix4<>+0x1cc8(SB)/8, $1644 // bitrev[921] = 1644
DATA bitrev_size4096_radix4<>+0x1cd0(SB)/8, $2668 // bitrev[922] = 2668
DATA bitrev_size4096_radix4<>+0x1cd8(SB)/8, $3692 // bitrev[923] = 3692
DATA bitrev_size4096_radix4<>+0x1ce0(SB)/8, $876  // bitrev[924] = 876
DATA bitrev_size4096_radix4<>+0x1ce8(SB)/8, $1900 // bitrev[925] = 1900
DATA bitrev_size4096_radix4<>+0x1cf0(SB)/8, $2924 // bitrev[926] = 2924
DATA bitrev_size4096_radix4<>+0x1cf8(SB)/8, $3948 // bitrev[927] = 3948
DATA bitrev_size4096_radix4<>+0x1d00(SB)/8, $172  // bitrev[928] = 172
DATA bitrev_size4096_radix4<>+0x1d08(SB)/8, $1196 // bitrev[929] = 1196
DATA bitrev_size4096_radix4<>+0x1d10(SB)/8, $2220 // bitrev[930] = 2220
DATA bitrev_size4096_radix4<>+0x1d18(SB)/8, $3244 // bitrev[931] = 3244
DATA bitrev_size4096_radix4<>+0x1d20(SB)/8, $428  // bitrev[932] = 428
DATA bitrev_size4096_radix4<>+0x1d28(SB)/8, $1452 // bitrev[933] = 1452
DATA bitrev_size4096_radix4<>+0x1d30(SB)/8, $2476 // bitrev[934] = 2476
DATA bitrev_size4096_radix4<>+0x1d38(SB)/8, $3500 // bitrev[935] = 3500
DATA bitrev_size4096_radix4<>+0x1d40(SB)/8, $684  // bitrev[936] = 684
DATA bitrev_size4096_radix4<>+0x1d48(SB)/8, $1708 // bitrev[937] = 1708
DATA bitrev_size4096_radix4<>+0x1d50(SB)/8, $2732 // bitrev[938] = 2732
DATA bitrev_size4096_radix4<>+0x1d58(SB)/8, $3756 // bitrev[939] = 3756
DATA bitrev_size4096_radix4<>+0x1d60(SB)/8, $940  // bitrev[940] = 940
DATA bitrev_size4096_radix4<>+0x1d68(SB)/8, $1964 // bitrev[941] = 1964
DATA bitrev_size4096_radix4<>+0x1d70(SB)/8, $2988 // bitrev[942] = 2988
DATA bitrev_size4096_radix4<>+0x1d78(SB)/8, $4012 // bitrev[943] = 4012
DATA bitrev_size4096_radix4<>+0x1d80(SB)/8, $236  // bitrev[944] = 236
DATA bitrev_size4096_radix4<>+0x1d88(SB)/8, $1260 // bitrev[945] = 1260
DATA bitrev_size4096_radix4<>+0x1d90(SB)/8, $2284 // bitrev[946] = 2284
DATA bitrev_size4096_radix4<>+0x1d98(SB)/8, $3308 // bitrev[947] = 3308
DATA bitrev_size4096_radix4<>+0x1da0(SB)/8, $492  // bitrev[948] = 492
DATA bitrev_size4096_radix4<>+0x1da8(SB)/8, $1516 // bitrev[949] = 1516
DATA bitrev_size4096_radix4<>+0x1db0(SB)/8, $2540 // bitrev[950] = 2540
DATA bitrev_size4096_radix4<>+0x1db8(SB)/8, $3564 // bitrev[951] = 3564
DATA bitrev_size4096_radix4<>+0x1dc0(SB)/8, $748  // bitrev[952] = 748
DATA bitrev_size4096_radix4<>+0x1dc8(SB)/8, $1772 // bitrev[953] = 1772
DATA bitrev_size4096_radix4<>+0x1dd0(SB)/8, $2796 // bitrev[954] = 2796
DATA bitrev_size4096_radix4<>+0x1dd8(SB)/8, $3820 // bitrev[955] = 3820
DATA bitrev_size4096_radix4<>+0x1de0(SB)/8, $1004 // bitrev[956] = 1004
DATA bitrev_size4096_radix4<>+0x1de8(SB)/8, $2028 // bitrev[957] = 2028
DATA bitrev_size4096_radix4<>+0x1df0(SB)/8, $3052 // bitrev[958] = 3052
DATA bitrev_size4096_radix4<>+0x1df8(SB)/8, $4076 // bitrev[959] = 4076
DATA bitrev_size4096_radix4<>+0x1e00(SB)/8, $60   // bitrev[960] = 60
DATA bitrev_size4096_radix4<>+0x1e08(SB)/8, $1084 // bitrev[961] = 1084
DATA bitrev_size4096_radix4<>+0x1e10(SB)/8, $2108 // bitrev[962] = 2108
DATA bitrev_size4096_radix4<>+0x1e18(SB)/8, $3132 // bitrev[963] = 3132
DATA bitrev_size4096_radix4<>+0x1e20(SB)/8, $316  // bitrev[964] = 316
DATA bitrev_size4096_radix4<>+0x1e28(SB)/8, $1340 // bitrev[965] = 1340
DATA bitrev_size4096_radix4<>+0x1e30(SB)/8, $2364 // bitrev[966] = 2364
DATA bitrev_size4096_radix4<>+0x1e38(SB)/8, $3388 // bitrev[967] = 3388
DATA bitrev_size4096_radix4<>+0x1e40(SB)/8, $572  // bitrev[968] = 572
DATA bitrev_size4096_radix4<>+0x1e48(SB)/8, $1596 // bitrev[969] = 1596
DATA bitrev_size4096_radix4<>+0x1e50(SB)/8, $2620 // bitrev[970] = 2620
DATA bitrev_size4096_radix4<>+0x1e58(SB)/8, $3644 // bitrev[971] = 3644
DATA bitrev_size4096_radix4<>+0x1e60(SB)/8, $828  // bitrev[972] = 828
DATA bitrev_size4096_radix4<>+0x1e68(SB)/8, $1852 // bitrev[973] = 1852
DATA bitrev_size4096_radix4<>+0x1e70(SB)/8, $2876 // bitrev[974] = 2876
DATA bitrev_size4096_radix4<>+0x1e78(SB)/8, $3900 // bitrev[975] = 3900
DATA bitrev_size4096_radix4<>+0x1e80(SB)/8, $124  // bitrev[976] = 124
DATA bitrev_size4096_radix4<>+0x1e88(SB)/8, $1148 // bitrev[977] = 1148
DATA bitrev_size4096_radix4<>+0x1e90(SB)/8, $2172 // bitrev[978] = 2172
DATA bitrev_size4096_radix4<>+0x1e98(SB)/8, $3196 // bitrev[979] = 3196
DATA bitrev_size4096_radix4<>+0x1ea0(SB)/8, $380  // bitrev[980] = 380
DATA bitrev_size4096_radix4<>+0x1ea8(SB)/8, $1404 // bitrev[981] = 1404
DATA bitrev_size4096_radix4<>+0x1eb0(SB)/8, $2428 // bitrev[982] = 2428
DATA bitrev_size4096_radix4<>+0x1eb8(SB)/8, $3452 // bitrev[983] = 3452
DATA bitrev_size4096_radix4<>+0x1ec0(SB)/8, $636  // bitrev[984] = 636
DATA bitrev_size4096_radix4<>+0x1ec8(SB)/8, $1660 // bitrev[985] = 1660
DATA bitrev_size4096_radix4<>+0x1ed0(SB)/8, $2684 // bitrev[986] = 2684
DATA bitrev_size4096_radix4<>+0x1ed8(SB)/8, $3708 // bitrev[987] = 3708
DATA bitrev_size4096_radix4<>+0x1ee0(SB)/8, $892  // bitrev[988] = 892
DATA bitrev_size4096_radix4<>+0x1ee8(SB)/8, $1916 // bitrev[989] = 1916
DATA bitrev_size4096_radix4<>+0x1ef0(SB)/8, $2940 // bitrev[990] = 2940
DATA bitrev_size4096_radix4<>+0x1ef8(SB)/8, $3964 // bitrev[991] = 3964
DATA bitrev_size4096_radix4<>+0x1f00(SB)/8, $188  // bitrev[992] = 188
DATA bitrev_size4096_radix4<>+0x1f08(SB)/8, $1212 // bitrev[993] = 1212
DATA bitrev_size4096_radix4<>+0x1f10(SB)/8, $2236 // bitrev[994] = 2236
DATA bitrev_size4096_radix4<>+0x1f18(SB)/8, $3260 // bitrev[995] = 3260
DATA bitrev_size4096_radix4<>+0x1f20(SB)/8, $444  // bitrev[996] = 444
DATA bitrev_size4096_radix4<>+0x1f28(SB)/8, $1468 // bitrev[997] = 1468
DATA bitrev_size4096_radix4<>+0x1f30(SB)/8, $2492 // bitrev[998] = 2492
DATA bitrev_size4096_radix4<>+0x1f38(SB)/8, $3516 // bitrev[999] = 3516
DATA bitrev_size4096_radix4<>+0x1f40(SB)/8, $700  // bitrev[1000] = 700
DATA bitrev_size4096_radix4<>+0x1f48(SB)/8, $1724 // bitrev[1001] = 1724
DATA bitrev_size4096_radix4<>+0x1f50(SB)/8, $2748 // bitrev[1002] = 2748
DATA bitrev_size4096_radix4<>+0x1f58(SB)/8, $3772 // bitrev[1003] = 3772
DATA bitrev_size4096_radix4<>+0x1f60(SB)/8, $956  // bitrev[1004] = 956
DATA bitrev_size4096_radix4<>+0x1f68(SB)/8, $1980 // bitrev[1005] = 1980
DATA bitrev_size4096_radix4<>+0x1f70(SB)/8, $3004 // bitrev[1006] = 3004
DATA bitrev_size4096_radix4<>+0x1f78(SB)/8, $4028 // bitrev[1007] = 4028
DATA bitrev_size4096_radix4<>+0x1f80(SB)/8, $252  // bitrev[1008] = 252
DATA bitrev_size4096_radix4<>+0x1f88(SB)/8, $1276 // bitrev[1009] = 1276
DATA bitrev_size4096_radix4<>+0x1f90(SB)/8, $2300 // bitrev[1010] = 2300
DATA bitrev_size4096_radix4<>+0x1f98(SB)/8, $3324 // bitrev[1011] = 3324
DATA bitrev_size4096_radix4<>+0x1fa0(SB)/8, $508  // bitrev[1012] = 508
DATA bitrev_size4096_radix4<>+0x1fa8(SB)/8, $1532 // bitrev[1013] = 1532
DATA bitrev_size4096_radix4<>+0x1fb0(SB)/8, $2556 // bitrev[1014] = 2556
DATA bitrev_size4096_radix4<>+0x1fb8(SB)/8, $3580 // bitrev[1015] = 3580
DATA bitrev_size4096_radix4<>+0x1fc0(SB)/8, $764  // bitrev[1016] = 764
DATA bitrev_size4096_radix4<>+0x1fc8(SB)/8, $1788 // bitrev[1017] = 1788
DATA bitrev_size4096_radix4<>+0x1fd0(SB)/8, $2812 // bitrev[1018] = 2812
DATA bitrev_size4096_radix4<>+0x1fd8(SB)/8, $3836 // bitrev[1019] = 3836
DATA bitrev_size4096_radix4<>+0x1fe0(SB)/8, $1020 // bitrev[1020] = 1020
DATA bitrev_size4096_radix4<>+0x1fe8(SB)/8, $2044 // bitrev[1021] = 2044
DATA bitrev_size4096_radix4<>+0x1ff0(SB)/8, $3068 // bitrev[1022] = 3068
DATA bitrev_size4096_radix4<>+0x1ff8(SB)/8, $4092 // bitrev[1023] = 4092
DATA bitrev_size4096_radix4<>+0x2000(SB)/8, $1    // bitrev[1024] = 1
DATA bitrev_size4096_radix4<>+0x2008(SB)/8, $1025 // bitrev[1025] = 1025
DATA bitrev_size4096_radix4<>+0x2010(SB)/8, $2049 // bitrev[1026] = 2049
DATA bitrev_size4096_radix4<>+0x2018(SB)/8, $3073 // bitrev[1027] = 3073
DATA bitrev_size4096_radix4<>+0x2020(SB)/8, $257  // bitrev[1028] = 257
DATA bitrev_size4096_radix4<>+0x2028(SB)/8, $1281 // bitrev[1029] = 1281
DATA bitrev_size4096_radix4<>+0x2030(SB)/8, $2305 // bitrev[1030] = 2305
DATA bitrev_size4096_radix4<>+0x2038(SB)/8, $3329 // bitrev[1031] = 3329
DATA bitrev_size4096_radix4<>+0x2040(SB)/8, $513  // bitrev[1032] = 513
DATA bitrev_size4096_radix4<>+0x2048(SB)/8, $1537 // bitrev[1033] = 1537
DATA bitrev_size4096_radix4<>+0x2050(SB)/8, $2561 // bitrev[1034] = 2561
DATA bitrev_size4096_radix4<>+0x2058(SB)/8, $3585 // bitrev[1035] = 3585
DATA bitrev_size4096_radix4<>+0x2060(SB)/8, $769  // bitrev[1036] = 769
DATA bitrev_size4096_radix4<>+0x2068(SB)/8, $1793 // bitrev[1037] = 1793
DATA bitrev_size4096_radix4<>+0x2070(SB)/8, $2817 // bitrev[1038] = 2817
DATA bitrev_size4096_radix4<>+0x2078(SB)/8, $3841 // bitrev[1039] = 3841
DATA bitrev_size4096_radix4<>+0x2080(SB)/8, $65   // bitrev[1040] = 65
DATA bitrev_size4096_radix4<>+0x2088(SB)/8, $1089 // bitrev[1041] = 1089
DATA bitrev_size4096_radix4<>+0x2090(SB)/8, $2113 // bitrev[1042] = 2113
DATA bitrev_size4096_radix4<>+0x2098(SB)/8, $3137 // bitrev[1043] = 3137
DATA bitrev_size4096_radix4<>+0x20a0(SB)/8, $321  // bitrev[1044] = 321
DATA bitrev_size4096_radix4<>+0x20a8(SB)/8, $1345 // bitrev[1045] = 1345
DATA bitrev_size4096_radix4<>+0x20b0(SB)/8, $2369 // bitrev[1046] = 2369
DATA bitrev_size4096_radix4<>+0x20b8(SB)/8, $3393 // bitrev[1047] = 3393
DATA bitrev_size4096_radix4<>+0x20c0(SB)/8, $577  // bitrev[1048] = 577
DATA bitrev_size4096_radix4<>+0x20c8(SB)/8, $1601 // bitrev[1049] = 1601
DATA bitrev_size4096_radix4<>+0x20d0(SB)/8, $2625 // bitrev[1050] = 2625
DATA bitrev_size4096_radix4<>+0x20d8(SB)/8, $3649 // bitrev[1051] = 3649
DATA bitrev_size4096_radix4<>+0x20e0(SB)/8, $833  // bitrev[1052] = 833
DATA bitrev_size4096_radix4<>+0x20e8(SB)/8, $1857 // bitrev[1053] = 1857
DATA bitrev_size4096_radix4<>+0x20f0(SB)/8, $2881 // bitrev[1054] = 2881
DATA bitrev_size4096_radix4<>+0x20f8(SB)/8, $3905 // bitrev[1055] = 3905
DATA bitrev_size4096_radix4<>+0x2100(SB)/8, $129  // bitrev[1056] = 129
DATA bitrev_size4096_radix4<>+0x2108(SB)/8, $1153 // bitrev[1057] = 1153
DATA bitrev_size4096_radix4<>+0x2110(SB)/8, $2177 // bitrev[1058] = 2177
DATA bitrev_size4096_radix4<>+0x2118(SB)/8, $3201 // bitrev[1059] = 3201
DATA bitrev_size4096_radix4<>+0x2120(SB)/8, $385  // bitrev[1060] = 385
DATA bitrev_size4096_radix4<>+0x2128(SB)/8, $1409 // bitrev[1061] = 1409
DATA bitrev_size4096_radix4<>+0x2130(SB)/8, $2433 // bitrev[1062] = 2433
DATA bitrev_size4096_radix4<>+0x2138(SB)/8, $3457 // bitrev[1063] = 3457
DATA bitrev_size4096_radix4<>+0x2140(SB)/8, $641  // bitrev[1064] = 641
DATA bitrev_size4096_radix4<>+0x2148(SB)/8, $1665 // bitrev[1065] = 1665
DATA bitrev_size4096_radix4<>+0x2150(SB)/8, $2689 // bitrev[1066] = 2689
DATA bitrev_size4096_radix4<>+0x2158(SB)/8, $3713 // bitrev[1067] = 3713
DATA bitrev_size4096_radix4<>+0x2160(SB)/8, $897  // bitrev[1068] = 897
DATA bitrev_size4096_radix4<>+0x2168(SB)/8, $1921 // bitrev[1069] = 1921
DATA bitrev_size4096_radix4<>+0x2170(SB)/8, $2945 // bitrev[1070] = 2945
DATA bitrev_size4096_radix4<>+0x2178(SB)/8, $3969 // bitrev[1071] = 3969
DATA bitrev_size4096_radix4<>+0x2180(SB)/8, $193  // bitrev[1072] = 193
DATA bitrev_size4096_radix4<>+0x2188(SB)/8, $1217 // bitrev[1073] = 1217
DATA bitrev_size4096_radix4<>+0x2190(SB)/8, $2241 // bitrev[1074] = 2241
DATA bitrev_size4096_radix4<>+0x2198(SB)/8, $3265 // bitrev[1075] = 3265
DATA bitrev_size4096_radix4<>+0x21a0(SB)/8, $449  // bitrev[1076] = 449
DATA bitrev_size4096_radix4<>+0x21a8(SB)/8, $1473 // bitrev[1077] = 1473
DATA bitrev_size4096_radix4<>+0x21b0(SB)/8, $2497 // bitrev[1078] = 2497
DATA bitrev_size4096_radix4<>+0x21b8(SB)/8, $3521 // bitrev[1079] = 3521
DATA bitrev_size4096_radix4<>+0x21c0(SB)/8, $705  // bitrev[1080] = 705
DATA bitrev_size4096_radix4<>+0x21c8(SB)/8, $1729 // bitrev[1081] = 1729
DATA bitrev_size4096_radix4<>+0x21d0(SB)/8, $2753 // bitrev[1082] = 2753
DATA bitrev_size4096_radix4<>+0x21d8(SB)/8, $3777 // bitrev[1083] = 3777
DATA bitrev_size4096_radix4<>+0x21e0(SB)/8, $961  // bitrev[1084] = 961
DATA bitrev_size4096_radix4<>+0x21e8(SB)/8, $1985 // bitrev[1085] = 1985
DATA bitrev_size4096_radix4<>+0x21f0(SB)/8, $3009 // bitrev[1086] = 3009
DATA bitrev_size4096_radix4<>+0x21f8(SB)/8, $4033 // bitrev[1087] = 4033
DATA bitrev_size4096_radix4<>+0x2200(SB)/8, $17   // bitrev[1088] = 17
DATA bitrev_size4096_radix4<>+0x2208(SB)/8, $1041 // bitrev[1089] = 1041
DATA bitrev_size4096_radix4<>+0x2210(SB)/8, $2065 // bitrev[1090] = 2065
DATA bitrev_size4096_radix4<>+0x2218(SB)/8, $3089 // bitrev[1091] = 3089
DATA bitrev_size4096_radix4<>+0x2220(SB)/8, $273  // bitrev[1092] = 273
DATA bitrev_size4096_radix4<>+0x2228(SB)/8, $1297 // bitrev[1093] = 1297
DATA bitrev_size4096_radix4<>+0x2230(SB)/8, $2321 // bitrev[1094] = 2321
DATA bitrev_size4096_radix4<>+0x2238(SB)/8, $3345 // bitrev[1095] = 3345
DATA bitrev_size4096_radix4<>+0x2240(SB)/8, $529  // bitrev[1096] = 529
DATA bitrev_size4096_radix4<>+0x2248(SB)/8, $1553 // bitrev[1097] = 1553
DATA bitrev_size4096_radix4<>+0x2250(SB)/8, $2577 // bitrev[1098] = 2577
DATA bitrev_size4096_radix4<>+0x2258(SB)/8, $3601 // bitrev[1099] = 3601
DATA bitrev_size4096_radix4<>+0x2260(SB)/8, $785  // bitrev[1100] = 785
DATA bitrev_size4096_radix4<>+0x2268(SB)/8, $1809 // bitrev[1101] = 1809
DATA bitrev_size4096_radix4<>+0x2270(SB)/8, $2833 // bitrev[1102] = 2833
DATA bitrev_size4096_radix4<>+0x2278(SB)/8, $3857 // bitrev[1103] = 3857
DATA bitrev_size4096_radix4<>+0x2280(SB)/8, $81   // bitrev[1104] = 81
DATA bitrev_size4096_radix4<>+0x2288(SB)/8, $1105 // bitrev[1105] = 1105
DATA bitrev_size4096_radix4<>+0x2290(SB)/8, $2129 // bitrev[1106] = 2129
DATA bitrev_size4096_radix4<>+0x2298(SB)/8, $3153 // bitrev[1107] = 3153
DATA bitrev_size4096_radix4<>+0x22a0(SB)/8, $337  // bitrev[1108] = 337
DATA bitrev_size4096_radix4<>+0x22a8(SB)/8, $1361 // bitrev[1109] = 1361
DATA bitrev_size4096_radix4<>+0x22b0(SB)/8, $2385 // bitrev[1110] = 2385
DATA bitrev_size4096_radix4<>+0x22b8(SB)/8, $3409 // bitrev[1111] = 3409
DATA bitrev_size4096_radix4<>+0x22c0(SB)/8, $593  // bitrev[1112] = 593
DATA bitrev_size4096_radix4<>+0x22c8(SB)/8, $1617 // bitrev[1113] = 1617
DATA bitrev_size4096_radix4<>+0x22d0(SB)/8, $2641 // bitrev[1114] = 2641
DATA bitrev_size4096_radix4<>+0x22d8(SB)/8, $3665 // bitrev[1115] = 3665
DATA bitrev_size4096_radix4<>+0x22e0(SB)/8, $849  // bitrev[1116] = 849
DATA bitrev_size4096_radix4<>+0x22e8(SB)/8, $1873 // bitrev[1117] = 1873
DATA bitrev_size4096_radix4<>+0x22f0(SB)/8, $2897 // bitrev[1118] = 2897
DATA bitrev_size4096_radix4<>+0x22f8(SB)/8, $3921 // bitrev[1119] = 3921
DATA bitrev_size4096_radix4<>+0x2300(SB)/8, $145  // bitrev[1120] = 145
DATA bitrev_size4096_radix4<>+0x2308(SB)/8, $1169 // bitrev[1121] = 1169
DATA bitrev_size4096_radix4<>+0x2310(SB)/8, $2193 // bitrev[1122] = 2193
DATA bitrev_size4096_radix4<>+0x2318(SB)/8, $3217 // bitrev[1123] = 3217
DATA bitrev_size4096_radix4<>+0x2320(SB)/8, $401  // bitrev[1124] = 401
DATA bitrev_size4096_radix4<>+0x2328(SB)/8, $1425 // bitrev[1125] = 1425
DATA bitrev_size4096_radix4<>+0x2330(SB)/8, $2449 // bitrev[1126] = 2449
DATA bitrev_size4096_radix4<>+0x2338(SB)/8, $3473 // bitrev[1127] = 3473
DATA bitrev_size4096_radix4<>+0x2340(SB)/8, $657  // bitrev[1128] = 657
DATA bitrev_size4096_radix4<>+0x2348(SB)/8, $1681 // bitrev[1129] = 1681
DATA bitrev_size4096_radix4<>+0x2350(SB)/8, $2705 // bitrev[1130] = 2705
DATA bitrev_size4096_radix4<>+0x2358(SB)/8, $3729 // bitrev[1131] = 3729
DATA bitrev_size4096_radix4<>+0x2360(SB)/8, $913  // bitrev[1132] = 913
DATA bitrev_size4096_radix4<>+0x2368(SB)/8, $1937 // bitrev[1133] = 1937
DATA bitrev_size4096_radix4<>+0x2370(SB)/8, $2961 // bitrev[1134] = 2961
DATA bitrev_size4096_radix4<>+0x2378(SB)/8, $3985 // bitrev[1135] = 3985
DATA bitrev_size4096_radix4<>+0x2380(SB)/8, $209  // bitrev[1136] = 209
DATA bitrev_size4096_radix4<>+0x2388(SB)/8, $1233 // bitrev[1137] = 1233
DATA bitrev_size4096_radix4<>+0x2390(SB)/8, $2257 // bitrev[1138] = 2257
DATA bitrev_size4096_radix4<>+0x2398(SB)/8, $3281 // bitrev[1139] = 3281
DATA bitrev_size4096_radix4<>+0x23a0(SB)/8, $465  // bitrev[1140] = 465
DATA bitrev_size4096_radix4<>+0x23a8(SB)/8, $1489 // bitrev[1141] = 1489
DATA bitrev_size4096_radix4<>+0x23b0(SB)/8, $2513 // bitrev[1142] = 2513
DATA bitrev_size4096_radix4<>+0x23b8(SB)/8, $3537 // bitrev[1143] = 3537
DATA bitrev_size4096_radix4<>+0x23c0(SB)/8, $721  // bitrev[1144] = 721
DATA bitrev_size4096_radix4<>+0x23c8(SB)/8, $1745 // bitrev[1145] = 1745
DATA bitrev_size4096_radix4<>+0x23d0(SB)/8, $2769 // bitrev[1146] = 2769
DATA bitrev_size4096_radix4<>+0x23d8(SB)/8, $3793 // bitrev[1147] = 3793
DATA bitrev_size4096_radix4<>+0x23e0(SB)/8, $977  // bitrev[1148] = 977
DATA bitrev_size4096_radix4<>+0x23e8(SB)/8, $2001 // bitrev[1149] = 2001
DATA bitrev_size4096_radix4<>+0x23f0(SB)/8, $3025 // bitrev[1150] = 3025
DATA bitrev_size4096_radix4<>+0x23f8(SB)/8, $4049 // bitrev[1151] = 4049
DATA bitrev_size4096_radix4<>+0x2400(SB)/8, $33   // bitrev[1152] = 33
DATA bitrev_size4096_radix4<>+0x2408(SB)/8, $1057 // bitrev[1153] = 1057
DATA bitrev_size4096_radix4<>+0x2410(SB)/8, $2081 // bitrev[1154] = 2081
DATA bitrev_size4096_radix4<>+0x2418(SB)/8, $3105 // bitrev[1155] = 3105
DATA bitrev_size4096_radix4<>+0x2420(SB)/8, $289  // bitrev[1156] = 289
DATA bitrev_size4096_radix4<>+0x2428(SB)/8, $1313 // bitrev[1157] = 1313
DATA bitrev_size4096_radix4<>+0x2430(SB)/8, $2337 // bitrev[1158] = 2337
DATA bitrev_size4096_radix4<>+0x2438(SB)/8, $3361 // bitrev[1159] = 3361
DATA bitrev_size4096_radix4<>+0x2440(SB)/8, $545  // bitrev[1160] = 545
DATA bitrev_size4096_radix4<>+0x2448(SB)/8, $1569 // bitrev[1161] = 1569
DATA bitrev_size4096_radix4<>+0x2450(SB)/8, $2593 // bitrev[1162] = 2593
DATA bitrev_size4096_radix4<>+0x2458(SB)/8, $3617 // bitrev[1163] = 3617
DATA bitrev_size4096_radix4<>+0x2460(SB)/8, $801  // bitrev[1164] = 801
DATA bitrev_size4096_radix4<>+0x2468(SB)/8, $1825 // bitrev[1165] = 1825
DATA bitrev_size4096_radix4<>+0x2470(SB)/8, $2849 // bitrev[1166] = 2849
DATA bitrev_size4096_radix4<>+0x2478(SB)/8, $3873 // bitrev[1167] = 3873
DATA bitrev_size4096_radix4<>+0x2480(SB)/8, $97   // bitrev[1168] = 97
DATA bitrev_size4096_radix4<>+0x2488(SB)/8, $1121 // bitrev[1169] = 1121
DATA bitrev_size4096_radix4<>+0x2490(SB)/8, $2145 // bitrev[1170] = 2145
DATA bitrev_size4096_radix4<>+0x2498(SB)/8, $3169 // bitrev[1171] = 3169
DATA bitrev_size4096_radix4<>+0x24a0(SB)/8, $353  // bitrev[1172] = 353
DATA bitrev_size4096_radix4<>+0x24a8(SB)/8, $1377 // bitrev[1173] = 1377
DATA bitrev_size4096_radix4<>+0x24b0(SB)/8, $2401 // bitrev[1174] = 2401
DATA bitrev_size4096_radix4<>+0x24b8(SB)/8, $3425 // bitrev[1175] = 3425
DATA bitrev_size4096_radix4<>+0x24c0(SB)/8, $609  // bitrev[1176] = 609
DATA bitrev_size4096_radix4<>+0x24c8(SB)/8, $1633 // bitrev[1177] = 1633
DATA bitrev_size4096_radix4<>+0x24d0(SB)/8, $2657 // bitrev[1178] = 2657
DATA bitrev_size4096_radix4<>+0x24d8(SB)/8, $3681 // bitrev[1179] = 3681
DATA bitrev_size4096_radix4<>+0x24e0(SB)/8, $865  // bitrev[1180] = 865
DATA bitrev_size4096_radix4<>+0x24e8(SB)/8, $1889 // bitrev[1181] = 1889
DATA bitrev_size4096_radix4<>+0x24f0(SB)/8, $2913 // bitrev[1182] = 2913
DATA bitrev_size4096_radix4<>+0x24f8(SB)/8, $3937 // bitrev[1183] = 3937
DATA bitrev_size4096_radix4<>+0x2500(SB)/8, $161  // bitrev[1184] = 161
DATA bitrev_size4096_radix4<>+0x2508(SB)/8, $1185 // bitrev[1185] = 1185
DATA bitrev_size4096_radix4<>+0x2510(SB)/8, $2209 // bitrev[1186] = 2209
DATA bitrev_size4096_radix4<>+0x2518(SB)/8, $3233 // bitrev[1187] = 3233
DATA bitrev_size4096_radix4<>+0x2520(SB)/8, $417  // bitrev[1188] = 417
DATA bitrev_size4096_radix4<>+0x2528(SB)/8, $1441 // bitrev[1189] = 1441
DATA bitrev_size4096_radix4<>+0x2530(SB)/8, $2465 // bitrev[1190] = 2465
DATA bitrev_size4096_radix4<>+0x2538(SB)/8, $3489 // bitrev[1191] = 3489
DATA bitrev_size4096_radix4<>+0x2540(SB)/8, $673  // bitrev[1192] = 673
DATA bitrev_size4096_radix4<>+0x2548(SB)/8, $1697 // bitrev[1193] = 1697
DATA bitrev_size4096_radix4<>+0x2550(SB)/8, $2721 // bitrev[1194] = 2721
DATA bitrev_size4096_radix4<>+0x2558(SB)/8, $3745 // bitrev[1195] = 3745
DATA bitrev_size4096_radix4<>+0x2560(SB)/8, $929  // bitrev[1196] = 929
DATA bitrev_size4096_radix4<>+0x2568(SB)/8, $1953 // bitrev[1197] = 1953
DATA bitrev_size4096_radix4<>+0x2570(SB)/8, $2977 // bitrev[1198] = 2977
DATA bitrev_size4096_radix4<>+0x2578(SB)/8, $4001 // bitrev[1199] = 4001
DATA bitrev_size4096_radix4<>+0x2580(SB)/8, $225  // bitrev[1200] = 225
DATA bitrev_size4096_radix4<>+0x2588(SB)/8, $1249 // bitrev[1201] = 1249
DATA bitrev_size4096_radix4<>+0x2590(SB)/8, $2273 // bitrev[1202] = 2273
DATA bitrev_size4096_radix4<>+0x2598(SB)/8, $3297 // bitrev[1203] = 3297
DATA bitrev_size4096_radix4<>+0x25a0(SB)/8, $481  // bitrev[1204] = 481
DATA bitrev_size4096_radix4<>+0x25a8(SB)/8, $1505 // bitrev[1205] = 1505
DATA bitrev_size4096_radix4<>+0x25b0(SB)/8, $2529 // bitrev[1206] = 2529
DATA bitrev_size4096_radix4<>+0x25b8(SB)/8, $3553 // bitrev[1207] = 3553
DATA bitrev_size4096_radix4<>+0x25c0(SB)/8, $737  // bitrev[1208] = 737
DATA bitrev_size4096_radix4<>+0x25c8(SB)/8, $1761 // bitrev[1209] = 1761
DATA bitrev_size4096_radix4<>+0x25d0(SB)/8, $2785 // bitrev[1210] = 2785
DATA bitrev_size4096_radix4<>+0x25d8(SB)/8, $3809 // bitrev[1211] = 3809
DATA bitrev_size4096_radix4<>+0x25e0(SB)/8, $993  // bitrev[1212] = 993
DATA bitrev_size4096_radix4<>+0x25e8(SB)/8, $2017 // bitrev[1213] = 2017
DATA bitrev_size4096_radix4<>+0x25f0(SB)/8, $3041 // bitrev[1214] = 3041
DATA bitrev_size4096_radix4<>+0x25f8(SB)/8, $4065 // bitrev[1215] = 4065
DATA bitrev_size4096_radix4<>+0x2600(SB)/8, $49   // bitrev[1216] = 49
DATA bitrev_size4096_radix4<>+0x2608(SB)/8, $1073 // bitrev[1217] = 1073
DATA bitrev_size4096_radix4<>+0x2610(SB)/8, $2097 // bitrev[1218] = 2097
DATA bitrev_size4096_radix4<>+0x2618(SB)/8, $3121 // bitrev[1219] = 3121
DATA bitrev_size4096_radix4<>+0x2620(SB)/8, $305  // bitrev[1220] = 305
DATA bitrev_size4096_radix4<>+0x2628(SB)/8, $1329 // bitrev[1221] = 1329
DATA bitrev_size4096_radix4<>+0x2630(SB)/8, $2353 // bitrev[1222] = 2353
DATA bitrev_size4096_radix4<>+0x2638(SB)/8, $3377 // bitrev[1223] = 3377
DATA bitrev_size4096_radix4<>+0x2640(SB)/8, $561  // bitrev[1224] = 561
DATA bitrev_size4096_radix4<>+0x2648(SB)/8, $1585 // bitrev[1225] = 1585
DATA bitrev_size4096_radix4<>+0x2650(SB)/8, $2609 // bitrev[1226] = 2609
DATA bitrev_size4096_radix4<>+0x2658(SB)/8, $3633 // bitrev[1227] = 3633
DATA bitrev_size4096_radix4<>+0x2660(SB)/8, $817  // bitrev[1228] = 817
DATA bitrev_size4096_radix4<>+0x2668(SB)/8, $1841 // bitrev[1229] = 1841
DATA bitrev_size4096_radix4<>+0x2670(SB)/8, $2865 // bitrev[1230] = 2865
DATA bitrev_size4096_radix4<>+0x2678(SB)/8, $3889 // bitrev[1231] = 3889
DATA bitrev_size4096_radix4<>+0x2680(SB)/8, $113  // bitrev[1232] = 113
DATA bitrev_size4096_radix4<>+0x2688(SB)/8, $1137 // bitrev[1233] = 1137
DATA bitrev_size4096_radix4<>+0x2690(SB)/8, $2161 // bitrev[1234] = 2161
DATA bitrev_size4096_radix4<>+0x2698(SB)/8, $3185 // bitrev[1235] = 3185
DATA bitrev_size4096_radix4<>+0x26a0(SB)/8, $369  // bitrev[1236] = 369
DATA bitrev_size4096_radix4<>+0x26a8(SB)/8, $1393 // bitrev[1237] = 1393
DATA bitrev_size4096_radix4<>+0x26b0(SB)/8, $2417 // bitrev[1238] = 2417
DATA bitrev_size4096_radix4<>+0x26b8(SB)/8, $3441 // bitrev[1239] = 3441
DATA bitrev_size4096_radix4<>+0x26c0(SB)/8, $625  // bitrev[1240] = 625
DATA bitrev_size4096_radix4<>+0x26c8(SB)/8, $1649 // bitrev[1241] = 1649
DATA bitrev_size4096_radix4<>+0x26d0(SB)/8, $2673 // bitrev[1242] = 2673
DATA bitrev_size4096_radix4<>+0x26d8(SB)/8, $3697 // bitrev[1243] = 3697
DATA bitrev_size4096_radix4<>+0x26e0(SB)/8, $881  // bitrev[1244] = 881
DATA bitrev_size4096_radix4<>+0x26e8(SB)/8, $1905 // bitrev[1245] = 1905
DATA bitrev_size4096_radix4<>+0x26f0(SB)/8, $2929 // bitrev[1246] = 2929
DATA bitrev_size4096_radix4<>+0x26f8(SB)/8, $3953 // bitrev[1247] = 3953
DATA bitrev_size4096_radix4<>+0x2700(SB)/8, $177  // bitrev[1248] = 177
DATA bitrev_size4096_radix4<>+0x2708(SB)/8, $1201 // bitrev[1249] = 1201
DATA bitrev_size4096_radix4<>+0x2710(SB)/8, $2225 // bitrev[1250] = 2225
DATA bitrev_size4096_radix4<>+0x2718(SB)/8, $3249 // bitrev[1251] = 3249
DATA bitrev_size4096_radix4<>+0x2720(SB)/8, $433  // bitrev[1252] = 433
DATA bitrev_size4096_radix4<>+0x2728(SB)/8, $1457 // bitrev[1253] = 1457
DATA bitrev_size4096_radix4<>+0x2730(SB)/8, $2481 // bitrev[1254] = 2481
DATA bitrev_size4096_radix4<>+0x2738(SB)/8, $3505 // bitrev[1255] = 3505
DATA bitrev_size4096_radix4<>+0x2740(SB)/8, $689  // bitrev[1256] = 689
DATA bitrev_size4096_radix4<>+0x2748(SB)/8, $1713 // bitrev[1257] = 1713
DATA bitrev_size4096_radix4<>+0x2750(SB)/8, $2737 // bitrev[1258] = 2737
DATA bitrev_size4096_radix4<>+0x2758(SB)/8, $3761 // bitrev[1259] = 3761
DATA bitrev_size4096_radix4<>+0x2760(SB)/8, $945  // bitrev[1260] = 945
DATA bitrev_size4096_radix4<>+0x2768(SB)/8, $1969 // bitrev[1261] = 1969
DATA bitrev_size4096_radix4<>+0x2770(SB)/8, $2993 // bitrev[1262] = 2993
DATA bitrev_size4096_radix4<>+0x2778(SB)/8, $4017 // bitrev[1263] = 4017
DATA bitrev_size4096_radix4<>+0x2780(SB)/8, $241  // bitrev[1264] = 241
DATA bitrev_size4096_radix4<>+0x2788(SB)/8, $1265 // bitrev[1265] = 1265
DATA bitrev_size4096_radix4<>+0x2790(SB)/8, $2289 // bitrev[1266] = 2289
DATA bitrev_size4096_radix4<>+0x2798(SB)/8, $3313 // bitrev[1267] = 3313
DATA bitrev_size4096_radix4<>+0x27a0(SB)/8, $497  // bitrev[1268] = 497
DATA bitrev_size4096_radix4<>+0x27a8(SB)/8, $1521 // bitrev[1269] = 1521
DATA bitrev_size4096_radix4<>+0x27b0(SB)/8, $2545 // bitrev[1270] = 2545
DATA bitrev_size4096_radix4<>+0x27b8(SB)/8, $3569 // bitrev[1271] = 3569
DATA bitrev_size4096_radix4<>+0x27c0(SB)/8, $753  // bitrev[1272] = 753
DATA bitrev_size4096_radix4<>+0x27c8(SB)/8, $1777 // bitrev[1273] = 1777
DATA bitrev_size4096_radix4<>+0x27d0(SB)/8, $2801 // bitrev[1274] = 2801
DATA bitrev_size4096_radix4<>+0x27d8(SB)/8, $3825 // bitrev[1275] = 3825
DATA bitrev_size4096_radix4<>+0x27e0(SB)/8, $1009 // bitrev[1276] = 1009
DATA bitrev_size4096_radix4<>+0x27e8(SB)/8, $2033 // bitrev[1277] = 2033
DATA bitrev_size4096_radix4<>+0x27f0(SB)/8, $3057 // bitrev[1278] = 3057
DATA bitrev_size4096_radix4<>+0x27f8(SB)/8, $4081 // bitrev[1279] = 4081
DATA bitrev_size4096_radix4<>+0x2800(SB)/8, $5    // bitrev[1280] = 5
DATA bitrev_size4096_radix4<>+0x2808(SB)/8, $1029 // bitrev[1281] = 1029
DATA bitrev_size4096_radix4<>+0x2810(SB)/8, $2053 // bitrev[1282] = 2053
DATA bitrev_size4096_radix4<>+0x2818(SB)/8, $3077 // bitrev[1283] = 3077
DATA bitrev_size4096_radix4<>+0x2820(SB)/8, $261  // bitrev[1284] = 261
DATA bitrev_size4096_radix4<>+0x2828(SB)/8, $1285 // bitrev[1285] = 1285
DATA bitrev_size4096_radix4<>+0x2830(SB)/8, $2309 // bitrev[1286] = 2309
DATA bitrev_size4096_radix4<>+0x2838(SB)/8, $3333 // bitrev[1287] = 3333
DATA bitrev_size4096_radix4<>+0x2840(SB)/8, $517  // bitrev[1288] = 517
DATA bitrev_size4096_radix4<>+0x2848(SB)/8, $1541 // bitrev[1289] = 1541
DATA bitrev_size4096_radix4<>+0x2850(SB)/8, $2565 // bitrev[1290] = 2565
DATA bitrev_size4096_radix4<>+0x2858(SB)/8, $3589 // bitrev[1291] = 3589
DATA bitrev_size4096_radix4<>+0x2860(SB)/8, $773  // bitrev[1292] = 773
DATA bitrev_size4096_radix4<>+0x2868(SB)/8, $1797 // bitrev[1293] = 1797
DATA bitrev_size4096_radix4<>+0x2870(SB)/8, $2821 // bitrev[1294] = 2821
DATA bitrev_size4096_radix4<>+0x2878(SB)/8, $3845 // bitrev[1295] = 3845
DATA bitrev_size4096_radix4<>+0x2880(SB)/8, $69   // bitrev[1296] = 69
DATA bitrev_size4096_radix4<>+0x2888(SB)/8, $1093 // bitrev[1297] = 1093
DATA bitrev_size4096_radix4<>+0x2890(SB)/8, $2117 // bitrev[1298] = 2117
DATA bitrev_size4096_radix4<>+0x2898(SB)/8, $3141 // bitrev[1299] = 3141
DATA bitrev_size4096_radix4<>+0x28a0(SB)/8, $325  // bitrev[1300] = 325
DATA bitrev_size4096_radix4<>+0x28a8(SB)/8, $1349 // bitrev[1301] = 1349
DATA bitrev_size4096_radix4<>+0x28b0(SB)/8, $2373 // bitrev[1302] = 2373
DATA bitrev_size4096_radix4<>+0x28b8(SB)/8, $3397 // bitrev[1303] = 3397
DATA bitrev_size4096_radix4<>+0x28c0(SB)/8, $581  // bitrev[1304] = 581
DATA bitrev_size4096_radix4<>+0x28c8(SB)/8, $1605 // bitrev[1305] = 1605
DATA bitrev_size4096_radix4<>+0x28d0(SB)/8, $2629 // bitrev[1306] = 2629
DATA bitrev_size4096_radix4<>+0x28d8(SB)/8, $3653 // bitrev[1307] = 3653
DATA bitrev_size4096_radix4<>+0x28e0(SB)/8, $837  // bitrev[1308] = 837
DATA bitrev_size4096_radix4<>+0x28e8(SB)/8, $1861 // bitrev[1309] = 1861
DATA bitrev_size4096_radix4<>+0x28f0(SB)/8, $2885 // bitrev[1310] = 2885
DATA bitrev_size4096_radix4<>+0x28f8(SB)/8, $3909 // bitrev[1311] = 3909
DATA bitrev_size4096_radix4<>+0x2900(SB)/8, $133  // bitrev[1312] = 133
DATA bitrev_size4096_radix4<>+0x2908(SB)/8, $1157 // bitrev[1313] = 1157
DATA bitrev_size4096_radix4<>+0x2910(SB)/8, $2181 // bitrev[1314] = 2181
DATA bitrev_size4096_radix4<>+0x2918(SB)/8, $3205 // bitrev[1315] = 3205
DATA bitrev_size4096_radix4<>+0x2920(SB)/8, $389  // bitrev[1316] = 389
DATA bitrev_size4096_radix4<>+0x2928(SB)/8, $1413 // bitrev[1317] = 1413
DATA bitrev_size4096_radix4<>+0x2930(SB)/8, $2437 // bitrev[1318] = 2437
DATA bitrev_size4096_radix4<>+0x2938(SB)/8, $3461 // bitrev[1319] = 3461
DATA bitrev_size4096_radix4<>+0x2940(SB)/8, $645  // bitrev[1320] = 645
DATA bitrev_size4096_radix4<>+0x2948(SB)/8, $1669 // bitrev[1321] = 1669
DATA bitrev_size4096_radix4<>+0x2950(SB)/8, $2693 // bitrev[1322] = 2693
DATA bitrev_size4096_radix4<>+0x2958(SB)/8, $3717 // bitrev[1323] = 3717
DATA bitrev_size4096_radix4<>+0x2960(SB)/8, $901  // bitrev[1324] = 901
DATA bitrev_size4096_radix4<>+0x2968(SB)/8, $1925 // bitrev[1325] = 1925
DATA bitrev_size4096_radix4<>+0x2970(SB)/8, $2949 // bitrev[1326] = 2949
DATA bitrev_size4096_radix4<>+0x2978(SB)/8, $3973 // bitrev[1327] = 3973
DATA bitrev_size4096_radix4<>+0x2980(SB)/8, $197  // bitrev[1328] = 197
DATA bitrev_size4096_radix4<>+0x2988(SB)/8, $1221 // bitrev[1329] = 1221
DATA bitrev_size4096_radix4<>+0x2990(SB)/8, $2245 // bitrev[1330] = 2245
DATA bitrev_size4096_radix4<>+0x2998(SB)/8, $3269 // bitrev[1331] = 3269
DATA bitrev_size4096_radix4<>+0x29a0(SB)/8, $453  // bitrev[1332] = 453
DATA bitrev_size4096_radix4<>+0x29a8(SB)/8, $1477 // bitrev[1333] = 1477
DATA bitrev_size4096_radix4<>+0x29b0(SB)/8, $2501 // bitrev[1334] = 2501
DATA bitrev_size4096_radix4<>+0x29b8(SB)/8, $3525 // bitrev[1335] = 3525
DATA bitrev_size4096_radix4<>+0x29c0(SB)/8, $709  // bitrev[1336] = 709
DATA bitrev_size4096_radix4<>+0x29c8(SB)/8, $1733 // bitrev[1337] = 1733
DATA bitrev_size4096_radix4<>+0x29d0(SB)/8, $2757 // bitrev[1338] = 2757
DATA bitrev_size4096_radix4<>+0x29d8(SB)/8, $3781 // bitrev[1339] = 3781
DATA bitrev_size4096_radix4<>+0x29e0(SB)/8, $965  // bitrev[1340] = 965
DATA bitrev_size4096_radix4<>+0x29e8(SB)/8, $1989 // bitrev[1341] = 1989
DATA bitrev_size4096_radix4<>+0x29f0(SB)/8, $3013 // bitrev[1342] = 3013
DATA bitrev_size4096_radix4<>+0x29f8(SB)/8, $4037 // bitrev[1343] = 4037
DATA bitrev_size4096_radix4<>+0x2a00(SB)/8, $21   // bitrev[1344] = 21
DATA bitrev_size4096_radix4<>+0x2a08(SB)/8, $1045 // bitrev[1345] = 1045
DATA bitrev_size4096_radix4<>+0x2a10(SB)/8, $2069 // bitrev[1346] = 2069
DATA bitrev_size4096_radix4<>+0x2a18(SB)/8, $3093 // bitrev[1347] = 3093
DATA bitrev_size4096_radix4<>+0x2a20(SB)/8, $277  // bitrev[1348] = 277
DATA bitrev_size4096_radix4<>+0x2a28(SB)/8, $1301 // bitrev[1349] = 1301
DATA bitrev_size4096_radix4<>+0x2a30(SB)/8, $2325 // bitrev[1350] = 2325
DATA bitrev_size4096_radix4<>+0x2a38(SB)/8, $3349 // bitrev[1351] = 3349
DATA bitrev_size4096_radix4<>+0x2a40(SB)/8, $533  // bitrev[1352] = 533
DATA bitrev_size4096_radix4<>+0x2a48(SB)/8, $1557 // bitrev[1353] = 1557
DATA bitrev_size4096_radix4<>+0x2a50(SB)/8, $2581 // bitrev[1354] = 2581
DATA bitrev_size4096_radix4<>+0x2a58(SB)/8, $3605 // bitrev[1355] = 3605
DATA bitrev_size4096_radix4<>+0x2a60(SB)/8, $789  // bitrev[1356] = 789
DATA bitrev_size4096_radix4<>+0x2a68(SB)/8, $1813 // bitrev[1357] = 1813
DATA bitrev_size4096_radix4<>+0x2a70(SB)/8, $2837 // bitrev[1358] = 2837
DATA bitrev_size4096_radix4<>+0x2a78(SB)/8, $3861 // bitrev[1359] = 3861
DATA bitrev_size4096_radix4<>+0x2a80(SB)/8, $85   // bitrev[1360] = 85
DATA bitrev_size4096_radix4<>+0x2a88(SB)/8, $1109 // bitrev[1361] = 1109
DATA bitrev_size4096_radix4<>+0x2a90(SB)/8, $2133 // bitrev[1362] = 2133
DATA bitrev_size4096_radix4<>+0x2a98(SB)/8, $3157 // bitrev[1363] = 3157
DATA bitrev_size4096_radix4<>+0x2aa0(SB)/8, $341  // bitrev[1364] = 341
DATA bitrev_size4096_radix4<>+0x2aa8(SB)/8, $1365 // bitrev[1365] = 1365
DATA bitrev_size4096_radix4<>+0x2ab0(SB)/8, $2389 // bitrev[1366] = 2389
DATA bitrev_size4096_radix4<>+0x2ab8(SB)/8, $3413 // bitrev[1367] = 3413
DATA bitrev_size4096_radix4<>+0x2ac0(SB)/8, $597  // bitrev[1368] = 597
DATA bitrev_size4096_radix4<>+0x2ac8(SB)/8, $1621 // bitrev[1369] = 1621
DATA bitrev_size4096_radix4<>+0x2ad0(SB)/8, $2645 // bitrev[1370] = 2645
DATA bitrev_size4096_radix4<>+0x2ad8(SB)/8, $3669 // bitrev[1371] = 3669
DATA bitrev_size4096_radix4<>+0x2ae0(SB)/8, $853  // bitrev[1372] = 853
DATA bitrev_size4096_radix4<>+0x2ae8(SB)/8, $1877 // bitrev[1373] = 1877
DATA bitrev_size4096_radix4<>+0x2af0(SB)/8, $2901 // bitrev[1374] = 2901
DATA bitrev_size4096_radix4<>+0x2af8(SB)/8, $3925 // bitrev[1375] = 3925
DATA bitrev_size4096_radix4<>+0x2b00(SB)/8, $149  // bitrev[1376] = 149
DATA bitrev_size4096_radix4<>+0x2b08(SB)/8, $1173 // bitrev[1377] = 1173
DATA bitrev_size4096_radix4<>+0x2b10(SB)/8, $2197 // bitrev[1378] = 2197
DATA bitrev_size4096_radix4<>+0x2b18(SB)/8, $3221 // bitrev[1379] = 3221
DATA bitrev_size4096_radix4<>+0x2b20(SB)/8, $405  // bitrev[1380] = 405
DATA bitrev_size4096_radix4<>+0x2b28(SB)/8, $1429 // bitrev[1381] = 1429
DATA bitrev_size4096_radix4<>+0x2b30(SB)/8, $2453 // bitrev[1382] = 2453
DATA bitrev_size4096_radix4<>+0x2b38(SB)/8, $3477 // bitrev[1383] = 3477
DATA bitrev_size4096_radix4<>+0x2b40(SB)/8, $661  // bitrev[1384] = 661
DATA bitrev_size4096_radix4<>+0x2b48(SB)/8, $1685 // bitrev[1385] = 1685
DATA bitrev_size4096_radix4<>+0x2b50(SB)/8, $2709 // bitrev[1386] = 2709
DATA bitrev_size4096_radix4<>+0x2b58(SB)/8, $3733 // bitrev[1387] = 3733
DATA bitrev_size4096_radix4<>+0x2b60(SB)/8, $917  // bitrev[1388] = 917
DATA bitrev_size4096_radix4<>+0x2b68(SB)/8, $1941 // bitrev[1389] = 1941
DATA bitrev_size4096_radix4<>+0x2b70(SB)/8, $2965 // bitrev[1390] = 2965
DATA bitrev_size4096_radix4<>+0x2b78(SB)/8, $3989 // bitrev[1391] = 3989
DATA bitrev_size4096_radix4<>+0x2b80(SB)/8, $213  // bitrev[1392] = 213
DATA bitrev_size4096_radix4<>+0x2b88(SB)/8, $1237 // bitrev[1393] = 1237
DATA bitrev_size4096_radix4<>+0x2b90(SB)/8, $2261 // bitrev[1394] = 2261
DATA bitrev_size4096_radix4<>+0x2b98(SB)/8, $3285 // bitrev[1395] = 3285
DATA bitrev_size4096_radix4<>+0x2ba0(SB)/8, $469  // bitrev[1396] = 469
DATA bitrev_size4096_radix4<>+0x2ba8(SB)/8, $1493 // bitrev[1397] = 1493
DATA bitrev_size4096_radix4<>+0x2bb0(SB)/8, $2517 // bitrev[1398] = 2517
DATA bitrev_size4096_radix4<>+0x2bb8(SB)/8, $3541 // bitrev[1399] = 3541
DATA bitrev_size4096_radix4<>+0x2bc0(SB)/8, $725  // bitrev[1400] = 725
DATA bitrev_size4096_radix4<>+0x2bc8(SB)/8, $1749 // bitrev[1401] = 1749
DATA bitrev_size4096_radix4<>+0x2bd0(SB)/8, $2773 // bitrev[1402] = 2773
DATA bitrev_size4096_radix4<>+0x2bd8(SB)/8, $3797 // bitrev[1403] = 3797
DATA bitrev_size4096_radix4<>+0x2be0(SB)/8, $981  // bitrev[1404] = 981
DATA bitrev_size4096_radix4<>+0x2be8(SB)/8, $2005 // bitrev[1405] = 2005
DATA bitrev_size4096_radix4<>+0x2bf0(SB)/8, $3029 // bitrev[1406] = 3029
DATA bitrev_size4096_radix4<>+0x2bf8(SB)/8, $4053 // bitrev[1407] = 4053
DATA bitrev_size4096_radix4<>+0x2c00(SB)/8, $37   // bitrev[1408] = 37
DATA bitrev_size4096_radix4<>+0x2c08(SB)/8, $1061 // bitrev[1409] = 1061
DATA bitrev_size4096_radix4<>+0x2c10(SB)/8, $2085 // bitrev[1410] = 2085
DATA bitrev_size4096_radix4<>+0x2c18(SB)/8, $3109 // bitrev[1411] = 3109
DATA bitrev_size4096_radix4<>+0x2c20(SB)/8, $293  // bitrev[1412] = 293
DATA bitrev_size4096_radix4<>+0x2c28(SB)/8, $1317 // bitrev[1413] = 1317
DATA bitrev_size4096_radix4<>+0x2c30(SB)/8, $2341 // bitrev[1414] = 2341
DATA bitrev_size4096_radix4<>+0x2c38(SB)/8, $3365 // bitrev[1415] = 3365
DATA bitrev_size4096_radix4<>+0x2c40(SB)/8, $549  // bitrev[1416] = 549
DATA bitrev_size4096_radix4<>+0x2c48(SB)/8, $1573 // bitrev[1417] = 1573
DATA bitrev_size4096_radix4<>+0x2c50(SB)/8, $2597 // bitrev[1418] = 2597
DATA bitrev_size4096_radix4<>+0x2c58(SB)/8, $3621 // bitrev[1419] = 3621
DATA bitrev_size4096_radix4<>+0x2c60(SB)/8, $805  // bitrev[1420] = 805
DATA bitrev_size4096_radix4<>+0x2c68(SB)/8, $1829 // bitrev[1421] = 1829
DATA bitrev_size4096_radix4<>+0x2c70(SB)/8, $2853 // bitrev[1422] = 2853
DATA bitrev_size4096_radix4<>+0x2c78(SB)/8, $3877 // bitrev[1423] = 3877
DATA bitrev_size4096_radix4<>+0x2c80(SB)/8, $101  // bitrev[1424] = 101
DATA bitrev_size4096_radix4<>+0x2c88(SB)/8, $1125 // bitrev[1425] = 1125
DATA bitrev_size4096_radix4<>+0x2c90(SB)/8, $2149 // bitrev[1426] = 2149
DATA bitrev_size4096_radix4<>+0x2c98(SB)/8, $3173 // bitrev[1427] = 3173
DATA bitrev_size4096_radix4<>+0x2ca0(SB)/8, $357  // bitrev[1428] = 357
DATA bitrev_size4096_radix4<>+0x2ca8(SB)/8, $1381 // bitrev[1429] = 1381
DATA bitrev_size4096_radix4<>+0x2cb0(SB)/8, $2405 // bitrev[1430] = 2405
DATA bitrev_size4096_radix4<>+0x2cb8(SB)/8, $3429 // bitrev[1431] = 3429
DATA bitrev_size4096_radix4<>+0x2cc0(SB)/8, $613  // bitrev[1432] = 613
DATA bitrev_size4096_radix4<>+0x2cc8(SB)/8, $1637 // bitrev[1433] = 1637
DATA bitrev_size4096_radix4<>+0x2cd0(SB)/8, $2661 // bitrev[1434] = 2661
DATA bitrev_size4096_radix4<>+0x2cd8(SB)/8, $3685 // bitrev[1435] = 3685
DATA bitrev_size4096_radix4<>+0x2ce0(SB)/8, $869  // bitrev[1436] = 869
DATA bitrev_size4096_radix4<>+0x2ce8(SB)/8, $1893 // bitrev[1437] = 1893
DATA bitrev_size4096_radix4<>+0x2cf0(SB)/8, $2917 // bitrev[1438] = 2917
DATA bitrev_size4096_radix4<>+0x2cf8(SB)/8, $3941 // bitrev[1439] = 3941
DATA bitrev_size4096_radix4<>+0x2d00(SB)/8, $165  // bitrev[1440] = 165
DATA bitrev_size4096_radix4<>+0x2d08(SB)/8, $1189 // bitrev[1441] = 1189
DATA bitrev_size4096_radix4<>+0x2d10(SB)/8, $2213 // bitrev[1442] = 2213
DATA bitrev_size4096_radix4<>+0x2d18(SB)/8, $3237 // bitrev[1443] = 3237
DATA bitrev_size4096_radix4<>+0x2d20(SB)/8, $421  // bitrev[1444] = 421
DATA bitrev_size4096_radix4<>+0x2d28(SB)/8, $1445 // bitrev[1445] = 1445
DATA bitrev_size4096_radix4<>+0x2d30(SB)/8, $2469 // bitrev[1446] = 2469
DATA bitrev_size4096_radix4<>+0x2d38(SB)/8, $3493 // bitrev[1447] = 3493
DATA bitrev_size4096_radix4<>+0x2d40(SB)/8, $677  // bitrev[1448] = 677
DATA bitrev_size4096_radix4<>+0x2d48(SB)/8, $1701 // bitrev[1449] = 1701
DATA bitrev_size4096_radix4<>+0x2d50(SB)/8, $2725 // bitrev[1450] = 2725
DATA bitrev_size4096_radix4<>+0x2d58(SB)/8, $3749 // bitrev[1451] = 3749
DATA bitrev_size4096_radix4<>+0x2d60(SB)/8, $933  // bitrev[1452] = 933
DATA bitrev_size4096_radix4<>+0x2d68(SB)/8, $1957 // bitrev[1453] = 1957
DATA bitrev_size4096_radix4<>+0x2d70(SB)/8, $2981 // bitrev[1454] = 2981
DATA bitrev_size4096_radix4<>+0x2d78(SB)/8, $4005 // bitrev[1455] = 4005
DATA bitrev_size4096_radix4<>+0x2d80(SB)/8, $229  // bitrev[1456] = 229
DATA bitrev_size4096_radix4<>+0x2d88(SB)/8, $1253 // bitrev[1457] = 1253
DATA bitrev_size4096_radix4<>+0x2d90(SB)/8, $2277 // bitrev[1458] = 2277
DATA bitrev_size4096_radix4<>+0x2d98(SB)/8, $3301 // bitrev[1459] = 3301
DATA bitrev_size4096_radix4<>+0x2da0(SB)/8, $485  // bitrev[1460] = 485
DATA bitrev_size4096_radix4<>+0x2da8(SB)/8, $1509 // bitrev[1461] = 1509
DATA bitrev_size4096_radix4<>+0x2db0(SB)/8, $2533 // bitrev[1462] = 2533
DATA bitrev_size4096_radix4<>+0x2db8(SB)/8, $3557 // bitrev[1463] = 3557
DATA bitrev_size4096_radix4<>+0x2dc0(SB)/8, $741  // bitrev[1464] = 741
DATA bitrev_size4096_radix4<>+0x2dc8(SB)/8, $1765 // bitrev[1465] = 1765
DATA bitrev_size4096_radix4<>+0x2dd0(SB)/8, $2789 // bitrev[1466] = 2789
DATA bitrev_size4096_radix4<>+0x2dd8(SB)/8, $3813 // bitrev[1467] = 3813
DATA bitrev_size4096_radix4<>+0x2de0(SB)/8, $997  // bitrev[1468] = 997
DATA bitrev_size4096_radix4<>+0x2de8(SB)/8, $2021 // bitrev[1469] = 2021
DATA bitrev_size4096_radix4<>+0x2df0(SB)/8, $3045 // bitrev[1470] = 3045
DATA bitrev_size4096_radix4<>+0x2df8(SB)/8, $4069 // bitrev[1471] = 4069
DATA bitrev_size4096_radix4<>+0x2e00(SB)/8, $53   // bitrev[1472] = 53
DATA bitrev_size4096_radix4<>+0x2e08(SB)/8, $1077 // bitrev[1473] = 1077
DATA bitrev_size4096_radix4<>+0x2e10(SB)/8, $2101 // bitrev[1474] = 2101
DATA bitrev_size4096_radix4<>+0x2e18(SB)/8, $3125 // bitrev[1475] = 3125
DATA bitrev_size4096_radix4<>+0x2e20(SB)/8, $309  // bitrev[1476] = 309
DATA bitrev_size4096_radix4<>+0x2e28(SB)/8, $1333 // bitrev[1477] = 1333
DATA bitrev_size4096_radix4<>+0x2e30(SB)/8, $2357 // bitrev[1478] = 2357
DATA bitrev_size4096_radix4<>+0x2e38(SB)/8, $3381 // bitrev[1479] = 3381
DATA bitrev_size4096_radix4<>+0x2e40(SB)/8, $565  // bitrev[1480] = 565
DATA bitrev_size4096_radix4<>+0x2e48(SB)/8, $1589 // bitrev[1481] = 1589
DATA bitrev_size4096_radix4<>+0x2e50(SB)/8, $2613 // bitrev[1482] = 2613
DATA bitrev_size4096_radix4<>+0x2e58(SB)/8, $3637 // bitrev[1483] = 3637
DATA bitrev_size4096_radix4<>+0x2e60(SB)/8, $821  // bitrev[1484] = 821
DATA bitrev_size4096_radix4<>+0x2e68(SB)/8, $1845 // bitrev[1485] = 1845
DATA bitrev_size4096_radix4<>+0x2e70(SB)/8, $2869 // bitrev[1486] = 2869
DATA bitrev_size4096_radix4<>+0x2e78(SB)/8, $3893 // bitrev[1487] = 3893
DATA bitrev_size4096_radix4<>+0x2e80(SB)/8, $117  // bitrev[1488] = 117
DATA bitrev_size4096_radix4<>+0x2e88(SB)/8, $1141 // bitrev[1489] = 1141
DATA bitrev_size4096_radix4<>+0x2e90(SB)/8, $2165 // bitrev[1490] = 2165
DATA bitrev_size4096_radix4<>+0x2e98(SB)/8, $3189 // bitrev[1491] = 3189
DATA bitrev_size4096_radix4<>+0x2ea0(SB)/8, $373  // bitrev[1492] = 373
DATA bitrev_size4096_radix4<>+0x2ea8(SB)/8, $1397 // bitrev[1493] = 1397
DATA bitrev_size4096_radix4<>+0x2eb0(SB)/8, $2421 // bitrev[1494] = 2421
DATA bitrev_size4096_radix4<>+0x2eb8(SB)/8, $3445 // bitrev[1495] = 3445
DATA bitrev_size4096_radix4<>+0x2ec0(SB)/8, $629  // bitrev[1496] = 629
DATA bitrev_size4096_radix4<>+0x2ec8(SB)/8, $1653 // bitrev[1497] = 1653
DATA bitrev_size4096_radix4<>+0x2ed0(SB)/8, $2677 // bitrev[1498] = 2677
DATA bitrev_size4096_radix4<>+0x2ed8(SB)/8, $3701 // bitrev[1499] = 3701
DATA bitrev_size4096_radix4<>+0x2ee0(SB)/8, $885  // bitrev[1500] = 885
DATA bitrev_size4096_radix4<>+0x2ee8(SB)/8, $1909 // bitrev[1501] = 1909
DATA bitrev_size4096_radix4<>+0x2ef0(SB)/8, $2933 // bitrev[1502] = 2933
DATA bitrev_size4096_radix4<>+0x2ef8(SB)/8, $3957 // bitrev[1503] = 3957
DATA bitrev_size4096_radix4<>+0x2f00(SB)/8, $181  // bitrev[1504] = 181
DATA bitrev_size4096_radix4<>+0x2f08(SB)/8, $1205 // bitrev[1505] = 1205
DATA bitrev_size4096_radix4<>+0x2f10(SB)/8, $2229 // bitrev[1506] = 2229
DATA bitrev_size4096_radix4<>+0x2f18(SB)/8, $3253 // bitrev[1507] = 3253
DATA bitrev_size4096_radix4<>+0x2f20(SB)/8, $437  // bitrev[1508] = 437
DATA bitrev_size4096_radix4<>+0x2f28(SB)/8, $1461 // bitrev[1509] = 1461
DATA bitrev_size4096_radix4<>+0x2f30(SB)/8, $2485 // bitrev[1510] = 2485
DATA bitrev_size4096_radix4<>+0x2f38(SB)/8, $3509 // bitrev[1511] = 3509
DATA bitrev_size4096_radix4<>+0x2f40(SB)/8, $693  // bitrev[1512] = 693
DATA bitrev_size4096_radix4<>+0x2f48(SB)/8, $1717 // bitrev[1513] = 1717
DATA bitrev_size4096_radix4<>+0x2f50(SB)/8, $2741 // bitrev[1514] = 2741
DATA bitrev_size4096_radix4<>+0x2f58(SB)/8, $3765 // bitrev[1515] = 3765
DATA bitrev_size4096_radix4<>+0x2f60(SB)/8, $949  // bitrev[1516] = 949
DATA bitrev_size4096_radix4<>+0x2f68(SB)/8, $1973 // bitrev[1517] = 1973
DATA bitrev_size4096_radix4<>+0x2f70(SB)/8, $2997 // bitrev[1518] = 2997
DATA bitrev_size4096_radix4<>+0x2f78(SB)/8, $4021 // bitrev[1519] = 4021
DATA bitrev_size4096_radix4<>+0x2f80(SB)/8, $245  // bitrev[1520] = 245
DATA bitrev_size4096_radix4<>+0x2f88(SB)/8, $1269 // bitrev[1521] = 1269
DATA bitrev_size4096_radix4<>+0x2f90(SB)/8, $2293 // bitrev[1522] = 2293
DATA bitrev_size4096_radix4<>+0x2f98(SB)/8, $3317 // bitrev[1523] = 3317
DATA bitrev_size4096_radix4<>+0x2fa0(SB)/8, $501  // bitrev[1524] = 501
DATA bitrev_size4096_radix4<>+0x2fa8(SB)/8, $1525 // bitrev[1525] = 1525
DATA bitrev_size4096_radix4<>+0x2fb0(SB)/8, $2549 // bitrev[1526] = 2549
DATA bitrev_size4096_radix4<>+0x2fb8(SB)/8, $3573 // bitrev[1527] = 3573
DATA bitrev_size4096_radix4<>+0x2fc0(SB)/8, $757  // bitrev[1528] = 757
DATA bitrev_size4096_radix4<>+0x2fc8(SB)/8, $1781 // bitrev[1529] = 1781
DATA bitrev_size4096_radix4<>+0x2fd0(SB)/8, $2805 // bitrev[1530] = 2805
DATA bitrev_size4096_radix4<>+0x2fd8(SB)/8, $3829 // bitrev[1531] = 3829
DATA bitrev_size4096_radix4<>+0x2fe0(SB)/8, $1013 // bitrev[1532] = 1013
DATA bitrev_size4096_radix4<>+0x2fe8(SB)/8, $2037 // bitrev[1533] = 2037
DATA bitrev_size4096_radix4<>+0x2ff0(SB)/8, $3061 // bitrev[1534] = 3061
DATA bitrev_size4096_radix4<>+0x2ff8(SB)/8, $4085 // bitrev[1535] = 4085
DATA bitrev_size4096_radix4<>+0x3000(SB)/8, $9    // bitrev[1536] = 9
DATA bitrev_size4096_radix4<>+0x3008(SB)/8, $1033 // bitrev[1537] = 1033
DATA bitrev_size4096_radix4<>+0x3010(SB)/8, $2057 // bitrev[1538] = 2057
DATA bitrev_size4096_radix4<>+0x3018(SB)/8, $3081 // bitrev[1539] = 3081
DATA bitrev_size4096_radix4<>+0x3020(SB)/8, $265  // bitrev[1540] = 265
DATA bitrev_size4096_radix4<>+0x3028(SB)/8, $1289 // bitrev[1541] = 1289
DATA bitrev_size4096_radix4<>+0x3030(SB)/8, $2313 // bitrev[1542] = 2313
DATA bitrev_size4096_radix4<>+0x3038(SB)/8, $3337 // bitrev[1543] = 3337
DATA bitrev_size4096_radix4<>+0x3040(SB)/8, $521  // bitrev[1544] = 521
DATA bitrev_size4096_radix4<>+0x3048(SB)/8, $1545 // bitrev[1545] = 1545
DATA bitrev_size4096_radix4<>+0x3050(SB)/8, $2569 // bitrev[1546] = 2569
DATA bitrev_size4096_radix4<>+0x3058(SB)/8, $3593 // bitrev[1547] = 3593
DATA bitrev_size4096_radix4<>+0x3060(SB)/8, $777  // bitrev[1548] = 777
DATA bitrev_size4096_radix4<>+0x3068(SB)/8, $1801 // bitrev[1549] = 1801
DATA bitrev_size4096_radix4<>+0x3070(SB)/8, $2825 // bitrev[1550] = 2825
DATA bitrev_size4096_radix4<>+0x3078(SB)/8, $3849 // bitrev[1551] = 3849
DATA bitrev_size4096_radix4<>+0x3080(SB)/8, $73   // bitrev[1552] = 73
DATA bitrev_size4096_radix4<>+0x3088(SB)/8, $1097 // bitrev[1553] = 1097
DATA bitrev_size4096_radix4<>+0x3090(SB)/8, $2121 // bitrev[1554] = 2121
DATA bitrev_size4096_radix4<>+0x3098(SB)/8, $3145 // bitrev[1555] = 3145
DATA bitrev_size4096_radix4<>+0x30a0(SB)/8, $329  // bitrev[1556] = 329
DATA bitrev_size4096_radix4<>+0x30a8(SB)/8, $1353 // bitrev[1557] = 1353
DATA bitrev_size4096_radix4<>+0x30b0(SB)/8, $2377 // bitrev[1558] = 2377
DATA bitrev_size4096_radix4<>+0x30b8(SB)/8, $3401 // bitrev[1559] = 3401
DATA bitrev_size4096_radix4<>+0x30c0(SB)/8, $585  // bitrev[1560] = 585
DATA bitrev_size4096_radix4<>+0x30c8(SB)/8, $1609 // bitrev[1561] = 1609
DATA bitrev_size4096_radix4<>+0x30d0(SB)/8, $2633 // bitrev[1562] = 2633
DATA bitrev_size4096_radix4<>+0x30d8(SB)/8, $3657 // bitrev[1563] = 3657
DATA bitrev_size4096_radix4<>+0x30e0(SB)/8, $841  // bitrev[1564] = 841
DATA bitrev_size4096_radix4<>+0x30e8(SB)/8, $1865 // bitrev[1565] = 1865
DATA bitrev_size4096_radix4<>+0x30f0(SB)/8, $2889 // bitrev[1566] = 2889
DATA bitrev_size4096_radix4<>+0x30f8(SB)/8, $3913 // bitrev[1567] = 3913
DATA bitrev_size4096_radix4<>+0x3100(SB)/8, $137  // bitrev[1568] = 137
DATA bitrev_size4096_radix4<>+0x3108(SB)/8, $1161 // bitrev[1569] = 1161
DATA bitrev_size4096_radix4<>+0x3110(SB)/8, $2185 // bitrev[1570] = 2185
DATA bitrev_size4096_radix4<>+0x3118(SB)/8, $3209 // bitrev[1571] = 3209
DATA bitrev_size4096_radix4<>+0x3120(SB)/8, $393  // bitrev[1572] = 393
DATA bitrev_size4096_radix4<>+0x3128(SB)/8, $1417 // bitrev[1573] = 1417
DATA bitrev_size4096_radix4<>+0x3130(SB)/8, $2441 // bitrev[1574] = 2441
DATA bitrev_size4096_radix4<>+0x3138(SB)/8, $3465 // bitrev[1575] = 3465
DATA bitrev_size4096_radix4<>+0x3140(SB)/8, $649  // bitrev[1576] = 649
DATA bitrev_size4096_radix4<>+0x3148(SB)/8, $1673 // bitrev[1577] = 1673
DATA bitrev_size4096_radix4<>+0x3150(SB)/8, $2697 // bitrev[1578] = 2697
DATA bitrev_size4096_radix4<>+0x3158(SB)/8, $3721 // bitrev[1579] = 3721
DATA bitrev_size4096_radix4<>+0x3160(SB)/8, $905  // bitrev[1580] = 905
DATA bitrev_size4096_radix4<>+0x3168(SB)/8, $1929 // bitrev[1581] = 1929
DATA bitrev_size4096_radix4<>+0x3170(SB)/8, $2953 // bitrev[1582] = 2953
DATA bitrev_size4096_radix4<>+0x3178(SB)/8, $3977 // bitrev[1583] = 3977
DATA bitrev_size4096_radix4<>+0x3180(SB)/8, $201  // bitrev[1584] = 201
DATA bitrev_size4096_radix4<>+0x3188(SB)/8, $1225 // bitrev[1585] = 1225
DATA bitrev_size4096_radix4<>+0x3190(SB)/8, $2249 // bitrev[1586] = 2249
DATA bitrev_size4096_radix4<>+0x3198(SB)/8, $3273 // bitrev[1587] = 3273
DATA bitrev_size4096_radix4<>+0x31a0(SB)/8, $457  // bitrev[1588] = 457
DATA bitrev_size4096_radix4<>+0x31a8(SB)/8, $1481 // bitrev[1589] = 1481
DATA bitrev_size4096_radix4<>+0x31b0(SB)/8, $2505 // bitrev[1590] = 2505
DATA bitrev_size4096_radix4<>+0x31b8(SB)/8, $3529 // bitrev[1591] = 3529
DATA bitrev_size4096_radix4<>+0x31c0(SB)/8, $713  // bitrev[1592] = 713
DATA bitrev_size4096_radix4<>+0x31c8(SB)/8, $1737 // bitrev[1593] = 1737
DATA bitrev_size4096_radix4<>+0x31d0(SB)/8, $2761 // bitrev[1594] = 2761
DATA bitrev_size4096_radix4<>+0x31d8(SB)/8, $3785 // bitrev[1595] = 3785
DATA bitrev_size4096_radix4<>+0x31e0(SB)/8, $969  // bitrev[1596] = 969
DATA bitrev_size4096_radix4<>+0x31e8(SB)/8, $1993 // bitrev[1597] = 1993
DATA bitrev_size4096_radix4<>+0x31f0(SB)/8, $3017 // bitrev[1598] = 3017
DATA bitrev_size4096_radix4<>+0x31f8(SB)/8, $4041 // bitrev[1599] = 4041
DATA bitrev_size4096_radix4<>+0x3200(SB)/8, $25   // bitrev[1600] = 25
DATA bitrev_size4096_radix4<>+0x3208(SB)/8, $1049 // bitrev[1601] = 1049
DATA bitrev_size4096_radix4<>+0x3210(SB)/8, $2073 // bitrev[1602] = 2073
DATA bitrev_size4096_radix4<>+0x3218(SB)/8, $3097 // bitrev[1603] = 3097
DATA bitrev_size4096_radix4<>+0x3220(SB)/8, $281  // bitrev[1604] = 281
DATA bitrev_size4096_radix4<>+0x3228(SB)/8, $1305 // bitrev[1605] = 1305
DATA bitrev_size4096_radix4<>+0x3230(SB)/8, $2329 // bitrev[1606] = 2329
DATA bitrev_size4096_radix4<>+0x3238(SB)/8, $3353 // bitrev[1607] = 3353
DATA bitrev_size4096_radix4<>+0x3240(SB)/8, $537  // bitrev[1608] = 537
DATA bitrev_size4096_radix4<>+0x3248(SB)/8, $1561 // bitrev[1609] = 1561
DATA bitrev_size4096_radix4<>+0x3250(SB)/8, $2585 // bitrev[1610] = 2585
DATA bitrev_size4096_radix4<>+0x3258(SB)/8, $3609 // bitrev[1611] = 3609
DATA bitrev_size4096_radix4<>+0x3260(SB)/8, $793  // bitrev[1612] = 793
DATA bitrev_size4096_radix4<>+0x3268(SB)/8, $1817 // bitrev[1613] = 1817
DATA bitrev_size4096_radix4<>+0x3270(SB)/8, $2841 // bitrev[1614] = 2841
DATA bitrev_size4096_radix4<>+0x3278(SB)/8, $3865 // bitrev[1615] = 3865
DATA bitrev_size4096_radix4<>+0x3280(SB)/8, $89   // bitrev[1616] = 89
DATA bitrev_size4096_radix4<>+0x3288(SB)/8, $1113 // bitrev[1617] = 1113
DATA bitrev_size4096_radix4<>+0x3290(SB)/8, $2137 // bitrev[1618] = 2137
DATA bitrev_size4096_radix4<>+0x3298(SB)/8, $3161 // bitrev[1619] = 3161
DATA bitrev_size4096_radix4<>+0x32a0(SB)/8, $345  // bitrev[1620] = 345
DATA bitrev_size4096_radix4<>+0x32a8(SB)/8, $1369 // bitrev[1621] = 1369
DATA bitrev_size4096_radix4<>+0x32b0(SB)/8, $2393 // bitrev[1622] = 2393
DATA bitrev_size4096_radix4<>+0x32b8(SB)/8, $3417 // bitrev[1623] = 3417
DATA bitrev_size4096_radix4<>+0x32c0(SB)/8, $601  // bitrev[1624] = 601
DATA bitrev_size4096_radix4<>+0x32c8(SB)/8, $1625 // bitrev[1625] = 1625
DATA bitrev_size4096_radix4<>+0x32d0(SB)/8, $2649 // bitrev[1626] = 2649
DATA bitrev_size4096_radix4<>+0x32d8(SB)/8, $3673 // bitrev[1627] = 3673
DATA bitrev_size4096_radix4<>+0x32e0(SB)/8, $857  // bitrev[1628] = 857
DATA bitrev_size4096_radix4<>+0x32e8(SB)/8, $1881 // bitrev[1629] = 1881
DATA bitrev_size4096_radix4<>+0x32f0(SB)/8, $2905 // bitrev[1630] = 2905
DATA bitrev_size4096_radix4<>+0x32f8(SB)/8, $3929 // bitrev[1631] = 3929
DATA bitrev_size4096_radix4<>+0x3300(SB)/8, $153  // bitrev[1632] = 153
DATA bitrev_size4096_radix4<>+0x3308(SB)/8, $1177 // bitrev[1633] = 1177
DATA bitrev_size4096_radix4<>+0x3310(SB)/8, $2201 // bitrev[1634] = 2201
DATA bitrev_size4096_radix4<>+0x3318(SB)/8, $3225 // bitrev[1635] = 3225
DATA bitrev_size4096_radix4<>+0x3320(SB)/8, $409  // bitrev[1636] = 409
DATA bitrev_size4096_radix4<>+0x3328(SB)/8, $1433 // bitrev[1637] = 1433
DATA bitrev_size4096_radix4<>+0x3330(SB)/8, $2457 // bitrev[1638] = 2457
DATA bitrev_size4096_radix4<>+0x3338(SB)/8, $3481 // bitrev[1639] = 3481
DATA bitrev_size4096_radix4<>+0x3340(SB)/8, $665  // bitrev[1640] = 665
DATA bitrev_size4096_radix4<>+0x3348(SB)/8, $1689 // bitrev[1641] = 1689
DATA bitrev_size4096_radix4<>+0x3350(SB)/8, $2713 // bitrev[1642] = 2713
DATA bitrev_size4096_radix4<>+0x3358(SB)/8, $3737 // bitrev[1643] = 3737
DATA bitrev_size4096_radix4<>+0x3360(SB)/8, $921  // bitrev[1644] = 921
DATA bitrev_size4096_radix4<>+0x3368(SB)/8, $1945 // bitrev[1645] = 1945
DATA bitrev_size4096_radix4<>+0x3370(SB)/8, $2969 // bitrev[1646] = 2969
DATA bitrev_size4096_radix4<>+0x3378(SB)/8, $3993 // bitrev[1647] = 3993
DATA bitrev_size4096_radix4<>+0x3380(SB)/8, $217  // bitrev[1648] = 217
DATA bitrev_size4096_radix4<>+0x3388(SB)/8, $1241 // bitrev[1649] = 1241
DATA bitrev_size4096_radix4<>+0x3390(SB)/8, $2265 // bitrev[1650] = 2265
DATA bitrev_size4096_radix4<>+0x3398(SB)/8, $3289 // bitrev[1651] = 3289
DATA bitrev_size4096_radix4<>+0x33a0(SB)/8, $473  // bitrev[1652] = 473
DATA bitrev_size4096_radix4<>+0x33a8(SB)/8, $1497 // bitrev[1653] = 1497
DATA bitrev_size4096_radix4<>+0x33b0(SB)/8, $2521 // bitrev[1654] = 2521
DATA bitrev_size4096_radix4<>+0x33b8(SB)/8, $3545 // bitrev[1655] = 3545
DATA bitrev_size4096_radix4<>+0x33c0(SB)/8, $729  // bitrev[1656] = 729
DATA bitrev_size4096_radix4<>+0x33c8(SB)/8, $1753 // bitrev[1657] = 1753
DATA bitrev_size4096_radix4<>+0x33d0(SB)/8, $2777 // bitrev[1658] = 2777
DATA bitrev_size4096_radix4<>+0x33d8(SB)/8, $3801 // bitrev[1659] = 3801
DATA bitrev_size4096_radix4<>+0x33e0(SB)/8, $985  // bitrev[1660] = 985
DATA bitrev_size4096_radix4<>+0x33e8(SB)/8, $2009 // bitrev[1661] = 2009
DATA bitrev_size4096_radix4<>+0x33f0(SB)/8, $3033 // bitrev[1662] = 3033
DATA bitrev_size4096_radix4<>+0x33f8(SB)/8, $4057 // bitrev[1663] = 4057
DATA bitrev_size4096_radix4<>+0x3400(SB)/8, $41   // bitrev[1664] = 41
DATA bitrev_size4096_radix4<>+0x3408(SB)/8, $1065 // bitrev[1665] = 1065
DATA bitrev_size4096_radix4<>+0x3410(SB)/8, $2089 // bitrev[1666] = 2089
DATA bitrev_size4096_radix4<>+0x3418(SB)/8, $3113 // bitrev[1667] = 3113
DATA bitrev_size4096_radix4<>+0x3420(SB)/8, $297  // bitrev[1668] = 297
DATA bitrev_size4096_radix4<>+0x3428(SB)/8, $1321 // bitrev[1669] = 1321
DATA bitrev_size4096_radix4<>+0x3430(SB)/8, $2345 // bitrev[1670] = 2345
DATA bitrev_size4096_radix4<>+0x3438(SB)/8, $3369 // bitrev[1671] = 3369
DATA bitrev_size4096_radix4<>+0x3440(SB)/8, $553  // bitrev[1672] = 553
DATA bitrev_size4096_radix4<>+0x3448(SB)/8, $1577 // bitrev[1673] = 1577
DATA bitrev_size4096_radix4<>+0x3450(SB)/8, $2601 // bitrev[1674] = 2601
DATA bitrev_size4096_radix4<>+0x3458(SB)/8, $3625 // bitrev[1675] = 3625
DATA bitrev_size4096_radix4<>+0x3460(SB)/8, $809  // bitrev[1676] = 809
DATA bitrev_size4096_radix4<>+0x3468(SB)/8, $1833 // bitrev[1677] = 1833
DATA bitrev_size4096_radix4<>+0x3470(SB)/8, $2857 // bitrev[1678] = 2857
DATA bitrev_size4096_radix4<>+0x3478(SB)/8, $3881 // bitrev[1679] = 3881
DATA bitrev_size4096_radix4<>+0x3480(SB)/8, $105  // bitrev[1680] = 105
DATA bitrev_size4096_radix4<>+0x3488(SB)/8, $1129 // bitrev[1681] = 1129
DATA bitrev_size4096_radix4<>+0x3490(SB)/8, $2153 // bitrev[1682] = 2153
DATA bitrev_size4096_radix4<>+0x3498(SB)/8, $3177 // bitrev[1683] = 3177
DATA bitrev_size4096_radix4<>+0x34a0(SB)/8, $361  // bitrev[1684] = 361
DATA bitrev_size4096_radix4<>+0x34a8(SB)/8, $1385 // bitrev[1685] = 1385
DATA bitrev_size4096_radix4<>+0x34b0(SB)/8, $2409 // bitrev[1686] = 2409
DATA bitrev_size4096_radix4<>+0x34b8(SB)/8, $3433 // bitrev[1687] = 3433
DATA bitrev_size4096_radix4<>+0x34c0(SB)/8, $617  // bitrev[1688] = 617
DATA bitrev_size4096_radix4<>+0x34c8(SB)/8, $1641 // bitrev[1689] = 1641
DATA bitrev_size4096_radix4<>+0x34d0(SB)/8, $2665 // bitrev[1690] = 2665
DATA bitrev_size4096_radix4<>+0x34d8(SB)/8, $3689 // bitrev[1691] = 3689
DATA bitrev_size4096_radix4<>+0x34e0(SB)/8, $873  // bitrev[1692] = 873
DATA bitrev_size4096_radix4<>+0x34e8(SB)/8, $1897 // bitrev[1693] = 1897
DATA bitrev_size4096_radix4<>+0x34f0(SB)/8, $2921 // bitrev[1694] = 2921
DATA bitrev_size4096_radix4<>+0x34f8(SB)/8, $3945 // bitrev[1695] = 3945
DATA bitrev_size4096_radix4<>+0x3500(SB)/8, $169  // bitrev[1696] = 169
DATA bitrev_size4096_radix4<>+0x3508(SB)/8, $1193 // bitrev[1697] = 1193
DATA bitrev_size4096_radix4<>+0x3510(SB)/8, $2217 // bitrev[1698] = 2217
DATA bitrev_size4096_radix4<>+0x3518(SB)/8, $3241 // bitrev[1699] = 3241
DATA bitrev_size4096_radix4<>+0x3520(SB)/8, $425  // bitrev[1700] = 425
DATA bitrev_size4096_radix4<>+0x3528(SB)/8, $1449 // bitrev[1701] = 1449
DATA bitrev_size4096_radix4<>+0x3530(SB)/8, $2473 // bitrev[1702] = 2473
DATA bitrev_size4096_radix4<>+0x3538(SB)/8, $3497 // bitrev[1703] = 3497
DATA bitrev_size4096_radix4<>+0x3540(SB)/8, $681  // bitrev[1704] = 681
DATA bitrev_size4096_radix4<>+0x3548(SB)/8, $1705 // bitrev[1705] = 1705
DATA bitrev_size4096_radix4<>+0x3550(SB)/8, $2729 // bitrev[1706] = 2729
DATA bitrev_size4096_radix4<>+0x3558(SB)/8, $3753 // bitrev[1707] = 3753
DATA bitrev_size4096_radix4<>+0x3560(SB)/8, $937  // bitrev[1708] = 937
DATA bitrev_size4096_radix4<>+0x3568(SB)/8, $1961 // bitrev[1709] = 1961
DATA bitrev_size4096_radix4<>+0x3570(SB)/8, $2985 // bitrev[1710] = 2985
DATA bitrev_size4096_radix4<>+0x3578(SB)/8, $4009 // bitrev[1711] = 4009
DATA bitrev_size4096_radix4<>+0x3580(SB)/8, $233  // bitrev[1712] = 233
DATA bitrev_size4096_radix4<>+0x3588(SB)/8, $1257 // bitrev[1713] = 1257
DATA bitrev_size4096_radix4<>+0x3590(SB)/8, $2281 // bitrev[1714] = 2281
DATA bitrev_size4096_radix4<>+0x3598(SB)/8, $3305 // bitrev[1715] = 3305
DATA bitrev_size4096_radix4<>+0x35a0(SB)/8, $489  // bitrev[1716] = 489
DATA bitrev_size4096_radix4<>+0x35a8(SB)/8, $1513 // bitrev[1717] = 1513
DATA bitrev_size4096_radix4<>+0x35b0(SB)/8, $2537 // bitrev[1718] = 2537
DATA bitrev_size4096_radix4<>+0x35b8(SB)/8, $3561 // bitrev[1719] = 3561
DATA bitrev_size4096_radix4<>+0x35c0(SB)/8, $745  // bitrev[1720] = 745
DATA bitrev_size4096_radix4<>+0x35c8(SB)/8, $1769 // bitrev[1721] = 1769
DATA bitrev_size4096_radix4<>+0x35d0(SB)/8, $2793 // bitrev[1722] = 2793
DATA bitrev_size4096_radix4<>+0x35d8(SB)/8, $3817 // bitrev[1723] = 3817
DATA bitrev_size4096_radix4<>+0x35e0(SB)/8, $1001 // bitrev[1724] = 1001
DATA bitrev_size4096_radix4<>+0x35e8(SB)/8, $2025 // bitrev[1725] = 2025
DATA bitrev_size4096_radix4<>+0x35f0(SB)/8, $3049 // bitrev[1726] = 3049
DATA bitrev_size4096_radix4<>+0x35f8(SB)/8, $4073 // bitrev[1727] = 4073
DATA bitrev_size4096_radix4<>+0x3600(SB)/8, $57   // bitrev[1728] = 57
DATA bitrev_size4096_radix4<>+0x3608(SB)/8, $1081 // bitrev[1729] = 1081
DATA bitrev_size4096_radix4<>+0x3610(SB)/8, $2105 // bitrev[1730] = 2105
DATA bitrev_size4096_radix4<>+0x3618(SB)/8, $3129 // bitrev[1731] = 3129
DATA bitrev_size4096_radix4<>+0x3620(SB)/8, $313  // bitrev[1732] = 313
DATA bitrev_size4096_radix4<>+0x3628(SB)/8, $1337 // bitrev[1733] = 1337
DATA bitrev_size4096_radix4<>+0x3630(SB)/8, $2361 // bitrev[1734] = 2361
DATA bitrev_size4096_radix4<>+0x3638(SB)/8, $3385 // bitrev[1735] = 3385
DATA bitrev_size4096_radix4<>+0x3640(SB)/8, $569  // bitrev[1736] = 569
DATA bitrev_size4096_radix4<>+0x3648(SB)/8, $1593 // bitrev[1737] = 1593
DATA bitrev_size4096_radix4<>+0x3650(SB)/8, $2617 // bitrev[1738] = 2617
DATA bitrev_size4096_radix4<>+0x3658(SB)/8, $3641 // bitrev[1739] = 3641
DATA bitrev_size4096_radix4<>+0x3660(SB)/8, $825  // bitrev[1740] = 825
DATA bitrev_size4096_radix4<>+0x3668(SB)/8, $1849 // bitrev[1741] = 1849
DATA bitrev_size4096_radix4<>+0x3670(SB)/8, $2873 // bitrev[1742] = 2873
DATA bitrev_size4096_radix4<>+0x3678(SB)/8, $3897 // bitrev[1743] = 3897
DATA bitrev_size4096_radix4<>+0x3680(SB)/8, $121  // bitrev[1744] = 121
DATA bitrev_size4096_radix4<>+0x3688(SB)/8, $1145 // bitrev[1745] = 1145
DATA bitrev_size4096_radix4<>+0x3690(SB)/8, $2169 // bitrev[1746] = 2169
DATA bitrev_size4096_radix4<>+0x3698(SB)/8, $3193 // bitrev[1747] = 3193
DATA bitrev_size4096_radix4<>+0x36a0(SB)/8, $377  // bitrev[1748] = 377
DATA bitrev_size4096_radix4<>+0x36a8(SB)/8, $1401 // bitrev[1749] = 1401
DATA bitrev_size4096_radix4<>+0x36b0(SB)/8, $2425 // bitrev[1750] = 2425
DATA bitrev_size4096_radix4<>+0x36b8(SB)/8, $3449 // bitrev[1751] = 3449
DATA bitrev_size4096_radix4<>+0x36c0(SB)/8, $633  // bitrev[1752] = 633
DATA bitrev_size4096_radix4<>+0x36c8(SB)/8, $1657 // bitrev[1753] = 1657
DATA bitrev_size4096_radix4<>+0x36d0(SB)/8, $2681 // bitrev[1754] = 2681
DATA bitrev_size4096_radix4<>+0x36d8(SB)/8, $3705 // bitrev[1755] = 3705
DATA bitrev_size4096_radix4<>+0x36e0(SB)/8, $889  // bitrev[1756] = 889
DATA bitrev_size4096_radix4<>+0x36e8(SB)/8, $1913 // bitrev[1757] = 1913
DATA bitrev_size4096_radix4<>+0x36f0(SB)/8, $2937 // bitrev[1758] = 2937
DATA bitrev_size4096_radix4<>+0x36f8(SB)/8, $3961 // bitrev[1759] = 3961
DATA bitrev_size4096_radix4<>+0x3700(SB)/8, $185  // bitrev[1760] = 185
DATA bitrev_size4096_radix4<>+0x3708(SB)/8, $1209 // bitrev[1761] = 1209
DATA bitrev_size4096_radix4<>+0x3710(SB)/8, $2233 // bitrev[1762] = 2233
DATA bitrev_size4096_radix4<>+0x3718(SB)/8, $3257 // bitrev[1763] = 3257
DATA bitrev_size4096_radix4<>+0x3720(SB)/8, $441  // bitrev[1764] = 441
DATA bitrev_size4096_radix4<>+0x3728(SB)/8, $1465 // bitrev[1765] = 1465
DATA bitrev_size4096_radix4<>+0x3730(SB)/8, $2489 // bitrev[1766] = 2489
DATA bitrev_size4096_radix4<>+0x3738(SB)/8, $3513 // bitrev[1767] = 3513
DATA bitrev_size4096_radix4<>+0x3740(SB)/8, $697  // bitrev[1768] = 697
DATA bitrev_size4096_radix4<>+0x3748(SB)/8, $1721 // bitrev[1769] = 1721
DATA bitrev_size4096_radix4<>+0x3750(SB)/8, $2745 // bitrev[1770] = 2745
DATA bitrev_size4096_radix4<>+0x3758(SB)/8, $3769 // bitrev[1771] = 3769
DATA bitrev_size4096_radix4<>+0x3760(SB)/8, $953  // bitrev[1772] = 953
DATA bitrev_size4096_radix4<>+0x3768(SB)/8, $1977 // bitrev[1773] = 1977
DATA bitrev_size4096_radix4<>+0x3770(SB)/8, $3001 // bitrev[1774] = 3001
DATA bitrev_size4096_radix4<>+0x3778(SB)/8, $4025 // bitrev[1775] = 4025
DATA bitrev_size4096_radix4<>+0x3780(SB)/8, $249  // bitrev[1776] = 249
DATA bitrev_size4096_radix4<>+0x3788(SB)/8, $1273 // bitrev[1777] = 1273
DATA bitrev_size4096_radix4<>+0x3790(SB)/8, $2297 // bitrev[1778] = 2297
DATA bitrev_size4096_radix4<>+0x3798(SB)/8, $3321 // bitrev[1779] = 3321
DATA bitrev_size4096_radix4<>+0x37a0(SB)/8, $505  // bitrev[1780] = 505
DATA bitrev_size4096_radix4<>+0x37a8(SB)/8, $1529 // bitrev[1781] = 1529
DATA bitrev_size4096_radix4<>+0x37b0(SB)/8, $2553 // bitrev[1782] = 2553
DATA bitrev_size4096_radix4<>+0x37b8(SB)/8, $3577 // bitrev[1783] = 3577
DATA bitrev_size4096_radix4<>+0x37c0(SB)/8, $761  // bitrev[1784] = 761
DATA bitrev_size4096_radix4<>+0x37c8(SB)/8, $1785 // bitrev[1785] = 1785
DATA bitrev_size4096_radix4<>+0x37d0(SB)/8, $2809 // bitrev[1786] = 2809
DATA bitrev_size4096_radix4<>+0x37d8(SB)/8, $3833 // bitrev[1787] = 3833
DATA bitrev_size4096_radix4<>+0x37e0(SB)/8, $1017 // bitrev[1788] = 1017
DATA bitrev_size4096_radix4<>+0x37e8(SB)/8, $2041 // bitrev[1789] = 2041
DATA bitrev_size4096_radix4<>+0x37f0(SB)/8, $3065 // bitrev[1790] = 3065
DATA bitrev_size4096_radix4<>+0x37f8(SB)/8, $4089 // bitrev[1791] = 4089
DATA bitrev_size4096_radix4<>+0x3800(SB)/8, $13   // bitrev[1792] = 13
DATA bitrev_size4096_radix4<>+0x3808(SB)/8, $1037 // bitrev[1793] = 1037
DATA bitrev_size4096_radix4<>+0x3810(SB)/8, $2061 // bitrev[1794] = 2061
DATA bitrev_size4096_radix4<>+0x3818(SB)/8, $3085 // bitrev[1795] = 3085
DATA bitrev_size4096_radix4<>+0x3820(SB)/8, $269  // bitrev[1796] = 269
DATA bitrev_size4096_radix4<>+0x3828(SB)/8, $1293 // bitrev[1797] = 1293
DATA bitrev_size4096_radix4<>+0x3830(SB)/8, $2317 // bitrev[1798] = 2317
DATA bitrev_size4096_radix4<>+0x3838(SB)/8, $3341 // bitrev[1799] = 3341
DATA bitrev_size4096_radix4<>+0x3840(SB)/8, $525  // bitrev[1800] = 525
DATA bitrev_size4096_radix4<>+0x3848(SB)/8, $1549 // bitrev[1801] = 1549
DATA bitrev_size4096_radix4<>+0x3850(SB)/8, $2573 // bitrev[1802] = 2573
DATA bitrev_size4096_radix4<>+0x3858(SB)/8, $3597 // bitrev[1803] = 3597
DATA bitrev_size4096_radix4<>+0x3860(SB)/8, $781  // bitrev[1804] = 781
DATA bitrev_size4096_radix4<>+0x3868(SB)/8, $1805 // bitrev[1805] = 1805
DATA bitrev_size4096_radix4<>+0x3870(SB)/8, $2829 // bitrev[1806] = 2829
DATA bitrev_size4096_radix4<>+0x3878(SB)/8, $3853 // bitrev[1807] = 3853
DATA bitrev_size4096_radix4<>+0x3880(SB)/8, $77   // bitrev[1808] = 77
DATA bitrev_size4096_radix4<>+0x3888(SB)/8, $1101 // bitrev[1809] = 1101
DATA bitrev_size4096_radix4<>+0x3890(SB)/8, $2125 // bitrev[1810] = 2125
DATA bitrev_size4096_radix4<>+0x3898(SB)/8, $3149 // bitrev[1811] = 3149
DATA bitrev_size4096_radix4<>+0x38a0(SB)/8, $333  // bitrev[1812] = 333
DATA bitrev_size4096_radix4<>+0x38a8(SB)/8, $1357 // bitrev[1813] = 1357
DATA bitrev_size4096_radix4<>+0x38b0(SB)/8, $2381 // bitrev[1814] = 2381
DATA bitrev_size4096_radix4<>+0x38b8(SB)/8, $3405 // bitrev[1815] = 3405
DATA bitrev_size4096_radix4<>+0x38c0(SB)/8, $589  // bitrev[1816] = 589
DATA bitrev_size4096_radix4<>+0x38c8(SB)/8, $1613 // bitrev[1817] = 1613
DATA bitrev_size4096_radix4<>+0x38d0(SB)/8, $2637 // bitrev[1818] = 2637
DATA bitrev_size4096_radix4<>+0x38d8(SB)/8, $3661 // bitrev[1819] = 3661
DATA bitrev_size4096_radix4<>+0x38e0(SB)/8, $845  // bitrev[1820] = 845
DATA bitrev_size4096_radix4<>+0x38e8(SB)/8, $1869 // bitrev[1821] = 1869
DATA bitrev_size4096_radix4<>+0x38f0(SB)/8, $2893 // bitrev[1822] = 2893
DATA bitrev_size4096_radix4<>+0x38f8(SB)/8, $3917 // bitrev[1823] = 3917
DATA bitrev_size4096_radix4<>+0x3900(SB)/8, $141  // bitrev[1824] = 141
DATA bitrev_size4096_radix4<>+0x3908(SB)/8, $1165 // bitrev[1825] = 1165
DATA bitrev_size4096_radix4<>+0x3910(SB)/8, $2189 // bitrev[1826] = 2189
DATA bitrev_size4096_radix4<>+0x3918(SB)/8, $3213 // bitrev[1827] = 3213
DATA bitrev_size4096_radix4<>+0x3920(SB)/8, $397  // bitrev[1828] = 397
DATA bitrev_size4096_radix4<>+0x3928(SB)/8, $1421 // bitrev[1829] = 1421
DATA bitrev_size4096_radix4<>+0x3930(SB)/8, $2445 // bitrev[1830] = 2445
DATA bitrev_size4096_radix4<>+0x3938(SB)/8, $3469 // bitrev[1831] = 3469
DATA bitrev_size4096_radix4<>+0x3940(SB)/8, $653  // bitrev[1832] = 653
DATA bitrev_size4096_radix4<>+0x3948(SB)/8, $1677 // bitrev[1833] = 1677
DATA bitrev_size4096_radix4<>+0x3950(SB)/8, $2701 // bitrev[1834] = 2701
DATA bitrev_size4096_radix4<>+0x3958(SB)/8, $3725 // bitrev[1835] = 3725
DATA bitrev_size4096_radix4<>+0x3960(SB)/8, $909  // bitrev[1836] = 909
DATA bitrev_size4096_radix4<>+0x3968(SB)/8, $1933 // bitrev[1837] = 1933
DATA bitrev_size4096_radix4<>+0x3970(SB)/8, $2957 // bitrev[1838] = 2957
DATA bitrev_size4096_radix4<>+0x3978(SB)/8, $3981 // bitrev[1839] = 3981
DATA bitrev_size4096_radix4<>+0x3980(SB)/8, $205  // bitrev[1840] = 205
DATA bitrev_size4096_radix4<>+0x3988(SB)/8, $1229 // bitrev[1841] = 1229
DATA bitrev_size4096_radix4<>+0x3990(SB)/8, $2253 // bitrev[1842] = 2253
DATA bitrev_size4096_radix4<>+0x3998(SB)/8, $3277 // bitrev[1843] = 3277
DATA bitrev_size4096_radix4<>+0x39a0(SB)/8, $461  // bitrev[1844] = 461
DATA bitrev_size4096_radix4<>+0x39a8(SB)/8, $1485 // bitrev[1845] = 1485
DATA bitrev_size4096_radix4<>+0x39b0(SB)/8, $2509 // bitrev[1846] = 2509
DATA bitrev_size4096_radix4<>+0x39b8(SB)/8, $3533 // bitrev[1847] = 3533
DATA bitrev_size4096_radix4<>+0x39c0(SB)/8, $717  // bitrev[1848] = 717
DATA bitrev_size4096_radix4<>+0x39c8(SB)/8, $1741 // bitrev[1849] = 1741
DATA bitrev_size4096_radix4<>+0x39d0(SB)/8, $2765 // bitrev[1850] = 2765
DATA bitrev_size4096_radix4<>+0x39d8(SB)/8, $3789 // bitrev[1851] = 3789
DATA bitrev_size4096_radix4<>+0x39e0(SB)/8, $973  // bitrev[1852] = 973
DATA bitrev_size4096_radix4<>+0x39e8(SB)/8, $1997 // bitrev[1853] = 1997
DATA bitrev_size4096_radix4<>+0x39f0(SB)/8, $3021 // bitrev[1854] = 3021
DATA bitrev_size4096_radix4<>+0x39f8(SB)/8, $4045 // bitrev[1855] = 4045
DATA bitrev_size4096_radix4<>+0x3a00(SB)/8, $29   // bitrev[1856] = 29
DATA bitrev_size4096_radix4<>+0x3a08(SB)/8, $1053 // bitrev[1857] = 1053
DATA bitrev_size4096_radix4<>+0x3a10(SB)/8, $2077 // bitrev[1858] = 2077
DATA bitrev_size4096_radix4<>+0x3a18(SB)/8, $3101 // bitrev[1859] = 3101
DATA bitrev_size4096_radix4<>+0x3a20(SB)/8, $285  // bitrev[1860] = 285
DATA bitrev_size4096_radix4<>+0x3a28(SB)/8, $1309 // bitrev[1861] = 1309
DATA bitrev_size4096_radix4<>+0x3a30(SB)/8, $2333 // bitrev[1862] = 2333
DATA bitrev_size4096_radix4<>+0x3a38(SB)/8, $3357 // bitrev[1863] = 3357
DATA bitrev_size4096_radix4<>+0x3a40(SB)/8, $541  // bitrev[1864] = 541
DATA bitrev_size4096_radix4<>+0x3a48(SB)/8, $1565 // bitrev[1865] = 1565
DATA bitrev_size4096_radix4<>+0x3a50(SB)/8, $2589 // bitrev[1866] = 2589
DATA bitrev_size4096_radix4<>+0x3a58(SB)/8, $3613 // bitrev[1867] = 3613
DATA bitrev_size4096_radix4<>+0x3a60(SB)/8, $797  // bitrev[1868] = 797
DATA bitrev_size4096_radix4<>+0x3a68(SB)/8, $1821 // bitrev[1869] = 1821
DATA bitrev_size4096_radix4<>+0x3a70(SB)/8, $2845 // bitrev[1870] = 2845
DATA bitrev_size4096_radix4<>+0x3a78(SB)/8, $3869 // bitrev[1871] = 3869
DATA bitrev_size4096_radix4<>+0x3a80(SB)/8, $93   // bitrev[1872] = 93
DATA bitrev_size4096_radix4<>+0x3a88(SB)/8, $1117 // bitrev[1873] = 1117
DATA bitrev_size4096_radix4<>+0x3a90(SB)/8, $2141 // bitrev[1874] = 2141
DATA bitrev_size4096_radix4<>+0x3a98(SB)/8, $3165 // bitrev[1875] = 3165
DATA bitrev_size4096_radix4<>+0x3aa0(SB)/8, $349  // bitrev[1876] = 349
DATA bitrev_size4096_radix4<>+0x3aa8(SB)/8, $1373 // bitrev[1877] = 1373
DATA bitrev_size4096_radix4<>+0x3ab0(SB)/8, $2397 // bitrev[1878] = 2397
DATA bitrev_size4096_radix4<>+0x3ab8(SB)/8, $3421 // bitrev[1879] = 3421
DATA bitrev_size4096_radix4<>+0x3ac0(SB)/8, $605  // bitrev[1880] = 605
DATA bitrev_size4096_radix4<>+0x3ac8(SB)/8, $1629 // bitrev[1881] = 1629
DATA bitrev_size4096_radix4<>+0x3ad0(SB)/8, $2653 // bitrev[1882] = 2653
DATA bitrev_size4096_radix4<>+0x3ad8(SB)/8, $3677 // bitrev[1883] = 3677
DATA bitrev_size4096_radix4<>+0x3ae0(SB)/8, $861  // bitrev[1884] = 861
DATA bitrev_size4096_radix4<>+0x3ae8(SB)/8, $1885 // bitrev[1885] = 1885
DATA bitrev_size4096_radix4<>+0x3af0(SB)/8, $2909 // bitrev[1886] = 2909
DATA bitrev_size4096_radix4<>+0x3af8(SB)/8, $3933 // bitrev[1887] = 3933
DATA bitrev_size4096_radix4<>+0x3b00(SB)/8, $157  // bitrev[1888] = 157
DATA bitrev_size4096_radix4<>+0x3b08(SB)/8, $1181 // bitrev[1889] = 1181
DATA bitrev_size4096_radix4<>+0x3b10(SB)/8, $2205 // bitrev[1890] = 2205
DATA bitrev_size4096_radix4<>+0x3b18(SB)/8, $3229 // bitrev[1891] = 3229
DATA bitrev_size4096_radix4<>+0x3b20(SB)/8, $413  // bitrev[1892] = 413
DATA bitrev_size4096_radix4<>+0x3b28(SB)/8, $1437 // bitrev[1893] = 1437
DATA bitrev_size4096_radix4<>+0x3b30(SB)/8, $2461 // bitrev[1894] = 2461
DATA bitrev_size4096_radix4<>+0x3b38(SB)/8, $3485 // bitrev[1895] = 3485
DATA bitrev_size4096_radix4<>+0x3b40(SB)/8, $669  // bitrev[1896] = 669
DATA bitrev_size4096_radix4<>+0x3b48(SB)/8, $1693 // bitrev[1897] = 1693
DATA bitrev_size4096_radix4<>+0x3b50(SB)/8, $2717 // bitrev[1898] = 2717
DATA bitrev_size4096_radix4<>+0x3b58(SB)/8, $3741 // bitrev[1899] = 3741
DATA bitrev_size4096_radix4<>+0x3b60(SB)/8, $925  // bitrev[1900] = 925
DATA bitrev_size4096_radix4<>+0x3b68(SB)/8, $1949 // bitrev[1901] = 1949
DATA bitrev_size4096_radix4<>+0x3b70(SB)/8, $2973 // bitrev[1902] = 2973
DATA bitrev_size4096_radix4<>+0x3b78(SB)/8, $3997 // bitrev[1903] = 3997
DATA bitrev_size4096_radix4<>+0x3b80(SB)/8, $221  // bitrev[1904] = 221
DATA bitrev_size4096_radix4<>+0x3b88(SB)/8, $1245 // bitrev[1905] = 1245
DATA bitrev_size4096_radix4<>+0x3b90(SB)/8, $2269 // bitrev[1906] = 2269
DATA bitrev_size4096_radix4<>+0x3b98(SB)/8, $3293 // bitrev[1907] = 3293
DATA bitrev_size4096_radix4<>+0x3ba0(SB)/8, $477  // bitrev[1908] = 477
DATA bitrev_size4096_radix4<>+0x3ba8(SB)/8, $1501 // bitrev[1909] = 1501
DATA bitrev_size4096_radix4<>+0x3bb0(SB)/8, $2525 // bitrev[1910] = 2525
DATA bitrev_size4096_radix4<>+0x3bb8(SB)/8, $3549 // bitrev[1911] = 3549
DATA bitrev_size4096_radix4<>+0x3bc0(SB)/8, $733  // bitrev[1912] = 733
DATA bitrev_size4096_radix4<>+0x3bc8(SB)/8, $1757 // bitrev[1913] = 1757
DATA bitrev_size4096_radix4<>+0x3bd0(SB)/8, $2781 // bitrev[1914] = 2781
DATA bitrev_size4096_radix4<>+0x3bd8(SB)/8, $3805 // bitrev[1915] = 3805
DATA bitrev_size4096_radix4<>+0x3be0(SB)/8, $989  // bitrev[1916] = 989
DATA bitrev_size4096_radix4<>+0x3be8(SB)/8, $2013 // bitrev[1917] = 2013
DATA bitrev_size4096_radix4<>+0x3bf0(SB)/8, $3037 // bitrev[1918] = 3037
DATA bitrev_size4096_radix4<>+0x3bf8(SB)/8, $4061 // bitrev[1919] = 4061
DATA bitrev_size4096_radix4<>+0x3c00(SB)/8, $45   // bitrev[1920] = 45
DATA bitrev_size4096_radix4<>+0x3c08(SB)/8, $1069 // bitrev[1921] = 1069
DATA bitrev_size4096_radix4<>+0x3c10(SB)/8, $2093 // bitrev[1922] = 2093
DATA bitrev_size4096_radix4<>+0x3c18(SB)/8, $3117 // bitrev[1923] = 3117
DATA bitrev_size4096_radix4<>+0x3c20(SB)/8, $301  // bitrev[1924] = 301
DATA bitrev_size4096_radix4<>+0x3c28(SB)/8, $1325 // bitrev[1925] = 1325
DATA bitrev_size4096_radix4<>+0x3c30(SB)/8, $2349 // bitrev[1926] = 2349
DATA bitrev_size4096_radix4<>+0x3c38(SB)/8, $3373 // bitrev[1927] = 3373
DATA bitrev_size4096_radix4<>+0x3c40(SB)/8, $557  // bitrev[1928] = 557
DATA bitrev_size4096_radix4<>+0x3c48(SB)/8, $1581 // bitrev[1929] = 1581
DATA bitrev_size4096_radix4<>+0x3c50(SB)/8, $2605 // bitrev[1930] = 2605
DATA bitrev_size4096_radix4<>+0x3c58(SB)/8, $3629 // bitrev[1931] = 3629
DATA bitrev_size4096_radix4<>+0x3c60(SB)/8, $813  // bitrev[1932] = 813
DATA bitrev_size4096_radix4<>+0x3c68(SB)/8, $1837 // bitrev[1933] = 1837
DATA bitrev_size4096_radix4<>+0x3c70(SB)/8, $2861 // bitrev[1934] = 2861
DATA bitrev_size4096_radix4<>+0x3c78(SB)/8, $3885 // bitrev[1935] = 3885
DATA bitrev_size4096_radix4<>+0x3c80(SB)/8, $109  // bitrev[1936] = 109
DATA bitrev_size4096_radix4<>+0x3c88(SB)/8, $1133 // bitrev[1937] = 1133
DATA bitrev_size4096_radix4<>+0x3c90(SB)/8, $2157 // bitrev[1938] = 2157
DATA bitrev_size4096_radix4<>+0x3c98(SB)/8, $3181 // bitrev[1939] = 3181
DATA bitrev_size4096_radix4<>+0x3ca0(SB)/8, $365  // bitrev[1940] = 365
DATA bitrev_size4096_radix4<>+0x3ca8(SB)/8, $1389 // bitrev[1941] = 1389
DATA bitrev_size4096_radix4<>+0x3cb0(SB)/8, $2413 // bitrev[1942] = 2413
DATA bitrev_size4096_radix4<>+0x3cb8(SB)/8, $3437 // bitrev[1943] = 3437
DATA bitrev_size4096_radix4<>+0x3cc0(SB)/8, $621  // bitrev[1944] = 621
DATA bitrev_size4096_radix4<>+0x3cc8(SB)/8, $1645 // bitrev[1945] = 1645
DATA bitrev_size4096_radix4<>+0x3cd0(SB)/8, $2669 // bitrev[1946] = 2669
DATA bitrev_size4096_radix4<>+0x3cd8(SB)/8, $3693 // bitrev[1947] = 3693
DATA bitrev_size4096_radix4<>+0x3ce0(SB)/8, $877  // bitrev[1948] = 877
DATA bitrev_size4096_radix4<>+0x3ce8(SB)/8, $1901 // bitrev[1949] = 1901
DATA bitrev_size4096_radix4<>+0x3cf0(SB)/8, $2925 // bitrev[1950] = 2925
DATA bitrev_size4096_radix4<>+0x3cf8(SB)/8, $3949 // bitrev[1951] = 3949
DATA bitrev_size4096_radix4<>+0x3d00(SB)/8, $173  // bitrev[1952] = 173
DATA bitrev_size4096_radix4<>+0x3d08(SB)/8, $1197 // bitrev[1953] = 1197
DATA bitrev_size4096_radix4<>+0x3d10(SB)/8, $2221 // bitrev[1954] = 2221
DATA bitrev_size4096_radix4<>+0x3d18(SB)/8, $3245 // bitrev[1955] = 3245
DATA bitrev_size4096_radix4<>+0x3d20(SB)/8, $429  // bitrev[1956] = 429
DATA bitrev_size4096_radix4<>+0x3d28(SB)/8, $1453 // bitrev[1957] = 1453
DATA bitrev_size4096_radix4<>+0x3d30(SB)/8, $2477 // bitrev[1958] = 2477
DATA bitrev_size4096_radix4<>+0x3d38(SB)/8, $3501 // bitrev[1959] = 3501
DATA bitrev_size4096_radix4<>+0x3d40(SB)/8, $685  // bitrev[1960] = 685
DATA bitrev_size4096_radix4<>+0x3d48(SB)/8, $1709 // bitrev[1961] = 1709
DATA bitrev_size4096_radix4<>+0x3d50(SB)/8, $2733 // bitrev[1962] = 2733
DATA bitrev_size4096_radix4<>+0x3d58(SB)/8, $3757 // bitrev[1963] = 3757
DATA bitrev_size4096_radix4<>+0x3d60(SB)/8, $941  // bitrev[1964] = 941
DATA bitrev_size4096_radix4<>+0x3d68(SB)/8, $1965 // bitrev[1965] = 1965
DATA bitrev_size4096_radix4<>+0x3d70(SB)/8, $2989 // bitrev[1966] = 2989
DATA bitrev_size4096_radix4<>+0x3d78(SB)/8, $4013 // bitrev[1967] = 4013
DATA bitrev_size4096_radix4<>+0x3d80(SB)/8, $237  // bitrev[1968] = 237
DATA bitrev_size4096_radix4<>+0x3d88(SB)/8, $1261 // bitrev[1969] = 1261
DATA bitrev_size4096_radix4<>+0x3d90(SB)/8, $2285 // bitrev[1970] = 2285
DATA bitrev_size4096_radix4<>+0x3d98(SB)/8, $3309 // bitrev[1971] = 3309
DATA bitrev_size4096_radix4<>+0x3da0(SB)/8, $493  // bitrev[1972] = 493
DATA bitrev_size4096_radix4<>+0x3da8(SB)/8, $1517 // bitrev[1973] = 1517
DATA bitrev_size4096_radix4<>+0x3db0(SB)/8, $2541 // bitrev[1974] = 2541
DATA bitrev_size4096_radix4<>+0x3db8(SB)/8, $3565 // bitrev[1975] = 3565
DATA bitrev_size4096_radix4<>+0x3dc0(SB)/8, $749  // bitrev[1976] = 749
DATA bitrev_size4096_radix4<>+0x3dc8(SB)/8, $1773 // bitrev[1977] = 1773
DATA bitrev_size4096_radix4<>+0x3dd0(SB)/8, $2797 // bitrev[1978] = 2797
DATA bitrev_size4096_radix4<>+0x3dd8(SB)/8, $3821 // bitrev[1979] = 3821
DATA bitrev_size4096_radix4<>+0x3de0(SB)/8, $1005 // bitrev[1980] = 1005
DATA bitrev_size4096_radix4<>+0x3de8(SB)/8, $2029 // bitrev[1981] = 2029
DATA bitrev_size4096_radix4<>+0x3df0(SB)/8, $3053 // bitrev[1982] = 3053
DATA bitrev_size4096_radix4<>+0x3df8(SB)/8, $4077 // bitrev[1983] = 4077
DATA bitrev_size4096_radix4<>+0x3e00(SB)/8, $61   // bitrev[1984] = 61
DATA bitrev_size4096_radix4<>+0x3e08(SB)/8, $1085 // bitrev[1985] = 1085
DATA bitrev_size4096_radix4<>+0x3e10(SB)/8, $2109 // bitrev[1986] = 2109
DATA bitrev_size4096_radix4<>+0x3e18(SB)/8, $3133 // bitrev[1987] = 3133
DATA bitrev_size4096_radix4<>+0x3e20(SB)/8, $317  // bitrev[1988] = 317
DATA bitrev_size4096_radix4<>+0x3e28(SB)/8, $1341 // bitrev[1989] = 1341
DATA bitrev_size4096_radix4<>+0x3e30(SB)/8, $2365 // bitrev[1990] = 2365
DATA bitrev_size4096_radix4<>+0x3e38(SB)/8, $3389 // bitrev[1991] = 3389
DATA bitrev_size4096_radix4<>+0x3e40(SB)/8, $573  // bitrev[1992] = 573
DATA bitrev_size4096_radix4<>+0x3e48(SB)/8, $1597 // bitrev[1993] = 1597
DATA bitrev_size4096_radix4<>+0x3e50(SB)/8, $2621 // bitrev[1994] = 2621
DATA bitrev_size4096_radix4<>+0x3e58(SB)/8, $3645 // bitrev[1995] = 3645
DATA bitrev_size4096_radix4<>+0x3e60(SB)/8, $829  // bitrev[1996] = 829
DATA bitrev_size4096_radix4<>+0x3e68(SB)/8, $1853 // bitrev[1997] = 1853
DATA bitrev_size4096_radix4<>+0x3e70(SB)/8, $2877 // bitrev[1998] = 2877
DATA bitrev_size4096_radix4<>+0x3e78(SB)/8, $3901 // bitrev[1999] = 3901
DATA bitrev_size4096_radix4<>+0x3e80(SB)/8, $125  // bitrev[2000] = 125
DATA bitrev_size4096_radix4<>+0x3e88(SB)/8, $1149 // bitrev[2001] = 1149
DATA bitrev_size4096_radix4<>+0x3e90(SB)/8, $2173 // bitrev[2002] = 2173
DATA bitrev_size4096_radix4<>+0x3e98(SB)/8, $3197 // bitrev[2003] = 3197
DATA bitrev_size4096_radix4<>+0x3ea0(SB)/8, $381  // bitrev[2004] = 381
DATA bitrev_size4096_radix4<>+0x3ea8(SB)/8, $1405 // bitrev[2005] = 1405
DATA bitrev_size4096_radix4<>+0x3eb0(SB)/8, $2429 // bitrev[2006] = 2429
DATA bitrev_size4096_radix4<>+0x3eb8(SB)/8, $3453 // bitrev[2007] = 3453
DATA bitrev_size4096_radix4<>+0x3ec0(SB)/8, $637  // bitrev[2008] = 637
DATA bitrev_size4096_radix4<>+0x3ec8(SB)/8, $1661 // bitrev[2009] = 1661
DATA bitrev_size4096_radix4<>+0x3ed0(SB)/8, $2685 // bitrev[2010] = 2685
DATA bitrev_size4096_radix4<>+0x3ed8(SB)/8, $3709 // bitrev[2011] = 3709
DATA bitrev_size4096_radix4<>+0x3ee0(SB)/8, $893  // bitrev[2012] = 893
DATA bitrev_size4096_radix4<>+0x3ee8(SB)/8, $1917 // bitrev[2013] = 1917
DATA bitrev_size4096_radix4<>+0x3ef0(SB)/8, $2941 // bitrev[2014] = 2941
DATA bitrev_size4096_radix4<>+0x3ef8(SB)/8, $3965 // bitrev[2015] = 3965
DATA bitrev_size4096_radix4<>+0x3f00(SB)/8, $189  // bitrev[2016] = 189
DATA bitrev_size4096_radix4<>+0x3f08(SB)/8, $1213 // bitrev[2017] = 1213
DATA bitrev_size4096_radix4<>+0x3f10(SB)/8, $2237 // bitrev[2018] = 2237
DATA bitrev_size4096_radix4<>+0x3f18(SB)/8, $3261 // bitrev[2019] = 3261
DATA bitrev_size4096_radix4<>+0x3f20(SB)/8, $445  // bitrev[2020] = 445
DATA bitrev_size4096_radix4<>+0x3f28(SB)/8, $1469 // bitrev[2021] = 1469
DATA bitrev_size4096_radix4<>+0x3f30(SB)/8, $2493 // bitrev[2022] = 2493
DATA bitrev_size4096_radix4<>+0x3f38(SB)/8, $3517 // bitrev[2023] = 3517
DATA bitrev_size4096_radix4<>+0x3f40(SB)/8, $701  // bitrev[2024] = 701
DATA bitrev_size4096_radix4<>+0x3f48(SB)/8, $1725 // bitrev[2025] = 1725
DATA bitrev_size4096_radix4<>+0x3f50(SB)/8, $2749 // bitrev[2026] = 2749
DATA bitrev_size4096_radix4<>+0x3f58(SB)/8, $3773 // bitrev[2027] = 3773
DATA bitrev_size4096_radix4<>+0x3f60(SB)/8, $957  // bitrev[2028] = 957
DATA bitrev_size4096_radix4<>+0x3f68(SB)/8, $1981 // bitrev[2029] = 1981
DATA bitrev_size4096_radix4<>+0x3f70(SB)/8, $3005 // bitrev[2030] = 3005
DATA bitrev_size4096_radix4<>+0x3f78(SB)/8, $4029 // bitrev[2031] = 4029
DATA bitrev_size4096_radix4<>+0x3f80(SB)/8, $253  // bitrev[2032] = 253
DATA bitrev_size4096_radix4<>+0x3f88(SB)/8, $1277 // bitrev[2033] = 1277
DATA bitrev_size4096_radix4<>+0x3f90(SB)/8, $2301 // bitrev[2034] = 2301
DATA bitrev_size4096_radix4<>+0x3f98(SB)/8, $3325 // bitrev[2035] = 3325
DATA bitrev_size4096_radix4<>+0x3fa0(SB)/8, $509  // bitrev[2036] = 509
DATA bitrev_size4096_radix4<>+0x3fa8(SB)/8, $1533 // bitrev[2037] = 1533
DATA bitrev_size4096_radix4<>+0x3fb0(SB)/8, $2557 // bitrev[2038] = 2557
DATA bitrev_size4096_radix4<>+0x3fb8(SB)/8, $3581 // bitrev[2039] = 3581
DATA bitrev_size4096_radix4<>+0x3fc0(SB)/8, $765  // bitrev[2040] = 765
DATA bitrev_size4096_radix4<>+0x3fc8(SB)/8, $1789 // bitrev[2041] = 1789
DATA bitrev_size4096_radix4<>+0x3fd0(SB)/8, $2813 // bitrev[2042] = 2813
DATA bitrev_size4096_radix4<>+0x3fd8(SB)/8, $3837 // bitrev[2043] = 3837
DATA bitrev_size4096_radix4<>+0x3fe0(SB)/8, $1021 // bitrev[2044] = 1021
DATA bitrev_size4096_radix4<>+0x3fe8(SB)/8, $2045 // bitrev[2045] = 2045
DATA bitrev_size4096_radix4<>+0x3ff0(SB)/8, $3069 // bitrev[2046] = 3069
DATA bitrev_size4096_radix4<>+0x3ff8(SB)/8, $4093 // bitrev[2047] = 4093
DATA bitrev_size4096_radix4<>+0x4000(SB)/8, $2    // bitrev[2048] = 2
DATA bitrev_size4096_radix4<>+0x4008(SB)/8, $1026 // bitrev[2049] = 1026
DATA bitrev_size4096_radix4<>+0x4010(SB)/8, $2050 // bitrev[2050] = 2050
DATA bitrev_size4096_radix4<>+0x4018(SB)/8, $3074 // bitrev[2051] = 3074
DATA bitrev_size4096_radix4<>+0x4020(SB)/8, $258  // bitrev[2052] = 258
DATA bitrev_size4096_radix4<>+0x4028(SB)/8, $1282 // bitrev[2053] = 1282
DATA bitrev_size4096_radix4<>+0x4030(SB)/8, $2306 // bitrev[2054] = 2306
DATA bitrev_size4096_radix4<>+0x4038(SB)/8, $3330 // bitrev[2055] = 3330
DATA bitrev_size4096_radix4<>+0x4040(SB)/8, $514  // bitrev[2056] = 514
DATA bitrev_size4096_radix4<>+0x4048(SB)/8, $1538 // bitrev[2057] = 1538
DATA bitrev_size4096_radix4<>+0x4050(SB)/8, $2562 // bitrev[2058] = 2562
DATA bitrev_size4096_radix4<>+0x4058(SB)/8, $3586 // bitrev[2059] = 3586
DATA bitrev_size4096_radix4<>+0x4060(SB)/8, $770  // bitrev[2060] = 770
DATA bitrev_size4096_radix4<>+0x4068(SB)/8, $1794 // bitrev[2061] = 1794
DATA bitrev_size4096_radix4<>+0x4070(SB)/8, $2818 // bitrev[2062] = 2818
DATA bitrev_size4096_radix4<>+0x4078(SB)/8, $3842 // bitrev[2063] = 3842
DATA bitrev_size4096_radix4<>+0x4080(SB)/8, $66   // bitrev[2064] = 66
DATA bitrev_size4096_radix4<>+0x4088(SB)/8, $1090 // bitrev[2065] = 1090
DATA bitrev_size4096_radix4<>+0x4090(SB)/8, $2114 // bitrev[2066] = 2114
DATA bitrev_size4096_radix4<>+0x4098(SB)/8, $3138 // bitrev[2067] = 3138
DATA bitrev_size4096_radix4<>+0x40a0(SB)/8, $322  // bitrev[2068] = 322
DATA bitrev_size4096_radix4<>+0x40a8(SB)/8, $1346 // bitrev[2069] = 1346
DATA bitrev_size4096_radix4<>+0x40b0(SB)/8, $2370 // bitrev[2070] = 2370
DATA bitrev_size4096_radix4<>+0x40b8(SB)/8, $3394 // bitrev[2071] = 3394
DATA bitrev_size4096_radix4<>+0x40c0(SB)/8, $578  // bitrev[2072] = 578
DATA bitrev_size4096_radix4<>+0x40c8(SB)/8, $1602 // bitrev[2073] = 1602
DATA bitrev_size4096_radix4<>+0x40d0(SB)/8, $2626 // bitrev[2074] = 2626
DATA bitrev_size4096_radix4<>+0x40d8(SB)/8, $3650 // bitrev[2075] = 3650
DATA bitrev_size4096_radix4<>+0x40e0(SB)/8, $834  // bitrev[2076] = 834
DATA bitrev_size4096_radix4<>+0x40e8(SB)/8, $1858 // bitrev[2077] = 1858
DATA bitrev_size4096_radix4<>+0x40f0(SB)/8, $2882 // bitrev[2078] = 2882
DATA bitrev_size4096_radix4<>+0x40f8(SB)/8, $3906 // bitrev[2079] = 3906
DATA bitrev_size4096_radix4<>+0x4100(SB)/8, $130  // bitrev[2080] = 130
DATA bitrev_size4096_radix4<>+0x4108(SB)/8, $1154 // bitrev[2081] = 1154
DATA bitrev_size4096_radix4<>+0x4110(SB)/8, $2178 // bitrev[2082] = 2178
DATA bitrev_size4096_radix4<>+0x4118(SB)/8, $3202 // bitrev[2083] = 3202
DATA bitrev_size4096_radix4<>+0x4120(SB)/8, $386  // bitrev[2084] = 386
DATA bitrev_size4096_radix4<>+0x4128(SB)/8, $1410 // bitrev[2085] = 1410
DATA bitrev_size4096_radix4<>+0x4130(SB)/8, $2434 // bitrev[2086] = 2434
DATA bitrev_size4096_radix4<>+0x4138(SB)/8, $3458 // bitrev[2087] = 3458
DATA bitrev_size4096_radix4<>+0x4140(SB)/8, $642  // bitrev[2088] = 642
DATA bitrev_size4096_radix4<>+0x4148(SB)/8, $1666 // bitrev[2089] = 1666
DATA bitrev_size4096_radix4<>+0x4150(SB)/8, $2690 // bitrev[2090] = 2690
DATA bitrev_size4096_radix4<>+0x4158(SB)/8, $3714 // bitrev[2091] = 3714
DATA bitrev_size4096_radix4<>+0x4160(SB)/8, $898  // bitrev[2092] = 898
DATA bitrev_size4096_radix4<>+0x4168(SB)/8, $1922 // bitrev[2093] = 1922
DATA bitrev_size4096_radix4<>+0x4170(SB)/8, $2946 // bitrev[2094] = 2946
DATA bitrev_size4096_radix4<>+0x4178(SB)/8, $3970 // bitrev[2095] = 3970
DATA bitrev_size4096_radix4<>+0x4180(SB)/8, $194  // bitrev[2096] = 194
DATA bitrev_size4096_radix4<>+0x4188(SB)/8, $1218 // bitrev[2097] = 1218
DATA bitrev_size4096_radix4<>+0x4190(SB)/8, $2242 // bitrev[2098] = 2242
DATA bitrev_size4096_radix4<>+0x4198(SB)/8, $3266 // bitrev[2099] = 3266
DATA bitrev_size4096_radix4<>+0x41a0(SB)/8, $450  // bitrev[2100] = 450
DATA bitrev_size4096_radix4<>+0x41a8(SB)/8, $1474 // bitrev[2101] = 1474
DATA bitrev_size4096_radix4<>+0x41b0(SB)/8, $2498 // bitrev[2102] = 2498
DATA bitrev_size4096_radix4<>+0x41b8(SB)/8, $3522 // bitrev[2103] = 3522
DATA bitrev_size4096_radix4<>+0x41c0(SB)/8, $706  // bitrev[2104] = 706
DATA bitrev_size4096_radix4<>+0x41c8(SB)/8, $1730 // bitrev[2105] = 1730
DATA bitrev_size4096_radix4<>+0x41d0(SB)/8, $2754 // bitrev[2106] = 2754
DATA bitrev_size4096_radix4<>+0x41d8(SB)/8, $3778 // bitrev[2107] = 3778
DATA bitrev_size4096_radix4<>+0x41e0(SB)/8, $962  // bitrev[2108] = 962
DATA bitrev_size4096_radix4<>+0x41e8(SB)/8, $1986 // bitrev[2109] = 1986
DATA bitrev_size4096_radix4<>+0x41f0(SB)/8, $3010 // bitrev[2110] = 3010
DATA bitrev_size4096_radix4<>+0x41f8(SB)/8, $4034 // bitrev[2111] = 4034
DATA bitrev_size4096_radix4<>+0x4200(SB)/8, $18   // bitrev[2112] = 18
DATA bitrev_size4096_radix4<>+0x4208(SB)/8, $1042 // bitrev[2113] = 1042
DATA bitrev_size4096_radix4<>+0x4210(SB)/8, $2066 // bitrev[2114] = 2066
DATA bitrev_size4096_radix4<>+0x4218(SB)/8, $3090 // bitrev[2115] = 3090
DATA bitrev_size4096_radix4<>+0x4220(SB)/8, $274  // bitrev[2116] = 274
DATA bitrev_size4096_radix4<>+0x4228(SB)/8, $1298 // bitrev[2117] = 1298
DATA bitrev_size4096_radix4<>+0x4230(SB)/8, $2322 // bitrev[2118] = 2322
DATA bitrev_size4096_radix4<>+0x4238(SB)/8, $3346 // bitrev[2119] = 3346
DATA bitrev_size4096_radix4<>+0x4240(SB)/8, $530  // bitrev[2120] = 530
DATA bitrev_size4096_radix4<>+0x4248(SB)/8, $1554 // bitrev[2121] = 1554
DATA bitrev_size4096_radix4<>+0x4250(SB)/8, $2578 // bitrev[2122] = 2578
DATA bitrev_size4096_radix4<>+0x4258(SB)/8, $3602 // bitrev[2123] = 3602
DATA bitrev_size4096_radix4<>+0x4260(SB)/8, $786  // bitrev[2124] = 786
DATA bitrev_size4096_radix4<>+0x4268(SB)/8, $1810 // bitrev[2125] = 1810
DATA bitrev_size4096_radix4<>+0x4270(SB)/8, $2834 // bitrev[2126] = 2834
DATA bitrev_size4096_radix4<>+0x4278(SB)/8, $3858 // bitrev[2127] = 3858
DATA bitrev_size4096_radix4<>+0x4280(SB)/8, $82   // bitrev[2128] = 82
DATA bitrev_size4096_radix4<>+0x4288(SB)/8, $1106 // bitrev[2129] = 1106
DATA bitrev_size4096_radix4<>+0x4290(SB)/8, $2130 // bitrev[2130] = 2130
DATA bitrev_size4096_radix4<>+0x4298(SB)/8, $3154 // bitrev[2131] = 3154
DATA bitrev_size4096_radix4<>+0x42a0(SB)/8, $338  // bitrev[2132] = 338
DATA bitrev_size4096_radix4<>+0x42a8(SB)/8, $1362 // bitrev[2133] = 1362
DATA bitrev_size4096_radix4<>+0x42b0(SB)/8, $2386 // bitrev[2134] = 2386
DATA bitrev_size4096_radix4<>+0x42b8(SB)/8, $3410 // bitrev[2135] = 3410
DATA bitrev_size4096_radix4<>+0x42c0(SB)/8, $594  // bitrev[2136] = 594
DATA bitrev_size4096_radix4<>+0x42c8(SB)/8, $1618 // bitrev[2137] = 1618
DATA bitrev_size4096_radix4<>+0x42d0(SB)/8, $2642 // bitrev[2138] = 2642
DATA bitrev_size4096_radix4<>+0x42d8(SB)/8, $3666 // bitrev[2139] = 3666
DATA bitrev_size4096_radix4<>+0x42e0(SB)/8, $850  // bitrev[2140] = 850
DATA bitrev_size4096_radix4<>+0x42e8(SB)/8, $1874 // bitrev[2141] = 1874
DATA bitrev_size4096_radix4<>+0x42f0(SB)/8, $2898 // bitrev[2142] = 2898
DATA bitrev_size4096_radix4<>+0x42f8(SB)/8, $3922 // bitrev[2143] = 3922
DATA bitrev_size4096_radix4<>+0x4300(SB)/8, $146  // bitrev[2144] = 146
DATA bitrev_size4096_radix4<>+0x4308(SB)/8, $1170 // bitrev[2145] = 1170
DATA bitrev_size4096_radix4<>+0x4310(SB)/8, $2194 // bitrev[2146] = 2194
DATA bitrev_size4096_radix4<>+0x4318(SB)/8, $3218 // bitrev[2147] = 3218
DATA bitrev_size4096_radix4<>+0x4320(SB)/8, $402  // bitrev[2148] = 402
DATA bitrev_size4096_radix4<>+0x4328(SB)/8, $1426 // bitrev[2149] = 1426
DATA bitrev_size4096_radix4<>+0x4330(SB)/8, $2450 // bitrev[2150] = 2450
DATA bitrev_size4096_radix4<>+0x4338(SB)/8, $3474 // bitrev[2151] = 3474
DATA bitrev_size4096_radix4<>+0x4340(SB)/8, $658  // bitrev[2152] = 658
DATA bitrev_size4096_radix4<>+0x4348(SB)/8, $1682 // bitrev[2153] = 1682
DATA bitrev_size4096_radix4<>+0x4350(SB)/8, $2706 // bitrev[2154] = 2706
DATA bitrev_size4096_radix4<>+0x4358(SB)/8, $3730 // bitrev[2155] = 3730
DATA bitrev_size4096_radix4<>+0x4360(SB)/8, $914  // bitrev[2156] = 914
DATA bitrev_size4096_radix4<>+0x4368(SB)/8, $1938 // bitrev[2157] = 1938
DATA bitrev_size4096_radix4<>+0x4370(SB)/8, $2962 // bitrev[2158] = 2962
DATA bitrev_size4096_radix4<>+0x4378(SB)/8, $3986 // bitrev[2159] = 3986
DATA bitrev_size4096_radix4<>+0x4380(SB)/8, $210  // bitrev[2160] = 210
DATA bitrev_size4096_radix4<>+0x4388(SB)/8, $1234 // bitrev[2161] = 1234
DATA bitrev_size4096_radix4<>+0x4390(SB)/8, $2258 // bitrev[2162] = 2258
DATA bitrev_size4096_radix4<>+0x4398(SB)/8, $3282 // bitrev[2163] = 3282
DATA bitrev_size4096_radix4<>+0x43a0(SB)/8, $466  // bitrev[2164] = 466
DATA bitrev_size4096_radix4<>+0x43a8(SB)/8, $1490 // bitrev[2165] = 1490
DATA bitrev_size4096_radix4<>+0x43b0(SB)/8, $2514 // bitrev[2166] = 2514
DATA bitrev_size4096_radix4<>+0x43b8(SB)/8, $3538 // bitrev[2167] = 3538
DATA bitrev_size4096_radix4<>+0x43c0(SB)/8, $722  // bitrev[2168] = 722
DATA bitrev_size4096_radix4<>+0x43c8(SB)/8, $1746 // bitrev[2169] = 1746
DATA bitrev_size4096_radix4<>+0x43d0(SB)/8, $2770 // bitrev[2170] = 2770
DATA bitrev_size4096_radix4<>+0x43d8(SB)/8, $3794 // bitrev[2171] = 3794
DATA bitrev_size4096_radix4<>+0x43e0(SB)/8, $978  // bitrev[2172] = 978
DATA bitrev_size4096_radix4<>+0x43e8(SB)/8, $2002 // bitrev[2173] = 2002
DATA bitrev_size4096_radix4<>+0x43f0(SB)/8, $3026 // bitrev[2174] = 3026
DATA bitrev_size4096_radix4<>+0x43f8(SB)/8, $4050 // bitrev[2175] = 4050
DATA bitrev_size4096_radix4<>+0x4400(SB)/8, $34   // bitrev[2176] = 34
DATA bitrev_size4096_radix4<>+0x4408(SB)/8, $1058 // bitrev[2177] = 1058
DATA bitrev_size4096_radix4<>+0x4410(SB)/8, $2082 // bitrev[2178] = 2082
DATA bitrev_size4096_radix4<>+0x4418(SB)/8, $3106 // bitrev[2179] = 3106
DATA bitrev_size4096_radix4<>+0x4420(SB)/8, $290  // bitrev[2180] = 290
DATA bitrev_size4096_radix4<>+0x4428(SB)/8, $1314 // bitrev[2181] = 1314
DATA bitrev_size4096_radix4<>+0x4430(SB)/8, $2338 // bitrev[2182] = 2338
DATA bitrev_size4096_radix4<>+0x4438(SB)/8, $3362 // bitrev[2183] = 3362
DATA bitrev_size4096_radix4<>+0x4440(SB)/8, $546  // bitrev[2184] = 546
DATA bitrev_size4096_radix4<>+0x4448(SB)/8, $1570 // bitrev[2185] = 1570
DATA bitrev_size4096_radix4<>+0x4450(SB)/8, $2594 // bitrev[2186] = 2594
DATA bitrev_size4096_radix4<>+0x4458(SB)/8, $3618 // bitrev[2187] = 3618
DATA bitrev_size4096_radix4<>+0x4460(SB)/8, $802  // bitrev[2188] = 802
DATA bitrev_size4096_radix4<>+0x4468(SB)/8, $1826 // bitrev[2189] = 1826
DATA bitrev_size4096_radix4<>+0x4470(SB)/8, $2850 // bitrev[2190] = 2850
DATA bitrev_size4096_radix4<>+0x4478(SB)/8, $3874 // bitrev[2191] = 3874
DATA bitrev_size4096_radix4<>+0x4480(SB)/8, $98   // bitrev[2192] = 98
DATA bitrev_size4096_radix4<>+0x4488(SB)/8, $1122 // bitrev[2193] = 1122
DATA bitrev_size4096_radix4<>+0x4490(SB)/8, $2146 // bitrev[2194] = 2146
DATA bitrev_size4096_radix4<>+0x4498(SB)/8, $3170 // bitrev[2195] = 3170
DATA bitrev_size4096_radix4<>+0x44a0(SB)/8, $354  // bitrev[2196] = 354
DATA bitrev_size4096_radix4<>+0x44a8(SB)/8, $1378 // bitrev[2197] = 1378
DATA bitrev_size4096_radix4<>+0x44b0(SB)/8, $2402 // bitrev[2198] = 2402
DATA bitrev_size4096_radix4<>+0x44b8(SB)/8, $3426 // bitrev[2199] = 3426
DATA bitrev_size4096_radix4<>+0x44c0(SB)/8, $610  // bitrev[2200] = 610
DATA bitrev_size4096_radix4<>+0x44c8(SB)/8, $1634 // bitrev[2201] = 1634
DATA bitrev_size4096_radix4<>+0x44d0(SB)/8, $2658 // bitrev[2202] = 2658
DATA bitrev_size4096_radix4<>+0x44d8(SB)/8, $3682 // bitrev[2203] = 3682
DATA bitrev_size4096_radix4<>+0x44e0(SB)/8, $866  // bitrev[2204] = 866
DATA bitrev_size4096_radix4<>+0x44e8(SB)/8, $1890 // bitrev[2205] = 1890
DATA bitrev_size4096_radix4<>+0x44f0(SB)/8, $2914 // bitrev[2206] = 2914
DATA bitrev_size4096_radix4<>+0x44f8(SB)/8, $3938 // bitrev[2207] = 3938
DATA bitrev_size4096_radix4<>+0x4500(SB)/8, $162  // bitrev[2208] = 162
DATA bitrev_size4096_radix4<>+0x4508(SB)/8, $1186 // bitrev[2209] = 1186
DATA bitrev_size4096_radix4<>+0x4510(SB)/8, $2210 // bitrev[2210] = 2210
DATA bitrev_size4096_radix4<>+0x4518(SB)/8, $3234 // bitrev[2211] = 3234
DATA bitrev_size4096_radix4<>+0x4520(SB)/8, $418  // bitrev[2212] = 418
DATA bitrev_size4096_radix4<>+0x4528(SB)/8, $1442 // bitrev[2213] = 1442
DATA bitrev_size4096_radix4<>+0x4530(SB)/8, $2466 // bitrev[2214] = 2466
DATA bitrev_size4096_radix4<>+0x4538(SB)/8, $3490 // bitrev[2215] = 3490
DATA bitrev_size4096_radix4<>+0x4540(SB)/8, $674  // bitrev[2216] = 674
DATA bitrev_size4096_radix4<>+0x4548(SB)/8, $1698 // bitrev[2217] = 1698
DATA bitrev_size4096_radix4<>+0x4550(SB)/8, $2722 // bitrev[2218] = 2722
DATA bitrev_size4096_radix4<>+0x4558(SB)/8, $3746 // bitrev[2219] = 3746
DATA bitrev_size4096_radix4<>+0x4560(SB)/8, $930  // bitrev[2220] = 930
DATA bitrev_size4096_radix4<>+0x4568(SB)/8, $1954 // bitrev[2221] = 1954
DATA bitrev_size4096_radix4<>+0x4570(SB)/8, $2978 // bitrev[2222] = 2978
DATA bitrev_size4096_radix4<>+0x4578(SB)/8, $4002 // bitrev[2223] = 4002
DATA bitrev_size4096_radix4<>+0x4580(SB)/8, $226  // bitrev[2224] = 226
DATA bitrev_size4096_radix4<>+0x4588(SB)/8, $1250 // bitrev[2225] = 1250
DATA bitrev_size4096_radix4<>+0x4590(SB)/8, $2274 // bitrev[2226] = 2274
DATA bitrev_size4096_radix4<>+0x4598(SB)/8, $3298 // bitrev[2227] = 3298
DATA bitrev_size4096_radix4<>+0x45a0(SB)/8, $482  // bitrev[2228] = 482
DATA bitrev_size4096_radix4<>+0x45a8(SB)/8, $1506 // bitrev[2229] = 1506
DATA bitrev_size4096_radix4<>+0x45b0(SB)/8, $2530 // bitrev[2230] = 2530
DATA bitrev_size4096_radix4<>+0x45b8(SB)/8, $3554 // bitrev[2231] = 3554
DATA bitrev_size4096_radix4<>+0x45c0(SB)/8, $738  // bitrev[2232] = 738
DATA bitrev_size4096_radix4<>+0x45c8(SB)/8, $1762 // bitrev[2233] = 1762
DATA bitrev_size4096_radix4<>+0x45d0(SB)/8, $2786 // bitrev[2234] = 2786
DATA bitrev_size4096_radix4<>+0x45d8(SB)/8, $3810 // bitrev[2235] = 3810
DATA bitrev_size4096_radix4<>+0x45e0(SB)/8, $994  // bitrev[2236] = 994
DATA bitrev_size4096_radix4<>+0x45e8(SB)/8, $2018 // bitrev[2237] = 2018
DATA bitrev_size4096_radix4<>+0x45f0(SB)/8, $3042 // bitrev[2238] = 3042
DATA bitrev_size4096_radix4<>+0x45f8(SB)/8, $4066 // bitrev[2239] = 4066
DATA bitrev_size4096_radix4<>+0x4600(SB)/8, $50   // bitrev[2240] = 50
DATA bitrev_size4096_radix4<>+0x4608(SB)/8, $1074 // bitrev[2241] = 1074
DATA bitrev_size4096_radix4<>+0x4610(SB)/8, $2098 // bitrev[2242] = 2098
DATA bitrev_size4096_radix4<>+0x4618(SB)/8, $3122 // bitrev[2243] = 3122
DATA bitrev_size4096_radix4<>+0x4620(SB)/8, $306  // bitrev[2244] = 306
DATA bitrev_size4096_radix4<>+0x4628(SB)/8, $1330 // bitrev[2245] = 1330
DATA bitrev_size4096_radix4<>+0x4630(SB)/8, $2354 // bitrev[2246] = 2354
DATA bitrev_size4096_radix4<>+0x4638(SB)/8, $3378 // bitrev[2247] = 3378
DATA bitrev_size4096_radix4<>+0x4640(SB)/8, $562  // bitrev[2248] = 562
DATA bitrev_size4096_radix4<>+0x4648(SB)/8, $1586 // bitrev[2249] = 1586
DATA bitrev_size4096_radix4<>+0x4650(SB)/8, $2610 // bitrev[2250] = 2610
DATA bitrev_size4096_radix4<>+0x4658(SB)/8, $3634 // bitrev[2251] = 3634
DATA bitrev_size4096_radix4<>+0x4660(SB)/8, $818  // bitrev[2252] = 818
DATA bitrev_size4096_radix4<>+0x4668(SB)/8, $1842 // bitrev[2253] = 1842
DATA bitrev_size4096_radix4<>+0x4670(SB)/8, $2866 // bitrev[2254] = 2866
DATA bitrev_size4096_radix4<>+0x4678(SB)/8, $3890 // bitrev[2255] = 3890
DATA bitrev_size4096_radix4<>+0x4680(SB)/8, $114  // bitrev[2256] = 114
DATA bitrev_size4096_radix4<>+0x4688(SB)/8, $1138 // bitrev[2257] = 1138
DATA bitrev_size4096_radix4<>+0x4690(SB)/8, $2162 // bitrev[2258] = 2162
DATA bitrev_size4096_radix4<>+0x4698(SB)/8, $3186 // bitrev[2259] = 3186
DATA bitrev_size4096_radix4<>+0x46a0(SB)/8, $370  // bitrev[2260] = 370
DATA bitrev_size4096_radix4<>+0x46a8(SB)/8, $1394 // bitrev[2261] = 1394
DATA bitrev_size4096_radix4<>+0x46b0(SB)/8, $2418 // bitrev[2262] = 2418
DATA bitrev_size4096_radix4<>+0x46b8(SB)/8, $3442 // bitrev[2263] = 3442
DATA bitrev_size4096_radix4<>+0x46c0(SB)/8, $626  // bitrev[2264] = 626
DATA bitrev_size4096_radix4<>+0x46c8(SB)/8, $1650 // bitrev[2265] = 1650
DATA bitrev_size4096_radix4<>+0x46d0(SB)/8, $2674 // bitrev[2266] = 2674
DATA bitrev_size4096_radix4<>+0x46d8(SB)/8, $3698 // bitrev[2267] = 3698
DATA bitrev_size4096_radix4<>+0x46e0(SB)/8, $882  // bitrev[2268] = 882
DATA bitrev_size4096_radix4<>+0x46e8(SB)/8, $1906 // bitrev[2269] = 1906
DATA bitrev_size4096_radix4<>+0x46f0(SB)/8, $2930 // bitrev[2270] = 2930
DATA bitrev_size4096_radix4<>+0x46f8(SB)/8, $3954 // bitrev[2271] = 3954
DATA bitrev_size4096_radix4<>+0x4700(SB)/8, $178  // bitrev[2272] = 178
DATA bitrev_size4096_radix4<>+0x4708(SB)/8, $1202 // bitrev[2273] = 1202
DATA bitrev_size4096_radix4<>+0x4710(SB)/8, $2226 // bitrev[2274] = 2226
DATA bitrev_size4096_radix4<>+0x4718(SB)/8, $3250 // bitrev[2275] = 3250
DATA bitrev_size4096_radix4<>+0x4720(SB)/8, $434  // bitrev[2276] = 434
DATA bitrev_size4096_radix4<>+0x4728(SB)/8, $1458 // bitrev[2277] = 1458
DATA bitrev_size4096_radix4<>+0x4730(SB)/8, $2482 // bitrev[2278] = 2482
DATA bitrev_size4096_radix4<>+0x4738(SB)/8, $3506 // bitrev[2279] = 3506
DATA bitrev_size4096_radix4<>+0x4740(SB)/8, $690  // bitrev[2280] = 690
DATA bitrev_size4096_radix4<>+0x4748(SB)/8, $1714 // bitrev[2281] = 1714
DATA bitrev_size4096_radix4<>+0x4750(SB)/8, $2738 // bitrev[2282] = 2738
DATA bitrev_size4096_radix4<>+0x4758(SB)/8, $3762 // bitrev[2283] = 3762
DATA bitrev_size4096_radix4<>+0x4760(SB)/8, $946  // bitrev[2284] = 946
DATA bitrev_size4096_radix4<>+0x4768(SB)/8, $1970 // bitrev[2285] = 1970
DATA bitrev_size4096_radix4<>+0x4770(SB)/8, $2994 // bitrev[2286] = 2994
DATA bitrev_size4096_radix4<>+0x4778(SB)/8, $4018 // bitrev[2287] = 4018
DATA bitrev_size4096_radix4<>+0x4780(SB)/8, $242  // bitrev[2288] = 242
DATA bitrev_size4096_radix4<>+0x4788(SB)/8, $1266 // bitrev[2289] = 1266
DATA bitrev_size4096_radix4<>+0x4790(SB)/8, $2290 // bitrev[2290] = 2290
DATA bitrev_size4096_radix4<>+0x4798(SB)/8, $3314 // bitrev[2291] = 3314
DATA bitrev_size4096_radix4<>+0x47a0(SB)/8, $498  // bitrev[2292] = 498
DATA bitrev_size4096_radix4<>+0x47a8(SB)/8, $1522 // bitrev[2293] = 1522
DATA bitrev_size4096_radix4<>+0x47b0(SB)/8, $2546 // bitrev[2294] = 2546
DATA bitrev_size4096_radix4<>+0x47b8(SB)/8, $3570 // bitrev[2295] = 3570
DATA bitrev_size4096_radix4<>+0x47c0(SB)/8, $754  // bitrev[2296] = 754
DATA bitrev_size4096_radix4<>+0x47c8(SB)/8, $1778 // bitrev[2297] = 1778
DATA bitrev_size4096_radix4<>+0x47d0(SB)/8, $2802 // bitrev[2298] = 2802
DATA bitrev_size4096_radix4<>+0x47d8(SB)/8, $3826 // bitrev[2299] = 3826
DATA bitrev_size4096_radix4<>+0x47e0(SB)/8, $1010 // bitrev[2300] = 1010
DATA bitrev_size4096_radix4<>+0x47e8(SB)/8, $2034 // bitrev[2301] = 2034
DATA bitrev_size4096_radix4<>+0x47f0(SB)/8, $3058 // bitrev[2302] = 3058
DATA bitrev_size4096_radix4<>+0x47f8(SB)/8, $4082 // bitrev[2303] = 4082
DATA bitrev_size4096_radix4<>+0x4800(SB)/8, $6    // bitrev[2304] = 6
DATA bitrev_size4096_radix4<>+0x4808(SB)/8, $1030 // bitrev[2305] = 1030
DATA bitrev_size4096_radix4<>+0x4810(SB)/8, $2054 // bitrev[2306] = 2054
DATA bitrev_size4096_radix4<>+0x4818(SB)/8, $3078 // bitrev[2307] = 3078
DATA bitrev_size4096_radix4<>+0x4820(SB)/8, $262  // bitrev[2308] = 262
DATA bitrev_size4096_radix4<>+0x4828(SB)/8, $1286 // bitrev[2309] = 1286
DATA bitrev_size4096_radix4<>+0x4830(SB)/8, $2310 // bitrev[2310] = 2310
DATA bitrev_size4096_radix4<>+0x4838(SB)/8, $3334 // bitrev[2311] = 3334
DATA bitrev_size4096_radix4<>+0x4840(SB)/8, $518  // bitrev[2312] = 518
DATA bitrev_size4096_radix4<>+0x4848(SB)/8, $1542 // bitrev[2313] = 1542
DATA bitrev_size4096_radix4<>+0x4850(SB)/8, $2566 // bitrev[2314] = 2566
DATA bitrev_size4096_radix4<>+0x4858(SB)/8, $3590 // bitrev[2315] = 3590
DATA bitrev_size4096_radix4<>+0x4860(SB)/8, $774  // bitrev[2316] = 774
DATA bitrev_size4096_radix4<>+0x4868(SB)/8, $1798 // bitrev[2317] = 1798
DATA bitrev_size4096_radix4<>+0x4870(SB)/8, $2822 // bitrev[2318] = 2822
DATA bitrev_size4096_radix4<>+0x4878(SB)/8, $3846 // bitrev[2319] = 3846
DATA bitrev_size4096_radix4<>+0x4880(SB)/8, $70   // bitrev[2320] = 70
DATA bitrev_size4096_radix4<>+0x4888(SB)/8, $1094 // bitrev[2321] = 1094
DATA bitrev_size4096_radix4<>+0x4890(SB)/8, $2118 // bitrev[2322] = 2118
DATA bitrev_size4096_radix4<>+0x4898(SB)/8, $3142 // bitrev[2323] = 3142
DATA bitrev_size4096_radix4<>+0x48a0(SB)/8, $326  // bitrev[2324] = 326
DATA bitrev_size4096_radix4<>+0x48a8(SB)/8, $1350 // bitrev[2325] = 1350
DATA bitrev_size4096_radix4<>+0x48b0(SB)/8, $2374 // bitrev[2326] = 2374
DATA bitrev_size4096_radix4<>+0x48b8(SB)/8, $3398 // bitrev[2327] = 3398
DATA bitrev_size4096_radix4<>+0x48c0(SB)/8, $582  // bitrev[2328] = 582
DATA bitrev_size4096_radix4<>+0x48c8(SB)/8, $1606 // bitrev[2329] = 1606
DATA bitrev_size4096_radix4<>+0x48d0(SB)/8, $2630 // bitrev[2330] = 2630
DATA bitrev_size4096_radix4<>+0x48d8(SB)/8, $3654 // bitrev[2331] = 3654
DATA bitrev_size4096_radix4<>+0x48e0(SB)/8, $838  // bitrev[2332] = 838
DATA bitrev_size4096_radix4<>+0x48e8(SB)/8, $1862 // bitrev[2333] = 1862
DATA bitrev_size4096_radix4<>+0x48f0(SB)/8, $2886 // bitrev[2334] = 2886
DATA bitrev_size4096_radix4<>+0x48f8(SB)/8, $3910 // bitrev[2335] = 3910
DATA bitrev_size4096_radix4<>+0x4900(SB)/8, $134  // bitrev[2336] = 134
DATA bitrev_size4096_radix4<>+0x4908(SB)/8, $1158 // bitrev[2337] = 1158
DATA bitrev_size4096_radix4<>+0x4910(SB)/8, $2182 // bitrev[2338] = 2182
DATA bitrev_size4096_radix4<>+0x4918(SB)/8, $3206 // bitrev[2339] = 3206
DATA bitrev_size4096_radix4<>+0x4920(SB)/8, $390  // bitrev[2340] = 390
DATA bitrev_size4096_radix4<>+0x4928(SB)/8, $1414 // bitrev[2341] = 1414
DATA bitrev_size4096_radix4<>+0x4930(SB)/8, $2438 // bitrev[2342] = 2438
DATA bitrev_size4096_radix4<>+0x4938(SB)/8, $3462 // bitrev[2343] = 3462
DATA bitrev_size4096_radix4<>+0x4940(SB)/8, $646  // bitrev[2344] = 646
DATA bitrev_size4096_radix4<>+0x4948(SB)/8, $1670 // bitrev[2345] = 1670
DATA bitrev_size4096_radix4<>+0x4950(SB)/8, $2694 // bitrev[2346] = 2694
DATA bitrev_size4096_radix4<>+0x4958(SB)/8, $3718 // bitrev[2347] = 3718
DATA bitrev_size4096_radix4<>+0x4960(SB)/8, $902  // bitrev[2348] = 902
DATA bitrev_size4096_radix4<>+0x4968(SB)/8, $1926 // bitrev[2349] = 1926
DATA bitrev_size4096_radix4<>+0x4970(SB)/8, $2950 // bitrev[2350] = 2950
DATA bitrev_size4096_radix4<>+0x4978(SB)/8, $3974 // bitrev[2351] = 3974
DATA bitrev_size4096_radix4<>+0x4980(SB)/8, $198  // bitrev[2352] = 198
DATA bitrev_size4096_radix4<>+0x4988(SB)/8, $1222 // bitrev[2353] = 1222
DATA bitrev_size4096_radix4<>+0x4990(SB)/8, $2246 // bitrev[2354] = 2246
DATA bitrev_size4096_radix4<>+0x4998(SB)/8, $3270 // bitrev[2355] = 3270
DATA bitrev_size4096_radix4<>+0x49a0(SB)/8, $454  // bitrev[2356] = 454
DATA bitrev_size4096_radix4<>+0x49a8(SB)/8, $1478 // bitrev[2357] = 1478
DATA bitrev_size4096_radix4<>+0x49b0(SB)/8, $2502 // bitrev[2358] = 2502
DATA bitrev_size4096_radix4<>+0x49b8(SB)/8, $3526 // bitrev[2359] = 3526
DATA bitrev_size4096_radix4<>+0x49c0(SB)/8, $710  // bitrev[2360] = 710
DATA bitrev_size4096_radix4<>+0x49c8(SB)/8, $1734 // bitrev[2361] = 1734
DATA bitrev_size4096_radix4<>+0x49d0(SB)/8, $2758 // bitrev[2362] = 2758
DATA bitrev_size4096_radix4<>+0x49d8(SB)/8, $3782 // bitrev[2363] = 3782
DATA bitrev_size4096_radix4<>+0x49e0(SB)/8, $966  // bitrev[2364] = 966
DATA bitrev_size4096_radix4<>+0x49e8(SB)/8, $1990 // bitrev[2365] = 1990
DATA bitrev_size4096_radix4<>+0x49f0(SB)/8, $3014 // bitrev[2366] = 3014
DATA bitrev_size4096_radix4<>+0x49f8(SB)/8, $4038 // bitrev[2367] = 4038
DATA bitrev_size4096_radix4<>+0x4a00(SB)/8, $22   // bitrev[2368] = 22
DATA bitrev_size4096_radix4<>+0x4a08(SB)/8, $1046 // bitrev[2369] = 1046
DATA bitrev_size4096_radix4<>+0x4a10(SB)/8, $2070 // bitrev[2370] = 2070
DATA bitrev_size4096_radix4<>+0x4a18(SB)/8, $3094 // bitrev[2371] = 3094
DATA bitrev_size4096_radix4<>+0x4a20(SB)/8, $278  // bitrev[2372] = 278
DATA bitrev_size4096_radix4<>+0x4a28(SB)/8, $1302 // bitrev[2373] = 1302
DATA bitrev_size4096_radix4<>+0x4a30(SB)/8, $2326 // bitrev[2374] = 2326
DATA bitrev_size4096_radix4<>+0x4a38(SB)/8, $3350 // bitrev[2375] = 3350
DATA bitrev_size4096_radix4<>+0x4a40(SB)/8, $534  // bitrev[2376] = 534
DATA bitrev_size4096_radix4<>+0x4a48(SB)/8, $1558 // bitrev[2377] = 1558
DATA bitrev_size4096_radix4<>+0x4a50(SB)/8, $2582 // bitrev[2378] = 2582
DATA bitrev_size4096_radix4<>+0x4a58(SB)/8, $3606 // bitrev[2379] = 3606
DATA bitrev_size4096_radix4<>+0x4a60(SB)/8, $790  // bitrev[2380] = 790
DATA bitrev_size4096_radix4<>+0x4a68(SB)/8, $1814 // bitrev[2381] = 1814
DATA bitrev_size4096_radix4<>+0x4a70(SB)/8, $2838 // bitrev[2382] = 2838
DATA bitrev_size4096_radix4<>+0x4a78(SB)/8, $3862 // bitrev[2383] = 3862
DATA bitrev_size4096_radix4<>+0x4a80(SB)/8, $86   // bitrev[2384] = 86
DATA bitrev_size4096_radix4<>+0x4a88(SB)/8, $1110 // bitrev[2385] = 1110
DATA bitrev_size4096_radix4<>+0x4a90(SB)/8, $2134 // bitrev[2386] = 2134
DATA bitrev_size4096_radix4<>+0x4a98(SB)/8, $3158 // bitrev[2387] = 3158
DATA bitrev_size4096_radix4<>+0x4aa0(SB)/8, $342  // bitrev[2388] = 342
DATA bitrev_size4096_radix4<>+0x4aa8(SB)/8, $1366 // bitrev[2389] = 1366
DATA bitrev_size4096_radix4<>+0x4ab0(SB)/8, $2390 // bitrev[2390] = 2390
DATA bitrev_size4096_radix4<>+0x4ab8(SB)/8, $3414 // bitrev[2391] = 3414
DATA bitrev_size4096_radix4<>+0x4ac0(SB)/8, $598  // bitrev[2392] = 598
DATA bitrev_size4096_radix4<>+0x4ac8(SB)/8, $1622 // bitrev[2393] = 1622
DATA bitrev_size4096_radix4<>+0x4ad0(SB)/8, $2646 // bitrev[2394] = 2646
DATA bitrev_size4096_radix4<>+0x4ad8(SB)/8, $3670 // bitrev[2395] = 3670
DATA bitrev_size4096_radix4<>+0x4ae0(SB)/8, $854  // bitrev[2396] = 854
DATA bitrev_size4096_radix4<>+0x4ae8(SB)/8, $1878 // bitrev[2397] = 1878
DATA bitrev_size4096_radix4<>+0x4af0(SB)/8, $2902 // bitrev[2398] = 2902
DATA bitrev_size4096_radix4<>+0x4af8(SB)/8, $3926 // bitrev[2399] = 3926
DATA bitrev_size4096_radix4<>+0x4b00(SB)/8, $150  // bitrev[2400] = 150
DATA bitrev_size4096_radix4<>+0x4b08(SB)/8, $1174 // bitrev[2401] = 1174
DATA bitrev_size4096_radix4<>+0x4b10(SB)/8, $2198 // bitrev[2402] = 2198
DATA bitrev_size4096_radix4<>+0x4b18(SB)/8, $3222 // bitrev[2403] = 3222
DATA bitrev_size4096_radix4<>+0x4b20(SB)/8, $406  // bitrev[2404] = 406
DATA bitrev_size4096_radix4<>+0x4b28(SB)/8, $1430 // bitrev[2405] = 1430
DATA bitrev_size4096_radix4<>+0x4b30(SB)/8, $2454 // bitrev[2406] = 2454
DATA bitrev_size4096_radix4<>+0x4b38(SB)/8, $3478 // bitrev[2407] = 3478
DATA bitrev_size4096_radix4<>+0x4b40(SB)/8, $662  // bitrev[2408] = 662
DATA bitrev_size4096_radix4<>+0x4b48(SB)/8, $1686 // bitrev[2409] = 1686
DATA bitrev_size4096_radix4<>+0x4b50(SB)/8, $2710 // bitrev[2410] = 2710
DATA bitrev_size4096_radix4<>+0x4b58(SB)/8, $3734 // bitrev[2411] = 3734
DATA bitrev_size4096_radix4<>+0x4b60(SB)/8, $918  // bitrev[2412] = 918
DATA bitrev_size4096_radix4<>+0x4b68(SB)/8, $1942 // bitrev[2413] = 1942
DATA bitrev_size4096_radix4<>+0x4b70(SB)/8, $2966 // bitrev[2414] = 2966
DATA bitrev_size4096_radix4<>+0x4b78(SB)/8, $3990 // bitrev[2415] = 3990
DATA bitrev_size4096_radix4<>+0x4b80(SB)/8, $214  // bitrev[2416] = 214
DATA bitrev_size4096_radix4<>+0x4b88(SB)/8, $1238 // bitrev[2417] = 1238
DATA bitrev_size4096_radix4<>+0x4b90(SB)/8, $2262 // bitrev[2418] = 2262
DATA bitrev_size4096_radix4<>+0x4b98(SB)/8, $3286 // bitrev[2419] = 3286
DATA bitrev_size4096_radix4<>+0x4ba0(SB)/8, $470  // bitrev[2420] = 470
DATA bitrev_size4096_radix4<>+0x4ba8(SB)/8, $1494 // bitrev[2421] = 1494
DATA bitrev_size4096_radix4<>+0x4bb0(SB)/8, $2518 // bitrev[2422] = 2518
DATA bitrev_size4096_radix4<>+0x4bb8(SB)/8, $3542 // bitrev[2423] = 3542
DATA bitrev_size4096_radix4<>+0x4bc0(SB)/8, $726  // bitrev[2424] = 726
DATA bitrev_size4096_radix4<>+0x4bc8(SB)/8, $1750 // bitrev[2425] = 1750
DATA bitrev_size4096_radix4<>+0x4bd0(SB)/8, $2774 // bitrev[2426] = 2774
DATA bitrev_size4096_radix4<>+0x4bd8(SB)/8, $3798 // bitrev[2427] = 3798
DATA bitrev_size4096_radix4<>+0x4be0(SB)/8, $982  // bitrev[2428] = 982
DATA bitrev_size4096_radix4<>+0x4be8(SB)/8, $2006 // bitrev[2429] = 2006
DATA bitrev_size4096_radix4<>+0x4bf0(SB)/8, $3030 // bitrev[2430] = 3030
DATA bitrev_size4096_radix4<>+0x4bf8(SB)/8, $4054 // bitrev[2431] = 4054
DATA bitrev_size4096_radix4<>+0x4c00(SB)/8, $38   // bitrev[2432] = 38
DATA bitrev_size4096_radix4<>+0x4c08(SB)/8, $1062 // bitrev[2433] = 1062
DATA bitrev_size4096_radix4<>+0x4c10(SB)/8, $2086 // bitrev[2434] = 2086
DATA bitrev_size4096_radix4<>+0x4c18(SB)/8, $3110 // bitrev[2435] = 3110
DATA bitrev_size4096_radix4<>+0x4c20(SB)/8, $294  // bitrev[2436] = 294
DATA bitrev_size4096_radix4<>+0x4c28(SB)/8, $1318 // bitrev[2437] = 1318
DATA bitrev_size4096_radix4<>+0x4c30(SB)/8, $2342 // bitrev[2438] = 2342
DATA bitrev_size4096_radix4<>+0x4c38(SB)/8, $3366 // bitrev[2439] = 3366
DATA bitrev_size4096_radix4<>+0x4c40(SB)/8, $550  // bitrev[2440] = 550
DATA bitrev_size4096_radix4<>+0x4c48(SB)/8, $1574 // bitrev[2441] = 1574
DATA bitrev_size4096_radix4<>+0x4c50(SB)/8, $2598 // bitrev[2442] = 2598
DATA bitrev_size4096_radix4<>+0x4c58(SB)/8, $3622 // bitrev[2443] = 3622
DATA bitrev_size4096_radix4<>+0x4c60(SB)/8, $806  // bitrev[2444] = 806
DATA bitrev_size4096_radix4<>+0x4c68(SB)/8, $1830 // bitrev[2445] = 1830
DATA bitrev_size4096_radix4<>+0x4c70(SB)/8, $2854 // bitrev[2446] = 2854
DATA bitrev_size4096_radix4<>+0x4c78(SB)/8, $3878 // bitrev[2447] = 3878
DATA bitrev_size4096_radix4<>+0x4c80(SB)/8, $102  // bitrev[2448] = 102
DATA bitrev_size4096_radix4<>+0x4c88(SB)/8, $1126 // bitrev[2449] = 1126
DATA bitrev_size4096_radix4<>+0x4c90(SB)/8, $2150 // bitrev[2450] = 2150
DATA bitrev_size4096_radix4<>+0x4c98(SB)/8, $3174 // bitrev[2451] = 3174
DATA bitrev_size4096_radix4<>+0x4ca0(SB)/8, $358  // bitrev[2452] = 358
DATA bitrev_size4096_radix4<>+0x4ca8(SB)/8, $1382 // bitrev[2453] = 1382
DATA bitrev_size4096_radix4<>+0x4cb0(SB)/8, $2406 // bitrev[2454] = 2406
DATA bitrev_size4096_radix4<>+0x4cb8(SB)/8, $3430 // bitrev[2455] = 3430
DATA bitrev_size4096_radix4<>+0x4cc0(SB)/8, $614  // bitrev[2456] = 614
DATA bitrev_size4096_radix4<>+0x4cc8(SB)/8, $1638 // bitrev[2457] = 1638
DATA bitrev_size4096_radix4<>+0x4cd0(SB)/8, $2662 // bitrev[2458] = 2662
DATA bitrev_size4096_radix4<>+0x4cd8(SB)/8, $3686 // bitrev[2459] = 3686
DATA bitrev_size4096_radix4<>+0x4ce0(SB)/8, $870  // bitrev[2460] = 870
DATA bitrev_size4096_radix4<>+0x4ce8(SB)/8, $1894 // bitrev[2461] = 1894
DATA bitrev_size4096_radix4<>+0x4cf0(SB)/8, $2918 // bitrev[2462] = 2918
DATA bitrev_size4096_radix4<>+0x4cf8(SB)/8, $3942 // bitrev[2463] = 3942
DATA bitrev_size4096_radix4<>+0x4d00(SB)/8, $166  // bitrev[2464] = 166
DATA bitrev_size4096_radix4<>+0x4d08(SB)/8, $1190 // bitrev[2465] = 1190
DATA bitrev_size4096_radix4<>+0x4d10(SB)/8, $2214 // bitrev[2466] = 2214
DATA bitrev_size4096_radix4<>+0x4d18(SB)/8, $3238 // bitrev[2467] = 3238
DATA bitrev_size4096_radix4<>+0x4d20(SB)/8, $422  // bitrev[2468] = 422
DATA bitrev_size4096_radix4<>+0x4d28(SB)/8, $1446 // bitrev[2469] = 1446
DATA bitrev_size4096_radix4<>+0x4d30(SB)/8, $2470 // bitrev[2470] = 2470
DATA bitrev_size4096_radix4<>+0x4d38(SB)/8, $3494 // bitrev[2471] = 3494
DATA bitrev_size4096_radix4<>+0x4d40(SB)/8, $678  // bitrev[2472] = 678
DATA bitrev_size4096_radix4<>+0x4d48(SB)/8, $1702 // bitrev[2473] = 1702
DATA bitrev_size4096_radix4<>+0x4d50(SB)/8, $2726 // bitrev[2474] = 2726
DATA bitrev_size4096_radix4<>+0x4d58(SB)/8, $3750 // bitrev[2475] = 3750
DATA bitrev_size4096_radix4<>+0x4d60(SB)/8, $934  // bitrev[2476] = 934
DATA bitrev_size4096_radix4<>+0x4d68(SB)/8, $1958 // bitrev[2477] = 1958
DATA bitrev_size4096_radix4<>+0x4d70(SB)/8, $2982 // bitrev[2478] = 2982
DATA bitrev_size4096_radix4<>+0x4d78(SB)/8, $4006 // bitrev[2479] = 4006
DATA bitrev_size4096_radix4<>+0x4d80(SB)/8, $230  // bitrev[2480] = 230
DATA bitrev_size4096_radix4<>+0x4d88(SB)/8, $1254 // bitrev[2481] = 1254
DATA bitrev_size4096_radix4<>+0x4d90(SB)/8, $2278 // bitrev[2482] = 2278
DATA bitrev_size4096_radix4<>+0x4d98(SB)/8, $3302 // bitrev[2483] = 3302
DATA bitrev_size4096_radix4<>+0x4da0(SB)/8, $486  // bitrev[2484] = 486
DATA bitrev_size4096_radix4<>+0x4da8(SB)/8, $1510 // bitrev[2485] = 1510
DATA bitrev_size4096_radix4<>+0x4db0(SB)/8, $2534 // bitrev[2486] = 2534
DATA bitrev_size4096_radix4<>+0x4db8(SB)/8, $3558 // bitrev[2487] = 3558
DATA bitrev_size4096_radix4<>+0x4dc0(SB)/8, $742  // bitrev[2488] = 742
DATA bitrev_size4096_radix4<>+0x4dc8(SB)/8, $1766 // bitrev[2489] = 1766
DATA bitrev_size4096_radix4<>+0x4dd0(SB)/8, $2790 // bitrev[2490] = 2790
DATA bitrev_size4096_radix4<>+0x4dd8(SB)/8, $3814 // bitrev[2491] = 3814
DATA bitrev_size4096_radix4<>+0x4de0(SB)/8, $998  // bitrev[2492] = 998
DATA bitrev_size4096_radix4<>+0x4de8(SB)/8, $2022 // bitrev[2493] = 2022
DATA bitrev_size4096_radix4<>+0x4df0(SB)/8, $3046 // bitrev[2494] = 3046
DATA bitrev_size4096_radix4<>+0x4df8(SB)/8, $4070 // bitrev[2495] = 4070
DATA bitrev_size4096_radix4<>+0x4e00(SB)/8, $54   // bitrev[2496] = 54
DATA bitrev_size4096_radix4<>+0x4e08(SB)/8, $1078 // bitrev[2497] = 1078
DATA bitrev_size4096_radix4<>+0x4e10(SB)/8, $2102 // bitrev[2498] = 2102
DATA bitrev_size4096_radix4<>+0x4e18(SB)/8, $3126 // bitrev[2499] = 3126
DATA bitrev_size4096_radix4<>+0x4e20(SB)/8, $310  // bitrev[2500] = 310
DATA bitrev_size4096_radix4<>+0x4e28(SB)/8, $1334 // bitrev[2501] = 1334
DATA bitrev_size4096_radix4<>+0x4e30(SB)/8, $2358 // bitrev[2502] = 2358
DATA bitrev_size4096_radix4<>+0x4e38(SB)/8, $3382 // bitrev[2503] = 3382
DATA bitrev_size4096_radix4<>+0x4e40(SB)/8, $566  // bitrev[2504] = 566
DATA bitrev_size4096_radix4<>+0x4e48(SB)/8, $1590 // bitrev[2505] = 1590
DATA bitrev_size4096_radix4<>+0x4e50(SB)/8, $2614 // bitrev[2506] = 2614
DATA bitrev_size4096_radix4<>+0x4e58(SB)/8, $3638 // bitrev[2507] = 3638
DATA bitrev_size4096_radix4<>+0x4e60(SB)/8, $822  // bitrev[2508] = 822
DATA bitrev_size4096_radix4<>+0x4e68(SB)/8, $1846 // bitrev[2509] = 1846
DATA bitrev_size4096_radix4<>+0x4e70(SB)/8, $2870 // bitrev[2510] = 2870
DATA bitrev_size4096_radix4<>+0x4e78(SB)/8, $3894 // bitrev[2511] = 3894
DATA bitrev_size4096_radix4<>+0x4e80(SB)/8, $118  // bitrev[2512] = 118
DATA bitrev_size4096_radix4<>+0x4e88(SB)/8, $1142 // bitrev[2513] = 1142
DATA bitrev_size4096_radix4<>+0x4e90(SB)/8, $2166 // bitrev[2514] = 2166
DATA bitrev_size4096_radix4<>+0x4e98(SB)/8, $3190 // bitrev[2515] = 3190
DATA bitrev_size4096_radix4<>+0x4ea0(SB)/8, $374  // bitrev[2516] = 374
DATA bitrev_size4096_radix4<>+0x4ea8(SB)/8, $1398 // bitrev[2517] = 1398
DATA bitrev_size4096_radix4<>+0x4eb0(SB)/8, $2422 // bitrev[2518] = 2422
DATA bitrev_size4096_radix4<>+0x4eb8(SB)/8, $3446 // bitrev[2519] = 3446
DATA bitrev_size4096_radix4<>+0x4ec0(SB)/8, $630  // bitrev[2520] = 630
DATA bitrev_size4096_radix4<>+0x4ec8(SB)/8, $1654 // bitrev[2521] = 1654
DATA bitrev_size4096_radix4<>+0x4ed0(SB)/8, $2678 // bitrev[2522] = 2678
DATA bitrev_size4096_radix4<>+0x4ed8(SB)/8, $3702 // bitrev[2523] = 3702
DATA bitrev_size4096_radix4<>+0x4ee0(SB)/8, $886  // bitrev[2524] = 886
DATA bitrev_size4096_radix4<>+0x4ee8(SB)/8, $1910 // bitrev[2525] = 1910
DATA bitrev_size4096_radix4<>+0x4ef0(SB)/8, $2934 // bitrev[2526] = 2934
DATA bitrev_size4096_radix4<>+0x4ef8(SB)/8, $3958 // bitrev[2527] = 3958
DATA bitrev_size4096_radix4<>+0x4f00(SB)/8, $182  // bitrev[2528] = 182
DATA bitrev_size4096_radix4<>+0x4f08(SB)/8, $1206 // bitrev[2529] = 1206
DATA bitrev_size4096_radix4<>+0x4f10(SB)/8, $2230 // bitrev[2530] = 2230
DATA bitrev_size4096_radix4<>+0x4f18(SB)/8, $3254 // bitrev[2531] = 3254
DATA bitrev_size4096_radix4<>+0x4f20(SB)/8, $438  // bitrev[2532] = 438
DATA bitrev_size4096_radix4<>+0x4f28(SB)/8, $1462 // bitrev[2533] = 1462
DATA bitrev_size4096_radix4<>+0x4f30(SB)/8, $2486 // bitrev[2534] = 2486
DATA bitrev_size4096_radix4<>+0x4f38(SB)/8, $3510 // bitrev[2535] = 3510
DATA bitrev_size4096_radix4<>+0x4f40(SB)/8, $694  // bitrev[2536] = 694
DATA bitrev_size4096_radix4<>+0x4f48(SB)/8, $1718 // bitrev[2537] = 1718
DATA bitrev_size4096_radix4<>+0x4f50(SB)/8, $2742 // bitrev[2538] = 2742
DATA bitrev_size4096_radix4<>+0x4f58(SB)/8, $3766 // bitrev[2539] = 3766
DATA bitrev_size4096_radix4<>+0x4f60(SB)/8, $950  // bitrev[2540] = 950
DATA bitrev_size4096_radix4<>+0x4f68(SB)/8, $1974 // bitrev[2541] = 1974
DATA bitrev_size4096_radix4<>+0x4f70(SB)/8, $2998 // bitrev[2542] = 2998
DATA bitrev_size4096_radix4<>+0x4f78(SB)/8, $4022 // bitrev[2543] = 4022
DATA bitrev_size4096_radix4<>+0x4f80(SB)/8, $246  // bitrev[2544] = 246
DATA bitrev_size4096_radix4<>+0x4f88(SB)/8, $1270 // bitrev[2545] = 1270
DATA bitrev_size4096_radix4<>+0x4f90(SB)/8, $2294 // bitrev[2546] = 2294
DATA bitrev_size4096_radix4<>+0x4f98(SB)/8, $3318 // bitrev[2547] = 3318
DATA bitrev_size4096_radix4<>+0x4fa0(SB)/8, $502  // bitrev[2548] = 502
DATA bitrev_size4096_radix4<>+0x4fa8(SB)/8, $1526 // bitrev[2549] = 1526
DATA bitrev_size4096_radix4<>+0x4fb0(SB)/8, $2550 // bitrev[2550] = 2550
DATA bitrev_size4096_radix4<>+0x4fb8(SB)/8, $3574 // bitrev[2551] = 3574
DATA bitrev_size4096_radix4<>+0x4fc0(SB)/8, $758  // bitrev[2552] = 758
DATA bitrev_size4096_radix4<>+0x4fc8(SB)/8, $1782 // bitrev[2553] = 1782
DATA bitrev_size4096_radix4<>+0x4fd0(SB)/8, $2806 // bitrev[2554] = 2806
DATA bitrev_size4096_radix4<>+0x4fd8(SB)/8, $3830 // bitrev[2555] = 3830
DATA bitrev_size4096_radix4<>+0x4fe0(SB)/8, $1014 // bitrev[2556] = 1014
DATA bitrev_size4096_radix4<>+0x4fe8(SB)/8, $2038 // bitrev[2557] = 2038
DATA bitrev_size4096_radix4<>+0x4ff0(SB)/8, $3062 // bitrev[2558] = 3062
DATA bitrev_size4096_radix4<>+0x4ff8(SB)/8, $4086 // bitrev[2559] = 4086
DATA bitrev_size4096_radix4<>+0x5000(SB)/8, $10   // bitrev[2560] = 10
DATA bitrev_size4096_radix4<>+0x5008(SB)/8, $1034 // bitrev[2561] = 1034
DATA bitrev_size4096_radix4<>+0x5010(SB)/8, $2058 // bitrev[2562] = 2058
DATA bitrev_size4096_radix4<>+0x5018(SB)/8, $3082 // bitrev[2563] = 3082
DATA bitrev_size4096_radix4<>+0x5020(SB)/8, $266  // bitrev[2564] = 266
DATA bitrev_size4096_radix4<>+0x5028(SB)/8, $1290 // bitrev[2565] = 1290
DATA bitrev_size4096_radix4<>+0x5030(SB)/8, $2314 // bitrev[2566] = 2314
DATA bitrev_size4096_radix4<>+0x5038(SB)/8, $3338 // bitrev[2567] = 3338
DATA bitrev_size4096_radix4<>+0x5040(SB)/8, $522  // bitrev[2568] = 522
DATA bitrev_size4096_radix4<>+0x5048(SB)/8, $1546 // bitrev[2569] = 1546
DATA bitrev_size4096_radix4<>+0x5050(SB)/8, $2570 // bitrev[2570] = 2570
DATA bitrev_size4096_radix4<>+0x5058(SB)/8, $3594 // bitrev[2571] = 3594
DATA bitrev_size4096_radix4<>+0x5060(SB)/8, $778  // bitrev[2572] = 778
DATA bitrev_size4096_radix4<>+0x5068(SB)/8, $1802 // bitrev[2573] = 1802
DATA bitrev_size4096_radix4<>+0x5070(SB)/8, $2826 // bitrev[2574] = 2826
DATA bitrev_size4096_radix4<>+0x5078(SB)/8, $3850 // bitrev[2575] = 3850
DATA bitrev_size4096_radix4<>+0x5080(SB)/8, $74   // bitrev[2576] = 74
DATA bitrev_size4096_radix4<>+0x5088(SB)/8, $1098 // bitrev[2577] = 1098
DATA bitrev_size4096_radix4<>+0x5090(SB)/8, $2122 // bitrev[2578] = 2122
DATA bitrev_size4096_radix4<>+0x5098(SB)/8, $3146 // bitrev[2579] = 3146
DATA bitrev_size4096_radix4<>+0x50a0(SB)/8, $330  // bitrev[2580] = 330
DATA bitrev_size4096_radix4<>+0x50a8(SB)/8, $1354 // bitrev[2581] = 1354
DATA bitrev_size4096_radix4<>+0x50b0(SB)/8, $2378 // bitrev[2582] = 2378
DATA bitrev_size4096_radix4<>+0x50b8(SB)/8, $3402 // bitrev[2583] = 3402
DATA bitrev_size4096_radix4<>+0x50c0(SB)/8, $586  // bitrev[2584] = 586
DATA bitrev_size4096_radix4<>+0x50c8(SB)/8, $1610 // bitrev[2585] = 1610
DATA bitrev_size4096_radix4<>+0x50d0(SB)/8, $2634 // bitrev[2586] = 2634
DATA bitrev_size4096_radix4<>+0x50d8(SB)/8, $3658 // bitrev[2587] = 3658
DATA bitrev_size4096_radix4<>+0x50e0(SB)/8, $842  // bitrev[2588] = 842
DATA bitrev_size4096_radix4<>+0x50e8(SB)/8, $1866 // bitrev[2589] = 1866
DATA bitrev_size4096_radix4<>+0x50f0(SB)/8, $2890 // bitrev[2590] = 2890
DATA bitrev_size4096_radix4<>+0x50f8(SB)/8, $3914 // bitrev[2591] = 3914
DATA bitrev_size4096_radix4<>+0x5100(SB)/8, $138  // bitrev[2592] = 138
DATA bitrev_size4096_radix4<>+0x5108(SB)/8, $1162 // bitrev[2593] = 1162
DATA bitrev_size4096_radix4<>+0x5110(SB)/8, $2186 // bitrev[2594] = 2186
DATA bitrev_size4096_radix4<>+0x5118(SB)/8, $3210 // bitrev[2595] = 3210
DATA bitrev_size4096_radix4<>+0x5120(SB)/8, $394  // bitrev[2596] = 394
DATA bitrev_size4096_radix4<>+0x5128(SB)/8, $1418 // bitrev[2597] = 1418
DATA bitrev_size4096_radix4<>+0x5130(SB)/8, $2442 // bitrev[2598] = 2442
DATA bitrev_size4096_radix4<>+0x5138(SB)/8, $3466 // bitrev[2599] = 3466
DATA bitrev_size4096_radix4<>+0x5140(SB)/8, $650  // bitrev[2600] = 650
DATA bitrev_size4096_radix4<>+0x5148(SB)/8, $1674 // bitrev[2601] = 1674
DATA bitrev_size4096_radix4<>+0x5150(SB)/8, $2698 // bitrev[2602] = 2698
DATA bitrev_size4096_radix4<>+0x5158(SB)/8, $3722 // bitrev[2603] = 3722
DATA bitrev_size4096_radix4<>+0x5160(SB)/8, $906  // bitrev[2604] = 906
DATA bitrev_size4096_radix4<>+0x5168(SB)/8, $1930 // bitrev[2605] = 1930
DATA bitrev_size4096_radix4<>+0x5170(SB)/8, $2954 // bitrev[2606] = 2954
DATA bitrev_size4096_radix4<>+0x5178(SB)/8, $3978 // bitrev[2607] = 3978
DATA bitrev_size4096_radix4<>+0x5180(SB)/8, $202  // bitrev[2608] = 202
DATA bitrev_size4096_radix4<>+0x5188(SB)/8, $1226 // bitrev[2609] = 1226
DATA bitrev_size4096_radix4<>+0x5190(SB)/8, $2250 // bitrev[2610] = 2250
DATA bitrev_size4096_radix4<>+0x5198(SB)/8, $3274 // bitrev[2611] = 3274
DATA bitrev_size4096_radix4<>+0x51a0(SB)/8, $458  // bitrev[2612] = 458
DATA bitrev_size4096_radix4<>+0x51a8(SB)/8, $1482 // bitrev[2613] = 1482
DATA bitrev_size4096_radix4<>+0x51b0(SB)/8, $2506 // bitrev[2614] = 2506
DATA bitrev_size4096_radix4<>+0x51b8(SB)/8, $3530 // bitrev[2615] = 3530
DATA bitrev_size4096_radix4<>+0x51c0(SB)/8, $714  // bitrev[2616] = 714
DATA bitrev_size4096_radix4<>+0x51c8(SB)/8, $1738 // bitrev[2617] = 1738
DATA bitrev_size4096_radix4<>+0x51d0(SB)/8, $2762 // bitrev[2618] = 2762
DATA bitrev_size4096_radix4<>+0x51d8(SB)/8, $3786 // bitrev[2619] = 3786
DATA bitrev_size4096_radix4<>+0x51e0(SB)/8, $970  // bitrev[2620] = 970
DATA bitrev_size4096_radix4<>+0x51e8(SB)/8, $1994 // bitrev[2621] = 1994
DATA bitrev_size4096_radix4<>+0x51f0(SB)/8, $3018 // bitrev[2622] = 3018
DATA bitrev_size4096_radix4<>+0x51f8(SB)/8, $4042 // bitrev[2623] = 4042
DATA bitrev_size4096_radix4<>+0x5200(SB)/8, $26   // bitrev[2624] = 26
DATA bitrev_size4096_radix4<>+0x5208(SB)/8, $1050 // bitrev[2625] = 1050
DATA bitrev_size4096_radix4<>+0x5210(SB)/8, $2074 // bitrev[2626] = 2074
DATA bitrev_size4096_radix4<>+0x5218(SB)/8, $3098 // bitrev[2627] = 3098
DATA bitrev_size4096_radix4<>+0x5220(SB)/8, $282  // bitrev[2628] = 282
DATA bitrev_size4096_radix4<>+0x5228(SB)/8, $1306 // bitrev[2629] = 1306
DATA bitrev_size4096_radix4<>+0x5230(SB)/8, $2330 // bitrev[2630] = 2330
DATA bitrev_size4096_radix4<>+0x5238(SB)/8, $3354 // bitrev[2631] = 3354
DATA bitrev_size4096_radix4<>+0x5240(SB)/8, $538  // bitrev[2632] = 538
DATA bitrev_size4096_radix4<>+0x5248(SB)/8, $1562 // bitrev[2633] = 1562
DATA bitrev_size4096_radix4<>+0x5250(SB)/8, $2586 // bitrev[2634] = 2586
DATA bitrev_size4096_radix4<>+0x5258(SB)/8, $3610 // bitrev[2635] = 3610
DATA bitrev_size4096_radix4<>+0x5260(SB)/8, $794  // bitrev[2636] = 794
DATA bitrev_size4096_radix4<>+0x5268(SB)/8, $1818 // bitrev[2637] = 1818
DATA bitrev_size4096_radix4<>+0x5270(SB)/8, $2842 // bitrev[2638] = 2842
DATA bitrev_size4096_radix4<>+0x5278(SB)/8, $3866 // bitrev[2639] = 3866
DATA bitrev_size4096_radix4<>+0x5280(SB)/8, $90   // bitrev[2640] = 90
DATA bitrev_size4096_radix4<>+0x5288(SB)/8, $1114 // bitrev[2641] = 1114
DATA bitrev_size4096_radix4<>+0x5290(SB)/8, $2138 // bitrev[2642] = 2138
DATA bitrev_size4096_radix4<>+0x5298(SB)/8, $3162 // bitrev[2643] = 3162
DATA bitrev_size4096_radix4<>+0x52a0(SB)/8, $346  // bitrev[2644] = 346
DATA bitrev_size4096_radix4<>+0x52a8(SB)/8, $1370 // bitrev[2645] = 1370
DATA bitrev_size4096_radix4<>+0x52b0(SB)/8, $2394 // bitrev[2646] = 2394
DATA bitrev_size4096_radix4<>+0x52b8(SB)/8, $3418 // bitrev[2647] = 3418
DATA bitrev_size4096_radix4<>+0x52c0(SB)/8, $602  // bitrev[2648] = 602
DATA bitrev_size4096_radix4<>+0x52c8(SB)/8, $1626 // bitrev[2649] = 1626
DATA bitrev_size4096_radix4<>+0x52d0(SB)/8, $2650 // bitrev[2650] = 2650
DATA bitrev_size4096_radix4<>+0x52d8(SB)/8, $3674 // bitrev[2651] = 3674
DATA bitrev_size4096_radix4<>+0x52e0(SB)/8, $858  // bitrev[2652] = 858
DATA bitrev_size4096_radix4<>+0x52e8(SB)/8, $1882 // bitrev[2653] = 1882
DATA bitrev_size4096_radix4<>+0x52f0(SB)/8, $2906 // bitrev[2654] = 2906
DATA bitrev_size4096_radix4<>+0x52f8(SB)/8, $3930 // bitrev[2655] = 3930
DATA bitrev_size4096_radix4<>+0x5300(SB)/8, $154  // bitrev[2656] = 154
DATA bitrev_size4096_radix4<>+0x5308(SB)/8, $1178 // bitrev[2657] = 1178
DATA bitrev_size4096_radix4<>+0x5310(SB)/8, $2202 // bitrev[2658] = 2202
DATA bitrev_size4096_radix4<>+0x5318(SB)/8, $3226 // bitrev[2659] = 3226
DATA bitrev_size4096_radix4<>+0x5320(SB)/8, $410  // bitrev[2660] = 410
DATA bitrev_size4096_radix4<>+0x5328(SB)/8, $1434 // bitrev[2661] = 1434
DATA bitrev_size4096_radix4<>+0x5330(SB)/8, $2458 // bitrev[2662] = 2458
DATA bitrev_size4096_radix4<>+0x5338(SB)/8, $3482 // bitrev[2663] = 3482
DATA bitrev_size4096_radix4<>+0x5340(SB)/8, $666  // bitrev[2664] = 666
DATA bitrev_size4096_radix4<>+0x5348(SB)/8, $1690 // bitrev[2665] = 1690
DATA bitrev_size4096_radix4<>+0x5350(SB)/8, $2714 // bitrev[2666] = 2714
DATA bitrev_size4096_radix4<>+0x5358(SB)/8, $3738 // bitrev[2667] = 3738
DATA bitrev_size4096_radix4<>+0x5360(SB)/8, $922  // bitrev[2668] = 922
DATA bitrev_size4096_radix4<>+0x5368(SB)/8, $1946 // bitrev[2669] = 1946
DATA bitrev_size4096_radix4<>+0x5370(SB)/8, $2970 // bitrev[2670] = 2970
DATA bitrev_size4096_radix4<>+0x5378(SB)/8, $3994 // bitrev[2671] = 3994
DATA bitrev_size4096_radix4<>+0x5380(SB)/8, $218  // bitrev[2672] = 218
DATA bitrev_size4096_radix4<>+0x5388(SB)/8, $1242 // bitrev[2673] = 1242
DATA bitrev_size4096_radix4<>+0x5390(SB)/8, $2266 // bitrev[2674] = 2266
DATA bitrev_size4096_radix4<>+0x5398(SB)/8, $3290 // bitrev[2675] = 3290
DATA bitrev_size4096_radix4<>+0x53a0(SB)/8, $474  // bitrev[2676] = 474
DATA bitrev_size4096_radix4<>+0x53a8(SB)/8, $1498 // bitrev[2677] = 1498
DATA bitrev_size4096_radix4<>+0x53b0(SB)/8, $2522 // bitrev[2678] = 2522
DATA bitrev_size4096_radix4<>+0x53b8(SB)/8, $3546 // bitrev[2679] = 3546
DATA bitrev_size4096_radix4<>+0x53c0(SB)/8, $730  // bitrev[2680] = 730
DATA bitrev_size4096_radix4<>+0x53c8(SB)/8, $1754 // bitrev[2681] = 1754
DATA bitrev_size4096_radix4<>+0x53d0(SB)/8, $2778 // bitrev[2682] = 2778
DATA bitrev_size4096_radix4<>+0x53d8(SB)/8, $3802 // bitrev[2683] = 3802
DATA bitrev_size4096_radix4<>+0x53e0(SB)/8, $986  // bitrev[2684] = 986
DATA bitrev_size4096_radix4<>+0x53e8(SB)/8, $2010 // bitrev[2685] = 2010
DATA bitrev_size4096_radix4<>+0x53f0(SB)/8, $3034 // bitrev[2686] = 3034
DATA bitrev_size4096_radix4<>+0x53f8(SB)/8, $4058 // bitrev[2687] = 4058
DATA bitrev_size4096_radix4<>+0x5400(SB)/8, $42   // bitrev[2688] = 42
DATA bitrev_size4096_radix4<>+0x5408(SB)/8, $1066 // bitrev[2689] = 1066
DATA bitrev_size4096_radix4<>+0x5410(SB)/8, $2090 // bitrev[2690] = 2090
DATA bitrev_size4096_radix4<>+0x5418(SB)/8, $3114 // bitrev[2691] = 3114
DATA bitrev_size4096_radix4<>+0x5420(SB)/8, $298  // bitrev[2692] = 298
DATA bitrev_size4096_radix4<>+0x5428(SB)/8, $1322 // bitrev[2693] = 1322
DATA bitrev_size4096_radix4<>+0x5430(SB)/8, $2346 // bitrev[2694] = 2346
DATA bitrev_size4096_radix4<>+0x5438(SB)/8, $3370 // bitrev[2695] = 3370
DATA bitrev_size4096_radix4<>+0x5440(SB)/8, $554  // bitrev[2696] = 554
DATA bitrev_size4096_radix4<>+0x5448(SB)/8, $1578 // bitrev[2697] = 1578
DATA bitrev_size4096_radix4<>+0x5450(SB)/8, $2602 // bitrev[2698] = 2602
DATA bitrev_size4096_radix4<>+0x5458(SB)/8, $3626 // bitrev[2699] = 3626
DATA bitrev_size4096_radix4<>+0x5460(SB)/8, $810  // bitrev[2700] = 810
DATA bitrev_size4096_radix4<>+0x5468(SB)/8, $1834 // bitrev[2701] = 1834
DATA bitrev_size4096_radix4<>+0x5470(SB)/8, $2858 // bitrev[2702] = 2858
DATA bitrev_size4096_radix4<>+0x5478(SB)/8, $3882 // bitrev[2703] = 3882
DATA bitrev_size4096_radix4<>+0x5480(SB)/8, $106  // bitrev[2704] = 106
DATA bitrev_size4096_radix4<>+0x5488(SB)/8, $1130 // bitrev[2705] = 1130
DATA bitrev_size4096_radix4<>+0x5490(SB)/8, $2154 // bitrev[2706] = 2154
DATA bitrev_size4096_radix4<>+0x5498(SB)/8, $3178 // bitrev[2707] = 3178
DATA bitrev_size4096_radix4<>+0x54a0(SB)/8, $362  // bitrev[2708] = 362
DATA bitrev_size4096_radix4<>+0x54a8(SB)/8, $1386 // bitrev[2709] = 1386
DATA bitrev_size4096_radix4<>+0x54b0(SB)/8, $2410 // bitrev[2710] = 2410
DATA bitrev_size4096_radix4<>+0x54b8(SB)/8, $3434 // bitrev[2711] = 3434
DATA bitrev_size4096_radix4<>+0x54c0(SB)/8, $618  // bitrev[2712] = 618
DATA bitrev_size4096_radix4<>+0x54c8(SB)/8, $1642 // bitrev[2713] = 1642
DATA bitrev_size4096_radix4<>+0x54d0(SB)/8, $2666 // bitrev[2714] = 2666
DATA bitrev_size4096_radix4<>+0x54d8(SB)/8, $3690 // bitrev[2715] = 3690
DATA bitrev_size4096_radix4<>+0x54e0(SB)/8, $874  // bitrev[2716] = 874
DATA bitrev_size4096_radix4<>+0x54e8(SB)/8, $1898 // bitrev[2717] = 1898
DATA bitrev_size4096_radix4<>+0x54f0(SB)/8, $2922 // bitrev[2718] = 2922
DATA bitrev_size4096_radix4<>+0x54f8(SB)/8, $3946 // bitrev[2719] = 3946
DATA bitrev_size4096_radix4<>+0x5500(SB)/8, $170  // bitrev[2720] = 170
DATA bitrev_size4096_radix4<>+0x5508(SB)/8, $1194 // bitrev[2721] = 1194
DATA bitrev_size4096_radix4<>+0x5510(SB)/8, $2218 // bitrev[2722] = 2218
DATA bitrev_size4096_radix4<>+0x5518(SB)/8, $3242 // bitrev[2723] = 3242
DATA bitrev_size4096_radix4<>+0x5520(SB)/8, $426  // bitrev[2724] = 426
DATA bitrev_size4096_radix4<>+0x5528(SB)/8, $1450 // bitrev[2725] = 1450
DATA bitrev_size4096_radix4<>+0x5530(SB)/8, $2474 // bitrev[2726] = 2474
DATA bitrev_size4096_radix4<>+0x5538(SB)/8, $3498 // bitrev[2727] = 3498
DATA bitrev_size4096_radix4<>+0x5540(SB)/8, $682  // bitrev[2728] = 682
DATA bitrev_size4096_radix4<>+0x5548(SB)/8, $1706 // bitrev[2729] = 1706
DATA bitrev_size4096_radix4<>+0x5550(SB)/8, $2730 // bitrev[2730] = 2730
DATA bitrev_size4096_radix4<>+0x5558(SB)/8, $3754 // bitrev[2731] = 3754
DATA bitrev_size4096_radix4<>+0x5560(SB)/8, $938  // bitrev[2732] = 938
DATA bitrev_size4096_radix4<>+0x5568(SB)/8, $1962 // bitrev[2733] = 1962
DATA bitrev_size4096_radix4<>+0x5570(SB)/8, $2986 // bitrev[2734] = 2986
DATA bitrev_size4096_radix4<>+0x5578(SB)/8, $4010 // bitrev[2735] = 4010
DATA bitrev_size4096_radix4<>+0x5580(SB)/8, $234  // bitrev[2736] = 234
DATA bitrev_size4096_radix4<>+0x5588(SB)/8, $1258 // bitrev[2737] = 1258
DATA bitrev_size4096_radix4<>+0x5590(SB)/8, $2282 // bitrev[2738] = 2282
DATA bitrev_size4096_radix4<>+0x5598(SB)/8, $3306 // bitrev[2739] = 3306
DATA bitrev_size4096_radix4<>+0x55a0(SB)/8, $490  // bitrev[2740] = 490
DATA bitrev_size4096_radix4<>+0x55a8(SB)/8, $1514 // bitrev[2741] = 1514
DATA bitrev_size4096_radix4<>+0x55b0(SB)/8, $2538 // bitrev[2742] = 2538
DATA bitrev_size4096_radix4<>+0x55b8(SB)/8, $3562 // bitrev[2743] = 3562
DATA bitrev_size4096_radix4<>+0x55c0(SB)/8, $746  // bitrev[2744] = 746
DATA bitrev_size4096_radix4<>+0x55c8(SB)/8, $1770 // bitrev[2745] = 1770
DATA bitrev_size4096_radix4<>+0x55d0(SB)/8, $2794 // bitrev[2746] = 2794
DATA bitrev_size4096_radix4<>+0x55d8(SB)/8, $3818 // bitrev[2747] = 3818
DATA bitrev_size4096_radix4<>+0x55e0(SB)/8, $1002 // bitrev[2748] = 1002
DATA bitrev_size4096_radix4<>+0x55e8(SB)/8, $2026 // bitrev[2749] = 2026
DATA bitrev_size4096_radix4<>+0x55f0(SB)/8, $3050 // bitrev[2750] = 3050
DATA bitrev_size4096_radix4<>+0x55f8(SB)/8, $4074 // bitrev[2751] = 4074
DATA bitrev_size4096_radix4<>+0x5600(SB)/8, $58   // bitrev[2752] = 58
DATA bitrev_size4096_radix4<>+0x5608(SB)/8, $1082 // bitrev[2753] = 1082
DATA bitrev_size4096_radix4<>+0x5610(SB)/8, $2106 // bitrev[2754] = 2106
DATA bitrev_size4096_radix4<>+0x5618(SB)/8, $3130 // bitrev[2755] = 3130
DATA bitrev_size4096_radix4<>+0x5620(SB)/8, $314  // bitrev[2756] = 314
DATA bitrev_size4096_radix4<>+0x5628(SB)/8, $1338 // bitrev[2757] = 1338
DATA bitrev_size4096_radix4<>+0x5630(SB)/8, $2362 // bitrev[2758] = 2362
DATA bitrev_size4096_radix4<>+0x5638(SB)/8, $3386 // bitrev[2759] = 3386
DATA bitrev_size4096_radix4<>+0x5640(SB)/8, $570  // bitrev[2760] = 570
DATA bitrev_size4096_radix4<>+0x5648(SB)/8, $1594 // bitrev[2761] = 1594
DATA bitrev_size4096_radix4<>+0x5650(SB)/8, $2618 // bitrev[2762] = 2618
DATA bitrev_size4096_radix4<>+0x5658(SB)/8, $3642 // bitrev[2763] = 3642
DATA bitrev_size4096_radix4<>+0x5660(SB)/8, $826  // bitrev[2764] = 826
DATA bitrev_size4096_radix4<>+0x5668(SB)/8, $1850 // bitrev[2765] = 1850
DATA bitrev_size4096_radix4<>+0x5670(SB)/8, $2874 // bitrev[2766] = 2874
DATA bitrev_size4096_radix4<>+0x5678(SB)/8, $3898 // bitrev[2767] = 3898
DATA bitrev_size4096_radix4<>+0x5680(SB)/8, $122  // bitrev[2768] = 122
DATA bitrev_size4096_radix4<>+0x5688(SB)/8, $1146 // bitrev[2769] = 1146
DATA bitrev_size4096_radix4<>+0x5690(SB)/8, $2170 // bitrev[2770] = 2170
DATA bitrev_size4096_radix4<>+0x5698(SB)/8, $3194 // bitrev[2771] = 3194
DATA bitrev_size4096_radix4<>+0x56a0(SB)/8, $378  // bitrev[2772] = 378
DATA bitrev_size4096_radix4<>+0x56a8(SB)/8, $1402 // bitrev[2773] = 1402
DATA bitrev_size4096_radix4<>+0x56b0(SB)/8, $2426 // bitrev[2774] = 2426
DATA bitrev_size4096_radix4<>+0x56b8(SB)/8, $3450 // bitrev[2775] = 3450
DATA bitrev_size4096_radix4<>+0x56c0(SB)/8, $634  // bitrev[2776] = 634
DATA bitrev_size4096_radix4<>+0x56c8(SB)/8, $1658 // bitrev[2777] = 1658
DATA bitrev_size4096_radix4<>+0x56d0(SB)/8, $2682 // bitrev[2778] = 2682
DATA bitrev_size4096_radix4<>+0x56d8(SB)/8, $3706 // bitrev[2779] = 3706
DATA bitrev_size4096_radix4<>+0x56e0(SB)/8, $890  // bitrev[2780] = 890
DATA bitrev_size4096_radix4<>+0x56e8(SB)/8, $1914 // bitrev[2781] = 1914
DATA bitrev_size4096_radix4<>+0x56f0(SB)/8, $2938 // bitrev[2782] = 2938
DATA bitrev_size4096_radix4<>+0x56f8(SB)/8, $3962 // bitrev[2783] = 3962
DATA bitrev_size4096_radix4<>+0x5700(SB)/8, $186  // bitrev[2784] = 186
DATA bitrev_size4096_radix4<>+0x5708(SB)/8, $1210 // bitrev[2785] = 1210
DATA bitrev_size4096_radix4<>+0x5710(SB)/8, $2234 // bitrev[2786] = 2234
DATA bitrev_size4096_radix4<>+0x5718(SB)/8, $3258 // bitrev[2787] = 3258
DATA bitrev_size4096_radix4<>+0x5720(SB)/8, $442  // bitrev[2788] = 442
DATA bitrev_size4096_radix4<>+0x5728(SB)/8, $1466 // bitrev[2789] = 1466
DATA bitrev_size4096_radix4<>+0x5730(SB)/8, $2490 // bitrev[2790] = 2490
DATA bitrev_size4096_radix4<>+0x5738(SB)/8, $3514 // bitrev[2791] = 3514
DATA bitrev_size4096_radix4<>+0x5740(SB)/8, $698  // bitrev[2792] = 698
DATA bitrev_size4096_radix4<>+0x5748(SB)/8, $1722 // bitrev[2793] = 1722
DATA bitrev_size4096_radix4<>+0x5750(SB)/8, $2746 // bitrev[2794] = 2746
DATA bitrev_size4096_radix4<>+0x5758(SB)/8, $3770 // bitrev[2795] = 3770
DATA bitrev_size4096_radix4<>+0x5760(SB)/8, $954  // bitrev[2796] = 954
DATA bitrev_size4096_radix4<>+0x5768(SB)/8, $1978 // bitrev[2797] = 1978
DATA bitrev_size4096_radix4<>+0x5770(SB)/8, $3002 // bitrev[2798] = 3002
DATA bitrev_size4096_radix4<>+0x5778(SB)/8, $4026 // bitrev[2799] = 4026
DATA bitrev_size4096_radix4<>+0x5780(SB)/8, $250  // bitrev[2800] = 250
DATA bitrev_size4096_radix4<>+0x5788(SB)/8, $1274 // bitrev[2801] = 1274
DATA bitrev_size4096_radix4<>+0x5790(SB)/8, $2298 // bitrev[2802] = 2298
DATA bitrev_size4096_radix4<>+0x5798(SB)/8, $3322 // bitrev[2803] = 3322
DATA bitrev_size4096_radix4<>+0x57a0(SB)/8, $506  // bitrev[2804] = 506
DATA bitrev_size4096_radix4<>+0x57a8(SB)/8, $1530 // bitrev[2805] = 1530
DATA bitrev_size4096_radix4<>+0x57b0(SB)/8, $2554 // bitrev[2806] = 2554
DATA bitrev_size4096_radix4<>+0x57b8(SB)/8, $3578 // bitrev[2807] = 3578
DATA bitrev_size4096_radix4<>+0x57c0(SB)/8, $762  // bitrev[2808] = 762
DATA bitrev_size4096_radix4<>+0x57c8(SB)/8, $1786 // bitrev[2809] = 1786
DATA bitrev_size4096_radix4<>+0x57d0(SB)/8, $2810 // bitrev[2810] = 2810
DATA bitrev_size4096_radix4<>+0x57d8(SB)/8, $3834 // bitrev[2811] = 3834
DATA bitrev_size4096_radix4<>+0x57e0(SB)/8, $1018 // bitrev[2812] = 1018
DATA bitrev_size4096_radix4<>+0x57e8(SB)/8, $2042 // bitrev[2813] = 2042
DATA bitrev_size4096_radix4<>+0x57f0(SB)/8, $3066 // bitrev[2814] = 3066
DATA bitrev_size4096_radix4<>+0x57f8(SB)/8, $4090 // bitrev[2815] = 4090
DATA bitrev_size4096_radix4<>+0x5800(SB)/8, $14   // bitrev[2816] = 14
DATA bitrev_size4096_radix4<>+0x5808(SB)/8, $1038 // bitrev[2817] = 1038
DATA bitrev_size4096_radix4<>+0x5810(SB)/8, $2062 // bitrev[2818] = 2062
DATA bitrev_size4096_radix4<>+0x5818(SB)/8, $3086 // bitrev[2819] = 3086
DATA bitrev_size4096_radix4<>+0x5820(SB)/8, $270  // bitrev[2820] = 270
DATA bitrev_size4096_radix4<>+0x5828(SB)/8, $1294 // bitrev[2821] = 1294
DATA bitrev_size4096_radix4<>+0x5830(SB)/8, $2318 // bitrev[2822] = 2318
DATA bitrev_size4096_radix4<>+0x5838(SB)/8, $3342 // bitrev[2823] = 3342
DATA bitrev_size4096_radix4<>+0x5840(SB)/8, $526  // bitrev[2824] = 526
DATA bitrev_size4096_radix4<>+0x5848(SB)/8, $1550 // bitrev[2825] = 1550
DATA bitrev_size4096_radix4<>+0x5850(SB)/8, $2574 // bitrev[2826] = 2574
DATA bitrev_size4096_radix4<>+0x5858(SB)/8, $3598 // bitrev[2827] = 3598
DATA bitrev_size4096_radix4<>+0x5860(SB)/8, $782  // bitrev[2828] = 782
DATA bitrev_size4096_radix4<>+0x5868(SB)/8, $1806 // bitrev[2829] = 1806
DATA bitrev_size4096_radix4<>+0x5870(SB)/8, $2830 // bitrev[2830] = 2830
DATA bitrev_size4096_radix4<>+0x5878(SB)/8, $3854 // bitrev[2831] = 3854
DATA bitrev_size4096_radix4<>+0x5880(SB)/8, $78   // bitrev[2832] = 78
DATA bitrev_size4096_radix4<>+0x5888(SB)/8, $1102 // bitrev[2833] = 1102
DATA bitrev_size4096_radix4<>+0x5890(SB)/8, $2126 // bitrev[2834] = 2126
DATA bitrev_size4096_radix4<>+0x5898(SB)/8, $3150 // bitrev[2835] = 3150
DATA bitrev_size4096_radix4<>+0x58a0(SB)/8, $334  // bitrev[2836] = 334
DATA bitrev_size4096_radix4<>+0x58a8(SB)/8, $1358 // bitrev[2837] = 1358
DATA bitrev_size4096_radix4<>+0x58b0(SB)/8, $2382 // bitrev[2838] = 2382
DATA bitrev_size4096_radix4<>+0x58b8(SB)/8, $3406 // bitrev[2839] = 3406
DATA bitrev_size4096_radix4<>+0x58c0(SB)/8, $590  // bitrev[2840] = 590
DATA bitrev_size4096_radix4<>+0x58c8(SB)/8, $1614 // bitrev[2841] = 1614
DATA bitrev_size4096_radix4<>+0x58d0(SB)/8, $2638 // bitrev[2842] = 2638
DATA bitrev_size4096_radix4<>+0x58d8(SB)/8, $3662 // bitrev[2843] = 3662
DATA bitrev_size4096_radix4<>+0x58e0(SB)/8, $846  // bitrev[2844] = 846
DATA bitrev_size4096_radix4<>+0x58e8(SB)/8, $1870 // bitrev[2845] = 1870
DATA bitrev_size4096_radix4<>+0x58f0(SB)/8, $2894 // bitrev[2846] = 2894
DATA bitrev_size4096_radix4<>+0x58f8(SB)/8, $3918 // bitrev[2847] = 3918
DATA bitrev_size4096_radix4<>+0x5900(SB)/8, $142  // bitrev[2848] = 142
DATA bitrev_size4096_radix4<>+0x5908(SB)/8, $1166 // bitrev[2849] = 1166
DATA bitrev_size4096_radix4<>+0x5910(SB)/8, $2190 // bitrev[2850] = 2190
DATA bitrev_size4096_radix4<>+0x5918(SB)/8, $3214 // bitrev[2851] = 3214
DATA bitrev_size4096_radix4<>+0x5920(SB)/8, $398  // bitrev[2852] = 398
DATA bitrev_size4096_radix4<>+0x5928(SB)/8, $1422 // bitrev[2853] = 1422
DATA bitrev_size4096_radix4<>+0x5930(SB)/8, $2446 // bitrev[2854] = 2446
DATA bitrev_size4096_radix4<>+0x5938(SB)/8, $3470 // bitrev[2855] = 3470
DATA bitrev_size4096_radix4<>+0x5940(SB)/8, $654  // bitrev[2856] = 654
DATA bitrev_size4096_radix4<>+0x5948(SB)/8, $1678 // bitrev[2857] = 1678
DATA bitrev_size4096_radix4<>+0x5950(SB)/8, $2702 // bitrev[2858] = 2702
DATA bitrev_size4096_radix4<>+0x5958(SB)/8, $3726 // bitrev[2859] = 3726
DATA bitrev_size4096_radix4<>+0x5960(SB)/8, $910  // bitrev[2860] = 910
DATA bitrev_size4096_radix4<>+0x5968(SB)/8, $1934 // bitrev[2861] = 1934
DATA bitrev_size4096_radix4<>+0x5970(SB)/8, $2958 // bitrev[2862] = 2958
DATA bitrev_size4096_radix4<>+0x5978(SB)/8, $3982 // bitrev[2863] = 3982
DATA bitrev_size4096_radix4<>+0x5980(SB)/8, $206  // bitrev[2864] = 206
DATA bitrev_size4096_radix4<>+0x5988(SB)/8, $1230 // bitrev[2865] = 1230
DATA bitrev_size4096_radix4<>+0x5990(SB)/8, $2254 // bitrev[2866] = 2254
DATA bitrev_size4096_radix4<>+0x5998(SB)/8, $3278 // bitrev[2867] = 3278
DATA bitrev_size4096_radix4<>+0x59a0(SB)/8, $462  // bitrev[2868] = 462
DATA bitrev_size4096_radix4<>+0x59a8(SB)/8, $1486 // bitrev[2869] = 1486
DATA bitrev_size4096_radix4<>+0x59b0(SB)/8, $2510 // bitrev[2870] = 2510
DATA bitrev_size4096_radix4<>+0x59b8(SB)/8, $3534 // bitrev[2871] = 3534
DATA bitrev_size4096_radix4<>+0x59c0(SB)/8, $718  // bitrev[2872] = 718
DATA bitrev_size4096_radix4<>+0x59c8(SB)/8, $1742 // bitrev[2873] = 1742
DATA bitrev_size4096_radix4<>+0x59d0(SB)/8, $2766 // bitrev[2874] = 2766
DATA bitrev_size4096_radix4<>+0x59d8(SB)/8, $3790 // bitrev[2875] = 3790
DATA bitrev_size4096_radix4<>+0x59e0(SB)/8, $974  // bitrev[2876] = 974
DATA bitrev_size4096_radix4<>+0x59e8(SB)/8, $1998 // bitrev[2877] = 1998
DATA bitrev_size4096_radix4<>+0x59f0(SB)/8, $3022 // bitrev[2878] = 3022
DATA bitrev_size4096_radix4<>+0x59f8(SB)/8, $4046 // bitrev[2879] = 4046
DATA bitrev_size4096_radix4<>+0x5a00(SB)/8, $30   // bitrev[2880] = 30
DATA bitrev_size4096_radix4<>+0x5a08(SB)/8, $1054 // bitrev[2881] = 1054
DATA bitrev_size4096_radix4<>+0x5a10(SB)/8, $2078 // bitrev[2882] = 2078
DATA bitrev_size4096_radix4<>+0x5a18(SB)/8, $3102 // bitrev[2883] = 3102
DATA bitrev_size4096_radix4<>+0x5a20(SB)/8, $286  // bitrev[2884] = 286
DATA bitrev_size4096_radix4<>+0x5a28(SB)/8, $1310 // bitrev[2885] = 1310
DATA bitrev_size4096_radix4<>+0x5a30(SB)/8, $2334 // bitrev[2886] = 2334
DATA bitrev_size4096_radix4<>+0x5a38(SB)/8, $3358 // bitrev[2887] = 3358
DATA bitrev_size4096_radix4<>+0x5a40(SB)/8, $542  // bitrev[2888] = 542
DATA bitrev_size4096_radix4<>+0x5a48(SB)/8, $1566 // bitrev[2889] = 1566
DATA bitrev_size4096_radix4<>+0x5a50(SB)/8, $2590 // bitrev[2890] = 2590
DATA bitrev_size4096_radix4<>+0x5a58(SB)/8, $3614 // bitrev[2891] = 3614
DATA bitrev_size4096_radix4<>+0x5a60(SB)/8, $798  // bitrev[2892] = 798
DATA bitrev_size4096_radix4<>+0x5a68(SB)/8, $1822 // bitrev[2893] = 1822
DATA bitrev_size4096_radix4<>+0x5a70(SB)/8, $2846 // bitrev[2894] = 2846
DATA bitrev_size4096_radix4<>+0x5a78(SB)/8, $3870 // bitrev[2895] = 3870
DATA bitrev_size4096_radix4<>+0x5a80(SB)/8, $94   // bitrev[2896] = 94
DATA bitrev_size4096_radix4<>+0x5a88(SB)/8, $1118 // bitrev[2897] = 1118
DATA bitrev_size4096_radix4<>+0x5a90(SB)/8, $2142 // bitrev[2898] = 2142
DATA bitrev_size4096_radix4<>+0x5a98(SB)/8, $3166 // bitrev[2899] = 3166
DATA bitrev_size4096_radix4<>+0x5aa0(SB)/8, $350  // bitrev[2900] = 350
DATA bitrev_size4096_radix4<>+0x5aa8(SB)/8, $1374 // bitrev[2901] = 1374
DATA bitrev_size4096_radix4<>+0x5ab0(SB)/8, $2398 // bitrev[2902] = 2398
DATA bitrev_size4096_radix4<>+0x5ab8(SB)/8, $3422 // bitrev[2903] = 3422
DATA bitrev_size4096_radix4<>+0x5ac0(SB)/8, $606  // bitrev[2904] = 606
DATA bitrev_size4096_radix4<>+0x5ac8(SB)/8, $1630 // bitrev[2905] = 1630
DATA bitrev_size4096_radix4<>+0x5ad0(SB)/8, $2654 // bitrev[2906] = 2654
DATA bitrev_size4096_radix4<>+0x5ad8(SB)/8, $3678 // bitrev[2907] = 3678
DATA bitrev_size4096_radix4<>+0x5ae0(SB)/8, $862  // bitrev[2908] = 862
DATA bitrev_size4096_radix4<>+0x5ae8(SB)/8, $1886 // bitrev[2909] = 1886
DATA bitrev_size4096_radix4<>+0x5af0(SB)/8, $2910 // bitrev[2910] = 2910
DATA bitrev_size4096_radix4<>+0x5af8(SB)/8, $3934 // bitrev[2911] = 3934
DATA bitrev_size4096_radix4<>+0x5b00(SB)/8, $158  // bitrev[2912] = 158
DATA bitrev_size4096_radix4<>+0x5b08(SB)/8, $1182 // bitrev[2913] = 1182
DATA bitrev_size4096_radix4<>+0x5b10(SB)/8, $2206 // bitrev[2914] = 2206
DATA bitrev_size4096_radix4<>+0x5b18(SB)/8, $3230 // bitrev[2915] = 3230
DATA bitrev_size4096_radix4<>+0x5b20(SB)/8, $414  // bitrev[2916] = 414
DATA bitrev_size4096_radix4<>+0x5b28(SB)/8, $1438 // bitrev[2917] = 1438
DATA bitrev_size4096_radix4<>+0x5b30(SB)/8, $2462 // bitrev[2918] = 2462
DATA bitrev_size4096_radix4<>+0x5b38(SB)/8, $3486 // bitrev[2919] = 3486
DATA bitrev_size4096_radix4<>+0x5b40(SB)/8, $670  // bitrev[2920] = 670
DATA bitrev_size4096_radix4<>+0x5b48(SB)/8, $1694 // bitrev[2921] = 1694
DATA bitrev_size4096_radix4<>+0x5b50(SB)/8, $2718 // bitrev[2922] = 2718
DATA bitrev_size4096_radix4<>+0x5b58(SB)/8, $3742 // bitrev[2923] = 3742
DATA bitrev_size4096_radix4<>+0x5b60(SB)/8, $926  // bitrev[2924] = 926
DATA bitrev_size4096_radix4<>+0x5b68(SB)/8, $1950 // bitrev[2925] = 1950
DATA bitrev_size4096_radix4<>+0x5b70(SB)/8, $2974 // bitrev[2926] = 2974
DATA bitrev_size4096_radix4<>+0x5b78(SB)/8, $3998 // bitrev[2927] = 3998
DATA bitrev_size4096_radix4<>+0x5b80(SB)/8, $222  // bitrev[2928] = 222
DATA bitrev_size4096_radix4<>+0x5b88(SB)/8, $1246 // bitrev[2929] = 1246
DATA bitrev_size4096_radix4<>+0x5b90(SB)/8, $2270 // bitrev[2930] = 2270
DATA bitrev_size4096_radix4<>+0x5b98(SB)/8, $3294 // bitrev[2931] = 3294
DATA bitrev_size4096_radix4<>+0x5ba0(SB)/8, $478  // bitrev[2932] = 478
DATA bitrev_size4096_radix4<>+0x5ba8(SB)/8, $1502 // bitrev[2933] = 1502
DATA bitrev_size4096_radix4<>+0x5bb0(SB)/8, $2526 // bitrev[2934] = 2526
DATA bitrev_size4096_radix4<>+0x5bb8(SB)/8, $3550 // bitrev[2935] = 3550
DATA bitrev_size4096_radix4<>+0x5bc0(SB)/8, $734  // bitrev[2936] = 734
DATA bitrev_size4096_radix4<>+0x5bc8(SB)/8, $1758 // bitrev[2937] = 1758
DATA bitrev_size4096_radix4<>+0x5bd0(SB)/8, $2782 // bitrev[2938] = 2782
DATA bitrev_size4096_radix4<>+0x5bd8(SB)/8, $3806 // bitrev[2939] = 3806
DATA bitrev_size4096_radix4<>+0x5be0(SB)/8, $990  // bitrev[2940] = 990
DATA bitrev_size4096_radix4<>+0x5be8(SB)/8, $2014 // bitrev[2941] = 2014
DATA bitrev_size4096_radix4<>+0x5bf0(SB)/8, $3038 // bitrev[2942] = 3038
DATA bitrev_size4096_radix4<>+0x5bf8(SB)/8, $4062 // bitrev[2943] = 4062
DATA bitrev_size4096_radix4<>+0x5c00(SB)/8, $46   // bitrev[2944] = 46
DATA bitrev_size4096_radix4<>+0x5c08(SB)/8, $1070 // bitrev[2945] = 1070
DATA bitrev_size4096_radix4<>+0x5c10(SB)/8, $2094 // bitrev[2946] = 2094
DATA bitrev_size4096_radix4<>+0x5c18(SB)/8, $3118 // bitrev[2947] = 3118
DATA bitrev_size4096_radix4<>+0x5c20(SB)/8, $302  // bitrev[2948] = 302
DATA bitrev_size4096_radix4<>+0x5c28(SB)/8, $1326 // bitrev[2949] = 1326
DATA bitrev_size4096_radix4<>+0x5c30(SB)/8, $2350 // bitrev[2950] = 2350
DATA bitrev_size4096_radix4<>+0x5c38(SB)/8, $3374 // bitrev[2951] = 3374
DATA bitrev_size4096_radix4<>+0x5c40(SB)/8, $558  // bitrev[2952] = 558
DATA bitrev_size4096_radix4<>+0x5c48(SB)/8, $1582 // bitrev[2953] = 1582
DATA bitrev_size4096_radix4<>+0x5c50(SB)/8, $2606 // bitrev[2954] = 2606
DATA bitrev_size4096_radix4<>+0x5c58(SB)/8, $3630 // bitrev[2955] = 3630
DATA bitrev_size4096_radix4<>+0x5c60(SB)/8, $814  // bitrev[2956] = 814
DATA bitrev_size4096_radix4<>+0x5c68(SB)/8, $1838 // bitrev[2957] = 1838
DATA bitrev_size4096_radix4<>+0x5c70(SB)/8, $2862 // bitrev[2958] = 2862
DATA bitrev_size4096_radix4<>+0x5c78(SB)/8, $3886 // bitrev[2959] = 3886
DATA bitrev_size4096_radix4<>+0x5c80(SB)/8, $110  // bitrev[2960] = 110
DATA bitrev_size4096_radix4<>+0x5c88(SB)/8, $1134 // bitrev[2961] = 1134
DATA bitrev_size4096_radix4<>+0x5c90(SB)/8, $2158 // bitrev[2962] = 2158
DATA bitrev_size4096_radix4<>+0x5c98(SB)/8, $3182 // bitrev[2963] = 3182
DATA bitrev_size4096_radix4<>+0x5ca0(SB)/8, $366  // bitrev[2964] = 366
DATA bitrev_size4096_radix4<>+0x5ca8(SB)/8, $1390 // bitrev[2965] = 1390
DATA bitrev_size4096_radix4<>+0x5cb0(SB)/8, $2414 // bitrev[2966] = 2414
DATA bitrev_size4096_radix4<>+0x5cb8(SB)/8, $3438 // bitrev[2967] = 3438
DATA bitrev_size4096_radix4<>+0x5cc0(SB)/8, $622  // bitrev[2968] = 622
DATA bitrev_size4096_radix4<>+0x5cc8(SB)/8, $1646 // bitrev[2969] = 1646
DATA bitrev_size4096_radix4<>+0x5cd0(SB)/8, $2670 // bitrev[2970] = 2670
DATA bitrev_size4096_radix4<>+0x5cd8(SB)/8, $3694 // bitrev[2971] = 3694
DATA bitrev_size4096_radix4<>+0x5ce0(SB)/8, $878  // bitrev[2972] = 878
DATA bitrev_size4096_radix4<>+0x5ce8(SB)/8, $1902 // bitrev[2973] = 1902
DATA bitrev_size4096_radix4<>+0x5cf0(SB)/8, $2926 // bitrev[2974] = 2926
DATA bitrev_size4096_radix4<>+0x5cf8(SB)/8, $3950 // bitrev[2975] = 3950
DATA bitrev_size4096_radix4<>+0x5d00(SB)/8, $174  // bitrev[2976] = 174
DATA bitrev_size4096_radix4<>+0x5d08(SB)/8, $1198 // bitrev[2977] = 1198
DATA bitrev_size4096_radix4<>+0x5d10(SB)/8, $2222 // bitrev[2978] = 2222
DATA bitrev_size4096_radix4<>+0x5d18(SB)/8, $3246 // bitrev[2979] = 3246
DATA bitrev_size4096_radix4<>+0x5d20(SB)/8, $430  // bitrev[2980] = 430
DATA bitrev_size4096_radix4<>+0x5d28(SB)/8, $1454 // bitrev[2981] = 1454
DATA bitrev_size4096_radix4<>+0x5d30(SB)/8, $2478 // bitrev[2982] = 2478
DATA bitrev_size4096_radix4<>+0x5d38(SB)/8, $3502 // bitrev[2983] = 3502
DATA bitrev_size4096_radix4<>+0x5d40(SB)/8, $686  // bitrev[2984] = 686
DATA bitrev_size4096_radix4<>+0x5d48(SB)/8, $1710 // bitrev[2985] = 1710
DATA bitrev_size4096_radix4<>+0x5d50(SB)/8, $2734 // bitrev[2986] = 2734
DATA bitrev_size4096_radix4<>+0x5d58(SB)/8, $3758 // bitrev[2987] = 3758
DATA bitrev_size4096_radix4<>+0x5d60(SB)/8, $942  // bitrev[2988] = 942
DATA bitrev_size4096_radix4<>+0x5d68(SB)/8, $1966 // bitrev[2989] = 1966
DATA bitrev_size4096_radix4<>+0x5d70(SB)/8, $2990 // bitrev[2990] = 2990
DATA bitrev_size4096_radix4<>+0x5d78(SB)/8, $4014 // bitrev[2991] = 4014
DATA bitrev_size4096_radix4<>+0x5d80(SB)/8, $238  // bitrev[2992] = 238
DATA bitrev_size4096_radix4<>+0x5d88(SB)/8, $1262 // bitrev[2993] = 1262
DATA bitrev_size4096_radix4<>+0x5d90(SB)/8, $2286 // bitrev[2994] = 2286
DATA bitrev_size4096_radix4<>+0x5d98(SB)/8, $3310 // bitrev[2995] = 3310
DATA bitrev_size4096_radix4<>+0x5da0(SB)/8, $494  // bitrev[2996] = 494
DATA bitrev_size4096_radix4<>+0x5da8(SB)/8, $1518 // bitrev[2997] = 1518
DATA bitrev_size4096_radix4<>+0x5db0(SB)/8, $2542 // bitrev[2998] = 2542
DATA bitrev_size4096_radix4<>+0x5db8(SB)/8, $3566 // bitrev[2999] = 3566
DATA bitrev_size4096_radix4<>+0x5dc0(SB)/8, $750  // bitrev[3000] = 750
DATA bitrev_size4096_radix4<>+0x5dc8(SB)/8, $1774 // bitrev[3001] = 1774
DATA bitrev_size4096_radix4<>+0x5dd0(SB)/8, $2798 // bitrev[3002] = 2798
DATA bitrev_size4096_radix4<>+0x5dd8(SB)/8, $3822 // bitrev[3003] = 3822
DATA bitrev_size4096_radix4<>+0x5de0(SB)/8, $1006 // bitrev[3004] = 1006
DATA bitrev_size4096_radix4<>+0x5de8(SB)/8, $2030 // bitrev[3005] = 2030
DATA bitrev_size4096_radix4<>+0x5df0(SB)/8, $3054 // bitrev[3006] = 3054
DATA bitrev_size4096_radix4<>+0x5df8(SB)/8, $4078 // bitrev[3007] = 4078
DATA bitrev_size4096_radix4<>+0x5e00(SB)/8, $62   // bitrev[3008] = 62
DATA bitrev_size4096_radix4<>+0x5e08(SB)/8, $1086 // bitrev[3009] = 1086
DATA bitrev_size4096_radix4<>+0x5e10(SB)/8, $2110 // bitrev[3010] = 2110
DATA bitrev_size4096_radix4<>+0x5e18(SB)/8, $3134 // bitrev[3011] = 3134
DATA bitrev_size4096_radix4<>+0x5e20(SB)/8, $318  // bitrev[3012] = 318
DATA bitrev_size4096_radix4<>+0x5e28(SB)/8, $1342 // bitrev[3013] = 1342
DATA bitrev_size4096_radix4<>+0x5e30(SB)/8, $2366 // bitrev[3014] = 2366
DATA bitrev_size4096_radix4<>+0x5e38(SB)/8, $3390 // bitrev[3015] = 3390
DATA bitrev_size4096_radix4<>+0x5e40(SB)/8, $574  // bitrev[3016] = 574
DATA bitrev_size4096_radix4<>+0x5e48(SB)/8, $1598 // bitrev[3017] = 1598
DATA bitrev_size4096_radix4<>+0x5e50(SB)/8, $2622 // bitrev[3018] = 2622
DATA bitrev_size4096_radix4<>+0x5e58(SB)/8, $3646 // bitrev[3019] = 3646
DATA bitrev_size4096_radix4<>+0x5e60(SB)/8, $830  // bitrev[3020] = 830
DATA bitrev_size4096_radix4<>+0x5e68(SB)/8, $1854 // bitrev[3021] = 1854
DATA bitrev_size4096_radix4<>+0x5e70(SB)/8, $2878 // bitrev[3022] = 2878
DATA bitrev_size4096_radix4<>+0x5e78(SB)/8, $3902 // bitrev[3023] = 3902
DATA bitrev_size4096_radix4<>+0x5e80(SB)/8, $126  // bitrev[3024] = 126
DATA bitrev_size4096_radix4<>+0x5e88(SB)/8, $1150 // bitrev[3025] = 1150
DATA bitrev_size4096_radix4<>+0x5e90(SB)/8, $2174 // bitrev[3026] = 2174
DATA bitrev_size4096_radix4<>+0x5e98(SB)/8, $3198 // bitrev[3027] = 3198
DATA bitrev_size4096_radix4<>+0x5ea0(SB)/8, $382  // bitrev[3028] = 382
DATA bitrev_size4096_radix4<>+0x5ea8(SB)/8, $1406 // bitrev[3029] = 1406
DATA bitrev_size4096_radix4<>+0x5eb0(SB)/8, $2430 // bitrev[3030] = 2430
DATA bitrev_size4096_radix4<>+0x5eb8(SB)/8, $3454 // bitrev[3031] = 3454
DATA bitrev_size4096_radix4<>+0x5ec0(SB)/8, $638  // bitrev[3032] = 638
DATA bitrev_size4096_radix4<>+0x5ec8(SB)/8, $1662 // bitrev[3033] = 1662
DATA bitrev_size4096_radix4<>+0x5ed0(SB)/8, $2686 // bitrev[3034] = 2686
DATA bitrev_size4096_radix4<>+0x5ed8(SB)/8, $3710 // bitrev[3035] = 3710
DATA bitrev_size4096_radix4<>+0x5ee0(SB)/8, $894  // bitrev[3036] = 894
DATA bitrev_size4096_radix4<>+0x5ee8(SB)/8, $1918 // bitrev[3037] = 1918
DATA bitrev_size4096_radix4<>+0x5ef0(SB)/8, $2942 // bitrev[3038] = 2942
DATA bitrev_size4096_radix4<>+0x5ef8(SB)/8, $3966 // bitrev[3039] = 3966
DATA bitrev_size4096_radix4<>+0x5f00(SB)/8, $190  // bitrev[3040] = 190
DATA bitrev_size4096_radix4<>+0x5f08(SB)/8, $1214 // bitrev[3041] = 1214
DATA bitrev_size4096_radix4<>+0x5f10(SB)/8, $2238 // bitrev[3042] = 2238
DATA bitrev_size4096_radix4<>+0x5f18(SB)/8, $3262 // bitrev[3043] = 3262
DATA bitrev_size4096_radix4<>+0x5f20(SB)/8, $446  // bitrev[3044] = 446
DATA bitrev_size4096_radix4<>+0x5f28(SB)/8, $1470 // bitrev[3045] = 1470
DATA bitrev_size4096_radix4<>+0x5f30(SB)/8, $2494 // bitrev[3046] = 2494
DATA bitrev_size4096_radix4<>+0x5f38(SB)/8, $3518 // bitrev[3047] = 3518
DATA bitrev_size4096_radix4<>+0x5f40(SB)/8, $702  // bitrev[3048] = 702
DATA bitrev_size4096_radix4<>+0x5f48(SB)/8, $1726 // bitrev[3049] = 1726
DATA bitrev_size4096_radix4<>+0x5f50(SB)/8, $2750 // bitrev[3050] = 2750
DATA bitrev_size4096_radix4<>+0x5f58(SB)/8, $3774 // bitrev[3051] = 3774
DATA bitrev_size4096_radix4<>+0x5f60(SB)/8, $958  // bitrev[3052] = 958
DATA bitrev_size4096_radix4<>+0x5f68(SB)/8, $1982 // bitrev[3053] = 1982
DATA bitrev_size4096_radix4<>+0x5f70(SB)/8, $3006 // bitrev[3054] = 3006
DATA bitrev_size4096_radix4<>+0x5f78(SB)/8, $4030 // bitrev[3055] = 4030
DATA bitrev_size4096_radix4<>+0x5f80(SB)/8, $254  // bitrev[3056] = 254
DATA bitrev_size4096_radix4<>+0x5f88(SB)/8, $1278 // bitrev[3057] = 1278
DATA bitrev_size4096_radix4<>+0x5f90(SB)/8, $2302 // bitrev[3058] = 2302
DATA bitrev_size4096_radix4<>+0x5f98(SB)/8, $3326 // bitrev[3059] = 3326
DATA bitrev_size4096_radix4<>+0x5fa0(SB)/8, $510  // bitrev[3060] = 510
DATA bitrev_size4096_radix4<>+0x5fa8(SB)/8, $1534 // bitrev[3061] = 1534
DATA bitrev_size4096_radix4<>+0x5fb0(SB)/8, $2558 // bitrev[3062] = 2558
DATA bitrev_size4096_radix4<>+0x5fb8(SB)/8, $3582 // bitrev[3063] = 3582
DATA bitrev_size4096_radix4<>+0x5fc0(SB)/8, $766  // bitrev[3064] = 766
DATA bitrev_size4096_radix4<>+0x5fc8(SB)/8, $1790 // bitrev[3065] = 1790
DATA bitrev_size4096_radix4<>+0x5fd0(SB)/8, $2814 // bitrev[3066] = 2814
DATA bitrev_size4096_radix4<>+0x5fd8(SB)/8, $3838 // bitrev[3067] = 3838
DATA bitrev_size4096_radix4<>+0x5fe0(SB)/8, $1022 // bitrev[3068] = 1022
DATA bitrev_size4096_radix4<>+0x5fe8(SB)/8, $2046 // bitrev[3069] = 2046
DATA bitrev_size4096_radix4<>+0x5ff0(SB)/8, $3070 // bitrev[3070] = 3070
DATA bitrev_size4096_radix4<>+0x5ff8(SB)/8, $4094 // bitrev[3071] = 4094
DATA bitrev_size4096_radix4<>+0x6000(SB)/8, $3    // bitrev[3072] = 3
DATA bitrev_size4096_radix4<>+0x6008(SB)/8, $1027 // bitrev[3073] = 1027
DATA bitrev_size4096_radix4<>+0x6010(SB)/8, $2051 // bitrev[3074] = 2051
DATA bitrev_size4096_radix4<>+0x6018(SB)/8, $3075 // bitrev[3075] = 3075
DATA bitrev_size4096_radix4<>+0x6020(SB)/8, $259  // bitrev[3076] = 259
DATA bitrev_size4096_radix4<>+0x6028(SB)/8, $1283 // bitrev[3077] = 1283
DATA bitrev_size4096_radix4<>+0x6030(SB)/8, $2307 // bitrev[3078] = 2307
DATA bitrev_size4096_radix4<>+0x6038(SB)/8, $3331 // bitrev[3079] = 3331
DATA bitrev_size4096_radix4<>+0x6040(SB)/8, $515  // bitrev[3080] = 515
DATA bitrev_size4096_radix4<>+0x6048(SB)/8, $1539 // bitrev[3081] = 1539
DATA bitrev_size4096_radix4<>+0x6050(SB)/8, $2563 // bitrev[3082] = 2563
DATA bitrev_size4096_radix4<>+0x6058(SB)/8, $3587 // bitrev[3083] = 3587
DATA bitrev_size4096_radix4<>+0x6060(SB)/8, $771  // bitrev[3084] = 771
DATA bitrev_size4096_radix4<>+0x6068(SB)/8, $1795 // bitrev[3085] = 1795
DATA bitrev_size4096_radix4<>+0x6070(SB)/8, $2819 // bitrev[3086] = 2819
DATA bitrev_size4096_radix4<>+0x6078(SB)/8, $3843 // bitrev[3087] = 3843
DATA bitrev_size4096_radix4<>+0x6080(SB)/8, $67   // bitrev[3088] = 67
DATA bitrev_size4096_radix4<>+0x6088(SB)/8, $1091 // bitrev[3089] = 1091
DATA bitrev_size4096_radix4<>+0x6090(SB)/8, $2115 // bitrev[3090] = 2115
DATA bitrev_size4096_radix4<>+0x6098(SB)/8, $3139 // bitrev[3091] = 3139
DATA bitrev_size4096_radix4<>+0x60a0(SB)/8, $323  // bitrev[3092] = 323
DATA bitrev_size4096_radix4<>+0x60a8(SB)/8, $1347 // bitrev[3093] = 1347
DATA bitrev_size4096_radix4<>+0x60b0(SB)/8, $2371 // bitrev[3094] = 2371
DATA bitrev_size4096_radix4<>+0x60b8(SB)/8, $3395 // bitrev[3095] = 3395
DATA bitrev_size4096_radix4<>+0x60c0(SB)/8, $579  // bitrev[3096] = 579
DATA bitrev_size4096_radix4<>+0x60c8(SB)/8, $1603 // bitrev[3097] = 1603
DATA bitrev_size4096_radix4<>+0x60d0(SB)/8, $2627 // bitrev[3098] = 2627
DATA bitrev_size4096_radix4<>+0x60d8(SB)/8, $3651 // bitrev[3099] = 3651
DATA bitrev_size4096_radix4<>+0x60e0(SB)/8, $835  // bitrev[3100] = 835
DATA bitrev_size4096_radix4<>+0x60e8(SB)/8, $1859 // bitrev[3101] = 1859
DATA bitrev_size4096_radix4<>+0x60f0(SB)/8, $2883 // bitrev[3102] = 2883
DATA bitrev_size4096_radix4<>+0x60f8(SB)/8, $3907 // bitrev[3103] = 3907
DATA bitrev_size4096_radix4<>+0x6100(SB)/8, $131  // bitrev[3104] = 131
DATA bitrev_size4096_radix4<>+0x6108(SB)/8, $1155 // bitrev[3105] = 1155
DATA bitrev_size4096_radix4<>+0x6110(SB)/8, $2179 // bitrev[3106] = 2179
DATA bitrev_size4096_radix4<>+0x6118(SB)/8, $3203 // bitrev[3107] = 3203
DATA bitrev_size4096_radix4<>+0x6120(SB)/8, $387  // bitrev[3108] = 387
DATA bitrev_size4096_radix4<>+0x6128(SB)/8, $1411 // bitrev[3109] = 1411
DATA bitrev_size4096_radix4<>+0x6130(SB)/8, $2435 // bitrev[3110] = 2435
DATA bitrev_size4096_radix4<>+0x6138(SB)/8, $3459 // bitrev[3111] = 3459
DATA bitrev_size4096_radix4<>+0x6140(SB)/8, $643  // bitrev[3112] = 643
DATA bitrev_size4096_radix4<>+0x6148(SB)/8, $1667 // bitrev[3113] = 1667
DATA bitrev_size4096_radix4<>+0x6150(SB)/8, $2691 // bitrev[3114] = 2691
DATA bitrev_size4096_radix4<>+0x6158(SB)/8, $3715 // bitrev[3115] = 3715
DATA bitrev_size4096_radix4<>+0x6160(SB)/8, $899  // bitrev[3116] = 899
DATA bitrev_size4096_radix4<>+0x6168(SB)/8, $1923 // bitrev[3117] = 1923
DATA bitrev_size4096_radix4<>+0x6170(SB)/8, $2947 // bitrev[3118] = 2947
DATA bitrev_size4096_radix4<>+0x6178(SB)/8, $3971 // bitrev[3119] = 3971
DATA bitrev_size4096_radix4<>+0x6180(SB)/8, $195  // bitrev[3120] = 195
DATA bitrev_size4096_radix4<>+0x6188(SB)/8, $1219 // bitrev[3121] = 1219
DATA bitrev_size4096_radix4<>+0x6190(SB)/8, $2243 // bitrev[3122] = 2243
DATA bitrev_size4096_radix4<>+0x6198(SB)/8, $3267 // bitrev[3123] = 3267
DATA bitrev_size4096_radix4<>+0x61a0(SB)/8, $451  // bitrev[3124] = 451
DATA bitrev_size4096_radix4<>+0x61a8(SB)/8, $1475 // bitrev[3125] = 1475
DATA bitrev_size4096_radix4<>+0x61b0(SB)/8, $2499 // bitrev[3126] = 2499
DATA bitrev_size4096_radix4<>+0x61b8(SB)/8, $3523 // bitrev[3127] = 3523
DATA bitrev_size4096_radix4<>+0x61c0(SB)/8, $707  // bitrev[3128] = 707
DATA bitrev_size4096_radix4<>+0x61c8(SB)/8, $1731 // bitrev[3129] = 1731
DATA bitrev_size4096_radix4<>+0x61d0(SB)/8, $2755 // bitrev[3130] = 2755
DATA bitrev_size4096_radix4<>+0x61d8(SB)/8, $3779 // bitrev[3131] = 3779
DATA bitrev_size4096_radix4<>+0x61e0(SB)/8, $963  // bitrev[3132] = 963
DATA bitrev_size4096_radix4<>+0x61e8(SB)/8, $1987 // bitrev[3133] = 1987
DATA bitrev_size4096_radix4<>+0x61f0(SB)/8, $3011 // bitrev[3134] = 3011
DATA bitrev_size4096_radix4<>+0x61f8(SB)/8, $4035 // bitrev[3135] = 4035
DATA bitrev_size4096_radix4<>+0x6200(SB)/8, $19   // bitrev[3136] = 19
DATA bitrev_size4096_radix4<>+0x6208(SB)/8, $1043 // bitrev[3137] = 1043
DATA bitrev_size4096_radix4<>+0x6210(SB)/8, $2067 // bitrev[3138] = 2067
DATA bitrev_size4096_radix4<>+0x6218(SB)/8, $3091 // bitrev[3139] = 3091
DATA bitrev_size4096_radix4<>+0x6220(SB)/8, $275  // bitrev[3140] = 275
DATA bitrev_size4096_radix4<>+0x6228(SB)/8, $1299 // bitrev[3141] = 1299
DATA bitrev_size4096_radix4<>+0x6230(SB)/8, $2323 // bitrev[3142] = 2323
DATA bitrev_size4096_radix4<>+0x6238(SB)/8, $3347 // bitrev[3143] = 3347
DATA bitrev_size4096_radix4<>+0x6240(SB)/8, $531  // bitrev[3144] = 531
DATA bitrev_size4096_radix4<>+0x6248(SB)/8, $1555 // bitrev[3145] = 1555
DATA bitrev_size4096_radix4<>+0x6250(SB)/8, $2579 // bitrev[3146] = 2579
DATA bitrev_size4096_radix4<>+0x6258(SB)/8, $3603 // bitrev[3147] = 3603
DATA bitrev_size4096_radix4<>+0x6260(SB)/8, $787  // bitrev[3148] = 787
DATA bitrev_size4096_radix4<>+0x6268(SB)/8, $1811 // bitrev[3149] = 1811
DATA bitrev_size4096_radix4<>+0x6270(SB)/8, $2835 // bitrev[3150] = 2835
DATA bitrev_size4096_radix4<>+0x6278(SB)/8, $3859 // bitrev[3151] = 3859
DATA bitrev_size4096_radix4<>+0x6280(SB)/8, $83   // bitrev[3152] = 83
DATA bitrev_size4096_radix4<>+0x6288(SB)/8, $1107 // bitrev[3153] = 1107
DATA bitrev_size4096_radix4<>+0x6290(SB)/8, $2131 // bitrev[3154] = 2131
DATA bitrev_size4096_radix4<>+0x6298(SB)/8, $3155 // bitrev[3155] = 3155
DATA bitrev_size4096_radix4<>+0x62a0(SB)/8, $339  // bitrev[3156] = 339
DATA bitrev_size4096_radix4<>+0x62a8(SB)/8, $1363 // bitrev[3157] = 1363
DATA bitrev_size4096_radix4<>+0x62b0(SB)/8, $2387 // bitrev[3158] = 2387
DATA bitrev_size4096_radix4<>+0x62b8(SB)/8, $3411 // bitrev[3159] = 3411
DATA bitrev_size4096_radix4<>+0x62c0(SB)/8, $595  // bitrev[3160] = 595
DATA bitrev_size4096_radix4<>+0x62c8(SB)/8, $1619 // bitrev[3161] = 1619
DATA bitrev_size4096_radix4<>+0x62d0(SB)/8, $2643 // bitrev[3162] = 2643
DATA bitrev_size4096_radix4<>+0x62d8(SB)/8, $3667 // bitrev[3163] = 3667
DATA bitrev_size4096_radix4<>+0x62e0(SB)/8, $851  // bitrev[3164] = 851
DATA bitrev_size4096_radix4<>+0x62e8(SB)/8, $1875 // bitrev[3165] = 1875
DATA bitrev_size4096_radix4<>+0x62f0(SB)/8, $2899 // bitrev[3166] = 2899
DATA bitrev_size4096_radix4<>+0x62f8(SB)/8, $3923 // bitrev[3167] = 3923
DATA bitrev_size4096_radix4<>+0x6300(SB)/8, $147  // bitrev[3168] = 147
DATA bitrev_size4096_radix4<>+0x6308(SB)/8, $1171 // bitrev[3169] = 1171
DATA bitrev_size4096_radix4<>+0x6310(SB)/8, $2195 // bitrev[3170] = 2195
DATA bitrev_size4096_radix4<>+0x6318(SB)/8, $3219 // bitrev[3171] = 3219
DATA bitrev_size4096_radix4<>+0x6320(SB)/8, $403  // bitrev[3172] = 403
DATA bitrev_size4096_radix4<>+0x6328(SB)/8, $1427 // bitrev[3173] = 1427
DATA bitrev_size4096_radix4<>+0x6330(SB)/8, $2451 // bitrev[3174] = 2451
DATA bitrev_size4096_radix4<>+0x6338(SB)/8, $3475 // bitrev[3175] = 3475
DATA bitrev_size4096_radix4<>+0x6340(SB)/8, $659  // bitrev[3176] = 659
DATA bitrev_size4096_radix4<>+0x6348(SB)/8, $1683 // bitrev[3177] = 1683
DATA bitrev_size4096_radix4<>+0x6350(SB)/8, $2707 // bitrev[3178] = 2707
DATA bitrev_size4096_radix4<>+0x6358(SB)/8, $3731 // bitrev[3179] = 3731
DATA bitrev_size4096_radix4<>+0x6360(SB)/8, $915  // bitrev[3180] = 915
DATA bitrev_size4096_radix4<>+0x6368(SB)/8, $1939 // bitrev[3181] = 1939
DATA bitrev_size4096_radix4<>+0x6370(SB)/8, $2963 // bitrev[3182] = 2963
DATA bitrev_size4096_radix4<>+0x6378(SB)/8, $3987 // bitrev[3183] = 3987
DATA bitrev_size4096_radix4<>+0x6380(SB)/8, $211  // bitrev[3184] = 211
DATA bitrev_size4096_radix4<>+0x6388(SB)/8, $1235 // bitrev[3185] = 1235
DATA bitrev_size4096_radix4<>+0x6390(SB)/8, $2259 // bitrev[3186] = 2259
DATA bitrev_size4096_radix4<>+0x6398(SB)/8, $3283 // bitrev[3187] = 3283
DATA bitrev_size4096_radix4<>+0x63a0(SB)/8, $467  // bitrev[3188] = 467
DATA bitrev_size4096_radix4<>+0x63a8(SB)/8, $1491 // bitrev[3189] = 1491
DATA bitrev_size4096_radix4<>+0x63b0(SB)/8, $2515 // bitrev[3190] = 2515
DATA bitrev_size4096_radix4<>+0x63b8(SB)/8, $3539 // bitrev[3191] = 3539
DATA bitrev_size4096_radix4<>+0x63c0(SB)/8, $723  // bitrev[3192] = 723
DATA bitrev_size4096_radix4<>+0x63c8(SB)/8, $1747 // bitrev[3193] = 1747
DATA bitrev_size4096_radix4<>+0x63d0(SB)/8, $2771 // bitrev[3194] = 2771
DATA bitrev_size4096_radix4<>+0x63d8(SB)/8, $3795 // bitrev[3195] = 3795
DATA bitrev_size4096_radix4<>+0x63e0(SB)/8, $979  // bitrev[3196] = 979
DATA bitrev_size4096_radix4<>+0x63e8(SB)/8, $2003 // bitrev[3197] = 2003
DATA bitrev_size4096_radix4<>+0x63f0(SB)/8, $3027 // bitrev[3198] = 3027
DATA bitrev_size4096_radix4<>+0x63f8(SB)/8, $4051 // bitrev[3199] = 4051
DATA bitrev_size4096_radix4<>+0x6400(SB)/8, $35   // bitrev[3200] = 35
DATA bitrev_size4096_radix4<>+0x6408(SB)/8, $1059 // bitrev[3201] = 1059
DATA bitrev_size4096_radix4<>+0x6410(SB)/8, $2083 // bitrev[3202] = 2083
DATA bitrev_size4096_radix4<>+0x6418(SB)/8, $3107 // bitrev[3203] = 3107
DATA bitrev_size4096_radix4<>+0x6420(SB)/8, $291  // bitrev[3204] = 291
DATA bitrev_size4096_radix4<>+0x6428(SB)/8, $1315 // bitrev[3205] = 1315
DATA bitrev_size4096_radix4<>+0x6430(SB)/8, $2339 // bitrev[3206] = 2339
DATA bitrev_size4096_radix4<>+0x6438(SB)/8, $3363 // bitrev[3207] = 3363
DATA bitrev_size4096_radix4<>+0x6440(SB)/8, $547  // bitrev[3208] = 547
DATA bitrev_size4096_radix4<>+0x6448(SB)/8, $1571 // bitrev[3209] = 1571
DATA bitrev_size4096_radix4<>+0x6450(SB)/8, $2595 // bitrev[3210] = 2595
DATA bitrev_size4096_radix4<>+0x6458(SB)/8, $3619 // bitrev[3211] = 3619
DATA bitrev_size4096_radix4<>+0x6460(SB)/8, $803  // bitrev[3212] = 803
DATA bitrev_size4096_radix4<>+0x6468(SB)/8, $1827 // bitrev[3213] = 1827
DATA bitrev_size4096_radix4<>+0x6470(SB)/8, $2851 // bitrev[3214] = 2851
DATA bitrev_size4096_radix4<>+0x6478(SB)/8, $3875 // bitrev[3215] = 3875
DATA bitrev_size4096_radix4<>+0x6480(SB)/8, $99   // bitrev[3216] = 99
DATA bitrev_size4096_radix4<>+0x6488(SB)/8, $1123 // bitrev[3217] = 1123
DATA bitrev_size4096_radix4<>+0x6490(SB)/8, $2147 // bitrev[3218] = 2147
DATA bitrev_size4096_radix4<>+0x6498(SB)/8, $3171 // bitrev[3219] = 3171
DATA bitrev_size4096_radix4<>+0x64a0(SB)/8, $355  // bitrev[3220] = 355
DATA bitrev_size4096_radix4<>+0x64a8(SB)/8, $1379 // bitrev[3221] = 1379
DATA bitrev_size4096_radix4<>+0x64b0(SB)/8, $2403 // bitrev[3222] = 2403
DATA bitrev_size4096_radix4<>+0x64b8(SB)/8, $3427 // bitrev[3223] = 3427
DATA bitrev_size4096_radix4<>+0x64c0(SB)/8, $611  // bitrev[3224] = 611
DATA bitrev_size4096_radix4<>+0x64c8(SB)/8, $1635 // bitrev[3225] = 1635
DATA bitrev_size4096_radix4<>+0x64d0(SB)/8, $2659 // bitrev[3226] = 2659
DATA bitrev_size4096_radix4<>+0x64d8(SB)/8, $3683 // bitrev[3227] = 3683
DATA bitrev_size4096_radix4<>+0x64e0(SB)/8, $867  // bitrev[3228] = 867
DATA bitrev_size4096_radix4<>+0x64e8(SB)/8, $1891 // bitrev[3229] = 1891
DATA bitrev_size4096_radix4<>+0x64f0(SB)/8, $2915 // bitrev[3230] = 2915
DATA bitrev_size4096_radix4<>+0x64f8(SB)/8, $3939 // bitrev[3231] = 3939
DATA bitrev_size4096_radix4<>+0x6500(SB)/8, $163  // bitrev[3232] = 163
DATA bitrev_size4096_radix4<>+0x6508(SB)/8, $1187 // bitrev[3233] = 1187
DATA bitrev_size4096_radix4<>+0x6510(SB)/8, $2211 // bitrev[3234] = 2211
DATA bitrev_size4096_radix4<>+0x6518(SB)/8, $3235 // bitrev[3235] = 3235
DATA bitrev_size4096_radix4<>+0x6520(SB)/8, $419  // bitrev[3236] = 419
DATA bitrev_size4096_radix4<>+0x6528(SB)/8, $1443 // bitrev[3237] = 1443
DATA bitrev_size4096_radix4<>+0x6530(SB)/8, $2467 // bitrev[3238] = 2467
DATA bitrev_size4096_radix4<>+0x6538(SB)/8, $3491 // bitrev[3239] = 3491
DATA bitrev_size4096_radix4<>+0x6540(SB)/8, $675  // bitrev[3240] = 675
DATA bitrev_size4096_radix4<>+0x6548(SB)/8, $1699 // bitrev[3241] = 1699
DATA bitrev_size4096_radix4<>+0x6550(SB)/8, $2723 // bitrev[3242] = 2723
DATA bitrev_size4096_radix4<>+0x6558(SB)/8, $3747 // bitrev[3243] = 3747
DATA bitrev_size4096_radix4<>+0x6560(SB)/8, $931  // bitrev[3244] = 931
DATA bitrev_size4096_radix4<>+0x6568(SB)/8, $1955 // bitrev[3245] = 1955
DATA bitrev_size4096_radix4<>+0x6570(SB)/8, $2979 // bitrev[3246] = 2979
DATA bitrev_size4096_radix4<>+0x6578(SB)/8, $4003 // bitrev[3247] = 4003
DATA bitrev_size4096_radix4<>+0x6580(SB)/8, $227  // bitrev[3248] = 227
DATA bitrev_size4096_radix4<>+0x6588(SB)/8, $1251 // bitrev[3249] = 1251
DATA bitrev_size4096_radix4<>+0x6590(SB)/8, $2275 // bitrev[3250] = 2275
DATA bitrev_size4096_radix4<>+0x6598(SB)/8, $3299 // bitrev[3251] = 3299
DATA bitrev_size4096_radix4<>+0x65a0(SB)/8, $483  // bitrev[3252] = 483
DATA bitrev_size4096_radix4<>+0x65a8(SB)/8, $1507 // bitrev[3253] = 1507
DATA bitrev_size4096_radix4<>+0x65b0(SB)/8, $2531 // bitrev[3254] = 2531
DATA bitrev_size4096_radix4<>+0x65b8(SB)/8, $3555 // bitrev[3255] = 3555
DATA bitrev_size4096_radix4<>+0x65c0(SB)/8, $739  // bitrev[3256] = 739
DATA bitrev_size4096_radix4<>+0x65c8(SB)/8, $1763 // bitrev[3257] = 1763
DATA bitrev_size4096_radix4<>+0x65d0(SB)/8, $2787 // bitrev[3258] = 2787
DATA bitrev_size4096_radix4<>+0x65d8(SB)/8, $3811 // bitrev[3259] = 3811
DATA bitrev_size4096_radix4<>+0x65e0(SB)/8, $995  // bitrev[3260] = 995
DATA bitrev_size4096_radix4<>+0x65e8(SB)/8, $2019 // bitrev[3261] = 2019
DATA bitrev_size4096_radix4<>+0x65f0(SB)/8, $3043 // bitrev[3262] = 3043
DATA bitrev_size4096_radix4<>+0x65f8(SB)/8, $4067 // bitrev[3263] = 4067
DATA bitrev_size4096_radix4<>+0x6600(SB)/8, $51   // bitrev[3264] = 51
DATA bitrev_size4096_radix4<>+0x6608(SB)/8, $1075 // bitrev[3265] = 1075
DATA bitrev_size4096_radix4<>+0x6610(SB)/8, $2099 // bitrev[3266] = 2099
DATA bitrev_size4096_radix4<>+0x6618(SB)/8, $3123 // bitrev[3267] = 3123
DATA bitrev_size4096_radix4<>+0x6620(SB)/8, $307  // bitrev[3268] = 307
DATA bitrev_size4096_radix4<>+0x6628(SB)/8, $1331 // bitrev[3269] = 1331
DATA bitrev_size4096_radix4<>+0x6630(SB)/8, $2355 // bitrev[3270] = 2355
DATA bitrev_size4096_radix4<>+0x6638(SB)/8, $3379 // bitrev[3271] = 3379
DATA bitrev_size4096_radix4<>+0x6640(SB)/8, $563  // bitrev[3272] = 563
DATA bitrev_size4096_radix4<>+0x6648(SB)/8, $1587 // bitrev[3273] = 1587
DATA bitrev_size4096_radix4<>+0x6650(SB)/8, $2611 // bitrev[3274] = 2611
DATA bitrev_size4096_radix4<>+0x6658(SB)/8, $3635 // bitrev[3275] = 3635
DATA bitrev_size4096_radix4<>+0x6660(SB)/8, $819  // bitrev[3276] = 819
DATA bitrev_size4096_radix4<>+0x6668(SB)/8, $1843 // bitrev[3277] = 1843
DATA bitrev_size4096_radix4<>+0x6670(SB)/8, $2867 // bitrev[3278] = 2867
DATA bitrev_size4096_radix4<>+0x6678(SB)/8, $3891 // bitrev[3279] = 3891
DATA bitrev_size4096_radix4<>+0x6680(SB)/8, $115  // bitrev[3280] = 115
DATA bitrev_size4096_radix4<>+0x6688(SB)/8, $1139 // bitrev[3281] = 1139
DATA bitrev_size4096_radix4<>+0x6690(SB)/8, $2163 // bitrev[3282] = 2163
DATA bitrev_size4096_radix4<>+0x6698(SB)/8, $3187 // bitrev[3283] = 3187
DATA bitrev_size4096_radix4<>+0x66a0(SB)/8, $371  // bitrev[3284] = 371
DATA bitrev_size4096_radix4<>+0x66a8(SB)/8, $1395 // bitrev[3285] = 1395
DATA bitrev_size4096_radix4<>+0x66b0(SB)/8, $2419 // bitrev[3286] = 2419
DATA bitrev_size4096_radix4<>+0x66b8(SB)/8, $3443 // bitrev[3287] = 3443
DATA bitrev_size4096_radix4<>+0x66c0(SB)/8, $627  // bitrev[3288] = 627
DATA bitrev_size4096_radix4<>+0x66c8(SB)/8, $1651 // bitrev[3289] = 1651
DATA bitrev_size4096_radix4<>+0x66d0(SB)/8, $2675 // bitrev[3290] = 2675
DATA bitrev_size4096_radix4<>+0x66d8(SB)/8, $3699 // bitrev[3291] = 3699
DATA bitrev_size4096_radix4<>+0x66e0(SB)/8, $883  // bitrev[3292] = 883
DATA bitrev_size4096_radix4<>+0x66e8(SB)/8, $1907 // bitrev[3293] = 1907
DATA bitrev_size4096_radix4<>+0x66f0(SB)/8, $2931 // bitrev[3294] = 2931
DATA bitrev_size4096_radix4<>+0x66f8(SB)/8, $3955 // bitrev[3295] = 3955
DATA bitrev_size4096_radix4<>+0x6700(SB)/8, $179  // bitrev[3296] = 179
DATA bitrev_size4096_radix4<>+0x6708(SB)/8, $1203 // bitrev[3297] = 1203
DATA bitrev_size4096_radix4<>+0x6710(SB)/8, $2227 // bitrev[3298] = 2227
DATA bitrev_size4096_radix4<>+0x6718(SB)/8, $3251 // bitrev[3299] = 3251
DATA bitrev_size4096_radix4<>+0x6720(SB)/8, $435  // bitrev[3300] = 435
DATA bitrev_size4096_radix4<>+0x6728(SB)/8, $1459 // bitrev[3301] = 1459
DATA bitrev_size4096_radix4<>+0x6730(SB)/8, $2483 // bitrev[3302] = 2483
DATA bitrev_size4096_radix4<>+0x6738(SB)/8, $3507 // bitrev[3303] = 3507
DATA bitrev_size4096_radix4<>+0x6740(SB)/8, $691  // bitrev[3304] = 691
DATA bitrev_size4096_radix4<>+0x6748(SB)/8, $1715 // bitrev[3305] = 1715
DATA bitrev_size4096_radix4<>+0x6750(SB)/8, $2739 // bitrev[3306] = 2739
DATA bitrev_size4096_radix4<>+0x6758(SB)/8, $3763 // bitrev[3307] = 3763
DATA bitrev_size4096_radix4<>+0x6760(SB)/8, $947  // bitrev[3308] = 947
DATA bitrev_size4096_radix4<>+0x6768(SB)/8, $1971 // bitrev[3309] = 1971
DATA bitrev_size4096_radix4<>+0x6770(SB)/8, $2995 // bitrev[3310] = 2995
DATA bitrev_size4096_radix4<>+0x6778(SB)/8, $4019 // bitrev[3311] = 4019
DATA bitrev_size4096_radix4<>+0x6780(SB)/8, $243  // bitrev[3312] = 243
DATA bitrev_size4096_radix4<>+0x6788(SB)/8, $1267 // bitrev[3313] = 1267
DATA bitrev_size4096_radix4<>+0x6790(SB)/8, $2291 // bitrev[3314] = 2291
DATA bitrev_size4096_radix4<>+0x6798(SB)/8, $3315 // bitrev[3315] = 3315
DATA bitrev_size4096_radix4<>+0x67a0(SB)/8, $499  // bitrev[3316] = 499
DATA bitrev_size4096_radix4<>+0x67a8(SB)/8, $1523 // bitrev[3317] = 1523
DATA bitrev_size4096_radix4<>+0x67b0(SB)/8, $2547 // bitrev[3318] = 2547
DATA bitrev_size4096_radix4<>+0x67b8(SB)/8, $3571 // bitrev[3319] = 3571
DATA bitrev_size4096_radix4<>+0x67c0(SB)/8, $755  // bitrev[3320] = 755
DATA bitrev_size4096_radix4<>+0x67c8(SB)/8, $1779 // bitrev[3321] = 1779
DATA bitrev_size4096_radix4<>+0x67d0(SB)/8, $2803 // bitrev[3322] = 2803
DATA bitrev_size4096_radix4<>+0x67d8(SB)/8, $3827 // bitrev[3323] = 3827
DATA bitrev_size4096_radix4<>+0x67e0(SB)/8, $1011 // bitrev[3324] = 1011
DATA bitrev_size4096_radix4<>+0x67e8(SB)/8, $2035 // bitrev[3325] = 2035
DATA bitrev_size4096_radix4<>+0x67f0(SB)/8, $3059 // bitrev[3326] = 3059
DATA bitrev_size4096_radix4<>+0x67f8(SB)/8, $4083 // bitrev[3327] = 4083
DATA bitrev_size4096_radix4<>+0x6800(SB)/8, $7    // bitrev[3328] = 7
DATA bitrev_size4096_radix4<>+0x6808(SB)/8, $1031 // bitrev[3329] = 1031
DATA bitrev_size4096_radix4<>+0x6810(SB)/8, $2055 // bitrev[3330] = 2055
DATA bitrev_size4096_radix4<>+0x6818(SB)/8, $3079 // bitrev[3331] = 3079
DATA bitrev_size4096_radix4<>+0x6820(SB)/8, $263  // bitrev[3332] = 263
DATA bitrev_size4096_radix4<>+0x6828(SB)/8, $1287 // bitrev[3333] = 1287
DATA bitrev_size4096_radix4<>+0x6830(SB)/8, $2311 // bitrev[3334] = 2311
DATA bitrev_size4096_radix4<>+0x6838(SB)/8, $3335 // bitrev[3335] = 3335
DATA bitrev_size4096_radix4<>+0x6840(SB)/8, $519  // bitrev[3336] = 519
DATA bitrev_size4096_radix4<>+0x6848(SB)/8, $1543 // bitrev[3337] = 1543
DATA bitrev_size4096_radix4<>+0x6850(SB)/8, $2567 // bitrev[3338] = 2567
DATA bitrev_size4096_radix4<>+0x6858(SB)/8, $3591 // bitrev[3339] = 3591
DATA bitrev_size4096_radix4<>+0x6860(SB)/8, $775  // bitrev[3340] = 775
DATA bitrev_size4096_radix4<>+0x6868(SB)/8, $1799 // bitrev[3341] = 1799
DATA bitrev_size4096_radix4<>+0x6870(SB)/8, $2823 // bitrev[3342] = 2823
DATA bitrev_size4096_radix4<>+0x6878(SB)/8, $3847 // bitrev[3343] = 3847
DATA bitrev_size4096_radix4<>+0x6880(SB)/8, $71   // bitrev[3344] = 71
DATA bitrev_size4096_radix4<>+0x6888(SB)/8, $1095 // bitrev[3345] = 1095
DATA bitrev_size4096_radix4<>+0x6890(SB)/8, $2119 // bitrev[3346] = 2119
DATA bitrev_size4096_radix4<>+0x6898(SB)/8, $3143 // bitrev[3347] = 3143
DATA bitrev_size4096_radix4<>+0x68a0(SB)/8, $327  // bitrev[3348] = 327
DATA bitrev_size4096_radix4<>+0x68a8(SB)/8, $1351 // bitrev[3349] = 1351
DATA bitrev_size4096_radix4<>+0x68b0(SB)/8, $2375 // bitrev[3350] = 2375
DATA bitrev_size4096_radix4<>+0x68b8(SB)/8, $3399 // bitrev[3351] = 3399
DATA bitrev_size4096_radix4<>+0x68c0(SB)/8, $583  // bitrev[3352] = 583
DATA bitrev_size4096_radix4<>+0x68c8(SB)/8, $1607 // bitrev[3353] = 1607
DATA bitrev_size4096_radix4<>+0x68d0(SB)/8, $2631 // bitrev[3354] = 2631
DATA bitrev_size4096_radix4<>+0x68d8(SB)/8, $3655 // bitrev[3355] = 3655
DATA bitrev_size4096_radix4<>+0x68e0(SB)/8, $839  // bitrev[3356] = 839
DATA bitrev_size4096_radix4<>+0x68e8(SB)/8, $1863 // bitrev[3357] = 1863
DATA bitrev_size4096_radix4<>+0x68f0(SB)/8, $2887 // bitrev[3358] = 2887
DATA bitrev_size4096_radix4<>+0x68f8(SB)/8, $3911 // bitrev[3359] = 3911
DATA bitrev_size4096_radix4<>+0x6900(SB)/8, $135  // bitrev[3360] = 135
DATA bitrev_size4096_radix4<>+0x6908(SB)/8, $1159 // bitrev[3361] = 1159
DATA bitrev_size4096_radix4<>+0x6910(SB)/8, $2183 // bitrev[3362] = 2183
DATA bitrev_size4096_radix4<>+0x6918(SB)/8, $3207 // bitrev[3363] = 3207
DATA bitrev_size4096_radix4<>+0x6920(SB)/8, $391  // bitrev[3364] = 391
DATA bitrev_size4096_radix4<>+0x6928(SB)/8, $1415 // bitrev[3365] = 1415
DATA bitrev_size4096_radix4<>+0x6930(SB)/8, $2439 // bitrev[3366] = 2439
DATA bitrev_size4096_radix4<>+0x6938(SB)/8, $3463 // bitrev[3367] = 3463
DATA bitrev_size4096_radix4<>+0x6940(SB)/8, $647  // bitrev[3368] = 647
DATA bitrev_size4096_radix4<>+0x6948(SB)/8, $1671 // bitrev[3369] = 1671
DATA bitrev_size4096_radix4<>+0x6950(SB)/8, $2695 // bitrev[3370] = 2695
DATA bitrev_size4096_radix4<>+0x6958(SB)/8, $3719 // bitrev[3371] = 3719
DATA bitrev_size4096_radix4<>+0x6960(SB)/8, $903  // bitrev[3372] = 903
DATA bitrev_size4096_radix4<>+0x6968(SB)/8, $1927 // bitrev[3373] = 1927
DATA bitrev_size4096_radix4<>+0x6970(SB)/8, $2951 // bitrev[3374] = 2951
DATA bitrev_size4096_radix4<>+0x6978(SB)/8, $3975 // bitrev[3375] = 3975
DATA bitrev_size4096_radix4<>+0x6980(SB)/8, $199  // bitrev[3376] = 199
DATA bitrev_size4096_radix4<>+0x6988(SB)/8, $1223 // bitrev[3377] = 1223
DATA bitrev_size4096_radix4<>+0x6990(SB)/8, $2247 // bitrev[3378] = 2247
DATA bitrev_size4096_radix4<>+0x6998(SB)/8, $3271 // bitrev[3379] = 3271
DATA bitrev_size4096_radix4<>+0x69a0(SB)/8, $455  // bitrev[3380] = 455
DATA bitrev_size4096_radix4<>+0x69a8(SB)/8, $1479 // bitrev[3381] = 1479
DATA bitrev_size4096_radix4<>+0x69b0(SB)/8, $2503 // bitrev[3382] = 2503
DATA bitrev_size4096_radix4<>+0x69b8(SB)/8, $3527 // bitrev[3383] = 3527
DATA bitrev_size4096_radix4<>+0x69c0(SB)/8, $711  // bitrev[3384] = 711
DATA bitrev_size4096_radix4<>+0x69c8(SB)/8, $1735 // bitrev[3385] = 1735
DATA bitrev_size4096_radix4<>+0x69d0(SB)/8, $2759 // bitrev[3386] = 2759
DATA bitrev_size4096_radix4<>+0x69d8(SB)/8, $3783 // bitrev[3387] = 3783
DATA bitrev_size4096_radix4<>+0x69e0(SB)/8, $967  // bitrev[3388] = 967
DATA bitrev_size4096_radix4<>+0x69e8(SB)/8, $1991 // bitrev[3389] = 1991
DATA bitrev_size4096_radix4<>+0x69f0(SB)/8, $3015 // bitrev[3390] = 3015
DATA bitrev_size4096_radix4<>+0x69f8(SB)/8, $4039 // bitrev[3391] = 4039
DATA bitrev_size4096_radix4<>+0x6a00(SB)/8, $23   // bitrev[3392] = 23
DATA bitrev_size4096_radix4<>+0x6a08(SB)/8, $1047 // bitrev[3393] = 1047
DATA bitrev_size4096_radix4<>+0x6a10(SB)/8, $2071 // bitrev[3394] = 2071
DATA bitrev_size4096_radix4<>+0x6a18(SB)/8, $3095 // bitrev[3395] = 3095
DATA bitrev_size4096_radix4<>+0x6a20(SB)/8, $279  // bitrev[3396] = 279
DATA bitrev_size4096_radix4<>+0x6a28(SB)/8, $1303 // bitrev[3397] = 1303
DATA bitrev_size4096_radix4<>+0x6a30(SB)/8, $2327 // bitrev[3398] = 2327
DATA bitrev_size4096_radix4<>+0x6a38(SB)/8, $3351 // bitrev[3399] = 3351
DATA bitrev_size4096_radix4<>+0x6a40(SB)/8, $535  // bitrev[3400] = 535
DATA bitrev_size4096_radix4<>+0x6a48(SB)/8, $1559 // bitrev[3401] = 1559
DATA bitrev_size4096_radix4<>+0x6a50(SB)/8, $2583 // bitrev[3402] = 2583
DATA bitrev_size4096_radix4<>+0x6a58(SB)/8, $3607 // bitrev[3403] = 3607
DATA bitrev_size4096_radix4<>+0x6a60(SB)/8, $791  // bitrev[3404] = 791
DATA bitrev_size4096_radix4<>+0x6a68(SB)/8, $1815 // bitrev[3405] = 1815
DATA bitrev_size4096_radix4<>+0x6a70(SB)/8, $2839 // bitrev[3406] = 2839
DATA bitrev_size4096_radix4<>+0x6a78(SB)/8, $3863 // bitrev[3407] = 3863
DATA bitrev_size4096_radix4<>+0x6a80(SB)/8, $87   // bitrev[3408] = 87
DATA bitrev_size4096_radix4<>+0x6a88(SB)/8, $1111 // bitrev[3409] = 1111
DATA bitrev_size4096_radix4<>+0x6a90(SB)/8, $2135 // bitrev[3410] = 2135
DATA bitrev_size4096_radix4<>+0x6a98(SB)/8, $3159 // bitrev[3411] = 3159
DATA bitrev_size4096_radix4<>+0x6aa0(SB)/8, $343  // bitrev[3412] = 343
DATA bitrev_size4096_radix4<>+0x6aa8(SB)/8, $1367 // bitrev[3413] = 1367
DATA bitrev_size4096_radix4<>+0x6ab0(SB)/8, $2391 // bitrev[3414] = 2391
DATA bitrev_size4096_radix4<>+0x6ab8(SB)/8, $3415 // bitrev[3415] = 3415
DATA bitrev_size4096_radix4<>+0x6ac0(SB)/8, $599  // bitrev[3416] = 599
DATA bitrev_size4096_radix4<>+0x6ac8(SB)/8, $1623 // bitrev[3417] = 1623
DATA bitrev_size4096_radix4<>+0x6ad0(SB)/8, $2647 // bitrev[3418] = 2647
DATA bitrev_size4096_radix4<>+0x6ad8(SB)/8, $3671 // bitrev[3419] = 3671
DATA bitrev_size4096_radix4<>+0x6ae0(SB)/8, $855  // bitrev[3420] = 855
DATA bitrev_size4096_radix4<>+0x6ae8(SB)/8, $1879 // bitrev[3421] = 1879
DATA bitrev_size4096_radix4<>+0x6af0(SB)/8, $2903 // bitrev[3422] = 2903
DATA bitrev_size4096_radix4<>+0x6af8(SB)/8, $3927 // bitrev[3423] = 3927
DATA bitrev_size4096_radix4<>+0x6b00(SB)/8, $151  // bitrev[3424] = 151
DATA bitrev_size4096_radix4<>+0x6b08(SB)/8, $1175 // bitrev[3425] = 1175
DATA bitrev_size4096_radix4<>+0x6b10(SB)/8, $2199 // bitrev[3426] = 2199
DATA bitrev_size4096_radix4<>+0x6b18(SB)/8, $3223 // bitrev[3427] = 3223
DATA bitrev_size4096_radix4<>+0x6b20(SB)/8, $407  // bitrev[3428] = 407
DATA bitrev_size4096_radix4<>+0x6b28(SB)/8, $1431 // bitrev[3429] = 1431
DATA bitrev_size4096_radix4<>+0x6b30(SB)/8, $2455 // bitrev[3430] = 2455
DATA bitrev_size4096_radix4<>+0x6b38(SB)/8, $3479 // bitrev[3431] = 3479
DATA bitrev_size4096_radix4<>+0x6b40(SB)/8, $663  // bitrev[3432] = 663
DATA bitrev_size4096_radix4<>+0x6b48(SB)/8, $1687 // bitrev[3433] = 1687
DATA bitrev_size4096_radix4<>+0x6b50(SB)/8, $2711 // bitrev[3434] = 2711
DATA bitrev_size4096_radix4<>+0x6b58(SB)/8, $3735 // bitrev[3435] = 3735
DATA bitrev_size4096_radix4<>+0x6b60(SB)/8, $919  // bitrev[3436] = 919
DATA bitrev_size4096_radix4<>+0x6b68(SB)/8, $1943 // bitrev[3437] = 1943
DATA bitrev_size4096_radix4<>+0x6b70(SB)/8, $2967 // bitrev[3438] = 2967
DATA bitrev_size4096_radix4<>+0x6b78(SB)/8, $3991 // bitrev[3439] = 3991
DATA bitrev_size4096_radix4<>+0x6b80(SB)/8, $215  // bitrev[3440] = 215
DATA bitrev_size4096_radix4<>+0x6b88(SB)/8, $1239 // bitrev[3441] = 1239
DATA bitrev_size4096_radix4<>+0x6b90(SB)/8, $2263 // bitrev[3442] = 2263
DATA bitrev_size4096_radix4<>+0x6b98(SB)/8, $3287 // bitrev[3443] = 3287
DATA bitrev_size4096_radix4<>+0x6ba0(SB)/8, $471  // bitrev[3444] = 471
DATA bitrev_size4096_radix4<>+0x6ba8(SB)/8, $1495 // bitrev[3445] = 1495
DATA bitrev_size4096_radix4<>+0x6bb0(SB)/8, $2519 // bitrev[3446] = 2519
DATA bitrev_size4096_radix4<>+0x6bb8(SB)/8, $3543 // bitrev[3447] = 3543
DATA bitrev_size4096_radix4<>+0x6bc0(SB)/8, $727  // bitrev[3448] = 727
DATA bitrev_size4096_radix4<>+0x6bc8(SB)/8, $1751 // bitrev[3449] = 1751
DATA bitrev_size4096_radix4<>+0x6bd0(SB)/8, $2775 // bitrev[3450] = 2775
DATA bitrev_size4096_radix4<>+0x6bd8(SB)/8, $3799 // bitrev[3451] = 3799
DATA bitrev_size4096_radix4<>+0x6be0(SB)/8, $983  // bitrev[3452] = 983
DATA bitrev_size4096_radix4<>+0x6be8(SB)/8, $2007 // bitrev[3453] = 2007
DATA bitrev_size4096_radix4<>+0x6bf0(SB)/8, $3031 // bitrev[3454] = 3031
DATA bitrev_size4096_radix4<>+0x6bf8(SB)/8, $4055 // bitrev[3455] = 4055
DATA bitrev_size4096_radix4<>+0x6c00(SB)/8, $39   // bitrev[3456] = 39
DATA bitrev_size4096_radix4<>+0x6c08(SB)/8, $1063 // bitrev[3457] = 1063
DATA bitrev_size4096_radix4<>+0x6c10(SB)/8, $2087 // bitrev[3458] = 2087
DATA bitrev_size4096_radix4<>+0x6c18(SB)/8, $3111 // bitrev[3459] = 3111
DATA bitrev_size4096_radix4<>+0x6c20(SB)/8, $295  // bitrev[3460] = 295
DATA bitrev_size4096_radix4<>+0x6c28(SB)/8, $1319 // bitrev[3461] = 1319
DATA bitrev_size4096_radix4<>+0x6c30(SB)/8, $2343 // bitrev[3462] = 2343
DATA bitrev_size4096_radix4<>+0x6c38(SB)/8, $3367 // bitrev[3463] = 3367
DATA bitrev_size4096_radix4<>+0x6c40(SB)/8, $551  // bitrev[3464] = 551
DATA bitrev_size4096_radix4<>+0x6c48(SB)/8, $1575 // bitrev[3465] = 1575
DATA bitrev_size4096_radix4<>+0x6c50(SB)/8, $2599 // bitrev[3466] = 2599
DATA bitrev_size4096_radix4<>+0x6c58(SB)/8, $3623 // bitrev[3467] = 3623
DATA bitrev_size4096_radix4<>+0x6c60(SB)/8, $807  // bitrev[3468] = 807
DATA bitrev_size4096_radix4<>+0x6c68(SB)/8, $1831 // bitrev[3469] = 1831
DATA bitrev_size4096_radix4<>+0x6c70(SB)/8, $2855 // bitrev[3470] = 2855
DATA bitrev_size4096_radix4<>+0x6c78(SB)/8, $3879 // bitrev[3471] = 3879
DATA bitrev_size4096_radix4<>+0x6c80(SB)/8, $103  // bitrev[3472] = 103
DATA bitrev_size4096_radix4<>+0x6c88(SB)/8, $1127 // bitrev[3473] = 1127
DATA bitrev_size4096_radix4<>+0x6c90(SB)/8, $2151 // bitrev[3474] = 2151
DATA bitrev_size4096_radix4<>+0x6c98(SB)/8, $3175 // bitrev[3475] = 3175
DATA bitrev_size4096_radix4<>+0x6ca0(SB)/8, $359  // bitrev[3476] = 359
DATA bitrev_size4096_radix4<>+0x6ca8(SB)/8, $1383 // bitrev[3477] = 1383
DATA bitrev_size4096_radix4<>+0x6cb0(SB)/8, $2407 // bitrev[3478] = 2407
DATA bitrev_size4096_radix4<>+0x6cb8(SB)/8, $3431 // bitrev[3479] = 3431
DATA bitrev_size4096_radix4<>+0x6cc0(SB)/8, $615  // bitrev[3480] = 615
DATA bitrev_size4096_radix4<>+0x6cc8(SB)/8, $1639 // bitrev[3481] = 1639
DATA bitrev_size4096_radix4<>+0x6cd0(SB)/8, $2663 // bitrev[3482] = 2663
DATA bitrev_size4096_radix4<>+0x6cd8(SB)/8, $3687 // bitrev[3483] = 3687
DATA bitrev_size4096_radix4<>+0x6ce0(SB)/8, $871  // bitrev[3484] = 871
DATA bitrev_size4096_radix4<>+0x6ce8(SB)/8, $1895 // bitrev[3485] = 1895
DATA bitrev_size4096_radix4<>+0x6cf0(SB)/8, $2919 // bitrev[3486] = 2919
DATA bitrev_size4096_radix4<>+0x6cf8(SB)/8, $3943 // bitrev[3487] = 3943
DATA bitrev_size4096_radix4<>+0x6d00(SB)/8, $167  // bitrev[3488] = 167
DATA bitrev_size4096_radix4<>+0x6d08(SB)/8, $1191 // bitrev[3489] = 1191
DATA bitrev_size4096_radix4<>+0x6d10(SB)/8, $2215 // bitrev[3490] = 2215
DATA bitrev_size4096_radix4<>+0x6d18(SB)/8, $3239 // bitrev[3491] = 3239
DATA bitrev_size4096_radix4<>+0x6d20(SB)/8, $423  // bitrev[3492] = 423
DATA bitrev_size4096_radix4<>+0x6d28(SB)/8, $1447 // bitrev[3493] = 1447
DATA bitrev_size4096_radix4<>+0x6d30(SB)/8, $2471 // bitrev[3494] = 2471
DATA bitrev_size4096_radix4<>+0x6d38(SB)/8, $3495 // bitrev[3495] = 3495
DATA bitrev_size4096_radix4<>+0x6d40(SB)/8, $679  // bitrev[3496] = 679
DATA bitrev_size4096_radix4<>+0x6d48(SB)/8, $1703 // bitrev[3497] = 1703
DATA bitrev_size4096_radix4<>+0x6d50(SB)/8, $2727 // bitrev[3498] = 2727
DATA bitrev_size4096_radix4<>+0x6d58(SB)/8, $3751 // bitrev[3499] = 3751
DATA bitrev_size4096_radix4<>+0x6d60(SB)/8, $935  // bitrev[3500] = 935
DATA bitrev_size4096_radix4<>+0x6d68(SB)/8, $1959 // bitrev[3501] = 1959
DATA bitrev_size4096_radix4<>+0x6d70(SB)/8, $2983 // bitrev[3502] = 2983
DATA bitrev_size4096_radix4<>+0x6d78(SB)/8, $4007 // bitrev[3503] = 4007
DATA bitrev_size4096_radix4<>+0x6d80(SB)/8, $231  // bitrev[3504] = 231
DATA bitrev_size4096_radix4<>+0x6d88(SB)/8, $1255 // bitrev[3505] = 1255
DATA bitrev_size4096_radix4<>+0x6d90(SB)/8, $2279 // bitrev[3506] = 2279
DATA bitrev_size4096_radix4<>+0x6d98(SB)/8, $3303 // bitrev[3507] = 3303
DATA bitrev_size4096_radix4<>+0x6da0(SB)/8, $487  // bitrev[3508] = 487
DATA bitrev_size4096_radix4<>+0x6da8(SB)/8, $1511 // bitrev[3509] = 1511
DATA bitrev_size4096_radix4<>+0x6db0(SB)/8, $2535 // bitrev[3510] = 2535
DATA bitrev_size4096_radix4<>+0x6db8(SB)/8, $3559 // bitrev[3511] = 3559
DATA bitrev_size4096_radix4<>+0x6dc0(SB)/8, $743  // bitrev[3512] = 743
DATA bitrev_size4096_radix4<>+0x6dc8(SB)/8, $1767 // bitrev[3513] = 1767
DATA bitrev_size4096_radix4<>+0x6dd0(SB)/8, $2791 // bitrev[3514] = 2791
DATA bitrev_size4096_radix4<>+0x6dd8(SB)/8, $3815 // bitrev[3515] = 3815
DATA bitrev_size4096_radix4<>+0x6de0(SB)/8, $999  // bitrev[3516] = 999
DATA bitrev_size4096_radix4<>+0x6de8(SB)/8, $2023 // bitrev[3517] = 2023
DATA bitrev_size4096_radix4<>+0x6df0(SB)/8, $3047 // bitrev[3518] = 3047
DATA bitrev_size4096_radix4<>+0x6df8(SB)/8, $4071 // bitrev[3519] = 4071
DATA bitrev_size4096_radix4<>+0x6e00(SB)/8, $55   // bitrev[3520] = 55
DATA bitrev_size4096_radix4<>+0x6e08(SB)/8, $1079 // bitrev[3521] = 1079
DATA bitrev_size4096_radix4<>+0x6e10(SB)/8, $2103 // bitrev[3522] = 2103
DATA bitrev_size4096_radix4<>+0x6e18(SB)/8, $3127 // bitrev[3523] = 3127
DATA bitrev_size4096_radix4<>+0x6e20(SB)/8, $311  // bitrev[3524] = 311
DATA bitrev_size4096_radix4<>+0x6e28(SB)/8, $1335 // bitrev[3525] = 1335
DATA bitrev_size4096_radix4<>+0x6e30(SB)/8, $2359 // bitrev[3526] = 2359
DATA bitrev_size4096_radix4<>+0x6e38(SB)/8, $3383 // bitrev[3527] = 3383
DATA bitrev_size4096_radix4<>+0x6e40(SB)/8, $567  // bitrev[3528] = 567
DATA bitrev_size4096_radix4<>+0x6e48(SB)/8, $1591 // bitrev[3529] = 1591
DATA bitrev_size4096_radix4<>+0x6e50(SB)/8, $2615 // bitrev[3530] = 2615
DATA bitrev_size4096_radix4<>+0x6e58(SB)/8, $3639 // bitrev[3531] = 3639
DATA bitrev_size4096_radix4<>+0x6e60(SB)/8, $823  // bitrev[3532] = 823
DATA bitrev_size4096_radix4<>+0x6e68(SB)/8, $1847 // bitrev[3533] = 1847
DATA bitrev_size4096_radix4<>+0x6e70(SB)/8, $2871 // bitrev[3534] = 2871
DATA bitrev_size4096_radix4<>+0x6e78(SB)/8, $3895 // bitrev[3535] = 3895
DATA bitrev_size4096_radix4<>+0x6e80(SB)/8, $119  // bitrev[3536] = 119
DATA bitrev_size4096_radix4<>+0x6e88(SB)/8, $1143 // bitrev[3537] = 1143
DATA bitrev_size4096_radix4<>+0x6e90(SB)/8, $2167 // bitrev[3538] = 2167
DATA bitrev_size4096_radix4<>+0x6e98(SB)/8, $3191 // bitrev[3539] = 3191
DATA bitrev_size4096_radix4<>+0x6ea0(SB)/8, $375  // bitrev[3540] = 375
DATA bitrev_size4096_radix4<>+0x6ea8(SB)/8, $1399 // bitrev[3541] = 1399
DATA bitrev_size4096_radix4<>+0x6eb0(SB)/8, $2423 // bitrev[3542] = 2423
DATA bitrev_size4096_radix4<>+0x6eb8(SB)/8, $3447 // bitrev[3543] = 3447
DATA bitrev_size4096_radix4<>+0x6ec0(SB)/8, $631  // bitrev[3544] = 631
DATA bitrev_size4096_radix4<>+0x6ec8(SB)/8, $1655 // bitrev[3545] = 1655
DATA bitrev_size4096_radix4<>+0x6ed0(SB)/8, $2679 // bitrev[3546] = 2679
DATA bitrev_size4096_radix4<>+0x6ed8(SB)/8, $3703 // bitrev[3547] = 3703
DATA bitrev_size4096_radix4<>+0x6ee0(SB)/8, $887  // bitrev[3548] = 887
DATA bitrev_size4096_radix4<>+0x6ee8(SB)/8, $1911 // bitrev[3549] = 1911
DATA bitrev_size4096_radix4<>+0x6ef0(SB)/8, $2935 // bitrev[3550] = 2935
DATA bitrev_size4096_radix4<>+0x6ef8(SB)/8, $3959 // bitrev[3551] = 3959
DATA bitrev_size4096_radix4<>+0x6f00(SB)/8, $183  // bitrev[3552] = 183
DATA bitrev_size4096_radix4<>+0x6f08(SB)/8, $1207 // bitrev[3553] = 1207
DATA bitrev_size4096_radix4<>+0x6f10(SB)/8, $2231 // bitrev[3554] = 2231
DATA bitrev_size4096_radix4<>+0x6f18(SB)/8, $3255 // bitrev[3555] = 3255
DATA bitrev_size4096_radix4<>+0x6f20(SB)/8, $439  // bitrev[3556] = 439
DATA bitrev_size4096_radix4<>+0x6f28(SB)/8, $1463 // bitrev[3557] = 1463
DATA bitrev_size4096_radix4<>+0x6f30(SB)/8, $2487 // bitrev[3558] = 2487
DATA bitrev_size4096_radix4<>+0x6f38(SB)/8, $3511 // bitrev[3559] = 3511
DATA bitrev_size4096_radix4<>+0x6f40(SB)/8, $695  // bitrev[3560] = 695
DATA bitrev_size4096_radix4<>+0x6f48(SB)/8, $1719 // bitrev[3561] = 1719
DATA bitrev_size4096_radix4<>+0x6f50(SB)/8, $2743 // bitrev[3562] = 2743
DATA bitrev_size4096_radix4<>+0x6f58(SB)/8, $3767 // bitrev[3563] = 3767
DATA bitrev_size4096_radix4<>+0x6f60(SB)/8, $951  // bitrev[3564] = 951
DATA bitrev_size4096_radix4<>+0x6f68(SB)/8, $1975 // bitrev[3565] = 1975
DATA bitrev_size4096_radix4<>+0x6f70(SB)/8, $2999 // bitrev[3566] = 2999
DATA bitrev_size4096_radix4<>+0x6f78(SB)/8, $4023 // bitrev[3567] = 4023
DATA bitrev_size4096_radix4<>+0x6f80(SB)/8, $247  // bitrev[3568] = 247
DATA bitrev_size4096_radix4<>+0x6f88(SB)/8, $1271 // bitrev[3569] = 1271
DATA bitrev_size4096_radix4<>+0x6f90(SB)/8, $2295 // bitrev[3570] = 2295
DATA bitrev_size4096_radix4<>+0x6f98(SB)/8, $3319 // bitrev[3571] = 3319
DATA bitrev_size4096_radix4<>+0x6fa0(SB)/8, $503  // bitrev[3572] = 503
DATA bitrev_size4096_radix4<>+0x6fa8(SB)/8, $1527 // bitrev[3573] = 1527
DATA bitrev_size4096_radix4<>+0x6fb0(SB)/8, $2551 // bitrev[3574] = 2551
DATA bitrev_size4096_radix4<>+0x6fb8(SB)/8, $3575 // bitrev[3575] = 3575
DATA bitrev_size4096_radix4<>+0x6fc0(SB)/8, $759  // bitrev[3576] = 759
DATA bitrev_size4096_radix4<>+0x6fc8(SB)/8, $1783 // bitrev[3577] = 1783
DATA bitrev_size4096_radix4<>+0x6fd0(SB)/8, $2807 // bitrev[3578] = 2807
DATA bitrev_size4096_radix4<>+0x6fd8(SB)/8, $3831 // bitrev[3579] = 3831
DATA bitrev_size4096_radix4<>+0x6fe0(SB)/8, $1015 // bitrev[3580] = 1015
DATA bitrev_size4096_radix4<>+0x6fe8(SB)/8, $2039 // bitrev[3581] = 2039
DATA bitrev_size4096_radix4<>+0x6ff0(SB)/8, $3063 // bitrev[3582] = 3063
DATA bitrev_size4096_radix4<>+0x6ff8(SB)/8, $4087 // bitrev[3583] = 4087
DATA bitrev_size4096_radix4<>+0x7000(SB)/8, $11   // bitrev[3584] = 11
DATA bitrev_size4096_radix4<>+0x7008(SB)/8, $1035 // bitrev[3585] = 1035
DATA bitrev_size4096_radix4<>+0x7010(SB)/8, $2059 // bitrev[3586] = 2059
DATA bitrev_size4096_radix4<>+0x7018(SB)/8, $3083 // bitrev[3587] = 3083
DATA bitrev_size4096_radix4<>+0x7020(SB)/8, $267  // bitrev[3588] = 267
DATA bitrev_size4096_radix4<>+0x7028(SB)/8, $1291 // bitrev[3589] = 1291
DATA bitrev_size4096_radix4<>+0x7030(SB)/8, $2315 // bitrev[3590] = 2315
DATA bitrev_size4096_radix4<>+0x7038(SB)/8, $3339 // bitrev[3591] = 3339
DATA bitrev_size4096_radix4<>+0x7040(SB)/8, $523  // bitrev[3592] = 523
DATA bitrev_size4096_radix4<>+0x7048(SB)/8, $1547 // bitrev[3593] = 1547
DATA bitrev_size4096_radix4<>+0x7050(SB)/8, $2571 // bitrev[3594] = 2571
DATA bitrev_size4096_radix4<>+0x7058(SB)/8, $3595 // bitrev[3595] = 3595
DATA bitrev_size4096_radix4<>+0x7060(SB)/8, $779  // bitrev[3596] = 779
DATA bitrev_size4096_radix4<>+0x7068(SB)/8, $1803 // bitrev[3597] = 1803
DATA bitrev_size4096_radix4<>+0x7070(SB)/8, $2827 // bitrev[3598] = 2827
DATA bitrev_size4096_radix4<>+0x7078(SB)/8, $3851 // bitrev[3599] = 3851
DATA bitrev_size4096_radix4<>+0x7080(SB)/8, $75   // bitrev[3600] = 75
DATA bitrev_size4096_radix4<>+0x7088(SB)/8, $1099 // bitrev[3601] = 1099
DATA bitrev_size4096_radix4<>+0x7090(SB)/8, $2123 // bitrev[3602] = 2123
DATA bitrev_size4096_radix4<>+0x7098(SB)/8, $3147 // bitrev[3603] = 3147
DATA bitrev_size4096_radix4<>+0x70a0(SB)/8, $331  // bitrev[3604] = 331
DATA bitrev_size4096_radix4<>+0x70a8(SB)/8, $1355 // bitrev[3605] = 1355
DATA bitrev_size4096_radix4<>+0x70b0(SB)/8, $2379 // bitrev[3606] = 2379
DATA bitrev_size4096_radix4<>+0x70b8(SB)/8, $3403 // bitrev[3607] = 3403
DATA bitrev_size4096_radix4<>+0x70c0(SB)/8, $587  // bitrev[3608] = 587
DATA bitrev_size4096_radix4<>+0x70c8(SB)/8, $1611 // bitrev[3609] = 1611
DATA bitrev_size4096_radix4<>+0x70d0(SB)/8, $2635 // bitrev[3610] = 2635
DATA bitrev_size4096_radix4<>+0x70d8(SB)/8, $3659 // bitrev[3611] = 3659
DATA bitrev_size4096_radix4<>+0x70e0(SB)/8, $843  // bitrev[3612] = 843
DATA bitrev_size4096_radix4<>+0x70e8(SB)/8, $1867 // bitrev[3613] = 1867
DATA bitrev_size4096_radix4<>+0x70f0(SB)/8, $2891 // bitrev[3614] = 2891
DATA bitrev_size4096_radix4<>+0x70f8(SB)/8, $3915 // bitrev[3615] = 3915
DATA bitrev_size4096_radix4<>+0x7100(SB)/8, $139  // bitrev[3616] = 139
DATA bitrev_size4096_radix4<>+0x7108(SB)/8, $1163 // bitrev[3617] = 1163
DATA bitrev_size4096_radix4<>+0x7110(SB)/8, $2187 // bitrev[3618] = 2187
DATA bitrev_size4096_radix4<>+0x7118(SB)/8, $3211 // bitrev[3619] = 3211
DATA bitrev_size4096_radix4<>+0x7120(SB)/8, $395  // bitrev[3620] = 395
DATA bitrev_size4096_radix4<>+0x7128(SB)/8, $1419 // bitrev[3621] = 1419
DATA bitrev_size4096_radix4<>+0x7130(SB)/8, $2443 // bitrev[3622] = 2443
DATA bitrev_size4096_radix4<>+0x7138(SB)/8, $3467 // bitrev[3623] = 3467
DATA bitrev_size4096_radix4<>+0x7140(SB)/8, $651  // bitrev[3624] = 651
DATA bitrev_size4096_radix4<>+0x7148(SB)/8, $1675 // bitrev[3625] = 1675
DATA bitrev_size4096_radix4<>+0x7150(SB)/8, $2699 // bitrev[3626] = 2699
DATA bitrev_size4096_radix4<>+0x7158(SB)/8, $3723 // bitrev[3627] = 3723
DATA bitrev_size4096_radix4<>+0x7160(SB)/8, $907  // bitrev[3628] = 907
DATA bitrev_size4096_radix4<>+0x7168(SB)/8, $1931 // bitrev[3629] = 1931
DATA bitrev_size4096_radix4<>+0x7170(SB)/8, $2955 // bitrev[3630] = 2955
DATA bitrev_size4096_radix4<>+0x7178(SB)/8, $3979 // bitrev[3631] = 3979
DATA bitrev_size4096_radix4<>+0x7180(SB)/8, $203  // bitrev[3632] = 203
DATA bitrev_size4096_radix4<>+0x7188(SB)/8, $1227 // bitrev[3633] = 1227
DATA bitrev_size4096_radix4<>+0x7190(SB)/8, $2251 // bitrev[3634] = 2251
DATA bitrev_size4096_radix4<>+0x7198(SB)/8, $3275 // bitrev[3635] = 3275
DATA bitrev_size4096_radix4<>+0x71a0(SB)/8, $459  // bitrev[3636] = 459
DATA bitrev_size4096_radix4<>+0x71a8(SB)/8, $1483 // bitrev[3637] = 1483
DATA bitrev_size4096_radix4<>+0x71b0(SB)/8, $2507 // bitrev[3638] = 2507
DATA bitrev_size4096_radix4<>+0x71b8(SB)/8, $3531 // bitrev[3639] = 3531
DATA bitrev_size4096_radix4<>+0x71c0(SB)/8, $715  // bitrev[3640] = 715
DATA bitrev_size4096_radix4<>+0x71c8(SB)/8, $1739 // bitrev[3641] = 1739
DATA bitrev_size4096_radix4<>+0x71d0(SB)/8, $2763 // bitrev[3642] = 2763
DATA bitrev_size4096_radix4<>+0x71d8(SB)/8, $3787 // bitrev[3643] = 3787
DATA bitrev_size4096_radix4<>+0x71e0(SB)/8, $971  // bitrev[3644] = 971
DATA bitrev_size4096_radix4<>+0x71e8(SB)/8, $1995 // bitrev[3645] = 1995
DATA bitrev_size4096_radix4<>+0x71f0(SB)/8, $3019 // bitrev[3646] = 3019
DATA bitrev_size4096_radix4<>+0x71f8(SB)/8, $4043 // bitrev[3647] = 4043
DATA bitrev_size4096_radix4<>+0x7200(SB)/8, $27   // bitrev[3648] = 27
DATA bitrev_size4096_radix4<>+0x7208(SB)/8, $1051 // bitrev[3649] = 1051
DATA bitrev_size4096_radix4<>+0x7210(SB)/8, $2075 // bitrev[3650] = 2075
DATA bitrev_size4096_radix4<>+0x7218(SB)/8, $3099 // bitrev[3651] = 3099
DATA bitrev_size4096_radix4<>+0x7220(SB)/8, $283  // bitrev[3652] = 283
DATA bitrev_size4096_radix4<>+0x7228(SB)/8, $1307 // bitrev[3653] = 1307
DATA bitrev_size4096_radix4<>+0x7230(SB)/8, $2331 // bitrev[3654] = 2331
DATA bitrev_size4096_radix4<>+0x7238(SB)/8, $3355 // bitrev[3655] = 3355
DATA bitrev_size4096_radix4<>+0x7240(SB)/8, $539  // bitrev[3656] = 539
DATA bitrev_size4096_radix4<>+0x7248(SB)/8, $1563 // bitrev[3657] = 1563
DATA bitrev_size4096_radix4<>+0x7250(SB)/8, $2587 // bitrev[3658] = 2587
DATA bitrev_size4096_radix4<>+0x7258(SB)/8, $3611 // bitrev[3659] = 3611
DATA bitrev_size4096_radix4<>+0x7260(SB)/8, $795  // bitrev[3660] = 795
DATA bitrev_size4096_radix4<>+0x7268(SB)/8, $1819 // bitrev[3661] = 1819
DATA bitrev_size4096_radix4<>+0x7270(SB)/8, $2843 // bitrev[3662] = 2843
DATA bitrev_size4096_radix4<>+0x7278(SB)/8, $3867 // bitrev[3663] = 3867
DATA bitrev_size4096_radix4<>+0x7280(SB)/8, $91   // bitrev[3664] = 91
DATA bitrev_size4096_radix4<>+0x7288(SB)/8, $1115 // bitrev[3665] = 1115
DATA bitrev_size4096_radix4<>+0x7290(SB)/8, $2139 // bitrev[3666] = 2139
DATA bitrev_size4096_radix4<>+0x7298(SB)/8, $3163 // bitrev[3667] = 3163
DATA bitrev_size4096_radix4<>+0x72a0(SB)/8, $347  // bitrev[3668] = 347
DATA bitrev_size4096_radix4<>+0x72a8(SB)/8, $1371 // bitrev[3669] = 1371
DATA bitrev_size4096_radix4<>+0x72b0(SB)/8, $2395 // bitrev[3670] = 2395
DATA bitrev_size4096_radix4<>+0x72b8(SB)/8, $3419 // bitrev[3671] = 3419
DATA bitrev_size4096_radix4<>+0x72c0(SB)/8, $603  // bitrev[3672] = 603
DATA bitrev_size4096_radix4<>+0x72c8(SB)/8, $1627 // bitrev[3673] = 1627
DATA bitrev_size4096_radix4<>+0x72d0(SB)/8, $2651 // bitrev[3674] = 2651
DATA bitrev_size4096_radix4<>+0x72d8(SB)/8, $3675 // bitrev[3675] = 3675
DATA bitrev_size4096_radix4<>+0x72e0(SB)/8, $859  // bitrev[3676] = 859
DATA bitrev_size4096_radix4<>+0x72e8(SB)/8, $1883 // bitrev[3677] = 1883
DATA bitrev_size4096_radix4<>+0x72f0(SB)/8, $2907 // bitrev[3678] = 2907
DATA bitrev_size4096_radix4<>+0x72f8(SB)/8, $3931 // bitrev[3679] = 3931
DATA bitrev_size4096_radix4<>+0x7300(SB)/8, $155  // bitrev[3680] = 155
DATA bitrev_size4096_radix4<>+0x7308(SB)/8, $1179 // bitrev[3681] = 1179
DATA bitrev_size4096_radix4<>+0x7310(SB)/8, $2203 // bitrev[3682] = 2203
DATA bitrev_size4096_radix4<>+0x7318(SB)/8, $3227 // bitrev[3683] = 3227
DATA bitrev_size4096_radix4<>+0x7320(SB)/8, $411  // bitrev[3684] = 411
DATA bitrev_size4096_radix4<>+0x7328(SB)/8, $1435 // bitrev[3685] = 1435
DATA bitrev_size4096_radix4<>+0x7330(SB)/8, $2459 // bitrev[3686] = 2459
DATA bitrev_size4096_radix4<>+0x7338(SB)/8, $3483 // bitrev[3687] = 3483
DATA bitrev_size4096_radix4<>+0x7340(SB)/8, $667  // bitrev[3688] = 667
DATA bitrev_size4096_radix4<>+0x7348(SB)/8, $1691 // bitrev[3689] = 1691
DATA bitrev_size4096_radix4<>+0x7350(SB)/8, $2715 // bitrev[3690] = 2715
DATA bitrev_size4096_radix4<>+0x7358(SB)/8, $3739 // bitrev[3691] = 3739
DATA bitrev_size4096_radix4<>+0x7360(SB)/8, $923  // bitrev[3692] = 923
DATA bitrev_size4096_radix4<>+0x7368(SB)/8, $1947 // bitrev[3693] = 1947
DATA bitrev_size4096_radix4<>+0x7370(SB)/8, $2971 // bitrev[3694] = 2971
DATA bitrev_size4096_radix4<>+0x7378(SB)/8, $3995 // bitrev[3695] = 3995
DATA bitrev_size4096_radix4<>+0x7380(SB)/8, $219  // bitrev[3696] = 219
DATA bitrev_size4096_radix4<>+0x7388(SB)/8, $1243 // bitrev[3697] = 1243
DATA bitrev_size4096_radix4<>+0x7390(SB)/8, $2267 // bitrev[3698] = 2267
DATA bitrev_size4096_radix4<>+0x7398(SB)/8, $3291 // bitrev[3699] = 3291
DATA bitrev_size4096_radix4<>+0x73a0(SB)/8, $475  // bitrev[3700] = 475
DATA bitrev_size4096_radix4<>+0x73a8(SB)/8, $1499 // bitrev[3701] = 1499
DATA bitrev_size4096_radix4<>+0x73b0(SB)/8, $2523 // bitrev[3702] = 2523
DATA bitrev_size4096_radix4<>+0x73b8(SB)/8, $3547 // bitrev[3703] = 3547
DATA bitrev_size4096_radix4<>+0x73c0(SB)/8, $731  // bitrev[3704] = 731
DATA bitrev_size4096_radix4<>+0x73c8(SB)/8, $1755 // bitrev[3705] = 1755
DATA bitrev_size4096_radix4<>+0x73d0(SB)/8, $2779 // bitrev[3706] = 2779
DATA bitrev_size4096_radix4<>+0x73d8(SB)/8, $3803 // bitrev[3707] = 3803
DATA bitrev_size4096_radix4<>+0x73e0(SB)/8, $987  // bitrev[3708] = 987
DATA bitrev_size4096_radix4<>+0x73e8(SB)/8, $2011 // bitrev[3709] = 2011
DATA bitrev_size4096_radix4<>+0x73f0(SB)/8, $3035 // bitrev[3710] = 3035
DATA bitrev_size4096_radix4<>+0x73f8(SB)/8, $4059 // bitrev[3711] = 4059
DATA bitrev_size4096_radix4<>+0x7400(SB)/8, $43   // bitrev[3712] = 43
DATA bitrev_size4096_radix4<>+0x7408(SB)/8, $1067 // bitrev[3713] = 1067
DATA bitrev_size4096_radix4<>+0x7410(SB)/8, $2091 // bitrev[3714] = 2091
DATA bitrev_size4096_radix4<>+0x7418(SB)/8, $3115 // bitrev[3715] = 3115
DATA bitrev_size4096_radix4<>+0x7420(SB)/8, $299  // bitrev[3716] = 299
DATA bitrev_size4096_radix4<>+0x7428(SB)/8, $1323 // bitrev[3717] = 1323
DATA bitrev_size4096_radix4<>+0x7430(SB)/8, $2347 // bitrev[3718] = 2347
DATA bitrev_size4096_radix4<>+0x7438(SB)/8, $3371 // bitrev[3719] = 3371
DATA bitrev_size4096_radix4<>+0x7440(SB)/8, $555  // bitrev[3720] = 555
DATA bitrev_size4096_radix4<>+0x7448(SB)/8, $1579 // bitrev[3721] = 1579
DATA bitrev_size4096_radix4<>+0x7450(SB)/8, $2603 // bitrev[3722] = 2603
DATA bitrev_size4096_radix4<>+0x7458(SB)/8, $3627 // bitrev[3723] = 3627
DATA bitrev_size4096_radix4<>+0x7460(SB)/8, $811  // bitrev[3724] = 811
DATA bitrev_size4096_radix4<>+0x7468(SB)/8, $1835 // bitrev[3725] = 1835
DATA bitrev_size4096_radix4<>+0x7470(SB)/8, $2859 // bitrev[3726] = 2859
DATA bitrev_size4096_radix4<>+0x7478(SB)/8, $3883 // bitrev[3727] = 3883
DATA bitrev_size4096_radix4<>+0x7480(SB)/8, $107  // bitrev[3728] = 107
DATA bitrev_size4096_radix4<>+0x7488(SB)/8, $1131 // bitrev[3729] = 1131
DATA bitrev_size4096_radix4<>+0x7490(SB)/8, $2155 // bitrev[3730] = 2155
DATA bitrev_size4096_radix4<>+0x7498(SB)/8, $3179 // bitrev[3731] = 3179
DATA bitrev_size4096_radix4<>+0x74a0(SB)/8, $363  // bitrev[3732] = 363
DATA bitrev_size4096_radix4<>+0x74a8(SB)/8, $1387 // bitrev[3733] = 1387
DATA bitrev_size4096_radix4<>+0x74b0(SB)/8, $2411 // bitrev[3734] = 2411
DATA bitrev_size4096_radix4<>+0x74b8(SB)/8, $3435 // bitrev[3735] = 3435
DATA bitrev_size4096_radix4<>+0x74c0(SB)/8, $619  // bitrev[3736] = 619
DATA bitrev_size4096_radix4<>+0x74c8(SB)/8, $1643 // bitrev[3737] = 1643
DATA bitrev_size4096_radix4<>+0x74d0(SB)/8, $2667 // bitrev[3738] = 2667
DATA bitrev_size4096_radix4<>+0x74d8(SB)/8, $3691 // bitrev[3739] = 3691
DATA bitrev_size4096_radix4<>+0x74e0(SB)/8, $875  // bitrev[3740] = 875
DATA bitrev_size4096_radix4<>+0x74e8(SB)/8, $1899 // bitrev[3741] = 1899
DATA bitrev_size4096_radix4<>+0x74f0(SB)/8, $2923 // bitrev[3742] = 2923
DATA bitrev_size4096_radix4<>+0x74f8(SB)/8, $3947 // bitrev[3743] = 3947
DATA bitrev_size4096_radix4<>+0x7500(SB)/8, $171  // bitrev[3744] = 171
DATA bitrev_size4096_radix4<>+0x7508(SB)/8, $1195 // bitrev[3745] = 1195
DATA bitrev_size4096_radix4<>+0x7510(SB)/8, $2219 // bitrev[3746] = 2219
DATA bitrev_size4096_radix4<>+0x7518(SB)/8, $3243 // bitrev[3747] = 3243
DATA bitrev_size4096_radix4<>+0x7520(SB)/8, $427  // bitrev[3748] = 427
DATA bitrev_size4096_radix4<>+0x7528(SB)/8, $1451 // bitrev[3749] = 1451
DATA bitrev_size4096_radix4<>+0x7530(SB)/8, $2475 // bitrev[3750] = 2475
DATA bitrev_size4096_radix4<>+0x7538(SB)/8, $3499 // bitrev[3751] = 3499
DATA bitrev_size4096_radix4<>+0x7540(SB)/8, $683  // bitrev[3752] = 683
DATA bitrev_size4096_radix4<>+0x7548(SB)/8, $1707 // bitrev[3753] = 1707
DATA bitrev_size4096_radix4<>+0x7550(SB)/8, $2731 // bitrev[3754] = 2731
DATA bitrev_size4096_radix4<>+0x7558(SB)/8, $3755 // bitrev[3755] = 3755
DATA bitrev_size4096_radix4<>+0x7560(SB)/8, $939  // bitrev[3756] = 939
DATA bitrev_size4096_radix4<>+0x7568(SB)/8, $1963 // bitrev[3757] = 1963
DATA bitrev_size4096_radix4<>+0x7570(SB)/8, $2987 // bitrev[3758] = 2987
DATA bitrev_size4096_radix4<>+0x7578(SB)/8, $4011 // bitrev[3759] = 4011
DATA bitrev_size4096_radix4<>+0x7580(SB)/8, $235  // bitrev[3760] = 235
DATA bitrev_size4096_radix4<>+0x7588(SB)/8, $1259 // bitrev[3761] = 1259
DATA bitrev_size4096_radix4<>+0x7590(SB)/8, $2283 // bitrev[3762] = 2283
DATA bitrev_size4096_radix4<>+0x7598(SB)/8, $3307 // bitrev[3763] = 3307
DATA bitrev_size4096_radix4<>+0x75a0(SB)/8, $491  // bitrev[3764] = 491
DATA bitrev_size4096_radix4<>+0x75a8(SB)/8, $1515 // bitrev[3765] = 1515
DATA bitrev_size4096_radix4<>+0x75b0(SB)/8, $2539 // bitrev[3766] = 2539
DATA bitrev_size4096_radix4<>+0x75b8(SB)/8, $3563 // bitrev[3767] = 3563
DATA bitrev_size4096_radix4<>+0x75c0(SB)/8, $747  // bitrev[3768] = 747
DATA bitrev_size4096_radix4<>+0x75c8(SB)/8, $1771 // bitrev[3769] = 1771
DATA bitrev_size4096_radix4<>+0x75d0(SB)/8, $2795 // bitrev[3770] = 2795
DATA bitrev_size4096_radix4<>+0x75d8(SB)/8, $3819 // bitrev[3771] = 3819
DATA bitrev_size4096_radix4<>+0x75e0(SB)/8, $1003 // bitrev[3772] = 1003
DATA bitrev_size4096_radix4<>+0x75e8(SB)/8, $2027 // bitrev[3773] = 2027
DATA bitrev_size4096_radix4<>+0x75f0(SB)/8, $3051 // bitrev[3774] = 3051
DATA bitrev_size4096_radix4<>+0x75f8(SB)/8, $4075 // bitrev[3775] = 4075
DATA bitrev_size4096_radix4<>+0x7600(SB)/8, $59   // bitrev[3776] = 59
DATA bitrev_size4096_radix4<>+0x7608(SB)/8, $1083 // bitrev[3777] = 1083
DATA bitrev_size4096_radix4<>+0x7610(SB)/8, $2107 // bitrev[3778] = 2107
DATA bitrev_size4096_radix4<>+0x7618(SB)/8, $3131 // bitrev[3779] = 3131
DATA bitrev_size4096_radix4<>+0x7620(SB)/8, $315  // bitrev[3780] = 315
DATA bitrev_size4096_radix4<>+0x7628(SB)/8, $1339 // bitrev[3781] = 1339
DATA bitrev_size4096_radix4<>+0x7630(SB)/8, $2363 // bitrev[3782] = 2363
DATA bitrev_size4096_radix4<>+0x7638(SB)/8, $3387 // bitrev[3783] = 3387
DATA bitrev_size4096_radix4<>+0x7640(SB)/8, $571  // bitrev[3784] = 571
DATA bitrev_size4096_radix4<>+0x7648(SB)/8, $1595 // bitrev[3785] = 1595
DATA bitrev_size4096_radix4<>+0x7650(SB)/8, $2619 // bitrev[3786] = 2619
DATA bitrev_size4096_radix4<>+0x7658(SB)/8, $3643 // bitrev[3787] = 3643
DATA bitrev_size4096_radix4<>+0x7660(SB)/8, $827  // bitrev[3788] = 827
DATA bitrev_size4096_radix4<>+0x7668(SB)/8, $1851 // bitrev[3789] = 1851
DATA bitrev_size4096_radix4<>+0x7670(SB)/8, $2875 // bitrev[3790] = 2875
DATA bitrev_size4096_radix4<>+0x7678(SB)/8, $3899 // bitrev[3791] = 3899
DATA bitrev_size4096_radix4<>+0x7680(SB)/8, $123  // bitrev[3792] = 123
DATA bitrev_size4096_radix4<>+0x7688(SB)/8, $1147 // bitrev[3793] = 1147
DATA bitrev_size4096_radix4<>+0x7690(SB)/8, $2171 // bitrev[3794] = 2171
DATA bitrev_size4096_radix4<>+0x7698(SB)/8, $3195 // bitrev[3795] = 3195
DATA bitrev_size4096_radix4<>+0x76a0(SB)/8, $379  // bitrev[3796] = 379
DATA bitrev_size4096_radix4<>+0x76a8(SB)/8, $1403 // bitrev[3797] = 1403
DATA bitrev_size4096_radix4<>+0x76b0(SB)/8, $2427 // bitrev[3798] = 2427
DATA bitrev_size4096_radix4<>+0x76b8(SB)/8, $3451 // bitrev[3799] = 3451
DATA bitrev_size4096_radix4<>+0x76c0(SB)/8, $635  // bitrev[3800] = 635
DATA bitrev_size4096_radix4<>+0x76c8(SB)/8, $1659 // bitrev[3801] = 1659
DATA bitrev_size4096_radix4<>+0x76d0(SB)/8, $2683 // bitrev[3802] = 2683
DATA bitrev_size4096_radix4<>+0x76d8(SB)/8, $3707 // bitrev[3803] = 3707
DATA bitrev_size4096_radix4<>+0x76e0(SB)/8, $891  // bitrev[3804] = 891
DATA bitrev_size4096_radix4<>+0x76e8(SB)/8, $1915 // bitrev[3805] = 1915
DATA bitrev_size4096_radix4<>+0x76f0(SB)/8, $2939 // bitrev[3806] = 2939
DATA bitrev_size4096_radix4<>+0x76f8(SB)/8, $3963 // bitrev[3807] = 3963
DATA bitrev_size4096_radix4<>+0x7700(SB)/8, $187  // bitrev[3808] = 187
DATA bitrev_size4096_radix4<>+0x7708(SB)/8, $1211 // bitrev[3809] = 1211
DATA bitrev_size4096_radix4<>+0x7710(SB)/8, $2235 // bitrev[3810] = 2235
DATA bitrev_size4096_radix4<>+0x7718(SB)/8, $3259 // bitrev[3811] = 3259
DATA bitrev_size4096_radix4<>+0x7720(SB)/8, $443  // bitrev[3812] = 443
DATA bitrev_size4096_radix4<>+0x7728(SB)/8, $1467 // bitrev[3813] = 1467
DATA bitrev_size4096_radix4<>+0x7730(SB)/8, $2491 // bitrev[3814] = 2491
DATA bitrev_size4096_radix4<>+0x7738(SB)/8, $3515 // bitrev[3815] = 3515
DATA bitrev_size4096_radix4<>+0x7740(SB)/8, $699  // bitrev[3816] = 699
DATA bitrev_size4096_radix4<>+0x7748(SB)/8, $1723 // bitrev[3817] = 1723
DATA bitrev_size4096_radix4<>+0x7750(SB)/8, $2747 // bitrev[3818] = 2747
DATA bitrev_size4096_radix4<>+0x7758(SB)/8, $3771 // bitrev[3819] = 3771
DATA bitrev_size4096_radix4<>+0x7760(SB)/8, $955  // bitrev[3820] = 955
DATA bitrev_size4096_radix4<>+0x7768(SB)/8, $1979 // bitrev[3821] = 1979
DATA bitrev_size4096_radix4<>+0x7770(SB)/8, $3003 // bitrev[3822] = 3003
DATA bitrev_size4096_radix4<>+0x7778(SB)/8, $4027 // bitrev[3823] = 4027
DATA bitrev_size4096_radix4<>+0x7780(SB)/8, $251  // bitrev[3824] = 251
DATA bitrev_size4096_radix4<>+0x7788(SB)/8, $1275 // bitrev[3825] = 1275
DATA bitrev_size4096_radix4<>+0x7790(SB)/8, $2299 // bitrev[3826] = 2299
DATA bitrev_size4096_radix4<>+0x7798(SB)/8, $3323 // bitrev[3827] = 3323
DATA bitrev_size4096_radix4<>+0x77a0(SB)/8, $507  // bitrev[3828] = 507
DATA bitrev_size4096_radix4<>+0x77a8(SB)/8, $1531 // bitrev[3829] = 1531
DATA bitrev_size4096_radix4<>+0x77b0(SB)/8, $2555 // bitrev[3830] = 2555
DATA bitrev_size4096_radix4<>+0x77b8(SB)/8, $3579 // bitrev[3831] = 3579
DATA bitrev_size4096_radix4<>+0x77c0(SB)/8, $763  // bitrev[3832] = 763
DATA bitrev_size4096_radix4<>+0x77c8(SB)/8, $1787 // bitrev[3833] = 1787
DATA bitrev_size4096_radix4<>+0x77d0(SB)/8, $2811 // bitrev[3834] = 2811
DATA bitrev_size4096_radix4<>+0x77d8(SB)/8, $3835 // bitrev[3835] = 3835
DATA bitrev_size4096_radix4<>+0x77e0(SB)/8, $1019 // bitrev[3836] = 1019
DATA bitrev_size4096_radix4<>+0x77e8(SB)/8, $2043 // bitrev[3837] = 2043
DATA bitrev_size4096_radix4<>+0x77f0(SB)/8, $3067 // bitrev[3838] = 3067
DATA bitrev_size4096_radix4<>+0x77f8(SB)/8, $4091 // bitrev[3839] = 4091
DATA bitrev_size4096_radix4<>+0x7800(SB)/8, $15   // bitrev[3840] = 15
DATA bitrev_size4096_radix4<>+0x7808(SB)/8, $1039 // bitrev[3841] = 1039
DATA bitrev_size4096_radix4<>+0x7810(SB)/8, $2063 // bitrev[3842] = 2063
DATA bitrev_size4096_radix4<>+0x7818(SB)/8, $3087 // bitrev[3843] = 3087
DATA bitrev_size4096_radix4<>+0x7820(SB)/8, $271  // bitrev[3844] = 271
DATA bitrev_size4096_radix4<>+0x7828(SB)/8, $1295 // bitrev[3845] = 1295
DATA bitrev_size4096_radix4<>+0x7830(SB)/8, $2319 // bitrev[3846] = 2319
DATA bitrev_size4096_radix4<>+0x7838(SB)/8, $3343 // bitrev[3847] = 3343
DATA bitrev_size4096_radix4<>+0x7840(SB)/8, $527  // bitrev[3848] = 527
DATA bitrev_size4096_radix4<>+0x7848(SB)/8, $1551 // bitrev[3849] = 1551
DATA bitrev_size4096_radix4<>+0x7850(SB)/8, $2575 // bitrev[3850] = 2575
DATA bitrev_size4096_radix4<>+0x7858(SB)/8, $3599 // bitrev[3851] = 3599
DATA bitrev_size4096_radix4<>+0x7860(SB)/8, $783  // bitrev[3852] = 783
DATA bitrev_size4096_radix4<>+0x7868(SB)/8, $1807 // bitrev[3853] = 1807
DATA bitrev_size4096_radix4<>+0x7870(SB)/8, $2831 // bitrev[3854] = 2831
DATA bitrev_size4096_radix4<>+0x7878(SB)/8, $3855 // bitrev[3855] = 3855
DATA bitrev_size4096_radix4<>+0x7880(SB)/8, $79   // bitrev[3856] = 79
DATA bitrev_size4096_radix4<>+0x7888(SB)/8, $1103 // bitrev[3857] = 1103
DATA bitrev_size4096_radix4<>+0x7890(SB)/8, $2127 // bitrev[3858] = 2127
DATA bitrev_size4096_radix4<>+0x7898(SB)/8, $3151 // bitrev[3859] = 3151
DATA bitrev_size4096_radix4<>+0x78a0(SB)/8, $335  // bitrev[3860] = 335
DATA bitrev_size4096_radix4<>+0x78a8(SB)/8, $1359 // bitrev[3861] = 1359
DATA bitrev_size4096_radix4<>+0x78b0(SB)/8, $2383 // bitrev[3862] = 2383
DATA bitrev_size4096_radix4<>+0x78b8(SB)/8, $3407 // bitrev[3863] = 3407
DATA bitrev_size4096_radix4<>+0x78c0(SB)/8, $591  // bitrev[3864] = 591
DATA bitrev_size4096_radix4<>+0x78c8(SB)/8, $1615 // bitrev[3865] = 1615
DATA bitrev_size4096_radix4<>+0x78d0(SB)/8, $2639 // bitrev[3866] = 2639
DATA bitrev_size4096_radix4<>+0x78d8(SB)/8, $3663 // bitrev[3867] = 3663
DATA bitrev_size4096_radix4<>+0x78e0(SB)/8, $847  // bitrev[3868] = 847
DATA bitrev_size4096_radix4<>+0x78e8(SB)/8, $1871 // bitrev[3869] = 1871
DATA bitrev_size4096_radix4<>+0x78f0(SB)/8, $2895 // bitrev[3870] = 2895
DATA bitrev_size4096_radix4<>+0x78f8(SB)/8, $3919 // bitrev[3871] = 3919
DATA bitrev_size4096_radix4<>+0x7900(SB)/8, $143  // bitrev[3872] = 143
DATA bitrev_size4096_radix4<>+0x7908(SB)/8, $1167 // bitrev[3873] = 1167
DATA bitrev_size4096_radix4<>+0x7910(SB)/8, $2191 // bitrev[3874] = 2191
DATA bitrev_size4096_radix4<>+0x7918(SB)/8, $3215 // bitrev[3875] = 3215
DATA bitrev_size4096_radix4<>+0x7920(SB)/8, $399  // bitrev[3876] = 399
DATA bitrev_size4096_radix4<>+0x7928(SB)/8, $1423 // bitrev[3877] = 1423
DATA bitrev_size4096_radix4<>+0x7930(SB)/8, $2447 // bitrev[3878] = 2447
DATA bitrev_size4096_radix4<>+0x7938(SB)/8, $3471 // bitrev[3879] = 3471
DATA bitrev_size4096_radix4<>+0x7940(SB)/8, $655  // bitrev[3880] = 655
DATA bitrev_size4096_radix4<>+0x7948(SB)/8, $1679 // bitrev[3881] = 1679
DATA bitrev_size4096_radix4<>+0x7950(SB)/8, $2703 // bitrev[3882] = 2703
DATA bitrev_size4096_radix4<>+0x7958(SB)/8, $3727 // bitrev[3883] = 3727
DATA bitrev_size4096_radix4<>+0x7960(SB)/8, $911  // bitrev[3884] = 911
DATA bitrev_size4096_radix4<>+0x7968(SB)/8, $1935 // bitrev[3885] = 1935
DATA bitrev_size4096_radix4<>+0x7970(SB)/8, $2959 // bitrev[3886] = 2959
DATA bitrev_size4096_radix4<>+0x7978(SB)/8, $3983 // bitrev[3887] = 3983
DATA bitrev_size4096_radix4<>+0x7980(SB)/8, $207  // bitrev[3888] = 207
DATA bitrev_size4096_radix4<>+0x7988(SB)/8, $1231 // bitrev[3889] = 1231
DATA bitrev_size4096_radix4<>+0x7990(SB)/8, $2255 // bitrev[3890] = 2255
DATA bitrev_size4096_radix4<>+0x7998(SB)/8, $3279 // bitrev[3891] = 3279
DATA bitrev_size4096_radix4<>+0x79a0(SB)/8, $463  // bitrev[3892] = 463
DATA bitrev_size4096_radix4<>+0x79a8(SB)/8, $1487 // bitrev[3893] = 1487
DATA bitrev_size4096_radix4<>+0x79b0(SB)/8, $2511 // bitrev[3894] = 2511
DATA bitrev_size4096_radix4<>+0x79b8(SB)/8, $3535 // bitrev[3895] = 3535
DATA bitrev_size4096_radix4<>+0x79c0(SB)/8, $719  // bitrev[3896] = 719
DATA bitrev_size4096_radix4<>+0x79c8(SB)/8, $1743 // bitrev[3897] = 1743
DATA bitrev_size4096_radix4<>+0x79d0(SB)/8, $2767 // bitrev[3898] = 2767
DATA bitrev_size4096_radix4<>+0x79d8(SB)/8, $3791 // bitrev[3899] = 3791
DATA bitrev_size4096_radix4<>+0x79e0(SB)/8, $975  // bitrev[3900] = 975
DATA bitrev_size4096_radix4<>+0x79e8(SB)/8, $1999 // bitrev[3901] = 1999
DATA bitrev_size4096_radix4<>+0x79f0(SB)/8, $3023 // bitrev[3902] = 3023
DATA bitrev_size4096_radix4<>+0x79f8(SB)/8, $4047 // bitrev[3903] = 4047
DATA bitrev_size4096_radix4<>+0x7a00(SB)/8, $31   // bitrev[3904] = 31
DATA bitrev_size4096_radix4<>+0x7a08(SB)/8, $1055 // bitrev[3905] = 1055
DATA bitrev_size4096_radix4<>+0x7a10(SB)/8, $2079 // bitrev[3906] = 2079
DATA bitrev_size4096_radix4<>+0x7a18(SB)/8, $3103 // bitrev[3907] = 3103
DATA bitrev_size4096_radix4<>+0x7a20(SB)/8, $287  // bitrev[3908] = 287
DATA bitrev_size4096_radix4<>+0x7a28(SB)/8, $1311 // bitrev[3909] = 1311
DATA bitrev_size4096_radix4<>+0x7a30(SB)/8, $2335 // bitrev[3910] = 2335
DATA bitrev_size4096_radix4<>+0x7a38(SB)/8, $3359 // bitrev[3911] = 3359
DATA bitrev_size4096_radix4<>+0x7a40(SB)/8, $543  // bitrev[3912] = 543
DATA bitrev_size4096_radix4<>+0x7a48(SB)/8, $1567 // bitrev[3913] = 1567
DATA bitrev_size4096_radix4<>+0x7a50(SB)/8, $2591 // bitrev[3914] = 2591
DATA bitrev_size4096_radix4<>+0x7a58(SB)/8, $3615 // bitrev[3915] = 3615
DATA bitrev_size4096_radix4<>+0x7a60(SB)/8, $799  // bitrev[3916] = 799
DATA bitrev_size4096_radix4<>+0x7a68(SB)/8, $1823 // bitrev[3917] = 1823
DATA bitrev_size4096_radix4<>+0x7a70(SB)/8, $2847 // bitrev[3918] = 2847
DATA bitrev_size4096_radix4<>+0x7a78(SB)/8, $3871 // bitrev[3919] = 3871
DATA bitrev_size4096_radix4<>+0x7a80(SB)/8, $95   // bitrev[3920] = 95
DATA bitrev_size4096_radix4<>+0x7a88(SB)/8, $1119 // bitrev[3921] = 1119
DATA bitrev_size4096_radix4<>+0x7a90(SB)/8, $2143 // bitrev[3922] = 2143
DATA bitrev_size4096_radix4<>+0x7a98(SB)/8, $3167 // bitrev[3923] = 3167
DATA bitrev_size4096_radix4<>+0x7aa0(SB)/8, $351  // bitrev[3924] = 351
DATA bitrev_size4096_radix4<>+0x7aa8(SB)/8, $1375 // bitrev[3925] = 1375
DATA bitrev_size4096_radix4<>+0x7ab0(SB)/8, $2399 // bitrev[3926] = 2399
DATA bitrev_size4096_radix4<>+0x7ab8(SB)/8, $3423 // bitrev[3927] = 3423
DATA bitrev_size4096_radix4<>+0x7ac0(SB)/8, $607  // bitrev[3928] = 607
DATA bitrev_size4096_radix4<>+0x7ac8(SB)/8, $1631 // bitrev[3929] = 1631
DATA bitrev_size4096_radix4<>+0x7ad0(SB)/8, $2655 // bitrev[3930] = 2655
DATA bitrev_size4096_radix4<>+0x7ad8(SB)/8, $3679 // bitrev[3931] = 3679
DATA bitrev_size4096_radix4<>+0x7ae0(SB)/8, $863  // bitrev[3932] = 863
DATA bitrev_size4096_radix4<>+0x7ae8(SB)/8, $1887 // bitrev[3933] = 1887
DATA bitrev_size4096_radix4<>+0x7af0(SB)/8, $2911 // bitrev[3934] = 2911
DATA bitrev_size4096_radix4<>+0x7af8(SB)/8, $3935 // bitrev[3935] = 3935
DATA bitrev_size4096_radix4<>+0x7b00(SB)/8, $159  // bitrev[3936] = 159
DATA bitrev_size4096_radix4<>+0x7b08(SB)/8, $1183 // bitrev[3937] = 1183
DATA bitrev_size4096_radix4<>+0x7b10(SB)/8, $2207 // bitrev[3938] = 2207
DATA bitrev_size4096_radix4<>+0x7b18(SB)/8, $3231 // bitrev[3939] = 3231
DATA bitrev_size4096_radix4<>+0x7b20(SB)/8, $415  // bitrev[3940] = 415
DATA bitrev_size4096_radix4<>+0x7b28(SB)/8, $1439 // bitrev[3941] = 1439
DATA bitrev_size4096_radix4<>+0x7b30(SB)/8, $2463 // bitrev[3942] = 2463
DATA bitrev_size4096_radix4<>+0x7b38(SB)/8, $3487 // bitrev[3943] = 3487
DATA bitrev_size4096_radix4<>+0x7b40(SB)/8, $671  // bitrev[3944] = 671
DATA bitrev_size4096_radix4<>+0x7b48(SB)/8, $1695 // bitrev[3945] = 1695
DATA bitrev_size4096_radix4<>+0x7b50(SB)/8, $2719 // bitrev[3946] = 2719
DATA bitrev_size4096_radix4<>+0x7b58(SB)/8, $3743 // bitrev[3947] = 3743
DATA bitrev_size4096_radix4<>+0x7b60(SB)/8, $927  // bitrev[3948] = 927
DATA bitrev_size4096_radix4<>+0x7b68(SB)/8, $1951 // bitrev[3949] = 1951
DATA bitrev_size4096_radix4<>+0x7b70(SB)/8, $2975 // bitrev[3950] = 2975
DATA bitrev_size4096_radix4<>+0x7b78(SB)/8, $3999 // bitrev[3951] = 3999
DATA bitrev_size4096_radix4<>+0x7b80(SB)/8, $223  // bitrev[3952] = 223
DATA bitrev_size4096_radix4<>+0x7b88(SB)/8, $1247 // bitrev[3953] = 1247
DATA bitrev_size4096_radix4<>+0x7b90(SB)/8, $2271 // bitrev[3954] = 2271
DATA bitrev_size4096_radix4<>+0x7b98(SB)/8, $3295 // bitrev[3955] = 3295
DATA bitrev_size4096_radix4<>+0x7ba0(SB)/8, $479  // bitrev[3956] = 479
DATA bitrev_size4096_radix4<>+0x7ba8(SB)/8, $1503 // bitrev[3957] = 1503
DATA bitrev_size4096_radix4<>+0x7bb0(SB)/8, $2527 // bitrev[3958] = 2527
DATA bitrev_size4096_radix4<>+0x7bb8(SB)/8, $3551 // bitrev[3959] = 3551
DATA bitrev_size4096_radix4<>+0x7bc0(SB)/8, $735  // bitrev[3960] = 735
DATA bitrev_size4096_radix4<>+0x7bc8(SB)/8, $1759 // bitrev[3961] = 1759
DATA bitrev_size4096_radix4<>+0x7bd0(SB)/8, $2783 // bitrev[3962] = 2783
DATA bitrev_size4096_radix4<>+0x7bd8(SB)/8, $3807 // bitrev[3963] = 3807
DATA bitrev_size4096_radix4<>+0x7be0(SB)/8, $991  // bitrev[3964] = 991
DATA bitrev_size4096_radix4<>+0x7be8(SB)/8, $2015 // bitrev[3965] = 2015
DATA bitrev_size4096_radix4<>+0x7bf0(SB)/8, $3039 // bitrev[3966] = 3039
DATA bitrev_size4096_radix4<>+0x7bf8(SB)/8, $4063 // bitrev[3967] = 4063
DATA bitrev_size4096_radix4<>+0x7c00(SB)/8, $47   // bitrev[3968] = 47
DATA bitrev_size4096_radix4<>+0x7c08(SB)/8, $1071 // bitrev[3969] = 1071
DATA bitrev_size4096_radix4<>+0x7c10(SB)/8, $2095 // bitrev[3970] = 2095
DATA bitrev_size4096_radix4<>+0x7c18(SB)/8, $3119 // bitrev[3971] = 3119
DATA bitrev_size4096_radix4<>+0x7c20(SB)/8, $303  // bitrev[3972] = 303
DATA bitrev_size4096_radix4<>+0x7c28(SB)/8, $1327 // bitrev[3973] = 1327
DATA bitrev_size4096_radix4<>+0x7c30(SB)/8, $2351 // bitrev[3974] = 2351
DATA bitrev_size4096_radix4<>+0x7c38(SB)/8, $3375 // bitrev[3975] = 3375
DATA bitrev_size4096_radix4<>+0x7c40(SB)/8, $559  // bitrev[3976] = 559
DATA bitrev_size4096_radix4<>+0x7c48(SB)/8, $1583 // bitrev[3977] = 1583
DATA bitrev_size4096_radix4<>+0x7c50(SB)/8, $2607 // bitrev[3978] = 2607
DATA bitrev_size4096_radix4<>+0x7c58(SB)/8, $3631 // bitrev[3979] = 3631
DATA bitrev_size4096_radix4<>+0x7c60(SB)/8, $815  // bitrev[3980] = 815
DATA bitrev_size4096_radix4<>+0x7c68(SB)/8, $1839 // bitrev[3981] = 1839
DATA bitrev_size4096_radix4<>+0x7c70(SB)/8, $2863 // bitrev[3982] = 2863
DATA bitrev_size4096_radix4<>+0x7c78(SB)/8, $3887 // bitrev[3983] = 3887
DATA bitrev_size4096_radix4<>+0x7c80(SB)/8, $111  // bitrev[3984] = 111
DATA bitrev_size4096_radix4<>+0x7c88(SB)/8, $1135 // bitrev[3985] = 1135
DATA bitrev_size4096_radix4<>+0x7c90(SB)/8, $2159 // bitrev[3986] = 2159
DATA bitrev_size4096_radix4<>+0x7c98(SB)/8, $3183 // bitrev[3987] = 3183
DATA bitrev_size4096_radix4<>+0x7ca0(SB)/8, $367  // bitrev[3988] = 367
DATA bitrev_size4096_radix4<>+0x7ca8(SB)/8, $1391 // bitrev[3989] = 1391
DATA bitrev_size4096_radix4<>+0x7cb0(SB)/8, $2415 // bitrev[3990] = 2415
DATA bitrev_size4096_radix4<>+0x7cb8(SB)/8, $3439 // bitrev[3991] = 3439
DATA bitrev_size4096_radix4<>+0x7cc0(SB)/8, $623  // bitrev[3992] = 623
DATA bitrev_size4096_radix4<>+0x7cc8(SB)/8, $1647 // bitrev[3993] = 1647
DATA bitrev_size4096_radix4<>+0x7cd0(SB)/8, $2671 // bitrev[3994] = 2671
DATA bitrev_size4096_radix4<>+0x7cd8(SB)/8, $3695 // bitrev[3995] = 3695
DATA bitrev_size4096_radix4<>+0x7ce0(SB)/8, $879  // bitrev[3996] = 879
DATA bitrev_size4096_radix4<>+0x7ce8(SB)/8, $1903 // bitrev[3997] = 1903
DATA bitrev_size4096_radix4<>+0x7cf0(SB)/8, $2927 // bitrev[3998] = 2927
DATA bitrev_size4096_radix4<>+0x7cf8(SB)/8, $3951 // bitrev[3999] = 3951
DATA bitrev_size4096_radix4<>+0x7d00(SB)/8, $175  // bitrev[4000] = 175
DATA bitrev_size4096_radix4<>+0x7d08(SB)/8, $1199 // bitrev[4001] = 1199
DATA bitrev_size4096_radix4<>+0x7d10(SB)/8, $2223 // bitrev[4002] = 2223
DATA bitrev_size4096_radix4<>+0x7d18(SB)/8, $3247 // bitrev[4003] = 3247
DATA bitrev_size4096_radix4<>+0x7d20(SB)/8, $431  // bitrev[4004] = 431
DATA bitrev_size4096_radix4<>+0x7d28(SB)/8, $1455 // bitrev[4005] = 1455
DATA bitrev_size4096_radix4<>+0x7d30(SB)/8, $2479 // bitrev[4006] = 2479
DATA bitrev_size4096_radix4<>+0x7d38(SB)/8, $3503 // bitrev[4007] = 3503
DATA bitrev_size4096_radix4<>+0x7d40(SB)/8, $687  // bitrev[4008] = 687
DATA bitrev_size4096_radix4<>+0x7d48(SB)/8, $1711 // bitrev[4009] = 1711
DATA bitrev_size4096_radix4<>+0x7d50(SB)/8, $2735 // bitrev[4010] = 2735
DATA bitrev_size4096_radix4<>+0x7d58(SB)/8, $3759 // bitrev[4011] = 3759
DATA bitrev_size4096_radix4<>+0x7d60(SB)/8, $943  // bitrev[4012] = 943
DATA bitrev_size4096_radix4<>+0x7d68(SB)/8, $1967 // bitrev[4013] = 1967
DATA bitrev_size4096_radix4<>+0x7d70(SB)/8, $2991 // bitrev[4014] = 2991
DATA bitrev_size4096_radix4<>+0x7d78(SB)/8, $4015 // bitrev[4015] = 4015
DATA bitrev_size4096_radix4<>+0x7d80(SB)/8, $239  // bitrev[4016] = 239
DATA bitrev_size4096_radix4<>+0x7d88(SB)/8, $1263 // bitrev[4017] = 1263
DATA bitrev_size4096_radix4<>+0x7d90(SB)/8, $2287 // bitrev[4018] = 2287
DATA bitrev_size4096_radix4<>+0x7d98(SB)/8, $3311 // bitrev[4019] = 3311
DATA bitrev_size4096_radix4<>+0x7da0(SB)/8, $495  // bitrev[4020] = 495
DATA bitrev_size4096_radix4<>+0x7da8(SB)/8, $1519 // bitrev[4021] = 1519
DATA bitrev_size4096_radix4<>+0x7db0(SB)/8, $2543 // bitrev[4022] = 2543
DATA bitrev_size4096_radix4<>+0x7db8(SB)/8, $3567 // bitrev[4023] = 3567
DATA bitrev_size4096_radix4<>+0x7dc0(SB)/8, $751  // bitrev[4024] = 751
DATA bitrev_size4096_radix4<>+0x7dc8(SB)/8, $1775 // bitrev[4025] = 1775
DATA bitrev_size4096_radix4<>+0x7dd0(SB)/8, $2799 // bitrev[4026] = 2799
DATA bitrev_size4096_radix4<>+0x7dd8(SB)/8, $3823 // bitrev[4027] = 3823
DATA bitrev_size4096_radix4<>+0x7de0(SB)/8, $1007 // bitrev[4028] = 1007
DATA bitrev_size4096_radix4<>+0x7de8(SB)/8, $2031 // bitrev[4029] = 2031
DATA bitrev_size4096_radix4<>+0x7df0(SB)/8, $3055 // bitrev[4030] = 3055
DATA bitrev_size4096_radix4<>+0x7df8(SB)/8, $4079 // bitrev[4031] = 4079
DATA bitrev_size4096_radix4<>+0x7e00(SB)/8, $63   // bitrev[4032] = 63
DATA bitrev_size4096_radix4<>+0x7e08(SB)/8, $1087 // bitrev[4033] = 1087
DATA bitrev_size4096_radix4<>+0x7e10(SB)/8, $2111 // bitrev[4034] = 2111
DATA bitrev_size4096_radix4<>+0x7e18(SB)/8, $3135 // bitrev[4035] = 3135
DATA bitrev_size4096_radix4<>+0x7e20(SB)/8, $319  // bitrev[4036] = 319
DATA bitrev_size4096_radix4<>+0x7e28(SB)/8, $1343 // bitrev[4037] = 1343
DATA bitrev_size4096_radix4<>+0x7e30(SB)/8, $2367 // bitrev[4038] = 2367
DATA bitrev_size4096_radix4<>+0x7e38(SB)/8, $3391 // bitrev[4039] = 3391
DATA bitrev_size4096_radix4<>+0x7e40(SB)/8, $575  // bitrev[4040] = 575
DATA bitrev_size4096_radix4<>+0x7e48(SB)/8, $1599 // bitrev[4041] = 1599
DATA bitrev_size4096_radix4<>+0x7e50(SB)/8, $2623 // bitrev[4042] = 2623
DATA bitrev_size4096_radix4<>+0x7e58(SB)/8, $3647 // bitrev[4043] = 3647
DATA bitrev_size4096_radix4<>+0x7e60(SB)/8, $831  // bitrev[4044] = 831
DATA bitrev_size4096_radix4<>+0x7e68(SB)/8, $1855 // bitrev[4045] = 1855
DATA bitrev_size4096_radix4<>+0x7e70(SB)/8, $2879 // bitrev[4046] = 2879
DATA bitrev_size4096_radix4<>+0x7e78(SB)/8, $3903 // bitrev[4047] = 3903
DATA bitrev_size4096_radix4<>+0x7e80(SB)/8, $127  // bitrev[4048] = 127
DATA bitrev_size4096_radix4<>+0x7e88(SB)/8, $1151 // bitrev[4049] = 1151
DATA bitrev_size4096_radix4<>+0x7e90(SB)/8, $2175 // bitrev[4050] = 2175
DATA bitrev_size4096_radix4<>+0x7e98(SB)/8, $3199 // bitrev[4051] = 3199
DATA bitrev_size4096_radix4<>+0x7ea0(SB)/8, $383  // bitrev[4052] = 383
DATA bitrev_size4096_radix4<>+0x7ea8(SB)/8, $1407 // bitrev[4053] = 1407
DATA bitrev_size4096_radix4<>+0x7eb0(SB)/8, $2431 // bitrev[4054] = 2431
DATA bitrev_size4096_radix4<>+0x7eb8(SB)/8, $3455 // bitrev[4055] = 3455
DATA bitrev_size4096_radix4<>+0x7ec0(SB)/8, $639  // bitrev[4056] = 639
DATA bitrev_size4096_radix4<>+0x7ec8(SB)/8, $1663 // bitrev[4057] = 1663
DATA bitrev_size4096_radix4<>+0x7ed0(SB)/8, $2687 // bitrev[4058] = 2687
DATA bitrev_size4096_radix4<>+0x7ed8(SB)/8, $3711 // bitrev[4059] = 3711
DATA bitrev_size4096_radix4<>+0x7ee0(SB)/8, $895  // bitrev[4060] = 895
DATA bitrev_size4096_radix4<>+0x7ee8(SB)/8, $1919 // bitrev[4061] = 1919
DATA bitrev_size4096_radix4<>+0x7ef0(SB)/8, $2943 // bitrev[4062] = 2943
DATA bitrev_size4096_radix4<>+0x7ef8(SB)/8, $3967 // bitrev[4063] = 3967
DATA bitrev_size4096_radix4<>+0x7f00(SB)/8, $191  // bitrev[4064] = 191
DATA bitrev_size4096_radix4<>+0x7f08(SB)/8, $1215 // bitrev[4065] = 1215
DATA bitrev_size4096_radix4<>+0x7f10(SB)/8, $2239 // bitrev[4066] = 2239
DATA bitrev_size4096_radix4<>+0x7f18(SB)/8, $3263 // bitrev[4067] = 3263
DATA bitrev_size4096_radix4<>+0x7f20(SB)/8, $447  // bitrev[4068] = 447
DATA bitrev_size4096_radix4<>+0x7f28(SB)/8, $1471 // bitrev[4069] = 1471
DATA bitrev_size4096_radix4<>+0x7f30(SB)/8, $2495 // bitrev[4070] = 2495
DATA bitrev_size4096_radix4<>+0x7f38(SB)/8, $3519 // bitrev[4071] = 3519
DATA bitrev_size4096_radix4<>+0x7f40(SB)/8, $703  // bitrev[4072] = 703
DATA bitrev_size4096_radix4<>+0x7f48(SB)/8, $1727 // bitrev[4073] = 1727
DATA bitrev_size4096_radix4<>+0x7f50(SB)/8, $2751 // bitrev[4074] = 2751
DATA bitrev_size4096_radix4<>+0x7f58(SB)/8, $3775 // bitrev[4075] = 3775
DATA bitrev_size4096_radix4<>+0x7f60(SB)/8, $959  // bitrev[4076] = 959
DATA bitrev_size4096_radix4<>+0x7f68(SB)/8, $1983 // bitrev[4077] = 1983
DATA bitrev_size4096_radix4<>+0x7f70(SB)/8, $3007 // bitrev[4078] = 3007
DATA bitrev_size4096_radix4<>+0x7f78(SB)/8, $4031 // bitrev[4079] = 4031
DATA bitrev_size4096_radix4<>+0x7f80(SB)/8, $255  // bitrev[4080] = 255
DATA bitrev_size4096_radix4<>+0x7f88(SB)/8, $1279 // bitrev[4081] = 1279
DATA bitrev_size4096_radix4<>+0x7f90(SB)/8, $2303 // bitrev[4082] = 2303
DATA bitrev_size4096_radix4<>+0x7f98(SB)/8, $3327 // bitrev[4083] = 3327
DATA bitrev_size4096_radix4<>+0x7fa0(SB)/8, $511  // bitrev[4084] = 511
DATA bitrev_size4096_radix4<>+0x7fa8(SB)/8, $1535 // bitrev[4085] = 1535
DATA bitrev_size4096_radix4<>+0x7fb0(SB)/8, $2559 // bitrev[4086] = 2559
DATA bitrev_size4096_radix4<>+0x7fb8(SB)/8, $3583 // bitrev[4087] = 3583
DATA bitrev_size4096_radix4<>+0x7fc0(SB)/8, $767  // bitrev[4088] = 767
DATA bitrev_size4096_radix4<>+0x7fc8(SB)/8, $1791 // bitrev[4089] = 1791
DATA bitrev_size4096_radix4<>+0x7fd0(SB)/8, $2815 // bitrev[4090] = 2815
DATA bitrev_size4096_radix4<>+0x7fd8(SB)/8, $3839 // bitrev[4091] = 3839
DATA bitrev_size4096_radix4<>+0x7fe0(SB)/8, $1023 // bitrev[4092] = 1023
DATA bitrev_size4096_radix4<>+0x7fe8(SB)/8, $2047 // bitrev[4093] = 2047
DATA bitrev_size4096_radix4<>+0x7ff0(SB)/8, $3071 // bitrev[4094] = 3071
DATA bitrev_size4096_radix4<>+0x7ff8(SB)/8, $4095 // bitrev[4095] = 4095
GLOBL bitrev_size4096_radix4<>(SB), RODATA, $32768
