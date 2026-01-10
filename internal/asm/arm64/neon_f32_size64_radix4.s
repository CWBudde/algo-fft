//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-64 Radix-4 FFT Kernels for ARM64
// ===========================================================================
//
// Size 64 = 4^3, radix-4 algorithm uses 3 stages:
//   Stage 1: 16 butterflies, stride=4
//   Stage 2: 4 groups × 4 butterflies, twiddle step=4
//   Stage 3: 1 group × 16 butterflies, twiddle step=1
//
// ===========================================================================

#include "textflag.h"

DATA ·neonInv64+0(SB)/4, $0x3c800000 // 1/64
GLOBL ·neonInv64(SB), RODATA, $4

// Forward transform, size 64, complex64, radix-4 variant
TEXT ·ForwardNEONSize64Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $64, R13
	BNE  neon64r4_return_false

	MOVD dst+8(FP), R0
	CMP  $64, R0
	BLT  neon64r4_return_false

	MOVD twiddle+56(FP), R0
	CMP  $64, R0
	BLT  neon64r4_return_false

	MOVD scratch+80(FP), R0
	CMP  $64, R0
	BLT  neon64r4_return_false

	// Load static bit-reversal table
	MOVD $bitrev_size64_radix4<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon64r4_use_dst
	MOVD R11, R8

neon64r4_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon64r4_bitrev_loop:
	CMP  $64, R0
	BGE  neon64r4_stage1

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
	B    neon64r4_bitrev_loop

neon64r4_stage1:
	// Stage 1: 16 radix-4 butterflies
	MOVD $0, R0

neon64r4_stage1_loop:
	CMP  $64, R0
	BGE  neon64r4_stage2

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
	FMOVS F22, 8(R1)
	FMOVS F23, 12(R1)
	FMOVS F18, 16(R1)
	FMOVS F19, 20(R1)
	FMOVS F26, 24(R1)
	FMOVS F27, 28(R1)

	ADD  $4, R0, R0
	B    neon64r4_stage1_loop

neon64r4_stage2:
	// Stage 2: 4 groups × 4 butterflies, twiddle step=4
	MOVD $0, R0                 // base

neon64r4_stage2_outer:
	CMP  $64, R0
	BGE  neon64r4_stage3

	MOVD $0, R1                 // j

neon64r4_stage2_inner:
	CMP  $4, R1
	BGE  neon64r4_stage2_next

	// idx0=base+j, idx1=idx0+4, idx2=idx0+8, idx3=idx0+12
	ADD  R0, R1, R2             // idx0
	ADD  $4, R2, R3             // idx1
	ADD  $8, R2, R4             // idx2
	ADD  $12, R2, R5            // idx3

	// twiddle indices: j*4, j*8, j*12
	LSL  $2, R1, R6             // j*4
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F0
	FMOVS 4(R6), F1

	LSL  $3, R1, R7             // j*8
	LSL  $3, R7, R7
	ADD  R10, R7, R7
	FMOVS 0(R7), F2
	FMOVS 4(R7), F3

	LSL  $2, R1, R6             // j*4
	LSL  $3, R1, R7             // j*8
	ADD  R6, R7, R6             // j*12
	LSL  $3, R6, R6
	ADD  R10, R6, R6
	FMOVS 0(R6), F4
	FMOVS 4(R6), F5

	// Load a0..a3
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

	// Radix-4 butterfly
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

	// Store results
	LSL  $3, R2, R6
	ADD  R8, R6, R6
	FMOVS F22, 0(R6)
	FMOVS F23, 4(R6)

	LSL  $3, R3, R6
	ADD  R8, R6, R6
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	ADD  $1, R1, R1
	B    neon64r4_stage2_inner

neon64r4_stage2_next:
	ADD  $16, R0, R0
	B    neon64r4_stage2_outer

neon64r4_stage3:
	// Stage 3: 1 group × 16 butterflies, twiddle step=1
	MOVD $0, R0

neon64r4_stage3_loop:
	CMP  $16, R0
	BGE  neon64r4_done

	// idx0=j, idx1=j+16, idx2=j+32, idx3=j+48
	MOVD R0, R1
	ADD  $16, R1, R2
	ADD  $32, R1, R3
	ADD  $48, R1, R4

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
	FMOVS F28, 0(R5)
	FMOVS F29, 4(R5)

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS F24, 0(R5)
	FMOVS F25, 4(R5)

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS F20, 0(R5)
	FMOVS F21, 4(R5)

	ADD  $1, R0, R0
	B    neon64r4_stage3_loop

neon64r4_done:
	// Copy back if we used scratch
	CMP  R8, R20
	BEQ  neon64r4_return_true

	MOVD $0, R0
neon64r4_copy_loop:
	CMP  $64, R0
	BGE  neon64r4_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon64r4_copy_loop

neon64r4_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon64r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 64, complex64, radix-4 variant
TEXT ·InverseNEONSize64Radix4Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $64, R13
	BNE  neon64r4_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $64, R0
	BLT  neon64r4_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $64, R0
	BLT  neon64r4_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $64, R0
	BLT  neon64r4_inv_return_false

	// Load static bit-reversal table
	MOVD $bitrev_size64_radix4<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon64r4_inv_use_dst
	MOVD R11, R8

neon64r4_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon64r4_inv_bitrev_loop:
	CMP  $64, R0
	BGE  neon64r4_inv_stage1

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
	B    neon64r4_inv_bitrev_loop

neon64r4_inv_stage1:
	// Stage 1 (same as forward)
	MOVD $0, R0

neon64r4_inv_stage1_loop:
	CMP  $64, R0
	BGE  neon64r4_inv_stage2

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
	FMOVS F22, 8(R1)
	FMOVS F23, 12(R1)
	FMOVS F18, 16(R1)
	FMOVS F19, 20(R1)
	FMOVS F26, 24(R1)
	FMOVS F27, 28(R1)

	ADD  $4, R0, R0
	B    neon64r4_inv_stage1_loop

neon64r4_inv_stage2:
	// Stage 2 with conjugated twiddles
	MOVD $0, R0

neon64r4_inv_stage2_outer:
	CMP  $64, R0
	BGE  neon64r4_inv_stage3

	MOVD $0, R1

neon64r4_inv_stage2_inner:
	CMP  $4, R1
	BGE  neon64r4_inv_stage2_next

	ADD  R0, R1, R2
	ADD  $4, R2, R3
	ADD  $8, R2, R4
	ADD  $12, R2, R5

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
	FMOVS F28, 0(R6)
	FMOVS F29, 4(R6)

	LSL  $3, R4, R6
	ADD  R8, R6, R6
	FMOVS F24, 0(R6)
	FMOVS F25, 4(R6)

	LSL  $3, R5, R6
	ADD  R8, R6, R6
	FMOVS F20, 0(R6)
	FMOVS F21, 4(R6)

	ADD  $1, R1, R1
	B    neon64r4_inv_stage2_inner

neon64r4_inv_stage2_next:
	ADD  $16, R0, R0
	B    neon64r4_inv_stage2_outer

neon64r4_inv_stage3:
	// Stage 3 with conjugated twiddles
	MOVD $0, R0

neon64r4_inv_stage3_loop:
	CMP  $16, R0
	BGE  neon64r4_inv_done

	MOVD R0, R1
	ADD  $16, R1, R2
	ADD  $32, R1, R3
	ADD  $48, R1, R4

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
	FMOVS F28, 0(R5)
	FMOVS F29, 4(R5)

	LSL  $3, R3, R5
	ADD  R8, R5, R5
	FMOVS F24, 0(R5)
	FMOVS F25, 4(R5)

	LSL  $3, R4, R5
	ADD  R8, R5, R5
	FMOVS F20, 0(R5)
	FMOVS F21, 4(R5)

	ADD  $1, R0, R0
	B    neon64r4_inv_stage3_loop

neon64r4_inv_done:
	// Copy back if we used scratch
	CMP  R8, R20
	BEQ  neon64r4_inv_scale

	MOVD $0, R0
neon64r4_inv_copy_loop:
	CMP  $64, R0
	BGE  neon64r4_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon64r4_inv_copy_loop

neon64r4_inv_scale:
	MOVD $·neonInv64(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon64r4_inv_scale_loop:
	CMP  $64, R0
	BGE  neon64r4_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon64r4_inv_scale_loop

neon64r4_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon64r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Bit-reversal lookup table for size-64 radix-4
// Pattern: [0 16 32 48 4 20 36 52 8 24 40 56 12 28 44 60 1 17 33 49 5 21 37 53 9 25 41 57 13 29 45 61 2 18 34 50 6 22 38 54 10 26 42 58 14 30 46 62 3 19 35 51 7 23 39 55 11 27 43 59 15 31 47 63]
DATA bitrev_size64_radix4<>+0x000(SB)/8, $0   // bitrev[0] = 0
DATA bitrev_size64_radix4<>+0x008(SB)/8, $16  // bitrev[1] = 16
DATA bitrev_size64_radix4<>+0x010(SB)/8, $32  // bitrev[2] = 32
DATA bitrev_size64_radix4<>+0x018(SB)/8, $48  // bitrev[3] = 48
DATA bitrev_size64_radix4<>+0x020(SB)/8, $4   // bitrev[4] = 4
DATA bitrev_size64_radix4<>+0x028(SB)/8, $20  // bitrev[5] = 20
DATA bitrev_size64_radix4<>+0x030(SB)/8, $36  // bitrev[6] = 36
DATA bitrev_size64_radix4<>+0x038(SB)/8, $52  // bitrev[7] = 52
DATA bitrev_size64_radix4<>+0x040(SB)/8, $8   // bitrev[8] = 8
DATA bitrev_size64_radix4<>+0x048(SB)/8, $24  // bitrev[9] = 24
DATA bitrev_size64_radix4<>+0x050(SB)/8, $40  // bitrev[10] = 40
DATA bitrev_size64_radix4<>+0x058(SB)/8, $56  // bitrev[11] = 56
DATA bitrev_size64_radix4<>+0x060(SB)/8, $12  // bitrev[12] = 12
DATA bitrev_size64_radix4<>+0x068(SB)/8, $28  // bitrev[13] = 28
DATA bitrev_size64_radix4<>+0x070(SB)/8, $44  // bitrev[14] = 44
DATA bitrev_size64_radix4<>+0x078(SB)/8, $60  // bitrev[15] = 60
DATA bitrev_size64_radix4<>+0x080(SB)/8, $1   // bitrev[16] = 1
DATA bitrev_size64_radix4<>+0x088(SB)/8, $17  // bitrev[17] = 17
DATA bitrev_size64_radix4<>+0x090(SB)/8, $33  // bitrev[18] = 33
DATA bitrev_size64_radix4<>+0x098(SB)/8, $49  // bitrev[19] = 49
DATA bitrev_size64_radix4<>+0x0a0(SB)/8, $5   // bitrev[20] = 5
DATA bitrev_size64_radix4<>+0x0a8(SB)/8, $21  // bitrev[21] = 21
DATA bitrev_size64_radix4<>+0x0b0(SB)/8, $37  // bitrev[22] = 37
DATA bitrev_size64_radix4<>+0x0b8(SB)/8, $53  // bitrev[23] = 53
DATA bitrev_size64_radix4<>+0x0c0(SB)/8, $9   // bitrev[24] = 9
DATA bitrev_size64_radix4<>+0x0c8(SB)/8, $25  // bitrev[25] = 25
DATA bitrev_size64_radix4<>+0x0d0(SB)/8, $41  // bitrev[26] = 41
DATA bitrev_size64_radix4<>+0x0d8(SB)/8, $57  // bitrev[27] = 57
DATA bitrev_size64_radix4<>+0x0e0(SB)/8, $13  // bitrev[28] = 13
DATA bitrev_size64_radix4<>+0x0e8(SB)/8, $29  // bitrev[29] = 29
DATA bitrev_size64_radix4<>+0x0f0(SB)/8, $45  // bitrev[30] = 45
DATA bitrev_size64_radix4<>+0x0f8(SB)/8, $61  // bitrev[31] = 61
DATA bitrev_size64_radix4<>+0x100(SB)/8, $2   // bitrev[32] = 2
DATA bitrev_size64_radix4<>+0x108(SB)/8, $18  // bitrev[33] = 18
DATA bitrev_size64_radix4<>+0x110(SB)/8, $34  // bitrev[34] = 34
DATA bitrev_size64_radix4<>+0x118(SB)/8, $50  // bitrev[35] = 50
DATA bitrev_size64_radix4<>+0x120(SB)/8, $6   // bitrev[36] = 6
DATA bitrev_size64_radix4<>+0x128(SB)/8, $22  // bitrev[37] = 22
DATA bitrev_size64_radix4<>+0x130(SB)/8, $38  // bitrev[38] = 38
DATA bitrev_size64_radix4<>+0x138(SB)/8, $54  // bitrev[39] = 54
DATA bitrev_size64_radix4<>+0x140(SB)/8, $10  // bitrev[40] = 10
DATA bitrev_size64_radix4<>+0x148(SB)/8, $26  // bitrev[41] = 26
DATA bitrev_size64_radix4<>+0x150(SB)/8, $42  // bitrev[42] = 42
DATA bitrev_size64_radix4<>+0x158(SB)/8, $58  // bitrev[43] = 58
DATA bitrev_size64_radix4<>+0x160(SB)/8, $14  // bitrev[44] = 14
DATA bitrev_size64_radix4<>+0x168(SB)/8, $30  // bitrev[45] = 30
DATA bitrev_size64_radix4<>+0x170(SB)/8, $46  // bitrev[46] = 46
DATA bitrev_size64_radix4<>+0x178(SB)/8, $62  // bitrev[47] = 62
DATA bitrev_size64_radix4<>+0x180(SB)/8, $3   // bitrev[48] = 3
DATA bitrev_size64_radix4<>+0x188(SB)/8, $19  // bitrev[49] = 19
DATA bitrev_size64_radix4<>+0x190(SB)/8, $35  // bitrev[50] = 35
DATA bitrev_size64_radix4<>+0x198(SB)/8, $51  // bitrev[51] = 51
DATA bitrev_size64_radix4<>+0x1a0(SB)/8, $7   // bitrev[52] = 7
DATA bitrev_size64_radix4<>+0x1a8(SB)/8, $23  // bitrev[53] = 23
DATA bitrev_size64_radix4<>+0x1b0(SB)/8, $39  // bitrev[54] = 39
DATA bitrev_size64_radix4<>+0x1b8(SB)/8, $55  // bitrev[55] = 55
DATA bitrev_size64_radix4<>+0x1c0(SB)/8, $11  // bitrev[56] = 11
DATA bitrev_size64_radix4<>+0x1c8(SB)/8, $27  // bitrev[57] = 27
DATA bitrev_size64_radix4<>+0x1d0(SB)/8, $43  // bitrev[58] = 43
DATA bitrev_size64_radix4<>+0x1d8(SB)/8, $59  // bitrev[59] = 59
DATA bitrev_size64_radix4<>+0x1e0(SB)/8, $15  // bitrev[60] = 15
DATA bitrev_size64_radix4<>+0x1e8(SB)/8, $31  // bitrev[61] = 31
DATA bitrev_size64_radix4<>+0x1f0(SB)/8, $47  // bitrev[62] = 47
DATA bitrev_size64_radix4<>+0x1f8(SB)/8, $63  // bitrev[63] = 63

GLOBL bitrev_size64_radix4<>(SB), RODATA, $512
