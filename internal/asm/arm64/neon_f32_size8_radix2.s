//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Size-8 Radix-2 FFT Kernels for ARM64
// ===========================================================================

#include "textflag.h"

// Forward transform, size 8, complex64, radix-2
TEXT ·ForwardNEONSize8Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $8, R13
	BNE  neon8r2_return_false

	MOVD dst+8(FP), R0
	CMP  $8, R0
	BLT  neon8r2_return_false

	MOVD twiddle+56(FP), R0
	CMP  $8, R0
	BLT  neon8r2_return_false

	MOVD scratch+80(FP), R0
	CMP  $8, R0
	BLT  neon8r2_return_false

	// Load static bit-reversal indices
	MOVD $bitrev_size8_radix2<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon8r2_use_dst
	MOVD R11, R8

neon8r2_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon8r2_bitrev_loop:
	CMP  $8, R0
	BGE  neon8r2_stage1

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
	B    neon8r2_bitrev_loop

neon8r2_stage1:
	// size=2, step=4
	MOVD $0, R0

neon8r2_stage1_loop:
	CMP  $8, R0
	BGE  neon8r2_stage2

	LSL  $3, R0, R1
	ADD  R8, R1, R1
	FMOVS 0(R1), F0
	FMOVS 4(R1), F1
	FMOVS 8(R1), F2
	FMOVS 12(R1), F3

	FADDS F2, F0, F4
	FADDS F3, F1, F5
	FSUBS F2, F0, F6
	FSUBS F3, F1, F7

	FMOVS F4, 0(R1)
	FMOVS F5, 4(R1)
	FMOVS F6, 8(R1)
	FMOVS F7, 12(R1)

	ADD  $2, R0, R0
	B    neon8r2_stage1_loop

neon8r2_stage2:
	// size=4, step=2
	MOVD $0, R0                 // base

neon8r2_stage2_outer:
	CMP  $8, R0
	BGE  neon8r2_stage3

	MOVD $0, R1                 // j

neon8r2_stage2_inner:
	CMP  $2, R1
	BGE  neon8r2_stage2_next

	ADD  R0, R1, R2             // idx0
	ADD  $2, R2, R3             // idx1

	// twiddle index = j*2
	LSL  $1, R1, R4
	LSL  $3, R4, R4
	ADD  R10, R4, R4
	FMOVS 0(R4), F0
	FMOVS 4(R4), F1

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F2
	FMOVS 4(R4), F3

	LSL  $3, R3, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F4
	FMOVS 4(R4), F5

	// wb = w * b
	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	FADDS F6, F2, F9
	FADDS F7, F3, F10
	FSUBS F6, F2, F11
	FSUBS F7, F3, F12

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS F9, 0(R4)
	FMOVS F10, 4(R4)

	LSL  $3, R3, R4
	ADD  R8, R4, R4
	FMOVS F11, 0(R4)
	FMOVS F12, 4(R4)

	ADD  $1, R1, R1
	B    neon8r2_stage2_inner

neon8r2_stage2_next:
	ADD  $4, R0, R0
	B    neon8r2_stage2_outer

neon8r2_stage3:
	// size=8, step=1
	MOVD $0, R0

neon8r2_stage3_loop:
	CMP  $4, R0
	BGE  neon8r2_done

	MOVD R0, R1                 // idx0
	ADD  $4, R1, R2             // idx1

	LSL  $3, R0, R3
	ADD  R10, R3, R3
	FMOVS 0(R3), F0
	FMOVS 4(R3), F1

	LSL  $3, R1, R3
	ADD  R8, R3, R3
	FMOVS 0(R3), F2
	FMOVS 4(R3), F3

	LSL  $3, R2, R3
	ADD  R8, R3, R3
	FMOVS 0(R3), F4
	FMOVS 4(R3), F5

	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	FADDS F6, F2, F9
	FADDS F7, F3, F10
	FSUBS F6, F2, F11
	FSUBS F7, F3, F12

	LSL  $3, R1, R3
	ADD  R8, R3, R3
	FMOVS F9, 0(R3)
	FMOVS F10, 4(R3)

	LSL  $3, R2, R3
	ADD  R8, R3, R3
	FMOVS F11, 0(R3)
	FMOVS F12, 4(R3)

	ADD  $1, R0, R0
	B    neon8r2_stage3_loop

neon8r2_done:
	CMP  R8, R20
	BEQ  neon8r2_return_true

	MOVD $0, R0
neon8r2_copy_loop:
	CMP  $8, R0
	BGE  neon8r2_return_true
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon8r2_copy_loop

neon8r2_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r2_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Inverse transform, size 8, complex64, radix-2
TEXT ·InverseNEONSize8Radix2Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src+32(FP), R13

	CMP  $8, R13
	BNE  neon8r2_inv_return_false

	MOVD dst+8(FP), R0
	CMP  $8, R0
	BLT  neon8r2_inv_return_false

	MOVD twiddle+56(FP), R0
	CMP  $8, R0
	BLT  neon8r2_inv_return_false

	MOVD scratch+80(FP), R0
	CMP  $8, R0
	BLT  neon8r2_inv_return_false

	// Load static bit-reversal indices
	MOVD $bitrev_size8_radix2<>(SB), R12

	MOVD R8, R20
	CMP  R8, R9
	BNE  neon8r2_inv_use_dst
	MOVD R11, R8

neon8r2_inv_use_dst:
	// Bit-reversal permutation
	MOVD $0, R0

neon8r2_inv_bitrev_loop:
	CMP  $8, R0
	BGE  neon8r2_inv_stage1

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
	B    neon8r2_inv_bitrev_loop

neon8r2_inv_stage1:
	// size=2, step=4
	MOVD $0, R0

neon8r2_inv_stage1_loop:
	CMP  $8, R0
	BGE  neon8r2_inv_stage2

	LSL  $3, R0, R1
	ADD  R8, R1, R1
	FMOVS 0(R1), F0
	FMOVS 4(R1), F1
	FMOVS 8(R1), F2
	FMOVS 12(R1), F3

	FADDS F2, F0, F4
	FADDS F3, F1, F5
	FSUBS F2, F0, F6
	FSUBS F3, F1, F7

	FMOVS F4, 0(R1)
	FMOVS F5, 4(R1)
	FMOVS F6, 8(R1)
	FMOVS F7, 12(R1)

	ADD  $2, R0, R0
	B    neon8r2_inv_stage1_loop

neon8r2_inv_stage2:
	// size=4, step=2 (conjugated twiddles)
	MOVD $0, R0

neon8r2_inv_stage2_outer:
	CMP  $8, R0
	BGE  neon8r2_inv_stage3

	MOVD $0, R1

neon8r2_inv_stage2_inner:
	CMP  $2, R1
	BGE  neon8r2_inv_stage2_next

	ADD  R0, R1, R2
	ADD  $2, R2, R3

	LSL  $1, R1, R4
	LSL  $3, R4, R4
	ADD  R10, R4, R4
	FMOVS 0(R4), F0
	FMOVS 4(R4), F1
	FNEGS  F1, F1

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F2
	FMOVS 4(R4), F3

	LSL  $3, R3, R4
	ADD  R8, R4, R4
	FMOVS 0(R4), F4
	FMOVS 4(R4), F5

	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	FADDS F6, F2, F9
	FADDS F7, F3, F10
	FSUBS F6, F2, F11
	FSUBS F7, F3, F12

	LSL  $3, R2, R4
	ADD  R8, R4, R4
	FMOVS F9, 0(R4)
	FMOVS F10, 4(R4)

	LSL  $3, R3, R4
	ADD  R8, R4, R4
	FMOVS F11, 0(R4)
	FMOVS F12, 4(R4)

	ADD  $1, R1, R1
	B    neon8r2_inv_stage2_inner

neon8r2_inv_stage2_next:
	ADD  $4, R0, R0
	B    neon8r2_inv_stage2_outer

neon8r2_inv_stage3:
	// size=8, step=1 (conjugated twiddles)
	MOVD $0, R0

neon8r2_inv_stage3_loop:
	CMP  $4, R0
	BGE  neon8r2_inv_done

	MOVD R0, R1
	ADD  $4, R1, R2

	LSL  $3, R0, R3
	ADD  R10, R3, R3
	FMOVS 0(R3), F0
	FMOVS 4(R3), F1
	FNEGS  F1, F1

	LSL  $3, R1, R3
	ADD  R8, R3, R3
	FMOVS 0(R3), F2
	FMOVS 4(R3), F3

	LSL  $3, R2, R3
	ADD  R8, R3, R3
	FMOVS 0(R3), F4
	FMOVS 4(R3), F5

	FMULS F0, F4, F6
	FMULS F1, F5, F7
	FSUBS F7, F6, F6
	FMULS F0, F5, F7
	FMULS F1, F4, F8
	FADDS F8, F7, F7

	FADDS F6, F2, F9
	FADDS F7, F3, F10
	FSUBS F6, F2, F11
	FSUBS F7, F3, F12

	LSL  $3, R1, R3
	ADD  R8, R3, R3
	FMOVS F9, 0(R3)
	FMOVS F10, 4(R3)

	LSL  $3, R2, R3
	ADD  R8, R3, R3
	FMOVS F11, 0(R3)
	FMOVS F12, 4(R3)

	ADD  $1, R0, R0
	B    neon8r2_inv_stage3_loop

neon8r2_inv_done:
	CMP  R8, R20
	BEQ  neon8r2_inv_scale

	MOVD $0, R0
neon8r2_inv_copy_loop:
	CMP  $8, R0
	BGE  neon8r2_inv_scale
	LSL  $3, R0, R1
	ADD  R8, R1, R1
	MOVD (R1), R2
	ADD  R20, R1, R3
	MOVD R2, (R3)
	ADD  $1, R0, R0
	B    neon8r2_inv_copy_loop

neon8r2_inv_scale:
	MOVD $·neonInv8(SB), R1
	FMOVS (R1), F0
	MOVD $0, R0

neon8r2_inv_scale_loop:
	CMP  $8, R0
	BGE  neon8r2_inv_return_true
	LSL  $3, R0, R1
	ADD  R20, R1, R1
	FMOVS 0(R1), F2
	FMOVS 4(R1), F3
	FMULS F0, F2, F2
	FMULS F0, F3, F3
	FMOVS F2, 0(R1)
	FMOVS F3, 4(R1)
	ADD  $1, R0, R0
	B    neon8r2_inv_scale_loop

neon8r2_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r2_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Data Section: Bit-Reversal Indices for Size-8 Radix-2
// ===========================================================================

// Bit-reversal permutation for size 8: [0, 4, 2, 6, 1, 5, 3, 7]
DATA bitrev_size8_radix2<>+0x00(SB)/8, $0  // bitrev[0] = 0
DATA bitrev_size8_radix2<>+0x08(SB)/8, $4  // bitrev[1] = 4
DATA bitrev_size8_radix2<>+0x10(SB)/8, $2  // bitrev[2] = 2
DATA bitrev_size8_radix2<>+0x18(SB)/8, $6  // bitrev[3] = 6
DATA bitrev_size8_radix2<>+0x20(SB)/8, $1  // bitrev[4] = 1
DATA bitrev_size8_radix2<>+0x28(SB)/8, $5  // bitrev[5] = 5
DATA bitrev_size8_radix2<>+0x30(SB)/8, $3  // bitrev[6] = 3
DATA bitrev_size8_radix2<>+0x38(SB)/8, $7  // bitrev[7] = 7
GLOBL bitrev_size8_radix2<>(SB), RODATA, $64
