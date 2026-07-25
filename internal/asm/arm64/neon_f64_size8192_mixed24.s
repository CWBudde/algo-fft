//go:build arm64 && !purego

// ===========================================================================
// NEON Size-8192 Mixed-Radix (Radix-4 + Radix-2) FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// Size 8192 = 4 * 4 * 4 * 4 * 4 * 4 * 2, mixed-radix algorithm:
//   Stage 1: 2048 radix-4 butterflies (no twiddles), stride=4
//   Stage 2: radix-4 with twiddles, 512 groups, step=16
//   Stage 3: radix-4 with twiddles, 128 groups, step=64
//   Stage 4: radix-4 with twiddles, 32 groups, step=256
//   Stage 5: radix-4 with twiddles, 8 groups, step=1024
//   Stage 6: radix-4 with twiddles, 2 groups, step=4096
//   Stage 7: radix-2 with twiddles, size=8192, step=1
//
// Each complex128 element is 16 bytes (real f64 + imag f64).
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 8192, complex128, mixed radix
// func ForwardNEONSize8192Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·ForwardNEONSize8192Radix4Then2Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $8192, R13
	BNE  neon8192m24f64_return_false

	MOVD dst_len+8(FP), R0
	CMP  $8192, R0
	BLT  neon8192m24f64_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $8192, R0
	BLT  neon8192m24f64_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $8192, R0
	BLT  neon8192m24f64_return_false

	MOVD $bitrev_size8192_mixed24_f64<>(SB), R12
	MOVD R8, R20

	CMP  R8, R9
	BNE  neon8192m24f64_use_dst
	MOVD R11, R8

neon8192m24f64_use_dst:
	// =========================================================================
	// Bit-reversal permutation
	// =========================================================================
	MOVD $0, R0

neon8192m24f64_bitrev_loop:
	CMP  $8192, R0
	BGE  neon8192m24f64_stage1

	LSL  $3, R0, R1
	ADD  R12, R1, R1
	MOVD (R1), R2

	LSL  $4, R2, R3
	ADD  R9, R3, R3
	MOVD (R3), R4
	MOVD 8(R3), R5

	LSL  $4, R0, R3
	ADD  R8, R3, R3
	MOVD R4, (R3)
	MOVD R5, 8(R3)

	ADD  $1, R0, R0
	B    neon8192m24f64_bitrev_loop

neon8192m24f64_stage1:
	// =========================================================================
	// Stage 1: 2048 radix-4 butterflies (no twiddles)
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_stage1_loop:
	CMP  $8192, R14
	BGE  neon8192m24f64_stage2

	LSL  $4, R14, R1
	ADD  R8, R1, R1

	FMOVD 0(R1), F0
	FMOVD 8(R1), F1
	FMOVD 16(R1), F2
	FMOVD 24(R1), F3
	FMOVD 32(R1), F4
	FMOVD 40(R1), F5
	FMOVD 48(R1), F6
	FMOVD 56(R1), F7

	FADDD F4, F0, F8
	FADDD F5, F1, F9
	FSUBD F4, F0, F10
	FSUBD F5, F1, F11

	FADDD F6, F2, F12
	FADDD F7, F3, F13
	FSUBD F6, F2, F14
	FSUBD F7, F3, F15

	FADDD F12, F8, F16
	FADDD F13, F9, F17
	FSUBD F12, F8, F18
	FSUBD F13, F9, F19

	FMOVD F15, F20
	FNEGD F14, F21

	FADDD F20, F10, F22
	FADDD F21, F11, F23

	FNEGD F15, F24
	FMOVD F14, F25

	FADDD F24, F10, F26
	FADDD F25, F11, F27

	FMOVD F16, 0(R1)
	FMOVD F17, 8(R1)
	FMOVD F22, 16(R1)
	FMOVD F23, 24(R1)
	FMOVD F18, 32(R1)
	FMOVD F19, 40(R1)
	FMOVD F26, 48(R1)
	FMOVD F27, 56(R1)

	ADD  $4, R14, R14
	B    neon8192m24f64_stage1_loop

neon8192m24f64_stage2:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=16, blocks=512, butterflies/block=4
	// Twiddle indices: j*512, j*1024, j*1536
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_stage2_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_stage3

	MOVD $0, R15

neon8192m24f64_stage2_j:
	CMP  $4, R15
	BGE  neon8192m24f64_stage2_next

	ADD  R14, R15, R0
	ADD  $4, R0, R1
	ADD  $8, R0, R2
	ADD  $12, R0, R3

	// Twiddles: w1=tw[j*512], w2=tw[j*1024], w3=tw[j*1536]
	LSL  $9, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $10, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $1536, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FADDD F21, F16, F26
	FSUBD F20, F17, F27

	FSUBD F21, F16, F28
	FADDD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_stage2_j

neon8192m24f64_stage2_next:
	ADD  $16, R14, R14
	B    neon8192m24f64_stage2_base

neon8192m24f64_stage3:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=64, blocks=128, butterflies/block=16
	// Twiddle indices: j*128, j*256, j*384
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_stage3_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_stage4

	MOVD $0, R15

neon8192m24f64_stage3_j:
	CMP  $16, R15
	BGE  neon8192m24f64_stage3_next

	ADD  R14, R15, R0
	ADD  $16, R0, R1
	ADD  $32, R0, R2
	ADD  $48, R0, R3

	// Twiddles: w1=tw[j*128], w2=tw[j*256], w3=tw[j*384]
	LSL  $7, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $8, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $384, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FADDD F21, F16, F26
	FSUBD F20, F17, F27

	FSUBD F21, F16, F28
	FADDD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_stage3_j

neon8192m24f64_stage3_next:
	ADD  $64, R14, R14
	B    neon8192m24f64_stage3_base

neon8192m24f64_stage4:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=256, blocks=32, butterflies/block=64
	// Twiddle indices: j*32, j*64, j*96
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_stage4_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_stage5

	MOVD $0, R15

neon8192m24f64_stage4_j:
	CMP  $64, R15
	BGE  neon8192m24f64_stage4_next

	ADD  R14, R15, R0
	ADD  $64, R0, R1
	ADD  $128, R0, R2
	ADD  $192, R0, R3

	// Twiddles: w1=tw[j*32], w2=tw[j*64], w3=tw[j*96]
	LSL  $5, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $6, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $96, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FADDD F21, F16, F26
	FSUBD F20, F17, F27

	FSUBD F21, F16, F28
	FADDD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_stage4_j

neon8192m24f64_stage4_next:
	ADD  $256, R14, R14
	B    neon8192m24f64_stage4_base

neon8192m24f64_stage5:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=1024, blocks=8, butterflies/block=256
	// Twiddle indices: j*8, j*16, j*24
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_stage5_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_stage6

	MOVD $0, R15

neon8192m24f64_stage5_j:
	CMP  $256, R15
	BGE  neon8192m24f64_stage5_next

	ADD  R14, R15, R0
	ADD  $256, R0, R1
	ADD  $512, R0, R2
	ADD  $768, R0, R3

	// Twiddles: w1=tw[j*8], w2=tw[j*16], w3=tw[j*24]
	LSL  $3, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $4, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $24, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FADDD F21, F16, F26
	FSUBD F20, F17, F27

	FSUBD F21, F16, F28
	FADDD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_stage5_j

neon8192m24f64_stage5_next:
	ADD  $1024, R14, R14
	B    neon8192m24f64_stage5_base

neon8192m24f64_stage6:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=4096, blocks=2, butterflies/block=1024
	// Twiddle indices: j*2, j*4, j*6
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_stage6_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_stage7

	MOVD $0, R15

neon8192m24f64_stage6_j:
	CMP  $1024, R15
	BGE  neon8192m24f64_stage6_next

	ADD  R14, R15, R0
	ADD  $1024, R0, R1
	ADD  $2048, R0, R2
	ADD  $3072, R0, R3

	// Twiddles: w1=tw[j*2], w2=tw[j*4], w3=tw[j*6]
	LSL  $1, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1

	LSL  $2, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3

	MOVD $6, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FADDD F21, F16, F26
	FSUBD F20, F17, F27

	FSUBD F21, F16, F28
	FADDD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_stage6_j

neon8192m24f64_stage6_next:
	ADD  $4096, R14, R14
	B    neon8192m24f64_stage6_base

neon8192m24f64_stage7:
	// =========================================================================
	// Final stage: radix-2 with twiddles, size=8192, step=1
	// =========================================================================
	MOVD $0, R0

neon8192m24f64_stage7_loop:
	CMP  $4096, R0
	BGE  neon8192m24f64_done

	ADD  $4096, R0, R1

	// Load twiddle[j]
	LSL  $4, R0, R2
	ADD  R10, R2, R2
	FMOVD 0(R2), F0
	FMOVD 8(R2), F1

	// Load a, b
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F2
	FMOVD 8(R2), F3

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F4
	FMOVD 8(R2), F5

	// wb = w * b
	FMULD F0, F4, F6
	FMULD F1, F5, F7
	FSUBD F7, F6, F6
	FMULD F0, F5, F7
	FMULD F1, F4, F8
	FADDD F8, F7, F7

	// Butterfly
	FADDD F6, F2, F8
	FADDD F7, F3, F9
	FSUBD F6, F2, F10
	FSUBD F7, F3, F11

	// Store
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD F8, 0(R2)
	FMOVD F9, 8(R2)

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD F10, 0(R2)
	FMOVD F11, 8(R2)

	ADD  $1, R0, R0
	B    neon8192m24f64_stage7_loop

neon8192m24f64_done:
	CMP  R8, R20
	BEQ  neon8192m24f64_return_true

	MOVD $0, R0
neon8192m24f64_copy_loop:
	CMP  $8192, R0
	BGE  neon8192m24f64_return_true
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon8192m24f64_copy_loop

neon8192m24f64_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8192m24f64_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform
// ===========================================================================
// func InverseNEONSize8192Radix4Then2Complex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·InverseNEONSize8192Radix4Then2Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD scratch+72(FP), R11
	MOVD src_len+32(FP), R13

	CMP  $8192, R13
	BNE  neon8192m24f64_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $8192, R0
	BLT  neon8192m24f64_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $8192, R0
	BLT  neon8192m24f64_inv_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $8192, R0
	BLT  neon8192m24f64_inv_return_false

	MOVD $bitrev_size8192_mixed24_f64<>(SB), R12
	MOVD R8, R20

	CMP  R8, R9
	BNE  neon8192m24f64_inv_use_dst
	MOVD R11, R8

neon8192m24f64_inv_use_dst:
	// =========================================================================
	// Bit-reversal permutation
	// =========================================================================
	MOVD $0, R0

neon8192m24f64_inv_bitrev_loop:
	CMP  $8192, R0
	BGE  neon8192m24f64_inv_stage1

	LSL  $3, R0, R1
	ADD  R12, R1, R1
	MOVD (R1), R2

	LSL  $4, R2, R3
	ADD  R9, R3, R3
	MOVD (R3), R4
	MOVD 8(R3), R5

	LSL  $4, R0, R3
	ADD  R8, R3, R3
	MOVD R4, (R3)
	MOVD R5, 8(R3)

	ADD  $1, R0, R0
	B    neon8192m24f64_inv_bitrev_loop

neon8192m24f64_inv_stage1:
	// =========================================================================
	// Stage 1: 2048 radix-4 butterflies (no twiddles)
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_inv_stage1_loop:
	CMP  $8192, R14
	BGE  neon8192m24f64_inv_stage2

	LSL  $4, R14, R1
	ADD  R8, R1, R1

	FMOVD 0(R1), F0
	FMOVD 8(R1), F1
	FMOVD 16(R1), F2
	FMOVD 24(R1), F3
	FMOVD 32(R1), F4
	FMOVD 40(R1), F5
	FMOVD 48(R1), F6
	FMOVD 56(R1), F7

	FADDD F4, F0, F8
	FADDD F5, F1, F9
	FSUBD F4, F0, F10
	FSUBD F5, F1, F11

	FADDD F6, F2, F12
	FADDD F7, F3, F13
	FSUBD F6, F2, F14
	FSUBD F7, F3, F15

	FADDD F12, F8, F16
	FADDD F13, F9, F17
	FSUBD F12, F8, F18
	FSUBD F13, F9, F19

	// For inverse: i * t3
	FNEGD F15, F20
	FMOVD F14, F21

	FADDD F20, F10, F22
	FADDD F21, F11, F23

	// For inverse: (-i) * t3
	FMOVD F15, F24
	FNEGD F14, F25

	FADDD F24, F10, F26
	FADDD F25, F11, F27

	FMOVD F16, 0(R1)
	FMOVD F17, 8(R1)
	FMOVD F22, 16(R1)
	FMOVD F23, 24(R1)
	FMOVD F18, 32(R1)
	FMOVD F19, 40(R1)
	FMOVD F26, 48(R1)
	FMOVD F27, 56(R1)

	ADD  $4, R14, R14
	B    neon8192m24f64_inv_stage1_loop

neon8192m24f64_inv_stage2:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=16, blocks=512, butterflies/block=4
	// Twiddle indices: j*512, j*1024, j*1536
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_inv_stage2_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_inv_stage3

	MOVD $0, R15

neon8192m24f64_inv_stage2_j:
	CMP  $4, R15
	BGE  neon8192m24f64_inv_stage2_next

	ADD  R14, R15, R0
	ADD  $4, R0, R1
	ADD  $8, R0, R2
	ADD  $12, R0, R3

	// Twiddles: w1=tw[j*512], w2=tw[j*1024], w3=tw[j*1536]
	LSL  $9, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $10, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $1536, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_inv_stage2_j

neon8192m24f64_inv_stage2_next:
	ADD  $16, R14, R14
	B    neon8192m24f64_inv_stage2_base

neon8192m24f64_inv_stage3:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=64, blocks=128, butterflies/block=16
	// Twiddle indices: j*128, j*256, j*384
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_inv_stage3_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_inv_stage4

	MOVD $0, R15

neon8192m24f64_inv_stage3_j:
	CMP  $16, R15
	BGE  neon8192m24f64_inv_stage3_next

	ADD  R14, R15, R0
	ADD  $16, R0, R1
	ADD  $32, R0, R2
	ADD  $48, R0, R3

	// Twiddles: w1=tw[j*128], w2=tw[j*256], w3=tw[j*384]
	LSL  $7, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $8, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $384, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_inv_stage3_j

neon8192m24f64_inv_stage3_next:
	ADD  $64, R14, R14
	B    neon8192m24f64_inv_stage3_base

neon8192m24f64_inv_stage4:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=256, blocks=32, butterflies/block=64
	// Twiddle indices: j*32, j*64, j*96
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_inv_stage4_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_inv_stage5

	MOVD $0, R15

neon8192m24f64_inv_stage4_j:
	CMP  $64, R15
	BGE  neon8192m24f64_inv_stage4_next

	ADD  R14, R15, R0
	ADD  $64, R0, R1
	ADD  $128, R0, R2
	ADD  $192, R0, R3

	// Twiddles: w1=tw[j*32], w2=tw[j*64], w3=tw[j*96]
	LSL  $5, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $6, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $96, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_inv_stage4_j

neon8192m24f64_inv_stage4_next:
	ADD  $256, R14, R14
	B    neon8192m24f64_inv_stage4_base

neon8192m24f64_inv_stage5:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=1024, blocks=8, butterflies/block=256
	// Twiddle indices: j*8, j*16, j*24
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_inv_stage5_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_inv_stage6

	MOVD $0, R15

neon8192m24f64_inv_stage5_j:
	CMP  $256, R15
	BGE  neon8192m24f64_inv_stage5_next

	ADD  R14, R15, R0
	ADD  $256, R0, R1
	ADD  $512, R0, R2
	ADD  $768, R0, R3

	// Twiddles: w1=tw[j*8], w2=tw[j*16], w3=tw[j*24]
	LSL  $3, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $4, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $24, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_inv_stage5_j

neon8192m24f64_inv_stage5_next:
	ADD  $1024, R14, R14
	B    neon8192m24f64_inv_stage5_base

neon8192m24f64_inv_stage6:
	// =========================================================================
	// Stage: radix-4 with twiddles, step=4096, blocks=2, butterflies/block=1024
	// Twiddle indices: j*2, j*4, j*6
	// =========================================================================
	MOVD $0, R14

neon8192m24f64_inv_stage6_base:
	CMP  $8192, R14
	BGE  neon8192m24f64_inv_stage7

	MOVD $0, R15

neon8192m24f64_inv_stage6_j:
	CMP  $1024, R15
	BGE  neon8192m24f64_inv_stage6_next

	ADD  R14, R15, R0
	ADD  $1024, R0, R1
	ADD  $2048, R0, R2
	ADD  $3072, R0, R3

	// Twiddles: w1=tw[j*2], w2=tw[j*4], w3=tw[j*6]
	LSL  $1, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F0
	FMOVD 8(R5), F1
	FNEGD F1, F1

	LSL  $2, R15, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F2
	FMOVD 8(R5), F3
	FNEGD F3, F3

	MOVD $6, R6
	MUL  R15, R6, R4
	LSL  $4, R4, R5
	ADD  R10, R5, R5
	FMOVD 0(R5), F4
	FMOVD 8(R5), F5
	FNEGD F5, F5

	// Load values
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F6
	FMOVD 8(R7), F7

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F8
	FMOVD 8(R7), F9

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F10
	FMOVD 8(R7), F11

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD 0(R7), F12
	FMOVD 8(R7), F13

	// Apply twiddles
	FMULD F0, F8, F14
	FMULD F1, F9, F15
	FSUBD F15, F14, F14
	FMULD F0, F9, F15
	FMULD F1, F8, F16
	FADDD F16, F15, F15
	FMOVD F14, F8
	FMOVD F15, F9

	FMULD F2, F10, F14
	FMULD F3, F11, F15
	FSUBD F15, F14, F14
	FMULD F2, F11, F15
	FMULD F3, F10, F16
	FADDD F16, F15, F15
	FMOVD F14, F10
	FMOVD F15, F11

	FMULD F4, F12, F14
	FMULD F5, F13, F15
	FSUBD F15, F14, F14
	FMULD F4, F13, F15
	FMULD F5, F12, F16
	FADDD F16, F15, F15
	FMOVD F14, F12
	FMOVD F15, F13

	// Radix-4 butterfly
	FADDD F10, F6, F14
	FADDD F11, F7, F15
	FSUBD F10, F6, F16
	FSUBD F11, F7, F17

	FADDD F12, F8, F18
	FADDD F13, F9, F19
	FSUBD F12, F8, F20
	FSUBD F13, F9, F21

	FADDD F18, F14, F22
	FADDD F19, F15, F23
	FSUBD F18, F14, F24
	FSUBD F19, F15, F25

	FSUBD F21, F16, F26
	FADDD F20, F17, F27

	FADDD F21, F16, F28
	FSUBD F20, F17, F29

	// Store
	LSL  $4, R0, R7
	ADD  R8, R7, R7
	FMOVD F22, 0(R7)
	FMOVD F23, 8(R7)

	LSL  $4, R1, R7
	ADD  R8, R7, R7
	FMOVD F26, 0(R7)
	FMOVD F27, 8(R7)

	LSL  $4, R2, R7
	ADD  R8, R7, R7
	FMOVD F24, 0(R7)
	FMOVD F25, 8(R7)

	LSL  $4, R3, R7
	ADD  R8, R7, R7
	FMOVD F28, 0(R7)
	FMOVD F29, 8(R7)

	ADD  $1, R15, R15
	B    neon8192m24f64_inv_stage6_j

neon8192m24f64_inv_stage6_next:
	ADD  $4096, R14, R14
	B    neon8192m24f64_inv_stage6_base

neon8192m24f64_inv_stage7:
	// =========================================================================
	// Final stage: radix-2 with twiddles, size=8192, step=1
	// =========================================================================
	MOVD $0, R0

neon8192m24f64_inv_stage7_loop:
	CMP  $4096, R0
	BGE  neon8192m24f64_inv_done

	ADD  $4096, R0, R1

	// Load twiddle[j]
	LSL  $4, R0, R2
	ADD  R10, R2, R2
	FMOVD 0(R2), F0
	FMOVD 8(R2), F1
	FNEGD F1, F1

	// Load a, b
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F2
	FMOVD 8(R2), F3

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD 0(R2), F4
	FMOVD 8(R2), F5

	// wb = w * b
	FMULD F0, F4, F6
	FMULD F1, F5, F7
	FSUBD F7, F6, F6
	FMULD F0, F5, F7
	FMULD F1, F4, F8
	FADDD F8, F7, F7

	// Butterfly
	FADDD F6, F2, F8
	FADDD F7, F3, F9
	FSUBD F6, F2, F10
	FSUBD F7, F3, F11

	// Store
	LSL  $4, R0, R2
	ADD  R8, R2, R2
	FMOVD F8, 0(R2)
	FMOVD F9, 8(R2)

	LSL  $4, R1, R2
	ADD  R8, R2, R2
	FMOVD F10, 0(R2)
	FMOVD F11, 8(R2)

	ADD  $1, R0, R0
	B    neon8192m24f64_inv_stage7_loop

neon8192m24f64_inv_done:
	CMP  R8, R20
	BEQ  neon8192m24f64_inv_scale

	MOVD $0, R0
neon8192m24f64_inv_copy_loop:
	CMP  $8192, R0
	BGE  neon8192m24f64_inv_scale
	LSL  $4, R0, R1
	ADD  R8, R1, R2
	MOVD (R2), R3
	MOVD 8(R2), R4
	ADD  R20, R1, R5
	MOVD R3, (R5)
	MOVD R4, 8(R5)
	ADD  $1, R0, R0
	B    neon8192m24f64_inv_copy_loop

neon8192m24f64_inv_scale:
	MOVD $·neonInv8192F64(SB), R1
	FMOVD (R1), F0
	MOVD $0, R0

neon8192m24f64_inv_scale_loop:
	CMP  $8192, R0
	BGE  neon8192m24f64_inv_return_true
	LSL  $4, R0, R1
	ADD  R20, R1, R1
	FMOVD 0(R1), F2
	FMOVD 8(R1), F3
	FMULD F0, F2, F2
	FMULD F0, F3, F3
	FMOVD F2, 0(R1)
	FMOVD F3, 8(R1)
	ADD  $1, R0, R0
	B    neon8192m24f64_inv_scale_loop

neon8192m24f64_inv_return_true:
	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8192m24f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// =========================================================================
// Bit-reversal table for size 8192 mixed-radix 4,4,...,4,2
// =========================================================================
DATA bitrev_size8192_mixed24_f64<>+0x000(SB)/8, $0
DATA bitrev_size8192_mixed24_f64<>+0x008(SB)/8, $2048
DATA bitrev_size8192_mixed24_f64<>+0x010(SB)/8, $4096
DATA bitrev_size8192_mixed24_f64<>+0x018(SB)/8, $6144
DATA bitrev_size8192_mixed24_f64<>+0x020(SB)/8, $512
DATA bitrev_size8192_mixed24_f64<>+0x028(SB)/8, $2560
DATA bitrev_size8192_mixed24_f64<>+0x030(SB)/8, $4608
DATA bitrev_size8192_mixed24_f64<>+0x038(SB)/8, $6656
DATA bitrev_size8192_mixed24_f64<>+0x040(SB)/8, $1024
DATA bitrev_size8192_mixed24_f64<>+0x048(SB)/8, $3072
DATA bitrev_size8192_mixed24_f64<>+0x050(SB)/8, $5120
DATA bitrev_size8192_mixed24_f64<>+0x058(SB)/8, $7168
DATA bitrev_size8192_mixed24_f64<>+0x060(SB)/8, $1536
DATA bitrev_size8192_mixed24_f64<>+0x068(SB)/8, $3584
DATA bitrev_size8192_mixed24_f64<>+0x070(SB)/8, $5632
DATA bitrev_size8192_mixed24_f64<>+0x078(SB)/8, $7680
DATA bitrev_size8192_mixed24_f64<>+0x080(SB)/8, $128
DATA bitrev_size8192_mixed24_f64<>+0x088(SB)/8, $2176
DATA bitrev_size8192_mixed24_f64<>+0x090(SB)/8, $4224
DATA bitrev_size8192_mixed24_f64<>+0x098(SB)/8, $6272
DATA bitrev_size8192_mixed24_f64<>+0x0A0(SB)/8, $640
DATA bitrev_size8192_mixed24_f64<>+0x0A8(SB)/8, $2688
DATA bitrev_size8192_mixed24_f64<>+0x0B0(SB)/8, $4736
DATA bitrev_size8192_mixed24_f64<>+0x0B8(SB)/8, $6784
DATA bitrev_size8192_mixed24_f64<>+0x0C0(SB)/8, $1152
DATA bitrev_size8192_mixed24_f64<>+0x0C8(SB)/8, $3200
DATA bitrev_size8192_mixed24_f64<>+0x0D0(SB)/8, $5248
DATA bitrev_size8192_mixed24_f64<>+0x0D8(SB)/8, $7296
DATA bitrev_size8192_mixed24_f64<>+0x0E0(SB)/8, $1664
DATA bitrev_size8192_mixed24_f64<>+0x0E8(SB)/8, $3712
DATA bitrev_size8192_mixed24_f64<>+0x0F0(SB)/8, $5760
DATA bitrev_size8192_mixed24_f64<>+0x0F8(SB)/8, $7808
DATA bitrev_size8192_mixed24_f64<>+0x100(SB)/8, $256
DATA bitrev_size8192_mixed24_f64<>+0x108(SB)/8, $2304
DATA bitrev_size8192_mixed24_f64<>+0x110(SB)/8, $4352
DATA bitrev_size8192_mixed24_f64<>+0x118(SB)/8, $6400
DATA bitrev_size8192_mixed24_f64<>+0x120(SB)/8, $768
DATA bitrev_size8192_mixed24_f64<>+0x128(SB)/8, $2816
DATA bitrev_size8192_mixed24_f64<>+0x130(SB)/8, $4864
DATA bitrev_size8192_mixed24_f64<>+0x138(SB)/8, $6912
DATA bitrev_size8192_mixed24_f64<>+0x140(SB)/8, $1280
DATA bitrev_size8192_mixed24_f64<>+0x148(SB)/8, $3328
DATA bitrev_size8192_mixed24_f64<>+0x150(SB)/8, $5376
DATA bitrev_size8192_mixed24_f64<>+0x158(SB)/8, $7424
DATA bitrev_size8192_mixed24_f64<>+0x160(SB)/8, $1792
DATA bitrev_size8192_mixed24_f64<>+0x168(SB)/8, $3840
DATA bitrev_size8192_mixed24_f64<>+0x170(SB)/8, $5888
DATA bitrev_size8192_mixed24_f64<>+0x178(SB)/8, $7936
DATA bitrev_size8192_mixed24_f64<>+0x180(SB)/8, $384
DATA bitrev_size8192_mixed24_f64<>+0x188(SB)/8, $2432
DATA bitrev_size8192_mixed24_f64<>+0x190(SB)/8, $4480
DATA bitrev_size8192_mixed24_f64<>+0x198(SB)/8, $6528
DATA bitrev_size8192_mixed24_f64<>+0x1A0(SB)/8, $896
DATA bitrev_size8192_mixed24_f64<>+0x1A8(SB)/8, $2944
DATA bitrev_size8192_mixed24_f64<>+0x1B0(SB)/8, $4992
DATA bitrev_size8192_mixed24_f64<>+0x1B8(SB)/8, $7040
DATA bitrev_size8192_mixed24_f64<>+0x1C0(SB)/8, $1408
DATA bitrev_size8192_mixed24_f64<>+0x1C8(SB)/8, $3456
DATA bitrev_size8192_mixed24_f64<>+0x1D0(SB)/8, $5504
DATA bitrev_size8192_mixed24_f64<>+0x1D8(SB)/8, $7552
DATA bitrev_size8192_mixed24_f64<>+0x1E0(SB)/8, $1920
DATA bitrev_size8192_mixed24_f64<>+0x1E8(SB)/8, $3968
DATA bitrev_size8192_mixed24_f64<>+0x1F0(SB)/8, $6016
DATA bitrev_size8192_mixed24_f64<>+0x1F8(SB)/8, $8064
DATA bitrev_size8192_mixed24_f64<>+0x200(SB)/8, $32
DATA bitrev_size8192_mixed24_f64<>+0x208(SB)/8, $2080
DATA bitrev_size8192_mixed24_f64<>+0x210(SB)/8, $4128
DATA bitrev_size8192_mixed24_f64<>+0x218(SB)/8, $6176
DATA bitrev_size8192_mixed24_f64<>+0x220(SB)/8, $544
DATA bitrev_size8192_mixed24_f64<>+0x228(SB)/8, $2592
DATA bitrev_size8192_mixed24_f64<>+0x230(SB)/8, $4640
DATA bitrev_size8192_mixed24_f64<>+0x238(SB)/8, $6688
DATA bitrev_size8192_mixed24_f64<>+0x240(SB)/8, $1056
DATA bitrev_size8192_mixed24_f64<>+0x248(SB)/8, $3104
DATA bitrev_size8192_mixed24_f64<>+0x250(SB)/8, $5152
DATA bitrev_size8192_mixed24_f64<>+0x258(SB)/8, $7200
DATA bitrev_size8192_mixed24_f64<>+0x260(SB)/8, $1568
DATA bitrev_size8192_mixed24_f64<>+0x268(SB)/8, $3616
DATA bitrev_size8192_mixed24_f64<>+0x270(SB)/8, $5664
DATA bitrev_size8192_mixed24_f64<>+0x278(SB)/8, $7712
DATA bitrev_size8192_mixed24_f64<>+0x280(SB)/8, $160
DATA bitrev_size8192_mixed24_f64<>+0x288(SB)/8, $2208
DATA bitrev_size8192_mixed24_f64<>+0x290(SB)/8, $4256
DATA bitrev_size8192_mixed24_f64<>+0x298(SB)/8, $6304
DATA bitrev_size8192_mixed24_f64<>+0x2A0(SB)/8, $672
DATA bitrev_size8192_mixed24_f64<>+0x2A8(SB)/8, $2720
DATA bitrev_size8192_mixed24_f64<>+0x2B0(SB)/8, $4768
DATA bitrev_size8192_mixed24_f64<>+0x2B8(SB)/8, $6816
DATA bitrev_size8192_mixed24_f64<>+0x2C0(SB)/8, $1184
DATA bitrev_size8192_mixed24_f64<>+0x2C8(SB)/8, $3232
DATA bitrev_size8192_mixed24_f64<>+0x2D0(SB)/8, $5280
DATA bitrev_size8192_mixed24_f64<>+0x2D8(SB)/8, $7328
DATA bitrev_size8192_mixed24_f64<>+0x2E0(SB)/8, $1696
DATA bitrev_size8192_mixed24_f64<>+0x2E8(SB)/8, $3744
DATA bitrev_size8192_mixed24_f64<>+0x2F0(SB)/8, $5792
DATA bitrev_size8192_mixed24_f64<>+0x2F8(SB)/8, $7840
DATA bitrev_size8192_mixed24_f64<>+0x300(SB)/8, $288
DATA bitrev_size8192_mixed24_f64<>+0x308(SB)/8, $2336
DATA bitrev_size8192_mixed24_f64<>+0x310(SB)/8, $4384
DATA bitrev_size8192_mixed24_f64<>+0x318(SB)/8, $6432
DATA bitrev_size8192_mixed24_f64<>+0x320(SB)/8, $800
DATA bitrev_size8192_mixed24_f64<>+0x328(SB)/8, $2848
DATA bitrev_size8192_mixed24_f64<>+0x330(SB)/8, $4896
DATA bitrev_size8192_mixed24_f64<>+0x338(SB)/8, $6944
DATA bitrev_size8192_mixed24_f64<>+0x340(SB)/8, $1312
DATA bitrev_size8192_mixed24_f64<>+0x348(SB)/8, $3360
DATA bitrev_size8192_mixed24_f64<>+0x350(SB)/8, $5408
DATA bitrev_size8192_mixed24_f64<>+0x358(SB)/8, $7456
DATA bitrev_size8192_mixed24_f64<>+0x360(SB)/8, $1824
DATA bitrev_size8192_mixed24_f64<>+0x368(SB)/8, $3872
DATA bitrev_size8192_mixed24_f64<>+0x370(SB)/8, $5920
DATA bitrev_size8192_mixed24_f64<>+0x378(SB)/8, $7968
DATA bitrev_size8192_mixed24_f64<>+0x380(SB)/8, $416
DATA bitrev_size8192_mixed24_f64<>+0x388(SB)/8, $2464
DATA bitrev_size8192_mixed24_f64<>+0x390(SB)/8, $4512
DATA bitrev_size8192_mixed24_f64<>+0x398(SB)/8, $6560
DATA bitrev_size8192_mixed24_f64<>+0x3A0(SB)/8, $928
DATA bitrev_size8192_mixed24_f64<>+0x3A8(SB)/8, $2976
DATA bitrev_size8192_mixed24_f64<>+0x3B0(SB)/8, $5024
DATA bitrev_size8192_mixed24_f64<>+0x3B8(SB)/8, $7072
DATA bitrev_size8192_mixed24_f64<>+0x3C0(SB)/8, $1440
DATA bitrev_size8192_mixed24_f64<>+0x3C8(SB)/8, $3488
DATA bitrev_size8192_mixed24_f64<>+0x3D0(SB)/8, $5536
DATA bitrev_size8192_mixed24_f64<>+0x3D8(SB)/8, $7584
DATA bitrev_size8192_mixed24_f64<>+0x3E0(SB)/8, $1952
DATA bitrev_size8192_mixed24_f64<>+0x3E8(SB)/8, $4000
DATA bitrev_size8192_mixed24_f64<>+0x3F0(SB)/8, $6048
DATA bitrev_size8192_mixed24_f64<>+0x3F8(SB)/8, $8096
DATA bitrev_size8192_mixed24_f64<>+0x400(SB)/8, $64
DATA bitrev_size8192_mixed24_f64<>+0x408(SB)/8, $2112
DATA bitrev_size8192_mixed24_f64<>+0x410(SB)/8, $4160
DATA bitrev_size8192_mixed24_f64<>+0x418(SB)/8, $6208
DATA bitrev_size8192_mixed24_f64<>+0x420(SB)/8, $576
DATA bitrev_size8192_mixed24_f64<>+0x428(SB)/8, $2624
DATA bitrev_size8192_mixed24_f64<>+0x430(SB)/8, $4672
DATA bitrev_size8192_mixed24_f64<>+0x438(SB)/8, $6720
DATA bitrev_size8192_mixed24_f64<>+0x440(SB)/8, $1088
DATA bitrev_size8192_mixed24_f64<>+0x448(SB)/8, $3136
DATA bitrev_size8192_mixed24_f64<>+0x450(SB)/8, $5184
DATA bitrev_size8192_mixed24_f64<>+0x458(SB)/8, $7232
DATA bitrev_size8192_mixed24_f64<>+0x460(SB)/8, $1600
DATA bitrev_size8192_mixed24_f64<>+0x468(SB)/8, $3648
DATA bitrev_size8192_mixed24_f64<>+0x470(SB)/8, $5696
DATA bitrev_size8192_mixed24_f64<>+0x478(SB)/8, $7744
DATA bitrev_size8192_mixed24_f64<>+0x480(SB)/8, $192
DATA bitrev_size8192_mixed24_f64<>+0x488(SB)/8, $2240
DATA bitrev_size8192_mixed24_f64<>+0x490(SB)/8, $4288
DATA bitrev_size8192_mixed24_f64<>+0x498(SB)/8, $6336
DATA bitrev_size8192_mixed24_f64<>+0x4A0(SB)/8, $704
DATA bitrev_size8192_mixed24_f64<>+0x4A8(SB)/8, $2752
DATA bitrev_size8192_mixed24_f64<>+0x4B0(SB)/8, $4800
DATA bitrev_size8192_mixed24_f64<>+0x4B8(SB)/8, $6848
DATA bitrev_size8192_mixed24_f64<>+0x4C0(SB)/8, $1216
DATA bitrev_size8192_mixed24_f64<>+0x4C8(SB)/8, $3264
DATA bitrev_size8192_mixed24_f64<>+0x4D0(SB)/8, $5312
DATA bitrev_size8192_mixed24_f64<>+0x4D8(SB)/8, $7360
DATA bitrev_size8192_mixed24_f64<>+0x4E0(SB)/8, $1728
DATA bitrev_size8192_mixed24_f64<>+0x4E8(SB)/8, $3776
DATA bitrev_size8192_mixed24_f64<>+0x4F0(SB)/8, $5824
DATA bitrev_size8192_mixed24_f64<>+0x4F8(SB)/8, $7872
DATA bitrev_size8192_mixed24_f64<>+0x500(SB)/8, $320
DATA bitrev_size8192_mixed24_f64<>+0x508(SB)/8, $2368
DATA bitrev_size8192_mixed24_f64<>+0x510(SB)/8, $4416
DATA bitrev_size8192_mixed24_f64<>+0x518(SB)/8, $6464
DATA bitrev_size8192_mixed24_f64<>+0x520(SB)/8, $832
DATA bitrev_size8192_mixed24_f64<>+0x528(SB)/8, $2880
DATA bitrev_size8192_mixed24_f64<>+0x530(SB)/8, $4928
DATA bitrev_size8192_mixed24_f64<>+0x538(SB)/8, $6976
DATA bitrev_size8192_mixed24_f64<>+0x540(SB)/8, $1344
DATA bitrev_size8192_mixed24_f64<>+0x548(SB)/8, $3392
DATA bitrev_size8192_mixed24_f64<>+0x550(SB)/8, $5440
DATA bitrev_size8192_mixed24_f64<>+0x558(SB)/8, $7488
DATA bitrev_size8192_mixed24_f64<>+0x560(SB)/8, $1856
DATA bitrev_size8192_mixed24_f64<>+0x568(SB)/8, $3904
DATA bitrev_size8192_mixed24_f64<>+0x570(SB)/8, $5952
DATA bitrev_size8192_mixed24_f64<>+0x578(SB)/8, $8000
DATA bitrev_size8192_mixed24_f64<>+0x580(SB)/8, $448
DATA bitrev_size8192_mixed24_f64<>+0x588(SB)/8, $2496
DATA bitrev_size8192_mixed24_f64<>+0x590(SB)/8, $4544
DATA bitrev_size8192_mixed24_f64<>+0x598(SB)/8, $6592
DATA bitrev_size8192_mixed24_f64<>+0x5A0(SB)/8, $960
DATA bitrev_size8192_mixed24_f64<>+0x5A8(SB)/8, $3008
DATA bitrev_size8192_mixed24_f64<>+0x5B0(SB)/8, $5056
DATA bitrev_size8192_mixed24_f64<>+0x5B8(SB)/8, $7104
DATA bitrev_size8192_mixed24_f64<>+0x5C0(SB)/8, $1472
DATA bitrev_size8192_mixed24_f64<>+0x5C8(SB)/8, $3520
DATA bitrev_size8192_mixed24_f64<>+0x5D0(SB)/8, $5568
DATA bitrev_size8192_mixed24_f64<>+0x5D8(SB)/8, $7616
DATA bitrev_size8192_mixed24_f64<>+0x5E0(SB)/8, $1984
DATA bitrev_size8192_mixed24_f64<>+0x5E8(SB)/8, $4032
DATA bitrev_size8192_mixed24_f64<>+0x5F0(SB)/8, $6080
DATA bitrev_size8192_mixed24_f64<>+0x5F8(SB)/8, $8128
DATA bitrev_size8192_mixed24_f64<>+0x600(SB)/8, $96
DATA bitrev_size8192_mixed24_f64<>+0x608(SB)/8, $2144
DATA bitrev_size8192_mixed24_f64<>+0x610(SB)/8, $4192
DATA bitrev_size8192_mixed24_f64<>+0x618(SB)/8, $6240
DATA bitrev_size8192_mixed24_f64<>+0x620(SB)/8, $608
DATA bitrev_size8192_mixed24_f64<>+0x628(SB)/8, $2656
DATA bitrev_size8192_mixed24_f64<>+0x630(SB)/8, $4704
DATA bitrev_size8192_mixed24_f64<>+0x638(SB)/8, $6752
DATA bitrev_size8192_mixed24_f64<>+0x640(SB)/8, $1120
DATA bitrev_size8192_mixed24_f64<>+0x648(SB)/8, $3168
DATA bitrev_size8192_mixed24_f64<>+0x650(SB)/8, $5216
DATA bitrev_size8192_mixed24_f64<>+0x658(SB)/8, $7264
DATA bitrev_size8192_mixed24_f64<>+0x660(SB)/8, $1632
DATA bitrev_size8192_mixed24_f64<>+0x668(SB)/8, $3680
DATA bitrev_size8192_mixed24_f64<>+0x670(SB)/8, $5728
DATA bitrev_size8192_mixed24_f64<>+0x678(SB)/8, $7776
DATA bitrev_size8192_mixed24_f64<>+0x680(SB)/8, $224
DATA bitrev_size8192_mixed24_f64<>+0x688(SB)/8, $2272
DATA bitrev_size8192_mixed24_f64<>+0x690(SB)/8, $4320
DATA bitrev_size8192_mixed24_f64<>+0x698(SB)/8, $6368
DATA bitrev_size8192_mixed24_f64<>+0x6A0(SB)/8, $736
DATA bitrev_size8192_mixed24_f64<>+0x6A8(SB)/8, $2784
DATA bitrev_size8192_mixed24_f64<>+0x6B0(SB)/8, $4832
DATA bitrev_size8192_mixed24_f64<>+0x6B8(SB)/8, $6880
DATA bitrev_size8192_mixed24_f64<>+0x6C0(SB)/8, $1248
DATA bitrev_size8192_mixed24_f64<>+0x6C8(SB)/8, $3296
DATA bitrev_size8192_mixed24_f64<>+0x6D0(SB)/8, $5344
DATA bitrev_size8192_mixed24_f64<>+0x6D8(SB)/8, $7392
DATA bitrev_size8192_mixed24_f64<>+0x6E0(SB)/8, $1760
DATA bitrev_size8192_mixed24_f64<>+0x6E8(SB)/8, $3808
DATA bitrev_size8192_mixed24_f64<>+0x6F0(SB)/8, $5856
DATA bitrev_size8192_mixed24_f64<>+0x6F8(SB)/8, $7904
DATA bitrev_size8192_mixed24_f64<>+0x700(SB)/8, $352
DATA bitrev_size8192_mixed24_f64<>+0x708(SB)/8, $2400
DATA bitrev_size8192_mixed24_f64<>+0x710(SB)/8, $4448
DATA bitrev_size8192_mixed24_f64<>+0x718(SB)/8, $6496
DATA bitrev_size8192_mixed24_f64<>+0x720(SB)/8, $864
DATA bitrev_size8192_mixed24_f64<>+0x728(SB)/8, $2912
DATA bitrev_size8192_mixed24_f64<>+0x730(SB)/8, $4960
DATA bitrev_size8192_mixed24_f64<>+0x738(SB)/8, $7008
DATA bitrev_size8192_mixed24_f64<>+0x740(SB)/8, $1376
DATA bitrev_size8192_mixed24_f64<>+0x748(SB)/8, $3424
DATA bitrev_size8192_mixed24_f64<>+0x750(SB)/8, $5472
DATA bitrev_size8192_mixed24_f64<>+0x758(SB)/8, $7520
DATA bitrev_size8192_mixed24_f64<>+0x760(SB)/8, $1888
DATA bitrev_size8192_mixed24_f64<>+0x768(SB)/8, $3936
DATA bitrev_size8192_mixed24_f64<>+0x770(SB)/8, $5984
DATA bitrev_size8192_mixed24_f64<>+0x778(SB)/8, $8032
DATA bitrev_size8192_mixed24_f64<>+0x780(SB)/8, $480
DATA bitrev_size8192_mixed24_f64<>+0x788(SB)/8, $2528
DATA bitrev_size8192_mixed24_f64<>+0x790(SB)/8, $4576
DATA bitrev_size8192_mixed24_f64<>+0x798(SB)/8, $6624
DATA bitrev_size8192_mixed24_f64<>+0x7A0(SB)/8, $992
DATA bitrev_size8192_mixed24_f64<>+0x7A8(SB)/8, $3040
DATA bitrev_size8192_mixed24_f64<>+0x7B0(SB)/8, $5088
DATA bitrev_size8192_mixed24_f64<>+0x7B8(SB)/8, $7136
DATA bitrev_size8192_mixed24_f64<>+0x7C0(SB)/8, $1504
DATA bitrev_size8192_mixed24_f64<>+0x7C8(SB)/8, $3552
DATA bitrev_size8192_mixed24_f64<>+0x7D0(SB)/8, $5600
DATA bitrev_size8192_mixed24_f64<>+0x7D8(SB)/8, $7648
DATA bitrev_size8192_mixed24_f64<>+0x7E0(SB)/8, $2016
DATA bitrev_size8192_mixed24_f64<>+0x7E8(SB)/8, $4064
DATA bitrev_size8192_mixed24_f64<>+0x7F0(SB)/8, $6112
DATA bitrev_size8192_mixed24_f64<>+0x7F8(SB)/8, $8160
DATA bitrev_size8192_mixed24_f64<>+0x800(SB)/8, $8
DATA bitrev_size8192_mixed24_f64<>+0x808(SB)/8, $2056
DATA bitrev_size8192_mixed24_f64<>+0x810(SB)/8, $4104
DATA bitrev_size8192_mixed24_f64<>+0x818(SB)/8, $6152
DATA bitrev_size8192_mixed24_f64<>+0x820(SB)/8, $520
DATA bitrev_size8192_mixed24_f64<>+0x828(SB)/8, $2568
DATA bitrev_size8192_mixed24_f64<>+0x830(SB)/8, $4616
DATA bitrev_size8192_mixed24_f64<>+0x838(SB)/8, $6664
DATA bitrev_size8192_mixed24_f64<>+0x840(SB)/8, $1032
DATA bitrev_size8192_mixed24_f64<>+0x848(SB)/8, $3080
DATA bitrev_size8192_mixed24_f64<>+0x850(SB)/8, $5128
DATA bitrev_size8192_mixed24_f64<>+0x858(SB)/8, $7176
DATA bitrev_size8192_mixed24_f64<>+0x860(SB)/8, $1544
DATA bitrev_size8192_mixed24_f64<>+0x868(SB)/8, $3592
DATA bitrev_size8192_mixed24_f64<>+0x870(SB)/8, $5640
DATA bitrev_size8192_mixed24_f64<>+0x878(SB)/8, $7688
DATA bitrev_size8192_mixed24_f64<>+0x880(SB)/8, $136
DATA bitrev_size8192_mixed24_f64<>+0x888(SB)/8, $2184
DATA bitrev_size8192_mixed24_f64<>+0x890(SB)/8, $4232
DATA bitrev_size8192_mixed24_f64<>+0x898(SB)/8, $6280
DATA bitrev_size8192_mixed24_f64<>+0x8A0(SB)/8, $648
DATA bitrev_size8192_mixed24_f64<>+0x8A8(SB)/8, $2696
DATA bitrev_size8192_mixed24_f64<>+0x8B0(SB)/8, $4744
DATA bitrev_size8192_mixed24_f64<>+0x8B8(SB)/8, $6792
DATA bitrev_size8192_mixed24_f64<>+0x8C0(SB)/8, $1160
DATA bitrev_size8192_mixed24_f64<>+0x8C8(SB)/8, $3208
DATA bitrev_size8192_mixed24_f64<>+0x8D0(SB)/8, $5256
DATA bitrev_size8192_mixed24_f64<>+0x8D8(SB)/8, $7304
DATA bitrev_size8192_mixed24_f64<>+0x8E0(SB)/8, $1672
DATA bitrev_size8192_mixed24_f64<>+0x8E8(SB)/8, $3720
DATA bitrev_size8192_mixed24_f64<>+0x8F0(SB)/8, $5768
DATA bitrev_size8192_mixed24_f64<>+0x8F8(SB)/8, $7816
DATA bitrev_size8192_mixed24_f64<>+0x900(SB)/8, $264
DATA bitrev_size8192_mixed24_f64<>+0x908(SB)/8, $2312
DATA bitrev_size8192_mixed24_f64<>+0x910(SB)/8, $4360
DATA bitrev_size8192_mixed24_f64<>+0x918(SB)/8, $6408
DATA bitrev_size8192_mixed24_f64<>+0x920(SB)/8, $776
DATA bitrev_size8192_mixed24_f64<>+0x928(SB)/8, $2824
DATA bitrev_size8192_mixed24_f64<>+0x930(SB)/8, $4872
DATA bitrev_size8192_mixed24_f64<>+0x938(SB)/8, $6920
DATA bitrev_size8192_mixed24_f64<>+0x940(SB)/8, $1288
DATA bitrev_size8192_mixed24_f64<>+0x948(SB)/8, $3336
DATA bitrev_size8192_mixed24_f64<>+0x950(SB)/8, $5384
DATA bitrev_size8192_mixed24_f64<>+0x958(SB)/8, $7432
DATA bitrev_size8192_mixed24_f64<>+0x960(SB)/8, $1800
DATA bitrev_size8192_mixed24_f64<>+0x968(SB)/8, $3848
DATA bitrev_size8192_mixed24_f64<>+0x970(SB)/8, $5896
DATA bitrev_size8192_mixed24_f64<>+0x978(SB)/8, $7944
DATA bitrev_size8192_mixed24_f64<>+0x980(SB)/8, $392
DATA bitrev_size8192_mixed24_f64<>+0x988(SB)/8, $2440
DATA bitrev_size8192_mixed24_f64<>+0x990(SB)/8, $4488
DATA bitrev_size8192_mixed24_f64<>+0x998(SB)/8, $6536
DATA bitrev_size8192_mixed24_f64<>+0x9A0(SB)/8, $904
DATA bitrev_size8192_mixed24_f64<>+0x9A8(SB)/8, $2952
DATA bitrev_size8192_mixed24_f64<>+0x9B0(SB)/8, $5000
DATA bitrev_size8192_mixed24_f64<>+0x9B8(SB)/8, $7048
DATA bitrev_size8192_mixed24_f64<>+0x9C0(SB)/8, $1416
DATA bitrev_size8192_mixed24_f64<>+0x9C8(SB)/8, $3464
DATA bitrev_size8192_mixed24_f64<>+0x9D0(SB)/8, $5512
DATA bitrev_size8192_mixed24_f64<>+0x9D8(SB)/8, $7560
DATA bitrev_size8192_mixed24_f64<>+0x9E0(SB)/8, $1928
DATA bitrev_size8192_mixed24_f64<>+0x9E8(SB)/8, $3976
DATA bitrev_size8192_mixed24_f64<>+0x9F0(SB)/8, $6024
DATA bitrev_size8192_mixed24_f64<>+0x9F8(SB)/8, $8072
DATA bitrev_size8192_mixed24_f64<>+0xA00(SB)/8, $40
DATA bitrev_size8192_mixed24_f64<>+0xA08(SB)/8, $2088
DATA bitrev_size8192_mixed24_f64<>+0xA10(SB)/8, $4136
DATA bitrev_size8192_mixed24_f64<>+0xA18(SB)/8, $6184
DATA bitrev_size8192_mixed24_f64<>+0xA20(SB)/8, $552
DATA bitrev_size8192_mixed24_f64<>+0xA28(SB)/8, $2600
DATA bitrev_size8192_mixed24_f64<>+0xA30(SB)/8, $4648
DATA bitrev_size8192_mixed24_f64<>+0xA38(SB)/8, $6696
DATA bitrev_size8192_mixed24_f64<>+0xA40(SB)/8, $1064
DATA bitrev_size8192_mixed24_f64<>+0xA48(SB)/8, $3112
DATA bitrev_size8192_mixed24_f64<>+0xA50(SB)/8, $5160
DATA bitrev_size8192_mixed24_f64<>+0xA58(SB)/8, $7208
DATA bitrev_size8192_mixed24_f64<>+0xA60(SB)/8, $1576
DATA bitrev_size8192_mixed24_f64<>+0xA68(SB)/8, $3624
DATA bitrev_size8192_mixed24_f64<>+0xA70(SB)/8, $5672
DATA bitrev_size8192_mixed24_f64<>+0xA78(SB)/8, $7720
DATA bitrev_size8192_mixed24_f64<>+0xA80(SB)/8, $168
DATA bitrev_size8192_mixed24_f64<>+0xA88(SB)/8, $2216
DATA bitrev_size8192_mixed24_f64<>+0xA90(SB)/8, $4264
DATA bitrev_size8192_mixed24_f64<>+0xA98(SB)/8, $6312
DATA bitrev_size8192_mixed24_f64<>+0xAA0(SB)/8, $680
DATA bitrev_size8192_mixed24_f64<>+0xAA8(SB)/8, $2728
DATA bitrev_size8192_mixed24_f64<>+0xAB0(SB)/8, $4776
DATA bitrev_size8192_mixed24_f64<>+0xAB8(SB)/8, $6824
DATA bitrev_size8192_mixed24_f64<>+0xAC0(SB)/8, $1192
DATA bitrev_size8192_mixed24_f64<>+0xAC8(SB)/8, $3240
DATA bitrev_size8192_mixed24_f64<>+0xAD0(SB)/8, $5288
DATA bitrev_size8192_mixed24_f64<>+0xAD8(SB)/8, $7336
DATA bitrev_size8192_mixed24_f64<>+0xAE0(SB)/8, $1704
DATA bitrev_size8192_mixed24_f64<>+0xAE8(SB)/8, $3752
DATA bitrev_size8192_mixed24_f64<>+0xAF0(SB)/8, $5800
DATA bitrev_size8192_mixed24_f64<>+0xAF8(SB)/8, $7848
DATA bitrev_size8192_mixed24_f64<>+0xB00(SB)/8, $296
DATA bitrev_size8192_mixed24_f64<>+0xB08(SB)/8, $2344
DATA bitrev_size8192_mixed24_f64<>+0xB10(SB)/8, $4392
DATA bitrev_size8192_mixed24_f64<>+0xB18(SB)/8, $6440
DATA bitrev_size8192_mixed24_f64<>+0xB20(SB)/8, $808
DATA bitrev_size8192_mixed24_f64<>+0xB28(SB)/8, $2856
DATA bitrev_size8192_mixed24_f64<>+0xB30(SB)/8, $4904
DATA bitrev_size8192_mixed24_f64<>+0xB38(SB)/8, $6952
DATA bitrev_size8192_mixed24_f64<>+0xB40(SB)/8, $1320
DATA bitrev_size8192_mixed24_f64<>+0xB48(SB)/8, $3368
DATA bitrev_size8192_mixed24_f64<>+0xB50(SB)/8, $5416
DATA bitrev_size8192_mixed24_f64<>+0xB58(SB)/8, $7464
DATA bitrev_size8192_mixed24_f64<>+0xB60(SB)/8, $1832
DATA bitrev_size8192_mixed24_f64<>+0xB68(SB)/8, $3880
DATA bitrev_size8192_mixed24_f64<>+0xB70(SB)/8, $5928
DATA bitrev_size8192_mixed24_f64<>+0xB78(SB)/8, $7976
DATA bitrev_size8192_mixed24_f64<>+0xB80(SB)/8, $424
DATA bitrev_size8192_mixed24_f64<>+0xB88(SB)/8, $2472
DATA bitrev_size8192_mixed24_f64<>+0xB90(SB)/8, $4520
DATA bitrev_size8192_mixed24_f64<>+0xB98(SB)/8, $6568
DATA bitrev_size8192_mixed24_f64<>+0xBA0(SB)/8, $936
DATA bitrev_size8192_mixed24_f64<>+0xBA8(SB)/8, $2984
DATA bitrev_size8192_mixed24_f64<>+0xBB0(SB)/8, $5032
DATA bitrev_size8192_mixed24_f64<>+0xBB8(SB)/8, $7080
DATA bitrev_size8192_mixed24_f64<>+0xBC0(SB)/8, $1448
DATA bitrev_size8192_mixed24_f64<>+0xBC8(SB)/8, $3496
DATA bitrev_size8192_mixed24_f64<>+0xBD0(SB)/8, $5544
DATA bitrev_size8192_mixed24_f64<>+0xBD8(SB)/8, $7592
DATA bitrev_size8192_mixed24_f64<>+0xBE0(SB)/8, $1960
DATA bitrev_size8192_mixed24_f64<>+0xBE8(SB)/8, $4008
DATA bitrev_size8192_mixed24_f64<>+0xBF0(SB)/8, $6056
DATA bitrev_size8192_mixed24_f64<>+0xBF8(SB)/8, $8104
DATA bitrev_size8192_mixed24_f64<>+0xC00(SB)/8, $72
DATA bitrev_size8192_mixed24_f64<>+0xC08(SB)/8, $2120
DATA bitrev_size8192_mixed24_f64<>+0xC10(SB)/8, $4168
DATA bitrev_size8192_mixed24_f64<>+0xC18(SB)/8, $6216
DATA bitrev_size8192_mixed24_f64<>+0xC20(SB)/8, $584
DATA bitrev_size8192_mixed24_f64<>+0xC28(SB)/8, $2632
DATA bitrev_size8192_mixed24_f64<>+0xC30(SB)/8, $4680
DATA bitrev_size8192_mixed24_f64<>+0xC38(SB)/8, $6728
DATA bitrev_size8192_mixed24_f64<>+0xC40(SB)/8, $1096
DATA bitrev_size8192_mixed24_f64<>+0xC48(SB)/8, $3144
DATA bitrev_size8192_mixed24_f64<>+0xC50(SB)/8, $5192
DATA bitrev_size8192_mixed24_f64<>+0xC58(SB)/8, $7240
DATA bitrev_size8192_mixed24_f64<>+0xC60(SB)/8, $1608
DATA bitrev_size8192_mixed24_f64<>+0xC68(SB)/8, $3656
DATA bitrev_size8192_mixed24_f64<>+0xC70(SB)/8, $5704
DATA bitrev_size8192_mixed24_f64<>+0xC78(SB)/8, $7752
DATA bitrev_size8192_mixed24_f64<>+0xC80(SB)/8, $200
DATA bitrev_size8192_mixed24_f64<>+0xC88(SB)/8, $2248
DATA bitrev_size8192_mixed24_f64<>+0xC90(SB)/8, $4296
DATA bitrev_size8192_mixed24_f64<>+0xC98(SB)/8, $6344
DATA bitrev_size8192_mixed24_f64<>+0xCA0(SB)/8, $712
DATA bitrev_size8192_mixed24_f64<>+0xCA8(SB)/8, $2760
DATA bitrev_size8192_mixed24_f64<>+0xCB0(SB)/8, $4808
DATA bitrev_size8192_mixed24_f64<>+0xCB8(SB)/8, $6856
DATA bitrev_size8192_mixed24_f64<>+0xCC0(SB)/8, $1224
DATA bitrev_size8192_mixed24_f64<>+0xCC8(SB)/8, $3272
DATA bitrev_size8192_mixed24_f64<>+0xCD0(SB)/8, $5320
DATA bitrev_size8192_mixed24_f64<>+0xCD8(SB)/8, $7368
DATA bitrev_size8192_mixed24_f64<>+0xCE0(SB)/8, $1736
DATA bitrev_size8192_mixed24_f64<>+0xCE8(SB)/8, $3784
DATA bitrev_size8192_mixed24_f64<>+0xCF0(SB)/8, $5832
DATA bitrev_size8192_mixed24_f64<>+0xCF8(SB)/8, $7880
DATA bitrev_size8192_mixed24_f64<>+0xD00(SB)/8, $328
DATA bitrev_size8192_mixed24_f64<>+0xD08(SB)/8, $2376
DATA bitrev_size8192_mixed24_f64<>+0xD10(SB)/8, $4424
DATA bitrev_size8192_mixed24_f64<>+0xD18(SB)/8, $6472
DATA bitrev_size8192_mixed24_f64<>+0xD20(SB)/8, $840
DATA bitrev_size8192_mixed24_f64<>+0xD28(SB)/8, $2888
DATA bitrev_size8192_mixed24_f64<>+0xD30(SB)/8, $4936
DATA bitrev_size8192_mixed24_f64<>+0xD38(SB)/8, $6984
DATA bitrev_size8192_mixed24_f64<>+0xD40(SB)/8, $1352
DATA bitrev_size8192_mixed24_f64<>+0xD48(SB)/8, $3400
DATA bitrev_size8192_mixed24_f64<>+0xD50(SB)/8, $5448
DATA bitrev_size8192_mixed24_f64<>+0xD58(SB)/8, $7496
DATA bitrev_size8192_mixed24_f64<>+0xD60(SB)/8, $1864
DATA bitrev_size8192_mixed24_f64<>+0xD68(SB)/8, $3912
DATA bitrev_size8192_mixed24_f64<>+0xD70(SB)/8, $5960
DATA bitrev_size8192_mixed24_f64<>+0xD78(SB)/8, $8008
DATA bitrev_size8192_mixed24_f64<>+0xD80(SB)/8, $456
DATA bitrev_size8192_mixed24_f64<>+0xD88(SB)/8, $2504
DATA bitrev_size8192_mixed24_f64<>+0xD90(SB)/8, $4552
DATA bitrev_size8192_mixed24_f64<>+0xD98(SB)/8, $6600
DATA bitrev_size8192_mixed24_f64<>+0xDA0(SB)/8, $968
DATA bitrev_size8192_mixed24_f64<>+0xDA8(SB)/8, $3016
DATA bitrev_size8192_mixed24_f64<>+0xDB0(SB)/8, $5064
DATA bitrev_size8192_mixed24_f64<>+0xDB8(SB)/8, $7112
DATA bitrev_size8192_mixed24_f64<>+0xDC0(SB)/8, $1480
DATA bitrev_size8192_mixed24_f64<>+0xDC8(SB)/8, $3528
DATA bitrev_size8192_mixed24_f64<>+0xDD0(SB)/8, $5576
DATA bitrev_size8192_mixed24_f64<>+0xDD8(SB)/8, $7624
DATA bitrev_size8192_mixed24_f64<>+0xDE0(SB)/8, $1992
DATA bitrev_size8192_mixed24_f64<>+0xDE8(SB)/8, $4040
DATA bitrev_size8192_mixed24_f64<>+0xDF0(SB)/8, $6088
DATA bitrev_size8192_mixed24_f64<>+0xDF8(SB)/8, $8136
DATA bitrev_size8192_mixed24_f64<>+0xE00(SB)/8, $104
DATA bitrev_size8192_mixed24_f64<>+0xE08(SB)/8, $2152
DATA bitrev_size8192_mixed24_f64<>+0xE10(SB)/8, $4200
DATA bitrev_size8192_mixed24_f64<>+0xE18(SB)/8, $6248
DATA bitrev_size8192_mixed24_f64<>+0xE20(SB)/8, $616
DATA bitrev_size8192_mixed24_f64<>+0xE28(SB)/8, $2664
DATA bitrev_size8192_mixed24_f64<>+0xE30(SB)/8, $4712
DATA bitrev_size8192_mixed24_f64<>+0xE38(SB)/8, $6760
DATA bitrev_size8192_mixed24_f64<>+0xE40(SB)/8, $1128
DATA bitrev_size8192_mixed24_f64<>+0xE48(SB)/8, $3176
DATA bitrev_size8192_mixed24_f64<>+0xE50(SB)/8, $5224
DATA bitrev_size8192_mixed24_f64<>+0xE58(SB)/8, $7272
DATA bitrev_size8192_mixed24_f64<>+0xE60(SB)/8, $1640
DATA bitrev_size8192_mixed24_f64<>+0xE68(SB)/8, $3688
DATA bitrev_size8192_mixed24_f64<>+0xE70(SB)/8, $5736
DATA bitrev_size8192_mixed24_f64<>+0xE78(SB)/8, $7784
DATA bitrev_size8192_mixed24_f64<>+0xE80(SB)/8, $232
DATA bitrev_size8192_mixed24_f64<>+0xE88(SB)/8, $2280
DATA bitrev_size8192_mixed24_f64<>+0xE90(SB)/8, $4328
DATA bitrev_size8192_mixed24_f64<>+0xE98(SB)/8, $6376
DATA bitrev_size8192_mixed24_f64<>+0xEA0(SB)/8, $744
DATA bitrev_size8192_mixed24_f64<>+0xEA8(SB)/8, $2792
DATA bitrev_size8192_mixed24_f64<>+0xEB0(SB)/8, $4840
DATA bitrev_size8192_mixed24_f64<>+0xEB8(SB)/8, $6888
DATA bitrev_size8192_mixed24_f64<>+0xEC0(SB)/8, $1256
DATA bitrev_size8192_mixed24_f64<>+0xEC8(SB)/8, $3304
DATA bitrev_size8192_mixed24_f64<>+0xED0(SB)/8, $5352
DATA bitrev_size8192_mixed24_f64<>+0xED8(SB)/8, $7400
DATA bitrev_size8192_mixed24_f64<>+0xEE0(SB)/8, $1768
DATA bitrev_size8192_mixed24_f64<>+0xEE8(SB)/8, $3816
DATA bitrev_size8192_mixed24_f64<>+0xEF0(SB)/8, $5864
DATA bitrev_size8192_mixed24_f64<>+0xEF8(SB)/8, $7912
DATA bitrev_size8192_mixed24_f64<>+0xF00(SB)/8, $360
DATA bitrev_size8192_mixed24_f64<>+0xF08(SB)/8, $2408
DATA bitrev_size8192_mixed24_f64<>+0xF10(SB)/8, $4456
DATA bitrev_size8192_mixed24_f64<>+0xF18(SB)/8, $6504
DATA bitrev_size8192_mixed24_f64<>+0xF20(SB)/8, $872
DATA bitrev_size8192_mixed24_f64<>+0xF28(SB)/8, $2920
DATA bitrev_size8192_mixed24_f64<>+0xF30(SB)/8, $4968
DATA bitrev_size8192_mixed24_f64<>+0xF38(SB)/8, $7016
DATA bitrev_size8192_mixed24_f64<>+0xF40(SB)/8, $1384
DATA bitrev_size8192_mixed24_f64<>+0xF48(SB)/8, $3432
DATA bitrev_size8192_mixed24_f64<>+0xF50(SB)/8, $5480
DATA bitrev_size8192_mixed24_f64<>+0xF58(SB)/8, $7528
DATA bitrev_size8192_mixed24_f64<>+0xF60(SB)/8, $1896
DATA bitrev_size8192_mixed24_f64<>+0xF68(SB)/8, $3944
DATA bitrev_size8192_mixed24_f64<>+0xF70(SB)/8, $5992
DATA bitrev_size8192_mixed24_f64<>+0xF78(SB)/8, $8040
DATA bitrev_size8192_mixed24_f64<>+0xF80(SB)/8, $488
DATA bitrev_size8192_mixed24_f64<>+0xF88(SB)/8, $2536
DATA bitrev_size8192_mixed24_f64<>+0xF90(SB)/8, $4584
DATA bitrev_size8192_mixed24_f64<>+0xF98(SB)/8, $6632
DATA bitrev_size8192_mixed24_f64<>+0xFA0(SB)/8, $1000
DATA bitrev_size8192_mixed24_f64<>+0xFA8(SB)/8, $3048
DATA bitrev_size8192_mixed24_f64<>+0xFB0(SB)/8, $5096
DATA bitrev_size8192_mixed24_f64<>+0xFB8(SB)/8, $7144
DATA bitrev_size8192_mixed24_f64<>+0xFC0(SB)/8, $1512
DATA bitrev_size8192_mixed24_f64<>+0xFC8(SB)/8, $3560
DATA bitrev_size8192_mixed24_f64<>+0xFD0(SB)/8, $5608
DATA bitrev_size8192_mixed24_f64<>+0xFD8(SB)/8, $7656
DATA bitrev_size8192_mixed24_f64<>+0xFE0(SB)/8, $2024
DATA bitrev_size8192_mixed24_f64<>+0xFE8(SB)/8, $4072
DATA bitrev_size8192_mixed24_f64<>+0xFF0(SB)/8, $6120
DATA bitrev_size8192_mixed24_f64<>+0xFF8(SB)/8, $8168
DATA bitrev_size8192_mixed24_f64<>+0x1000(SB)/8, $16
DATA bitrev_size8192_mixed24_f64<>+0x1008(SB)/8, $2064
DATA bitrev_size8192_mixed24_f64<>+0x1010(SB)/8, $4112
DATA bitrev_size8192_mixed24_f64<>+0x1018(SB)/8, $6160
DATA bitrev_size8192_mixed24_f64<>+0x1020(SB)/8, $528
DATA bitrev_size8192_mixed24_f64<>+0x1028(SB)/8, $2576
DATA bitrev_size8192_mixed24_f64<>+0x1030(SB)/8, $4624
DATA bitrev_size8192_mixed24_f64<>+0x1038(SB)/8, $6672
DATA bitrev_size8192_mixed24_f64<>+0x1040(SB)/8, $1040
DATA bitrev_size8192_mixed24_f64<>+0x1048(SB)/8, $3088
DATA bitrev_size8192_mixed24_f64<>+0x1050(SB)/8, $5136
DATA bitrev_size8192_mixed24_f64<>+0x1058(SB)/8, $7184
DATA bitrev_size8192_mixed24_f64<>+0x1060(SB)/8, $1552
DATA bitrev_size8192_mixed24_f64<>+0x1068(SB)/8, $3600
DATA bitrev_size8192_mixed24_f64<>+0x1070(SB)/8, $5648
DATA bitrev_size8192_mixed24_f64<>+0x1078(SB)/8, $7696
DATA bitrev_size8192_mixed24_f64<>+0x1080(SB)/8, $144
DATA bitrev_size8192_mixed24_f64<>+0x1088(SB)/8, $2192
DATA bitrev_size8192_mixed24_f64<>+0x1090(SB)/8, $4240
DATA bitrev_size8192_mixed24_f64<>+0x1098(SB)/8, $6288
DATA bitrev_size8192_mixed24_f64<>+0x10A0(SB)/8, $656
DATA bitrev_size8192_mixed24_f64<>+0x10A8(SB)/8, $2704
DATA bitrev_size8192_mixed24_f64<>+0x10B0(SB)/8, $4752
DATA bitrev_size8192_mixed24_f64<>+0x10B8(SB)/8, $6800
DATA bitrev_size8192_mixed24_f64<>+0x10C0(SB)/8, $1168
DATA bitrev_size8192_mixed24_f64<>+0x10C8(SB)/8, $3216
DATA bitrev_size8192_mixed24_f64<>+0x10D0(SB)/8, $5264
DATA bitrev_size8192_mixed24_f64<>+0x10D8(SB)/8, $7312
DATA bitrev_size8192_mixed24_f64<>+0x10E0(SB)/8, $1680
DATA bitrev_size8192_mixed24_f64<>+0x10E8(SB)/8, $3728
DATA bitrev_size8192_mixed24_f64<>+0x10F0(SB)/8, $5776
DATA bitrev_size8192_mixed24_f64<>+0x10F8(SB)/8, $7824
DATA bitrev_size8192_mixed24_f64<>+0x1100(SB)/8, $272
DATA bitrev_size8192_mixed24_f64<>+0x1108(SB)/8, $2320
DATA bitrev_size8192_mixed24_f64<>+0x1110(SB)/8, $4368
DATA bitrev_size8192_mixed24_f64<>+0x1118(SB)/8, $6416
DATA bitrev_size8192_mixed24_f64<>+0x1120(SB)/8, $784
DATA bitrev_size8192_mixed24_f64<>+0x1128(SB)/8, $2832
DATA bitrev_size8192_mixed24_f64<>+0x1130(SB)/8, $4880
DATA bitrev_size8192_mixed24_f64<>+0x1138(SB)/8, $6928
DATA bitrev_size8192_mixed24_f64<>+0x1140(SB)/8, $1296
DATA bitrev_size8192_mixed24_f64<>+0x1148(SB)/8, $3344
DATA bitrev_size8192_mixed24_f64<>+0x1150(SB)/8, $5392
DATA bitrev_size8192_mixed24_f64<>+0x1158(SB)/8, $7440
DATA bitrev_size8192_mixed24_f64<>+0x1160(SB)/8, $1808
DATA bitrev_size8192_mixed24_f64<>+0x1168(SB)/8, $3856
DATA bitrev_size8192_mixed24_f64<>+0x1170(SB)/8, $5904
DATA bitrev_size8192_mixed24_f64<>+0x1178(SB)/8, $7952
DATA bitrev_size8192_mixed24_f64<>+0x1180(SB)/8, $400
DATA bitrev_size8192_mixed24_f64<>+0x1188(SB)/8, $2448
DATA bitrev_size8192_mixed24_f64<>+0x1190(SB)/8, $4496
DATA bitrev_size8192_mixed24_f64<>+0x1198(SB)/8, $6544
DATA bitrev_size8192_mixed24_f64<>+0x11A0(SB)/8, $912
DATA bitrev_size8192_mixed24_f64<>+0x11A8(SB)/8, $2960
DATA bitrev_size8192_mixed24_f64<>+0x11B0(SB)/8, $5008
DATA bitrev_size8192_mixed24_f64<>+0x11B8(SB)/8, $7056
DATA bitrev_size8192_mixed24_f64<>+0x11C0(SB)/8, $1424
DATA bitrev_size8192_mixed24_f64<>+0x11C8(SB)/8, $3472
DATA bitrev_size8192_mixed24_f64<>+0x11D0(SB)/8, $5520
DATA bitrev_size8192_mixed24_f64<>+0x11D8(SB)/8, $7568
DATA bitrev_size8192_mixed24_f64<>+0x11E0(SB)/8, $1936
DATA bitrev_size8192_mixed24_f64<>+0x11E8(SB)/8, $3984
DATA bitrev_size8192_mixed24_f64<>+0x11F0(SB)/8, $6032
DATA bitrev_size8192_mixed24_f64<>+0x11F8(SB)/8, $8080
DATA bitrev_size8192_mixed24_f64<>+0x1200(SB)/8, $48
DATA bitrev_size8192_mixed24_f64<>+0x1208(SB)/8, $2096
DATA bitrev_size8192_mixed24_f64<>+0x1210(SB)/8, $4144
DATA bitrev_size8192_mixed24_f64<>+0x1218(SB)/8, $6192
DATA bitrev_size8192_mixed24_f64<>+0x1220(SB)/8, $560
DATA bitrev_size8192_mixed24_f64<>+0x1228(SB)/8, $2608
DATA bitrev_size8192_mixed24_f64<>+0x1230(SB)/8, $4656
DATA bitrev_size8192_mixed24_f64<>+0x1238(SB)/8, $6704
DATA bitrev_size8192_mixed24_f64<>+0x1240(SB)/8, $1072
DATA bitrev_size8192_mixed24_f64<>+0x1248(SB)/8, $3120
DATA bitrev_size8192_mixed24_f64<>+0x1250(SB)/8, $5168
DATA bitrev_size8192_mixed24_f64<>+0x1258(SB)/8, $7216
DATA bitrev_size8192_mixed24_f64<>+0x1260(SB)/8, $1584
DATA bitrev_size8192_mixed24_f64<>+0x1268(SB)/8, $3632
DATA bitrev_size8192_mixed24_f64<>+0x1270(SB)/8, $5680
DATA bitrev_size8192_mixed24_f64<>+0x1278(SB)/8, $7728
DATA bitrev_size8192_mixed24_f64<>+0x1280(SB)/8, $176
DATA bitrev_size8192_mixed24_f64<>+0x1288(SB)/8, $2224
DATA bitrev_size8192_mixed24_f64<>+0x1290(SB)/8, $4272
DATA bitrev_size8192_mixed24_f64<>+0x1298(SB)/8, $6320
DATA bitrev_size8192_mixed24_f64<>+0x12A0(SB)/8, $688
DATA bitrev_size8192_mixed24_f64<>+0x12A8(SB)/8, $2736
DATA bitrev_size8192_mixed24_f64<>+0x12B0(SB)/8, $4784
DATA bitrev_size8192_mixed24_f64<>+0x12B8(SB)/8, $6832
DATA bitrev_size8192_mixed24_f64<>+0x12C0(SB)/8, $1200
DATA bitrev_size8192_mixed24_f64<>+0x12C8(SB)/8, $3248
DATA bitrev_size8192_mixed24_f64<>+0x12D0(SB)/8, $5296
DATA bitrev_size8192_mixed24_f64<>+0x12D8(SB)/8, $7344
DATA bitrev_size8192_mixed24_f64<>+0x12E0(SB)/8, $1712
DATA bitrev_size8192_mixed24_f64<>+0x12E8(SB)/8, $3760
DATA bitrev_size8192_mixed24_f64<>+0x12F0(SB)/8, $5808
DATA bitrev_size8192_mixed24_f64<>+0x12F8(SB)/8, $7856
DATA bitrev_size8192_mixed24_f64<>+0x1300(SB)/8, $304
DATA bitrev_size8192_mixed24_f64<>+0x1308(SB)/8, $2352
DATA bitrev_size8192_mixed24_f64<>+0x1310(SB)/8, $4400
DATA bitrev_size8192_mixed24_f64<>+0x1318(SB)/8, $6448
DATA bitrev_size8192_mixed24_f64<>+0x1320(SB)/8, $816
DATA bitrev_size8192_mixed24_f64<>+0x1328(SB)/8, $2864
DATA bitrev_size8192_mixed24_f64<>+0x1330(SB)/8, $4912
DATA bitrev_size8192_mixed24_f64<>+0x1338(SB)/8, $6960
DATA bitrev_size8192_mixed24_f64<>+0x1340(SB)/8, $1328
DATA bitrev_size8192_mixed24_f64<>+0x1348(SB)/8, $3376
DATA bitrev_size8192_mixed24_f64<>+0x1350(SB)/8, $5424
DATA bitrev_size8192_mixed24_f64<>+0x1358(SB)/8, $7472
DATA bitrev_size8192_mixed24_f64<>+0x1360(SB)/8, $1840
DATA bitrev_size8192_mixed24_f64<>+0x1368(SB)/8, $3888
DATA bitrev_size8192_mixed24_f64<>+0x1370(SB)/8, $5936
DATA bitrev_size8192_mixed24_f64<>+0x1378(SB)/8, $7984
DATA bitrev_size8192_mixed24_f64<>+0x1380(SB)/8, $432
DATA bitrev_size8192_mixed24_f64<>+0x1388(SB)/8, $2480
DATA bitrev_size8192_mixed24_f64<>+0x1390(SB)/8, $4528
DATA bitrev_size8192_mixed24_f64<>+0x1398(SB)/8, $6576
DATA bitrev_size8192_mixed24_f64<>+0x13A0(SB)/8, $944
DATA bitrev_size8192_mixed24_f64<>+0x13A8(SB)/8, $2992
DATA bitrev_size8192_mixed24_f64<>+0x13B0(SB)/8, $5040
DATA bitrev_size8192_mixed24_f64<>+0x13B8(SB)/8, $7088
DATA bitrev_size8192_mixed24_f64<>+0x13C0(SB)/8, $1456
DATA bitrev_size8192_mixed24_f64<>+0x13C8(SB)/8, $3504
DATA bitrev_size8192_mixed24_f64<>+0x13D0(SB)/8, $5552
DATA bitrev_size8192_mixed24_f64<>+0x13D8(SB)/8, $7600
DATA bitrev_size8192_mixed24_f64<>+0x13E0(SB)/8, $1968
DATA bitrev_size8192_mixed24_f64<>+0x13E8(SB)/8, $4016
DATA bitrev_size8192_mixed24_f64<>+0x13F0(SB)/8, $6064
DATA bitrev_size8192_mixed24_f64<>+0x13F8(SB)/8, $8112
DATA bitrev_size8192_mixed24_f64<>+0x1400(SB)/8, $80
DATA bitrev_size8192_mixed24_f64<>+0x1408(SB)/8, $2128
DATA bitrev_size8192_mixed24_f64<>+0x1410(SB)/8, $4176
DATA bitrev_size8192_mixed24_f64<>+0x1418(SB)/8, $6224
DATA bitrev_size8192_mixed24_f64<>+0x1420(SB)/8, $592
DATA bitrev_size8192_mixed24_f64<>+0x1428(SB)/8, $2640
DATA bitrev_size8192_mixed24_f64<>+0x1430(SB)/8, $4688
DATA bitrev_size8192_mixed24_f64<>+0x1438(SB)/8, $6736
DATA bitrev_size8192_mixed24_f64<>+0x1440(SB)/8, $1104
DATA bitrev_size8192_mixed24_f64<>+0x1448(SB)/8, $3152
DATA bitrev_size8192_mixed24_f64<>+0x1450(SB)/8, $5200
DATA bitrev_size8192_mixed24_f64<>+0x1458(SB)/8, $7248
DATA bitrev_size8192_mixed24_f64<>+0x1460(SB)/8, $1616
DATA bitrev_size8192_mixed24_f64<>+0x1468(SB)/8, $3664
DATA bitrev_size8192_mixed24_f64<>+0x1470(SB)/8, $5712
DATA bitrev_size8192_mixed24_f64<>+0x1478(SB)/8, $7760
DATA bitrev_size8192_mixed24_f64<>+0x1480(SB)/8, $208
DATA bitrev_size8192_mixed24_f64<>+0x1488(SB)/8, $2256
DATA bitrev_size8192_mixed24_f64<>+0x1490(SB)/8, $4304
DATA bitrev_size8192_mixed24_f64<>+0x1498(SB)/8, $6352
DATA bitrev_size8192_mixed24_f64<>+0x14A0(SB)/8, $720
DATA bitrev_size8192_mixed24_f64<>+0x14A8(SB)/8, $2768
DATA bitrev_size8192_mixed24_f64<>+0x14B0(SB)/8, $4816
DATA bitrev_size8192_mixed24_f64<>+0x14B8(SB)/8, $6864
DATA bitrev_size8192_mixed24_f64<>+0x14C0(SB)/8, $1232
DATA bitrev_size8192_mixed24_f64<>+0x14C8(SB)/8, $3280
DATA bitrev_size8192_mixed24_f64<>+0x14D0(SB)/8, $5328
DATA bitrev_size8192_mixed24_f64<>+0x14D8(SB)/8, $7376
DATA bitrev_size8192_mixed24_f64<>+0x14E0(SB)/8, $1744
DATA bitrev_size8192_mixed24_f64<>+0x14E8(SB)/8, $3792
DATA bitrev_size8192_mixed24_f64<>+0x14F0(SB)/8, $5840
DATA bitrev_size8192_mixed24_f64<>+0x14F8(SB)/8, $7888
DATA bitrev_size8192_mixed24_f64<>+0x1500(SB)/8, $336
DATA bitrev_size8192_mixed24_f64<>+0x1508(SB)/8, $2384
DATA bitrev_size8192_mixed24_f64<>+0x1510(SB)/8, $4432
DATA bitrev_size8192_mixed24_f64<>+0x1518(SB)/8, $6480
DATA bitrev_size8192_mixed24_f64<>+0x1520(SB)/8, $848
DATA bitrev_size8192_mixed24_f64<>+0x1528(SB)/8, $2896
DATA bitrev_size8192_mixed24_f64<>+0x1530(SB)/8, $4944
DATA bitrev_size8192_mixed24_f64<>+0x1538(SB)/8, $6992
DATA bitrev_size8192_mixed24_f64<>+0x1540(SB)/8, $1360
DATA bitrev_size8192_mixed24_f64<>+0x1548(SB)/8, $3408
DATA bitrev_size8192_mixed24_f64<>+0x1550(SB)/8, $5456
DATA bitrev_size8192_mixed24_f64<>+0x1558(SB)/8, $7504
DATA bitrev_size8192_mixed24_f64<>+0x1560(SB)/8, $1872
DATA bitrev_size8192_mixed24_f64<>+0x1568(SB)/8, $3920
DATA bitrev_size8192_mixed24_f64<>+0x1570(SB)/8, $5968
DATA bitrev_size8192_mixed24_f64<>+0x1578(SB)/8, $8016
DATA bitrev_size8192_mixed24_f64<>+0x1580(SB)/8, $464
DATA bitrev_size8192_mixed24_f64<>+0x1588(SB)/8, $2512
DATA bitrev_size8192_mixed24_f64<>+0x1590(SB)/8, $4560
DATA bitrev_size8192_mixed24_f64<>+0x1598(SB)/8, $6608
DATA bitrev_size8192_mixed24_f64<>+0x15A0(SB)/8, $976
DATA bitrev_size8192_mixed24_f64<>+0x15A8(SB)/8, $3024
DATA bitrev_size8192_mixed24_f64<>+0x15B0(SB)/8, $5072
DATA bitrev_size8192_mixed24_f64<>+0x15B8(SB)/8, $7120
DATA bitrev_size8192_mixed24_f64<>+0x15C0(SB)/8, $1488
DATA bitrev_size8192_mixed24_f64<>+0x15C8(SB)/8, $3536
DATA bitrev_size8192_mixed24_f64<>+0x15D0(SB)/8, $5584
DATA bitrev_size8192_mixed24_f64<>+0x15D8(SB)/8, $7632
DATA bitrev_size8192_mixed24_f64<>+0x15E0(SB)/8, $2000
DATA bitrev_size8192_mixed24_f64<>+0x15E8(SB)/8, $4048
DATA bitrev_size8192_mixed24_f64<>+0x15F0(SB)/8, $6096
DATA bitrev_size8192_mixed24_f64<>+0x15F8(SB)/8, $8144
DATA bitrev_size8192_mixed24_f64<>+0x1600(SB)/8, $112
DATA bitrev_size8192_mixed24_f64<>+0x1608(SB)/8, $2160
DATA bitrev_size8192_mixed24_f64<>+0x1610(SB)/8, $4208
DATA bitrev_size8192_mixed24_f64<>+0x1618(SB)/8, $6256
DATA bitrev_size8192_mixed24_f64<>+0x1620(SB)/8, $624
DATA bitrev_size8192_mixed24_f64<>+0x1628(SB)/8, $2672
DATA bitrev_size8192_mixed24_f64<>+0x1630(SB)/8, $4720
DATA bitrev_size8192_mixed24_f64<>+0x1638(SB)/8, $6768
DATA bitrev_size8192_mixed24_f64<>+0x1640(SB)/8, $1136
DATA bitrev_size8192_mixed24_f64<>+0x1648(SB)/8, $3184
DATA bitrev_size8192_mixed24_f64<>+0x1650(SB)/8, $5232
DATA bitrev_size8192_mixed24_f64<>+0x1658(SB)/8, $7280
DATA bitrev_size8192_mixed24_f64<>+0x1660(SB)/8, $1648
DATA bitrev_size8192_mixed24_f64<>+0x1668(SB)/8, $3696
DATA bitrev_size8192_mixed24_f64<>+0x1670(SB)/8, $5744
DATA bitrev_size8192_mixed24_f64<>+0x1678(SB)/8, $7792
DATA bitrev_size8192_mixed24_f64<>+0x1680(SB)/8, $240
DATA bitrev_size8192_mixed24_f64<>+0x1688(SB)/8, $2288
DATA bitrev_size8192_mixed24_f64<>+0x1690(SB)/8, $4336
DATA bitrev_size8192_mixed24_f64<>+0x1698(SB)/8, $6384
DATA bitrev_size8192_mixed24_f64<>+0x16A0(SB)/8, $752
DATA bitrev_size8192_mixed24_f64<>+0x16A8(SB)/8, $2800
DATA bitrev_size8192_mixed24_f64<>+0x16B0(SB)/8, $4848
DATA bitrev_size8192_mixed24_f64<>+0x16B8(SB)/8, $6896
DATA bitrev_size8192_mixed24_f64<>+0x16C0(SB)/8, $1264
DATA bitrev_size8192_mixed24_f64<>+0x16C8(SB)/8, $3312
DATA bitrev_size8192_mixed24_f64<>+0x16D0(SB)/8, $5360
DATA bitrev_size8192_mixed24_f64<>+0x16D8(SB)/8, $7408
DATA bitrev_size8192_mixed24_f64<>+0x16E0(SB)/8, $1776
DATA bitrev_size8192_mixed24_f64<>+0x16E8(SB)/8, $3824
DATA bitrev_size8192_mixed24_f64<>+0x16F0(SB)/8, $5872
DATA bitrev_size8192_mixed24_f64<>+0x16F8(SB)/8, $7920
DATA bitrev_size8192_mixed24_f64<>+0x1700(SB)/8, $368
DATA bitrev_size8192_mixed24_f64<>+0x1708(SB)/8, $2416
DATA bitrev_size8192_mixed24_f64<>+0x1710(SB)/8, $4464
DATA bitrev_size8192_mixed24_f64<>+0x1718(SB)/8, $6512
DATA bitrev_size8192_mixed24_f64<>+0x1720(SB)/8, $880
DATA bitrev_size8192_mixed24_f64<>+0x1728(SB)/8, $2928
DATA bitrev_size8192_mixed24_f64<>+0x1730(SB)/8, $4976
DATA bitrev_size8192_mixed24_f64<>+0x1738(SB)/8, $7024
DATA bitrev_size8192_mixed24_f64<>+0x1740(SB)/8, $1392
DATA bitrev_size8192_mixed24_f64<>+0x1748(SB)/8, $3440
DATA bitrev_size8192_mixed24_f64<>+0x1750(SB)/8, $5488
DATA bitrev_size8192_mixed24_f64<>+0x1758(SB)/8, $7536
DATA bitrev_size8192_mixed24_f64<>+0x1760(SB)/8, $1904
DATA bitrev_size8192_mixed24_f64<>+0x1768(SB)/8, $3952
DATA bitrev_size8192_mixed24_f64<>+0x1770(SB)/8, $6000
DATA bitrev_size8192_mixed24_f64<>+0x1778(SB)/8, $8048
DATA bitrev_size8192_mixed24_f64<>+0x1780(SB)/8, $496
DATA bitrev_size8192_mixed24_f64<>+0x1788(SB)/8, $2544
DATA bitrev_size8192_mixed24_f64<>+0x1790(SB)/8, $4592
DATA bitrev_size8192_mixed24_f64<>+0x1798(SB)/8, $6640
DATA bitrev_size8192_mixed24_f64<>+0x17A0(SB)/8, $1008
DATA bitrev_size8192_mixed24_f64<>+0x17A8(SB)/8, $3056
DATA bitrev_size8192_mixed24_f64<>+0x17B0(SB)/8, $5104
DATA bitrev_size8192_mixed24_f64<>+0x17B8(SB)/8, $7152
DATA bitrev_size8192_mixed24_f64<>+0x17C0(SB)/8, $1520
DATA bitrev_size8192_mixed24_f64<>+0x17C8(SB)/8, $3568
DATA bitrev_size8192_mixed24_f64<>+0x17D0(SB)/8, $5616
DATA bitrev_size8192_mixed24_f64<>+0x17D8(SB)/8, $7664
DATA bitrev_size8192_mixed24_f64<>+0x17E0(SB)/8, $2032
DATA bitrev_size8192_mixed24_f64<>+0x17E8(SB)/8, $4080
DATA bitrev_size8192_mixed24_f64<>+0x17F0(SB)/8, $6128
DATA bitrev_size8192_mixed24_f64<>+0x17F8(SB)/8, $8176
DATA bitrev_size8192_mixed24_f64<>+0x1800(SB)/8, $24
DATA bitrev_size8192_mixed24_f64<>+0x1808(SB)/8, $2072
DATA bitrev_size8192_mixed24_f64<>+0x1810(SB)/8, $4120
DATA bitrev_size8192_mixed24_f64<>+0x1818(SB)/8, $6168
DATA bitrev_size8192_mixed24_f64<>+0x1820(SB)/8, $536
DATA bitrev_size8192_mixed24_f64<>+0x1828(SB)/8, $2584
DATA bitrev_size8192_mixed24_f64<>+0x1830(SB)/8, $4632
DATA bitrev_size8192_mixed24_f64<>+0x1838(SB)/8, $6680
DATA bitrev_size8192_mixed24_f64<>+0x1840(SB)/8, $1048
DATA bitrev_size8192_mixed24_f64<>+0x1848(SB)/8, $3096
DATA bitrev_size8192_mixed24_f64<>+0x1850(SB)/8, $5144
DATA bitrev_size8192_mixed24_f64<>+0x1858(SB)/8, $7192
DATA bitrev_size8192_mixed24_f64<>+0x1860(SB)/8, $1560
DATA bitrev_size8192_mixed24_f64<>+0x1868(SB)/8, $3608
DATA bitrev_size8192_mixed24_f64<>+0x1870(SB)/8, $5656
DATA bitrev_size8192_mixed24_f64<>+0x1878(SB)/8, $7704
DATA bitrev_size8192_mixed24_f64<>+0x1880(SB)/8, $152
DATA bitrev_size8192_mixed24_f64<>+0x1888(SB)/8, $2200
DATA bitrev_size8192_mixed24_f64<>+0x1890(SB)/8, $4248
DATA bitrev_size8192_mixed24_f64<>+0x1898(SB)/8, $6296
DATA bitrev_size8192_mixed24_f64<>+0x18A0(SB)/8, $664
DATA bitrev_size8192_mixed24_f64<>+0x18A8(SB)/8, $2712
DATA bitrev_size8192_mixed24_f64<>+0x18B0(SB)/8, $4760
DATA bitrev_size8192_mixed24_f64<>+0x18B8(SB)/8, $6808
DATA bitrev_size8192_mixed24_f64<>+0x18C0(SB)/8, $1176
DATA bitrev_size8192_mixed24_f64<>+0x18C8(SB)/8, $3224
DATA bitrev_size8192_mixed24_f64<>+0x18D0(SB)/8, $5272
DATA bitrev_size8192_mixed24_f64<>+0x18D8(SB)/8, $7320
DATA bitrev_size8192_mixed24_f64<>+0x18E0(SB)/8, $1688
DATA bitrev_size8192_mixed24_f64<>+0x18E8(SB)/8, $3736
DATA bitrev_size8192_mixed24_f64<>+0x18F0(SB)/8, $5784
DATA bitrev_size8192_mixed24_f64<>+0x18F8(SB)/8, $7832
DATA bitrev_size8192_mixed24_f64<>+0x1900(SB)/8, $280
DATA bitrev_size8192_mixed24_f64<>+0x1908(SB)/8, $2328
DATA bitrev_size8192_mixed24_f64<>+0x1910(SB)/8, $4376
DATA bitrev_size8192_mixed24_f64<>+0x1918(SB)/8, $6424
DATA bitrev_size8192_mixed24_f64<>+0x1920(SB)/8, $792
DATA bitrev_size8192_mixed24_f64<>+0x1928(SB)/8, $2840
DATA bitrev_size8192_mixed24_f64<>+0x1930(SB)/8, $4888
DATA bitrev_size8192_mixed24_f64<>+0x1938(SB)/8, $6936
DATA bitrev_size8192_mixed24_f64<>+0x1940(SB)/8, $1304
DATA bitrev_size8192_mixed24_f64<>+0x1948(SB)/8, $3352
DATA bitrev_size8192_mixed24_f64<>+0x1950(SB)/8, $5400
DATA bitrev_size8192_mixed24_f64<>+0x1958(SB)/8, $7448
DATA bitrev_size8192_mixed24_f64<>+0x1960(SB)/8, $1816
DATA bitrev_size8192_mixed24_f64<>+0x1968(SB)/8, $3864
DATA bitrev_size8192_mixed24_f64<>+0x1970(SB)/8, $5912
DATA bitrev_size8192_mixed24_f64<>+0x1978(SB)/8, $7960
DATA bitrev_size8192_mixed24_f64<>+0x1980(SB)/8, $408
DATA bitrev_size8192_mixed24_f64<>+0x1988(SB)/8, $2456
DATA bitrev_size8192_mixed24_f64<>+0x1990(SB)/8, $4504
DATA bitrev_size8192_mixed24_f64<>+0x1998(SB)/8, $6552
DATA bitrev_size8192_mixed24_f64<>+0x19A0(SB)/8, $920
DATA bitrev_size8192_mixed24_f64<>+0x19A8(SB)/8, $2968
DATA bitrev_size8192_mixed24_f64<>+0x19B0(SB)/8, $5016
DATA bitrev_size8192_mixed24_f64<>+0x19B8(SB)/8, $7064
DATA bitrev_size8192_mixed24_f64<>+0x19C0(SB)/8, $1432
DATA bitrev_size8192_mixed24_f64<>+0x19C8(SB)/8, $3480
DATA bitrev_size8192_mixed24_f64<>+0x19D0(SB)/8, $5528
DATA bitrev_size8192_mixed24_f64<>+0x19D8(SB)/8, $7576
DATA bitrev_size8192_mixed24_f64<>+0x19E0(SB)/8, $1944
DATA bitrev_size8192_mixed24_f64<>+0x19E8(SB)/8, $3992
DATA bitrev_size8192_mixed24_f64<>+0x19F0(SB)/8, $6040
DATA bitrev_size8192_mixed24_f64<>+0x19F8(SB)/8, $8088
DATA bitrev_size8192_mixed24_f64<>+0x1A00(SB)/8, $56
DATA bitrev_size8192_mixed24_f64<>+0x1A08(SB)/8, $2104
DATA bitrev_size8192_mixed24_f64<>+0x1A10(SB)/8, $4152
DATA bitrev_size8192_mixed24_f64<>+0x1A18(SB)/8, $6200
DATA bitrev_size8192_mixed24_f64<>+0x1A20(SB)/8, $568
DATA bitrev_size8192_mixed24_f64<>+0x1A28(SB)/8, $2616
DATA bitrev_size8192_mixed24_f64<>+0x1A30(SB)/8, $4664
DATA bitrev_size8192_mixed24_f64<>+0x1A38(SB)/8, $6712
DATA bitrev_size8192_mixed24_f64<>+0x1A40(SB)/8, $1080
DATA bitrev_size8192_mixed24_f64<>+0x1A48(SB)/8, $3128
DATA bitrev_size8192_mixed24_f64<>+0x1A50(SB)/8, $5176
DATA bitrev_size8192_mixed24_f64<>+0x1A58(SB)/8, $7224
DATA bitrev_size8192_mixed24_f64<>+0x1A60(SB)/8, $1592
DATA bitrev_size8192_mixed24_f64<>+0x1A68(SB)/8, $3640
DATA bitrev_size8192_mixed24_f64<>+0x1A70(SB)/8, $5688
DATA bitrev_size8192_mixed24_f64<>+0x1A78(SB)/8, $7736
DATA bitrev_size8192_mixed24_f64<>+0x1A80(SB)/8, $184
DATA bitrev_size8192_mixed24_f64<>+0x1A88(SB)/8, $2232
DATA bitrev_size8192_mixed24_f64<>+0x1A90(SB)/8, $4280
DATA bitrev_size8192_mixed24_f64<>+0x1A98(SB)/8, $6328
DATA bitrev_size8192_mixed24_f64<>+0x1AA0(SB)/8, $696
DATA bitrev_size8192_mixed24_f64<>+0x1AA8(SB)/8, $2744
DATA bitrev_size8192_mixed24_f64<>+0x1AB0(SB)/8, $4792
DATA bitrev_size8192_mixed24_f64<>+0x1AB8(SB)/8, $6840
DATA bitrev_size8192_mixed24_f64<>+0x1AC0(SB)/8, $1208
DATA bitrev_size8192_mixed24_f64<>+0x1AC8(SB)/8, $3256
DATA bitrev_size8192_mixed24_f64<>+0x1AD0(SB)/8, $5304
DATA bitrev_size8192_mixed24_f64<>+0x1AD8(SB)/8, $7352
DATA bitrev_size8192_mixed24_f64<>+0x1AE0(SB)/8, $1720
DATA bitrev_size8192_mixed24_f64<>+0x1AE8(SB)/8, $3768
DATA bitrev_size8192_mixed24_f64<>+0x1AF0(SB)/8, $5816
DATA bitrev_size8192_mixed24_f64<>+0x1AF8(SB)/8, $7864
DATA bitrev_size8192_mixed24_f64<>+0x1B00(SB)/8, $312
DATA bitrev_size8192_mixed24_f64<>+0x1B08(SB)/8, $2360
DATA bitrev_size8192_mixed24_f64<>+0x1B10(SB)/8, $4408
DATA bitrev_size8192_mixed24_f64<>+0x1B18(SB)/8, $6456
DATA bitrev_size8192_mixed24_f64<>+0x1B20(SB)/8, $824
DATA bitrev_size8192_mixed24_f64<>+0x1B28(SB)/8, $2872
DATA bitrev_size8192_mixed24_f64<>+0x1B30(SB)/8, $4920
DATA bitrev_size8192_mixed24_f64<>+0x1B38(SB)/8, $6968
DATA bitrev_size8192_mixed24_f64<>+0x1B40(SB)/8, $1336
DATA bitrev_size8192_mixed24_f64<>+0x1B48(SB)/8, $3384
DATA bitrev_size8192_mixed24_f64<>+0x1B50(SB)/8, $5432
DATA bitrev_size8192_mixed24_f64<>+0x1B58(SB)/8, $7480
DATA bitrev_size8192_mixed24_f64<>+0x1B60(SB)/8, $1848
DATA bitrev_size8192_mixed24_f64<>+0x1B68(SB)/8, $3896
DATA bitrev_size8192_mixed24_f64<>+0x1B70(SB)/8, $5944
DATA bitrev_size8192_mixed24_f64<>+0x1B78(SB)/8, $7992
DATA bitrev_size8192_mixed24_f64<>+0x1B80(SB)/8, $440
DATA bitrev_size8192_mixed24_f64<>+0x1B88(SB)/8, $2488
DATA bitrev_size8192_mixed24_f64<>+0x1B90(SB)/8, $4536
DATA bitrev_size8192_mixed24_f64<>+0x1B98(SB)/8, $6584
DATA bitrev_size8192_mixed24_f64<>+0x1BA0(SB)/8, $952
DATA bitrev_size8192_mixed24_f64<>+0x1BA8(SB)/8, $3000
DATA bitrev_size8192_mixed24_f64<>+0x1BB0(SB)/8, $5048
DATA bitrev_size8192_mixed24_f64<>+0x1BB8(SB)/8, $7096
DATA bitrev_size8192_mixed24_f64<>+0x1BC0(SB)/8, $1464
DATA bitrev_size8192_mixed24_f64<>+0x1BC8(SB)/8, $3512
DATA bitrev_size8192_mixed24_f64<>+0x1BD0(SB)/8, $5560
DATA bitrev_size8192_mixed24_f64<>+0x1BD8(SB)/8, $7608
DATA bitrev_size8192_mixed24_f64<>+0x1BE0(SB)/8, $1976
DATA bitrev_size8192_mixed24_f64<>+0x1BE8(SB)/8, $4024
DATA bitrev_size8192_mixed24_f64<>+0x1BF0(SB)/8, $6072
DATA bitrev_size8192_mixed24_f64<>+0x1BF8(SB)/8, $8120
DATA bitrev_size8192_mixed24_f64<>+0x1C00(SB)/8, $88
DATA bitrev_size8192_mixed24_f64<>+0x1C08(SB)/8, $2136
DATA bitrev_size8192_mixed24_f64<>+0x1C10(SB)/8, $4184
DATA bitrev_size8192_mixed24_f64<>+0x1C18(SB)/8, $6232
DATA bitrev_size8192_mixed24_f64<>+0x1C20(SB)/8, $600
DATA bitrev_size8192_mixed24_f64<>+0x1C28(SB)/8, $2648
DATA bitrev_size8192_mixed24_f64<>+0x1C30(SB)/8, $4696
DATA bitrev_size8192_mixed24_f64<>+0x1C38(SB)/8, $6744
DATA bitrev_size8192_mixed24_f64<>+0x1C40(SB)/8, $1112
DATA bitrev_size8192_mixed24_f64<>+0x1C48(SB)/8, $3160
DATA bitrev_size8192_mixed24_f64<>+0x1C50(SB)/8, $5208
DATA bitrev_size8192_mixed24_f64<>+0x1C58(SB)/8, $7256
DATA bitrev_size8192_mixed24_f64<>+0x1C60(SB)/8, $1624
DATA bitrev_size8192_mixed24_f64<>+0x1C68(SB)/8, $3672
DATA bitrev_size8192_mixed24_f64<>+0x1C70(SB)/8, $5720
DATA bitrev_size8192_mixed24_f64<>+0x1C78(SB)/8, $7768
DATA bitrev_size8192_mixed24_f64<>+0x1C80(SB)/8, $216
DATA bitrev_size8192_mixed24_f64<>+0x1C88(SB)/8, $2264
DATA bitrev_size8192_mixed24_f64<>+0x1C90(SB)/8, $4312
DATA bitrev_size8192_mixed24_f64<>+0x1C98(SB)/8, $6360
DATA bitrev_size8192_mixed24_f64<>+0x1CA0(SB)/8, $728
DATA bitrev_size8192_mixed24_f64<>+0x1CA8(SB)/8, $2776
DATA bitrev_size8192_mixed24_f64<>+0x1CB0(SB)/8, $4824
DATA bitrev_size8192_mixed24_f64<>+0x1CB8(SB)/8, $6872
DATA bitrev_size8192_mixed24_f64<>+0x1CC0(SB)/8, $1240
DATA bitrev_size8192_mixed24_f64<>+0x1CC8(SB)/8, $3288
DATA bitrev_size8192_mixed24_f64<>+0x1CD0(SB)/8, $5336
DATA bitrev_size8192_mixed24_f64<>+0x1CD8(SB)/8, $7384
DATA bitrev_size8192_mixed24_f64<>+0x1CE0(SB)/8, $1752
DATA bitrev_size8192_mixed24_f64<>+0x1CE8(SB)/8, $3800
DATA bitrev_size8192_mixed24_f64<>+0x1CF0(SB)/8, $5848
DATA bitrev_size8192_mixed24_f64<>+0x1CF8(SB)/8, $7896
DATA bitrev_size8192_mixed24_f64<>+0x1D00(SB)/8, $344
DATA bitrev_size8192_mixed24_f64<>+0x1D08(SB)/8, $2392
DATA bitrev_size8192_mixed24_f64<>+0x1D10(SB)/8, $4440
DATA bitrev_size8192_mixed24_f64<>+0x1D18(SB)/8, $6488
DATA bitrev_size8192_mixed24_f64<>+0x1D20(SB)/8, $856
DATA bitrev_size8192_mixed24_f64<>+0x1D28(SB)/8, $2904
DATA bitrev_size8192_mixed24_f64<>+0x1D30(SB)/8, $4952
DATA bitrev_size8192_mixed24_f64<>+0x1D38(SB)/8, $7000
DATA bitrev_size8192_mixed24_f64<>+0x1D40(SB)/8, $1368
DATA bitrev_size8192_mixed24_f64<>+0x1D48(SB)/8, $3416
DATA bitrev_size8192_mixed24_f64<>+0x1D50(SB)/8, $5464
DATA bitrev_size8192_mixed24_f64<>+0x1D58(SB)/8, $7512
DATA bitrev_size8192_mixed24_f64<>+0x1D60(SB)/8, $1880
DATA bitrev_size8192_mixed24_f64<>+0x1D68(SB)/8, $3928
DATA bitrev_size8192_mixed24_f64<>+0x1D70(SB)/8, $5976
DATA bitrev_size8192_mixed24_f64<>+0x1D78(SB)/8, $8024
DATA bitrev_size8192_mixed24_f64<>+0x1D80(SB)/8, $472
DATA bitrev_size8192_mixed24_f64<>+0x1D88(SB)/8, $2520
DATA bitrev_size8192_mixed24_f64<>+0x1D90(SB)/8, $4568
DATA bitrev_size8192_mixed24_f64<>+0x1D98(SB)/8, $6616
DATA bitrev_size8192_mixed24_f64<>+0x1DA0(SB)/8, $984
DATA bitrev_size8192_mixed24_f64<>+0x1DA8(SB)/8, $3032
DATA bitrev_size8192_mixed24_f64<>+0x1DB0(SB)/8, $5080
DATA bitrev_size8192_mixed24_f64<>+0x1DB8(SB)/8, $7128
DATA bitrev_size8192_mixed24_f64<>+0x1DC0(SB)/8, $1496
DATA bitrev_size8192_mixed24_f64<>+0x1DC8(SB)/8, $3544
DATA bitrev_size8192_mixed24_f64<>+0x1DD0(SB)/8, $5592
DATA bitrev_size8192_mixed24_f64<>+0x1DD8(SB)/8, $7640
DATA bitrev_size8192_mixed24_f64<>+0x1DE0(SB)/8, $2008
DATA bitrev_size8192_mixed24_f64<>+0x1DE8(SB)/8, $4056
DATA bitrev_size8192_mixed24_f64<>+0x1DF0(SB)/8, $6104
DATA bitrev_size8192_mixed24_f64<>+0x1DF8(SB)/8, $8152
DATA bitrev_size8192_mixed24_f64<>+0x1E00(SB)/8, $120
DATA bitrev_size8192_mixed24_f64<>+0x1E08(SB)/8, $2168
DATA bitrev_size8192_mixed24_f64<>+0x1E10(SB)/8, $4216
DATA bitrev_size8192_mixed24_f64<>+0x1E18(SB)/8, $6264
DATA bitrev_size8192_mixed24_f64<>+0x1E20(SB)/8, $632
DATA bitrev_size8192_mixed24_f64<>+0x1E28(SB)/8, $2680
DATA bitrev_size8192_mixed24_f64<>+0x1E30(SB)/8, $4728
DATA bitrev_size8192_mixed24_f64<>+0x1E38(SB)/8, $6776
DATA bitrev_size8192_mixed24_f64<>+0x1E40(SB)/8, $1144
DATA bitrev_size8192_mixed24_f64<>+0x1E48(SB)/8, $3192
DATA bitrev_size8192_mixed24_f64<>+0x1E50(SB)/8, $5240
DATA bitrev_size8192_mixed24_f64<>+0x1E58(SB)/8, $7288
DATA bitrev_size8192_mixed24_f64<>+0x1E60(SB)/8, $1656
DATA bitrev_size8192_mixed24_f64<>+0x1E68(SB)/8, $3704
DATA bitrev_size8192_mixed24_f64<>+0x1E70(SB)/8, $5752
DATA bitrev_size8192_mixed24_f64<>+0x1E78(SB)/8, $7800
DATA bitrev_size8192_mixed24_f64<>+0x1E80(SB)/8, $248
DATA bitrev_size8192_mixed24_f64<>+0x1E88(SB)/8, $2296
DATA bitrev_size8192_mixed24_f64<>+0x1E90(SB)/8, $4344
DATA bitrev_size8192_mixed24_f64<>+0x1E98(SB)/8, $6392
DATA bitrev_size8192_mixed24_f64<>+0x1EA0(SB)/8, $760
DATA bitrev_size8192_mixed24_f64<>+0x1EA8(SB)/8, $2808
DATA bitrev_size8192_mixed24_f64<>+0x1EB0(SB)/8, $4856
DATA bitrev_size8192_mixed24_f64<>+0x1EB8(SB)/8, $6904
DATA bitrev_size8192_mixed24_f64<>+0x1EC0(SB)/8, $1272
DATA bitrev_size8192_mixed24_f64<>+0x1EC8(SB)/8, $3320
DATA bitrev_size8192_mixed24_f64<>+0x1ED0(SB)/8, $5368
DATA bitrev_size8192_mixed24_f64<>+0x1ED8(SB)/8, $7416
DATA bitrev_size8192_mixed24_f64<>+0x1EE0(SB)/8, $1784
DATA bitrev_size8192_mixed24_f64<>+0x1EE8(SB)/8, $3832
DATA bitrev_size8192_mixed24_f64<>+0x1EF0(SB)/8, $5880
DATA bitrev_size8192_mixed24_f64<>+0x1EF8(SB)/8, $7928
DATA bitrev_size8192_mixed24_f64<>+0x1F00(SB)/8, $376
DATA bitrev_size8192_mixed24_f64<>+0x1F08(SB)/8, $2424
DATA bitrev_size8192_mixed24_f64<>+0x1F10(SB)/8, $4472
DATA bitrev_size8192_mixed24_f64<>+0x1F18(SB)/8, $6520
DATA bitrev_size8192_mixed24_f64<>+0x1F20(SB)/8, $888
DATA bitrev_size8192_mixed24_f64<>+0x1F28(SB)/8, $2936
DATA bitrev_size8192_mixed24_f64<>+0x1F30(SB)/8, $4984
DATA bitrev_size8192_mixed24_f64<>+0x1F38(SB)/8, $7032
DATA bitrev_size8192_mixed24_f64<>+0x1F40(SB)/8, $1400
DATA bitrev_size8192_mixed24_f64<>+0x1F48(SB)/8, $3448
DATA bitrev_size8192_mixed24_f64<>+0x1F50(SB)/8, $5496
DATA bitrev_size8192_mixed24_f64<>+0x1F58(SB)/8, $7544
DATA bitrev_size8192_mixed24_f64<>+0x1F60(SB)/8, $1912
DATA bitrev_size8192_mixed24_f64<>+0x1F68(SB)/8, $3960
DATA bitrev_size8192_mixed24_f64<>+0x1F70(SB)/8, $6008
DATA bitrev_size8192_mixed24_f64<>+0x1F78(SB)/8, $8056
DATA bitrev_size8192_mixed24_f64<>+0x1F80(SB)/8, $504
DATA bitrev_size8192_mixed24_f64<>+0x1F88(SB)/8, $2552
DATA bitrev_size8192_mixed24_f64<>+0x1F90(SB)/8, $4600
DATA bitrev_size8192_mixed24_f64<>+0x1F98(SB)/8, $6648
DATA bitrev_size8192_mixed24_f64<>+0x1FA0(SB)/8, $1016
DATA bitrev_size8192_mixed24_f64<>+0x1FA8(SB)/8, $3064
DATA bitrev_size8192_mixed24_f64<>+0x1FB0(SB)/8, $5112
DATA bitrev_size8192_mixed24_f64<>+0x1FB8(SB)/8, $7160
DATA bitrev_size8192_mixed24_f64<>+0x1FC0(SB)/8, $1528
DATA bitrev_size8192_mixed24_f64<>+0x1FC8(SB)/8, $3576
DATA bitrev_size8192_mixed24_f64<>+0x1FD0(SB)/8, $5624
DATA bitrev_size8192_mixed24_f64<>+0x1FD8(SB)/8, $7672
DATA bitrev_size8192_mixed24_f64<>+0x1FE0(SB)/8, $2040
DATA bitrev_size8192_mixed24_f64<>+0x1FE8(SB)/8, $4088
DATA bitrev_size8192_mixed24_f64<>+0x1FF0(SB)/8, $6136
DATA bitrev_size8192_mixed24_f64<>+0x1FF8(SB)/8, $8184
DATA bitrev_size8192_mixed24_f64<>+0x2000(SB)/8, $2
DATA bitrev_size8192_mixed24_f64<>+0x2008(SB)/8, $2050
DATA bitrev_size8192_mixed24_f64<>+0x2010(SB)/8, $4098
DATA bitrev_size8192_mixed24_f64<>+0x2018(SB)/8, $6146
DATA bitrev_size8192_mixed24_f64<>+0x2020(SB)/8, $514
DATA bitrev_size8192_mixed24_f64<>+0x2028(SB)/8, $2562
DATA bitrev_size8192_mixed24_f64<>+0x2030(SB)/8, $4610
DATA bitrev_size8192_mixed24_f64<>+0x2038(SB)/8, $6658
DATA bitrev_size8192_mixed24_f64<>+0x2040(SB)/8, $1026
DATA bitrev_size8192_mixed24_f64<>+0x2048(SB)/8, $3074
DATA bitrev_size8192_mixed24_f64<>+0x2050(SB)/8, $5122
DATA bitrev_size8192_mixed24_f64<>+0x2058(SB)/8, $7170
DATA bitrev_size8192_mixed24_f64<>+0x2060(SB)/8, $1538
DATA bitrev_size8192_mixed24_f64<>+0x2068(SB)/8, $3586
DATA bitrev_size8192_mixed24_f64<>+0x2070(SB)/8, $5634
DATA bitrev_size8192_mixed24_f64<>+0x2078(SB)/8, $7682
DATA bitrev_size8192_mixed24_f64<>+0x2080(SB)/8, $130
DATA bitrev_size8192_mixed24_f64<>+0x2088(SB)/8, $2178
DATA bitrev_size8192_mixed24_f64<>+0x2090(SB)/8, $4226
DATA bitrev_size8192_mixed24_f64<>+0x2098(SB)/8, $6274
DATA bitrev_size8192_mixed24_f64<>+0x20A0(SB)/8, $642
DATA bitrev_size8192_mixed24_f64<>+0x20A8(SB)/8, $2690
DATA bitrev_size8192_mixed24_f64<>+0x20B0(SB)/8, $4738
DATA bitrev_size8192_mixed24_f64<>+0x20B8(SB)/8, $6786
DATA bitrev_size8192_mixed24_f64<>+0x20C0(SB)/8, $1154
DATA bitrev_size8192_mixed24_f64<>+0x20C8(SB)/8, $3202
DATA bitrev_size8192_mixed24_f64<>+0x20D0(SB)/8, $5250
DATA bitrev_size8192_mixed24_f64<>+0x20D8(SB)/8, $7298
DATA bitrev_size8192_mixed24_f64<>+0x20E0(SB)/8, $1666
DATA bitrev_size8192_mixed24_f64<>+0x20E8(SB)/8, $3714
DATA bitrev_size8192_mixed24_f64<>+0x20F0(SB)/8, $5762
DATA bitrev_size8192_mixed24_f64<>+0x20F8(SB)/8, $7810
DATA bitrev_size8192_mixed24_f64<>+0x2100(SB)/8, $258
DATA bitrev_size8192_mixed24_f64<>+0x2108(SB)/8, $2306
DATA bitrev_size8192_mixed24_f64<>+0x2110(SB)/8, $4354
DATA bitrev_size8192_mixed24_f64<>+0x2118(SB)/8, $6402
DATA bitrev_size8192_mixed24_f64<>+0x2120(SB)/8, $770
DATA bitrev_size8192_mixed24_f64<>+0x2128(SB)/8, $2818
DATA bitrev_size8192_mixed24_f64<>+0x2130(SB)/8, $4866
DATA bitrev_size8192_mixed24_f64<>+0x2138(SB)/8, $6914
DATA bitrev_size8192_mixed24_f64<>+0x2140(SB)/8, $1282
DATA bitrev_size8192_mixed24_f64<>+0x2148(SB)/8, $3330
DATA bitrev_size8192_mixed24_f64<>+0x2150(SB)/8, $5378
DATA bitrev_size8192_mixed24_f64<>+0x2158(SB)/8, $7426
DATA bitrev_size8192_mixed24_f64<>+0x2160(SB)/8, $1794
DATA bitrev_size8192_mixed24_f64<>+0x2168(SB)/8, $3842
DATA bitrev_size8192_mixed24_f64<>+0x2170(SB)/8, $5890
DATA bitrev_size8192_mixed24_f64<>+0x2178(SB)/8, $7938
DATA bitrev_size8192_mixed24_f64<>+0x2180(SB)/8, $386
DATA bitrev_size8192_mixed24_f64<>+0x2188(SB)/8, $2434
DATA bitrev_size8192_mixed24_f64<>+0x2190(SB)/8, $4482
DATA bitrev_size8192_mixed24_f64<>+0x2198(SB)/8, $6530
DATA bitrev_size8192_mixed24_f64<>+0x21A0(SB)/8, $898
DATA bitrev_size8192_mixed24_f64<>+0x21A8(SB)/8, $2946
DATA bitrev_size8192_mixed24_f64<>+0x21B0(SB)/8, $4994
DATA bitrev_size8192_mixed24_f64<>+0x21B8(SB)/8, $7042
DATA bitrev_size8192_mixed24_f64<>+0x21C0(SB)/8, $1410
DATA bitrev_size8192_mixed24_f64<>+0x21C8(SB)/8, $3458
DATA bitrev_size8192_mixed24_f64<>+0x21D0(SB)/8, $5506
DATA bitrev_size8192_mixed24_f64<>+0x21D8(SB)/8, $7554
DATA bitrev_size8192_mixed24_f64<>+0x21E0(SB)/8, $1922
DATA bitrev_size8192_mixed24_f64<>+0x21E8(SB)/8, $3970
DATA bitrev_size8192_mixed24_f64<>+0x21F0(SB)/8, $6018
DATA bitrev_size8192_mixed24_f64<>+0x21F8(SB)/8, $8066
DATA bitrev_size8192_mixed24_f64<>+0x2200(SB)/8, $34
DATA bitrev_size8192_mixed24_f64<>+0x2208(SB)/8, $2082
DATA bitrev_size8192_mixed24_f64<>+0x2210(SB)/8, $4130
DATA bitrev_size8192_mixed24_f64<>+0x2218(SB)/8, $6178
DATA bitrev_size8192_mixed24_f64<>+0x2220(SB)/8, $546
DATA bitrev_size8192_mixed24_f64<>+0x2228(SB)/8, $2594
DATA bitrev_size8192_mixed24_f64<>+0x2230(SB)/8, $4642
DATA bitrev_size8192_mixed24_f64<>+0x2238(SB)/8, $6690
DATA bitrev_size8192_mixed24_f64<>+0x2240(SB)/8, $1058
DATA bitrev_size8192_mixed24_f64<>+0x2248(SB)/8, $3106
DATA bitrev_size8192_mixed24_f64<>+0x2250(SB)/8, $5154
DATA bitrev_size8192_mixed24_f64<>+0x2258(SB)/8, $7202
DATA bitrev_size8192_mixed24_f64<>+0x2260(SB)/8, $1570
DATA bitrev_size8192_mixed24_f64<>+0x2268(SB)/8, $3618
DATA bitrev_size8192_mixed24_f64<>+0x2270(SB)/8, $5666
DATA bitrev_size8192_mixed24_f64<>+0x2278(SB)/8, $7714
DATA bitrev_size8192_mixed24_f64<>+0x2280(SB)/8, $162
DATA bitrev_size8192_mixed24_f64<>+0x2288(SB)/8, $2210
DATA bitrev_size8192_mixed24_f64<>+0x2290(SB)/8, $4258
DATA bitrev_size8192_mixed24_f64<>+0x2298(SB)/8, $6306
DATA bitrev_size8192_mixed24_f64<>+0x22A0(SB)/8, $674
DATA bitrev_size8192_mixed24_f64<>+0x22A8(SB)/8, $2722
DATA bitrev_size8192_mixed24_f64<>+0x22B0(SB)/8, $4770
DATA bitrev_size8192_mixed24_f64<>+0x22B8(SB)/8, $6818
DATA bitrev_size8192_mixed24_f64<>+0x22C0(SB)/8, $1186
DATA bitrev_size8192_mixed24_f64<>+0x22C8(SB)/8, $3234
DATA bitrev_size8192_mixed24_f64<>+0x22D0(SB)/8, $5282
DATA bitrev_size8192_mixed24_f64<>+0x22D8(SB)/8, $7330
DATA bitrev_size8192_mixed24_f64<>+0x22E0(SB)/8, $1698
DATA bitrev_size8192_mixed24_f64<>+0x22E8(SB)/8, $3746
DATA bitrev_size8192_mixed24_f64<>+0x22F0(SB)/8, $5794
DATA bitrev_size8192_mixed24_f64<>+0x22F8(SB)/8, $7842
DATA bitrev_size8192_mixed24_f64<>+0x2300(SB)/8, $290
DATA bitrev_size8192_mixed24_f64<>+0x2308(SB)/8, $2338
DATA bitrev_size8192_mixed24_f64<>+0x2310(SB)/8, $4386
DATA bitrev_size8192_mixed24_f64<>+0x2318(SB)/8, $6434
DATA bitrev_size8192_mixed24_f64<>+0x2320(SB)/8, $802
DATA bitrev_size8192_mixed24_f64<>+0x2328(SB)/8, $2850
DATA bitrev_size8192_mixed24_f64<>+0x2330(SB)/8, $4898
DATA bitrev_size8192_mixed24_f64<>+0x2338(SB)/8, $6946
DATA bitrev_size8192_mixed24_f64<>+0x2340(SB)/8, $1314
DATA bitrev_size8192_mixed24_f64<>+0x2348(SB)/8, $3362
DATA bitrev_size8192_mixed24_f64<>+0x2350(SB)/8, $5410
DATA bitrev_size8192_mixed24_f64<>+0x2358(SB)/8, $7458
DATA bitrev_size8192_mixed24_f64<>+0x2360(SB)/8, $1826
DATA bitrev_size8192_mixed24_f64<>+0x2368(SB)/8, $3874
DATA bitrev_size8192_mixed24_f64<>+0x2370(SB)/8, $5922
DATA bitrev_size8192_mixed24_f64<>+0x2378(SB)/8, $7970
DATA bitrev_size8192_mixed24_f64<>+0x2380(SB)/8, $418
DATA bitrev_size8192_mixed24_f64<>+0x2388(SB)/8, $2466
DATA bitrev_size8192_mixed24_f64<>+0x2390(SB)/8, $4514
DATA bitrev_size8192_mixed24_f64<>+0x2398(SB)/8, $6562
DATA bitrev_size8192_mixed24_f64<>+0x23A0(SB)/8, $930
DATA bitrev_size8192_mixed24_f64<>+0x23A8(SB)/8, $2978
DATA bitrev_size8192_mixed24_f64<>+0x23B0(SB)/8, $5026
DATA bitrev_size8192_mixed24_f64<>+0x23B8(SB)/8, $7074
DATA bitrev_size8192_mixed24_f64<>+0x23C0(SB)/8, $1442
DATA bitrev_size8192_mixed24_f64<>+0x23C8(SB)/8, $3490
DATA bitrev_size8192_mixed24_f64<>+0x23D0(SB)/8, $5538
DATA bitrev_size8192_mixed24_f64<>+0x23D8(SB)/8, $7586
DATA bitrev_size8192_mixed24_f64<>+0x23E0(SB)/8, $1954
DATA bitrev_size8192_mixed24_f64<>+0x23E8(SB)/8, $4002
DATA bitrev_size8192_mixed24_f64<>+0x23F0(SB)/8, $6050
DATA bitrev_size8192_mixed24_f64<>+0x23F8(SB)/8, $8098
DATA bitrev_size8192_mixed24_f64<>+0x2400(SB)/8, $66
DATA bitrev_size8192_mixed24_f64<>+0x2408(SB)/8, $2114
DATA bitrev_size8192_mixed24_f64<>+0x2410(SB)/8, $4162
DATA bitrev_size8192_mixed24_f64<>+0x2418(SB)/8, $6210
DATA bitrev_size8192_mixed24_f64<>+0x2420(SB)/8, $578
DATA bitrev_size8192_mixed24_f64<>+0x2428(SB)/8, $2626
DATA bitrev_size8192_mixed24_f64<>+0x2430(SB)/8, $4674
DATA bitrev_size8192_mixed24_f64<>+0x2438(SB)/8, $6722
DATA bitrev_size8192_mixed24_f64<>+0x2440(SB)/8, $1090
DATA bitrev_size8192_mixed24_f64<>+0x2448(SB)/8, $3138
DATA bitrev_size8192_mixed24_f64<>+0x2450(SB)/8, $5186
DATA bitrev_size8192_mixed24_f64<>+0x2458(SB)/8, $7234
DATA bitrev_size8192_mixed24_f64<>+0x2460(SB)/8, $1602
DATA bitrev_size8192_mixed24_f64<>+0x2468(SB)/8, $3650
DATA bitrev_size8192_mixed24_f64<>+0x2470(SB)/8, $5698
DATA bitrev_size8192_mixed24_f64<>+0x2478(SB)/8, $7746
DATA bitrev_size8192_mixed24_f64<>+0x2480(SB)/8, $194
DATA bitrev_size8192_mixed24_f64<>+0x2488(SB)/8, $2242
DATA bitrev_size8192_mixed24_f64<>+0x2490(SB)/8, $4290
DATA bitrev_size8192_mixed24_f64<>+0x2498(SB)/8, $6338
DATA bitrev_size8192_mixed24_f64<>+0x24A0(SB)/8, $706
DATA bitrev_size8192_mixed24_f64<>+0x24A8(SB)/8, $2754
DATA bitrev_size8192_mixed24_f64<>+0x24B0(SB)/8, $4802
DATA bitrev_size8192_mixed24_f64<>+0x24B8(SB)/8, $6850
DATA bitrev_size8192_mixed24_f64<>+0x24C0(SB)/8, $1218
DATA bitrev_size8192_mixed24_f64<>+0x24C8(SB)/8, $3266
DATA bitrev_size8192_mixed24_f64<>+0x24D0(SB)/8, $5314
DATA bitrev_size8192_mixed24_f64<>+0x24D8(SB)/8, $7362
DATA bitrev_size8192_mixed24_f64<>+0x24E0(SB)/8, $1730
DATA bitrev_size8192_mixed24_f64<>+0x24E8(SB)/8, $3778
DATA bitrev_size8192_mixed24_f64<>+0x24F0(SB)/8, $5826
DATA bitrev_size8192_mixed24_f64<>+0x24F8(SB)/8, $7874
DATA bitrev_size8192_mixed24_f64<>+0x2500(SB)/8, $322
DATA bitrev_size8192_mixed24_f64<>+0x2508(SB)/8, $2370
DATA bitrev_size8192_mixed24_f64<>+0x2510(SB)/8, $4418
DATA bitrev_size8192_mixed24_f64<>+0x2518(SB)/8, $6466
DATA bitrev_size8192_mixed24_f64<>+0x2520(SB)/8, $834
DATA bitrev_size8192_mixed24_f64<>+0x2528(SB)/8, $2882
DATA bitrev_size8192_mixed24_f64<>+0x2530(SB)/8, $4930
DATA bitrev_size8192_mixed24_f64<>+0x2538(SB)/8, $6978
DATA bitrev_size8192_mixed24_f64<>+0x2540(SB)/8, $1346
DATA bitrev_size8192_mixed24_f64<>+0x2548(SB)/8, $3394
DATA bitrev_size8192_mixed24_f64<>+0x2550(SB)/8, $5442
DATA bitrev_size8192_mixed24_f64<>+0x2558(SB)/8, $7490
DATA bitrev_size8192_mixed24_f64<>+0x2560(SB)/8, $1858
DATA bitrev_size8192_mixed24_f64<>+0x2568(SB)/8, $3906
DATA bitrev_size8192_mixed24_f64<>+0x2570(SB)/8, $5954
DATA bitrev_size8192_mixed24_f64<>+0x2578(SB)/8, $8002
DATA bitrev_size8192_mixed24_f64<>+0x2580(SB)/8, $450
DATA bitrev_size8192_mixed24_f64<>+0x2588(SB)/8, $2498
DATA bitrev_size8192_mixed24_f64<>+0x2590(SB)/8, $4546
DATA bitrev_size8192_mixed24_f64<>+0x2598(SB)/8, $6594
DATA bitrev_size8192_mixed24_f64<>+0x25A0(SB)/8, $962
DATA bitrev_size8192_mixed24_f64<>+0x25A8(SB)/8, $3010
DATA bitrev_size8192_mixed24_f64<>+0x25B0(SB)/8, $5058
DATA bitrev_size8192_mixed24_f64<>+0x25B8(SB)/8, $7106
DATA bitrev_size8192_mixed24_f64<>+0x25C0(SB)/8, $1474
DATA bitrev_size8192_mixed24_f64<>+0x25C8(SB)/8, $3522
DATA bitrev_size8192_mixed24_f64<>+0x25D0(SB)/8, $5570
DATA bitrev_size8192_mixed24_f64<>+0x25D8(SB)/8, $7618
DATA bitrev_size8192_mixed24_f64<>+0x25E0(SB)/8, $1986
DATA bitrev_size8192_mixed24_f64<>+0x25E8(SB)/8, $4034
DATA bitrev_size8192_mixed24_f64<>+0x25F0(SB)/8, $6082
DATA bitrev_size8192_mixed24_f64<>+0x25F8(SB)/8, $8130
DATA bitrev_size8192_mixed24_f64<>+0x2600(SB)/8, $98
DATA bitrev_size8192_mixed24_f64<>+0x2608(SB)/8, $2146
DATA bitrev_size8192_mixed24_f64<>+0x2610(SB)/8, $4194
DATA bitrev_size8192_mixed24_f64<>+0x2618(SB)/8, $6242
DATA bitrev_size8192_mixed24_f64<>+0x2620(SB)/8, $610
DATA bitrev_size8192_mixed24_f64<>+0x2628(SB)/8, $2658
DATA bitrev_size8192_mixed24_f64<>+0x2630(SB)/8, $4706
DATA bitrev_size8192_mixed24_f64<>+0x2638(SB)/8, $6754
DATA bitrev_size8192_mixed24_f64<>+0x2640(SB)/8, $1122
DATA bitrev_size8192_mixed24_f64<>+0x2648(SB)/8, $3170
DATA bitrev_size8192_mixed24_f64<>+0x2650(SB)/8, $5218
DATA bitrev_size8192_mixed24_f64<>+0x2658(SB)/8, $7266
DATA bitrev_size8192_mixed24_f64<>+0x2660(SB)/8, $1634
DATA bitrev_size8192_mixed24_f64<>+0x2668(SB)/8, $3682
DATA bitrev_size8192_mixed24_f64<>+0x2670(SB)/8, $5730
DATA bitrev_size8192_mixed24_f64<>+0x2678(SB)/8, $7778
DATA bitrev_size8192_mixed24_f64<>+0x2680(SB)/8, $226
DATA bitrev_size8192_mixed24_f64<>+0x2688(SB)/8, $2274
DATA bitrev_size8192_mixed24_f64<>+0x2690(SB)/8, $4322
DATA bitrev_size8192_mixed24_f64<>+0x2698(SB)/8, $6370
DATA bitrev_size8192_mixed24_f64<>+0x26A0(SB)/8, $738
DATA bitrev_size8192_mixed24_f64<>+0x26A8(SB)/8, $2786
DATA bitrev_size8192_mixed24_f64<>+0x26B0(SB)/8, $4834
DATA bitrev_size8192_mixed24_f64<>+0x26B8(SB)/8, $6882
DATA bitrev_size8192_mixed24_f64<>+0x26C0(SB)/8, $1250
DATA bitrev_size8192_mixed24_f64<>+0x26C8(SB)/8, $3298
DATA bitrev_size8192_mixed24_f64<>+0x26D0(SB)/8, $5346
DATA bitrev_size8192_mixed24_f64<>+0x26D8(SB)/8, $7394
DATA bitrev_size8192_mixed24_f64<>+0x26E0(SB)/8, $1762
DATA bitrev_size8192_mixed24_f64<>+0x26E8(SB)/8, $3810
DATA bitrev_size8192_mixed24_f64<>+0x26F0(SB)/8, $5858
DATA bitrev_size8192_mixed24_f64<>+0x26F8(SB)/8, $7906
DATA bitrev_size8192_mixed24_f64<>+0x2700(SB)/8, $354
DATA bitrev_size8192_mixed24_f64<>+0x2708(SB)/8, $2402
DATA bitrev_size8192_mixed24_f64<>+0x2710(SB)/8, $4450
DATA bitrev_size8192_mixed24_f64<>+0x2718(SB)/8, $6498
DATA bitrev_size8192_mixed24_f64<>+0x2720(SB)/8, $866
DATA bitrev_size8192_mixed24_f64<>+0x2728(SB)/8, $2914
DATA bitrev_size8192_mixed24_f64<>+0x2730(SB)/8, $4962
DATA bitrev_size8192_mixed24_f64<>+0x2738(SB)/8, $7010
DATA bitrev_size8192_mixed24_f64<>+0x2740(SB)/8, $1378
DATA bitrev_size8192_mixed24_f64<>+0x2748(SB)/8, $3426
DATA bitrev_size8192_mixed24_f64<>+0x2750(SB)/8, $5474
DATA bitrev_size8192_mixed24_f64<>+0x2758(SB)/8, $7522
DATA bitrev_size8192_mixed24_f64<>+0x2760(SB)/8, $1890
DATA bitrev_size8192_mixed24_f64<>+0x2768(SB)/8, $3938
DATA bitrev_size8192_mixed24_f64<>+0x2770(SB)/8, $5986
DATA bitrev_size8192_mixed24_f64<>+0x2778(SB)/8, $8034
DATA bitrev_size8192_mixed24_f64<>+0x2780(SB)/8, $482
DATA bitrev_size8192_mixed24_f64<>+0x2788(SB)/8, $2530
DATA bitrev_size8192_mixed24_f64<>+0x2790(SB)/8, $4578
DATA bitrev_size8192_mixed24_f64<>+0x2798(SB)/8, $6626
DATA bitrev_size8192_mixed24_f64<>+0x27A0(SB)/8, $994
DATA bitrev_size8192_mixed24_f64<>+0x27A8(SB)/8, $3042
DATA bitrev_size8192_mixed24_f64<>+0x27B0(SB)/8, $5090
DATA bitrev_size8192_mixed24_f64<>+0x27B8(SB)/8, $7138
DATA bitrev_size8192_mixed24_f64<>+0x27C0(SB)/8, $1506
DATA bitrev_size8192_mixed24_f64<>+0x27C8(SB)/8, $3554
DATA bitrev_size8192_mixed24_f64<>+0x27D0(SB)/8, $5602
DATA bitrev_size8192_mixed24_f64<>+0x27D8(SB)/8, $7650
DATA bitrev_size8192_mixed24_f64<>+0x27E0(SB)/8, $2018
DATA bitrev_size8192_mixed24_f64<>+0x27E8(SB)/8, $4066
DATA bitrev_size8192_mixed24_f64<>+0x27F0(SB)/8, $6114
DATA bitrev_size8192_mixed24_f64<>+0x27F8(SB)/8, $8162
DATA bitrev_size8192_mixed24_f64<>+0x2800(SB)/8, $10
DATA bitrev_size8192_mixed24_f64<>+0x2808(SB)/8, $2058
DATA bitrev_size8192_mixed24_f64<>+0x2810(SB)/8, $4106
DATA bitrev_size8192_mixed24_f64<>+0x2818(SB)/8, $6154
DATA bitrev_size8192_mixed24_f64<>+0x2820(SB)/8, $522
DATA bitrev_size8192_mixed24_f64<>+0x2828(SB)/8, $2570
DATA bitrev_size8192_mixed24_f64<>+0x2830(SB)/8, $4618
DATA bitrev_size8192_mixed24_f64<>+0x2838(SB)/8, $6666
DATA bitrev_size8192_mixed24_f64<>+0x2840(SB)/8, $1034
DATA bitrev_size8192_mixed24_f64<>+0x2848(SB)/8, $3082
DATA bitrev_size8192_mixed24_f64<>+0x2850(SB)/8, $5130
DATA bitrev_size8192_mixed24_f64<>+0x2858(SB)/8, $7178
DATA bitrev_size8192_mixed24_f64<>+0x2860(SB)/8, $1546
DATA bitrev_size8192_mixed24_f64<>+0x2868(SB)/8, $3594
DATA bitrev_size8192_mixed24_f64<>+0x2870(SB)/8, $5642
DATA bitrev_size8192_mixed24_f64<>+0x2878(SB)/8, $7690
DATA bitrev_size8192_mixed24_f64<>+0x2880(SB)/8, $138
DATA bitrev_size8192_mixed24_f64<>+0x2888(SB)/8, $2186
DATA bitrev_size8192_mixed24_f64<>+0x2890(SB)/8, $4234
DATA bitrev_size8192_mixed24_f64<>+0x2898(SB)/8, $6282
DATA bitrev_size8192_mixed24_f64<>+0x28A0(SB)/8, $650
DATA bitrev_size8192_mixed24_f64<>+0x28A8(SB)/8, $2698
DATA bitrev_size8192_mixed24_f64<>+0x28B0(SB)/8, $4746
DATA bitrev_size8192_mixed24_f64<>+0x28B8(SB)/8, $6794
DATA bitrev_size8192_mixed24_f64<>+0x28C0(SB)/8, $1162
DATA bitrev_size8192_mixed24_f64<>+0x28C8(SB)/8, $3210
DATA bitrev_size8192_mixed24_f64<>+0x28D0(SB)/8, $5258
DATA bitrev_size8192_mixed24_f64<>+0x28D8(SB)/8, $7306
DATA bitrev_size8192_mixed24_f64<>+0x28E0(SB)/8, $1674
DATA bitrev_size8192_mixed24_f64<>+0x28E8(SB)/8, $3722
DATA bitrev_size8192_mixed24_f64<>+0x28F0(SB)/8, $5770
DATA bitrev_size8192_mixed24_f64<>+0x28F8(SB)/8, $7818
DATA bitrev_size8192_mixed24_f64<>+0x2900(SB)/8, $266
DATA bitrev_size8192_mixed24_f64<>+0x2908(SB)/8, $2314
DATA bitrev_size8192_mixed24_f64<>+0x2910(SB)/8, $4362
DATA bitrev_size8192_mixed24_f64<>+0x2918(SB)/8, $6410
DATA bitrev_size8192_mixed24_f64<>+0x2920(SB)/8, $778
DATA bitrev_size8192_mixed24_f64<>+0x2928(SB)/8, $2826
DATA bitrev_size8192_mixed24_f64<>+0x2930(SB)/8, $4874
DATA bitrev_size8192_mixed24_f64<>+0x2938(SB)/8, $6922
DATA bitrev_size8192_mixed24_f64<>+0x2940(SB)/8, $1290
DATA bitrev_size8192_mixed24_f64<>+0x2948(SB)/8, $3338
DATA bitrev_size8192_mixed24_f64<>+0x2950(SB)/8, $5386
DATA bitrev_size8192_mixed24_f64<>+0x2958(SB)/8, $7434
DATA bitrev_size8192_mixed24_f64<>+0x2960(SB)/8, $1802
DATA bitrev_size8192_mixed24_f64<>+0x2968(SB)/8, $3850
DATA bitrev_size8192_mixed24_f64<>+0x2970(SB)/8, $5898
DATA bitrev_size8192_mixed24_f64<>+0x2978(SB)/8, $7946
DATA bitrev_size8192_mixed24_f64<>+0x2980(SB)/8, $394
DATA bitrev_size8192_mixed24_f64<>+0x2988(SB)/8, $2442
DATA bitrev_size8192_mixed24_f64<>+0x2990(SB)/8, $4490
DATA bitrev_size8192_mixed24_f64<>+0x2998(SB)/8, $6538
DATA bitrev_size8192_mixed24_f64<>+0x29A0(SB)/8, $906
DATA bitrev_size8192_mixed24_f64<>+0x29A8(SB)/8, $2954
DATA bitrev_size8192_mixed24_f64<>+0x29B0(SB)/8, $5002
DATA bitrev_size8192_mixed24_f64<>+0x29B8(SB)/8, $7050
DATA bitrev_size8192_mixed24_f64<>+0x29C0(SB)/8, $1418
DATA bitrev_size8192_mixed24_f64<>+0x29C8(SB)/8, $3466
DATA bitrev_size8192_mixed24_f64<>+0x29D0(SB)/8, $5514
DATA bitrev_size8192_mixed24_f64<>+0x29D8(SB)/8, $7562
DATA bitrev_size8192_mixed24_f64<>+0x29E0(SB)/8, $1930
DATA bitrev_size8192_mixed24_f64<>+0x29E8(SB)/8, $3978
DATA bitrev_size8192_mixed24_f64<>+0x29F0(SB)/8, $6026
DATA bitrev_size8192_mixed24_f64<>+0x29F8(SB)/8, $8074
DATA bitrev_size8192_mixed24_f64<>+0x2A00(SB)/8, $42
DATA bitrev_size8192_mixed24_f64<>+0x2A08(SB)/8, $2090
DATA bitrev_size8192_mixed24_f64<>+0x2A10(SB)/8, $4138
DATA bitrev_size8192_mixed24_f64<>+0x2A18(SB)/8, $6186
DATA bitrev_size8192_mixed24_f64<>+0x2A20(SB)/8, $554
DATA bitrev_size8192_mixed24_f64<>+0x2A28(SB)/8, $2602
DATA bitrev_size8192_mixed24_f64<>+0x2A30(SB)/8, $4650
DATA bitrev_size8192_mixed24_f64<>+0x2A38(SB)/8, $6698
DATA bitrev_size8192_mixed24_f64<>+0x2A40(SB)/8, $1066
DATA bitrev_size8192_mixed24_f64<>+0x2A48(SB)/8, $3114
DATA bitrev_size8192_mixed24_f64<>+0x2A50(SB)/8, $5162
DATA bitrev_size8192_mixed24_f64<>+0x2A58(SB)/8, $7210
DATA bitrev_size8192_mixed24_f64<>+0x2A60(SB)/8, $1578
DATA bitrev_size8192_mixed24_f64<>+0x2A68(SB)/8, $3626
DATA bitrev_size8192_mixed24_f64<>+0x2A70(SB)/8, $5674
DATA bitrev_size8192_mixed24_f64<>+0x2A78(SB)/8, $7722
DATA bitrev_size8192_mixed24_f64<>+0x2A80(SB)/8, $170
DATA bitrev_size8192_mixed24_f64<>+0x2A88(SB)/8, $2218
DATA bitrev_size8192_mixed24_f64<>+0x2A90(SB)/8, $4266
DATA bitrev_size8192_mixed24_f64<>+0x2A98(SB)/8, $6314
DATA bitrev_size8192_mixed24_f64<>+0x2AA0(SB)/8, $682
DATA bitrev_size8192_mixed24_f64<>+0x2AA8(SB)/8, $2730
DATA bitrev_size8192_mixed24_f64<>+0x2AB0(SB)/8, $4778
DATA bitrev_size8192_mixed24_f64<>+0x2AB8(SB)/8, $6826
DATA bitrev_size8192_mixed24_f64<>+0x2AC0(SB)/8, $1194
DATA bitrev_size8192_mixed24_f64<>+0x2AC8(SB)/8, $3242
DATA bitrev_size8192_mixed24_f64<>+0x2AD0(SB)/8, $5290
DATA bitrev_size8192_mixed24_f64<>+0x2AD8(SB)/8, $7338
DATA bitrev_size8192_mixed24_f64<>+0x2AE0(SB)/8, $1706
DATA bitrev_size8192_mixed24_f64<>+0x2AE8(SB)/8, $3754
DATA bitrev_size8192_mixed24_f64<>+0x2AF0(SB)/8, $5802
DATA bitrev_size8192_mixed24_f64<>+0x2AF8(SB)/8, $7850
DATA bitrev_size8192_mixed24_f64<>+0x2B00(SB)/8, $298
DATA bitrev_size8192_mixed24_f64<>+0x2B08(SB)/8, $2346
DATA bitrev_size8192_mixed24_f64<>+0x2B10(SB)/8, $4394
DATA bitrev_size8192_mixed24_f64<>+0x2B18(SB)/8, $6442
DATA bitrev_size8192_mixed24_f64<>+0x2B20(SB)/8, $810
DATA bitrev_size8192_mixed24_f64<>+0x2B28(SB)/8, $2858
DATA bitrev_size8192_mixed24_f64<>+0x2B30(SB)/8, $4906
DATA bitrev_size8192_mixed24_f64<>+0x2B38(SB)/8, $6954
DATA bitrev_size8192_mixed24_f64<>+0x2B40(SB)/8, $1322
DATA bitrev_size8192_mixed24_f64<>+0x2B48(SB)/8, $3370
DATA bitrev_size8192_mixed24_f64<>+0x2B50(SB)/8, $5418
DATA bitrev_size8192_mixed24_f64<>+0x2B58(SB)/8, $7466
DATA bitrev_size8192_mixed24_f64<>+0x2B60(SB)/8, $1834
DATA bitrev_size8192_mixed24_f64<>+0x2B68(SB)/8, $3882
DATA bitrev_size8192_mixed24_f64<>+0x2B70(SB)/8, $5930
DATA bitrev_size8192_mixed24_f64<>+0x2B78(SB)/8, $7978
DATA bitrev_size8192_mixed24_f64<>+0x2B80(SB)/8, $426
DATA bitrev_size8192_mixed24_f64<>+0x2B88(SB)/8, $2474
DATA bitrev_size8192_mixed24_f64<>+0x2B90(SB)/8, $4522
DATA bitrev_size8192_mixed24_f64<>+0x2B98(SB)/8, $6570
DATA bitrev_size8192_mixed24_f64<>+0x2BA0(SB)/8, $938
DATA bitrev_size8192_mixed24_f64<>+0x2BA8(SB)/8, $2986
DATA bitrev_size8192_mixed24_f64<>+0x2BB0(SB)/8, $5034
DATA bitrev_size8192_mixed24_f64<>+0x2BB8(SB)/8, $7082
DATA bitrev_size8192_mixed24_f64<>+0x2BC0(SB)/8, $1450
DATA bitrev_size8192_mixed24_f64<>+0x2BC8(SB)/8, $3498
DATA bitrev_size8192_mixed24_f64<>+0x2BD0(SB)/8, $5546
DATA bitrev_size8192_mixed24_f64<>+0x2BD8(SB)/8, $7594
DATA bitrev_size8192_mixed24_f64<>+0x2BE0(SB)/8, $1962
DATA bitrev_size8192_mixed24_f64<>+0x2BE8(SB)/8, $4010
DATA bitrev_size8192_mixed24_f64<>+0x2BF0(SB)/8, $6058
DATA bitrev_size8192_mixed24_f64<>+0x2BF8(SB)/8, $8106
DATA bitrev_size8192_mixed24_f64<>+0x2C00(SB)/8, $74
DATA bitrev_size8192_mixed24_f64<>+0x2C08(SB)/8, $2122
DATA bitrev_size8192_mixed24_f64<>+0x2C10(SB)/8, $4170
DATA bitrev_size8192_mixed24_f64<>+0x2C18(SB)/8, $6218
DATA bitrev_size8192_mixed24_f64<>+0x2C20(SB)/8, $586
DATA bitrev_size8192_mixed24_f64<>+0x2C28(SB)/8, $2634
DATA bitrev_size8192_mixed24_f64<>+0x2C30(SB)/8, $4682
DATA bitrev_size8192_mixed24_f64<>+0x2C38(SB)/8, $6730
DATA bitrev_size8192_mixed24_f64<>+0x2C40(SB)/8, $1098
DATA bitrev_size8192_mixed24_f64<>+0x2C48(SB)/8, $3146
DATA bitrev_size8192_mixed24_f64<>+0x2C50(SB)/8, $5194
DATA bitrev_size8192_mixed24_f64<>+0x2C58(SB)/8, $7242
DATA bitrev_size8192_mixed24_f64<>+0x2C60(SB)/8, $1610
DATA bitrev_size8192_mixed24_f64<>+0x2C68(SB)/8, $3658
DATA bitrev_size8192_mixed24_f64<>+0x2C70(SB)/8, $5706
DATA bitrev_size8192_mixed24_f64<>+0x2C78(SB)/8, $7754
DATA bitrev_size8192_mixed24_f64<>+0x2C80(SB)/8, $202
DATA bitrev_size8192_mixed24_f64<>+0x2C88(SB)/8, $2250
DATA bitrev_size8192_mixed24_f64<>+0x2C90(SB)/8, $4298
DATA bitrev_size8192_mixed24_f64<>+0x2C98(SB)/8, $6346
DATA bitrev_size8192_mixed24_f64<>+0x2CA0(SB)/8, $714
DATA bitrev_size8192_mixed24_f64<>+0x2CA8(SB)/8, $2762
DATA bitrev_size8192_mixed24_f64<>+0x2CB0(SB)/8, $4810
DATA bitrev_size8192_mixed24_f64<>+0x2CB8(SB)/8, $6858
DATA bitrev_size8192_mixed24_f64<>+0x2CC0(SB)/8, $1226
DATA bitrev_size8192_mixed24_f64<>+0x2CC8(SB)/8, $3274
DATA bitrev_size8192_mixed24_f64<>+0x2CD0(SB)/8, $5322
DATA bitrev_size8192_mixed24_f64<>+0x2CD8(SB)/8, $7370
DATA bitrev_size8192_mixed24_f64<>+0x2CE0(SB)/8, $1738
DATA bitrev_size8192_mixed24_f64<>+0x2CE8(SB)/8, $3786
DATA bitrev_size8192_mixed24_f64<>+0x2CF0(SB)/8, $5834
DATA bitrev_size8192_mixed24_f64<>+0x2CF8(SB)/8, $7882
DATA bitrev_size8192_mixed24_f64<>+0x2D00(SB)/8, $330
DATA bitrev_size8192_mixed24_f64<>+0x2D08(SB)/8, $2378
DATA bitrev_size8192_mixed24_f64<>+0x2D10(SB)/8, $4426
DATA bitrev_size8192_mixed24_f64<>+0x2D18(SB)/8, $6474
DATA bitrev_size8192_mixed24_f64<>+0x2D20(SB)/8, $842
DATA bitrev_size8192_mixed24_f64<>+0x2D28(SB)/8, $2890
DATA bitrev_size8192_mixed24_f64<>+0x2D30(SB)/8, $4938
DATA bitrev_size8192_mixed24_f64<>+0x2D38(SB)/8, $6986
DATA bitrev_size8192_mixed24_f64<>+0x2D40(SB)/8, $1354
DATA bitrev_size8192_mixed24_f64<>+0x2D48(SB)/8, $3402
DATA bitrev_size8192_mixed24_f64<>+0x2D50(SB)/8, $5450
DATA bitrev_size8192_mixed24_f64<>+0x2D58(SB)/8, $7498
DATA bitrev_size8192_mixed24_f64<>+0x2D60(SB)/8, $1866
DATA bitrev_size8192_mixed24_f64<>+0x2D68(SB)/8, $3914
DATA bitrev_size8192_mixed24_f64<>+0x2D70(SB)/8, $5962
DATA bitrev_size8192_mixed24_f64<>+0x2D78(SB)/8, $8010
DATA bitrev_size8192_mixed24_f64<>+0x2D80(SB)/8, $458
DATA bitrev_size8192_mixed24_f64<>+0x2D88(SB)/8, $2506
DATA bitrev_size8192_mixed24_f64<>+0x2D90(SB)/8, $4554
DATA bitrev_size8192_mixed24_f64<>+0x2D98(SB)/8, $6602
DATA bitrev_size8192_mixed24_f64<>+0x2DA0(SB)/8, $970
DATA bitrev_size8192_mixed24_f64<>+0x2DA8(SB)/8, $3018
DATA bitrev_size8192_mixed24_f64<>+0x2DB0(SB)/8, $5066
DATA bitrev_size8192_mixed24_f64<>+0x2DB8(SB)/8, $7114
DATA bitrev_size8192_mixed24_f64<>+0x2DC0(SB)/8, $1482
DATA bitrev_size8192_mixed24_f64<>+0x2DC8(SB)/8, $3530
DATA bitrev_size8192_mixed24_f64<>+0x2DD0(SB)/8, $5578
DATA bitrev_size8192_mixed24_f64<>+0x2DD8(SB)/8, $7626
DATA bitrev_size8192_mixed24_f64<>+0x2DE0(SB)/8, $1994
DATA bitrev_size8192_mixed24_f64<>+0x2DE8(SB)/8, $4042
DATA bitrev_size8192_mixed24_f64<>+0x2DF0(SB)/8, $6090
DATA bitrev_size8192_mixed24_f64<>+0x2DF8(SB)/8, $8138
DATA bitrev_size8192_mixed24_f64<>+0x2E00(SB)/8, $106
DATA bitrev_size8192_mixed24_f64<>+0x2E08(SB)/8, $2154
DATA bitrev_size8192_mixed24_f64<>+0x2E10(SB)/8, $4202
DATA bitrev_size8192_mixed24_f64<>+0x2E18(SB)/8, $6250
DATA bitrev_size8192_mixed24_f64<>+0x2E20(SB)/8, $618
DATA bitrev_size8192_mixed24_f64<>+0x2E28(SB)/8, $2666
DATA bitrev_size8192_mixed24_f64<>+0x2E30(SB)/8, $4714
DATA bitrev_size8192_mixed24_f64<>+0x2E38(SB)/8, $6762
DATA bitrev_size8192_mixed24_f64<>+0x2E40(SB)/8, $1130
DATA bitrev_size8192_mixed24_f64<>+0x2E48(SB)/8, $3178
DATA bitrev_size8192_mixed24_f64<>+0x2E50(SB)/8, $5226
DATA bitrev_size8192_mixed24_f64<>+0x2E58(SB)/8, $7274
DATA bitrev_size8192_mixed24_f64<>+0x2E60(SB)/8, $1642
DATA bitrev_size8192_mixed24_f64<>+0x2E68(SB)/8, $3690
DATA bitrev_size8192_mixed24_f64<>+0x2E70(SB)/8, $5738
DATA bitrev_size8192_mixed24_f64<>+0x2E78(SB)/8, $7786
DATA bitrev_size8192_mixed24_f64<>+0x2E80(SB)/8, $234
DATA bitrev_size8192_mixed24_f64<>+0x2E88(SB)/8, $2282
DATA bitrev_size8192_mixed24_f64<>+0x2E90(SB)/8, $4330
DATA bitrev_size8192_mixed24_f64<>+0x2E98(SB)/8, $6378
DATA bitrev_size8192_mixed24_f64<>+0x2EA0(SB)/8, $746
DATA bitrev_size8192_mixed24_f64<>+0x2EA8(SB)/8, $2794
DATA bitrev_size8192_mixed24_f64<>+0x2EB0(SB)/8, $4842
DATA bitrev_size8192_mixed24_f64<>+0x2EB8(SB)/8, $6890
DATA bitrev_size8192_mixed24_f64<>+0x2EC0(SB)/8, $1258
DATA bitrev_size8192_mixed24_f64<>+0x2EC8(SB)/8, $3306
DATA bitrev_size8192_mixed24_f64<>+0x2ED0(SB)/8, $5354
DATA bitrev_size8192_mixed24_f64<>+0x2ED8(SB)/8, $7402
DATA bitrev_size8192_mixed24_f64<>+0x2EE0(SB)/8, $1770
DATA bitrev_size8192_mixed24_f64<>+0x2EE8(SB)/8, $3818
DATA bitrev_size8192_mixed24_f64<>+0x2EF0(SB)/8, $5866
DATA bitrev_size8192_mixed24_f64<>+0x2EF8(SB)/8, $7914
DATA bitrev_size8192_mixed24_f64<>+0x2F00(SB)/8, $362
DATA bitrev_size8192_mixed24_f64<>+0x2F08(SB)/8, $2410
DATA bitrev_size8192_mixed24_f64<>+0x2F10(SB)/8, $4458
DATA bitrev_size8192_mixed24_f64<>+0x2F18(SB)/8, $6506
DATA bitrev_size8192_mixed24_f64<>+0x2F20(SB)/8, $874
DATA bitrev_size8192_mixed24_f64<>+0x2F28(SB)/8, $2922
DATA bitrev_size8192_mixed24_f64<>+0x2F30(SB)/8, $4970
DATA bitrev_size8192_mixed24_f64<>+0x2F38(SB)/8, $7018
DATA bitrev_size8192_mixed24_f64<>+0x2F40(SB)/8, $1386
DATA bitrev_size8192_mixed24_f64<>+0x2F48(SB)/8, $3434
DATA bitrev_size8192_mixed24_f64<>+0x2F50(SB)/8, $5482
DATA bitrev_size8192_mixed24_f64<>+0x2F58(SB)/8, $7530
DATA bitrev_size8192_mixed24_f64<>+0x2F60(SB)/8, $1898
DATA bitrev_size8192_mixed24_f64<>+0x2F68(SB)/8, $3946
DATA bitrev_size8192_mixed24_f64<>+0x2F70(SB)/8, $5994
DATA bitrev_size8192_mixed24_f64<>+0x2F78(SB)/8, $8042
DATA bitrev_size8192_mixed24_f64<>+0x2F80(SB)/8, $490
DATA bitrev_size8192_mixed24_f64<>+0x2F88(SB)/8, $2538
DATA bitrev_size8192_mixed24_f64<>+0x2F90(SB)/8, $4586
DATA bitrev_size8192_mixed24_f64<>+0x2F98(SB)/8, $6634
DATA bitrev_size8192_mixed24_f64<>+0x2FA0(SB)/8, $1002
DATA bitrev_size8192_mixed24_f64<>+0x2FA8(SB)/8, $3050
DATA bitrev_size8192_mixed24_f64<>+0x2FB0(SB)/8, $5098
DATA bitrev_size8192_mixed24_f64<>+0x2FB8(SB)/8, $7146
DATA bitrev_size8192_mixed24_f64<>+0x2FC0(SB)/8, $1514
DATA bitrev_size8192_mixed24_f64<>+0x2FC8(SB)/8, $3562
DATA bitrev_size8192_mixed24_f64<>+0x2FD0(SB)/8, $5610
DATA bitrev_size8192_mixed24_f64<>+0x2FD8(SB)/8, $7658
DATA bitrev_size8192_mixed24_f64<>+0x2FE0(SB)/8, $2026
DATA bitrev_size8192_mixed24_f64<>+0x2FE8(SB)/8, $4074
DATA bitrev_size8192_mixed24_f64<>+0x2FF0(SB)/8, $6122
DATA bitrev_size8192_mixed24_f64<>+0x2FF8(SB)/8, $8170
DATA bitrev_size8192_mixed24_f64<>+0x3000(SB)/8, $18
DATA bitrev_size8192_mixed24_f64<>+0x3008(SB)/8, $2066
DATA bitrev_size8192_mixed24_f64<>+0x3010(SB)/8, $4114
DATA bitrev_size8192_mixed24_f64<>+0x3018(SB)/8, $6162
DATA bitrev_size8192_mixed24_f64<>+0x3020(SB)/8, $530
DATA bitrev_size8192_mixed24_f64<>+0x3028(SB)/8, $2578
DATA bitrev_size8192_mixed24_f64<>+0x3030(SB)/8, $4626
DATA bitrev_size8192_mixed24_f64<>+0x3038(SB)/8, $6674
DATA bitrev_size8192_mixed24_f64<>+0x3040(SB)/8, $1042
DATA bitrev_size8192_mixed24_f64<>+0x3048(SB)/8, $3090
DATA bitrev_size8192_mixed24_f64<>+0x3050(SB)/8, $5138
DATA bitrev_size8192_mixed24_f64<>+0x3058(SB)/8, $7186
DATA bitrev_size8192_mixed24_f64<>+0x3060(SB)/8, $1554
DATA bitrev_size8192_mixed24_f64<>+0x3068(SB)/8, $3602
DATA bitrev_size8192_mixed24_f64<>+0x3070(SB)/8, $5650
DATA bitrev_size8192_mixed24_f64<>+0x3078(SB)/8, $7698
DATA bitrev_size8192_mixed24_f64<>+0x3080(SB)/8, $146
DATA bitrev_size8192_mixed24_f64<>+0x3088(SB)/8, $2194
DATA bitrev_size8192_mixed24_f64<>+0x3090(SB)/8, $4242
DATA bitrev_size8192_mixed24_f64<>+0x3098(SB)/8, $6290
DATA bitrev_size8192_mixed24_f64<>+0x30A0(SB)/8, $658
DATA bitrev_size8192_mixed24_f64<>+0x30A8(SB)/8, $2706
DATA bitrev_size8192_mixed24_f64<>+0x30B0(SB)/8, $4754
DATA bitrev_size8192_mixed24_f64<>+0x30B8(SB)/8, $6802
DATA bitrev_size8192_mixed24_f64<>+0x30C0(SB)/8, $1170
DATA bitrev_size8192_mixed24_f64<>+0x30C8(SB)/8, $3218
DATA bitrev_size8192_mixed24_f64<>+0x30D0(SB)/8, $5266
DATA bitrev_size8192_mixed24_f64<>+0x30D8(SB)/8, $7314
DATA bitrev_size8192_mixed24_f64<>+0x30E0(SB)/8, $1682
DATA bitrev_size8192_mixed24_f64<>+0x30E8(SB)/8, $3730
DATA bitrev_size8192_mixed24_f64<>+0x30F0(SB)/8, $5778
DATA bitrev_size8192_mixed24_f64<>+0x30F8(SB)/8, $7826
DATA bitrev_size8192_mixed24_f64<>+0x3100(SB)/8, $274
DATA bitrev_size8192_mixed24_f64<>+0x3108(SB)/8, $2322
DATA bitrev_size8192_mixed24_f64<>+0x3110(SB)/8, $4370
DATA bitrev_size8192_mixed24_f64<>+0x3118(SB)/8, $6418
DATA bitrev_size8192_mixed24_f64<>+0x3120(SB)/8, $786
DATA bitrev_size8192_mixed24_f64<>+0x3128(SB)/8, $2834
DATA bitrev_size8192_mixed24_f64<>+0x3130(SB)/8, $4882
DATA bitrev_size8192_mixed24_f64<>+0x3138(SB)/8, $6930
DATA bitrev_size8192_mixed24_f64<>+0x3140(SB)/8, $1298
DATA bitrev_size8192_mixed24_f64<>+0x3148(SB)/8, $3346
DATA bitrev_size8192_mixed24_f64<>+0x3150(SB)/8, $5394
DATA bitrev_size8192_mixed24_f64<>+0x3158(SB)/8, $7442
DATA bitrev_size8192_mixed24_f64<>+0x3160(SB)/8, $1810
DATA bitrev_size8192_mixed24_f64<>+0x3168(SB)/8, $3858
DATA bitrev_size8192_mixed24_f64<>+0x3170(SB)/8, $5906
DATA bitrev_size8192_mixed24_f64<>+0x3178(SB)/8, $7954
DATA bitrev_size8192_mixed24_f64<>+0x3180(SB)/8, $402
DATA bitrev_size8192_mixed24_f64<>+0x3188(SB)/8, $2450
DATA bitrev_size8192_mixed24_f64<>+0x3190(SB)/8, $4498
DATA bitrev_size8192_mixed24_f64<>+0x3198(SB)/8, $6546
DATA bitrev_size8192_mixed24_f64<>+0x31A0(SB)/8, $914
DATA bitrev_size8192_mixed24_f64<>+0x31A8(SB)/8, $2962
DATA bitrev_size8192_mixed24_f64<>+0x31B0(SB)/8, $5010
DATA bitrev_size8192_mixed24_f64<>+0x31B8(SB)/8, $7058
DATA bitrev_size8192_mixed24_f64<>+0x31C0(SB)/8, $1426
DATA bitrev_size8192_mixed24_f64<>+0x31C8(SB)/8, $3474
DATA bitrev_size8192_mixed24_f64<>+0x31D0(SB)/8, $5522
DATA bitrev_size8192_mixed24_f64<>+0x31D8(SB)/8, $7570
DATA bitrev_size8192_mixed24_f64<>+0x31E0(SB)/8, $1938
DATA bitrev_size8192_mixed24_f64<>+0x31E8(SB)/8, $3986
DATA bitrev_size8192_mixed24_f64<>+0x31F0(SB)/8, $6034
DATA bitrev_size8192_mixed24_f64<>+0x31F8(SB)/8, $8082
DATA bitrev_size8192_mixed24_f64<>+0x3200(SB)/8, $50
DATA bitrev_size8192_mixed24_f64<>+0x3208(SB)/8, $2098
DATA bitrev_size8192_mixed24_f64<>+0x3210(SB)/8, $4146
DATA bitrev_size8192_mixed24_f64<>+0x3218(SB)/8, $6194
DATA bitrev_size8192_mixed24_f64<>+0x3220(SB)/8, $562
DATA bitrev_size8192_mixed24_f64<>+0x3228(SB)/8, $2610
DATA bitrev_size8192_mixed24_f64<>+0x3230(SB)/8, $4658
DATA bitrev_size8192_mixed24_f64<>+0x3238(SB)/8, $6706
DATA bitrev_size8192_mixed24_f64<>+0x3240(SB)/8, $1074
DATA bitrev_size8192_mixed24_f64<>+0x3248(SB)/8, $3122
DATA bitrev_size8192_mixed24_f64<>+0x3250(SB)/8, $5170
DATA bitrev_size8192_mixed24_f64<>+0x3258(SB)/8, $7218
DATA bitrev_size8192_mixed24_f64<>+0x3260(SB)/8, $1586
DATA bitrev_size8192_mixed24_f64<>+0x3268(SB)/8, $3634
DATA bitrev_size8192_mixed24_f64<>+0x3270(SB)/8, $5682
DATA bitrev_size8192_mixed24_f64<>+0x3278(SB)/8, $7730
DATA bitrev_size8192_mixed24_f64<>+0x3280(SB)/8, $178
DATA bitrev_size8192_mixed24_f64<>+0x3288(SB)/8, $2226
DATA bitrev_size8192_mixed24_f64<>+0x3290(SB)/8, $4274
DATA bitrev_size8192_mixed24_f64<>+0x3298(SB)/8, $6322
DATA bitrev_size8192_mixed24_f64<>+0x32A0(SB)/8, $690
DATA bitrev_size8192_mixed24_f64<>+0x32A8(SB)/8, $2738
DATA bitrev_size8192_mixed24_f64<>+0x32B0(SB)/8, $4786
DATA bitrev_size8192_mixed24_f64<>+0x32B8(SB)/8, $6834
DATA bitrev_size8192_mixed24_f64<>+0x32C0(SB)/8, $1202
DATA bitrev_size8192_mixed24_f64<>+0x32C8(SB)/8, $3250
DATA bitrev_size8192_mixed24_f64<>+0x32D0(SB)/8, $5298
DATA bitrev_size8192_mixed24_f64<>+0x32D8(SB)/8, $7346
DATA bitrev_size8192_mixed24_f64<>+0x32E0(SB)/8, $1714
DATA bitrev_size8192_mixed24_f64<>+0x32E8(SB)/8, $3762
DATA bitrev_size8192_mixed24_f64<>+0x32F0(SB)/8, $5810
DATA bitrev_size8192_mixed24_f64<>+0x32F8(SB)/8, $7858
DATA bitrev_size8192_mixed24_f64<>+0x3300(SB)/8, $306
DATA bitrev_size8192_mixed24_f64<>+0x3308(SB)/8, $2354
DATA bitrev_size8192_mixed24_f64<>+0x3310(SB)/8, $4402
DATA bitrev_size8192_mixed24_f64<>+0x3318(SB)/8, $6450
DATA bitrev_size8192_mixed24_f64<>+0x3320(SB)/8, $818
DATA bitrev_size8192_mixed24_f64<>+0x3328(SB)/8, $2866
DATA bitrev_size8192_mixed24_f64<>+0x3330(SB)/8, $4914
DATA bitrev_size8192_mixed24_f64<>+0x3338(SB)/8, $6962
DATA bitrev_size8192_mixed24_f64<>+0x3340(SB)/8, $1330
DATA bitrev_size8192_mixed24_f64<>+0x3348(SB)/8, $3378
DATA bitrev_size8192_mixed24_f64<>+0x3350(SB)/8, $5426
DATA bitrev_size8192_mixed24_f64<>+0x3358(SB)/8, $7474
DATA bitrev_size8192_mixed24_f64<>+0x3360(SB)/8, $1842
DATA bitrev_size8192_mixed24_f64<>+0x3368(SB)/8, $3890
DATA bitrev_size8192_mixed24_f64<>+0x3370(SB)/8, $5938
DATA bitrev_size8192_mixed24_f64<>+0x3378(SB)/8, $7986
DATA bitrev_size8192_mixed24_f64<>+0x3380(SB)/8, $434
DATA bitrev_size8192_mixed24_f64<>+0x3388(SB)/8, $2482
DATA bitrev_size8192_mixed24_f64<>+0x3390(SB)/8, $4530
DATA bitrev_size8192_mixed24_f64<>+0x3398(SB)/8, $6578
DATA bitrev_size8192_mixed24_f64<>+0x33A0(SB)/8, $946
DATA bitrev_size8192_mixed24_f64<>+0x33A8(SB)/8, $2994
DATA bitrev_size8192_mixed24_f64<>+0x33B0(SB)/8, $5042
DATA bitrev_size8192_mixed24_f64<>+0x33B8(SB)/8, $7090
DATA bitrev_size8192_mixed24_f64<>+0x33C0(SB)/8, $1458
DATA bitrev_size8192_mixed24_f64<>+0x33C8(SB)/8, $3506
DATA bitrev_size8192_mixed24_f64<>+0x33D0(SB)/8, $5554
DATA bitrev_size8192_mixed24_f64<>+0x33D8(SB)/8, $7602
DATA bitrev_size8192_mixed24_f64<>+0x33E0(SB)/8, $1970
DATA bitrev_size8192_mixed24_f64<>+0x33E8(SB)/8, $4018
DATA bitrev_size8192_mixed24_f64<>+0x33F0(SB)/8, $6066
DATA bitrev_size8192_mixed24_f64<>+0x33F8(SB)/8, $8114
DATA bitrev_size8192_mixed24_f64<>+0x3400(SB)/8, $82
DATA bitrev_size8192_mixed24_f64<>+0x3408(SB)/8, $2130
DATA bitrev_size8192_mixed24_f64<>+0x3410(SB)/8, $4178
DATA bitrev_size8192_mixed24_f64<>+0x3418(SB)/8, $6226
DATA bitrev_size8192_mixed24_f64<>+0x3420(SB)/8, $594
DATA bitrev_size8192_mixed24_f64<>+0x3428(SB)/8, $2642
DATA bitrev_size8192_mixed24_f64<>+0x3430(SB)/8, $4690
DATA bitrev_size8192_mixed24_f64<>+0x3438(SB)/8, $6738
DATA bitrev_size8192_mixed24_f64<>+0x3440(SB)/8, $1106
DATA bitrev_size8192_mixed24_f64<>+0x3448(SB)/8, $3154
DATA bitrev_size8192_mixed24_f64<>+0x3450(SB)/8, $5202
DATA bitrev_size8192_mixed24_f64<>+0x3458(SB)/8, $7250
DATA bitrev_size8192_mixed24_f64<>+0x3460(SB)/8, $1618
DATA bitrev_size8192_mixed24_f64<>+0x3468(SB)/8, $3666
DATA bitrev_size8192_mixed24_f64<>+0x3470(SB)/8, $5714
DATA bitrev_size8192_mixed24_f64<>+0x3478(SB)/8, $7762
DATA bitrev_size8192_mixed24_f64<>+0x3480(SB)/8, $210
DATA bitrev_size8192_mixed24_f64<>+0x3488(SB)/8, $2258
DATA bitrev_size8192_mixed24_f64<>+0x3490(SB)/8, $4306
DATA bitrev_size8192_mixed24_f64<>+0x3498(SB)/8, $6354
DATA bitrev_size8192_mixed24_f64<>+0x34A0(SB)/8, $722
DATA bitrev_size8192_mixed24_f64<>+0x34A8(SB)/8, $2770
DATA bitrev_size8192_mixed24_f64<>+0x34B0(SB)/8, $4818
DATA bitrev_size8192_mixed24_f64<>+0x34B8(SB)/8, $6866
DATA bitrev_size8192_mixed24_f64<>+0x34C0(SB)/8, $1234
DATA bitrev_size8192_mixed24_f64<>+0x34C8(SB)/8, $3282
DATA bitrev_size8192_mixed24_f64<>+0x34D0(SB)/8, $5330
DATA bitrev_size8192_mixed24_f64<>+0x34D8(SB)/8, $7378
DATA bitrev_size8192_mixed24_f64<>+0x34E0(SB)/8, $1746
DATA bitrev_size8192_mixed24_f64<>+0x34E8(SB)/8, $3794
DATA bitrev_size8192_mixed24_f64<>+0x34F0(SB)/8, $5842
DATA bitrev_size8192_mixed24_f64<>+0x34F8(SB)/8, $7890
DATA bitrev_size8192_mixed24_f64<>+0x3500(SB)/8, $338
DATA bitrev_size8192_mixed24_f64<>+0x3508(SB)/8, $2386
DATA bitrev_size8192_mixed24_f64<>+0x3510(SB)/8, $4434
DATA bitrev_size8192_mixed24_f64<>+0x3518(SB)/8, $6482
DATA bitrev_size8192_mixed24_f64<>+0x3520(SB)/8, $850
DATA bitrev_size8192_mixed24_f64<>+0x3528(SB)/8, $2898
DATA bitrev_size8192_mixed24_f64<>+0x3530(SB)/8, $4946
DATA bitrev_size8192_mixed24_f64<>+0x3538(SB)/8, $6994
DATA bitrev_size8192_mixed24_f64<>+0x3540(SB)/8, $1362
DATA bitrev_size8192_mixed24_f64<>+0x3548(SB)/8, $3410
DATA bitrev_size8192_mixed24_f64<>+0x3550(SB)/8, $5458
DATA bitrev_size8192_mixed24_f64<>+0x3558(SB)/8, $7506
DATA bitrev_size8192_mixed24_f64<>+0x3560(SB)/8, $1874
DATA bitrev_size8192_mixed24_f64<>+0x3568(SB)/8, $3922
DATA bitrev_size8192_mixed24_f64<>+0x3570(SB)/8, $5970
DATA bitrev_size8192_mixed24_f64<>+0x3578(SB)/8, $8018
DATA bitrev_size8192_mixed24_f64<>+0x3580(SB)/8, $466
DATA bitrev_size8192_mixed24_f64<>+0x3588(SB)/8, $2514
DATA bitrev_size8192_mixed24_f64<>+0x3590(SB)/8, $4562
DATA bitrev_size8192_mixed24_f64<>+0x3598(SB)/8, $6610
DATA bitrev_size8192_mixed24_f64<>+0x35A0(SB)/8, $978
DATA bitrev_size8192_mixed24_f64<>+0x35A8(SB)/8, $3026
DATA bitrev_size8192_mixed24_f64<>+0x35B0(SB)/8, $5074
DATA bitrev_size8192_mixed24_f64<>+0x35B8(SB)/8, $7122
DATA bitrev_size8192_mixed24_f64<>+0x35C0(SB)/8, $1490
DATA bitrev_size8192_mixed24_f64<>+0x35C8(SB)/8, $3538
DATA bitrev_size8192_mixed24_f64<>+0x35D0(SB)/8, $5586
DATA bitrev_size8192_mixed24_f64<>+0x35D8(SB)/8, $7634
DATA bitrev_size8192_mixed24_f64<>+0x35E0(SB)/8, $2002
DATA bitrev_size8192_mixed24_f64<>+0x35E8(SB)/8, $4050
DATA bitrev_size8192_mixed24_f64<>+0x35F0(SB)/8, $6098
DATA bitrev_size8192_mixed24_f64<>+0x35F8(SB)/8, $8146
DATA bitrev_size8192_mixed24_f64<>+0x3600(SB)/8, $114
DATA bitrev_size8192_mixed24_f64<>+0x3608(SB)/8, $2162
DATA bitrev_size8192_mixed24_f64<>+0x3610(SB)/8, $4210
DATA bitrev_size8192_mixed24_f64<>+0x3618(SB)/8, $6258
DATA bitrev_size8192_mixed24_f64<>+0x3620(SB)/8, $626
DATA bitrev_size8192_mixed24_f64<>+0x3628(SB)/8, $2674
DATA bitrev_size8192_mixed24_f64<>+0x3630(SB)/8, $4722
DATA bitrev_size8192_mixed24_f64<>+0x3638(SB)/8, $6770
DATA bitrev_size8192_mixed24_f64<>+0x3640(SB)/8, $1138
DATA bitrev_size8192_mixed24_f64<>+0x3648(SB)/8, $3186
DATA bitrev_size8192_mixed24_f64<>+0x3650(SB)/8, $5234
DATA bitrev_size8192_mixed24_f64<>+0x3658(SB)/8, $7282
DATA bitrev_size8192_mixed24_f64<>+0x3660(SB)/8, $1650
DATA bitrev_size8192_mixed24_f64<>+0x3668(SB)/8, $3698
DATA bitrev_size8192_mixed24_f64<>+0x3670(SB)/8, $5746
DATA bitrev_size8192_mixed24_f64<>+0x3678(SB)/8, $7794
DATA bitrev_size8192_mixed24_f64<>+0x3680(SB)/8, $242
DATA bitrev_size8192_mixed24_f64<>+0x3688(SB)/8, $2290
DATA bitrev_size8192_mixed24_f64<>+0x3690(SB)/8, $4338
DATA bitrev_size8192_mixed24_f64<>+0x3698(SB)/8, $6386
DATA bitrev_size8192_mixed24_f64<>+0x36A0(SB)/8, $754
DATA bitrev_size8192_mixed24_f64<>+0x36A8(SB)/8, $2802
DATA bitrev_size8192_mixed24_f64<>+0x36B0(SB)/8, $4850
DATA bitrev_size8192_mixed24_f64<>+0x36B8(SB)/8, $6898
DATA bitrev_size8192_mixed24_f64<>+0x36C0(SB)/8, $1266
DATA bitrev_size8192_mixed24_f64<>+0x36C8(SB)/8, $3314
DATA bitrev_size8192_mixed24_f64<>+0x36D0(SB)/8, $5362
DATA bitrev_size8192_mixed24_f64<>+0x36D8(SB)/8, $7410
DATA bitrev_size8192_mixed24_f64<>+0x36E0(SB)/8, $1778
DATA bitrev_size8192_mixed24_f64<>+0x36E8(SB)/8, $3826
DATA bitrev_size8192_mixed24_f64<>+0x36F0(SB)/8, $5874
DATA bitrev_size8192_mixed24_f64<>+0x36F8(SB)/8, $7922
DATA bitrev_size8192_mixed24_f64<>+0x3700(SB)/8, $370
DATA bitrev_size8192_mixed24_f64<>+0x3708(SB)/8, $2418
DATA bitrev_size8192_mixed24_f64<>+0x3710(SB)/8, $4466
DATA bitrev_size8192_mixed24_f64<>+0x3718(SB)/8, $6514
DATA bitrev_size8192_mixed24_f64<>+0x3720(SB)/8, $882
DATA bitrev_size8192_mixed24_f64<>+0x3728(SB)/8, $2930
DATA bitrev_size8192_mixed24_f64<>+0x3730(SB)/8, $4978
DATA bitrev_size8192_mixed24_f64<>+0x3738(SB)/8, $7026
DATA bitrev_size8192_mixed24_f64<>+0x3740(SB)/8, $1394
DATA bitrev_size8192_mixed24_f64<>+0x3748(SB)/8, $3442
DATA bitrev_size8192_mixed24_f64<>+0x3750(SB)/8, $5490
DATA bitrev_size8192_mixed24_f64<>+0x3758(SB)/8, $7538
DATA bitrev_size8192_mixed24_f64<>+0x3760(SB)/8, $1906
DATA bitrev_size8192_mixed24_f64<>+0x3768(SB)/8, $3954
DATA bitrev_size8192_mixed24_f64<>+0x3770(SB)/8, $6002
DATA bitrev_size8192_mixed24_f64<>+0x3778(SB)/8, $8050
DATA bitrev_size8192_mixed24_f64<>+0x3780(SB)/8, $498
DATA bitrev_size8192_mixed24_f64<>+0x3788(SB)/8, $2546
DATA bitrev_size8192_mixed24_f64<>+0x3790(SB)/8, $4594
DATA bitrev_size8192_mixed24_f64<>+0x3798(SB)/8, $6642
DATA bitrev_size8192_mixed24_f64<>+0x37A0(SB)/8, $1010
DATA bitrev_size8192_mixed24_f64<>+0x37A8(SB)/8, $3058
DATA bitrev_size8192_mixed24_f64<>+0x37B0(SB)/8, $5106
DATA bitrev_size8192_mixed24_f64<>+0x37B8(SB)/8, $7154
DATA bitrev_size8192_mixed24_f64<>+0x37C0(SB)/8, $1522
DATA bitrev_size8192_mixed24_f64<>+0x37C8(SB)/8, $3570
DATA bitrev_size8192_mixed24_f64<>+0x37D0(SB)/8, $5618
DATA bitrev_size8192_mixed24_f64<>+0x37D8(SB)/8, $7666
DATA bitrev_size8192_mixed24_f64<>+0x37E0(SB)/8, $2034
DATA bitrev_size8192_mixed24_f64<>+0x37E8(SB)/8, $4082
DATA bitrev_size8192_mixed24_f64<>+0x37F0(SB)/8, $6130
DATA bitrev_size8192_mixed24_f64<>+0x37F8(SB)/8, $8178
DATA bitrev_size8192_mixed24_f64<>+0x3800(SB)/8, $26
DATA bitrev_size8192_mixed24_f64<>+0x3808(SB)/8, $2074
DATA bitrev_size8192_mixed24_f64<>+0x3810(SB)/8, $4122
DATA bitrev_size8192_mixed24_f64<>+0x3818(SB)/8, $6170
DATA bitrev_size8192_mixed24_f64<>+0x3820(SB)/8, $538
DATA bitrev_size8192_mixed24_f64<>+0x3828(SB)/8, $2586
DATA bitrev_size8192_mixed24_f64<>+0x3830(SB)/8, $4634
DATA bitrev_size8192_mixed24_f64<>+0x3838(SB)/8, $6682
DATA bitrev_size8192_mixed24_f64<>+0x3840(SB)/8, $1050
DATA bitrev_size8192_mixed24_f64<>+0x3848(SB)/8, $3098
DATA bitrev_size8192_mixed24_f64<>+0x3850(SB)/8, $5146
DATA bitrev_size8192_mixed24_f64<>+0x3858(SB)/8, $7194
DATA bitrev_size8192_mixed24_f64<>+0x3860(SB)/8, $1562
DATA bitrev_size8192_mixed24_f64<>+0x3868(SB)/8, $3610
DATA bitrev_size8192_mixed24_f64<>+0x3870(SB)/8, $5658
DATA bitrev_size8192_mixed24_f64<>+0x3878(SB)/8, $7706
DATA bitrev_size8192_mixed24_f64<>+0x3880(SB)/8, $154
DATA bitrev_size8192_mixed24_f64<>+0x3888(SB)/8, $2202
DATA bitrev_size8192_mixed24_f64<>+0x3890(SB)/8, $4250
DATA bitrev_size8192_mixed24_f64<>+0x3898(SB)/8, $6298
DATA bitrev_size8192_mixed24_f64<>+0x38A0(SB)/8, $666
DATA bitrev_size8192_mixed24_f64<>+0x38A8(SB)/8, $2714
DATA bitrev_size8192_mixed24_f64<>+0x38B0(SB)/8, $4762
DATA bitrev_size8192_mixed24_f64<>+0x38B8(SB)/8, $6810
DATA bitrev_size8192_mixed24_f64<>+0x38C0(SB)/8, $1178
DATA bitrev_size8192_mixed24_f64<>+0x38C8(SB)/8, $3226
DATA bitrev_size8192_mixed24_f64<>+0x38D0(SB)/8, $5274
DATA bitrev_size8192_mixed24_f64<>+0x38D8(SB)/8, $7322
DATA bitrev_size8192_mixed24_f64<>+0x38E0(SB)/8, $1690
DATA bitrev_size8192_mixed24_f64<>+0x38E8(SB)/8, $3738
DATA bitrev_size8192_mixed24_f64<>+0x38F0(SB)/8, $5786
DATA bitrev_size8192_mixed24_f64<>+0x38F8(SB)/8, $7834
DATA bitrev_size8192_mixed24_f64<>+0x3900(SB)/8, $282
DATA bitrev_size8192_mixed24_f64<>+0x3908(SB)/8, $2330
DATA bitrev_size8192_mixed24_f64<>+0x3910(SB)/8, $4378
DATA bitrev_size8192_mixed24_f64<>+0x3918(SB)/8, $6426
DATA bitrev_size8192_mixed24_f64<>+0x3920(SB)/8, $794
DATA bitrev_size8192_mixed24_f64<>+0x3928(SB)/8, $2842
DATA bitrev_size8192_mixed24_f64<>+0x3930(SB)/8, $4890
DATA bitrev_size8192_mixed24_f64<>+0x3938(SB)/8, $6938
DATA bitrev_size8192_mixed24_f64<>+0x3940(SB)/8, $1306
DATA bitrev_size8192_mixed24_f64<>+0x3948(SB)/8, $3354
DATA bitrev_size8192_mixed24_f64<>+0x3950(SB)/8, $5402
DATA bitrev_size8192_mixed24_f64<>+0x3958(SB)/8, $7450
DATA bitrev_size8192_mixed24_f64<>+0x3960(SB)/8, $1818
DATA bitrev_size8192_mixed24_f64<>+0x3968(SB)/8, $3866
DATA bitrev_size8192_mixed24_f64<>+0x3970(SB)/8, $5914
DATA bitrev_size8192_mixed24_f64<>+0x3978(SB)/8, $7962
DATA bitrev_size8192_mixed24_f64<>+0x3980(SB)/8, $410
DATA bitrev_size8192_mixed24_f64<>+0x3988(SB)/8, $2458
DATA bitrev_size8192_mixed24_f64<>+0x3990(SB)/8, $4506
DATA bitrev_size8192_mixed24_f64<>+0x3998(SB)/8, $6554
DATA bitrev_size8192_mixed24_f64<>+0x39A0(SB)/8, $922
DATA bitrev_size8192_mixed24_f64<>+0x39A8(SB)/8, $2970
DATA bitrev_size8192_mixed24_f64<>+0x39B0(SB)/8, $5018
DATA bitrev_size8192_mixed24_f64<>+0x39B8(SB)/8, $7066
DATA bitrev_size8192_mixed24_f64<>+0x39C0(SB)/8, $1434
DATA bitrev_size8192_mixed24_f64<>+0x39C8(SB)/8, $3482
DATA bitrev_size8192_mixed24_f64<>+0x39D0(SB)/8, $5530
DATA bitrev_size8192_mixed24_f64<>+0x39D8(SB)/8, $7578
DATA bitrev_size8192_mixed24_f64<>+0x39E0(SB)/8, $1946
DATA bitrev_size8192_mixed24_f64<>+0x39E8(SB)/8, $3994
DATA bitrev_size8192_mixed24_f64<>+0x39F0(SB)/8, $6042
DATA bitrev_size8192_mixed24_f64<>+0x39F8(SB)/8, $8090
DATA bitrev_size8192_mixed24_f64<>+0x3A00(SB)/8, $58
DATA bitrev_size8192_mixed24_f64<>+0x3A08(SB)/8, $2106
DATA bitrev_size8192_mixed24_f64<>+0x3A10(SB)/8, $4154
DATA bitrev_size8192_mixed24_f64<>+0x3A18(SB)/8, $6202
DATA bitrev_size8192_mixed24_f64<>+0x3A20(SB)/8, $570
DATA bitrev_size8192_mixed24_f64<>+0x3A28(SB)/8, $2618
DATA bitrev_size8192_mixed24_f64<>+0x3A30(SB)/8, $4666
DATA bitrev_size8192_mixed24_f64<>+0x3A38(SB)/8, $6714
DATA bitrev_size8192_mixed24_f64<>+0x3A40(SB)/8, $1082
DATA bitrev_size8192_mixed24_f64<>+0x3A48(SB)/8, $3130
DATA bitrev_size8192_mixed24_f64<>+0x3A50(SB)/8, $5178
DATA bitrev_size8192_mixed24_f64<>+0x3A58(SB)/8, $7226
DATA bitrev_size8192_mixed24_f64<>+0x3A60(SB)/8, $1594
DATA bitrev_size8192_mixed24_f64<>+0x3A68(SB)/8, $3642
DATA bitrev_size8192_mixed24_f64<>+0x3A70(SB)/8, $5690
DATA bitrev_size8192_mixed24_f64<>+0x3A78(SB)/8, $7738
DATA bitrev_size8192_mixed24_f64<>+0x3A80(SB)/8, $186
DATA bitrev_size8192_mixed24_f64<>+0x3A88(SB)/8, $2234
DATA bitrev_size8192_mixed24_f64<>+0x3A90(SB)/8, $4282
DATA bitrev_size8192_mixed24_f64<>+0x3A98(SB)/8, $6330
DATA bitrev_size8192_mixed24_f64<>+0x3AA0(SB)/8, $698
DATA bitrev_size8192_mixed24_f64<>+0x3AA8(SB)/8, $2746
DATA bitrev_size8192_mixed24_f64<>+0x3AB0(SB)/8, $4794
DATA bitrev_size8192_mixed24_f64<>+0x3AB8(SB)/8, $6842
DATA bitrev_size8192_mixed24_f64<>+0x3AC0(SB)/8, $1210
DATA bitrev_size8192_mixed24_f64<>+0x3AC8(SB)/8, $3258
DATA bitrev_size8192_mixed24_f64<>+0x3AD0(SB)/8, $5306
DATA bitrev_size8192_mixed24_f64<>+0x3AD8(SB)/8, $7354
DATA bitrev_size8192_mixed24_f64<>+0x3AE0(SB)/8, $1722
DATA bitrev_size8192_mixed24_f64<>+0x3AE8(SB)/8, $3770
DATA bitrev_size8192_mixed24_f64<>+0x3AF0(SB)/8, $5818
DATA bitrev_size8192_mixed24_f64<>+0x3AF8(SB)/8, $7866
DATA bitrev_size8192_mixed24_f64<>+0x3B00(SB)/8, $314
DATA bitrev_size8192_mixed24_f64<>+0x3B08(SB)/8, $2362
DATA bitrev_size8192_mixed24_f64<>+0x3B10(SB)/8, $4410
DATA bitrev_size8192_mixed24_f64<>+0x3B18(SB)/8, $6458
DATA bitrev_size8192_mixed24_f64<>+0x3B20(SB)/8, $826
DATA bitrev_size8192_mixed24_f64<>+0x3B28(SB)/8, $2874
DATA bitrev_size8192_mixed24_f64<>+0x3B30(SB)/8, $4922
DATA bitrev_size8192_mixed24_f64<>+0x3B38(SB)/8, $6970
DATA bitrev_size8192_mixed24_f64<>+0x3B40(SB)/8, $1338
DATA bitrev_size8192_mixed24_f64<>+0x3B48(SB)/8, $3386
DATA bitrev_size8192_mixed24_f64<>+0x3B50(SB)/8, $5434
DATA bitrev_size8192_mixed24_f64<>+0x3B58(SB)/8, $7482
DATA bitrev_size8192_mixed24_f64<>+0x3B60(SB)/8, $1850
DATA bitrev_size8192_mixed24_f64<>+0x3B68(SB)/8, $3898
DATA bitrev_size8192_mixed24_f64<>+0x3B70(SB)/8, $5946
DATA bitrev_size8192_mixed24_f64<>+0x3B78(SB)/8, $7994
DATA bitrev_size8192_mixed24_f64<>+0x3B80(SB)/8, $442
DATA bitrev_size8192_mixed24_f64<>+0x3B88(SB)/8, $2490
DATA bitrev_size8192_mixed24_f64<>+0x3B90(SB)/8, $4538
DATA bitrev_size8192_mixed24_f64<>+0x3B98(SB)/8, $6586
DATA bitrev_size8192_mixed24_f64<>+0x3BA0(SB)/8, $954
DATA bitrev_size8192_mixed24_f64<>+0x3BA8(SB)/8, $3002
DATA bitrev_size8192_mixed24_f64<>+0x3BB0(SB)/8, $5050
DATA bitrev_size8192_mixed24_f64<>+0x3BB8(SB)/8, $7098
DATA bitrev_size8192_mixed24_f64<>+0x3BC0(SB)/8, $1466
DATA bitrev_size8192_mixed24_f64<>+0x3BC8(SB)/8, $3514
DATA bitrev_size8192_mixed24_f64<>+0x3BD0(SB)/8, $5562
DATA bitrev_size8192_mixed24_f64<>+0x3BD8(SB)/8, $7610
DATA bitrev_size8192_mixed24_f64<>+0x3BE0(SB)/8, $1978
DATA bitrev_size8192_mixed24_f64<>+0x3BE8(SB)/8, $4026
DATA bitrev_size8192_mixed24_f64<>+0x3BF0(SB)/8, $6074
DATA bitrev_size8192_mixed24_f64<>+0x3BF8(SB)/8, $8122
DATA bitrev_size8192_mixed24_f64<>+0x3C00(SB)/8, $90
DATA bitrev_size8192_mixed24_f64<>+0x3C08(SB)/8, $2138
DATA bitrev_size8192_mixed24_f64<>+0x3C10(SB)/8, $4186
DATA bitrev_size8192_mixed24_f64<>+0x3C18(SB)/8, $6234
DATA bitrev_size8192_mixed24_f64<>+0x3C20(SB)/8, $602
DATA bitrev_size8192_mixed24_f64<>+0x3C28(SB)/8, $2650
DATA bitrev_size8192_mixed24_f64<>+0x3C30(SB)/8, $4698
DATA bitrev_size8192_mixed24_f64<>+0x3C38(SB)/8, $6746
DATA bitrev_size8192_mixed24_f64<>+0x3C40(SB)/8, $1114
DATA bitrev_size8192_mixed24_f64<>+0x3C48(SB)/8, $3162
DATA bitrev_size8192_mixed24_f64<>+0x3C50(SB)/8, $5210
DATA bitrev_size8192_mixed24_f64<>+0x3C58(SB)/8, $7258
DATA bitrev_size8192_mixed24_f64<>+0x3C60(SB)/8, $1626
DATA bitrev_size8192_mixed24_f64<>+0x3C68(SB)/8, $3674
DATA bitrev_size8192_mixed24_f64<>+0x3C70(SB)/8, $5722
DATA bitrev_size8192_mixed24_f64<>+0x3C78(SB)/8, $7770
DATA bitrev_size8192_mixed24_f64<>+0x3C80(SB)/8, $218
DATA bitrev_size8192_mixed24_f64<>+0x3C88(SB)/8, $2266
DATA bitrev_size8192_mixed24_f64<>+0x3C90(SB)/8, $4314
DATA bitrev_size8192_mixed24_f64<>+0x3C98(SB)/8, $6362
DATA bitrev_size8192_mixed24_f64<>+0x3CA0(SB)/8, $730
DATA bitrev_size8192_mixed24_f64<>+0x3CA8(SB)/8, $2778
DATA bitrev_size8192_mixed24_f64<>+0x3CB0(SB)/8, $4826
DATA bitrev_size8192_mixed24_f64<>+0x3CB8(SB)/8, $6874
DATA bitrev_size8192_mixed24_f64<>+0x3CC0(SB)/8, $1242
DATA bitrev_size8192_mixed24_f64<>+0x3CC8(SB)/8, $3290
DATA bitrev_size8192_mixed24_f64<>+0x3CD0(SB)/8, $5338
DATA bitrev_size8192_mixed24_f64<>+0x3CD8(SB)/8, $7386
DATA bitrev_size8192_mixed24_f64<>+0x3CE0(SB)/8, $1754
DATA bitrev_size8192_mixed24_f64<>+0x3CE8(SB)/8, $3802
DATA bitrev_size8192_mixed24_f64<>+0x3CF0(SB)/8, $5850
DATA bitrev_size8192_mixed24_f64<>+0x3CF8(SB)/8, $7898
DATA bitrev_size8192_mixed24_f64<>+0x3D00(SB)/8, $346
DATA bitrev_size8192_mixed24_f64<>+0x3D08(SB)/8, $2394
DATA bitrev_size8192_mixed24_f64<>+0x3D10(SB)/8, $4442
DATA bitrev_size8192_mixed24_f64<>+0x3D18(SB)/8, $6490
DATA bitrev_size8192_mixed24_f64<>+0x3D20(SB)/8, $858
DATA bitrev_size8192_mixed24_f64<>+0x3D28(SB)/8, $2906
DATA bitrev_size8192_mixed24_f64<>+0x3D30(SB)/8, $4954
DATA bitrev_size8192_mixed24_f64<>+0x3D38(SB)/8, $7002
DATA bitrev_size8192_mixed24_f64<>+0x3D40(SB)/8, $1370
DATA bitrev_size8192_mixed24_f64<>+0x3D48(SB)/8, $3418
DATA bitrev_size8192_mixed24_f64<>+0x3D50(SB)/8, $5466
DATA bitrev_size8192_mixed24_f64<>+0x3D58(SB)/8, $7514
DATA bitrev_size8192_mixed24_f64<>+0x3D60(SB)/8, $1882
DATA bitrev_size8192_mixed24_f64<>+0x3D68(SB)/8, $3930
DATA bitrev_size8192_mixed24_f64<>+0x3D70(SB)/8, $5978
DATA bitrev_size8192_mixed24_f64<>+0x3D78(SB)/8, $8026
DATA bitrev_size8192_mixed24_f64<>+0x3D80(SB)/8, $474
DATA bitrev_size8192_mixed24_f64<>+0x3D88(SB)/8, $2522
DATA bitrev_size8192_mixed24_f64<>+0x3D90(SB)/8, $4570
DATA bitrev_size8192_mixed24_f64<>+0x3D98(SB)/8, $6618
DATA bitrev_size8192_mixed24_f64<>+0x3DA0(SB)/8, $986
DATA bitrev_size8192_mixed24_f64<>+0x3DA8(SB)/8, $3034
DATA bitrev_size8192_mixed24_f64<>+0x3DB0(SB)/8, $5082
DATA bitrev_size8192_mixed24_f64<>+0x3DB8(SB)/8, $7130
DATA bitrev_size8192_mixed24_f64<>+0x3DC0(SB)/8, $1498
DATA bitrev_size8192_mixed24_f64<>+0x3DC8(SB)/8, $3546
DATA bitrev_size8192_mixed24_f64<>+0x3DD0(SB)/8, $5594
DATA bitrev_size8192_mixed24_f64<>+0x3DD8(SB)/8, $7642
DATA bitrev_size8192_mixed24_f64<>+0x3DE0(SB)/8, $2010
DATA bitrev_size8192_mixed24_f64<>+0x3DE8(SB)/8, $4058
DATA bitrev_size8192_mixed24_f64<>+0x3DF0(SB)/8, $6106
DATA bitrev_size8192_mixed24_f64<>+0x3DF8(SB)/8, $8154
DATA bitrev_size8192_mixed24_f64<>+0x3E00(SB)/8, $122
DATA bitrev_size8192_mixed24_f64<>+0x3E08(SB)/8, $2170
DATA bitrev_size8192_mixed24_f64<>+0x3E10(SB)/8, $4218
DATA bitrev_size8192_mixed24_f64<>+0x3E18(SB)/8, $6266
DATA bitrev_size8192_mixed24_f64<>+0x3E20(SB)/8, $634
DATA bitrev_size8192_mixed24_f64<>+0x3E28(SB)/8, $2682
DATA bitrev_size8192_mixed24_f64<>+0x3E30(SB)/8, $4730
DATA bitrev_size8192_mixed24_f64<>+0x3E38(SB)/8, $6778
DATA bitrev_size8192_mixed24_f64<>+0x3E40(SB)/8, $1146
DATA bitrev_size8192_mixed24_f64<>+0x3E48(SB)/8, $3194
DATA bitrev_size8192_mixed24_f64<>+0x3E50(SB)/8, $5242
DATA bitrev_size8192_mixed24_f64<>+0x3E58(SB)/8, $7290
DATA bitrev_size8192_mixed24_f64<>+0x3E60(SB)/8, $1658
DATA bitrev_size8192_mixed24_f64<>+0x3E68(SB)/8, $3706
DATA bitrev_size8192_mixed24_f64<>+0x3E70(SB)/8, $5754
DATA bitrev_size8192_mixed24_f64<>+0x3E78(SB)/8, $7802
DATA bitrev_size8192_mixed24_f64<>+0x3E80(SB)/8, $250
DATA bitrev_size8192_mixed24_f64<>+0x3E88(SB)/8, $2298
DATA bitrev_size8192_mixed24_f64<>+0x3E90(SB)/8, $4346
DATA bitrev_size8192_mixed24_f64<>+0x3E98(SB)/8, $6394
DATA bitrev_size8192_mixed24_f64<>+0x3EA0(SB)/8, $762
DATA bitrev_size8192_mixed24_f64<>+0x3EA8(SB)/8, $2810
DATA bitrev_size8192_mixed24_f64<>+0x3EB0(SB)/8, $4858
DATA bitrev_size8192_mixed24_f64<>+0x3EB8(SB)/8, $6906
DATA bitrev_size8192_mixed24_f64<>+0x3EC0(SB)/8, $1274
DATA bitrev_size8192_mixed24_f64<>+0x3EC8(SB)/8, $3322
DATA bitrev_size8192_mixed24_f64<>+0x3ED0(SB)/8, $5370
DATA bitrev_size8192_mixed24_f64<>+0x3ED8(SB)/8, $7418
DATA bitrev_size8192_mixed24_f64<>+0x3EE0(SB)/8, $1786
DATA bitrev_size8192_mixed24_f64<>+0x3EE8(SB)/8, $3834
DATA bitrev_size8192_mixed24_f64<>+0x3EF0(SB)/8, $5882
DATA bitrev_size8192_mixed24_f64<>+0x3EF8(SB)/8, $7930
DATA bitrev_size8192_mixed24_f64<>+0x3F00(SB)/8, $378
DATA bitrev_size8192_mixed24_f64<>+0x3F08(SB)/8, $2426
DATA bitrev_size8192_mixed24_f64<>+0x3F10(SB)/8, $4474
DATA bitrev_size8192_mixed24_f64<>+0x3F18(SB)/8, $6522
DATA bitrev_size8192_mixed24_f64<>+0x3F20(SB)/8, $890
DATA bitrev_size8192_mixed24_f64<>+0x3F28(SB)/8, $2938
DATA bitrev_size8192_mixed24_f64<>+0x3F30(SB)/8, $4986
DATA bitrev_size8192_mixed24_f64<>+0x3F38(SB)/8, $7034
DATA bitrev_size8192_mixed24_f64<>+0x3F40(SB)/8, $1402
DATA bitrev_size8192_mixed24_f64<>+0x3F48(SB)/8, $3450
DATA bitrev_size8192_mixed24_f64<>+0x3F50(SB)/8, $5498
DATA bitrev_size8192_mixed24_f64<>+0x3F58(SB)/8, $7546
DATA bitrev_size8192_mixed24_f64<>+0x3F60(SB)/8, $1914
DATA bitrev_size8192_mixed24_f64<>+0x3F68(SB)/8, $3962
DATA bitrev_size8192_mixed24_f64<>+0x3F70(SB)/8, $6010
DATA bitrev_size8192_mixed24_f64<>+0x3F78(SB)/8, $8058
DATA bitrev_size8192_mixed24_f64<>+0x3F80(SB)/8, $506
DATA bitrev_size8192_mixed24_f64<>+0x3F88(SB)/8, $2554
DATA bitrev_size8192_mixed24_f64<>+0x3F90(SB)/8, $4602
DATA bitrev_size8192_mixed24_f64<>+0x3F98(SB)/8, $6650
DATA bitrev_size8192_mixed24_f64<>+0x3FA0(SB)/8, $1018
DATA bitrev_size8192_mixed24_f64<>+0x3FA8(SB)/8, $3066
DATA bitrev_size8192_mixed24_f64<>+0x3FB0(SB)/8, $5114
DATA bitrev_size8192_mixed24_f64<>+0x3FB8(SB)/8, $7162
DATA bitrev_size8192_mixed24_f64<>+0x3FC0(SB)/8, $1530
DATA bitrev_size8192_mixed24_f64<>+0x3FC8(SB)/8, $3578
DATA bitrev_size8192_mixed24_f64<>+0x3FD0(SB)/8, $5626
DATA bitrev_size8192_mixed24_f64<>+0x3FD8(SB)/8, $7674
DATA bitrev_size8192_mixed24_f64<>+0x3FE0(SB)/8, $2042
DATA bitrev_size8192_mixed24_f64<>+0x3FE8(SB)/8, $4090
DATA bitrev_size8192_mixed24_f64<>+0x3FF0(SB)/8, $6138
DATA bitrev_size8192_mixed24_f64<>+0x3FF8(SB)/8, $8186
DATA bitrev_size8192_mixed24_f64<>+0x4000(SB)/8, $4
DATA bitrev_size8192_mixed24_f64<>+0x4008(SB)/8, $2052
DATA bitrev_size8192_mixed24_f64<>+0x4010(SB)/8, $4100
DATA bitrev_size8192_mixed24_f64<>+0x4018(SB)/8, $6148
DATA bitrev_size8192_mixed24_f64<>+0x4020(SB)/8, $516
DATA bitrev_size8192_mixed24_f64<>+0x4028(SB)/8, $2564
DATA bitrev_size8192_mixed24_f64<>+0x4030(SB)/8, $4612
DATA bitrev_size8192_mixed24_f64<>+0x4038(SB)/8, $6660
DATA bitrev_size8192_mixed24_f64<>+0x4040(SB)/8, $1028
DATA bitrev_size8192_mixed24_f64<>+0x4048(SB)/8, $3076
DATA bitrev_size8192_mixed24_f64<>+0x4050(SB)/8, $5124
DATA bitrev_size8192_mixed24_f64<>+0x4058(SB)/8, $7172
DATA bitrev_size8192_mixed24_f64<>+0x4060(SB)/8, $1540
DATA bitrev_size8192_mixed24_f64<>+0x4068(SB)/8, $3588
DATA bitrev_size8192_mixed24_f64<>+0x4070(SB)/8, $5636
DATA bitrev_size8192_mixed24_f64<>+0x4078(SB)/8, $7684
DATA bitrev_size8192_mixed24_f64<>+0x4080(SB)/8, $132
DATA bitrev_size8192_mixed24_f64<>+0x4088(SB)/8, $2180
DATA bitrev_size8192_mixed24_f64<>+0x4090(SB)/8, $4228
DATA bitrev_size8192_mixed24_f64<>+0x4098(SB)/8, $6276
DATA bitrev_size8192_mixed24_f64<>+0x40A0(SB)/8, $644
DATA bitrev_size8192_mixed24_f64<>+0x40A8(SB)/8, $2692
DATA bitrev_size8192_mixed24_f64<>+0x40B0(SB)/8, $4740
DATA bitrev_size8192_mixed24_f64<>+0x40B8(SB)/8, $6788
DATA bitrev_size8192_mixed24_f64<>+0x40C0(SB)/8, $1156
DATA bitrev_size8192_mixed24_f64<>+0x40C8(SB)/8, $3204
DATA bitrev_size8192_mixed24_f64<>+0x40D0(SB)/8, $5252
DATA bitrev_size8192_mixed24_f64<>+0x40D8(SB)/8, $7300
DATA bitrev_size8192_mixed24_f64<>+0x40E0(SB)/8, $1668
DATA bitrev_size8192_mixed24_f64<>+0x40E8(SB)/8, $3716
DATA bitrev_size8192_mixed24_f64<>+0x40F0(SB)/8, $5764
DATA bitrev_size8192_mixed24_f64<>+0x40F8(SB)/8, $7812
DATA bitrev_size8192_mixed24_f64<>+0x4100(SB)/8, $260
DATA bitrev_size8192_mixed24_f64<>+0x4108(SB)/8, $2308
DATA bitrev_size8192_mixed24_f64<>+0x4110(SB)/8, $4356
DATA bitrev_size8192_mixed24_f64<>+0x4118(SB)/8, $6404
DATA bitrev_size8192_mixed24_f64<>+0x4120(SB)/8, $772
DATA bitrev_size8192_mixed24_f64<>+0x4128(SB)/8, $2820
DATA bitrev_size8192_mixed24_f64<>+0x4130(SB)/8, $4868
DATA bitrev_size8192_mixed24_f64<>+0x4138(SB)/8, $6916
DATA bitrev_size8192_mixed24_f64<>+0x4140(SB)/8, $1284
DATA bitrev_size8192_mixed24_f64<>+0x4148(SB)/8, $3332
DATA bitrev_size8192_mixed24_f64<>+0x4150(SB)/8, $5380
DATA bitrev_size8192_mixed24_f64<>+0x4158(SB)/8, $7428
DATA bitrev_size8192_mixed24_f64<>+0x4160(SB)/8, $1796
DATA bitrev_size8192_mixed24_f64<>+0x4168(SB)/8, $3844
DATA bitrev_size8192_mixed24_f64<>+0x4170(SB)/8, $5892
DATA bitrev_size8192_mixed24_f64<>+0x4178(SB)/8, $7940
DATA bitrev_size8192_mixed24_f64<>+0x4180(SB)/8, $388
DATA bitrev_size8192_mixed24_f64<>+0x4188(SB)/8, $2436
DATA bitrev_size8192_mixed24_f64<>+0x4190(SB)/8, $4484
DATA bitrev_size8192_mixed24_f64<>+0x4198(SB)/8, $6532
DATA bitrev_size8192_mixed24_f64<>+0x41A0(SB)/8, $900
DATA bitrev_size8192_mixed24_f64<>+0x41A8(SB)/8, $2948
DATA bitrev_size8192_mixed24_f64<>+0x41B0(SB)/8, $4996
DATA bitrev_size8192_mixed24_f64<>+0x41B8(SB)/8, $7044
DATA bitrev_size8192_mixed24_f64<>+0x41C0(SB)/8, $1412
DATA bitrev_size8192_mixed24_f64<>+0x41C8(SB)/8, $3460
DATA bitrev_size8192_mixed24_f64<>+0x41D0(SB)/8, $5508
DATA bitrev_size8192_mixed24_f64<>+0x41D8(SB)/8, $7556
DATA bitrev_size8192_mixed24_f64<>+0x41E0(SB)/8, $1924
DATA bitrev_size8192_mixed24_f64<>+0x41E8(SB)/8, $3972
DATA bitrev_size8192_mixed24_f64<>+0x41F0(SB)/8, $6020
DATA bitrev_size8192_mixed24_f64<>+0x41F8(SB)/8, $8068
DATA bitrev_size8192_mixed24_f64<>+0x4200(SB)/8, $36
DATA bitrev_size8192_mixed24_f64<>+0x4208(SB)/8, $2084
DATA bitrev_size8192_mixed24_f64<>+0x4210(SB)/8, $4132
DATA bitrev_size8192_mixed24_f64<>+0x4218(SB)/8, $6180
DATA bitrev_size8192_mixed24_f64<>+0x4220(SB)/8, $548
DATA bitrev_size8192_mixed24_f64<>+0x4228(SB)/8, $2596
DATA bitrev_size8192_mixed24_f64<>+0x4230(SB)/8, $4644
DATA bitrev_size8192_mixed24_f64<>+0x4238(SB)/8, $6692
DATA bitrev_size8192_mixed24_f64<>+0x4240(SB)/8, $1060
DATA bitrev_size8192_mixed24_f64<>+0x4248(SB)/8, $3108
DATA bitrev_size8192_mixed24_f64<>+0x4250(SB)/8, $5156
DATA bitrev_size8192_mixed24_f64<>+0x4258(SB)/8, $7204
DATA bitrev_size8192_mixed24_f64<>+0x4260(SB)/8, $1572
DATA bitrev_size8192_mixed24_f64<>+0x4268(SB)/8, $3620
DATA bitrev_size8192_mixed24_f64<>+0x4270(SB)/8, $5668
DATA bitrev_size8192_mixed24_f64<>+0x4278(SB)/8, $7716
DATA bitrev_size8192_mixed24_f64<>+0x4280(SB)/8, $164
DATA bitrev_size8192_mixed24_f64<>+0x4288(SB)/8, $2212
DATA bitrev_size8192_mixed24_f64<>+0x4290(SB)/8, $4260
DATA bitrev_size8192_mixed24_f64<>+0x4298(SB)/8, $6308
DATA bitrev_size8192_mixed24_f64<>+0x42A0(SB)/8, $676
DATA bitrev_size8192_mixed24_f64<>+0x42A8(SB)/8, $2724
DATA bitrev_size8192_mixed24_f64<>+0x42B0(SB)/8, $4772
DATA bitrev_size8192_mixed24_f64<>+0x42B8(SB)/8, $6820
DATA bitrev_size8192_mixed24_f64<>+0x42C0(SB)/8, $1188
DATA bitrev_size8192_mixed24_f64<>+0x42C8(SB)/8, $3236
DATA bitrev_size8192_mixed24_f64<>+0x42D0(SB)/8, $5284
DATA bitrev_size8192_mixed24_f64<>+0x42D8(SB)/8, $7332
DATA bitrev_size8192_mixed24_f64<>+0x42E0(SB)/8, $1700
DATA bitrev_size8192_mixed24_f64<>+0x42E8(SB)/8, $3748
DATA bitrev_size8192_mixed24_f64<>+0x42F0(SB)/8, $5796
DATA bitrev_size8192_mixed24_f64<>+0x42F8(SB)/8, $7844
DATA bitrev_size8192_mixed24_f64<>+0x4300(SB)/8, $292
DATA bitrev_size8192_mixed24_f64<>+0x4308(SB)/8, $2340
DATA bitrev_size8192_mixed24_f64<>+0x4310(SB)/8, $4388
DATA bitrev_size8192_mixed24_f64<>+0x4318(SB)/8, $6436
DATA bitrev_size8192_mixed24_f64<>+0x4320(SB)/8, $804
DATA bitrev_size8192_mixed24_f64<>+0x4328(SB)/8, $2852
DATA bitrev_size8192_mixed24_f64<>+0x4330(SB)/8, $4900
DATA bitrev_size8192_mixed24_f64<>+0x4338(SB)/8, $6948
DATA bitrev_size8192_mixed24_f64<>+0x4340(SB)/8, $1316
DATA bitrev_size8192_mixed24_f64<>+0x4348(SB)/8, $3364
DATA bitrev_size8192_mixed24_f64<>+0x4350(SB)/8, $5412
DATA bitrev_size8192_mixed24_f64<>+0x4358(SB)/8, $7460
DATA bitrev_size8192_mixed24_f64<>+0x4360(SB)/8, $1828
DATA bitrev_size8192_mixed24_f64<>+0x4368(SB)/8, $3876
DATA bitrev_size8192_mixed24_f64<>+0x4370(SB)/8, $5924
DATA bitrev_size8192_mixed24_f64<>+0x4378(SB)/8, $7972
DATA bitrev_size8192_mixed24_f64<>+0x4380(SB)/8, $420
DATA bitrev_size8192_mixed24_f64<>+0x4388(SB)/8, $2468
DATA bitrev_size8192_mixed24_f64<>+0x4390(SB)/8, $4516
DATA bitrev_size8192_mixed24_f64<>+0x4398(SB)/8, $6564
DATA bitrev_size8192_mixed24_f64<>+0x43A0(SB)/8, $932
DATA bitrev_size8192_mixed24_f64<>+0x43A8(SB)/8, $2980
DATA bitrev_size8192_mixed24_f64<>+0x43B0(SB)/8, $5028
DATA bitrev_size8192_mixed24_f64<>+0x43B8(SB)/8, $7076
DATA bitrev_size8192_mixed24_f64<>+0x43C0(SB)/8, $1444
DATA bitrev_size8192_mixed24_f64<>+0x43C8(SB)/8, $3492
DATA bitrev_size8192_mixed24_f64<>+0x43D0(SB)/8, $5540
DATA bitrev_size8192_mixed24_f64<>+0x43D8(SB)/8, $7588
DATA bitrev_size8192_mixed24_f64<>+0x43E0(SB)/8, $1956
DATA bitrev_size8192_mixed24_f64<>+0x43E8(SB)/8, $4004
DATA bitrev_size8192_mixed24_f64<>+0x43F0(SB)/8, $6052
DATA bitrev_size8192_mixed24_f64<>+0x43F8(SB)/8, $8100
DATA bitrev_size8192_mixed24_f64<>+0x4400(SB)/8, $68
DATA bitrev_size8192_mixed24_f64<>+0x4408(SB)/8, $2116
DATA bitrev_size8192_mixed24_f64<>+0x4410(SB)/8, $4164
DATA bitrev_size8192_mixed24_f64<>+0x4418(SB)/8, $6212
DATA bitrev_size8192_mixed24_f64<>+0x4420(SB)/8, $580
DATA bitrev_size8192_mixed24_f64<>+0x4428(SB)/8, $2628
DATA bitrev_size8192_mixed24_f64<>+0x4430(SB)/8, $4676
DATA bitrev_size8192_mixed24_f64<>+0x4438(SB)/8, $6724
DATA bitrev_size8192_mixed24_f64<>+0x4440(SB)/8, $1092
DATA bitrev_size8192_mixed24_f64<>+0x4448(SB)/8, $3140
DATA bitrev_size8192_mixed24_f64<>+0x4450(SB)/8, $5188
DATA bitrev_size8192_mixed24_f64<>+0x4458(SB)/8, $7236
DATA bitrev_size8192_mixed24_f64<>+0x4460(SB)/8, $1604
DATA bitrev_size8192_mixed24_f64<>+0x4468(SB)/8, $3652
DATA bitrev_size8192_mixed24_f64<>+0x4470(SB)/8, $5700
DATA bitrev_size8192_mixed24_f64<>+0x4478(SB)/8, $7748
DATA bitrev_size8192_mixed24_f64<>+0x4480(SB)/8, $196
DATA bitrev_size8192_mixed24_f64<>+0x4488(SB)/8, $2244
DATA bitrev_size8192_mixed24_f64<>+0x4490(SB)/8, $4292
DATA bitrev_size8192_mixed24_f64<>+0x4498(SB)/8, $6340
DATA bitrev_size8192_mixed24_f64<>+0x44A0(SB)/8, $708
DATA bitrev_size8192_mixed24_f64<>+0x44A8(SB)/8, $2756
DATA bitrev_size8192_mixed24_f64<>+0x44B0(SB)/8, $4804
DATA bitrev_size8192_mixed24_f64<>+0x44B8(SB)/8, $6852
DATA bitrev_size8192_mixed24_f64<>+0x44C0(SB)/8, $1220
DATA bitrev_size8192_mixed24_f64<>+0x44C8(SB)/8, $3268
DATA bitrev_size8192_mixed24_f64<>+0x44D0(SB)/8, $5316
DATA bitrev_size8192_mixed24_f64<>+0x44D8(SB)/8, $7364
DATA bitrev_size8192_mixed24_f64<>+0x44E0(SB)/8, $1732
DATA bitrev_size8192_mixed24_f64<>+0x44E8(SB)/8, $3780
DATA bitrev_size8192_mixed24_f64<>+0x44F0(SB)/8, $5828
DATA bitrev_size8192_mixed24_f64<>+0x44F8(SB)/8, $7876
DATA bitrev_size8192_mixed24_f64<>+0x4500(SB)/8, $324
DATA bitrev_size8192_mixed24_f64<>+0x4508(SB)/8, $2372
DATA bitrev_size8192_mixed24_f64<>+0x4510(SB)/8, $4420
DATA bitrev_size8192_mixed24_f64<>+0x4518(SB)/8, $6468
DATA bitrev_size8192_mixed24_f64<>+0x4520(SB)/8, $836
DATA bitrev_size8192_mixed24_f64<>+0x4528(SB)/8, $2884
DATA bitrev_size8192_mixed24_f64<>+0x4530(SB)/8, $4932
DATA bitrev_size8192_mixed24_f64<>+0x4538(SB)/8, $6980
DATA bitrev_size8192_mixed24_f64<>+0x4540(SB)/8, $1348
DATA bitrev_size8192_mixed24_f64<>+0x4548(SB)/8, $3396
DATA bitrev_size8192_mixed24_f64<>+0x4550(SB)/8, $5444
DATA bitrev_size8192_mixed24_f64<>+0x4558(SB)/8, $7492
DATA bitrev_size8192_mixed24_f64<>+0x4560(SB)/8, $1860
DATA bitrev_size8192_mixed24_f64<>+0x4568(SB)/8, $3908
DATA bitrev_size8192_mixed24_f64<>+0x4570(SB)/8, $5956
DATA bitrev_size8192_mixed24_f64<>+0x4578(SB)/8, $8004
DATA bitrev_size8192_mixed24_f64<>+0x4580(SB)/8, $452
DATA bitrev_size8192_mixed24_f64<>+0x4588(SB)/8, $2500
DATA bitrev_size8192_mixed24_f64<>+0x4590(SB)/8, $4548
DATA bitrev_size8192_mixed24_f64<>+0x4598(SB)/8, $6596
DATA bitrev_size8192_mixed24_f64<>+0x45A0(SB)/8, $964
DATA bitrev_size8192_mixed24_f64<>+0x45A8(SB)/8, $3012
DATA bitrev_size8192_mixed24_f64<>+0x45B0(SB)/8, $5060
DATA bitrev_size8192_mixed24_f64<>+0x45B8(SB)/8, $7108
DATA bitrev_size8192_mixed24_f64<>+0x45C0(SB)/8, $1476
DATA bitrev_size8192_mixed24_f64<>+0x45C8(SB)/8, $3524
DATA bitrev_size8192_mixed24_f64<>+0x45D0(SB)/8, $5572
DATA bitrev_size8192_mixed24_f64<>+0x45D8(SB)/8, $7620
DATA bitrev_size8192_mixed24_f64<>+0x45E0(SB)/8, $1988
DATA bitrev_size8192_mixed24_f64<>+0x45E8(SB)/8, $4036
DATA bitrev_size8192_mixed24_f64<>+0x45F0(SB)/8, $6084
DATA bitrev_size8192_mixed24_f64<>+0x45F8(SB)/8, $8132
DATA bitrev_size8192_mixed24_f64<>+0x4600(SB)/8, $100
DATA bitrev_size8192_mixed24_f64<>+0x4608(SB)/8, $2148
DATA bitrev_size8192_mixed24_f64<>+0x4610(SB)/8, $4196
DATA bitrev_size8192_mixed24_f64<>+0x4618(SB)/8, $6244
DATA bitrev_size8192_mixed24_f64<>+0x4620(SB)/8, $612
DATA bitrev_size8192_mixed24_f64<>+0x4628(SB)/8, $2660
DATA bitrev_size8192_mixed24_f64<>+0x4630(SB)/8, $4708
DATA bitrev_size8192_mixed24_f64<>+0x4638(SB)/8, $6756
DATA bitrev_size8192_mixed24_f64<>+0x4640(SB)/8, $1124
DATA bitrev_size8192_mixed24_f64<>+0x4648(SB)/8, $3172
DATA bitrev_size8192_mixed24_f64<>+0x4650(SB)/8, $5220
DATA bitrev_size8192_mixed24_f64<>+0x4658(SB)/8, $7268
DATA bitrev_size8192_mixed24_f64<>+0x4660(SB)/8, $1636
DATA bitrev_size8192_mixed24_f64<>+0x4668(SB)/8, $3684
DATA bitrev_size8192_mixed24_f64<>+0x4670(SB)/8, $5732
DATA bitrev_size8192_mixed24_f64<>+0x4678(SB)/8, $7780
DATA bitrev_size8192_mixed24_f64<>+0x4680(SB)/8, $228
DATA bitrev_size8192_mixed24_f64<>+0x4688(SB)/8, $2276
DATA bitrev_size8192_mixed24_f64<>+0x4690(SB)/8, $4324
DATA bitrev_size8192_mixed24_f64<>+0x4698(SB)/8, $6372
DATA bitrev_size8192_mixed24_f64<>+0x46A0(SB)/8, $740
DATA bitrev_size8192_mixed24_f64<>+0x46A8(SB)/8, $2788
DATA bitrev_size8192_mixed24_f64<>+0x46B0(SB)/8, $4836
DATA bitrev_size8192_mixed24_f64<>+0x46B8(SB)/8, $6884
DATA bitrev_size8192_mixed24_f64<>+0x46C0(SB)/8, $1252
DATA bitrev_size8192_mixed24_f64<>+0x46C8(SB)/8, $3300
DATA bitrev_size8192_mixed24_f64<>+0x46D0(SB)/8, $5348
DATA bitrev_size8192_mixed24_f64<>+0x46D8(SB)/8, $7396
DATA bitrev_size8192_mixed24_f64<>+0x46E0(SB)/8, $1764
DATA bitrev_size8192_mixed24_f64<>+0x46E8(SB)/8, $3812
DATA bitrev_size8192_mixed24_f64<>+0x46F0(SB)/8, $5860
DATA bitrev_size8192_mixed24_f64<>+0x46F8(SB)/8, $7908
DATA bitrev_size8192_mixed24_f64<>+0x4700(SB)/8, $356
DATA bitrev_size8192_mixed24_f64<>+0x4708(SB)/8, $2404
DATA bitrev_size8192_mixed24_f64<>+0x4710(SB)/8, $4452
DATA bitrev_size8192_mixed24_f64<>+0x4718(SB)/8, $6500
DATA bitrev_size8192_mixed24_f64<>+0x4720(SB)/8, $868
DATA bitrev_size8192_mixed24_f64<>+0x4728(SB)/8, $2916
DATA bitrev_size8192_mixed24_f64<>+0x4730(SB)/8, $4964
DATA bitrev_size8192_mixed24_f64<>+0x4738(SB)/8, $7012
DATA bitrev_size8192_mixed24_f64<>+0x4740(SB)/8, $1380
DATA bitrev_size8192_mixed24_f64<>+0x4748(SB)/8, $3428
DATA bitrev_size8192_mixed24_f64<>+0x4750(SB)/8, $5476
DATA bitrev_size8192_mixed24_f64<>+0x4758(SB)/8, $7524
DATA bitrev_size8192_mixed24_f64<>+0x4760(SB)/8, $1892
DATA bitrev_size8192_mixed24_f64<>+0x4768(SB)/8, $3940
DATA bitrev_size8192_mixed24_f64<>+0x4770(SB)/8, $5988
DATA bitrev_size8192_mixed24_f64<>+0x4778(SB)/8, $8036
DATA bitrev_size8192_mixed24_f64<>+0x4780(SB)/8, $484
DATA bitrev_size8192_mixed24_f64<>+0x4788(SB)/8, $2532
DATA bitrev_size8192_mixed24_f64<>+0x4790(SB)/8, $4580
DATA bitrev_size8192_mixed24_f64<>+0x4798(SB)/8, $6628
DATA bitrev_size8192_mixed24_f64<>+0x47A0(SB)/8, $996
DATA bitrev_size8192_mixed24_f64<>+0x47A8(SB)/8, $3044
DATA bitrev_size8192_mixed24_f64<>+0x47B0(SB)/8, $5092
DATA bitrev_size8192_mixed24_f64<>+0x47B8(SB)/8, $7140
DATA bitrev_size8192_mixed24_f64<>+0x47C0(SB)/8, $1508
DATA bitrev_size8192_mixed24_f64<>+0x47C8(SB)/8, $3556
DATA bitrev_size8192_mixed24_f64<>+0x47D0(SB)/8, $5604
DATA bitrev_size8192_mixed24_f64<>+0x47D8(SB)/8, $7652
DATA bitrev_size8192_mixed24_f64<>+0x47E0(SB)/8, $2020
DATA bitrev_size8192_mixed24_f64<>+0x47E8(SB)/8, $4068
DATA bitrev_size8192_mixed24_f64<>+0x47F0(SB)/8, $6116
DATA bitrev_size8192_mixed24_f64<>+0x47F8(SB)/8, $8164
DATA bitrev_size8192_mixed24_f64<>+0x4800(SB)/8, $12
DATA bitrev_size8192_mixed24_f64<>+0x4808(SB)/8, $2060
DATA bitrev_size8192_mixed24_f64<>+0x4810(SB)/8, $4108
DATA bitrev_size8192_mixed24_f64<>+0x4818(SB)/8, $6156
DATA bitrev_size8192_mixed24_f64<>+0x4820(SB)/8, $524
DATA bitrev_size8192_mixed24_f64<>+0x4828(SB)/8, $2572
DATA bitrev_size8192_mixed24_f64<>+0x4830(SB)/8, $4620
DATA bitrev_size8192_mixed24_f64<>+0x4838(SB)/8, $6668
DATA bitrev_size8192_mixed24_f64<>+0x4840(SB)/8, $1036
DATA bitrev_size8192_mixed24_f64<>+0x4848(SB)/8, $3084
DATA bitrev_size8192_mixed24_f64<>+0x4850(SB)/8, $5132
DATA bitrev_size8192_mixed24_f64<>+0x4858(SB)/8, $7180
DATA bitrev_size8192_mixed24_f64<>+0x4860(SB)/8, $1548
DATA bitrev_size8192_mixed24_f64<>+0x4868(SB)/8, $3596
DATA bitrev_size8192_mixed24_f64<>+0x4870(SB)/8, $5644
DATA bitrev_size8192_mixed24_f64<>+0x4878(SB)/8, $7692
DATA bitrev_size8192_mixed24_f64<>+0x4880(SB)/8, $140
DATA bitrev_size8192_mixed24_f64<>+0x4888(SB)/8, $2188
DATA bitrev_size8192_mixed24_f64<>+0x4890(SB)/8, $4236
DATA bitrev_size8192_mixed24_f64<>+0x4898(SB)/8, $6284
DATA bitrev_size8192_mixed24_f64<>+0x48A0(SB)/8, $652
DATA bitrev_size8192_mixed24_f64<>+0x48A8(SB)/8, $2700
DATA bitrev_size8192_mixed24_f64<>+0x48B0(SB)/8, $4748
DATA bitrev_size8192_mixed24_f64<>+0x48B8(SB)/8, $6796
DATA bitrev_size8192_mixed24_f64<>+0x48C0(SB)/8, $1164
DATA bitrev_size8192_mixed24_f64<>+0x48C8(SB)/8, $3212
DATA bitrev_size8192_mixed24_f64<>+0x48D0(SB)/8, $5260
DATA bitrev_size8192_mixed24_f64<>+0x48D8(SB)/8, $7308
DATA bitrev_size8192_mixed24_f64<>+0x48E0(SB)/8, $1676
DATA bitrev_size8192_mixed24_f64<>+0x48E8(SB)/8, $3724
DATA bitrev_size8192_mixed24_f64<>+0x48F0(SB)/8, $5772
DATA bitrev_size8192_mixed24_f64<>+0x48F8(SB)/8, $7820
DATA bitrev_size8192_mixed24_f64<>+0x4900(SB)/8, $268
DATA bitrev_size8192_mixed24_f64<>+0x4908(SB)/8, $2316
DATA bitrev_size8192_mixed24_f64<>+0x4910(SB)/8, $4364
DATA bitrev_size8192_mixed24_f64<>+0x4918(SB)/8, $6412
DATA bitrev_size8192_mixed24_f64<>+0x4920(SB)/8, $780
DATA bitrev_size8192_mixed24_f64<>+0x4928(SB)/8, $2828
DATA bitrev_size8192_mixed24_f64<>+0x4930(SB)/8, $4876
DATA bitrev_size8192_mixed24_f64<>+0x4938(SB)/8, $6924
DATA bitrev_size8192_mixed24_f64<>+0x4940(SB)/8, $1292
DATA bitrev_size8192_mixed24_f64<>+0x4948(SB)/8, $3340
DATA bitrev_size8192_mixed24_f64<>+0x4950(SB)/8, $5388
DATA bitrev_size8192_mixed24_f64<>+0x4958(SB)/8, $7436
DATA bitrev_size8192_mixed24_f64<>+0x4960(SB)/8, $1804
DATA bitrev_size8192_mixed24_f64<>+0x4968(SB)/8, $3852
DATA bitrev_size8192_mixed24_f64<>+0x4970(SB)/8, $5900
DATA bitrev_size8192_mixed24_f64<>+0x4978(SB)/8, $7948
DATA bitrev_size8192_mixed24_f64<>+0x4980(SB)/8, $396
DATA bitrev_size8192_mixed24_f64<>+0x4988(SB)/8, $2444
DATA bitrev_size8192_mixed24_f64<>+0x4990(SB)/8, $4492
DATA bitrev_size8192_mixed24_f64<>+0x4998(SB)/8, $6540
DATA bitrev_size8192_mixed24_f64<>+0x49A0(SB)/8, $908
DATA bitrev_size8192_mixed24_f64<>+0x49A8(SB)/8, $2956
DATA bitrev_size8192_mixed24_f64<>+0x49B0(SB)/8, $5004
DATA bitrev_size8192_mixed24_f64<>+0x49B8(SB)/8, $7052
DATA bitrev_size8192_mixed24_f64<>+0x49C0(SB)/8, $1420
DATA bitrev_size8192_mixed24_f64<>+0x49C8(SB)/8, $3468
DATA bitrev_size8192_mixed24_f64<>+0x49D0(SB)/8, $5516
DATA bitrev_size8192_mixed24_f64<>+0x49D8(SB)/8, $7564
DATA bitrev_size8192_mixed24_f64<>+0x49E0(SB)/8, $1932
DATA bitrev_size8192_mixed24_f64<>+0x49E8(SB)/8, $3980
DATA bitrev_size8192_mixed24_f64<>+0x49F0(SB)/8, $6028
DATA bitrev_size8192_mixed24_f64<>+0x49F8(SB)/8, $8076
DATA bitrev_size8192_mixed24_f64<>+0x4A00(SB)/8, $44
DATA bitrev_size8192_mixed24_f64<>+0x4A08(SB)/8, $2092
DATA bitrev_size8192_mixed24_f64<>+0x4A10(SB)/8, $4140
DATA bitrev_size8192_mixed24_f64<>+0x4A18(SB)/8, $6188
DATA bitrev_size8192_mixed24_f64<>+0x4A20(SB)/8, $556
DATA bitrev_size8192_mixed24_f64<>+0x4A28(SB)/8, $2604
DATA bitrev_size8192_mixed24_f64<>+0x4A30(SB)/8, $4652
DATA bitrev_size8192_mixed24_f64<>+0x4A38(SB)/8, $6700
DATA bitrev_size8192_mixed24_f64<>+0x4A40(SB)/8, $1068
DATA bitrev_size8192_mixed24_f64<>+0x4A48(SB)/8, $3116
DATA bitrev_size8192_mixed24_f64<>+0x4A50(SB)/8, $5164
DATA bitrev_size8192_mixed24_f64<>+0x4A58(SB)/8, $7212
DATA bitrev_size8192_mixed24_f64<>+0x4A60(SB)/8, $1580
DATA bitrev_size8192_mixed24_f64<>+0x4A68(SB)/8, $3628
DATA bitrev_size8192_mixed24_f64<>+0x4A70(SB)/8, $5676
DATA bitrev_size8192_mixed24_f64<>+0x4A78(SB)/8, $7724
DATA bitrev_size8192_mixed24_f64<>+0x4A80(SB)/8, $172
DATA bitrev_size8192_mixed24_f64<>+0x4A88(SB)/8, $2220
DATA bitrev_size8192_mixed24_f64<>+0x4A90(SB)/8, $4268
DATA bitrev_size8192_mixed24_f64<>+0x4A98(SB)/8, $6316
DATA bitrev_size8192_mixed24_f64<>+0x4AA0(SB)/8, $684
DATA bitrev_size8192_mixed24_f64<>+0x4AA8(SB)/8, $2732
DATA bitrev_size8192_mixed24_f64<>+0x4AB0(SB)/8, $4780
DATA bitrev_size8192_mixed24_f64<>+0x4AB8(SB)/8, $6828
DATA bitrev_size8192_mixed24_f64<>+0x4AC0(SB)/8, $1196
DATA bitrev_size8192_mixed24_f64<>+0x4AC8(SB)/8, $3244
DATA bitrev_size8192_mixed24_f64<>+0x4AD0(SB)/8, $5292
DATA bitrev_size8192_mixed24_f64<>+0x4AD8(SB)/8, $7340
DATA bitrev_size8192_mixed24_f64<>+0x4AE0(SB)/8, $1708
DATA bitrev_size8192_mixed24_f64<>+0x4AE8(SB)/8, $3756
DATA bitrev_size8192_mixed24_f64<>+0x4AF0(SB)/8, $5804
DATA bitrev_size8192_mixed24_f64<>+0x4AF8(SB)/8, $7852
DATA bitrev_size8192_mixed24_f64<>+0x4B00(SB)/8, $300
DATA bitrev_size8192_mixed24_f64<>+0x4B08(SB)/8, $2348
DATA bitrev_size8192_mixed24_f64<>+0x4B10(SB)/8, $4396
DATA bitrev_size8192_mixed24_f64<>+0x4B18(SB)/8, $6444
DATA bitrev_size8192_mixed24_f64<>+0x4B20(SB)/8, $812
DATA bitrev_size8192_mixed24_f64<>+0x4B28(SB)/8, $2860
DATA bitrev_size8192_mixed24_f64<>+0x4B30(SB)/8, $4908
DATA bitrev_size8192_mixed24_f64<>+0x4B38(SB)/8, $6956
DATA bitrev_size8192_mixed24_f64<>+0x4B40(SB)/8, $1324
DATA bitrev_size8192_mixed24_f64<>+0x4B48(SB)/8, $3372
DATA bitrev_size8192_mixed24_f64<>+0x4B50(SB)/8, $5420
DATA bitrev_size8192_mixed24_f64<>+0x4B58(SB)/8, $7468
DATA bitrev_size8192_mixed24_f64<>+0x4B60(SB)/8, $1836
DATA bitrev_size8192_mixed24_f64<>+0x4B68(SB)/8, $3884
DATA bitrev_size8192_mixed24_f64<>+0x4B70(SB)/8, $5932
DATA bitrev_size8192_mixed24_f64<>+0x4B78(SB)/8, $7980
DATA bitrev_size8192_mixed24_f64<>+0x4B80(SB)/8, $428
DATA bitrev_size8192_mixed24_f64<>+0x4B88(SB)/8, $2476
DATA bitrev_size8192_mixed24_f64<>+0x4B90(SB)/8, $4524
DATA bitrev_size8192_mixed24_f64<>+0x4B98(SB)/8, $6572
DATA bitrev_size8192_mixed24_f64<>+0x4BA0(SB)/8, $940
DATA bitrev_size8192_mixed24_f64<>+0x4BA8(SB)/8, $2988
DATA bitrev_size8192_mixed24_f64<>+0x4BB0(SB)/8, $5036
DATA bitrev_size8192_mixed24_f64<>+0x4BB8(SB)/8, $7084
DATA bitrev_size8192_mixed24_f64<>+0x4BC0(SB)/8, $1452
DATA bitrev_size8192_mixed24_f64<>+0x4BC8(SB)/8, $3500
DATA bitrev_size8192_mixed24_f64<>+0x4BD0(SB)/8, $5548
DATA bitrev_size8192_mixed24_f64<>+0x4BD8(SB)/8, $7596
DATA bitrev_size8192_mixed24_f64<>+0x4BE0(SB)/8, $1964
DATA bitrev_size8192_mixed24_f64<>+0x4BE8(SB)/8, $4012
DATA bitrev_size8192_mixed24_f64<>+0x4BF0(SB)/8, $6060
DATA bitrev_size8192_mixed24_f64<>+0x4BF8(SB)/8, $8108
DATA bitrev_size8192_mixed24_f64<>+0x4C00(SB)/8, $76
DATA bitrev_size8192_mixed24_f64<>+0x4C08(SB)/8, $2124
DATA bitrev_size8192_mixed24_f64<>+0x4C10(SB)/8, $4172
DATA bitrev_size8192_mixed24_f64<>+0x4C18(SB)/8, $6220
DATA bitrev_size8192_mixed24_f64<>+0x4C20(SB)/8, $588
DATA bitrev_size8192_mixed24_f64<>+0x4C28(SB)/8, $2636
DATA bitrev_size8192_mixed24_f64<>+0x4C30(SB)/8, $4684
DATA bitrev_size8192_mixed24_f64<>+0x4C38(SB)/8, $6732
DATA bitrev_size8192_mixed24_f64<>+0x4C40(SB)/8, $1100
DATA bitrev_size8192_mixed24_f64<>+0x4C48(SB)/8, $3148
DATA bitrev_size8192_mixed24_f64<>+0x4C50(SB)/8, $5196
DATA bitrev_size8192_mixed24_f64<>+0x4C58(SB)/8, $7244
DATA bitrev_size8192_mixed24_f64<>+0x4C60(SB)/8, $1612
DATA bitrev_size8192_mixed24_f64<>+0x4C68(SB)/8, $3660
DATA bitrev_size8192_mixed24_f64<>+0x4C70(SB)/8, $5708
DATA bitrev_size8192_mixed24_f64<>+0x4C78(SB)/8, $7756
DATA bitrev_size8192_mixed24_f64<>+0x4C80(SB)/8, $204
DATA bitrev_size8192_mixed24_f64<>+0x4C88(SB)/8, $2252
DATA bitrev_size8192_mixed24_f64<>+0x4C90(SB)/8, $4300
DATA bitrev_size8192_mixed24_f64<>+0x4C98(SB)/8, $6348
DATA bitrev_size8192_mixed24_f64<>+0x4CA0(SB)/8, $716
DATA bitrev_size8192_mixed24_f64<>+0x4CA8(SB)/8, $2764
DATA bitrev_size8192_mixed24_f64<>+0x4CB0(SB)/8, $4812
DATA bitrev_size8192_mixed24_f64<>+0x4CB8(SB)/8, $6860
DATA bitrev_size8192_mixed24_f64<>+0x4CC0(SB)/8, $1228
DATA bitrev_size8192_mixed24_f64<>+0x4CC8(SB)/8, $3276
DATA bitrev_size8192_mixed24_f64<>+0x4CD0(SB)/8, $5324
DATA bitrev_size8192_mixed24_f64<>+0x4CD8(SB)/8, $7372
DATA bitrev_size8192_mixed24_f64<>+0x4CE0(SB)/8, $1740
DATA bitrev_size8192_mixed24_f64<>+0x4CE8(SB)/8, $3788
DATA bitrev_size8192_mixed24_f64<>+0x4CF0(SB)/8, $5836
DATA bitrev_size8192_mixed24_f64<>+0x4CF8(SB)/8, $7884
DATA bitrev_size8192_mixed24_f64<>+0x4D00(SB)/8, $332
DATA bitrev_size8192_mixed24_f64<>+0x4D08(SB)/8, $2380
DATA bitrev_size8192_mixed24_f64<>+0x4D10(SB)/8, $4428
DATA bitrev_size8192_mixed24_f64<>+0x4D18(SB)/8, $6476
DATA bitrev_size8192_mixed24_f64<>+0x4D20(SB)/8, $844
DATA bitrev_size8192_mixed24_f64<>+0x4D28(SB)/8, $2892
DATA bitrev_size8192_mixed24_f64<>+0x4D30(SB)/8, $4940
DATA bitrev_size8192_mixed24_f64<>+0x4D38(SB)/8, $6988
DATA bitrev_size8192_mixed24_f64<>+0x4D40(SB)/8, $1356
DATA bitrev_size8192_mixed24_f64<>+0x4D48(SB)/8, $3404
DATA bitrev_size8192_mixed24_f64<>+0x4D50(SB)/8, $5452
DATA bitrev_size8192_mixed24_f64<>+0x4D58(SB)/8, $7500
DATA bitrev_size8192_mixed24_f64<>+0x4D60(SB)/8, $1868
DATA bitrev_size8192_mixed24_f64<>+0x4D68(SB)/8, $3916
DATA bitrev_size8192_mixed24_f64<>+0x4D70(SB)/8, $5964
DATA bitrev_size8192_mixed24_f64<>+0x4D78(SB)/8, $8012
DATA bitrev_size8192_mixed24_f64<>+0x4D80(SB)/8, $460
DATA bitrev_size8192_mixed24_f64<>+0x4D88(SB)/8, $2508
DATA bitrev_size8192_mixed24_f64<>+0x4D90(SB)/8, $4556
DATA bitrev_size8192_mixed24_f64<>+0x4D98(SB)/8, $6604
DATA bitrev_size8192_mixed24_f64<>+0x4DA0(SB)/8, $972
DATA bitrev_size8192_mixed24_f64<>+0x4DA8(SB)/8, $3020
DATA bitrev_size8192_mixed24_f64<>+0x4DB0(SB)/8, $5068
DATA bitrev_size8192_mixed24_f64<>+0x4DB8(SB)/8, $7116
DATA bitrev_size8192_mixed24_f64<>+0x4DC0(SB)/8, $1484
DATA bitrev_size8192_mixed24_f64<>+0x4DC8(SB)/8, $3532
DATA bitrev_size8192_mixed24_f64<>+0x4DD0(SB)/8, $5580
DATA bitrev_size8192_mixed24_f64<>+0x4DD8(SB)/8, $7628
DATA bitrev_size8192_mixed24_f64<>+0x4DE0(SB)/8, $1996
DATA bitrev_size8192_mixed24_f64<>+0x4DE8(SB)/8, $4044
DATA bitrev_size8192_mixed24_f64<>+0x4DF0(SB)/8, $6092
DATA bitrev_size8192_mixed24_f64<>+0x4DF8(SB)/8, $8140
DATA bitrev_size8192_mixed24_f64<>+0x4E00(SB)/8, $108
DATA bitrev_size8192_mixed24_f64<>+0x4E08(SB)/8, $2156
DATA bitrev_size8192_mixed24_f64<>+0x4E10(SB)/8, $4204
DATA bitrev_size8192_mixed24_f64<>+0x4E18(SB)/8, $6252
DATA bitrev_size8192_mixed24_f64<>+0x4E20(SB)/8, $620
DATA bitrev_size8192_mixed24_f64<>+0x4E28(SB)/8, $2668
DATA bitrev_size8192_mixed24_f64<>+0x4E30(SB)/8, $4716
DATA bitrev_size8192_mixed24_f64<>+0x4E38(SB)/8, $6764
DATA bitrev_size8192_mixed24_f64<>+0x4E40(SB)/8, $1132
DATA bitrev_size8192_mixed24_f64<>+0x4E48(SB)/8, $3180
DATA bitrev_size8192_mixed24_f64<>+0x4E50(SB)/8, $5228
DATA bitrev_size8192_mixed24_f64<>+0x4E58(SB)/8, $7276
DATA bitrev_size8192_mixed24_f64<>+0x4E60(SB)/8, $1644
DATA bitrev_size8192_mixed24_f64<>+0x4E68(SB)/8, $3692
DATA bitrev_size8192_mixed24_f64<>+0x4E70(SB)/8, $5740
DATA bitrev_size8192_mixed24_f64<>+0x4E78(SB)/8, $7788
DATA bitrev_size8192_mixed24_f64<>+0x4E80(SB)/8, $236
DATA bitrev_size8192_mixed24_f64<>+0x4E88(SB)/8, $2284
DATA bitrev_size8192_mixed24_f64<>+0x4E90(SB)/8, $4332
DATA bitrev_size8192_mixed24_f64<>+0x4E98(SB)/8, $6380
DATA bitrev_size8192_mixed24_f64<>+0x4EA0(SB)/8, $748
DATA bitrev_size8192_mixed24_f64<>+0x4EA8(SB)/8, $2796
DATA bitrev_size8192_mixed24_f64<>+0x4EB0(SB)/8, $4844
DATA bitrev_size8192_mixed24_f64<>+0x4EB8(SB)/8, $6892
DATA bitrev_size8192_mixed24_f64<>+0x4EC0(SB)/8, $1260
DATA bitrev_size8192_mixed24_f64<>+0x4EC8(SB)/8, $3308
DATA bitrev_size8192_mixed24_f64<>+0x4ED0(SB)/8, $5356
DATA bitrev_size8192_mixed24_f64<>+0x4ED8(SB)/8, $7404
DATA bitrev_size8192_mixed24_f64<>+0x4EE0(SB)/8, $1772
DATA bitrev_size8192_mixed24_f64<>+0x4EE8(SB)/8, $3820
DATA bitrev_size8192_mixed24_f64<>+0x4EF0(SB)/8, $5868
DATA bitrev_size8192_mixed24_f64<>+0x4EF8(SB)/8, $7916
DATA bitrev_size8192_mixed24_f64<>+0x4F00(SB)/8, $364
DATA bitrev_size8192_mixed24_f64<>+0x4F08(SB)/8, $2412
DATA bitrev_size8192_mixed24_f64<>+0x4F10(SB)/8, $4460
DATA bitrev_size8192_mixed24_f64<>+0x4F18(SB)/8, $6508
DATA bitrev_size8192_mixed24_f64<>+0x4F20(SB)/8, $876
DATA bitrev_size8192_mixed24_f64<>+0x4F28(SB)/8, $2924
DATA bitrev_size8192_mixed24_f64<>+0x4F30(SB)/8, $4972
DATA bitrev_size8192_mixed24_f64<>+0x4F38(SB)/8, $7020
DATA bitrev_size8192_mixed24_f64<>+0x4F40(SB)/8, $1388
DATA bitrev_size8192_mixed24_f64<>+0x4F48(SB)/8, $3436
DATA bitrev_size8192_mixed24_f64<>+0x4F50(SB)/8, $5484
DATA bitrev_size8192_mixed24_f64<>+0x4F58(SB)/8, $7532
DATA bitrev_size8192_mixed24_f64<>+0x4F60(SB)/8, $1900
DATA bitrev_size8192_mixed24_f64<>+0x4F68(SB)/8, $3948
DATA bitrev_size8192_mixed24_f64<>+0x4F70(SB)/8, $5996
DATA bitrev_size8192_mixed24_f64<>+0x4F78(SB)/8, $8044
DATA bitrev_size8192_mixed24_f64<>+0x4F80(SB)/8, $492
DATA bitrev_size8192_mixed24_f64<>+0x4F88(SB)/8, $2540
DATA bitrev_size8192_mixed24_f64<>+0x4F90(SB)/8, $4588
DATA bitrev_size8192_mixed24_f64<>+0x4F98(SB)/8, $6636
DATA bitrev_size8192_mixed24_f64<>+0x4FA0(SB)/8, $1004
DATA bitrev_size8192_mixed24_f64<>+0x4FA8(SB)/8, $3052
DATA bitrev_size8192_mixed24_f64<>+0x4FB0(SB)/8, $5100
DATA bitrev_size8192_mixed24_f64<>+0x4FB8(SB)/8, $7148
DATA bitrev_size8192_mixed24_f64<>+0x4FC0(SB)/8, $1516
DATA bitrev_size8192_mixed24_f64<>+0x4FC8(SB)/8, $3564
DATA bitrev_size8192_mixed24_f64<>+0x4FD0(SB)/8, $5612
DATA bitrev_size8192_mixed24_f64<>+0x4FD8(SB)/8, $7660
DATA bitrev_size8192_mixed24_f64<>+0x4FE0(SB)/8, $2028
DATA bitrev_size8192_mixed24_f64<>+0x4FE8(SB)/8, $4076
DATA bitrev_size8192_mixed24_f64<>+0x4FF0(SB)/8, $6124
DATA bitrev_size8192_mixed24_f64<>+0x4FF8(SB)/8, $8172
DATA bitrev_size8192_mixed24_f64<>+0x5000(SB)/8, $20
DATA bitrev_size8192_mixed24_f64<>+0x5008(SB)/8, $2068
DATA bitrev_size8192_mixed24_f64<>+0x5010(SB)/8, $4116
DATA bitrev_size8192_mixed24_f64<>+0x5018(SB)/8, $6164
DATA bitrev_size8192_mixed24_f64<>+0x5020(SB)/8, $532
DATA bitrev_size8192_mixed24_f64<>+0x5028(SB)/8, $2580
DATA bitrev_size8192_mixed24_f64<>+0x5030(SB)/8, $4628
DATA bitrev_size8192_mixed24_f64<>+0x5038(SB)/8, $6676
DATA bitrev_size8192_mixed24_f64<>+0x5040(SB)/8, $1044
DATA bitrev_size8192_mixed24_f64<>+0x5048(SB)/8, $3092
DATA bitrev_size8192_mixed24_f64<>+0x5050(SB)/8, $5140
DATA bitrev_size8192_mixed24_f64<>+0x5058(SB)/8, $7188
DATA bitrev_size8192_mixed24_f64<>+0x5060(SB)/8, $1556
DATA bitrev_size8192_mixed24_f64<>+0x5068(SB)/8, $3604
DATA bitrev_size8192_mixed24_f64<>+0x5070(SB)/8, $5652
DATA bitrev_size8192_mixed24_f64<>+0x5078(SB)/8, $7700
DATA bitrev_size8192_mixed24_f64<>+0x5080(SB)/8, $148
DATA bitrev_size8192_mixed24_f64<>+0x5088(SB)/8, $2196
DATA bitrev_size8192_mixed24_f64<>+0x5090(SB)/8, $4244
DATA bitrev_size8192_mixed24_f64<>+0x5098(SB)/8, $6292
DATA bitrev_size8192_mixed24_f64<>+0x50A0(SB)/8, $660
DATA bitrev_size8192_mixed24_f64<>+0x50A8(SB)/8, $2708
DATA bitrev_size8192_mixed24_f64<>+0x50B0(SB)/8, $4756
DATA bitrev_size8192_mixed24_f64<>+0x50B8(SB)/8, $6804
DATA bitrev_size8192_mixed24_f64<>+0x50C0(SB)/8, $1172
DATA bitrev_size8192_mixed24_f64<>+0x50C8(SB)/8, $3220
DATA bitrev_size8192_mixed24_f64<>+0x50D0(SB)/8, $5268
DATA bitrev_size8192_mixed24_f64<>+0x50D8(SB)/8, $7316
DATA bitrev_size8192_mixed24_f64<>+0x50E0(SB)/8, $1684
DATA bitrev_size8192_mixed24_f64<>+0x50E8(SB)/8, $3732
DATA bitrev_size8192_mixed24_f64<>+0x50F0(SB)/8, $5780
DATA bitrev_size8192_mixed24_f64<>+0x50F8(SB)/8, $7828
DATA bitrev_size8192_mixed24_f64<>+0x5100(SB)/8, $276
DATA bitrev_size8192_mixed24_f64<>+0x5108(SB)/8, $2324
DATA bitrev_size8192_mixed24_f64<>+0x5110(SB)/8, $4372
DATA bitrev_size8192_mixed24_f64<>+0x5118(SB)/8, $6420
DATA bitrev_size8192_mixed24_f64<>+0x5120(SB)/8, $788
DATA bitrev_size8192_mixed24_f64<>+0x5128(SB)/8, $2836
DATA bitrev_size8192_mixed24_f64<>+0x5130(SB)/8, $4884
DATA bitrev_size8192_mixed24_f64<>+0x5138(SB)/8, $6932
DATA bitrev_size8192_mixed24_f64<>+0x5140(SB)/8, $1300
DATA bitrev_size8192_mixed24_f64<>+0x5148(SB)/8, $3348
DATA bitrev_size8192_mixed24_f64<>+0x5150(SB)/8, $5396
DATA bitrev_size8192_mixed24_f64<>+0x5158(SB)/8, $7444
DATA bitrev_size8192_mixed24_f64<>+0x5160(SB)/8, $1812
DATA bitrev_size8192_mixed24_f64<>+0x5168(SB)/8, $3860
DATA bitrev_size8192_mixed24_f64<>+0x5170(SB)/8, $5908
DATA bitrev_size8192_mixed24_f64<>+0x5178(SB)/8, $7956
DATA bitrev_size8192_mixed24_f64<>+0x5180(SB)/8, $404
DATA bitrev_size8192_mixed24_f64<>+0x5188(SB)/8, $2452
DATA bitrev_size8192_mixed24_f64<>+0x5190(SB)/8, $4500
DATA bitrev_size8192_mixed24_f64<>+0x5198(SB)/8, $6548
DATA bitrev_size8192_mixed24_f64<>+0x51A0(SB)/8, $916
DATA bitrev_size8192_mixed24_f64<>+0x51A8(SB)/8, $2964
DATA bitrev_size8192_mixed24_f64<>+0x51B0(SB)/8, $5012
DATA bitrev_size8192_mixed24_f64<>+0x51B8(SB)/8, $7060
DATA bitrev_size8192_mixed24_f64<>+0x51C0(SB)/8, $1428
DATA bitrev_size8192_mixed24_f64<>+0x51C8(SB)/8, $3476
DATA bitrev_size8192_mixed24_f64<>+0x51D0(SB)/8, $5524
DATA bitrev_size8192_mixed24_f64<>+0x51D8(SB)/8, $7572
DATA bitrev_size8192_mixed24_f64<>+0x51E0(SB)/8, $1940
DATA bitrev_size8192_mixed24_f64<>+0x51E8(SB)/8, $3988
DATA bitrev_size8192_mixed24_f64<>+0x51F0(SB)/8, $6036
DATA bitrev_size8192_mixed24_f64<>+0x51F8(SB)/8, $8084
DATA bitrev_size8192_mixed24_f64<>+0x5200(SB)/8, $52
DATA bitrev_size8192_mixed24_f64<>+0x5208(SB)/8, $2100
DATA bitrev_size8192_mixed24_f64<>+0x5210(SB)/8, $4148
DATA bitrev_size8192_mixed24_f64<>+0x5218(SB)/8, $6196
DATA bitrev_size8192_mixed24_f64<>+0x5220(SB)/8, $564
DATA bitrev_size8192_mixed24_f64<>+0x5228(SB)/8, $2612
DATA bitrev_size8192_mixed24_f64<>+0x5230(SB)/8, $4660
DATA bitrev_size8192_mixed24_f64<>+0x5238(SB)/8, $6708
DATA bitrev_size8192_mixed24_f64<>+0x5240(SB)/8, $1076
DATA bitrev_size8192_mixed24_f64<>+0x5248(SB)/8, $3124
DATA bitrev_size8192_mixed24_f64<>+0x5250(SB)/8, $5172
DATA bitrev_size8192_mixed24_f64<>+0x5258(SB)/8, $7220
DATA bitrev_size8192_mixed24_f64<>+0x5260(SB)/8, $1588
DATA bitrev_size8192_mixed24_f64<>+0x5268(SB)/8, $3636
DATA bitrev_size8192_mixed24_f64<>+0x5270(SB)/8, $5684
DATA bitrev_size8192_mixed24_f64<>+0x5278(SB)/8, $7732
DATA bitrev_size8192_mixed24_f64<>+0x5280(SB)/8, $180
DATA bitrev_size8192_mixed24_f64<>+0x5288(SB)/8, $2228
DATA bitrev_size8192_mixed24_f64<>+0x5290(SB)/8, $4276
DATA bitrev_size8192_mixed24_f64<>+0x5298(SB)/8, $6324
DATA bitrev_size8192_mixed24_f64<>+0x52A0(SB)/8, $692
DATA bitrev_size8192_mixed24_f64<>+0x52A8(SB)/8, $2740
DATA bitrev_size8192_mixed24_f64<>+0x52B0(SB)/8, $4788
DATA bitrev_size8192_mixed24_f64<>+0x52B8(SB)/8, $6836
DATA bitrev_size8192_mixed24_f64<>+0x52C0(SB)/8, $1204
DATA bitrev_size8192_mixed24_f64<>+0x52C8(SB)/8, $3252
DATA bitrev_size8192_mixed24_f64<>+0x52D0(SB)/8, $5300
DATA bitrev_size8192_mixed24_f64<>+0x52D8(SB)/8, $7348
DATA bitrev_size8192_mixed24_f64<>+0x52E0(SB)/8, $1716
DATA bitrev_size8192_mixed24_f64<>+0x52E8(SB)/8, $3764
DATA bitrev_size8192_mixed24_f64<>+0x52F0(SB)/8, $5812
DATA bitrev_size8192_mixed24_f64<>+0x52F8(SB)/8, $7860
DATA bitrev_size8192_mixed24_f64<>+0x5300(SB)/8, $308
DATA bitrev_size8192_mixed24_f64<>+0x5308(SB)/8, $2356
DATA bitrev_size8192_mixed24_f64<>+0x5310(SB)/8, $4404
DATA bitrev_size8192_mixed24_f64<>+0x5318(SB)/8, $6452
DATA bitrev_size8192_mixed24_f64<>+0x5320(SB)/8, $820
DATA bitrev_size8192_mixed24_f64<>+0x5328(SB)/8, $2868
DATA bitrev_size8192_mixed24_f64<>+0x5330(SB)/8, $4916
DATA bitrev_size8192_mixed24_f64<>+0x5338(SB)/8, $6964
DATA bitrev_size8192_mixed24_f64<>+0x5340(SB)/8, $1332
DATA bitrev_size8192_mixed24_f64<>+0x5348(SB)/8, $3380
DATA bitrev_size8192_mixed24_f64<>+0x5350(SB)/8, $5428
DATA bitrev_size8192_mixed24_f64<>+0x5358(SB)/8, $7476
DATA bitrev_size8192_mixed24_f64<>+0x5360(SB)/8, $1844
DATA bitrev_size8192_mixed24_f64<>+0x5368(SB)/8, $3892
DATA bitrev_size8192_mixed24_f64<>+0x5370(SB)/8, $5940
DATA bitrev_size8192_mixed24_f64<>+0x5378(SB)/8, $7988
DATA bitrev_size8192_mixed24_f64<>+0x5380(SB)/8, $436
DATA bitrev_size8192_mixed24_f64<>+0x5388(SB)/8, $2484
DATA bitrev_size8192_mixed24_f64<>+0x5390(SB)/8, $4532
DATA bitrev_size8192_mixed24_f64<>+0x5398(SB)/8, $6580
DATA bitrev_size8192_mixed24_f64<>+0x53A0(SB)/8, $948
DATA bitrev_size8192_mixed24_f64<>+0x53A8(SB)/8, $2996
DATA bitrev_size8192_mixed24_f64<>+0x53B0(SB)/8, $5044
DATA bitrev_size8192_mixed24_f64<>+0x53B8(SB)/8, $7092
DATA bitrev_size8192_mixed24_f64<>+0x53C0(SB)/8, $1460
DATA bitrev_size8192_mixed24_f64<>+0x53C8(SB)/8, $3508
DATA bitrev_size8192_mixed24_f64<>+0x53D0(SB)/8, $5556
DATA bitrev_size8192_mixed24_f64<>+0x53D8(SB)/8, $7604
DATA bitrev_size8192_mixed24_f64<>+0x53E0(SB)/8, $1972
DATA bitrev_size8192_mixed24_f64<>+0x53E8(SB)/8, $4020
DATA bitrev_size8192_mixed24_f64<>+0x53F0(SB)/8, $6068
DATA bitrev_size8192_mixed24_f64<>+0x53F8(SB)/8, $8116
DATA bitrev_size8192_mixed24_f64<>+0x5400(SB)/8, $84
DATA bitrev_size8192_mixed24_f64<>+0x5408(SB)/8, $2132
DATA bitrev_size8192_mixed24_f64<>+0x5410(SB)/8, $4180
DATA bitrev_size8192_mixed24_f64<>+0x5418(SB)/8, $6228
DATA bitrev_size8192_mixed24_f64<>+0x5420(SB)/8, $596
DATA bitrev_size8192_mixed24_f64<>+0x5428(SB)/8, $2644
DATA bitrev_size8192_mixed24_f64<>+0x5430(SB)/8, $4692
DATA bitrev_size8192_mixed24_f64<>+0x5438(SB)/8, $6740
DATA bitrev_size8192_mixed24_f64<>+0x5440(SB)/8, $1108
DATA bitrev_size8192_mixed24_f64<>+0x5448(SB)/8, $3156
DATA bitrev_size8192_mixed24_f64<>+0x5450(SB)/8, $5204
DATA bitrev_size8192_mixed24_f64<>+0x5458(SB)/8, $7252
DATA bitrev_size8192_mixed24_f64<>+0x5460(SB)/8, $1620
DATA bitrev_size8192_mixed24_f64<>+0x5468(SB)/8, $3668
DATA bitrev_size8192_mixed24_f64<>+0x5470(SB)/8, $5716
DATA bitrev_size8192_mixed24_f64<>+0x5478(SB)/8, $7764
DATA bitrev_size8192_mixed24_f64<>+0x5480(SB)/8, $212
DATA bitrev_size8192_mixed24_f64<>+0x5488(SB)/8, $2260
DATA bitrev_size8192_mixed24_f64<>+0x5490(SB)/8, $4308
DATA bitrev_size8192_mixed24_f64<>+0x5498(SB)/8, $6356
DATA bitrev_size8192_mixed24_f64<>+0x54A0(SB)/8, $724
DATA bitrev_size8192_mixed24_f64<>+0x54A8(SB)/8, $2772
DATA bitrev_size8192_mixed24_f64<>+0x54B0(SB)/8, $4820
DATA bitrev_size8192_mixed24_f64<>+0x54B8(SB)/8, $6868
DATA bitrev_size8192_mixed24_f64<>+0x54C0(SB)/8, $1236
DATA bitrev_size8192_mixed24_f64<>+0x54C8(SB)/8, $3284
DATA bitrev_size8192_mixed24_f64<>+0x54D0(SB)/8, $5332
DATA bitrev_size8192_mixed24_f64<>+0x54D8(SB)/8, $7380
DATA bitrev_size8192_mixed24_f64<>+0x54E0(SB)/8, $1748
DATA bitrev_size8192_mixed24_f64<>+0x54E8(SB)/8, $3796
DATA bitrev_size8192_mixed24_f64<>+0x54F0(SB)/8, $5844
DATA bitrev_size8192_mixed24_f64<>+0x54F8(SB)/8, $7892
DATA bitrev_size8192_mixed24_f64<>+0x5500(SB)/8, $340
DATA bitrev_size8192_mixed24_f64<>+0x5508(SB)/8, $2388
DATA bitrev_size8192_mixed24_f64<>+0x5510(SB)/8, $4436
DATA bitrev_size8192_mixed24_f64<>+0x5518(SB)/8, $6484
DATA bitrev_size8192_mixed24_f64<>+0x5520(SB)/8, $852
DATA bitrev_size8192_mixed24_f64<>+0x5528(SB)/8, $2900
DATA bitrev_size8192_mixed24_f64<>+0x5530(SB)/8, $4948
DATA bitrev_size8192_mixed24_f64<>+0x5538(SB)/8, $6996
DATA bitrev_size8192_mixed24_f64<>+0x5540(SB)/8, $1364
DATA bitrev_size8192_mixed24_f64<>+0x5548(SB)/8, $3412
DATA bitrev_size8192_mixed24_f64<>+0x5550(SB)/8, $5460
DATA bitrev_size8192_mixed24_f64<>+0x5558(SB)/8, $7508
DATA bitrev_size8192_mixed24_f64<>+0x5560(SB)/8, $1876
DATA bitrev_size8192_mixed24_f64<>+0x5568(SB)/8, $3924
DATA bitrev_size8192_mixed24_f64<>+0x5570(SB)/8, $5972
DATA bitrev_size8192_mixed24_f64<>+0x5578(SB)/8, $8020
DATA bitrev_size8192_mixed24_f64<>+0x5580(SB)/8, $468
DATA bitrev_size8192_mixed24_f64<>+0x5588(SB)/8, $2516
DATA bitrev_size8192_mixed24_f64<>+0x5590(SB)/8, $4564
DATA bitrev_size8192_mixed24_f64<>+0x5598(SB)/8, $6612
DATA bitrev_size8192_mixed24_f64<>+0x55A0(SB)/8, $980
DATA bitrev_size8192_mixed24_f64<>+0x55A8(SB)/8, $3028
DATA bitrev_size8192_mixed24_f64<>+0x55B0(SB)/8, $5076
DATA bitrev_size8192_mixed24_f64<>+0x55B8(SB)/8, $7124
DATA bitrev_size8192_mixed24_f64<>+0x55C0(SB)/8, $1492
DATA bitrev_size8192_mixed24_f64<>+0x55C8(SB)/8, $3540
DATA bitrev_size8192_mixed24_f64<>+0x55D0(SB)/8, $5588
DATA bitrev_size8192_mixed24_f64<>+0x55D8(SB)/8, $7636
DATA bitrev_size8192_mixed24_f64<>+0x55E0(SB)/8, $2004
DATA bitrev_size8192_mixed24_f64<>+0x55E8(SB)/8, $4052
DATA bitrev_size8192_mixed24_f64<>+0x55F0(SB)/8, $6100
DATA bitrev_size8192_mixed24_f64<>+0x55F8(SB)/8, $8148
DATA bitrev_size8192_mixed24_f64<>+0x5600(SB)/8, $116
DATA bitrev_size8192_mixed24_f64<>+0x5608(SB)/8, $2164
DATA bitrev_size8192_mixed24_f64<>+0x5610(SB)/8, $4212
DATA bitrev_size8192_mixed24_f64<>+0x5618(SB)/8, $6260
DATA bitrev_size8192_mixed24_f64<>+0x5620(SB)/8, $628
DATA bitrev_size8192_mixed24_f64<>+0x5628(SB)/8, $2676
DATA bitrev_size8192_mixed24_f64<>+0x5630(SB)/8, $4724
DATA bitrev_size8192_mixed24_f64<>+0x5638(SB)/8, $6772
DATA bitrev_size8192_mixed24_f64<>+0x5640(SB)/8, $1140
DATA bitrev_size8192_mixed24_f64<>+0x5648(SB)/8, $3188
DATA bitrev_size8192_mixed24_f64<>+0x5650(SB)/8, $5236
DATA bitrev_size8192_mixed24_f64<>+0x5658(SB)/8, $7284
DATA bitrev_size8192_mixed24_f64<>+0x5660(SB)/8, $1652
DATA bitrev_size8192_mixed24_f64<>+0x5668(SB)/8, $3700
DATA bitrev_size8192_mixed24_f64<>+0x5670(SB)/8, $5748
DATA bitrev_size8192_mixed24_f64<>+0x5678(SB)/8, $7796
DATA bitrev_size8192_mixed24_f64<>+0x5680(SB)/8, $244
DATA bitrev_size8192_mixed24_f64<>+0x5688(SB)/8, $2292
DATA bitrev_size8192_mixed24_f64<>+0x5690(SB)/8, $4340
DATA bitrev_size8192_mixed24_f64<>+0x5698(SB)/8, $6388
DATA bitrev_size8192_mixed24_f64<>+0x56A0(SB)/8, $756
DATA bitrev_size8192_mixed24_f64<>+0x56A8(SB)/8, $2804
DATA bitrev_size8192_mixed24_f64<>+0x56B0(SB)/8, $4852
DATA bitrev_size8192_mixed24_f64<>+0x56B8(SB)/8, $6900
DATA bitrev_size8192_mixed24_f64<>+0x56C0(SB)/8, $1268
DATA bitrev_size8192_mixed24_f64<>+0x56C8(SB)/8, $3316
DATA bitrev_size8192_mixed24_f64<>+0x56D0(SB)/8, $5364
DATA bitrev_size8192_mixed24_f64<>+0x56D8(SB)/8, $7412
DATA bitrev_size8192_mixed24_f64<>+0x56E0(SB)/8, $1780
DATA bitrev_size8192_mixed24_f64<>+0x56E8(SB)/8, $3828
DATA bitrev_size8192_mixed24_f64<>+0x56F0(SB)/8, $5876
DATA bitrev_size8192_mixed24_f64<>+0x56F8(SB)/8, $7924
DATA bitrev_size8192_mixed24_f64<>+0x5700(SB)/8, $372
DATA bitrev_size8192_mixed24_f64<>+0x5708(SB)/8, $2420
DATA bitrev_size8192_mixed24_f64<>+0x5710(SB)/8, $4468
DATA bitrev_size8192_mixed24_f64<>+0x5718(SB)/8, $6516
DATA bitrev_size8192_mixed24_f64<>+0x5720(SB)/8, $884
DATA bitrev_size8192_mixed24_f64<>+0x5728(SB)/8, $2932
DATA bitrev_size8192_mixed24_f64<>+0x5730(SB)/8, $4980
DATA bitrev_size8192_mixed24_f64<>+0x5738(SB)/8, $7028
DATA bitrev_size8192_mixed24_f64<>+0x5740(SB)/8, $1396
DATA bitrev_size8192_mixed24_f64<>+0x5748(SB)/8, $3444
DATA bitrev_size8192_mixed24_f64<>+0x5750(SB)/8, $5492
DATA bitrev_size8192_mixed24_f64<>+0x5758(SB)/8, $7540
DATA bitrev_size8192_mixed24_f64<>+0x5760(SB)/8, $1908
DATA bitrev_size8192_mixed24_f64<>+0x5768(SB)/8, $3956
DATA bitrev_size8192_mixed24_f64<>+0x5770(SB)/8, $6004
DATA bitrev_size8192_mixed24_f64<>+0x5778(SB)/8, $8052
DATA bitrev_size8192_mixed24_f64<>+0x5780(SB)/8, $500
DATA bitrev_size8192_mixed24_f64<>+0x5788(SB)/8, $2548
DATA bitrev_size8192_mixed24_f64<>+0x5790(SB)/8, $4596
DATA bitrev_size8192_mixed24_f64<>+0x5798(SB)/8, $6644
DATA bitrev_size8192_mixed24_f64<>+0x57A0(SB)/8, $1012
DATA bitrev_size8192_mixed24_f64<>+0x57A8(SB)/8, $3060
DATA bitrev_size8192_mixed24_f64<>+0x57B0(SB)/8, $5108
DATA bitrev_size8192_mixed24_f64<>+0x57B8(SB)/8, $7156
DATA bitrev_size8192_mixed24_f64<>+0x57C0(SB)/8, $1524
DATA bitrev_size8192_mixed24_f64<>+0x57C8(SB)/8, $3572
DATA bitrev_size8192_mixed24_f64<>+0x57D0(SB)/8, $5620
DATA bitrev_size8192_mixed24_f64<>+0x57D8(SB)/8, $7668
DATA bitrev_size8192_mixed24_f64<>+0x57E0(SB)/8, $2036
DATA bitrev_size8192_mixed24_f64<>+0x57E8(SB)/8, $4084
DATA bitrev_size8192_mixed24_f64<>+0x57F0(SB)/8, $6132
DATA bitrev_size8192_mixed24_f64<>+0x57F8(SB)/8, $8180
DATA bitrev_size8192_mixed24_f64<>+0x5800(SB)/8, $28
DATA bitrev_size8192_mixed24_f64<>+0x5808(SB)/8, $2076
DATA bitrev_size8192_mixed24_f64<>+0x5810(SB)/8, $4124
DATA bitrev_size8192_mixed24_f64<>+0x5818(SB)/8, $6172
DATA bitrev_size8192_mixed24_f64<>+0x5820(SB)/8, $540
DATA bitrev_size8192_mixed24_f64<>+0x5828(SB)/8, $2588
DATA bitrev_size8192_mixed24_f64<>+0x5830(SB)/8, $4636
DATA bitrev_size8192_mixed24_f64<>+0x5838(SB)/8, $6684
DATA bitrev_size8192_mixed24_f64<>+0x5840(SB)/8, $1052
DATA bitrev_size8192_mixed24_f64<>+0x5848(SB)/8, $3100
DATA bitrev_size8192_mixed24_f64<>+0x5850(SB)/8, $5148
DATA bitrev_size8192_mixed24_f64<>+0x5858(SB)/8, $7196
DATA bitrev_size8192_mixed24_f64<>+0x5860(SB)/8, $1564
DATA bitrev_size8192_mixed24_f64<>+0x5868(SB)/8, $3612
DATA bitrev_size8192_mixed24_f64<>+0x5870(SB)/8, $5660
DATA bitrev_size8192_mixed24_f64<>+0x5878(SB)/8, $7708
DATA bitrev_size8192_mixed24_f64<>+0x5880(SB)/8, $156
DATA bitrev_size8192_mixed24_f64<>+0x5888(SB)/8, $2204
DATA bitrev_size8192_mixed24_f64<>+0x5890(SB)/8, $4252
DATA bitrev_size8192_mixed24_f64<>+0x5898(SB)/8, $6300
DATA bitrev_size8192_mixed24_f64<>+0x58A0(SB)/8, $668
DATA bitrev_size8192_mixed24_f64<>+0x58A8(SB)/8, $2716
DATA bitrev_size8192_mixed24_f64<>+0x58B0(SB)/8, $4764
DATA bitrev_size8192_mixed24_f64<>+0x58B8(SB)/8, $6812
DATA bitrev_size8192_mixed24_f64<>+0x58C0(SB)/8, $1180
DATA bitrev_size8192_mixed24_f64<>+0x58C8(SB)/8, $3228
DATA bitrev_size8192_mixed24_f64<>+0x58D0(SB)/8, $5276
DATA bitrev_size8192_mixed24_f64<>+0x58D8(SB)/8, $7324
DATA bitrev_size8192_mixed24_f64<>+0x58E0(SB)/8, $1692
DATA bitrev_size8192_mixed24_f64<>+0x58E8(SB)/8, $3740
DATA bitrev_size8192_mixed24_f64<>+0x58F0(SB)/8, $5788
DATA bitrev_size8192_mixed24_f64<>+0x58F8(SB)/8, $7836
DATA bitrev_size8192_mixed24_f64<>+0x5900(SB)/8, $284
DATA bitrev_size8192_mixed24_f64<>+0x5908(SB)/8, $2332
DATA bitrev_size8192_mixed24_f64<>+0x5910(SB)/8, $4380
DATA bitrev_size8192_mixed24_f64<>+0x5918(SB)/8, $6428
DATA bitrev_size8192_mixed24_f64<>+0x5920(SB)/8, $796
DATA bitrev_size8192_mixed24_f64<>+0x5928(SB)/8, $2844
DATA bitrev_size8192_mixed24_f64<>+0x5930(SB)/8, $4892
DATA bitrev_size8192_mixed24_f64<>+0x5938(SB)/8, $6940
DATA bitrev_size8192_mixed24_f64<>+0x5940(SB)/8, $1308
DATA bitrev_size8192_mixed24_f64<>+0x5948(SB)/8, $3356
DATA bitrev_size8192_mixed24_f64<>+0x5950(SB)/8, $5404
DATA bitrev_size8192_mixed24_f64<>+0x5958(SB)/8, $7452
DATA bitrev_size8192_mixed24_f64<>+0x5960(SB)/8, $1820
DATA bitrev_size8192_mixed24_f64<>+0x5968(SB)/8, $3868
DATA bitrev_size8192_mixed24_f64<>+0x5970(SB)/8, $5916
DATA bitrev_size8192_mixed24_f64<>+0x5978(SB)/8, $7964
DATA bitrev_size8192_mixed24_f64<>+0x5980(SB)/8, $412
DATA bitrev_size8192_mixed24_f64<>+0x5988(SB)/8, $2460
DATA bitrev_size8192_mixed24_f64<>+0x5990(SB)/8, $4508
DATA bitrev_size8192_mixed24_f64<>+0x5998(SB)/8, $6556
DATA bitrev_size8192_mixed24_f64<>+0x59A0(SB)/8, $924
DATA bitrev_size8192_mixed24_f64<>+0x59A8(SB)/8, $2972
DATA bitrev_size8192_mixed24_f64<>+0x59B0(SB)/8, $5020
DATA bitrev_size8192_mixed24_f64<>+0x59B8(SB)/8, $7068
DATA bitrev_size8192_mixed24_f64<>+0x59C0(SB)/8, $1436
DATA bitrev_size8192_mixed24_f64<>+0x59C8(SB)/8, $3484
DATA bitrev_size8192_mixed24_f64<>+0x59D0(SB)/8, $5532
DATA bitrev_size8192_mixed24_f64<>+0x59D8(SB)/8, $7580
DATA bitrev_size8192_mixed24_f64<>+0x59E0(SB)/8, $1948
DATA bitrev_size8192_mixed24_f64<>+0x59E8(SB)/8, $3996
DATA bitrev_size8192_mixed24_f64<>+0x59F0(SB)/8, $6044
DATA bitrev_size8192_mixed24_f64<>+0x59F8(SB)/8, $8092
DATA bitrev_size8192_mixed24_f64<>+0x5A00(SB)/8, $60
DATA bitrev_size8192_mixed24_f64<>+0x5A08(SB)/8, $2108
DATA bitrev_size8192_mixed24_f64<>+0x5A10(SB)/8, $4156
DATA bitrev_size8192_mixed24_f64<>+0x5A18(SB)/8, $6204
DATA bitrev_size8192_mixed24_f64<>+0x5A20(SB)/8, $572
DATA bitrev_size8192_mixed24_f64<>+0x5A28(SB)/8, $2620
DATA bitrev_size8192_mixed24_f64<>+0x5A30(SB)/8, $4668
DATA bitrev_size8192_mixed24_f64<>+0x5A38(SB)/8, $6716
DATA bitrev_size8192_mixed24_f64<>+0x5A40(SB)/8, $1084
DATA bitrev_size8192_mixed24_f64<>+0x5A48(SB)/8, $3132
DATA bitrev_size8192_mixed24_f64<>+0x5A50(SB)/8, $5180
DATA bitrev_size8192_mixed24_f64<>+0x5A58(SB)/8, $7228
DATA bitrev_size8192_mixed24_f64<>+0x5A60(SB)/8, $1596
DATA bitrev_size8192_mixed24_f64<>+0x5A68(SB)/8, $3644
DATA bitrev_size8192_mixed24_f64<>+0x5A70(SB)/8, $5692
DATA bitrev_size8192_mixed24_f64<>+0x5A78(SB)/8, $7740
DATA bitrev_size8192_mixed24_f64<>+0x5A80(SB)/8, $188
DATA bitrev_size8192_mixed24_f64<>+0x5A88(SB)/8, $2236
DATA bitrev_size8192_mixed24_f64<>+0x5A90(SB)/8, $4284
DATA bitrev_size8192_mixed24_f64<>+0x5A98(SB)/8, $6332
DATA bitrev_size8192_mixed24_f64<>+0x5AA0(SB)/8, $700
DATA bitrev_size8192_mixed24_f64<>+0x5AA8(SB)/8, $2748
DATA bitrev_size8192_mixed24_f64<>+0x5AB0(SB)/8, $4796
DATA bitrev_size8192_mixed24_f64<>+0x5AB8(SB)/8, $6844
DATA bitrev_size8192_mixed24_f64<>+0x5AC0(SB)/8, $1212
DATA bitrev_size8192_mixed24_f64<>+0x5AC8(SB)/8, $3260
DATA bitrev_size8192_mixed24_f64<>+0x5AD0(SB)/8, $5308
DATA bitrev_size8192_mixed24_f64<>+0x5AD8(SB)/8, $7356
DATA bitrev_size8192_mixed24_f64<>+0x5AE0(SB)/8, $1724
DATA bitrev_size8192_mixed24_f64<>+0x5AE8(SB)/8, $3772
DATA bitrev_size8192_mixed24_f64<>+0x5AF0(SB)/8, $5820
DATA bitrev_size8192_mixed24_f64<>+0x5AF8(SB)/8, $7868
DATA bitrev_size8192_mixed24_f64<>+0x5B00(SB)/8, $316
DATA bitrev_size8192_mixed24_f64<>+0x5B08(SB)/8, $2364
DATA bitrev_size8192_mixed24_f64<>+0x5B10(SB)/8, $4412
DATA bitrev_size8192_mixed24_f64<>+0x5B18(SB)/8, $6460
DATA bitrev_size8192_mixed24_f64<>+0x5B20(SB)/8, $828
DATA bitrev_size8192_mixed24_f64<>+0x5B28(SB)/8, $2876
DATA bitrev_size8192_mixed24_f64<>+0x5B30(SB)/8, $4924
DATA bitrev_size8192_mixed24_f64<>+0x5B38(SB)/8, $6972
DATA bitrev_size8192_mixed24_f64<>+0x5B40(SB)/8, $1340
DATA bitrev_size8192_mixed24_f64<>+0x5B48(SB)/8, $3388
DATA bitrev_size8192_mixed24_f64<>+0x5B50(SB)/8, $5436
DATA bitrev_size8192_mixed24_f64<>+0x5B58(SB)/8, $7484
DATA bitrev_size8192_mixed24_f64<>+0x5B60(SB)/8, $1852
DATA bitrev_size8192_mixed24_f64<>+0x5B68(SB)/8, $3900
DATA bitrev_size8192_mixed24_f64<>+0x5B70(SB)/8, $5948
DATA bitrev_size8192_mixed24_f64<>+0x5B78(SB)/8, $7996
DATA bitrev_size8192_mixed24_f64<>+0x5B80(SB)/8, $444
DATA bitrev_size8192_mixed24_f64<>+0x5B88(SB)/8, $2492
DATA bitrev_size8192_mixed24_f64<>+0x5B90(SB)/8, $4540
DATA bitrev_size8192_mixed24_f64<>+0x5B98(SB)/8, $6588
DATA bitrev_size8192_mixed24_f64<>+0x5BA0(SB)/8, $956
DATA bitrev_size8192_mixed24_f64<>+0x5BA8(SB)/8, $3004
DATA bitrev_size8192_mixed24_f64<>+0x5BB0(SB)/8, $5052
DATA bitrev_size8192_mixed24_f64<>+0x5BB8(SB)/8, $7100
DATA bitrev_size8192_mixed24_f64<>+0x5BC0(SB)/8, $1468
DATA bitrev_size8192_mixed24_f64<>+0x5BC8(SB)/8, $3516
DATA bitrev_size8192_mixed24_f64<>+0x5BD0(SB)/8, $5564
DATA bitrev_size8192_mixed24_f64<>+0x5BD8(SB)/8, $7612
DATA bitrev_size8192_mixed24_f64<>+0x5BE0(SB)/8, $1980
DATA bitrev_size8192_mixed24_f64<>+0x5BE8(SB)/8, $4028
DATA bitrev_size8192_mixed24_f64<>+0x5BF0(SB)/8, $6076
DATA bitrev_size8192_mixed24_f64<>+0x5BF8(SB)/8, $8124
DATA bitrev_size8192_mixed24_f64<>+0x5C00(SB)/8, $92
DATA bitrev_size8192_mixed24_f64<>+0x5C08(SB)/8, $2140
DATA bitrev_size8192_mixed24_f64<>+0x5C10(SB)/8, $4188
DATA bitrev_size8192_mixed24_f64<>+0x5C18(SB)/8, $6236
DATA bitrev_size8192_mixed24_f64<>+0x5C20(SB)/8, $604
DATA bitrev_size8192_mixed24_f64<>+0x5C28(SB)/8, $2652
DATA bitrev_size8192_mixed24_f64<>+0x5C30(SB)/8, $4700
DATA bitrev_size8192_mixed24_f64<>+0x5C38(SB)/8, $6748
DATA bitrev_size8192_mixed24_f64<>+0x5C40(SB)/8, $1116
DATA bitrev_size8192_mixed24_f64<>+0x5C48(SB)/8, $3164
DATA bitrev_size8192_mixed24_f64<>+0x5C50(SB)/8, $5212
DATA bitrev_size8192_mixed24_f64<>+0x5C58(SB)/8, $7260
DATA bitrev_size8192_mixed24_f64<>+0x5C60(SB)/8, $1628
DATA bitrev_size8192_mixed24_f64<>+0x5C68(SB)/8, $3676
DATA bitrev_size8192_mixed24_f64<>+0x5C70(SB)/8, $5724
DATA bitrev_size8192_mixed24_f64<>+0x5C78(SB)/8, $7772
DATA bitrev_size8192_mixed24_f64<>+0x5C80(SB)/8, $220
DATA bitrev_size8192_mixed24_f64<>+0x5C88(SB)/8, $2268
DATA bitrev_size8192_mixed24_f64<>+0x5C90(SB)/8, $4316
DATA bitrev_size8192_mixed24_f64<>+0x5C98(SB)/8, $6364
DATA bitrev_size8192_mixed24_f64<>+0x5CA0(SB)/8, $732
DATA bitrev_size8192_mixed24_f64<>+0x5CA8(SB)/8, $2780
DATA bitrev_size8192_mixed24_f64<>+0x5CB0(SB)/8, $4828
DATA bitrev_size8192_mixed24_f64<>+0x5CB8(SB)/8, $6876
DATA bitrev_size8192_mixed24_f64<>+0x5CC0(SB)/8, $1244
DATA bitrev_size8192_mixed24_f64<>+0x5CC8(SB)/8, $3292
DATA bitrev_size8192_mixed24_f64<>+0x5CD0(SB)/8, $5340
DATA bitrev_size8192_mixed24_f64<>+0x5CD8(SB)/8, $7388
DATA bitrev_size8192_mixed24_f64<>+0x5CE0(SB)/8, $1756
DATA bitrev_size8192_mixed24_f64<>+0x5CE8(SB)/8, $3804
DATA bitrev_size8192_mixed24_f64<>+0x5CF0(SB)/8, $5852
DATA bitrev_size8192_mixed24_f64<>+0x5CF8(SB)/8, $7900
DATA bitrev_size8192_mixed24_f64<>+0x5D00(SB)/8, $348
DATA bitrev_size8192_mixed24_f64<>+0x5D08(SB)/8, $2396
DATA bitrev_size8192_mixed24_f64<>+0x5D10(SB)/8, $4444
DATA bitrev_size8192_mixed24_f64<>+0x5D18(SB)/8, $6492
DATA bitrev_size8192_mixed24_f64<>+0x5D20(SB)/8, $860
DATA bitrev_size8192_mixed24_f64<>+0x5D28(SB)/8, $2908
DATA bitrev_size8192_mixed24_f64<>+0x5D30(SB)/8, $4956
DATA bitrev_size8192_mixed24_f64<>+0x5D38(SB)/8, $7004
DATA bitrev_size8192_mixed24_f64<>+0x5D40(SB)/8, $1372
DATA bitrev_size8192_mixed24_f64<>+0x5D48(SB)/8, $3420
DATA bitrev_size8192_mixed24_f64<>+0x5D50(SB)/8, $5468
DATA bitrev_size8192_mixed24_f64<>+0x5D58(SB)/8, $7516
DATA bitrev_size8192_mixed24_f64<>+0x5D60(SB)/8, $1884
DATA bitrev_size8192_mixed24_f64<>+0x5D68(SB)/8, $3932
DATA bitrev_size8192_mixed24_f64<>+0x5D70(SB)/8, $5980
DATA bitrev_size8192_mixed24_f64<>+0x5D78(SB)/8, $8028
DATA bitrev_size8192_mixed24_f64<>+0x5D80(SB)/8, $476
DATA bitrev_size8192_mixed24_f64<>+0x5D88(SB)/8, $2524
DATA bitrev_size8192_mixed24_f64<>+0x5D90(SB)/8, $4572
DATA bitrev_size8192_mixed24_f64<>+0x5D98(SB)/8, $6620
DATA bitrev_size8192_mixed24_f64<>+0x5DA0(SB)/8, $988
DATA bitrev_size8192_mixed24_f64<>+0x5DA8(SB)/8, $3036
DATA bitrev_size8192_mixed24_f64<>+0x5DB0(SB)/8, $5084
DATA bitrev_size8192_mixed24_f64<>+0x5DB8(SB)/8, $7132
DATA bitrev_size8192_mixed24_f64<>+0x5DC0(SB)/8, $1500
DATA bitrev_size8192_mixed24_f64<>+0x5DC8(SB)/8, $3548
DATA bitrev_size8192_mixed24_f64<>+0x5DD0(SB)/8, $5596
DATA bitrev_size8192_mixed24_f64<>+0x5DD8(SB)/8, $7644
DATA bitrev_size8192_mixed24_f64<>+0x5DE0(SB)/8, $2012
DATA bitrev_size8192_mixed24_f64<>+0x5DE8(SB)/8, $4060
DATA bitrev_size8192_mixed24_f64<>+0x5DF0(SB)/8, $6108
DATA bitrev_size8192_mixed24_f64<>+0x5DF8(SB)/8, $8156
DATA bitrev_size8192_mixed24_f64<>+0x5E00(SB)/8, $124
DATA bitrev_size8192_mixed24_f64<>+0x5E08(SB)/8, $2172
DATA bitrev_size8192_mixed24_f64<>+0x5E10(SB)/8, $4220
DATA bitrev_size8192_mixed24_f64<>+0x5E18(SB)/8, $6268
DATA bitrev_size8192_mixed24_f64<>+0x5E20(SB)/8, $636
DATA bitrev_size8192_mixed24_f64<>+0x5E28(SB)/8, $2684
DATA bitrev_size8192_mixed24_f64<>+0x5E30(SB)/8, $4732
DATA bitrev_size8192_mixed24_f64<>+0x5E38(SB)/8, $6780
DATA bitrev_size8192_mixed24_f64<>+0x5E40(SB)/8, $1148
DATA bitrev_size8192_mixed24_f64<>+0x5E48(SB)/8, $3196
DATA bitrev_size8192_mixed24_f64<>+0x5E50(SB)/8, $5244
DATA bitrev_size8192_mixed24_f64<>+0x5E58(SB)/8, $7292
DATA bitrev_size8192_mixed24_f64<>+0x5E60(SB)/8, $1660
DATA bitrev_size8192_mixed24_f64<>+0x5E68(SB)/8, $3708
DATA bitrev_size8192_mixed24_f64<>+0x5E70(SB)/8, $5756
DATA bitrev_size8192_mixed24_f64<>+0x5E78(SB)/8, $7804
DATA bitrev_size8192_mixed24_f64<>+0x5E80(SB)/8, $252
DATA bitrev_size8192_mixed24_f64<>+0x5E88(SB)/8, $2300
DATA bitrev_size8192_mixed24_f64<>+0x5E90(SB)/8, $4348
DATA bitrev_size8192_mixed24_f64<>+0x5E98(SB)/8, $6396
DATA bitrev_size8192_mixed24_f64<>+0x5EA0(SB)/8, $764
DATA bitrev_size8192_mixed24_f64<>+0x5EA8(SB)/8, $2812
DATA bitrev_size8192_mixed24_f64<>+0x5EB0(SB)/8, $4860
DATA bitrev_size8192_mixed24_f64<>+0x5EB8(SB)/8, $6908
DATA bitrev_size8192_mixed24_f64<>+0x5EC0(SB)/8, $1276
DATA bitrev_size8192_mixed24_f64<>+0x5EC8(SB)/8, $3324
DATA bitrev_size8192_mixed24_f64<>+0x5ED0(SB)/8, $5372
DATA bitrev_size8192_mixed24_f64<>+0x5ED8(SB)/8, $7420
DATA bitrev_size8192_mixed24_f64<>+0x5EE0(SB)/8, $1788
DATA bitrev_size8192_mixed24_f64<>+0x5EE8(SB)/8, $3836
DATA bitrev_size8192_mixed24_f64<>+0x5EF0(SB)/8, $5884
DATA bitrev_size8192_mixed24_f64<>+0x5EF8(SB)/8, $7932
DATA bitrev_size8192_mixed24_f64<>+0x5F00(SB)/8, $380
DATA bitrev_size8192_mixed24_f64<>+0x5F08(SB)/8, $2428
DATA bitrev_size8192_mixed24_f64<>+0x5F10(SB)/8, $4476
DATA bitrev_size8192_mixed24_f64<>+0x5F18(SB)/8, $6524
DATA bitrev_size8192_mixed24_f64<>+0x5F20(SB)/8, $892
DATA bitrev_size8192_mixed24_f64<>+0x5F28(SB)/8, $2940
DATA bitrev_size8192_mixed24_f64<>+0x5F30(SB)/8, $4988
DATA bitrev_size8192_mixed24_f64<>+0x5F38(SB)/8, $7036
DATA bitrev_size8192_mixed24_f64<>+0x5F40(SB)/8, $1404
DATA bitrev_size8192_mixed24_f64<>+0x5F48(SB)/8, $3452
DATA bitrev_size8192_mixed24_f64<>+0x5F50(SB)/8, $5500
DATA bitrev_size8192_mixed24_f64<>+0x5F58(SB)/8, $7548
DATA bitrev_size8192_mixed24_f64<>+0x5F60(SB)/8, $1916
DATA bitrev_size8192_mixed24_f64<>+0x5F68(SB)/8, $3964
DATA bitrev_size8192_mixed24_f64<>+0x5F70(SB)/8, $6012
DATA bitrev_size8192_mixed24_f64<>+0x5F78(SB)/8, $8060
DATA bitrev_size8192_mixed24_f64<>+0x5F80(SB)/8, $508
DATA bitrev_size8192_mixed24_f64<>+0x5F88(SB)/8, $2556
DATA bitrev_size8192_mixed24_f64<>+0x5F90(SB)/8, $4604
DATA bitrev_size8192_mixed24_f64<>+0x5F98(SB)/8, $6652
DATA bitrev_size8192_mixed24_f64<>+0x5FA0(SB)/8, $1020
DATA bitrev_size8192_mixed24_f64<>+0x5FA8(SB)/8, $3068
DATA bitrev_size8192_mixed24_f64<>+0x5FB0(SB)/8, $5116
DATA bitrev_size8192_mixed24_f64<>+0x5FB8(SB)/8, $7164
DATA bitrev_size8192_mixed24_f64<>+0x5FC0(SB)/8, $1532
DATA bitrev_size8192_mixed24_f64<>+0x5FC8(SB)/8, $3580
DATA bitrev_size8192_mixed24_f64<>+0x5FD0(SB)/8, $5628
DATA bitrev_size8192_mixed24_f64<>+0x5FD8(SB)/8, $7676
DATA bitrev_size8192_mixed24_f64<>+0x5FE0(SB)/8, $2044
DATA bitrev_size8192_mixed24_f64<>+0x5FE8(SB)/8, $4092
DATA bitrev_size8192_mixed24_f64<>+0x5FF0(SB)/8, $6140
DATA bitrev_size8192_mixed24_f64<>+0x5FF8(SB)/8, $8188
DATA bitrev_size8192_mixed24_f64<>+0x6000(SB)/8, $6
DATA bitrev_size8192_mixed24_f64<>+0x6008(SB)/8, $2054
DATA bitrev_size8192_mixed24_f64<>+0x6010(SB)/8, $4102
DATA bitrev_size8192_mixed24_f64<>+0x6018(SB)/8, $6150
DATA bitrev_size8192_mixed24_f64<>+0x6020(SB)/8, $518
DATA bitrev_size8192_mixed24_f64<>+0x6028(SB)/8, $2566
DATA bitrev_size8192_mixed24_f64<>+0x6030(SB)/8, $4614
DATA bitrev_size8192_mixed24_f64<>+0x6038(SB)/8, $6662
DATA bitrev_size8192_mixed24_f64<>+0x6040(SB)/8, $1030
DATA bitrev_size8192_mixed24_f64<>+0x6048(SB)/8, $3078
DATA bitrev_size8192_mixed24_f64<>+0x6050(SB)/8, $5126
DATA bitrev_size8192_mixed24_f64<>+0x6058(SB)/8, $7174
DATA bitrev_size8192_mixed24_f64<>+0x6060(SB)/8, $1542
DATA bitrev_size8192_mixed24_f64<>+0x6068(SB)/8, $3590
DATA bitrev_size8192_mixed24_f64<>+0x6070(SB)/8, $5638
DATA bitrev_size8192_mixed24_f64<>+0x6078(SB)/8, $7686
DATA bitrev_size8192_mixed24_f64<>+0x6080(SB)/8, $134
DATA bitrev_size8192_mixed24_f64<>+0x6088(SB)/8, $2182
DATA bitrev_size8192_mixed24_f64<>+0x6090(SB)/8, $4230
DATA bitrev_size8192_mixed24_f64<>+0x6098(SB)/8, $6278
DATA bitrev_size8192_mixed24_f64<>+0x60A0(SB)/8, $646
DATA bitrev_size8192_mixed24_f64<>+0x60A8(SB)/8, $2694
DATA bitrev_size8192_mixed24_f64<>+0x60B0(SB)/8, $4742
DATA bitrev_size8192_mixed24_f64<>+0x60B8(SB)/8, $6790
DATA bitrev_size8192_mixed24_f64<>+0x60C0(SB)/8, $1158
DATA bitrev_size8192_mixed24_f64<>+0x60C8(SB)/8, $3206
DATA bitrev_size8192_mixed24_f64<>+0x60D0(SB)/8, $5254
DATA bitrev_size8192_mixed24_f64<>+0x60D8(SB)/8, $7302
DATA bitrev_size8192_mixed24_f64<>+0x60E0(SB)/8, $1670
DATA bitrev_size8192_mixed24_f64<>+0x60E8(SB)/8, $3718
DATA bitrev_size8192_mixed24_f64<>+0x60F0(SB)/8, $5766
DATA bitrev_size8192_mixed24_f64<>+0x60F8(SB)/8, $7814
DATA bitrev_size8192_mixed24_f64<>+0x6100(SB)/8, $262
DATA bitrev_size8192_mixed24_f64<>+0x6108(SB)/8, $2310
DATA bitrev_size8192_mixed24_f64<>+0x6110(SB)/8, $4358
DATA bitrev_size8192_mixed24_f64<>+0x6118(SB)/8, $6406
DATA bitrev_size8192_mixed24_f64<>+0x6120(SB)/8, $774
DATA bitrev_size8192_mixed24_f64<>+0x6128(SB)/8, $2822
DATA bitrev_size8192_mixed24_f64<>+0x6130(SB)/8, $4870
DATA bitrev_size8192_mixed24_f64<>+0x6138(SB)/8, $6918
DATA bitrev_size8192_mixed24_f64<>+0x6140(SB)/8, $1286
DATA bitrev_size8192_mixed24_f64<>+0x6148(SB)/8, $3334
DATA bitrev_size8192_mixed24_f64<>+0x6150(SB)/8, $5382
DATA bitrev_size8192_mixed24_f64<>+0x6158(SB)/8, $7430
DATA bitrev_size8192_mixed24_f64<>+0x6160(SB)/8, $1798
DATA bitrev_size8192_mixed24_f64<>+0x6168(SB)/8, $3846
DATA bitrev_size8192_mixed24_f64<>+0x6170(SB)/8, $5894
DATA bitrev_size8192_mixed24_f64<>+0x6178(SB)/8, $7942
DATA bitrev_size8192_mixed24_f64<>+0x6180(SB)/8, $390
DATA bitrev_size8192_mixed24_f64<>+0x6188(SB)/8, $2438
DATA bitrev_size8192_mixed24_f64<>+0x6190(SB)/8, $4486
DATA bitrev_size8192_mixed24_f64<>+0x6198(SB)/8, $6534
DATA bitrev_size8192_mixed24_f64<>+0x61A0(SB)/8, $902
DATA bitrev_size8192_mixed24_f64<>+0x61A8(SB)/8, $2950
DATA bitrev_size8192_mixed24_f64<>+0x61B0(SB)/8, $4998
DATA bitrev_size8192_mixed24_f64<>+0x61B8(SB)/8, $7046
DATA bitrev_size8192_mixed24_f64<>+0x61C0(SB)/8, $1414
DATA bitrev_size8192_mixed24_f64<>+0x61C8(SB)/8, $3462
DATA bitrev_size8192_mixed24_f64<>+0x61D0(SB)/8, $5510
DATA bitrev_size8192_mixed24_f64<>+0x61D8(SB)/8, $7558
DATA bitrev_size8192_mixed24_f64<>+0x61E0(SB)/8, $1926
DATA bitrev_size8192_mixed24_f64<>+0x61E8(SB)/8, $3974
DATA bitrev_size8192_mixed24_f64<>+0x61F0(SB)/8, $6022
DATA bitrev_size8192_mixed24_f64<>+0x61F8(SB)/8, $8070
DATA bitrev_size8192_mixed24_f64<>+0x6200(SB)/8, $38
DATA bitrev_size8192_mixed24_f64<>+0x6208(SB)/8, $2086
DATA bitrev_size8192_mixed24_f64<>+0x6210(SB)/8, $4134
DATA bitrev_size8192_mixed24_f64<>+0x6218(SB)/8, $6182
DATA bitrev_size8192_mixed24_f64<>+0x6220(SB)/8, $550
DATA bitrev_size8192_mixed24_f64<>+0x6228(SB)/8, $2598
DATA bitrev_size8192_mixed24_f64<>+0x6230(SB)/8, $4646
DATA bitrev_size8192_mixed24_f64<>+0x6238(SB)/8, $6694
DATA bitrev_size8192_mixed24_f64<>+0x6240(SB)/8, $1062
DATA bitrev_size8192_mixed24_f64<>+0x6248(SB)/8, $3110
DATA bitrev_size8192_mixed24_f64<>+0x6250(SB)/8, $5158
DATA bitrev_size8192_mixed24_f64<>+0x6258(SB)/8, $7206
DATA bitrev_size8192_mixed24_f64<>+0x6260(SB)/8, $1574
DATA bitrev_size8192_mixed24_f64<>+0x6268(SB)/8, $3622
DATA bitrev_size8192_mixed24_f64<>+0x6270(SB)/8, $5670
DATA bitrev_size8192_mixed24_f64<>+0x6278(SB)/8, $7718
DATA bitrev_size8192_mixed24_f64<>+0x6280(SB)/8, $166
DATA bitrev_size8192_mixed24_f64<>+0x6288(SB)/8, $2214
DATA bitrev_size8192_mixed24_f64<>+0x6290(SB)/8, $4262
DATA bitrev_size8192_mixed24_f64<>+0x6298(SB)/8, $6310
DATA bitrev_size8192_mixed24_f64<>+0x62A0(SB)/8, $678
DATA bitrev_size8192_mixed24_f64<>+0x62A8(SB)/8, $2726
DATA bitrev_size8192_mixed24_f64<>+0x62B0(SB)/8, $4774
DATA bitrev_size8192_mixed24_f64<>+0x62B8(SB)/8, $6822
DATA bitrev_size8192_mixed24_f64<>+0x62C0(SB)/8, $1190
DATA bitrev_size8192_mixed24_f64<>+0x62C8(SB)/8, $3238
DATA bitrev_size8192_mixed24_f64<>+0x62D0(SB)/8, $5286
DATA bitrev_size8192_mixed24_f64<>+0x62D8(SB)/8, $7334
DATA bitrev_size8192_mixed24_f64<>+0x62E0(SB)/8, $1702
DATA bitrev_size8192_mixed24_f64<>+0x62E8(SB)/8, $3750
DATA bitrev_size8192_mixed24_f64<>+0x62F0(SB)/8, $5798
DATA bitrev_size8192_mixed24_f64<>+0x62F8(SB)/8, $7846
DATA bitrev_size8192_mixed24_f64<>+0x6300(SB)/8, $294
DATA bitrev_size8192_mixed24_f64<>+0x6308(SB)/8, $2342
DATA bitrev_size8192_mixed24_f64<>+0x6310(SB)/8, $4390
DATA bitrev_size8192_mixed24_f64<>+0x6318(SB)/8, $6438
DATA bitrev_size8192_mixed24_f64<>+0x6320(SB)/8, $806
DATA bitrev_size8192_mixed24_f64<>+0x6328(SB)/8, $2854
DATA bitrev_size8192_mixed24_f64<>+0x6330(SB)/8, $4902
DATA bitrev_size8192_mixed24_f64<>+0x6338(SB)/8, $6950
DATA bitrev_size8192_mixed24_f64<>+0x6340(SB)/8, $1318
DATA bitrev_size8192_mixed24_f64<>+0x6348(SB)/8, $3366
DATA bitrev_size8192_mixed24_f64<>+0x6350(SB)/8, $5414
DATA bitrev_size8192_mixed24_f64<>+0x6358(SB)/8, $7462
DATA bitrev_size8192_mixed24_f64<>+0x6360(SB)/8, $1830
DATA bitrev_size8192_mixed24_f64<>+0x6368(SB)/8, $3878
DATA bitrev_size8192_mixed24_f64<>+0x6370(SB)/8, $5926
DATA bitrev_size8192_mixed24_f64<>+0x6378(SB)/8, $7974
DATA bitrev_size8192_mixed24_f64<>+0x6380(SB)/8, $422
DATA bitrev_size8192_mixed24_f64<>+0x6388(SB)/8, $2470
DATA bitrev_size8192_mixed24_f64<>+0x6390(SB)/8, $4518
DATA bitrev_size8192_mixed24_f64<>+0x6398(SB)/8, $6566
DATA bitrev_size8192_mixed24_f64<>+0x63A0(SB)/8, $934
DATA bitrev_size8192_mixed24_f64<>+0x63A8(SB)/8, $2982
DATA bitrev_size8192_mixed24_f64<>+0x63B0(SB)/8, $5030
DATA bitrev_size8192_mixed24_f64<>+0x63B8(SB)/8, $7078
DATA bitrev_size8192_mixed24_f64<>+0x63C0(SB)/8, $1446
DATA bitrev_size8192_mixed24_f64<>+0x63C8(SB)/8, $3494
DATA bitrev_size8192_mixed24_f64<>+0x63D0(SB)/8, $5542
DATA bitrev_size8192_mixed24_f64<>+0x63D8(SB)/8, $7590
DATA bitrev_size8192_mixed24_f64<>+0x63E0(SB)/8, $1958
DATA bitrev_size8192_mixed24_f64<>+0x63E8(SB)/8, $4006
DATA bitrev_size8192_mixed24_f64<>+0x63F0(SB)/8, $6054
DATA bitrev_size8192_mixed24_f64<>+0x63F8(SB)/8, $8102
DATA bitrev_size8192_mixed24_f64<>+0x6400(SB)/8, $70
DATA bitrev_size8192_mixed24_f64<>+0x6408(SB)/8, $2118
DATA bitrev_size8192_mixed24_f64<>+0x6410(SB)/8, $4166
DATA bitrev_size8192_mixed24_f64<>+0x6418(SB)/8, $6214
DATA bitrev_size8192_mixed24_f64<>+0x6420(SB)/8, $582
DATA bitrev_size8192_mixed24_f64<>+0x6428(SB)/8, $2630
DATA bitrev_size8192_mixed24_f64<>+0x6430(SB)/8, $4678
DATA bitrev_size8192_mixed24_f64<>+0x6438(SB)/8, $6726
DATA bitrev_size8192_mixed24_f64<>+0x6440(SB)/8, $1094
DATA bitrev_size8192_mixed24_f64<>+0x6448(SB)/8, $3142
DATA bitrev_size8192_mixed24_f64<>+0x6450(SB)/8, $5190
DATA bitrev_size8192_mixed24_f64<>+0x6458(SB)/8, $7238
DATA bitrev_size8192_mixed24_f64<>+0x6460(SB)/8, $1606
DATA bitrev_size8192_mixed24_f64<>+0x6468(SB)/8, $3654
DATA bitrev_size8192_mixed24_f64<>+0x6470(SB)/8, $5702
DATA bitrev_size8192_mixed24_f64<>+0x6478(SB)/8, $7750
DATA bitrev_size8192_mixed24_f64<>+0x6480(SB)/8, $198
DATA bitrev_size8192_mixed24_f64<>+0x6488(SB)/8, $2246
DATA bitrev_size8192_mixed24_f64<>+0x6490(SB)/8, $4294
DATA bitrev_size8192_mixed24_f64<>+0x6498(SB)/8, $6342
DATA bitrev_size8192_mixed24_f64<>+0x64A0(SB)/8, $710
DATA bitrev_size8192_mixed24_f64<>+0x64A8(SB)/8, $2758
DATA bitrev_size8192_mixed24_f64<>+0x64B0(SB)/8, $4806
DATA bitrev_size8192_mixed24_f64<>+0x64B8(SB)/8, $6854
DATA bitrev_size8192_mixed24_f64<>+0x64C0(SB)/8, $1222
DATA bitrev_size8192_mixed24_f64<>+0x64C8(SB)/8, $3270
DATA bitrev_size8192_mixed24_f64<>+0x64D0(SB)/8, $5318
DATA bitrev_size8192_mixed24_f64<>+0x64D8(SB)/8, $7366
DATA bitrev_size8192_mixed24_f64<>+0x64E0(SB)/8, $1734
DATA bitrev_size8192_mixed24_f64<>+0x64E8(SB)/8, $3782
DATA bitrev_size8192_mixed24_f64<>+0x64F0(SB)/8, $5830
DATA bitrev_size8192_mixed24_f64<>+0x64F8(SB)/8, $7878
DATA bitrev_size8192_mixed24_f64<>+0x6500(SB)/8, $326
DATA bitrev_size8192_mixed24_f64<>+0x6508(SB)/8, $2374
DATA bitrev_size8192_mixed24_f64<>+0x6510(SB)/8, $4422
DATA bitrev_size8192_mixed24_f64<>+0x6518(SB)/8, $6470
DATA bitrev_size8192_mixed24_f64<>+0x6520(SB)/8, $838
DATA bitrev_size8192_mixed24_f64<>+0x6528(SB)/8, $2886
DATA bitrev_size8192_mixed24_f64<>+0x6530(SB)/8, $4934
DATA bitrev_size8192_mixed24_f64<>+0x6538(SB)/8, $6982
DATA bitrev_size8192_mixed24_f64<>+0x6540(SB)/8, $1350
DATA bitrev_size8192_mixed24_f64<>+0x6548(SB)/8, $3398
DATA bitrev_size8192_mixed24_f64<>+0x6550(SB)/8, $5446
DATA bitrev_size8192_mixed24_f64<>+0x6558(SB)/8, $7494
DATA bitrev_size8192_mixed24_f64<>+0x6560(SB)/8, $1862
DATA bitrev_size8192_mixed24_f64<>+0x6568(SB)/8, $3910
DATA bitrev_size8192_mixed24_f64<>+0x6570(SB)/8, $5958
DATA bitrev_size8192_mixed24_f64<>+0x6578(SB)/8, $8006
DATA bitrev_size8192_mixed24_f64<>+0x6580(SB)/8, $454
DATA bitrev_size8192_mixed24_f64<>+0x6588(SB)/8, $2502
DATA bitrev_size8192_mixed24_f64<>+0x6590(SB)/8, $4550
DATA bitrev_size8192_mixed24_f64<>+0x6598(SB)/8, $6598
DATA bitrev_size8192_mixed24_f64<>+0x65A0(SB)/8, $966
DATA bitrev_size8192_mixed24_f64<>+0x65A8(SB)/8, $3014
DATA bitrev_size8192_mixed24_f64<>+0x65B0(SB)/8, $5062
DATA bitrev_size8192_mixed24_f64<>+0x65B8(SB)/8, $7110
DATA bitrev_size8192_mixed24_f64<>+0x65C0(SB)/8, $1478
DATA bitrev_size8192_mixed24_f64<>+0x65C8(SB)/8, $3526
DATA bitrev_size8192_mixed24_f64<>+0x65D0(SB)/8, $5574
DATA bitrev_size8192_mixed24_f64<>+0x65D8(SB)/8, $7622
DATA bitrev_size8192_mixed24_f64<>+0x65E0(SB)/8, $1990
DATA bitrev_size8192_mixed24_f64<>+0x65E8(SB)/8, $4038
DATA bitrev_size8192_mixed24_f64<>+0x65F0(SB)/8, $6086
DATA bitrev_size8192_mixed24_f64<>+0x65F8(SB)/8, $8134
DATA bitrev_size8192_mixed24_f64<>+0x6600(SB)/8, $102
DATA bitrev_size8192_mixed24_f64<>+0x6608(SB)/8, $2150
DATA bitrev_size8192_mixed24_f64<>+0x6610(SB)/8, $4198
DATA bitrev_size8192_mixed24_f64<>+0x6618(SB)/8, $6246
DATA bitrev_size8192_mixed24_f64<>+0x6620(SB)/8, $614
DATA bitrev_size8192_mixed24_f64<>+0x6628(SB)/8, $2662
DATA bitrev_size8192_mixed24_f64<>+0x6630(SB)/8, $4710
DATA bitrev_size8192_mixed24_f64<>+0x6638(SB)/8, $6758
DATA bitrev_size8192_mixed24_f64<>+0x6640(SB)/8, $1126
DATA bitrev_size8192_mixed24_f64<>+0x6648(SB)/8, $3174
DATA bitrev_size8192_mixed24_f64<>+0x6650(SB)/8, $5222
DATA bitrev_size8192_mixed24_f64<>+0x6658(SB)/8, $7270
DATA bitrev_size8192_mixed24_f64<>+0x6660(SB)/8, $1638
DATA bitrev_size8192_mixed24_f64<>+0x6668(SB)/8, $3686
DATA bitrev_size8192_mixed24_f64<>+0x6670(SB)/8, $5734
DATA bitrev_size8192_mixed24_f64<>+0x6678(SB)/8, $7782
DATA bitrev_size8192_mixed24_f64<>+0x6680(SB)/8, $230
DATA bitrev_size8192_mixed24_f64<>+0x6688(SB)/8, $2278
DATA bitrev_size8192_mixed24_f64<>+0x6690(SB)/8, $4326
DATA bitrev_size8192_mixed24_f64<>+0x6698(SB)/8, $6374
DATA bitrev_size8192_mixed24_f64<>+0x66A0(SB)/8, $742
DATA bitrev_size8192_mixed24_f64<>+0x66A8(SB)/8, $2790
DATA bitrev_size8192_mixed24_f64<>+0x66B0(SB)/8, $4838
DATA bitrev_size8192_mixed24_f64<>+0x66B8(SB)/8, $6886
DATA bitrev_size8192_mixed24_f64<>+0x66C0(SB)/8, $1254
DATA bitrev_size8192_mixed24_f64<>+0x66C8(SB)/8, $3302
DATA bitrev_size8192_mixed24_f64<>+0x66D0(SB)/8, $5350
DATA bitrev_size8192_mixed24_f64<>+0x66D8(SB)/8, $7398
DATA bitrev_size8192_mixed24_f64<>+0x66E0(SB)/8, $1766
DATA bitrev_size8192_mixed24_f64<>+0x66E8(SB)/8, $3814
DATA bitrev_size8192_mixed24_f64<>+0x66F0(SB)/8, $5862
DATA bitrev_size8192_mixed24_f64<>+0x66F8(SB)/8, $7910
DATA bitrev_size8192_mixed24_f64<>+0x6700(SB)/8, $358
DATA bitrev_size8192_mixed24_f64<>+0x6708(SB)/8, $2406
DATA bitrev_size8192_mixed24_f64<>+0x6710(SB)/8, $4454
DATA bitrev_size8192_mixed24_f64<>+0x6718(SB)/8, $6502
DATA bitrev_size8192_mixed24_f64<>+0x6720(SB)/8, $870
DATA bitrev_size8192_mixed24_f64<>+0x6728(SB)/8, $2918
DATA bitrev_size8192_mixed24_f64<>+0x6730(SB)/8, $4966
DATA bitrev_size8192_mixed24_f64<>+0x6738(SB)/8, $7014
DATA bitrev_size8192_mixed24_f64<>+0x6740(SB)/8, $1382
DATA bitrev_size8192_mixed24_f64<>+0x6748(SB)/8, $3430
DATA bitrev_size8192_mixed24_f64<>+0x6750(SB)/8, $5478
DATA bitrev_size8192_mixed24_f64<>+0x6758(SB)/8, $7526
DATA bitrev_size8192_mixed24_f64<>+0x6760(SB)/8, $1894
DATA bitrev_size8192_mixed24_f64<>+0x6768(SB)/8, $3942
DATA bitrev_size8192_mixed24_f64<>+0x6770(SB)/8, $5990
DATA bitrev_size8192_mixed24_f64<>+0x6778(SB)/8, $8038
DATA bitrev_size8192_mixed24_f64<>+0x6780(SB)/8, $486
DATA bitrev_size8192_mixed24_f64<>+0x6788(SB)/8, $2534
DATA bitrev_size8192_mixed24_f64<>+0x6790(SB)/8, $4582
DATA bitrev_size8192_mixed24_f64<>+0x6798(SB)/8, $6630
DATA bitrev_size8192_mixed24_f64<>+0x67A0(SB)/8, $998
DATA bitrev_size8192_mixed24_f64<>+0x67A8(SB)/8, $3046
DATA bitrev_size8192_mixed24_f64<>+0x67B0(SB)/8, $5094
DATA bitrev_size8192_mixed24_f64<>+0x67B8(SB)/8, $7142
DATA bitrev_size8192_mixed24_f64<>+0x67C0(SB)/8, $1510
DATA bitrev_size8192_mixed24_f64<>+0x67C8(SB)/8, $3558
DATA bitrev_size8192_mixed24_f64<>+0x67D0(SB)/8, $5606
DATA bitrev_size8192_mixed24_f64<>+0x67D8(SB)/8, $7654
DATA bitrev_size8192_mixed24_f64<>+0x67E0(SB)/8, $2022
DATA bitrev_size8192_mixed24_f64<>+0x67E8(SB)/8, $4070
DATA bitrev_size8192_mixed24_f64<>+0x67F0(SB)/8, $6118
DATA bitrev_size8192_mixed24_f64<>+0x67F8(SB)/8, $8166
DATA bitrev_size8192_mixed24_f64<>+0x6800(SB)/8, $14
DATA bitrev_size8192_mixed24_f64<>+0x6808(SB)/8, $2062
DATA bitrev_size8192_mixed24_f64<>+0x6810(SB)/8, $4110
DATA bitrev_size8192_mixed24_f64<>+0x6818(SB)/8, $6158
DATA bitrev_size8192_mixed24_f64<>+0x6820(SB)/8, $526
DATA bitrev_size8192_mixed24_f64<>+0x6828(SB)/8, $2574
DATA bitrev_size8192_mixed24_f64<>+0x6830(SB)/8, $4622
DATA bitrev_size8192_mixed24_f64<>+0x6838(SB)/8, $6670
DATA bitrev_size8192_mixed24_f64<>+0x6840(SB)/8, $1038
DATA bitrev_size8192_mixed24_f64<>+0x6848(SB)/8, $3086
DATA bitrev_size8192_mixed24_f64<>+0x6850(SB)/8, $5134
DATA bitrev_size8192_mixed24_f64<>+0x6858(SB)/8, $7182
DATA bitrev_size8192_mixed24_f64<>+0x6860(SB)/8, $1550
DATA bitrev_size8192_mixed24_f64<>+0x6868(SB)/8, $3598
DATA bitrev_size8192_mixed24_f64<>+0x6870(SB)/8, $5646
DATA bitrev_size8192_mixed24_f64<>+0x6878(SB)/8, $7694
DATA bitrev_size8192_mixed24_f64<>+0x6880(SB)/8, $142
DATA bitrev_size8192_mixed24_f64<>+0x6888(SB)/8, $2190
DATA bitrev_size8192_mixed24_f64<>+0x6890(SB)/8, $4238
DATA bitrev_size8192_mixed24_f64<>+0x6898(SB)/8, $6286
DATA bitrev_size8192_mixed24_f64<>+0x68A0(SB)/8, $654
DATA bitrev_size8192_mixed24_f64<>+0x68A8(SB)/8, $2702
DATA bitrev_size8192_mixed24_f64<>+0x68B0(SB)/8, $4750
DATA bitrev_size8192_mixed24_f64<>+0x68B8(SB)/8, $6798
DATA bitrev_size8192_mixed24_f64<>+0x68C0(SB)/8, $1166
DATA bitrev_size8192_mixed24_f64<>+0x68C8(SB)/8, $3214
DATA bitrev_size8192_mixed24_f64<>+0x68D0(SB)/8, $5262
DATA bitrev_size8192_mixed24_f64<>+0x68D8(SB)/8, $7310
DATA bitrev_size8192_mixed24_f64<>+0x68E0(SB)/8, $1678
DATA bitrev_size8192_mixed24_f64<>+0x68E8(SB)/8, $3726
DATA bitrev_size8192_mixed24_f64<>+0x68F0(SB)/8, $5774
DATA bitrev_size8192_mixed24_f64<>+0x68F8(SB)/8, $7822
DATA bitrev_size8192_mixed24_f64<>+0x6900(SB)/8, $270
DATA bitrev_size8192_mixed24_f64<>+0x6908(SB)/8, $2318
DATA bitrev_size8192_mixed24_f64<>+0x6910(SB)/8, $4366
DATA bitrev_size8192_mixed24_f64<>+0x6918(SB)/8, $6414
DATA bitrev_size8192_mixed24_f64<>+0x6920(SB)/8, $782
DATA bitrev_size8192_mixed24_f64<>+0x6928(SB)/8, $2830
DATA bitrev_size8192_mixed24_f64<>+0x6930(SB)/8, $4878
DATA bitrev_size8192_mixed24_f64<>+0x6938(SB)/8, $6926
DATA bitrev_size8192_mixed24_f64<>+0x6940(SB)/8, $1294
DATA bitrev_size8192_mixed24_f64<>+0x6948(SB)/8, $3342
DATA bitrev_size8192_mixed24_f64<>+0x6950(SB)/8, $5390
DATA bitrev_size8192_mixed24_f64<>+0x6958(SB)/8, $7438
DATA bitrev_size8192_mixed24_f64<>+0x6960(SB)/8, $1806
DATA bitrev_size8192_mixed24_f64<>+0x6968(SB)/8, $3854
DATA bitrev_size8192_mixed24_f64<>+0x6970(SB)/8, $5902
DATA bitrev_size8192_mixed24_f64<>+0x6978(SB)/8, $7950
DATA bitrev_size8192_mixed24_f64<>+0x6980(SB)/8, $398
DATA bitrev_size8192_mixed24_f64<>+0x6988(SB)/8, $2446
DATA bitrev_size8192_mixed24_f64<>+0x6990(SB)/8, $4494
DATA bitrev_size8192_mixed24_f64<>+0x6998(SB)/8, $6542
DATA bitrev_size8192_mixed24_f64<>+0x69A0(SB)/8, $910
DATA bitrev_size8192_mixed24_f64<>+0x69A8(SB)/8, $2958
DATA bitrev_size8192_mixed24_f64<>+0x69B0(SB)/8, $5006
DATA bitrev_size8192_mixed24_f64<>+0x69B8(SB)/8, $7054
DATA bitrev_size8192_mixed24_f64<>+0x69C0(SB)/8, $1422
DATA bitrev_size8192_mixed24_f64<>+0x69C8(SB)/8, $3470
DATA bitrev_size8192_mixed24_f64<>+0x69D0(SB)/8, $5518
DATA bitrev_size8192_mixed24_f64<>+0x69D8(SB)/8, $7566
DATA bitrev_size8192_mixed24_f64<>+0x69E0(SB)/8, $1934
DATA bitrev_size8192_mixed24_f64<>+0x69E8(SB)/8, $3982
DATA bitrev_size8192_mixed24_f64<>+0x69F0(SB)/8, $6030
DATA bitrev_size8192_mixed24_f64<>+0x69F8(SB)/8, $8078
DATA bitrev_size8192_mixed24_f64<>+0x6A00(SB)/8, $46
DATA bitrev_size8192_mixed24_f64<>+0x6A08(SB)/8, $2094
DATA bitrev_size8192_mixed24_f64<>+0x6A10(SB)/8, $4142
DATA bitrev_size8192_mixed24_f64<>+0x6A18(SB)/8, $6190
DATA bitrev_size8192_mixed24_f64<>+0x6A20(SB)/8, $558
DATA bitrev_size8192_mixed24_f64<>+0x6A28(SB)/8, $2606
DATA bitrev_size8192_mixed24_f64<>+0x6A30(SB)/8, $4654
DATA bitrev_size8192_mixed24_f64<>+0x6A38(SB)/8, $6702
DATA bitrev_size8192_mixed24_f64<>+0x6A40(SB)/8, $1070
DATA bitrev_size8192_mixed24_f64<>+0x6A48(SB)/8, $3118
DATA bitrev_size8192_mixed24_f64<>+0x6A50(SB)/8, $5166
DATA bitrev_size8192_mixed24_f64<>+0x6A58(SB)/8, $7214
DATA bitrev_size8192_mixed24_f64<>+0x6A60(SB)/8, $1582
DATA bitrev_size8192_mixed24_f64<>+0x6A68(SB)/8, $3630
DATA bitrev_size8192_mixed24_f64<>+0x6A70(SB)/8, $5678
DATA bitrev_size8192_mixed24_f64<>+0x6A78(SB)/8, $7726
DATA bitrev_size8192_mixed24_f64<>+0x6A80(SB)/8, $174
DATA bitrev_size8192_mixed24_f64<>+0x6A88(SB)/8, $2222
DATA bitrev_size8192_mixed24_f64<>+0x6A90(SB)/8, $4270
DATA bitrev_size8192_mixed24_f64<>+0x6A98(SB)/8, $6318
DATA bitrev_size8192_mixed24_f64<>+0x6AA0(SB)/8, $686
DATA bitrev_size8192_mixed24_f64<>+0x6AA8(SB)/8, $2734
DATA bitrev_size8192_mixed24_f64<>+0x6AB0(SB)/8, $4782
DATA bitrev_size8192_mixed24_f64<>+0x6AB8(SB)/8, $6830
DATA bitrev_size8192_mixed24_f64<>+0x6AC0(SB)/8, $1198
DATA bitrev_size8192_mixed24_f64<>+0x6AC8(SB)/8, $3246
DATA bitrev_size8192_mixed24_f64<>+0x6AD0(SB)/8, $5294
DATA bitrev_size8192_mixed24_f64<>+0x6AD8(SB)/8, $7342
DATA bitrev_size8192_mixed24_f64<>+0x6AE0(SB)/8, $1710
DATA bitrev_size8192_mixed24_f64<>+0x6AE8(SB)/8, $3758
DATA bitrev_size8192_mixed24_f64<>+0x6AF0(SB)/8, $5806
DATA bitrev_size8192_mixed24_f64<>+0x6AF8(SB)/8, $7854
DATA bitrev_size8192_mixed24_f64<>+0x6B00(SB)/8, $302
DATA bitrev_size8192_mixed24_f64<>+0x6B08(SB)/8, $2350
DATA bitrev_size8192_mixed24_f64<>+0x6B10(SB)/8, $4398
DATA bitrev_size8192_mixed24_f64<>+0x6B18(SB)/8, $6446
DATA bitrev_size8192_mixed24_f64<>+0x6B20(SB)/8, $814
DATA bitrev_size8192_mixed24_f64<>+0x6B28(SB)/8, $2862
DATA bitrev_size8192_mixed24_f64<>+0x6B30(SB)/8, $4910
DATA bitrev_size8192_mixed24_f64<>+0x6B38(SB)/8, $6958
DATA bitrev_size8192_mixed24_f64<>+0x6B40(SB)/8, $1326
DATA bitrev_size8192_mixed24_f64<>+0x6B48(SB)/8, $3374
DATA bitrev_size8192_mixed24_f64<>+0x6B50(SB)/8, $5422
DATA bitrev_size8192_mixed24_f64<>+0x6B58(SB)/8, $7470
DATA bitrev_size8192_mixed24_f64<>+0x6B60(SB)/8, $1838
DATA bitrev_size8192_mixed24_f64<>+0x6B68(SB)/8, $3886
DATA bitrev_size8192_mixed24_f64<>+0x6B70(SB)/8, $5934
DATA bitrev_size8192_mixed24_f64<>+0x6B78(SB)/8, $7982
DATA bitrev_size8192_mixed24_f64<>+0x6B80(SB)/8, $430
DATA bitrev_size8192_mixed24_f64<>+0x6B88(SB)/8, $2478
DATA bitrev_size8192_mixed24_f64<>+0x6B90(SB)/8, $4526
DATA bitrev_size8192_mixed24_f64<>+0x6B98(SB)/8, $6574
DATA bitrev_size8192_mixed24_f64<>+0x6BA0(SB)/8, $942
DATA bitrev_size8192_mixed24_f64<>+0x6BA8(SB)/8, $2990
DATA bitrev_size8192_mixed24_f64<>+0x6BB0(SB)/8, $5038
DATA bitrev_size8192_mixed24_f64<>+0x6BB8(SB)/8, $7086
DATA bitrev_size8192_mixed24_f64<>+0x6BC0(SB)/8, $1454
DATA bitrev_size8192_mixed24_f64<>+0x6BC8(SB)/8, $3502
DATA bitrev_size8192_mixed24_f64<>+0x6BD0(SB)/8, $5550
DATA bitrev_size8192_mixed24_f64<>+0x6BD8(SB)/8, $7598
DATA bitrev_size8192_mixed24_f64<>+0x6BE0(SB)/8, $1966
DATA bitrev_size8192_mixed24_f64<>+0x6BE8(SB)/8, $4014
DATA bitrev_size8192_mixed24_f64<>+0x6BF0(SB)/8, $6062
DATA bitrev_size8192_mixed24_f64<>+0x6BF8(SB)/8, $8110
DATA bitrev_size8192_mixed24_f64<>+0x6C00(SB)/8, $78
DATA bitrev_size8192_mixed24_f64<>+0x6C08(SB)/8, $2126
DATA bitrev_size8192_mixed24_f64<>+0x6C10(SB)/8, $4174
DATA bitrev_size8192_mixed24_f64<>+0x6C18(SB)/8, $6222
DATA bitrev_size8192_mixed24_f64<>+0x6C20(SB)/8, $590
DATA bitrev_size8192_mixed24_f64<>+0x6C28(SB)/8, $2638
DATA bitrev_size8192_mixed24_f64<>+0x6C30(SB)/8, $4686
DATA bitrev_size8192_mixed24_f64<>+0x6C38(SB)/8, $6734
DATA bitrev_size8192_mixed24_f64<>+0x6C40(SB)/8, $1102
DATA bitrev_size8192_mixed24_f64<>+0x6C48(SB)/8, $3150
DATA bitrev_size8192_mixed24_f64<>+0x6C50(SB)/8, $5198
DATA bitrev_size8192_mixed24_f64<>+0x6C58(SB)/8, $7246
DATA bitrev_size8192_mixed24_f64<>+0x6C60(SB)/8, $1614
DATA bitrev_size8192_mixed24_f64<>+0x6C68(SB)/8, $3662
DATA bitrev_size8192_mixed24_f64<>+0x6C70(SB)/8, $5710
DATA bitrev_size8192_mixed24_f64<>+0x6C78(SB)/8, $7758
DATA bitrev_size8192_mixed24_f64<>+0x6C80(SB)/8, $206
DATA bitrev_size8192_mixed24_f64<>+0x6C88(SB)/8, $2254
DATA bitrev_size8192_mixed24_f64<>+0x6C90(SB)/8, $4302
DATA bitrev_size8192_mixed24_f64<>+0x6C98(SB)/8, $6350
DATA bitrev_size8192_mixed24_f64<>+0x6CA0(SB)/8, $718
DATA bitrev_size8192_mixed24_f64<>+0x6CA8(SB)/8, $2766
DATA bitrev_size8192_mixed24_f64<>+0x6CB0(SB)/8, $4814
DATA bitrev_size8192_mixed24_f64<>+0x6CB8(SB)/8, $6862
DATA bitrev_size8192_mixed24_f64<>+0x6CC0(SB)/8, $1230
DATA bitrev_size8192_mixed24_f64<>+0x6CC8(SB)/8, $3278
DATA bitrev_size8192_mixed24_f64<>+0x6CD0(SB)/8, $5326
DATA bitrev_size8192_mixed24_f64<>+0x6CD8(SB)/8, $7374
DATA bitrev_size8192_mixed24_f64<>+0x6CE0(SB)/8, $1742
DATA bitrev_size8192_mixed24_f64<>+0x6CE8(SB)/8, $3790
DATA bitrev_size8192_mixed24_f64<>+0x6CF0(SB)/8, $5838
DATA bitrev_size8192_mixed24_f64<>+0x6CF8(SB)/8, $7886
DATA bitrev_size8192_mixed24_f64<>+0x6D00(SB)/8, $334
DATA bitrev_size8192_mixed24_f64<>+0x6D08(SB)/8, $2382
DATA bitrev_size8192_mixed24_f64<>+0x6D10(SB)/8, $4430
DATA bitrev_size8192_mixed24_f64<>+0x6D18(SB)/8, $6478
DATA bitrev_size8192_mixed24_f64<>+0x6D20(SB)/8, $846
DATA bitrev_size8192_mixed24_f64<>+0x6D28(SB)/8, $2894
DATA bitrev_size8192_mixed24_f64<>+0x6D30(SB)/8, $4942
DATA bitrev_size8192_mixed24_f64<>+0x6D38(SB)/8, $6990
DATA bitrev_size8192_mixed24_f64<>+0x6D40(SB)/8, $1358
DATA bitrev_size8192_mixed24_f64<>+0x6D48(SB)/8, $3406
DATA bitrev_size8192_mixed24_f64<>+0x6D50(SB)/8, $5454
DATA bitrev_size8192_mixed24_f64<>+0x6D58(SB)/8, $7502
DATA bitrev_size8192_mixed24_f64<>+0x6D60(SB)/8, $1870
DATA bitrev_size8192_mixed24_f64<>+0x6D68(SB)/8, $3918
DATA bitrev_size8192_mixed24_f64<>+0x6D70(SB)/8, $5966
DATA bitrev_size8192_mixed24_f64<>+0x6D78(SB)/8, $8014
DATA bitrev_size8192_mixed24_f64<>+0x6D80(SB)/8, $462
DATA bitrev_size8192_mixed24_f64<>+0x6D88(SB)/8, $2510
DATA bitrev_size8192_mixed24_f64<>+0x6D90(SB)/8, $4558
DATA bitrev_size8192_mixed24_f64<>+0x6D98(SB)/8, $6606
DATA bitrev_size8192_mixed24_f64<>+0x6DA0(SB)/8, $974
DATA bitrev_size8192_mixed24_f64<>+0x6DA8(SB)/8, $3022
DATA bitrev_size8192_mixed24_f64<>+0x6DB0(SB)/8, $5070
DATA bitrev_size8192_mixed24_f64<>+0x6DB8(SB)/8, $7118
DATA bitrev_size8192_mixed24_f64<>+0x6DC0(SB)/8, $1486
DATA bitrev_size8192_mixed24_f64<>+0x6DC8(SB)/8, $3534
DATA bitrev_size8192_mixed24_f64<>+0x6DD0(SB)/8, $5582
DATA bitrev_size8192_mixed24_f64<>+0x6DD8(SB)/8, $7630
DATA bitrev_size8192_mixed24_f64<>+0x6DE0(SB)/8, $1998
DATA bitrev_size8192_mixed24_f64<>+0x6DE8(SB)/8, $4046
DATA bitrev_size8192_mixed24_f64<>+0x6DF0(SB)/8, $6094
DATA bitrev_size8192_mixed24_f64<>+0x6DF8(SB)/8, $8142
DATA bitrev_size8192_mixed24_f64<>+0x6E00(SB)/8, $110
DATA bitrev_size8192_mixed24_f64<>+0x6E08(SB)/8, $2158
DATA bitrev_size8192_mixed24_f64<>+0x6E10(SB)/8, $4206
DATA bitrev_size8192_mixed24_f64<>+0x6E18(SB)/8, $6254
DATA bitrev_size8192_mixed24_f64<>+0x6E20(SB)/8, $622
DATA bitrev_size8192_mixed24_f64<>+0x6E28(SB)/8, $2670
DATA bitrev_size8192_mixed24_f64<>+0x6E30(SB)/8, $4718
DATA bitrev_size8192_mixed24_f64<>+0x6E38(SB)/8, $6766
DATA bitrev_size8192_mixed24_f64<>+0x6E40(SB)/8, $1134
DATA bitrev_size8192_mixed24_f64<>+0x6E48(SB)/8, $3182
DATA bitrev_size8192_mixed24_f64<>+0x6E50(SB)/8, $5230
DATA bitrev_size8192_mixed24_f64<>+0x6E58(SB)/8, $7278
DATA bitrev_size8192_mixed24_f64<>+0x6E60(SB)/8, $1646
DATA bitrev_size8192_mixed24_f64<>+0x6E68(SB)/8, $3694
DATA bitrev_size8192_mixed24_f64<>+0x6E70(SB)/8, $5742
DATA bitrev_size8192_mixed24_f64<>+0x6E78(SB)/8, $7790
DATA bitrev_size8192_mixed24_f64<>+0x6E80(SB)/8, $238
DATA bitrev_size8192_mixed24_f64<>+0x6E88(SB)/8, $2286
DATA bitrev_size8192_mixed24_f64<>+0x6E90(SB)/8, $4334
DATA bitrev_size8192_mixed24_f64<>+0x6E98(SB)/8, $6382
DATA bitrev_size8192_mixed24_f64<>+0x6EA0(SB)/8, $750
DATA bitrev_size8192_mixed24_f64<>+0x6EA8(SB)/8, $2798
DATA bitrev_size8192_mixed24_f64<>+0x6EB0(SB)/8, $4846
DATA bitrev_size8192_mixed24_f64<>+0x6EB8(SB)/8, $6894
DATA bitrev_size8192_mixed24_f64<>+0x6EC0(SB)/8, $1262
DATA bitrev_size8192_mixed24_f64<>+0x6EC8(SB)/8, $3310
DATA bitrev_size8192_mixed24_f64<>+0x6ED0(SB)/8, $5358
DATA bitrev_size8192_mixed24_f64<>+0x6ED8(SB)/8, $7406
DATA bitrev_size8192_mixed24_f64<>+0x6EE0(SB)/8, $1774
DATA bitrev_size8192_mixed24_f64<>+0x6EE8(SB)/8, $3822
DATA bitrev_size8192_mixed24_f64<>+0x6EF0(SB)/8, $5870
DATA bitrev_size8192_mixed24_f64<>+0x6EF8(SB)/8, $7918
DATA bitrev_size8192_mixed24_f64<>+0x6F00(SB)/8, $366
DATA bitrev_size8192_mixed24_f64<>+0x6F08(SB)/8, $2414
DATA bitrev_size8192_mixed24_f64<>+0x6F10(SB)/8, $4462
DATA bitrev_size8192_mixed24_f64<>+0x6F18(SB)/8, $6510
DATA bitrev_size8192_mixed24_f64<>+0x6F20(SB)/8, $878
DATA bitrev_size8192_mixed24_f64<>+0x6F28(SB)/8, $2926
DATA bitrev_size8192_mixed24_f64<>+0x6F30(SB)/8, $4974
DATA bitrev_size8192_mixed24_f64<>+0x6F38(SB)/8, $7022
DATA bitrev_size8192_mixed24_f64<>+0x6F40(SB)/8, $1390
DATA bitrev_size8192_mixed24_f64<>+0x6F48(SB)/8, $3438
DATA bitrev_size8192_mixed24_f64<>+0x6F50(SB)/8, $5486
DATA bitrev_size8192_mixed24_f64<>+0x6F58(SB)/8, $7534
DATA bitrev_size8192_mixed24_f64<>+0x6F60(SB)/8, $1902
DATA bitrev_size8192_mixed24_f64<>+0x6F68(SB)/8, $3950
DATA bitrev_size8192_mixed24_f64<>+0x6F70(SB)/8, $5998
DATA bitrev_size8192_mixed24_f64<>+0x6F78(SB)/8, $8046
DATA bitrev_size8192_mixed24_f64<>+0x6F80(SB)/8, $494
DATA bitrev_size8192_mixed24_f64<>+0x6F88(SB)/8, $2542
DATA bitrev_size8192_mixed24_f64<>+0x6F90(SB)/8, $4590
DATA bitrev_size8192_mixed24_f64<>+0x6F98(SB)/8, $6638
DATA bitrev_size8192_mixed24_f64<>+0x6FA0(SB)/8, $1006
DATA bitrev_size8192_mixed24_f64<>+0x6FA8(SB)/8, $3054
DATA bitrev_size8192_mixed24_f64<>+0x6FB0(SB)/8, $5102
DATA bitrev_size8192_mixed24_f64<>+0x6FB8(SB)/8, $7150
DATA bitrev_size8192_mixed24_f64<>+0x6FC0(SB)/8, $1518
DATA bitrev_size8192_mixed24_f64<>+0x6FC8(SB)/8, $3566
DATA bitrev_size8192_mixed24_f64<>+0x6FD0(SB)/8, $5614
DATA bitrev_size8192_mixed24_f64<>+0x6FD8(SB)/8, $7662
DATA bitrev_size8192_mixed24_f64<>+0x6FE0(SB)/8, $2030
DATA bitrev_size8192_mixed24_f64<>+0x6FE8(SB)/8, $4078
DATA bitrev_size8192_mixed24_f64<>+0x6FF0(SB)/8, $6126
DATA bitrev_size8192_mixed24_f64<>+0x6FF8(SB)/8, $8174
DATA bitrev_size8192_mixed24_f64<>+0x7000(SB)/8, $22
DATA bitrev_size8192_mixed24_f64<>+0x7008(SB)/8, $2070
DATA bitrev_size8192_mixed24_f64<>+0x7010(SB)/8, $4118
DATA bitrev_size8192_mixed24_f64<>+0x7018(SB)/8, $6166
DATA bitrev_size8192_mixed24_f64<>+0x7020(SB)/8, $534
DATA bitrev_size8192_mixed24_f64<>+0x7028(SB)/8, $2582
DATA bitrev_size8192_mixed24_f64<>+0x7030(SB)/8, $4630
DATA bitrev_size8192_mixed24_f64<>+0x7038(SB)/8, $6678
DATA bitrev_size8192_mixed24_f64<>+0x7040(SB)/8, $1046
DATA bitrev_size8192_mixed24_f64<>+0x7048(SB)/8, $3094
DATA bitrev_size8192_mixed24_f64<>+0x7050(SB)/8, $5142
DATA bitrev_size8192_mixed24_f64<>+0x7058(SB)/8, $7190
DATA bitrev_size8192_mixed24_f64<>+0x7060(SB)/8, $1558
DATA bitrev_size8192_mixed24_f64<>+0x7068(SB)/8, $3606
DATA bitrev_size8192_mixed24_f64<>+0x7070(SB)/8, $5654
DATA bitrev_size8192_mixed24_f64<>+0x7078(SB)/8, $7702
DATA bitrev_size8192_mixed24_f64<>+0x7080(SB)/8, $150
DATA bitrev_size8192_mixed24_f64<>+0x7088(SB)/8, $2198
DATA bitrev_size8192_mixed24_f64<>+0x7090(SB)/8, $4246
DATA bitrev_size8192_mixed24_f64<>+0x7098(SB)/8, $6294
DATA bitrev_size8192_mixed24_f64<>+0x70A0(SB)/8, $662
DATA bitrev_size8192_mixed24_f64<>+0x70A8(SB)/8, $2710
DATA bitrev_size8192_mixed24_f64<>+0x70B0(SB)/8, $4758
DATA bitrev_size8192_mixed24_f64<>+0x70B8(SB)/8, $6806
DATA bitrev_size8192_mixed24_f64<>+0x70C0(SB)/8, $1174
DATA bitrev_size8192_mixed24_f64<>+0x70C8(SB)/8, $3222
DATA bitrev_size8192_mixed24_f64<>+0x70D0(SB)/8, $5270
DATA bitrev_size8192_mixed24_f64<>+0x70D8(SB)/8, $7318
DATA bitrev_size8192_mixed24_f64<>+0x70E0(SB)/8, $1686
DATA bitrev_size8192_mixed24_f64<>+0x70E8(SB)/8, $3734
DATA bitrev_size8192_mixed24_f64<>+0x70F0(SB)/8, $5782
DATA bitrev_size8192_mixed24_f64<>+0x70F8(SB)/8, $7830
DATA bitrev_size8192_mixed24_f64<>+0x7100(SB)/8, $278
DATA bitrev_size8192_mixed24_f64<>+0x7108(SB)/8, $2326
DATA bitrev_size8192_mixed24_f64<>+0x7110(SB)/8, $4374
DATA bitrev_size8192_mixed24_f64<>+0x7118(SB)/8, $6422
DATA bitrev_size8192_mixed24_f64<>+0x7120(SB)/8, $790
DATA bitrev_size8192_mixed24_f64<>+0x7128(SB)/8, $2838
DATA bitrev_size8192_mixed24_f64<>+0x7130(SB)/8, $4886
DATA bitrev_size8192_mixed24_f64<>+0x7138(SB)/8, $6934
DATA bitrev_size8192_mixed24_f64<>+0x7140(SB)/8, $1302
DATA bitrev_size8192_mixed24_f64<>+0x7148(SB)/8, $3350
DATA bitrev_size8192_mixed24_f64<>+0x7150(SB)/8, $5398
DATA bitrev_size8192_mixed24_f64<>+0x7158(SB)/8, $7446
DATA bitrev_size8192_mixed24_f64<>+0x7160(SB)/8, $1814
DATA bitrev_size8192_mixed24_f64<>+0x7168(SB)/8, $3862
DATA bitrev_size8192_mixed24_f64<>+0x7170(SB)/8, $5910
DATA bitrev_size8192_mixed24_f64<>+0x7178(SB)/8, $7958
DATA bitrev_size8192_mixed24_f64<>+0x7180(SB)/8, $406
DATA bitrev_size8192_mixed24_f64<>+0x7188(SB)/8, $2454
DATA bitrev_size8192_mixed24_f64<>+0x7190(SB)/8, $4502
DATA bitrev_size8192_mixed24_f64<>+0x7198(SB)/8, $6550
DATA bitrev_size8192_mixed24_f64<>+0x71A0(SB)/8, $918
DATA bitrev_size8192_mixed24_f64<>+0x71A8(SB)/8, $2966
DATA bitrev_size8192_mixed24_f64<>+0x71B0(SB)/8, $5014
DATA bitrev_size8192_mixed24_f64<>+0x71B8(SB)/8, $7062
DATA bitrev_size8192_mixed24_f64<>+0x71C0(SB)/8, $1430
DATA bitrev_size8192_mixed24_f64<>+0x71C8(SB)/8, $3478
DATA bitrev_size8192_mixed24_f64<>+0x71D0(SB)/8, $5526
DATA bitrev_size8192_mixed24_f64<>+0x71D8(SB)/8, $7574
DATA bitrev_size8192_mixed24_f64<>+0x71E0(SB)/8, $1942
DATA bitrev_size8192_mixed24_f64<>+0x71E8(SB)/8, $3990
DATA bitrev_size8192_mixed24_f64<>+0x71F0(SB)/8, $6038
DATA bitrev_size8192_mixed24_f64<>+0x71F8(SB)/8, $8086
DATA bitrev_size8192_mixed24_f64<>+0x7200(SB)/8, $54
DATA bitrev_size8192_mixed24_f64<>+0x7208(SB)/8, $2102
DATA bitrev_size8192_mixed24_f64<>+0x7210(SB)/8, $4150
DATA bitrev_size8192_mixed24_f64<>+0x7218(SB)/8, $6198
DATA bitrev_size8192_mixed24_f64<>+0x7220(SB)/8, $566
DATA bitrev_size8192_mixed24_f64<>+0x7228(SB)/8, $2614
DATA bitrev_size8192_mixed24_f64<>+0x7230(SB)/8, $4662
DATA bitrev_size8192_mixed24_f64<>+0x7238(SB)/8, $6710
DATA bitrev_size8192_mixed24_f64<>+0x7240(SB)/8, $1078
DATA bitrev_size8192_mixed24_f64<>+0x7248(SB)/8, $3126
DATA bitrev_size8192_mixed24_f64<>+0x7250(SB)/8, $5174
DATA bitrev_size8192_mixed24_f64<>+0x7258(SB)/8, $7222
DATA bitrev_size8192_mixed24_f64<>+0x7260(SB)/8, $1590
DATA bitrev_size8192_mixed24_f64<>+0x7268(SB)/8, $3638
DATA bitrev_size8192_mixed24_f64<>+0x7270(SB)/8, $5686
DATA bitrev_size8192_mixed24_f64<>+0x7278(SB)/8, $7734
DATA bitrev_size8192_mixed24_f64<>+0x7280(SB)/8, $182
DATA bitrev_size8192_mixed24_f64<>+0x7288(SB)/8, $2230
DATA bitrev_size8192_mixed24_f64<>+0x7290(SB)/8, $4278
DATA bitrev_size8192_mixed24_f64<>+0x7298(SB)/8, $6326
DATA bitrev_size8192_mixed24_f64<>+0x72A0(SB)/8, $694
DATA bitrev_size8192_mixed24_f64<>+0x72A8(SB)/8, $2742
DATA bitrev_size8192_mixed24_f64<>+0x72B0(SB)/8, $4790
DATA bitrev_size8192_mixed24_f64<>+0x72B8(SB)/8, $6838
DATA bitrev_size8192_mixed24_f64<>+0x72C0(SB)/8, $1206
DATA bitrev_size8192_mixed24_f64<>+0x72C8(SB)/8, $3254
DATA bitrev_size8192_mixed24_f64<>+0x72D0(SB)/8, $5302
DATA bitrev_size8192_mixed24_f64<>+0x72D8(SB)/8, $7350
DATA bitrev_size8192_mixed24_f64<>+0x72E0(SB)/8, $1718
DATA bitrev_size8192_mixed24_f64<>+0x72E8(SB)/8, $3766
DATA bitrev_size8192_mixed24_f64<>+0x72F0(SB)/8, $5814
DATA bitrev_size8192_mixed24_f64<>+0x72F8(SB)/8, $7862
DATA bitrev_size8192_mixed24_f64<>+0x7300(SB)/8, $310
DATA bitrev_size8192_mixed24_f64<>+0x7308(SB)/8, $2358
DATA bitrev_size8192_mixed24_f64<>+0x7310(SB)/8, $4406
DATA bitrev_size8192_mixed24_f64<>+0x7318(SB)/8, $6454
DATA bitrev_size8192_mixed24_f64<>+0x7320(SB)/8, $822
DATA bitrev_size8192_mixed24_f64<>+0x7328(SB)/8, $2870
DATA bitrev_size8192_mixed24_f64<>+0x7330(SB)/8, $4918
DATA bitrev_size8192_mixed24_f64<>+0x7338(SB)/8, $6966
DATA bitrev_size8192_mixed24_f64<>+0x7340(SB)/8, $1334
DATA bitrev_size8192_mixed24_f64<>+0x7348(SB)/8, $3382
DATA bitrev_size8192_mixed24_f64<>+0x7350(SB)/8, $5430
DATA bitrev_size8192_mixed24_f64<>+0x7358(SB)/8, $7478
DATA bitrev_size8192_mixed24_f64<>+0x7360(SB)/8, $1846
DATA bitrev_size8192_mixed24_f64<>+0x7368(SB)/8, $3894
DATA bitrev_size8192_mixed24_f64<>+0x7370(SB)/8, $5942
DATA bitrev_size8192_mixed24_f64<>+0x7378(SB)/8, $7990
DATA bitrev_size8192_mixed24_f64<>+0x7380(SB)/8, $438
DATA bitrev_size8192_mixed24_f64<>+0x7388(SB)/8, $2486
DATA bitrev_size8192_mixed24_f64<>+0x7390(SB)/8, $4534
DATA bitrev_size8192_mixed24_f64<>+0x7398(SB)/8, $6582
DATA bitrev_size8192_mixed24_f64<>+0x73A0(SB)/8, $950
DATA bitrev_size8192_mixed24_f64<>+0x73A8(SB)/8, $2998
DATA bitrev_size8192_mixed24_f64<>+0x73B0(SB)/8, $5046
DATA bitrev_size8192_mixed24_f64<>+0x73B8(SB)/8, $7094
DATA bitrev_size8192_mixed24_f64<>+0x73C0(SB)/8, $1462
DATA bitrev_size8192_mixed24_f64<>+0x73C8(SB)/8, $3510
DATA bitrev_size8192_mixed24_f64<>+0x73D0(SB)/8, $5558
DATA bitrev_size8192_mixed24_f64<>+0x73D8(SB)/8, $7606
DATA bitrev_size8192_mixed24_f64<>+0x73E0(SB)/8, $1974
DATA bitrev_size8192_mixed24_f64<>+0x73E8(SB)/8, $4022
DATA bitrev_size8192_mixed24_f64<>+0x73F0(SB)/8, $6070
DATA bitrev_size8192_mixed24_f64<>+0x73F8(SB)/8, $8118
DATA bitrev_size8192_mixed24_f64<>+0x7400(SB)/8, $86
DATA bitrev_size8192_mixed24_f64<>+0x7408(SB)/8, $2134
DATA bitrev_size8192_mixed24_f64<>+0x7410(SB)/8, $4182
DATA bitrev_size8192_mixed24_f64<>+0x7418(SB)/8, $6230
DATA bitrev_size8192_mixed24_f64<>+0x7420(SB)/8, $598
DATA bitrev_size8192_mixed24_f64<>+0x7428(SB)/8, $2646
DATA bitrev_size8192_mixed24_f64<>+0x7430(SB)/8, $4694
DATA bitrev_size8192_mixed24_f64<>+0x7438(SB)/8, $6742
DATA bitrev_size8192_mixed24_f64<>+0x7440(SB)/8, $1110
DATA bitrev_size8192_mixed24_f64<>+0x7448(SB)/8, $3158
DATA bitrev_size8192_mixed24_f64<>+0x7450(SB)/8, $5206
DATA bitrev_size8192_mixed24_f64<>+0x7458(SB)/8, $7254
DATA bitrev_size8192_mixed24_f64<>+0x7460(SB)/8, $1622
DATA bitrev_size8192_mixed24_f64<>+0x7468(SB)/8, $3670
DATA bitrev_size8192_mixed24_f64<>+0x7470(SB)/8, $5718
DATA bitrev_size8192_mixed24_f64<>+0x7478(SB)/8, $7766
DATA bitrev_size8192_mixed24_f64<>+0x7480(SB)/8, $214
DATA bitrev_size8192_mixed24_f64<>+0x7488(SB)/8, $2262
DATA bitrev_size8192_mixed24_f64<>+0x7490(SB)/8, $4310
DATA bitrev_size8192_mixed24_f64<>+0x7498(SB)/8, $6358
DATA bitrev_size8192_mixed24_f64<>+0x74A0(SB)/8, $726
DATA bitrev_size8192_mixed24_f64<>+0x74A8(SB)/8, $2774
DATA bitrev_size8192_mixed24_f64<>+0x74B0(SB)/8, $4822
DATA bitrev_size8192_mixed24_f64<>+0x74B8(SB)/8, $6870
DATA bitrev_size8192_mixed24_f64<>+0x74C0(SB)/8, $1238
DATA bitrev_size8192_mixed24_f64<>+0x74C8(SB)/8, $3286
DATA bitrev_size8192_mixed24_f64<>+0x74D0(SB)/8, $5334
DATA bitrev_size8192_mixed24_f64<>+0x74D8(SB)/8, $7382
DATA bitrev_size8192_mixed24_f64<>+0x74E0(SB)/8, $1750
DATA bitrev_size8192_mixed24_f64<>+0x74E8(SB)/8, $3798
DATA bitrev_size8192_mixed24_f64<>+0x74F0(SB)/8, $5846
DATA bitrev_size8192_mixed24_f64<>+0x74F8(SB)/8, $7894
DATA bitrev_size8192_mixed24_f64<>+0x7500(SB)/8, $342
DATA bitrev_size8192_mixed24_f64<>+0x7508(SB)/8, $2390
DATA bitrev_size8192_mixed24_f64<>+0x7510(SB)/8, $4438
DATA bitrev_size8192_mixed24_f64<>+0x7518(SB)/8, $6486
DATA bitrev_size8192_mixed24_f64<>+0x7520(SB)/8, $854
DATA bitrev_size8192_mixed24_f64<>+0x7528(SB)/8, $2902
DATA bitrev_size8192_mixed24_f64<>+0x7530(SB)/8, $4950
DATA bitrev_size8192_mixed24_f64<>+0x7538(SB)/8, $6998
DATA bitrev_size8192_mixed24_f64<>+0x7540(SB)/8, $1366
DATA bitrev_size8192_mixed24_f64<>+0x7548(SB)/8, $3414
DATA bitrev_size8192_mixed24_f64<>+0x7550(SB)/8, $5462
DATA bitrev_size8192_mixed24_f64<>+0x7558(SB)/8, $7510
DATA bitrev_size8192_mixed24_f64<>+0x7560(SB)/8, $1878
DATA bitrev_size8192_mixed24_f64<>+0x7568(SB)/8, $3926
DATA bitrev_size8192_mixed24_f64<>+0x7570(SB)/8, $5974
DATA bitrev_size8192_mixed24_f64<>+0x7578(SB)/8, $8022
DATA bitrev_size8192_mixed24_f64<>+0x7580(SB)/8, $470
DATA bitrev_size8192_mixed24_f64<>+0x7588(SB)/8, $2518
DATA bitrev_size8192_mixed24_f64<>+0x7590(SB)/8, $4566
DATA bitrev_size8192_mixed24_f64<>+0x7598(SB)/8, $6614
DATA bitrev_size8192_mixed24_f64<>+0x75A0(SB)/8, $982
DATA bitrev_size8192_mixed24_f64<>+0x75A8(SB)/8, $3030
DATA bitrev_size8192_mixed24_f64<>+0x75B0(SB)/8, $5078
DATA bitrev_size8192_mixed24_f64<>+0x75B8(SB)/8, $7126
DATA bitrev_size8192_mixed24_f64<>+0x75C0(SB)/8, $1494
DATA bitrev_size8192_mixed24_f64<>+0x75C8(SB)/8, $3542
DATA bitrev_size8192_mixed24_f64<>+0x75D0(SB)/8, $5590
DATA bitrev_size8192_mixed24_f64<>+0x75D8(SB)/8, $7638
DATA bitrev_size8192_mixed24_f64<>+0x75E0(SB)/8, $2006
DATA bitrev_size8192_mixed24_f64<>+0x75E8(SB)/8, $4054
DATA bitrev_size8192_mixed24_f64<>+0x75F0(SB)/8, $6102
DATA bitrev_size8192_mixed24_f64<>+0x75F8(SB)/8, $8150
DATA bitrev_size8192_mixed24_f64<>+0x7600(SB)/8, $118
DATA bitrev_size8192_mixed24_f64<>+0x7608(SB)/8, $2166
DATA bitrev_size8192_mixed24_f64<>+0x7610(SB)/8, $4214
DATA bitrev_size8192_mixed24_f64<>+0x7618(SB)/8, $6262
DATA bitrev_size8192_mixed24_f64<>+0x7620(SB)/8, $630
DATA bitrev_size8192_mixed24_f64<>+0x7628(SB)/8, $2678
DATA bitrev_size8192_mixed24_f64<>+0x7630(SB)/8, $4726
DATA bitrev_size8192_mixed24_f64<>+0x7638(SB)/8, $6774
DATA bitrev_size8192_mixed24_f64<>+0x7640(SB)/8, $1142
DATA bitrev_size8192_mixed24_f64<>+0x7648(SB)/8, $3190
DATA bitrev_size8192_mixed24_f64<>+0x7650(SB)/8, $5238
DATA bitrev_size8192_mixed24_f64<>+0x7658(SB)/8, $7286
DATA bitrev_size8192_mixed24_f64<>+0x7660(SB)/8, $1654
DATA bitrev_size8192_mixed24_f64<>+0x7668(SB)/8, $3702
DATA bitrev_size8192_mixed24_f64<>+0x7670(SB)/8, $5750
DATA bitrev_size8192_mixed24_f64<>+0x7678(SB)/8, $7798
DATA bitrev_size8192_mixed24_f64<>+0x7680(SB)/8, $246
DATA bitrev_size8192_mixed24_f64<>+0x7688(SB)/8, $2294
DATA bitrev_size8192_mixed24_f64<>+0x7690(SB)/8, $4342
DATA bitrev_size8192_mixed24_f64<>+0x7698(SB)/8, $6390
DATA bitrev_size8192_mixed24_f64<>+0x76A0(SB)/8, $758
DATA bitrev_size8192_mixed24_f64<>+0x76A8(SB)/8, $2806
DATA bitrev_size8192_mixed24_f64<>+0x76B0(SB)/8, $4854
DATA bitrev_size8192_mixed24_f64<>+0x76B8(SB)/8, $6902
DATA bitrev_size8192_mixed24_f64<>+0x76C0(SB)/8, $1270
DATA bitrev_size8192_mixed24_f64<>+0x76C8(SB)/8, $3318
DATA bitrev_size8192_mixed24_f64<>+0x76D0(SB)/8, $5366
DATA bitrev_size8192_mixed24_f64<>+0x76D8(SB)/8, $7414
DATA bitrev_size8192_mixed24_f64<>+0x76E0(SB)/8, $1782
DATA bitrev_size8192_mixed24_f64<>+0x76E8(SB)/8, $3830
DATA bitrev_size8192_mixed24_f64<>+0x76F0(SB)/8, $5878
DATA bitrev_size8192_mixed24_f64<>+0x76F8(SB)/8, $7926
DATA bitrev_size8192_mixed24_f64<>+0x7700(SB)/8, $374
DATA bitrev_size8192_mixed24_f64<>+0x7708(SB)/8, $2422
DATA bitrev_size8192_mixed24_f64<>+0x7710(SB)/8, $4470
DATA bitrev_size8192_mixed24_f64<>+0x7718(SB)/8, $6518
DATA bitrev_size8192_mixed24_f64<>+0x7720(SB)/8, $886
DATA bitrev_size8192_mixed24_f64<>+0x7728(SB)/8, $2934
DATA bitrev_size8192_mixed24_f64<>+0x7730(SB)/8, $4982
DATA bitrev_size8192_mixed24_f64<>+0x7738(SB)/8, $7030
DATA bitrev_size8192_mixed24_f64<>+0x7740(SB)/8, $1398
DATA bitrev_size8192_mixed24_f64<>+0x7748(SB)/8, $3446
DATA bitrev_size8192_mixed24_f64<>+0x7750(SB)/8, $5494
DATA bitrev_size8192_mixed24_f64<>+0x7758(SB)/8, $7542
DATA bitrev_size8192_mixed24_f64<>+0x7760(SB)/8, $1910
DATA bitrev_size8192_mixed24_f64<>+0x7768(SB)/8, $3958
DATA bitrev_size8192_mixed24_f64<>+0x7770(SB)/8, $6006
DATA bitrev_size8192_mixed24_f64<>+0x7778(SB)/8, $8054
DATA bitrev_size8192_mixed24_f64<>+0x7780(SB)/8, $502
DATA bitrev_size8192_mixed24_f64<>+0x7788(SB)/8, $2550
DATA bitrev_size8192_mixed24_f64<>+0x7790(SB)/8, $4598
DATA bitrev_size8192_mixed24_f64<>+0x7798(SB)/8, $6646
DATA bitrev_size8192_mixed24_f64<>+0x77A0(SB)/8, $1014
DATA bitrev_size8192_mixed24_f64<>+0x77A8(SB)/8, $3062
DATA bitrev_size8192_mixed24_f64<>+0x77B0(SB)/8, $5110
DATA bitrev_size8192_mixed24_f64<>+0x77B8(SB)/8, $7158
DATA bitrev_size8192_mixed24_f64<>+0x77C0(SB)/8, $1526
DATA bitrev_size8192_mixed24_f64<>+0x77C8(SB)/8, $3574
DATA bitrev_size8192_mixed24_f64<>+0x77D0(SB)/8, $5622
DATA bitrev_size8192_mixed24_f64<>+0x77D8(SB)/8, $7670
DATA bitrev_size8192_mixed24_f64<>+0x77E0(SB)/8, $2038
DATA bitrev_size8192_mixed24_f64<>+0x77E8(SB)/8, $4086
DATA bitrev_size8192_mixed24_f64<>+0x77F0(SB)/8, $6134
DATA bitrev_size8192_mixed24_f64<>+0x77F8(SB)/8, $8182
DATA bitrev_size8192_mixed24_f64<>+0x7800(SB)/8, $30
DATA bitrev_size8192_mixed24_f64<>+0x7808(SB)/8, $2078
DATA bitrev_size8192_mixed24_f64<>+0x7810(SB)/8, $4126
DATA bitrev_size8192_mixed24_f64<>+0x7818(SB)/8, $6174
DATA bitrev_size8192_mixed24_f64<>+0x7820(SB)/8, $542
DATA bitrev_size8192_mixed24_f64<>+0x7828(SB)/8, $2590
DATA bitrev_size8192_mixed24_f64<>+0x7830(SB)/8, $4638
DATA bitrev_size8192_mixed24_f64<>+0x7838(SB)/8, $6686
DATA bitrev_size8192_mixed24_f64<>+0x7840(SB)/8, $1054
DATA bitrev_size8192_mixed24_f64<>+0x7848(SB)/8, $3102
DATA bitrev_size8192_mixed24_f64<>+0x7850(SB)/8, $5150
DATA bitrev_size8192_mixed24_f64<>+0x7858(SB)/8, $7198
DATA bitrev_size8192_mixed24_f64<>+0x7860(SB)/8, $1566
DATA bitrev_size8192_mixed24_f64<>+0x7868(SB)/8, $3614
DATA bitrev_size8192_mixed24_f64<>+0x7870(SB)/8, $5662
DATA bitrev_size8192_mixed24_f64<>+0x7878(SB)/8, $7710
DATA bitrev_size8192_mixed24_f64<>+0x7880(SB)/8, $158
DATA bitrev_size8192_mixed24_f64<>+0x7888(SB)/8, $2206
DATA bitrev_size8192_mixed24_f64<>+0x7890(SB)/8, $4254
DATA bitrev_size8192_mixed24_f64<>+0x7898(SB)/8, $6302
DATA bitrev_size8192_mixed24_f64<>+0x78A0(SB)/8, $670
DATA bitrev_size8192_mixed24_f64<>+0x78A8(SB)/8, $2718
DATA bitrev_size8192_mixed24_f64<>+0x78B0(SB)/8, $4766
DATA bitrev_size8192_mixed24_f64<>+0x78B8(SB)/8, $6814
DATA bitrev_size8192_mixed24_f64<>+0x78C0(SB)/8, $1182
DATA bitrev_size8192_mixed24_f64<>+0x78C8(SB)/8, $3230
DATA bitrev_size8192_mixed24_f64<>+0x78D0(SB)/8, $5278
DATA bitrev_size8192_mixed24_f64<>+0x78D8(SB)/8, $7326
DATA bitrev_size8192_mixed24_f64<>+0x78E0(SB)/8, $1694
DATA bitrev_size8192_mixed24_f64<>+0x78E8(SB)/8, $3742
DATA bitrev_size8192_mixed24_f64<>+0x78F0(SB)/8, $5790
DATA bitrev_size8192_mixed24_f64<>+0x78F8(SB)/8, $7838
DATA bitrev_size8192_mixed24_f64<>+0x7900(SB)/8, $286
DATA bitrev_size8192_mixed24_f64<>+0x7908(SB)/8, $2334
DATA bitrev_size8192_mixed24_f64<>+0x7910(SB)/8, $4382
DATA bitrev_size8192_mixed24_f64<>+0x7918(SB)/8, $6430
DATA bitrev_size8192_mixed24_f64<>+0x7920(SB)/8, $798
DATA bitrev_size8192_mixed24_f64<>+0x7928(SB)/8, $2846
DATA bitrev_size8192_mixed24_f64<>+0x7930(SB)/8, $4894
DATA bitrev_size8192_mixed24_f64<>+0x7938(SB)/8, $6942
DATA bitrev_size8192_mixed24_f64<>+0x7940(SB)/8, $1310
DATA bitrev_size8192_mixed24_f64<>+0x7948(SB)/8, $3358
DATA bitrev_size8192_mixed24_f64<>+0x7950(SB)/8, $5406
DATA bitrev_size8192_mixed24_f64<>+0x7958(SB)/8, $7454
DATA bitrev_size8192_mixed24_f64<>+0x7960(SB)/8, $1822
DATA bitrev_size8192_mixed24_f64<>+0x7968(SB)/8, $3870
DATA bitrev_size8192_mixed24_f64<>+0x7970(SB)/8, $5918
DATA bitrev_size8192_mixed24_f64<>+0x7978(SB)/8, $7966
DATA bitrev_size8192_mixed24_f64<>+0x7980(SB)/8, $414
DATA bitrev_size8192_mixed24_f64<>+0x7988(SB)/8, $2462
DATA bitrev_size8192_mixed24_f64<>+0x7990(SB)/8, $4510
DATA bitrev_size8192_mixed24_f64<>+0x7998(SB)/8, $6558
DATA bitrev_size8192_mixed24_f64<>+0x79A0(SB)/8, $926
DATA bitrev_size8192_mixed24_f64<>+0x79A8(SB)/8, $2974
DATA bitrev_size8192_mixed24_f64<>+0x79B0(SB)/8, $5022
DATA bitrev_size8192_mixed24_f64<>+0x79B8(SB)/8, $7070
DATA bitrev_size8192_mixed24_f64<>+0x79C0(SB)/8, $1438
DATA bitrev_size8192_mixed24_f64<>+0x79C8(SB)/8, $3486
DATA bitrev_size8192_mixed24_f64<>+0x79D0(SB)/8, $5534
DATA bitrev_size8192_mixed24_f64<>+0x79D8(SB)/8, $7582
DATA bitrev_size8192_mixed24_f64<>+0x79E0(SB)/8, $1950
DATA bitrev_size8192_mixed24_f64<>+0x79E8(SB)/8, $3998
DATA bitrev_size8192_mixed24_f64<>+0x79F0(SB)/8, $6046
DATA bitrev_size8192_mixed24_f64<>+0x79F8(SB)/8, $8094
DATA bitrev_size8192_mixed24_f64<>+0x7A00(SB)/8, $62
DATA bitrev_size8192_mixed24_f64<>+0x7A08(SB)/8, $2110
DATA bitrev_size8192_mixed24_f64<>+0x7A10(SB)/8, $4158
DATA bitrev_size8192_mixed24_f64<>+0x7A18(SB)/8, $6206
DATA bitrev_size8192_mixed24_f64<>+0x7A20(SB)/8, $574
DATA bitrev_size8192_mixed24_f64<>+0x7A28(SB)/8, $2622
DATA bitrev_size8192_mixed24_f64<>+0x7A30(SB)/8, $4670
DATA bitrev_size8192_mixed24_f64<>+0x7A38(SB)/8, $6718
DATA bitrev_size8192_mixed24_f64<>+0x7A40(SB)/8, $1086
DATA bitrev_size8192_mixed24_f64<>+0x7A48(SB)/8, $3134
DATA bitrev_size8192_mixed24_f64<>+0x7A50(SB)/8, $5182
DATA bitrev_size8192_mixed24_f64<>+0x7A58(SB)/8, $7230
DATA bitrev_size8192_mixed24_f64<>+0x7A60(SB)/8, $1598
DATA bitrev_size8192_mixed24_f64<>+0x7A68(SB)/8, $3646
DATA bitrev_size8192_mixed24_f64<>+0x7A70(SB)/8, $5694
DATA bitrev_size8192_mixed24_f64<>+0x7A78(SB)/8, $7742
DATA bitrev_size8192_mixed24_f64<>+0x7A80(SB)/8, $190
DATA bitrev_size8192_mixed24_f64<>+0x7A88(SB)/8, $2238
DATA bitrev_size8192_mixed24_f64<>+0x7A90(SB)/8, $4286
DATA bitrev_size8192_mixed24_f64<>+0x7A98(SB)/8, $6334
DATA bitrev_size8192_mixed24_f64<>+0x7AA0(SB)/8, $702
DATA bitrev_size8192_mixed24_f64<>+0x7AA8(SB)/8, $2750
DATA bitrev_size8192_mixed24_f64<>+0x7AB0(SB)/8, $4798
DATA bitrev_size8192_mixed24_f64<>+0x7AB8(SB)/8, $6846
DATA bitrev_size8192_mixed24_f64<>+0x7AC0(SB)/8, $1214
DATA bitrev_size8192_mixed24_f64<>+0x7AC8(SB)/8, $3262
DATA bitrev_size8192_mixed24_f64<>+0x7AD0(SB)/8, $5310
DATA bitrev_size8192_mixed24_f64<>+0x7AD8(SB)/8, $7358
DATA bitrev_size8192_mixed24_f64<>+0x7AE0(SB)/8, $1726
DATA bitrev_size8192_mixed24_f64<>+0x7AE8(SB)/8, $3774
DATA bitrev_size8192_mixed24_f64<>+0x7AF0(SB)/8, $5822
DATA bitrev_size8192_mixed24_f64<>+0x7AF8(SB)/8, $7870
DATA bitrev_size8192_mixed24_f64<>+0x7B00(SB)/8, $318
DATA bitrev_size8192_mixed24_f64<>+0x7B08(SB)/8, $2366
DATA bitrev_size8192_mixed24_f64<>+0x7B10(SB)/8, $4414
DATA bitrev_size8192_mixed24_f64<>+0x7B18(SB)/8, $6462
DATA bitrev_size8192_mixed24_f64<>+0x7B20(SB)/8, $830
DATA bitrev_size8192_mixed24_f64<>+0x7B28(SB)/8, $2878
DATA bitrev_size8192_mixed24_f64<>+0x7B30(SB)/8, $4926
DATA bitrev_size8192_mixed24_f64<>+0x7B38(SB)/8, $6974
DATA bitrev_size8192_mixed24_f64<>+0x7B40(SB)/8, $1342
DATA bitrev_size8192_mixed24_f64<>+0x7B48(SB)/8, $3390
DATA bitrev_size8192_mixed24_f64<>+0x7B50(SB)/8, $5438
DATA bitrev_size8192_mixed24_f64<>+0x7B58(SB)/8, $7486
DATA bitrev_size8192_mixed24_f64<>+0x7B60(SB)/8, $1854
DATA bitrev_size8192_mixed24_f64<>+0x7B68(SB)/8, $3902
DATA bitrev_size8192_mixed24_f64<>+0x7B70(SB)/8, $5950
DATA bitrev_size8192_mixed24_f64<>+0x7B78(SB)/8, $7998
DATA bitrev_size8192_mixed24_f64<>+0x7B80(SB)/8, $446
DATA bitrev_size8192_mixed24_f64<>+0x7B88(SB)/8, $2494
DATA bitrev_size8192_mixed24_f64<>+0x7B90(SB)/8, $4542
DATA bitrev_size8192_mixed24_f64<>+0x7B98(SB)/8, $6590
DATA bitrev_size8192_mixed24_f64<>+0x7BA0(SB)/8, $958
DATA bitrev_size8192_mixed24_f64<>+0x7BA8(SB)/8, $3006
DATA bitrev_size8192_mixed24_f64<>+0x7BB0(SB)/8, $5054
DATA bitrev_size8192_mixed24_f64<>+0x7BB8(SB)/8, $7102
DATA bitrev_size8192_mixed24_f64<>+0x7BC0(SB)/8, $1470
DATA bitrev_size8192_mixed24_f64<>+0x7BC8(SB)/8, $3518
DATA bitrev_size8192_mixed24_f64<>+0x7BD0(SB)/8, $5566
DATA bitrev_size8192_mixed24_f64<>+0x7BD8(SB)/8, $7614
DATA bitrev_size8192_mixed24_f64<>+0x7BE0(SB)/8, $1982
DATA bitrev_size8192_mixed24_f64<>+0x7BE8(SB)/8, $4030
DATA bitrev_size8192_mixed24_f64<>+0x7BF0(SB)/8, $6078
DATA bitrev_size8192_mixed24_f64<>+0x7BF8(SB)/8, $8126
DATA bitrev_size8192_mixed24_f64<>+0x7C00(SB)/8, $94
DATA bitrev_size8192_mixed24_f64<>+0x7C08(SB)/8, $2142
DATA bitrev_size8192_mixed24_f64<>+0x7C10(SB)/8, $4190
DATA bitrev_size8192_mixed24_f64<>+0x7C18(SB)/8, $6238
DATA bitrev_size8192_mixed24_f64<>+0x7C20(SB)/8, $606
DATA bitrev_size8192_mixed24_f64<>+0x7C28(SB)/8, $2654
DATA bitrev_size8192_mixed24_f64<>+0x7C30(SB)/8, $4702
DATA bitrev_size8192_mixed24_f64<>+0x7C38(SB)/8, $6750
DATA bitrev_size8192_mixed24_f64<>+0x7C40(SB)/8, $1118
DATA bitrev_size8192_mixed24_f64<>+0x7C48(SB)/8, $3166
DATA bitrev_size8192_mixed24_f64<>+0x7C50(SB)/8, $5214
DATA bitrev_size8192_mixed24_f64<>+0x7C58(SB)/8, $7262
DATA bitrev_size8192_mixed24_f64<>+0x7C60(SB)/8, $1630
DATA bitrev_size8192_mixed24_f64<>+0x7C68(SB)/8, $3678
DATA bitrev_size8192_mixed24_f64<>+0x7C70(SB)/8, $5726
DATA bitrev_size8192_mixed24_f64<>+0x7C78(SB)/8, $7774
DATA bitrev_size8192_mixed24_f64<>+0x7C80(SB)/8, $222
DATA bitrev_size8192_mixed24_f64<>+0x7C88(SB)/8, $2270
DATA bitrev_size8192_mixed24_f64<>+0x7C90(SB)/8, $4318
DATA bitrev_size8192_mixed24_f64<>+0x7C98(SB)/8, $6366
DATA bitrev_size8192_mixed24_f64<>+0x7CA0(SB)/8, $734
DATA bitrev_size8192_mixed24_f64<>+0x7CA8(SB)/8, $2782
DATA bitrev_size8192_mixed24_f64<>+0x7CB0(SB)/8, $4830
DATA bitrev_size8192_mixed24_f64<>+0x7CB8(SB)/8, $6878
DATA bitrev_size8192_mixed24_f64<>+0x7CC0(SB)/8, $1246
DATA bitrev_size8192_mixed24_f64<>+0x7CC8(SB)/8, $3294
DATA bitrev_size8192_mixed24_f64<>+0x7CD0(SB)/8, $5342
DATA bitrev_size8192_mixed24_f64<>+0x7CD8(SB)/8, $7390
DATA bitrev_size8192_mixed24_f64<>+0x7CE0(SB)/8, $1758
DATA bitrev_size8192_mixed24_f64<>+0x7CE8(SB)/8, $3806
DATA bitrev_size8192_mixed24_f64<>+0x7CF0(SB)/8, $5854
DATA bitrev_size8192_mixed24_f64<>+0x7CF8(SB)/8, $7902
DATA bitrev_size8192_mixed24_f64<>+0x7D00(SB)/8, $350
DATA bitrev_size8192_mixed24_f64<>+0x7D08(SB)/8, $2398
DATA bitrev_size8192_mixed24_f64<>+0x7D10(SB)/8, $4446
DATA bitrev_size8192_mixed24_f64<>+0x7D18(SB)/8, $6494
DATA bitrev_size8192_mixed24_f64<>+0x7D20(SB)/8, $862
DATA bitrev_size8192_mixed24_f64<>+0x7D28(SB)/8, $2910
DATA bitrev_size8192_mixed24_f64<>+0x7D30(SB)/8, $4958
DATA bitrev_size8192_mixed24_f64<>+0x7D38(SB)/8, $7006
DATA bitrev_size8192_mixed24_f64<>+0x7D40(SB)/8, $1374
DATA bitrev_size8192_mixed24_f64<>+0x7D48(SB)/8, $3422
DATA bitrev_size8192_mixed24_f64<>+0x7D50(SB)/8, $5470
DATA bitrev_size8192_mixed24_f64<>+0x7D58(SB)/8, $7518
DATA bitrev_size8192_mixed24_f64<>+0x7D60(SB)/8, $1886
DATA bitrev_size8192_mixed24_f64<>+0x7D68(SB)/8, $3934
DATA bitrev_size8192_mixed24_f64<>+0x7D70(SB)/8, $5982
DATA bitrev_size8192_mixed24_f64<>+0x7D78(SB)/8, $8030
DATA bitrev_size8192_mixed24_f64<>+0x7D80(SB)/8, $478
DATA bitrev_size8192_mixed24_f64<>+0x7D88(SB)/8, $2526
DATA bitrev_size8192_mixed24_f64<>+0x7D90(SB)/8, $4574
DATA bitrev_size8192_mixed24_f64<>+0x7D98(SB)/8, $6622
DATA bitrev_size8192_mixed24_f64<>+0x7DA0(SB)/8, $990
DATA bitrev_size8192_mixed24_f64<>+0x7DA8(SB)/8, $3038
DATA bitrev_size8192_mixed24_f64<>+0x7DB0(SB)/8, $5086
DATA bitrev_size8192_mixed24_f64<>+0x7DB8(SB)/8, $7134
DATA bitrev_size8192_mixed24_f64<>+0x7DC0(SB)/8, $1502
DATA bitrev_size8192_mixed24_f64<>+0x7DC8(SB)/8, $3550
DATA bitrev_size8192_mixed24_f64<>+0x7DD0(SB)/8, $5598
DATA bitrev_size8192_mixed24_f64<>+0x7DD8(SB)/8, $7646
DATA bitrev_size8192_mixed24_f64<>+0x7DE0(SB)/8, $2014
DATA bitrev_size8192_mixed24_f64<>+0x7DE8(SB)/8, $4062
DATA bitrev_size8192_mixed24_f64<>+0x7DF0(SB)/8, $6110
DATA bitrev_size8192_mixed24_f64<>+0x7DF8(SB)/8, $8158
DATA bitrev_size8192_mixed24_f64<>+0x7E00(SB)/8, $126
DATA bitrev_size8192_mixed24_f64<>+0x7E08(SB)/8, $2174
DATA bitrev_size8192_mixed24_f64<>+0x7E10(SB)/8, $4222
DATA bitrev_size8192_mixed24_f64<>+0x7E18(SB)/8, $6270
DATA bitrev_size8192_mixed24_f64<>+0x7E20(SB)/8, $638
DATA bitrev_size8192_mixed24_f64<>+0x7E28(SB)/8, $2686
DATA bitrev_size8192_mixed24_f64<>+0x7E30(SB)/8, $4734
DATA bitrev_size8192_mixed24_f64<>+0x7E38(SB)/8, $6782
DATA bitrev_size8192_mixed24_f64<>+0x7E40(SB)/8, $1150
DATA bitrev_size8192_mixed24_f64<>+0x7E48(SB)/8, $3198
DATA bitrev_size8192_mixed24_f64<>+0x7E50(SB)/8, $5246
DATA bitrev_size8192_mixed24_f64<>+0x7E58(SB)/8, $7294
DATA bitrev_size8192_mixed24_f64<>+0x7E60(SB)/8, $1662
DATA bitrev_size8192_mixed24_f64<>+0x7E68(SB)/8, $3710
DATA bitrev_size8192_mixed24_f64<>+0x7E70(SB)/8, $5758
DATA bitrev_size8192_mixed24_f64<>+0x7E78(SB)/8, $7806
DATA bitrev_size8192_mixed24_f64<>+0x7E80(SB)/8, $254
DATA bitrev_size8192_mixed24_f64<>+0x7E88(SB)/8, $2302
DATA bitrev_size8192_mixed24_f64<>+0x7E90(SB)/8, $4350
DATA bitrev_size8192_mixed24_f64<>+0x7E98(SB)/8, $6398
DATA bitrev_size8192_mixed24_f64<>+0x7EA0(SB)/8, $766
DATA bitrev_size8192_mixed24_f64<>+0x7EA8(SB)/8, $2814
DATA bitrev_size8192_mixed24_f64<>+0x7EB0(SB)/8, $4862
DATA bitrev_size8192_mixed24_f64<>+0x7EB8(SB)/8, $6910
DATA bitrev_size8192_mixed24_f64<>+0x7EC0(SB)/8, $1278
DATA bitrev_size8192_mixed24_f64<>+0x7EC8(SB)/8, $3326
DATA bitrev_size8192_mixed24_f64<>+0x7ED0(SB)/8, $5374
DATA bitrev_size8192_mixed24_f64<>+0x7ED8(SB)/8, $7422
DATA bitrev_size8192_mixed24_f64<>+0x7EE0(SB)/8, $1790
DATA bitrev_size8192_mixed24_f64<>+0x7EE8(SB)/8, $3838
DATA bitrev_size8192_mixed24_f64<>+0x7EF0(SB)/8, $5886
DATA bitrev_size8192_mixed24_f64<>+0x7EF8(SB)/8, $7934
DATA bitrev_size8192_mixed24_f64<>+0x7F00(SB)/8, $382
DATA bitrev_size8192_mixed24_f64<>+0x7F08(SB)/8, $2430
DATA bitrev_size8192_mixed24_f64<>+0x7F10(SB)/8, $4478
DATA bitrev_size8192_mixed24_f64<>+0x7F18(SB)/8, $6526
DATA bitrev_size8192_mixed24_f64<>+0x7F20(SB)/8, $894
DATA bitrev_size8192_mixed24_f64<>+0x7F28(SB)/8, $2942
DATA bitrev_size8192_mixed24_f64<>+0x7F30(SB)/8, $4990
DATA bitrev_size8192_mixed24_f64<>+0x7F38(SB)/8, $7038
DATA bitrev_size8192_mixed24_f64<>+0x7F40(SB)/8, $1406
DATA bitrev_size8192_mixed24_f64<>+0x7F48(SB)/8, $3454
DATA bitrev_size8192_mixed24_f64<>+0x7F50(SB)/8, $5502
DATA bitrev_size8192_mixed24_f64<>+0x7F58(SB)/8, $7550
DATA bitrev_size8192_mixed24_f64<>+0x7F60(SB)/8, $1918
DATA bitrev_size8192_mixed24_f64<>+0x7F68(SB)/8, $3966
DATA bitrev_size8192_mixed24_f64<>+0x7F70(SB)/8, $6014
DATA bitrev_size8192_mixed24_f64<>+0x7F78(SB)/8, $8062
DATA bitrev_size8192_mixed24_f64<>+0x7F80(SB)/8, $510
DATA bitrev_size8192_mixed24_f64<>+0x7F88(SB)/8, $2558
DATA bitrev_size8192_mixed24_f64<>+0x7F90(SB)/8, $4606
DATA bitrev_size8192_mixed24_f64<>+0x7F98(SB)/8, $6654
DATA bitrev_size8192_mixed24_f64<>+0x7FA0(SB)/8, $1022
DATA bitrev_size8192_mixed24_f64<>+0x7FA8(SB)/8, $3070
DATA bitrev_size8192_mixed24_f64<>+0x7FB0(SB)/8, $5118
DATA bitrev_size8192_mixed24_f64<>+0x7FB8(SB)/8, $7166
DATA bitrev_size8192_mixed24_f64<>+0x7FC0(SB)/8, $1534
DATA bitrev_size8192_mixed24_f64<>+0x7FC8(SB)/8, $3582
DATA bitrev_size8192_mixed24_f64<>+0x7FD0(SB)/8, $5630
DATA bitrev_size8192_mixed24_f64<>+0x7FD8(SB)/8, $7678
DATA bitrev_size8192_mixed24_f64<>+0x7FE0(SB)/8, $2046
DATA bitrev_size8192_mixed24_f64<>+0x7FE8(SB)/8, $4094
DATA bitrev_size8192_mixed24_f64<>+0x7FF0(SB)/8, $6142
DATA bitrev_size8192_mixed24_f64<>+0x7FF8(SB)/8, $8190
DATA bitrev_size8192_mixed24_f64<>+0x8000(SB)/8, $1
DATA bitrev_size8192_mixed24_f64<>+0x8008(SB)/8, $2049
DATA bitrev_size8192_mixed24_f64<>+0x8010(SB)/8, $4097
DATA bitrev_size8192_mixed24_f64<>+0x8018(SB)/8, $6145
DATA bitrev_size8192_mixed24_f64<>+0x8020(SB)/8, $513
DATA bitrev_size8192_mixed24_f64<>+0x8028(SB)/8, $2561
DATA bitrev_size8192_mixed24_f64<>+0x8030(SB)/8, $4609
DATA bitrev_size8192_mixed24_f64<>+0x8038(SB)/8, $6657
DATA bitrev_size8192_mixed24_f64<>+0x8040(SB)/8, $1025
DATA bitrev_size8192_mixed24_f64<>+0x8048(SB)/8, $3073
DATA bitrev_size8192_mixed24_f64<>+0x8050(SB)/8, $5121
DATA bitrev_size8192_mixed24_f64<>+0x8058(SB)/8, $7169
DATA bitrev_size8192_mixed24_f64<>+0x8060(SB)/8, $1537
DATA bitrev_size8192_mixed24_f64<>+0x8068(SB)/8, $3585
DATA bitrev_size8192_mixed24_f64<>+0x8070(SB)/8, $5633
DATA bitrev_size8192_mixed24_f64<>+0x8078(SB)/8, $7681
DATA bitrev_size8192_mixed24_f64<>+0x8080(SB)/8, $129
DATA bitrev_size8192_mixed24_f64<>+0x8088(SB)/8, $2177
DATA bitrev_size8192_mixed24_f64<>+0x8090(SB)/8, $4225
DATA bitrev_size8192_mixed24_f64<>+0x8098(SB)/8, $6273
DATA bitrev_size8192_mixed24_f64<>+0x80A0(SB)/8, $641
DATA bitrev_size8192_mixed24_f64<>+0x80A8(SB)/8, $2689
DATA bitrev_size8192_mixed24_f64<>+0x80B0(SB)/8, $4737
DATA bitrev_size8192_mixed24_f64<>+0x80B8(SB)/8, $6785
DATA bitrev_size8192_mixed24_f64<>+0x80C0(SB)/8, $1153
DATA bitrev_size8192_mixed24_f64<>+0x80C8(SB)/8, $3201
DATA bitrev_size8192_mixed24_f64<>+0x80D0(SB)/8, $5249
DATA bitrev_size8192_mixed24_f64<>+0x80D8(SB)/8, $7297
DATA bitrev_size8192_mixed24_f64<>+0x80E0(SB)/8, $1665
DATA bitrev_size8192_mixed24_f64<>+0x80E8(SB)/8, $3713
DATA bitrev_size8192_mixed24_f64<>+0x80F0(SB)/8, $5761
DATA bitrev_size8192_mixed24_f64<>+0x80F8(SB)/8, $7809
DATA bitrev_size8192_mixed24_f64<>+0x8100(SB)/8, $257
DATA bitrev_size8192_mixed24_f64<>+0x8108(SB)/8, $2305
DATA bitrev_size8192_mixed24_f64<>+0x8110(SB)/8, $4353
DATA bitrev_size8192_mixed24_f64<>+0x8118(SB)/8, $6401
DATA bitrev_size8192_mixed24_f64<>+0x8120(SB)/8, $769
DATA bitrev_size8192_mixed24_f64<>+0x8128(SB)/8, $2817
DATA bitrev_size8192_mixed24_f64<>+0x8130(SB)/8, $4865
DATA bitrev_size8192_mixed24_f64<>+0x8138(SB)/8, $6913
DATA bitrev_size8192_mixed24_f64<>+0x8140(SB)/8, $1281
DATA bitrev_size8192_mixed24_f64<>+0x8148(SB)/8, $3329
DATA bitrev_size8192_mixed24_f64<>+0x8150(SB)/8, $5377
DATA bitrev_size8192_mixed24_f64<>+0x8158(SB)/8, $7425
DATA bitrev_size8192_mixed24_f64<>+0x8160(SB)/8, $1793
DATA bitrev_size8192_mixed24_f64<>+0x8168(SB)/8, $3841
DATA bitrev_size8192_mixed24_f64<>+0x8170(SB)/8, $5889
DATA bitrev_size8192_mixed24_f64<>+0x8178(SB)/8, $7937
DATA bitrev_size8192_mixed24_f64<>+0x8180(SB)/8, $385
DATA bitrev_size8192_mixed24_f64<>+0x8188(SB)/8, $2433
DATA bitrev_size8192_mixed24_f64<>+0x8190(SB)/8, $4481
DATA bitrev_size8192_mixed24_f64<>+0x8198(SB)/8, $6529
DATA bitrev_size8192_mixed24_f64<>+0x81A0(SB)/8, $897
DATA bitrev_size8192_mixed24_f64<>+0x81A8(SB)/8, $2945
DATA bitrev_size8192_mixed24_f64<>+0x81B0(SB)/8, $4993
DATA bitrev_size8192_mixed24_f64<>+0x81B8(SB)/8, $7041
DATA bitrev_size8192_mixed24_f64<>+0x81C0(SB)/8, $1409
DATA bitrev_size8192_mixed24_f64<>+0x81C8(SB)/8, $3457
DATA bitrev_size8192_mixed24_f64<>+0x81D0(SB)/8, $5505
DATA bitrev_size8192_mixed24_f64<>+0x81D8(SB)/8, $7553
DATA bitrev_size8192_mixed24_f64<>+0x81E0(SB)/8, $1921
DATA bitrev_size8192_mixed24_f64<>+0x81E8(SB)/8, $3969
DATA bitrev_size8192_mixed24_f64<>+0x81F0(SB)/8, $6017
DATA bitrev_size8192_mixed24_f64<>+0x81F8(SB)/8, $8065
DATA bitrev_size8192_mixed24_f64<>+0x8200(SB)/8, $33
DATA bitrev_size8192_mixed24_f64<>+0x8208(SB)/8, $2081
DATA bitrev_size8192_mixed24_f64<>+0x8210(SB)/8, $4129
DATA bitrev_size8192_mixed24_f64<>+0x8218(SB)/8, $6177
DATA bitrev_size8192_mixed24_f64<>+0x8220(SB)/8, $545
DATA bitrev_size8192_mixed24_f64<>+0x8228(SB)/8, $2593
DATA bitrev_size8192_mixed24_f64<>+0x8230(SB)/8, $4641
DATA bitrev_size8192_mixed24_f64<>+0x8238(SB)/8, $6689
DATA bitrev_size8192_mixed24_f64<>+0x8240(SB)/8, $1057
DATA bitrev_size8192_mixed24_f64<>+0x8248(SB)/8, $3105
DATA bitrev_size8192_mixed24_f64<>+0x8250(SB)/8, $5153
DATA bitrev_size8192_mixed24_f64<>+0x8258(SB)/8, $7201
DATA bitrev_size8192_mixed24_f64<>+0x8260(SB)/8, $1569
DATA bitrev_size8192_mixed24_f64<>+0x8268(SB)/8, $3617
DATA bitrev_size8192_mixed24_f64<>+0x8270(SB)/8, $5665
DATA bitrev_size8192_mixed24_f64<>+0x8278(SB)/8, $7713
DATA bitrev_size8192_mixed24_f64<>+0x8280(SB)/8, $161
DATA bitrev_size8192_mixed24_f64<>+0x8288(SB)/8, $2209
DATA bitrev_size8192_mixed24_f64<>+0x8290(SB)/8, $4257
DATA bitrev_size8192_mixed24_f64<>+0x8298(SB)/8, $6305
DATA bitrev_size8192_mixed24_f64<>+0x82A0(SB)/8, $673
DATA bitrev_size8192_mixed24_f64<>+0x82A8(SB)/8, $2721
DATA bitrev_size8192_mixed24_f64<>+0x82B0(SB)/8, $4769
DATA bitrev_size8192_mixed24_f64<>+0x82B8(SB)/8, $6817
DATA bitrev_size8192_mixed24_f64<>+0x82C0(SB)/8, $1185
DATA bitrev_size8192_mixed24_f64<>+0x82C8(SB)/8, $3233
DATA bitrev_size8192_mixed24_f64<>+0x82D0(SB)/8, $5281
DATA bitrev_size8192_mixed24_f64<>+0x82D8(SB)/8, $7329
DATA bitrev_size8192_mixed24_f64<>+0x82E0(SB)/8, $1697
DATA bitrev_size8192_mixed24_f64<>+0x82E8(SB)/8, $3745
DATA bitrev_size8192_mixed24_f64<>+0x82F0(SB)/8, $5793
DATA bitrev_size8192_mixed24_f64<>+0x82F8(SB)/8, $7841
DATA bitrev_size8192_mixed24_f64<>+0x8300(SB)/8, $289
DATA bitrev_size8192_mixed24_f64<>+0x8308(SB)/8, $2337
DATA bitrev_size8192_mixed24_f64<>+0x8310(SB)/8, $4385
DATA bitrev_size8192_mixed24_f64<>+0x8318(SB)/8, $6433
DATA bitrev_size8192_mixed24_f64<>+0x8320(SB)/8, $801
DATA bitrev_size8192_mixed24_f64<>+0x8328(SB)/8, $2849
DATA bitrev_size8192_mixed24_f64<>+0x8330(SB)/8, $4897
DATA bitrev_size8192_mixed24_f64<>+0x8338(SB)/8, $6945
DATA bitrev_size8192_mixed24_f64<>+0x8340(SB)/8, $1313
DATA bitrev_size8192_mixed24_f64<>+0x8348(SB)/8, $3361
DATA bitrev_size8192_mixed24_f64<>+0x8350(SB)/8, $5409
DATA bitrev_size8192_mixed24_f64<>+0x8358(SB)/8, $7457
DATA bitrev_size8192_mixed24_f64<>+0x8360(SB)/8, $1825
DATA bitrev_size8192_mixed24_f64<>+0x8368(SB)/8, $3873
DATA bitrev_size8192_mixed24_f64<>+0x8370(SB)/8, $5921
DATA bitrev_size8192_mixed24_f64<>+0x8378(SB)/8, $7969
DATA bitrev_size8192_mixed24_f64<>+0x8380(SB)/8, $417
DATA bitrev_size8192_mixed24_f64<>+0x8388(SB)/8, $2465
DATA bitrev_size8192_mixed24_f64<>+0x8390(SB)/8, $4513
DATA bitrev_size8192_mixed24_f64<>+0x8398(SB)/8, $6561
DATA bitrev_size8192_mixed24_f64<>+0x83A0(SB)/8, $929
DATA bitrev_size8192_mixed24_f64<>+0x83A8(SB)/8, $2977
DATA bitrev_size8192_mixed24_f64<>+0x83B0(SB)/8, $5025
DATA bitrev_size8192_mixed24_f64<>+0x83B8(SB)/8, $7073
DATA bitrev_size8192_mixed24_f64<>+0x83C0(SB)/8, $1441
DATA bitrev_size8192_mixed24_f64<>+0x83C8(SB)/8, $3489
DATA bitrev_size8192_mixed24_f64<>+0x83D0(SB)/8, $5537
DATA bitrev_size8192_mixed24_f64<>+0x83D8(SB)/8, $7585
DATA bitrev_size8192_mixed24_f64<>+0x83E0(SB)/8, $1953
DATA bitrev_size8192_mixed24_f64<>+0x83E8(SB)/8, $4001
DATA bitrev_size8192_mixed24_f64<>+0x83F0(SB)/8, $6049
DATA bitrev_size8192_mixed24_f64<>+0x83F8(SB)/8, $8097
DATA bitrev_size8192_mixed24_f64<>+0x8400(SB)/8, $65
DATA bitrev_size8192_mixed24_f64<>+0x8408(SB)/8, $2113
DATA bitrev_size8192_mixed24_f64<>+0x8410(SB)/8, $4161
DATA bitrev_size8192_mixed24_f64<>+0x8418(SB)/8, $6209
DATA bitrev_size8192_mixed24_f64<>+0x8420(SB)/8, $577
DATA bitrev_size8192_mixed24_f64<>+0x8428(SB)/8, $2625
DATA bitrev_size8192_mixed24_f64<>+0x8430(SB)/8, $4673
DATA bitrev_size8192_mixed24_f64<>+0x8438(SB)/8, $6721
DATA bitrev_size8192_mixed24_f64<>+0x8440(SB)/8, $1089
DATA bitrev_size8192_mixed24_f64<>+0x8448(SB)/8, $3137
DATA bitrev_size8192_mixed24_f64<>+0x8450(SB)/8, $5185
DATA bitrev_size8192_mixed24_f64<>+0x8458(SB)/8, $7233
DATA bitrev_size8192_mixed24_f64<>+0x8460(SB)/8, $1601
DATA bitrev_size8192_mixed24_f64<>+0x8468(SB)/8, $3649
DATA bitrev_size8192_mixed24_f64<>+0x8470(SB)/8, $5697
DATA bitrev_size8192_mixed24_f64<>+0x8478(SB)/8, $7745
DATA bitrev_size8192_mixed24_f64<>+0x8480(SB)/8, $193
DATA bitrev_size8192_mixed24_f64<>+0x8488(SB)/8, $2241
DATA bitrev_size8192_mixed24_f64<>+0x8490(SB)/8, $4289
DATA bitrev_size8192_mixed24_f64<>+0x8498(SB)/8, $6337
DATA bitrev_size8192_mixed24_f64<>+0x84A0(SB)/8, $705
DATA bitrev_size8192_mixed24_f64<>+0x84A8(SB)/8, $2753
DATA bitrev_size8192_mixed24_f64<>+0x84B0(SB)/8, $4801
DATA bitrev_size8192_mixed24_f64<>+0x84B8(SB)/8, $6849
DATA bitrev_size8192_mixed24_f64<>+0x84C0(SB)/8, $1217
DATA bitrev_size8192_mixed24_f64<>+0x84C8(SB)/8, $3265
DATA bitrev_size8192_mixed24_f64<>+0x84D0(SB)/8, $5313
DATA bitrev_size8192_mixed24_f64<>+0x84D8(SB)/8, $7361
DATA bitrev_size8192_mixed24_f64<>+0x84E0(SB)/8, $1729
DATA bitrev_size8192_mixed24_f64<>+0x84E8(SB)/8, $3777
DATA bitrev_size8192_mixed24_f64<>+0x84F0(SB)/8, $5825
DATA bitrev_size8192_mixed24_f64<>+0x84F8(SB)/8, $7873
DATA bitrev_size8192_mixed24_f64<>+0x8500(SB)/8, $321
DATA bitrev_size8192_mixed24_f64<>+0x8508(SB)/8, $2369
DATA bitrev_size8192_mixed24_f64<>+0x8510(SB)/8, $4417
DATA bitrev_size8192_mixed24_f64<>+0x8518(SB)/8, $6465
DATA bitrev_size8192_mixed24_f64<>+0x8520(SB)/8, $833
DATA bitrev_size8192_mixed24_f64<>+0x8528(SB)/8, $2881
DATA bitrev_size8192_mixed24_f64<>+0x8530(SB)/8, $4929
DATA bitrev_size8192_mixed24_f64<>+0x8538(SB)/8, $6977
DATA bitrev_size8192_mixed24_f64<>+0x8540(SB)/8, $1345
DATA bitrev_size8192_mixed24_f64<>+0x8548(SB)/8, $3393
DATA bitrev_size8192_mixed24_f64<>+0x8550(SB)/8, $5441
DATA bitrev_size8192_mixed24_f64<>+0x8558(SB)/8, $7489
DATA bitrev_size8192_mixed24_f64<>+0x8560(SB)/8, $1857
DATA bitrev_size8192_mixed24_f64<>+0x8568(SB)/8, $3905
DATA bitrev_size8192_mixed24_f64<>+0x8570(SB)/8, $5953
DATA bitrev_size8192_mixed24_f64<>+0x8578(SB)/8, $8001
DATA bitrev_size8192_mixed24_f64<>+0x8580(SB)/8, $449
DATA bitrev_size8192_mixed24_f64<>+0x8588(SB)/8, $2497
DATA bitrev_size8192_mixed24_f64<>+0x8590(SB)/8, $4545
DATA bitrev_size8192_mixed24_f64<>+0x8598(SB)/8, $6593
DATA bitrev_size8192_mixed24_f64<>+0x85A0(SB)/8, $961
DATA bitrev_size8192_mixed24_f64<>+0x85A8(SB)/8, $3009
DATA bitrev_size8192_mixed24_f64<>+0x85B0(SB)/8, $5057
DATA bitrev_size8192_mixed24_f64<>+0x85B8(SB)/8, $7105
DATA bitrev_size8192_mixed24_f64<>+0x85C0(SB)/8, $1473
DATA bitrev_size8192_mixed24_f64<>+0x85C8(SB)/8, $3521
DATA bitrev_size8192_mixed24_f64<>+0x85D0(SB)/8, $5569
DATA bitrev_size8192_mixed24_f64<>+0x85D8(SB)/8, $7617
DATA bitrev_size8192_mixed24_f64<>+0x85E0(SB)/8, $1985
DATA bitrev_size8192_mixed24_f64<>+0x85E8(SB)/8, $4033
DATA bitrev_size8192_mixed24_f64<>+0x85F0(SB)/8, $6081
DATA bitrev_size8192_mixed24_f64<>+0x85F8(SB)/8, $8129
DATA bitrev_size8192_mixed24_f64<>+0x8600(SB)/8, $97
DATA bitrev_size8192_mixed24_f64<>+0x8608(SB)/8, $2145
DATA bitrev_size8192_mixed24_f64<>+0x8610(SB)/8, $4193
DATA bitrev_size8192_mixed24_f64<>+0x8618(SB)/8, $6241
DATA bitrev_size8192_mixed24_f64<>+0x8620(SB)/8, $609
DATA bitrev_size8192_mixed24_f64<>+0x8628(SB)/8, $2657
DATA bitrev_size8192_mixed24_f64<>+0x8630(SB)/8, $4705
DATA bitrev_size8192_mixed24_f64<>+0x8638(SB)/8, $6753
DATA bitrev_size8192_mixed24_f64<>+0x8640(SB)/8, $1121
DATA bitrev_size8192_mixed24_f64<>+0x8648(SB)/8, $3169
DATA bitrev_size8192_mixed24_f64<>+0x8650(SB)/8, $5217
DATA bitrev_size8192_mixed24_f64<>+0x8658(SB)/8, $7265
DATA bitrev_size8192_mixed24_f64<>+0x8660(SB)/8, $1633
DATA bitrev_size8192_mixed24_f64<>+0x8668(SB)/8, $3681
DATA bitrev_size8192_mixed24_f64<>+0x8670(SB)/8, $5729
DATA bitrev_size8192_mixed24_f64<>+0x8678(SB)/8, $7777
DATA bitrev_size8192_mixed24_f64<>+0x8680(SB)/8, $225
DATA bitrev_size8192_mixed24_f64<>+0x8688(SB)/8, $2273
DATA bitrev_size8192_mixed24_f64<>+0x8690(SB)/8, $4321
DATA bitrev_size8192_mixed24_f64<>+0x8698(SB)/8, $6369
DATA bitrev_size8192_mixed24_f64<>+0x86A0(SB)/8, $737
DATA bitrev_size8192_mixed24_f64<>+0x86A8(SB)/8, $2785
DATA bitrev_size8192_mixed24_f64<>+0x86B0(SB)/8, $4833
DATA bitrev_size8192_mixed24_f64<>+0x86B8(SB)/8, $6881
DATA bitrev_size8192_mixed24_f64<>+0x86C0(SB)/8, $1249
DATA bitrev_size8192_mixed24_f64<>+0x86C8(SB)/8, $3297
DATA bitrev_size8192_mixed24_f64<>+0x86D0(SB)/8, $5345
DATA bitrev_size8192_mixed24_f64<>+0x86D8(SB)/8, $7393
DATA bitrev_size8192_mixed24_f64<>+0x86E0(SB)/8, $1761
DATA bitrev_size8192_mixed24_f64<>+0x86E8(SB)/8, $3809
DATA bitrev_size8192_mixed24_f64<>+0x86F0(SB)/8, $5857
DATA bitrev_size8192_mixed24_f64<>+0x86F8(SB)/8, $7905
DATA bitrev_size8192_mixed24_f64<>+0x8700(SB)/8, $353
DATA bitrev_size8192_mixed24_f64<>+0x8708(SB)/8, $2401
DATA bitrev_size8192_mixed24_f64<>+0x8710(SB)/8, $4449
DATA bitrev_size8192_mixed24_f64<>+0x8718(SB)/8, $6497
DATA bitrev_size8192_mixed24_f64<>+0x8720(SB)/8, $865
DATA bitrev_size8192_mixed24_f64<>+0x8728(SB)/8, $2913
DATA bitrev_size8192_mixed24_f64<>+0x8730(SB)/8, $4961
DATA bitrev_size8192_mixed24_f64<>+0x8738(SB)/8, $7009
DATA bitrev_size8192_mixed24_f64<>+0x8740(SB)/8, $1377
DATA bitrev_size8192_mixed24_f64<>+0x8748(SB)/8, $3425
DATA bitrev_size8192_mixed24_f64<>+0x8750(SB)/8, $5473
DATA bitrev_size8192_mixed24_f64<>+0x8758(SB)/8, $7521
DATA bitrev_size8192_mixed24_f64<>+0x8760(SB)/8, $1889
DATA bitrev_size8192_mixed24_f64<>+0x8768(SB)/8, $3937
DATA bitrev_size8192_mixed24_f64<>+0x8770(SB)/8, $5985
DATA bitrev_size8192_mixed24_f64<>+0x8778(SB)/8, $8033
DATA bitrev_size8192_mixed24_f64<>+0x8780(SB)/8, $481
DATA bitrev_size8192_mixed24_f64<>+0x8788(SB)/8, $2529
DATA bitrev_size8192_mixed24_f64<>+0x8790(SB)/8, $4577
DATA bitrev_size8192_mixed24_f64<>+0x8798(SB)/8, $6625
DATA bitrev_size8192_mixed24_f64<>+0x87A0(SB)/8, $993
DATA bitrev_size8192_mixed24_f64<>+0x87A8(SB)/8, $3041
DATA bitrev_size8192_mixed24_f64<>+0x87B0(SB)/8, $5089
DATA bitrev_size8192_mixed24_f64<>+0x87B8(SB)/8, $7137
DATA bitrev_size8192_mixed24_f64<>+0x87C0(SB)/8, $1505
DATA bitrev_size8192_mixed24_f64<>+0x87C8(SB)/8, $3553
DATA bitrev_size8192_mixed24_f64<>+0x87D0(SB)/8, $5601
DATA bitrev_size8192_mixed24_f64<>+0x87D8(SB)/8, $7649
DATA bitrev_size8192_mixed24_f64<>+0x87E0(SB)/8, $2017
DATA bitrev_size8192_mixed24_f64<>+0x87E8(SB)/8, $4065
DATA bitrev_size8192_mixed24_f64<>+0x87F0(SB)/8, $6113
DATA bitrev_size8192_mixed24_f64<>+0x87F8(SB)/8, $8161
DATA bitrev_size8192_mixed24_f64<>+0x8800(SB)/8, $9
DATA bitrev_size8192_mixed24_f64<>+0x8808(SB)/8, $2057
DATA bitrev_size8192_mixed24_f64<>+0x8810(SB)/8, $4105
DATA bitrev_size8192_mixed24_f64<>+0x8818(SB)/8, $6153
DATA bitrev_size8192_mixed24_f64<>+0x8820(SB)/8, $521
DATA bitrev_size8192_mixed24_f64<>+0x8828(SB)/8, $2569
DATA bitrev_size8192_mixed24_f64<>+0x8830(SB)/8, $4617
DATA bitrev_size8192_mixed24_f64<>+0x8838(SB)/8, $6665
DATA bitrev_size8192_mixed24_f64<>+0x8840(SB)/8, $1033
DATA bitrev_size8192_mixed24_f64<>+0x8848(SB)/8, $3081
DATA bitrev_size8192_mixed24_f64<>+0x8850(SB)/8, $5129
DATA bitrev_size8192_mixed24_f64<>+0x8858(SB)/8, $7177
DATA bitrev_size8192_mixed24_f64<>+0x8860(SB)/8, $1545
DATA bitrev_size8192_mixed24_f64<>+0x8868(SB)/8, $3593
DATA bitrev_size8192_mixed24_f64<>+0x8870(SB)/8, $5641
DATA bitrev_size8192_mixed24_f64<>+0x8878(SB)/8, $7689
DATA bitrev_size8192_mixed24_f64<>+0x8880(SB)/8, $137
DATA bitrev_size8192_mixed24_f64<>+0x8888(SB)/8, $2185
DATA bitrev_size8192_mixed24_f64<>+0x8890(SB)/8, $4233
DATA bitrev_size8192_mixed24_f64<>+0x8898(SB)/8, $6281
DATA bitrev_size8192_mixed24_f64<>+0x88A0(SB)/8, $649
DATA bitrev_size8192_mixed24_f64<>+0x88A8(SB)/8, $2697
DATA bitrev_size8192_mixed24_f64<>+0x88B0(SB)/8, $4745
DATA bitrev_size8192_mixed24_f64<>+0x88B8(SB)/8, $6793
DATA bitrev_size8192_mixed24_f64<>+0x88C0(SB)/8, $1161
DATA bitrev_size8192_mixed24_f64<>+0x88C8(SB)/8, $3209
DATA bitrev_size8192_mixed24_f64<>+0x88D0(SB)/8, $5257
DATA bitrev_size8192_mixed24_f64<>+0x88D8(SB)/8, $7305
DATA bitrev_size8192_mixed24_f64<>+0x88E0(SB)/8, $1673
DATA bitrev_size8192_mixed24_f64<>+0x88E8(SB)/8, $3721
DATA bitrev_size8192_mixed24_f64<>+0x88F0(SB)/8, $5769
DATA bitrev_size8192_mixed24_f64<>+0x88F8(SB)/8, $7817
DATA bitrev_size8192_mixed24_f64<>+0x8900(SB)/8, $265
DATA bitrev_size8192_mixed24_f64<>+0x8908(SB)/8, $2313
DATA bitrev_size8192_mixed24_f64<>+0x8910(SB)/8, $4361
DATA bitrev_size8192_mixed24_f64<>+0x8918(SB)/8, $6409
DATA bitrev_size8192_mixed24_f64<>+0x8920(SB)/8, $777
DATA bitrev_size8192_mixed24_f64<>+0x8928(SB)/8, $2825
DATA bitrev_size8192_mixed24_f64<>+0x8930(SB)/8, $4873
DATA bitrev_size8192_mixed24_f64<>+0x8938(SB)/8, $6921
DATA bitrev_size8192_mixed24_f64<>+0x8940(SB)/8, $1289
DATA bitrev_size8192_mixed24_f64<>+0x8948(SB)/8, $3337
DATA bitrev_size8192_mixed24_f64<>+0x8950(SB)/8, $5385
DATA bitrev_size8192_mixed24_f64<>+0x8958(SB)/8, $7433
DATA bitrev_size8192_mixed24_f64<>+0x8960(SB)/8, $1801
DATA bitrev_size8192_mixed24_f64<>+0x8968(SB)/8, $3849
DATA bitrev_size8192_mixed24_f64<>+0x8970(SB)/8, $5897
DATA bitrev_size8192_mixed24_f64<>+0x8978(SB)/8, $7945
DATA bitrev_size8192_mixed24_f64<>+0x8980(SB)/8, $393
DATA bitrev_size8192_mixed24_f64<>+0x8988(SB)/8, $2441
DATA bitrev_size8192_mixed24_f64<>+0x8990(SB)/8, $4489
DATA bitrev_size8192_mixed24_f64<>+0x8998(SB)/8, $6537
DATA bitrev_size8192_mixed24_f64<>+0x89A0(SB)/8, $905
DATA bitrev_size8192_mixed24_f64<>+0x89A8(SB)/8, $2953
DATA bitrev_size8192_mixed24_f64<>+0x89B0(SB)/8, $5001
DATA bitrev_size8192_mixed24_f64<>+0x89B8(SB)/8, $7049
DATA bitrev_size8192_mixed24_f64<>+0x89C0(SB)/8, $1417
DATA bitrev_size8192_mixed24_f64<>+0x89C8(SB)/8, $3465
DATA bitrev_size8192_mixed24_f64<>+0x89D0(SB)/8, $5513
DATA bitrev_size8192_mixed24_f64<>+0x89D8(SB)/8, $7561
DATA bitrev_size8192_mixed24_f64<>+0x89E0(SB)/8, $1929
DATA bitrev_size8192_mixed24_f64<>+0x89E8(SB)/8, $3977
DATA bitrev_size8192_mixed24_f64<>+0x89F0(SB)/8, $6025
DATA bitrev_size8192_mixed24_f64<>+0x89F8(SB)/8, $8073
DATA bitrev_size8192_mixed24_f64<>+0x8A00(SB)/8, $41
DATA bitrev_size8192_mixed24_f64<>+0x8A08(SB)/8, $2089
DATA bitrev_size8192_mixed24_f64<>+0x8A10(SB)/8, $4137
DATA bitrev_size8192_mixed24_f64<>+0x8A18(SB)/8, $6185
DATA bitrev_size8192_mixed24_f64<>+0x8A20(SB)/8, $553
DATA bitrev_size8192_mixed24_f64<>+0x8A28(SB)/8, $2601
DATA bitrev_size8192_mixed24_f64<>+0x8A30(SB)/8, $4649
DATA bitrev_size8192_mixed24_f64<>+0x8A38(SB)/8, $6697
DATA bitrev_size8192_mixed24_f64<>+0x8A40(SB)/8, $1065
DATA bitrev_size8192_mixed24_f64<>+0x8A48(SB)/8, $3113
DATA bitrev_size8192_mixed24_f64<>+0x8A50(SB)/8, $5161
DATA bitrev_size8192_mixed24_f64<>+0x8A58(SB)/8, $7209
DATA bitrev_size8192_mixed24_f64<>+0x8A60(SB)/8, $1577
DATA bitrev_size8192_mixed24_f64<>+0x8A68(SB)/8, $3625
DATA bitrev_size8192_mixed24_f64<>+0x8A70(SB)/8, $5673
DATA bitrev_size8192_mixed24_f64<>+0x8A78(SB)/8, $7721
DATA bitrev_size8192_mixed24_f64<>+0x8A80(SB)/8, $169
DATA bitrev_size8192_mixed24_f64<>+0x8A88(SB)/8, $2217
DATA bitrev_size8192_mixed24_f64<>+0x8A90(SB)/8, $4265
DATA bitrev_size8192_mixed24_f64<>+0x8A98(SB)/8, $6313
DATA bitrev_size8192_mixed24_f64<>+0x8AA0(SB)/8, $681
DATA bitrev_size8192_mixed24_f64<>+0x8AA8(SB)/8, $2729
DATA bitrev_size8192_mixed24_f64<>+0x8AB0(SB)/8, $4777
DATA bitrev_size8192_mixed24_f64<>+0x8AB8(SB)/8, $6825
DATA bitrev_size8192_mixed24_f64<>+0x8AC0(SB)/8, $1193
DATA bitrev_size8192_mixed24_f64<>+0x8AC8(SB)/8, $3241
DATA bitrev_size8192_mixed24_f64<>+0x8AD0(SB)/8, $5289
DATA bitrev_size8192_mixed24_f64<>+0x8AD8(SB)/8, $7337
DATA bitrev_size8192_mixed24_f64<>+0x8AE0(SB)/8, $1705
DATA bitrev_size8192_mixed24_f64<>+0x8AE8(SB)/8, $3753
DATA bitrev_size8192_mixed24_f64<>+0x8AF0(SB)/8, $5801
DATA bitrev_size8192_mixed24_f64<>+0x8AF8(SB)/8, $7849
DATA bitrev_size8192_mixed24_f64<>+0x8B00(SB)/8, $297
DATA bitrev_size8192_mixed24_f64<>+0x8B08(SB)/8, $2345
DATA bitrev_size8192_mixed24_f64<>+0x8B10(SB)/8, $4393
DATA bitrev_size8192_mixed24_f64<>+0x8B18(SB)/8, $6441
DATA bitrev_size8192_mixed24_f64<>+0x8B20(SB)/8, $809
DATA bitrev_size8192_mixed24_f64<>+0x8B28(SB)/8, $2857
DATA bitrev_size8192_mixed24_f64<>+0x8B30(SB)/8, $4905
DATA bitrev_size8192_mixed24_f64<>+0x8B38(SB)/8, $6953
DATA bitrev_size8192_mixed24_f64<>+0x8B40(SB)/8, $1321
DATA bitrev_size8192_mixed24_f64<>+0x8B48(SB)/8, $3369
DATA bitrev_size8192_mixed24_f64<>+0x8B50(SB)/8, $5417
DATA bitrev_size8192_mixed24_f64<>+0x8B58(SB)/8, $7465
DATA bitrev_size8192_mixed24_f64<>+0x8B60(SB)/8, $1833
DATA bitrev_size8192_mixed24_f64<>+0x8B68(SB)/8, $3881
DATA bitrev_size8192_mixed24_f64<>+0x8B70(SB)/8, $5929
DATA bitrev_size8192_mixed24_f64<>+0x8B78(SB)/8, $7977
DATA bitrev_size8192_mixed24_f64<>+0x8B80(SB)/8, $425
DATA bitrev_size8192_mixed24_f64<>+0x8B88(SB)/8, $2473
DATA bitrev_size8192_mixed24_f64<>+0x8B90(SB)/8, $4521
DATA bitrev_size8192_mixed24_f64<>+0x8B98(SB)/8, $6569
DATA bitrev_size8192_mixed24_f64<>+0x8BA0(SB)/8, $937
DATA bitrev_size8192_mixed24_f64<>+0x8BA8(SB)/8, $2985
DATA bitrev_size8192_mixed24_f64<>+0x8BB0(SB)/8, $5033
DATA bitrev_size8192_mixed24_f64<>+0x8BB8(SB)/8, $7081
DATA bitrev_size8192_mixed24_f64<>+0x8BC0(SB)/8, $1449
DATA bitrev_size8192_mixed24_f64<>+0x8BC8(SB)/8, $3497
DATA bitrev_size8192_mixed24_f64<>+0x8BD0(SB)/8, $5545
DATA bitrev_size8192_mixed24_f64<>+0x8BD8(SB)/8, $7593
DATA bitrev_size8192_mixed24_f64<>+0x8BE0(SB)/8, $1961
DATA bitrev_size8192_mixed24_f64<>+0x8BE8(SB)/8, $4009
DATA bitrev_size8192_mixed24_f64<>+0x8BF0(SB)/8, $6057
DATA bitrev_size8192_mixed24_f64<>+0x8BF8(SB)/8, $8105
DATA bitrev_size8192_mixed24_f64<>+0x8C00(SB)/8, $73
DATA bitrev_size8192_mixed24_f64<>+0x8C08(SB)/8, $2121
DATA bitrev_size8192_mixed24_f64<>+0x8C10(SB)/8, $4169
DATA bitrev_size8192_mixed24_f64<>+0x8C18(SB)/8, $6217
DATA bitrev_size8192_mixed24_f64<>+0x8C20(SB)/8, $585
DATA bitrev_size8192_mixed24_f64<>+0x8C28(SB)/8, $2633
DATA bitrev_size8192_mixed24_f64<>+0x8C30(SB)/8, $4681
DATA bitrev_size8192_mixed24_f64<>+0x8C38(SB)/8, $6729
DATA bitrev_size8192_mixed24_f64<>+0x8C40(SB)/8, $1097
DATA bitrev_size8192_mixed24_f64<>+0x8C48(SB)/8, $3145
DATA bitrev_size8192_mixed24_f64<>+0x8C50(SB)/8, $5193
DATA bitrev_size8192_mixed24_f64<>+0x8C58(SB)/8, $7241
DATA bitrev_size8192_mixed24_f64<>+0x8C60(SB)/8, $1609
DATA bitrev_size8192_mixed24_f64<>+0x8C68(SB)/8, $3657
DATA bitrev_size8192_mixed24_f64<>+0x8C70(SB)/8, $5705
DATA bitrev_size8192_mixed24_f64<>+0x8C78(SB)/8, $7753
DATA bitrev_size8192_mixed24_f64<>+0x8C80(SB)/8, $201
DATA bitrev_size8192_mixed24_f64<>+0x8C88(SB)/8, $2249
DATA bitrev_size8192_mixed24_f64<>+0x8C90(SB)/8, $4297
DATA bitrev_size8192_mixed24_f64<>+0x8C98(SB)/8, $6345
DATA bitrev_size8192_mixed24_f64<>+0x8CA0(SB)/8, $713
DATA bitrev_size8192_mixed24_f64<>+0x8CA8(SB)/8, $2761
DATA bitrev_size8192_mixed24_f64<>+0x8CB0(SB)/8, $4809
DATA bitrev_size8192_mixed24_f64<>+0x8CB8(SB)/8, $6857
DATA bitrev_size8192_mixed24_f64<>+0x8CC0(SB)/8, $1225
DATA bitrev_size8192_mixed24_f64<>+0x8CC8(SB)/8, $3273
DATA bitrev_size8192_mixed24_f64<>+0x8CD0(SB)/8, $5321
DATA bitrev_size8192_mixed24_f64<>+0x8CD8(SB)/8, $7369
DATA bitrev_size8192_mixed24_f64<>+0x8CE0(SB)/8, $1737
DATA bitrev_size8192_mixed24_f64<>+0x8CE8(SB)/8, $3785
DATA bitrev_size8192_mixed24_f64<>+0x8CF0(SB)/8, $5833
DATA bitrev_size8192_mixed24_f64<>+0x8CF8(SB)/8, $7881
DATA bitrev_size8192_mixed24_f64<>+0x8D00(SB)/8, $329
DATA bitrev_size8192_mixed24_f64<>+0x8D08(SB)/8, $2377
DATA bitrev_size8192_mixed24_f64<>+0x8D10(SB)/8, $4425
DATA bitrev_size8192_mixed24_f64<>+0x8D18(SB)/8, $6473
DATA bitrev_size8192_mixed24_f64<>+0x8D20(SB)/8, $841
DATA bitrev_size8192_mixed24_f64<>+0x8D28(SB)/8, $2889
DATA bitrev_size8192_mixed24_f64<>+0x8D30(SB)/8, $4937
DATA bitrev_size8192_mixed24_f64<>+0x8D38(SB)/8, $6985
DATA bitrev_size8192_mixed24_f64<>+0x8D40(SB)/8, $1353
DATA bitrev_size8192_mixed24_f64<>+0x8D48(SB)/8, $3401
DATA bitrev_size8192_mixed24_f64<>+0x8D50(SB)/8, $5449
DATA bitrev_size8192_mixed24_f64<>+0x8D58(SB)/8, $7497
DATA bitrev_size8192_mixed24_f64<>+0x8D60(SB)/8, $1865
DATA bitrev_size8192_mixed24_f64<>+0x8D68(SB)/8, $3913
DATA bitrev_size8192_mixed24_f64<>+0x8D70(SB)/8, $5961
DATA bitrev_size8192_mixed24_f64<>+0x8D78(SB)/8, $8009
DATA bitrev_size8192_mixed24_f64<>+0x8D80(SB)/8, $457
DATA bitrev_size8192_mixed24_f64<>+0x8D88(SB)/8, $2505
DATA bitrev_size8192_mixed24_f64<>+0x8D90(SB)/8, $4553
DATA bitrev_size8192_mixed24_f64<>+0x8D98(SB)/8, $6601
DATA bitrev_size8192_mixed24_f64<>+0x8DA0(SB)/8, $969
DATA bitrev_size8192_mixed24_f64<>+0x8DA8(SB)/8, $3017
DATA bitrev_size8192_mixed24_f64<>+0x8DB0(SB)/8, $5065
DATA bitrev_size8192_mixed24_f64<>+0x8DB8(SB)/8, $7113
DATA bitrev_size8192_mixed24_f64<>+0x8DC0(SB)/8, $1481
DATA bitrev_size8192_mixed24_f64<>+0x8DC8(SB)/8, $3529
DATA bitrev_size8192_mixed24_f64<>+0x8DD0(SB)/8, $5577
DATA bitrev_size8192_mixed24_f64<>+0x8DD8(SB)/8, $7625
DATA bitrev_size8192_mixed24_f64<>+0x8DE0(SB)/8, $1993
DATA bitrev_size8192_mixed24_f64<>+0x8DE8(SB)/8, $4041
DATA bitrev_size8192_mixed24_f64<>+0x8DF0(SB)/8, $6089
DATA bitrev_size8192_mixed24_f64<>+0x8DF8(SB)/8, $8137
DATA bitrev_size8192_mixed24_f64<>+0x8E00(SB)/8, $105
DATA bitrev_size8192_mixed24_f64<>+0x8E08(SB)/8, $2153
DATA bitrev_size8192_mixed24_f64<>+0x8E10(SB)/8, $4201
DATA bitrev_size8192_mixed24_f64<>+0x8E18(SB)/8, $6249
DATA bitrev_size8192_mixed24_f64<>+0x8E20(SB)/8, $617
DATA bitrev_size8192_mixed24_f64<>+0x8E28(SB)/8, $2665
DATA bitrev_size8192_mixed24_f64<>+0x8E30(SB)/8, $4713
DATA bitrev_size8192_mixed24_f64<>+0x8E38(SB)/8, $6761
DATA bitrev_size8192_mixed24_f64<>+0x8E40(SB)/8, $1129
DATA bitrev_size8192_mixed24_f64<>+0x8E48(SB)/8, $3177
DATA bitrev_size8192_mixed24_f64<>+0x8E50(SB)/8, $5225
DATA bitrev_size8192_mixed24_f64<>+0x8E58(SB)/8, $7273
DATA bitrev_size8192_mixed24_f64<>+0x8E60(SB)/8, $1641
DATA bitrev_size8192_mixed24_f64<>+0x8E68(SB)/8, $3689
DATA bitrev_size8192_mixed24_f64<>+0x8E70(SB)/8, $5737
DATA bitrev_size8192_mixed24_f64<>+0x8E78(SB)/8, $7785
DATA bitrev_size8192_mixed24_f64<>+0x8E80(SB)/8, $233
DATA bitrev_size8192_mixed24_f64<>+0x8E88(SB)/8, $2281
DATA bitrev_size8192_mixed24_f64<>+0x8E90(SB)/8, $4329
DATA bitrev_size8192_mixed24_f64<>+0x8E98(SB)/8, $6377
DATA bitrev_size8192_mixed24_f64<>+0x8EA0(SB)/8, $745
DATA bitrev_size8192_mixed24_f64<>+0x8EA8(SB)/8, $2793
DATA bitrev_size8192_mixed24_f64<>+0x8EB0(SB)/8, $4841
DATA bitrev_size8192_mixed24_f64<>+0x8EB8(SB)/8, $6889
DATA bitrev_size8192_mixed24_f64<>+0x8EC0(SB)/8, $1257
DATA bitrev_size8192_mixed24_f64<>+0x8EC8(SB)/8, $3305
DATA bitrev_size8192_mixed24_f64<>+0x8ED0(SB)/8, $5353
DATA bitrev_size8192_mixed24_f64<>+0x8ED8(SB)/8, $7401
DATA bitrev_size8192_mixed24_f64<>+0x8EE0(SB)/8, $1769
DATA bitrev_size8192_mixed24_f64<>+0x8EE8(SB)/8, $3817
DATA bitrev_size8192_mixed24_f64<>+0x8EF0(SB)/8, $5865
DATA bitrev_size8192_mixed24_f64<>+0x8EF8(SB)/8, $7913
DATA bitrev_size8192_mixed24_f64<>+0x8F00(SB)/8, $361
DATA bitrev_size8192_mixed24_f64<>+0x8F08(SB)/8, $2409
DATA bitrev_size8192_mixed24_f64<>+0x8F10(SB)/8, $4457
DATA bitrev_size8192_mixed24_f64<>+0x8F18(SB)/8, $6505
DATA bitrev_size8192_mixed24_f64<>+0x8F20(SB)/8, $873
DATA bitrev_size8192_mixed24_f64<>+0x8F28(SB)/8, $2921
DATA bitrev_size8192_mixed24_f64<>+0x8F30(SB)/8, $4969
DATA bitrev_size8192_mixed24_f64<>+0x8F38(SB)/8, $7017
DATA bitrev_size8192_mixed24_f64<>+0x8F40(SB)/8, $1385
DATA bitrev_size8192_mixed24_f64<>+0x8F48(SB)/8, $3433
DATA bitrev_size8192_mixed24_f64<>+0x8F50(SB)/8, $5481
DATA bitrev_size8192_mixed24_f64<>+0x8F58(SB)/8, $7529
DATA bitrev_size8192_mixed24_f64<>+0x8F60(SB)/8, $1897
DATA bitrev_size8192_mixed24_f64<>+0x8F68(SB)/8, $3945
DATA bitrev_size8192_mixed24_f64<>+0x8F70(SB)/8, $5993
DATA bitrev_size8192_mixed24_f64<>+0x8F78(SB)/8, $8041
DATA bitrev_size8192_mixed24_f64<>+0x8F80(SB)/8, $489
DATA bitrev_size8192_mixed24_f64<>+0x8F88(SB)/8, $2537
DATA bitrev_size8192_mixed24_f64<>+0x8F90(SB)/8, $4585
DATA bitrev_size8192_mixed24_f64<>+0x8F98(SB)/8, $6633
DATA bitrev_size8192_mixed24_f64<>+0x8FA0(SB)/8, $1001
DATA bitrev_size8192_mixed24_f64<>+0x8FA8(SB)/8, $3049
DATA bitrev_size8192_mixed24_f64<>+0x8FB0(SB)/8, $5097
DATA bitrev_size8192_mixed24_f64<>+0x8FB8(SB)/8, $7145
DATA bitrev_size8192_mixed24_f64<>+0x8FC0(SB)/8, $1513
DATA bitrev_size8192_mixed24_f64<>+0x8FC8(SB)/8, $3561
DATA bitrev_size8192_mixed24_f64<>+0x8FD0(SB)/8, $5609
DATA bitrev_size8192_mixed24_f64<>+0x8FD8(SB)/8, $7657
DATA bitrev_size8192_mixed24_f64<>+0x8FE0(SB)/8, $2025
DATA bitrev_size8192_mixed24_f64<>+0x8FE8(SB)/8, $4073
DATA bitrev_size8192_mixed24_f64<>+0x8FF0(SB)/8, $6121
DATA bitrev_size8192_mixed24_f64<>+0x8FF8(SB)/8, $8169
DATA bitrev_size8192_mixed24_f64<>+0x9000(SB)/8, $17
DATA bitrev_size8192_mixed24_f64<>+0x9008(SB)/8, $2065
DATA bitrev_size8192_mixed24_f64<>+0x9010(SB)/8, $4113
DATA bitrev_size8192_mixed24_f64<>+0x9018(SB)/8, $6161
DATA bitrev_size8192_mixed24_f64<>+0x9020(SB)/8, $529
DATA bitrev_size8192_mixed24_f64<>+0x9028(SB)/8, $2577
DATA bitrev_size8192_mixed24_f64<>+0x9030(SB)/8, $4625
DATA bitrev_size8192_mixed24_f64<>+0x9038(SB)/8, $6673
DATA bitrev_size8192_mixed24_f64<>+0x9040(SB)/8, $1041
DATA bitrev_size8192_mixed24_f64<>+0x9048(SB)/8, $3089
DATA bitrev_size8192_mixed24_f64<>+0x9050(SB)/8, $5137
DATA bitrev_size8192_mixed24_f64<>+0x9058(SB)/8, $7185
DATA bitrev_size8192_mixed24_f64<>+0x9060(SB)/8, $1553
DATA bitrev_size8192_mixed24_f64<>+0x9068(SB)/8, $3601
DATA bitrev_size8192_mixed24_f64<>+0x9070(SB)/8, $5649
DATA bitrev_size8192_mixed24_f64<>+0x9078(SB)/8, $7697
DATA bitrev_size8192_mixed24_f64<>+0x9080(SB)/8, $145
DATA bitrev_size8192_mixed24_f64<>+0x9088(SB)/8, $2193
DATA bitrev_size8192_mixed24_f64<>+0x9090(SB)/8, $4241
DATA bitrev_size8192_mixed24_f64<>+0x9098(SB)/8, $6289
DATA bitrev_size8192_mixed24_f64<>+0x90A0(SB)/8, $657
DATA bitrev_size8192_mixed24_f64<>+0x90A8(SB)/8, $2705
DATA bitrev_size8192_mixed24_f64<>+0x90B0(SB)/8, $4753
DATA bitrev_size8192_mixed24_f64<>+0x90B8(SB)/8, $6801
DATA bitrev_size8192_mixed24_f64<>+0x90C0(SB)/8, $1169
DATA bitrev_size8192_mixed24_f64<>+0x90C8(SB)/8, $3217
DATA bitrev_size8192_mixed24_f64<>+0x90D0(SB)/8, $5265
DATA bitrev_size8192_mixed24_f64<>+0x90D8(SB)/8, $7313
DATA bitrev_size8192_mixed24_f64<>+0x90E0(SB)/8, $1681
DATA bitrev_size8192_mixed24_f64<>+0x90E8(SB)/8, $3729
DATA bitrev_size8192_mixed24_f64<>+0x90F0(SB)/8, $5777
DATA bitrev_size8192_mixed24_f64<>+0x90F8(SB)/8, $7825
DATA bitrev_size8192_mixed24_f64<>+0x9100(SB)/8, $273
DATA bitrev_size8192_mixed24_f64<>+0x9108(SB)/8, $2321
DATA bitrev_size8192_mixed24_f64<>+0x9110(SB)/8, $4369
DATA bitrev_size8192_mixed24_f64<>+0x9118(SB)/8, $6417
DATA bitrev_size8192_mixed24_f64<>+0x9120(SB)/8, $785
DATA bitrev_size8192_mixed24_f64<>+0x9128(SB)/8, $2833
DATA bitrev_size8192_mixed24_f64<>+0x9130(SB)/8, $4881
DATA bitrev_size8192_mixed24_f64<>+0x9138(SB)/8, $6929
DATA bitrev_size8192_mixed24_f64<>+0x9140(SB)/8, $1297
DATA bitrev_size8192_mixed24_f64<>+0x9148(SB)/8, $3345
DATA bitrev_size8192_mixed24_f64<>+0x9150(SB)/8, $5393
DATA bitrev_size8192_mixed24_f64<>+0x9158(SB)/8, $7441
DATA bitrev_size8192_mixed24_f64<>+0x9160(SB)/8, $1809
DATA bitrev_size8192_mixed24_f64<>+0x9168(SB)/8, $3857
DATA bitrev_size8192_mixed24_f64<>+0x9170(SB)/8, $5905
DATA bitrev_size8192_mixed24_f64<>+0x9178(SB)/8, $7953
DATA bitrev_size8192_mixed24_f64<>+0x9180(SB)/8, $401
DATA bitrev_size8192_mixed24_f64<>+0x9188(SB)/8, $2449
DATA bitrev_size8192_mixed24_f64<>+0x9190(SB)/8, $4497
DATA bitrev_size8192_mixed24_f64<>+0x9198(SB)/8, $6545
DATA bitrev_size8192_mixed24_f64<>+0x91A0(SB)/8, $913
DATA bitrev_size8192_mixed24_f64<>+0x91A8(SB)/8, $2961
DATA bitrev_size8192_mixed24_f64<>+0x91B0(SB)/8, $5009
DATA bitrev_size8192_mixed24_f64<>+0x91B8(SB)/8, $7057
DATA bitrev_size8192_mixed24_f64<>+0x91C0(SB)/8, $1425
DATA bitrev_size8192_mixed24_f64<>+0x91C8(SB)/8, $3473
DATA bitrev_size8192_mixed24_f64<>+0x91D0(SB)/8, $5521
DATA bitrev_size8192_mixed24_f64<>+0x91D8(SB)/8, $7569
DATA bitrev_size8192_mixed24_f64<>+0x91E0(SB)/8, $1937
DATA bitrev_size8192_mixed24_f64<>+0x91E8(SB)/8, $3985
DATA bitrev_size8192_mixed24_f64<>+0x91F0(SB)/8, $6033
DATA bitrev_size8192_mixed24_f64<>+0x91F8(SB)/8, $8081
DATA bitrev_size8192_mixed24_f64<>+0x9200(SB)/8, $49
DATA bitrev_size8192_mixed24_f64<>+0x9208(SB)/8, $2097
DATA bitrev_size8192_mixed24_f64<>+0x9210(SB)/8, $4145
DATA bitrev_size8192_mixed24_f64<>+0x9218(SB)/8, $6193
DATA bitrev_size8192_mixed24_f64<>+0x9220(SB)/8, $561
DATA bitrev_size8192_mixed24_f64<>+0x9228(SB)/8, $2609
DATA bitrev_size8192_mixed24_f64<>+0x9230(SB)/8, $4657
DATA bitrev_size8192_mixed24_f64<>+0x9238(SB)/8, $6705
DATA bitrev_size8192_mixed24_f64<>+0x9240(SB)/8, $1073
DATA bitrev_size8192_mixed24_f64<>+0x9248(SB)/8, $3121
DATA bitrev_size8192_mixed24_f64<>+0x9250(SB)/8, $5169
DATA bitrev_size8192_mixed24_f64<>+0x9258(SB)/8, $7217
DATA bitrev_size8192_mixed24_f64<>+0x9260(SB)/8, $1585
DATA bitrev_size8192_mixed24_f64<>+0x9268(SB)/8, $3633
DATA bitrev_size8192_mixed24_f64<>+0x9270(SB)/8, $5681
DATA bitrev_size8192_mixed24_f64<>+0x9278(SB)/8, $7729
DATA bitrev_size8192_mixed24_f64<>+0x9280(SB)/8, $177
DATA bitrev_size8192_mixed24_f64<>+0x9288(SB)/8, $2225
DATA bitrev_size8192_mixed24_f64<>+0x9290(SB)/8, $4273
DATA bitrev_size8192_mixed24_f64<>+0x9298(SB)/8, $6321
DATA bitrev_size8192_mixed24_f64<>+0x92A0(SB)/8, $689
DATA bitrev_size8192_mixed24_f64<>+0x92A8(SB)/8, $2737
DATA bitrev_size8192_mixed24_f64<>+0x92B0(SB)/8, $4785
DATA bitrev_size8192_mixed24_f64<>+0x92B8(SB)/8, $6833
DATA bitrev_size8192_mixed24_f64<>+0x92C0(SB)/8, $1201
DATA bitrev_size8192_mixed24_f64<>+0x92C8(SB)/8, $3249
DATA bitrev_size8192_mixed24_f64<>+0x92D0(SB)/8, $5297
DATA bitrev_size8192_mixed24_f64<>+0x92D8(SB)/8, $7345
DATA bitrev_size8192_mixed24_f64<>+0x92E0(SB)/8, $1713
DATA bitrev_size8192_mixed24_f64<>+0x92E8(SB)/8, $3761
DATA bitrev_size8192_mixed24_f64<>+0x92F0(SB)/8, $5809
DATA bitrev_size8192_mixed24_f64<>+0x92F8(SB)/8, $7857
DATA bitrev_size8192_mixed24_f64<>+0x9300(SB)/8, $305
DATA bitrev_size8192_mixed24_f64<>+0x9308(SB)/8, $2353
DATA bitrev_size8192_mixed24_f64<>+0x9310(SB)/8, $4401
DATA bitrev_size8192_mixed24_f64<>+0x9318(SB)/8, $6449
DATA bitrev_size8192_mixed24_f64<>+0x9320(SB)/8, $817
DATA bitrev_size8192_mixed24_f64<>+0x9328(SB)/8, $2865
DATA bitrev_size8192_mixed24_f64<>+0x9330(SB)/8, $4913
DATA bitrev_size8192_mixed24_f64<>+0x9338(SB)/8, $6961
DATA bitrev_size8192_mixed24_f64<>+0x9340(SB)/8, $1329
DATA bitrev_size8192_mixed24_f64<>+0x9348(SB)/8, $3377
DATA bitrev_size8192_mixed24_f64<>+0x9350(SB)/8, $5425
DATA bitrev_size8192_mixed24_f64<>+0x9358(SB)/8, $7473
DATA bitrev_size8192_mixed24_f64<>+0x9360(SB)/8, $1841
DATA bitrev_size8192_mixed24_f64<>+0x9368(SB)/8, $3889
DATA bitrev_size8192_mixed24_f64<>+0x9370(SB)/8, $5937
DATA bitrev_size8192_mixed24_f64<>+0x9378(SB)/8, $7985
DATA bitrev_size8192_mixed24_f64<>+0x9380(SB)/8, $433
DATA bitrev_size8192_mixed24_f64<>+0x9388(SB)/8, $2481
DATA bitrev_size8192_mixed24_f64<>+0x9390(SB)/8, $4529
DATA bitrev_size8192_mixed24_f64<>+0x9398(SB)/8, $6577
DATA bitrev_size8192_mixed24_f64<>+0x93A0(SB)/8, $945
DATA bitrev_size8192_mixed24_f64<>+0x93A8(SB)/8, $2993
DATA bitrev_size8192_mixed24_f64<>+0x93B0(SB)/8, $5041
DATA bitrev_size8192_mixed24_f64<>+0x93B8(SB)/8, $7089
DATA bitrev_size8192_mixed24_f64<>+0x93C0(SB)/8, $1457
DATA bitrev_size8192_mixed24_f64<>+0x93C8(SB)/8, $3505
DATA bitrev_size8192_mixed24_f64<>+0x93D0(SB)/8, $5553
DATA bitrev_size8192_mixed24_f64<>+0x93D8(SB)/8, $7601
DATA bitrev_size8192_mixed24_f64<>+0x93E0(SB)/8, $1969
DATA bitrev_size8192_mixed24_f64<>+0x93E8(SB)/8, $4017
DATA bitrev_size8192_mixed24_f64<>+0x93F0(SB)/8, $6065
DATA bitrev_size8192_mixed24_f64<>+0x93F8(SB)/8, $8113
DATA bitrev_size8192_mixed24_f64<>+0x9400(SB)/8, $81
DATA bitrev_size8192_mixed24_f64<>+0x9408(SB)/8, $2129
DATA bitrev_size8192_mixed24_f64<>+0x9410(SB)/8, $4177
DATA bitrev_size8192_mixed24_f64<>+0x9418(SB)/8, $6225
DATA bitrev_size8192_mixed24_f64<>+0x9420(SB)/8, $593
DATA bitrev_size8192_mixed24_f64<>+0x9428(SB)/8, $2641
DATA bitrev_size8192_mixed24_f64<>+0x9430(SB)/8, $4689
DATA bitrev_size8192_mixed24_f64<>+0x9438(SB)/8, $6737
DATA bitrev_size8192_mixed24_f64<>+0x9440(SB)/8, $1105
DATA bitrev_size8192_mixed24_f64<>+0x9448(SB)/8, $3153
DATA bitrev_size8192_mixed24_f64<>+0x9450(SB)/8, $5201
DATA bitrev_size8192_mixed24_f64<>+0x9458(SB)/8, $7249
DATA bitrev_size8192_mixed24_f64<>+0x9460(SB)/8, $1617
DATA bitrev_size8192_mixed24_f64<>+0x9468(SB)/8, $3665
DATA bitrev_size8192_mixed24_f64<>+0x9470(SB)/8, $5713
DATA bitrev_size8192_mixed24_f64<>+0x9478(SB)/8, $7761
DATA bitrev_size8192_mixed24_f64<>+0x9480(SB)/8, $209
DATA bitrev_size8192_mixed24_f64<>+0x9488(SB)/8, $2257
DATA bitrev_size8192_mixed24_f64<>+0x9490(SB)/8, $4305
DATA bitrev_size8192_mixed24_f64<>+0x9498(SB)/8, $6353
DATA bitrev_size8192_mixed24_f64<>+0x94A0(SB)/8, $721
DATA bitrev_size8192_mixed24_f64<>+0x94A8(SB)/8, $2769
DATA bitrev_size8192_mixed24_f64<>+0x94B0(SB)/8, $4817
DATA bitrev_size8192_mixed24_f64<>+0x94B8(SB)/8, $6865
DATA bitrev_size8192_mixed24_f64<>+0x94C0(SB)/8, $1233
DATA bitrev_size8192_mixed24_f64<>+0x94C8(SB)/8, $3281
DATA bitrev_size8192_mixed24_f64<>+0x94D0(SB)/8, $5329
DATA bitrev_size8192_mixed24_f64<>+0x94D8(SB)/8, $7377
DATA bitrev_size8192_mixed24_f64<>+0x94E0(SB)/8, $1745
DATA bitrev_size8192_mixed24_f64<>+0x94E8(SB)/8, $3793
DATA bitrev_size8192_mixed24_f64<>+0x94F0(SB)/8, $5841
DATA bitrev_size8192_mixed24_f64<>+0x94F8(SB)/8, $7889
DATA bitrev_size8192_mixed24_f64<>+0x9500(SB)/8, $337
DATA bitrev_size8192_mixed24_f64<>+0x9508(SB)/8, $2385
DATA bitrev_size8192_mixed24_f64<>+0x9510(SB)/8, $4433
DATA bitrev_size8192_mixed24_f64<>+0x9518(SB)/8, $6481
DATA bitrev_size8192_mixed24_f64<>+0x9520(SB)/8, $849
DATA bitrev_size8192_mixed24_f64<>+0x9528(SB)/8, $2897
DATA bitrev_size8192_mixed24_f64<>+0x9530(SB)/8, $4945
DATA bitrev_size8192_mixed24_f64<>+0x9538(SB)/8, $6993
DATA bitrev_size8192_mixed24_f64<>+0x9540(SB)/8, $1361
DATA bitrev_size8192_mixed24_f64<>+0x9548(SB)/8, $3409
DATA bitrev_size8192_mixed24_f64<>+0x9550(SB)/8, $5457
DATA bitrev_size8192_mixed24_f64<>+0x9558(SB)/8, $7505
DATA bitrev_size8192_mixed24_f64<>+0x9560(SB)/8, $1873
DATA bitrev_size8192_mixed24_f64<>+0x9568(SB)/8, $3921
DATA bitrev_size8192_mixed24_f64<>+0x9570(SB)/8, $5969
DATA bitrev_size8192_mixed24_f64<>+0x9578(SB)/8, $8017
DATA bitrev_size8192_mixed24_f64<>+0x9580(SB)/8, $465
DATA bitrev_size8192_mixed24_f64<>+0x9588(SB)/8, $2513
DATA bitrev_size8192_mixed24_f64<>+0x9590(SB)/8, $4561
DATA bitrev_size8192_mixed24_f64<>+0x9598(SB)/8, $6609
DATA bitrev_size8192_mixed24_f64<>+0x95A0(SB)/8, $977
DATA bitrev_size8192_mixed24_f64<>+0x95A8(SB)/8, $3025
DATA bitrev_size8192_mixed24_f64<>+0x95B0(SB)/8, $5073
DATA bitrev_size8192_mixed24_f64<>+0x95B8(SB)/8, $7121
DATA bitrev_size8192_mixed24_f64<>+0x95C0(SB)/8, $1489
DATA bitrev_size8192_mixed24_f64<>+0x95C8(SB)/8, $3537
DATA bitrev_size8192_mixed24_f64<>+0x95D0(SB)/8, $5585
DATA bitrev_size8192_mixed24_f64<>+0x95D8(SB)/8, $7633
DATA bitrev_size8192_mixed24_f64<>+0x95E0(SB)/8, $2001
DATA bitrev_size8192_mixed24_f64<>+0x95E8(SB)/8, $4049
DATA bitrev_size8192_mixed24_f64<>+0x95F0(SB)/8, $6097
DATA bitrev_size8192_mixed24_f64<>+0x95F8(SB)/8, $8145
DATA bitrev_size8192_mixed24_f64<>+0x9600(SB)/8, $113
DATA bitrev_size8192_mixed24_f64<>+0x9608(SB)/8, $2161
DATA bitrev_size8192_mixed24_f64<>+0x9610(SB)/8, $4209
DATA bitrev_size8192_mixed24_f64<>+0x9618(SB)/8, $6257
DATA bitrev_size8192_mixed24_f64<>+0x9620(SB)/8, $625
DATA bitrev_size8192_mixed24_f64<>+0x9628(SB)/8, $2673
DATA bitrev_size8192_mixed24_f64<>+0x9630(SB)/8, $4721
DATA bitrev_size8192_mixed24_f64<>+0x9638(SB)/8, $6769
DATA bitrev_size8192_mixed24_f64<>+0x9640(SB)/8, $1137
DATA bitrev_size8192_mixed24_f64<>+0x9648(SB)/8, $3185
DATA bitrev_size8192_mixed24_f64<>+0x9650(SB)/8, $5233
DATA bitrev_size8192_mixed24_f64<>+0x9658(SB)/8, $7281
DATA bitrev_size8192_mixed24_f64<>+0x9660(SB)/8, $1649
DATA bitrev_size8192_mixed24_f64<>+0x9668(SB)/8, $3697
DATA bitrev_size8192_mixed24_f64<>+0x9670(SB)/8, $5745
DATA bitrev_size8192_mixed24_f64<>+0x9678(SB)/8, $7793
DATA bitrev_size8192_mixed24_f64<>+0x9680(SB)/8, $241
DATA bitrev_size8192_mixed24_f64<>+0x9688(SB)/8, $2289
DATA bitrev_size8192_mixed24_f64<>+0x9690(SB)/8, $4337
DATA bitrev_size8192_mixed24_f64<>+0x9698(SB)/8, $6385
DATA bitrev_size8192_mixed24_f64<>+0x96A0(SB)/8, $753
DATA bitrev_size8192_mixed24_f64<>+0x96A8(SB)/8, $2801
DATA bitrev_size8192_mixed24_f64<>+0x96B0(SB)/8, $4849
DATA bitrev_size8192_mixed24_f64<>+0x96B8(SB)/8, $6897
DATA bitrev_size8192_mixed24_f64<>+0x96C0(SB)/8, $1265
DATA bitrev_size8192_mixed24_f64<>+0x96C8(SB)/8, $3313
DATA bitrev_size8192_mixed24_f64<>+0x96D0(SB)/8, $5361
DATA bitrev_size8192_mixed24_f64<>+0x96D8(SB)/8, $7409
DATA bitrev_size8192_mixed24_f64<>+0x96E0(SB)/8, $1777
DATA bitrev_size8192_mixed24_f64<>+0x96E8(SB)/8, $3825
DATA bitrev_size8192_mixed24_f64<>+0x96F0(SB)/8, $5873
DATA bitrev_size8192_mixed24_f64<>+0x96F8(SB)/8, $7921
DATA bitrev_size8192_mixed24_f64<>+0x9700(SB)/8, $369
DATA bitrev_size8192_mixed24_f64<>+0x9708(SB)/8, $2417
DATA bitrev_size8192_mixed24_f64<>+0x9710(SB)/8, $4465
DATA bitrev_size8192_mixed24_f64<>+0x9718(SB)/8, $6513
DATA bitrev_size8192_mixed24_f64<>+0x9720(SB)/8, $881
DATA bitrev_size8192_mixed24_f64<>+0x9728(SB)/8, $2929
DATA bitrev_size8192_mixed24_f64<>+0x9730(SB)/8, $4977
DATA bitrev_size8192_mixed24_f64<>+0x9738(SB)/8, $7025
DATA bitrev_size8192_mixed24_f64<>+0x9740(SB)/8, $1393
DATA bitrev_size8192_mixed24_f64<>+0x9748(SB)/8, $3441
DATA bitrev_size8192_mixed24_f64<>+0x9750(SB)/8, $5489
DATA bitrev_size8192_mixed24_f64<>+0x9758(SB)/8, $7537
DATA bitrev_size8192_mixed24_f64<>+0x9760(SB)/8, $1905
DATA bitrev_size8192_mixed24_f64<>+0x9768(SB)/8, $3953
DATA bitrev_size8192_mixed24_f64<>+0x9770(SB)/8, $6001
DATA bitrev_size8192_mixed24_f64<>+0x9778(SB)/8, $8049
DATA bitrev_size8192_mixed24_f64<>+0x9780(SB)/8, $497
DATA bitrev_size8192_mixed24_f64<>+0x9788(SB)/8, $2545
DATA bitrev_size8192_mixed24_f64<>+0x9790(SB)/8, $4593
DATA bitrev_size8192_mixed24_f64<>+0x9798(SB)/8, $6641
DATA bitrev_size8192_mixed24_f64<>+0x97A0(SB)/8, $1009
DATA bitrev_size8192_mixed24_f64<>+0x97A8(SB)/8, $3057
DATA bitrev_size8192_mixed24_f64<>+0x97B0(SB)/8, $5105
DATA bitrev_size8192_mixed24_f64<>+0x97B8(SB)/8, $7153
DATA bitrev_size8192_mixed24_f64<>+0x97C0(SB)/8, $1521
DATA bitrev_size8192_mixed24_f64<>+0x97C8(SB)/8, $3569
DATA bitrev_size8192_mixed24_f64<>+0x97D0(SB)/8, $5617
DATA bitrev_size8192_mixed24_f64<>+0x97D8(SB)/8, $7665
DATA bitrev_size8192_mixed24_f64<>+0x97E0(SB)/8, $2033
DATA bitrev_size8192_mixed24_f64<>+0x97E8(SB)/8, $4081
DATA bitrev_size8192_mixed24_f64<>+0x97F0(SB)/8, $6129
DATA bitrev_size8192_mixed24_f64<>+0x97F8(SB)/8, $8177
DATA bitrev_size8192_mixed24_f64<>+0x9800(SB)/8, $25
DATA bitrev_size8192_mixed24_f64<>+0x9808(SB)/8, $2073
DATA bitrev_size8192_mixed24_f64<>+0x9810(SB)/8, $4121
DATA bitrev_size8192_mixed24_f64<>+0x9818(SB)/8, $6169
DATA bitrev_size8192_mixed24_f64<>+0x9820(SB)/8, $537
DATA bitrev_size8192_mixed24_f64<>+0x9828(SB)/8, $2585
DATA bitrev_size8192_mixed24_f64<>+0x9830(SB)/8, $4633
DATA bitrev_size8192_mixed24_f64<>+0x9838(SB)/8, $6681
DATA bitrev_size8192_mixed24_f64<>+0x9840(SB)/8, $1049
DATA bitrev_size8192_mixed24_f64<>+0x9848(SB)/8, $3097
DATA bitrev_size8192_mixed24_f64<>+0x9850(SB)/8, $5145
DATA bitrev_size8192_mixed24_f64<>+0x9858(SB)/8, $7193
DATA bitrev_size8192_mixed24_f64<>+0x9860(SB)/8, $1561
DATA bitrev_size8192_mixed24_f64<>+0x9868(SB)/8, $3609
DATA bitrev_size8192_mixed24_f64<>+0x9870(SB)/8, $5657
DATA bitrev_size8192_mixed24_f64<>+0x9878(SB)/8, $7705
DATA bitrev_size8192_mixed24_f64<>+0x9880(SB)/8, $153
DATA bitrev_size8192_mixed24_f64<>+0x9888(SB)/8, $2201
DATA bitrev_size8192_mixed24_f64<>+0x9890(SB)/8, $4249
DATA bitrev_size8192_mixed24_f64<>+0x9898(SB)/8, $6297
DATA bitrev_size8192_mixed24_f64<>+0x98A0(SB)/8, $665
DATA bitrev_size8192_mixed24_f64<>+0x98A8(SB)/8, $2713
DATA bitrev_size8192_mixed24_f64<>+0x98B0(SB)/8, $4761
DATA bitrev_size8192_mixed24_f64<>+0x98B8(SB)/8, $6809
DATA bitrev_size8192_mixed24_f64<>+0x98C0(SB)/8, $1177
DATA bitrev_size8192_mixed24_f64<>+0x98C8(SB)/8, $3225
DATA bitrev_size8192_mixed24_f64<>+0x98D0(SB)/8, $5273
DATA bitrev_size8192_mixed24_f64<>+0x98D8(SB)/8, $7321
DATA bitrev_size8192_mixed24_f64<>+0x98E0(SB)/8, $1689
DATA bitrev_size8192_mixed24_f64<>+0x98E8(SB)/8, $3737
DATA bitrev_size8192_mixed24_f64<>+0x98F0(SB)/8, $5785
DATA bitrev_size8192_mixed24_f64<>+0x98F8(SB)/8, $7833
DATA bitrev_size8192_mixed24_f64<>+0x9900(SB)/8, $281
DATA bitrev_size8192_mixed24_f64<>+0x9908(SB)/8, $2329
DATA bitrev_size8192_mixed24_f64<>+0x9910(SB)/8, $4377
DATA bitrev_size8192_mixed24_f64<>+0x9918(SB)/8, $6425
DATA bitrev_size8192_mixed24_f64<>+0x9920(SB)/8, $793
DATA bitrev_size8192_mixed24_f64<>+0x9928(SB)/8, $2841
DATA bitrev_size8192_mixed24_f64<>+0x9930(SB)/8, $4889
DATA bitrev_size8192_mixed24_f64<>+0x9938(SB)/8, $6937
DATA bitrev_size8192_mixed24_f64<>+0x9940(SB)/8, $1305
DATA bitrev_size8192_mixed24_f64<>+0x9948(SB)/8, $3353
DATA bitrev_size8192_mixed24_f64<>+0x9950(SB)/8, $5401
DATA bitrev_size8192_mixed24_f64<>+0x9958(SB)/8, $7449
DATA bitrev_size8192_mixed24_f64<>+0x9960(SB)/8, $1817
DATA bitrev_size8192_mixed24_f64<>+0x9968(SB)/8, $3865
DATA bitrev_size8192_mixed24_f64<>+0x9970(SB)/8, $5913
DATA bitrev_size8192_mixed24_f64<>+0x9978(SB)/8, $7961
DATA bitrev_size8192_mixed24_f64<>+0x9980(SB)/8, $409
DATA bitrev_size8192_mixed24_f64<>+0x9988(SB)/8, $2457
DATA bitrev_size8192_mixed24_f64<>+0x9990(SB)/8, $4505
DATA bitrev_size8192_mixed24_f64<>+0x9998(SB)/8, $6553
DATA bitrev_size8192_mixed24_f64<>+0x99A0(SB)/8, $921
DATA bitrev_size8192_mixed24_f64<>+0x99A8(SB)/8, $2969
DATA bitrev_size8192_mixed24_f64<>+0x99B0(SB)/8, $5017
DATA bitrev_size8192_mixed24_f64<>+0x99B8(SB)/8, $7065
DATA bitrev_size8192_mixed24_f64<>+0x99C0(SB)/8, $1433
DATA bitrev_size8192_mixed24_f64<>+0x99C8(SB)/8, $3481
DATA bitrev_size8192_mixed24_f64<>+0x99D0(SB)/8, $5529
DATA bitrev_size8192_mixed24_f64<>+0x99D8(SB)/8, $7577
DATA bitrev_size8192_mixed24_f64<>+0x99E0(SB)/8, $1945
DATA bitrev_size8192_mixed24_f64<>+0x99E8(SB)/8, $3993
DATA bitrev_size8192_mixed24_f64<>+0x99F0(SB)/8, $6041
DATA bitrev_size8192_mixed24_f64<>+0x99F8(SB)/8, $8089
DATA bitrev_size8192_mixed24_f64<>+0x9A00(SB)/8, $57
DATA bitrev_size8192_mixed24_f64<>+0x9A08(SB)/8, $2105
DATA bitrev_size8192_mixed24_f64<>+0x9A10(SB)/8, $4153
DATA bitrev_size8192_mixed24_f64<>+0x9A18(SB)/8, $6201
DATA bitrev_size8192_mixed24_f64<>+0x9A20(SB)/8, $569
DATA bitrev_size8192_mixed24_f64<>+0x9A28(SB)/8, $2617
DATA bitrev_size8192_mixed24_f64<>+0x9A30(SB)/8, $4665
DATA bitrev_size8192_mixed24_f64<>+0x9A38(SB)/8, $6713
DATA bitrev_size8192_mixed24_f64<>+0x9A40(SB)/8, $1081
DATA bitrev_size8192_mixed24_f64<>+0x9A48(SB)/8, $3129
DATA bitrev_size8192_mixed24_f64<>+0x9A50(SB)/8, $5177
DATA bitrev_size8192_mixed24_f64<>+0x9A58(SB)/8, $7225
DATA bitrev_size8192_mixed24_f64<>+0x9A60(SB)/8, $1593
DATA bitrev_size8192_mixed24_f64<>+0x9A68(SB)/8, $3641
DATA bitrev_size8192_mixed24_f64<>+0x9A70(SB)/8, $5689
DATA bitrev_size8192_mixed24_f64<>+0x9A78(SB)/8, $7737
DATA bitrev_size8192_mixed24_f64<>+0x9A80(SB)/8, $185
DATA bitrev_size8192_mixed24_f64<>+0x9A88(SB)/8, $2233
DATA bitrev_size8192_mixed24_f64<>+0x9A90(SB)/8, $4281
DATA bitrev_size8192_mixed24_f64<>+0x9A98(SB)/8, $6329
DATA bitrev_size8192_mixed24_f64<>+0x9AA0(SB)/8, $697
DATA bitrev_size8192_mixed24_f64<>+0x9AA8(SB)/8, $2745
DATA bitrev_size8192_mixed24_f64<>+0x9AB0(SB)/8, $4793
DATA bitrev_size8192_mixed24_f64<>+0x9AB8(SB)/8, $6841
DATA bitrev_size8192_mixed24_f64<>+0x9AC0(SB)/8, $1209
DATA bitrev_size8192_mixed24_f64<>+0x9AC8(SB)/8, $3257
DATA bitrev_size8192_mixed24_f64<>+0x9AD0(SB)/8, $5305
DATA bitrev_size8192_mixed24_f64<>+0x9AD8(SB)/8, $7353
DATA bitrev_size8192_mixed24_f64<>+0x9AE0(SB)/8, $1721
DATA bitrev_size8192_mixed24_f64<>+0x9AE8(SB)/8, $3769
DATA bitrev_size8192_mixed24_f64<>+0x9AF0(SB)/8, $5817
DATA bitrev_size8192_mixed24_f64<>+0x9AF8(SB)/8, $7865
DATA bitrev_size8192_mixed24_f64<>+0x9B00(SB)/8, $313
DATA bitrev_size8192_mixed24_f64<>+0x9B08(SB)/8, $2361
DATA bitrev_size8192_mixed24_f64<>+0x9B10(SB)/8, $4409
DATA bitrev_size8192_mixed24_f64<>+0x9B18(SB)/8, $6457
DATA bitrev_size8192_mixed24_f64<>+0x9B20(SB)/8, $825
DATA bitrev_size8192_mixed24_f64<>+0x9B28(SB)/8, $2873
DATA bitrev_size8192_mixed24_f64<>+0x9B30(SB)/8, $4921
DATA bitrev_size8192_mixed24_f64<>+0x9B38(SB)/8, $6969
DATA bitrev_size8192_mixed24_f64<>+0x9B40(SB)/8, $1337
DATA bitrev_size8192_mixed24_f64<>+0x9B48(SB)/8, $3385
DATA bitrev_size8192_mixed24_f64<>+0x9B50(SB)/8, $5433
DATA bitrev_size8192_mixed24_f64<>+0x9B58(SB)/8, $7481
DATA bitrev_size8192_mixed24_f64<>+0x9B60(SB)/8, $1849
DATA bitrev_size8192_mixed24_f64<>+0x9B68(SB)/8, $3897
DATA bitrev_size8192_mixed24_f64<>+0x9B70(SB)/8, $5945
DATA bitrev_size8192_mixed24_f64<>+0x9B78(SB)/8, $7993
DATA bitrev_size8192_mixed24_f64<>+0x9B80(SB)/8, $441
DATA bitrev_size8192_mixed24_f64<>+0x9B88(SB)/8, $2489
DATA bitrev_size8192_mixed24_f64<>+0x9B90(SB)/8, $4537
DATA bitrev_size8192_mixed24_f64<>+0x9B98(SB)/8, $6585
DATA bitrev_size8192_mixed24_f64<>+0x9BA0(SB)/8, $953
DATA bitrev_size8192_mixed24_f64<>+0x9BA8(SB)/8, $3001
DATA bitrev_size8192_mixed24_f64<>+0x9BB0(SB)/8, $5049
DATA bitrev_size8192_mixed24_f64<>+0x9BB8(SB)/8, $7097
DATA bitrev_size8192_mixed24_f64<>+0x9BC0(SB)/8, $1465
DATA bitrev_size8192_mixed24_f64<>+0x9BC8(SB)/8, $3513
DATA bitrev_size8192_mixed24_f64<>+0x9BD0(SB)/8, $5561
DATA bitrev_size8192_mixed24_f64<>+0x9BD8(SB)/8, $7609
DATA bitrev_size8192_mixed24_f64<>+0x9BE0(SB)/8, $1977
DATA bitrev_size8192_mixed24_f64<>+0x9BE8(SB)/8, $4025
DATA bitrev_size8192_mixed24_f64<>+0x9BF0(SB)/8, $6073
DATA bitrev_size8192_mixed24_f64<>+0x9BF8(SB)/8, $8121
DATA bitrev_size8192_mixed24_f64<>+0x9C00(SB)/8, $89
DATA bitrev_size8192_mixed24_f64<>+0x9C08(SB)/8, $2137
DATA bitrev_size8192_mixed24_f64<>+0x9C10(SB)/8, $4185
DATA bitrev_size8192_mixed24_f64<>+0x9C18(SB)/8, $6233
DATA bitrev_size8192_mixed24_f64<>+0x9C20(SB)/8, $601
DATA bitrev_size8192_mixed24_f64<>+0x9C28(SB)/8, $2649
DATA bitrev_size8192_mixed24_f64<>+0x9C30(SB)/8, $4697
DATA bitrev_size8192_mixed24_f64<>+0x9C38(SB)/8, $6745
DATA bitrev_size8192_mixed24_f64<>+0x9C40(SB)/8, $1113
DATA bitrev_size8192_mixed24_f64<>+0x9C48(SB)/8, $3161
DATA bitrev_size8192_mixed24_f64<>+0x9C50(SB)/8, $5209
DATA bitrev_size8192_mixed24_f64<>+0x9C58(SB)/8, $7257
DATA bitrev_size8192_mixed24_f64<>+0x9C60(SB)/8, $1625
DATA bitrev_size8192_mixed24_f64<>+0x9C68(SB)/8, $3673
DATA bitrev_size8192_mixed24_f64<>+0x9C70(SB)/8, $5721
DATA bitrev_size8192_mixed24_f64<>+0x9C78(SB)/8, $7769
DATA bitrev_size8192_mixed24_f64<>+0x9C80(SB)/8, $217
DATA bitrev_size8192_mixed24_f64<>+0x9C88(SB)/8, $2265
DATA bitrev_size8192_mixed24_f64<>+0x9C90(SB)/8, $4313
DATA bitrev_size8192_mixed24_f64<>+0x9C98(SB)/8, $6361
DATA bitrev_size8192_mixed24_f64<>+0x9CA0(SB)/8, $729
DATA bitrev_size8192_mixed24_f64<>+0x9CA8(SB)/8, $2777
DATA bitrev_size8192_mixed24_f64<>+0x9CB0(SB)/8, $4825
DATA bitrev_size8192_mixed24_f64<>+0x9CB8(SB)/8, $6873
DATA bitrev_size8192_mixed24_f64<>+0x9CC0(SB)/8, $1241
DATA bitrev_size8192_mixed24_f64<>+0x9CC8(SB)/8, $3289
DATA bitrev_size8192_mixed24_f64<>+0x9CD0(SB)/8, $5337
DATA bitrev_size8192_mixed24_f64<>+0x9CD8(SB)/8, $7385
DATA bitrev_size8192_mixed24_f64<>+0x9CE0(SB)/8, $1753
DATA bitrev_size8192_mixed24_f64<>+0x9CE8(SB)/8, $3801
DATA bitrev_size8192_mixed24_f64<>+0x9CF0(SB)/8, $5849
DATA bitrev_size8192_mixed24_f64<>+0x9CF8(SB)/8, $7897
DATA bitrev_size8192_mixed24_f64<>+0x9D00(SB)/8, $345
DATA bitrev_size8192_mixed24_f64<>+0x9D08(SB)/8, $2393
DATA bitrev_size8192_mixed24_f64<>+0x9D10(SB)/8, $4441
DATA bitrev_size8192_mixed24_f64<>+0x9D18(SB)/8, $6489
DATA bitrev_size8192_mixed24_f64<>+0x9D20(SB)/8, $857
DATA bitrev_size8192_mixed24_f64<>+0x9D28(SB)/8, $2905
DATA bitrev_size8192_mixed24_f64<>+0x9D30(SB)/8, $4953
DATA bitrev_size8192_mixed24_f64<>+0x9D38(SB)/8, $7001
DATA bitrev_size8192_mixed24_f64<>+0x9D40(SB)/8, $1369
DATA bitrev_size8192_mixed24_f64<>+0x9D48(SB)/8, $3417
DATA bitrev_size8192_mixed24_f64<>+0x9D50(SB)/8, $5465
DATA bitrev_size8192_mixed24_f64<>+0x9D58(SB)/8, $7513
DATA bitrev_size8192_mixed24_f64<>+0x9D60(SB)/8, $1881
DATA bitrev_size8192_mixed24_f64<>+0x9D68(SB)/8, $3929
DATA bitrev_size8192_mixed24_f64<>+0x9D70(SB)/8, $5977
DATA bitrev_size8192_mixed24_f64<>+0x9D78(SB)/8, $8025
DATA bitrev_size8192_mixed24_f64<>+0x9D80(SB)/8, $473
DATA bitrev_size8192_mixed24_f64<>+0x9D88(SB)/8, $2521
DATA bitrev_size8192_mixed24_f64<>+0x9D90(SB)/8, $4569
DATA bitrev_size8192_mixed24_f64<>+0x9D98(SB)/8, $6617
DATA bitrev_size8192_mixed24_f64<>+0x9DA0(SB)/8, $985
DATA bitrev_size8192_mixed24_f64<>+0x9DA8(SB)/8, $3033
DATA bitrev_size8192_mixed24_f64<>+0x9DB0(SB)/8, $5081
DATA bitrev_size8192_mixed24_f64<>+0x9DB8(SB)/8, $7129
DATA bitrev_size8192_mixed24_f64<>+0x9DC0(SB)/8, $1497
DATA bitrev_size8192_mixed24_f64<>+0x9DC8(SB)/8, $3545
DATA bitrev_size8192_mixed24_f64<>+0x9DD0(SB)/8, $5593
DATA bitrev_size8192_mixed24_f64<>+0x9DD8(SB)/8, $7641
DATA bitrev_size8192_mixed24_f64<>+0x9DE0(SB)/8, $2009
DATA bitrev_size8192_mixed24_f64<>+0x9DE8(SB)/8, $4057
DATA bitrev_size8192_mixed24_f64<>+0x9DF0(SB)/8, $6105
DATA bitrev_size8192_mixed24_f64<>+0x9DF8(SB)/8, $8153
DATA bitrev_size8192_mixed24_f64<>+0x9E00(SB)/8, $121
DATA bitrev_size8192_mixed24_f64<>+0x9E08(SB)/8, $2169
DATA bitrev_size8192_mixed24_f64<>+0x9E10(SB)/8, $4217
DATA bitrev_size8192_mixed24_f64<>+0x9E18(SB)/8, $6265
DATA bitrev_size8192_mixed24_f64<>+0x9E20(SB)/8, $633
DATA bitrev_size8192_mixed24_f64<>+0x9E28(SB)/8, $2681
DATA bitrev_size8192_mixed24_f64<>+0x9E30(SB)/8, $4729
DATA bitrev_size8192_mixed24_f64<>+0x9E38(SB)/8, $6777
DATA bitrev_size8192_mixed24_f64<>+0x9E40(SB)/8, $1145
DATA bitrev_size8192_mixed24_f64<>+0x9E48(SB)/8, $3193
DATA bitrev_size8192_mixed24_f64<>+0x9E50(SB)/8, $5241
DATA bitrev_size8192_mixed24_f64<>+0x9E58(SB)/8, $7289
DATA bitrev_size8192_mixed24_f64<>+0x9E60(SB)/8, $1657
DATA bitrev_size8192_mixed24_f64<>+0x9E68(SB)/8, $3705
DATA bitrev_size8192_mixed24_f64<>+0x9E70(SB)/8, $5753
DATA bitrev_size8192_mixed24_f64<>+0x9E78(SB)/8, $7801
DATA bitrev_size8192_mixed24_f64<>+0x9E80(SB)/8, $249
DATA bitrev_size8192_mixed24_f64<>+0x9E88(SB)/8, $2297
DATA bitrev_size8192_mixed24_f64<>+0x9E90(SB)/8, $4345
DATA bitrev_size8192_mixed24_f64<>+0x9E98(SB)/8, $6393
DATA bitrev_size8192_mixed24_f64<>+0x9EA0(SB)/8, $761
DATA bitrev_size8192_mixed24_f64<>+0x9EA8(SB)/8, $2809
DATA bitrev_size8192_mixed24_f64<>+0x9EB0(SB)/8, $4857
DATA bitrev_size8192_mixed24_f64<>+0x9EB8(SB)/8, $6905
DATA bitrev_size8192_mixed24_f64<>+0x9EC0(SB)/8, $1273
DATA bitrev_size8192_mixed24_f64<>+0x9EC8(SB)/8, $3321
DATA bitrev_size8192_mixed24_f64<>+0x9ED0(SB)/8, $5369
DATA bitrev_size8192_mixed24_f64<>+0x9ED8(SB)/8, $7417
DATA bitrev_size8192_mixed24_f64<>+0x9EE0(SB)/8, $1785
DATA bitrev_size8192_mixed24_f64<>+0x9EE8(SB)/8, $3833
DATA bitrev_size8192_mixed24_f64<>+0x9EF0(SB)/8, $5881
DATA bitrev_size8192_mixed24_f64<>+0x9EF8(SB)/8, $7929
DATA bitrev_size8192_mixed24_f64<>+0x9F00(SB)/8, $377
DATA bitrev_size8192_mixed24_f64<>+0x9F08(SB)/8, $2425
DATA bitrev_size8192_mixed24_f64<>+0x9F10(SB)/8, $4473
DATA bitrev_size8192_mixed24_f64<>+0x9F18(SB)/8, $6521
DATA bitrev_size8192_mixed24_f64<>+0x9F20(SB)/8, $889
DATA bitrev_size8192_mixed24_f64<>+0x9F28(SB)/8, $2937
DATA bitrev_size8192_mixed24_f64<>+0x9F30(SB)/8, $4985
DATA bitrev_size8192_mixed24_f64<>+0x9F38(SB)/8, $7033
DATA bitrev_size8192_mixed24_f64<>+0x9F40(SB)/8, $1401
DATA bitrev_size8192_mixed24_f64<>+0x9F48(SB)/8, $3449
DATA bitrev_size8192_mixed24_f64<>+0x9F50(SB)/8, $5497
DATA bitrev_size8192_mixed24_f64<>+0x9F58(SB)/8, $7545
DATA bitrev_size8192_mixed24_f64<>+0x9F60(SB)/8, $1913
DATA bitrev_size8192_mixed24_f64<>+0x9F68(SB)/8, $3961
DATA bitrev_size8192_mixed24_f64<>+0x9F70(SB)/8, $6009
DATA bitrev_size8192_mixed24_f64<>+0x9F78(SB)/8, $8057
DATA bitrev_size8192_mixed24_f64<>+0x9F80(SB)/8, $505
DATA bitrev_size8192_mixed24_f64<>+0x9F88(SB)/8, $2553
DATA bitrev_size8192_mixed24_f64<>+0x9F90(SB)/8, $4601
DATA bitrev_size8192_mixed24_f64<>+0x9F98(SB)/8, $6649
DATA bitrev_size8192_mixed24_f64<>+0x9FA0(SB)/8, $1017
DATA bitrev_size8192_mixed24_f64<>+0x9FA8(SB)/8, $3065
DATA bitrev_size8192_mixed24_f64<>+0x9FB0(SB)/8, $5113
DATA bitrev_size8192_mixed24_f64<>+0x9FB8(SB)/8, $7161
DATA bitrev_size8192_mixed24_f64<>+0x9FC0(SB)/8, $1529
DATA bitrev_size8192_mixed24_f64<>+0x9FC8(SB)/8, $3577
DATA bitrev_size8192_mixed24_f64<>+0x9FD0(SB)/8, $5625
DATA bitrev_size8192_mixed24_f64<>+0x9FD8(SB)/8, $7673
DATA bitrev_size8192_mixed24_f64<>+0x9FE0(SB)/8, $2041
DATA bitrev_size8192_mixed24_f64<>+0x9FE8(SB)/8, $4089
DATA bitrev_size8192_mixed24_f64<>+0x9FF0(SB)/8, $6137
DATA bitrev_size8192_mixed24_f64<>+0x9FF8(SB)/8, $8185
DATA bitrev_size8192_mixed24_f64<>+0xA000(SB)/8, $3
DATA bitrev_size8192_mixed24_f64<>+0xA008(SB)/8, $2051
DATA bitrev_size8192_mixed24_f64<>+0xA010(SB)/8, $4099
DATA bitrev_size8192_mixed24_f64<>+0xA018(SB)/8, $6147
DATA bitrev_size8192_mixed24_f64<>+0xA020(SB)/8, $515
DATA bitrev_size8192_mixed24_f64<>+0xA028(SB)/8, $2563
DATA bitrev_size8192_mixed24_f64<>+0xA030(SB)/8, $4611
DATA bitrev_size8192_mixed24_f64<>+0xA038(SB)/8, $6659
DATA bitrev_size8192_mixed24_f64<>+0xA040(SB)/8, $1027
DATA bitrev_size8192_mixed24_f64<>+0xA048(SB)/8, $3075
DATA bitrev_size8192_mixed24_f64<>+0xA050(SB)/8, $5123
DATA bitrev_size8192_mixed24_f64<>+0xA058(SB)/8, $7171
DATA bitrev_size8192_mixed24_f64<>+0xA060(SB)/8, $1539
DATA bitrev_size8192_mixed24_f64<>+0xA068(SB)/8, $3587
DATA bitrev_size8192_mixed24_f64<>+0xA070(SB)/8, $5635
DATA bitrev_size8192_mixed24_f64<>+0xA078(SB)/8, $7683
DATA bitrev_size8192_mixed24_f64<>+0xA080(SB)/8, $131
DATA bitrev_size8192_mixed24_f64<>+0xA088(SB)/8, $2179
DATA bitrev_size8192_mixed24_f64<>+0xA090(SB)/8, $4227
DATA bitrev_size8192_mixed24_f64<>+0xA098(SB)/8, $6275
DATA bitrev_size8192_mixed24_f64<>+0xA0A0(SB)/8, $643
DATA bitrev_size8192_mixed24_f64<>+0xA0A8(SB)/8, $2691
DATA bitrev_size8192_mixed24_f64<>+0xA0B0(SB)/8, $4739
DATA bitrev_size8192_mixed24_f64<>+0xA0B8(SB)/8, $6787
DATA bitrev_size8192_mixed24_f64<>+0xA0C0(SB)/8, $1155
DATA bitrev_size8192_mixed24_f64<>+0xA0C8(SB)/8, $3203
DATA bitrev_size8192_mixed24_f64<>+0xA0D0(SB)/8, $5251
DATA bitrev_size8192_mixed24_f64<>+0xA0D8(SB)/8, $7299
DATA bitrev_size8192_mixed24_f64<>+0xA0E0(SB)/8, $1667
DATA bitrev_size8192_mixed24_f64<>+0xA0E8(SB)/8, $3715
DATA bitrev_size8192_mixed24_f64<>+0xA0F0(SB)/8, $5763
DATA bitrev_size8192_mixed24_f64<>+0xA0F8(SB)/8, $7811
DATA bitrev_size8192_mixed24_f64<>+0xA100(SB)/8, $259
DATA bitrev_size8192_mixed24_f64<>+0xA108(SB)/8, $2307
DATA bitrev_size8192_mixed24_f64<>+0xA110(SB)/8, $4355
DATA bitrev_size8192_mixed24_f64<>+0xA118(SB)/8, $6403
DATA bitrev_size8192_mixed24_f64<>+0xA120(SB)/8, $771
DATA bitrev_size8192_mixed24_f64<>+0xA128(SB)/8, $2819
DATA bitrev_size8192_mixed24_f64<>+0xA130(SB)/8, $4867
DATA bitrev_size8192_mixed24_f64<>+0xA138(SB)/8, $6915
DATA bitrev_size8192_mixed24_f64<>+0xA140(SB)/8, $1283
DATA bitrev_size8192_mixed24_f64<>+0xA148(SB)/8, $3331
DATA bitrev_size8192_mixed24_f64<>+0xA150(SB)/8, $5379
DATA bitrev_size8192_mixed24_f64<>+0xA158(SB)/8, $7427
DATA bitrev_size8192_mixed24_f64<>+0xA160(SB)/8, $1795
DATA bitrev_size8192_mixed24_f64<>+0xA168(SB)/8, $3843
DATA bitrev_size8192_mixed24_f64<>+0xA170(SB)/8, $5891
DATA bitrev_size8192_mixed24_f64<>+0xA178(SB)/8, $7939
DATA bitrev_size8192_mixed24_f64<>+0xA180(SB)/8, $387
DATA bitrev_size8192_mixed24_f64<>+0xA188(SB)/8, $2435
DATA bitrev_size8192_mixed24_f64<>+0xA190(SB)/8, $4483
DATA bitrev_size8192_mixed24_f64<>+0xA198(SB)/8, $6531
DATA bitrev_size8192_mixed24_f64<>+0xA1A0(SB)/8, $899
DATA bitrev_size8192_mixed24_f64<>+0xA1A8(SB)/8, $2947
DATA bitrev_size8192_mixed24_f64<>+0xA1B0(SB)/8, $4995
DATA bitrev_size8192_mixed24_f64<>+0xA1B8(SB)/8, $7043
DATA bitrev_size8192_mixed24_f64<>+0xA1C0(SB)/8, $1411
DATA bitrev_size8192_mixed24_f64<>+0xA1C8(SB)/8, $3459
DATA bitrev_size8192_mixed24_f64<>+0xA1D0(SB)/8, $5507
DATA bitrev_size8192_mixed24_f64<>+0xA1D8(SB)/8, $7555
DATA bitrev_size8192_mixed24_f64<>+0xA1E0(SB)/8, $1923
DATA bitrev_size8192_mixed24_f64<>+0xA1E8(SB)/8, $3971
DATA bitrev_size8192_mixed24_f64<>+0xA1F0(SB)/8, $6019
DATA bitrev_size8192_mixed24_f64<>+0xA1F8(SB)/8, $8067
DATA bitrev_size8192_mixed24_f64<>+0xA200(SB)/8, $35
DATA bitrev_size8192_mixed24_f64<>+0xA208(SB)/8, $2083
DATA bitrev_size8192_mixed24_f64<>+0xA210(SB)/8, $4131
DATA bitrev_size8192_mixed24_f64<>+0xA218(SB)/8, $6179
DATA bitrev_size8192_mixed24_f64<>+0xA220(SB)/8, $547
DATA bitrev_size8192_mixed24_f64<>+0xA228(SB)/8, $2595
DATA bitrev_size8192_mixed24_f64<>+0xA230(SB)/8, $4643
DATA bitrev_size8192_mixed24_f64<>+0xA238(SB)/8, $6691
DATA bitrev_size8192_mixed24_f64<>+0xA240(SB)/8, $1059
DATA bitrev_size8192_mixed24_f64<>+0xA248(SB)/8, $3107
DATA bitrev_size8192_mixed24_f64<>+0xA250(SB)/8, $5155
DATA bitrev_size8192_mixed24_f64<>+0xA258(SB)/8, $7203
DATA bitrev_size8192_mixed24_f64<>+0xA260(SB)/8, $1571
DATA bitrev_size8192_mixed24_f64<>+0xA268(SB)/8, $3619
DATA bitrev_size8192_mixed24_f64<>+0xA270(SB)/8, $5667
DATA bitrev_size8192_mixed24_f64<>+0xA278(SB)/8, $7715
DATA bitrev_size8192_mixed24_f64<>+0xA280(SB)/8, $163
DATA bitrev_size8192_mixed24_f64<>+0xA288(SB)/8, $2211
DATA bitrev_size8192_mixed24_f64<>+0xA290(SB)/8, $4259
DATA bitrev_size8192_mixed24_f64<>+0xA298(SB)/8, $6307
DATA bitrev_size8192_mixed24_f64<>+0xA2A0(SB)/8, $675
DATA bitrev_size8192_mixed24_f64<>+0xA2A8(SB)/8, $2723
DATA bitrev_size8192_mixed24_f64<>+0xA2B0(SB)/8, $4771
DATA bitrev_size8192_mixed24_f64<>+0xA2B8(SB)/8, $6819
DATA bitrev_size8192_mixed24_f64<>+0xA2C0(SB)/8, $1187
DATA bitrev_size8192_mixed24_f64<>+0xA2C8(SB)/8, $3235
DATA bitrev_size8192_mixed24_f64<>+0xA2D0(SB)/8, $5283
DATA bitrev_size8192_mixed24_f64<>+0xA2D8(SB)/8, $7331
DATA bitrev_size8192_mixed24_f64<>+0xA2E0(SB)/8, $1699
DATA bitrev_size8192_mixed24_f64<>+0xA2E8(SB)/8, $3747
DATA bitrev_size8192_mixed24_f64<>+0xA2F0(SB)/8, $5795
DATA bitrev_size8192_mixed24_f64<>+0xA2F8(SB)/8, $7843
DATA bitrev_size8192_mixed24_f64<>+0xA300(SB)/8, $291
DATA bitrev_size8192_mixed24_f64<>+0xA308(SB)/8, $2339
DATA bitrev_size8192_mixed24_f64<>+0xA310(SB)/8, $4387
DATA bitrev_size8192_mixed24_f64<>+0xA318(SB)/8, $6435
DATA bitrev_size8192_mixed24_f64<>+0xA320(SB)/8, $803
DATA bitrev_size8192_mixed24_f64<>+0xA328(SB)/8, $2851
DATA bitrev_size8192_mixed24_f64<>+0xA330(SB)/8, $4899
DATA bitrev_size8192_mixed24_f64<>+0xA338(SB)/8, $6947
DATA bitrev_size8192_mixed24_f64<>+0xA340(SB)/8, $1315
DATA bitrev_size8192_mixed24_f64<>+0xA348(SB)/8, $3363
DATA bitrev_size8192_mixed24_f64<>+0xA350(SB)/8, $5411
DATA bitrev_size8192_mixed24_f64<>+0xA358(SB)/8, $7459
DATA bitrev_size8192_mixed24_f64<>+0xA360(SB)/8, $1827
DATA bitrev_size8192_mixed24_f64<>+0xA368(SB)/8, $3875
DATA bitrev_size8192_mixed24_f64<>+0xA370(SB)/8, $5923
DATA bitrev_size8192_mixed24_f64<>+0xA378(SB)/8, $7971
DATA bitrev_size8192_mixed24_f64<>+0xA380(SB)/8, $419
DATA bitrev_size8192_mixed24_f64<>+0xA388(SB)/8, $2467
DATA bitrev_size8192_mixed24_f64<>+0xA390(SB)/8, $4515
DATA bitrev_size8192_mixed24_f64<>+0xA398(SB)/8, $6563
DATA bitrev_size8192_mixed24_f64<>+0xA3A0(SB)/8, $931
DATA bitrev_size8192_mixed24_f64<>+0xA3A8(SB)/8, $2979
DATA bitrev_size8192_mixed24_f64<>+0xA3B0(SB)/8, $5027
DATA bitrev_size8192_mixed24_f64<>+0xA3B8(SB)/8, $7075
DATA bitrev_size8192_mixed24_f64<>+0xA3C0(SB)/8, $1443
DATA bitrev_size8192_mixed24_f64<>+0xA3C8(SB)/8, $3491
DATA bitrev_size8192_mixed24_f64<>+0xA3D0(SB)/8, $5539
DATA bitrev_size8192_mixed24_f64<>+0xA3D8(SB)/8, $7587
DATA bitrev_size8192_mixed24_f64<>+0xA3E0(SB)/8, $1955
DATA bitrev_size8192_mixed24_f64<>+0xA3E8(SB)/8, $4003
DATA bitrev_size8192_mixed24_f64<>+0xA3F0(SB)/8, $6051
DATA bitrev_size8192_mixed24_f64<>+0xA3F8(SB)/8, $8099
DATA bitrev_size8192_mixed24_f64<>+0xA400(SB)/8, $67
DATA bitrev_size8192_mixed24_f64<>+0xA408(SB)/8, $2115
DATA bitrev_size8192_mixed24_f64<>+0xA410(SB)/8, $4163
DATA bitrev_size8192_mixed24_f64<>+0xA418(SB)/8, $6211
DATA bitrev_size8192_mixed24_f64<>+0xA420(SB)/8, $579
DATA bitrev_size8192_mixed24_f64<>+0xA428(SB)/8, $2627
DATA bitrev_size8192_mixed24_f64<>+0xA430(SB)/8, $4675
DATA bitrev_size8192_mixed24_f64<>+0xA438(SB)/8, $6723
DATA bitrev_size8192_mixed24_f64<>+0xA440(SB)/8, $1091
DATA bitrev_size8192_mixed24_f64<>+0xA448(SB)/8, $3139
DATA bitrev_size8192_mixed24_f64<>+0xA450(SB)/8, $5187
DATA bitrev_size8192_mixed24_f64<>+0xA458(SB)/8, $7235
DATA bitrev_size8192_mixed24_f64<>+0xA460(SB)/8, $1603
DATA bitrev_size8192_mixed24_f64<>+0xA468(SB)/8, $3651
DATA bitrev_size8192_mixed24_f64<>+0xA470(SB)/8, $5699
DATA bitrev_size8192_mixed24_f64<>+0xA478(SB)/8, $7747
DATA bitrev_size8192_mixed24_f64<>+0xA480(SB)/8, $195
DATA bitrev_size8192_mixed24_f64<>+0xA488(SB)/8, $2243
DATA bitrev_size8192_mixed24_f64<>+0xA490(SB)/8, $4291
DATA bitrev_size8192_mixed24_f64<>+0xA498(SB)/8, $6339
DATA bitrev_size8192_mixed24_f64<>+0xA4A0(SB)/8, $707
DATA bitrev_size8192_mixed24_f64<>+0xA4A8(SB)/8, $2755
DATA bitrev_size8192_mixed24_f64<>+0xA4B0(SB)/8, $4803
DATA bitrev_size8192_mixed24_f64<>+0xA4B8(SB)/8, $6851
DATA bitrev_size8192_mixed24_f64<>+0xA4C0(SB)/8, $1219
DATA bitrev_size8192_mixed24_f64<>+0xA4C8(SB)/8, $3267
DATA bitrev_size8192_mixed24_f64<>+0xA4D0(SB)/8, $5315
DATA bitrev_size8192_mixed24_f64<>+0xA4D8(SB)/8, $7363
DATA bitrev_size8192_mixed24_f64<>+0xA4E0(SB)/8, $1731
DATA bitrev_size8192_mixed24_f64<>+0xA4E8(SB)/8, $3779
DATA bitrev_size8192_mixed24_f64<>+0xA4F0(SB)/8, $5827
DATA bitrev_size8192_mixed24_f64<>+0xA4F8(SB)/8, $7875
DATA bitrev_size8192_mixed24_f64<>+0xA500(SB)/8, $323
DATA bitrev_size8192_mixed24_f64<>+0xA508(SB)/8, $2371
DATA bitrev_size8192_mixed24_f64<>+0xA510(SB)/8, $4419
DATA bitrev_size8192_mixed24_f64<>+0xA518(SB)/8, $6467
DATA bitrev_size8192_mixed24_f64<>+0xA520(SB)/8, $835
DATA bitrev_size8192_mixed24_f64<>+0xA528(SB)/8, $2883
DATA bitrev_size8192_mixed24_f64<>+0xA530(SB)/8, $4931
DATA bitrev_size8192_mixed24_f64<>+0xA538(SB)/8, $6979
DATA bitrev_size8192_mixed24_f64<>+0xA540(SB)/8, $1347
DATA bitrev_size8192_mixed24_f64<>+0xA548(SB)/8, $3395
DATA bitrev_size8192_mixed24_f64<>+0xA550(SB)/8, $5443
DATA bitrev_size8192_mixed24_f64<>+0xA558(SB)/8, $7491
DATA bitrev_size8192_mixed24_f64<>+0xA560(SB)/8, $1859
DATA bitrev_size8192_mixed24_f64<>+0xA568(SB)/8, $3907
DATA bitrev_size8192_mixed24_f64<>+0xA570(SB)/8, $5955
DATA bitrev_size8192_mixed24_f64<>+0xA578(SB)/8, $8003
DATA bitrev_size8192_mixed24_f64<>+0xA580(SB)/8, $451
DATA bitrev_size8192_mixed24_f64<>+0xA588(SB)/8, $2499
DATA bitrev_size8192_mixed24_f64<>+0xA590(SB)/8, $4547
DATA bitrev_size8192_mixed24_f64<>+0xA598(SB)/8, $6595
DATA bitrev_size8192_mixed24_f64<>+0xA5A0(SB)/8, $963
DATA bitrev_size8192_mixed24_f64<>+0xA5A8(SB)/8, $3011
DATA bitrev_size8192_mixed24_f64<>+0xA5B0(SB)/8, $5059
DATA bitrev_size8192_mixed24_f64<>+0xA5B8(SB)/8, $7107
DATA bitrev_size8192_mixed24_f64<>+0xA5C0(SB)/8, $1475
DATA bitrev_size8192_mixed24_f64<>+0xA5C8(SB)/8, $3523
DATA bitrev_size8192_mixed24_f64<>+0xA5D0(SB)/8, $5571
DATA bitrev_size8192_mixed24_f64<>+0xA5D8(SB)/8, $7619
DATA bitrev_size8192_mixed24_f64<>+0xA5E0(SB)/8, $1987
DATA bitrev_size8192_mixed24_f64<>+0xA5E8(SB)/8, $4035
DATA bitrev_size8192_mixed24_f64<>+0xA5F0(SB)/8, $6083
DATA bitrev_size8192_mixed24_f64<>+0xA5F8(SB)/8, $8131
DATA bitrev_size8192_mixed24_f64<>+0xA600(SB)/8, $99
DATA bitrev_size8192_mixed24_f64<>+0xA608(SB)/8, $2147
DATA bitrev_size8192_mixed24_f64<>+0xA610(SB)/8, $4195
DATA bitrev_size8192_mixed24_f64<>+0xA618(SB)/8, $6243
DATA bitrev_size8192_mixed24_f64<>+0xA620(SB)/8, $611
DATA bitrev_size8192_mixed24_f64<>+0xA628(SB)/8, $2659
DATA bitrev_size8192_mixed24_f64<>+0xA630(SB)/8, $4707
DATA bitrev_size8192_mixed24_f64<>+0xA638(SB)/8, $6755
DATA bitrev_size8192_mixed24_f64<>+0xA640(SB)/8, $1123
DATA bitrev_size8192_mixed24_f64<>+0xA648(SB)/8, $3171
DATA bitrev_size8192_mixed24_f64<>+0xA650(SB)/8, $5219
DATA bitrev_size8192_mixed24_f64<>+0xA658(SB)/8, $7267
DATA bitrev_size8192_mixed24_f64<>+0xA660(SB)/8, $1635
DATA bitrev_size8192_mixed24_f64<>+0xA668(SB)/8, $3683
DATA bitrev_size8192_mixed24_f64<>+0xA670(SB)/8, $5731
DATA bitrev_size8192_mixed24_f64<>+0xA678(SB)/8, $7779
DATA bitrev_size8192_mixed24_f64<>+0xA680(SB)/8, $227
DATA bitrev_size8192_mixed24_f64<>+0xA688(SB)/8, $2275
DATA bitrev_size8192_mixed24_f64<>+0xA690(SB)/8, $4323
DATA bitrev_size8192_mixed24_f64<>+0xA698(SB)/8, $6371
DATA bitrev_size8192_mixed24_f64<>+0xA6A0(SB)/8, $739
DATA bitrev_size8192_mixed24_f64<>+0xA6A8(SB)/8, $2787
DATA bitrev_size8192_mixed24_f64<>+0xA6B0(SB)/8, $4835
DATA bitrev_size8192_mixed24_f64<>+0xA6B8(SB)/8, $6883
DATA bitrev_size8192_mixed24_f64<>+0xA6C0(SB)/8, $1251
DATA bitrev_size8192_mixed24_f64<>+0xA6C8(SB)/8, $3299
DATA bitrev_size8192_mixed24_f64<>+0xA6D0(SB)/8, $5347
DATA bitrev_size8192_mixed24_f64<>+0xA6D8(SB)/8, $7395
DATA bitrev_size8192_mixed24_f64<>+0xA6E0(SB)/8, $1763
DATA bitrev_size8192_mixed24_f64<>+0xA6E8(SB)/8, $3811
DATA bitrev_size8192_mixed24_f64<>+0xA6F0(SB)/8, $5859
DATA bitrev_size8192_mixed24_f64<>+0xA6F8(SB)/8, $7907
DATA bitrev_size8192_mixed24_f64<>+0xA700(SB)/8, $355
DATA bitrev_size8192_mixed24_f64<>+0xA708(SB)/8, $2403
DATA bitrev_size8192_mixed24_f64<>+0xA710(SB)/8, $4451
DATA bitrev_size8192_mixed24_f64<>+0xA718(SB)/8, $6499
DATA bitrev_size8192_mixed24_f64<>+0xA720(SB)/8, $867
DATA bitrev_size8192_mixed24_f64<>+0xA728(SB)/8, $2915
DATA bitrev_size8192_mixed24_f64<>+0xA730(SB)/8, $4963
DATA bitrev_size8192_mixed24_f64<>+0xA738(SB)/8, $7011
DATA bitrev_size8192_mixed24_f64<>+0xA740(SB)/8, $1379
DATA bitrev_size8192_mixed24_f64<>+0xA748(SB)/8, $3427
DATA bitrev_size8192_mixed24_f64<>+0xA750(SB)/8, $5475
DATA bitrev_size8192_mixed24_f64<>+0xA758(SB)/8, $7523
DATA bitrev_size8192_mixed24_f64<>+0xA760(SB)/8, $1891
DATA bitrev_size8192_mixed24_f64<>+0xA768(SB)/8, $3939
DATA bitrev_size8192_mixed24_f64<>+0xA770(SB)/8, $5987
DATA bitrev_size8192_mixed24_f64<>+0xA778(SB)/8, $8035
DATA bitrev_size8192_mixed24_f64<>+0xA780(SB)/8, $483
DATA bitrev_size8192_mixed24_f64<>+0xA788(SB)/8, $2531
DATA bitrev_size8192_mixed24_f64<>+0xA790(SB)/8, $4579
DATA bitrev_size8192_mixed24_f64<>+0xA798(SB)/8, $6627
DATA bitrev_size8192_mixed24_f64<>+0xA7A0(SB)/8, $995
DATA bitrev_size8192_mixed24_f64<>+0xA7A8(SB)/8, $3043
DATA bitrev_size8192_mixed24_f64<>+0xA7B0(SB)/8, $5091
DATA bitrev_size8192_mixed24_f64<>+0xA7B8(SB)/8, $7139
DATA bitrev_size8192_mixed24_f64<>+0xA7C0(SB)/8, $1507
DATA bitrev_size8192_mixed24_f64<>+0xA7C8(SB)/8, $3555
DATA bitrev_size8192_mixed24_f64<>+0xA7D0(SB)/8, $5603
DATA bitrev_size8192_mixed24_f64<>+0xA7D8(SB)/8, $7651
DATA bitrev_size8192_mixed24_f64<>+0xA7E0(SB)/8, $2019
DATA bitrev_size8192_mixed24_f64<>+0xA7E8(SB)/8, $4067
DATA bitrev_size8192_mixed24_f64<>+0xA7F0(SB)/8, $6115
DATA bitrev_size8192_mixed24_f64<>+0xA7F8(SB)/8, $8163
DATA bitrev_size8192_mixed24_f64<>+0xA800(SB)/8, $11
DATA bitrev_size8192_mixed24_f64<>+0xA808(SB)/8, $2059
DATA bitrev_size8192_mixed24_f64<>+0xA810(SB)/8, $4107
DATA bitrev_size8192_mixed24_f64<>+0xA818(SB)/8, $6155
DATA bitrev_size8192_mixed24_f64<>+0xA820(SB)/8, $523
DATA bitrev_size8192_mixed24_f64<>+0xA828(SB)/8, $2571
DATA bitrev_size8192_mixed24_f64<>+0xA830(SB)/8, $4619
DATA bitrev_size8192_mixed24_f64<>+0xA838(SB)/8, $6667
DATA bitrev_size8192_mixed24_f64<>+0xA840(SB)/8, $1035
DATA bitrev_size8192_mixed24_f64<>+0xA848(SB)/8, $3083
DATA bitrev_size8192_mixed24_f64<>+0xA850(SB)/8, $5131
DATA bitrev_size8192_mixed24_f64<>+0xA858(SB)/8, $7179
DATA bitrev_size8192_mixed24_f64<>+0xA860(SB)/8, $1547
DATA bitrev_size8192_mixed24_f64<>+0xA868(SB)/8, $3595
DATA bitrev_size8192_mixed24_f64<>+0xA870(SB)/8, $5643
DATA bitrev_size8192_mixed24_f64<>+0xA878(SB)/8, $7691
DATA bitrev_size8192_mixed24_f64<>+0xA880(SB)/8, $139
DATA bitrev_size8192_mixed24_f64<>+0xA888(SB)/8, $2187
DATA bitrev_size8192_mixed24_f64<>+0xA890(SB)/8, $4235
DATA bitrev_size8192_mixed24_f64<>+0xA898(SB)/8, $6283
DATA bitrev_size8192_mixed24_f64<>+0xA8A0(SB)/8, $651
DATA bitrev_size8192_mixed24_f64<>+0xA8A8(SB)/8, $2699
DATA bitrev_size8192_mixed24_f64<>+0xA8B0(SB)/8, $4747
DATA bitrev_size8192_mixed24_f64<>+0xA8B8(SB)/8, $6795
DATA bitrev_size8192_mixed24_f64<>+0xA8C0(SB)/8, $1163
DATA bitrev_size8192_mixed24_f64<>+0xA8C8(SB)/8, $3211
DATA bitrev_size8192_mixed24_f64<>+0xA8D0(SB)/8, $5259
DATA bitrev_size8192_mixed24_f64<>+0xA8D8(SB)/8, $7307
DATA bitrev_size8192_mixed24_f64<>+0xA8E0(SB)/8, $1675
DATA bitrev_size8192_mixed24_f64<>+0xA8E8(SB)/8, $3723
DATA bitrev_size8192_mixed24_f64<>+0xA8F0(SB)/8, $5771
DATA bitrev_size8192_mixed24_f64<>+0xA8F8(SB)/8, $7819
DATA bitrev_size8192_mixed24_f64<>+0xA900(SB)/8, $267
DATA bitrev_size8192_mixed24_f64<>+0xA908(SB)/8, $2315
DATA bitrev_size8192_mixed24_f64<>+0xA910(SB)/8, $4363
DATA bitrev_size8192_mixed24_f64<>+0xA918(SB)/8, $6411
DATA bitrev_size8192_mixed24_f64<>+0xA920(SB)/8, $779
DATA bitrev_size8192_mixed24_f64<>+0xA928(SB)/8, $2827
DATA bitrev_size8192_mixed24_f64<>+0xA930(SB)/8, $4875
DATA bitrev_size8192_mixed24_f64<>+0xA938(SB)/8, $6923
DATA bitrev_size8192_mixed24_f64<>+0xA940(SB)/8, $1291
DATA bitrev_size8192_mixed24_f64<>+0xA948(SB)/8, $3339
DATA bitrev_size8192_mixed24_f64<>+0xA950(SB)/8, $5387
DATA bitrev_size8192_mixed24_f64<>+0xA958(SB)/8, $7435
DATA bitrev_size8192_mixed24_f64<>+0xA960(SB)/8, $1803
DATA bitrev_size8192_mixed24_f64<>+0xA968(SB)/8, $3851
DATA bitrev_size8192_mixed24_f64<>+0xA970(SB)/8, $5899
DATA bitrev_size8192_mixed24_f64<>+0xA978(SB)/8, $7947
DATA bitrev_size8192_mixed24_f64<>+0xA980(SB)/8, $395
DATA bitrev_size8192_mixed24_f64<>+0xA988(SB)/8, $2443
DATA bitrev_size8192_mixed24_f64<>+0xA990(SB)/8, $4491
DATA bitrev_size8192_mixed24_f64<>+0xA998(SB)/8, $6539
DATA bitrev_size8192_mixed24_f64<>+0xA9A0(SB)/8, $907
DATA bitrev_size8192_mixed24_f64<>+0xA9A8(SB)/8, $2955
DATA bitrev_size8192_mixed24_f64<>+0xA9B0(SB)/8, $5003
DATA bitrev_size8192_mixed24_f64<>+0xA9B8(SB)/8, $7051
DATA bitrev_size8192_mixed24_f64<>+0xA9C0(SB)/8, $1419
DATA bitrev_size8192_mixed24_f64<>+0xA9C8(SB)/8, $3467
DATA bitrev_size8192_mixed24_f64<>+0xA9D0(SB)/8, $5515
DATA bitrev_size8192_mixed24_f64<>+0xA9D8(SB)/8, $7563
DATA bitrev_size8192_mixed24_f64<>+0xA9E0(SB)/8, $1931
DATA bitrev_size8192_mixed24_f64<>+0xA9E8(SB)/8, $3979
DATA bitrev_size8192_mixed24_f64<>+0xA9F0(SB)/8, $6027
DATA bitrev_size8192_mixed24_f64<>+0xA9F8(SB)/8, $8075
DATA bitrev_size8192_mixed24_f64<>+0xAA00(SB)/8, $43
DATA bitrev_size8192_mixed24_f64<>+0xAA08(SB)/8, $2091
DATA bitrev_size8192_mixed24_f64<>+0xAA10(SB)/8, $4139
DATA bitrev_size8192_mixed24_f64<>+0xAA18(SB)/8, $6187
DATA bitrev_size8192_mixed24_f64<>+0xAA20(SB)/8, $555
DATA bitrev_size8192_mixed24_f64<>+0xAA28(SB)/8, $2603
DATA bitrev_size8192_mixed24_f64<>+0xAA30(SB)/8, $4651
DATA bitrev_size8192_mixed24_f64<>+0xAA38(SB)/8, $6699
DATA bitrev_size8192_mixed24_f64<>+0xAA40(SB)/8, $1067
DATA bitrev_size8192_mixed24_f64<>+0xAA48(SB)/8, $3115
DATA bitrev_size8192_mixed24_f64<>+0xAA50(SB)/8, $5163
DATA bitrev_size8192_mixed24_f64<>+0xAA58(SB)/8, $7211
DATA bitrev_size8192_mixed24_f64<>+0xAA60(SB)/8, $1579
DATA bitrev_size8192_mixed24_f64<>+0xAA68(SB)/8, $3627
DATA bitrev_size8192_mixed24_f64<>+0xAA70(SB)/8, $5675
DATA bitrev_size8192_mixed24_f64<>+0xAA78(SB)/8, $7723
DATA bitrev_size8192_mixed24_f64<>+0xAA80(SB)/8, $171
DATA bitrev_size8192_mixed24_f64<>+0xAA88(SB)/8, $2219
DATA bitrev_size8192_mixed24_f64<>+0xAA90(SB)/8, $4267
DATA bitrev_size8192_mixed24_f64<>+0xAA98(SB)/8, $6315
DATA bitrev_size8192_mixed24_f64<>+0xAAA0(SB)/8, $683
DATA bitrev_size8192_mixed24_f64<>+0xAAA8(SB)/8, $2731
DATA bitrev_size8192_mixed24_f64<>+0xAAB0(SB)/8, $4779
DATA bitrev_size8192_mixed24_f64<>+0xAAB8(SB)/8, $6827
DATA bitrev_size8192_mixed24_f64<>+0xAAC0(SB)/8, $1195
DATA bitrev_size8192_mixed24_f64<>+0xAAC8(SB)/8, $3243
DATA bitrev_size8192_mixed24_f64<>+0xAAD0(SB)/8, $5291
DATA bitrev_size8192_mixed24_f64<>+0xAAD8(SB)/8, $7339
DATA bitrev_size8192_mixed24_f64<>+0xAAE0(SB)/8, $1707
DATA bitrev_size8192_mixed24_f64<>+0xAAE8(SB)/8, $3755
DATA bitrev_size8192_mixed24_f64<>+0xAAF0(SB)/8, $5803
DATA bitrev_size8192_mixed24_f64<>+0xAAF8(SB)/8, $7851
DATA bitrev_size8192_mixed24_f64<>+0xAB00(SB)/8, $299
DATA bitrev_size8192_mixed24_f64<>+0xAB08(SB)/8, $2347
DATA bitrev_size8192_mixed24_f64<>+0xAB10(SB)/8, $4395
DATA bitrev_size8192_mixed24_f64<>+0xAB18(SB)/8, $6443
DATA bitrev_size8192_mixed24_f64<>+0xAB20(SB)/8, $811
DATA bitrev_size8192_mixed24_f64<>+0xAB28(SB)/8, $2859
DATA bitrev_size8192_mixed24_f64<>+0xAB30(SB)/8, $4907
DATA bitrev_size8192_mixed24_f64<>+0xAB38(SB)/8, $6955
DATA bitrev_size8192_mixed24_f64<>+0xAB40(SB)/8, $1323
DATA bitrev_size8192_mixed24_f64<>+0xAB48(SB)/8, $3371
DATA bitrev_size8192_mixed24_f64<>+0xAB50(SB)/8, $5419
DATA bitrev_size8192_mixed24_f64<>+0xAB58(SB)/8, $7467
DATA bitrev_size8192_mixed24_f64<>+0xAB60(SB)/8, $1835
DATA bitrev_size8192_mixed24_f64<>+0xAB68(SB)/8, $3883
DATA bitrev_size8192_mixed24_f64<>+0xAB70(SB)/8, $5931
DATA bitrev_size8192_mixed24_f64<>+0xAB78(SB)/8, $7979
DATA bitrev_size8192_mixed24_f64<>+0xAB80(SB)/8, $427
DATA bitrev_size8192_mixed24_f64<>+0xAB88(SB)/8, $2475
DATA bitrev_size8192_mixed24_f64<>+0xAB90(SB)/8, $4523
DATA bitrev_size8192_mixed24_f64<>+0xAB98(SB)/8, $6571
DATA bitrev_size8192_mixed24_f64<>+0xABA0(SB)/8, $939
DATA bitrev_size8192_mixed24_f64<>+0xABA8(SB)/8, $2987
DATA bitrev_size8192_mixed24_f64<>+0xABB0(SB)/8, $5035
DATA bitrev_size8192_mixed24_f64<>+0xABB8(SB)/8, $7083
DATA bitrev_size8192_mixed24_f64<>+0xABC0(SB)/8, $1451
DATA bitrev_size8192_mixed24_f64<>+0xABC8(SB)/8, $3499
DATA bitrev_size8192_mixed24_f64<>+0xABD0(SB)/8, $5547
DATA bitrev_size8192_mixed24_f64<>+0xABD8(SB)/8, $7595
DATA bitrev_size8192_mixed24_f64<>+0xABE0(SB)/8, $1963
DATA bitrev_size8192_mixed24_f64<>+0xABE8(SB)/8, $4011
DATA bitrev_size8192_mixed24_f64<>+0xABF0(SB)/8, $6059
DATA bitrev_size8192_mixed24_f64<>+0xABF8(SB)/8, $8107
DATA bitrev_size8192_mixed24_f64<>+0xAC00(SB)/8, $75
DATA bitrev_size8192_mixed24_f64<>+0xAC08(SB)/8, $2123
DATA bitrev_size8192_mixed24_f64<>+0xAC10(SB)/8, $4171
DATA bitrev_size8192_mixed24_f64<>+0xAC18(SB)/8, $6219
DATA bitrev_size8192_mixed24_f64<>+0xAC20(SB)/8, $587
DATA bitrev_size8192_mixed24_f64<>+0xAC28(SB)/8, $2635
DATA bitrev_size8192_mixed24_f64<>+0xAC30(SB)/8, $4683
DATA bitrev_size8192_mixed24_f64<>+0xAC38(SB)/8, $6731
DATA bitrev_size8192_mixed24_f64<>+0xAC40(SB)/8, $1099
DATA bitrev_size8192_mixed24_f64<>+0xAC48(SB)/8, $3147
DATA bitrev_size8192_mixed24_f64<>+0xAC50(SB)/8, $5195
DATA bitrev_size8192_mixed24_f64<>+0xAC58(SB)/8, $7243
DATA bitrev_size8192_mixed24_f64<>+0xAC60(SB)/8, $1611
DATA bitrev_size8192_mixed24_f64<>+0xAC68(SB)/8, $3659
DATA bitrev_size8192_mixed24_f64<>+0xAC70(SB)/8, $5707
DATA bitrev_size8192_mixed24_f64<>+0xAC78(SB)/8, $7755
DATA bitrev_size8192_mixed24_f64<>+0xAC80(SB)/8, $203
DATA bitrev_size8192_mixed24_f64<>+0xAC88(SB)/8, $2251
DATA bitrev_size8192_mixed24_f64<>+0xAC90(SB)/8, $4299
DATA bitrev_size8192_mixed24_f64<>+0xAC98(SB)/8, $6347
DATA bitrev_size8192_mixed24_f64<>+0xACA0(SB)/8, $715
DATA bitrev_size8192_mixed24_f64<>+0xACA8(SB)/8, $2763
DATA bitrev_size8192_mixed24_f64<>+0xACB0(SB)/8, $4811
DATA bitrev_size8192_mixed24_f64<>+0xACB8(SB)/8, $6859
DATA bitrev_size8192_mixed24_f64<>+0xACC0(SB)/8, $1227
DATA bitrev_size8192_mixed24_f64<>+0xACC8(SB)/8, $3275
DATA bitrev_size8192_mixed24_f64<>+0xACD0(SB)/8, $5323
DATA bitrev_size8192_mixed24_f64<>+0xACD8(SB)/8, $7371
DATA bitrev_size8192_mixed24_f64<>+0xACE0(SB)/8, $1739
DATA bitrev_size8192_mixed24_f64<>+0xACE8(SB)/8, $3787
DATA bitrev_size8192_mixed24_f64<>+0xACF0(SB)/8, $5835
DATA bitrev_size8192_mixed24_f64<>+0xACF8(SB)/8, $7883
DATA bitrev_size8192_mixed24_f64<>+0xAD00(SB)/8, $331
DATA bitrev_size8192_mixed24_f64<>+0xAD08(SB)/8, $2379
DATA bitrev_size8192_mixed24_f64<>+0xAD10(SB)/8, $4427
DATA bitrev_size8192_mixed24_f64<>+0xAD18(SB)/8, $6475
DATA bitrev_size8192_mixed24_f64<>+0xAD20(SB)/8, $843
DATA bitrev_size8192_mixed24_f64<>+0xAD28(SB)/8, $2891
DATA bitrev_size8192_mixed24_f64<>+0xAD30(SB)/8, $4939
DATA bitrev_size8192_mixed24_f64<>+0xAD38(SB)/8, $6987
DATA bitrev_size8192_mixed24_f64<>+0xAD40(SB)/8, $1355
DATA bitrev_size8192_mixed24_f64<>+0xAD48(SB)/8, $3403
DATA bitrev_size8192_mixed24_f64<>+0xAD50(SB)/8, $5451
DATA bitrev_size8192_mixed24_f64<>+0xAD58(SB)/8, $7499
DATA bitrev_size8192_mixed24_f64<>+0xAD60(SB)/8, $1867
DATA bitrev_size8192_mixed24_f64<>+0xAD68(SB)/8, $3915
DATA bitrev_size8192_mixed24_f64<>+0xAD70(SB)/8, $5963
DATA bitrev_size8192_mixed24_f64<>+0xAD78(SB)/8, $8011
DATA bitrev_size8192_mixed24_f64<>+0xAD80(SB)/8, $459
DATA bitrev_size8192_mixed24_f64<>+0xAD88(SB)/8, $2507
DATA bitrev_size8192_mixed24_f64<>+0xAD90(SB)/8, $4555
DATA bitrev_size8192_mixed24_f64<>+0xAD98(SB)/8, $6603
DATA bitrev_size8192_mixed24_f64<>+0xADA0(SB)/8, $971
DATA bitrev_size8192_mixed24_f64<>+0xADA8(SB)/8, $3019
DATA bitrev_size8192_mixed24_f64<>+0xADB0(SB)/8, $5067
DATA bitrev_size8192_mixed24_f64<>+0xADB8(SB)/8, $7115
DATA bitrev_size8192_mixed24_f64<>+0xADC0(SB)/8, $1483
DATA bitrev_size8192_mixed24_f64<>+0xADC8(SB)/8, $3531
DATA bitrev_size8192_mixed24_f64<>+0xADD0(SB)/8, $5579
DATA bitrev_size8192_mixed24_f64<>+0xADD8(SB)/8, $7627
DATA bitrev_size8192_mixed24_f64<>+0xADE0(SB)/8, $1995
DATA bitrev_size8192_mixed24_f64<>+0xADE8(SB)/8, $4043
DATA bitrev_size8192_mixed24_f64<>+0xADF0(SB)/8, $6091
DATA bitrev_size8192_mixed24_f64<>+0xADF8(SB)/8, $8139
DATA bitrev_size8192_mixed24_f64<>+0xAE00(SB)/8, $107
DATA bitrev_size8192_mixed24_f64<>+0xAE08(SB)/8, $2155
DATA bitrev_size8192_mixed24_f64<>+0xAE10(SB)/8, $4203
DATA bitrev_size8192_mixed24_f64<>+0xAE18(SB)/8, $6251
DATA bitrev_size8192_mixed24_f64<>+0xAE20(SB)/8, $619
DATA bitrev_size8192_mixed24_f64<>+0xAE28(SB)/8, $2667
DATA bitrev_size8192_mixed24_f64<>+0xAE30(SB)/8, $4715
DATA bitrev_size8192_mixed24_f64<>+0xAE38(SB)/8, $6763
DATA bitrev_size8192_mixed24_f64<>+0xAE40(SB)/8, $1131
DATA bitrev_size8192_mixed24_f64<>+0xAE48(SB)/8, $3179
DATA bitrev_size8192_mixed24_f64<>+0xAE50(SB)/8, $5227
DATA bitrev_size8192_mixed24_f64<>+0xAE58(SB)/8, $7275
DATA bitrev_size8192_mixed24_f64<>+0xAE60(SB)/8, $1643
DATA bitrev_size8192_mixed24_f64<>+0xAE68(SB)/8, $3691
DATA bitrev_size8192_mixed24_f64<>+0xAE70(SB)/8, $5739
DATA bitrev_size8192_mixed24_f64<>+0xAE78(SB)/8, $7787
DATA bitrev_size8192_mixed24_f64<>+0xAE80(SB)/8, $235
DATA bitrev_size8192_mixed24_f64<>+0xAE88(SB)/8, $2283
DATA bitrev_size8192_mixed24_f64<>+0xAE90(SB)/8, $4331
DATA bitrev_size8192_mixed24_f64<>+0xAE98(SB)/8, $6379
DATA bitrev_size8192_mixed24_f64<>+0xAEA0(SB)/8, $747
DATA bitrev_size8192_mixed24_f64<>+0xAEA8(SB)/8, $2795
DATA bitrev_size8192_mixed24_f64<>+0xAEB0(SB)/8, $4843
DATA bitrev_size8192_mixed24_f64<>+0xAEB8(SB)/8, $6891
DATA bitrev_size8192_mixed24_f64<>+0xAEC0(SB)/8, $1259
DATA bitrev_size8192_mixed24_f64<>+0xAEC8(SB)/8, $3307
DATA bitrev_size8192_mixed24_f64<>+0xAED0(SB)/8, $5355
DATA bitrev_size8192_mixed24_f64<>+0xAED8(SB)/8, $7403
DATA bitrev_size8192_mixed24_f64<>+0xAEE0(SB)/8, $1771
DATA bitrev_size8192_mixed24_f64<>+0xAEE8(SB)/8, $3819
DATA bitrev_size8192_mixed24_f64<>+0xAEF0(SB)/8, $5867
DATA bitrev_size8192_mixed24_f64<>+0xAEF8(SB)/8, $7915
DATA bitrev_size8192_mixed24_f64<>+0xAF00(SB)/8, $363
DATA bitrev_size8192_mixed24_f64<>+0xAF08(SB)/8, $2411
DATA bitrev_size8192_mixed24_f64<>+0xAF10(SB)/8, $4459
DATA bitrev_size8192_mixed24_f64<>+0xAF18(SB)/8, $6507
DATA bitrev_size8192_mixed24_f64<>+0xAF20(SB)/8, $875
DATA bitrev_size8192_mixed24_f64<>+0xAF28(SB)/8, $2923
DATA bitrev_size8192_mixed24_f64<>+0xAF30(SB)/8, $4971
DATA bitrev_size8192_mixed24_f64<>+0xAF38(SB)/8, $7019
DATA bitrev_size8192_mixed24_f64<>+0xAF40(SB)/8, $1387
DATA bitrev_size8192_mixed24_f64<>+0xAF48(SB)/8, $3435
DATA bitrev_size8192_mixed24_f64<>+0xAF50(SB)/8, $5483
DATA bitrev_size8192_mixed24_f64<>+0xAF58(SB)/8, $7531
DATA bitrev_size8192_mixed24_f64<>+0xAF60(SB)/8, $1899
DATA bitrev_size8192_mixed24_f64<>+0xAF68(SB)/8, $3947
DATA bitrev_size8192_mixed24_f64<>+0xAF70(SB)/8, $5995
DATA bitrev_size8192_mixed24_f64<>+0xAF78(SB)/8, $8043
DATA bitrev_size8192_mixed24_f64<>+0xAF80(SB)/8, $491
DATA bitrev_size8192_mixed24_f64<>+0xAF88(SB)/8, $2539
DATA bitrev_size8192_mixed24_f64<>+0xAF90(SB)/8, $4587
DATA bitrev_size8192_mixed24_f64<>+0xAF98(SB)/8, $6635
DATA bitrev_size8192_mixed24_f64<>+0xAFA0(SB)/8, $1003
DATA bitrev_size8192_mixed24_f64<>+0xAFA8(SB)/8, $3051
DATA bitrev_size8192_mixed24_f64<>+0xAFB0(SB)/8, $5099
DATA bitrev_size8192_mixed24_f64<>+0xAFB8(SB)/8, $7147
DATA bitrev_size8192_mixed24_f64<>+0xAFC0(SB)/8, $1515
DATA bitrev_size8192_mixed24_f64<>+0xAFC8(SB)/8, $3563
DATA bitrev_size8192_mixed24_f64<>+0xAFD0(SB)/8, $5611
DATA bitrev_size8192_mixed24_f64<>+0xAFD8(SB)/8, $7659
DATA bitrev_size8192_mixed24_f64<>+0xAFE0(SB)/8, $2027
DATA bitrev_size8192_mixed24_f64<>+0xAFE8(SB)/8, $4075
DATA bitrev_size8192_mixed24_f64<>+0xAFF0(SB)/8, $6123
DATA bitrev_size8192_mixed24_f64<>+0xAFF8(SB)/8, $8171
DATA bitrev_size8192_mixed24_f64<>+0xB000(SB)/8, $19
DATA bitrev_size8192_mixed24_f64<>+0xB008(SB)/8, $2067
DATA bitrev_size8192_mixed24_f64<>+0xB010(SB)/8, $4115
DATA bitrev_size8192_mixed24_f64<>+0xB018(SB)/8, $6163
DATA bitrev_size8192_mixed24_f64<>+0xB020(SB)/8, $531
DATA bitrev_size8192_mixed24_f64<>+0xB028(SB)/8, $2579
DATA bitrev_size8192_mixed24_f64<>+0xB030(SB)/8, $4627
DATA bitrev_size8192_mixed24_f64<>+0xB038(SB)/8, $6675
DATA bitrev_size8192_mixed24_f64<>+0xB040(SB)/8, $1043
DATA bitrev_size8192_mixed24_f64<>+0xB048(SB)/8, $3091
DATA bitrev_size8192_mixed24_f64<>+0xB050(SB)/8, $5139
DATA bitrev_size8192_mixed24_f64<>+0xB058(SB)/8, $7187
DATA bitrev_size8192_mixed24_f64<>+0xB060(SB)/8, $1555
DATA bitrev_size8192_mixed24_f64<>+0xB068(SB)/8, $3603
DATA bitrev_size8192_mixed24_f64<>+0xB070(SB)/8, $5651
DATA bitrev_size8192_mixed24_f64<>+0xB078(SB)/8, $7699
DATA bitrev_size8192_mixed24_f64<>+0xB080(SB)/8, $147
DATA bitrev_size8192_mixed24_f64<>+0xB088(SB)/8, $2195
DATA bitrev_size8192_mixed24_f64<>+0xB090(SB)/8, $4243
DATA bitrev_size8192_mixed24_f64<>+0xB098(SB)/8, $6291
DATA bitrev_size8192_mixed24_f64<>+0xB0A0(SB)/8, $659
DATA bitrev_size8192_mixed24_f64<>+0xB0A8(SB)/8, $2707
DATA bitrev_size8192_mixed24_f64<>+0xB0B0(SB)/8, $4755
DATA bitrev_size8192_mixed24_f64<>+0xB0B8(SB)/8, $6803
DATA bitrev_size8192_mixed24_f64<>+0xB0C0(SB)/8, $1171
DATA bitrev_size8192_mixed24_f64<>+0xB0C8(SB)/8, $3219
DATA bitrev_size8192_mixed24_f64<>+0xB0D0(SB)/8, $5267
DATA bitrev_size8192_mixed24_f64<>+0xB0D8(SB)/8, $7315
DATA bitrev_size8192_mixed24_f64<>+0xB0E0(SB)/8, $1683
DATA bitrev_size8192_mixed24_f64<>+0xB0E8(SB)/8, $3731
DATA bitrev_size8192_mixed24_f64<>+0xB0F0(SB)/8, $5779
DATA bitrev_size8192_mixed24_f64<>+0xB0F8(SB)/8, $7827
DATA bitrev_size8192_mixed24_f64<>+0xB100(SB)/8, $275
DATA bitrev_size8192_mixed24_f64<>+0xB108(SB)/8, $2323
DATA bitrev_size8192_mixed24_f64<>+0xB110(SB)/8, $4371
DATA bitrev_size8192_mixed24_f64<>+0xB118(SB)/8, $6419
DATA bitrev_size8192_mixed24_f64<>+0xB120(SB)/8, $787
DATA bitrev_size8192_mixed24_f64<>+0xB128(SB)/8, $2835
DATA bitrev_size8192_mixed24_f64<>+0xB130(SB)/8, $4883
DATA bitrev_size8192_mixed24_f64<>+0xB138(SB)/8, $6931
DATA bitrev_size8192_mixed24_f64<>+0xB140(SB)/8, $1299
DATA bitrev_size8192_mixed24_f64<>+0xB148(SB)/8, $3347
DATA bitrev_size8192_mixed24_f64<>+0xB150(SB)/8, $5395
DATA bitrev_size8192_mixed24_f64<>+0xB158(SB)/8, $7443
DATA bitrev_size8192_mixed24_f64<>+0xB160(SB)/8, $1811
DATA bitrev_size8192_mixed24_f64<>+0xB168(SB)/8, $3859
DATA bitrev_size8192_mixed24_f64<>+0xB170(SB)/8, $5907
DATA bitrev_size8192_mixed24_f64<>+0xB178(SB)/8, $7955
DATA bitrev_size8192_mixed24_f64<>+0xB180(SB)/8, $403
DATA bitrev_size8192_mixed24_f64<>+0xB188(SB)/8, $2451
DATA bitrev_size8192_mixed24_f64<>+0xB190(SB)/8, $4499
DATA bitrev_size8192_mixed24_f64<>+0xB198(SB)/8, $6547
DATA bitrev_size8192_mixed24_f64<>+0xB1A0(SB)/8, $915
DATA bitrev_size8192_mixed24_f64<>+0xB1A8(SB)/8, $2963
DATA bitrev_size8192_mixed24_f64<>+0xB1B0(SB)/8, $5011
DATA bitrev_size8192_mixed24_f64<>+0xB1B8(SB)/8, $7059
DATA bitrev_size8192_mixed24_f64<>+0xB1C0(SB)/8, $1427
DATA bitrev_size8192_mixed24_f64<>+0xB1C8(SB)/8, $3475
DATA bitrev_size8192_mixed24_f64<>+0xB1D0(SB)/8, $5523
DATA bitrev_size8192_mixed24_f64<>+0xB1D8(SB)/8, $7571
DATA bitrev_size8192_mixed24_f64<>+0xB1E0(SB)/8, $1939
DATA bitrev_size8192_mixed24_f64<>+0xB1E8(SB)/8, $3987
DATA bitrev_size8192_mixed24_f64<>+0xB1F0(SB)/8, $6035
DATA bitrev_size8192_mixed24_f64<>+0xB1F8(SB)/8, $8083
DATA bitrev_size8192_mixed24_f64<>+0xB200(SB)/8, $51
DATA bitrev_size8192_mixed24_f64<>+0xB208(SB)/8, $2099
DATA bitrev_size8192_mixed24_f64<>+0xB210(SB)/8, $4147
DATA bitrev_size8192_mixed24_f64<>+0xB218(SB)/8, $6195
DATA bitrev_size8192_mixed24_f64<>+0xB220(SB)/8, $563
DATA bitrev_size8192_mixed24_f64<>+0xB228(SB)/8, $2611
DATA bitrev_size8192_mixed24_f64<>+0xB230(SB)/8, $4659
DATA bitrev_size8192_mixed24_f64<>+0xB238(SB)/8, $6707
DATA bitrev_size8192_mixed24_f64<>+0xB240(SB)/8, $1075
DATA bitrev_size8192_mixed24_f64<>+0xB248(SB)/8, $3123
DATA bitrev_size8192_mixed24_f64<>+0xB250(SB)/8, $5171
DATA bitrev_size8192_mixed24_f64<>+0xB258(SB)/8, $7219
DATA bitrev_size8192_mixed24_f64<>+0xB260(SB)/8, $1587
DATA bitrev_size8192_mixed24_f64<>+0xB268(SB)/8, $3635
DATA bitrev_size8192_mixed24_f64<>+0xB270(SB)/8, $5683
DATA bitrev_size8192_mixed24_f64<>+0xB278(SB)/8, $7731
DATA bitrev_size8192_mixed24_f64<>+0xB280(SB)/8, $179
DATA bitrev_size8192_mixed24_f64<>+0xB288(SB)/8, $2227
DATA bitrev_size8192_mixed24_f64<>+0xB290(SB)/8, $4275
DATA bitrev_size8192_mixed24_f64<>+0xB298(SB)/8, $6323
DATA bitrev_size8192_mixed24_f64<>+0xB2A0(SB)/8, $691
DATA bitrev_size8192_mixed24_f64<>+0xB2A8(SB)/8, $2739
DATA bitrev_size8192_mixed24_f64<>+0xB2B0(SB)/8, $4787
DATA bitrev_size8192_mixed24_f64<>+0xB2B8(SB)/8, $6835
DATA bitrev_size8192_mixed24_f64<>+0xB2C0(SB)/8, $1203
DATA bitrev_size8192_mixed24_f64<>+0xB2C8(SB)/8, $3251
DATA bitrev_size8192_mixed24_f64<>+0xB2D0(SB)/8, $5299
DATA bitrev_size8192_mixed24_f64<>+0xB2D8(SB)/8, $7347
DATA bitrev_size8192_mixed24_f64<>+0xB2E0(SB)/8, $1715
DATA bitrev_size8192_mixed24_f64<>+0xB2E8(SB)/8, $3763
DATA bitrev_size8192_mixed24_f64<>+0xB2F0(SB)/8, $5811
DATA bitrev_size8192_mixed24_f64<>+0xB2F8(SB)/8, $7859
DATA bitrev_size8192_mixed24_f64<>+0xB300(SB)/8, $307
DATA bitrev_size8192_mixed24_f64<>+0xB308(SB)/8, $2355
DATA bitrev_size8192_mixed24_f64<>+0xB310(SB)/8, $4403
DATA bitrev_size8192_mixed24_f64<>+0xB318(SB)/8, $6451
DATA bitrev_size8192_mixed24_f64<>+0xB320(SB)/8, $819
DATA bitrev_size8192_mixed24_f64<>+0xB328(SB)/8, $2867
DATA bitrev_size8192_mixed24_f64<>+0xB330(SB)/8, $4915
DATA bitrev_size8192_mixed24_f64<>+0xB338(SB)/8, $6963
DATA bitrev_size8192_mixed24_f64<>+0xB340(SB)/8, $1331
DATA bitrev_size8192_mixed24_f64<>+0xB348(SB)/8, $3379
DATA bitrev_size8192_mixed24_f64<>+0xB350(SB)/8, $5427
DATA bitrev_size8192_mixed24_f64<>+0xB358(SB)/8, $7475
DATA bitrev_size8192_mixed24_f64<>+0xB360(SB)/8, $1843
DATA bitrev_size8192_mixed24_f64<>+0xB368(SB)/8, $3891
DATA bitrev_size8192_mixed24_f64<>+0xB370(SB)/8, $5939
DATA bitrev_size8192_mixed24_f64<>+0xB378(SB)/8, $7987
DATA bitrev_size8192_mixed24_f64<>+0xB380(SB)/8, $435
DATA bitrev_size8192_mixed24_f64<>+0xB388(SB)/8, $2483
DATA bitrev_size8192_mixed24_f64<>+0xB390(SB)/8, $4531
DATA bitrev_size8192_mixed24_f64<>+0xB398(SB)/8, $6579
DATA bitrev_size8192_mixed24_f64<>+0xB3A0(SB)/8, $947
DATA bitrev_size8192_mixed24_f64<>+0xB3A8(SB)/8, $2995
DATA bitrev_size8192_mixed24_f64<>+0xB3B0(SB)/8, $5043
DATA bitrev_size8192_mixed24_f64<>+0xB3B8(SB)/8, $7091
DATA bitrev_size8192_mixed24_f64<>+0xB3C0(SB)/8, $1459
DATA bitrev_size8192_mixed24_f64<>+0xB3C8(SB)/8, $3507
DATA bitrev_size8192_mixed24_f64<>+0xB3D0(SB)/8, $5555
DATA bitrev_size8192_mixed24_f64<>+0xB3D8(SB)/8, $7603
DATA bitrev_size8192_mixed24_f64<>+0xB3E0(SB)/8, $1971
DATA bitrev_size8192_mixed24_f64<>+0xB3E8(SB)/8, $4019
DATA bitrev_size8192_mixed24_f64<>+0xB3F0(SB)/8, $6067
DATA bitrev_size8192_mixed24_f64<>+0xB3F8(SB)/8, $8115
DATA bitrev_size8192_mixed24_f64<>+0xB400(SB)/8, $83
DATA bitrev_size8192_mixed24_f64<>+0xB408(SB)/8, $2131
DATA bitrev_size8192_mixed24_f64<>+0xB410(SB)/8, $4179
DATA bitrev_size8192_mixed24_f64<>+0xB418(SB)/8, $6227
DATA bitrev_size8192_mixed24_f64<>+0xB420(SB)/8, $595
DATA bitrev_size8192_mixed24_f64<>+0xB428(SB)/8, $2643
DATA bitrev_size8192_mixed24_f64<>+0xB430(SB)/8, $4691
DATA bitrev_size8192_mixed24_f64<>+0xB438(SB)/8, $6739
DATA bitrev_size8192_mixed24_f64<>+0xB440(SB)/8, $1107
DATA bitrev_size8192_mixed24_f64<>+0xB448(SB)/8, $3155
DATA bitrev_size8192_mixed24_f64<>+0xB450(SB)/8, $5203
DATA bitrev_size8192_mixed24_f64<>+0xB458(SB)/8, $7251
DATA bitrev_size8192_mixed24_f64<>+0xB460(SB)/8, $1619
DATA bitrev_size8192_mixed24_f64<>+0xB468(SB)/8, $3667
DATA bitrev_size8192_mixed24_f64<>+0xB470(SB)/8, $5715
DATA bitrev_size8192_mixed24_f64<>+0xB478(SB)/8, $7763
DATA bitrev_size8192_mixed24_f64<>+0xB480(SB)/8, $211
DATA bitrev_size8192_mixed24_f64<>+0xB488(SB)/8, $2259
DATA bitrev_size8192_mixed24_f64<>+0xB490(SB)/8, $4307
DATA bitrev_size8192_mixed24_f64<>+0xB498(SB)/8, $6355
DATA bitrev_size8192_mixed24_f64<>+0xB4A0(SB)/8, $723
DATA bitrev_size8192_mixed24_f64<>+0xB4A8(SB)/8, $2771
DATA bitrev_size8192_mixed24_f64<>+0xB4B0(SB)/8, $4819
DATA bitrev_size8192_mixed24_f64<>+0xB4B8(SB)/8, $6867
DATA bitrev_size8192_mixed24_f64<>+0xB4C0(SB)/8, $1235
DATA bitrev_size8192_mixed24_f64<>+0xB4C8(SB)/8, $3283
DATA bitrev_size8192_mixed24_f64<>+0xB4D0(SB)/8, $5331
DATA bitrev_size8192_mixed24_f64<>+0xB4D8(SB)/8, $7379
DATA bitrev_size8192_mixed24_f64<>+0xB4E0(SB)/8, $1747
DATA bitrev_size8192_mixed24_f64<>+0xB4E8(SB)/8, $3795
DATA bitrev_size8192_mixed24_f64<>+0xB4F0(SB)/8, $5843
DATA bitrev_size8192_mixed24_f64<>+0xB4F8(SB)/8, $7891
DATA bitrev_size8192_mixed24_f64<>+0xB500(SB)/8, $339
DATA bitrev_size8192_mixed24_f64<>+0xB508(SB)/8, $2387
DATA bitrev_size8192_mixed24_f64<>+0xB510(SB)/8, $4435
DATA bitrev_size8192_mixed24_f64<>+0xB518(SB)/8, $6483
DATA bitrev_size8192_mixed24_f64<>+0xB520(SB)/8, $851
DATA bitrev_size8192_mixed24_f64<>+0xB528(SB)/8, $2899
DATA bitrev_size8192_mixed24_f64<>+0xB530(SB)/8, $4947
DATA bitrev_size8192_mixed24_f64<>+0xB538(SB)/8, $6995
DATA bitrev_size8192_mixed24_f64<>+0xB540(SB)/8, $1363
DATA bitrev_size8192_mixed24_f64<>+0xB548(SB)/8, $3411
DATA bitrev_size8192_mixed24_f64<>+0xB550(SB)/8, $5459
DATA bitrev_size8192_mixed24_f64<>+0xB558(SB)/8, $7507
DATA bitrev_size8192_mixed24_f64<>+0xB560(SB)/8, $1875
DATA bitrev_size8192_mixed24_f64<>+0xB568(SB)/8, $3923
DATA bitrev_size8192_mixed24_f64<>+0xB570(SB)/8, $5971
DATA bitrev_size8192_mixed24_f64<>+0xB578(SB)/8, $8019
DATA bitrev_size8192_mixed24_f64<>+0xB580(SB)/8, $467
DATA bitrev_size8192_mixed24_f64<>+0xB588(SB)/8, $2515
DATA bitrev_size8192_mixed24_f64<>+0xB590(SB)/8, $4563
DATA bitrev_size8192_mixed24_f64<>+0xB598(SB)/8, $6611
DATA bitrev_size8192_mixed24_f64<>+0xB5A0(SB)/8, $979
DATA bitrev_size8192_mixed24_f64<>+0xB5A8(SB)/8, $3027
DATA bitrev_size8192_mixed24_f64<>+0xB5B0(SB)/8, $5075
DATA bitrev_size8192_mixed24_f64<>+0xB5B8(SB)/8, $7123
DATA bitrev_size8192_mixed24_f64<>+0xB5C0(SB)/8, $1491
DATA bitrev_size8192_mixed24_f64<>+0xB5C8(SB)/8, $3539
DATA bitrev_size8192_mixed24_f64<>+0xB5D0(SB)/8, $5587
DATA bitrev_size8192_mixed24_f64<>+0xB5D8(SB)/8, $7635
DATA bitrev_size8192_mixed24_f64<>+0xB5E0(SB)/8, $2003
DATA bitrev_size8192_mixed24_f64<>+0xB5E8(SB)/8, $4051
DATA bitrev_size8192_mixed24_f64<>+0xB5F0(SB)/8, $6099
DATA bitrev_size8192_mixed24_f64<>+0xB5F8(SB)/8, $8147
DATA bitrev_size8192_mixed24_f64<>+0xB600(SB)/8, $115
DATA bitrev_size8192_mixed24_f64<>+0xB608(SB)/8, $2163
DATA bitrev_size8192_mixed24_f64<>+0xB610(SB)/8, $4211
DATA bitrev_size8192_mixed24_f64<>+0xB618(SB)/8, $6259
DATA bitrev_size8192_mixed24_f64<>+0xB620(SB)/8, $627
DATA bitrev_size8192_mixed24_f64<>+0xB628(SB)/8, $2675
DATA bitrev_size8192_mixed24_f64<>+0xB630(SB)/8, $4723
DATA bitrev_size8192_mixed24_f64<>+0xB638(SB)/8, $6771
DATA bitrev_size8192_mixed24_f64<>+0xB640(SB)/8, $1139
DATA bitrev_size8192_mixed24_f64<>+0xB648(SB)/8, $3187
DATA bitrev_size8192_mixed24_f64<>+0xB650(SB)/8, $5235
DATA bitrev_size8192_mixed24_f64<>+0xB658(SB)/8, $7283
DATA bitrev_size8192_mixed24_f64<>+0xB660(SB)/8, $1651
DATA bitrev_size8192_mixed24_f64<>+0xB668(SB)/8, $3699
DATA bitrev_size8192_mixed24_f64<>+0xB670(SB)/8, $5747
DATA bitrev_size8192_mixed24_f64<>+0xB678(SB)/8, $7795
DATA bitrev_size8192_mixed24_f64<>+0xB680(SB)/8, $243
DATA bitrev_size8192_mixed24_f64<>+0xB688(SB)/8, $2291
DATA bitrev_size8192_mixed24_f64<>+0xB690(SB)/8, $4339
DATA bitrev_size8192_mixed24_f64<>+0xB698(SB)/8, $6387
DATA bitrev_size8192_mixed24_f64<>+0xB6A0(SB)/8, $755
DATA bitrev_size8192_mixed24_f64<>+0xB6A8(SB)/8, $2803
DATA bitrev_size8192_mixed24_f64<>+0xB6B0(SB)/8, $4851
DATA bitrev_size8192_mixed24_f64<>+0xB6B8(SB)/8, $6899
DATA bitrev_size8192_mixed24_f64<>+0xB6C0(SB)/8, $1267
DATA bitrev_size8192_mixed24_f64<>+0xB6C8(SB)/8, $3315
DATA bitrev_size8192_mixed24_f64<>+0xB6D0(SB)/8, $5363
DATA bitrev_size8192_mixed24_f64<>+0xB6D8(SB)/8, $7411
DATA bitrev_size8192_mixed24_f64<>+0xB6E0(SB)/8, $1779
DATA bitrev_size8192_mixed24_f64<>+0xB6E8(SB)/8, $3827
DATA bitrev_size8192_mixed24_f64<>+0xB6F0(SB)/8, $5875
DATA bitrev_size8192_mixed24_f64<>+0xB6F8(SB)/8, $7923
DATA bitrev_size8192_mixed24_f64<>+0xB700(SB)/8, $371
DATA bitrev_size8192_mixed24_f64<>+0xB708(SB)/8, $2419
DATA bitrev_size8192_mixed24_f64<>+0xB710(SB)/8, $4467
DATA bitrev_size8192_mixed24_f64<>+0xB718(SB)/8, $6515
DATA bitrev_size8192_mixed24_f64<>+0xB720(SB)/8, $883
DATA bitrev_size8192_mixed24_f64<>+0xB728(SB)/8, $2931
DATA bitrev_size8192_mixed24_f64<>+0xB730(SB)/8, $4979
DATA bitrev_size8192_mixed24_f64<>+0xB738(SB)/8, $7027
DATA bitrev_size8192_mixed24_f64<>+0xB740(SB)/8, $1395
DATA bitrev_size8192_mixed24_f64<>+0xB748(SB)/8, $3443
DATA bitrev_size8192_mixed24_f64<>+0xB750(SB)/8, $5491
DATA bitrev_size8192_mixed24_f64<>+0xB758(SB)/8, $7539
DATA bitrev_size8192_mixed24_f64<>+0xB760(SB)/8, $1907
DATA bitrev_size8192_mixed24_f64<>+0xB768(SB)/8, $3955
DATA bitrev_size8192_mixed24_f64<>+0xB770(SB)/8, $6003
DATA bitrev_size8192_mixed24_f64<>+0xB778(SB)/8, $8051
DATA bitrev_size8192_mixed24_f64<>+0xB780(SB)/8, $499
DATA bitrev_size8192_mixed24_f64<>+0xB788(SB)/8, $2547
DATA bitrev_size8192_mixed24_f64<>+0xB790(SB)/8, $4595
DATA bitrev_size8192_mixed24_f64<>+0xB798(SB)/8, $6643
DATA bitrev_size8192_mixed24_f64<>+0xB7A0(SB)/8, $1011
DATA bitrev_size8192_mixed24_f64<>+0xB7A8(SB)/8, $3059
DATA bitrev_size8192_mixed24_f64<>+0xB7B0(SB)/8, $5107
DATA bitrev_size8192_mixed24_f64<>+0xB7B8(SB)/8, $7155
DATA bitrev_size8192_mixed24_f64<>+0xB7C0(SB)/8, $1523
DATA bitrev_size8192_mixed24_f64<>+0xB7C8(SB)/8, $3571
DATA bitrev_size8192_mixed24_f64<>+0xB7D0(SB)/8, $5619
DATA bitrev_size8192_mixed24_f64<>+0xB7D8(SB)/8, $7667
DATA bitrev_size8192_mixed24_f64<>+0xB7E0(SB)/8, $2035
DATA bitrev_size8192_mixed24_f64<>+0xB7E8(SB)/8, $4083
DATA bitrev_size8192_mixed24_f64<>+0xB7F0(SB)/8, $6131
DATA bitrev_size8192_mixed24_f64<>+0xB7F8(SB)/8, $8179
DATA bitrev_size8192_mixed24_f64<>+0xB800(SB)/8, $27
DATA bitrev_size8192_mixed24_f64<>+0xB808(SB)/8, $2075
DATA bitrev_size8192_mixed24_f64<>+0xB810(SB)/8, $4123
DATA bitrev_size8192_mixed24_f64<>+0xB818(SB)/8, $6171
DATA bitrev_size8192_mixed24_f64<>+0xB820(SB)/8, $539
DATA bitrev_size8192_mixed24_f64<>+0xB828(SB)/8, $2587
DATA bitrev_size8192_mixed24_f64<>+0xB830(SB)/8, $4635
DATA bitrev_size8192_mixed24_f64<>+0xB838(SB)/8, $6683
DATA bitrev_size8192_mixed24_f64<>+0xB840(SB)/8, $1051
DATA bitrev_size8192_mixed24_f64<>+0xB848(SB)/8, $3099
DATA bitrev_size8192_mixed24_f64<>+0xB850(SB)/8, $5147
DATA bitrev_size8192_mixed24_f64<>+0xB858(SB)/8, $7195
DATA bitrev_size8192_mixed24_f64<>+0xB860(SB)/8, $1563
DATA bitrev_size8192_mixed24_f64<>+0xB868(SB)/8, $3611
DATA bitrev_size8192_mixed24_f64<>+0xB870(SB)/8, $5659
DATA bitrev_size8192_mixed24_f64<>+0xB878(SB)/8, $7707
DATA bitrev_size8192_mixed24_f64<>+0xB880(SB)/8, $155
DATA bitrev_size8192_mixed24_f64<>+0xB888(SB)/8, $2203
DATA bitrev_size8192_mixed24_f64<>+0xB890(SB)/8, $4251
DATA bitrev_size8192_mixed24_f64<>+0xB898(SB)/8, $6299
DATA bitrev_size8192_mixed24_f64<>+0xB8A0(SB)/8, $667
DATA bitrev_size8192_mixed24_f64<>+0xB8A8(SB)/8, $2715
DATA bitrev_size8192_mixed24_f64<>+0xB8B0(SB)/8, $4763
DATA bitrev_size8192_mixed24_f64<>+0xB8B8(SB)/8, $6811
DATA bitrev_size8192_mixed24_f64<>+0xB8C0(SB)/8, $1179
DATA bitrev_size8192_mixed24_f64<>+0xB8C8(SB)/8, $3227
DATA bitrev_size8192_mixed24_f64<>+0xB8D0(SB)/8, $5275
DATA bitrev_size8192_mixed24_f64<>+0xB8D8(SB)/8, $7323
DATA bitrev_size8192_mixed24_f64<>+0xB8E0(SB)/8, $1691
DATA bitrev_size8192_mixed24_f64<>+0xB8E8(SB)/8, $3739
DATA bitrev_size8192_mixed24_f64<>+0xB8F0(SB)/8, $5787
DATA bitrev_size8192_mixed24_f64<>+0xB8F8(SB)/8, $7835
DATA bitrev_size8192_mixed24_f64<>+0xB900(SB)/8, $283
DATA bitrev_size8192_mixed24_f64<>+0xB908(SB)/8, $2331
DATA bitrev_size8192_mixed24_f64<>+0xB910(SB)/8, $4379
DATA bitrev_size8192_mixed24_f64<>+0xB918(SB)/8, $6427
DATA bitrev_size8192_mixed24_f64<>+0xB920(SB)/8, $795
DATA bitrev_size8192_mixed24_f64<>+0xB928(SB)/8, $2843
DATA bitrev_size8192_mixed24_f64<>+0xB930(SB)/8, $4891
DATA bitrev_size8192_mixed24_f64<>+0xB938(SB)/8, $6939
DATA bitrev_size8192_mixed24_f64<>+0xB940(SB)/8, $1307
DATA bitrev_size8192_mixed24_f64<>+0xB948(SB)/8, $3355
DATA bitrev_size8192_mixed24_f64<>+0xB950(SB)/8, $5403
DATA bitrev_size8192_mixed24_f64<>+0xB958(SB)/8, $7451
DATA bitrev_size8192_mixed24_f64<>+0xB960(SB)/8, $1819
DATA bitrev_size8192_mixed24_f64<>+0xB968(SB)/8, $3867
DATA bitrev_size8192_mixed24_f64<>+0xB970(SB)/8, $5915
DATA bitrev_size8192_mixed24_f64<>+0xB978(SB)/8, $7963
DATA bitrev_size8192_mixed24_f64<>+0xB980(SB)/8, $411
DATA bitrev_size8192_mixed24_f64<>+0xB988(SB)/8, $2459
DATA bitrev_size8192_mixed24_f64<>+0xB990(SB)/8, $4507
DATA bitrev_size8192_mixed24_f64<>+0xB998(SB)/8, $6555
DATA bitrev_size8192_mixed24_f64<>+0xB9A0(SB)/8, $923
DATA bitrev_size8192_mixed24_f64<>+0xB9A8(SB)/8, $2971
DATA bitrev_size8192_mixed24_f64<>+0xB9B0(SB)/8, $5019
DATA bitrev_size8192_mixed24_f64<>+0xB9B8(SB)/8, $7067
DATA bitrev_size8192_mixed24_f64<>+0xB9C0(SB)/8, $1435
DATA bitrev_size8192_mixed24_f64<>+0xB9C8(SB)/8, $3483
DATA bitrev_size8192_mixed24_f64<>+0xB9D0(SB)/8, $5531
DATA bitrev_size8192_mixed24_f64<>+0xB9D8(SB)/8, $7579
DATA bitrev_size8192_mixed24_f64<>+0xB9E0(SB)/8, $1947
DATA bitrev_size8192_mixed24_f64<>+0xB9E8(SB)/8, $3995
DATA bitrev_size8192_mixed24_f64<>+0xB9F0(SB)/8, $6043
DATA bitrev_size8192_mixed24_f64<>+0xB9F8(SB)/8, $8091
DATA bitrev_size8192_mixed24_f64<>+0xBA00(SB)/8, $59
DATA bitrev_size8192_mixed24_f64<>+0xBA08(SB)/8, $2107
DATA bitrev_size8192_mixed24_f64<>+0xBA10(SB)/8, $4155
DATA bitrev_size8192_mixed24_f64<>+0xBA18(SB)/8, $6203
DATA bitrev_size8192_mixed24_f64<>+0xBA20(SB)/8, $571
DATA bitrev_size8192_mixed24_f64<>+0xBA28(SB)/8, $2619
DATA bitrev_size8192_mixed24_f64<>+0xBA30(SB)/8, $4667
DATA bitrev_size8192_mixed24_f64<>+0xBA38(SB)/8, $6715
DATA bitrev_size8192_mixed24_f64<>+0xBA40(SB)/8, $1083
DATA bitrev_size8192_mixed24_f64<>+0xBA48(SB)/8, $3131
DATA bitrev_size8192_mixed24_f64<>+0xBA50(SB)/8, $5179
DATA bitrev_size8192_mixed24_f64<>+0xBA58(SB)/8, $7227
DATA bitrev_size8192_mixed24_f64<>+0xBA60(SB)/8, $1595
DATA bitrev_size8192_mixed24_f64<>+0xBA68(SB)/8, $3643
DATA bitrev_size8192_mixed24_f64<>+0xBA70(SB)/8, $5691
DATA bitrev_size8192_mixed24_f64<>+0xBA78(SB)/8, $7739
DATA bitrev_size8192_mixed24_f64<>+0xBA80(SB)/8, $187
DATA bitrev_size8192_mixed24_f64<>+0xBA88(SB)/8, $2235
DATA bitrev_size8192_mixed24_f64<>+0xBA90(SB)/8, $4283
DATA bitrev_size8192_mixed24_f64<>+0xBA98(SB)/8, $6331
DATA bitrev_size8192_mixed24_f64<>+0xBAA0(SB)/8, $699
DATA bitrev_size8192_mixed24_f64<>+0xBAA8(SB)/8, $2747
DATA bitrev_size8192_mixed24_f64<>+0xBAB0(SB)/8, $4795
DATA bitrev_size8192_mixed24_f64<>+0xBAB8(SB)/8, $6843
DATA bitrev_size8192_mixed24_f64<>+0xBAC0(SB)/8, $1211
DATA bitrev_size8192_mixed24_f64<>+0xBAC8(SB)/8, $3259
DATA bitrev_size8192_mixed24_f64<>+0xBAD0(SB)/8, $5307
DATA bitrev_size8192_mixed24_f64<>+0xBAD8(SB)/8, $7355
DATA bitrev_size8192_mixed24_f64<>+0xBAE0(SB)/8, $1723
DATA bitrev_size8192_mixed24_f64<>+0xBAE8(SB)/8, $3771
DATA bitrev_size8192_mixed24_f64<>+0xBAF0(SB)/8, $5819
DATA bitrev_size8192_mixed24_f64<>+0xBAF8(SB)/8, $7867
DATA bitrev_size8192_mixed24_f64<>+0xBB00(SB)/8, $315
DATA bitrev_size8192_mixed24_f64<>+0xBB08(SB)/8, $2363
DATA bitrev_size8192_mixed24_f64<>+0xBB10(SB)/8, $4411
DATA bitrev_size8192_mixed24_f64<>+0xBB18(SB)/8, $6459
DATA bitrev_size8192_mixed24_f64<>+0xBB20(SB)/8, $827
DATA bitrev_size8192_mixed24_f64<>+0xBB28(SB)/8, $2875
DATA bitrev_size8192_mixed24_f64<>+0xBB30(SB)/8, $4923
DATA bitrev_size8192_mixed24_f64<>+0xBB38(SB)/8, $6971
DATA bitrev_size8192_mixed24_f64<>+0xBB40(SB)/8, $1339
DATA bitrev_size8192_mixed24_f64<>+0xBB48(SB)/8, $3387
DATA bitrev_size8192_mixed24_f64<>+0xBB50(SB)/8, $5435
DATA bitrev_size8192_mixed24_f64<>+0xBB58(SB)/8, $7483
DATA bitrev_size8192_mixed24_f64<>+0xBB60(SB)/8, $1851
DATA bitrev_size8192_mixed24_f64<>+0xBB68(SB)/8, $3899
DATA bitrev_size8192_mixed24_f64<>+0xBB70(SB)/8, $5947
DATA bitrev_size8192_mixed24_f64<>+0xBB78(SB)/8, $7995
DATA bitrev_size8192_mixed24_f64<>+0xBB80(SB)/8, $443
DATA bitrev_size8192_mixed24_f64<>+0xBB88(SB)/8, $2491
DATA bitrev_size8192_mixed24_f64<>+0xBB90(SB)/8, $4539
DATA bitrev_size8192_mixed24_f64<>+0xBB98(SB)/8, $6587
DATA bitrev_size8192_mixed24_f64<>+0xBBA0(SB)/8, $955
DATA bitrev_size8192_mixed24_f64<>+0xBBA8(SB)/8, $3003
DATA bitrev_size8192_mixed24_f64<>+0xBBB0(SB)/8, $5051
DATA bitrev_size8192_mixed24_f64<>+0xBBB8(SB)/8, $7099
DATA bitrev_size8192_mixed24_f64<>+0xBBC0(SB)/8, $1467
DATA bitrev_size8192_mixed24_f64<>+0xBBC8(SB)/8, $3515
DATA bitrev_size8192_mixed24_f64<>+0xBBD0(SB)/8, $5563
DATA bitrev_size8192_mixed24_f64<>+0xBBD8(SB)/8, $7611
DATA bitrev_size8192_mixed24_f64<>+0xBBE0(SB)/8, $1979
DATA bitrev_size8192_mixed24_f64<>+0xBBE8(SB)/8, $4027
DATA bitrev_size8192_mixed24_f64<>+0xBBF0(SB)/8, $6075
DATA bitrev_size8192_mixed24_f64<>+0xBBF8(SB)/8, $8123
DATA bitrev_size8192_mixed24_f64<>+0xBC00(SB)/8, $91
DATA bitrev_size8192_mixed24_f64<>+0xBC08(SB)/8, $2139
DATA bitrev_size8192_mixed24_f64<>+0xBC10(SB)/8, $4187
DATA bitrev_size8192_mixed24_f64<>+0xBC18(SB)/8, $6235
DATA bitrev_size8192_mixed24_f64<>+0xBC20(SB)/8, $603
DATA bitrev_size8192_mixed24_f64<>+0xBC28(SB)/8, $2651
DATA bitrev_size8192_mixed24_f64<>+0xBC30(SB)/8, $4699
DATA bitrev_size8192_mixed24_f64<>+0xBC38(SB)/8, $6747
DATA bitrev_size8192_mixed24_f64<>+0xBC40(SB)/8, $1115
DATA bitrev_size8192_mixed24_f64<>+0xBC48(SB)/8, $3163
DATA bitrev_size8192_mixed24_f64<>+0xBC50(SB)/8, $5211
DATA bitrev_size8192_mixed24_f64<>+0xBC58(SB)/8, $7259
DATA bitrev_size8192_mixed24_f64<>+0xBC60(SB)/8, $1627
DATA bitrev_size8192_mixed24_f64<>+0xBC68(SB)/8, $3675
DATA bitrev_size8192_mixed24_f64<>+0xBC70(SB)/8, $5723
DATA bitrev_size8192_mixed24_f64<>+0xBC78(SB)/8, $7771
DATA bitrev_size8192_mixed24_f64<>+0xBC80(SB)/8, $219
DATA bitrev_size8192_mixed24_f64<>+0xBC88(SB)/8, $2267
DATA bitrev_size8192_mixed24_f64<>+0xBC90(SB)/8, $4315
DATA bitrev_size8192_mixed24_f64<>+0xBC98(SB)/8, $6363
DATA bitrev_size8192_mixed24_f64<>+0xBCA0(SB)/8, $731
DATA bitrev_size8192_mixed24_f64<>+0xBCA8(SB)/8, $2779
DATA bitrev_size8192_mixed24_f64<>+0xBCB0(SB)/8, $4827
DATA bitrev_size8192_mixed24_f64<>+0xBCB8(SB)/8, $6875
DATA bitrev_size8192_mixed24_f64<>+0xBCC0(SB)/8, $1243
DATA bitrev_size8192_mixed24_f64<>+0xBCC8(SB)/8, $3291
DATA bitrev_size8192_mixed24_f64<>+0xBCD0(SB)/8, $5339
DATA bitrev_size8192_mixed24_f64<>+0xBCD8(SB)/8, $7387
DATA bitrev_size8192_mixed24_f64<>+0xBCE0(SB)/8, $1755
DATA bitrev_size8192_mixed24_f64<>+0xBCE8(SB)/8, $3803
DATA bitrev_size8192_mixed24_f64<>+0xBCF0(SB)/8, $5851
DATA bitrev_size8192_mixed24_f64<>+0xBCF8(SB)/8, $7899
DATA bitrev_size8192_mixed24_f64<>+0xBD00(SB)/8, $347
DATA bitrev_size8192_mixed24_f64<>+0xBD08(SB)/8, $2395
DATA bitrev_size8192_mixed24_f64<>+0xBD10(SB)/8, $4443
DATA bitrev_size8192_mixed24_f64<>+0xBD18(SB)/8, $6491
DATA bitrev_size8192_mixed24_f64<>+0xBD20(SB)/8, $859
DATA bitrev_size8192_mixed24_f64<>+0xBD28(SB)/8, $2907
DATA bitrev_size8192_mixed24_f64<>+0xBD30(SB)/8, $4955
DATA bitrev_size8192_mixed24_f64<>+0xBD38(SB)/8, $7003
DATA bitrev_size8192_mixed24_f64<>+0xBD40(SB)/8, $1371
DATA bitrev_size8192_mixed24_f64<>+0xBD48(SB)/8, $3419
DATA bitrev_size8192_mixed24_f64<>+0xBD50(SB)/8, $5467
DATA bitrev_size8192_mixed24_f64<>+0xBD58(SB)/8, $7515
DATA bitrev_size8192_mixed24_f64<>+0xBD60(SB)/8, $1883
DATA bitrev_size8192_mixed24_f64<>+0xBD68(SB)/8, $3931
DATA bitrev_size8192_mixed24_f64<>+0xBD70(SB)/8, $5979
DATA bitrev_size8192_mixed24_f64<>+0xBD78(SB)/8, $8027
DATA bitrev_size8192_mixed24_f64<>+0xBD80(SB)/8, $475
DATA bitrev_size8192_mixed24_f64<>+0xBD88(SB)/8, $2523
DATA bitrev_size8192_mixed24_f64<>+0xBD90(SB)/8, $4571
DATA bitrev_size8192_mixed24_f64<>+0xBD98(SB)/8, $6619
DATA bitrev_size8192_mixed24_f64<>+0xBDA0(SB)/8, $987
DATA bitrev_size8192_mixed24_f64<>+0xBDA8(SB)/8, $3035
DATA bitrev_size8192_mixed24_f64<>+0xBDB0(SB)/8, $5083
DATA bitrev_size8192_mixed24_f64<>+0xBDB8(SB)/8, $7131
DATA bitrev_size8192_mixed24_f64<>+0xBDC0(SB)/8, $1499
DATA bitrev_size8192_mixed24_f64<>+0xBDC8(SB)/8, $3547
DATA bitrev_size8192_mixed24_f64<>+0xBDD0(SB)/8, $5595
DATA bitrev_size8192_mixed24_f64<>+0xBDD8(SB)/8, $7643
DATA bitrev_size8192_mixed24_f64<>+0xBDE0(SB)/8, $2011
DATA bitrev_size8192_mixed24_f64<>+0xBDE8(SB)/8, $4059
DATA bitrev_size8192_mixed24_f64<>+0xBDF0(SB)/8, $6107
DATA bitrev_size8192_mixed24_f64<>+0xBDF8(SB)/8, $8155
DATA bitrev_size8192_mixed24_f64<>+0xBE00(SB)/8, $123
DATA bitrev_size8192_mixed24_f64<>+0xBE08(SB)/8, $2171
DATA bitrev_size8192_mixed24_f64<>+0xBE10(SB)/8, $4219
DATA bitrev_size8192_mixed24_f64<>+0xBE18(SB)/8, $6267
DATA bitrev_size8192_mixed24_f64<>+0xBE20(SB)/8, $635
DATA bitrev_size8192_mixed24_f64<>+0xBE28(SB)/8, $2683
DATA bitrev_size8192_mixed24_f64<>+0xBE30(SB)/8, $4731
DATA bitrev_size8192_mixed24_f64<>+0xBE38(SB)/8, $6779
DATA bitrev_size8192_mixed24_f64<>+0xBE40(SB)/8, $1147
DATA bitrev_size8192_mixed24_f64<>+0xBE48(SB)/8, $3195
DATA bitrev_size8192_mixed24_f64<>+0xBE50(SB)/8, $5243
DATA bitrev_size8192_mixed24_f64<>+0xBE58(SB)/8, $7291
DATA bitrev_size8192_mixed24_f64<>+0xBE60(SB)/8, $1659
DATA bitrev_size8192_mixed24_f64<>+0xBE68(SB)/8, $3707
DATA bitrev_size8192_mixed24_f64<>+0xBE70(SB)/8, $5755
DATA bitrev_size8192_mixed24_f64<>+0xBE78(SB)/8, $7803
DATA bitrev_size8192_mixed24_f64<>+0xBE80(SB)/8, $251
DATA bitrev_size8192_mixed24_f64<>+0xBE88(SB)/8, $2299
DATA bitrev_size8192_mixed24_f64<>+0xBE90(SB)/8, $4347
DATA bitrev_size8192_mixed24_f64<>+0xBE98(SB)/8, $6395
DATA bitrev_size8192_mixed24_f64<>+0xBEA0(SB)/8, $763
DATA bitrev_size8192_mixed24_f64<>+0xBEA8(SB)/8, $2811
DATA bitrev_size8192_mixed24_f64<>+0xBEB0(SB)/8, $4859
DATA bitrev_size8192_mixed24_f64<>+0xBEB8(SB)/8, $6907
DATA bitrev_size8192_mixed24_f64<>+0xBEC0(SB)/8, $1275
DATA bitrev_size8192_mixed24_f64<>+0xBEC8(SB)/8, $3323
DATA bitrev_size8192_mixed24_f64<>+0xBED0(SB)/8, $5371
DATA bitrev_size8192_mixed24_f64<>+0xBED8(SB)/8, $7419
DATA bitrev_size8192_mixed24_f64<>+0xBEE0(SB)/8, $1787
DATA bitrev_size8192_mixed24_f64<>+0xBEE8(SB)/8, $3835
DATA bitrev_size8192_mixed24_f64<>+0xBEF0(SB)/8, $5883
DATA bitrev_size8192_mixed24_f64<>+0xBEF8(SB)/8, $7931
DATA bitrev_size8192_mixed24_f64<>+0xBF00(SB)/8, $379
DATA bitrev_size8192_mixed24_f64<>+0xBF08(SB)/8, $2427
DATA bitrev_size8192_mixed24_f64<>+0xBF10(SB)/8, $4475
DATA bitrev_size8192_mixed24_f64<>+0xBF18(SB)/8, $6523
DATA bitrev_size8192_mixed24_f64<>+0xBF20(SB)/8, $891
DATA bitrev_size8192_mixed24_f64<>+0xBF28(SB)/8, $2939
DATA bitrev_size8192_mixed24_f64<>+0xBF30(SB)/8, $4987
DATA bitrev_size8192_mixed24_f64<>+0xBF38(SB)/8, $7035
DATA bitrev_size8192_mixed24_f64<>+0xBF40(SB)/8, $1403
DATA bitrev_size8192_mixed24_f64<>+0xBF48(SB)/8, $3451
DATA bitrev_size8192_mixed24_f64<>+0xBF50(SB)/8, $5499
DATA bitrev_size8192_mixed24_f64<>+0xBF58(SB)/8, $7547
DATA bitrev_size8192_mixed24_f64<>+0xBF60(SB)/8, $1915
DATA bitrev_size8192_mixed24_f64<>+0xBF68(SB)/8, $3963
DATA bitrev_size8192_mixed24_f64<>+0xBF70(SB)/8, $6011
DATA bitrev_size8192_mixed24_f64<>+0xBF78(SB)/8, $8059
DATA bitrev_size8192_mixed24_f64<>+0xBF80(SB)/8, $507
DATA bitrev_size8192_mixed24_f64<>+0xBF88(SB)/8, $2555
DATA bitrev_size8192_mixed24_f64<>+0xBF90(SB)/8, $4603
DATA bitrev_size8192_mixed24_f64<>+0xBF98(SB)/8, $6651
DATA bitrev_size8192_mixed24_f64<>+0xBFA0(SB)/8, $1019
DATA bitrev_size8192_mixed24_f64<>+0xBFA8(SB)/8, $3067
DATA bitrev_size8192_mixed24_f64<>+0xBFB0(SB)/8, $5115
DATA bitrev_size8192_mixed24_f64<>+0xBFB8(SB)/8, $7163
DATA bitrev_size8192_mixed24_f64<>+0xBFC0(SB)/8, $1531
DATA bitrev_size8192_mixed24_f64<>+0xBFC8(SB)/8, $3579
DATA bitrev_size8192_mixed24_f64<>+0xBFD0(SB)/8, $5627
DATA bitrev_size8192_mixed24_f64<>+0xBFD8(SB)/8, $7675
DATA bitrev_size8192_mixed24_f64<>+0xBFE0(SB)/8, $2043
DATA bitrev_size8192_mixed24_f64<>+0xBFE8(SB)/8, $4091
DATA bitrev_size8192_mixed24_f64<>+0xBFF0(SB)/8, $6139
DATA bitrev_size8192_mixed24_f64<>+0xBFF8(SB)/8, $8187
DATA bitrev_size8192_mixed24_f64<>+0xC000(SB)/8, $5
DATA bitrev_size8192_mixed24_f64<>+0xC008(SB)/8, $2053
DATA bitrev_size8192_mixed24_f64<>+0xC010(SB)/8, $4101
DATA bitrev_size8192_mixed24_f64<>+0xC018(SB)/8, $6149
DATA bitrev_size8192_mixed24_f64<>+0xC020(SB)/8, $517
DATA bitrev_size8192_mixed24_f64<>+0xC028(SB)/8, $2565
DATA bitrev_size8192_mixed24_f64<>+0xC030(SB)/8, $4613
DATA bitrev_size8192_mixed24_f64<>+0xC038(SB)/8, $6661
DATA bitrev_size8192_mixed24_f64<>+0xC040(SB)/8, $1029
DATA bitrev_size8192_mixed24_f64<>+0xC048(SB)/8, $3077
DATA bitrev_size8192_mixed24_f64<>+0xC050(SB)/8, $5125
DATA bitrev_size8192_mixed24_f64<>+0xC058(SB)/8, $7173
DATA bitrev_size8192_mixed24_f64<>+0xC060(SB)/8, $1541
DATA bitrev_size8192_mixed24_f64<>+0xC068(SB)/8, $3589
DATA bitrev_size8192_mixed24_f64<>+0xC070(SB)/8, $5637
DATA bitrev_size8192_mixed24_f64<>+0xC078(SB)/8, $7685
DATA bitrev_size8192_mixed24_f64<>+0xC080(SB)/8, $133
DATA bitrev_size8192_mixed24_f64<>+0xC088(SB)/8, $2181
DATA bitrev_size8192_mixed24_f64<>+0xC090(SB)/8, $4229
DATA bitrev_size8192_mixed24_f64<>+0xC098(SB)/8, $6277
DATA bitrev_size8192_mixed24_f64<>+0xC0A0(SB)/8, $645
DATA bitrev_size8192_mixed24_f64<>+0xC0A8(SB)/8, $2693
DATA bitrev_size8192_mixed24_f64<>+0xC0B0(SB)/8, $4741
DATA bitrev_size8192_mixed24_f64<>+0xC0B8(SB)/8, $6789
DATA bitrev_size8192_mixed24_f64<>+0xC0C0(SB)/8, $1157
DATA bitrev_size8192_mixed24_f64<>+0xC0C8(SB)/8, $3205
DATA bitrev_size8192_mixed24_f64<>+0xC0D0(SB)/8, $5253
DATA bitrev_size8192_mixed24_f64<>+0xC0D8(SB)/8, $7301
DATA bitrev_size8192_mixed24_f64<>+0xC0E0(SB)/8, $1669
DATA bitrev_size8192_mixed24_f64<>+0xC0E8(SB)/8, $3717
DATA bitrev_size8192_mixed24_f64<>+0xC0F0(SB)/8, $5765
DATA bitrev_size8192_mixed24_f64<>+0xC0F8(SB)/8, $7813
DATA bitrev_size8192_mixed24_f64<>+0xC100(SB)/8, $261
DATA bitrev_size8192_mixed24_f64<>+0xC108(SB)/8, $2309
DATA bitrev_size8192_mixed24_f64<>+0xC110(SB)/8, $4357
DATA bitrev_size8192_mixed24_f64<>+0xC118(SB)/8, $6405
DATA bitrev_size8192_mixed24_f64<>+0xC120(SB)/8, $773
DATA bitrev_size8192_mixed24_f64<>+0xC128(SB)/8, $2821
DATA bitrev_size8192_mixed24_f64<>+0xC130(SB)/8, $4869
DATA bitrev_size8192_mixed24_f64<>+0xC138(SB)/8, $6917
DATA bitrev_size8192_mixed24_f64<>+0xC140(SB)/8, $1285
DATA bitrev_size8192_mixed24_f64<>+0xC148(SB)/8, $3333
DATA bitrev_size8192_mixed24_f64<>+0xC150(SB)/8, $5381
DATA bitrev_size8192_mixed24_f64<>+0xC158(SB)/8, $7429
DATA bitrev_size8192_mixed24_f64<>+0xC160(SB)/8, $1797
DATA bitrev_size8192_mixed24_f64<>+0xC168(SB)/8, $3845
DATA bitrev_size8192_mixed24_f64<>+0xC170(SB)/8, $5893
DATA bitrev_size8192_mixed24_f64<>+0xC178(SB)/8, $7941
DATA bitrev_size8192_mixed24_f64<>+0xC180(SB)/8, $389
DATA bitrev_size8192_mixed24_f64<>+0xC188(SB)/8, $2437
DATA bitrev_size8192_mixed24_f64<>+0xC190(SB)/8, $4485
DATA bitrev_size8192_mixed24_f64<>+0xC198(SB)/8, $6533
DATA bitrev_size8192_mixed24_f64<>+0xC1A0(SB)/8, $901
DATA bitrev_size8192_mixed24_f64<>+0xC1A8(SB)/8, $2949
DATA bitrev_size8192_mixed24_f64<>+0xC1B0(SB)/8, $4997
DATA bitrev_size8192_mixed24_f64<>+0xC1B8(SB)/8, $7045
DATA bitrev_size8192_mixed24_f64<>+0xC1C0(SB)/8, $1413
DATA bitrev_size8192_mixed24_f64<>+0xC1C8(SB)/8, $3461
DATA bitrev_size8192_mixed24_f64<>+0xC1D0(SB)/8, $5509
DATA bitrev_size8192_mixed24_f64<>+0xC1D8(SB)/8, $7557
DATA bitrev_size8192_mixed24_f64<>+0xC1E0(SB)/8, $1925
DATA bitrev_size8192_mixed24_f64<>+0xC1E8(SB)/8, $3973
DATA bitrev_size8192_mixed24_f64<>+0xC1F0(SB)/8, $6021
DATA bitrev_size8192_mixed24_f64<>+0xC1F8(SB)/8, $8069
DATA bitrev_size8192_mixed24_f64<>+0xC200(SB)/8, $37
DATA bitrev_size8192_mixed24_f64<>+0xC208(SB)/8, $2085
DATA bitrev_size8192_mixed24_f64<>+0xC210(SB)/8, $4133
DATA bitrev_size8192_mixed24_f64<>+0xC218(SB)/8, $6181
DATA bitrev_size8192_mixed24_f64<>+0xC220(SB)/8, $549
DATA bitrev_size8192_mixed24_f64<>+0xC228(SB)/8, $2597
DATA bitrev_size8192_mixed24_f64<>+0xC230(SB)/8, $4645
DATA bitrev_size8192_mixed24_f64<>+0xC238(SB)/8, $6693
DATA bitrev_size8192_mixed24_f64<>+0xC240(SB)/8, $1061
DATA bitrev_size8192_mixed24_f64<>+0xC248(SB)/8, $3109
DATA bitrev_size8192_mixed24_f64<>+0xC250(SB)/8, $5157
DATA bitrev_size8192_mixed24_f64<>+0xC258(SB)/8, $7205
DATA bitrev_size8192_mixed24_f64<>+0xC260(SB)/8, $1573
DATA bitrev_size8192_mixed24_f64<>+0xC268(SB)/8, $3621
DATA bitrev_size8192_mixed24_f64<>+0xC270(SB)/8, $5669
DATA bitrev_size8192_mixed24_f64<>+0xC278(SB)/8, $7717
DATA bitrev_size8192_mixed24_f64<>+0xC280(SB)/8, $165
DATA bitrev_size8192_mixed24_f64<>+0xC288(SB)/8, $2213
DATA bitrev_size8192_mixed24_f64<>+0xC290(SB)/8, $4261
DATA bitrev_size8192_mixed24_f64<>+0xC298(SB)/8, $6309
DATA bitrev_size8192_mixed24_f64<>+0xC2A0(SB)/8, $677
DATA bitrev_size8192_mixed24_f64<>+0xC2A8(SB)/8, $2725
DATA bitrev_size8192_mixed24_f64<>+0xC2B0(SB)/8, $4773
DATA bitrev_size8192_mixed24_f64<>+0xC2B8(SB)/8, $6821
DATA bitrev_size8192_mixed24_f64<>+0xC2C0(SB)/8, $1189
DATA bitrev_size8192_mixed24_f64<>+0xC2C8(SB)/8, $3237
DATA bitrev_size8192_mixed24_f64<>+0xC2D0(SB)/8, $5285
DATA bitrev_size8192_mixed24_f64<>+0xC2D8(SB)/8, $7333
DATA bitrev_size8192_mixed24_f64<>+0xC2E0(SB)/8, $1701
DATA bitrev_size8192_mixed24_f64<>+0xC2E8(SB)/8, $3749
DATA bitrev_size8192_mixed24_f64<>+0xC2F0(SB)/8, $5797
DATA bitrev_size8192_mixed24_f64<>+0xC2F8(SB)/8, $7845
DATA bitrev_size8192_mixed24_f64<>+0xC300(SB)/8, $293
DATA bitrev_size8192_mixed24_f64<>+0xC308(SB)/8, $2341
DATA bitrev_size8192_mixed24_f64<>+0xC310(SB)/8, $4389
DATA bitrev_size8192_mixed24_f64<>+0xC318(SB)/8, $6437
DATA bitrev_size8192_mixed24_f64<>+0xC320(SB)/8, $805
DATA bitrev_size8192_mixed24_f64<>+0xC328(SB)/8, $2853
DATA bitrev_size8192_mixed24_f64<>+0xC330(SB)/8, $4901
DATA bitrev_size8192_mixed24_f64<>+0xC338(SB)/8, $6949
DATA bitrev_size8192_mixed24_f64<>+0xC340(SB)/8, $1317
DATA bitrev_size8192_mixed24_f64<>+0xC348(SB)/8, $3365
DATA bitrev_size8192_mixed24_f64<>+0xC350(SB)/8, $5413
DATA bitrev_size8192_mixed24_f64<>+0xC358(SB)/8, $7461
DATA bitrev_size8192_mixed24_f64<>+0xC360(SB)/8, $1829
DATA bitrev_size8192_mixed24_f64<>+0xC368(SB)/8, $3877
DATA bitrev_size8192_mixed24_f64<>+0xC370(SB)/8, $5925
DATA bitrev_size8192_mixed24_f64<>+0xC378(SB)/8, $7973
DATA bitrev_size8192_mixed24_f64<>+0xC380(SB)/8, $421
DATA bitrev_size8192_mixed24_f64<>+0xC388(SB)/8, $2469
DATA bitrev_size8192_mixed24_f64<>+0xC390(SB)/8, $4517
DATA bitrev_size8192_mixed24_f64<>+0xC398(SB)/8, $6565
DATA bitrev_size8192_mixed24_f64<>+0xC3A0(SB)/8, $933
DATA bitrev_size8192_mixed24_f64<>+0xC3A8(SB)/8, $2981
DATA bitrev_size8192_mixed24_f64<>+0xC3B0(SB)/8, $5029
DATA bitrev_size8192_mixed24_f64<>+0xC3B8(SB)/8, $7077
DATA bitrev_size8192_mixed24_f64<>+0xC3C0(SB)/8, $1445
DATA bitrev_size8192_mixed24_f64<>+0xC3C8(SB)/8, $3493
DATA bitrev_size8192_mixed24_f64<>+0xC3D0(SB)/8, $5541
DATA bitrev_size8192_mixed24_f64<>+0xC3D8(SB)/8, $7589
DATA bitrev_size8192_mixed24_f64<>+0xC3E0(SB)/8, $1957
DATA bitrev_size8192_mixed24_f64<>+0xC3E8(SB)/8, $4005
DATA bitrev_size8192_mixed24_f64<>+0xC3F0(SB)/8, $6053
DATA bitrev_size8192_mixed24_f64<>+0xC3F8(SB)/8, $8101
DATA bitrev_size8192_mixed24_f64<>+0xC400(SB)/8, $69
DATA bitrev_size8192_mixed24_f64<>+0xC408(SB)/8, $2117
DATA bitrev_size8192_mixed24_f64<>+0xC410(SB)/8, $4165
DATA bitrev_size8192_mixed24_f64<>+0xC418(SB)/8, $6213
DATA bitrev_size8192_mixed24_f64<>+0xC420(SB)/8, $581
DATA bitrev_size8192_mixed24_f64<>+0xC428(SB)/8, $2629
DATA bitrev_size8192_mixed24_f64<>+0xC430(SB)/8, $4677
DATA bitrev_size8192_mixed24_f64<>+0xC438(SB)/8, $6725
DATA bitrev_size8192_mixed24_f64<>+0xC440(SB)/8, $1093
DATA bitrev_size8192_mixed24_f64<>+0xC448(SB)/8, $3141
DATA bitrev_size8192_mixed24_f64<>+0xC450(SB)/8, $5189
DATA bitrev_size8192_mixed24_f64<>+0xC458(SB)/8, $7237
DATA bitrev_size8192_mixed24_f64<>+0xC460(SB)/8, $1605
DATA bitrev_size8192_mixed24_f64<>+0xC468(SB)/8, $3653
DATA bitrev_size8192_mixed24_f64<>+0xC470(SB)/8, $5701
DATA bitrev_size8192_mixed24_f64<>+0xC478(SB)/8, $7749
DATA bitrev_size8192_mixed24_f64<>+0xC480(SB)/8, $197
DATA bitrev_size8192_mixed24_f64<>+0xC488(SB)/8, $2245
DATA bitrev_size8192_mixed24_f64<>+0xC490(SB)/8, $4293
DATA bitrev_size8192_mixed24_f64<>+0xC498(SB)/8, $6341
DATA bitrev_size8192_mixed24_f64<>+0xC4A0(SB)/8, $709
DATA bitrev_size8192_mixed24_f64<>+0xC4A8(SB)/8, $2757
DATA bitrev_size8192_mixed24_f64<>+0xC4B0(SB)/8, $4805
DATA bitrev_size8192_mixed24_f64<>+0xC4B8(SB)/8, $6853
DATA bitrev_size8192_mixed24_f64<>+0xC4C0(SB)/8, $1221
DATA bitrev_size8192_mixed24_f64<>+0xC4C8(SB)/8, $3269
DATA bitrev_size8192_mixed24_f64<>+0xC4D0(SB)/8, $5317
DATA bitrev_size8192_mixed24_f64<>+0xC4D8(SB)/8, $7365
DATA bitrev_size8192_mixed24_f64<>+0xC4E0(SB)/8, $1733
DATA bitrev_size8192_mixed24_f64<>+0xC4E8(SB)/8, $3781
DATA bitrev_size8192_mixed24_f64<>+0xC4F0(SB)/8, $5829
DATA bitrev_size8192_mixed24_f64<>+0xC4F8(SB)/8, $7877
DATA bitrev_size8192_mixed24_f64<>+0xC500(SB)/8, $325
DATA bitrev_size8192_mixed24_f64<>+0xC508(SB)/8, $2373
DATA bitrev_size8192_mixed24_f64<>+0xC510(SB)/8, $4421
DATA bitrev_size8192_mixed24_f64<>+0xC518(SB)/8, $6469
DATA bitrev_size8192_mixed24_f64<>+0xC520(SB)/8, $837
DATA bitrev_size8192_mixed24_f64<>+0xC528(SB)/8, $2885
DATA bitrev_size8192_mixed24_f64<>+0xC530(SB)/8, $4933
DATA bitrev_size8192_mixed24_f64<>+0xC538(SB)/8, $6981
DATA bitrev_size8192_mixed24_f64<>+0xC540(SB)/8, $1349
DATA bitrev_size8192_mixed24_f64<>+0xC548(SB)/8, $3397
DATA bitrev_size8192_mixed24_f64<>+0xC550(SB)/8, $5445
DATA bitrev_size8192_mixed24_f64<>+0xC558(SB)/8, $7493
DATA bitrev_size8192_mixed24_f64<>+0xC560(SB)/8, $1861
DATA bitrev_size8192_mixed24_f64<>+0xC568(SB)/8, $3909
DATA bitrev_size8192_mixed24_f64<>+0xC570(SB)/8, $5957
DATA bitrev_size8192_mixed24_f64<>+0xC578(SB)/8, $8005
DATA bitrev_size8192_mixed24_f64<>+0xC580(SB)/8, $453
DATA bitrev_size8192_mixed24_f64<>+0xC588(SB)/8, $2501
DATA bitrev_size8192_mixed24_f64<>+0xC590(SB)/8, $4549
DATA bitrev_size8192_mixed24_f64<>+0xC598(SB)/8, $6597
DATA bitrev_size8192_mixed24_f64<>+0xC5A0(SB)/8, $965
DATA bitrev_size8192_mixed24_f64<>+0xC5A8(SB)/8, $3013
DATA bitrev_size8192_mixed24_f64<>+0xC5B0(SB)/8, $5061
DATA bitrev_size8192_mixed24_f64<>+0xC5B8(SB)/8, $7109
DATA bitrev_size8192_mixed24_f64<>+0xC5C0(SB)/8, $1477
DATA bitrev_size8192_mixed24_f64<>+0xC5C8(SB)/8, $3525
DATA bitrev_size8192_mixed24_f64<>+0xC5D0(SB)/8, $5573
DATA bitrev_size8192_mixed24_f64<>+0xC5D8(SB)/8, $7621
DATA bitrev_size8192_mixed24_f64<>+0xC5E0(SB)/8, $1989
DATA bitrev_size8192_mixed24_f64<>+0xC5E8(SB)/8, $4037
DATA bitrev_size8192_mixed24_f64<>+0xC5F0(SB)/8, $6085
DATA bitrev_size8192_mixed24_f64<>+0xC5F8(SB)/8, $8133
DATA bitrev_size8192_mixed24_f64<>+0xC600(SB)/8, $101
DATA bitrev_size8192_mixed24_f64<>+0xC608(SB)/8, $2149
DATA bitrev_size8192_mixed24_f64<>+0xC610(SB)/8, $4197
DATA bitrev_size8192_mixed24_f64<>+0xC618(SB)/8, $6245
DATA bitrev_size8192_mixed24_f64<>+0xC620(SB)/8, $613
DATA bitrev_size8192_mixed24_f64<>+0xC628(SB)/8, $2661
DATA bitrev_size8192_mixed24_f64<>+0xC630(SB)/8, $4709
DATA bitrev_size8192_mixed24_f64<>+0xC638(SB)/8, $6757
DATA bitrev_size8192_mixed24_f64<>+0xC640(SB)/8, $1125
DATA bitrev_size8192_mixed24_f64<>+0xC648(SB)/8, $3173
DATA bitrev_size8192_mixed24_f64<>+0xC650(SB)/8, $5221
DATA bitrev_size8192_mixed24_f64<>+0xC658(SB)/8, $7269
DATA bitrev_size8192_mixed24_f64<>+0xC660(SB)/8, $1637
DATA bitrev_size8192_mixed24_f64<>+0xC668(SB)/8, $3685
DATA bitrev_size8192_mixed24_f64<>+0xC670(SB)/8, $5733
DATA bitrev_size8192_mixed24_f64<>+0xC678(SB)/8, $7781
DATA bitrev_size8192_mixed24_f64<>+0xC680(SB)/8, $229
DATA bitrev_size8192_mixed24_f64<>+0xC688(SB)/8, $2277
DATA bitrev_size8192_mixed24_f64<>+0xC690(SB)/8, $4325
DATA bitrev_size8192_mixed24_f64<>+0xC698(SB)/8, $6373
DATA bitrev_size8192_mixed24_f64<>+0xC6A0(SB)/8, $741
DATA bitrev_size8192_mixed24_f64<>+0xC6A8(SB)/8, $2789
DATA bitrev_size8192_mixed24_f64<>+0xC6B0(SB)/8, $4837
DATA bitrev_size8192_mixed24_f64<>+0xC6B8(SB)/8, $6885
DATA bitrev_size8192_mixed24_f64<>+0xC6C0(SB)/8, $1253
DATA bitrev_size8192_mixed24_f64<>+0xC6C8(SB)/8, $3301
DATA bitrev_size8192_mixed24_f64<>+0xC6D0(SB)/8, $5349
DATA bitrev_size8192_mixed24_f64<>+0xC6D8(SB)/8, $7397
DATA bitrev_size8192_mixed24_f64<>+0xC6E0(SB)/8, $1765
DATA bitrev_size8192_mixed24_f64<>+0xC6E8(SB)/8, $3813
DATA bitrev_size8192_mixed24_f64<>+0xC6F0(SB)/8, $5861
DATA bitrev_size8192_mixed24_f64<>+0xC6F8(SB)/8, $7909
DATA bitrev_size8192_mixed24_f64<>+0xC700(SB)/8, $357
DATA bitrev_size8192_mixed24_f64<>+0xC708(SB)/8, $2405
DATA bitrev_size8192_mixed24_f64<>+0xC710(SB)/8, $4453
DATA bitrev_size8192_mixed24_f64<>+0xC718(SB)/8, $6501
DATA bitrev_size8192_mixed24_f64<>+0xC720(SB)/8, $869
DATA bitrev_size8192_mixed24_f64<>+0xC728(SB)/8, $2917
DATA bitrev_size8192_mixed24_f64<>+0xC730(SB)/8, $4965
DATA bitrev_size8192_mixed24_f64<>+0xC738(SB)/8, $7013
DATA bitrev_size8192_mixed24_f64<>+0xC740(SB)/8, $1381
DATA bitrev_size8192_mixed24_f64<>+0xC748(SB)/8, $3429
DATA bitrev_size8192_mixed24_f64<>+0xC750(SB)/8, $5477
DATA bitrev_size8192_mixed24_f64<>+0xC758(SB)/8, $7525
DATA bitrev_size8192_mixed24_f64<>+0xC760(SB)/8, $1893
DATA bitrev_size8192_mixed24_f64<>+0xC768(SB)/8, $3941
DATA bitrev_size8192_mixed24_f64<>+0xC770(SB)/8, $5989
DATA bitrev_size8192_mixed24_f64<>+0xC778(SB)/8, $8037
DATA bitrev_size8192_mixed24_f64<>+0xC780(SB)/8, $485
DATA bitrev_size8192_mixed24_f64<>+0xC788(SB)/8, $2533
DATA bitrev_size8192_mixed24_f64<>+0xC790(SB)/8, $4581
DATA bitrev_size8192_mixed24_f64<>+0xC798(SB)/8, $6629
DATA bitrev_size8192_mixed24_f64<>+0xC7A0(SB)/8, $997
DATA bitrev_size8192_mixed24_f64<>+0xC7A8(SB)/8, $3045
DATA bitrev_size8192_mixed24_f64<>+0xC7B0(SB)/8, $5093
DATA bitrev_size8192_mixed24_f64<>+0xC7B8(SB)/8, $7141
DATA bitrev_size8192_mixed24_f64<>+0xC7C0(SB)/8, $1509
DATA bitrev_size8192_mixed24_f64<>+0xC7C8(SB)/8, $3557
DATA bitrev_size8192_mixed24_f64<>+0xC7D0(SB)/8, $5605
DATA bitrev_size8192_mixed24_f64<>+0xC7D8(SB)/8, $7653
DATA bitrev_size8192_mixed24_f64<>+0xC7E0(SB)/8, $2021
DATA bitrev_size8192_mixed24_f64<>+0xC7E8(SB)/8, $4069
DATA bitrev_size8192_mixed24_f64<>+0xC7F0(SB)/8, $6117
DATA bitrev_size8192_mixed24_f64<>+0xC7F8(SB)/8, $8165
DATA bitrev_size8192_mixed24_f64<>+0xC800(SB)/8, $13
DATA bitrev_size8192_mixed24_f64<>+0xC808(SB)/8, $2061
DATA bitrev_size8192_mixed24_f64<>+0xC810(SB)/8, $4109
DATA bitrev_size8192_mixed24_f64<>+0xC818(SB)/8, $6157
DATA bitrev_size8192_mixed24_f64<>+0xC820(SB)/8, $525
DATA bitrev_size8192_mixed24_f64<>+0xC828(SB)/8, $2573
DATA bitrev_size8192_mixed24_f64<>+0xC830(SB)/8, $4621
DATA bitrev_size8192_mixed24_f64<>+0xC838(SB)/8, $6669
DATA bitrev_size8192_mixed24_f64<>+0xC840(SB)/8, $1037
DATA bitrev_size8192_mixed24_f64<>+0xC848(SB)/8, $3085
DATA bitrev_size8192_mixed24_f64<>+0xC850(SB)/8, $5133
DATA bitrev_size8192_mixed24_f64<>+0xC858(SB)/8, $7181
DATA bitrev_size8192_mixed24_f64<>+0xC860(SB)/8, $1549
DATA bitrev_size8192_mixed24_f64<>+0xC868(SB)/8, $3597
DATA bitrev_size8192_mixed24_f64<>+0xC870(SB)/8, $5645
DATA bitrev_size8192_mixed24_f64<>+0xC878(SB)/8, $7693
DATA bitrev_size8192_mixed24_f64<>+0xC880(SB)/8, $141
DATA bitrev_size8192_mixed24_f64<>+0xC888(SB)/8, $2189
DATA bitrev_size8192_mixed24_f64<>+0xC890(SB)/8, $4237
DATA bitrev_size8192_mixed24_f64<>+0xC898(SB)/8, $6285
DATA bitrev_size8192_mixed24_f64<>+0xC8A0(SB)/8, $653
DATA bitrev_size8192_mixed24_f64<>+0xC8A8(SB)/8, $2701
DATA bitrev_size8192_mixed24_f64<>+0xC8B0(SB)/8, $4749
DATA bitrev_size8192_mixed24_f64<>+0xC8B8(SB)/8, $6797
DATA bitrev_size8192_mixed24_f64<>+0xC8C0(SB)/8, $1165
DATA bitrev_size8192_mixed24_f64<>+0xC8C8(SB)/8, $3213
DATA bitrev_size8192_mixed24_f64<>+0xC8D0(SB)/8, $5261
DATA bitrev_size8192_mixed24_f64<>+0xC8D8(SB)/8, $7309
DATA bitrev_size8192_mixed24_f64<>+0xC8E0(SB)/8, $1677
DATA bitrev_size8192_mixed24_f64<>+0xC8E8(SB)/8, $3725
DATA bitrev_size8192_mixed24_f64<>+0xC8F0(SB)/8, $5773
DATA bitrev_size8192_mixed24_f64<>+0xC8F8(SB)/8, $7821
DATA bitrev_size8192_mixed24_f64<>+0xC900(SB)/8, $269
DATA bitrev_size8192_mixed24_f64<>+0xC908(SB)/8, $2317
DATA bitrev_size8192_mixed24_f64<>+0xC910(SB)/8, $4365
DATA bitrev_size8192_mixed24_f64<>+0xC918(SB)/8, $6413
DATA bitrev_size8192_mixed24_f64<>+0xC920(SB)/8, $781
DATA bitrev_size8192_mixed24_f64<>+0xC928(SB)/8, $2829
DATA bitrev_size8192_mixed24_f64<>+0xC930(SB)/8, $4877
DATA bitrev_size8192_mixed24_f64<>+0xC938(SB)/8, $6925
DATA bitrev_size8192_mixed24_f64<>+0xC940(SB)/8, $1293
DATA bitrev_size8192_mixed24_f64<>+0xC948(SB)/8, $3341
DATA bitrev_size8192_mixed24_f64<>+0xC950(SB)/8, $5389
DATA bitrev_size8192_mixed24_f64<>+0xC958(SB)/8, $7437
DATA bitrev_size8192_mixed24_f64<>+0xC960(SB)/8, $1805
DATA bitrev_size8192_mixed24_f64<>+0xC968(SB)/8, $3853
DATA bitrev_size8192_mixed24_f64<>+0xC970(SB)/8, $5901
DATA bitrev_size8192_mixed24_f64<>+0xC978(SB)/8, $7949
DATA bitrev_size8192_mixed24_f64<>+0xC980(SB)/8, $397
DATA bitrev_size8192_mixed24_f64<>+0xC988(SB)/8, $2445
DATA bitrev_size8192_mixed24_f64<>+0xC990(SB)/8, $4493
DATA bitrev_size8192_mixed24_f64<>+0xC998(SB)/8, $6541
DATA bitrev_size8192_mixed24_f64<>+0xC9A0(SB)/8, $909
DATA bitrev_size8192_mixed24_f64<>+0xC9A8(SB)/8, $2957
DATA bitrev_size8192_mixed24_f64<>+0xC9B0(SB)/8, $5005
DATA bitrev_size8192_mixed24_f64<>+0xC9B8(SB)/8, $7053
DATA bitrev_size8192_mixed24_f64<>+0xC9C0(SB)/8, $1421
DATA bitrev_size8192_mixed24_f64<>+0xC9C8(SB)/8, $3469
DATA bitrev_size8192_mixed24_f64<>+0xC9D0(SB)/8, $5517
DATA bitrev_size8192_mixed24_f64<>+0xC9D8(SB)/8, $7565
DATA bitrev_size8192_mixed24_f64<>+0xC9E0(SB)/8, $1933
DATA bitrev_size8192_mixed24_f64<>+0xC9E8(SB)/8, $3981
DATA bitrev_size8192_mixed24_f64<>+0xC9F0(SB)/8, $6029
DATA bitrev_size8192_mixed24_f64<>+0xC9F8(SB)/8, $8077
DATA bitrev_size8192_mixed24_f64<>+0xCA00(SB)/8, $45
DATA bitrev_size8192_mixed24_f64<>+0xCA08(SB)/8, $2093
DATA bitrev_size8192_mixed24_f64<>+0xCA10(SB)/8, $4141
DATA bitrev_size8192_mixed24_f64<>+0xCA18(SB)/8, $6189
DATA bitrev_size8192_mixed24_f64<>+0xCA20(SB)/8, $557
DATA bitrev_size8192_mixed24_f64<>+0xCA28(SB)/8, $2605
DATA bitrev_size8192_mixed24_f64<>+0xCA30(SB)/8, $4653
DATA bitrev_size8192_mixed24_f64<>+0xCA38(SB)/8, $6701
DATA bitrev_size8192_mixed24_f64<>+0xCA40(SB)/8, $1069
DATA bitrev_size8192_mixed24_f64<>+0xCA48(SB)/8, $3117
DATA bitrev_size8192_mixed24_f64<>+0xCA50(SB)/8, $5165
DATA bitrev_size8192_mixed24_f64<>+0xCA58(SB)/8, $7213
DATA bitrev_size8192_mixed24_f64<>+0xCA60(SB)/8, $1581
DATA bitrev_size8192_mixed24_f64<>+0xCA68(SB)/8, $3629
DATA bitrev_size8192_mixed24_f64<>+0xCA70(SB)/8, $5677
DATA bitrev_size8192_mixed24_f64<>+0xCA78(SB)/8, $7725
DATA bitrev_size8192_mixed24_f64<>+0xCA80(SB)/8, $173
DATA bitrev_size8192_mixed24_f64<>+0xCA88(SB)/8, $2221
DATA bitrev_size8192_mixed24_f64<>+0xCA90(SB)/8, $4269
DATA bitrev_size8192_mixed24_f64<>+0xCA98(SB)/8, $6317
DATA bitrev_size8192_mixed24_f64<>+0xCAA0(SB)/8, $685
DATA bitrev_size8192_mixed24_f64<>+0xCAA8(SB)/8, $2733
DATA bitrev_size8192_mixed24_f64<>+0xCAB0(SB)/8, $4781
DATA bitrev_size8192_mixed24_f64<>+0xCAB8(SB)/8, $6829
DATA bitrev_size8192_mixed24_f64<>+0xCAC0(SB)/8, $1197
DATA bitrev_size8192_mixed24_f64<>+0xCAC8(SB)/8, $3245
DATA bitrev_size8192_mixed24_f64<>+0xCAD0(SB)/8, $5293
DATA bitrev_size8192_mixed24_f64<>+0xCAD8(SB)/8, $7341
DATA bitrev_size8192_mixed24_f64<>+0xCAE0(SB)/8, $1709
DATA bitrev_size8192_mixed24_f64<>+0xCAE8(SB)/8, $3757
DATA bitrev_size8192_mixed24_f64<>+0xCAF0(SB)/8, $5805
DATA bitrev_size8192_mixed24_f64<>+0xCAF8(SB)/8, $7853
DATA bitrev_size8192_mixed24_f64<>+0xCB00(SB)/8, $301
DATA bitrev_size8192_mixed24_f64<>+0xCB08(SB)/8, $2349
DATA bitrev_size8192_mixed24_f64<>+0xCB10(SB)/8, $4397
DATA bitrev_size8192_mixed24_f64<>+0xCB18(SB)/8, $6445
DATA bitrev_size8192_mixed24_f64<>+0xCB20(SB)/8, $813
DATA bitrev_size8192_mixed24_f64<>+0xCB28(SB)/8, $2861
DATA bitrev_size8192_mixed24_f64<>+0xCB30(SB)/8, $4909
DATA bitrev_size8192_mixed24_f64<>+0xCB38(SB)/8, $6957
DATA bitrev_size8192_mixed24_f64<>+0xCB40(SB)/8, $1325
DATA bitrev_size8192_mixed24_f64<>+0xCB48(SB)/8, $3373
DATA bitrev_size8192_mixed24_f64<>+0xCB50(SB)/8, $5421
DATA bitrev_size8192_mixed24_f64<>+0xCB58(SB)/8, $7469
DATA bitrev_size8192_mixed24_f64<>+0xCB60(SB)/8, $1837
DATA bitrev_size8192_mixed24_f64<>+0xCB68(SB)/8, $3885
DATA bitrev_size8192_mixed24_f64<>+0xCB70(SB)/8, $5933
DATA bitrev_size8192_mixed24_f64<>+0xCB78(SB)/8, $7981
DATA bitrev_size8192_mixed24_f64<>+0xCB80(SB)/8, $429
DATA bitrev_size8192_mixed24_f64<>+0xCB88(SB)/8, $2477
DATA bitrev_size8192_mixed24_f64<>+0xCB90(SB)/8, $4525
DATA bitrev_size8192_mixed24_f64<>+0xCB98(SB)/8, $6573
DATA bitrev_size8192_mixed24_f64<>+0xCBA0(SB)/8, $941
DATA bitrev_size8192_mixed24_f64<>+0xCBA8(SB)/8, $2989
DATA bitrev_size8192_mixed24_f64<>+0xCBB0(SB)/8, $5037
DATA bitrev_size8192_mixed24_f64<>+0xCBB8(SB)/8, $7085
DATA bitrev_size8192_mixed24_f64<>+0xCBC0(SB)/8, $1453
DATA bitrev_size8192_mixed24_f64<>+0xCBC8(SB)/8, $3501
DATA bitrev_size8192_mixed24_f64<>+0xCBD0(SB)/8, $5549
DATA bitrev_size8192_mixed24_f64<>+0xCBD8(SB)/8, $7597
DATA bitrev_size8192_mixed24_f64<>+0xCBE0(SB)/8, $1965
DATA bitrev_size8192_mixed24_f64<>+0xCBE8(SB)/8, $4013
DATA bitrev_size8192_mixed24_f64<>+0xCBF0(SB)/8, $6061
DATA bitrev_size8192_mixed24_f64<>+0xCBF8(SB)/8, $8109
DATA bitrev_size8192_mixed24_f64<>+0xCC00(SB)/8, $77
DATA bitrev_size8192_mixed24_f64<>+0xCC08(SB)/8, $2125
DATA bitrev_size8192_mixed24_f64<>+0xCC10(SB)/8, $4173
DATA bitrev_size8192_mixed24_f64<>+0xCC18(SB)/8, $6221
DATA bitrev_size8192_mixed24_f64<>+0xCC20(SB)/8, $589
DATA bitrev_size8192_mixed24_f64<>+0xCC28(SB)/8, $2637
DATA bitrev_size8192_mixed24_f64<>+0xCC30(SB)/8, $4685
DATA bitrev_size8192_mixed24_f64<>+0xCC38(SB)/8, $6733
DATA bitrev_size8192_mixed24_f64<>+0xCC40(SB)/8, $1101
DATA bitrev_size8192_mixed24_f64<>+0xCC48(SB)/8, $3149
DATA bitrev_size8192_mixed24_f64<>+0xCC50(SB)/8, $5197
DATA bitrev_size8192_mixed24_f64<>+0xCC58(SB)/8, $7245
DATA bitrev_size8192_mixed24_f64<>+0xCC60(SB)/8, $1613
DATA bitrev_size8192_mixed24_f64<>+0xCC68(SB)/8, $3661
DATA bitrev_size8192_mixed24_f64<>+0xCC70(SB)/8, $5709
DATA bitrev_size8192_mixed24_f64<>+0xCC78(SB)/8, $7757
DATA bitrev_size8192_mixed24_f64<>+0xCC80(SB)/8, $205
DATA bitrev_size8192_mixed24_f64<>+0xCC88(SB)/8, $2253
DATA bitrev_size8192_mixed24_f64<>+0xCC90(SB)/8, $4301
DATA bitrev_size8192_mixed24_f64<>+0xCC98(SB)/8, $6349
DATA bitrev_size8192_mixed24_f64<>+0xCCA0(SB)/8, $717
DATA bitrev_size8192_mixed24_f64<>+0xCCA8(SB)/8, $2765
DATA bitrev_size8192_mixed24_f64<>+0xCCB0(SB)/8, $4813
DATA bitrev_size8192_mixed24_f64<>+0xCCB8(SB)/8, $6861
DATA bitrev_size8192_mixed24_f64<>+0xCCC0(SB)/8, $1229
DATA bitrev_size8192_mixed24_f64<>+0xCCC8(SB)/8, $3277
DATA bitrev_size8192_mixed24_f64<>+0xCCD0(SB)/8, $5325
DATA bitrev_size8192_mixed24_f64<>+0xCCD8(SB)/8, $7373
DATA bitrev_size8192_mixed24_f64<>+0xCCE0(SB)/8, $1741
DATA bitrev_size8192_mixed24_f64<>+0xCCE8(SB)/8, $3789
DATA bitrev_size8192_mixed24_f64<>+0xCCF0(SB)/8, $5837
DATA bitrev_size8192_mixed24_f64<>+0xCCF8(SB)/8, $7885
DATA bitrev_size8192_mixed24_f64<>+0xCD00(SB)/8, $333
DATA bitrev_size8192_mixed24_f64<>+0xCD08(SB)/8, $2381
DATA bitrev_size8192_mixed24_f64<>+0xCD10(SB)/8, $4429
DATA bitrev_size8192_mixed24_f64<>+0xCD18(SB)/8, $6477
DATA bitrev_size8192_mixed24_f64<>+0xCD20(SB)/8, $845
DATA bitrev_size8192_mixed24_f64<>+0xCD28(SB)/8, $2893
DATA bitrev_size8192_mixed24_f64<>+0xCD30(SB)/8, $4941
DATA bitrev_size8192_mixed24_f64<>+0xCD38(SB)/8, $6989
DATA bitrev_size8192_mixed24_f64<>+0xCD40(SB)/8, $1357
DATA bitrev_size8192_mixed24_f64<>+0xCD48(SB)/8, $3405
DATA bitrev_size8192_mixed24_f64<>+0xCD50(SB)/8, $5453
DATA bitrev_size8192_mixed24_f64<>+0xCD58(SB)/8, $7501
DATA bitrev_size8192_mixed24_f64<>+0xCD60(SB)/8, $1869
DATA bitrev_size8192_mixed24_f64<>+0xCD68(SB)/8, $3917
DATA bitrev_size8192_mixed24_f64<>+0xCD70(SB)/8, $5965
DATA bitrev_size8192_mixed24_f64<>+0xCD78(SB)/8, $8013
DATA bitrev_size8192_mixed24_f64<>+0xCD80(SB)/8, $461
DATA bitrev_size8192_mixed24_f64<>+0xCD88(SB)/8, $2509
DATA bitrev_size8192_mixed24_f64<>+0xCD90(SB)/8, $4557
DATA bitrev_size8192_mixed24_f64<>+0xCD98(SB)/8, $6605
DATA bitrev_size8192_mixed24_f64<>+0xCDA0(SB)/8, $973
DATA bitrev_size8192_mixed24_f64<>+0xCDA8(SB)/8, $3021
DATA bitrev_size8192_mixed24_f64<>+0xCDB0(SB)/8, $5069
DATA bitrev_size8192_mixed24_f64<>+0xCDB8(SB)/8, $7117
DATA bitrev_size8192_mixed24_f64<>+0xCDC0(SB)/8, $1485
DATA bitrev_size8192_mixed24_f64<>+0xCDC8(SB)/8, $3533
DATA bitrev_size8192_mixed24_f64<>+0xCDD0(SB)/8, $5581
DATA bitrev_size8192_mixed24_f64<>+0xCDD8(SB)/8, $7629
DATA bitrev_size8192_mixed24_f64<>+0xCDE0(SB)/8, $1997
DATA bitrev_size8192_mixed24_f64<>+0xCDE8(SB)/8, $4045
DATA bitrev_size8192_mixed24_f64<>+0xCDF0(SB)/8, $6093
DATA bitrev_size8192_mixed24_f64<>+0xCDF8(SB)/8, $8141
DATA bitrev_size8192_mixed24_f64<>+0xCE00(SB)/8, $109
DATA bitrev_size8192_mixed24_f64<>+0xCE08(SB)/8, $2157
DATA bitrev_size8192_mixed24_f64<>+0xCE10(SB)/8, $4205
DATA bitrev_size8192_mixed24_f64<>+0xCE18(SB)/8, $6253
DATA bitrev_size8192_mixed24_f64<>+0xCE20(SB)/8, $621
DATA bitrev_size8192_mixed24_f64<>+0xCE28(SB)/8, $2669
DATA bitrev_size8192_mixed24_f64<>+0xCE30(SB)/8, $4717
DATA bitrev_size8192_mixed24_f64<>+0xCE38(SB)/8, $6765
DATA bitrev_size8192_mixed24_f64<>+0xCE40(SB)/8, $1133
DATA bitrev_size8192_mixed24_f64<>+0xCE48(SB)/8, $3181
DATA bitrev_size8192_mixed24_f64<>+0xCE50(SB)/8, $5229
DATA bitrev_size8192_mixed24_f64<>+0xCE58(SB)/8, $7277
DATA bitrev_size8192_mixed24_f64<>+0xCE60(SB)/8, $1645
DATA bitrev_size8192_mixed24_f64<>+0xCE68(SB)/8, $3693
DATA bitrev_size8192_mixed24_f64<>+0xCE70(SB)/8, $5741
DATA bitrev_size8192_mixed24_f64<>+0xCE78(SB)/8, $7789
DATA bitrev_size8192_mixed24_f64<>+0xCE80(SB)/8, $237
DATA bitrev_size8192_mixed24_f64<>+0xCE88(SB)/8, $2285
DATA bitrev_size8192_mixed24_f64<>+0xCE90(SB)/8, $4333
DATA bitrev_size8192_mixed24_f64<>+0xCE98(SB)/8, $6381
DATA bitrev_size8192_mixed24_f64<>+0xCEA0(SB)/8, $749
DATA bitrev_size8192_mixed24_f64<>+0xCEA8(SB)/8, $2797
DATA bitrev_size8192_mixed24_f64<>+0xCEB0(SB)/8, $4845
DATA bitrev_size8192_mixed24_f64<>+0xCEB8(SB)/8, $6893
DATA bitrev_size8192_mixed24_f64<>+0xCEC0(SB)/8, $1261
DATA bitrev_size8192_mixed24_f64<>+0xCEC8(SB)/8, $3309
DATA bitrev_size8192_mixed24_f64<>+0xCED0(SB)/8, $5357
DATA bitrev_size8192_mixed24_f64<>+0xCED8(SB)/8, $7405
DATA bitrev_size8192_mixed24_f64<>+0xCEE0(SB)/8, $1773
DATA bitrev_size8192_mixed24_f64<>+0xCEE8(SB)/8, $3821
DATA bitrev_size8192_mixed24_f64<>+0xCEF0(SB)/8, $5869
DATA bitrev_size8192_mixed24_f64<>+0xCEF8(SB)/8, $7917
DATA bitrev_size8192_mixed24_f64<>+0xCF00(SB)/8, $365
DATA bitrev_size8192_mixed24_f64<>+0xCF08(SB)/8, $2413
DATA bitrev_size8192_mixed24_f64<>+0xCF10(SB)/8, $4461
DATA bitrev_size8192_mixed24_f64<>+0xCF18(SB)/8, $6509
DATA bitrev_size8192_mixed24_f64<>+0xCF20(SB)/8, $877
DATA bitrev_size8192_mixed24_f64<>+0xCF28(SB)/8, $2925
DATA bitrev_size8192_mixed24_f64<>+0xCF30(SB)/8, $4973
DATA bitrev_size8192_mixed24_f64<>+0xCF38(SB)/8, $7021
DATA bitrev_size8192_mixed24_f64<>+0xCF40(SB)/8, $1389
DATA bitrev_size8192_mixed24_f64<>+0xCF48(SB)/8, $3437
DATA bitrev_size8192_mixed24_f64<>+0xCF50(SB)/8, $5485
DATA bitrev_size8192_mixed24_f64<>+0xCF58(SB)/8, $7533
DATA bitrev_size8192_mixed24_f64<>+0xCF60(SB)/8, $1901
DATA bitrev_size8192_mixed24_f64<>+0xCF68(SB)/8, $3949
DATA bitrev_size8192_mixed24_f64<>+0xCF70(SB)/8, $5997
DATA bitrev_size8192_mixed24_f64<>+0xCF78(SB)/8, $8045
DATA bitrev_size8192_mixed24_f64<>+0xCF80(SB)/8, $493
DATA bitrev_size8192_mixed24_f64<>+0xCF88(SB)/8, $2541
DATA bitrev_size8192_mixed24_f64<>+0xCF90(SB)/8, $4589
DATA bitrev_size8192_mixed24_f64<>+0xCF98(SB)/8, $6637
DATA bitrev_size8192_mixed24_f64<>+0xCFA0(SB)/8, $1005
DATA bitrev_size8192_mixed24_f64<>+0xCFA8(SB)/8, $3053
DATA bitrev_size8192_mixed24_f64<>+0xCFB0(SB)/8, $5101
DATA bitrev_size8192_mixed24_f64<>+0xCFB8(SB)/8, $7149
DATA bitrev_size8192_mixed24_f64<>+0xCFC0(SB)/8, $1517
DATA bitrev_size8192_mixed24_f64<>+0xCFC8(SB)/8, $3565
DATA bitrev_size8192_mixed24_f64<>+0xCFD0(SB)/8, $5613
DATA bitrev_size8192_mixed24_f64<>+0xCFD8(SB)/8, $7661
DATA bitrev_size8192_mixed24_f64<>+0xCFE0(SB)/8, $2029
DATA bitrev_size8192_mixed24_f64<>+0xCFE8(SB)/8, $4077
DATA bitrev_size8192_mixed24_f64<>+0xCFF0(SB)/8, $6125
DATA bitrev_size8192_mixed24_f64<>+0xCFF8(SB)/8, $8173
DATA bitrev_size8192_mixed24_f64<>+0xD000(SB)/8, $21
DATA bitrev_size8192_mixed24_f64<>+0xD008(SB)/8, $2069
DATA bitrev_size8192_mixed24_f64<>+0xD010(SB)/8, $4117
DATA bitrev_size8192_mixed24_f64<>+0xD018(SB)/8, $6165
DATA bitrev_size8192_mixed24_f64<>+0xD020(SB)/8, $533
DATA bitrev_size8192_mixed24_f64<>+0xD028(SB)/8, $2581
DATA bitrev_size8192_mixed24_f64<>+0xD030(SB)/8, $4629
DATA bitrev_size8192_mixed24_f64<>+0xD038(SB)/8, $6677
DATA bitrev_size8192_mixed24_f64<>+0xD040(SB)/8, $1045
DATA bitrev_size8192_mixed24_f64<>+0xD048(SB)/8, $3093
DATA bitrev_size8192_mixed24_f64<>+0xD050(SB)/8, $5141
DATA bitrev_size8192_mixed24_f64<>+0xD058(SB)/8, $7189
DATA bitrev_size8192_mixed24_f64<>+0xD060(SB)/8, $1557
DATA bitrev_size8192_mixed24_f64<>+0xD068(SB)/8, $3605
DATA bitrev_size8192_mixed24_f64<>+0xD070(SB)/8, $5653
DATA bitrev_size8192_mixed24_f64<>+0xD078(SB)/8, $7701
DATA bitrev_size8192_mixed24_f64<>+0xD080(SB)/8, $149
DATA bitrev_size8192_mixed24_f64<>+0xD088(SB)/8, $2197
DATA bitrev_size8192_mixed24_f64<>+0xD090(SB)/8, $4245
DATA bitrev_size8192_mixed24_f64<>+0xD098(SB)/8, $6293
DATA bitrev_size8192_mixed24_f64<>+0xD0A0(SB)/8, $661
DATA bitrev_size8192_mixed24_f64<>+0xD0A8(SB)/8, $2709
DATA bitrev_size8192_mixed24_f64<>+0xD0B0(SB)/8, $4757
DATA bitrev_size8192_mixed24_f64<>+0xD0B8(SB)/8, $6805
DATA bitrev_size8192_mixed24_f64<>+0xD0C0(SB)/8, $1173
DATA bitrev_size8192_mixed24_f64<>+0xD0C8(SB)/8, $3221
DATA bitrev_size8192_mixed24_f64<>+0xD0D0(SB)/8, $5269
DATA bitrev_size8192_mixed24_f64<>+0xD0D8(SB)/8, $7317
DATA bitrev_size8192_mixed24_f64<>+0xD0E0(SB)/8, $1685
DATA bitrev_size8192_mixed24_f64<>+0xD0E8(SB)/8, $3733
DATA bitrev_size8192_mixed24_f64<>+0xD0F0(SB)/8, $5781
DATA bitrev_size8192_mixed24_f64<>+0xD0F8(SB)/8, $7829
DATA bitrev_size8192_mixed24_f64<>+0xD100(SB)/8, $277
DATA bitrev_size8192_mixed24_f64<>+0xD108(SB)/8, $2325
DATA bitrev_size8192_mixed24_f64<>+0xD110(SB)/8, $4373
DATA bitrev_size8192_mixed24_f64<>+0xD118(SB)/8, $6421
DATA bitrev_size8192_mixed24_f64<>+0xD120(SB)/8, $789
DATA bitrev_size8192_mixed24_f64<>+0xD128(SB)/8, $2837
DATA bitrev_size8192_mixed24_f64<>+0xD130(SB)/8, $4885
DATA bitrev_size8192_mixed24_f64<>+0xD138(SB)/8, $6933
DATA bitrev_size8192_mixed24_f64<>+0xD140(SB)/8, $1301
DATA bitrev_size8192_mixed24_f64<>+0xD148(SB)/8, $3349
DATA bitrev_size8192_mixed24_f64<>+0xD150(SB)/8, $5397
DATA bitrev_size8192_mixed24_f64<>+0xD158(SB)/8, $7445
DATA bitrev_size8192_mixed24_f64<>+0xD160(SB)/8, $1813
DATA bitrev_size8192_mixed24_f64<>+0xD168(SB)/8, $3861
DATA bitrev_size8192_mixed24_f64<>+0xD170(SB)/8, $5909
DATA bitrev_size8192_mixed24_f64<>+0xD178(SB)/8, $7957
DATA bitrev_size8192_mixed24_f64<>+0xD180(SB)/8, $405
DATA bitrev_size8192_mixed24_f64<>+0xD188(SB)/8, $2453
DATA bitrev_size8192_mixed24_f64<>+0xD190(SB)/8, $4501
DATA bitrev_size8192_mixed24_f64<>+0xD198(SB)/8, $6549
DATA bitrev_size8192_mixed24_f64<>+0xD1A0(SB)/8, $917
DATA bitrev_size8192_mixed24_f64<>+0xD1A8(SB)/8, $2965
DATA bitrev_size8192_mixed24_f64<>+0xD1B0(SB)/8, $5013
DATA bitrev_size8192_mixed24_f64<>+0xD1B8(SB)/8, $7061
DATA bitrev_size8192_mixed24_f64<>+0xD1C0(SB)/8, $1429
DATA bitrev_size8192_mixed24_f64<>+0xD1C8(SB)/8, $3477
DATA bitrev_size8192_mixed24_f64<>+0xD1D0(SB)/8, $5525
DATA bitrev_size8192_mixed24_f64<>+0xD1D8(SB)/8, $7573
DATA bitrev_size8192_mixed24_f64<>+0xD1E0(SB)/8, $1941
DATA bitrev_size8192_mixed24_f64<>+0xD1E8(SB)/8, $3989
DATA bitrev_size8192_mixed24_f64<>+0xD1F0(SB)/8, $6037
DATA bitrev_size8192_mixed24_f64<>+0xD1F8(SB)/8, $8085
DATA bitrev_size8192_mixed24_f64<>+0xD200(SB)/8, $53
DATA bitrev_size8192_mixed24_f64<>+0xD208(SB)/8, $2101
DATA bitrev_size8192_mixed24_f64<>+0xD210(SB)/8, $4149
DATA bitrev_size8192_mixed24_f64<>+0xD218(SB)/8, $6197
DATA bitrev_size8192_mixed24_f64<>+0xD220(SB)/8, $565
DATA bitrev_size8192_mixed24_f64<>+0xD228(SB)/8, $2613
DATA bitrev_size8192_mixed24_f64<>+0xD230(SB)/8, $4661
DATA bitrev_size8192_mixed24_f64<>+0xD238(SB)/8, $6709
DATA bitrev_size8192_mixed24_f64<>+0xD240(SB)/8, $1077
DATA bitrev_size8192_mixed24_f64<>+0xD248(SB)/8, $3125
DATA bitrev_size8192_mixed24_f64<>+0xD250(SB)/8, $5173
DATA bitrev_size8192_mixed24_f64<>+0xD258(SB)/8, $7221
DATA bitrev_size8192_mixed24_f64<>+0xD260(SB)/8, $1589
DATA bitrev_size8192_mixed24_f64<>+0xD268(SB)/8, $3637
DATA bitrev_size8192_mixed24_f64<>+0xD270(SB)/8, $5685
DATA bitrev_size8192_mixed24_f64<>+0xD278(SB)/8, $7733
DATA bitrev_size8192_mixed24_f64<>+0xD280(SB)/8, $181
DATA bitrev_size8192_mixed24_f64<>+0xD288(SB)/8, $2229
DATA bitrev_size8192_mixed24_f64<>+0xD290(SB)/8, $4277
DATA bitrev_size8192_mixed24_f64<>+0xD298(SB)/8, $6325
DATA bitrev_size8192_mixed24_f64<>+0xD2A0(SB)/8, $693
DATA bitrev_size8192_mixed24_f64<>+0xD2A8(SB)/8, $2741
DATA bitrev_size8192_mixed24_f64<>+0xD2B0(SB)/8, $4789
DATA bitrev_size8192_mixed24_f64<>+0xD2B8(SB)/8, $6837
DATA bitrev_size8192_mixed24_f64<>+0xD2C0(SB)/8, $1205
DATA bitrev_size8192_mixed24_f64<>+0xD2C8(SB)/8, $3253
DATA bitrev_size8192_mixed24_f64<>+0xD2D0(SB)/8, $5301
DATA bitrev_size8192_mixed24_f64<>+0xD2D8(SB)/8, $7349
DATA bitrev_size8192_mixed24_f64<>+0xD2E0(SB)/8, $1717
DATA bitrev_size8192_mixed24_f64<>+0xD2E8(SB)/8, $3765
DATA bitrev_size8192_mixed24_f64<>+0xD2F0(SB)/8, $5813
DATA bitrev_size8192_mixed24_f64<>+0xD2F8(SB)/8, $7861
DATA bitrev_size8192_mixed24_f64<>+0xD300(SB)/8, $309
DATA bitrev_size8192_mixed24_f64<>+0xD308(SB)/8, $2357
DATA bitrev_size8192_mixed24_f64<>+0xD310(SB)/8, $4405
DATA bitrev_size8192_mixed24_f64<>+0xD318(SB)/8, $6453
DATA bitrev_size8192_mixed24_f64<>+0xD320(SB)/8, $821
DATA bitrev_size8192_mixed24_f64<>+0xD328(SB)/8, $2869
DATA bitrev_size8192_mixed24_f64<>+0xD330(SB)/8, $4917
DATA bitrev_size8192_mixed24_f64<>+0xD338(SB)/8, $6965
DATA bitrev_size8192_mixed24_f64<>+0xD340(SB)/8, $1333
DATA bitrev_size8192_mixed24_f64<>+0xD348(SB)/8, $3381
DATA bitrev_size8192_mixed24_f64<>+0xD350(SB)/8, $5429
DATA bitrev_size8192_mixed24_f64<>+0xD358(SB)/8, $7477
DATA bitrev_size8192_mixed24_f64<>+0xD360(SB)/8, $1845
DATA bitrev_size8192_mixed24_f64<>+0xD368(SB)/8, $3893
DATA bitrev_size8192_mixed24_f64<>+0xD370(SB)/8, $5941
DATA bitrev_size8192_mixed24_f64<>+0xD378(SB)/8, $7989
DATA bitrev_size8192_mixed24_f64<>+0xD380(SB)/8, $437
DATA bitrev_size8192_mixed24_f64<>+0xD388(SB)/8, $2485
DATA bitrev_size8192_mixed24_f64<>+0xD390(SB)/8, $4533
DATA bitrev_size8192_mixed24_f64<>+0xD398(SB)/8, $6581
DATA bitrev_size8192_mixed24_f64<>+0xD3A0(SB)/8, $949
DATA bitrev_size8192_mixed24_f64<>+0xD3A8(SB)/8, $2997
DATA bitrev_size8192_mixed24_f64<>+0xD3B0(SB)/8, $5045
DATA bitrev_size8192_mixed24_f64<>+0xD3B8(SB)/8, $7093
DATA bitrev_size8192_mixed24_f64<>+0xD3C0(SB)/8, $1461
DATA bitrev_size8192_mixed24_f64<>+0xD3C8(SB)/8, $3509
DATA bitrev_size8192_mixed24_f64<>+0xD3D0(SB)/8, $5557
DATA bitrev_size8192_mixed24_f64<>+0xD3D8(SB)/8, $7605
DATA bitrev_size8192_mixed24_f64<>+0xD3E0(SB)/8, $1973
DATA bitrev_size8192_mixed24_f64<>+0xD3E8(SB)/8, $4021
DATA bitrev_size8192_mixed24_f64<>+0xD3F0(SB)/8, $6069
DATA bitrev_size8192_mixed24_f64<>+0xD3F8(SB)/8, $8117
DATA bitrev_size8192_mixed24_f64<>+0xD400(SB)/8, $85
DATA bitrev_size8192_mixed24_f64<>+0xD408(SB)/8, $2133
DATA bitrev_size8192_mixed24_f64<>+0xD410(SB)/8, $4181
DATA bitrev_size8192_mixed24_f64<>+0xD418(SB)/8, $6229
DATA bitrev_size8192_mixed24_f64<>+0xD420(SB)/8, $597
DATA bitrev_size8192_mixed24_f64<>+0xD428(SB)/8, $2645
DATA bitrev_size8192_mixed24_f64<>+0xD430(SB)/8, $4693
DATA bitrev_size8192_mixed24_f64<>+0xD438(SB)/8, $6741
DATA bitrev_size8192_mixed24_f64<>+0xD440(SB)/8, $1109
DATA bitrev_size8192_mixed24_f64<>+0xD448(SB)/8, $3157
DATA bitrev_size8192_mixed24_f64<>+0xD450(SB)/8, $5205
DATA bitrev_size8192_mixed24_f64<>+0xD458(SB)/8, $7253
DATA bitrev_size8192_mixed24_f64<>+0xD460(SB)/8, $1621
DATA bitrev_size8192_mixed24_f64<>+0xD468(SB)/8, $3669
DATA bitrev_size8192_mixed24_f64<>+0xD470(SB)/8, $5717
DATA bitrev_size8192_mixed24_f64<>+0xD478(SB)/8, $7765
DATA bitrev_size8192_mixed24_f64<>+0xD480(SB)/8, $213
DATA bitrev_size8192_mixed24_f64<>+0xD488(SB)/8, $2261
DATA bitrev_size8192_mixed24_f64<>+0xD490(SB)/8, $4309
DATA bitrev_size8192_mixed24_f64<>+0xD498(SB)/8, $6357
DATA bitrev_size8192_mixed24_f64<>+0xD4A0(SB)/8, $725
DATA bitrev_size8192_mixed24_f64<>+0xD4A8(SB)/8, $2773
DATA bitrev_size8192_mixed24_f64<>+0xD4B0(SB)/8, $4821
DATA bitrev_size8192_mixed24_f64<>+0xD4B8(SB)/8, $6869
DATA bitrev_size8192_mixed24_f64<>+0xD4C0(SB)/8, $1237
DATA bitrev_size8192_mixed24_f64<>+0xD4C8(SB)/8, $3285
DATA bitrev_size8192_mixed24_f64<>+0xD4D0(SB)/8, $5333
DATA bitrev_size8192_mixed24_f64<>+0xD4D8(SB)/8, $7381
DATA bitrev_size8192_mixed24_f64<>+0xD4E0(SB)/8, $1749
DATA bitrev_size8192_mixed24_f64<>+0xD4E8(SB)/8, $3797
DATA bitrev_size8192_mixed24_f64<>+0xD4F0(SB)/8, $5845
DATA bitrev_size8192_mixed24_f64<>+0xD4F8(SB)/8, $7893
DATA bitrev_size8192_mixed24_f64<>+0xD500(SB)/8, $341
DATA bitrev_size8192_mixed24_f64<>+0xD508(SB)/8, $2389
DATA bitrev_size8192_mixed24_f64<>+0xD510(SB)/8, $4437
DATA bitrev_size8192_mixed24_f64<>+0xD518(SB)/8, $6485
DATA bitrev_size8192_mixed24_f64<>+0xD520(SB)/8, $853
DATA bitrev_size8192_mixed24_f64<>+0xD528(SB)/8, $2901
DATA bitrev_size8192_mixed24_f64<>+0xD530(SB)/8, $4949
DATA bitrev_size8192_mixed24_f64<>+0xD538(SB)/8, $6997
DATA bitrev_size8192_mixed24_f64<>+0xD540(SB)/8, $1365
DATA bitrev_size8192_mixed24_f64<>+0xD548(SB)/8, $3413
DATA bitrev_size8192_mixed24_f64<>+0xD550(SB)/8, $5461
DATA bitrev_size8192_mixed24_f64<>+0xD558(SB)/8, $7509
DATA bitrev_size8192_mixed24_f64<>+0xD560(SB)/8, $1877
DATA bitrev_size8192_mixed24_f64<>+0xD568(SB)/8, $3925
DATA bitrev_size8192_mixed24_f64<>+0xD570(SB)/8, $5973
DATA bitrev_size8192_mixed24_f64<>+0xD578(SB)/8, $8021
DATA bitrev_size8192_mixed24_f64<>+0xD580(SB)/8, $469
DATA bitrev_size8192_mixed24_f64<>+0xD588(SB)/8, $2517
DATA bitrev_size8192_mixed24_f64<>+0xD590(SB)/8, $4565
DATA bitrev_size8192_mixed24_f64<>+0xD598(SB)/8, $6613
DATA bitrev_size8192_mixed24_f64<>+0xD5A0(SB)/8, $981
DATA bitrev_size8192_mixed24_f64<>+0xD5A8(SB)/8, $3029
DATA bitrev_size8192_mixed24_f64<>+0xD5B0(SB)/8, $5077
DATA bitrev_size8192_mixed24_f64<>+0xD5B8(SB)/8, $7125
DATA bitrev_size8192_mixed24_f64<>+0xD5C0(SB)/8, $1493
DATA bitrev_size8192_mixed24_f64<>+0xD5C8(SB)/8, $3541
DATA bitrev_size8192_mixed24_f64<>+0xD5D0(SB)/8, $5589
DATA bitrev_size8192_mixed24_f64<>+0xD5D8(SB)/8, $7637
DATA bitrev_size8192_mixed24_f64<>+0xD5E0(SB)/8, $2005
DATA bitrev_size8192_mixed24_f64<>+0xD5E8(SB)/8, $4053
DATA bitrev_size8192_mixed24_f64<>+0xD5F0(SB)/8, $6101
DATA bitrev_size8192_mixed24_f64<>+0xD5F8(SB)/8, $8149
DATA bitrev_size8192_mixed24_f64<>+0xD600(SB)/8, $117
DATA bitrev_size8192_mixed24_f64<>+0xD608(SB)/8, $2165
DATA bitrev_size8192_mixed24_f64<>+0xD610(SB)/8, $4213
DATA bitrev_size8192_mixed24_f64<>+0xD618(SB)/8, $6261
DATA bitrev_size8192_mixed24_f64<>+0xD620(SB)/8, $629
DATA bitrev_size8192_mixed24_f64<>+0xD628(SB)/8, $2677
DATA bitrev_size8192_mixed24_f64<>+0xD630(SB)/8, $4725
DATA bitrev_size8192_mixed24_f64<>+0xD638(SB)/8, $6773
DATA bitrev_size8192_mixed24_f64<>+0xD640(SB)/8, $1141
DATA bitrev_size8192_mixed24_f64<>+0xD648(SB)/8, $3189
DATA bitrev_size8192_mixed24_f64<>+0xD650(SB)/8, $5237
DATA bitrev_size8192_mixed24_f64<>+0xD658(SB)/8, $7285
DATA bitrev_size8192_mixed24_f64<>+0xD660(SB)/8, $1653
DATA bitrev_size8192_mixed24_f64<>+0xD668(SB)/8, $3701
DATA bitrev_size8192_mixed24_f64<>+0xD670(SB)/8, $5749
DATA bitrev_size8192_mixed24_f64<>+0xD678(SB)/8, $7797
DATA bitrev_size8192_mixed24_f64<>+0xD680(SB)/8, $245
DATA bitrev_size8192_mixed24_f64<>+0xD688(SB)/8, $2293
DATA bitrev_size8192_mixed24_f64<>+0xD690(SB)/8, $4341
DATA bitrev_size8192_mixed24_f64<>+0xD698(SB)/8, $6389
DATA bitrev_size8192_mixed24_f64<>+0xD6A0(SB)/8, $757
DATA bitrev_size8192_mixed24_f64<>+0xD6A8(SB)/8, $2805
DATA bitrev_size8192_mixed24_f64<>+0xD6B0(SB)/8, $4853
DATA bitrev_size8192_mixed24_f64<>+0xD6B8(SB)/8, $6901
DATA bitrev_size8192_mixed24_f64<>+0xD6C0(SB)/8, $1269
DATA bitrev_size8192_mixed24_f64<>+0xD6C8(SB)/8, $3317
DATA bitrev_size8192_mixed24_f64<>+0xD6D0(SB)/8, $5365
DATA bitrev_size8192_mixed24_f64<>+0xD6D8(SB)/8, $7413
DATA bitrev_size8192_mixed24_f64<>+0xD6E0(SB)/8, $1781
DATA bitrev_size8192_mixed24_f64<>+0xD6E8(SB)/8, $3829
DATA bitrev_size8192_mixed24_f64<>+0xD6F0(SB)/8, $5877
DATA bitrev_size8192_mixed24_f64<>+0xD6F8(SB)/8, $7925
DATA bitrev_size8192_mixed24_f64<>+0xD700(SB)/8, $373
DATA bitrev_size8192_mixed24_f64<>+0xD708(SB)/8, $2421
DATA bitrev_size8192_mixed24_f64<>+0xD710(SB)/8, $4469
DATA bitrev_size8192_mixed24_f64<>+0xD718(SB)/8, $6517
DATA bitrev_size8192_mixed24_f64<>+0xD720(SB)/8, $885
DATA bitrev_size8192_mixed24_f64<>+0xD728(SB)/8, $2933
DATA bitrev_size8192_mixed24_f64<>+0xD730(SB)/8, $4981
DATA bitrev_size8192_mixed24_f64<>+0xD738(SB)/8, $7029
DATA bitrev_size8192_mixed24_f64<>+0xD740(SB)/8, $1397
DATA bitrev_size8192_mixed24_f64<>+0xD748(SB)/8, $3445
DATA bitrev_size8192_mixed24_f64<>+0xD750(SB)/8, $5493
DATA bitrev_size8192_mixed24_f64<>+0xD758(SB)/8, $7541
DATA bitrev_size8192_mixed24_f64<>+0xD760(SB)/8, $1909
DATA bitrev_size8192_mixed24_f64<>+0xD768(SB)/8, $3957
DATA bitrev_size8192_mixed24_f64<>+0xD770(SB)/8, $6005
DATA bitrev_size8192_mixed24_f64<>+0xD778(SB)/8, $8053
DATA bitrev_size8192_mixed24_f64<>+0xD780(SB)/8, $501
DATA bitrev_size8192_mixed24_f64<>+0xD788(SB)/8, $2549
DATA bitrev_size8192_mixed24_f64<>+0xD790(SB)/8, $4597
DATA bitrev_size8192_mixed24_f64<>+0xD798(SB)/8, $6645
DATA bitrev_size8192_mixed24_f64<>+0xD7A0(SB)/8, $1013
DATA bitrev_size8192_mixed24_f64<>+0xD7A8(SB)/8, $3061
DATA bitrev_size8192_mixed24_f64<>+0xD7B0(SB)/8, $5109
DATA bitrev_size8192_mixed24_f64<>+0xD7B8(SB)/8, $7157
DATA bitrev_size8192_mixed24_f64<>+0xD7C0(SB)/8, $1525
DATA bitrev_size8192_mixed24_f64<>+0xD7C8(SB)/8, $3573
DATA bitrev_size8192_mixed24_f64<>+0xD7D0(SB)/8, $5621
DATA bitrev_size8192_mixed24_f64<>+0xD7D8(SB)/8, $7669
DATA bitrev_size8192_mixed24_f64<>+0xD7E0(SB)/8, $2037
DATA bitrev_size8192_mixed24_f64<>+0xD7E8(SB)/8, $4085
DATA bitrev_size8192_mixed24_f64<>+0xD7F0(SB)/8, $6133
DATA bitrev_size8192_mixed24_f64<>+0xD7F8(SB)/8, $8181
DATA bitrev_size8192_mixed24_f64<>+0xD800(SB)/8, $29
DATA bitrev_size8192_mixed24_f64<>+0xD808(SB)/8, $2077
DATA bitrev_size8192_mixed24_f64<>+0xD810(SB)/8, $4125
DATA bitrev_size8192_mixed24_f64<>+0xD818(SB)/8, $6173
DATA bitrev_size8192_mixed24_f64<>+0xD820(SB)/8, $541
DATA bitrev_size8192_mixed24_f64<>+0xD828(SB)/8, $2589
DATA bitrev_size8192_mixed24_f64<>+0xD830(SB)/8, $4637
DATA bitrev_size8192_mixed24_f64<>+0xD838(SB)/8, $6685
DATA bitrev_size8192_mixed24_f64<>+0xD840(SB)/8, $1053
DATA bitrev_size8192_mixed24_f64<>+0xD848(SB)/8, $3101
DATA bitrev_size8192_mixed24_f64<>+0xD850(SB)/8, $5149
DATA bitrev_size8192_mixed24_f64<>+0xD858(SB)/8, $7197
DATA bitrev_size8192_mixed24_f64<>+0xD860(SB)/8, $1565
DATA bitrev_size8192_mixed24_f64<>+0xD868(SB)/8, $3613
DATA bitrev_size8192_mixed24_f64<>+0xD870(SB)/8, $5661
DATA bitrev_size8192_mixed24_f64<>+0xD878(SB)/8, $7709
DATA bitrev_size8192_mixed24_f64<>+0xD880(SB)/8, $157
DATA bitrev_size8192_mixed24_f64<>+0xD888(SB)/8, $2205
DATA bitrev_size8192_mixed24_f64<>+0xD890(SB)/8, $4253
DATA bitrev_size8192_mixed24_f64<>+0xD898(SB)/8, $6301
DATA bitrev_size8192_mixed24_f64<>+0xD8A0(SB)/8, $669
DATA bitrev_size8192_mixed24_f64<>+0xD8A8(SB)/8, $2717
DATA bitrev_size8192_mixed24_f64<>+0xD8B0(SB)/8, $4765
DATA bitrev_size8192_mixed24_f64<>+0xD8B8(SB)/8, $6813
DATA bitrev_size8192_mixed24_f64<>+0xD8C0(SB)/8, $1181
DATA bitrev_size8192_mixed24_f64<>+0xD8C8(SB)/8, $3229
DATA bitrev_size8192_mixed24_f64<>+0xD8D0(SB)/8, $5277
DATA bitrev_size8192_mixed24_f64<>+0xD8D8(SB)/8, $7325
DATA bitrev_size8192_mixed24_f64<>+0xD8E0(SB)/8, $1693
DATA bitrev_size8192_mixed24_f64<>+0xD8E8(SB)/8, $3741
DATA bitrev_size8192_mixed24_f64<>+0xD8F0(SB)/8, $5789
DATA bitrev_size8192_mixed24_f64<>+0xD8F8(SB)/8, $7837
DATA bitrev_size8192_mixed24_f64<>+0xD900(SB)/8, $285
DATA bitrev_size8192_mixed24_f64<>+0xD908(SB)/8, $2333
DATA bitrev_size8192_mixed24_f64<>+0xD910(SB)/8, $4381
DATA bitrev_size8192_mixed24_f64<>+0xD918(SB)/8, $6429
DATA bitrev_size8192_mixed24_f64<>+0xD920(SB)/8, $797
DATA bitrev_size8192_mixed24_f64<>+0xD928(SB)/8, $2845
DATA bitrev_size8192_mixed24_f64<>+0xD930(SB)/8, $4893
DATA bitrev_size8192_mixed24_f64<>+0xD938(SB)/8, $6941
DATA bitrev_size8192_mixed24_f64<>+0xD940(SB)/8, $1309
DATA bitrev_size8192_mixed24_f64<>+0xD948(SB)/8, $3357
DATA bitrev_size8192_mixed24_f64<>+0xD950(SB)/8, $5405
DATA bitrev_size8192_mixed24_f64<>+0xD958(SB)/8, $7453
DATA bitrev_size8192_mixed24_f64<>+0xD960(SB)/8, $1821
DATA bitrev_size8192_mixed24_f64<>+0xD968(SB)/8, $3869
DATA bitrev_size8192_mixed24_f64<>+0xD970(SB)/8, $5917
DATA bitrev_size8192_mixed24_f64<>+0xD978(SB)/8, $7965
DATA bitrev_size8192_mixed24_f64<>+0xD980(SB)/8, $413
DATA bitrev_size8192_mixed24_f64<>+0xD988(SB)/8, $2461
DATA bitrev_size8192_mixed24_f64<>+0xD990(SB)/8, $4509
DATA bitrev_size8192_mixed24_f64<>+0xD998(SB)/8, $6557
DATA bitrev_size8192_mixed24_f64<>+0xD9A0(SB)/8, $925
DATA bitrev_size8192_mixed24_f64<>+0xD9A8(SB)/8, $2973
DATA bitrev_size8192_mixed24_f64<>+0xD9B0(SB)/8, $5021
DATA bitrev_size8192_mixed24_f64<>+0xD9B8(SB)/8, $7069
DATA bitrev_size8192_mixed24_f64<>+0xD9C0(SB)/8, $1437
DATA bitrev_size8192_mixed24_f64<>+0xD9C8(SB)/8, $3485
DATA bitrev_size8192_mixed24_f64<>+0xD9D0(SB)/8, $5533
DATA bitrev_size8192_mixed24_f64<>+0xD9D8(SB)/8, $7581
DATA bitrev_size8192_mixed24_f64<>+0xD9E0(SB)/8, $1949
DATA bitrev_size8192_mixed24_f64<>+0xD9E8(SB)/8, $3997
DATA bitrev_size8192_mixed24_f64<>+0xD9F0(SB)/8, $6045
DATA bitrev_size8192_mixed24_f64<>+0xD9F8(SB)/8, $8093
DATA bitrev_size8192_mixed24_f64<>+0xDA00(SB)/8, $61
DATA bitrev_size8192_mixed24_f64<>+0xDA08(SB)/8, $2109
DATA bitrev_size8192_mixed24_f64<>+0xDA10(SB)/8, $4157
DATA bitrev_size8192_mixed24_f64<>+0xDA18(SB)/8, $6205
DATA bitrev_size8192_mixed24_f64<>+0xDA20(SB)/8, $573
DATA bitrev_size8192_mixed24_f64<>+0xDA28(SB)/8, $2621
DATA bitrev_size8192_mixed24_f64<>+0xDA30(SB)/8, $4669
DATA bitrev_size8192_mixed24_f64<>+0xDA38(SB)/8, $6717
DATA bitrev_size8192_mixed24_f64<>+0xDA40(SB)/8, $1085
DATA bitrev_size8192_mixed24_f64<>+0xDA48(SB)/8, $3133
DATA bitrev_size8192_mixed24_f64<>+0xDA50(SB)/8, $5181
DATA bitrev_size8192_mixed24_f64<>+0xDA58(SB)/8, $7229
DATA bitrev_size8192_mixed24_f64<>+0xDA60(SB)/8, $1597
DATA bitrev_size8192_mixed24_f64<>+0xDA68(SB)/8, $3645
DATA bitrev_size8192_mixed24_f64<>+0xDA70(SB)/8, $5693
DATA bitrev_size8192_mixed24_f64<>+0xDA78(SB)/8, $7741
DATA bitrev_size8192_mixed24_f64<>+0xDA80(SB)/8, $189
DATA bitrev_size8192_mixed24_f64<>+0xDA88(SB)/8, $2237
DATA bitrev_size8192_mixed24_f64<>+0xDA90(SB)/8, $4285
DATA bitrev_size8192_mixed24_f64<>+0xDA98(SB)/8, $6333
DATA bitrev_size8192_mixed24_f64<>+0xDAA0(SB)/8, $701
DATA bitrev_size8192_mixed24_f64<>+0xDAA8(SB)/8, $2749
DATA bitrev_size8192_mixed24_f64<>+0xDAB0(SB)/8, $4797
DATA bitrev_size8192_mixed24_f64<>+0xDAB8(SB)/8, $6845
DATA bitrev_size8192_mixed24_f64<>+0xDAC0(SB)/8, $1213
DATA bitrev_size8192_mixed24_f64<>+0xDAC8(SB)/8, $3261
DATA bitrev_size8192_mixed24_f64<>+0xDAD0(SB)/8, $5309
DATA bitrev_size8192_mixed24_f64<>+0xDAD8(SB)/8, $7357
DATA bitrev_size8192_mixed24_f64<>+0xDAE0(SB)/8, $1725
DATA bitrev_size8192_mixed24_f64<>+0xDAE8(SB)/8, $3773
DATA bitrev_size8192_mixed24_f64<>+0xDAF0(SB)/8, $5821
DATA bitrev_size8192_mixed24_f64<>+0xDAF8(SB)/8, $7869
DATA bitrev_size8192_mixed24_f64<>+0xDB00(SB)/8, $317
DATA bitrev_size8192_mixed24_f64<>+0xDB08(SB)/8, $2365
DATA bitrev_size8192_mixed24_f64<>+0xDB10(SB)/8, $4413
DATA bitrev_size8192_mixed24_f64<>+0xDB18(SB)/8, $6461
DATA bitrev_size8192_mixed24_f64<>+0xDB20(SB)/8, $829
DATA bitrev_size8192_mixed24_f64<>+0xDB28(SB)/8, $2877
DATA bitrev_size8192_mixed24_f64<>+0xDB30(SB)/8, $4925
DATA bitrev_size8192_mixed24_f64<>+0xDB38(SB)/8, $6973
DATA bitrev_size8192_mixed24_f64<>+0xDB40(SB)/8, $1341
DATA bitrev_size8192_mixed24_f64<>+0xDB48(SB)/8, $3389
DATA bitrev_size8192_mixed24_f64<>+0xDB50(SB)/8, $5437
DATA bitrev_size8192_mixed24_f64<>+0xDB58(SB)/8, $7485
DATA bitrev_size8192_mixed24_f64<>+0xDB60(SB)/8, $1853
DATA bitrev_size8192_mixed24_f64<>+0xDB68(SB)/8, $3901
DATA bitrev_size8192_mixed24_f64<>+0xDB70(SB)/8, $5949
DATA bitrev_size8192_mixed24_f64<>+0xDB78(SB)/8, $7997
DATA bitrev_size8192_mixed24_f64<>+0xDB80(SB)/8, $445
DATA bitrev_size8192_mixed24_f64<>+0xDB88(SB)/8, $2493
DATA bitrev_size8192_mixed24_f64<>+0xDB90(SB)/8, $4541
DATA bitrev_size8192_mixed24_f64<>+0xDB98(SB)/8, $6589
DATA bitrev_size8192_mixed24_f64<>+0xDBA0(SB)/8, $957
DATA bitrev_size8192_mixed24_f64<>+0xDBA8(SB)/8, $3005
DATA bitrev_size8192_mixed24_f64<>+0xDBB0(SB)/8, $5053
DATA bitrev_size8192_mixed24_f64<>+0xDBB8(SB)/8, $7101
DATA bitrev_size8192_mixed24_f64<>+0xDBC0(SB)/8, $1469
DATA bitrev_size8192_mixed24_f64<>+0xDBC8(SB)/8, $3517
DATA bitrev_size8192_mixed24_f64<>+0xDBD0(SB)/8, $5565
DATA bitrev_size8192_mixed24_f64<>+0xDBD8(SB)/8, $7613
DATA bitrev_size8192_mixed24_f64<>+0xDBE0(SB)/8, $1981
DATA bitrev_size8192_mixed24_f64<>+0xDBE8(SB)/8, $4029
DATA bitrev_size8192_mixed24_f64<>+0xDBF0(SB)/8, $6077
DATA bitrev_size8192_mixed24_f64<>+0xDBF8(SB)/8, $8125
DATA bitrev_size8192_mixed24_f64<>+0xDC00(SB)/8, $93
DATA bitrev_size8192_mixed24_f64<>+0xDC08(SB)/8, $2141
DATA bitrev_size8192_mixed24_f64<>+0xDC10(SB)/8, $4189
DATA bitrev_size8192_mixed24_f64<>+0xDC18(SB)/8, $6237
DATA bitrev_size8192_mixed24_f64<>+0xDC20(SB)/8, $605
DATA bitrev_size8192_mixed24_f64<>+0xDC28(SB)/8, $2653
DATA bitrev_size8192_mixed24_f64<>+0xDC30(SB)/8, $4701
DATA bitrev_size8192_mixed24_f64<>+0xDC38(SB)/8, $6749
DATA bitrev_size8192_mixed24_f64<>+0xDC40(SB)/8, $1117
DATA bitrev_size8192_mixed24_f64<>+0xDC48(SB)/8, $3165
DATA bitrev_size8192_mixed24_f64<>+0xDC50(SB)/8, $5213
DATA bitrev_size8192_mixed24_f64<>+0xDC58(SB)/8, $7261
DATA bitrev_size8192_mixed24_f64<>+0xDC60(SB)/8, $1629
DATA bitrev_size8192_mixed24_f64<>+0xDC68(SB)/8, $3677
DATA bitrev_size8192_mixed24_f64<>+0xDC70(SB)/8, $5725
DATA bitrev_size8192_mixed24_f64<>+0xDC78(SB)/8, $7773
DATA bitrev_size8192_mixed24_f64<>+0xDC80(SB)/8, $221
DATA bitrev_size8192_mixed24_f64<>+0xDC88(SB)/8, $2269
DATA bitrev_size8192_mixed24_f64<>+0xDC90(SB)/8, $4317
DATA bitrev_size8192_mixed24_f64<>+0xDC98(SB)/8, $6365
DATA bitrev_size8192_mixed24_f64<>+0xDCA0(SB)/8, $733
DATA bitrev_size8192_mixed24_f64<>+0xDCA8(SB)/8, $2781
DATA bitrev_size8192_mixed24_f64<>+0xDCB0(SB)/8, $4829
DATA bitrev_size8192_mixed24_f64<>+0xDCB8(SB)/8, $6877
DATA bitrev_size8192_mixed24_f64<>+0xDCC0(SB)/8, $1245
DATA bitrev_size8192_mixed24_f64<>+0xDCC8(SB)/8, $3293
DATA bitrev_size8192_mixed24_f64<>+0xDCD0(SB)/8, $5341
DATA bitrev_size8192_mixed24_f64<>+0xDCD8(SB)/8, $7389
DATA bitrev_size8192_mixed24_f64<>+0xDCE0(SB)/8, $1757
DATA bitrev_size8192_mixed24_f64<>+0xDCE8(SB)/8, $3805
DATA bitrev_size8192_mixed24_f64<>+0xDCF0(SB)/8, $5853
DATA bitrev_size8192_mixed24_f64<>+0xDCF8(SB)/8, $7901
DATA bitrev_size8192_mixed24_f64<>+0xDD00(SB)/8, $349
DATA bitrev_size8192_mixed24_f64<>+0xDD08(SB)/8, $2397
DATA bitrev_size8192_mixed24_f64<>+0xDD10(SB)/8, $4445
DATA bitrev_size8192_mixed24_f64<>+0xDD18(SB)/8, $6493
DATA bitrev_size8192_mixed24_f64<>+0xDD20(SB)/8, $861
DATA bitrev_size8192_mixed24_f64<>+0xDD28(SB)/8, $2909
DATA bitrev_size8192_mixed24_f64<>+0xDD30(SB)/8, $4957
DATA bitrev_size8192_mixed24_f64<>+0xDD38(SB)/8, $7005
DATA bitrev_size8192_mixed24_f64<>+0xDD40(SB)/8, $1373
DATA bitrev_size8192_mixed24_f64<>+0xDD48(SB)/8, $3421
DATA bitrev_size8192_mixed24_f64<>+0xDD50(SB)/8, $5469
DATA bitrev_size8192_mixed24_f64<>+0xDD58(SB)/8, $7517
DATA bitrev_size8192_mixed24_f64<>+0xDD60(SB)/8, $1885
DATA bitrev_size8192_mixed24_f64<>+0xDD68(SB)/8, $3933
DATA bitrev_size8192_mixed24_f64<>+0xDD70(SB)/8, $5981
DATA bitrev_size8192_mixed24_f64<>+0xDD78(SB)/8, $8029
DATA bitrev_size8192_mixed24_f64<>+0xDD80(SB)/8, $477
DATA bitrev_size8192_mixed24_f64<>+0xDD88(SB)/8, $2525
DATA bitrev_size8192_mixed24_f64<>+0xDD90(SB)/8, $4573
DATA bitrev_size8192_mixed24_f64<>+0xDD98(SB)/8, $6621
DATA bitrev_size8192_mixed24_f64<>+0xDDA0(SB)/8, $989
DATA bitrev_size8192_mixed24_f64<>+0xDDA8(SB)/8, $3037
DATA bitrev_size8192_mixed24_f64<>+0xDDB0(SB)/8, $5085
DATA bitrev_size8192_mixed24_f64<>+0xDDB8(SB)/8, $7133
DATA bitrev_size8192_mixed24_f64<>+0xDDC0(SB)/8, $1501
DATA bitrev_size8192_mixed24_f64<>+0xDDC8(SB)/8, $3549
DATA bitrev_size8192_mixed24_f64<>+0xDDD0(SB)/8, $5597
DATA bitrev_size8192_mixed24_f64<>+0xDDD8(SB)/8, $7645
DATA bitrev_size8192_mixed24_f64<>+0xDDE0(SB)/8, $2013
DATA bitrev_size8192_mixed24_f64<>+0xDDE8(SB)/8, $4061
DATA bitrev_size8192_mixed24_f64<>+0xDDF0(SB)/8, $6109
DATA bitrev_size8192_mixed24_f64<>+0xDDF8(SB)/8, $8157
DATA bitrev_size8192_mixed24_f64<>+0xDE00(SB)/8, $125
DATA bitrev_size8192_mixed24_f64<>+0xDE08(SB)/8, $2173
DATA bitrev_size8192_mixed24_f64<>+0xDE10(SB)/8, $4221
DATA bitrev_size8192_mixed24_f64<>+0xDE18(SB)/8, $6269
DATA bitrev_size8192_mixed24_f64<>+0xDE20(SB)/8, $637
DATA bitrev_size8192_mixed24_f64<>+0xDE28(SB)/8, $2685
DATA bitrev_size8192_mixed24_f64<>+0xDE30(SB)/8, $4733
DATA bitrev_size8192_mixed24_f64<>+0xDE38(SB)/8, $6781
DATA bitrev_size8192_mixed24_f64<>+0xDE40(SB)/8, $1149
DATA bitrev_size8192_mixed24_f64<>+0xDE48(SB)/8, $3197
DATA bitrev_size8192_mixed24_f64<>+0xDE50(SB)/8, $5245
DATA bitrev_size8192_mixed24_f64<>+0xDE58(SB)/8, $7293
DATA bitrev_size8192_mixed24_f64<>+0xDE60(SB)/8, $1661
DATA bitrev_size8192_mixed24_f64<>+0xDE68(SB)/8, $3709
DATA bitrev_size8192_mixed24_f64<>+0xDE70(SB)/8, $5757
DATA bitrev_size8192_mixed24_f64<>+0xDE78(SB)/8, $7805
DATA bitrev_size8192_mixed24_f64<>+0xDE80(SB)/8, $253
DATA bitrev_size8192_mixed24_f64<>+0xDE88(SB)/8, $2301
DATA bitrev_size8192_mixed24_f64<>+0xDE90(SB)/8, $4349
DATA bitrev_size8192_mixed24_f64<>+0xDE98(SB)/8, $6397
DATA bitrev_size8192_mixed24_f64<>+0xDEA0(SB)/8, $765
DATA bitrev_size8192_mixed24_f64<>+0xDEA8(SB)/8, $2813
DATA bitrev_size8192_mixed24_f64<>+0xDEB0(SB)/8, $4861
DATA bitrev_size8192_mixed24_f64<>+0xDEB8(SB)/8, $6909
DATA bitrev_size8192_mixed24_f64<>+0xDEC0(SB)/8, $1277
DATA bitrev_size8192_mixed24_f64<>+0xDEC8(SB)/8, $3325
DATA bitrev_size8192_mixed24_f64<>+0xDED0(SB)/8, $5373
DATA bitrev_size8192_mixed24_f64<>+0xDED8(SB)/8, $7421
DATA bitrev_size8192_mixed24_f64<>+0xDEE0(SB)/8, $1789
DATA bitrev_size8192_mixed24_f64<>+0xDEE8(SB)/8, $3837
DATA bitrev_size8192_mixed24_f64<>+0xDEF0(SB)/8, $5885
DATA bitrev_size8192_mixed24_f64<>+0xDEF8(SB)/8, $7933
DATA bitrev_size8192_mixed24_f64<>+0xDF00(SB)/8, $381
DATA bitrev_size8192_mixed24_f64<>+0xDF08(SB)/8, $2429
DATA bitrev_size8192_mixed24_f64<>+0xDF10(SB)/8, $4477
DATA bitrev_size8192_mixed24_f64<>+0xDF18(SB)/8, $6525
DATA bitrev_size8192_mixed24_f64<>+0xDF20(SB)/8, $893
DATA bitrev_size8192_mixed24_f64<>+0xDF28(SB)/8, $2941
DATA bitrev_size8192_mixed24_f64<>+0xDF30(SB)/8, $4989
DATA bitrev_size8192_mixed24_f64<>+0xDF38(SB)/8, $7037
DATA bitrev_size8192_mixed24_f64<>+0xDF40(SB)/8, $1405
DATA bitrev_size8192_mixed24_f64<>+0xDF48(SB)/8, $3453
DATA bitrev_size8192_mixed24_f64<>+0xDF50(SB)/8, $5501
DATA bitrev_size8192_mixed24_f64<>+0xDF58(SB)/8, $7549
DATA bitrev_size8192_mixed24_f64<>+0xDF60(SB)/8, $1917
DATA bitrev_size8192_mixed24_f64<>+0xDF68(SB)/8, $3965
DATA bitrev_size8192_mixed24_f64<>+0xDF70(SB)/8, $6013
DATA bitrev_size8192_mixed24_f64<>+0xDF78(SB)/8, $8061
DATA bitrev_size8192_mixed24_f64<>+0xDF80(SB)/8, $509
DATA bitrev_size8192_mixed24_f64<>+0xDF88(SB)/8, $2557
DATA bitrev_size8192_mixed24_f64<>+0xDF90(SB)/8, $4605
DATA bitrev_size8192_mixed24_f64<>+0xDF98(SB)/8, $6653
DATA bitrev_size8192_mixed24_f64<>+0xDFA0(SB)/8, $1021
DATA bitrev_size8192_mixed24_f64<>+0xDFA8(SB)/8, $3069
DATA bitrev_size8192_mixed24_f64<>+0xDFB0(SB)/8, $5117
DATA bitrev_size8192_mixed24_f64<>+0xDFB8(SB)/8, $7165
DATA bitrev_size8192_mixed24_f64<>+0xDFC0(SB)/8, $1533
DATA bitrev_size8192_mixed24_f64<>+0xDFC8(SB)/8, $3581
DATA bitrev_size8192_mixed24_f64<>+0xDFD0(SB)/8, $5629
DATA bitrev_size8192_mixed24_f64<>+0xDFD8(SB)/8, $7677
DATA bitrev_size8192_mixed24_f64<>+0xDFE0(SB)/8, $2045
DATA bitrev_size8192_mixed24_f64<>+0xDFE8(SB)/8, $4093
DATA bitrev_size8192_mixed24_f64<>+0xDFF0(SB)/8, $6141
DATA bitrev_size8192_mixed24_f64<>+0xDFF8(SB)/8, $8189
DATA bitrev_size8192_mixed24_f64<>+0xE000(SB)/8, $7
DATA bitrev_size8192_mixed24_f64<>+0xE008(SB)/8, $2055
DATA bitrev_size8192_mixed24_f64<>+0xE010(SB)/8, $4103
DATA bitrev_size8192_mixed24_f64<>+0xE018(SB)/8, $6151
DATA bitrev_size8192_mixed24_f64<>+0xE020(SB)/8, $519
DATA bitrev_size8192_mixed24_f64<>+0xE028(SB)/8, $2567
DATA bitrev_size8192_mixed24_f64<>+0xE030(SB)/8, $4615
DATA bitrev_size8192_mixed24_f64<>+0xE038(SB)/8, $6663
DATA bitrev_size8192_mixed24_f64<>+0xE040(SB)/8, $1031
DATA bitrev_size8192_mixed24_f64<>+0xE048(SB)/8, $3079
DATA bitrev_size8192_mixed24_f64<>+0xE050(SB)/8, $5127
DATA bitrev_size8192_mixed24_f64<>+0xE058(SB)/8, $7175
DATA bitrev_size8192_mixed24_f64<>+0xE060(SB)/8, $1543
DATA bitrev_size8192_mixed24_f64<>+0xE068(SB)/8, $3591
DATA bitrev_size8192_mixed24_f64<>+0xE070(SB)/8, $5639
DATA bitrev_size8192_mixed24_f64<>+0xE078(SB)/8, $7687
DATA bitrev_size8192_mixed24_f64<>+0xE080(SB)/8, $135
DATA bitrev_size8192_mixed24_f64<>+0xE088(SB)/8, $2183
DATA bitrev_size8192_mixed24_f64<>+0xE090(SB)/8, $4231
DATA bitrev_size8192_mixed24_f64<>+0xE098(SB)/8, $6279
DATA bitrev_size8192_mixed24_f64<>+0xE0A0(SB)/8, $647
DATA bitrev_size8192_mixed24_f64<>+0xE0A8(SB)/8, $2695
DATA bitrev_size8192_mixed24_f64<>+0xE0B0(SB)/8, $4743
DATA bitrev_size8192_mixed24_f64<>+0xE0B8(SB)/8, $6791
DATA bitrev_size8192_mixed24_f64<>+0xE0C0(SB)/8, $1159
DATA bitrev_size8192_mixed24_f64<>+0xE0C8(SB)/8, $3207
DATA bitrev_size8192_mixed24_f64<>+0xE0D0(SB)/8, $5255
DATA bitrev_size8192_mixed24_f64<>+0xE0D8(SB)/8, $7303
DATA bitrev_size8192_mixed24_f64<>+0xE0E0(SB)/8, $1671
DATA bitrev_size8192_mixed24_f64<>+0xE0E8(SB)/8, $3719
DATA bitrev_size8192_mixed24_f64<>+0xE0F0(SB)/8, $5767
DATA bitrev_size8192_mixed24_f64<>+0xE0F8(SB)/8, $7815
DATA bitrev_size8192_mixed24_f64<>+0xE100(SB)/8, $263
DATA bitrev_size8192_mixed24_f64<>+0xE108(SB)/8, $2311
DATA bitrev_size8192_mixed24_f64<>+0xE110(SB)/8, $4359
DATA bitrev_size8192_mixed24_f64<>+0xE118(SB)/8, $6407
DATA bitrev_size8192_mixed24_f64<>+0xE120(SB)/8, $775
DATA bitrev_size8192_mixed24_f64<>+0xE128(SB)/8, $2823
DATA bitrev_size8192_mixed24_f64<>+0xE130(SB)/8, $4871
DATA bitrev_size8192_mixed24_f64<>+0xE138(SB)/8, $6919
DATA bitrev_size8192_mixed24_f64<>+0xE140(SB)/8, $1287
DATA bitrev_size8192_mixed24_f64<>+0xE148(SB)/8, $3335
DATA bitrev_size8192_mixed24_f64<>+0xE150(SB)/8, $5383
DATA bitrev_size8192_mixed24_f64<>+0xE158(SB)/8, $7431
DATA bitrev_size8192_mixed24_f64<>+0xE160(SB)/8, $1799
DATA bitrev_size8192_mixed24_f64<>+0xE168(SB)/8, $3847
DATA bitrev_size8192_mixed24_f64<>+0xE170(SB)/8, $5895
DATA bitrev_size8192_mixed24_f64<>+0xE178(SB)/8, $7943
DATA bitrev_size8192_mixed24_f64<>+0xE180(SB)/8, $391
DATA bitrev_size8192_mixed24_f64<>+0xE188(SB)/8, $2439
DATA bitrev_size8192_mixed24_f64<>+0xE190(SB)/8, $4487
DATA bitrev_size8192_mixed24_f64<>+0xE198(SB)/8, $6535
DATA bitrev_size8192_mixed24_f64<>+0xE1A0(SB)/8, $903
DATA bitrev_size8192_mixed24_f64<>+0xE1A8(SB)/8, $2951
DATA bitrev_size8192_mixed24_f64<>+0xE1B0(SB)/8, $4999
DATA bitrev_size8192_mixed24_f64<>+0xE1B8(SB)/8, $7047
DATA bitrev_size8192_mixed24_f64<>+0xE1C0(SB)/8, $1415
DATA bitrev_size8192_mixed24_f64<>+0xE1C8(SB)/8, $3463
DATA bitrev_size8192_mixed24_f64<>+0xE1D0(SB)/8, $5511
DATA bitrev_size8192_mixed24_f64<>+0xE1D8(SB)/8, $7559
DATA bitrev_size8192_mixed24_f64<>+0xE1E0(SB)/8, $1927
DATA bitrev_size8192_mixed24_f64<>+0xE1E8(SB)/8, $3975
DATA bitrev_size8192_mixed24_f64<>+0xE1F0(SB)/8, $6023
DATA bitrev_size8192_mixed24_f64<>+0xE1F8(SB)/8, $8071
DATA bitrev_size8192_mixed24_f64<>+0xE200(SB)/8, $39
DATA bitrev_size8192_mixed24_f64<>+0xE208(SB)/8, $2087
DATA bitrev_size8192_mixed24_f64<>+0xE210(SB)/8, $4135
DATA bitrev_size8192_mixed24_f64<>+0xE218(SB)/8, $6183
DATA bitrev_size8192_mixed24_f64<>+0xE220(SB)/8, $551
DATA bitrev_size8192_mixed24_f64<>+0xE228(SB)/8, $2599
DATA bitrev_size8192_mixed24_f64<>+0xE230(SB)/8, $4647
DATA bitrev_size8192_mixed24_f64<>+0xE238(SB)/8, $6695
DATA bitrev_size8192_mixed24_f64<>+0xE240(SB)/8, $1063
DATA bitrev_size8192_mixed24_f64<>+0xE248(SB)/8, $3111
DATA bitrev_size8192_mixed24_f64<>+0xE250(SB)/8, $5159
DATA bitrev_size8192_mixed24_f64<>+0xE258(SB)/8, $7207
DATA bitrev_size8192_mixed24_f64<>+0xE260(SB)/8, $1575
DATA bitrev_size8192_mixed24_f64<>+0xE268(SB)/8, $3623
DATA bitrev_size8192_mixed24_f64<>+0xE270(SB)/8, $5671
DATA bitrev_size8192_mixed24_f64<>+0xE278(SB)/8, $7719
DATA bitrev_size8192_mixed24_f64<>+0xE280(SB)/8, $167
DATA bitrev_size8192_mixed24_f64<>+0xE288(SB)/8, $2215
DATA bitrev_size8192_mixed24_f64<>+0xE290(SB)/8, $4263
DATA bitrev_size8192_mixed24_f64<>+0xE298(SB)/8, $6311
DATA bitrev_size8192_mixed24_f64<>+0xE2A0(SB)/8, $679
DATA bitrev_size8192_mixed24_f64<>+0xE2A8(SB)/8, $2727
DATA bitrev_size8192_mixed24_f64<>+0xE2B0(SB)/8, $4775
DATA bitrev_size8192_mixed24_f64<>+0xE2B8(SB)/8, $6823
DATA bitrev_size8192_mixed24_f64<>+0xE2C0(SB)/8, $1191
DATA bitrev_size8192_mixed24_f64<>+0xE2C8(SB)/8, $3239
DATA bitrev_size8192_mixed24_f64<>+0xE2D0(SB)/8, $5287
DATA bitrev_size8192_mixed24_f64<>+0xE2D8(SB)/8, $7335
DATA bitrev_size8192_mixed24_f64<>+0xE2E0(SB)/8, $1703
DATA bitrev_size8192_mixed24_f64<>+0xE2E8(SB)/8, $3751
DATA bitrev_size8192_mixed24_f64<>+0xE2F0(SB)/8, $5799
DATA bitrev_size8192_mixed24_f64<>+0xE2F8(SB)/8, $7847
DATA bitrev_size8192_mixed24_f64<>+0xE300(SB)/8, $295
DATA bitrev_size8192_mixed24_f64<>+0xE308(SB)/8, $2343
DATA bitrev_size8192_mixed24_f64<>+0xE310(SB)/8, $4391
DATA bitrev_size8192_mixed24_f64<>+0xE318(SB)/8, $6439
DATA bitrev_size8192_mixed24_f64<>+0xE320(SB)/8, $807
DATA bitrev_size8192_mixed24_f64<>+0xE328(SB)/8, $2855
DATA bitrev_size8192_mixed24_f64<>+0xE330(SB)/8, $4903
DATA bitrev_size8192_mixed24_f64<>+0xE338(SB)/8, $6951
DATA bitrev_size8192_mixed24_f64<>+0xE340(SB)/8, $1319
DATA bitrev_size8192_mixed24_f64<>+0xE348(SB)/8, $3367
DATA bitrev_size8192_mixed24_f64<>+0xE350(SB)/8, $5415
DATA bitrev_size8192_mixed24_f64<>+0xE358(SB)/8, $7463
DATA bitrev_size8192_mixed24_f64<>+0xE360(SB)/8, $1831
DATA bitrev_size8192_mixed24_f64<>+0xE368(SB)/8, $3879
DATA bitrev_size8192_mixed24_f64<>+0xE370(SB)/8, $5927
DATA bitrev_size8192_mixed24_f64<>+0xE378(SB)/8, $7975
DATA bitrev_size8192_mixed24_f64<>+0xE380(SB)/8, $423
DATA bitrev_size8192_mixed24_f64<>+0xE388(SB)/8, $2471
DATA bitrev_size8192_mixed24_f64<>+0xE390(SB)/8, $4519
DATA bitrev_size8192_mixed24_f64<>+0xE398(SB)/8, $6567
DATA bitrev_size8192_mixed24_f64<>+0xE3A0(SB)/8, $935
DATA bitrev_size8192_mixed24_f64<>+0xE3A8(SB)/8, $2983
DATA bitrev_size8192_mixed24_f64<>+0xE3B0(SB)/8, $5031
DATA bitrev_size8192_mixed24_f64<>+0xE3B8(SB)/8, $7079
DATA bitrev_size8192_mixed24_f64<>+0xE3C0(SB)/8, $1447
DATA bitrev_size8192_mixed24_f64<>+0xE3C8(SB)/8, $3495
DATA bitrev_size8192_mixed24_f64<>+0xE3D0(SB)/8, $5543
DATA bitrev_size8192_mixed24_f64<>+0xE3D8(SB)/8, $7591
DATA bitrev_size8192_mixed24_f64<>+0xE3E0(SB)/8, $1959
DATA bitrev_size8192_mixed24_f64<>+0xE3E8(SB)/8, $4007
DATA bitrev_size8192_mixed24_f64<>+0xE3F0(SB)/8, $6055
DATA bitrev_size8192_mixed24_f64<>+0xE3F8(SB)/8, $8103
DATA bitrev_size8192_mixed24_f64<>+0xE400(SB)/8, $71
DATA bitrev_size8192_mixed24_f64<>+0xE408(SB)/8, $2119
DATA bitrev_size8192_mixed24_f64<>+0xE410(SB)/8, $4167
DATA bitrev_size8192_mixed24_f64<>+0xE418(SB)/8, $6215
DATA bitrev_size8192_mixed24_f64<>+0xE420(SB)/8, $583
DATA bitrev_size8192_mixed24_f64<>+0xE428(SB)/8, $2631
DATA bitrev_size8192_mixed24_f64<>+0xE430(SB)/8, $4679
DATA bitrev_size8192_mixed24_f64<>+0xE438(SB)/8, $6727
DATA bitrev_size8192_mixed24_f64<>+0xE440(SB)/8, $1095
DATA bitrev_size8192_mixed24_f64<>+0xE448(SB)/8, $3143
DATA bitrev_size8192_mixed24_f64<>+0xE450(SB)/8, $5191
DATA bitrev_size8192_mixed24_f64<>+0xE458(SB)/8, $7239
DATA bitrev_size8192_mixed24_f64<>+0xE460(SB)/8, $1607
DATA bitrev_size8192_mixed24_f64<>+0xE468(SB)/8, $3655
DATA bitrev_size8192_mixed24_f64<>+0xE470(SB)/8, $5703
DATA bitrev_size8192_mixed24_f64<>+0xE478(SB)/8, $7751
DATA bitrev_size8192_mixed24_f64<>+0xE480(SB)/8, $199
DATA bitrev_size8192_mixed24_f64<>+0xE488(SB)/8, $2247
DATA bitrev_size8192_mixed24_f64<>+0xE490(SB)/8, $4295
DATA bitrev_size8192_mixed24_f64<>+0xE498(SB)/8, $6343
DATA bitrev_size8192_mixed24_f64<>+0xE4A0(SB)/8, $711
DATA bitrev_size8192_mixed24_f64<>+0xE4A8(SB)/8, $2759
DATA bitrev_size8192_mixed24_f64<>+0xE4B0(SB)/8, $4807
DATA bitrev_size8192_mixed24_f64<>+0xE4B8(SB)/8, $6855
DATA bitrev_size8192_mixed24_f64<>+0xE4C0(SB)/8, $1223
DATA bitrev_size8192_mixed24_f64<>+0xE4C8(SB)/8, $3271
DATA bitrev_size8192_mixed24_f64<>+0xE4D0(SB)/8, $5319
DATA bitrev_size8192_mixed24_f64<>+0xE4D8(SB)/8, $7367
DATA bitrev_size8192_mixed24_f64<>+0xE4E0(SB)/8, $1735
DATA bitrev_size8192_mixed24_f64<>+0xE4E8(SB)/8, $3783
DATA bitrev_size8192_mixed24_f64<>+0xE4F0(SB)/8, $5831
DATA bitrev_size8192_mixed24_f64<>+0xE4F8(SB)/8, $7879
DATA bitrev_size8192_mixed24_f64<>+0xE500(SB)/8, $327
DATA bitrev_size8192_mixed24_f64<>+0xE508(SB)/8, $2375
DATA bitrev_size8192_mixed24_f64<>+0xE510(SB)/8, $4423
DATA bitrev_size8192_mixed24_f64<>+0xE518(SB)/8, $6471
DATA bitrev_size8192_mixed24_f64<>+0xE520(SB)/8, $839
DATA bitrev_size8192_mixed24_f64<>+0xE528(SB)/8, $2887
DATA bitrev_size8192_mixed24_f64<>+0xE530(SB)/8, $4935
DATA bitrev_size8192_mixed24_f64<>+0xE538(SB)/8, $6983
DATA bitrev_size8192_mixed24_f64<>+0xE540(SB)/8, $1351
DATA bitrev_size8192_mixed24_f64<>+0xE548(SB)/8, $3399
DATA bitrev_size8192_mixed24_f64<>+0xE550(SB)/8, $5447
DATA bitrev_size8192_mixed24_f64<>+0xE558(SB)/8, $7495
DATA bitrev_size8192_mixed24_f64<>+0xE560(SB)/8, $1863
DATA bitrev_size8192_mixed24_f64<>+0xE568(SB)/8, $3911
DATA bitrev_size8192_mixed24_f64<>+0xE570(SB)/8, $5959
DATA bitrev_size8192_mixed24_f64<>+0xE578(SB)/8, $8007
DATA bitrev_size8192_mixed24_f64<>+0xE580(SB)/8, $455
DATA bitrev_size8192_mixed24_f64<>+0xE588(SB)/8, $2503
DATA bitrev_size8192_mixed24_f64<>+0xE590(SB)/8, $4551
DATA bitrev_size8192_mixed24_f64<>+0xE598(SB)/8, $6599
DATA bitrev_size8192_mixed24_f64<>+0xE5A0(SB)/8, $967
DATA bitrev_size8192_mixed24_f64<>+0xE5A8(SB)/8, $3015
DATA bitrev_size8192_mixed24_f64<>+0xE5B0(SB)/8, $5063
DATA bitrev_size8192_mixed24_f64<>+0xE5B8(SB)/8, $7111
DATA bitrev_size8192_mixed24_f64<>+0xE5C0(SB)/8, $1479
DATA bitrev_size8192_mixed24_f64<>+0xE5C8(SB)/8, $3527
DATA bitrev_size8192_mixed24_f64<>+0xE5D0(SB)/8, $5575
DATA bitrev_size8192_mixed24_f64<>+0xE5D8(SB)/8, $7623
DATA bitrev_size8192_mixed24_f64<>+0xE5E0(SB)/8, $1991
DATA bitrev_size8192_mixed24_f64<>+0xE5E8(SB)/8, $4039
DATA bitrev_size8192_mixed24_f64<>+0xE5F0(SB)/8, $6087
DATA bitrev_size8192_mixed24_f64<>+0xE5F8(SB)/8, $8135
DATA bitrev_size8192_mixed24_f64<>+0xE600(SB)/8, $103
DATA bitrev_size8192_mixed24_f64<>+0xE608(SB)/8, $2151
DATA bitrev_size8192_mixed24_f64<>+0xE610(SB)/8, $4199
DATA bitrev_size8192_mixed24_f64<>+0xE618(SB)/8, $6247
DATA bitrev_size8192_mixed24_f64<>+0xE620(SB)/8, $615
DATA bitrev_size8192_mixed24_f64<>+0xE628(SB)/8, $2663
DATA bitrev_size8192_mixed24_f64<>+0xE630(SB)/8, $4711
DATA bitrev_size8192_mixed24_f64<>+0xE638(SB)/8, $6759
DATA bitrev_size8192_mixed24_f64<>+0xE640(SB)/8, $1127
DATA bitrev_size8192_mixed24_f64<>+0xE648(SB)/8, $3175
DATA bitrev_size8192_mixed24_f64<>+0xE650(SB)/8, $5223
DATA bitrev_size8192_mixed24_f64<>+0xE658(SB)/8, $7271
DATA bitrev_size8192_mixed24_f64<>+0xE660(SB)/8, $1639
DATA bitrev_size8192_mixed24_f64<>+0xE668(SB)/8, $3687
DATA bitrev_size8192_mixed24_f64<>+0xE670(SB)/8, $5735
DATA bitrev_size8192_mixed24_f64<>+0xE678(SB)/8, $7783
DATA bitrev_size8192_mixed24_f64<>+0xE680(SB)/8, $231
DATA bitrev_size8192_mixed24_f64<>+0xE688(SB)/8, $2279
DATA bitrev_size8192_mixed24_f64<>+0xE690(SB)/8, $4327
DATA bitrev_size8192_mixed24_f64<>+0xE698(SB)/8, $6375
DATA bitrev_size8192_mixed24_f64<>+0xE6A0(SB)/8, $743
DATA bitrev_size8192_mixed24_f64<>+0xE6A8(SB)/8, $2791
DATA bitrev_size8192_mixed24_f64<>+0xE6B0(SB)/8, $4839
DATA bitrev_size8192_mixed24_f64<>+0xE6B8(SB)/8, $6887
DATA bitrev_size8192_mixed24_f64<>+0xE6C0(SB)/8, $1255
DATA bitrev_size8192_mixed24_f64<>+0xE6C8(SB)/8, $3303
DATA bitrev_size8192_mixed24_f64<>+0xE6D0(SB)/8, $5351
DATA bitrev_size8192_mixed24_f64<>+0xE6D8(SB)/8, $7399
DATA bitrev_size8192_mixed24_f64<>+0xE6E0(SB)/8, $1767
DATA bitrev_size8192_mixed24_f64<>+0xE6E8(SB)/8, $3815
DATA bitrev_size8192_mixed24_f64<>+0xE6F0(SB)/8, $5863
DATA bitrev_size8192_mixed24_f64<>+0xE6F8(SB)/8, $7911
DATA bitrev_size8192_mixed24_f64<>+0xE700(SB)/8, $359
DATA bitrev_size8192_mixed24_f64<>+0xE708(SB)/8, $2407
DATA bitrev_size8192_mixed24_f64<>+0xE710(SB)/8, $4455
DATA bitrev_size8192_mixed24_f64<>+0xE718(SB)/8, $6503
DATA bitrev_size8192_mixed24_f64<>+0xE720(SB)/8, $871
DATA bitrev_size8192_mixed24_f64<>+0xE728(SB)/8, $2919
DATA bitrev_size8192_mixed24_f64<>+0xE730(SB)/8, $4967
DATA bitrev_size8192_mixed24_f64<>+0xE738(SB)/8, $7015
DATA bitrev_size8192_mixed24_f64<>+0xE740(SB)/8, $1383
DATA bitrev_size8192_mixed24_f64<>+0xE748(SB)/8, $3431
DATA bitrev_size8192_mixed24_f64<>+0xE750(SB)/8, $5479
DATA bitrev_size8192_mixed24_f64<>+0xE758(SB)/8, $7527
DATA bitrev_size8192_mixed24_f64<>+0xE760(SB)/8, $1895
DATA bitrev_size8192_mixed24_f64<>+0xE768(SB)/8, $3943
DATA bitrev_size8192_mixed24_f64<>+0xE770(SB)/8, $5991
DATA bitrev_size8192_mixed24_f64<>+0xE778(SB)/8, $8039
DATA bitrev_size8192_mixed24_f64<>+0xE780(SB)/8, $487
DATA bitrev_size8192_mixed24_f64<>+0xE788(SB)/8, $2535
DATA bitrev_size8192_mixed24_f64<>+0xE790(SB)/8, $4583
DATA bitrev_size8192_mixed24_f64<>+0xE798(SB)/8, $6631
DATA bitrev_size8192_mixed24_f64<>+0xE7A0(SB)/8, $999
DATA bitrev_size8192_mixed24_f64<>+0xE7A8(SB)/8, $3047
DATA bitrev_size8192_mixed24_f64<>+0xE7B0(SB)/8, $5095
DATA bitrev_size8192_mixed24_f64<>+0xE7B8(SB)/8, $7143
DATA bitrev_size8192_mixed24_f64<>+0xE7C0(SB)/8, $1511
DATA bitrev_size8192_mixed24_f64<>+0xE7C8(SB)/8, $3559
DATA bitrev_size8192_mixed24_f64<>+0xE7D0(SB)/8, $5607
DATA bitrev_size8192_mixed24_f64<>+0xE7D8(SB)/8, $7655
DATA bitrev_size8192_mixed24_f64<>+0xE7E0(SB)/8, $2023
DATA bitrev_size8192_mixed24_f64<>+0xE7E8(SB)/8, $4071
DATA bitrev_size8192_mixed24_f64<>+0xE7F0(SB)/8, $6119
DATA bitrev_size8192_mixed24_f64<>+0xE7F8(SB)/8, $8167
DATA bitrev_size8192_mixed24_f64<>+0xE800(SB)/8, $15
DATA bitrev_size8192_mixed24_f64<>+0xE808(SB)/8, $2063
DATA bitrev_size8192_mixed24_f64<>+0xE810(SB)/8, $4111
DATA bitrev_size8192_mixed24_f64<>+0xE818(SB)/8, $6159
DATA bitrev_size8192_mixed24_f64<>+0xE820(SB)/8, $527
DATA bitrev_size8192_mixed24_f64<>+0xE828(SB)/8, $2575
DATA bitrev_size8192_mixed24_f64<>+0xE830(SB)/8, $4623
DATA bitrev_size8192_mixed24_f64<>+0xE838(SB)/8, $6671
DATA bitrev_size8192_mixed24_f64<>+0xE840(SB)/8, $1039
DATA bitrev_size8192_mixed24_f64<>+0xE848(SB)/8, $3087
DATA bitrev_size8192_mixed24_f64<>+0xE850(SB)/8, $5135
DATA bitrev_size8192_mixed24_f64<>+0xE858(SB)/8, $7183
DATA bitrev_size8192_mixed24_f64<>+0xE860(SB)/8, $1551
DATA bitrev_size8192_mixed24_f64<>+0xE868(SB)/8, $3599
DATA bitrev_size8192_mixed24_f64<>+0xE870(SB)/8, $5647
DATA bitrev_size8192_mixed24_f64<>+0xE878(SB)/8, $7695
DATA bitrev_size8192_mixed24_f64<>+0xE880(SB)/8, $143
DATA bitrev_size8192_mixed24_f64<>+0xE888(SB)/8, $2191
DATA bitrev_size8192_mixed24_f64<>+0xE890(SB)/8, $4239
DATA bitrev_size8192_mixed24_f64<>+0xE898(SB)/8, $6287
DATA bitrev_size8192_mixed24_f64<>+0xE8A0(SB)/8, $655
DATA bitrev_size8192_mixed24_f64<>+0xE8A8(SB)/8, $2703
DATA bitrev_size8192_mixed24_f64<>+0xE8B0(SB)/8, $4751
DATA bitrev_size8192_mixed24_f64<>+0xE8B8(SB)/8, $6799
DATA bitrev_size8192_mixed24_f64<>+0xE8C0(SB)/8, $1167
DATA bitrev_size8192_mixed24_f64<>+0xE8C8(SB)/8, $3215
DATA bitrev_size8192_mixed24_f64<>+0xE8D0(SB)/8, $5263
DATA bitrev_size8192_mixed24_f64<>+0xE8D8(SB)/8, $7311
DATA bitrev_size8192_mixed24_f64<>+0xE8E0(SB)/8, $1679
DATA bitrev_size8192_mixed24_f64<>+0xE8E8(SB)/8, $3727
DATA bitrev_size8192_mixed24_f64<>+0xE8F0(SB)/8, $5775
DATA bitrev_size8192_mixed24_f64<>+0xE8F8(SB)/8, $7823
DATA bitrev_size8192_mixed24_f64<>+0xE900(SB)/8, $271
DATA bitrev_size8192_mixed24_f64<>+0xE908(SB)/8, $2319
DATA bitrev_size8192_mixed24_f64<>+0xE910(SB)/8, $4367
DATA bitrev_size8192_mixed24_f64<>+0xE918(SB)/8, $6415
DATA bitrev_size8192_mixed24_f64<>+0xE920(SB)/8, $783
DATA bitrev_size8192_mixed24_f64<>+0xE928(SB)/8, $2831
DATA bitrev_size8192_mixed24_f64<>+0xE930(SB)/8, $4879
DATA bitrev_size8192_mixed24_f64<>+0xE938(SB)/8, $6927
DATA bitrev_size8192_mixed24_f64<>+0xE940(SB)/8, $1295
DATA bitrev_size8192_mixed24_f64<>+0xE948(SB)/8, $3343
DATA bitrev_size8192_mixed24_f64<>+0xE950(SB)/8, $5391
DATA bitrev_size8192_mixed24_f64<>+0xE958(SB)/8, $7439
DATA bitrev_size8192_mixed24_f64<>+0xE960(SB)/8, $1807
DATA bitrev_size8192_mixed24_f64<>+0xE968(SB)/8, $3855
DATA bitrev_size8192_mixed24_f64<>+0xE970(SB)/8, $5903
DATA bitrev_size8192_mixed24_f64<>+0xE978(SB)/8, $7951
DATA bitrev_size8192_mixed24_f64<>+0xE980(SB)/8, $399
DATA bitrev_size8192_mixed24_f64<>+0xE988(SB)/8, $2447
DATA bitrev_size8192_mixed24_f64<>+0xE990(SB)/8, $4495
DATA bitrev_size8192_mixed24_f64<>+0xE998(SB)/8, $6543
DATA bitrev_size8192_mixed24_f64<>+0xE9A0(SB)/8, $911
DATA bitrev_size8192_mixed24_f64<>+0xE9A8(SB)/8, $2959
DATA bitrev_size8192_mixed24_f64<>+0xE9B0(SB)/8, $5007
DATA bitrev_size8192_mixed24_f64<>+0xE9B8(SB)/8, $7055
DATA bitrev_size8192_mixed24_f64<>+0xE9C0(SB)/8, $1423
DATA bitrev_size8192_mixed24_f64<>+0xE9C8(SB)/8, $3471
DATA bitrev_size8192_mixed24_f64<>+0xE9D0(SB)/8, $5519
DATA bitrev_size8192_mixed24_f64<>+0xE9D8(SB)/8, $7567
DATA bitrev_size8192_mixed24_f64<>+0xE9E0(SB)/8, $1935
DATA bitrev_size8192_mixed24_f64<>+0xE9E8(SB)/8, $3983
DATA bitrev_size8192_mixed24_f64<>+0xE9F0(SB)/8, $6031
DATA bitrev_size8192_mixed24_f64<>+0xE9F8(SB)/8, $8079
DATA bitrev_size8192_mixed24_f64<>+0xEA00(SB)/8, $47
DATA bitrev_size8192_mixed24_f64<>+0xEA08(SB)/8, $2095
DATA bitrev_size8192_mixed24_f64<>+0xEA10(SB)/8, $4143
DATA bitrev_size8192_mixed24_f64<>+0xEA18(SB)/8, $6191
DATA bitrev_size8192_mixed24_f64<>+0xEA20(SB)/8, $559
DATA bitrev_size8192_mixed24_f64<>+0xEA28(SB)/8, $2607
DATA bitrev_size8192_mixed24_f64<>+0xEA30(SB)/8, $4655
DATA bitrev_size8192_mixed24_f64<>+0xEA38(SB)/8, $6703
DATA bitrev_size8192_mixed24_f64<>+0xEA40(SB)/8, $1071
DATA bitrev_size8192_mixed24_f64<>+0xEA48(SB)/8, $3119
DATA bitrev_size8192_mixed24_f64<>+0xEA50(SB)/8, $5167
DATA bitrev_size8192_mixed24_f64<>+0xEA58(SB)/8, $7215
DATA bitrev_size8192_mixed24_f64<>+0xEA60(SB)/8, $1583
DATA bitrev_size8192_mixed24_f64<>+0xEA68(SB)/8, $3631
DATA bitrev_size8192_mixed24_f64<>+0xEA70(SB)/8, $5679
DATA bitrev_size8192_mixed24_f64<>+0xEA78(SB)/8, $7727
DATA bitrev_size8192_mixed24_f64<>+0xEA80(SB)/8, $175
DATA bitrev_size8192_mixed24_f64<>+0xEA88(SB)/8, $2223
DATA bitrev_size8192_mixed24_f64<>+0xEA90(SB)/8, $4271
DATA bitrev_size8192_mixed24_f64<>+0xEA98(SB)/8, $6319
DATA bitrev_size8192_mixed24_f64<>+0xEAA0(SB)/8, $687
DATA bitrev_size8192_mixed24_f64<>+0xEAA8(SB)/8, $2735
DATA bitrev_size8192_mixed24_f64<>+0xEAB0(SB)/8, $4783
DATA bitrev_size8192_mixed24_f64<>+0xEAB8(SB)/8, $6831
DATA bitrev_size8192_mixed24_f64<>+0xEAC0(SB)/8, $1199
DATA bitrev_size8192_mixed24_f64<>+0xEAC8(SB)/8, $3247
DATA bitrev_size8192_mixed24_f64<>+0xEAD0(SB)/8, $5295
DATA bitrev_size8192_mixed24_f64<>+0xEAD8(SB)/8, $7343
DATA bitrev_size8192_mixed24_f64<>+0xEAE0(SB)/8, $1711
DATA bitrev_size8192_mixed24_f64<>+0xEAE8(SB)/8, $3759
DATA bitrev_size8192_mixed24_f64<>+0xEAF0(SB)/8, $5807
DATA bitrev_size8192_mixed24_f64<>+0xEAF8(SB)/8, $7855
DATA bitrev_size8192_mixed24_f64<>+0xEB00(SB)/8, $303
DATA bitrev_size8192_mixed24_f64<>+0xEB08(SB)/8, $2351
DATA bitrev_size8192_mixed24_f64<>+0xEB10(SB)/8, $4399
DATA bitrev_size8192_mixed24_f64<>+0xEB18(SB)/8, $6447
DATA bitrev_size8192_mixed24_f64<>+0xEB20(SB)/8, $815
DATA bitrev_size8192_mixed24_f64<>+0xEB28(SB)/8, $2863
DATA bitrev_size8192_mixed24_f64<>+0xEB30(SB)/8, $4911
DATA bitrev_size8192_mixed24_f64<>+0xEB38(SB)/8, $6959
DATA bitrev_size8192_mixed24_f64<>+0xEB40(SB)/8, $1327
DATA bitrev_size8192_mixed24_f64<>+0xEB48(SB)/8, $3375
DATA bitrev_size8192_mixed24_f64<>+0xEB50(SB)/8, $5423
DATA bitrev_size8192_mixed24_f64<>+0xEB58(SB)/8, $7471
DATA bitrev_size8192_mixed24_f64<>+0xEB60(SB)/8, $1839
DATA bitrev_size8192_mixed24_f64<>+0xEB68(SB)/8, $3887
DATA bitrev_size8192_mixed24_f64<>+0xEB70(SB)/8, $5935
DATA bitrev_size8192_mixed24_f64<>+0xEB78(SB)/8, $7983
DATA bitrev_size8192_mixed24_f64<>+0xEB80(SB)/8, $431
DATA bitrev_size8192_mixed24_f64<>+0xEB88(SB)/8, $2479
DATA bitrev_size8192_mixed24_f64<>+0xEB90(SB)/8, $4527
DATA bitrev_size8192_mixed24_f64<>+0xEB98(SB)/8, $6575
DATA bitrev_size8192_mixed24_f64<>+0xEBA0(SB)/8, $943
DATA bitrev_size8192_mixed24_f64<>+0xEBA8(SB)/8, $2991
DATA bitrev_size8192_mixed24_f64<>+0xEBB0(SB)/8, $5039
DATA bitrev_size8192_mixed24_f64<>+0xEBB8(SB)/8, $7087
DATA bitrev_size8192_mixed24_f64<>+0xEBC0(SB)/8, $1455
DATA bitrev_size8192_mixed24_f64<>+0xEBC8(SB)/8, $3503
DATA bitrev_size8192_mixed24_f64<>+0xEBD0(SB)/8, $5551
DATA bitrev_size8192_mixed24_f64<>+0xEBD8(SB)/8, $7599
DATA bitrev_size8192_mixed24_f64<>+0xEBE0(SB)/8, $1967
DATA bitrev_size8192_mixed24_f64<>+0xEBE8(SB)/8, $4015
DATA bitrev_size8192_mixed24_f64<>+0xEBF0(SB)/8, $6063
DATA bitrev_size8192_mixed24_f64<>+0xEBF8(SB)/8, $8111
DATA bitrev_size8192_mixed24_f64<>+0xEC00(SB)/8, $79
DATA bitrev_size8192_mixed24_f64<>+0xEC08(SB)/8, $2127
DATA bitrev_size8192_mixed24_f64<>+0xEC10(SB)/8, $4175
DATA bitrev_size8192_mixed24_f64<>+0xEC18(SB)/8, $6223
DATA bitrev_size8192_mixed24_f64<>+0xEC20(SB)/8, $591
DATA bitrev_size8192_mixed24_f64<>+0xEC28(SB)/8, $2639
DATA bitrev_size8192_mixed24_f64<>+0xEC30(SB)/8, $4687
DATA bitrev_size8192_mixed24_f64<>+0xEC38(SB)/8, $6735
DATA bitrev_size8192_mixed24_f64<>+0xEC40(SB)/8, $1103
DATA bitrev_size8192_mixed24_f64<>+0xEC48(SB)/8, $3151
DATA bitrev_size8192_mixed24_f64<>+0xEC50(SB)/8, $5199
DATA bitrev_size8192_mixed24_f64<>+0xEC58(SB)/8, $7247
DATA bitrev_size8192_mixed24_f64<>+0xEC60(SB)/8, $1615
DATA bitrev_size8192_mixed24_f64<>+0xEC68(SB)/8, $3663
DATA bitrev_size8192_mixed24_f64<>+0xEC70(SB)/8, $5711
DATA bitrev_size8192_mixed24_f64<>+0xEC78(SB)/8, $7759
DATA bitrev_size8192_mixed24_f64<>+0xEC80(SB)/8, $207
DATA bitrev_size8192_mixed24_f64<>+0xEC88(SB)/8, $2255
DATA bitrev_size8192_mixed24_f64<>+0xEC90(SB)/8, $4303
DATA bitrev_size8192_mixed24_f64<>+0xEC98(SB)/8, $6351
DATA bitrev_size8192_mixed24_f64<>+0xECA0(SB)/8, $719
DATA bitrev_size8192_mixed24_f64<>+0xECA8(SB)/8, $2767
DATA bitrev_size8192_mixed24_f64<>+0xECB0(SB)/8, $4815
DATA bitrev_size8192_mixed24_f64<>+0xECB8(SB)/8, $6863
DATA bitrev_size8192_mixed24_f64<>+0xECC0(SB)/8, $1231
DATA bitrev_size8192_mixed24_f64<>+0xECC8(SB)/8, $3279
DATA bitrev_size8192_mixed24_f64<>+0xECD0(SB)/8, $5327
DATA bitrev_size8192_mixed24_f64<>+0xECD8(SB)/8, $7375
DATA bitrev_size8192_mixed24_f64<>+0xECE0(SB)/8, $1743
DATA bitrev_size8192_mixed24_f64<>+0xECE8(SB)/8, $3791
DATA bitrev_size8192_mixed24_f64<>+0xECF0(SB)/8, $5839
DATA bitrev_size8192_mixed24_f64<>+0xECF8(SB)/8, $7887
DATA bitrev_size8192_mixed24_f64<>+0xED00(SB)/8, $335
DATA bitrev_size8192_mixed24_f64<>+0xED08(SB)/8, $2383
DATA bitrev_size8192_mixed24_f64<>+0xED10(SB)/8, $4431
DATA bitrev_size8192_mixed24_f64<>+0xED18(SB)/8, $6479
DATA bitrev_size8192_mixed24_f64<>+0xED20(SB)/8, $847
DATA bitrev_size8192_mixed24_f64<>+0xED28(SB)/8, $2895
DATA bitrev_size8192_mixed24_f64<>+0xED30(SB)/8, $4943
DATA bitrev_size8192_mixed24_f64<>+0xED38(SB)/8, $6991
DATA bitrev_size8192_mixed24_f64<>+0xED40(SB)/8, $1359
DATA bitrev_size8192_mixed24_f64<>+0xED48(SB)/8, $3407
DATA bitrev_size8192_mixed24_f64<>+0xED50(SB)/8, $5455
DATA bitrev_size8192_mixed24_f64<>+0xED58(SB)/8, $7503
DATA bitrev_size8192_mixed24_f64<>+0xED60(SB)/8, $1871
DATA bitrev_size8192_mixed24_f64<>+0xED68(SB)/8, $3919
DATA bitrev_size8192_mixed24_f64<>+0xED70(SB)/8, $5967
DATA bitrev_size8192_mixed24_f64<>+0xED78(SB)/8, $8015
DATA bitrev_size8192_mixed24_f64<>+0xED80(SB)/8, $463
DATA bitrev_size8192_mixed24_f64<>+0xED88(SB)/8, $2511
DATA bitrev_size8192_mixed24_f64<>+0xED90(SB)/8, $4559
DATA bitrev_size8192_mixed24_f64<>+0xED98(SB)/8, $6607
DATA bitrev_size8192_mixed24_f64<>+0xEDA0(SB)/8, $975
DATA bitrev_size8192_mixed24_f64<>+0xEDA8(SB)/8, $3023
DATA bitrev_size8192_mixed24_f64<>+0xEDB0(SB)/8, $5071
DATA bitrev_size8192_mixed24_f64<>+0xEDB8(SB)/8, $7119
DATA bitrev_size8192_mixed24_f64<>+0xEDC0(SB)/8, $1487
DATA bitrev_size8192_mixed24_f64<>+0xEDC8(SB)/8, $3535
DATA bitrev_size8192_mixed24_f64<>+0xEDD0(SB)/8, $5583
DATA bitrev_size8192_mixed24_f64<>+0xEDD8(SB)/8, $7631
DATA bitrev_size8192_mixed24_f64<>+0xEDE0(SB)/8, $1999
DATA bitrev_size8192_mixed24_f64<>+0xEDE8(SB)/8, $4047
DATA bitrev_size8192_mixed24_f64<>+0xEDF0(SB)/8, $6095
DATA bitrev_size8192_mixed24_f64<>+0xEDF8(SB)/8, $8143
DATA bitrev_size8192_mixed24_f64<>+0xEE00(SB)/8, $111
DATA bitrev_size8192_mixed24_f64<>+0xEE08(SB)/8, $2159
DATA bitrev_size8192_mixed24_f64<>+0xEE10(SB)/8, $4207
DATA bitrev_size8192_mixed24_f64<>+0xEE18(SB)/8, $6255
DATA bitrev_size8192_mixed24_f64<>+0xEE20(SB)/8, $623
DATA bitrev_size8192_mixed24_f64<>+0xEE28(SB)/8, $2671
DATA bitrev_size8192_mixed24_f64<>+0xEE30(SB)/8, $4719
DATA bitrev_size8192_mixed24_f64<>+0xEE38(SB)/8, $6767
DATA bitrev_size8192_mixed24_f64<>+0xEE40(SB)/8, $1135
DATA bitrev_size8192_mixed24_f64<>+0xEE48(SB)/8, $3183
DATA bitrev_size8192_mixed24_f64<>+0xEE50(SB)/8, $5231
DATA bitrev_size8192_mixed24_f64<>+0xEE58(SB)/8, $7279
DATA bitrev_size8192_mixed24_f64<>+0xEE60(SB)/8, $1647
DATA bitrev_size8192_mixed24_f64<>+0xEE68(SB)/8, $3695
DATA bitrev_size8192_mixed24_f64<>+0xEE70(SB)/8, $5743
DATA bitrev_size8192_mixed24_f64<>+0xEE78(SB)/8, $7791
DATA bitrev_size8192_mixed24_f64<>+0xEE80(SB)/8, $239
DATA bitrev_size8192_mixed24_f64<>+0xEE88(SB)/8, $2287
DATA bitrev_size8192_mixed24_f64<>+0xEE90(SB)/8, $4335
DATA bitrev_size8192_mixed24_f64<>+0xEE98(SB)/8, $6383
DATA bitrev_size8192_mixed24_f64<>+0xEEA0(SB)/8, $751
DATA bitrev_size8192_mixed24_f64<>+0xEEA8(SB)/8, $2799
DATA bitrev_size8192_mixed24_f64<>+0xEEB0(SB)/8, $4847
DATA bitrev_size8192_mixed24_f64<>+0xEEB8(SB)/8, $6895
DATA bitrev_size8192_mixed24_f64<>+0xEEC0(SB)/8, $1263
DATA bitrev_size8192_mixed24_f64<>+0xEEC8(SB)/8, $3311
DATA bitrev_size8192_mixed24_f64<>+0xEED0(SB)/8, $5359
DATA bitrev_size8192_mixed24_f64<>+0xEED8(SB)/8, $7407
DATA bitrev_size8192_mixed24_f64<>+0xEEE0(SB)/8, $1775
DATA bitrev_size8192_mixed24_f64<>+0xEEE8(SB)/8, $3823
DATA bitrev_size8192_mixed24_f64<>+0xEEF0(SB)/8, $5871
DATA bitrev_size8192_mixed24_f64<>+0xEEF8(SB)/8, $7919
DATA bitrev_size8192_mixed24_f64<>+0xEF00(SB)/8, $367
DATA bitrev_size8192_mixed24_f64<>+0xEF08(SB)/8, $2415
DATA bitrev_size8192_mixed24_f64<>+0xEF10(SB)/8, $4463
DATA bitrev_size8192_mixed24_f64<>+0xEF18(SB)/8, $6511
DATA bitrev_size8192_mixed24_f64<>+0xEF20(SB)/8, $879
DATA bitrev_size8192_mixed24_f64<>+0xEF28(SB)/8, $2927
DATA bitrev_size8192_mixed24_f64<>+0xEF30(SB)/8, $4975
DATA bitrev_size8192_mixed24_f64<>+0xEF38(SB)/8, $7023
DATA bitrev_size8192_mixed24_f64<>+0xEF40(SB)/8, $1391
DATA bitrev_size8192_mixed24_f64<>+0xEF48(SB)/8, $3439
DATA bitrev_size8192_mixed24_f64<>+0xEF50(SB)/8, $5487
DATA bitrev_size8192_mixed24_f64<>+0xEF58(SB)/8, $7535
DATA bitrev_size8192_mixed24_f64<>+0xEF60(SB)/8, $1903
DATA bitrev_size8192_mixed24_f64<>+0xEF68(SB)/8, $3951
DATA bitrev_size8192_mixed24_f64<>+0xEF70(SB)/8, $5999
DATA bitrev_size8192_mixed24_f64<>+0xEF78(SB)/8, $8047
DATA bitrev_size8192_mixed24_f64<>+0xEF80(SB)/8, $495
DATA bitrev_size8192_mixed24_f64<>+0xEF88(SB)/8, $2543
DATA bitrev_size8192_mixed24_f64<>+0xEF90(SB)/8, $4591
DATA bitrev_size8192_mixed24_f64<>+0xEF98(SB)/8, $6639
DATA bitrev_size8192_mixed24_f64<>+0xEFA0(SB)/8, $1007
DATA bitrev_size8192_mixed24_f64<>+0xEFA8(SB)/8, $3055
DATA bitrev_size8192_mixed24_f64<>+0xEFB0(SB)/8, $5103
DATA bitrev_size8192_mixed24_f64<>+0xEFB8(SB)/8, $7151
DATA bitrev_size8192_mixed24_f64<>+0xEFC0(SB)/8, $1519
DATA bitrev_size8192_mixed24_f64<>+0xEFC8(SB)/8, $3567
DATA bitrev_size8192_mixed24_f64<>+0xEFD0(SB)/8, $5615
DATA bitrev_size8192_mixed24_f64<>+0xEFD8(SB)/8, $7663
DATA bitrev_size8192_mixed24_f64<>+0xEFE0(SB)/8, $2031
DATA bitrev_size8192_mixed24_f64<>+0xEFE8(SB)/8, $4079
DATA bitrev_size8192_mixed24_f64<>+0xEFF0(SB)/8, $6127
DATA bitrev_size8192_mixed24_f64<>+0xEFF8(SB)/8, $8175
DATA bitrev_size8192_mixed24_f64<>+0xF000(SB)/8, $23
DATA bitrev_size8192_mixed24_f64<>+0xF008(SB)/8, $2071
DATA bitrev_size8192_mixed24_f64<>+0xF010(SB)/8, $4119
DATA bitrev_size8192_mixed24_f64<>+0xF018(SB)/8, $6167
DATA bitrev_size8192_mixed24_f64<>+0xF020(SB)/8, $535
DATA bitrev_size8192_mixed24_f64<>+0xF028(SB)/8, $2583
DATA bitrev_size8192_mixed24_f64<>+0xF030(SB)/8, $4631
DATA bitrev_size8192_mixed24_f64<>+0xF038(SB)/8, $6679
DATA bitrev_size8192_mixed24_f64<>+0xF040(SB)/8, $1047
DATA bitrev_size8192_mixed24_f64<>+0xF048(SB)/8, $3095
DATA bitrev_size8192_mixed24_f64<>+0xF050(SB)/8, $5143
DATA bitrev_size8192_mixed24_f64<>+0xF058(SB)/8, $7191
DATA bitrev_size8192_mixed24_f64<>+0xF060(SB)/8, $1559
DATA bitrev_size8192_mixed24_f64<>+0xF068(SB)/8, $3607
DATA bitrev_size8192_mixed24_f64<>+0xF070(SB)/8, $5655
DATA bitrev_size8192_mixed24_f64<>+0xF078(SB)/8, $7703
DATA bitrev_size8192_mixed24_f64<>+0xF080(SB)/8, $151
DATA bitrev_size8192_mixed24_f64<>+0xF088(SB)/8, $2199
DATA bitrev_size8192_mixed24_f64<>+0xF090(SB)/8, $4247
DATA bitrev_size8192_mixed24_f64<>+0xF098(SB)/8, $6295
DATA bitrev_size8192_mixed24_f64<>+0xF0A0(SB)/8, $663
DATA bitrev_size8192_mixed24_f64<>+0xF0A8(SB)/8, $2711
DATA bitrev_size8192_mixed24_f64<>+0xF0B0(SB)/8, $4759
DATA bitrev_size8192_mixed24_f64<>+0xF0B8(SB)/8, $6807
DATA bitrev_size8192_mixed24_f64<>+0xF0C0(SB)/8, $1175
DATA bitrev_size8192_mixed24_f64<>+0xF0C8(SB)/8, $3223
DATA bitrev_size8192_mixed24_f64<>+0xF0D0(SB)/8, $5271
DATA bitrev_size8192_mixed24_f64<>+0xF0D8(SB)/8, $7319
DATA bitrev_size8192_mixed24_f64<>+0xF0E0(SB)/8, $1687
DATA bitrev_size8192_mixed24_f64<>+0xF0E8(SB)/8, $3735
DATA bitrev_size8192_mixed24_f64<>+0xF0F0(SB)/8, $5783
DATA bitrev_size8192_mixed24_f64<>+0xF0F8(SB)/8, $7831
DATA bitrev_size8192_mixed24_f64<>+0xF100(SB)/8, $279
DATA bitrev_size8192_mixed24_f64<>+0xF108(SB)/8, $2327
DATA bitrev_size8192_mixed24_f64<>+0xF110(SB)/8, $4375
DATA bitrev_size8192_mixed24_f64<>+0xF118(SB)/8, $6423
DATA bitrev_size8192_mixed24_f64<>+0xF120(SB)/8, $791
DATA bitrev_size8192_mixed24_f64<>+0xF128(SB)/8, $2839
DATA bitrev_size8192_mixed24_f64<>+0xF130(SB)/8, $4887
DATA bitrev_size8192_mixed24_f64<>+0xF138(SB)/8, $6935
DATA bitrev_size8192_mixed24_f64<>+0xF140(SB)/8, $1303
DATA bitrev_size8192_mixed24_f64<>+0xF148(SB)/8, $3351
DATA bitrev_size8192_mixed24_f64<>+0xF150(SB)/8, $5399
DATA bitrev_size8192_mixed24_f64<>+0xF158(SB)/8, $7447
DATA bitrev_size8192_mixed24_f64<>+0xF160(SB)/8, $1815
DATA bitrev_size8192_mixed24_f64<>+0xF168(SB)/8, $3863
DATA bitrev_size8192_mixed24_f64<>+0xF170(SB)/8, $5911
DATA bitrev_size8192_mixed24_f64<>+0xF178(SB)/8, $7959
DATA bitrev_size8192_mixed24_f64<>+0xF180(SB)/8, $407
DATA bitrev_size8192_mixed24_f64<>+0xF188(SB)/8, $2455
DATA bitrev_size8192_mixed24_f64<>+0xF190(SB)/8, $4503
DATA bitrev_size8192_mixed24_f64<>+0xF198(SB)/8, $6551
DATA bitrev_size8192_mixed24_f64<>+0xF1A0(SB)/8, $919
DATA bitrev_size8192_mixed24_f64<>+0xF1A8(SB)/8, $2967
DATA bitrev_size8192_mixed24_f64<>+0xF1B0(SB)/8, $5015
DATA bitrev_size8192_mixed24_f64<>+0xF1B8(SB)/8, $7063
DATA bitrev_size8192_mixed24_f64<>+0xF1C0(SB)/8, $1431
DATA bitrev_size8192_mixed24_f64<>+0xF1C8(SB)/8, $3479
DATA bitrev_size8192_mixed24_f64<>+0xF1D0(SB)/8, $5527
DATA bitrev_size8192_mixed24_f64<>+0xF1D8(SB)/8, $7575
DATA bitrev_size8192_mixed24_f64<>+0xF1E0(SB)/8, $1943
DATA bitrev_size8192_mixed24_f64<>+0xF1E8(SB)/8, $3991
DATA bitrev_size8192_mixed24_f64<>+0xF1F0(SB)/8, $6039
DATA bitrev_size8192_mixed24_f64<>+0xF1F8(SB)/8, $8087
DATA bitrev_size8192_mixed24_f64<>+0xF200(SB)/8, $55
DATA bitrev_size8192_mixed24_f64<>+0xF208(SB)/8, $2103
DATA bitrev_size8192_mixed24_f64<>+0xF210(SB)/8, $4151
DATA bitrev_size8192_mixed24_f64<>+0xF218(SB)/8, $6199
DATA bitrev_size8192_mixed24_f64<>+0xF220(SB)/8, $567
DATA bitrev_size8192_mixed24_f64<>+0xF228(SB)/8, $2615
DATA bitrev_size8192_mixed24_f64<>+0xF230(SB)/8, $4663
DATA bitrev_size8192_mixed24_f64<>+0xF238(SB)/8, $6711
DATA bitrev_size8192_mixed24_f64<>+0xF240(SB)/8, $1079
DATA bitrev_size8192_mixed24_f64<>+0xF248(SB)/8, $3127
DATA bitrev_size8192_mixed24_f64<>+0xF250(SB)/8, $5175
DATA bitrev_size8192_mixed24_f64<>+0xF258(SB)/8, $7223
DATA bitrev_size8192_mixed24_f64<>+0xF260(SB)/8, $1591
DATA bitrev_size8192_mixed24_f64<>+0xF268(SB)/8, $3639
DATA bitrev_size8192_mixed24_f64<>+0xF270(SB)/8, $5687
DATA bitrev_size8192_mixed24_f64<>+0xF278(SB)/8, $7735
DATA bitrev_size8192_mixed24_f64<>+0xF280(SB)/8, $183
DATA bitrev_size8192_mixed24_f64<>+0xF288(SB)/8, $2231
DATA bitrev_size8192_mixed24_f64<>+0xF290(SB)/8, $4279
DATA bitrev_size8192_mixed24_f64<>+0xF298(SB)/8, $6327
DATA bitrev_size8192_mixed24_f64<>+0xF2A0(SB)/8, $695
DATA bitrev_size8192_mixed24_f64<>+0xF2A8(SB)/8, $2743
DATA bitrev_size8192_mixed24_f64<>+0xF2B0(SB)/8, $4791
DATA bitrev_size8192_mixed24_f64<>+0xF2B8(SB)/8, $6839
DATA bitrev_size8192_mixed24_f64<>+0xF2C0(SB)/8, $1207
DATA bitrev_size8192_mixed24_f64<>+0xF2C8(SB)/8, $3255
DATA bitrev_size8192_mixed24_f64<>+0xF2D0(SB)/8, $5303
DATA bitrev_size8192_mixed24_f64<>+0xF2D8(SB)/8, $7351
DATA bitrev_size8192_mixed24_f64<>+0xF2E0(SB)/8, $1719
DATA bitrev_size8192_mixed24_f64<>+0xF2E8(SB)/8, $3767
DATA bitrev_size8192_mixed24_f64<>+0xF2F0(SB)/8, $5815
DATA bitrev_size8192_mixed24_f64<>+0xF2F8(SB)/8, $7863
DATA bitrev_size8192_mixed24_f64<>+0xF300(SB)/8, $311
DATA bitrev_size8192_mixed24_f64<>+0xF308(SB)/8, $2359
DATA bitrev_size8192_mixed24_f64<>+0xF310(SB)/8, $4407
DATA bitrev_size8192_mixed24_f64<>+0xF318(SB)/8, $6455
DATA bitrev_size8192_mixed24_f64<>+0xF320(SB)/8, $823
DATA bitrev_size8192_mixed24_f64<>+0xF328(SB)/8, $2871
DATA bitrev_size8192_mixed24_f64<>+0xF330(SB)/8, $4919
DATA bitrev_size8192_mixed24_f64<>+0xF338(SB)/8, $6967
DATA bitrev_size8192_mixed24_f64<>+0xF340(SB)/8, $1335
DATA bitrev_size8192_mixed24_f64<>+0xF348(SB)/8, $3383
DATA bitrev_size8192_mixed24_f64<>+0xF350(SB)/8, $5431
DATA bitrev_size8192_mixed24_f64<>+0xF358(SB)/8, $7479
DATA bitrev_size8192_mixed24_f64<>+0xF360(SB)/8, $1847
DATA bitrev_size8192_mixed24_f64<>+0xF368(SB)/8, $3895
DATA bitrev_size8192_mixed24_f64<>+0xF370(SB)/8, $5943
DATA bitrev_size8192_mixed24_f64<>+0xF378(SB)/8, $7991
DATA bitrev_size8192_mixed24_f64<>+0xF380(SB)/8, $439
DATA bitrev_size8192_mixed24_f64<>+0xF388(SB)/8, $2487
DATA bitrev_size8192_mixed24_f64<>+0xF390(SB)/8, $4535
DATA bitrev_size8192_mixed24_f64<>+0xF398(SB)/8, $6583
DATA bitrev_size8192_mixed24_f64<>+0xF3A0(SB)/8, $951
DATA bitrev_size8192_mixed24_f64<>+0xF3A8(SB)/8, $2999
DATA bitrev_size8192_mixed24_f64<>+0xF3B0(SB)/8, $5047
DATA bitrev_size8192_mixed24_f64<>+0xF3B8(SB)/8, $7095
DATA bitrev_size8192_mixed24_f64<>+0xF3C0(SB)/8, $1463
DATA bitrev_size8192_mixed24_f64<>+0xF3C8(SB)/8, $3511
DATA bitrev_size8192_mixed24_f64<>+0xF3D0(SB)/8, $5559
DATA bitrev_size8192_mixed24_f64<>+0xF3D8(SB)/8, $7607
DATA bitrev_size8192_mixed24_f64<>+0xF3E0(SB)/8, $1975
DATA bitrev_size8192_mixed24_f64<>+0xF3E8(SB)/8, $4023
DATA bitrev_size8192_mixed24_f64<>+0xF3F0(SB)/8, $6071
DATA bitrev_size8192_mixed24_f64<>+0xF3F8(SB)/8, $8119
DATA bitrev_size8192_mixed24_f64<>+0xF400(SB)/8, $87
DATA bitrev_size8192_mixed24_f64<>+0xF408(SB)/8, $2135
DATA bitrev_size8192_mixed24_f64<>+0xF410(SB)/8, $4183
DATA bitrev_size8192_mixed24_f64<>+0xF418(SB)/8, $6231
DATA bitrev_size8192_mixed24_f64<>+0xF420(SB)/8, $599
DATA bitrev_size8192_mixed24_f64<>+0xF428(SB)/8, $2647
DATA bitrev_size8192_mixed24_f64<>+0xF430(SB)/8, $4695
DATA bitrev_size8192_mixed24_f64<>+0xF438(SB)/8, $6743
DATA bitrev_size8192_mixed24_f64<>+0xF440(SB)/8, $1111
DATA bitrev_size8192_mixed24_f64<>+0xF448(SB)/8, $3159
DATA bitrev_size8192_mixed24_f64<>+0xF450(SB)/8, $5207
DATA bitrev_size8192_mixed24_f64<>+0xF458(SB)/8, $7255
DATA bitrev_size8192_mixed24_f64<>+0xF460(SB)/8, $1623
DATA bitrev_size8192_mixed24_f64<>+0xF468(SB)/8, $3671
DATA bitrev_size8192_mixed24_f64<>+0xF470(SB)/8, $5719
DATA bitrev_size8192_mixed24_f64<>+0xF478(SB)/8, $7767
DATA bitrev_size8192_mixed24_f64<>+0xF480(SB)/8, $215
DATA bitrev_size8192_mixed24_f64<>+0xF488(SB)/8, $2263
DATA bitrev_size8192_mixed24_f64<>+0xF490(SB)/8, $4311
DATA bitrev_size8192_mixed24_f64<>+0xF498(SB)/8, $6359
DATA bitrev_size8192_mixed24_f64<>+0xF4A0(SB)/8, $727
DATA bitrev_size8192_mixed24_f64<>+0xF4A8(SB)/8, $2775
DATA bitrev_size8192_mixed24_f64<>+0xF4B0(SB)/8, $4823
DATA bitrev_size8192_mixed24_f64<>+0xF4B8(SB)/8, $6871
DATA bitrev_size8192_mixed24_f64<>+0xF4C0(SB)/8, $1239
DATA bitrev_size8192_mixed24_f64<>+0xF4C8(SB)/8, $3287
DATA bitrev_size8192_mixed24_f64<>+0xF4D0(SB)/8, $5335
DATA bitrev_size8192_mixed24_f64<>+0xF4D8(SB)/8, $7383
DATA bitrev_size8192_mixed24_f64<>+0xF4E0(SB)/8, $1751
DATA bitrev_size8192_mixed24_f64<>+0xF4E8(SB)/8, $3799
DATA bitrev_size8192_mixed24_f64<>+0xF4F0(SB)/8, $5847
DATA bitrev_size8192_mixed24_f64<>+0xF4F8(SB)/8, $7895
DATA bitrev_size8192_mixed24_f64<>+0xF500(SB)/8, $343
DATA bitrev_size8192_mixed24_f64<>+0xF508(SB)/8, $2391
DATA bitrev_size8192_mixed24_f64<>+0xF510(SB)/8, $4439
DATA bitrev_size8192_mixed24_f64<>+0xF518(SB)/8, $6487
DATA bitrev_size8192_mixed24_f64<>+0xF520(SB)/8, $855
DATA bitrev_size8192_mixed24_f64<>+0xF528(SB)/8, $2903
DATA bitrev_size8192_mixed24_f64<>+0xF530(SB)/8, $4951
DATA bitrev_size8192_mixed24_f64<>+0xF538(SB)/8, $6999
DATA bitrev_size8192_mixed24_f64<>+0xF540(SB)/8, $1367
DATA bitrev_size8192_mixed24_f64<>+0xF548(SB)/8, $3415
DATA bitrev_size8192_mixed24_f64<>+0xF550(SB)/8, $5463
DATA bitrev_size8192_mixed24_f64<>+0xF558(SB)/8, $7511
DATA bitrev_size8192_mixed24_f64<>+0xF560(SB)/8, $1879
DATA bitrev_size8192_mixed24_f64<>+0xF568(SB)/8, $3927
DATA bitrev_size8192_mixed24_f64<>+0xF570(SB)/8, $5975
DATA bitrev_size8192_mixed24_f64<>+0xF578(SB)/8, $8023
DATA bitrev_size8192_mixed24_f64<>+0xF580(SB)/8, $471
DATA bitrev_size8192_mixed24_f64<>+0xF588(SB)/8, $2519
DATA bitrev_size8192_mixed24_f64<>+0xF590(SB)/8, $4567
DATA bitrev_size8192_mixed24_f64<>+0xF598(SB)/8, $6615
DATA bitrev_size8192_mixed24_f64<>+0xF5A0(SB)/8, $983
DATA bitrev_size8192_mixed24_f64<>+0xF5A8(SB)/8, $3031
DATA bitrev_size8192_mixed24_f64<>+0xF5B0(SB)/8, $5079
DATA bitrev_size8192_mixed24_f64<>+0xF5B8(SB)/8, $7127
DATA bitrev_size8192_mixed24_f64<>+0xF5C0(SB)/8, $1495
DATA bitrev_size8192_mixed24_f64<>+0xF5C8(SB)/8, $3543
DATA bitrev_size8192_mixed24_f64<>+0xF5D0(SB)/8, $5591
DATA bitrev_size8192_mixed24_f64<>+0xF5D8(SB)/8, $7639
DATA bitrev_size8192_mixed24_f64<>+0xF5E0(SB)/8, $2007
DATA bitrev_size8192_mixed24_f64<>+0xF5E8(SB)/8, $4055
DATA bitrev_size8192_mixed24_f64<>+0xF5F0(SB)/8, $6103
DATA bitrev_size8192_mixed24_f64<>+0xF5F8(SB)/8, $8151
DATA bitrev_size8192_mixed24_f64<>+0xF600(SB)/8, $119
DATA bitrev_size8192_mixed24_f64<>+0xF608(SB)/8, $2167
DATA bitrev_size8192_mixed24_f64<>+0xF610(SB)/8, $4215
DATA bitrev_size8192_mixed24_f64<>+0xF618(SB)/8, $6263
DATA bitrev_size8192_mixed24_f64<>+0xF620(SB)/8, $631
DATA bitrev_size8192_mixed24_f64<>+0xF628(SB)/8, $2679
DATA bitrev_size8192_mixed24_f64<>+0xF630(SB)/8, $4727
DATA bitrev_size8192_mixed24_f64<>+0xF638(SB)/8, $6775
DATA bitrev_size8192_mixed24_f64<>+0xF640(SB)/8, $1143
DATA bitrev_size8192_mixed24_f64<>+0xF648(SB)/8, $3191
DATA bitrev_size8192_mixed24_f64<>+0xF650(SB)/8, $5239
DATA bitrev_size8192_mixed24_f64<>+0xF658(SB)/8, $7287
DATA bitrev_size8192_mixed24_f64<>+0xF660(SB)/8, $1655
DATA bitrev_size8192_mixed24_f64<>+0xF668(SB)/8, $3703
DATA bitrev_size8192_mixed24_f64<>+0xF670(SB)/8, $5751
DATA bitrev_size8192_mixed24_f64<>+0xF678(SB)/8, $7799
DATA bitrev_size8192_mixed24_f64<>+0xF680(SB)/8, $247
DATA bitrev_size8192_mixed24_f64<>+0xF688(SB)/8, $2295
DATA bitrev_size8192_mixed24_f64<>+0xF690(SB)/8, $4343
DATA bitrev_size8192_mixed24_f64<>+0xF698(SB)/8, $6391
DATA bitrev_size8192_mixed24_f64<>+0xF6A0(SB)/8, $759
DATA bitrev_size8192_mixed24_f64<>+0xF6A8(SB)/8, $2807
DATA bitrev_size8192_mixed24_f64<>+0xF6B0(SB)/8, $4855
DATA bitrev_size8192_mixed24_f64<>+0xF6B8(SB)/8, $6903
DATA bitrev_size8192_mixed24_f64<>+0xF6C0(SB)/8, $1271
DATA bitrev_size8192_mixed24_f64<>+0xF6C8(SB)/8, $3319
DATA bitrev_size8192_mixed24_f64<>+0xF6D0(SB)/8, $5367
DATA bitrev_size8192_mixed24_f64<>+0xF6D8(SB)/8, $7415
DATA bitrev_size8192_mixed24_f64<>+0xF6E0(SB)/8, $1783
DATA bitrev_size8192_mixed24_f64<>+0xF6E8(SB)/8, $3831
DATA bitrev_size8192_mixed24_f64<>+0xF6F0(SB)/8, $5879
DATA bitrev_size8192_mixed24_f64<>+0xF6F8(SB)/8, $7927
DATA bitrev_size8192_mixed24_f64<>+0xF700(SB)/8, $375
DATA bitrev_size8192_mixed24_f64<>+0xF708(SB)/8, $2423
DATA bitrev_size8192_mixed24_f64<>+0xF710(SB)/8, $4471
DATA bitrev_size8192_mixed24_f64<>+0xF718(SB)/8, $6519
DATA bitrev_size8192_mixed24_f64<>+0xF720(SB)/8, $887
DATA bitrev_size8192_mixed24_f64<>+0xF728(SB)/8, $2935
DATA bitrev_size8192_mixed24_f64<>+0xF730(SB)/8, $4983
DATA bitrev_size8192_mixed24_f64<>+0xF738(SB)/8, $7031
DATA bitrev_size8192_mixed24_f64<>+0xF740(SB)/8, $1399
DATA bitrev_size8192_mixed24_f64<>+0xF748(SB)/8, $3447
DATA bitrev_size8192_mixed24_f64<>+0xF750(SB)/8, $5495
DATA bitrev_size8192_mixed24_f64<>+0xF758(SB)/8, $7543
DATA bitrev_size8192_mixed24_f64<>+0xF760(SB)/8, $1911
DATA bitrev_size8192_mixed24_f64<>+0xF768(SB)/8, $3959
DATA bitrev_size8192_mixed24_f64<>+0xF770(SB)/8, $6007
DATA bitrev_size8192_mixed24_f64<>+0xF778(SB)/8, $8055
DATA bitrev_size8192_mixed24_f64<>+0xF780(SB)/8, $503
DATA bitrev_size8192_mixed24_f64<>+0xF788(SB)/8, $2551
DATA bitrev_size8192_mixed24_f64<>+0xF790(SB)/8, $4599
DATA bitrev_size8192_mixed24_f64<>+0xF798(SB)/8, $6647
DATA bitrev_size8192_mixed24_f64<>+0xF7A0(SB)/8, $1015
DATA bitrev_size8192_mixed24_f64<>+0xF7A8(SB)/8, $3063
DATA bitrev_size8192_mixed24_f64<>+0xF7B0(SB)/8, $5111
DATA bitrev_size8192_mixed24_f64<>+0xF7B8(SB)/8, $7159
DATA bitrev_size8192_mixed24_f64<>+0xF7C0(SB)/8, $1527
DATA bitrev_size8192_mixed24_f64<>+0xF7C8(SB)/8, $3575
DATA bitrev_size8192_mixed24_f64<>+0xF7D0(SB)/8, $5623
DATA bitrev_size8192_mixed24_f64<>+0xF7D8(SB)/8, $7671
DATA bitrev_size8192_mixed24_f64<>+0xF7E0(SB)/8, $2039
DATA bitrev_size8192_mixed24_f64<>+0xF7E8(SB)/8, $4087
DATA bitrev_size8192_mixed24_f64<>+0xF7F0(SB)/8, $6135
DATA bitrev_size8192_mixed24_f64<>+0xF7F8(SB)/8, $8183
DATA bitrev_size8192_mixed24_f64<>+0xF800(SB)/8, $31
DATA bitrev_size8192_mixed24_f64<>+0xF808(SB)/8, $2079
DATA bitrev_size8192_mixed24_f64<>+0xF810(SB)/8, $4127
DATA bitrev_size8192_mixed24_f64<>+0xF818(SB)/8, $6175
DATA bitrev_size8192_mixed24_f64<>+0xF820(SB)/8, $543
DATA bitrev_size8192_mixed24_f64<>+0xF828(SB)/8, $2591
DATA bitrev_size8192_mixed24_f64<>+0xF830(SB)/8, $4639
DATA bitrev_size8192_mixed24_f64<>+0xF838(SB)/8, $6687
DATA bitrev_size8192_mixed24_f64<>+0xF840(SB)/8, $1055
DATA bitrev_size8192_mixed24_f64<>+0xF848(SB)/8, $3103
DATA bitrev_size8192_mixed24_f64<>+0xF850(SB)/8, $5151
DATA bitrev_size8192_mixed24_f64<>+0xF858(SB)/8, $7199
DATA bitrev_size8192_mixed24_f64<>+0xF860(SB)/8, $1567
DATA bitrev_size8192_mixed24_f64<>+0xF868(SB)/8, $3615
DATA bitrev_size8192_mixed24_f64<>+0xF870(SB)/8, $5663
DATA bitrev_size8192_mixed24_f64<>+0xF878(SB)/8, $7711
DATA bitrev_size8192_mixed24_f64<>+0xF880(SB)/8, $159
DATA bitrev_size8192_mixed24_f64<>+0xF888(SB)/8, $2207
DATA bitrev_size8192_mixed24_f64<>+0xF890(SB)/8, $4255
DATA bitrev_size8192_mixed24_f64<>+0xF898(SB)/8, $6303
DATA bitrev_size8192_mixed24_f64<>+0xF8A0(SB)/8, $671
DATA bitrev_size8192_mixed24_f64<>+0xF8A8(SB)/8, $2719
DATA bitrev_size8192_mixed24_f64<>+0xF8B0(SB)/8, $4767
DATA bitrev_size8192_mixed24_f64<>+0xF8B8(SB)/8, $6815
DATA bitrev_size8192_mixed24_f64<>+0xF8C0(SB)/8, $1183
DATA bitrev_size8192_mixed24_f64<>+0xF8C8(SB)/8, $3231
DATA bitrev_size8192_mixed24_f64<>+0xF8D0(SB)/8, $5279
DATA bitrev_size8192_mixed24_f64<>+0xF8D8(SB)/8, $7327
DATA bitrev_size8192_mixed24_f64<>+0xF8E0(SB)/8, $1695
DATA bitrev_size8192_mixed24_f64<>+0xF8E8(SB)/8, $3743
DATA bitrev_size8192_mixed24_f64<>+0xF8F0(SB)/8, $5791
DATA bitrev_size8192_mixed24_f64<>+0xF8F8(SB)/8, $7839
DATA bitrev_size8192_mixed24_f64<>+0xF900(SB)/8, $287
DATA bitrev_size8192_mixed24_f64<>+0xF908(SB)/8, $2335
DATA bitrev_size8192_mixed24_f64<>+0xF910(SB)/8, $4383
DATA bitrev_size8192_mixed24_f64<>+0xF918(SB)/8, $6431
DATA bitrev_size8192_mixed24_f64<>+0xF920(SB)/8, $799
DATA bitrev_size8192_mixed24_f64<>+0xF928(SB)/8, $2847
DATA bitrev_size8192_mixed24_f64<>+0xF930(SB)/8, $4895
DATA bitrev_size8192_mixed24_f64<>+0xF938(SB)/8, $6943
DATA bitrev_size8192_mixed24_f64<>+0xF940(SB)/8, $1311
DATA bitrev_size8192_mixed24_f64<>+0xF948(SB)/8, $3359
DATA bitrev_size8192_mixed24_f64<>+0xF950(SB)/8, $5407
DATA bitrev_size8192_mixed24_f64<>+0xF958(SB)/8, $7455
DATA bitrev_size8192_mixed24_f64<>+0xF960(SB)/8, $1823
DATA bitrev_size8192_mixed24_f64<>+0xF968(SB)/8, $3871
DATA bitrev_size8192_mixed24_f64<>+0xF970(SB)/8, $5919
DATA bitrev_size8192_mixed24_f64<>+0xF978(SB)/8, $7967
DATA bitrev_size8192_mixed24_f64<>+0xF980(SB)/8, $415
DATA bitrev_size8192_mixed24_f64<>+0xF988(SB)/8, $2463
DATA bitrev_size8192_mixed24_f64<>+0xF990(SB)/8, $4511
DATA bitrev_size8192_mixed24_f64<>+0xF998(SB)/8, $6559
DATA bitrev_size8192_mixed24_f64<>+0xF9A0(SB)/8, $927
DATA bitrev_size8192_mixed24_f64<>+0xF9A8(SB)/8, $2975
DATA bitrev_size8192_mixed24_f64<>+0xF9B0(SB)/8, $5023
DATA bitrev_size8192_mixed24_f64<>+0xF9B8(SB)/8, $7071
DATA bitrev_size8192_mixed24_f64<>+0xF9C0(SB)/8, $1439
DATA bitrev_size8192_mixed24_f64<>+0xF9C8(SB)/8, $3487
DATA bitrev_size8192_mixed24_f64<>+0xF9D0(SB)/8, $5535
DATA bitrev_size8192_mixed24_f64<>+0xF9D8(SB)/8, $7583
DATA bitrev_size8192_mixed24_f64<>+0xF9E0(SB)/8, $1951
DATA bitrev_size8192_mixed24_f64<>+0xF9E8(SB)/8, $3999
DATA bitrev_size8192_mixed24_f64<>+0xF9F0(SB)/8, $6047
DATA bitrev_size8192_mixed24_f64<>+0xF9F8(SB)/8, $8095
DATA bitrev_size8192_mixed24_f64<>+0xFA00(SB)/8, $63
DATA bitrev_size8192_mixed24_f64<>+0xFA08(SB)/8, $2111
DATA bitrev_size8192_mixed24_f64<>+0xFA10(SB)/8, $4159
DATA bitrev_size8192_mixed24_f64<>+0xFA18(SB)/8, $6207
DATA bitrev_size8192_mixed24_f64<>+0xFA20(SB)/8, $575
DATA bitrev_size8192_mixed24_f64<>+0xFA28(SB)/8, $2623
DATA bitrev_size8192_mixed24_f64<>+0xFA30(SB)/8, $4671
DATA bitrev_size8192_mixed24_f64<>+0xFA38(SB)/8, $6719
DATA bitrev_size8192_mixed24_f64<>+0xFA40(SB)/8, $1087
DATA bitrev_size8192_mixed24_f64<>+0xFA48(SB)/8, $3135
DATA bitrev_size8192_mixed24_f64<>+0xFA50(SB)/8, $5183
DATA bitrev_size8192_mixed24_f64<>+0xFA58(SB)/8, $7231
DATA bitrev_size8192_mixed24_f64<>+0xFA60(SB)/8, $1599
DATA bitrev_size8192_mixed24_f64<>+0xFA68(SB)/8, $3647
DATA bitrev_size8192_mixed24_f64<>+0xFA70(SB)/8, $5695
DATA bitrev_size8192_mixed24_f64<>+0xFA78(SB)/8, $7743
DATA bitrev_size8192_mixed24_f64<>+0xFA80(SB)/8, $191
DATA bitrev_size8192_mixed24_f64<>+0xFA88(SB)/8, $2239
DATA bitrev_size8192_mixed24_f64<>+0xFA90(SB)/8, $4287
DATA bitrev_size8192_mixed24_f64<>+0xFA98(SB)/8, $6335
DATA bitrev_size8192_mixed24_f64<>+0xFAA0(SB)/8, $703
DATA bitrev_size8192_mixed24_f64<>+0xFAA8(SB)/8, $2751
DATA bitrev_size8192_mixed24_f64<>+0xFAB0(SB)/8, $4799
DATA bitrev_size8192_mixed24_f64<>+0xFAB8(SB)/8, $6847
DATA bitrev_size8192_mixed24_f64<>+0xFAC0(SB)/8, $1215
DATA bitrev_size8192_mixed24_f64<>+0xFAC8(SB)/8, $3263
DATA bitrev_size8192_mixed24_f64<>+0xFAD0(SB)/8, $5311
DATA bitrev_size8192_mixed24_f64<>+0xFAD8(SB)/8, $7359
DATA bitrev_size8192_mixed24_f64<>+0xFAE0(SB)/8, $1727
DATA bitrev_size8192_mixed24_f64<>+0xFAE8(SB)/8, $3775
DATA bitrev_size8192_mixed24_f64<>+0xFAF0(SB)/8, $5823
DATA bitrev_size8192_mixed24_f64<>+0xFAF8(SB)/8, $7871
DATA bitrev_size8192_mixed24_f64<>+0xFB00(SB)/8, $319
DATA bitrev_size8192_mixed24_f64<>+0xFB08(SB)/8, $2367
DATA bitrev_size8192_mixed24_f64<>+0xFB10(SB)/8, $4415
DATA bitrev_size8192_mixed24_f64<>+0xFB18(SB)/8, $6463
DATA bitrev_size8192_mixed24_f64<>+0xFB20(SB)/8, $831
DATA bitrev_size8192_mixed24_f64<>+0xFB28(SB)/8, $2879
DATA bitrev_size8192_mixed24_f64<>+0xFB30(SB)/8, $4927
DATA bitrev_size8192_mixed24_f64<>+0xFB38(SB)/8, $6975
DATA bitrev_size8192_mixed24_f64<>+0xFB40(SB)/8, $1343
DATA bitrev_size8192_mixed24_f64<>+0xFB48(SB)/8, $3391
DATA bitrev_size8192_mixed24_f64<>+0xFB50(SB)/8, $5439
DATA bitrev_size8192_mixed24_f64<>+0xFB58(SB)/8, $7487
DATA bitrev_size8192_mixed24_f64<>+0xFB60(SB)/8, $1855
DATA bitrev_size8192_mixed24_f64<>+0xFB68(SB)/8, $3903
DATA bitrev_size8192_mixed24_f64<>+0xFB70(SB)/8, $5951
DATA bitrev_size8192_mixed24_f64<>+0xFB78(SB)/8, $7999
DATA bitrev_size8192_mixed24_f64<>+0xFB80(SB)/8, $447
DATA bitrev_size8192_mixed24_f64<>+0xFB88(SB)/8, $2495
DATA bitrev_size8192_mixed24_f64<>+0xFB90(SB)/8, $4543
DATA bitrev_size8192_mixed24_f64<>+0xFB98(SB)/8, $6591
DATA bitrev_size8192_mixed24_f64<>+0xFBA0(SB)/8, $959
DATA bitrev_size8192_mixed24_f64<>+0xFBA8(SB)/8, $3007
DATA bitrev_size8192_mixed24_f64<>+0xFBB0(SB)/8, $5055
DATA bitrev_size8192_mixed24_f64<>+0xFBB8(SB)/8, $7103
DATA bitrev_size8192_mixed24_f64<>+0xFBC0(SB)/8, $1471
DATA bitrev_size8192_mixed24_f64<>+0xFBC8(SB)/8, $3519
DATA bitrev_size8192_mixed24_f64<>+0xFBD0(SB)/8, $5567
DATA bitrev_size8192_mixed24_f64<>+0xFBD8(SB)/8, $7615
DATA bitrev_size8192_mixed24_f64<>+0xFBE0(SB)/8, $1983
DATA bitrev_size8192_mixed24_f64<>+0xFBE8(SB)/8, $4031
DATA bitrev_size8192_mixed24_f64<>+0xFBF0(SB)/8, $6079
DATA bitrev_size8192_mixed24_f64<>+0xFBF8(SB)/8, $8127
DATA bitrev_size8192_mixed24_f64<>+0xFC00(SB)/8, $95
DATA bitrev_size8192_mixed24_f64<>+0xFC08(SB)/8, $2143
DATA bitrev_size8192_mixed24_f64<>+0xFC10(SB)/8, $4191
DATA bitrev_size8192_mixed24_f64<>+0xFC18(SB)/8, $6239
DATA bitrev_size8192_mixed24_f64<>+0xFC20(SB)/8, $607
DATA bitrev_size8192_mixed24_f64<>+0xFC28(SB)/8, $2655
DATA bitrev_size8192_mixed24_f64<>+0xFC30(SB)/8, $4703
DATA bitrev_size8192_mixed24_f64<>+0xFC38(SB)/8, $6751
DATA bitrev_size8192_mixed24_f64<>+0xFC40(SB)/8, $1119
DATA bitrev_size8192_mixed24_f64<>+0xFC48(SB)/8, $3167
DATA bitrev_size8192_mixed24_f64<>+0xFC50(SB)/8, $5215
DATA bitrev_size8192_mixed24_f64<>+0xFC58(SB)/8, $7263
DATA bitrev_size8192_mixed24_f64<>+0xFC60(SB)/8, $1631
DATA bitrev_size8192_mixed24_f64<>+0xFC68(SB)/8, $3679
DATA bitrev_size8192_mixed24_f64<>+0xFC70(SB)/8, $5727
DATA bitrev_size8192_mixed24_f64<>+0xFC78(SB)/8, $7775
DATA bitrev_size8192_mixed24_f64<>+0xFC80(SB)/8, $223
DATA bitrev_size8192_mixed24_f64<>+0xFC88(SB)/8, $2271
DATA bitrev_size8192_mixed24_f64<>+0xFC90(SB)/8, $4319
DATA bitrev_size8192_mixed24_f64<>+0xFC98(SB)/8, $6367
DATA bitrev_size8192_mixed24_f64<>+0xFCA0(SB)/8, $735
DATA bitrev_size8192_mixed24_f64<>+0xFCA8(SB)/8, $2783
DATA bitrev_size8192_mixed24_f64<>+0xFCB0(SB)/8, $4831
DATA bitrev_size8192_mixed24_f64<>+0xFCB8(SB)/8, $6879
DATA bitrev_size8192_mixed24_f64<>+0xFCC0(SB)/8, $1247
DATA bitrev_size8192_mixed24_f64<>+0xFCC8(SB)/8, $3295
DATA bitrev_size8192_mixed24_f64<>+0xFCD0(SB)/8, $5343
DATA bitrev_size8192_mixed24_f64<>+0xFCD8(SB)/8, $7391
DATA bitrev_size8192_mixed24_f64<>+0xFCE0(SB)/8, $1759
DATA bitrev_size8192_mixed24_f64<>+0xFCE8(SB)/8, $3807
DATA bitrev_size8192_mixed24_f64<>+0xFCF0(SB)/8, $5855
DATA bitrev_size8192_mixed24_f64<>+0xFCF8(SB)/8, $7903
DATA bitrev_size8192_mixed24_f64<>+0xFD00(SB)/8, $351
DATA bitrev_size8192_mixed24_f64<>+0xFD08(SB)/8, $2399
DATA bitrev_size8192_mixed24_f64<>+0xFD10(SB)/8, $4447
DATA bitrev_size8192_mixed24_f64<>+0xFD18(SB)/8, $6495
DATA bitrev_size8192_mixed24_f64<>+0xFD20(SB)/8, $863
DATA bitrev_size8192_mixed24_f64<>+0xFD28(SB)/8, $2911
DATA bitrev_size8192_mixed24_f64<>+0xFD30(SB)/8, $4959
DATA bitrev_size8192_mixed24_f64<>+0xFD38(SB)/8, $7007
DATA bitrev_size8192_mixed24_f64<>+0xFD40(SB)/8, $1375
DATA bitrev_size8192_mixed24_f64<>+0xFD48(SB)/8, $3423
DATA bitrev_size8192_mixed24_f64<>+0xFD50(SB)/8, $5471
DATA bitrev_size8192_mixed24_f64<>+0xFD58(SB)/8, $7519
DATA bitrev_size8192_mixed24_f64<>+0xFD60(SB)/8, $1887
DATA bitrev_size8192_mixed24_f64<>+0xFD68(SB)/8, $3935
DATA bitrev_size8192_mixed24_f64<>+0xFD70(SB)/8, $5983
DATA bitrev_size8192_mixed24_f64<>+0xFD78(SB)/8, $8031
DATA bitrev_size8192_mixed24_f64<>+0xFD80(SB)/8, $479
DATA bitrev_size8192_mixed24_f64<>+0xFD88(SB)/8, $2527
DATA bitrev_size8192_mixed24_f64<>+0xFD90(SB)/8, $4575
DATA bitrev_size8192_mixed24_f64<>+0xFD98(SB)/8, $6623
DATA bitrev_size8192_mixed24_f64<>+0xFDA0(SB)/8, $991
DATA bitrev_size8192_mixed24_f64<>+0xFDA8(SB)/8, $3039
DATA bitrev_size8192_mixed24_f64<>+0xFDB0(SB)/8, $5087
DATA bitrev_size8192_mixed24_f64<>+0xFDB8(SB)/8, $7135
DATA bitrev_size8192_mixed24_f64<>+0xFDC0(SB)/8, $1503
DATA bitrev_size8192_mixed24_f64<>+0xFDC8(SB)/8, $3551
DATA bitrev_size8192_mixed24_f64<>+0xFDD0(SB)/8, $5599
DATA bitrev_size8192_mixed24_f64<>+0xFDD8(SB)/8, $7647
DATA bitrev_size8192_mixed24_f64<>+0xFDE0(SB)/8, $2015
DATA bitrev_size8192_mixed24_f64<>+0xFDE8(SB)/8, $4063
DATA bitrev_size8192_mixed24_f64<>+0xFDF0(SB)/8, $6111
DATA bitrev_size8192_mixed24_f64<>+0xFDF8(SB)/8, $8159
DATA bitrev_size8192_mixed24_f64<>+0xFE00(SB)/8, $127
DATA bitrev_size8192_mixed24_f64<>+0xFE08(SB)/8, $2175
DATA bitrev_size8192_mixed24_f64<>+0xFE10(SB)/8, $4223
DATA bitrev_size8192_mixed24_f64<>+0xFE18(SB)/8, $6271
DATA bitrev_size8192_mixed24_f64<>+0xFE20(SB)/8, $639
DATA bitrev_size8192_mixed24_f64<>+0xFE28(SB)/8, $2687
DATA bitrev_size8192_mixed24_f64<>+0xFE30(SB)/8, $4735
DATA bitrev_size8192_mixed24_f64<>+0xFE38(SB)/8, $6783
DATA bitrev_size8192_mixed24_f64<>+0xFE40(SB)/8, $1151
DATA bitrev_size8192_mixed24_f64<>+0xFE48(SB)/8, $3199
DATA bitrev_size8192_mixed24_f64<>+0xFE50(SB)/8, $5247
DATA bitrev_size8192_mixed24_f64<>+0xFE58(SB)/8, $7295
DATA bitrev_size8192_mixed24_f64<>+0xFE60(SB)/8, $1663
DATA bitrev_size8192_mixed24_f64<>+0xFE68(SB)/8, $3711
DATA bitrev_size8192_mixed24_f64<>+0xFE70(SB)/8, $5759
DATA bitrev_size8192_mixed24_f64<>+0xFE78(SB)/8, $7807
DATA bitrev_size8192_mixed24_f64<>+0xFE80(SB)/8, $255
DATA bitrev_size8192_mixed24_f64<>+0xFE88(SB)/8, $2303
DATA bitrev_size8192_mixed24_f64<>+0xFE90(SB)/8, $4351
DATA bitrev_size8192_mixed24_f64<>+0xFE98(SB)/8, $6399
DATA bitrev_size8192_mixed24_f64<>+0xFEA0(SB)/8, $767
DATA bitrev_size8192_mixed24_f64<>+0xFEA8(SB)/8, $2815
DATA bitrev_size8192_mixed24_f64<>+0xFEB0(SB)/8, $4863
DATA bitrev_size8192_mixed24_f64<>+0xFEB8(SB)/8, $6911
DATA bitrev_size8192_mixed24_f64<>+0xFEC0(SB)/8, $1279
DATA bitrev_size8192_mixed24_f64<>+0xFEC8(SB)/8, $3327
DATA bitrev_size8192_mixed24_f64<>+0xFED0(SB)/8, $5375
DATA bitrev_size8192_mixed24_f64<>+0xFED8(SB)/8, $7423
DATA bitrev_size8192_mixed24_f64<>+0xFEE0(SB)/8, $1791
DATA bitrev_size8192_mixed24_f64<>+0xFEE8(SB)/8, $3839
DATA bitrev_size8192_mixed24_f64<>+0xFEF0(SB)/8, $5887
DATA bitrev_size8192_mixed24_f64<>+0xFEF8(SB)/8, $7935
DATA bitrev_size8192_mixed24_f64<>+0xFF00(SB)/8, $383
DATA bitrev_size8192_mixed24_f64<>+0xFF08(SB)/8, $2431
DATA bitrev_size8192_mixed24_f64<>+0xFF10(SB)/8, $4479
DATA bitrev_size8192_mixed24_f64<>+0xFF18(SB)/8, $6527
DATA bitrev_size8192_mixed24_f64<>+0xFF20(SB)/8, $895
DATA bitrev_size8192_mixed24_f64<>+0xFF28(SB)/8, $2943
DATA bitrev_size8192_mixed24_f64<>+0xFF30(SB)/8, $4991
DATA bitrev_size8192_mixed24_f64<>+0xFF38(SB)/8, $7039
DATA bitrev_size8192_mixed24_f64<>+0xFF40(SB)/8, $1407
DATA bitrev_size8192_mixed24_f64<>+0xFF48(SB)/8, $3455
DATA bitrev_size8192_mixed24_f64<>+0xFF50(SB)/8, $5503
DATA bitrev_size8192_mixed24_f64<>+0xFF58(SB)/8, $7551
DATA bitrev_size8192_mixed24_f64<>+0xFF60(SB)/8, $1919
DATA bitrev_size8192_mixed24_f64<>+0xFF68(SB)/8, $3967
DATA bitrev_size8192_mixed24_f64<>+0xFF70(SB)/8, $6015
DATA bitrev_size8192_mixed24_f64<>+0xFF78(SB)/8, $8063
DATA bitrev_size8192_mixed24_f64<>+0xFF80(SB)/8, $511
DATA bitrev_size8192_mixed24_f64<>+0xFF88(SB)/8, $2559
DATA bitrev_size8192_mixed24_f64<>+0xFF90(SB)/8, $4607
DATA bitrev_size8192_mixed24_f64<>+0xFF98(SB)/8, $6655
DATA bitrev_size8192_mixed24_f64<>+0xFFA0(SB)/8, $1023
DATA bitrev_size8192_mixed24_f64<>+0xFFA8(SB)/8, $3071
DATA bitrev_size8192_mixed24_f64<>+0xFFB0(SB)/8, $5119
DATA bitrev_size8192_mixed24_f64<>+0xFFB8(SB)/8, $7167
DATA bitrev_size8192_mixed24_f64<>+0xFFC0(SB)/8, $1535
DATA bitrev_size8192_mixed24_f64<>+0xFFC8(SB)/8, $3583
DATA bitrev_size8192_mixed24_f64<>+0xFFD0(SB)/8, $5631
DATA bitrev_size8192_mixed24_f64<>+0xFFD8(SB)/8, $7679
DATA bitrev_size8192_mixed24_f64<>+0xFFE0(SB)/8, $2047
DATA bitrev_size8192_mixed24_f64<>+0xFFE8(SB)/8, $4095
DATA bitrev_size8192_mixed24_f64<>+0xFFF0(SB)/8, $6143
DATA bitrev_size8192_mixed24_f64<>+0xFFF8(SB)/8, $8191
GLOBL bitrev_size8192_mixed24_f64<>(SB), RODATA, $65536

// =========================================================================
// Inverse scale constant: 1/8192
// =========================================================================
DATA ·neonInv8192F64+0(SB)/8, $0x3F20000000000000 // 1/8192 = 0.0001220703125
GLOBL ·neonInv8192F64(SB), RODATA, $8
