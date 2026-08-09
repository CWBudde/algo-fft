//go:build arm64 && !purego

// Size-generic NEON radix-8 DIT kernel for complex64. The Go caller supplies
// the same packed twiddle planes and radix-8 digit-reversal group table used
// by the generic and AVX2 ladders. Two butterflies are carried in each vector.
// The minimum size is 32: one radix-8 stage followed by a radix-4 tail.

#include "textflag.h"
#include "neon_fp.h"

GLOBL ·r8nNegOdd<>(SB), RODATA|NOPTR, $16
DATA ·r8nNegOdd<>+0(SB)/8, $0x8000000000000000
DATA ·r8nNegOdd<>+8(SB)/8, $0x8000000000000000

GLOBL ·r8nNegEven<>(SB), RODATA|NOPTR, $16
DATA ·r8nNegEven<>+0(SB)/8, $0x0000000080000000
DATA ·r8nNegEven<>+8(SB)/8, $0x0000000080000000

GLOBL ·r8nRoot2<>(SB), RODATA|NOPTR, $4
DATA ·r8nRoot2<>+0(SB)/4, $0x3f3504f3

// Load one stream for two stage-1 groups. R0/R1 are the two digit-reversed
// base indices, R19 is the current stream base and R14 is n/8 complex values
// expressed in bytes.
#define R8_LOAD_STAGE1_PAIR(v) \
	MOVD (R19)(R0<<3), R20 \
	MOVD (R19)(R1<<3), R21 \
	VMOV R20, v.D[0]       \
	VMOV R21, v.D[1]       \
	ADD  R14, R19, R19

// x *= w for two interleaved complex64 values. V16..V21 are scratch.
#define R8_CMUL(x, w) \
	VUZP1 x.S4, x.S4, V16.S4 \
	VUZP2 x.S4, x.S4, V17.S4 \
	VUZP1 w.S4, w.S4, V18.S4 \
	VUZP2 w.S4, w.S4, V19.S4 \
	VMULF_S4(16, 18, 20)      \
	VFMSF_S4(17, 19, 20)      \
	VMULF_S4(16, 19, 21)      \
	VFMAF_S4(17, 18, 21)      \
	VZIP1 V21.S4, V20.S4, x.S4

// Radix-8 butterfly over V0..V7. Each register contains two interleaved
// complex64 values. V30 is the first +/-i rotation mask, V31 the opposite
// mask, and V29 is sqrt(2)/2. Results are normalized into V0..V7.
#define R8_BUTTERFLY \
	VADDF_S4(0, 4, 8)       \
	VSUBF_S4(0, 4, 9)       \
	VADDF_S4(2, 6, 10)      \
	VSUBF_S4(2, 6, 11)      \
	VADDF_S4(1, 5, 0)       \
	VSUBF_S4(1, 5, 2)       \
	VADDF_S4(3, 7, 4)       \
	VSUBF_S4(3, 7, 6)       \
	VADDF_S4(8, 10, 1)      \
	VSUBF_S4(8, 10, 3)      \
	VREV64 V11.S4, V17.S4   \
	VEOR V30.B16, V17.B16, V7.B16 \
	VEOR V31.B16, V17.B16, V5.B16 \
	VADDF_S4(9, 7, 7)       \
	VADDF_S4(9, 5, 5)       \
	VADDF_S4(0, 4, 8)       \
	VSUBF_S4(0, 4, 10)      \
	VREV64 V6.S4, V17.S4    \
	VEOR V30.B16, V17.B16, V11.B16 \
	VEOR V31.B16, V17.B16, V9.B16  \
	VADDF_S4(2, 11, 11)     \
	VADDF_S4(2, 9, 9)       \
	VREV64 V11.S4, V0.S4    \
	VEOR V30.B16, V0.B16, V0.B16 \
	VADDF_S4(11, 0, 0)      \
	VMULF_S4(0, 29, 0)      \
	VREV64 V10.S4, V2.S4    \
	VEOR V30.B16, V2.B16, V2.B16 \
	VREV64 V9.S4, V4.S4     \
	VEOR V30.B16, V4.B16, V4.B16 \
	VSUBF_S4(4, 9, 4)       \
	VMULF_S4(4, 29, 4)      \
	VADDF_S4(1, 8, 6)       \
	VSUBF_S4(1, 8, 1)       \
	VADDF_S4(7, 0, 9)       \
	VSUBF_S4(7, 0, 7)       \
	VADDF_S4(3, 2, 10)      \
	VSUBF_S4(3, 2, 3)       \
	VADDF_S4(5, 4, 11)      \
	VSUBF_S4(5, 4, 5)       \
	VMOVR(5, 12)            \
	VMOVR(6, 0)             \
	VMOVR(1, 4)             \
	VMOVR(9, 1)             \
	VMOVR(10, 2)            \
	VMOVR(3, 6)             \
	VMOVR(11, 3)            \
	VMOVR(7, 5)             \
	VMOVR(12, 7)

// func Radix8Complex64Asm(dst, src, twiddle, scratch []complex64, idx []int32,
// limit int, inverse bool, scale float32) bool
TEXT ·Radix8Complex64Asm(SB), NOSPLIT, $0-137
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD scratch+72(FP), R11
	MOVD idx+96(FP), R12
	MOVD src_len+32(FP), R13

	CMP  $32, R13
	BLT  r8n_false
	MOVD dst_len+8(FP), R0
	CMP  R13, R0
	BLT  r8n_false
	MOVD scratch_len+80(FP), R0
	CMP  R13, R0
	BLT  r8n_false
	MOVD twiddle_len+56(FP), R0
	ADD  $8, R13, R1
	CMP  R1, R0
	BLT  r8n_false
	MOVD idx_len+104(FP), R0
	LSR  $3, R13, R1
	CMP  R1, R0
	BLT  r8n_false

	MOVBU inverse+128(FP), R22
	CBNZ  R22, r8n_inverse_masks
	MOVD  $·r8nNegOdd<>(SB), R0
	VLD1  (R0), [V30.B16]
	MOVD  $·r8nNegEven<>(SB), R0
	VLD1  (R0), [V31.B16]
	B     r8n_masks_done

r8n_inverse_masks:
	MOVD $·r8nNegEven<>(SB), R0
	VLD1 (R0), [V30.B16]
	MOVD $·r8nNegOdd<>(SB), R0
	VLD1 (R0), [V31.B16]

r8n_masks_done:
	MOVD  $·r8nRoot2<>(SB), R0
	VLD1R (R0), [V29.S4]
	MOVWU scale+132(FP), R0
	VMOV  R0, V28.S[0]
	VMOV  R0, V28.S[1]
	VMOV  R0, V28.S[2]
	VMOV  R0, V28.S[3]

	CMP R8, R9
	BNE r8n_stage1_setup
	MOVD R11, R8

r8n_stage1_setup:
	MOVD R13, R17
	LSR  $3, R17
	MOVD R17, R14
	LSL  $3, R14
	MOVD R8, R16
	MOVD $0, R15

r8n_stage1_loop:
	CMP   R17, R15
	BGE   r8n_stages
	MOVWU (R12), R0
	MOVWU 4(R12), R1
	ADD   $8, R12, R12
	MOVD  R9, R19

	R8_LOAD_STAGE1_PAIR(V0)
	R8_LOAD_STAGE1_PAIR(V1)
	R8_LOAD_STAGE1_PAIR(V2)
	R8_LOAD_STAGE1_PAIR(V3)
	R8_LOAD_STAGE1_PAIR(V4)
	R8_LOAD_STAGE1_PAIR(V5)
	R8_LOAD_STAGE1_PAIR(V6)
	R8_LOAD_STAGE1_PAIR(V7)

	R8_BUTTERFLY
	CBZ R22, r8n_stage1_unscaled
	VMULF_S4(0, 28, 0)
	VMULF_S4(1, 28, 1)
	VMULF_S4(2, 28, 2)
	VMULF_S4(3, 28, 3)
	VMULF_S4(4, 28, 4)
	VMULF_S4(5, 28, 5)
	VMULF_S4(6, 28, 6)
	VMULF_S4(7, 28, 7)

r8n_stage1_unscaled:
	VZIP1 V1.D2, V0.D2, V16.D2
	VZIP2 V1.D2, V0.D2, V17.D2
	VZIP1 V3.D2, V2.D2, V18.D2
	VZIP2 V3.D2, V2.D2, V19.D2
	VZIP1 V5.D2, V4.D2, V20.D2
	VZIP2 V5.D2, V4.D2, V21.D2
	VZIP1 V7.D2, V6.D2, V22.D2
	VZIP2 V7.D2, V6.D2, V23.D2

	VST1.P [V16.D2], 16(R16)
	VST1.P [V18.D2], 16(R16)
	VST1.P [V20.D2], 16(R16)
	VST1.P [V22.D2], 16(R16)
	VST1.P [V17.D2], 16(R16)
	VST1.P [V19.D2], 16(R16)
	VST1.P [V21.D2], 16(R16)
	VST1.P [V23.D2], 16(R16)

	ADD $2, R15, R15
	B   r8n_stage1_loop

r8n_stages:
	MOVD twiddle+48(FP), R10
	MOVD R13, R11
	LSL  $3, R11
	ADD  R8, R11, R11
	MOVD $64, R6
	MOVD limit+120(FP), R7
	CMP  R7, R6
	BGT  r8n_tail

r8n_stage_setup:
	MOVD R6, R12
	LSL  $3, R12
	MOVD R8, R9

r8n_group_loop:
	MOVD R9, R0
	ADD  R6, R0, R19
	ADD  R6, R19, R1
	ADD  R6, R1, R20
	ADD  R6, R20, R2
	ADD  R6, R2, R21
	ADD  R6, R21, R3
	ADD  R6, R3, R23
	MOVD R10, R4
	ADD  R6, R4, R24
	ADD  R6, R24, R25
	ADD  R6, R25, R5
	ADD  R6, R5, R26
	ADD  R6, R26, R27
	ADD  R6, R27, R15
	MOVD R6, R14
	LSR  $3, R14

r8n_inner_loop:
	VLD1 (R0), [V0.S4]
	VLD1 (R19), [V1.S4]
	VLD1 (R4), [V22.S4]
	R8_CMUL(V1, V22)
	VLD1 (R1), [V2.S4]
	VLD1 (R24), [V22.S4]
	R8_CMUL(V2, V22)
	VLD1 (R20), [V3.S4]
	VLD1 (R25), [V22.S4]
	R8_CMUL(V3, V22)
	VLD1 (R2), [V4.S4]
	VLD1 (R5), [V22.S4]
	R8_CMUL(V4, V22)
	VLD1 (R21), [V5.S4]
	VLD1 (R26), [V22.S4]
	R8_CMUL(V5, V22)
	VLD1 (R3), [V6.S4]
	VLD1 (R27), [V22.S4]
	R8_CMUL(V6, V22)
	VLD1 (R23), [V7.S4]
	VLD1 (R15), [V22.S4]
	R8_CMUL(V7, V22)

	R8_BUTTERFLY
	VST1 [V0.S4], (R0)
	VST1 [V1.S4], (R19)
	VST1 [V2.S4], (R1)
	VST1 [V3.S4], (R20)
	VST1 [V4.S4], (R2)
	VST1 [V5.S4], (R21)
	VST1 [V6.S4], (R3)
	VST1 [V7.S4], (R23)

	ADD $16, R0, R0
	ADD $16, R19, R19
	ADD $16, R1, R1
	ADD $16, R20, R20
	ADD $16, R2, R2
	ADD $16, R21, R21
	ADD $16, R3, R3
	ADD $16, R23, R23
	ADD $16, R4, R4
	ADD $16, R24, R24
	ADD $16, R25, R25
	ADD $16, R5, R5
	ADD $16, R26, R26
	ADD $16, R27, R27
	ADD $16, R15, R15
	SUB $2, R14, R14
	CBNZ R14, r8n_inner_loop

	ADD R12, R9, R9
	CMP R11, R9
	BLT r8n_group_loop
	ADD R6<<3, R10, R10
	SUB R6, R10, R10
	LSL $3, R6, R6
	CMP R7, R6
	BLE r8n_stage_setup

r8n_tail:
	MOVD limit+120(FP), R6
	CMP  R13, R6
	BGE  r8n_copy_out
	LSL  $1, R6, R0
	CMP  R13, R0
	BNE  r8n_tail4

	MOVD R13, R14
	LSR  $1, R14
	MOVD R8, R0
	MOVD R14, R6
	LSL  $3, R6
	ADD  R6, R8, R1
	MOVD R6, R14

r8n_tail2_loop:
	VLD1 (R0), [V0.S4]
	VLD1 (R1), [V1.S4]
	VLD1 (R10), [V22.S4]
	R8_CMUL(V1, V22)
	VADDF_S4(0, 1, 2)
	VSUBF_S4(0, 1, 3)
	VST1.P [V2.S4], 16(R0)
	VST1.P [V3.S4], 16(R1)
	ADD $16, R10, R10
	SUB $16, R14, R14
	CBNZ R14, r8n_tail2_loop
	B r8n_copy_out

r8n_tail4:
	MOVD R13, R14
	LSR  $2, R14
	MOVD R14, R6
	LSL  $3, R6
	MOVD R8, R0
	ADD  R6, R0, R19
	ADD  R6, R19, R1
	ADD  R6, R1, R20
	MOVD R10, R4
	ADD  R6, R4, R24
	ADD  R6, R24, R25
	MOVD R6, R14

r8n_tail4_loop:
	VLD1 (R0), [V0.S4]
	VLD1 (R19), [V1.S4]
	VLD1 (R4), [V22.S4]
	R8_CMUL(V1, V22)
	VLD1 (R1), [V2.S4]
	VLD1 (R24), [V22.S4]
	R8_CMUL(V2, V22)
	VLD1 (R20), [V3.S4]
	VLD1 (R25), [V22.S4]
	R8_CMUL(V3, V22)
	VADDF_S4(0, 2, 4)
	VSUBF_S4(0, 2, 5)
	VADDF_S4(1, 3, 6)
	VSUBF_S4(1, 3, 7)
	VREV64 V7.S4, V16.S4
	VEOR V30.B16, V16.B16, V17.B16
	VEOR V31.B16, V16.B16, V16.B16
	VADDF_S4(4, 6, 0)
	VADDF_S4(5, 17, 1)
	VSUBF_S4(4, 6, 2)
	VADDF_S4(5, 16, 3)
	VST1 [V0.S4], (R0)
	VST1 [V1.S4], (R19)
	VST1 [V2.S4], (R1)
	VST1 [V3.S4], (R20)
	ADD $16, R0, R0
	ADD $16, R19, R19
	ADD $16, R1, R1
	ADD $16, R20, R20
	ADD $16, R4, R4
	ADD $16, R24, R24
	ADD $16, R25, R25
	SUB $16, R14, R14
	CBNZ R14, r8n_tail4_loop

r8n_copy_out:
	MOVD dst+0(FP), R9
	CMP  R9, R8
	BEQ  r8n_true
	MOVD R13, R14
	LSL  $3, R14

r8n_copy_loop:
	VLD1.P 16(R8), [V0.B16]
	VST1.P [V0.B16], 16(R9)
	SUB $16, R14, R14
	CBNZ R14, r8n_copy_loop

r8n_true:
	MOVD $1, R0
	MOVB R0, ret+136(FP)
	RET

r8n_false:
	MOVB ZR, ret+136(FP)
	RET
