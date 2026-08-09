//go:build arm64 && !purego

// Size-generic NEON radix-8 DIT probe for complex128. One interleaved
// complex128 value fills each vector; the 32-register AArch64 file still
// leaves enough scratch for all eight streams and the radix-8 butterfly.

#include "textflag.h"
#include "neon_fp.h"

GLOBL ·r8dNegOdd<>(SB), RODATA|NOPTR, $16
DATA ·r8dNegOdd<>+0(SB)/8, $0x0000000000000000
DATA ·r8dNegOdd<>+8(SB)/8, $0x8000000000000000

GLOBL ·r8dNegEven<>(SB), RODATA|NOPTR, $16
DATA ·r8dNegEven<>+0(SB)/8, $0x8000000000000000
DATA ·r8dNegEven<>+8(SB)/8, $0x0000000000000000

GLOBL ·r8dRoot2<>(SB), RODATA|NOPTR, $8
DATA ·r8dRoot2<>+0(SB)/8, $0x3fe6a09e667f3bcd

// x *= w for one interleaved complex128 value. V27 is the sign mask that
// turns [imag*wi, real*wi] into [-imag*wi, real*wi].
#define R8D_CMUL(x, xn, w) \
	VEXT $8, x.B16, x.B16, V16.B16 \
	VZIP1 w.D2, w.D2, V17.D2       \
	VZIP2 w.D2, w.D2, V18.D2       \
	VMULF_D2(xn, 17, 19)            \
	VMULF_D2(16, 18, 20)            \
	VEOR V27.B16, V20.B16, V20.B16 \
	VADDF_D2(19, 20, 19)            \
	VMOVR(19, xn)

// Radix-8 butterfly over V0..V7. V30 is the first +/-i rotation mask,
// V31 the opposite mask and V29 is sqrt(2)/2. Results finish in V0..V7.
#define R8D_BUTTERFLY \
	VADDF_D2(0, 4, 8)       \
	VSUBF_D2(0, 4, 9)       \
	VADDF_D2(2, 6, 10)      \
	VSUBF_D2(2, 6, 11)      \
	VADDF_D2(1, 5, 0)       \
	VSUBF_D2(1, 5, 2)       \
	VADDF_D2(3, 7, 4)       \
	VSUBF_D2(3, 7, 6)       \
	VADDF_D2(8, 10, 1)      \
	VSUBF_D2(8, 10, 3)      \
	VEXT $8, V11.B16, V11.B16, V17.B16 \
	VEOR V30.B16, V17.B16, V7.B16 \
	VEOR V31.B16, V17.B16, V5.B16 \
	VADDF_D2(9, 7, 7)       \
	VADDF_D2(9, 5, 5)       \
	VADDF_D2(0, 4, 8)       \
	VSUBF_D2(0, 4, 10)      \
	VEXT $8, V6.B16, V6.B16, V17.B16 \
	VEOR V30.B16, V17.B16, V11.B16 \
	VEOR V31.B16, V17.B16, V9.B16  \
	VADDF_D2(2, 11, 11)     \
	VADDF_D2(2, 9, 9)       \
	VEXT $8, V11.B16, V11.B16, V0.B16 \
	VEOR V30.B16, V0.B16, V0.B16 \
	VADDF_D2(11, 0, 0)      \
	VMULF_D2(0, 29, 0)      \
	VEXT $8, V10.B16, V10.B16, V2.B16 \
	VEOR V30.B16, V2.B16, V2.B16 \
	VEXT $8, V9.B16, V9.B16, V4.B16 \
	VEOR V30.B16, V4.B16, V4.B16 \
	VSUBF_D2(4, 9, 4)       \
	VMULF_D2(4, 29, 4)      \
	VADDF_D2(1, 8, 6)       \
	VSUBF_D2(1, 8, 1)       \
	VADDF_D2(7, 0, 9)       \
	VSUBF_D2(7, 0, 7)       \
	VADDF_D2(3, 2, 10)      \
	VSUBF_D2(3, 2, 3)       \
	VADDF_D2(5, 4, 11)      \
	VSUBF_D2(5, 4, 5)       \
	VMOVR(5, 12)            \
	VMOVR(6, 0)             \
	VMOVR(1, 4)             \
	VMOVR(9, 1)             \
	VMOVR(10, 2)            \
	VMOVR(3, 6)             \
	VMOVR(11, 3)            \
	VMOVR(7, 5)             \
	VMOVR(12, 7)

// func Radix8Complex128Asm(dst, src, twiddle, scratch []complex128,
// idx []int32, limit int, inverse bool, scale float64) bool
TEXT ·Radix8Complex128Asm(SB), NOSPLIT, $0-145
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD scratch+72(FP), R11
	MOVD idx+96(FP), R12
	MOVD src_len+32(FP), R13

	CMP  $64, R13
	BLT  r8d_false
	MOVD dst_len+8(FP), R0
	CMP  R13, R0
	BLT  r8d_false
	MOVD scratch_len+80(FP), R0
	CMP  R13, R0
	BLT  r8d_false
	MOVD twiddle_len+56(FP), R0
	ADD  $8, R13, R1
	CMP  R1, R0
	BLT  r8d_false
	MOVD idx_len+104(FP), R0
	LSR  $3, R13, R1
	CMP  R1, R0
	BLT  r8d_false

	MOVD $·r8dNegEven<>(SB), R0
	VLD1 (R0), [V27.B16]
	MOVBU inverse+128(FP), R22
	CBNZ  R22, r8d_inverse_masks
	MOVD  $·r8dNegOdd<>(SB), R0
	VLD1  (R0), [V30.B16]
	MOVD  $·r8dNegEven<>(SB), R0
	VLD1  (R0), [V31.B16]
	B     r8d_masks_done

r8d_inverse_masks:
	MOVD $·r8dNegEven<>(SB), R0
	VLD1 (R0), [V30.B16]
	MOVD $·r8dNegOdd<>(SB), R0
	VLD1 (R0), [V31.B16]

r8d_masks_done:
	MOVD  $·r8dRoot2<>(SB), R0
	VLD1R (R0), [V29.D2]
	MOVD  scale+136(FP), R0
	VMOV  R0, V28.D[0]
	VMOV  R0, V28.D[1]

	CMP R8, R9
	BNE r8d_stage1_setup
	MOVD R11, R8

r8d_stage1_setup:
	MOVD R13, R17
	LSR  $3, R17
	MOVD R17, R14
	LSL  $4, R14
	MOVD R8, R16
	MOVD $0, R15

r8d_stage1_loop:
	CMP   R17, R15
	BGE   r8d_stages
	MOVWU (R12), R0
	ADD   $4, R12, R12
	LSL   $4, R0, R1
	MOVD  R9, R19

	ADD R1, R19, R20
	VLD1 (R20), [V0.D2]
	ADD R14, R19, R19
	ADD R1, R19, R20
	VLD1 (R20), [V1.D2]
	ADD R14, R19, R19
	ADD R1, R19, R20
	VLD1 (R20), [V2.D2]
	ADD R14, R19, R19
	ADD R1, R19, R20
	VLD1 (R20), [V3.D2]
	ADD R14, R19, R19
	ADD R1, R19, R20
	VLD1 (R20), [V4.D2]
	ADD R14, R19, R19
	ADD R1, R19, R20
	VLD1 (R20), [V5.D2]
	ADD R14, R19, R19
	ADD R1, R19, R20
	VLD1 (R20), [V6.D2]
	ADD R14, R19, R19
	ADD R1, R19, R20
	VLD1 (R20), [V7.D2]

	R8D_BUTTERFLY
	CBZ R22, r8d_stage1_unscaled
	VMULF_D2(0, 28, 0)
	VMULF_D2(1, 28, 1)
	VMULF_D2(2, 28, 2)
	VMULF_D2(3, 28, 3)
	VMULF_D2(4, 28, 4)
	VMULF_D2(5, 28, 5)
	VMULF_D2(6, 28, 6)
	VMULF_D2(7, 28, 7)

r8d_stage1_unscaled:
	VST1.P [V0.D2], 16(R16)
	VST1.P [V1.D2], 16(R16)
	VST1.P [V2.D2], 16(R16)
	VST1.P [V3.D2], 16(R16)
	VST1.P [V4.D2], 16(R16)
	VST1.P [V5.D2], 16(R16)
	VST1.P [V6.D2], 16(R16)
	VST1.P [V7.D2], 16(R16)
	ADD $1, R15, R15
	B   r8d_stage1_loop

r8d_stages:
	MOVD twiddle+48(FP), R10
	MOVD R13, R11
	LSL  $4, R11
	ADD  R8, R11, R11
	MOVD $128, R6
	MOVD limit+120(FP), R7
	LSL  $1, R7
	CMP  R7, R6
	BGT  r8d_tail

r8d_stage_setup:
	MOVD R6, R12
	LSL  $3, R12
	MOVD R8, R9

r8d_group_loop:
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
	LSR  $4, R14

r8d_inner_loop:
	VLD1 (R0), [V0.D2]
	VLD1 (R19), [V1.D2]
	VLD1 (R4), [V22.D2]
	R8D_CMUL(V1, 1, V22)
	VLD1 (R1), [V2.D2]
	VLD1 (R24), [V22.D2]
	R8D_CMUL(V2, 2, V22)
	VLD1 (R20), [V3.D2]
	VLD1 (R25), [V22.D2]
	R8D_CMUL(V3, 3, V22)
	VLD1 (R2), [V4.D2]
	VLD1 (R5), [V22.D2]
	R8D_CMUL(V4, 4, V22)
	VLD1 (R21), [V5.D2]
	VLD1 (R26), [V22.D2]
	R8D_CMUL(V5, 5, V22)
	VLD1 (R3), [V6.D2]
	VLD1 (R27), [V22.D2]
	R8D_CMUL(V6, 6, V22)
	VLD1 (R23), [V7.D2]
	VLD1 (R15), [V22.D2]
	R8D_CMUL(V7, 7, V22)

	R8D_BUTTERFLY
	VST1 [V0.D2], (R0)
	VST1 [V1.D2], (R19)
	VST1 [V2.D2], (R1)
	VST1 [V3.D2], (R20)
	VST1 [V4.D2], (R2)
	VST1 [V5.D2], (R21)
	VST1 [V6.D2], (R3)
	VST1 [V7.D2], (R23)

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
	SUB $1, R14, R14
	CBNZ R14, r8d_inner_loop

	ADD R12, R9, R9
	CMP R11, R9
	BLT r8d_group_loop
	ADD R6<<3, R10, R10
	SUB R6, R10, R10
	LSL $3, R6, R6
	CMP R7, R6
	BLE r8d_stage_setup

r8d_tail:
	MOVD limit+120(FP), R6
	CMP  R13, R6
	BGE  r8d_copy_out
	LSL  $1, R6, R0
	CMP  R13, R0
	BNE  r8d_tail4

	MOVD R13, R14
	LSR  $1, R14
	MOVD R8, R0
	MOVD R14, R6
	LSL  $4, R6
	ADD  R6, R8, R1
	MOVD R6, R14

r8d_tail2_loop:
	VLD1 (R0), [V0.D2]
	VLD1 (R1), [V1.D2]
	VLD1 (R10), [V22.D2]
	R8D_CMUL(V1, 1, V22)
	VADDF_D2(0, 1, 2)
	VSUBF_D2(0, 1, 3)
	VST1.P [V2.D2], 16(R0)
	VST1.P [V3.D2], 16(R1)
	ADD $16, R10, R10
	SUB $16, R14, R14
	CBNZ R14, r8d_tail2_loop
	B r8d_copy_out

r8d_tail4:
	MOVD R13, R14
	LSR  $2, R14
	MOVD R14, R6
	LSL  $4, R6
	MOVD R8, R0
	ADD  R6, R0, R19
	ADD  R6, R19, R1
	ADD  R6, R1, R20
	MOVD R10, R4
	ADD  R6, R4, R24
	ADD  R6, R24, R25
	MOVD R6, R14

r8d_tail4_loop:
	VLD1 (R0), [V0.D2]
	VLD1 (R19), [V1.D2]
	VLD1 (R4), [V22.D2]
	R8D_CMUL(V1, 1, V22)
	VLD1 (R1), [V2.D2]
	VLD1 (R24), [V22.D2]
	R8D_CMUL(V2, 2, V22)
	VLD1 (R20), [V3.D2]
	VLD1 (R25), [V22.D2]
	R8D_CMUL(V3, 3, V22)
	VADDF_D2(0, 2, 4)
	VSUBF_D2(0, 2, 5)
	VADDF_D2(1, 3, 6)
	VSUBF_D2(1, 3, 7)
	VEXT $8, V7.B16, V7.B16, V16.B16
	VEOR V30.B16, V16.B16, V17.B16
	VEOR V31.B16, V16.B16, V16.B16
	VADDF_D2(4, 6, 0)
	VADDF_D2(5, 17, 1)
	VSUBF_D2(4, 6, 2)
	VADDF_D2(5, 16, 3)
	VST1 [V0.D2], (R0)
	VST1 [V1.D2], (R19)
	VST1 [V2.D2], (R1)
	VST1 [V3.D2], (R20)
	ADD $16, R0, R0
	ADD $16, R19, R19
	ADD $16, R1, R1
	ADD $16, R20, R20
	ADD $16, R4, R4
	ADD $16, R24, R24
	ADD $16, R25, R25
	SUB $16, R14, R14
	CBNZ R14, r8d_tail4_loop

r8d_copy_out:
	MOVD dst+0(FP), R9
	CMP  R9, R8
	BEQ  r8d_true
	MOVD R13, R14
	LSL  $4, R14

r8d_copy_loop:
	VLD1.P 16(R8), [V0.B16]
	VST1.P [V0.B16], 16(R9)
	SUB $16, R14, R14
	CBNZ R14, r8d_copy_loop

r8d_true:
	MOVD $1, R0
	MOVB R0, ret+144(FP)
	RET

r8d_false:
	MOVB ZR, ret+144(FP)
	RET
