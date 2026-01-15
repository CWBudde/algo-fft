//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2/FMA-optimized FFT Assembly for AMD64 - complex64 (float32)
// Radix-4 generic DIT path for power-of-4 sizes.
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// forwardAVX2Complex64Radix4Asm - Forward FFT for complex64 using radix-4 DIT
// ===========================================================================
TEXT ·ForwardAVX2Complex64Radix4Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    fwd_r4_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   fwd_r4_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   fwd_r4_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   fwd_r4_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  fwd_r4_check_power_of_2
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  fwd_r4_return_true

fwd_r4_check_power_of_2:
	// Verify n is power of 2: (n & (n-1)) == 0
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  fwd_r4_return_false

	// Require even log2 (power-of-4) and minimum size
	MOVQ R13, AX
	BSRQ AX, R14
	TESTQ $1, R14
	JNZ  fwd_r4_return_false

	CMPQ R13, $64
	JL   fwd_r4_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  fwd_r4_use_dst
	MOVQ R11, R8

fwd_r4_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation (base-4)
	// -----------------------------------------------------------------------
	XORQ CX, CX

fwd_r4_bitrev_loop:
	CMPQ CX, R13
	JGE  fwd_r4_stage_init

	MOVQ CX, DX
	XORQ BX, BX
	MOVQ R14, SI

fwd_r4_bitrev_inner:
	CMPQ SI, $0
	JE   fwd_r4_bitrev_store
	MOVQ DX, AX
	ANDQ $3, AX
	SHLQ $2, BX
	ORQ  AX, BX
	SHRQ $2, DX
	SUBQ $2, SI
	JMP  fwd_r4_bitrev_inner

fwd_r4_bitrev_store:
	MOVQ (R9)(BX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	JMP  fwd_r4_bitrev_loop

fwd_r4_stage_init:
	// -----------------------------------------------------------------------
	// Radix-4 stages (size = 4, 16, 64, ...)
	// -----------------------------------------------------------------------
	MOVQ $2, R12            // log2(size) starting at 2
	MOVQ $4, R14            // size

fwd_r4_stage_loop:
	CMPQ R14, R13
	JG   fwd_r4_copy_back

	MOVQ R14, R15
	SHRQ $2, R15            // quarter = size/4

	// step = n >> log2(size)
	MOVQ R13, BX
	MOVQ R12, CX
	SHRQ CL, BX

	XORQ CX, CX             // base

fwd_r4_stage_base:
	CMPQ CX, R13
	JGE  fwd_r4_stage_next

	XORQ DX, DX             // j

	// Fast path for contiguous twiddles on the final stage (step == 1).
	CMPQ BX, $1
	JNE  fwd_r4_stage_scalar
	CMPQ R15, $4
	JL   fwd_r4_stage_scalar

	MOVQ R15, R11
	SHLQ $3, R11            // quarter_bytes
	MOVQ R11, R9
	SHLQ $1, R9             // 2 * quarter_bytes

fwd_r4_step1_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   fwd_r4_stage_scalar

	// base offset
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI             // (base + j) * 8

	LEAQ (SI)(R11*1), DI    // + quarter
	LEAQ (SI)(R9*1), AX     // + 2*quarter
	LEAQ (AX)(R11*1), BP    // + 3*quarter

	VMOVUPS (R8)(SI*1), Y0
	VMOVUPS (R8)(DI*1), Y1
	VMOVUPS (R8)(AX*1), Y2
	VMOVUPS (R8)(BP*1), Y3

	// w1 = twiddle[j : j+4] (contiguous)
	MOVQ DX, AX
	SHLQ $3, AX
	VMOVUPS (R10)(AX*1), Y4

	// w2 = twiddle[2*j, 2*j+2, 2*j+4, 2*j+6]
	MOVQ DX, AX
	SHLQ $1, AX
	SHLQ $3, AX
	VMOVSD (R10)(AX*1), X5
	LEAQ 16(AX), DI
	VMOVSD (R10)(DI*1), X6
	LEAQ 16(DI), DI
	VMOVSD (R10)(DI*1), X7
	LEAQ 16(DI), DI
	VMOVSD (R10)(DI*1), X8
	VPUNPCKLQDQ X6, X5, X5
	VPUNPCKLQDQ X8, X7, X7
	VINSERTF128 $1, X7, Y5, Y5

	// w3 = twiddle[3*j, 3*j+3, 3*j+6, 3*j+9]
	LEAQ (DX)(DX*2), AX
	SHLQ $3, AX
	VMOVSD (R10)(AX*1), X9
	LEAQ 24(AX), DI
	VMOVSD (R10)(DI*1), X10
	LEAQ 24(DI), DI
	VMOVSD (R10)(DI*1), X11
	LEAQ 24(DI), DI
	VMOVSD (R10)(DI*1), X12
	VPUNPCKLQDQ X10, X9, X9
	VPUNPCKLQDQ X12, X11, X11
	VINSERTF128 $1, X11, Y6, Y6

	// Complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP Y4, Y7
	VMOVSHDUP Y4, Y8
	VSHUFPS $0xB1, Y1, Y1, Y9
	VMULPS Y8, Y9, Y9
	VFMADDSUB231PS Y7, Y1, Y9
	VMOVAPS Y9, Y1

	VMOVSLDUP Y5, Y7
	VMOVSHDUP Y5, Y8
	VSHUFPS $0xB1, Y2, Y2, Y9
	VMULPS Y8, Y9, Y9
	VFMADDSUB231PS Y7, Y2, Y9
	VMOVAPS Y9, Y2

	VMOVSLDUP Y6, Y7
	VMOVSHDUP Y6, Y8
	VSHUFPS $0xB1, Y3, Y3, Y9
	VMULPS Y8, Y9, Y9
	VFMADDSUB231PS Y7, Y3, Y9
	VMOVAPS Y9, Y3

	VADDPS Y0, Y2, Y10
	VSUBPS Y2, Y0, Y11
	VADDPS Y1, Y3, Y12
	VSUBPS Y3, Y1, Y13

	// (-i)*t3
	VPERMILPS $0xB1, Y13, Y14
	VXORPS Y15, Y15, Y15
	VSUBPS Y14, Y15, Y7
	VBLENDPS $0x02, Y7, Y14, Y14

	// i*t3
	VPERMILPS $0xB1, Y13, Y7
	VSUBPS Y7, Y15, Y8
	VBLENDPS $0x01, Y8, Y7, Y7

	VADDPS Y10, Y12, Y0
	VADDPS Y11, Y14, Y1
	VSUBPS Y12, Y10, Y2
	VADDPS Y11, Y7, Y3

	VMOVUPS Y0, (R8)(SI*1)
	VMOVUPS Y1, (R8)(DI*1)
	VMOVUPS Y2, (R8)(AX*1)
	VMOVUPS Y3, (R8)(BP*1)

	ADDQ $4, DX
	JMP  fwd_r4_step1_loop

fwd_r4_stage_scalar:
	CMPQ DX, R15
	JGE  fwd_r4_stage_base_next

	// indices
	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R15, DI
	MOVQ DI, R11
	ADDQ R15, R11
	MOVQ R11, R9
	ADDQ R15, R9

	// twiddle indices: w1 = twiddle[j*step], w2 = twiddle[2*j*step], w3 = twiddle[3*j*step]
	MOVQ DX, AX
	IMULQ BX, AX
	VMOVSD (R10)(AX*8), X8

	MOVQ AX, BP
	SHLQ $1, BP
	VMOVSD (R10)(BP*8), X9

	ADDQ AX, BP
	VMOVSD (R10)(BP*8), X10

	// load inputs
	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1
	VMOVSD (R8)(R11*8), X2
	VMOVSD (R8)(R9*8), X3

	// Complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// (-i)*t3
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8

	// i*t3
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11

	VADDPS X4, X6, X0
	VADDPS X5, X8, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X11, X3

	VMOVSD X0, (R8)(SI*8)
	VMOVSD X1, (R8)(DI*8)
	VMOVSD X2, (R8)(R11*8)
	VMOVSD X3, (R8)(R9*8)

	INCQ DX
	JMP  fwd_r4_stage_scalar

fwd_r4_stage_base_next:
	ADDQ R14, CX
	JMP  fwd_r4_stage_base

fwd_r4_stage_next:
	ADDQ $2, R12
	SHLQ $2, R14
	JMP  fwd_r4_stage_loop

fwd_r4_copy_back:
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   fwd_r4_return_true

	XORQ CX, CX

fwd_r4_copy_loop:
	CMPQ CX, R13
	JGE  fwd_r4_return_true
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  fwd_r4_copy_loop

fwd_r4_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

fwd_r4_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// inverseAVX2Complex64Radix4Asm - Inverse FFT for complex64 using radix-4 DIT
// ===========================================================================
TEXT ·InverseAVX2Complex64Radix4Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    inv_r4_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   inv_r4_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   inv_r4_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   inv_r4_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  inv_r4_check_power_of_2
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  inv_r4_return_true

inv_r4_check_power_of_2:
	// Verify n is power of 2: (n & (n-1)) == 0
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  inv_r4_return_false

	// Require even log2 (power-of-4) and minimum size
	MOVQ R13, AX
	BSRQ AX, R14
	TESTQ $1, R14
	JNZ  inv_r4_return_false

	CMPQ R13, $64
	JL   inv_r4_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  inv_r4_use_dst
	MOVQ R11, R8

inv_r4_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation (base-4)
	// -----------------------------------------------------------------------
	XORQ CX, CX

inv_r4_bitrev_loop:
	CMPQ CX, R13
	JGE  inv_r4_stage_init

	MOVQ CX, DX
	XORQ BX, BX
	MOVQ R14, SI

inv_r4_bitrev_inner:
	CMPQ SI, $0
	JE   inv_r4_bitrev_store
	MOVQ DX, AX
	ANDQ $3, AX
	SHLQ $2, BX
	ORQ  AX, BX
	SHRQ $2, DX
	SUBQ $2, SI
	JMP  inv_r4_bitrev_inner

inv_r4_bitrev_store:
	MOVQ (R9)(BX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	JMP  inv_r4_bitrev_loop

inv_r4_stage_init:
	// -----------------------------------------------------------------------
	// Radix-4 stages (size = 4, 16, 64, ...)
	// -----------------------------------------------------------------------
	MOVQ $2, R12            // log2(size) starting at 2
	MOVQ $4, R14            // size

inv_r4_stage_loop:
	CMPQ R14, R13
	JG   inv_r4_copy_back

	MOVQ R14, R15
	SHRQ $2, R15            // quarter = size/4

	// step = n >> log2(size)
	MOVQ R13, BX
	MOVQ R12, CX
	SHRQ CL, BX

	XORQ CX, CX             // base

inv_r4_stage_base:
	CMPQ CX, R13
	JGE  inv_r4_stage_next

	XORQ DX, DX             // j

	// Fast path for contiguous twiddles on the final stage (step == 1).
	CMPQ BX, $1
	JNE  inv_r4_stage_scalar
	CMPQ R15, $4
	JL   inv_r4_stage_scalar

	MOVQ R15, R11
	SHLQ $3, R11            // quarter_bytes
	MOVQ R11, R9
	SHLQ $1, R9             // 2 * quarter_bytes

inv_r4_step1_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   inv_r4_stage_scalar

	// base offset
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI             // (base + j) * 8

	LEAQ (SI)(R11*1), DI    // + quarter
	LEAQ (SI)(R9*1), AX     // + 2*quarter
	LEAQ (AX)(R11*1), BP    // + 3*quarter

	VMOVUPS (R8)(SI*1), Y0
	VMOVUPS (R8)(DI*1), Y1
	VMOVUPS (R8)(AX*1), Y2
	VMOVUPS (R8)(BP*1), Y3

	// w1 = twiddle[j : j+4] (contiguous)
	MOVQ DX, AX
	SHLQ $3, AX
	VMOVUPS (R10)(AX*1), Y4

	// w2 = twiddle[2*j, 2*j+2, 2*j+4, 2*j+6]
	MOVQ DX, AX
	SHLQ $1, AX
	SHLQ $3, AX
	VMOVSD (R10)(AX*1), X5
	LEAQ 16(AX), DI
	VMOVSD (R10)(DI*1), X6
	LEAQ 16(DI), DI
	VMOVSD (R10)(DI*1), X7
	LEAQ 16(DI), DI
	VMOVSD (R10)(DI*1), X8
	VPUNPCKLQDQ X6, X5, X5
	VPUNPCKLQDQ X8, X7, X7
	VINSERTF128 $1, X7, Y5, Y5

	// w3 = twiddle[3*j, 3*j+3, 3*j+6, 3*j+9]
	LEAQ (DX)(DX*2), AX
	SHLQ $3, AX
	VMOVSD (R10)(AX*1), X9
	LEAQ 24(AX), DI
	VMOVSD (R10)(DI*1), X10
	LEAQ 24(DI), DI
	VMOVSD (R10)(DI*1), X11
	LEAQ 24(DI), DI
	VMOVSD (R10)(DI*1), X12
	VPUNPCKLQDQ X10, X9, X9
	VPUNPCKLQDQ X12, X11, X11
	VINSERTF128 $1, X11, Y6, Y6

	// Conjugate complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP Y4, Y7
	VMOVSHDUP Y4, Y8
	VSHUFPS $0xB1, Y1, Y1, Y9
	VMULPS Y8, Y9, Y9
	VFMSUBADD231PS Y7, Y1, Y9
	VMOVAPS Y9, Y1

	VMOVSLDUP Y5, Y7
	VMOVSHDUP Y5, Y8
	VSHUFPS $0xB1, Y2, Y2, Y9
	VMULPS Y8, Y9, Y9
	VFMSUBADD231PS Y7, Y2, Y9
	VMOVAPS Y9, Y2

	VMOVSLDUP Y6, Y7
	VMOVSHDUP Y6, Y8
	VSHUFPS $0xB1, Y3, Y3, Y9
	VMULPS Y8, Y9, Y9
	VFMSUBADD231PS Y7, Y3, Y9
	VMOVAPS Y9, Y3

	VADDPS Y0, Y2, Y10
	VSUBPS Y2, Y0, Y11
	VADDPS Y1, Y3, Y12
	VSUBPS Y3, Y1, Y13

	// (-i)*t3
	VPERMILPS $0xB1, Y13, Y14
	VXORPS Y15, Y15, Y15
	VSUBPS Y14, Y15, Y7
	VBLENDPS $0x02, Y7, Y14, Y14

	// i*t3
	VPERMILPS $0xB1, Y13, Y7
	VSUBPS Y7, Y15, Y8
	VBLENDPS $0x01, Y8, Y7, Y7

	VADDPS Y10, Y12, Y0
	VADDPS Y11, Y7, Y1
	VSUBPS Y12, Y10, Y2
	VADDPS Y11, Y14, Y3

	VMOVUPS Y0, (R8)(SI*1)
	VMOVUPS Y1, (R8)(DI*1)
	VMOVUPS Y2, (R8)(AX*1)
	VMOVUPS Y3, (R8)(BP*1)

	ADDQ $4, DX
	JMP  inv_r4_step1_loop

inv_r4_stage_scalar:
	CMPQ DX, R15
	JGE  inv_r4_stage_base_next

	// indices
	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R15, DI
	MOVQ DI, R11
	ADDQ R15, R11
	MOVQ R11, R9
	ADDQ R15, R9

	// twiddle indices: w1 = twiddle[j*step], w2 = twiddle[2*j*step], w3 = twiddle[3*j*step]
	MOVQ DX, AX
	IMULQ BX, AX
	VMOVSD (R10)(AX*8), X8

	MOVQ AX, BP
	SHLQ $1, BP
	VMOVSD (R10)(BP*8), X9

	ADDQ AX, BP
	VMOVSD (R10)(BP*8), X10

	// load inputs
	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1
	VMOVSD (R8)(R11*8), X2
	VMOVSD (R8)(R9*8), X3

	// Conjugate complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// (-i)*t3
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8

	// i*t3
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11

	VADDPS X4, X6, X0
	VADDPS X5, X11, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X8, X3

	VMOVSD X0, (R8)(SI*8)
	VMOVSD X1, (R8)(DI*8)
	VMOVSD X2, (R8)(R11*8)
	VMOVSD X3, (R8)(R9*8)

	INCQ DX
	JMP  inv_r4_stage_scalar

inv_r4_stage_base_next:
	ADDQ R14, CX
	JMP  inv_r4_stage_base

inv_r4_stage_next:
	ADDQ $2, R12
	SHLQ $2, R14
	JMP  inv_r4_stage_loop

inv_r4_copy_back:
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   inv_r4_scale

	XORQ CX, CX

inv_r4_copy_loop:
	CMPQ CX, R13
	JGE  inv_r4_scale
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  inv_r4_copy_loop

inv_r4_scale:
	// scale by 1/n
	CVTSQ2SS R13, X0
	MOVSS    ·one32(SB), X1
	DIVSS    X0, X1
	SHUFPS   $0x00, X1, X1

	XORQ CX, CX

inv_r4_scale_loop:
	CMPQ CX, R13
	JGE  inv_r4_return_true
	MOVSD (AX)(CX*8), X0
	MULPS X1, X0
	MOVSD X0, (AX)(CX*8)
	INCQ CX
	JMP  inv_r4_scale_loop

inv_r4_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_r4_return_false:
	MOVB $0, ret+120(FP)
	RET
