//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2/FMA-optimized FFT Assembly for AMD64 - complex64 (float32)
// Mixed radix-4 + final radix-2 DIT path for odd log2 sizes.
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// forwardAVX2Complex64Radix4MixedAsm - Forward FFT using radix-4 stages and
// a final radix-2 stage (for odd log2 sizes).
// ===========================================================================
TEXT ·ForwardAVX2Complex64Radix4MixedAsm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    fwd_r4m_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   fwd_r4m_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   fwd_r4m_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   fwd_r4m_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  fwd_r4m_check_power_of_2
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  fwd_r4m_return_true

fwd_r4m_check_power_of_2:
	// Verify n is power of 2: (n & (n-1)) == 0
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  fwd_r4m_return_false

	// Require odd log2 and minimum size
	MOVQ R13, AX
	BSRQ AX, R12            // R12 = log2(n)
	TESTQ $1, R12
	JZ   fwd_r4m_return_false

	CMPQ R13, $64
	JL   fwd_r4m_return_false

	// Number of radix-4 stages: k = (log2(n)-1)/2
	MOVQ R12, R14
	SUBQ $1, R14
	SHRQ $1, R14            // R14 = k (radix-4 stage count)

	// Select working buffer
	CMPQ R8, R9
	JNE  fwd_r4m_use_dst
	MOVQ R11, R8

fwd_r4m_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation (mixed radix: base-4 digits, then top bit)
	// -----------------------------------------------------------------------
	XORQ CX, CX

fwd_r4m_bitrev_loop:
	CMPQ CX, R13
	JGE  fwd_r4m_stage_init

	MOVQ CX, DX
	XORQ BX, BX
	MOVQ R14, SI

fwd_r4m_bitrev_inner:
	CMPQ SI, $0
	JE   fwd_r4m_bitrev_store
	MOVQ DX, AX
	ANDQ $3, AX
	SHLQ $2, BX
	ORQ  AX, BX
	SHRQ $2, DX
	DECQ SI
	JMP  fwd_r4m_bitrev_inner

fwd_r4m_bitrev_store:
	SHLQ $1, BX
	ORQ  DX, BX             // append top bit
	MOVQ (R9)(BX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	JMP  fwd_r4m_bitrev_loop

fwd_r4m_stage_init:
	// -----------------------------------------------------------------------
	// Radix-4 stages (size = 4, 16, 64, ...)
	// -----------------------------------------------------------------------
	MOVQ $2, R12            // log2(size) starting at 2
	MOVQ $4, R15            // size

fwd_r4m_stage_loop:
	CMPQ R14, $0
	JE   fwd_r4m_radix2_stage

	MOVQ R15, R11
	SHRQ $2, R11            // quarter = size/4

	// step = n >> log2(size)
	MOVQ R13, BX
	MOVQ R12, CX
	SHRQ CL, BX

	XORQ CX, CX             // base

fwd_r4m_stage_base:
	CMPQ CX, R13
	JGE  fwd_r4m_stage_next

	XORQ DX, DX             // j

fwd_r4m_stage_inner:
	CMPQ DX, R11
	JGE  fwd_r4m_stage_base_next

	// twiddle indices: w1 = twiddle[j*step], w2 = twiddle[2*j*step], w3 = twiddle[3*j*step]
	MOVQ DX, AX
	IMULQ BX, AX
	VMOVSD (R10)(AX*8), X8

	MOVQ AX, R9
	SHLQ $1, R9
	VMOVSD (R10)(R9*8), X9

	ADDQ AX, R9
	VMOVSD (R10)(R9*8), X10

	// indices
	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R11, DI
	MOVQ DI, R9
	ADDQ R11, R9
	MOVQ R9, BP
	ADDQ R11, BP

	// load inputs
	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1
	VMOVSD (R8)(R9*8), X2
	VMOVSD (R8)(BP*8), X3

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
	VMOVSD X2, (R8)(R9*8)
	VMOVSD X3, (R8)(BP*8)

	INCQ DX
	JMP  fwd_r4m_stage_inner

fwd_r4m_stage_base_next:
	ADDQ R15, CX
	JMP  fwd_r4m_stage_base

fwd_r4m_stage_next:
	ADDQ $2, R12
	SHLQ $2, R15
	DECQ R14
	JMP  fwd_r4m_stage_loop

fwd_r4m_radix2_stage:
	// -----------------------------------------------------------------------
	// Final radix-2 stage (size = n, step = 1)
	// -----------------------------------------------------------------------
	MOVQ R13, R11
	SHRQ $1, R11            // half
	XORQ CX, CX             // base

fwd_r4m_r2_base:
	CMPQ CX, R13
	JGE  fwd_r4m_copy_back

	XORQ DX, DX             // j

fwd_r4m_r2_inner:
	CMPQ DX, R11
	JGE  fwd_r4m_r2_next

	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R11, DI

	MOVQ DX, AX
	VMOVSD (R10)(AX*8), X8

	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13

	VADDPS X13, X0, X2
	VSUBPS X13, X0, X3

	VMOVSD X2, (R8)(SI*8)
	VMOVSD X3, (R8)(DI*8)

	INCQ DX
	JMP  fwd_r4m_r2_inner

fwd_r4m_r2_next:
	ADDQ R13, CX
	JMP  fwd_r4m_r2_base

fwd_r4m_copy_back:
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   fwd_r4m_return_true

	XORQ CX, CX

fwd_r4m_copy_loop:
	CMPQ CX, R13
	JGE  fwd_r4m_return_true
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  fwd_r4m_copy_loop

fwd_r4m_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

fwd_r4m_return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// inverseAVX2Complex64Radix4MixedAsm - Inverse FFT using radix-4 stages and
// a final radix-2 stage (for odd log2 sizes).
// ===========================================================================
TEXT ·InverseAVX2Complex64Radix4MixedAsm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    inv_r4m_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   inv_r4m_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   inv_r4m_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   inv_r4m_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  inv_r4m_check_power_of_2
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  inv_r4m_return_true

inv_r4m_check_power_of_2:
	// Verify n is power of 2: (n & (n-1)) == 0
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  inv_r4m_return_false

	// Require odd log2 and minimum size
	MOVQ R13, AX
	BSRQ AX, R12            // R12 = log2(n)
	TESTQ $1, R12
	JZ   inv_r4m_return_false

	CMPQ R13, $64
	JL   inv_r4m_return_false

	// Number of radix-4 stages: k = (log2(n)-1)/2
	MOVQ R12, R14
	SUBQ $1, R14
	SHRQ $1, R14            // R14 = k (radix-4 stage count)

	// Select working buffer
	CMPQ R8, R9
	JNE  inv_r4m_use_dst
	MOVQ R11, R8

inv_r4m_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation (mixed radix: base-4 digits, then top bit)
	// -----------------------------------------------------------------------
	XORQ CX, CX

inv_r4m_bitrev_loop:
	CMPQ CX, R13
	JGE  inv_r4m_stage_init

	MOVQ CX, DX
	XORQ BX, BX
	MOVQ R14, SI

inv_r4m_bitrev_inner:
	CMPQ SI, $0
	JE   inv_r4m_bitrev_store
	MOVQ DX, AX
	ANDQ $3, AX
	SHLQ $2, BX
	ORQ  AX, BX
	SHRQ $2, DX
	DECQ SI
	JMP  inv_r4m_bitrev_inner

inv_r4m_bitrev_store:
	SHLQ $1, BX
	ORQ  DX, BX             // append top bit
	MOVQ (R9)(BX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	JMP  inv_r4m_bitrev_loop

inv_r4m_stage_init:
	// -----------------------------------------------------------------------
	// Radix-4 stages (size = 4, 16, 64, ...)
	// -----------------------------------------------------------------------
	MOVQ $2, R12            // log2(size) starting at 2
	MOVQ $4, R15            // size

inv_r4m_stage_loop:
	CMPQ R14, $0
	JE   inv_r4m_radix2_stage

	MOVQ R15, R11
	SHRQ $2, R11            // quarter = size/4

	// step = n >> log2(size)
	MOVQ R13, BX
	MOVQ R12, CX
	SHRQ CL, BX

	XORQ CX, CX             // base

inv_r4m_stage_base:
	CMPQ CX, R13
	JGE  inv_r4m_stage_next

	XORQ DX, DX             // j

inv_r4m_stage_inner:
	CMPQ DX, R11
	JGE  inv_r4m_stage_base_next

	// twiddle indices: w1 = twiddle[j*step], w2 = twiddle[2*j*step], w3 = twiddle[3*j*step]
	MOVQ DX, AX
	IMULQ BX, AX
	VMOVSD (R10)(AX*8), X8

	MOVQ AX, R9
	SHLQ $1, R9
	VMOVSD (R10)(R9*8), X9

	ADDQ AX, R9
	VMOVSD (R10)(R9*8), X10

	// indices
	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R11, DI
	MOVQ DI, R9
	ADDQ R11, R9
	MOVQ R9, BP
	ADDQ R11, BP

	// load inputs
	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1
	VMOVSD (R8)(R9*8), X2
	VMOVSD (R8)(BP*8), X3

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
	VMOVSD X2, (R8)(R9*8)
	VMOVSD X3, (R8)(BP*8)

	INCQ DX
	JMP  inv_r4m_stage_inner

inv_r4m_stage_base_next:
	ADDQ R15, CX
	JMP  inv_r4m_stage_base

inv_r4m_stage_next:
	ADDQ $2, R12
	SHLQ $2, R15
	DECQ R14
	JMP  inv_r4m_stage_loop

inv_r4m_radix2_stage:
	// -----------------------------------------------------------------------
	// Final radix-2 stage (size = n, step = 1, conjugated)
	// -----------------------------------------------------------------------
	MOVQ R13, R11
	SHRQ $1, R11            // half
	XORQ CX, CX             // base

inv_r4m_r2_base:
	CMPQ CX, R13
	JGE  inv_r4m_copy_back

	XORQ DX, DX             // j

inv_r4m_r2_inner:
	CMPQ DX, R11
	JGE  inv_r4m_r2_next

	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R11, DI

	MOVQ DX, AX
	VMOVSD (R10)(AX*8), X8

	VMOVSD (R8)(SI*8), X0
	VMOVSD (R8)(DI*8), X1

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X1, X13

	VADDPS X13, X0, X2
	VSUBPS X13, X0, X3

	VMOVSD X2, (R8)(SI*8)
	VMOVSD X3, (R8)(DI*8)

	INCQ DX
	JMP  inv_r4m_r2_inner

inv_r4m_r2_next:
	ADDQ R13, CX
	JMP  inv_r4m_r2_base

inv_r4m_copy_back:
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   inv_r4m_scale

	XORQ CX, CX

inv_r4m_copy_loop:
	CMPQ CX, R13
	JGE  inv_r4m_scale
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  inv_r4m_copy_loop

inv_r4m_scale:
	// scale by 1/n
	CVTSQ2SS R13, X0
	MOVSS    ·one32(SB), X1
	DIVSS    X0, X1
	SHUFPS   $0x00, X1, X1

	XORQ CX, CX

inv_r4m_scale_loop:
	CMPQ CX, R13
	JGE  inv_r4m_return_true
	MOVSD (AX)(CX*8), X0
	MULPS X1, X0
	MOVSD X0, (AX)(CX*8)
	INCQ CX
	JMP  inv_r4m_scale_loop

inv_r4m_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_r4m_return_false:
	MOVB $0, ret+120(FP)
	RET
