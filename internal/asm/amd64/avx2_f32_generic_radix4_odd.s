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

	// Fast path for vectorized strided twiddles.
	CMPQ R11, $4
	JL   fwd_r4m_stage_scalar

	MOVQ BX, R9
	SHLQ $3, R9             // stride1_bytes

	MOVQ DX, R15
	IMULQ BX, R15
	SHLQ $3, R15            // twiddle offset bytes (j*step*8)

fwd_r4m_stepn_loop:
	MOVQ R11, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   fwd_r4m_stage_scalar

	// indices (element offsets)
	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R11, DI
	MOVQ DI, BP
	ADDQ R11, BP
	MOVQ BP, AX
	ADDQ R11, AX

	VMOVUPS (R8)(SI*8), Y0
	VMOVUPS (R8)(DI*8), Y1
	VMOVUPS (R8)(BP*8), Y2
	VMOVUPS (R8)(AX*8), Y3

	// twiddle base offsets
	MOVQ R15, DI
	SHLQ $1, DI
	MOVQ R15, BP
	ADDQ DI, BP

	// w1 = twiddle[j*step + i*step] for i=0..3
	VMOVSD (R10)(R15*1), X4
	ADDQ R9, R15
	VMOVSD (R10)(R15*1), X5
	ADDQ R9, R15
	VMOVSD (R10)(R15*1), X6
	ADDQ R9, R15
	VMOVSD (R10)(R15*1), X7
	ADDQ R9, R15            // advance to next block
	VPUNPCKLQDQ X5, X4, X4
	VPUNPCKLQDQ X7, X6, X6
	VINSERTF128 $1, X6, Y4, Y4

	// w2 = twiddle[2*j*step + i*2*step]
	MOVQ R9, AX
	SHLQ $1, AX             // stride2_bytes
	VMOVSD (R10)(DI*1), X5
	ADDQ AX, DI
	VMOVSD (R10)(DI*1), X6
	ADDQ AX, DI
	VMOVSD (R10)(DI*1), X7
	ADDQ AX, DI
	VMOVSD (R10)(DI*1), X8
	VPUNPCKLQDQ X6, X5, X5
	VPUNPCKLQDQ X8, X7, X7
	VINSERTF128 $1, X7, Y5, Y5

	// w3 = twiddle[3*j*step + i*3*step]
	LEAQ (R9)(AX*1), AX     // stride3_bytes
	VMOVSD (R10)(BP*1), X6
	ADDQ AX, BP
	VMOVSD (R10)(BP*1), X7
	ADDQ AX, BP
	VMOVSD (R10)(BP*1), X8
	ADDQ AX, BP
	VMOVSD (R10)(BP*1), X9
	VPUNPCKLQDQ X7, X6, X6
	VPUNPCKLQDQ X9, X8, X8
	VINSERTF128 $1, X8, Y6, Y6

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
	VBLENDPS $0xAA, Y7, Y14, Y14

	// i*t3
	VPERMILPS $0xB1, Y13, Y7
	VSUBPS Y7, Y15, Y8
	VBLENDPS $0x55, Y8, Y7, Y7

	VADDPS Y10, Y12, Y0
	VADDPS Y11, Y14, Y1
	VSUBPS Y12, Y10, Y2
	VADDPS Y11, Y7, Y3

	// store results (recompute indices)
	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R11, DI
	MOVQ DI, BP
	ADDQ R11, BP
	MOVQ BP, AX
	ADDQ R11, AX

	VMOVUPS Y0, (R8)(SI*8)
	VMOVUPS Y1, (R8)(DI*8)
	VMOVUPS Y2, (R8)(BP*8)
	VMOVUPS Y3, (R8)(AX*8)

	ADDQ $4, DX
	JMP  fwd_r4m_stepn_loop

fwd_r4m_stage_scalar:
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
	MOVQ R11, R15
	SHLQ $2, R15
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

	// Vectorized path for radix-2 (4 complex per iteration).
	MOVQ R11, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   fwd_r4m_r2_scalar

	// base offsets in bytes
	MOVQ CX, SI
	SHLQ $3, SI             // base bytes

	MOVQ DX, DI
	SHLQ $3, DI
	LEAQ (R8)(SI*1), BP
	ADDQ DI, BP             // ptrA = base + j

	MOVQ DX, DI
	ADDQ R11, DI
	SHLQ $3, DI
	LEAQ (R8)(SI*1), R9
	ADDQ DI, R9             // ptrB = base + half + j

	LEAQ (R10)(DX*8), R14   // twiddle ptr

fwd_r4m_r2_vec:
	MOVQ R11, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   fwd_r4m_r2_scalar

	VMOVUPS (BP), Y0        // a
	VMOVUPS (R9), Y1        // b
	VMOVUPS (R14), Y2       // twiddle

	VMOVSLDUP Y2, Y3        // w.r
	VMOVSHDUP Y2, Y4        // w.i
	VSHUFPS $0xB1, Y1, Y1, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y1, Y5 // t = w * b

	VADDPS Y5, Y0, Y6
	VSUBPS Y5, Y0, Y7

	VMOVUPS Y6, (BP)
	VMOVUPS Y7, (R9)

	ADDQ $32, BP
	ADDQ $32, R9
	ADDQ $32, R14
	ADDQ $4, DX
	JMP  fwd_r4m_r2_vec

fwd_r4m_r2_scalar:

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

	// Fast path for vectorized strided twiddles.
	CMPQ R11, $4
	JL   inv_r4m_stage_scalar

	MOVQ BX, R9
	SHLQ $3, R9             // stride1_bytes

	MOVQ DX, R15
	IMULQ BX, R15
	SHLQ $3, R15            // twiddle offset bytes (j*step*8)

inv_r4m_stepn_loop:
	MOVQ R11, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   inv_r4m_stage_scalar

	// indices (element offsets)
	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R11, DI
	MOVQ DI, BP
	ADDQ R11, BP
	MOVQ BP, AX
	ADDQ R11, AX

	VMOVUPS (R8)(SI*8), Y0
	VMOVUPS (R8)(DI*8), Y1
	VMOVUPS (R8)(BP*8), Y2
	VMOVUPS (R8)(AX*8), Y3

	// twiddle base offsets
	MOVQ R15, DI
	SHLQ $1, DI
	MOVQ R15, BP
	ADDQ DI, BP

	// w1 = twiddle[j*step + i*step] for i=0..3
	VMOVSD (R10)(R15*1), X4
	ADDQ R9, R15
	VMOVSD (R10)(R15*1), X5
	ADDQ R9, R15
	VMOVSD (R10)(R15*1), X6
	ADDQ R9, R15
	VMOVSD (R10)(R15*1), X7
	ADDQ R9, R15            // advance to next block
	VPUNPCKLQDQ X5, X4, X4
	VPUNPCKLQDQ X7, X6, X6
	VINSERTF128 $1, X6, Y4, Y4

	// w2 = twiddle[2*j*step + i*2*step]
	MOVQ R9, AX
	SHLQ $1, AX             // stride2_bytes
	VMOVSD (R10)(DI*1), X5
	ADDQ AX, DI
	VMOVSD (R10)(DI*1), X6
	ADDQ AX, DI
	VMOVSD (R10)(DI*1), X7
	ADDQ AX, DI
	VMOVSD (R10)(DI*1), X8
	VPUNPCKLQDQ X6, X5, X5
	VPUNPCKLQDQ X8, X7, X7
	VINSERTF128 $1, X7, Y5, Y5

	// w3 = twiddle[3*j*step + i*3*step]
	LEAQ (R9)(AX*1), AX     // stride3_bytes
	VMOVSD (R10)(BP*1), X6
	ADDQ AX, BP
	VMOVSD (R10)(BP*1), X7
	ADDQ AX, BP
	VMOVSD (R10)(BP*1), X8
	ADDQ AX, BP
	VMOVSD (R10)(BP*1), X9
	VPUNPCKLQDQ X7, X6, X6
	VPUNPCKLQDQ X9, X8, X8
	VINSERTF128 $1, X8, Y6, Y6

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
	VBLENDPS $0xAA, Y7, Y14, Y14

	// i*t3
	VPERMILPS $0xB1, Y13, Y7
	VSUBPS Y7, Y15, Y8
	VBLENDPS $0x55, Y8, Y7, Y7

	VADDPS Y10, Y12, Y0
	VADDPS Y11, Y7, Y1
	VSUBPS Y12, Y10, Y2
	VADDPS Y11, Y14, Y3

	// store results (recompute indices)
	MOVQ CX, SI
	ADDQ DX, SI
	MOVQ SI, DI
	ADDQ R11, DI
	MOVQ DI, BP
	ADDQ R11, BP
	MOVQ BP, AX
	ADDQ R11, AX

	VMOVUPS Y0, (R8)(SI*8)
	VMOVUPS Y1, (R8)(DI*8)
	VMOVUPS Y2, (R8)(BP*8)
	VMOVUPS Y3, (R8)(AX*8)

	ADDQ $4, DX
	JMP  inv_r4m_stepn_loop

inv_r4m_stage_scalar:
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
	MOVQ R11, R15
	SHLQ $2, R15
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

	// Vectorized path for radix-2 (4 complex per iteration).
	MOVQ R11, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   inv_r4m_r2_scalar

	// base offsets in bytes
	MOVQ CX, SI
	SHLQ $3, SI             // base bytes

	MOVQ DX, DI
	SHLQ $3, DI
	LEAQ (R8)(SI*1), BP
	ADDQ DI, BP             // ptrA = base + j

	MOVQ DX, DI
	ADDQ R11, DI
	SHLQ $3, DI
	LEAQ (R8)(SI*1), R9
	ADDQ DI, R9             // ptrB = base + half + j

	LEAQ (R10)(DX*8), R14   // twiddle ptr

inv_r4m_r2_vec:
	MOVQ R11, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   inv_r4m_r2_scalar

	VMOVUPS (BP), Y0        // a
	VMOVUPS (R9), Y1        // b
	VMOVUPS (R14), Y2       // twiddle

	VMOVSLDUP Y2, Y3        // w.r
	VMOVSHDUP Y2, Y4        // w.i
	VSHUFPS $0xB1, Y1, Y1, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y1, Y5 // t = conj(w) * b

	VADDPS Y5, Y0, Y6
	VSUBPS Y5, Y0, Y7

	VMOVUPS Y6, (BP)
	VMOVUPS Y7, (R9)

	ADDQ $32, BP
	ADDQ $32, R9
	ADDQ $32, R14
	ADDQ $4, DX
	JMP  inv_r4m_r2_vec

inv_r4m_r2_scalar:

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
