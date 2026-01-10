//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2-optimized FFT Assembly for AMD64 (complex64/float32)
// ===========================================================================
//
// This file implements FFT transforms using SSE2 instructions for complex64
// (single-precision) data types as a fallback for systems without AVX2 support.
// SSE2 is available on all x86-64 CPUs.
//
// SIMD STRATEGY
// -------------
// - complex64:  Process 2 butterflies per iteration using XMM (128-bit) registers
//               Each complex64 = 8 bytes, so XMM holds 2 complex numbers
//
// See avx2_f32_generic.s for detailed algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardSSE2Complex64Asm(SB), NOSPLIT, $0-97
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n = len(src)

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    sse2_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   sse2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   sse2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   sse2_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  sse2_check_power
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  sse2_return_true

sse2_check_power:
	// Verify n is power of 2
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  sse2_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	CMPQ R8, R9
	JNE  sse2_use_dst
	MOVQ R11, R8             // In-place: use scratch

sse2_use_dst:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	// Compute nbits = log2(n)
	MOVQ R13, R12
	XORQ R15, R15

sse2_bitrev_bits_loop:
	CMPQ R12, $1
	JLE  sse2_bitrev_bits_done
	SHRQ $1, R12
	INCQ R15
	JMP  sse2_bitrev_bits_loop

sse2_bitrev_bits_done:
	XORQ CX, CX

sse2_bitrev_loop:
	CMPQ CX, R13
	JGE  sse2_bitrev_done
	MOVQ CX, AX
	XORQ DX, DX
	MOVQ R15, R14

sse2_bitrev_inner:
	TESTQ R14, R14
	JZ    sse2_bitrev_inner_done
	SHLQ $1, DX
	MOVQ AX, BX
	ANDQ $1, BX
	ORQ BX, DX
	SHRQ $1, AX
	DECQ R14
	JMP  sse2_bitrev_inner

sse2_bitrev_inner_done:
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	JMP  sse2_bitrev_loop

sse2_bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: Main DIT Butterfly Stages
	// -----------------------------------------------------------------------
	MOVQ $2, R14             // size = 2

sse2_size_loop:
	CMPQ R14, R13
	JG   sse2_transform_done

	MOVQ R14, R15
	SHRQ $1, R15             // half = size / 2

	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX              // step = n / size

	XORQ CX, CX              // base = 0

sse2_base_loop:
	CMPQ CX, R13
	JGE  sse2_next_size

	// SSE2 vector path only for contiguous twiddles and half >= 2
	CMPQ BX, $1
	JNE  sse2_scalar_butterflies
	CMPQ R15, $2
	JL   sse2_scalar_butterflies

	XORQ DX, DX              // j = 0

sse2_vec_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $2
	JL   sse2_scalar_remainder

	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI              // SI = (base + j) * 8

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI              // DI = (base + j + half) * 8

	// Load 2 complex64 values for a and b
	MOVUPS (R8)(SI*1), X0
	MOVUPS (R8)(DI*1), X1

	// Load contiguous twiddles (2 complex64)
	MOVQ DX, AX
	SHLQ $3, AX
	MOVUPS (R10)(AX*1), X2

	// t = w * b (SSE2, 2 complex64)
	MOVAPS X1, X3
	MULPS  X2, X3            // prod1 = b * w

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4     // w_shuf

	MOVAPS X1, X5
	MULPS  X4, X5            // prod2 = b * w_shuf

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	SUBPS  X6, X3            // real = prod1 - shuf(prod1)

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	ADDPS  X7, X5            // imag = prod2 + shuf(prod2)

	SHUFPS $0x88, X3, X3     // real lanes
	SHUFPS $0xDD, X5, X5     // imag lanes
	UNPCKLPS X5, X3          // t = [r0,i0,r1,i1]

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store results
	MOVUPS X6, (R8)(SI*1)
	MOVUPS X7, (R8)(DI*1)

	ADDQ $2, DX
	JMP  sse2_vec_loop

sse2_scalar_remainder:
	CMPQ DX, R15
	JGE  sse2_next_base

sse2_scalar_loop:
	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load single complex64 values
	XORPS X0, X0
	MOVSD (R8)(SI*1), X0     // a
	XORPS X1, X1
	MOVSD (R8)(DI*1), X1     // b

	// Load twiddle (stride for scalar path)
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	XORPS X2, X2
	MOVSD (R10)(AX*1), X2    // w

	// t = w * b (SSE2, 1 complex64)
	MOVAPS X1, X3
	MULPS  X2, X3

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4

	MOVAPS X1, X5
	MULPS  X4, X5

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	SUBPS  X6, X3

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	ADDPS  X7, X5

	SHUFPS $0x88, X3, X3
	SHUFPS $0xDD, X5, X5
	UNPCKLPS X5, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store
	MOVSD X6, (R8)(SI*1)
	MOVSD X7, (R8)(DI*1)

	INCQ DX
	CMPQ DX, R15
	JL   sse2_scalar_loop
	JMP  sse2_next_base

sse2_scalar_butterflies:
	// Pure scalar path for early/strided stages
	XORQ DX, DX

sse2_scalar_only_loop:
	CMPQ DX, R15
	JGE  sse2_next_base

	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load single complex64 values
	XORPS X0, X0
	MOVSD (R8)(SI*1), X0
	XORPS X1, X1
	MOVSD (R8)(DI*1), X1

	// Load twiddle with stride
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	XORPS X2, X2
	MOVSD (R10)(AX*1), X2

	// t = w * b
	MOVAPS X1, X3
	MULPS  X2, X3

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4

	MOVAPS X1, X5
	MULPS  X4, X5

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	SUBPS  X6, X3

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	ADDPS  X7, X5

	SHUFPS $0x88, X3, X3
	SHUFPS $0xDD, X5, X5
	UNPCKLPS X5, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store
	MOVSD X6, (R8)(SI*1)
	MOVSD X7, (R8)(DI*1)

	INCQ DX
	JMP  sse2_scalar_only_loop

sse2_next_base:
	ADDQ R14, CX
	JMP  sse2_base_loop

sse2_next_size:
	SHLQ $1, R14
	JMP  sse2_size_loop

sse2_transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Copy results back if we used scratch
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   sse2_return_true

	XORQ CX, CX

sse2_copy_loop:
	CMPQ CX, R13
	JGE  sse2_return_true
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  sse2_copy_loop

sse2_return_true:
	MOVB $1, ret+96(FP)
	RET

sse2_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// inverseSSE2Complex64Asm - Inverse FFT for complex64 using SSE2
// ===========================================================================
TEXT ·InverseSSE2Complex64Asm(SB), NOSPLIT, $0-97
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n = len(src)

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    inv_sse2_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   inv_sse2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   inv_sse2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   inv_sse2_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  inv_sse2_check_power
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  inv_sse2_return_true

inv_sse2_check_power:
	// Verify n is power of 2
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  inv_sse2_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select working buffer
	// -----------------------------------------------------------------------
	CMPQ R8, R9
	JNE  inv_sse2_use_dst
	MOVQ R11, R8             // In-place: use scratch

inv_sse2_use_dst:
	// -----------------------------------------------------------------------
	// PHASE 3: Bit-reversal permutation
	// -----------------------------------------------------------------------
	// Compute nbits = log2(n)
	MOVQ R13, R12
	XORQ R15, R15

inv_sse2_bitrev_bits_loop:
	CMPQ R12, $1
	JLE  inv_sse2_bitrev_bits_done
	SHRQ $1, R12
	INCQ R15
	JMP  inv_sse2_bitrev_bits_loop

inv_sse2_bitrev_bits_done:
	XORQ CX, CX

inv_sse2_bitrev_loop:
	CMPQ CX, R13
	JGE  inv_sse2_bitrev_done
	MOVQ CX, AX
	XORQ DX, DX
	MOVQ R15, R14

inv_sse2_bitrev_inner:
	TESTQ R14, R14
	JZ    inv_sse2_bitrev_inner_done
	SHLQ $1, DX
	MOVQ AX, BX
	ANDQ $1, BX
	ORQ BX, DX
	SHRQ $1, AX
	DECQ R14
	JMP  inv_sse2_bitrev_inner

inv_sse2_bitrev_inner_done:
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	JMP  inv_sse2_bitrev_loop

inv_sse2_bitrev_done:
	// -----------------------------------------------------------------------
	// PHASE 4: DIT butterfly stages with CONJUGATE twiddles
	// -----------------------------------------------------------------------
	MOVQ $2, R14

inv_sse2_size_loop:
	CMPQ R14, R13
	JG   inv_sse2_transform_done

	MOVQ R14, R15
	SHRQ $1, R15

	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX

	XORQ CX, CX

inv_sse2_base_loop:
	CMPQ CX, R13
	JGE  inv_sse2_next_size

	// SSE2 vector path only for contiguous twiddles and half >= 2
	CMPQ BX, $1
	JNE  inv_sse2_scalar_butterflies
	CMPQ R15, $2
	JL   inv_sse2_scalar_butterflies

	XORQ DX, DX

inv_sse2_vec_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $2
	JL   inv_sse2_scalar_remainder

	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load 2 complex64 values for a and b
	MOVUPS (R8)(SI*1), X0
	MOVUPS (R8)(DI*1), X1

	// Load contiguous twiddles (2 complex64)
	MOVQ DX, AX
	SHLQ $3, AX
	MOVUPS (R10)(AX*1), X2

	// t = conj(w) * b (SSE2, 2 complex64)
	MOVAPS X1, X3
	MULPS  X2, X3            // prod1 = b * w

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4     // w_shuf

	MOVAPS X1, X5
	MULPS  X4, X5            // prod2 = b * w_shuf

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	ADDPS  X6, X3            // real = prod1 + shuf(prod1)

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	SUBPS  X5, X7            // imag = shuf(prod2) - prod2

	SHUFPS $0x88, X3, X3
	SHUFPS $0x88, X7, X7
	UNPCKLPS X7, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store results
	MOVUPS X6, (R8)(SI*1)
	MOVUPS X7, (R8)(DI*1)

	ADDQ $2, DX
	JMP  inv_sse2_vec_loop

inv_sse2_scalar_remainder:
	CMPQ DX, R15
	JGE  inv_sse2_next_base

inv_sse2_scalar_loop:
	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load single complex64 values
	XORPS X0, X0
	MOVSD (R8)(SI*1), X0
	XORPS X1, X1
	MOVSD (R8)(DI*1), X1

	// Load twiddle with stride
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	XORPS X2, X2
	MOVSD (R10)(AX*1), X2

	// t = conj(w) * b
	MOVAPS X1, X3
	MULPS  X2, X3

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4

	MOVAPS X1, X5
	MULPS  X4, X5

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	ADDPS  X6, X3

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	SUBPS  X5, X7

	SHUFPS $0x88, X3, X3
	SHUFPS $0x88, X7, X7
	UNPCKLPS X7, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store
	MOVSD X6, (R8)(SI*1)
	MOVSD X7, (R8)(DI*1)

	INCQ DX
	CMPQ DX, R15
	JL   inv_sse2_scalar_loop
	JMP  inv_sse2_next_base

inv_sse2_scalar_butterflies:
	// Pure scalar path for early/strided stages
	XORQ DX, DX

inv_sse2_scalar_only_loop:
	CMPQ DX, R15
	JGE  inv_sse2_next_base

	// Compute byte offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load single complex64 values
	XORPS X0, X0
	MOVSD (R8)(SI*1), X0
	XORPS X1, X1
	MOVSD (R8)(DI*1), X1

	// Load twiddle with stride
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	XORPS X2, X2
	MOVSD (R10)(AX*1), X2

	// t = conj(w) * b
	MOVAPS X1, X3
	MULPS  X2, X3

	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4

	MOVAPS X1, X5
	MULPS  X4, X5

	MOVAPS X3, X6
	SHUFPS $0xB1, X6, X6
	ADDPS  X6, X3

	MOVAPS X5, X7
	SHUFPS $0xB1, X7, X7
	SUBPS  X5, X7

	SHUFPS $0x88, X3, X3
	SHUFPS $0x88, X7, X7
	UNPCKLPS X7, X3

	// Butterfly
	MOVAPS X0, X6
	ADDPS  X3, X6
	MOVAPS X0, X7
	SUBPS  X3, X7

	// Store
	MOVSD X6, (R8)(SI*1)
	MOVSD X7, (R8)(DI*1)

	INCQ DX
	JMP  inv_sse2_scalar_only_loop

inv_sse2_next_base:
	ADDQ R14, CX
	JMP  inv_sse2_base_loop

inv_sse2_next_size:
	SHLQ $1, R14
	JMP  inv_sse2_size_loop

inv_sse2_transform_done:
	// -----------------------------------------------------------------------
	// PHASE 5: Copy back (if needed) and Scale by 1/n
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   inv_sse2_scale

	XORQ CX, CX

inv_sse2_copy_loop:
	CMPQ CX, R13
	JGE  inv_sse2_scale
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  inv_sse2_copy_loop

inv_sse2_scale:
	// Scale output by 1/n
	MOVQ dst+0(FP), R8

	CVTSQ2SS R13, X0
	MOVSS    ·one32(SB), X1
	DIVSS    X0, X1
	SHUFPS   $0x00, X1, X1   // broadcast scale

	XORQ CX, CX

inv_sse2_scale_loop:
	CMPQ CX, R13
	JGE  inv_sse2_return_true

	XORPS X0, X0
	MOVSD (R8)(CX*8), X0
	MULPS X1, X0
	MOVSD X0, (R8)(CX*8)

	INCQ CX
	JMP  inv_sse2_scale_loop

inv_sse2_return_true:
	MOVB $1, ret+96(FP)
	RET

inv_sse2_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// forwardAVX2Complex128Asm - Forward FFT for complex128 using AVX2/FMA
// ===========================================================================
// Double-precision (float64) version of the forward FFT.
//
// Key differences from complex64 version:
//   - complex128 = 16 bytes (8 bytes real + 8 bytes imag)
//   - YMM holds 2 complex128 values (vs 4 complex64)
//   - Uses VMOVUPD (packed double) instead of VMOVUPS (packed single)
//   - Uses VMULPD, VADDPD, VSUBPD instead of PS variants
//   - Uses VFMADDSUB231PD instead of VFMADDSUB231PS
//   - Uses VMOVDDUP and VPERMILPD for component broadcast/shuffle
//   - Minimum size: n >= 8 (vs n >= 16 for complex64)
// ===========================================================================
