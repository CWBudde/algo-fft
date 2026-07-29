//go:build amd64 && !purego

#include "textflag.h"

// ===========================================================================
// AVX2 Stockham autosort FFT kernels for complex128
// ===========================================================================
// Same stage structure as the complex64 Stockham kernels in
// avx2_f32_generic.s: for each stage (m = n, n/2, ..., 2; half = m/2)
//
//	out[k*half + j]       = in[k*m + j] + in[k*m + j + half]
//	out[k*half + j + n/2] = (in[k*m + j] - in[k*m + j + half]) * w[j*step]
//
// with step = n/m, ping-ponging between dst and scratch. Each YMM register
// holds two complex128 values [r0 i0 r1 i1]. Complex multiply idiom:
//
//	VMOVDDUP  w, wr        // [wr0 wr0 wr1 wr1]
//	VPERMILPD $0xF, w, wi  // [wi0 wi0 wi1 wi1]
//	VPERMILPD $0x5, d, ds  // [di0 dr0 di1 dr1]
//	VMULPD    wi, ds, t    // [di*wi dr*wi ...]
//	VFMADDSUB231PD wr, d, t  // fwd: [dr*wr-di*wi, di*wr+dr*wi] = d*w
//	VFMSUBADD231PD wr, d, t  // inv: [dr*wr+di*wi, di*wr-dr*wi] = d*conj(w)
// ===========================================================================

// func ForwardAVX2StockhamComplex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·ForwardAVX2StockhamComplex128Asm(SB), NOSPLIT, $0-97
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src_len+32(FP), R13 // R13 = n = len(src)

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    stockham128_return_true

	// Validate all slice lengths are >= n
	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   stockham128_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, R13
	JL   stockham128_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   stockham128_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  stockham128_check_power
	VMOVUPS (R9), X0
	VMOVUPS X0, (R8)
	JMP  stockham128_return_true

stockham128_check_power:
	// Verify n is power of 2
	MOVQ  R13, AX
	LEAQ  -1(AX), BX
	TESTQ AX, BX
	JNZ   stockham128_return_false

	// Minimum size for AVX2 vectorization
	CMPQ R13, $16
	JL   stockham128_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select buffers (Stockham uses ping-pong)
	// -----------------------------------------------------------------------
	MOVQ R9, SI               // SI = in (src)
	MOVQ R8, DI               // DI = out (dst)
	CMPQ R8, R9
	JNE  stockham128_out_ready
	MOVQ R11, DI              // In-place: first out = scratch

stockham128_out_ready:
	// m starts at n and halves each stage
	MOVQ R13, R14             // R14 = m

stockham128_stage_loop:
	CMPQ R14, $1
	JLE  stockham128_done

	// step = n / m
	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX               // BX = step (also group count)

	XORQ CX, CX               // k = 0

stockham128_k_loop:
	CMPQ CX, BX
	JGE  stockham128_stage_done

	// half = m / 2
	MOVQ R14, R15
	SHRQ $1, R15

	// baseElem = k * m
	MOVQ  CX, AX
	IMULQ R14, AX

	// ptrA = in + baseElem*16
	SHLQ $4, AX              // AX = baseElem * 16
	LEAQ (SI)(AX*1), BP      // BP = ptrA = in + baseElem*16

	// ptrB = ptrA + half*16
	MOVQ R15, AX
	SHLQ $4, AX              // AX = half * 16
	ADDQ BP, AX              // AX = ptrB = ptrA + half*16

	// outBaseElem = k * half
	MOVQ  CX, DX
	IMULQ R15, DX

	// ptrOut0 = out + outBaseElem*16
	SHLQ $4, DX              // DX = outBaseElem * 16
	LEAQ (DI)(DX*1), R9      // R9 = ptrOut0 (DI = current output buffer)

	// ptrOut1 = ptrOut0 + (n/2)*16
	MOVQ R13, R12
	SHLQ $3, R12             // R12 = n * 8 bytes
	LEAQ (R9)(R12*1), R12    // R12 = ptrOut1

	// remaining = half
	MOVQ R15, DX             // DX = half

	// Fast path for contiguous twiddles (step == 1)
	CMPQ BX, $1
	JNE  stockham128_scalar_strided

	// twiddle offset for contiguous path
	XORQ R11, R11
	CMPQ DX, $2
	JL   stockham128_scalar_contig

stockham128_vec_loop:
	CMPQ DX, $2
	JL   stockham128_scalar_contig

	VMOVUPD (BP), Y0          // a (ptrA)
	VMOVUPD (AX), Y1          // b (ptrB)
	VMOVUPD (R10)(R11*1), Y2  // twiddle

	VADDPD Y1, Y0, Y3         // sum = a + b
	VSUBPD Y1, Y0, Y4         // diff = a - b

	VMOVDDUP  Y2, Y5          // w.r
	VPERMILPD $0xF, Y2, Y6    // w.i
	VPERMILPD $0x5, Y4, Y7    // diff swapped
	VMULPD    Y6, Y7, Y7
	VFMADDSUB231PD Y5, Y4, Y7 // t = diff * w

	VMOVUPD Y3, (R9)          // out0 = sum
	VMOVUPD Y7, (R12)         // out1 = diff * w

	ADDQ $32, BP
	ADDQ $32, AX
	ADDQ $32, R9
	ADDQ $32, R12
	ADDQ $32, R11
	SUBQ $2, DX
	JMP  stockham128_vec_loop

stockham128_scalar_contig:
	MOVQ $16, R15             // stride bytes for step==1
	JMP  stockham128_scalar_core

stockham128_scalar_strided:
	MOVQ BX, R15
	SHLQ $4, R15              // stride bytes = step * 16
	XORQ R11, R11             // twiddle offset bytes

	CMPQ DX, $2
	JL   stockham128_scalar_core

stockham128_strided_vec_loop:
	CMPQ DX, $2
	JL   stockham128_scalar_core

	VMOVUPD (BP), Y0          // a (ptrA)
	VMOVUPD (AX), Y1          // b (ptrB)

	// Gather 2 strided twiddles using running offset
	VMOVUPD (R10)(R11*1), X2
	ADDQ    R15, R11
	VMOVUPD (R10)(R11*1), X3
	ADDQ    R15, R11          // advance to next block
	VINSERTF128 $1, X3, Y2, Y2

	VADDPD Y1, Y0, Y3         // sum = a + b
	VSUBPD Y1, Y0, Y4         // diff = a - b

	VMOVDDUP  Y2, Y5          // w.r
	VPERMILPD $0xF, Y2, Y6    // w.i
	VPERMILPD $0x5, Y4, Y7    // diff swapped
	VMULPD    Y6, Y7, Y7
	VFMADDSUB231PD Y5, Y4, Y7 // t = diff * w

	VMOVUPD Y3, (R9)          // out0 = sum
	VMOVUPD Y7, (R12)         // out1 = diff * w

	ADDQ $32, BP
	ADDQ $32, AX
	ADDQ $32, R9
	ADDQ $32, R12
	SUBQ $2, DX
	JMP  stockham128_strided_vec_loop

stockham128_scalar_core:
	CMPQ DX, $0
	JLE  stockham128_k_done

stockham128_scalar_loop:
	VMOVSD (BP), X0  // a.r (ptrA)
	VMOVSD 8(BP), X1 // a.i
	VMOVSD (AX), X2  // b.r (ptrB)
	VMOVSD 8(AX), X3 // b.i

	// sum = a + b
	VMOVSD X0, X4, X4
	VADDSD X2, X4, X4
	VMOVSD X1, X5, X5
	VADDSD X3, X5, X5
	VMOVSD X4, (R9)
	VMOVSD X5, 8(R9)

	// diff = a - b
	VMOVSD X0, X6, X6
	VSUBSD X2, X6, X6
	VMOVSD X1, X7, X7
	VSUBSD X3, X7, X7

	// twiddle (strided)
	VMOVSD (R10)(R11*1), X8
	VMOVSD 8(R10)(R11*1), X9

	// t = diff * w
	VMOVSD X6, X10, X10
	VMULSD X8, X10, X10
	VMOVSD X7, X11, X11
	VMULSD X9, X11, X11
	VSUBSD X11, X10, X10

	VMOVSD X7, X12, X12
	VMULSD X8, X12, X12
	VMOVSD X6, X13, X13
	VMULSD X9, X13, X13
	VADDSD X13, X12, X12

	VMOVSD X10, (R12)
	VMOVSD X12, 8(R12)

	ADDQ $16, BP
	ADDQ $16, AX
	ADDQ $16, R9
	ADDQ $16, R12
	ADDQ R15, R11
	DECQ DX
	JNZ  stockham128_scalar_loop

stockham128_k_done:
	INCQ CX
	JMP  stockham128_k_loop

stockham128_stage_done:
	// Swap in/out buffers
	MOVQ DI, SI
	MOVQ dst+0(FP), AX
	CMPQ DI, AX
	JE   stockham128_out_to_scratch
	MOVQ AX, DI
	JMP  stockham128_stage_next

stockham128_out_to_scratch:
	MOVQ scratch+72(FP), DI

stockham128_stage_next:
	SHRQ $1, R14
	JMP  stockham128_stage_loop

stockham128_done:
	VZEROUPPER
	MOVQ dst+0(FP), AX
	CMPQ SI, AX
	JE   stockham128_return_true

	XORQ CX, CX

stockham128_copy_loop:
	CMPQ   CX, R13
	JGE    stockham128_return_true
	MOVQ   CX, DX
	SHLQ   $4, DX
	VMOVUPS (SI)(DX*1), X0
	VMOVUPS X0, (AX)(DX*1)
	INCQ   CX
	JMP    stockham128_copy_loop

stockham128_return_true:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

stockham128_return_false:
	MOVB $0, ret+96(FP)
	RET

// func InverseAVX2StockhamComplex128Asm(dst, src, twiddle, scratch []complex128) bool
TEXT ·InverseAVX2StockhamComplex128Asm(SB), NOSPLIT, $0-97
	// -----------------------------------------------------------------------
	// PHASE 1: Load parameters and validate inputs
	// -----------------------------------------------------------------------
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src_len+32(FP), R13 // R13 = n = len(src)

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    inv_stockham128_return_true

	// Validate all slice lengths are >= n
	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   inv_stockham128_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, R13
	JL   inv_stockham128_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   inv_stockham128_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  inv_stockham128_check_power
	VMOVUPS (R9), X0
	VMOVUPS X0, (R8)
	JMP  inv_stockham128_return_true

inv_stockham128_check_power:
	// Verify n is power of 2
	MOVQ  R13, AX
	LEAQ  -1(AX), BX
	TESTQ AX, BX
	JNZ   inv_stockham128_return_false

	// Minimum size for AVX2 vectorization
	CMPQ R13, $16
	JL   inv_stockham128_return_false

	// -----------------------------------------------------------------------
	// PHASE 2: Select buffers (Stockham uses ping-pong)
	// -----------------------------------------------------------------------
	MOVQ R9, SI               // SI = in (src)
	MOVQ R8, DI               // DI = out (dst)
	CMPQ R8, R9
	JNE  inv_stockham128_out_ready
	MOVQ R11, DI              // In-place: first out = scratch

inv_stockham128_out_ready:
	// m starts at n and halves each stage
	MOVQ R13, R14             // R14 = m

inv_stockham128_stage_loop:
	CMPQ R14, $1
	JLE  inv_stockham128_done

	// step = n / m
	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX               // BX = step (also group count)

	XORQ CX, CX               // k = 0

inv_stockham128_k_loop:
	CMPQ CX, BX
	JGE  inv_stockham128_stage_done

	// half = m / 2
	MOVQ R14, R15
	SHRQ $1, R15

	// baseElem = k * m
	MOVQ  CX, AX
	IMULQ R14, AX

	// ptrA = in + baseElem*16
	SHLQ $4, AX              // AX = baseElem * 16
	LEAQ (SI)(AX*1), BP      // BP = ptrA = in + baseElem*16

	// ptrB = ptrA + half*16
	MOVQ R15, AX
	SHLQ $4, AX              // AX = half * 16
	ADDQ BP, AX              // AX = ptrB = ptrA + half*16

	// outBaseElem = k * half
	MOVQ  CX, DX
	IMULQ R15, DX

	// ptrOut0 = out + outBaseElem*16
	SHLQ $4, DX              // DX = outBaseElem * 16
	LEAQ (DI)(DX*1), R9      // R9 = ptrOut0 (DI = current output buffer)

	// ptrOut1 = ptrOut0 + (n/2)*16
	MOVQ R13, R12
	SHLQ $3, R12             // R12 = n * 8 bytes
	LEAQ (R9)(R12*1), R12    // R12 = ptrOut1

	// remaining = half
	MOVQ R15, DX             // DX = half

	// Fast path for contiguous twiddles (step == 1)
	CMPQ BX, $1
	JNE  inv_stockham128_scalar_strided

	// twiddle offset for contiguous path
	XORQ R11, R11
	CMPQ DX, $2
	JL   inv_stockham128_scalar_contig

inv_stockham128_vec_loop:
	CMPQ DX, $2
	JL   inv_stockham128_scalar_contig

	VMOVUPD (BP), Y0          // a (ptrA)
	VMOVUPD (AX), Y1          // b (ptrB)
	VMOVUPD (R10)(R11*1), Y2  // twiddle

	VADDPD Y1, Y0, Y3         // sum = a + b
	VSUBPD Y1, Y0, Y4         // diff = a - b

	// Conjugate multiply: diff * conj(w)
	VMOVDDUP  Y2, Y5          // w.r
	VPERMILPD $0xF, Y2, Y6    // w.i
	VPERMILPD $0x5, Y4, Y7    // diff swapped
	VMULPD    Y6, Y7, Y7
	VFMSUBADD231PD Y5, Y4, Y7 // t = diff * conj(w)

	VMOVUPD Y3, (R9)          // out0 = sum
	VMOVUPD Y7, (R12)         // out1 = diff * conj(w)

	ADDQ $32, BP
	ADDQ $32, AX
	ADDQ $32, R9
	ADDQ $32, R12
	ADDQ $32, R11
	SUBQ $2, DX
	JMP  inv_stockham128_vec_loop

inv_stockham128_scalar_contig:
	MOVQ $16, R15             // stride bytes for step==1
	JMP  inv_stockham128_scalar_core

inv_stockham128_scalar_strided:
	MOVQ BX, R15
	SHLQ $4, R15              // stride bytes = step * 16
	XORQ R11, R11             // twiddle offset bytes

	CMPQ DX, $2
	JL   inv_stockham128_scalar_core

inv_stockham128_strided_vec_loop:
	CMPQ DX, $2
	JL   inv_stockham128_scalar_core

	VMOVUPD (BP), Y0          // a (ptrA)
	VMOVUPD (AX), Y1          // b (ptrB)

	// Gather 2 strided twiddles using running offset
	VMOVUPD (R10)(R11*1), X2
	ADDQ    R15, R11
	VMOVUPD (R10)(R11*1), X3
	ADDQ    R15, R11          // advance to next block
	VINSERTF128 $1, X3, Y2, Y2

	VADDPD Y1, Y0, Y3         // sum = a + b
	VSUBPD Y1, Y0, Y4         // diff = a - b

	// Conjugate multiply: diff * conj(w)
	VMOVDDUP  Y2, Y5          // w.r
	VPERMILPD $0xF, Y2, Y6    // w.i
	VPERMILPD $0x5, Y4, Y7    // diff swapped
	VMULPD    Y6, Y7, Y7
	VFMSUBADD231PD Y5, Y4, Y7 // t = diff * conj(w)

	VMOVUPD Y3, (R9)          // out0 = sum
	VMOVUPD Y7, (R12)         // out1 = diff * conj(w)

	ADDQ $32, BP
	ADDQ $32, AX
	ADDQ $32, R9
	ADDQ $32, R12
	SUBQ $2, DX
	JMP  inv_stockham128_strided_vec_loop

inv_stockham128_scalar_core:
	CMPQ DX, $0
	JLE  inv_stockham128_k_done

inv_stockham128_scalar_loop:
	VMOVSD (BP), X0  // a.r (ptrA)
	VMOVSD 8(BP), X1 // a.i
	VMOVSD (AX), X2  // b.r (ptrB)
	VMOVSD 8(AX), X3 // b.i

	// sum = a + b
	VMOVSD X0, X4, X4
	VADDSD X2, X4, X4
	VMOVSD X1, X5, X5
	VADDSD X3, X5, X5
	VMOVSD X4, (R9)
	VMOVSD X5, 8(R9)

	// diff = a - b
	VMOVSD X0, X6, X6
	VSUBSD X2, X6, X6
	VMOVSD X1, X7, X7
	VSUBSD X3, X7, X7

	// twiddle (conjugate, strided)
	VMOVSD (R10)(R11*1), X8
	VMOVSD 8(R10)(R11*1), X9

	// t = diff * conj(w)
	VMOVSD X6, X10, X10
	VMULSD X8, X10, X10
	VMOVSD X7, X11, X11
	VMULSD X9, X11, X11
	VADDSD X11, X10, X10

	VMOVSD X7, X12, X12
	VMULSD X8, X12, X12
	VMOVSD X6, X13, X13
	VMULSD X9, X13, X13
	VSUBSD X13, X12, X12

	VMOVSD X10, (R12)
	VMOVSD X12, 8(R12)

	ADDQ $16, BP
	ADDQ $16, AX
	ADDQ $16, R9
	ADDQ $16, R12
	ADDQ R15, R11
	DECQ DX
	JNZ  inv_stockham128_scalar_loop

inv_stockham128_k_done:
	INCQ CX
	JMP  inv_stockham128_k_loop

inv_stockham128_stage_done:
	// Swap in/out buffers
	MOVQ DI, SI
	MOVQ dst+0(FP), AX
	CMPQ DI, AX
	JE   inv_stockham128_out_to_scratch
	MOVQ AX, DI
	JMP  inv_stockham128_stage_next

inv_stockham128_out_to_scratch:
	MOVQ scratch+72(FP), DI

inv_stockham128_stage_next:
	SHRQ $1, R14
	JMP  inv_stockham128_stage_loop

inv_stockham128_done:
	VZEROUPPER
	MOVQ dst+0(FP), AX
	CMPQ SI, AX
	JE   inv_stockham128_scale

	XORQ CX, CX

inv_stockham128_copy_loop:
	CMPQ   CX, R13
	JGE    inv_stockham128_scale
	MOVQ   CX, DX
	SHLQ   $4, DX
	VMOVUPS (SI)(DX*1), X0
	VMOVUPS X0, (AX)(DX*1)
	INCQ   CX
	JMP    inv_stockham128_copy_loop

inv_stockham128_scale:
	// scale by 1/n
	VCVTSI2SDQ R13, X0, X0
	VMOVSD ·one64(SB), X1
	VDIVSD X0, X1, X1
	VMOVDDUP X1, X1

	XORQ CX, CX

inv_stockham128_scale_loop:
	CMPQ   CX, R13
	JGE    inv_stockham128_return_true
	MOVQ   CX, DX
	SHLQ   $4, DX
	VMOVUPS (AX)(DX*1), X0
	VMULPD X1, X0, X0
	VMOVUPS X0, (AX)(DX*1)
	INCQ   CX
	JMP    inv_stockham128_scale_loop

inv_stockham128_return_true:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

inv_stockham128_return_false:
	MOVB $0, ret+96(FP)
	RET
