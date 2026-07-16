//go:build amd64

// ===========================================================================
// SSE3 Forward Real FFT Recombination Helpers
// ===========================================================================
//
// SSE tier of avx2_real_recombine.s for hardware without AVX2:
//
//   X[k] = A[k] - U[k] * (A[k] - conj(A[half-k]))   for k = 1..half-1
//
// complex64 processes 2 bins per XMM iteration; complex128 processes 1 bin
// per iteration (a full XMM register). Both need SSE3 for MOVSLDUP/MOVSHDUP/
// MOVDDUP and ADDSUBPS/ADDSUBPD; the dispatcher falls back to the generic
// Go loop on SSE2-only hardware.
//
// The callers guarantee dst does not alias src and pass count = the number
// of bins to process starting at k=1 (a multiple of 2 for complex64) with
// count <= half-1, so the reversed loads stay within src[1..half-1].
// ===========================================================================

#include "textflag.h"

// func RecombineForwardComplex64SSE3Asm(dst, src, weight []complex64, count int)
TEXT ·RecombineForwardComplex64SSE3Asm(SB), NOSPLIT, $0-80
	MOVQ dst+0(FP), DI     // DI = dst base pointer
	MOVQ src+24(FP), SI    // SI = src base pointer
	MOVQ src_len+32(FP), R8 // R8 = half = len(src)
	MOVQ weight+48(FP), DX // DX = weight base pointer
	MOVQ count+72(FP), CX  // CX = count (multiple of 2, <= half-1)

	CMPQ CX, $2
	JL   sse3recomb64_done

	MOVQ  $·complex64ImagSignMaskY(SB), R9
	MOVUPS (R9), X5        // X5 = sign mask for imaginary float32 lanes (low 16 bytes)

	// Base byte offset for the reversed load: block at k reads
	// src[half-k-1 .. half-k], i.e. byte offset (half-1-k)*8.
	MOVQ R8, R11
	SUBQ $1, R11
	SHLQ $3, R11           // R11 = (half-1)*8

	MOVQ CX, R12
	SUBQ $1, R12           // R12 = last block start = count-1
	MOVQ $1, AX            // AX = k

sse3recomb64_loop:
	CMPQ AX, R12
	JG   sse3recomb64_done

	MOVQ AX, R9
	SHLQ $3, R9            // R9 = k*8

	MOVUPS (SI)(R9*1), X0  // X0 = a = src[k..k+1]

	MOVQ R11, R10
	SUBQ R9, R10           // R10 = (half-k-1)*8
	MOVUPS (SI)(R10*1), X1 // X1 = [src[half-k-1], src[half-k]]
	SHUFPD $0x1, X1, X1    // swap complex elements -> [src[half-k], src[half-k-1]]
	XORPS  X5, X1          // X1 = b = conj(src[half-k-j])

	MOVAPS X0, X2
	SUBPS  X1, X2          // X2 = t = a - b

	MOVUPS (DX)(R9*1), X3  // X3 = w = weight[k..k+1]
	MOVSLDUP X3, X6        // X6 = [w.r, w.r, ...]
	MOVSHDUP X3, X7        // X7 = [w.i, w.i, ...]
	MOVAPS X2, X4
	SHUFPS $0xB1, X4, X4   // X4 = [t.i, t.r, ...]
	MULPS  X7, X4          // X4 = [w.i*t.i, w.i*t.r, ...]
	MULPS  X2, X6          // X6 = [w.r*t.r, w.r*t.i, ...]
	ADDSUBPS X4, X6        // X6 = c = w*t (even: sub, odd: add)

	SUBPS  X6, X0          // X0 = a - c
	MOVUPS X0, (DI)(R9*1)  // dst[k..k+1] = a - c

	ADDQ $2, AX            // k += 2
	JMP  sse3recomb64_loop

sse3recomb64_done:
	RET

// func RecombineForwardComplex128SSE3Asm(dst, src, weight []complex128, count int)
TEXT ·RecombineForwardComplex128SSE3Asm(SB), NOSPLIT, $0-80
	MOVQ dst+0(FP), DI     // DI = dst base pointer
	MOVQ src+24(FP), SI    // SI = src base pointer
	MOVQ src_len+32(FP), R8 // R8 = half = len(src)
	MOVQ weight+48(FP), DX // DX = weight base pointer
	MOVQ count+72(FP), CX  // CX = count (<= half-1)

	CMPQ CX, $1
	JL   sse3recomb128_done

	MOVQ  $·maskNegHiPD(SB), R9
	MOVUPS (R9), X5        // X5 = sign mask for the imaginary float64 lane

	MOVQ R8, R11
	SHLQ $4, R11           // R11 = half*16

	MOVQ $1, AX            // AX = k

sse3recomb128_loop:
	CMPQ AX, CX
	JG   sse3recomb128_done

	MOVQ AX, R9
	SHLQ $4, R9            // R9 = k*16

	MOVUPS (SI)(R9*1), X0  // X0 = a = src[k]

	MOVQ R11, R10
	SUBQ R9, R10           // R10 = (half-k)*16
	MOVUPS (SI)(R10*1), X1 // X1 = src[half-k]
	XORPS  X5, X1          // X1 = b = conj(src[half-k])

	MOVAPS X0, X2
	SUBPD  X1, X2          // X2 = t = a - b

	MOVUPS (DX)(R9*1), X3  // X3 = w = weight[k]
	MOVDDUP X3, X6         // X6 = [w.r, w.r]
	MOVAPS X3, X7
	SHUFPD $0x3, X7, X7    // X7 = [w.i, w.i]
	MOVAPS X2, X4
	SHUFPD $0x1, X4, X4    // X4 = [t.i, t.r]
	MULPD  X7, X4          // X4 = [w.i*t.i, w.i*t.r]
	MULPD  X2, X6          // X6 = [w.r*t.r, w.r*t.i]
	ADDSUBPD X4, X6        // X6 = c = w*t (low: sub, high: add)

	SUBPD  X6, X0          // X0 = a - c
	MOVUPS X0, (DI)(R9*1)  // dst[k] = a - c

	ADDQ $1, AX            // k += 1
	JMP  sse3recomb128_loop

sse3recomb128_done:
	RET
