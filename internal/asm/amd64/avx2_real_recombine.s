//go:build amd64

// ===========================================================================
// AVX2 Forward Real FFT Recombination Helpers
// ===========================================================================
//
// After the half-size complex FFT of the packed even/odd samples, the real
// FFT spectrum is recovered per bin via
//
//   X[k] = A[k] - U[k] * (A[k] - conj(A[half-k]))   for k = 1..half-1
//
// where A = src (the half-size FFT output) and U = weight. Every bin is
// independent (reads from src, writes to dst), so the whole range is
// vectorized: 4 complex64 bins or 2 complex128 bins per iteration. The
// mirrored conj(A[half-k]) term is a contiguous reversed load: one unaligned
// vector load ending at index half-k, followed by an in-register reversal
// and a sign flip of the imaginary lanes.
//
// The callers guarantee dst does not alias src and pass count = the number
// of bins to process starting at k=1, always a whole number of vector
// blocks (multiple of 4 for complex64, of 2 for complex128) with
// count <= half-1, so the reversed loads stay within src[1..half-1].
// ===========================================================================

#include "textflag.h"

// func RecombineForwardComplex64AVX2Asm(dst, src, weight []complex64, count int)
TEXT ·RecombineForwardComplex64AVX2Asm(SB), NOSPLIT, $0-80
	MOVQ dst+0(FP), DI     // DI = dst base pointer
	MOVQ src+24(FP), SI    // SI = src base pointer
	MOVQ src_len+32(FP), R8 // R8 = half = len(src)
	MOVQ weight+48(FP), DX // DX = weight base pointer
	MOVQ count+72(FP), CX  // CX = count (multiple of 4, <= half-1)

	CMPQ CX, $4
	JL   recomb64_done

	MOVQ  $·complex64ImagSignMaskY(SB), R9
	VMOVUPS (R9), Y5       // Y5 = sign mask for imaginary float32 lanes

	// Base byte offset for the reversed load: block at k reads
	// src[half-k-3 .. half-k], i.e. byte offset (half-3-k)*8.
	MOVQ R8, R11
	SUBQ $3, R11
	SHLQ $3, R11           // R11 = (half-3)*8

	MOVQ CX, R12
	SUBQ $3, R12           // R12 = last block start = count-3
	MOVQ $1, AX            // AX = k

recomb64_loop:
	CMPQ AX, R12
	JG   recomb64_done

	MOVQ AX, R9
	SHLQ $3, R9            // R9 = k*8

	VMOVUPS (SI)(R9*1), Y0 // Y0 = a = src[k..k+3]

	MOVQ R11, R10
	SUBQ R9, R10           // R10 = (half-k-3)*8
	VMOVUPS (SI)(R10*1), Y1 // Y1 = src[half-k-3 .. half-k]
	VPERMQ $0x1B, Y1, Y1   // reverse complex order -> src[half-k], .., src[half-k-3]
	VXORPS Y5, Y1, Y1      // Y1 = b = conj(src[half-k-j])

	VSUBPS Y1, Y0, Y2      // Y2 = t = a - b

	VMOVUPS (DX)(R9*1), Y3 // Y3 = w = weight[k..k+3]
	VMOVSLDUP Y3, Y6       // Y6 = [w.r, w.r, ...]
	VMOVSHDUP Y3, Y7       // Y7 = [w.i, w.i, ...]
	VSHUFPS $0xB1, Y2, Y2, Y4 // Y4 = [t.i, t.r, ...]
	VMULPS Y7, Y4, Y4      // Y4 = [w.i*t.i, w.i*t.r, ...]
	VFMADDSUB231PS Y6, Y2, Y4 // Y4 = c = w.r*t -/+ Y4 = w*t

	VSUBPS Y4, Y0, Y0      // Y0 = a - c
	VMOVUPS Y0, (DI)(R9*1) // dst[k..k+3] = a - c

	ADDQ $4, AX            // k += 4
	JMP  recomb64_loop

recomb64_done:
	VZEROUPPER
	RET

// func RecombineForwardComplex128AVX2Asm(dst, src, weight []complex128, count int)
TEXT ·RecombineForwardComplex128AVX2Asm(SB), NOSPLIT, $0-80
	MOVQ dst+0(FP), DI     // DI = dst base pointer
	MOVQ src+24(FP), SI    // SI = src base pointer
	MOVQ src_len+32(FP), R8 // R8 = half = len(src)
	MOVQ weight+48(FP), DX // DX = weight base pointer
	MOVQ count+72(FP), CX  // CX = count (multiple of 2, <= half-1)

	CMPQ CX, $2
	JL   recomb128_done

	MOVQ  $·maskNegHiPD_YMM(SB), R9
	VMOVUPD (R9), Y5       // Y5 = sign mask for imaginary float64 lanes

	// Base byte offset for the reversed load: block at k reads
	// src[half-k-1 .. half-k], i.e. byte offset (half-1-k)*16.
	MOVQ R8, R11
	SUBQ $1, R11
	SHLQ $4, R11           // R11 = (half-1)*16

	MOVQ CX, R12
	SUBQ $1, R12           // R12 = last block start = count-1
	MOVQ $1, AX            // AX = k

recomb128_loop:
	CMPQ AX, R12
	JG   recomb128_done

	MOVQ AX, R9
	SHLQ $4, R9            // R9 = k*16

	VMOVUPD (SI)(R9*1), Y0 // Y0 = a = src[k..k+1]

	MOVQ R11, R10
	SUBQ R9, R10           // R10 = (half-k-1)*16
	VMOVUPD (SI)(R10*1), Y1 // Y1 = [src[half-k-1], src[half-k]]
	VPERM2F128 $0x01, Y1, Y1, Y1 // swap 128-bit halves -> [src[half-k], src[half-k-1]]
	VXORPD Y5, Y1, Y1      // Y1 = b = conj(src[half-k-j])

	VSUBPD Y1, Y0, Y2      // Y2 = t = a - b

	VMOVUPD (DX)(R9*1), Y3 // Y3 = w = weight[k..k+1]
	VMOVDDUP Y3, Y6        // Y6 = [w0.r, w0.r, w1.r, w1.r]
	VPERMILPD $0x0F, Y3, Y7 // Y7 = [w0.i, w0.i, w1.i, w1.i]
	VPERMILPD $0x05, Y2, Y4 // Y4 = [t0.i, t0.r, t1.i, t1.r]
	VMULPD Y7, Y4, Y4      // Y4 = [w.i*t.i, w.i*t.r, ...]
	VFMADDSUB231PD Y6, Y2, Y4 // Y4 = c = w.r*t -/+ Y4 = w*t

	VSUBPD Y4, Y0, Y0      // Y0 = a - c
	VMOVUPD Y0, (DI)(R9*1) // dst[k..k+1] = a - c

	ADDQ $2, AX            // k += 2
	JMP  recomb128_loop

recomb128_done:
	VZEROUPPER
	RET
