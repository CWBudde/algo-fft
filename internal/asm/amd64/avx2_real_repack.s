//go:build amd64

// ===========================================================================
// AVX2 Inverse Real FFT Repack Helper (complex64)
// ===========================================================================

#include "textflag.h"

DATA ·complex64OnesMaskY+0(SB)/4, $0x3f800000
DATA ·complex64OnesMaskY+4(SB)/4, $0x00000000
DATA ·complex64OnesMaskY+8(SB)/4, $0x3f800000
DATA ·complex64OnesMaskY+12(SB)/4, $0x00000000
DATA ·complex64OnesMaskY+16(SB)/4, $0x3f800000
DATA ·complex64OnesMaskY+20(SB)/4, $0x00000000
DATA ·complex64OnesMaskY+24(SB)/4, $0x3f800000
DATA ·complex64OnesMaskY+28(SB)/4, $0x00000000
GLOBL ·complex64OnesMaskY(SB), RODATA|NOPTR, $32

DATA ·float32TwosY+0(SB)/4, $0x40000000
DATA ·float32TwosY+4(SB)/4, $0x40000000
DATA ·float32TwosY+8(SB)/4, $0x40000000
DATA ·float32TwosY+12(SB)/4, $0x40000000
DATA ·float32TwosY+16(SB)/4, $0x40000000
DATA ·float32TwosY+20(SB)/4, $0x40000000
DATA ·float32TwosY+24(SB)/4, $0x40000000
DATA ·float32TwosY+28(SB)/4, $0x40000000
GLOBL ·float32TwosY(SB), RODATA|NOPTR, $32

DATA ·complex64ImagSignMaskY+0(SB)/4, $0x00000000
DATA ·complex64ImagSignMaskY+4(SB)/4, $0x80000000
DATA ·complex64ImagSignMaskY+8(SB)/4, $0x00000000
DATA ·complex64ImagSignMaskY+12(SB)/4, $0x80000000
DATA ·complex64ImagSignMaskY+16(SB)/4, $0x00000000
DATA ·complex64ImagSignMaskY+20(SB)/4, $0x80000000
DATA ·complex64ImagSignMaskY+24(SB)/4, $0x00000000
DATA ·complex64ImagSignMaskY+28(SB)/4, $0x80000000
GLOBL ·complex64ImagSignMaskY(SB), RODATA|NOPTR, $32

// func InverseRepackComplex64AVX2Asm(dst, src, weight []complex64, kStartMax int)
TEXT ·InverseRepackComplex64AVX2Asm(SB), NOSPLIT, $0-80
	MOVQ dst+0(FP), DI
	MOVQ dst_len+8(FP), R8
	MOVQ src+24(FP), SI
	MOVQ weight+48(FP), DX
	MOVQ kStartMax+72(FP), CX

	CMPQ CX, $1
	JL   avx2_repack_done

	MOVQ $·complex64OnesMaskY(SB), R9
	VMOVUPS (R9), X12
	MOVQ $·float32TwosY(SB), R10
	VMOVUPS (R10), X14

	MOVQ $1, AX

avx2_repack_loop:
	CMPQ AX, CX
	JG   avx2_repack_done

	// Reload scalar constants (X12=1.0, X14=2.0) each iteration.
	VMOVSS (R9), X12
	VMOVSS (R10), X14

	// kStart offset
	MOVQ AX, R12
	SHLQ $3, R12

	// mStart = half - kStart
	MOVQ R8, R13
	SUBQ AX, R13
	CMPQ AX, R13
	JG   avx2_repack_done
	MOVQ R13, R14
	SHLQ $3, R14

	// Load xk and xmk, conjugate xmk.
	VMOVSS (SI)(R12*1), X0         // xk.re
	VMOVSS 4(SI)(R12*1), X1        // xk.im
	VMOVSS (SI)(R14*1), X2         // xmk.re
	VMOVSS 4(SI)(R14*1), X3        // xmk.im
	VXORPS ·maskNegLoPS(SB), X3, X3    // xmk.im = -xmk.im

	// Load U.
	VMOVSS (DX)(R12*1), X4         // u.re
	VMOVSS 4(DX)(R12*1), X5        // u.im

	// oneMinusU = (1 - u.re, -u.im)
	VMOVSS X12, X6, X6
	VSUBSS X4, X6, X6
	VMOVSS X5, X7, X7
	VXORPS ·maskNegLoPS(SB), X7, X7

	// invDet = conj(1 - 2*u) = (1 - 2*u.re, 2*u.im)
	VMOVSS X4, X9, X9
	VMULSS X14, X9, X9                 // 2*u.re
	VMOVSS X12, X8, X8
	VSUBSS X9, X8, X8                  // invDet.re
	VMOVSS X5, X9, X9
	VMULSS X14, X9, X9                 // invDet.im

	// t0 = xk * oneMinusU
	VMOVSS X0, X10, X10
	VMULSS X6, X10, X10                 // xk.re * oneMinusU.re
	VMOVSS X1, X11, X11
	VMULSS X7, X11, X11                 // xk.im * oneMinusU.im
	VSUBSS X11, X10, X10                // t0.re
	VMOVSS X0, X11, X11
	VMULSS X7, X11, X11                 // xk.re * oneMinusU.im
	VMOVSS X1, X13, X13
	VMULSS X6, X13, X13                 // xk.im * oneMinusU.re
	VADDSS X13, X11, X11                // t0.im

	// t1 = xmkc * U
	VMOVSS X2, X13, X13
	VMULSS X4, X13, X13                 // xmk.re * u.re
	VMOVSS X3, X15, X15
	VMULSS X5, X15, X15                 // xmk.im * u.im
	VSUBSS X15, X13, X13                // t1.re
	VMOVSS X2, X15, X15
	VMULSS X5, X15, X15                 // xmk.re * u.im
	VMOVSS X3, X14, X14
	VMULSS X4, X14, X14                 // xmk.im * u.re
	VADDSS X14, X15, X15                // t1.im

	// a = (t0 - t1) * invDet
	VSUBSS X13, X10, X10                // a.re (pre)
	VSUBSS X15, X11, X11                // a.im (pre)
	VMOVSS X10, X13, X13
	VMULSS X8, X13, X13                 // a.re * invDet.re
	VMOVSS X11, X15, X15
	VMULSS X9, X15, X15                 // a.im * invDet.im
	VSUBSS X15, X13, X13                // a.re
	VMOVSS X10, X15, X15
	VMULSS X9, X15, X15                 // a.re * invDet.im
	VMOVSS X11, X14, X14
	VMULSS X8, X14, X14                 // a.im * invDet.re
	VADDSS X14, X15, X15                // a.im
	VMOVSS X13, (DI)(R12*1)
	VMOVSS X15, 4(DI)(R12*1)

	// t2 = xmkc * oneMinusU
	VMOVSS X2, X10, X10
	VMULSS X6, X10, X10                 // xmk.re * oneMinusU.re
	VMOVSS X3, X11, X11
	VMULSS X7, X11, X11                 // xmk.im * oneMinusU.im
	VSUBSS X11, X10, X10                // t2.re
	VMOVSS X2, X11, X11
	VMULSS X7, X11, X11                 // xmk.re * oneMinusU.im
	VMOVSS X3, X13, X13
	VMULSS X6, X13, X13                 // xmk.im * oneMinusU.re
	VADDSS X13, X11, X11                // t2.im

	// t3 = xk * U
	VMOVSS X0, X13, X13
	VMULSS X4, X13, X13                 // xk.re * u.re
	VMOVSS X1, X15, X15
	VMULSS X5, X15, X15                 // xk.im * u.im
	VSUBSS X15, X13, X13                // t3.re
	VMOVSS X0, X15, X15
	VMULSS X5, X15, X15                 // xk.re * u.im
	VMOVSS X1, X14, X14
	VMULSS X4, X14, X14                 // xk.im * u.re
	VADDSS X14, X15, X15                // t3.im

	// b = (t2 - t3) * invDet
	VSUBSS X13, X10, X10                // b.re (pre)
	VSUBSS X15, X11, X11                // b.im (pre)
	VMOVSS X10, X13, X13
	VMULSS X8, X13, X13                 // b.re * invDet.re
	VMOVSS X11, X15, X15
	VMULSS X9, X15, X15                 // b.im * invDet.im
	VSUBSS X15, X13, X13                // b.re
	VMOVSS X10, X15, X15
	VMULSS X9, X15, X15                 // b.re * invDet.im
	VMOVSS X11, X14, X14
	VMULSS X8, X14, X14                 // b.im * invDet.re
	VADDSS X14, X15, X15                // b.im
	CMPQ AX, R13
	JE   avx2_repack_next
	VXORPS ·maskNegLoPS(SB), X15, X15   // conj(b).im
	VMOVSS X13, (DI)(R14*1)
	VMOVSS X15, 4(DI)(R14*1)

avx2_repack_next:
	ADDQ $1, AX
	JMP  avx2_repack_loop

avx2_repack_done:
	VZEROUPPER
	RET

// ===========================================================================
// AVX2 Inverse Real FFT Repack Helper (complex128, vectorized)
// ===========================================================================
//
// Per pair (k, m = half-k) the inverse pre-pass computes
//
//   oneMinusU = 1 - U[k]
//   invDet    = conj(1 - 2*U[k])          // det is on the unit circle
//   dst[k]    = (X[k]*oneMinusU - conj(X[m])*U[k]) * invDet
//   dst[m]    = conj((oneMinusU*conj(X[m]) - U[k]*X[k]) * invDet)
//
// Two k-bins are processed per iteration in one YMM register; the mirrored
// X[m] pair is a single reversed load (one 32-byte load ending at index m,
// then a 128-bit lane swap) and the mirrored store reverses the same way.
// ===========================================================================

DATA ·complex128OnesMaskY+0(SB)/8, $0x3ff0000000000000
DATA ·complex128OnesMaskY+8(SB)/8, $0x0000000000000000
DATA ·complex128OnesMaskY+16(SB)/8, $0x3ff0000000000000
DATA ·complex128OnesMaskY+24(SB)/8, $0x0000000000000000
GLOBL ·complex128OnesMaskY(SB), RODATA|NOPTR, $32

// func InverseRepackComplex128AVX2Asm(dst, src, weight []complex128, count int)
//
// Processes k = 1..count in blocks of 2; the caller guarantees count is a
// multiple of 2 with count <= (half-1)/2, so the k-side and mirrored m-side
// load/store ranges never overlap and stay in bounds.
TEXT ·InverseRepackComplex128AVX2Asm(SB), NOSPLIT, $0-80
	MOVQ dst+0(FP), DI     // DI = dst base pointer
	MOVQ dst_len+8(FP), R8 // R8 = half = len(dst)
	MOVQ src+24(FP), SI    // SI = src base pointer
	MOVQ weight+48(FP), DX // DX = weight base pointer
	MOVQ count+72(FP), CX  // CX = count (multiple of 2, <= (half-1)/2)

	CMPQ CX, $2
	JL   repack128_done

	MOVQ  $·complex128OnesMaskY(SB), R9
	VMOVUPD (R9), Y13      // Y13 = [1.0, 0.0, 1.0, 0.0]
	MOVQ  $·maskNegHiPD_YMM(SB), R9
	VMOVUPD (R9), Y14      // Y14 = sign mask for imaginary float64 lanes

	// Base byte offset for the mirrored side: block at k accesses
	// src/dst[m-1 .. m] with m = half-k, i.e. byte offset (half-1-k)*16.
	MOVQ R8, R11
	SUBQ $1, R11
	SHLQ $4, R11           // R11 = (half-1)*16

	MOVQ CX, R12
	SUBQ $1, R12           // R12 = last block start = count-1
	MOVQ $1, AX            // AX = k

repack128_loop:
	CMPQ AX, R12
	JG   repack128_done

	MOVQ AX, R9
	SHLQ $4, R9            // R9 = k*16

	VMOVUPD (SI)(R9*1), Y0 // Y0 = xk = src[k..k+1]

	MOVQ R11, R10
	SUBQ R9, R10           // R10 = (half-k-1)*16
	VMOVUPD (SI)(R10*1), Y1 // Y1 = [src[m-1], src[m]]
	VPERM2F128 $0x01, Y1, Y1, Y1 // swap lanes -> [src[m], src[m-1]] = xmk per k-lane
	VXORPD Y14, Y1, Y1     // Y1 = xmkc = conj(xmk)

	VMOVUPD (DX)(R9*1), Y2 // Y2 = u = weight[k..k+1]
	VSUBPD Y2, Y13, Y3     // Y3 = oneMinusU = (1 - u.r, -u.i)

	VADDPD Y2, Y2, Y4      // Y4 = 2*u
	VSUBPD Y4, Y13, Y4     // Y4 = det = 1 - 2*u
	VXORPD Y14, Y4, Y4     // Y4 = invDet = conj(det)

	// t0 = xk * oneMinusU
	VMOVDDUP Y0, Y5        // [xk.r, xk.r]
	VPERMILPD $0x0F, Y0, Y6 // [xk.i, xk.i]
	VPERMILPD $0x05, Y3, Y7 // [oneMinusU.i, oneMinusU.r]
	VMULPD Y7, Y6, Y8      // xk.i * swapped
	VFMADDSUB231PD Y5, Y3, Y8 // Y8 = t0 = xk * oneMinusU

	// t1 = xmkc * u
	VMOVDDUP Y1, Y5
	VPERMILPD $0x0F, Y1, Y6
	VPERMILPD $0x05, Y2, Y7
	VMULPD Y7, Y6, Y9
	VFMADDSUB231PD Y5, Y2, Y9 // Y9 = t1 = xmkc * u

	VSUBPD Y9, Y8, Y8      // Y8 = t0 - t1

	// a = (t0 - t1) * invDet
	VMOVDDUP Y8, Y5
	VPERMILPD $0x0F, Y8, Y6
	VPERMILPD $0x05, Y4, Y7
	VMULPD Y7, Y6, Y10
	VFMADDSUB231PD Y5, Y4, Y10 // Y10 = a
	VMOVUPD Y10, (DI)(R9*1)   // dst[k..k+1] = a

	// t2 = xmkc * oneMinusU
	VMOVDDUP Y1, Y5
	VPERMILPD $0x0F, Y1, Y6
	VPERMILPD $0x05, Y3, Y7
	VMULPD Y7, Y6, Y8
	VFMADDSUB231PD Y5, Y3, Y8 // Y8 = t2 = xmkc * oneMinusU

	// t3 = xk * u
	VMOVDDUP Y0, Y5
	VPERMILPD $0x0F, Y0, Y6
	VPERMILPD $0x05, Y2, Y7
	VMULPD Y7, Y6, Y9
	VFMADDSUB231PD Y5, Y2, Y9 // Y9 = t3 = xk * u

	VSUBPD Y9, Y8, Y8      // Y8 = t2 - t3

	// b = (t2 - t3) * invDet
	VMOVDDUP Y8, Y5
	VPERMILPD $0x0F, Y8, Y6
	VPERMILPD $0x05, Y4, Y7
	VMULPD Y7, Y6, Y10
	VFMADDSUB231PD Y5, Y4, Y10 // Y10 = b

	VXORPD Y14, Y10, Y10   // conj(b)
	VPERM2F128 $0x01, Y10, Y10, Y10 // reverse lanes to mirrored order
	VMOVUPD Y10, (DI)(R10*1) // dst[m-1..m] = [conj(b_{k+1}), conj(b_k)]

	ADDQ $2, AX            // k += 2
	JMP  repack128_loop

repack128_done:
	VZEROUPPER
	RET
