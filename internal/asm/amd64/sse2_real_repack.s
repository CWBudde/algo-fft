//go:build amd64

// ===========================================================================
// SSE2 Inverse Real FFT Repack Helper (complex64)
// ===========================================================================

#include "textflag.h"

DATA ·complex64OnesMask+0(SB)/4, $0x3f800000
DATA ·complex64OnesMask+4(SB)/4, $0x00000000
DATA ·complex64OnesMask+8(SB)/4, $0x3f800000
DATA ·complex64OnesMask+12(SB)/4, $0x00000000
GLOBL ·complex64OnesMask(SB), RODATA|NOPTR, $16

DATA ·float32Twos+0(SB)/4, $0x40000000
DATA ·float32Twos+4(SB)/4, $0x40000000
DATA ·float32Twos+8(SB)/4, $0x40000000
DATA ·float32Twos+12(SB)/4, $0x40000000
GLOBL ·float32Twos(SB), RODATA|NOPTR, $16

DATA ·complex64ImagSignMask+0(SB)/4, $0x00000000
DATA ·complex64ImagSignMask+4(SB)/4, $0x80000000
DATA ·complex64ImagSignMask+8(SB)/4, $0x00000000
DATA ·complex64ImagSignMask+12(SB)/4, $0x80000000
GLOBL ·complex64ImagSignMask(SB), RODATA|NOPTR, $16

// func InverseRepackComplex64SSE2Asm(dst, src, weight []complex64, kStartMax int)
TEXT ·InverseRepackComplex64SSE2Asm(SB), NOSPLIT, $0-80
	MOVQ dst+0(FP), DI
	MOVQ dst+8(FP), R8
	MOVQ src+24(FP), SI
	MOVQ weight+48(FP), DX
	MOVQ kStartMax+72(FP), CX

	CMPQ CX, $1
	JL   sse2_repack_done

	MOVQ $·complex64OnesMask(SB), R9
	MOVUPS (R9), X12
	MOVQ $·float32Twos(SB), R10
	MOVUPS (R10), X14

	MOVQ $1, AX

sse2_repack_loop:
	CMPQ AX, CX
	JG   sse2_repack_done

	// Reload scalar constants (X12=1.0, X14=2.0) each iteration.
	MOVSS (R9), X12
	MOVSS (R10), X14

	// kStart offset
	MOVQ AX, R12
	SHLQ $3, R12

	// mStart = half - kStart
	MOVQ R8, R13
	SUBQ AX, R13
	CMPQ AX, R13
	JG   sse2_repack_done
	MOVQ R13, R14
	SHLQ $3, R14

	// Load xk and xmk, conjugate xmk.
	MOVSS (SI)(R12*1), X0         // xk.re
	MOVSS 4(SI)(R12*1), X1        // xk.im
	MOVSS (SI)(R14*1), X2         // xmk.re
	MOVSS 4(SI)(R14*1), X3        // xmk.im
	XORPS ·maskNegLoPS(SB), X3    // xmk.im = -xmk.im

	// Load U.
	MOVSS (DX)(R12*1), X4         // u.re
	MOVSS 4(DX)(R12*1), X5        // u.im

	// oneMinusU = (1 - u.re, -u.im)
	MOVSS X12, X6
	SUBSS X4, X6
	MOVSS X5, X7
	XORPS ·maskNegLoPS(SB), X7

	// invDet = conj(1 - 2*u) = (1 - 2*u.re, 2*u.im)
	MOVSS X4, X9
	MULSS X14, X9                 // 2*u.re
	MOVSS X12, X8
	SUBSS X9, X8                  // invDet.re
	MOVSS X5, X9
	MULSS X14, X9                 // invDet.im

	// t0 = xk * oneMinusU
	MOVSS X0, X10
	MULSS X6, X10                 // xk.re * oneMinusU.re
	MOVSS X1, X11
	MULSS X7, X11                 // xk.im * oneMinusU.im
	SUBSS X11, X10                // t0.re
	MOVSS X0, X11
	MULSS X7, X11                 // xk.re * oneMinusU.im
	MOVSS X1, X13
	MULSS X6, X13                 // xk.im * oneMinusU.re
	ADDSS X13, X11                // t0.im

	// t1 = xmkc * U
	MOVSS X2, X13
	MULSS X4, X13                 // xmk.re * u.re
	MOVSS X3, X15
	MULSS X5, X15                 // xmk.im * u.im
	SUBSS X15, X13                // t1.re
	MOVSS X2, X15
	MULSS X5, X15                 // xmk.re * u.im
	MOVSS X3, X14
	MULSS X4, X14                 // xmk.im * u.re
	ADDSS X14, X15                // t1.im

	// a = (t0 - t1) * invDet
	SUBSS X13, X10                // a.re (pre)
	SUBSS X15, X11                // a.im (pre)
	MOVSS X10, X13
	MULSS X8, X13                 // a.re * invDet.re
	MOVSS X11, X15
	MULSS X9, X15                 // a.im * invDet.im
	SUBSS X15, X13                // a.re
	MOVSS X10, X15
	MULSS X9, X15                 // a.re * invDet.im
	MOVSS X11, X14
	MULSS X8, X14                 // a.im * invDet.re
	ADDSS X14, X15                // a.im
	MOVSS X13, (DI)(R12*1)
	MOVSS X15, 4(DI)(R12*1)

	// t2 = xmkc * oneMinusU
	MOVSS X2, X10
	MULSS X6, X10                 // xmk.re * oneMinusU.re
	MOVSS X3, X11
	MULSS X7, X11                 // xmk.im * oneMinusU.im
	SUBSS X11, X10                // t2.re
	MOVSS X2, X11
	MULSS X7, X11                 // xmk.re * oneMinusU.im
	MOVSS X3, X13
	MULSS X6, X13                 // xmk.im * oneMinusU.re
	ADDSS X13, X11                // t2.im

	// t3 = xk * U
	MOVSS X0, X13
	MULSS X4, X13                 // xk.re * u.re
	MOVSS X1, X15
	MULSS X5, X15                 // xk.im * u.im
	SUBSS X15, X13                // t3.re
	MOVSS X0, X15
	MULSS X5, X15                 // xk.re * u.im
	MOVSS X1, X14
	MULSS X4, X14                 // xk.im * u.re
	ADDSS X14, X15                // t3.im

	// b = (t2 - t3) * invDet
	SUBSS X13, X10                // b.re (pre)
	SUBSS X15, X11                // b.im (pre)
	MOVSS X10, X13
	MULSS X8, X13                 // b.re * invDet.re
	MOVSS X11, X15
	MULSS X9, X15                 // b.im * invDet.im
	SUBSS X15, X13                // b.re
	MOVSS X10, X15
	MULSS X9, X15                 // b.re * invDet.im
	MOVSS X11, X14
	MULSS X8, X14                 // b.im * invDet.re
	ADDSS X14, X15                // b.im
	CMPQ AX, R13
	JE   sse2_repack_next
	XORPS ·maskNegLoPS(SB), X15   // conj(b).im
	MOVSS X13, (DI)(R14*1)
	MOVSS X15, 4(DI)(R14*1)

sse2_repack_next:
	ADDQ $1, AX
	JMP  sse2_repack_loop

sse2_repack_done:
	RET
