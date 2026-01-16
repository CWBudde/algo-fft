//go:build arm64 && asm && !purego

// ===========================================================================
// NEON Inverse Real FFT Repack Helper (complex64)
// ===========================================================================

#include "textflag.h"

DATA ·complex64OnesMaskNEON+0(SB)/4, $0x3f800000
DATA ·complex64OnesMaskNEON+4(SB)/4, $0x00000000
DATA ·complex64OnesMaskNEON+8(SB)/4, $0x3f800000
DATA ·complex64OnesMaskNEON+12(SB)/4, $0x00000000
GLOBL ·complex64OnesMaskNEON(SB), RODATA|NOPTR, $16

DATA ·float32TwosNEON+0(SB)/4, $0x40000000
DATA ·float32TwosNEON+4(SB)/4, $0x40000000
DATA ·float32TwosNEON+8(SB)/4, $0x40000000
DATA ·float32TwosNEON+12(SB)/4, $0x40000000
GLOBL ·float32TwosNEON(SB), RODATA|NOPTR, $16

DATA ·complex64ImagSignMaskNEON+0(SB)/4, $0x00000000
DATA ·complex64ImagSignMaskNEON+4(SB)/4, $0x80000000
DATA ·complex64ImagSignMaskNEON+8(SB)/4, $0x00000000
DATA ·complex64ImagSignMaskNEON+12(SB)/4, $0x80000000
GLOBL ·complex64ImagSignMaskNEON(SB), RODATA|NOPTR, $16

// func InverseRepackComplex64NEONAsm(dst, src, weight []complex64, kStartMax int)
TEXT ·InverseRepackComplex64NEONAsm(SB), NOSPLIT, $0-80
	MOVD dst+0(FP), R0       // dst base
	MOVD dst+8(FP), R8       // half (dst len)
	MOVD src+24(FP), R1      // src base
	MOVD weight+48(FP), R2   // weight base
	MOVD kStartMax+72(FP), R3

	CMP  $1, R3
	BLT  neon_repack_done

	// Load constant pointers
	MOVD $·complex64OnesMaskNEON(SB), R4
	FMOVS (R4), F28          // F28 = 1.0
	MOVD $·float32TwosNEON(SB), R5
	FMOVS (R5), F29          // F29 = 2.0

	MOVD $1, R9              // k = 1

neon_repack_loop:
	CMP  R3, R9
	BGT  neon_repack_done

	// kStart offset = k * 8
	LSL  $3, R9, R10

	// mStart = half - k
	SUB  R9, R8, R11
	CMP  R11, R9
	BGT  neon_repack_done
	LSL  $3, R11, R12        // mStart offset

	// Compute addresses
	ADD  R1, R10, R13        // src + kStart*8
	ADD  R1, R12, R14        // src + mStart*8
	ADD  R2, R10, R15        // weight + kStart*8

	// Load xk and xmk, conjugate xmk
	FMOVS 0(R13), F0         // xk.re
	FMOVS 4(R13), F1         // xk.im
	FMOVS 0(R14), F2         // xmk.re
	FMOVS 4(R14), F3         // xmk.im
	FNEGS F3, F3             // xmk.im = -xmk.im (conjugate)

	// Load U
	FMOVS 0(R15), F4         // u.re
	FMOVS 4(R15), F5         // u.im

	// oneMinusU = (1 - u.re, -u.im)
	FSUBS F4, F28, F6        // F6 = 1.0 - u.re
	FNEGS F5, F7             // F7 = -u.im

	// invDet = conj(1 - 2*u) = (1 - 2*u.re, 2*u.im)
	FMULS F29, F4, F9        // F9 = 2*u.re
	FSUBS F9, F28, F8        // F8 = invDet.re = 1 - 2*u.re
	FMULS F29, F5, F9        // F9 = invDet.im = 2*u.im

	// t0 = xk * oneMinusU
	FMULS F6, F0, F10        // F10 = xk.re * oneMinusU.re
	FMULS F7, F1, F11        // F11 = xk.im * oneMinusU.im
	FSUBS F11, F10, F10      // F10 = t0.re
	FMULS F7, F0, F11        // F11 = xk.re * oneMinusU.im
	FMULS F6, F1, F12        // F12 = xk.im * oneMinusU.re
	FADDS F12, F11, F11      // F11 = t0.im

	// t1 = xmkc * U
	FMULS F4, F2, F12        // F12 = xmk.re * u.re
	FMULS F5, F3, F13        // F13 = xmk.im * u.im
	FSUBS F13, F12, F12      // F12 = t1.re
	FMULS F5, F2, F13        // F13 = xmk.re * u.im
	FMULS F4, F3, F14        // F14 = xmk.im * u.re
	FADDS F14, F13, F13      // F13 = t1.im

	// a = (t0 - t1) * invDet
	FSUBS F12, F10, F10      // F10 = a.re (pre) = t0.re - t1.re
	FSUBS F13, F11, F11      // F11 = a.im (pre) = t0.im - t1.im
	FMULS F8, F10, F12       // F12 = a.re(pre) * invDet.re
	FMULS F9, F11, F14       // F14 = a.im(pre) * invDet.im
	FSUBS F14, F12, F12      // F12 = a.re
	FMULS F9, F10, F14       // F14 = a.re(pre) * invDet.im
	FMULS F8, F11, F15       // F15 = a.im(pre) * invDet.re
	FADDS F15, F14, F14      // F14 = a.im

	// Store a to dst[kStart]
	ADD  R0, R10, R16        // dst + kStart*8
	FMOVS F12, 0(R16)
	FMOVS F14, 4(R16)

	// t2 = xmkc * oneMinusU
	FMULS F6, F2, F10        // F10 = xmk.re * oneMinusU.re
	FMULS F7, F3, F11        // F11 = xmk.im * oneMinusU.im
	FSUBS F11, F10, F10      // F10 = t2.re
	FMULS F7, F2, F11        // F11 = xmk.re * oneMinusU.im
	FMULS F6, F3, F12        // F12 = xmk.im * oneMinusU.re
	FADDS F12, F11, F11      // F11 = t2.im

	// t3 = xk * U
	FMULS F4, F0, F12        // F12 = xk.re * u.re
	FMULS F5, F1, F13        // F13 = xk.im * u.im
	FSUBS F13, F12, F12      // F12 = t3.re
	FMULS F5, F0, F13        // F13 = xk.re * u.im
	FMULS F4, F1, F14        // F14 = xk.im * u.re
	FADDS F14, F13, F13      // F13 = t3.im

	// b = (t2 - t3) * invDet
	FSUBS F12, F10, F10      // F10 = b.re (pre)
	FSUBS F13, F11, F11      // F11 = b.im (pre)
	FMULS F8, F10, F12       // F12 = b.re(pre) * invDet.re
	FMULS F9, F11, F14       // F14 = b.im(pre) * invDet.im
	FSUBS F14, F12, F12      // F12 = b.re
	FMULS F9, F10, F14       // F14 = b.re(pre) * invDet.im
	FMULS F8, F11, F15       // F15 = b.im(pre) * invDet.re
	FADDS F15, F14, F14      // F14 = b.im

	// Skip storing b if k == mStart (middle element)
	CMP  R11, R9
	BEQ  neon_repack_next

	// Store conj(b) to dst[mStart]
	FNEGS F14, F14           // conj(b).im
	ADD  R0, R12, R16        // dst + mStart*8
	FMOVS F12, 0(R16)
	FMOVS F14, 4(R16)

neon_repack_next:
	ADD  $1, R9, R9
	B    neon_repack_loop

neon_repack_done:
	RET
