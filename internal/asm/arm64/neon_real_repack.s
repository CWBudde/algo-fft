//go:build arm64 && !purego

// ===========================================================================
// NEON Inverse Real FFT Repack Helpers (complex64 and complex128)
// ===========================================================================
//
// Per pair (k, m = half-k) the inverse pre-pass computes
//
//   oneMinusU = 1 - U[k]
//   invDet    = conj(1 - 2*U[k])          // det is on the unit circle
//   dst[k]    = (X[k]*oneMinusU - conj(X[m])*U[k]) * invDet
//   dst[m]    = conj((oneMinusU*conj(X[m]) - U[k]*X[k]) * invDet)
//
// See internal/fft/real_repack.go (inverseRepackComplex64Generic /
// inverseRepackComplex128Generic) for the scalar ground truth this mirrors,
// and internal/asm/amd64/avx2_real_repack.s for the complex128 vectorization
// this file follows the shape of (two k-bins processed per iteration, the
// mirrored X[m] side loaded/stored with one lane-reversed access).
//
// This file previously contained an "InverseRepackComplex64NEONAsm" with
// zero vector instructions -- plain scalar FMOVS/FADDS/FMULS despite the
// NEON name. Both kernels below are genuinely vectorized:
//
//   complex64  (·InverseRepackComplex64NEONAsm): a V.S4 register is 128 bits
//     = 4 x float32 = 2 complex64. Because complex64 values are stored
//     re/im-interleaved, each k-pair load is deinterleaved with VUZP1/VUZP2
//     into "duplicated" real/imag vectors ([re0,re1,re0,re1] /
//     [im0,im1,im0,im1]); all arithmetic runs on that duplicated form using
//     the WORD-encoded add/sub/mul macros from neon_fp.h, and only the final
//     store re-interleaves with VZIP1. The duplication costs nothing beyond
//     the initial VUZP/final VZIP, since every op in between applies
//     uniformly across all 4 lanes.
//
//   complex128 (·InverseRepackComplex128NEONAsm): a V.D2 register is 128
//     bits = 2 x float64 = 1 complex128, so 2 k-bins need a REGISTER PAIR
//     (one holding the two real parts, one the two imaginary parts) rather
//     than a single duplicated register. VLD2/VST2 do this deinterleaving
//     directly on load/store, so no VUZP/VZIP is needed at all -- every
//     intermediate is already in the genuinely-2-wide (not duplicated) form
//     neon_fp.h's macros operate on.
//
// Both kernels process k = 1..count in blocks of 2; the caller guarantees
// count is a multiple of 2 with count <= (half-1)/2, so the k-side and
// mirrored m-side load/store ranges never overlap, never touch the middle
// (self-mirror) element, and stay in bounds -- mirroring the AVX2 complex128
// contract exactly (see inverseRepackComplex128SIMD in
// internal/fft/real_repack_amd64.go).
//
// Go's arm64 assembler has NO mnemonic for vector FADD/FSUB/FMUL; the
// WORD-encoded macros in neon_fp.h (VADDF_S4/D2, VSUBF_S4/D2, VMULF_S4/D2,
// VFMAF_S4/D2, VFMSF_S4/D2) close that gap. VUZP1/VUZP2/VZIP1/VEXT/VEOR/VLD2/
// VST2 all have real Go assembler mnemonics and are used directly.
//
// ===========================================================================

#include "textflag.h"
#include "neon_fp.h"

// Complex multiply on "separated" real/imag vector pairs: (ar,ai) and
// (br,bi) hold the real and imaginary components of one or more complex
// values (duplicated-S4 or genuine-D2, see above), never interleaved.
// Computes (cr,ci) = (ar,ai) * (br,bi) using fused multiply-add/sub so no
// temporary register is needed. Register NUMBERS (see neon_fp.h).
#define CMULSEP_S4(ar, ai, br, bi, cr, ci) \
	VMULF_S4(ar, br, cr) \
	VFMSF_S4(ai, bi, cr) \
	VMULF_S4(ar, bi, ci) \
	VFMAF_S4(ai, br, ci)

#define CMULSEP_D2(ar, ai, br, bi, cr, ci) \
	VMULF_D2(ar, br, cr) \
	VFMSF_D2(ai, bi, cr) \
	VMULF_D2(ar, bi, ci) \
	VFMAF_D2(ai, br, ci)

// ---------------------------------------------------------------------------
// complex64 constants
// ---------------------------------------------------------------------------

// Interleaved [1.0, 0.0, 1.0, 0.0]: kept for the interleaved-form uses below
// (currently unused directly, retained for parity with the amd64 layout).
DATA ·complex64OnesMaskNEON+0(SB)/4, $0x3f800000
DATA ·complex64OnesMaskNEON+4(SB)/4, $0x00000000
DATA ·complex64OnesMaskNEON+8(SB)/4, $0x3f800000
DATA ·complex64OnesMaskNEON+12(SB)/4, $0x00000000
GLOBL ·complex64OnesMaskNEON(SB), RODATA|NOPTR, $16

// All-lanes 2.0: valid both while interleaved and on the duplicated
// separated form, since every lane needs the same "2*x" scale.
DATA ·float32TwosNEON+0(SB)/4, $0x40000000
DATA ·float32TwosNEON+4(SB)/4, $0x40000000
DATA ·float32TwosNEON+8(SB)/4, $0x40000000
DATA ·float32TwosNEON+12(SB)/4, $0x40000000
GLOBL ·float32TwosNEON(SB), RODATA|NOPTR, $16

// All-lanes 1.0: the "1" in oneMinusU/invDet once data is in the duplicated
// separated (real-only or imag-only) form, as opposed to complex64OnesMaskNEON
// above which is only valid while still interleaved.
DATA ·float32OnesAllNEON+0(SB)/4, $0x3f800000
DATA ·float32OnesAllNEON+4(SB)/4, $0x3f800000
DATA ·float32OnesAllNEON+8(SB)/4, $0x3f800000
DATA ·float32OnesAllNEON+12(SB)/4, $0x3f800000
GLOBL ·float32OnesAllNEON(SB), RODATA|NOPTR, $16

// Sign-flip mask for the imaginary lanes of an INTERLEAVED (re,im,re,im)
// complex64 pair; XOR negates only the imaginary components.
DATA ·complex64ImagSignMaskNEON+0(SB)/4, $0x00000000
DATA ·complex64ImagSignMaskNEON+4(SB)/4, $0x80000000
DATA ·complex64ImagSignMaskNEON+8(SB)/4, $0x00000000
DATA ·complex64ImagSignMaskNEON+12(SB)/4, $0x80000000
GLOBL ·complex64ImagSignMaskNEON(SB), RODATA|NOPTR, $16

// ---------------------------------------------------------------------------
// complex128 constants
// ---------------------------------------------------------------------------

DATA ·float64OnesNEON+0(SB)/8, $0x3ff0000000000000 // 1.0
GLOBL ·float64OnesNEON(SB), RODATA|NOPTR, $8

DATA ·float64TwosNEON+0(SB)/8, $0x4000000000000000 // 2.0
GLOBL ·float64TwosNEON(SB), RODATA|NOPTR, $8

// ---------------------------------------------------------------------------
// func InverseRepackComplex64NEONAsm(dst, src, weight []complex64, count int)
//
// Processes k = 1..count in blocks of 2; see the file header for the count
// contract. Register plan (V0-V27 working, V28-V31 constants/zero):
//   V0  raw xk load (interleaved)      V1  raw/conj xmk load (interleaved)
//   V2  raw u load (interleaved)       V3/V4   xkR/xkI (duplicated)
//   V5/V6   xmkcR/xmkcI (duplicated)   V7/V8   uR/uI (duplicated)
//   V9/V10  oneMinusUR/I               V11/V12 invDetR/I
//   V13/V14 xkOneR/I, later bR/bI      V15/V16 xkUR/I
//   V17/V18 xmkcOneR/I                 V19/V20 xmkcUR/I
//   V21/V22 aPreR/I                    V23/V24 bPreR/I
//   V25/V26 aR/aI                      V27 interleave/store scratch
//   V28 ones-all  V29 twos-all  V30 zero  V31 interleaved imag-sign mask
// ---------------------------------------------------------------------------
TEXT ·InverseRepackComplex64NEONAsm(SB), NOSPLIT, $0-80
	MOVD dst+0(FP), R0        // R0  = dst base
	MOVD dst_len+8(FP), R8    // R8  = half = len(dst)
	MOVD src+24(FP), R1       // R1  = src base
	MOVD weight+48(FP), R2    // R2  = weight base
	MOVD count+72(FP), R3     // R3  = count (multiple of 2, <= (half-1)/2)

	CMP  $2, R3
	BLT  neon64_repack_done

	MOVD  $·float32OnesAllNEON(SB), R4
	VLD1  (R4), [V28.S4]      // V28 = ones-all = [1,1,1,1]
	MOVD  $·float32TwosNEON(SB), R5
	VLD1  (R5), [V29.S4]      // V29 = twos-all = [2,2,2,2]
	VEOR  V30.B16, V30.B16, V30.B16 // V30 = zero
	MOVD  $·complex64ImagSignMaskNEON(SB), R6
	VLD1  (R6), [V31.S4]      // V31 = interleaved imag-sign mask

	SUB  $1, R3, R11          // R11 = count-1 (last block start)
	MOVD $1, R9               // R9  = k

neon64_repack_loop:
	CMP  R11, R9
	BGT  neon64_repack_done

	LSL  $3, R9, R12          // R12 = k*8 (complex64 byte offset)
	ADD  R1, R12, R13         // R13 = &src[k]
	ADD  R2, R12, R14         // R14 = &weight[k]
	ADD  R0, R12, R15         // R15 = &dst[k]

	SUB  R9, R8, R16          // R16 = m = half - k
	SUB  $1, R16, R16         // R16 = m-1
	LSL  $3, R16, R6          // R6  = (m-1)*8
	ADD  R1, R6, R7           // R7  = &src[m-1]
	ADD  R0, R6, R10          // R10 = &dst[m-1]

	VLD1 (R13), [V0.S4]       // V0 = xk interleaved: [re_k,im_k,re_k1,im_k1]

	VLD1 (R7), [V1.S4]        // V1 = [src[m-1], src[m]] interleaved
	VEXT $8, V1.B16, V1.B16, V1.B16 // swap halves -> [src[m], src[m-1]] = xmk for (k,k+1)
	VEOR V31.B16, V1.B16, V1.B16    // conjugate: negate imaginary lanes -> xmkc

	VLD1 (R14), [V2.S4]       // V2 = u interleaved

	// Deinterleave to duplicated real/imag form: [x0,x1,x0,x1].
	VUZP1 V0.S4, V0.S4, V3.S4 // V3 = xkR
	VUZP2 V0.S4, V0.S4, V4.S4 // V4 = xkI
	VUZP1 V1.S4, V1.S4, V5.S4 // V5 = xmkcR
	VUZP2 V1.S4, V1.S4, V6.S4 // V6 = xmkcI
	VUZP1 V2.S4, V2.S4, V7.S4 // V7 = uR
	VUZP2 V2.S4, V2.S4, V8.S4 // V8 = uI

	// oneMinusU = (1 - u.r, -u.i)
	VSUBF_S4(28, 7, 9)  // V9  = oneMinusUR = ones - uR
	VSUBF_S4(30, 8, 10) // V10 = oneMinusUI = 0 - uI

	// invDet = conj(1 - 2*u) = (1 - 2*u.r, 2*u.i)
	VMULF_S4(29, 7, 11)  // V11 = 2*uR
	VSUBF_S4(28, 11, 11) // V11 = invDetR = 1 - 2*uR
	VMULF_S4(29, 8, 12)  // V12 = invDetI = 2*uI

	CMULSEP_S4(3, 4, 9, 10, 13, 14)   // xkOne   = xk * oneMinusU
	CMULSEP_S4(3, 4, 7, 8, 15, 16)    // xkU     = xk * u
	CMULSEP_S4(5, 6, 9, 10, 17, 18)   // xmkcOne = xmkc * oneMinusU
	CMULSEP_S4(5, 6, 7, 8, 19, 20)    // xmkcU   = xmkc * u

	VSUBF_S4(13, 19, 21) // V21 = aPreR = xkOneR - xmkcUR
	VSUBF_S4(14, 20, 22) // V22 = aPreI = xkOneI - xmkcUI
	VSUBF_S4(17, 15, 23) // V23 = bPreR = xmkcOneR - xkUR
	VSUBF_S4(18, 16, 24) // V24 = bPreI = xmkcOneI - xkUI

	CMULSEP_S4(21, 22, 11, 12, 25, 26) // a = aPre * invDet
	CMULSEP_S4(23, 24, 11, 12, 13, 14) // b = bPre * invDet (reuses dead V13/V14)

	// dst[k], dst[k+1] = a (ascending order, no swap needed).
	VZIP1 V26.S4, V25.S4, V27.S4 // V27 = [aR_k,aI_k,aR_k1,aI_k1]
	VST1  [V27.S4], (R15)

	// dst[m-1], dst[m] = conj(b), reversed since b is in (k,k+1) order but
	// belongs at (m,m-1): conjugate, then swap the two complex64 halves.
	VZIP1 V14.S4, V13.S4, V27.S4        // V27 = [bR_k,bI_k,bR_k1,bI_k1]
	VEOR  V31.B16, V27.B16, V27.B16     // conjugate
	VEXT  $8, V27.B16, V27.B16, V27.B16 // -> [conj(b_k1),conj(b_k)] = (m-1,m) order
	VST1  [V27.S4], (R10)

	ADD  $2, R9, R9
	B    neon64_repack_loop

neon64_repack_done:
	RET

// ---------------------------------------------------------------------------
// func InverseRepackComplex128NEONAsm(dst, src, weight []complex128, count int)
//
// Same algorithm and count contract as the complex64 kernel above, but each
// V.D2 register already holds exactly one real/imag component pair for 2
// k-bins (no duplication needed): VLD2/VST2 deinterleave directly, so every
// intermediate stays in the genuinely-2-wide separated form throughout.
// Register plan (V0-V25 working, V28-V30 constants/zero):
//   V0/V1   xkR/xkI                    V2/V3   xmkR/xmkcI (conj in place)
//   V4/V5   uR/uI                      V6/V7   oneMinusUR/I
//   V8/V9   invDetR/I                  V10/V11 xkOneR/I
//   V12/V13 xkUR/I                     V14/V15 xmkcOneR/I
//   V16/V17 xmkcUR/I                   V18/V19 aPreR/I
//   V20/V21 bPreR/I                    V22/V23 aR/aI
//   V24/V25 bR/bI
//   V28 ones  V29 twos  V30 zero
// ---------------------------------------------------------------------------
TEXT ·InverseRepackComplex128NEONAsm(SB), NOSPLIT, $0-80
	MOVD dst+0(FP), R0        // R0  = dst base
	MOVD dst_len+8(FP), R8    // R8  = half = len(dst)
	MOVD src+24(FP), R1       // R1  = src base
	MOVD weight+48(FP), R2    // R2  = weight base
	MOVD count+72(FP), R3     // R3  = count (multiple of 2, <= (half-1)/2)

	CMP  $2, R3
	BLT  neon128_repack_done

	MOVD  $·float64OnesNEON(SB), R4
	VLD1R (R4), [V28.D2]      // V28 = ones = [1,1]
	MOVD  $·float64TwosNEON(SB), R5
	VLD1R (R5), [V29.D2]      // V29 = twos = [2,2]
	VEOR  V30.B16, V30.B16, V30.B16 // V30 = zero

	SUB  $1, R3, R11          // R11 = count-1 (last block start)
	MOVD $1, R9               // R9  = k

neon128_repack_loop:
	CMP  R11, R9
	BGT  neon128_repack_done

	LSL  $4, R9, R12          // R12 = k*16 (complex128 byte offset)
	ADD  R1, R12, R13         // R13 = &src[k]
	ADD  R2, R12, R14         // R14 = &weight[k]
	ADD  R0, R12, R15         // R15 = &dst[k]

	SUB  R9, R8, R16          // R16 = m = half - k
	SUB  $1, R16, R16         // R16 = m-1
	LSL  $4, R16, R6          // R6  = (m-1)*16
	ADD  R1, R6, R7           // R7  = &src[m-1]
	ADD  R0, R6, R10          // R10 = &dst[m-1]

	VLD2 (R13), [V0.D2, V1.D2] // V0=xkR,V1=xkI for (k,k+1)

	VLD2 (R7), [V2.D2, V3.D2]       // V2=[re(m-1),re(m)], V3=[im(m-1),im(m)]
	VEXT $8, V2.B16, V2.B16, V2.B16 // -> [re(m),re(m-1)] = xmkR for (k,k+1)
	VEXT $8, V3.B16, V3.B16, V3.B16 // -> [im(m),im(m-1)]
	VSUBF_D2(30, 3, 3)              // V3 = xmkcI = 0 - im (conjugate)

	VLD2 (R14), [V4.D2, V5.D2] // V4=uR,V5=uI

	VSUBF_D2(28, 4, 6) // V6 = oneMinusUR = ones - uR
	VSUBF_D2(30, 5, 7) // V7 = oneMinusUI = 0 - uI

	VMULF_D2(29, 4, 8) // V8 = 2*uR
	VSUBF_D2(28, 8, 8) // V8 = invDetR = 1 - 2*uR
	VMULF_D2(29, 5, 9) // V9 = invDetI = 2*uI

	CMULSEP_D2(0, 1, 6, 7, 10, 11)  // xkOne   = xk * oneMinusU
	CMULSEP_D2(0, 1, 4, 5, 12, 13)  // xkU     = xk * u
	CMULSEP_D2(2, 3, 6, 7, 14, 15)  // xmkcOne = xmkc * oneMinusU
	CMULSEP_D2(2, 3, 4, 5, 16, 17)  // xmkcU   = xmkc * u

	VSUBF_D2(10, 16, 18) // V18 = aPreR = xkOneR - xmkcUR
	VSUBF_D2(11, 17, 19) // V19 = aPreI = xkOneI - xmkcUI
	VSUBF_D2(14, 12, 20) // V20 = bPreR = xmkcOneR - xkUR
	VSUBF_D2(15, 13, 21) // V21 = bPreI = xmkcOneI - xkUI

	CMULSEP_D2(18, 19, 8, 9, 22, 23) // a = aPre * invDet
	CMULSEP_D2(20, 21, 8, 9, 24, 25) // b = bPre * invDet

	VST2 [V22.D2, V23.D2], (R15) // dst[k],dst[k+1] = a

	VSUBF_D2(30, 25, 25)                // conj(b): bI = 0 - bI
	VEXT $8, V24.B16, V24.B16, V24.B16  // swap -> (m-1,m) order
	VEXT $8, V25.B16, V25.B16, V25.B16
	VST2 [V24.D2, V25.D2], (R10) // dst[m-1],dst[m] = conj(b), reversed

	ADD  $2, R9, R9
	B    neon128_repack_loop

neon128_repack_done:
	RET
