//go:build arm64 && !purego

// ===========================================================================
// NEON Size-16 Radix-4 FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// This replaced an earlier scalar implementation that, despite the "NEON"
// name, contained no vector instructions at all — plain FMOVD/FADDD/FMULD
// with a bit-reversal loop — and measured 1.95x (forward) / 2.36x (inverse)
// SLOWER than the pure-Go codelet on an Apple M5. See docs/CODELET_BENCHMARKS.md.
//
// Algorithm — a 2x8 Cooley-Tukey, natural order in, natural order out, no
// bit-reversal pass. A Q register holds 2 float64, so a VLD2 pair holds 2
// complex128 with re/im split; n = 16 complex128 = 256 bytes = 8 such pairs,
// loaded at offsets 0,32,...,224.
//
// With n = n1 + 2*n2 (n1 in {0,1} = lane, n2 in {0..7} = vector) and
// k = 8*k1 + k2 (k1 in {0,1}, k2 in {0..7}):
//
//   X[8k1+k2] = SUM_{n1} W2^(n1*k1) * W16^(n1*k2) * [ SUM_{n2} x[n1+2*n2] * W8^(n2*k2) ]
//
//   A) DFT8 over n2   — vertical across the eight vector pairs, lane = n1.
//      Afterwards the vector index means k2, the lane still means n1.
//   B) twiddle by W16^(n1*k2) — lane n1=0 gets factor 1, lane n1=1 gets
//      W16^k2. k2 = 0 is all-ones and is skipped.
//   C) pair up k2 vectors two at a time and DFT2 over n1, which is the
//      *lane* index: pairing turns the horizontal DFT2 into a vertical one
//      and lands two consecutive outputs per register in the same step.
//   D) store — four VST2s of two consecutive outputs each.
//
// Every input is loaded into registers before the first store, so dst may
// alias src and no scratch buffer or copy-back is needed — all 16 source
// values live in registers before the DFT8 even starts.
//
// The DFT8 in (A) is built as a radix-2 decimation-in-frequency split: even
// n2 (0,2,4,6, i.e. registers holding n2=0,2,4,6) and odd n2 (1,3,5,7) each
// feed a 4-point DFT (itself two more radix-2 stages, so three radix-2
// stages total), then combined as
//
//   X[k]   = E[k] + W8^k * O[k]      for k = 0..3
//   X[k+4] = E[k] - W8^k * O[k]
//
// where E = DFT4(evens), O = DFT4(odds). The only non-trivial twiddles are
// W8^1 = c - i*c and W8^3 = -c - i*c (c = sqrt(2)/2); W8^2 = -i is a free
// re/im swap with a sign flip, folded directly into the add/sub that follows
// rather than costing a multiply.
//
// Go's arm64 assembler has NO mnemonic for vector FP add, subtract or
// multiply; the WORD-encoded macros in neon_fp.h (VADDF_D2, VSUBF_D2,
// VMULF_D2, VFMAF_D2, VFMSF_D2) close that gap directly, so this file does
// NOT use the older "VMOV + VFMLA against a ones vector" workaround that
// costs two instructions per add.
//
// ===========================================================================

#include "textflag.h"
#include "neon_fp.h"

// Forward DFT4 across four vectors (W4 = -i), operating on real parts
// ar0..ar3 and imaginary parts ai0..ai3 in place, clobbering the eight
// temporaries t0r..t3i. Register NUMBERS (see neon_fp.h).
#define VDFT4_FWD_D2(ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3, t0r, t1r, t2r, t3r, t0i, t1i, t2i, t3i) \
	VADDF_D2(ar0, ar2, t0r) \
	VSUBF_D2(ar0, ar2, t1r) \
	VADDF_D2(ar1, ar3, t2r) \
	VSUBF_D2(ar1, ar3, t3r) \
	VADDF_D2(ai0, ai2, t0i) \
	VSUBF_D2(ai0, ai2, t1i) \
	VADDF_D2(ai1, ai3, t2i) \
	VSUBF_D2(ai1, ai3, t3i) \
	VADDF_D2(t0r, t2r, ar0) \
	VADDF_D2(t0i, t2i, ai0) \
	VSUBF_D2(t0r, t2r, ar2) \
	VSUBF_D2(t0i, t2i, ai2) \
	VADDF_D2(t1r, t3i, ar1) \
	VSUBF_D2(t1i, t3r, ai1) \
	VSUBF_D2(t1r, t3i, ar3) \
	VADDF_D2(t1i, t3r, ai3)

// Inverse DFT4 across four vectors (W4 = +i): X1 and X3 swap relative to fwd.
#define VDFT4_INV_D2(ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3, t0r, t1r, t2r, t3r, t0i, t1i, t2i, t3i) \
	VADDF_D2(ar0, ar2, t0r) \
	VSUBF_D2(ar0, ar2, t1r) \
	VADDF_D2(ar1, ar3, t2r) \
	VSUBF_D2(ar1, ar3, t3r) \
	VADDF_D2(ai0, ai2, t0i) \
	VSUBF_D2(ai0, ai2, t1i) \
	VADDF_D2(ai1, ai3, t2i) \
	VSUBF_D2(ai1, ai3, t3i) \
	VADDF_D2(t0r, t2r, ar0) \
	VADDF_D2(t0i, t2i, ai0) \
	VSUBF_D2(t0r, t2r, ar2) \
	VSUBF_D2(t0i, t2i, ai2) \
	VSUBF_D2(t1r, t3i, ar1) \
	VADDF_D2(t1i, t3r, ai1) \
	VADDF_D2(t1r, t3i, ar3) \
	VSUBF_D2(t1i, t3r, ai3)

// dr,di *= (wr + i*wi), result back in dr,di; clobbers p, q. Register NUMBERS.
#define VCMUL_FWD_D2(dr, di, wr, wi, p, q) \
	VMULF_D2(dr, wr, p) \
	VFMSF_D2(di, wi, p) \
	VMULF_D2(dr, wi, q) \
	VFMAF_D2(di, wr, q) \
	VMOVR(p, dr)        \
	VMOVR(q, di)

// dr,di *= conj(wr + i*wi) — the inverse twiddle. Register NUMBERS.
#define VCMUL_INV_D2(dr, di, wr, wi, p, q) \
	VMULF_D2(dr, wr, p) \
	VFMAF_D2(di, wi, p) \
	VMULF_D2(di, wr, q) \
	VFMSF_D2(dr, wi, q) \
	VMOVR(p, dr)        \
	VMOVR(q, di)

// c = sqrt(2)/2, the only nontrivial DFT8 twiddle magnitude (W8^1, W8^3).
DATA ·neonSqrt2Half64+0(SB)/8, $0x3fe6a09e667f3bcd // sqrt(2)/2
GLOBL ·neonSqrt2Half64(SB), RODATA, $8

DATA ·neonInv16F64+0(SB)/8, $0x3fb0000000000000 // 1/16
GLOBL ·neonInv16F64(SB), RODATA, $8

// ---------------------------------------------------------------------------
// func ForwardNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool
// ---------------------------------------------------------------------------
TEXT ·ForwardNEONSize16Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $16, R13
	BNE  neon16r4f64_return_false

	MOVD dst_len+8(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_return_false

	MOVD  $·neonSqrt2Half64(SB), R0
	VLD1R (R0), [V30.D2]      // V30 = c = sqrt(2)/2, live for the whole DFT8

	// Load x[n1 + 2*n2]: vector n2, lane n1. VLD2 deinterleaves re/im, and its
	// register list must be contiguous, so re/im pairs are adjacent:
	// re[n2] = V0,V2,V4,...,V14   im[n2] = V1,V3,V5,...,V15
	VLD2 (R9), [V0.D2, V1.D2]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.D2, V3.D2]
	ADD  $64, R9, R1
	VLD2 (R1), [V4.D2, V5.D2]
	ADD  $96, R9, R1
	VLD2 (R1), [V6.D2, V7.D2]
	ADD  $128, R9, R1
	VLD2 (R1), [V8.D2, V9.D2]
	ADD  $160, R9, R1
	VLD2 (R1), [V10.D2, V11.D2]
	ADD  $192, R9, R1
	VLD2 (R1), [V12.D2, V13.D2]
	ADD  $224, R9, R1
	VLD2 (R1), [V14.D2, V15.D2]

	// (A) DFT8 over n2, split even/odd (radix-2 DIF): evens = n2 in {0,2,4,6}
	// at (V0,V4,V8,V12)/(V1,V5,V9,V13); odds = n2 in {1,3,5,7} at
	// (V2,V6,V10,V14)/(V3,V7,V11,V15). Each 4-point DFT is itself two radix-2
	// stages, so this is three radix-2 stages in total.
	VDFT4_FWD_D2(0, 4, 8, 12, 1, 5, 9, 13, 16, 17, 18, 19, 20, 21, 22, 23)  // E0..E3
	VDFT4_FWD_D2(2, 6, 10, 14, 3, 7, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23) // O0..O3

	// Combine: X[k]=E[k]+W8^k*O[k], X[k+4]=E[k]-W8^k*O[k]. Results land
	// directly in the k2 = 0..7 registers used by stages (B) and (C):
	//   k2=0:V16/17  k2=1:V18/19  k2=2:V20/21  k2=3:V22/23
	//   k2=4:V24/25  k2=5:V26/27  k2=6:V28/29  k2=7:V0/1 (reused once free)

	// r=0, W8^0=1: E0=(V0,V1), O0=(V2,V3).
	VADDF_D2(0, 2, 16) // k2=0 re
	VADDF_D2(1, 3, 17) // k2=0 im
	VSUBF_D2(0, 2, 24) // k2=4 re
	VSUBF_D2(1, 3, 25) // k2=4 im
	// V0-V3 now free.

	// r=1, W8^1=c-i*c: E1=(V4,V5), O1=(V6,V7). rot = c*(a+b) + i*c*(b-a).
	VADDF_D2(6, 7, 2) // V2 = a+b
	VSUBF_D2(7, 6, 3) // V3 = b-a
	VMULF_D2(2, 30, 2) // V2 = real(rot)
	VMULF_D2(3, 30, 3) // V3 = imag(rot)
	VADDF_D2(4, 2, 18) // k2=1 re
	VADDF_D2(5, 3, 19) // k2=1 im
	VSUBF_D2(4, 2, 26) // k2=5 re
	VSUBF_D2(5, 3, 27) // k2=5 im
	// V4-V7 now free.

	// r=2, W8^2=-i: E2=(V8,V9), O2=(V10,V11). rot=(-i)*O2=(im,-re) of O2,
	// folded directly into the add/sub — no multiply needed.
	VADDF_D2(8, 11, 20) // k2=2 re = E2re + O2im
	VSUBF_D2(9, 10, 21) // k2=2 im = E2im - O2re
	VSUBF_D2(8, 11, 28) // k2=6 re = E2re - O2im
	VADDF_D2(9, 10, 29) // k2=6 im = E2im + O2re
	// V8-V11 now free.

	// r=3, W8^3=-c-i*c: E3=(V12,V13), O3=(V14,V15).
	// rot = c*(b-a) - i*c*(a+b), a=O3re, b=O3im.
	VSUBF_D2(15, 14, 2) // V2 = b-a
	VADDF_D2(14, 15, 3) // V3 = a+b
	VMULF_D2(2, 30, 2)  // V2 = real(rot)
	VMULF_D2(3, 30, 3)  // V3 = c*(a+b); imag(rot) = -V3
	VADDF_D2(12, 2, 22) // k2=3 re = E3re + real(rot)
	VSUBF_D2(13, 3, 23) // k2=3 im = E3im - c*(a+b)
	VSUBF_D2(12, 2, 0)  // k2=7 re = E3re - real(rot)
	VADDF_D2(13, 3, 1)  // k2=7 im = E3im + c*(a+b)
	// V12-V15 now free.

	// (B) twiddle by W16^(n1*k2). k2=0 is all-ones, so it is skipped. Build
	// wr=[1.0, Re(W16^k2)], wi=[0.0, Im(W16^k2)] via broadcast + VZIP1
	// against ones/zero vectors, then VCMUL_FWD_D2 in place. 'c' in V30 is
	// dead once (A) is done, so V30 is safe to overwrite with 1.0 here.
	MOVD  $·neonOne64(SB), R0
	VLD1R (R0), [V30.D2]            // V30 = ones = [1.0, 1.0]
	VEOR  V31.B16, V31.B16, V31.B16 // V31 not used elsewhere here; zero vector

	// k2=1
	ADD   $16, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2 // wr = [1.0, Re(W16^1)]
	ADD   $24, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2 // wi = [0.0, Im(W16^1)]
	VCMUL_FWD_D2(18, 19, 4, 5, 6, 7)

	// k2=2
	ADD   $32, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $40, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_FWD_D2(20, 21, 4, 5, 6, 7)

	// k2=3
	ADD   $48, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $56, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_FWD_D2(22, 23, 4, 5, 6, 7)

	// k2=4
	ADD   $64, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $72, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_FWD_D2(24, 25, 4, 5, 6, 7)

	// k2=5
	ADD   $80, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $88, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_FWD_D2(26, 27, 4, 5, 6, 7)

	// k2=6
	ADD   $96, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $104, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_FWD_D2(28, 29, 4, 5, 6, 7)

	// k2=7
	ADD   $112, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $120, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_FWD_D2(0, 1, 4, 5, 6, 7)

	// (C) pair up k2 vectors and DFT2 over n1 (the lane index), storing each
	// pair as soon as it is ready to free registers for the next pair.

	// pair (0,1) -> X[0],X[1] (k1=0) and X[8],X[9] (k1=1)
	VZIP1 V18.D2, V16.D2, V2.D2 // p_re = [A0(n1=0), A1(n1=0)]
	VZIP2 V18.D2, V16.D2, V3.D2 // q_re = [A0(n1=1), A1(n1=1)]
	VZIP1 V19.D2, V17.D2, V4.D2 // p_im
	VZIP2 V19.D2, V17.D2, V5.D2 // q_im
	VADDF_D2(2, 3, 6)  // lo_re -> X[0],X[1]
	VADDF_D2(4, 5, 7)  // lo_im
	VSUBF_D2(2, 3, 8)  // hi_re -> X[8],X[9]
	VSUBF_D2(4, 5, 9)  // hi_im
	VST2 [V6.D2, V7.D2], (R8)
	ADD  $128, R8, R1
	VST2 [V8.D2, V9.D2], (R1)

	// pair (2,3) -> X[2],X[3] and X[10],X[11]
	VZIP1 V22.D2, V20.D2, V2.D2
	VZIP2 V22.D2, V20.D2, V3.D2
	VZIP1 V23.D2, V21.D2, V4.D2
	VZIP2 V23.D2, V21.D2, V5.D2
	VADDF_D2(2, 3, 6)
	VADDF_D2(4, 5, 7)
	VSUBF_D2(2, 3, 8)
	VSUBF_D2(4, 5, 9)
	ADD  $32, R8, R1
	VST2 [V6.D2, V7.D2], (R1)
	ADD  $160, R8, R1
	VST2 [V8.D2, V9.D2], (R1)

	// pair (4,5) -> X[4],X[5] and X[12],X[13]
	VZIP1 V26.D2, V24.D2, V2.D2
	VZIP2 V26.D2, V24.D2, V3.D2
	VZIP1 V27.D2, V25.D2, V4.D2
	VZIP2 V27.D2, V25.D2, V5.D2
	VADDF_D2(2, 3, 6)
	VADDF_D2(4, 5, 7)
	VSUBF_D2(2, 3, 8)
	VSUBF_D2(4, 5, 9)
	ADD  $64, R8, R1
	VST2 [V6.D2, V7.D2], (R1)
	ADD  $192, R8, R1
	VST2 [V8.D2, V9.D2], (R1)

	// pair (6,7) -> X[6],X[7] and X[14],X[15]
	VZIP1 V0.D2, V28.D2, V2.D2
	VZIP2 V0.D2, V28.D2, V3.D2
	VZIP1 V1.D2, V29.D2, V4.D2
	VZIP2 V1.D2, V29.D2, V5.D2
	VADDF_D2(2, 3, 6)
	VADDF_D2(4, 5, 7)
	VSUBF_D2(2, 3, 8)
	VSUBF_D2(4, 5, 9)
	ADD  $96, R8, R1
	VST2 [V6.D2, V7.D2], (R1)
	ADD  $224, R8, R1
	VST2 [V8.D2, V9.D2], (R1)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon16r4f64_return_false:
	MOVB ZR, ret+96(FP)
	RET

// ---------------------------------------------------------------------------
// func InverseNEONSize16Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool
// ---------------------------------------------------------------------------
TEXT ·InverseNEONSize16Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $16, R13
	BNE  neon16r4f64_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $16, R0
	BLT  neon16r4f64_inv_return_false

	MOVD  $·neonSqrt2Half64(SB), R0
	VLD1R (R0), [V30.D2] // V30 = c = sqrt(2)/2

	// Load x[n1 + 2*n2]: vector n2, lane n1.
	VLD2 (R9), [V0.D2, V1.D2]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.D2, V3.D2]
	ADD  $64, R9, R1
	VLD2 (R1), [V4.D2, V5.D2]
	ADD  $96, R9, R1
	VLD2 (R1), [V6.D2, V7.D2]
	ADD  $128, R9, R1
	VLD2 (R1), [V8.D2, V9.D2]
	ADD  $160, R9, R1
	VLD2 (R1), [V10.D2, V11.D2]
	ADD  $192, R9, R1
	VLD2 (R1), [V12.D2, V13.D2]
	ADD  $224, R9, R1
	VLD2 (R1), [V14.D2, V15.D2]

	// (A) inverse DFT8 over n2 — same even/odd split, conjugated twiddles.
	VDFT4_INV_D2(0, 4, 8, 12, 1, 5, 9, 13, 16, 17, 18, 19, 20, 21, 22, 23)  // E0..E3
	VDFT4_INV_D2(2, 6, 10, 14, 3, 7, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23) // O0..O3

	// r=0, conj(W8^0)=1.
	VADDF_D2(0, 2, 16) // k2=0 re
	VADDF_D2(1, 3, 17) // k2=0 im
	VSUBF_D2(0, 2, 24) // k2=4 re
	VSUBF_D2(1, 3, 25) // k2=4 im

	// r=1, conj(W8^1)=c+i*c: rot = c*(a-b) + i*c*(a+b), a=O1re,b=O1im.
	VSUBF_D2(6, 7, 2) // V2 = a-b
	VADDF_D2(6, 7, 3) // V3 = a+b
	VMULF_D2(2, 30, 2) // real(rot)
	VMULF_D2(3, 30, 3) // imag(rot)
	VADDF_D2(4, 2, 18) // k2=1 re
	VADDF_D2(5, 3, 19) // k2=1 im
	VSUBF_D2(4, 2, 26) // k2=5 re
	VSUBF_D2(5, 3, 27) // k2=5 im

	// r=2, conj(W8^2)=+i: rot=i*O2=(-im,re) of O2, folded into add/sub.
	VSUBF_D2(8, 11, 20) // k2=2 re = E2re - O2im
	VADDF_D2(9, 10, 21) // k2=2 im = E2im + O2re
	VADDF_D2(8, 11, 28) // k2=6 re = E2re + O2im
	VSUBF_D2(9, 10, 29) // k2=6 im = E2im - O2re

	// r=3, conj(W8^3)=-c+i*c: rot = -c*(a+b) + i*c*(a-b), a=O3re,b=O3im.
	VADDF_D2(14, 15, 2) // V2 = a+b
	VSUBF_D2(14, 15, 3) // V3 = a-b
	VMULF_D2(2, 30, 2)  // V2 = c*(a+b); real(rot) = -V2
	VMULF_D2(3, 30, 3)  // V3 = imag(rot)
	VSUBF_D2(12, 2, 22) // k2=3 re = E3re - c*(a+b)
	VADDF_D2(13, 3, 23) // k2=3 im = E3im + imag(rot)
	VADDF_D2(12, 2, 0)  // k2=7 re = E3re + c*(a+b)
	VSUBF_D2(13, 3, 1)  // k2=7 im = E3im - imag(rot)

	// (B) twiddle by conj(W16^(n1*k2)). k2=0 skipped.
	MOVD  $·neonOne64(SB), R0
	VLD1R (R0), [V30.D2]            // V30 = ones = [1.0, 1.0]
	VEOR  V31.B16, V31.B16, V31.B16 // V31 = zero vector

	ADD   $16, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $24, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_INV_D2(18, 19, 4, 5, 6, 7)

	ADD   $32, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $40, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_INV_D2(20, 21, 4, 5, 6, 7)

	ADD   $48, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $56, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_INV_D2(22, 23, 4, 5, 6, 7)

	ADD   $64, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $72, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_INV_D2(24, 25, 4, 5, 6, 7)

	ADD   $80, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $88, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_INV_D2(26, 27, 4, 5, 6, 7)

	ADD   $96, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $104, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_INV_D2(28, 29, 4, 5, 6, 7)

	ADD   $112, R10, R2
	VLD1R (R2), [V2.D2]
	VZIP1 V2.D2, V30.D2, V4.D2
	ADD   $120, R10, R3
	VLD1R (R3), [V2.D2]
	VZIP1 V2.D2, V31.D2, V5.D2
	VCMUL_INV_D2(0, 1, 4, 5, 6, 7)

	// (C) pair up k2 vectors and DFT2 over n1, applying the 1/16 scale on
	// the way out. Broadcast from memory — a register broadcast of a scalar
	// constant costs a fixed ~100ns and would dominate a kernel this small.
	// Use V31 (the zero vector from stage B, dead now) rather than V29,
	// which still holds live k2=6 imaginary data until the last pair below.
	MOVD  $·neonInv16F64(SB), R0
	VLD1R (R0), [V31.D2] // V31 = 1/16

	// pair (0,1) -> X[0],X[1] and X[8],X[9]
	VZIP1 V18.D2, V16.D2, V2.D2
	VZIP2 V18.D2, V16.D2, V3.D2
	VZIP1 V19.D2, V17.D2, V4.D2
	VZIP2 V19.D2, V17.D2, V5.D2
	VADDF_D2(2, 3, 6)
	VADDF_D2(4, 5, 7)
	VSUBF_D2(2, 3, 8)
	VSUBF_D2(4, 5, 9)
	VMULF_D2(6, 31, 6)
	VMULF_D2(7, 31, 7)
	VMULF_D2(8, 31, 8)
	VMULF_D2(9, 31, 9)
	VST2 [V6.D2, V7.D2], (R8)
	ADD  $128, R8, R1
	VST2 [V8.D2, V9.D2], (R1)

	// pair (2,3) -> X[2],X[3] and X[10],X[11]
	VZIP1 V22.D2, V20.D2, V2.D2
	VZIP2 V22.D2, V20.D2, V3.D2
	VZIP1 V23.D2, V21.D2, V4.D2
	VZIP2 V23.D2, V21.D2, V5.D2
	VADDF_D2(2, 3, 6)
	VADDF_D2(4, 5, 7)
	VSUBF_D2(2, 3, 8)
	VSUBF_D2(4, 5, 9)
	VMULF_D2(6, 31, 6)
	VMULF_D2(7, 31, 7)
	VMULF_D2(8, 31, 8)
	VMULF_D2(9, 31, 9)
	ADD  $32, R8, R1
	VST2 [V6.D2, V7.D2], (R1)
	ADD  $160, R8, R1
	VST2 [V8.D2, V9.D2], (R1)

	// pair (4,5) -> X[4],X[5] and X[12],X[13]
	VZIP1 V26.D2, V24.D2, V2.D2
	VZIP2 V26.D2, V24.D2, V3.D2
	VZIP1 V27.D2, V25.D2, V4.D2
	VZIP2 V27.D2, V25.D2, V5.D2
	VADDF_D2(2, 3, 6)
	VADDF_D2(4, 5, 7)
	VSUBF_D2(2, 3, 8)
	VSUBF_D2(4, 5, 9)
	VMULF_D2(6, 31, 6)
	VMULF_D2(7, 31, 7)
	VMULF_D2(8, 31, 8)
	VMULF_D2(9, 31, 9)
	ADD  $64, R8, R1
	VST2 [V6.D2, V7.D2], (R1)
	ADD  $192, R8, R1
	VST2 [V8.D2, V9.D2], (R1)

	// pair (6,7) -> X[6],X[7] and X[14],X[15]
	VZIP1 V0.D2, V28.D2, V2.D2
	VZIP2 V0.D2, V28.D2, V3.D2
	VZIP1 V1.D2, V29.D2, V4.D2
	VZIP2 V1.D2, V29.D2, V5.D2
	VADDF_D2(2, 3, 6)
	VADDF_D2(4, 5, 7)
	VSUBF_D2(2, 3, 8)
	VSUBF_D2(4, 5, 9)
	VMULF_D2(6, 31, 6)
	VMULF_D2(7, 31, 7)
	VMULF_D2(8, 31, 8)
	VMULF_D2(9, 31, 9)
	ADD  $96, R8, R1
	VST2 [V6.D2, V7.D2], (R1)
	ADD  $224, R8, R1
	VST2 [V8.D2, V9.D2], (R1)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon16r4f64_inv_return_false:
	MOVB ZR, ret+96(FP)
	RET
