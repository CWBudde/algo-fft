//go:build arm64 && !purego

// ===========================================================================
// NEON Size-8 Radix-4 FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// This replaced an earlier scalar implementation that, despite the "NEON"
// name, contained no vector instructions at all — plain FMOVD/FADDD/FMULD
// with a bit-reversal loop — and measured 3.76x (forward) / 3.69x (inverse)
// SLOWER than the pure-Go codelet on an Apple M5, the single worst cell in
// the whole NEON codelet table. See docs/CODELET_BENCHMARKS.md.
//
// Algorithm — a 2x4 Cooley-Tukey, natural order in, natural order out, no
// bit-reversal pass. A Q register holds 2 float64, so a VLD2 pair holds 2
// complex128 with re/im split; n = 8 complex128 = 128 bytes = 4 such pairs.
//
// With n = n1 + 2*n2 (n1 in {0,1} = lane, n2 in {0,1,2,3} = vector) and
// k = 4*k1 + k2 (k1 in {0,1}, k2 in {0,1,2,3}):
//
//   X[4k1+k2] = SUM_{n1} W2^(n1*k1) * W8^(n1*k2) * [ SUM_{n2} x[n1+2*n2] * W4^(n2*k2) ]
//
//   A) DFT4 over n2   — vertical across the four vector pairs, lane = n1.
//      Afterwards the vector index means k2, the lane still means n1.
//   B) twiddle by W8^(n1*k2) — lane n1=0 gets factor 1, lane n1=1 gets
//      W8^k2. k2 = 0 is all-ones and is skipped.
//   C) pair up k2 vectors two at a time and DFT2 over n1, which is the
//      *lane* index: pairing turns the horizontal DFT2 into a vertical one
//      and lands two consecutive outputs per register in the same step.
//   D) store — four VST2s of two consecutive outputs each.
//
// Every input is loaded into registers before the first store, so dst may
// alias src and no scratch buffer or copy-back is needed.
//
// Go's arm64 assembler has NO vector FADD/FSUB/FMUL — VFMLA and VFMLS are the
// only vector FP arithmetic mnemonics it accepts. Addition and subtraction are
// therefore synthesized against a vector of 1.0 and multiplication against a
// VEOR-zeroed accumulator, exactly as neon_f64_size4_radix4.s does. There is
// no float64 ·neonOnes; ·neonOne64 (core.s) is one 8-byte 1.0, broadcast to
// both lanes with VLD1R.
//
// ===========================================================================

#include "textflag.h"

// Note: neonInv8F64 is defined in neon_f64_size8_radix2.s to avoid duplicate
// symbols.

// V31 permanently holds [1.0, 1.0]; the add/sub macros depend on it.
#define ONES V31

// d = a + b, d = a - b (a, b, d are V-register names without arrangement).
#define VADDF(a, b, d) \
	VMOV  a.B16, d.B16    \
	VFMLA b.D2, ONES.D2, d.D2

#define VSUBF(a, b, d) \
	VMOV  a.B16, d.B16    \
	VFMLS b.D2, ONES.D2, d.D2

// Forward DFT4 across four vectors (W4 = -i):
//   X0 = t0+t2   X2 = t0-t2   X1 = t1 - i*t3   X3 = t1 + i*t3
// where t0 = a0+a2, t1 = a0-a2, t2 = a1+a3, t3 = a1-a3.
// Operates on real parts ar0..ar3 and imaginary parts ai0..ai3 in place,
// clobbering the eight temporaries.
#define VDFT4_FWD(ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3, t0r, t1r, t2r, t3r, t0i, t1i, t2i, t3i) \
	VADDF(ar0, ar2, t0r) \
	VSUBF(ar0, ar2, t1r) \
	VADDF(ar1, ar3, t2r) \
	VSUBF(ar1, ar3, t3r) \
	VADDF(ai0, ai2, t0i) \
	VSUBF(ai0, ai2, t1i) \
	VADDF(ai1, ai3, t2i) \
	VSUBF(ai1, ai3, t3i) \
	VADDF(t0r, t2r, ar0) \
	VADDF(t0i, t2i, ai0) \
	VSUBF(t0r, t2r, ar2) \
	VSUBF(t0i, t2i, ai2) \
	VADDF(t1r, t3i, ar1) \
	VSUBF(t1i, t3r, ai1) \
	VSUBF(t1r, t3i, ar3) \
	VADDF(t1i, t3r, ai3)

// Inverse DFT4 across four vectors (W4 = +i): X1 and X3 swap relative to fwd.
#define VDFT4_INV(ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3, t0r, t1r, t2r, t3r, t0i, t1i, t2i, t3i) \
	VADDF(ar0, ar2, t0r) \
	VSUBF(ar0, ar2, t1r) \
	VADDF(ar1, ar3, t2r) \
	VSUBF(ar1, ar3, t3r) \
	VADDF(ai0, ai2, t0i) \
	VSUBF(ai0, ai2, t1i) \
	VADDF(ai1, ai3, t2i) \
	VSUBF(ai1, ai3, t3i) \
	VADDF(t0r, t2r, ar0) \
	VADDF(t0i, t2i, ai0) \
	VSUBF(t0r, t2r, ar2) \
	VSUBF(t0i, t2i, ai2) \
	VSUBF(t1r, t3i, ar1) \
	VADDF(t1i, t3r, ai1) \
	VADDF(t1r, t3i, ar3) \
	VSUBF(t1i, t3r, ai3)

// dr,di *= (wr + i*wi), result back in dr,di; clobbers p, q.
#define VCMUL_FWD(dr, di, wr, wi, p, q) \
	VEOR  p.B16, p.B16, p.B16 \
	VFMLA dr.D2, wr.D2, p.D2  \
	VFMLS di.D2, wi.D2, p.D2  \
	VEOR  q.B16, q.B16, q.B16 \
	VFMLA dr.D2, wi.D2, q.D2  \
	VFMLA di.D2, wr.D2, q.D2  \
	VMOV  p.B16, dr.B16       \
	VMOV  q.B16, di.B16

// dr,di *= conj(wr + i*wi) — the inverse twiddle.
#define VCMUL_INV(dr, di, wr, wi, p, q) \
	VEOR  p.B16, p.B16, p.B16 \
	VFMLA dr.D2, wr.D2, p.D2  \
	VFMLA di.D2, wi.D2, p.D2  \
	VEOR  q.B16, q.B16, q.B16 \
	VFMLA di.D2, wr.D2, q.D2  \
	VFMLS dr.D2, wi.D2, q.D2  \
	VMOV  p.B16, dr.B16       \
	VMOV  q.B16, di.B16

// ---------------------------------------------------------------------------
// func ForwardNEONSize8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool
// ---------------------------------------------------------------------------
TEXT ·ForwardNEONSize8Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $8, R13
	BNE  neon8r4f64_return_false

	MOVD dst_len+8(FP), R0
	CMP  $8, R0
	BLT  neon8r4f64_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $8, R0
	BLT  neon8r4f64_return_false

	MOVD  $·neonOne64(SB), R0
	VLD1R (R0), [ONES.D2]
	VEOR  V30.B16, V30.B16, V30.B16 // V30 = [0.0, 0.0], used to build wi

	// Load x[n1 + 2*n2]: vector n2, lane n1. VLD2 deinterleaves re/im, and its
	// register list must be contiguous, so re/im pairs are adjacent:
	// re[n2] = V0,V2,V4,V6   im[n2] = V1,V3,V5,V7
	VLD2 (R9), [V0.D2, V1.D2]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.D2, V3.D2]
	ADD  $64, R9, R1
	VLD2 (R1), [V4.D2, V5.D2]
	ADD  $96, R9, R1
	VLD2 (R1), [V6.D2, V7.D2]

	// (A) DFT4 over n2. Vector index becomes k2.
	VDFT4_FWD(V0, V2, V4, V6, V1, V3, V5, V7, V8, V9, V10, V11, V12, V13, V14, V15)

	// (B) twiddle by W8^(n1*k2). k2 = 0 is all-ones, so it is skipped.
	ADD  $16, R10, R2
	VLD1R (R2), [V16.D2]     // t = Re(W8^1)
	VZIP1 V16.D2, ONES.D2, V17.D2 // wr = [1.0, Re(W8^1)]
	ADD  $8, R2, R3
	VLD1R (R3), [V16.D2]     // t = Im(W8^1)
	VZIP1 V16.D2, V30.D2, V18.D2  // wi = [0.0, Im(W8^1)]
	VCMUL_FWD(V2, V3, V17, V18, V8, V9)

	ADD  $32, R10, R2
	VLD1R (R2), [V16.D2]     // t = Re(W8^2)
	VZIP1 V16.D2, ONES.D2, V17.D2 // wr = [1.0, Re(W8^2)]
	ADD  $8, R2, R3
	VLD1R (R3), [V16.D2]     // t = Im(W8^2)
	VZIP1 V16.D2, V30.D2, V18.D2  // wi = [0.0, Im(W8^2)]
	VCMUL_FWD(V4, V5, V17, V18, V8, V9)

	ADD  $48, R10, R2
	VLD1R (R2), [V16.D2]     // t = Re(W8^3)
	VZIP1 V16.D2, ONES.D2, V17.D2 // wr = [1.0, Re(W8^3)]
	ADD  $8, R2, R3
	VLD1R (R3), [V16.D2]     // t = Im(W8^3)
	VZIP1 V16.D2, V30.D2, V18.D2  // wi = [0.0, Im(W8^3)]
	VCMUL_FWD(V6, V7, V17, V18, V8, V9)

	// (C) pair k2=0,1 -> DFT2 over n1 (the lane index). p = n1=0, q = n1=1.
	VZIP1 V2.D2, V0.D2, V20.D2 // p_re = [A0(n1=0), A1(n1=0)]
	VZIP2 V2.D2, V0.D2, V21.D2 // q_re = [A0(n1=1), A1(n1=1)]
	VZIP1 V3.D2, V1.D2, V22.D2 // p_im
	VZIP2 V3.D2, V1.D2, V23.D2 // q_im
	VADDF(V20, V21, V24)       // X0,X1 (k1=0) real
	VADDF(V22, V23, V25)       // X0,X1 imag
	VSUBF(V20, V21, V26)       // X4,X5 (k1=1) real
	VSUBF(V22, V23, V27)       // X4,X5 imag

	VST2 [V24.D2, V25.D2], (R8)
	ADD  $64, R8, R1
	VST2 [V26.D2, V27.D2], (R1)

	// pair k2=2,3 -> DFT2 over n1, giving X2,X3 (k1=0) and X6,X7 (k1=1).
	VZIP1 V6.D2, V4.D2, V20.D2
	VZIP2 V6.D2, V4.D2, V21.D2
	VZIP1 V7.D2, V5.D2, V22.D2
	VZIP2 V7.D2, V5.D2, V23.D2
	VADDF(V20, V21, V24)
	VADDF(V22, V23, V25)
	VSUBF(V20, V21, V26)
	VSUBF(V22, V23, V27)

	ADD  $32, R8, R1
	VST2 [V24.D2, V25.D2], (R1)
	ADD  $96, R8, R1
	VST2 [V26.D2, V27.D2], (R1)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r4f64_return_false:
	MOVB ZR, ret+96(FP)
	RET

// ---------------------------------------------------------------------------
// func InverseNEONSize8Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool
// ---------------------------------------------------------------------------
TEXT ·InverseNEONSize8Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $8, R13
	BNE  neon8r4f64_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $8, R0
	BLT  neon8r4f64_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $8, R0
	BLT  neon8r4f64_inv_return_false

	MOVD  $·neonOne64(SB), R0
	VLD1R (R0), [ONES.D2]
	VEOR  V30.B16, V30.B16, V30.B16 // V30 = [0.0, 0.0], used to build wi

	// Load x[n1 + 2*n2]: vector n2, lane n1.
	VLD2 (R9), [V0.D2, V1.D2]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.D2, V3.D2]
	ADD  $64, R9, R1
	VLD2 (R1), [V4.D2, V5.D2]
	ADD  $96, R9, R1
	VLD2 (R1), [V6.D2, V7.D2]

	// (A) DFT4 over n2. Vector index becomes k2.
	VDFT4_INV(V0, V2, V4, V6, V1, V3, V5, V7, V8, V9, V10, V11, V12, V13, V14, V15)

	// (B) twiddle by W8^(n1*k2), conjugated by VCMUL_INV. k2 = 0 skipped.
	ADD  $16, R10, R2
	VLD1R (R2), [V16.D2]
	VZIP1 V16.D2, ONES.D2, V17.D2
	ADD  $8, R2, R3
	VLD1R (R3), [V16.D2]
	VZIP1 V16.D2, V30.D2, V18.D2
	VCMUL_INV(V2, V3, V17, V18, V8, V9)

	ADD  $32, R10, R2
	VLD1R (R2), [V16.D2]
	VZIP1 V16.D2, ONES.D2, V17.D2
	ADD  $8, R2, R3
	VLD1R (R3), [V16.D2]
	VZIP1 V16.D2, V30.D2, V18.D2
	VCMUL_INV(V4, V5, V17, V18, V8, V9)

	ADD  $48, R10, R2
	VLD1R (R2), [V16.D2]
	VZIP1 V16.D2, ONES.D2, V17.D2
	ADD  $8, R2, R3
	VLD1R (R3), [V16.D2]
	VZIP1 V16.D2, V30.D2, V18.D2
	VCMUL_INV(V6, V7, V17, V18, V8, V9)

	// (C) pair k2=0,1 -> DFT2 over n1.
	VZIP1 V2.D2, V0.D2, V20.D2
	VZIP2 V2.D2, V0.D2, V21.D2
	VZIP1 V3.D2, V1.D2, V22.D2
	VZIP2 V3.D2, V1.D2, V23.D2
	VADDF(V20, V21, V24) // X0,X1 real
	VADDF(V22, V23, V25) // X0,X1 imag
	VSUBF(V20, V21, V26) // X4,X5 real
	VSUBF(V22, V23, V27) // X4,X5 imag

	// pair k2=2,3 -> DFT2 over n1.
	VZIP1 V6.D2, V4.D2, V0.D2
	VZIP2 V6.D2, V4.D2, V1.D2
	VZIP1 V7.D2, V5.D2, V2.D2
	VZIP2 V7.D2, V5.D2, V3.D2
	VADDF(V0, V1, V4) // X2,X3 real
	VADDF(V2, V3, V5) // X2,X3 imag
	VSUBF(V0, V1, V6) // X6,X7 real
	VSUBF(V2, V3, V7) // X6,X7 imag

	// Scale by 1/8. Broadcast from memory — a register broadcast of a scalar
	// constant costs a fixed ~100ns and would dominate a kernel this small.
	MOVD  $·neonInv8F64(SB), R0
	VLD1R (R0), [V29.D2]

	VEOR  V8.B16, V8.B16, V8.B16
	VFMLA V24.D2, V29.D2, V8.D2
	VEOR  V9.B16, V9.B16, V9.B16
	VFMLA V25.D2, V29.D2, V9.D2
	VST2  [V8.D2, V9.D2], (R8) // X0,X1

	VEOR  V10.B16, V10.B16, V10.B16
	VFMLA V4.D2, V29.D2, V10.D2
	VEOR  V11.B16, V11.B16, V11.B16
	VFMLA V5.D2, V29.D2, V11.D2
	ADD   $32, R8, R1
	VST2  [V10.D2, V11.D2], (R1) // X2,X3

	VEOR  V12.B16, V12.B16, V12.B16
	VFMLA V26.D2, V29.D2, V12.D2
	VEOR  V13.B16, V13.B16, V13.B16
	VFMLA V27.D2, V29.D2, V13.D2
	ADD   $64, R8, R1
	VST2  [V12.D2, V13.D2], (R1) // X4,X5

	VEOR  V14.B16, V14.B16, V14.B16
	VFMLA V6.D2, V29.D2, V14.D2
	VEOR  V15.B16, V15.B16, V15.B16
	VFMLA V7.D2, V29.D2, V15.D2
	ADD   $96, R8, R1
	VST2  [V14.D2, V15.D2], (R1) // X6,X7

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r4f64_inv_return_false:
	MOVB ZR, ret+96(FP)
	RET
