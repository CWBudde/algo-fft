//go:build arm64 && !purego

// ===========================================================================
// NEON Size-8 Radix-8 FFT Kernels for ARM64
// ===========================================================================
//
// This replaced an earlier scalar implementation that, despite the "NEON"
// name, contained no vector instructions at all — plain FMOVS/FADDS/FMULS
// over 389 lines. See docs/CODELET_BENCHMARKS.md.
//
// Algorithm — a 4x2 Cooley-Tukey with a mid-kernel re-lane. n = 8 complex64
// is 2 vector pairs of 4 lanes each (VLD2 deinterleaves re/im, 4 complex64
// per S4 pair). Element index n = n1 + 4*n2 (n1 = lane in 0..3, n2 = vector
// in 0..1); output index k = 2*k1 + k2 (k1 in 0..3, k2 in 0..1). Then
//
//   X[2k1+k2] = sum_{n1} W4^{n1 k1} W8^{n1 k2} sum_{n2} x[n1+4n2] W2^{n2 k2}
//
// (A) DFT2 over n2       — vertical, two vector pairs in, two out (k2 = 0,1)
// (B) twiddle k2=1 branch by W8^n1 — n1 is the lane, so the factor is the
//     first four entries of the twiddle table, a single free load
// (C) 2x4 -> 4x2 re-lane: interleave the k2=0/1 branches so lane 0/1 become
//     k2 = 0/1 and the vector index becomes n1 (lanes 2/3 hold discarded
//     duplicates)
// (D) DFT4 over n1        — vertical again, vector = k1, lanes (0,1) = k2
//
// Step D leaves vector k1 holding X[2k1+0..1] in its low 64 bits, so pairs
// of vectors (s0,s1) and (s2,s3) collect into four consecutive outputs via a
// D2 zip, and the two stores are VST2.
//
// Go's arm64 assembler has NO vector FADD/FSUB/FMUL — VFMLA and VFMLS are the
// only vector FP arithmetic mnemonics it accepts. Addition and subtraction are
// therefore synthesized against a vector of 1.0 (·neonOnes, core.s) and
// multiplication against a VEOR-zeroed accumulator, exactly as
// neon_f32_size16_radix4.s does.
//
// dst may alias src: every input is loaded into registers before the first
// store, so no scratch buffer or copy-back is needed.
//
// ===========================================================================

#include "textflag.h"

// V31 permanently holds [1.0, 1.0, 1.0, 1.0]; the add/sub macros depend on it.
#define ONES V31

// d = a + b, d = a - b (a, b, d are V-register names without arrangement).
#define VADDF(a, b, d) \
	VMOV  a.B16, d.B16    \
	VFMLA b.S4, ONES.S4, d.S4

#define VSUBF(a, b, d) \
	VMOV  a.B16, d.B16    \
	VFMLS b.S4, ONES.S4, d.S4

// Forward DFT4 across four vectors (W4 = -i):
//   X0 = t0+t2   X2 = t0-t2   X1 = t1 - i*t3   X3 = t1 + i*t3
// where t0 = a0+a2, t1 = a0-a2, t2 = a1+a3, t3 = a1-a3.
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
	VFMLA dr.S4, wr.S4, p.S4  \
	VFMLS di.S4, wi.S4, p.S4  \
	VEOR  q.B16, q.B16, q.B16 \
	VFMLA dr.S4, wi.S4, q.S4  \
	VFMLA di.S4, wr.S4, q.S4  \
	VMOV  p.B16, dr.B16       \
	VMOV  q.B16, di.B16

// dr,di *= conj(wr + i*wi) — the inverse twiddle.
#define VCMUL_INV(dr, di, wr, wi, p, q) \
	VEOR  p.B16, p.B16, p.B16 \
	VFMLA dr.S4, wr.S4, p.S4  \
	VFMLA di.S4, wi.S4, p.S4  \
	VEOR  q.B16, q.B16, q.B16 \
	VFMLA di.S4, wr.S4, q.S4  \
	VFMLS dr.S4, wi.S4, q.S4  \
	VMOV  p.B16, dr.B16       \
	VMOV  q.B16, di.B16

// (A) DFT2 over n2 + (B) twiddle + (C) re-lane, shared by forward and inverse
// (the twiddle multiply macro CMUL is passed in). Inputs V0/V1 = re/im of
// x[0..3], V2/V3 = re/im of x[4..7]; twiddle base in R10. Outputs r0..r3
// (real) and their imaginary twins in the *i registers named below, ready for
// VDFT4_*. Clobbers V4..V9, V16..V19.
#define VSTAGE_AB(CMUL) \
	VADDF(V0, V2, V4)                  \
	VADDF(V1, V3, V5)                  \
	VSUBF(V0, V2, V6)                  \
	VSUBF(V1, V3, V7)                  \
	VLD2  (R10), [V16.S4, V17.S4]      \
	CMUL(V6, V7, V16, V17, V8, V9)     \
	VZIP1 V6.S4, V4.S4, V18.S4         \
	VZIP2 V6.S4, V4.S4, V19.S4         \
	VMOV  V18.B16, V0.B16              \
	VEXT  $8, V18.B16, V18.B16, V1.B16 \
	VMOV  V19.B16, V2.B16              \
	VEXT  $8, V19.B16, V19.B16, V3.B16 \
	VZIP1 V7.S4, V5.S4, V18.S4         \
	VZIP2 V7.S4, V5.S4, V19.S4         \
	VMOV  V18.B16, V20.B16             \
	VEXT  $8, V18.B16, V18.B16, V21.B16 \
	VMOV  V19.B16, V22.B16             \
	VEXT  $8, V19.B16, V19.B16, V23.B16

// ---------------------------------------------------------------------------
// func ForwardNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool
// ---------------------------------------------------------------------------
TEXT ·ForwardNEONSize8Radix8Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $8, R13
	BNE  neon8r8_return_false

	MOVD dst_len+8(FP), R0
	CMP  $8, R0
	BLT  neon8r8_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $8, R0
	BLT  neon8r8_return_false

	MOVD $·neonOnes(SB), R0
	VLD1 (R0), [ONES.S4]

	// Load x[n1 + 4*n2]: vector n2, lane n1.
	VLD2 (R9), [V0.S4, V1.S4]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.S4, V3.S4]

	// (A)+(B)+(C): produces r0=V0,r1=V1,r2=V2,r3=V3 (real) and
	// r0i=V20,r1i=V21,r2i=V22,r3i=V23 (imag), lanes(0,1) = k2, vector = n1.
	VSTAGE_AB(VCMUL_FWD)

	// (D) DFT4 over n1. Vector index becomes k1, lanes(0,1) stay k2.
	VDFT4_FWD(V0, V1, V2, V3, V20, V21, V22, V23, V4, V5, V6, V7, V8, V9, V16, V17)

	// Vector k1 holds [X[2k1+0], X[2k1+1]] in its low 64 bits (D0). Pairs of
	// consecutive k1 collect into four consecutive outputs via a D2 zip.
	VZIP1 V1.D2, V0.D2, V24.D2
	VZIP1 V21.D2, V20.D2, V25.D2
	VST2  [V24.S4, V25.S4], (R8)

	VZIP1 V3.D2, V2.D2, V26.D2
	VZIP1 V23.D2, V22.D2, V27.D2
	ADD   $32, R8, R1
	VST2  [V26.S4, V27.S4], (R1)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r8_return_false:
	MOVB ZR, ret+96(FP)
	RET

// ---------------------------------------------------------------------------
// func InverseNEONSize8Radix8Complex64Asm(dst, src, twiddle, scratch []complex64) bool
// ---------------------------------------------------------------------------
TEXT ·InverseNEONSize8Radix8Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $8, R13
	BNE  neon8r8_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $8, R0
	BLT  neon8r8_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $8, R0
	BLT  neon8r8_inv_return_false

	MOVD $·neonOnes(SB), R0
	VLD1 (R0), [ONES.S4]

	// Load x[n1 + 4*n2]: vector n2, lane n1.
	VLD2 (R9), [V0.S4, V1.S4]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.S4, V3.S4]

	// (A)+(B)+(C): produces r0=V0,r1=V1,r2=V2,r3=V3 (real) and
	// r0i=V20,r1i=V21,r2i=V22,r3i=V23 (imag), lanes(0,1) = k2, vector = n1.
	VSTAGE_AB(VCMUL_INV)

	// (D) DFT4 over n1. Vector index becomes k1, lanes(0,1) stay k2.
	VDFT4_INV(V0, V1, V2, V3, V20, V21, V22, V23, V4, V5, V6, V7, V8, V9, V16, V17)

	// Scale by 1/8. Broadcast from memory — a register broadcast of a scalar
	// constant costs a fixed ~100ns and would dominate a kernel this small.
	MOVD  $·neonInv8(SB), R0
	VLD1R (R0), [V28.S4]

	VEOR  V24.B16, V24.B16, V24.B16
	VFMLA V0.S4, V28.S4, V24.S4
	VEOR  V25.B16, V25.B16, V25.B16
	VFMLA V20.S4, V28.S4, V25.S4
	VEOR  V26.B16, V26.B16, V26.B16
	VFMLA V1.S4, V28.S4, V26.S4
	VEOR  V27.B16, V27.B16, V27.B16
	VFMLA V21.S4, V28.S4, V27.S4

	VZIP1 V26.D2, V24.D2, V4.D2
	VZIP1 V27.D2, V25.D2, V5.D2
	VST2  [V4.S4, V5.S4], (R8)

	VEOR  V24.B16, V24.B16, V24.B16
	VFMLA V2.S4, V28.S4, V24.S4
	VEOR  V25.B16, V25.B16, V25.B16
	VFMLA V22.S4, V28.S4, V25.S4
	VEOR  V26.B16, V26.B16, V26.B16
	VFMLA V3.S4, V28.S4, V26.S4
	VEOR  V27.B16, V27.B16, V27.B16
	VFMLA V23.S4, V28.S4, V27.S4

	VZIP1 V26.D2, V24.D2, V6.D2
	VZIP1 V27.D2, V25.D2, V7.D2
	ADD   $32, R8, R1
	VST2  [V6.S4, V7.S4], (R1)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon8r8_inv_return_false:
	MOVB ZR, ret+96(FP)
	RET
