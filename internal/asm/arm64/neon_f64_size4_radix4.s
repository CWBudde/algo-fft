//go:build arm64 && !purego

// ===========================================================================
// NEON Size-4 Radix-4 FFT Kernels for ARM64 (complex128)
// ===========================================================================
//
// This replaced an earlier scalar implementation that, despite the "NEON"
// name, contained no vector instructions at all — plain FMOVD/FADDD/FSUBD
// with a copy-back loop. This version is the float64 analogue of
// neon_f32_size16_radix4.s: a 2x2 Cooley-Tukey using .D2 vector arrangements
// (a Q register holds 2 float64, i.e. 1 complex128 pair per re/im lane).
//
// Algorithm — 2x2 Cooley-Tukey, natural order in, natural order out. With
// n = n1 + 2*n2 and k = 2*k1 + k2:
//
//   X[2k1+k2] = sum_{n1} W2^{n1 k1} W4^{n1 k2} sum_{n2} x[n1+2n2] W2^{n2 k2}
//
//   A) DFT2 over n2  — vertical across the two vector pairs, lane = n1
//   B) multiply by W4^{n1 k2}: k2=0 is all-ones (skipped); k2=1 is
//      [W4^0, W4^1] = [1, -i], which is exactly twiddle[0], twiddle[1].
//   C) 2x2 transpose — lane becomes k2, vector becomes n1
//   D) DFT2 over n1  — vertical again, vector = k1, lane stays k2
//
// Step D leaves vector k1 holding X[2k1+0..1], i.e. two *consecutive*
// outputs, so each store is a single VST2. Every input is loaded into
// registers before the first store, so dst may alias src and no scratch
// buffer or copy-back is needed.
//
// Go's arm64 assembler has NO vector FADD/FSUB/FMUL — VFMLA and VFMLS are the
// only vector FP arithmetic mnemonics it accepts. Addition and subtraction are
// therefore synthesized against a vector of 1.0 and multiplication against a
// VEOR-zeroed accumulator, exactly as neon_f32_size16_radix4.s does. There is
// no float64 ·neonOnes; ·neonOne64 (core.s) is one 8-byte 1.0, broadcast to
// both lanes with VLD1R.
//
// ===========================================================================

#include "textflag.h"

// V31 permanently holds [1.0, 1.0]; the add/sub macros depend on it.
#define ONES V31

// d = a + b, d = a - b (a, b, d are V-register names without arrangement).
#define VADDF(a, b, d) \
	VMOV  a.B16, d.B16    \
	VFMLA b.D2, ONES.D2, d.D2

#define VSUBF(a, b, d) \
	VMOV  a.B16, d.B16    \
	VFMLS b.D2, ONES.D2, d.D2

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
// func ForwardNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool
// ---------------------------------------------------------------------------
TEXT ·ForwardNEONSize4Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $4, R13
	BNE  neon4r4f64_return_false

	MOVD dst_len+8(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_return_false

	MOVD $·neonOne64(SB), R0
	VLD1R (R0), [ONES.D2]

	// Load x[n1 + 2*n2]: vector n2, lane n1. VLD2 deinterleaves re/im, and its
	// register list must be contiguous, so re/im pairs are adjacent:
	// re[n2] = V0,V2   im[n2] = V1,V3
	VLD2 (R9), [V0.D2, V1.D2]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.D2, V3.D2]

	// (A) DFT2 over n2. Vector index becomes k2.
	VADDF(V0, V2, V4)
	VADDF(V1, V3, V5)
	VSUBF(V0, V2, V6)
	VSUBF(V1, V3, V7)

	// (B) twiddle by W4^(n1*k2). k2 = 0 is all-ones, so it is skipped.
	// k2 = 1: [tw0, tw1] = [1, -i] — loads for free from the twiddle table.
	VLD2 (R10), [V8.D2, V9.D2]
	VCMUL_FWD(V6, V7, V8, V9, V10, V11)

	// (C) 2x2 transpose: lane becomes k2, vector becomes n1.
	VZIP1 V6.D2, V4.D2, V12.D2 // n1=0 re: [u0r.D0, u1r.D0]
	VZIP2 V6.D2, V4.D2, V13.D2 // n1=1 re: [u0r.D1, u1r.D1]
	VZIP1 V7.D2, V5.D2, V14.D2 // n1=0 im
	VZIP2 V7.D2, V5.D2, V15.D2 // n1=1 im

	// (D) DFT2 over n1. Vector index becomes k1, lane stays k2.
	VADDF(V12, V13, V16) // X0, X1
	VADDF(V14, V15, V17)
	VSUBF(V12, V13, V18) // X2, X3
	VSUBF(V14, V15, V19)

	// Vector k1 holds X[2k1 + 0..1] — two consecutive outputs.
	VST2 [V16.D2, V17.D2], (R8)
	ADD  $32, R8, R1
	VST2 [V18.D2, V19.D2], (R1)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon4r4f64_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ---------------------------------------------------------------------------
// func InverseNEONSize4Radix4Complex128Asm(dst, src, twiddle, scratch []complex128) bool
// ---------------------------------------------------------------------------
TEXT ·InverseNEONSize4Radix4Complex128Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $4, R13
	BNE  neon4r4f64_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $4, R0
	BLT  neon4r4f64_inv_return_false

	MOVD $·neonOne64(SB), R0
	VLD1R (R0), [ONES.D2]

	// Load x[n1 + 2*n2]: vector n2, lane n1.
	VLD2 (R9), [V0.D2, V1.D2]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.D2, V3.D2]

	// (A) DFT2 over n2. Vector index becomes k2.
	VADDF(V0, V2, V4)
	VADDF(V1, V3, V5)
	VSUBF(V0, V2, V6)
	VSUBF(V1, V3, V7)

	// (B) twiddle by W4^(n1*k2), conjugated. k2 = 0 is all-ones, so it is
	// skipped. k2 = 1 uses the same loaded [1, -i] with VCMUL_INV, which
	// conjugates it to [1, +i].
	VLD2 (R10), [V8.D2, V9.D2]
	VCMUL_INV(V6, V7, V8, V9, V10, V11)

	// (C) 2x2 transpose: lane becomes k2, vector becomes n1.
	VZIP1 V6.D2, V4.D2, V12.D2
	VZIP2 V6.D2, V4.D2, V13.D2
	VZIP1 V7.D2, V5.D2, V14.D2
	VZIP2 V7.D2, V5.D2, V15.D2

	// (D) DFT2 over n1. Vector index becomes k1, lane stays k2.
	VADDF(V12, V13, V16)
	VADDF(V14, V15, V17)
	VSUBF(V12, V13, V18)
	VSUBF(V14, V15, V19)

	// Scale by 1/4. Broadcast from memory — a register broadcast of a scalar
	// constant costs a fixed ~100ns and would dominate a kernel this small.
	MOVD $·neonInv4F64(SB), R0
	VLD1R (R0), [V20.D2]

	VEOR V21.B16, V21.B16, V21.B16
	VFMLA V16.D2, V20.D2, V21.D2
	VEOR V22.B16, V22.B16, V22.B16
	VFMLA V17.D2, V20.D2, V22.D2
	VST2 [V21.D2, V22.D2], (R8)

	VEOR V23.B16, V23.B16, V23.B16
	VFMLA V18.D2, V20.D2, V23.D2
	VEOR V24.B16, V24.B16, V24.B16
	VFMLA V19.D2, V20.D2, V24.D2
	ADD  $32, R8, R1
	VST2 [V23.D2, V24.D2], (R1)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon4r4f64_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// 1/4, the inverse normalization.
DATA ·neonInv4F64+0(SB)/8, $0x3fd0000000000000 // 1/4
GLOBL ·neonInv4F64(SB), RODATA, $8
