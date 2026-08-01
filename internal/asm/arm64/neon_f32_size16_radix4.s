//go:build arm64 && !purego

// ===========================================================================
// NEON Size-16 Radix-4 FFT Kernels for ARM64
// ===========================================================================
//
// This replaced an earlier scalar implementation that, despite the "NEON"
// name, contained no vector instructions at all — plain FMOVS/FADDS/FMULS
// with a bit-reversal loop — and measured 1.83x (forward) / 2.19x (inverse)
// SLOWER than the pure-Go codelet on an Apple M5. This version is 3.76x /
// 3.69x faster than that one and 1.71x / 1.70x faster than the best pure-Go
// candidate. See docs/CODELET_BENCHMARKS.md.
//
// Algorithm — 4x4 Cooley-Tukey, natural order in, natural order out, no
// bit-reversal pass. With n = n1 + 4*n2 and k = 4*k1 + k2:
//
//   X[4k1+k2] = sum_{n1} W4^{n1 k1} W16^{n1 k2} sum_{n2} x[n1+4n2] W4^{n2 k2}
//
//   A) 4 DFT4s over n2      — vertical across 4 vectors, lane = n1
//   B) multiply by W16^{n1 k2}
//   C) 4x4 transpose        — lane becomes k2, vector becomes n1
//   D) 4 DFT4s over n1      — vertical again, vector = k1, lane = k2
//
// Step D leaves vector k1 holding X[4k1+0..3], i.e. four *consecutive*
// outputs, so each store is a single VST2. Every input is loaded into
// registers before the first store, so dst may alias src and no scratch
// buffer or copy-back is needed.
//
// Go's arm64 assembler has NO vector FADD/FSUB/FMUL — VFMLA and VFMLS are the
// only vector FP arithmetic mnemonics it accepts. Addition and subtraction are
// therefore synthesized against a vector of 1.0 (·neonOnes) and multiplication
// against a VEOR-zeroed accumulator, exactly as neon_f32_generic.s does. On
// Apple cores FMA has the same throughput as FADD, so this costs nothing.
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

// 4x4 float32 transpose of rows r0..r3 into o0..o3, clobbering t0..t3.
#define VTRANSPOSE4(r0, r1, r2, r3, t0, t1, t2, t3, o0, o1, o2, o3) \
	VTRN1 r1.S4, r0.S4, t0.S4 \
	VTRN2 r1.S4, r0.S4, t1.S4 \
	VTRN1 r3.S4, r2.S4, t2.S4 \
	VTRN2 r3.S4, r2.S4, t3.S4 \
	VZIP1 t2.D2, t0.D2, o0.D2 \
	VZIP1 t3.D2, t1.D2, o1.D2 \
	VZIP2 t2.D2, t0.D2, o2.D2 \
	VZIP2 t3.D2, t1.D2, o3.D2

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

// Gather the twiddle vector for column k2 = 3: [tw0, tw3, tw6, tw9].
// Stride 3 has no shuffle form, so it is a scalar gather into lanes.
#define VGATHER_TW3(base, wr, wi) \
	MOVW 0(base), R1    \
	MOVW 4(base), R2    \
	VMOV R1, wr.S[0]    \
	VMOV R2, wi.S[0]    \
	MOVW 24(base), R1   \
	MOVW 28(base), R2   \
	VMOV R1, wr.S[1]    \
	VMOV R2, wi.S[1]    \
	MOVW 48(base), R1   \
	MOVW 52(base), R2   \
	VMOV R1, wr.S[2]    \
	VMOV R2, wi.S[2]    \
	MOVW 72(base), R1   \
	MOVW 76(base), R2   \
	VMOV R1, wr.S[3]    \
	VMOV R2, wi.S[3]

// ---------------------------------------------------------------------------
// func ForwardNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool
// ---------------------------------------------------------------------------
TEXT ·ForwardNEONSize16Radix4Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $16, R13
	BNE  neon16r4_false

	MOVD dst_len+8(FP), R0
	CMP  $16, R0
	BLT  neon16r4_false

	MOVD twiddle_len+56(FP), R0
	CMP  $16, R0
	BLT  neon16r4_false

	MOVD $·neonOnes(SB), R0
	VLD1 (R0), [ONES.S4]

	// Load x[n1 + 4*n2]: vector n2, lane n1. VLD2 deinterleaves re/im, and its
	// register list must be contiguous, so re/im pairs are adjacent:
	// re[n2] = V0,V2,V4,V6   im[n2] = V1,V3,V5,V7
	VLD2 (R9), [V0.S4, V1.S4]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.S4, V3.S4]
	ADD  $64, R9, R1
	VLD2 (R1), [V4.S4, V5.S4]
	ADD  $96, R9, R1
	VLD2 (R1), [V6.S4, V7.S4]

	// (A) DFT4 over n2. Vector index becomes k2.
	VDFT4_FWD(V0, V2, V4, V6, V1, V3, V5, V7, V8, V9, V10, V11, V12, V13, V14, V15)

	// (B) twiddle by W16^(n1*k2). k2 = 0 is all-ones, so it is skipped.
	// k2 = 1: [tw0, tw1, tw2, tw3] — four consecutive entries.
	VLD2 (R10), [V16.S4, V17.S4]
	VCMUL_FWD(V2, V3, V16, V17, V8, V9)

	// k2 = 2: [tw0, tw2, tw4, tw6] — the even entries of tw[0..7].
	VLD2 (R10), [V18.S4, V19.S4]
	ADD  $32, R10, R1
	VLD2 (R1), [V20.S4, V21.S4]
	VUZP1 V20.S4, V18.S4, V22.S4
	VUZP1 V21.S4, V19.S4, V23.S4
	VCMUL_FWD(V4, V5, V22, V23, V8, V9)

	// k2 = 3: [tw0, tw3, tw6, tw9].
	VGATHER_TW3(R10, V25, V26)
	VCMUL_FWD(V6, V7, V25, V26, V8, V9)

	// (C) transpose: lane becomes k2, vector becomes n1. Outputs are laid out
	// as adjacent re/im pairs again so the stores can be VST2.
	VTRANSPOSE4(V0, V2, V4, V6, V8, V9, V10, V11, V16, V18, V20, V22)
	VTRANSPOSE4(V1, V3, V5, V7, V8, V9, V10, V11, V17, V19, V21, V23)

	// (D) DFT4 over n1. Vector index becomes k1, lane stays k2.
	VDFT4_FWD(V16, V18, V20, V22, V17, V19, V21, V23, V8, V9, V10, V11, V12, V13, V14, V15)

	// Vector k1 holds X[4k1 + 0..3] — four consecutive outputs.
	VST2 [V16.S4, V17.S4], (R8)
	ADD  $32, R8, R1
	VST2 [V18.S4, V19.S4], (R1)
	ADD  $64, R8, R1
	VST2 [V20.S4, V21.S4], (R1)
	ADD  $96, R8, R1
	VST2 [V22.S4, V23.S4], (R1)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon16r4_false:
	MOVB ZR, ret+96(FP)
	RET

// ---------------------------------------------------------------------------
// func InverseNEONSize16Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool
// ---------------------------------------------------------------------------
TEXT ·InverseNEONSize16Radix4Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD twiddle+48(FP), R10
	MOVD src_len+32(FP), R13

	CMP  $16, R13
	BNE  neon16r4_inv_false

	MOVD dst_len+8(FP), R0
	CMP  $16, R0
	BLT  neon16r4_inv_false

	MOVD twiddle_len+56(FP), R0
	CMP  $16, R0
	BLT  neon16r4_inv_false

	MOVD $·neonOnes(SB), R0
	VLD1 (R0), [ONES.S4]

	// Load x[n1 + 4*n2]: vector n2, lane n1. VLD2 deinterleaves re/im, and its
	// register list must be contiguous, so re/im pairs are adjacent:
	// re[n2] = V0,V2,V4,V6   im[n2] = V1,V3,V5,V7
	VLD2 (R9), [V0.S4, V1.S4]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.S4, V3.S4]
	ADD  $64, R9, R1
	VLD2 (R1), [V4.S4, V5.S4]
	ADD  $96, R9, R1
	VLD2 (R1), [V6.S4, V7.S4]

	// (A) DFT4 over n2. Vector index becomes k2.
	VDFT4_INV(V0, V2, V4, V6, V1, V3, V5, V7, V8, V9, V10, V11, V12, V13, V14, V15)

	// (B) twiddle by W16^(n1*k2). k2 = 0 is all-ones, so it is skipped.
	// k2 = 1: [tw0, tw1, tw2, tw3] — four consecutive entries.
	VLD2 (R10), [V16.S4, V17.S4]
	VCMUL_INV(V2, V3, V16, V17, V8, V9)

	// k2 = 2: [tw0, tw2, tw4, tw6] — the even entries of tw[0..7].
	VLD2 (R10), [V18.S4, V19.S4]
	ADD  $32, R10, R1
	VLD2 (R1), [V20.S4, V21.S4]
	VUZP1 V20.S4, V18.S4, V22.S4
	VUZP1 V21.S4, V19.S4, V23.S4
	VCMUL_INV(V4, V5, V22, V23, V8, V9)

	// k2 = 3: [tw0, tw3, tw6, tw9].
	VGATHER_TW3(R10, V25, V26)
	VCMUL_INV(V6, V7, V25, V26, V8, V9)

	// (C) transpose: lane becomes k2, vector becomes n1. Outputs are laid out
	// as adjacent re/im pairs again so the stores can be VST2.
	VTRANSPOSE4(V0, V2, V4, V6, V8, V9, V10, V11, V16, V18, V20, V22)
	VTRANSPOSE4(V1, V3, V5, V7, V8, V9, V10, V11, V17, V19, V21, V23)

	// (D) DFT4 over n1. Vector index becomes k1, lane stays k2.
	VDFT4_INV(V16, V18, V20, V22, V17, V19, V21, V23, V8, V9, V10, V11, V12, V13, V14, V15)

	// Scale by 1/16. Broadcast from memory — a register broadcast of a scalar
	// constant costs a fixed ~100ns and would dominate a kernel this small.
	MOVD $·neonInv16(SB), R0
	VLD1R (R0), [V24.S4]

	VEOR V8.B16, V8.B16, V8.B16
	VFMLA V16.S4, V24.S4, V8.S4
	VEOR V9.B16, V9.B16, V9.B16
	VFMLA V17.S4, V24.S4, V9.S4
	VST2 [V8.S4, V9.S4], (R8)

	VEOR V10.B16, V10.B16, V10.B16
	VFMLA V18.S4, V24.S4, V10.S4
	VEOR V11.B16, V11.B16, V11.B16
	VFMLA V19.S4, V24.S4, V11.S4
	ADD  $32, R8, R1
	VST2 [V10.S4, V11.S4], (R1)

	VEOR V12.B16, V12.B16, V12.B16
	VFMLA V20.S4, V24.S4, V12.S4
	VEOR V13.B16, V13.B16, V13.B16
	VFMLA V21.S4, V24.S4, V13.S4
	ADD  $64, R8, R1
	VST2 [V12.S4, V13.S4], (R1)

	VEOR V14.B16, V14.B16, V14.B16
	VFMLA V22.S4, V24.S4, V14.S4
	VEOR V15.B16, V15.B16, V15.B16
	VFMLA V23.S4, V24.S4, V15.S4
	ADD  $96, R8, R1
	VST2 [V14.S4, V15.S4], (R1)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon16r4_inv_false:
	MOVB ZR, ret+96(FP)
	RET

// 1/16, the inverse normalization. Also referenced by neon_f32_size16_radix2.s,
// which is why the definition lives here rather than beside its only user.
DATA ·neonInv16+0(SB)/4, $0x3d800000 // 1/16
GLOBL ·neonInv16(SB), RODATA, $4
