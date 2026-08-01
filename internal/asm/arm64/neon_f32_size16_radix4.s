//go:build arm64 && !purego

// ===========================================================================
// NEON Size-16 Radix-4 FFT Kernels for ARM64
// ===========================================================================
//
// This replaced an earlier scalar implementation that, despite the "NEON"
// name, contained no vector instructions at all — plain FMOVS/FADDS/FMULS
// with a bit-reversal loop — and measured 1.83x (forward) / 2.19x (inverse)
// SLOWER than the pure-Go codelet on an Apple M5. See
// docs/CODELET_BENCHMARKS.md.
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
// TWO MACRO FAMILIES, and the difference is load-bearing:
//
//   * The arithmetic macros (VDFT4_*, VCMUL_*) take register NUMBERS. They
//     expand to WORD-encoded FADD/FSUB/FMUL/FMLA, because Go's assembler has
//     no mnemonic for vector FP add, subtract or multiply, and its
//     preprocessor has no token pasting to turn a name into a number. See
//     neon_fp.h.
//   * The shuffle/load macros (VTRANSPOSE4, VGATHER_TW3) and all VLD2/VST2
//     take register NAMES, because those instructions do have mnemonics.
//
// So VDFT4_FWD(0, 2, 4, 6, ...) and VTRANSPOSE4(V0, V2, V4, V6, ...) name the
// same four registers. Nothing checks the correspondence; the registry-driven
// reference tests are what catch a wrong number.
//
// The earlier version of this kernel synthesized every add and subtract as a
// VMOV plus a VFMLA against a vector of 1.0, since that was the only vector FP
// arithmetic the assembler would accept. That cost two instructions per add
// and a register plus two prologue instructions for the constant. Emitting the
// real encodings halves the butterfly and frees V31.
//
// ===========================================================================

#include "textflag.h"
#include "neon_fp.h"

// 4x4 float32 transpose of rows r0..r3 into o0..o3, clobbering t0..t3.
// Takes register NAMES (VTRN/VZIP have real mnemonics).
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
// clobbering the eight temporaries. Takes register NUMBERS.
#define VDFT4_FWD(ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3, t0r, t1r, t2r, t3r, t0i, t1i, t2i, t3i) \
	VADDF_S4(ar0, ar2, t0r) \
	VSUBF_S4(ar0, ar2, t1r) \
	VADDF_S4(ar1, ar3, t2r) \
	VSUBF_S4(ar1, ar3, t3r) \
	VADDF_S4(ai0, ai2, t0i) \
	VSUBF_S4(ai0, ai2, t1i) \
	VADDF_S4(ai1, ai3, t2i) \
	VSUBF_S4(ai1, ai3, t3i) \
	VADDF_S4(t0r, t2r, ar0) \
	VADDF_S4(t0i, t2i, ai0) \
	VSUBF_S4(t0r, t2r, ar2) \
	VSUBF_S4(t0i, t2i, ai2) \
	VADDF_S4(t1r, t3i, ar1) \
	VSUBF_S4(t1i, t3r, ai1) \
	VSUBF_S4(t1r, t3i, ar3) \
	VADDF_S4(t1i, t3r, ai3)

// Inverse DFT4 across four vectors (W4 = +i): X1 and X3 swap relative to fwd.
#define VDFT4_INV(ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3, t0r, t1r, t2r, t3r, t0i, t1i, t2i, t3i) \
	VADDF_S4(ar0, ar2, t0r) \
	VSUBF_S4(ar0, ar2, t1r) \
	VADDF_S4(ar1, ar3, t2r) \
	VSUBF_S4(ar1, ar3, t3r) \
	VADDF_S4(ai0, ai2, t0i) \
	VSUBF_S4(ai0, ai2, t1i) \
	VADDF_S4(ai1, ai3, t2i) \
	VSUBF_S4(ai1, ai3, t3i) \
	VADDF_S4(t0r, t2r, ar0) \
	VADDF_S4(t0i, t2i, ai0) \
	VSUBF_S4(t0r, t2r, ar2) \
	VSUBF_S4(t0i, t2i, ai2) \
	VSUBF_S4(t1r, t3i, ar1) \
	VADDF_S4(t1i, t3r, ai1) \
	VADDF_S4(t1r, t3i, ar3) \
	VSUBF_S4(t1i, t3r, ai3)

// dr,di *= (wr + i*wi), result back in dr,di; clobbers p, q. Register NUMBERS.
#define VCMUL_FWD(dr, di, wr, wi, p, q) \
	VMULF_S4(dr, wr, p) \
	VFMSF_S4(di, wi, p) \
	VMULF_S4(dr, wi, q) \
	VFMAF_S4(di, wr, q) \
	VMOVR(p, dr)        \
	VMOVR(q, di)

// dr,di *= conj(wr + i*wi) — the inverse twiddle.
#define VCMUL_INV(dr, di, wr, wi, p, q) \
	VMULF_S4(dr, wr, p) \
	VFMAF_S4(di, wi, p) \
	VMULF_S4(di, wr, q) \
	VFMSF_S4(dr, wi, q) \
	VMOVR(p, dr)        \
	VMOVR(q, di)

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
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)

	// (B) twiddle by W16^(n1*k2). k2 = 0 is all-ones, so it is skipped.
	// k2 = 1: [tw0, tw1, tw2, tw3] — four consecutive entries.
	VLD2 (R10), [V16.S4, V17.S4]
	VCMUL_FWD(2, 3, 16, 17, 8, 9)

	// k2 = 2: [tw0, tw2, tw4, tw6] — the even entries of tw[0..7].
	VLD2 (R10), [V18.S4, V19.S4]
	ADD  $32, R10, R1
	VLD2 (R1), [V20.S4, V21.S4]
	VUZP1 V20.S4, V18.S4, V22.S4
	VUZP1 V21.S4, V19.S4, V23.S4
	VCMUL_FWD(4, 5, 22, 23, 8, 9)

	// k2 = 3: [tw0, tw3, tw6, tw9].
	VGATHER_TW3(R10, V25, V26)
	VCMUL_FWD(6, 7, 25, 26, 8, 9)

	// (C) transpose: lane becomes k2, vector becomes n1. Outputs are laid out
	// as adjacent re/im pairs again so the stores can be VST2.
	VTRANSPOSE4(V0, V2, V4, V6, V8, V9, V10, V11, V16, V18, V20, V22)
	VTRANSPOSE4(V1, V3, V5, V7, V8, V9, V10, V11, V17, V19, V21, V23)

	// (D) DFT4 over n1. Vector index becomes k1, lane stays k2.
	VDFT4_FWD(16, 18, 20, 22, 17, 19, 21, 23, 8, 9, 10, 11, 12, 13, 14, 15)

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

	// Load x[n1 + 4*n2]: vector n2, lane n1.
	VLD2 (R9), [V0.S4, V1.S4]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.S4, V3.S4]
	ADD  $64, R9, R1
	VLD2 (R1), [V4.S4, V5.S4]
	ADD  $96, R9, R1
	VLD2 (R1), [V6.S4, V7.S4]

	// (A) DFT4 over n2. Vector index becomes k2.
	VDFT4_INV(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)

	// (B) twiddle by W16^(n1*k2), conjugated. k2 = 0 is skipped.
	VLD2 (R10), [V16.S4, V17.S4]
	VCMUL_INV(2, 3, 16, 17, 8, 9)

	VLD2 (R10), [V18.S4, V19.S4]
	ADD  $32, R10, R1
	VLD2 (R1), [V20.S4, V21.S4]
	VUZP1 V20.S4, V18.S4, V22.S4
	VUZP1 V21.S4, V19.S4, V23.S4
	VCMUL_INV(4, 5, 22, 23, 8, 9)

	VGATHER_TW3(R10, V25, V26)
	VCMUL_INV(6, 7, 25, 26, 8, 9)

	// (C) transpose: lane becomes k2, vector becomes n1.
	VTRANSPOSE4(V0, V2, V4, V6, V8, V9, V10, V11, V16, V18, V20, V22)
	VTRANSPOSE4(V1, V3, V5, V7, V8, V9, V10, V11, V17, V19, V21, V23)

	// (D) DFT4 over n1. Vector index becomes k1, lane stays k2.
	VDFT4_INV(16, 18, 20, 22, 17, 19, 21, 23, 8, 9, 10, 11, 12, 13, 14, 15)

	// Scale by 1/16. Broadcast from memory — a register broadcast of a scalar
	// constant costs a fixed ~100ns and would dominate a kernel this small.
	MOVD  $·neonInv16(SB), R0
	VLD1R (R0), [V24.S4]

	VMULF_S4(16, 24, 8)
	VMULF_S4(17, 24, 9)
	VST2 [V8.S4, V9.S4], (R8)

	VMULF_S4(18, 24, 10)
	VMULF_S4(19, 24, 11)
	ADD  $32, R8, R1
	VST2 [V10.S4, V11.S4], (R1)

	VMULF_S4(20, 24, 12)
	VMULF_S4(21, 24, 13)
	ADD  $64, R8, R1
	VST2 [V12.S4, V13.S4], (R1)

	VMULF_S4(22, 24, 14)
	VMULF_S4(23, 24, 15)
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
