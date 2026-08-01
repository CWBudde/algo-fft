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
// Go's assembler has no mnemonic for vector FADD/FSUB/FMUL; the real
// encodings are emitted directly with WORD via the macros in neon_fp.h
// (VADDF_D2/VSUBF_D2/VMULF_D2/VFMAF_D2/VFMSF_D2), which take register
// NUMBERS rather than names because the assembler's preprocessor has no
// token pasting. See neon_fp.h for the full rationale and the encoding
// table.
//
// ===========================================================================

#include "textflag.h"
#include "neon_fp.h"

// dr,di *= (wr + i*wi), result back in dr,di; clobbers p, q. Register NUMBERS.
#define VCMUL_FWD(dr, di, wr, wi, p, q) \
	VMULF_D2(dr, wr, p) \
	VFMSF_D2(di, wi, p) \
	VMULF_D2(dr, wi, q) \
	VFMAF_D2(di, wr, q) \
	VMOVR(p, dr)        \
	VMOVR(q, di)

// dr,di *= conj(wr + i*wi) — the inverse twiddle.
#define VCMUL_INV(dr, di, wr, wi, p, q) \
	VMULF_D2(dr, wr, p) \
	VFMAF_D2(di, wi, p) \
	VMULF_D2(di, wr, q) \
	VFMSF_D2(dr, wi, q) \
	VMOVR(p, dr)        \
	VMOVR(q, di)

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

	// Load x[n1 + 2*n2]: vector n2, lane n1. VLD2 deinterleaves re/im, and its
	// register list must be contiguous, so re/im pairs are adjacent:
	// re[n2] = V0,V2   im[n2] = V1,V3
	VLD2 (R9), [V0.D2, V1.D2]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.D2, V3.D2]

	// (A) DFT2 over n2. Vector index becomes k2.
	VADDF_D2(0, 2, 4)
	VADDF_D2(1, 3, 5)
	VSUBF_D2(0, 2, 6)
	VSUBF_D2(1, 3, 7)

	// (B) twiddle by W4^(n1*k2). k2 = 0 is all-ones, so it is skipped.
	// k2 = 1: [tw0, tw1] = [1, -i] — loads for free from the twiddle table.
	VLD2 (R10), [V8.D2, V9.D2]
	VCMUL_FWD(6, 7, 8, 9, 10, 11)

	// (C) 2x2 transpose: lane becomes k2, vector becomes n1.
	VZIP1 V6.D2, V4.D2, V12.D2 // n1=0 re: [u0r.D0, u1r.D0]
	VZIP2 V6.D2, V4.D2, V13.D2 // n1=1 re: [u0r.D1, u1r.D1]
	VZIP1 V7.D2, V5.D2, V14.D2 // n1=0 im
	VZIP2 V7.D2, V5.D2, V15.D2 // n1=1 im

	// (D) DFT2 over n1. Vector index becomes k1, lane stays k2.
	VADDF_D2(12, 13, 16) // X0, X1
	VADDF_D2(14, 15, 17)
	VSUBF_D2(12, 13, 18) // X2, X3
	VSUBF_D2(14, 15, 19)

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

	// Load x[n1 + 2*n2]: vector n2, lane n1.
	VLD2 (R9), [V0.D2, V1.D2]
	ADD  $32, R9, R1
	VLD2 (R1), [V2.D2, V3.D2]

	// (A) DFT2 over n2. Vector index becomes k2.
	VADDF_D2(0, 2, 4)
	VADDF_D2(1, 3, 5)
	VSUBF_D2(0, 2, 6)
	VSUBF_D2(1, 3, 7)

	// (B) twiddle by W4^(n1*k2), conjugated. k2 = 0 is all-ones, so it is
	// skipped. k2 = 1 uses the same loaded [1, -i] with VCMUL_INV, which
	// conjugates it to [1, +i].
	VLD2 (R10), [V8.D2, V9.D2]
	VCMUL_INV(6, 7, 8, 9, 10, 11)

	// (C) 2x2 transpose: lane becomes k2, vector becomes n1.
	VZIP1 V6.D2, V4.D2, V12.D2
	VZIP2 V6.D2, V4.D2, V13.D2
	VZIP1 V7.D2, V5.D2, V14.D2
	VZIP2 V7.D2, V5.D2, V15.D2

	// (D) DFT2 over n1. Vector index becomes k1, lane stays k2.
	VADDF_D2(12, 13, 16)
	VADDF_D2(14, 15, 17)
	VSUBF_D2(12, 13, 18)
	VSUBF_D2(14, 15, 19)

	// Scale by 1/4. Broadcast from memory — a register broadcast of a scalar
	// constant costs a fixed ~100ns and would dominate a kernel this small.
	MOVD $·neonInv4F64(SB), R0
	VLD1R (R0), [V20.D2]

	VMULF_D2(16, 20, 21)
	VMULF_D2(17, 20, 22)
	VST2 [V21.D2, V22.D2], (R8)

	VMULF_D2(18, 20, 23)
	VMULF_D2(19, 20, 24)
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
