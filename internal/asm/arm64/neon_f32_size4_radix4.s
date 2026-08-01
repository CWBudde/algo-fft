//go:build arm64 && !purego

// ===========================================================================
// NEON Size-4 Radix-4 FFT Kernels for ARM64
// ===========================================================================
//
// This replaced an earlier scalar implementation that, despite the "NEON"
// name, contained no vector instructions at all — plain FMOVS/FADDS/FSUBS —
// and measured 3.23x SLOWER (inverse) than the pure-Go codelet on an Apple
// M5. See docs/CODELET_BENCHMARKS.md.
//
// n = 4 complex64 is only 32 bytes = 2 Q registers — too small for the 4x4
// vertical decomposition the size-16 kernel uses (VLD2 split layout, DFT4
// across four vectors, transpose). Instead this works directly on the raw
// interleaved (re,im,re,im,...) layout loaded as two S4 vectors:
//
//   V0 = [x0r, x0i, x1r, x1i]   V1 = [x2r, x2i, x3r, x3i]
//
// Radix-4 DIT butterfly:
//   t0 = x0+x2   t1 = x0-x2   t2 = x1+x3   t3 = x1-x3
//   X0 = t0+t2   X2 = t0-t2
//   X1 = t1 - i*t3   X3 = t1 + i*t3            (forward, W4 = -i)
//   X1 = t1 + i*t3   X3 = t1 - i*t3            (inverse, W4 = +i)
//
// A = t0,t2 (VADDF of V0,V1); B = t1,t3 (VSUBF of V0,V1). A D2-zip pair
// regroups these into C = [t0, t1] and D = [t2, t3]. A VREV64 + single-lane
// blend turns D into E = [t2r, t2i, t3i, t3r]; negating one lane (chosen by
// direction) yields ±i*t3 in the upper half, so Dp = [t2, ±i*t3] and
// C±Dp gives all four outputs directly, laid out as two consecutive
// complex pairs ready for a plain VST1.
//
// The twiddle argument is unused (all n=4 factors are trivial) but its
// length is still validated to keep the accepted-input contract unchanged.
//
// dst may alias src: both source vectors are loaded into registers before
// the first store, so nothing needs a scratch copy-back.
//
// Go's arm64 assembler has NO vector FADD/FSUB/FMUL — VFMLA and VFMLS are
// the only vector FP arithmetic mnemonics it accepts. Addition and
// subtraction are synthesized against a vector of 1.0 (·neonOnes), exactly
// as neon_f32_generic.s and neon_f32_size16_radix4.s do.
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

// ---------------------------------------------------------------------------
// func ForwardNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool
// ---------------------------------------------------------------------------
TEXT ·ForwardNEONSize4Radix4Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD src_len+32(FP), R13

	CMP  $4, R13
	BNE  neon4r4_return_false

	MOVD dst_len+8(FP), R0
	CMP  $4, R0
	BLT  neon4r4_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $4, R0
	BLT  neon4r4_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $4, R0
	BLT  neon4r4_return_false

	MOVD $·neonOnes(SB), R0
	VLD1 (R0), [ONES.S4]

	// Raw interleaved load: V0 = x0,x1  V1 = x2,x3.
	VLD1 (R9), [V0.S4, V1.S4]

	// A = [t0, t2] = x0+x2, x1+x3.  B = [t1, t3] = x0-x2, x1-x3.
	VADDF(V0, V1, V2)
	VSUBF(V0, V1, V3)

	// C = [t0, t1]  D = [t2, t3].
	VZIP1 V3.D2, V2.D2, V4.D2
	VZIP2 V3.D2, V2.D2, V5.D2

	// E = [t2i, t2r, t3i, t3r]; blend D's low half with E's high half to
	// get Dp = [t2r, t2i, t3i, t3r], then negate lane 3 (t3r -> -t3r) so
	// the upper half becomes -i*t3 = (t3i, -t3r).
	VREV64 V5.S4, V6.S4
	VMOV   V5.B16, V7.B16
	VMOV   V6.D[1], V7.D[1]

	MOVD $·neonSignLane3F32(SB), R0
	VLD1 (R0), [V16.S4]
	VEOR V16.B16, V7.B16, V7.B16

	// X_lo = C+Dp = [X0, X1]   X_hi = C-Dp = [X2, X3].
	VADDF(V4, V7, V8)
	VSUBF(V4, V7, V9)

	VST1 [V8.S4, V9.S4], (R8)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon4r4_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// ---------------------------------------------------------------------------
// func InverseNEONSize4Radix4Complex64Asm(dst, src, twiddle, scratch []complex64) bool
// ---------------------------------------------------------------------------
TEXT ·InverseNEONSize4Radix4Complex64Asm(SB), NOSPLIT, $0-97
	MOVD dst+0(FP), R8
	MOVD src+24(FP), R9
	MOVD src_len+32(FP), R13

	CMP  $4, R13
	BNE  neon4r4_inv_return_false

	MOVD dst_len+8(FP), R0
	CMP  $4, R0
	BLT  neon4r4_inv_return_false

	MOVD twiddle_len+56(FP), R0
	CMP  $4, R0
	BLT  neon4r4_inv_return_false

	MOVD scratch_len+80(FP), R0
	CMP  $4, R0
	BLT  neon4r4_inv_return_false

	MOVD $·neonOnes(SB), R0
	VLD1 (R0), [ONES.S4]

	// Raw interleaved load: V0 = x0,x1  V1 = x2,x3.
	VLD1 (R9), [V0.S4, V1.S4]

	// A = [t0, t2] = x0+x2, x1+x3.  B = [t1, t3] = x0-x2, x1-x3.
	VADDF(V0, V1, V2)
	VSUBF(V0, V1, V3)

	// C = [t0, t1]  D = [t2, t3].
	VZIP1 V3.D2, V2.D2, V4.D2
	VZIP2 V3.D2, V2.D2, V5.D2

	// E = [t2i, t2r, t3i, t3r]; blend D's low half with E's high half to
	// get Dp = [t2r, t2i, t3i, t3r], then negate lane 2 (t3i -> -t3i) so
	// the upper half becomes +i*t3 = (-t3i, t3r).
	VREV64 V5.S4, V6.S4
	VMOV   V5.B16, V7.B16
	VMOV   V6.D[1], V7.D[1]

	MOVD $·neonSignLane2F32(SB), R0
	VLD1 (R0), [V16.S4]
	VEOR V16.B16, V7.B16, V7.B16

	// X_lo = C+Dp = [X0, X1]   X_hi = C-Dp = [X2, X3].
	VADDF(V4, V7, V8)
	VSUBF(V4, V7, V9)

	// Scale by 1/4. Broadcast from memory — a register broadcast of a
	// scalar constant costs a fixed ~100ns and would dominate a kernel
	// this small.
	MOVD  $·neonInv4(SB), R0
	VLD1R (R0), [V17.S4]

	VEOR  V10.B16, V10.B16, V10.B16
	VFMLA V8.S4, V17.S4, V10.S4
	VEOR  V11.B16, V11.B16, V11.B16
	VFMLA V9.S4, V17.S4, V11.S4

	VST1 [V10.S4, V11.S4], (R8)

	MOVD $1, R0
	MOVB R0, ret+96(FP)
	RET

neon4r4_inv_return_false:
	MOVD $0, R0
	MOVB R0, ret+96(FP)
	RET

// Sign masks for the ±i*t3 blend: negate a single lane (the low 32 bits of
// the D-lane holding t3's real or imaginary part) via VEOR.
DATA ·neonSignLane3F32+0(SB)/4, $0x00000000
DATA ·neonSignLane3F32+4(SB)/4, $0x00000000
DATA ·neonSignLane3F32+8(SB)/4, $0x00000000
DATA ·neonSignLane3F32+12(SB)/4, $0x80000000
GLOBL ·neonSignLane3F32(SB), RODATA, $16

DATA ·neonSignLane2F32+0(SB)/4, $0x00000000
DATA ·neonSignLane2F32+4(SB)/4, $0x00000000
DATA ·neonSignLane2F32+8(SB)/4, $0x80000000
DATA ·neonSignLane2F32+12(SB)/4, $0x00000000
GLOBL ·neonSignLane2F32(SB), RODATA, $16
