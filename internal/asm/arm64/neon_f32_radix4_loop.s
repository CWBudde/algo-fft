//go:build arm64 && !purego

// ===========================================================================
// NEON Stockham radix-4 core for complex64, ARM64 — one looped kernel for
// every power-of-four length n >= 64, extended below to also serve every
// n = 2*4^k (32, 128, 512, 2048, 8192, 32768).
// ===========================================================================
//
// This replaces five fully-unrolled "NEON" codelets (sizes 64, 256, 1024,
// 4096 and 16384) that together came to roughly 28,500 lines of *scalar*
// assembly — FMOVS/FADDS/FMULS throughout, not one vector instruction, plus a
// runtime bit-reversal loop and 175 KB of bit-reversal tables. Every one of
// them lost to the pure-Go codelet on an Apple M5. At 18,000 lines the size-
// 16384 body could not fit any instruction cache, so the unrolling was
// actively harmful on top of the missing vectorization.
//
// ALGORITHM — Stockham autosort, radix 4, out of place, no bit-reversal.
//
// n is a power of four, n >= 64. Let s = log4(n). Start l = n/4, m = 1; after
// each stage l /= 4 and m *= 4. One stage reads x and writes y:
//
//   for j = 0 .. l-1:
//      w1 = tw[j*m] ; w2 = tw[2*j*m] ; w3 = tw[3*j*m]      tw[t] = W_n^t
//      for k = 0 .. m-1:
//         a0 = x[k + m*(j+0*l)] ; a1 = x[k + m*(j+1*l)]
//         a2 = x[k + m*(j+2*l)] ; a3 = x[k + m*(j+3*l)]
//         t0 = a0+a2 ; t1 = a0-a2 ; t2 = a1+a3 ; t3 = a1-a3
//         y[k + m*(4j+0)] = t0 + t2
//         y[k + m*(4j+1)] = (t1 - i*t3) * w1
//         y[k + m*(4j+2)] = (t0 - t2)   * w2
//         y[k + m*(4j+3)] = (t1 + i*t3) * w3
//
// 3*j*m <= 3*(n/4 - 1) < n, so every twiddle index is inside the table.
//
// In the split re/im layout VLD2 produces, -i*t3 costs nothing — no shuffle,
// just a crossed add/subtract. That is what VDFT4_FWD below computes.
//
// TWO VECTORIZATION REGIMES. A VLD2 register pair holds four complex64.
//
//   * Stage 0 (m = 1): the k-loop has length one, so we vectorize along j,
//     four consecutive j per iteration (l = n/4 is a multiple of 4 for
//     n >= 64). This is exactly the size-16 kernel
//     (neon_f32_size16_radix4.s) with the loads, the twiddle gathers and the
//     stores given a running base — including the strided w2 gather (two
//     VLD2 plus VUZP1) and the stride-3 w3 scalar gather. Outputs are
//     strided, so a 4x4 transpose turns each vector back into four
//     consecutive complex before the VST2.
//
//   * Stages 1 .. s-2 (m >= 4): vectorize along k, four k per iteration (m is
//     a power of four). j is fixed across the inner loop, so w1/w2/w3 are
//     scalars — two VLD1R broadcasts each — and both the loads and the stores
//     are already four consecutive complex. No shuffles at all. This is the
//     cheap case and most of the work happens here.
//
//   * Stage s-1 (l = 1): j = 0 only, so w1 = w2 = w3 = 1. It is a pure
//     butterfly with no complex multiplies, written out separately rather
//     than multiplying by one.
//
// BUFFERING. dst may alias src (the registry's in-place test does exactly
// that) and stage 0 reads locations that later iterations of the same stage
// would overwrite, so stage 0 ALWAYS writes scratch. After that the two
// buffers alternate. Whether the last stage lands in dst depends on the
// parity of s, so rather than special-casing each size the code tracks the
// pointers and does a runtime compare at the end, copying scratch -> dst when
// the result did not land there. s is odd (copy needed) for n = 64, 1024 and
// 16384; even (no copy) for n = 256 and 4096.
//
// INVERSE. Same structure with VDFT4_INV (+i*t3 / -i*t3 swapped) and
// VCMUL_INV (conjugated twiddles). The 1/n normalization is folded into
// stage 0's *inputs* rather than the final stage's outputs: scaling is
// linear, so it commutes with every stage, 1/n is an exact power of two so
// nothing is lost, and doing it on the input costs eight VMULs per stage-0
// iteration instead of needing a scaled variant of the final butterfly.
//
// EXTENSION TO n = 2*4^k (sizes 32, 128, 512, 2048, 8192, 32768). Put the
// radix-2 factor LAST rather than first: the radix schedule becomes
// [4, 4, ..., 4, 2] with k radix-4 stages followed by one radix-2 stage, so
// every m before the final stage stays a power of four and both existing
// vectorization regimes above are reused unchanged. Putting the 2 first
// instead would force an m = 2 stage — half the width a VLD2 pair holds for
// complex64 — wasting half the vector lanes in a hot stage.
//
// The invariant l*m*radix = n still holds every stage. For n = 2*4^k, the
// last RADIX-4 stage lands at l = 2 (not l = 1: the trailing factor of 2
// still has to come out of l somewhere), so it is NOT the trivial one-
// twiddle stage — j ranges over 0 and 1, so w1 = tw[m] is a real, non-unity
// twiddle. It runs through the ordinary k-vectorized middle-stage path
// unchanged. Only the stage after it — the final radix-2 stage proper, at
// l = 1, m = n/2 — is genuinely new code: j = 0 only, so w = 1, and it is a
// pure add/subtract butterfly over m = n/2 lanes with no complex multiply at
// all, k-vectorized exactly like the radix-4 final stage it sits after:
//
//   for k = 0 .. m-1:
//      a0 = x[k] ; a1 = x[k + m]
//      y[k]     = a0 + a1
//      y[k + m] = a0 - a1
//
// For a pure power of four, l descends through powers of four (4, 16, ...)
// and is never 2, so checking "was this stage's l exactly 2" cleanly
// distinguishes the two families with no size whitelist: the core stays a
// general "any n = 4^s" / "any n = 2*4^s" implementation, matching every
// existing Go wrapper without the wrappers needing to say which family they
// are.
//
// BUFFER PARITY reasoning for the extended family: stage 0 always writes
// scratch (dst may alias src). Every subsequent radix-4 stage — including
// the l = 2 one — swaps the two buffers exactly as before. The new radix-2
// stage is simply one more stage that swaps again: it reads whatever stage
// 0's opposite-parity chain left as input (R11) and writes the other buffer
// (R12), using the same "compare the final output pointer against dst and
// copy if they differ" logic already in place for the trivial radix-4 last
// stage — the copy-or-not decision was already computed at runtime rather
// than baked in per size, so adding one more stage to the chain needs no
// new parity bookkeeping at all, only that the new stage participates in
// the same buffer hand-off. Concretely: for n = 32 there are 3 total stages
// (stage 0, one radix-4 middle stage, one radix-2 final stage) — odd, so a
// copy is needed, matching the pattern already documented below for s odd.
// For n = 128 there are 4 stages — even, no copy. This alternates with k
// exactly as the s-odd/even rule already does for pure powers of four.
//
// FUSED TAIL. The size-specific fused wrappers pass n with its sign bit set;
// the core takes the absolute size and records the sign in R27. At l = 2 that
// path computes j = 0 and j = 1 together, keeps both DFT4 results in V0..V23,
// applies the final radix-2 sums/differences, and writes the eight output
// blocks directly. It removes the intermediate n-element store/reload pass.
// The fused result lands in the opposite ping-pong buffer from the two-pass
// form, but the existing runtime pointer comparison already handles that.
//
// TWO MACRO FAMILIES, and the difference is load-bearing:
//
//   * The arithmetic macros (VDFT4_*, VCMUL_*) take register NUMBERS. They
//     expand to WORD-encoded FADD/FSUB/FMUL/FMLA because Go's assembler has
//     no mnemonic for vector FP add, subtract or multiply. See neon_fp.h.
//   * The shuffle/gather macros and all VLD1R/VLD2/VST1/VST2 take register
//     NAMES, because those instructions do have mnemonics.
//
// So VDFT4_FWD(0, 2, 4, 6, ...) and VTRANSPOSE4(V0, V2, V4, V6, ...) name the
// same four registers. Nothing checks the correspondence; a wrong number
// assembles happily and the registry-driven reference tests are what catch
// it.
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

// Radix-2 final-stage butterfly (W2 = 1, so no twiddle multiply at all):
// or0/oi0 = a0+a1, or1/oi1 = a0-a1. Same for forward and inverse — there is
// no sign difference, since j = 0 is the only lane and w = 1 either way.
// Register NUMBERS.
#define VBFLY2(ar0, ai0, ar1, ai1, or0, oi0, or1, oi1) \
	VADDF_S4(ar0, ar1, or0) \
	VADDF_S4(ai0, ai1, oi0) \
	VSUBF_S4(ar0, ar1, or1) \
	VSUBF_S4(ai0, ai1, oi1)

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

// Gather the stage-0 w3 vector [tw[3j], tw[3j+3], tw[3j+6], tw[3j+9]] from
// base = &tw[3j]. Stride 3 has no shuffle form, so it is a scalar gather into
// lanes. Clobbers R1 and R2.
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

// Load the four radix-4 inputs from R4, stride R6 bytes, into V0..V7 as
// re/im pairs (VLD2's register list must be contiguous).
#define VLOAD4(from) \
	VLD2 (from), [V0.S4, V1.S4] \
	ADD  R6, from, R0           \
	VLD2 (R0), [V2.S4, V3.S4]   \
	ADD  R6, R0, R0             \
	VLD2 (R0), [V4.S4, V5.S4]   \
	ADD  R6, R0, R0             \
	VLD2 (R0), [V6.S4, V7.S4]

// Store V0..V7 to `to`, stride `stride` bytes.
#define VSTORE4(to, stride) \
	VST2 [V0.S4, V1.S4], (to) \
	ADD  stride, to, R0       \
	VST2 [V2.S4, V3.S4], (R0) \
	ADD  stride, R0, R0       \
	VST2 [V4.S4, V5.S4], (R0) \
	ADD  stride, R0, R0       \
	VST2 [V6.S4, V7.S4], (R0)

// Broadcast the scalar twiddles tw[j*m], tw[2*j*m], tw[3*j*m] into
// V16/V17, V18/V19, V20/V21. R21 = &tw[0], R24 = byte offset of tw[j*m].
// Clobbers R0 and R25.
#define VBCAST_TW \
	ADD   R24, R21, R0     \
	VLD1R (R0), [V16.S4]   \
	ADD   $4, R0, R0       \
	VLD1R (R0), [V17.S4]   \
	ADD   R24, R24, R25    \
	ADD   R25, R21, R0     \
	VLD1R (R0), [V18.S4]   \
	ADD   $4, R0, R0       \
	VLD1R (R0), [V19.S4]   \
	ADD   R24, R25, R25    \
	ADD   R25, R21, R0     \
	VLD1R (R0), [V20.S4]   \
	ADD   $4, R0, R0       \
	VLD1R (R0), [V21.S4]

// Broadcast the l=2 stage's non-unity j=1 twiddles into V24..V29 for the
// fused radix-4/radix-2 tail. R23 = 8*m is both one complex64 block and the
// byte offset of tw[m]. Clobbers R0 and R25.
#define VBCAST_FUSED_TW \
	ADD   R23, R21, R0     \
	VLD1R (R0), [V24.S4]   \
	ADD   $4, R0, R0       \
	VLD1R (R0), [V25.S4]   \
	LSL   $1, R23, R25     \
	ADD   R25, R21, R0     \
	VLD1R (R0), [V26.S4]   \
	ADD   $4, R0, R0       \
	VLD1R (R0), [V27.S4]   \
	ADD   R23, R25, R25    \
	ADD   R25, R21, R0     \
	VLD1R (R0), [V28.S4]   \
	ADD   $4, R0, R0       \
	VLD1R (R0), [V29.S4]

// Gather the three stage-0 twiddle vectors. R10 = &tw[j], R24 = &tw[2j],
// R25 = &tw[3j]. Leaves w1 in V16/V17, w2 in V22/V23, w3 in V25/V26 and
// clobbers V18..V21, R0, R1, R2.
#define VGATHER_STAGE0_TW \
	VLD2  (R10), [V16.S4, V17.S4]   \
	VLD2  (R24), [V18.S4, V19.S4]   \
	ADD   $32, R24, R0              \
	VLD2  (R0), [V20.S4, V21.S4]    \
	VUZP1 V20.S4, V18.S4, V22.S4    \
	VUZP1 V21.S4, V19.S4, V23.S4    \
	VGATHER_TW3(R25, V25, V26)

// Advance the stage-0 running pointers by one iteration (four j).
#define VSTEP_STAGE0 \
	ADD $32, R4, R4   \
	ADD $32, R10, R10 \
	ADD $64, R24, R24 \
	ADD $96, R25, R25 \
	ADD $128, R5, R5

// Register map, both kernels:
//
//   R0,R1,R2  temporaries       R19  dst base (original)
//   R3        inner k counter   R20  scratch base
//   R4        x running ptr     R21  twiddle base
//   R5        y running ptr     R22  n
//   R6        x stride = 2n     R23  8*m
//   R9        x base for j      R24  twiddle offset / stage-0 &tw[2j]
//   R10       y base for j      R25  temporary / stage-0 &tw[3j]
//   R11       stage input ptr   R15  j counter
//   R12       stage output ptr  R13  m
//   R1        32*m (stages>=1)  R14  l
//   R26       saved l, to detect the n = 2*4^k family's last radix-4 stage
//             (l == 2) after the stage runs and l/m have been updated
//   R27       fuse-tail flag (negative n from the fused-tail wrapper)

// ---------------------------------------------------------------------------
// func neonRadix4ForwardC64(dst, src, twiddle, scratch []complex64, n int) bool
// ---------------------------------------------------------------------------
TEXT ·neonRadix4ForwardC64(SB), NOSPLIT, $0-105
	MOVD n+96(FP), R22
	MOVD $0, R27
	CMP  $0, R22
	BGE  r4fwd_size_ready
	NEG  R22, R22
	MOVD $1, R27
r4fwd_size_ready:

	MOVD src_len+32(FP), R0
	CMP  R22, R0
	BNE  r4fwd_false

	MOVD dst_len+8(FP), R0
	CMP  R22, R0
	BLT  r4fwd_false

	MOVD twiddle_len+56(FP), R0
	CMP  R22, R0
	BLT  r4fwd_false

	MOVD scratch_len+80(FP), R0
	CMP  R22, R0
	BLT  r4fwd_false

	MOVD dst+0(FP), R19
	MOVD src+24(FP), R11
	MOVD twiddle+48(FP), R21
	MOVD scratch+72(FP), R20

	LSL $1, R22, R6 // x stride = 8*(n/4) = 2n bytes, constant for every stage

	// --- stage 0 (m = 1, l = n/4): vectorized along j, src -> scratch -----
	MOVD R11, R4  // x = src
	MOVD R20, R5  // y = scratch
	MOVD R21, R10 // &tw[j]
	MOVD R21, R24 // &tw[2j]
	MOVD R21, R25 // &tw[3j]
	LSR  $4, R22, R15

r4fwd_st0:
	VLOAD4(R4)
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)

	VGATHER_STAGE0_TW
	VCMUL_FWD(2, 3, 16, 17, 8, 9)
	VCMUL_FWD(4, 5, 22, 23, 8, 9)
	VCMUL_FWD(6, 7, 25, 26, 8, 9)

	// Outputs y[4(j+p)+q] are strided across the four vectors; transposing
	// puts four consecutive complex back into each vector so the stores are
	// plain VST2s.
	VTRANSPOSE4(V0, V2, V4, V6, V8, V9, V10, V11, V16, V18, V20, V22)
	VTRANSPOSE4(V1, V3, V5, V7, V8, V9, V10, V11, V17, V19, V21, V23)

	VST2 [V16.S4, V17.S4], (R5)
	ADD  $32, R5, R0
	VST2 [V18.S4, V19.S4], (R0)
	ADD  $32, R0, R0
	VST2 [V20.S4, V21.S4], (R0)
	ADD  $32, R0, R0
	VST2 [V22.S4, V23.S4], (R0)

	VSTEP_STAGE0
	SUBS $1, R15, R15
	BNE  r4fwd_st0

	MOVD R20, R11 // next stage reads scratch
	MOVD R19, R12 // ... and writes dst

	// --- stages 1 .. s-2 (m >= 4, l > 1): vectorized along k -------------
	MOVD $4, R13
	LSR  $4, R22, R14 // l = n/16

r4fwd_stage:
	CMP $1, R14
	BEQ r4fwd_last
	CMP $2, R14
	BNE r4fwd_stage_regular
	CMP $0, R27
	BNE r4fwd_fused_tail
r4fwd_stage_regular:
	MOVD R14, R26 // remember this stage's l, to spot the n = 2*4^k family's last radix-4 stage (l == 2) once l/m are updated below

	LSL  $3, R13, R23 // 8*m
	LSL  $5, R13, R1  // 32*m
	MOVD R14, R15
	MOVD R11, R9
	MOVD R12, R10
	MOVD $0, R24

r4fwd_jloop:
	VBCAST_TW

	MOVD R9, R4
	MOVD R10, R5
	LSR  $2, R13, R3

r4fwd_kloop:
	VLOAD4(R4)
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VCMUL_FWD(2, 3, 16, 17, 22, 23)
	VCMUL_FWD(4, 5, 18, 19, 22, 23)
	VCMUL_FWD(6, 7, 20, 21, 22, 23)
	VSTORE4(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4fwd_kloop

	ADD  R23, R9, R9
	ADD  R1, R10, R10
	ADD  R23, R24, R24
	SUBS $1, R15, R15
	BNE  r4fwd_jloop

	MOVD R11, R0 // swap the two buffers
	MOVD R12, R11
	MOVD R0, R12

	LSL $2, R13, R13
	LSR $2, R14, R14

	// n = 2*4^k: l == 2 was this stage's l (saved above), so it was the
	// last radix-4 stage (full twiddles, not the trivial l == 1 case
	// below). m has already been updated to n/2; go straight to the new
	// radix-2 final stage instead of looping (l/4 from 2 is not valid).
	CMP $2, R26
	BEQ r4fwd_final_radix2
	B   r4fwd_stage

	// --- stage s-1 (m = n/4, l = 1): j = 0, so every twiddle is 1 --------
r4fwd_last:
	LSL  $3, R13, R23
	MOVD R11, R4
	MOVD R12, R5
	LSR  $2, R13, R3

r4fwd_lastk:
	VLOAD4(R4)
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VSTORE4(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4fwd_lastk

	B r4fwd_finish

	// n = 2*4^k only: final radix-2 stage (l = 1, m = n/2). j = 0, so
	// w = 1 and there is no complex multiply — a pure add/subtract
	// butterfly, k-vectorized four lanes at a time exactly like the
	// radix-4 final stage above.
r4fwd_final_radix2:
	LSL  $3, R13, R23 // 8*m = byte offset between x[k] and x[k+m]
	MOVD R11, R4
	MOVD R12, R5
	LSR  $2, R13, R3 // m/4 iterations

r4fwd_radix2k:
	VLD2 (R4), [V0.S4, V1.S4] // a0 = x[k..k+3]
	ADD  R23, R4, R0
	VLD2 (R0), [V2.S4, V3.S4] // a1 = x[k+m..k+m+3]

	VBFLY2(0, 1, 2, 3, 4, 5, 6, 7)

	VST2 [V4.S4, V5.S4], (R5)
	ADD  R23, R5, R0
	VST2 [V6.S4, V7.S4], (R0)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4fwd_radix2k
	B    r4fwd_finish

	// Fused alternative for n = 2*4^k: compute both j values of the
	// l=2 radix-4 stage together, then perform the radix-2 add/sub between
	// corresponding outputs while all eight radix-4 results are live. This
	// removes the intermediate n-element store/reload pass.
r4fwd_fused_tail:
	LSL  $3, R13, R23 // one m-element block, and tw[m], in bytes
	VBCAST_FUSED_TW
	MOVD R11, R4
	MOVD R12, R5
	LSR  $2, R13, R3

r4fwd_fused_k:
	// j=0 reads blocks 0,2,4,6 and has unity twiddles.
	VLOAD4(R4)
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VMOVR(0, 16)
	VMOVR(1, 17)
	VMOVR(2, 18)
	VMOVR(3, 19)
	VMOVR(4, 20)
	VMOVR(5, 21)
	VMOVR(6, 22)
	VMOVR(7, 23)

	// j=1 reads blocks 1,3,5,7 and uses tw[m], tw[2m], tw[3m].
	ADD R23, R4, R0
	VLOAD4(R0)
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VCMUL_FWD(2, 3, 24, 25, 30, 31)
	VCMUL_FWD(4, 5, 26, 27, 30, 31)
	VCMUL_FWD(6, 7, 28, 29, 30, 31)

	// Blocks 0..3 are the radix-2 sums, blocks 4..7 the differences. Store
	// one output pair at a time so V24..V29 keep the twiddles across k.
	MOVD R5, R1
	LSL  $2, R23, R25
	ADD  R25, R5, R2
	VADDF_S4(16, 0, 30)
	VADDF_S4(17, 1, 31)
	VST2 [V30.S4, V31.S4], (R1)
	VSUBF_S4(16, 0, 30)
	VSUBF_S4(17, 1, 31)
	VST2 [V30.S4, V31.S4], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_S4(18, 2, 30)
	VADDF_S4(19, 3, 31)
	VST2 [V30.S4, V31.S4], (R1)
	VSUBF_S4(18, 2, 30)
	VSUBF_S4(19, 3, 31)
	VST2 [V30.S4, V31.S4], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_S4(20, 4, 30)
	VADDF_S4(21, 5, 31)
	VST2 [V30.S4, V31.S4], (R1)
	VSUBF_S4(20, 4, 30)
	VSUBF_S4(21, 5, 31)
	VST2 [V30.S4, V31.S4], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_S4(22, 6, 30)
	VADDF_S4(23, 7, 31)
	VST2 [V30.S4, V31.S4], (R1)
	VSUBF_S4(22, 6, 30)
	VSUBF_S4(23, 7, 31)
	VST2 [V30.S4, V31.S4], (R2)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4fwd_fused_k

r4fwd_finish:
	MOVD R12, R11 // the result is wherever the last stage wrote

	// The number of stages is odd (copy needed) for n = 64, 1024, 16384,
	// 32, 512 and 8192; even (no copy) for n = 256, 4096, 128, 2048 and
	// 32768. Stage 0 must write scratch because dst may alias src, so an
	// odd total lands in scratch and has to be copied out; this is a
	// runtime comparison rather than a per-size special case, so the new
	// radix-2 final stage (one more link in the same swap chain) needs no
	// separate parity bookkeeping.
	CMP R19, R11
	BEQ r4fwd_done

	LSR  $3, R22, R3 // 8n bytes / 64 per iteration
	MOVD R11, R4
	MOVD R19, R5

r4fwd_copy:
	VLD1.P 64(R4), [V0.B16, V1.B16, V2.B16, V3.B16]
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R5)
	SUBS   $1, R3, R3
	BNE    r4fwd_copy

r4fwd_done:
	MOVD $1, R0
	MOVB R0, ret+104(FP)
	RET

r4fwd_false:
	MOVB ZR, ret+104(FP)
	RET

// ---------------------------------------------------------------------------
// func neonRadix4InverseC64(dst, src, twiddle, scratch []complex64, n int, scale float32) bool
// ---------------------------------------------------------------------------
TEXT ·neonRadix4InverseC64(SB), NOSPLIT, $0-113
	MOVD n+96(FP), R22
	MOVD $0, R27
	CMP  $0, R22
	BGE  r4inv_size_ready
	NEG  R22, R22
	MOVD $1, R27
r4inv_size_ready:

	MOVD src_len+32(FP), R0
	CMP  R22, R0
	BNE  r4inv_false

	MOVD dst_len+8(FP), R0
	CMP  R22, R0
	BLT  r4inv_false

	MOVD twiddle_len+56(FP), R0
	CMP  R22, R0
	BLT  r4inv_false

	MOVD scratch_len+80(FP), R0
	CMP  R22, R0
	BLT  r4inv_false

	MOVD dst+0(FP), R19
	MOVD src+24(FP), R11
	MOVD twiddle+48(FP), R21
	MOVD scratch+72(FP), R20

	// 1/n, folded into stage 0's inputs (see the header): scaling is linear
	// and 1/n is an exact power of two, so this is free of error and saves a
	// scaled variant of the final butterfly.
	FMOVS scale+104(FP), F24
	VDUP  V24.S[0], V24.S4

	LSL $1, R22, R6

	// --- stage 0 (m = 1, l = n/4), src -> scratch ------------------------
	MOVD R11, R4
	MOVD R20, R5
	MOVD R21, R10
	MOVD R21, R24
	MOVD R21, R25
	LSR  $4, R22, R15

r4inv_st0:
	VLOAD4(R4)

	VMULF_S4(0, 24, 0)
	VMULF_S4(1, 24, 1)
	VMULF_S4(2, 24, 2)
	VMULF_S4(3, 24, 3)
	VMULF_S4(4, 24, 4)
	VMULF_S4(5, 24, 5)
	VMULF_S4(6, 24, 6)
	VMULF_S4(7, 24, 7)

	VDFT4_INV(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)

	VGATHER_STAGE0_TW
	VCMUL_INV(2, 3, 16, 17, 8, 9)
	VCMUL_INV(4, 5, 22, 23, 8, 9)
	VCMUL_INV(6, 7, 25, 26, 8, 9)

	VTRANSPOSE4(V0, V2, V4, V6, V8, V9, V10, V11, V16, V18, V20, V22)
	VTRANSPOSE4(V1, V3, V5, V7, V8, V9, V10, V11, V17, V19, V21, V23)

	VST2 [V16.S4, V17.S4], (R5)
	ADD  $32, R5, R0
	VST2 [V18.S4, V19.S4], (R0)
	ADD  $32, R0, R0
	VST2 [V20.S4, V21.S4], (R0)
	ADD  $32, R0, R0
	VST2 [V22.S4, V23.S4], (R0)

	VSTEP_STAGE0
	SUBS $1, R15, R15
	BNE  r4inv_st0

	MOVD R20, R11
	MOVD R19, R12

	// --- stages 1 .. s-2 -------------------------------------------------
	MOVD $4, R13
	LSR  $4, R22, R14

r4inv_stage:
	CMP $1, R14
	BEQ r4inv_last
	CMP $2, R14
	BNE r4inv_stage_regular
	CMP $0, R27
	BNE r4inv_fused_tail
r4inv_stage_regular:
	MOVD R14, R26 // remember this stage's l (see r4fwd_stage)

	LSL  $3, R13, R23
	LSL  $5, R13, R1
	MOVD R14, R15
	MOVD R11, R9
	MOVD R12, R10
	MOVD $0, R24

r4inv_jloop:
	VBCAST_TW

	MOVD R9, R4
	MOVD R10, R5
	LSR  $2, R13, R3

r4inv_kloop:
	VLOAD4(R4)
	VDFT4_INV(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VCMUL_INV(2, 3, 16, 17, 22, 23)
	VCMUL_INV(4, 5, 18, 19, 22, 23)
	VCMUL_INV(6, 7, 20, 21, 22, 23)
	VSTORE4(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4inv_kloop

	ADD  R23, R9, R9
	ADD  R1, R10, R10
	ADD  R23, R24, R24
	SUBS $1, R15, R15
	BNE  r4inv_jloop

	MOVD R11, R0
	MOVD R12, R11
	MOVD R0, R12

	LSL $2, R13, R13
	LSR $2, R14, R14

	// n = 2*4^k: see r4fwd_stage for why l == 2 here means "go straight to
	// the final radix-2 stage".
	CMP $2, R26
	BEQ r4inv_final_radix2
	B   r4inv_stage

	// --- stage s-1 (l = 1): every twiddle is 1 ---------------------------
r4inv_last:
	LSL  $3, R13, R23
	MOVD R11, R4
	MOVD R12, R5
	LSR  $2, R13, R3

r4inv_lastk:
	VLOAD4(R4)
	VDFT4_INV(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VSTORE4(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4inv_lastk

	B r4inv_finish

	// n = 2*4^k only: final radix-2 stage. The 1/n scaling was already
	// folded into stage 0's inputs (see the header), and scaling commutes
	// with a linear butterfly, so no extra scale multiply is needed here.
r4inv_final_radix2:
	LSL  $3, R13, R23
	MOVD R11, R4
	MOVD R12, R5
	LSR  $2, R13, R3

r4inv_radix2k:
	VLD2 (R4), [V0.S4, V1.S4]
	ADD  R23, R4, R0
	VLD2 (R0), [V2.S4, V3.S4]

	VBFLY2(0, 1, 2, 3, 4, 5, 6, 7)

	VST2 [V4.S4, V5.S4], (R5)
	ADD  R23, R5, R0
	VST2 [V6.S4, V7.S4], (R0)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4inv_radix2k
	B    r4inv_finish

r4inv_fused_tail:
	LSL  $3, R13, R23
	VBCAST_FUSED_TW
	MOVD R11, R4
	MOVD R12, R5
	LSR  $2, R13, R3

r4inv_fused_k:
	VLOAD4(R4)
	VDFT4_INV(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VMOVR(0, 16)
	VMOVR(1, 17)
	VMOVR(2, 18)
	VMOVR(3, 19)
	VMOVR(4, 20)
	VMOVR(5, 21)
	VMOVR(6, 22)
	VMOVR(7, 23)

	ADD R23, R4, R0
	VLOAD4(R0)
	VDFT4_INV(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VCMUL_INV(2, 3, 24, 25, 30, 31)
	VCMUL_INV(4, 5, 26, 27, 30, 31)
	VCMUL_INV(6, 7, 28, 29, 30, 31)

	MOVD R5, R1
	LSL  $2, R23, R25
	ADD  R25, R5, R2
	VADDF_S4(16, 0, 30)
	VADDF_S4(17, 1, 31)
	VST2 [V30.S4, V31.S4], (R1)
	VSUBF_S4(16, 0, 30)
	VSUBF_S4(17, 1, 31)
	VST2 [V30.S4, V31.S4], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_S4(18, 2, 30)
	VADDF_S4(19, 3, 31)
	VST2 [V30.S4, V31.S4], (R1)
	VSUBF_S4(18, 2, 30)
	VSUBF_S4(19, 3, 31)
	VST2 [V30.S4, V31.S4], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_S4(20, 4, 30)
	VADDF_S4(21, 5, 31)
	VST2 [V30.S4, V31.S4], (R1)
	VSUBF_S4(20, 4, 30)
	VSUBF_S4(21, 5, 31)
	VST2 [V30.S4, V31.S4], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_S4(22, 6, 30)
	VADDF_S4(23, 7, 31)
	VST2 [V30.S4, V31.S4], (R1)
	VSUBF_S4(22, 6, 30)
	VSUBF_S4(23, 7, 31)
	VST2 [V30.S4, V31.S4], (R2)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4inv_fused_k

r4inv_finish:
	MOVD R12, R11

	CMP R19, R11
	BEQ r4inv_done

	LSR  $3, R22, R3
	MOVD R11, R4
	MOVD R19, R5

r4inv_copy:
	VLD1.P 64(R4), [V0.B16, V1.B16, V2.B16, V3.B16]
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R5)
	SUBS   $1, R3, R3
	BNE    r4inv_copy

r4inv_done:
	MOVD $1, R0
	MOVB R0, ret+112(FP)
	RET

r4inv_false:
	MOVB ZR, ret+112(FP)
	RET
