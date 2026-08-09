//go:build arm64 && !purego

// ===========================================================================
// NEON Stockham radix-4 core for complex128, ARM64 — one looped kernel for
// every power-of-four length n >= 64, AND for every n = 2*4^k, 32 <= n <=
// 32768 (32, 128, 512, 2048, 8192, 32768).
// ===========================================================================
//
// This replaces five fully-unrolled "NEON" codelets (sizes 64, 256, 1024,
// 4096 and 16384) that together came to roughly 29,100 lines of *scalar*
// assembly — FMOVD/FADDD/FMULD throughout, not one vector instruction, plus a
// runtime bit-reversal loop and 175 KB of bit-reversal tables. Every one of
// them lost to the pure-Go codelet on an Apple M5, by 1.27x-1.63x. At 18,400
// lines the size-16384 body could not fit any instruction cache, so the
// unrolling was actively harmful on top of the missing vectorization.
//
// It is the complex128 twin of neon_f32_radix4_loop.s, but a re-derivation
// rather than a transliteration: a VLD2 register pair holds only TWO
// complex128 where the .4S pair holds four complex64, so both regimes
// vectorize two lanes wide, the index algebra differs, and the 4x4 transpose
// of the float32 core collapses to four ZIP1/ZIP2 pairs.
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
// TWO VECTORIZATION REGIMES. A VLD2 .D2 register pair holds two complex128.
//
//   * Stage 0 (m = 1): the k-loop has length one, so we vectorize along j,
//     TWO consecutive j per iteration (l = n/4 is even for n >= 8). The four
//     radix-4 inputs x[j + p*l] are two consecutive complex at stride
//     l*16 = 4n bytes, so VLOAD4 applies unchanged.
//
//     Twiddles: w1 = {tw[j], tw[j+1]} is one VLD2. w2 = {tw[2j], tw[2j+2]}
//     and w3 = {tw[3j], tw[3j+3]} are strided, but at two lanes a strided
//     gather is just two VLD1 of a whole complex (16 bytes = one .2D
//     register, [re,im]) plus a ZIP1/ZIP2 pair to split re from im — four
//     instructions, the same cost as the float32 core's VLD2+VUZP1 form and
//     far cheaper than its stride-3 scalar gather.
//
//     Outputs y[4j+q] and y[4(j+1)+q] sit in lane 0 and lane 1 of vector q,
//     so the eight consecutive complex the iteration produces are recovered
//     by four ZIP1/ZIP2 pairs (the .2D degenerate case of a 4x4 transpose)
//     before four plain VST2s.
//
//   * Stages 1 .. s-2 (m >= 4): vectorize along k, two k per iteration (m is
//     a power of four, so m/2 is exact). j is fixed across the inner loop, so
//     w1/w2/w3 are scalars — two VLD1R broadcasts each — and both the loads
//     and the stores are already two consecutive complex. No shuffles at all.
//     This is the cheap case and most of the work happens here.
//
//   * Stage s-1 (l = 1): j = 0 only, so w1 = w2 = w3 = 1. It is a pure
//     butterfly with no complex multiplies, written out separately rather
//     than multiplying by one.
//
// BYTE ALGEBRA, complex128 = 16 bytes:
//   x stride between the four radix-4 inputs   16*(n/4) = 4n      (R6)
//   one k-iteration / one j-pair advance       2*16 = 32
//   stage-0 y advance (8 complex)              128
//   stage j-base advance                       16*m               (R23)
//   stage y j-base advance                     16*4m = 64*m       (R1)
//   final copy, 64 bytes per iteration         16n/64 = n/4
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
// n = 2*4^k EXTENSION. n = 32*4^(k-2) factors as 4^k * 2: k radix-4 stages
// (stage 0 plus k-1 further m>=4 stages, both regimes unchanged from above)
// followed by ONE final radix-2 stage at l = 1, m = n/2. j = 0 there too, so
// w = 1 and it is the cheapest stage that exists: a pure add/sub butterfly,
// k-vectorized two lanes at a time exactly like the radix-4 final stage it
// replaces (VDFT2/VLOAD2/VSTORE2 below). The radix-2 stage is placed LAST,
// not first, specifically so every m stays a power of four and both
// vectorization regimes above apply unchanged; an m=2 first stage would be
// exactly one .D2 vector wide and leave no room for the k-loop they are
// built around.
//
// The l-sequence lets the transition be detected for free: for a
// power-of-four n, l walks 1, 4, 16, ... down to a final l = 1 (caught by
// the existing "CMP $1" check at the top of the stage loop). For n = 2*4^k,
// l instead walks down to l = 2 (never 1) after the last radix-4 stage —
// the one middle-loop iteration whose l was 2 is detected at the BOTTOM of
// the loop (after that stage has executed as an ordinary radix-4 stage) and
// branches directly to the radix-2 final stage instead of back to the top,
// with m already correctly carried forward (m *= 4, same as any radix-4
// stage's contribution to the next stage's m).
//
// BUFFER PARITY across the extra stage. The stage count is now k+1 (k
// radix-4 + 1 radix-2) instead of s = log4(n); whether that is odd or even
// depends on k, and does not need to be reasoned about separately, because
// the runtime pointer-identity compare in the BUFFERING paragraph above
// already generalizes: the radix-2 final stage sets the "result location"
// register exactly as the radix-4 final stage does, and the CMP against dst
// decides whether a copy is needed regardless of how many stages preceded
// it or what the last one was.
//
// FUSED TAIL. The size-specific fused wrappers pass a negative n as an
// internal flag. The core takes its absolute value and records the flag in
// R27. At l = 2 it computes both j branches together, keeps both DFT4 results
// in V0..V23, applies the final radix-2 sums/differences, and writes all eight
// output blocks directly. This removes the intermediate n-element
// store/reload pass. The result lands in the opposite ping-pong buffer from
// the two-pass form; the runtime pointer comparison handles either parity.
//
// TWO MACRO FAMILIES, and the difference is load-bearing:
//
//   * The arithmetic macros (VDFT4_*, VCMUL_*) take register NUMBERS. They
//     expand to WORD-encoded FADD/FSUB/FMUL/FMLA because Go's assembler has
//     no mnemonic for vector FP add, subtract or multiply. See neon_fp.h.
//   * The shuffle/gather macros and all VLD1R/VLD1/VLD2/VST1/VST2 take
//     register NAMES, because those instructions do have mnemonics.
//
// So VDFT4_FWD(0, 2, 4, 6, ...) and VZIP1 V2.D2, V0.D2, ... name the same
// registers. Nothing checks the correspondence; a wrong number assembles
// happily and the registry-driven reference tests are what catch it.
//
// ===========================================================================

#include "textflag.h"
#include "neon_fp.h"

// Forward DFT4 across four vectors (W4 = -i):
//   X0 = t0+t2   X2 = t0-t2   X1 = t1 - i*t3   X3 = t1 + i*t3
// where t0 = a0+a2, t1 = a0-a2, t2 = a1+a3, t3 = a1-a3.
// Operates on real parts ar0..ar3 and imaginary parts ai0..ai3 in place,
// clobbering the eight temporaries. Takes register NUMBERS.
#define VDFT4_FWD(ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3, t0r, t1r, t2r, t3r, t0i, t1i, t2i, t3i) \
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
#define VDFT4_INV(ar0, ar1, ar2, ar3, ai0, ai1, ai2, ai3, t0r, t1r, t2r, t3r, t0i, t1i, t2i, t3i) \
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

// Load the four radix-4 inputs from `from`, stride R6 bytes, into V0..V7 as
// re/im pairs (VLD2's register list must be contiguous).
#define VLOAD4(from) \
	VLD2 (from), [V0.D2, V1.D2] \
	ADD  R6, from, R0           \
	VLD2 (R0), [V2.D2, V3.D2]   \
	ADD  R6, R0, R0             \
	VLD2 (R0), [V4.D2, V5.D2]   \
	ADD  R6, R0, R0             \
	VLD2 (R0), [V6.D2, V7.D2]

// Store V0..V7 to `to`, stride `stride` bytes.
#define VSTORE4(to, stride) \
	VST2 [V0.D2, V1.D2], (to) \
	ADD  stride, to, R0       \
	VST2 [V2.D2, V3.D2], (R0) \
	ADD  stride, R0, R0       \
	VST2 [V4.D2, V5.D2], (R0) \
	ADD  stride, R0, R0       \
	VST2 [V6.D2, V7.D2], (R0)

// Broadcast the scalar twiddles tw[j*m], tw[2*j*m], tw[3*j*m] into
// V16/V17, V18/V19, V20/V21. R21 = &tw[0], R24 = byte offset of tw[j*m].
// Clobbers R0 and R25.
#define VBCAST_TW \
	ADD   R24, R21, R0     \
	VLD1R (R0), [V16.D2]   \
	ADD   $8, R0, R0       \
	VLD1R (R0), [V17.D2]   \
	ADD   R24, R24, R25    \
	ADD   R25, R21, R0     \
	VLD1R (R0), [V18.D2]   \
	ADD   $8, R0, R0       \
	VLD1R (R0), [V19.D2]   \
	ADD   R24, R25, R25    \
	ADD   R25, R21, R0     \
	VLD1R (R0), [V20.D2]   \
	ADD   $8, R0, R0       \
	VLD1R (R0), [V21.D2]

// Broadcast the l=2 stage's j=1 twiddles into V24..V29 for the fused tail.
// R23 = 16*m is both one complex128 block and the byte offset of tw[m].
#define VBCAST_FUSED_TW \
	ADD   R23, R21, R0     \
	VLD1R (R0), [V24.D2]   \
	ADD   $8, R0, R0       \
	VLD1R (R0), [V25.D2]   \
	LSL   $1, R23, R25     \
	ADD   R25, R21, R0     \
	VLD1R (R0), [V26.D2]   \
	ADD   $8, R0, R0       \
	VLD1R (R0), [V27.D2]   \
	ADD   R23, R25, R25    \
	ADD   R25, R21, R0     \
	VLD1R (R0), [V28.D2]   \
	ADD   $8, R0, R0       \
	VLD1R (R0), [V29.D2]

// Gather the three stage-0 twiddle vectors for the j-pair (j, j+1).
// R10 = &tw[j], R24 = &tw[2j], R25 = &tw[3j].
// Leaves w1 in V16/V17, w2 in V22/V23, w3 in V25/V26; clobbers V18..V21, R0.
//
// A whole complex128 is 16 bytes, i.e. exactly one .2D register holding
// [re, im], so a VLD1 pair plus VZIP1/VZIP2 splits two strided elements into
// a re vector and an im vector. VZIP1 Vm, Vn, Vd -> [Vn.D0, Vm.D0] (the two
// reals) and VZIP2 -> [Vn.D1, Vm.D1] (the two imaginaries).
#define VGATHER_STAGE0_TW \
	VLD2  (R10), [V16.D2, V17.D2] \
	VLD1  (R24), [V18.D2]         \
	ADD   $32, R24, R0            \
	VLD1  (R0), [V19.D2]          \
	VZIP1 V19.D2, V18.D2, V22.D2  \
	VZIP2 V19.D2, V18.D2, V23.D2  \
	VLD1  (R25), [V20.D2]         \
	ADD   $48, R25, R0            \
	VLD1  (R0), [V21.D2]          \
	VZIP1 V21.D2, V20.D2, V25.D2  \
	VZIP2 V21.D2, V20.D2, V26.D2

// Re-lane the stage-0 results and store the eight consecutive complex the
// iteration produced. Vector q holds y[4j+q] in lane 0 and y[4(j+1)+q] in
// lane 1, so ZIP1 collects the j outputs and ZIP2 the j+1 outputs.
// R5 = &y[4j]; clobbers V16..V23 and R0.
#define VSTORE_STAGE0 \
	VZIP1 V2.D2, V0.D2, V16.D2 \
	VZIP1 V3.D2, V1.D2, V17.D2 \
	VZIP1 V6.D2, V4.D2, V18.D2 \
	VZIP1 V7.D2, V5.D2, V19.D2 \
	VZIP2 V2.D2, V0.D2, V20.D2 \
	VZIP2 V3.D2, V1.D2, V21.D2 \
	VZIP2 V6.D2, V4.D2, V22.D2 \
	VZIP2 V7.D2, V5.D2, V23.D2 \
	VST2  [V16.D2, V17.D2], (R5) \
	ADD   $32, R5, R0            \
	VST2  [V18.D2, V19.D2], (R0) \
	ADD   $32, R0, R0            \
	VST2  [V20.D2, V21.D2], (R0) \
	ADD   $32, R0, R0            \
	VST2  [V22.D2, V23.D2], (R0)

// Advance the stage-0 running pointers by one iteration (two j).
#define VSTEP_STAGE0 \
	ADD $32, R4, R4   \
	ADD $32, R10, R10 \
	ADD $64, R24, R24 \
	ADD $96, R25, R25 \
	ADD $128, R5, R5

// Radix-2 final-stage butterfly (n = 2*4^k family: l = 1, j = 0, so the only
// twiddle is 1 — pure add/sub, no complex multiply). Register NUMBERS.
// Inputs: ar0/ai0 = a0 re/im, ar1/ai1 = a1 re/im. Outputs: or0/oi0 = y0
// re/im, or1/oi1 = y1 re/im.
#define VDFT2(ar0, ai0, ar1, ai1, or0, oi0, or1, oi1) \
	VADDF_D2(ar0, ar1, or0) \
	VADDF_D2(ai0, ai1, oi0) \
	VSUBF_D2(ar0, ar1, or1) \
	VSUBF_D2(ai0, ai1, oi1)

// Load the two radix-2 final-stage inputs from `from`, stride `stride`
// bytes, into V0/V1 (a0 re/im) and V2/V3 (a1 re/im). Clobbers R0.
#define VLOAD2(from, stride) \
	VLD2 (from), [V0.D2, V1.D2] \
	ADD  stride, from, R0       \
	VLD2 (R0), [V2.D2, V3.D2]

// Store V4/V5 (y0) and V6/V7 (y1) to `to`, stride `stride` bytes. Clobbers
// R0.
#define VSTORE2(to, stride) \
	VST2 [V4.D2, V5.D2], (to) \
	ADD  stride, to, R0       \
	VST2 [V6.D2, V7.D2], (R0)

// Register map, both kernels:
//
//   R0        temporary          R19  dst base (original)
//   R3        inner k counter    R20  scratch base
//   R4        x running ptr      R21  twiddle base
//   R5        y running ptr      R22  n
//   R6        x stride = 4n      R23  16*m
//   R9        x base for j       R24  twiddle offset / stage-0 &tw[2j]
//   R10       y base for j       R25  temporary / stage-0 &tw[3j]
//   R11       stage input ptr    R15  j counter
//   R12       stage output ptr   R13  m
//   R1        64*m (stages>=1)   R14  l
//   R27       fuse-tail flag (negative n from the fused-tail wrapper)

// ---------------------------------------------------------------------------
// func neonRadix4ForwardC128(dst, src, twiddle, scratch []complex128, n int) bool
// ---------------------------------------------------------------------------
TEXT ·neonRadix4ForwardC128(SB), NOSPLIT, $0-105
	MOVD n+96(FP), R22
	MOVD $0, R27
	CMP  $0, R22
	BGE  r4fwd64_size_ready
	NEG  R22, R22
	MOVD $1, R27
r4fwd64_size_ready:

	MOVD src_len+32(FP), R0
	CMP  R22, R0
	BNE  r4fwd64_false

	MOVD dst_len+8(FP), R0
	CMP  R22, R0
	BLT  r4fwd64_false

	MOVD twiddle_len+56(FP), R0
	CMP  R22, R0
	BLT  r4fwd64_false

	MOVD scratch_len+80(FP), R0
	CMP  R22, R0
	BLT  r4fwd64_false

	MOVD dst+0(FP), R19
	MOVD src+24(FP), R11
	MOVD twiddle+48(FP), R21
	MOVD scratch+72(FP), R20

	LSL $2, R22, R6 // x stride = 16*(n/4) = 4n bytes, constant for every stage

	// --- stage 0 (m = 1, l = n/4): vectorized along j, src -> scratch -----
	MOVD R11, R4  // x = src
	MOVD R20, R5  // y = scratch
	MOVD R21, R10 // &tw[j]
	MOVD R21, R24 // &tw[2j]
	MOVD R21, R25 // &tw[3j]
	LSR  $3, R22, R15 // l/2 = n/8 iterations, two j each

r4fwd64_st0:
	VLOAD4(R4)
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)

	VGATHER_STAGE0_TW
	VCMUL_FWD(2, 3, 16, 17, 8, 9)
	VCMUL_FWD(4, 5, 22, 23, 8, 9)
	VCMUL_FWD(6, 7, 25, 26, 8, 9)

	VSTORE_STAGE0

	VSTEP_STAGE0
	SUBS $1, R15, R15
	BNE  r4fwd64_st0

	MOVD R20, R11 // next stage reads scratch
	MOVD R19, R12 // ... and writes dst

	// --- stages 1 .. s-2 (m >= 4, l > 1): vectorized along k -------------
	MOVD $4, R13
	LSR  $4, R22, R14 // l = n/16

r4fwd64_stage:
	CMP $1, R14
	BEQ r4fwd64_last
	CMP $2, R14
	BNE r4fwd64_stage_regular
	CMP $0, R27
	BNE r4fwd64_fused_tail
r4fwd64_stage_regular:

	LSL  $4, R13, R23 // 16*m
	LSL  $6, R13, R1  // 64*m
	MOVD R14, R15
	MOVD R11, R9
	MOVD R12, R10
	MOVD $0, R24

r4fwd64_jloop:
	VBCAST_TW

	MOVD R9, R4
	MOVD R10, R5
	LSR  $1, R13, R3 // m/2 iterations, two k each

r4fwd64_kloop:
	VLOAD4(R4)
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VCMUL_FWD(2, 3, 16, 17, 22, 23)
	VCMUL_FWD(4, 5, 18, 19, 22, 23)
	VCMUL_FWD(6, 7, 20, 21, 22, 23)
	VSTORE4(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4fwd64_kloop

	ADD  R23, R9, R9
	ADD  R1, R10, R10
	ADD  R23, R24, R24
	SUBS $1, R15, R15
	BNE  r4fwd64_jloop

	MOVD R11, R0 // swap the two buffers
	MOVD R12, R11
	MOVD R0, R12

	CMP $2, R14           // was the stage we just finished l = 2?
	BEQ r4fwd64_last2      // then the final stage is radix-2 (n = 2*4^k)

	LSL $2, R13, R13
	LSR $2, R14, R14
	B   r4fwd64_stage

	// --- stage s-1 (m = n/4, l = 1): j = 0, so every twiddle is 1 --------
r4fwd64_last:
	LSL  $4, R13, R23
	MOVD R11, R4
	MOVD R12, R5
	LSR  $1, R13, R3

r4fwd64_lastk:
	VLOAD4(R4)
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VSTORE4(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4fwd64_lastk

	B r4fwd64_tail

	// --- n = 2*4^k final stage (m = n/2, l = 1, radix 2): j = 0, so the
	// only twiddle is 1 -- a pure add/sub butterfly, no complex multiply.
r4fwd64_last2:
	LSL  $2, R13, R13 // m *= 4 (the just-finished stage's radix): m = n/2
	LSL  $4, R13, R23 // stride = m*16 bytes
	MOVD R11, R4
	MOVD R12, R5
	LSR  $1, R13, R3  // m/2 iterations, two k each

r4fwd64_last2k:
	VLOAD2(R4, R23)
	VDFT2(0, 1, 2, 3, 4, 5, 6, 7)
	VSTORE2(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4fwd64_last2k
	B    r4fwd64_tail

r4fwd64_fused_tail:
	LSL  $4, R13, R23
	VBCAST_FUSED_TW
	MOVD R11, R4
	MOVD R12, R5
	LSR  $1, R13, R3

r4fwd64_fused_k:
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

	ADD R23, R4, R0
	VLOAD4(R0)
	VDFT4_FWD(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VCMUL_FWD(2, 3, 24, 25, 30, 31)
	VCMUL_FWD(4, 5, 26, 27, 30, 31)
	VCMUL_FWD(6, 7, 28, 29, 30, 31)

	MOVD R5, R1
	LSL  $2, R23, R25
	ADD  R25, R5, R2
	VADDF_D2(16, 0, 30)
	VADDF_D2(17, 1, 31)
	VST2 [V30.D2, V31.D2], (R1)
	VSUBF_D2(16, 0, 30)
	VSUBF_D2(17, 1, 31)
	VST2 [V30.D2, V31.D2], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_D2(18, 2, 30)
	VADDF_D2(19, 3, 31)
	VST2 [V30.D2, V31.D2], (R1)
	VSUBF_D2(18, 2, 30)
	VSUBF_D2(19, 3, 31)
	VST2 [V30.D2, V31.D2], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_D2(20, 4, 30)
	VADDF_D2(21, 5, 31)
	VST2 [V30.D2, V31.D2], (R1)
	VSUBF_D2(20, 4, 30)
	VSUBF_D2(21, 5, 31)
	VST2 [V30.D2, V31.D2], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_D2(22, 6, 30)
	VADDF_D2(23, 7, 31)
	VST2 [V30.D2, V31.D2], (R1)
	VSUBF_D2(22, 6, 30)
	VSUBF_D2(23, 7, 31)
	VST2 [V30.D2, V31.D2], (R2)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4fwd64_fused_k

r4fwd64_tail:
	MOVD R12, R11 // the result is wherever the last stage wrote

	// The stage count varies by family and size; rather than special-casing
	// each one, track the pointers and compare at runtime, copying
	// scratch -> dst when the result did not land there.
	CMP R19, R11
	BEQ r4fwd64_done

	LSR  $2, R22, R3 // 16n bytes / 64 per iteration
	MOVD R11, R4
	MOVD R19, R5

r4fwd64_copy:
	VLD1.P 64(R4), [V0.B16, V1.B16, V2.B16, V3.B16]
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R5)
	SUBS   $1, R3, R3
	BNE    r4fwd64_copy

r4fwd64_done:
	MOVD $1, R0
	MOVB R0, ret+104(FP)
	RET

r4fwd64_false:
	MOVB ZR, ret+104(FP)
	RET

// ---------------------------------------------------------------------------
// func neonRadix4InverseC128(dst, src, twiddle, scratch []complex128, n int, scale float64) bool
// ---------------------------------------------------------------------------
TEXT ·neonRadix4InverseC128(SB), NOSPLIT, $0-113
	MOVD n+96(FP), R22
	MOVD $0, R27
	CMP  $0, R22
	BGE  r4inv64_size_ready
	NEG  R22, R22
	MOVD $1, R27
r4inv64_size_ready:

	MOVD src_len+32(FP), R0
	CMP  R22, R0
	BNE  r4inv64_false

	MOVD dst_len+8(FP), R0
	CMP  R22, R0
	BLT  r4inv64_false

	MOVD twiddle_len+56(FP), R0
	CMP  R22, R0
	BLT  r4inv64_false

	MOVD scratch_len+80(FP), R0
	CMP  R22, R0
	BLT  r4inv64_false

	MOVD dst+0(FP), R19
	MOVD src+24(FP), R11
	MOVD twiddle+48(FP), R21
	MOVD scratch+72(FP), R20

	// 1/n, folded into stage 0's inputs (see the header): scaling is linear
	// and 1/n is an exact power of two, so this is free of error and saves a
	// scaled variant of the final butterfly.
	FMOVD scale+104(FP), F24
	VDUP  V24.D[0], V24.D2

	LSL $2, R22, R6

	// --- stage 0 (m = 1, l = n/4), src -> scratch ------------------------
	MOVD R11, R4
	MOVD R20, R5
	MOVD R21, R10
	MOVD R21, R24
	MOVD R21, R25
	LSR  $3, R22, R15

r4inv64_st0:
	VLOAD4(R4)

	VMULF_D2(0, 24, 0)
	VMULF_D2(1, 24, 1)
	VMULF_D2(2, 24, 2)
	VMULF_D2(3, 24, 3)
	VMULF_D2(4, 24, 4)
	VMULF_D2(5, 24, 5)
	VMULF_D2(6, 24, 6)
	VMULF_D2(7, 24, 7)

	VDFT4_INV(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)

	VGATHER_STAGE0_TW
	VCMUL_INV(2, 3, 16, 17, 8, 9)
	VCMUL_INV(4, 5, 22, 23, 8, 9)
	VCMUL_INV(6, 7, 25, 26, 8, 9)

	VSTORE_STAGE0

	VSTEP_STAGE0
	SUBS $1, R15, R15
	BNE  r4inv64_st0

	MOVD R20, R11
	MOVD R19, R12

	// --- stages 1 .. s-2 -------------------------------------------------
	MOVD $4, R13
	LSR  $4, R22, R14

r4inv64_stage:
	CMP $1, R14
	BEQ r4inv64_last
	CMP $2, R14
	BNE r4inv64_stage_regular
	CMP $0, R27
	BNE r4inv64_fused_tail
r4inv64_stage_regular:

	LSL  $4, R13, R23
	LSL  $6, R13, R1
	MOVD R14, R15
	MOVD R11, R9
	MOVD R12, R10
	MOVD $0, R24

r4inv64_jloop:
	VBCAST_TW

	MOVD R9, R4
	MOVD R10, R5
	LSR  $1, R13, R3

r4inv64_kloop:
	VLOAD4(R4)
	VDFT4_INV(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VCMUL_INV(2, 3, 16, 17, 22, 23)
	VCMUL_INV(4, 5, 18, 19, 22, 23)
	VCMUL_INV(6, 7, 20, 21, 22, 23)
	VSTORE4(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4inv64_kloop

	ADD  R23, R9, R9
	ADD  R1, R10, R10
	ADD  R23, R24, R24
	SUBS $1, R15, R15
	BNE  r4inv64_jloop

	MOVD R11, R0
	MOVD R12, R11
	MOVD R0, R12

	CMP $2, R14           // was the stage we just finished l = 2?
	BEQ r4inv64_last2      // then the final stage is radix-2 (n = 2*4^k)

	LSL $2, R13, R13
	LSR $2, R14, R14
	B   r4inv64_stage

	// --- stage s-1 (l = 1): every twiddle is 1 ---------------------------
r4inv64_last:
	LSL  $4, R13, R23
	MOVD R11, R4
	MOVD R12, R5
	LSR  $1, R13, R3

r4inv64_lastk:
	VLOAD4(R4)
	VDFT4_INV(0, 2, 4, 6, 1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15)
	VSTORE4(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4inv64_lastk

	B r4inv64_tail

	// --- n = 2*4^k final stage (m = n/2, l = 1, radix 2): every twiddle
	// is 1, same butterfly as the forward path (no conjugation needed for
	// a real add/sub).
r4inv64_last2:
	LSL  $2, R13, R13 // m *= 4: m = n/2
	LSL  $4, R13, R23 // stride = m*16 bytes
	MOVD R11, R4
	MOVD R12, R5
	LSR  $1, R13, R3  // m/2 iterations, two k each

r4inv64_last2k:
	VLOAD2(R4, R23)
	VDFT2(0, 1, 2, 3, 4, 5, 6, 7)
	VSTORE2(R5, R23)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4inv64_last2k
	B    r4inv64_tail

r4inv64_fused_tail:
	LSL  $4, R13, R23
	VBCAST_FUSED_TW
	MOVD R11, R4
	MOVD R12, R5
	LSR  $1, R13, R3

r4inv64_fused_k:
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
	VADDF_D2(16, 0, 30)
	VADDF_D2(17, 1, 31)
	VST2 [V30.D2, V31.D2], (R1)
	VSUBF_D2(16, 0, 30)
	VSUBF_D2(17, 1, 31)
	VST2 [V30.D2, V31.D2], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_D2(18, 2, 30)
	VADDF_D2(19, 3, 31)
	VST2 [V30.D2, V31.D2], (R1)
	VSUBF_D2(18, 2, 30)
	VSUBF_D2(19, 3, 31)
	VST2 [V30.D2, V31.D2], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_D2(20, 4, 30)
	VADDF_D2(21, 5, 31)
	VST2 [V30.D2, V31.D2], (R1)
	VSUBF_D2(20, 4, 30)
	VSUBF_D2(21, 5, 31)
	VST2 [V30.D2, V31.D2], (R2)
	ADD  R23, R1, R1
	ADD  R23, R2, R2
	VADDF_D2(22, 6, 30)
	VADDF_D2(23, 7, 31)
	VST2 [V30.D2, V31.D2], (R1)
	VSUBF_D2(22, 6, 30)
	VSUBF_D2(23, 7, 31)
	VST2 [V30.D2, V31.D2], (R2)

	ADD  $32, R4, R4
	ADD  $32, R5, R5
	SUBS $1, R3, R3
	BNE  r4inv64_fused_k

r4inv64_tail:
	MOVD R12, R11

	CMP R19, R11
	BEQ r4inv64_done

	LSR  $2, R22, R3
	MOVD R11, R4
	MOVD R19, R5

r4inv64_copy:
	VLD1.P 64(R4), [V0.B16, V1.B16, V2.B16, V3.B16]
	VST1.P [V0.B16, V1.B16, V2.B16, V3.B16], 64(R5)
	SUBS   $1, R3, R3
	BNE    r4inv64_copy

r4inv64_done:
	MOVD $1, R0
	MOVB R0, ret+112(FP)
	RET

r4inv64_false:
	MOVB ZR, ret+112(FP)
	RET
