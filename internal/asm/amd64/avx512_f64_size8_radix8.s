//go:build amd64 && !purego

// ===========================================================================
// AVX-512 Size-8 Radix-8 (complex128) FFT Kernels for AMD64
// ===========================================================================
//
// A ZMM register holds 4 complex128 values, so the whole size-8 transform is
// two ZMM registers: no stage touches memory between the two loads and the two
// stores, and `scratch` is never used (in-place dst == src is therefore safe
// without a copy).
//
// ALGORITHM (single radix-8 butterfly, identity input permutation)
// ---------------------------------------------------------------
// Stage 1 pairs x[j] with x[j+4] (the inter-register stage -- free):
//   s = [s0,s1,s2,s3] = x[0..3] + x[4..7]
//   d = [d0,d1,d2,d3] = x[0..3] - x[4..7]
//
// Stage 2 combines s0/s2, s1/s3 and d0/d2, d1/d3, with a -i twist on the d
// pairs. Both operands are gathered by two VPERMT2PD, which also folds in the
// real/imag swap that the -i multiply needs:
//   P = [s0, d0, s1, d1]
//   Q = [s2, -i*d2, s3, -i*d3]        (permute gives swap(d), XOR gives -i)
//   R = P + Q = [e0, e1, o0, o1]
//   S = P - Q = [e2, e3, o2, o3]
// with the classic radix-8 intermediates
//   e0 = s0+s2   e1 = d0-i*d2   e2 = s0-s2   e3 = d0+i*d2
//   o0 = s1+s3   o1 = d1-i*d3   o2 = s1-s3   o3 = d1+i*d3
//
// Two VSHUFF64X2 then regroup halves into E = [e0,e1,e2,e3] and
// O = [o0,o1,o2,o3], and stage 3 is a single vector butterfly:
//   X[k]   = E[k] + W8^k * O[k]
//   X[k+4] = E[k] - W8^k * O[k]        k = 0..3
//
// TWIDDLES
// --------
// At n = 8 the four twiddles are known exactly: W = [1, c-i*c, -i, -c-i*c]
// with c = sqrt(2)/2, so no general complex multiply and no twiddle table read
// is needed. Writing w = (wr, wi) and using swap(o) = (o.im, o.re),
//   w*o = A*o + B*swap(o),  A = [wr, wr],  B = [-wi, +wi]
// which makes A and B compile-time vectors (twA8/twB8 below): the whole
// stage-3 twiddle multiply is one VMULPD plus one VFMADD231PD. The inverse
// needs conj(w), i.e. B negated, which is a VFMSUB231PD instead -- and the 1/8
// normalisation is folded into the A/B constants (exact, they are powers of two
// apart) plus one VMULPD on E.
//
// The `twiddle` argument is therefore not read; only `src`/`dst` are touched.
//
// INSTRUCTION SET
// ---------------
// AVX512F only. In particular the sign flips use VPXORQ, not VXORPD: the EVEX
// forms of VXORPD/VXORPS are AVX512DQ, while the gate
// (golang.org/x/sys/cpu.X86.HasAVX512) is CPUID leaf 7 EBX bit 16 = AVX512F.
// The two are bit-identical here; D/Q vs PS/PD only changes masking
// granularity, and no masking is used.
//
// ===========================================================================

#include "textflag.h"

// VPERMT2PD index vectors (qword granularity; 0..7 select the destination
// register = table0 = s, 8..15 select the second source = table1 = d).
//
// idxP8r8: P = [s0, d0, s1, d1]
DATA zidxP8r8<>+0(SB)/8, $0
DATA zidxP8r8<>+8(SB)/8, $1
DATA zidxP8r8<>+16(SB)/8, $8
DATA zidxP8r8<>+24(SB)/8, $9
DATA zidxP8r8<>+32(SB)/8, $2
DATA zidxP8r8<>+40(SB)/8, $3
DATA zidxP8r8<>+48(SB)/8, $10
DATA zidxP8r8<>+56(SB)/8, $11
GLOBL zidxP8r8<>(SB), RODATA|NOPTR, $64

// idxQ8r8: Q = [s2, swap(d2), s3, swap(d3)] -- the swapped d lanes turn the
// following sign flip into a multiply by -i (forward) or +i (inverse).
DATA zidxQ8r8<>+0(SB)/8, $4
DATA zidxQ8r8<>+8(SB)/8, $5
DATA zidxQ8r8<>+16(SB)/8, $13
DATA zidxQ8r8<>+24(SB)/8, $12
DATA zidxQ8r8<>+32(SB)/8, $6
DATA zidxQ8r8<>+40(SB)/8, $7
DATA zidxQ8r8<>+48(SB)/8, $15
DATA zidxQ8r8<>+56(SB)/8, $14
GLOBL zidxQ8r8<>(SB), RODATA|NOPTR, $64

// Forward: negate doubles 3 and 7 -> (d.im, -d.re) = -i*d in lanes 1 and 3.
DATA zneg37<>+0(SB)/8, $0x0000000000000000
DATA zneg37<>+8(SB)/8, $0x0000000000000000
DATA zneg37<>+16(SB)/8, $0x0000000000000000
DATA zneg37<>+24(SB)/8, $0x8000000000000000
DATA zneg37<>+32(SB)/8, $0x0000000000000000
DATA zneg37<>+40(SB)/8, $0x0000000000000000
DATA zneg37<>+48(SB)/8, $0x0000000000000000
DATA zneg37<>+56(SB)/8, $0x8000000000000000
GLOBL zneg37<>(SB), RODATA|NOPTR, $64

// Inverse: negate doubles 2 and 6 -> (-d.im, d.re) = +i*d in lanes 1 and 3.
DATA zneg26<>+0(SB)/8, $0x0000000000000000
DATA zneg26<>+8(SB)/8, $0x0000000000000000
DATA zneg26<>+16(SB)/8, $0x8000000000000000
DATA zneg26<>+24(SB)/8, $0x0000000000000000
DATA zneg26<>+32(SB)/8, $0x0000000000000000
DATA zneg26<>+40(SB)/8, $0x0000000000000000
DATA zneg26<>+48(SB)/8, $0x8000000000000000
DATA zneg26<>+56(SB)/8, $0x0000000000000000
GLOBL zneg26<>(SB), RODATA|NOPTR, $64

// twA8 = [wr, wr] per lane for W = [1, c-i*c, -i, -c-i*c], c = sqrt(2)/2.
DATA ztwA8<>+0(SB)/8, $0x3ff0000000000000  // 1
DATA ztwA8<>+8(SB)/8, $0x3ff0000000000000  // 1
DATA ztwA8<>+16(SB)/8, $0x3fe6a09e667f3bcd // c
DATA ztwA8<>+24(SB)/8, $0x3fe6a09e667f3bcd // c
DATA ztwA8<>+32(SB)/8, $0x0000000000000000 // 0
DATA ztwA8<>+40(SB)/8, $0x0000000000000000 // 0
DATA ztwA8<>+48(SB)/8, $0xbfe6a09e667f3bcd // -c
DATA ztwA8<>+56(SB)/8, $0xbfe6a09e667f3bcd // -c
GLOBL ztwA8<>(SB), RODATA|NOPTR, $64

// twB8 = [-wi, +wi] per lane for the same W.
DATA ztwB8<>+0(SB)/8, $0x0000000000000000  // 0
DATA ztwB8<>+8(SB)/8, $0x0000000000000000  // 0
DATA ztwB8<>+16(SB)/8, $0x3fe6a09e667f3bcd // c
DATA ztwB8<>+24(SB)/8, $0xbfe6a09e667f3bcd // -c
DATA ztwB8<>+32(SB)/8, $0x3ff0000000000000 // 1
DATA ztwB8<>+40(SB)/8, $0xbff0000000000000 // -1
DATA ztwB8<>+48(SB)/8, $0x3fe6a09e667f3bcd // c
DATA ztwB8<>+56(SB)/8, $0xbfe6a09e667f3bcd // -c
GLOBL ztwB8<>(SB), RODATA|NOPTR, $64

// twA8 scaled by 1/8 for the inverse (exact: pure exponent change).
DATA ztwA8i<>+0(SB)/8, $0x3fc0000000000000  // 1/8
DATA ztwA8i<>+8(SB)/8, $0x3fc0000000000000  // 1/8
DATA ztwA8i<>+16(SB)/8, $0x3fb6a09e667f3bcd // c/8
DATA ztwA8i<>+24(SB)/8, $0x3fb6a09e667f3bcd // c/8
DATA ztwA8i<>+32(SB)/8, $0x0000000000000000 // 0
DATA ztwA8i<>+40(SB)/8, $0x0000000000000000 // 0
DATA ztwA8i<>+48(SB)/8, $0xbfb6a09e667f3bcd // -c/8
DATA ztwA8i<>+56(SB)/8, $0xbfb6a09e667f3bcd // -c/8
GLOBL ztwA8i<>(SB), RODATA|NOPTR, $64

// twB8 scaled by 1/8 for the inverse.
DATA ztwB8i<>+0(SB)/8, $0x0000000000000000  // 0
DATA ztwB8i<>+8(SB)/8, $0x0000000000000000  // 0
DATA ztwB8i<>+16(SB)/8, $0x3fb6a09e667f3bcd // c/8
DATA ztwB8i<>+24(SB)/8, $0xbfb6a09e667f3bcd // -c/8
DATA ztwB8i<>+32(SB)/8, $0x3fc0000000000000 // 1/8
DATA ztwB8i<>+40(SB)/8, $0xbfc0000000000000 // -1/8
DATA ztwB8i<>+48(SB)/8, $0x3fb6a09e667f3bcd // c/8
DATA ztwB8i<>+56(SB)/8, $0xbfb6a09e667f3bcd // -c/8
GLOBL ztwB8i<>(SB), RODATA|NOPTR, $64

// ===========================================================================
// Forward transform, size 8, complex128, radix-8 (AVX-512)
// ===========================================================================
TEXT ·ForwardAVX512Size8Radix8Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8 = dst pointer
	MOVQ src+24(FP), R9      // R9 = src pointer
	MOVQ src_len+32(FP), R13 // R13 = n

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_512_r8_fwd_return_false

	// Validate all slice lengths >= 8
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $8
	JL   size8_512_r8_fwd_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $8
	JL   size8_512_r8_fwd_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $8
	JL   size8_512_r8_fwd_return_false

	// Load the whole transform: 2 ZMM = 8 complex128, natural order.
	VMOVUPD 0(R9), Z0  // Z0 = [x0, x1, x2, x3]
	VMOVUPD 64(R9), Z1 // Z1 = [x4, x5, x6, x7]

	// Stage 1: s = x[0..3] + x[4..7], d = x[0..3] - x[4..7]
	VADDPD Z1, Z0, Z2 // Z2 = s = [s0, s1, s2, s3]
	VSUBPD Z1, Z0, Z3 // Z3 = d = [d0, d1, d2, d3]

	// Stage 2 operands. VPERMT2PD overwrites its table0 (= destination), so s
	// is copied first; the copy is handled by move elimination.
	VMOVUPD zidxP8r8<>(SB), Z4 // Z4 = index [0,1,8,9,2,3,10,11]
	VMOVUPD zidxQ8r8<>(SB), Z5 // Z5 = index [4,5,13,12,6,7,15,14]
	VMOVAPD Z2, Z6             // Z6 = s (second table0 copy)

	VPERMT2PD Z3, Z4, Z2 // Z2 = P = [s0, d0, s1, d1]
	VPERMT2PD Z3, Z5, Z6 // Z6 = [s2, swap(d2), s3, swap(d3)]

	VPXORQ zneg37<>(SB), Z6, Z6 // Z6 = Q = [s2, -i*d2, s3, -i*d3]

	VADDPD Z6, Z2, Z7 // Z7 = R = P + Q = [e0, e1, o0, o1]
	VSUBPD Z6, Z2, Z8 // Z8 = S = P - Q = [e2, e3, o2, o3]

	// Regroup: E from the low halves of R/S, O from the high halves.
	VSHUFF64X2 $0x44, Z8, Z7, Z9  // Z9  = E = [e0, e1, e2, e3]
	VSHUFF64X2 $0xEE, Z8, Z7, Z10 // Z10 = O = [o0, o1, o2, o3]

	// Stage 3 twiddle: W*O = A*O + B*swap(O) with the constant A/B vectors.
	VPERMILPD   $0x55, Z10, Z11        // Z11 = swap(O) = [(o.im, o.re) x 4]
	VMULPD      ztwB8<>(SB), Z11, Z11  // Z11 = B * swap(O)
	VFMADD231PD ztwA8<>(SB), Z10, Z11  // Z11 = A*O + B*swap(O) = W .* O

	VADDPD Z11, Z9, Z12 // Z12 = X[0..3] = E + W.*O
	VSUBPD Z11, Z9, Z13 // Z13 = X[4..7] = E - W.*O

	VMOVUPD Z12, 0(R8)  // store X[0..3]
	VMOVUPD Z13, 64(R8) // store X[4..7]

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size8_512_r8_fwd_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 8, complex128, radix-8 (AVX-512)
// Conjugated twiddles (+i twist, VFMSUB231PD) and 1/8 normalisation.
// ===========================================================================
TEXT ·InverseAVX512Size8Radix8Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8 = dst pointer
	MOVQ src+24(FP), R9      // R9 = src pointer
	MOVQ src_len+32(FP), R13 // R13 = n

	// Verify n == 8
	CMPQ R13, $8
	JNE  size8_512_r8_inv_return_false

	// Validate all slice lengths >= 8
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $8
	JL   size8_512_r8_inv_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $8
	JL   size8_512_r8_inv_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $8
	JL   size8_512_r8_inv_return_false

	// Load the whole transform: 2 ZMM = 8 complex128, natural order.
	VMOVUPD 0(R9), Z0  // Z0 = [x0, x1, x2, x3]
	VMOVUPD 64(R9), Z1 // Z1 = [x4, x5, x6, x7]

	// Stage 1: s = x[0..3] + x[4..7], d = x[0..3] - x[4..7]
	VADDPD Z1, Z0, Z2 // Z2 = s = [s0, s1, s2, s3]
	VSUBPD Z1, Z0, Z3 // Z3 = d = [d0, d1, d2, d3]

	// Stage 2 operands (same permutes as the forward transform).
	VMOVUPD zidxP8r8<>(SB), Z4 // Z4 = index [0,1,8,9,2,3,10,11]
	VMOVUPD zidxQ8r8<>(SB), Z5 // Z5 = index [4,5,13,12,6,7,15,14]
	VMOVAPD Z2, Z6             // Z6 = s (second table0 copy)

	VPERMT2PD Z3, Z4, Z2 // Z2 = P = [s0, d0, s1, d1]
	VPERMT2PD Z3, Z5, Z6 // Z6 = [s2, swap(d2), s3, swap(d3)]

	VPXORQ zneg26<>(SB), Z6, Z6 // Z6 = Q = [s2, +i*d2, s3, +i*d3]

	VADDPD Z6, Z2, Z7 // Z7 = R = P + Q = [e0, e1, o0, o1]
	VSUBPD Z6, Z2, Z8 // Z8 = S = P - Q = [e2, e3, o2, o3]

	// Regroup: E from the low halves of R/S, O from the high halves.
	VSHUFF64X2 $0x44, Z8, Z7, Z9  // Z9  = E = [e0, e1, e2, e3]
	VSHUFF64X2 $0xEE, Z8, Z7, Z10 // Z10 = O = [o0, o1, o2, o3]

	VMULPD.BCST ·eighth64(SB), Z9, Z9 // Z9 = E/8 (1/N normalisation, exact)

	// Stage 3 twiddle: conj(W)*O/8 = (A/8)*O - (B/8)*swap(O).
	VPERMILPD   $0x55, Z10, Z11         // Z11 = swap(O)
	VMULPD      ztwB8i<>(SB), Z11, Z11  // Z11 = (B/8) * swap(O)
	VFMSUB231PD ztwA8i<>(SB), Z10, Z11  // Z11 = (A/8)*O - (B/8)*swap(O)

	VADDPD Z11, Z9, Z12 // Z12 = x[0..3] = (E + conj(W).*O)/8
	VSUBPD Z11, Z9, Z13 // Z13 = x[4..7] = (E - conj(W).*O)/8

	VMOVUPD Z12, 0(R8)  // store x[0..3]
	VMOVUPD Z13, 64(R8) // store x[4..7]

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size8_512_r8_inv_return_false:
	MOVB $0, ret+96(FP)
	RET
