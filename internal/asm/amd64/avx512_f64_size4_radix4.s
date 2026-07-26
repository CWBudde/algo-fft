//go:build amd64 && !purego

// ===========================================================================
// AVX-512 Size-4 Radix-4 (complex128) FFT Kernels for AMD64
// ===========================================================================
//
// 4 complex128 = 64 bytes = exactly one ZMM register, so the transform is one
// load, one store, and no memory traffic in between. `twiddle` and `scratch`
// are not read (a size-4 radix-4 butterfly has no non-trivial twiddle), which
// also makes in-place (dst == src) safe without a copy.
//
// ALGORITHM (single radix-4 butterfly, no bit-reversal at size 4)
// --------------------------------------------------------------
//   t0 = x0 + x2   t1 = x0 - x2   t2 = x1 + x3   t3 = x1 - x3
//   y0 = t0 + t2   y1 = t1 - i*t3   y2 = t0 - t2   y3 = t1 + i*t3
// (inverse: -i <-> +i, plus 1/4 normalisation)
//
// Stage 1 broadcasts lane pairs so that both butterflies land element-wise:
//   A = [x0,x0,x1,x1]  B = [x2,x2,x3,x3]
//   t = A + (B ^ negate-lanes-1-and-3) = [t0, t1, t2, t3]
// Stage 2 does the same trick one level up, and folds the -i multiply into a
// selective real/imag swap plus one sign mask:
//   P = [t0,t1,t0,t1]
//   Q = [t2, swap(t3), t2, swap(t3)] ^ sign2  = [t2, -i*t3, -t2, +i*t3]
//   y = P + Q = [y0, y1, y2, y3]
// No multiply at all in the forward direction (the inverse has the single 1/4
// scaling multiply).
//
// INSTRUCTION SET
// ---------------
// AVX512F only. The sign flips use VPXORQ, not VXORPD: the EVEX forms of
// VXORPD/VXORPS are AVX512DQ, while the gate
// (golang.org/x/sys/cpu.X86.HasAVX512) is CPUID leaf 7 EBX bit 16 = AVX512F.
// Bit-identical here; D/Q vs PS/PD only changes masking granularity, and no
// masking is used.
//
// ===========================================================================

#include "textflag.h"

// Negate lanes 1 and 3 (doubles 2,3,6,7): turns an add into the (sum, diff)
// pair of stage 1.
DATA zneg4L13<>+0(SB)/8, $0x0000000000000000
DATA zneg4L13<>+8(SB)/8, $0x0000000000000000
DATA zneg4L13<>+16(SB)/8, $0x8000000000000000
DATA zneg4L13<>+24(SB)/8, $0x8000000000000000
DATA zneg4L13<>+32(SB)/8, $0x0000000000000000
DATA zneg4L13<>+40(SB)/8, $0x0000000000000000
DATA zneg4L13<>+48(SB)/8, $0x8000000000000000
DATA zneg4L13<>+56(SB)/8, $0x8000000000000000
GLOBL zneg4L13<>(SB), RODATA|NOPTR, $64

// Forward stage 2 sign mask, applied to [t2, swap(t3), t2, swap(t3)]:
//   lane0: keep  t2
//   lane1: negate double 3        -> (t3.im, -t3.re) = -i*t3
//   lane2: negate doubles 4,5     -> -t2
//   lane3: negate double 6        -> (-t3.im, t3.re) = +i*t3
DATA zsign4F<>+0(SB)/8, $0x0000000000000000
DATA zsign4F<>+8(SB)/8, $0x0000000000000000
DATA zsign4F<>+16(SB)/8, $0x0000000000000000
DATA zsign4F<>+24(SB)/8, $0x8000000000000000
DATA zsign4F<>+32(SB)/8, $0x8000000000000000
DATA zsign4F<>+40(SB)/8, $0x8000000000000000
DATA zsign4F<>+48(SB)/8, $0x8000000000000000
DATA zsign4F<>+56(SB)/8, $0x0000000000000000
GLOBL zsign4F<>(SB), RODATA|NOPTR, $64

// Inverse stage 2 sign mask: +i in lane 1, -i in lane 3.
DATA zsign4I<>+0(SB)/8, $0x0000000000000000
DATA zsign4I<>+8(SB)/8, $0x0000000000000000
DATA zsign4I<>+16(SB)/8, $0x8000000000000000
DATA zsign4I<>+24(SB)/8, $0x0000000000000000
DATA zsign4I<>+32(SB)/8, $0x8000000000000000
DATA zsign4I<>+40(SB)/8, $0x8000000000000000
DATA zsign4I<>+48(SB)/8, $0x0000000000000000
DATA zsign4I<>+56(SB)/8, $0x8000000000000000
GLOBL zsign4I<>(SB), RODATA|NOPTR, $64

// ===========================================================================
// Forward transform, size 4, complex128, radix-4 (AVX-512)
// ===========================================================================
TEXT ·ForwardAVX512Size4Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8 = dst pointer
	MOVQ src+24(FP), R9      // R9 = src pointer
	MOVQ src_len+32(FP), R13 // R13 = n

	// Verify n == 4
	CMPQ R13, $4
	JNE  size4_512_r4_fwd_return_false

	// Validate all slice lengths >= 4
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $4
	JL   size4_512_r4_fwd_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $4
	JL   size4_512_r4_fwd_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $4
	JL   size4_512_r4_fwd_return_false

	VMOVUPD 0(R9), Z0 // Z0 = [x0, x1, x2, x3]

	// Stage 1: t = [x0+x2, x0-x2, x1+x3, x1-x3]
	VSHUFF64X2 $0x50, Z0, Z0, Z1  // Z1 = [x0, x0, x1, x1]
	VSHUFF64X2 $0xFA, Z0, Z0, Z2  // Z2 = [x2, x2, x3, x3]
	VPXORQ     zneg4L13<>(SB), Z2, Z2 // Z2 = [x2, -x2, x3, -x3]
	VADDPD     Z2, Z1, Z3         // Z3 = t = [t0, t1, t2, t3]

	// Stage 2: y = [t0+t2, t1-i*t3, t0-t2, t1+i*t3]
	VSHUFF64X2 $0x44, Z3, Z3, Z4  // Z4 = P = [t0, t1, t0, t1]
	VSHUFF64X2 $0xEE, Z3, Z3, Z5  // Z5 = [t2, t3, t2, t3]
	VPERMILPD  $0x66, Z5, Z5      // Z5 = [t2, swap(t3), t2, swap(t3)]
	VPXORQ     zsign4F<>(SB), Z5, Z5 // Z5 = Q = [t2, -i*t3, -t2, +i*t3]
	VADDPD     Z5, Z4, Z6         // Z6 = y = P + Q

	VMOVUPD Z6, 0(R8) // store y[0..3]

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size4_512_r4_fwd_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 4, complex128, radix-4 (AVX-512)
// ===========================================================================
TEXT ·InverseAVX512Size4Radix4Complex128Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8 = dst pointer
	MOVQ src+24(FP), R9      // R9 = src pointer
	MOVQ src_len+32(FP), R13 // R13 = n

	// Verify n == 4
	CMPQ R13, $4
	JNE  size4_512_r4_inv_return_false

	// Validate all slice lengths >= 4
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $4
	JL   size4_512_r4_inv_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $4
	JL   size4_512_r4_inv_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $4
	JL   size4_512_r4_inv_return_false

	VMOVUPD 0(R9), Z0 // Z0 = [X0, X1, X2, X3]

	// Stage 1: t = [X0+X2, X0-X2, X1+X3, X1-X3]
	VSHUFF64X2 $0x50, Z0, Z0, Z1  // Z1 = [X0, X0, X1, X1]
	VSHUFF64X2 $0xFA, Z0, Z0, Z2  // Z2 = [X2, X2, X3, X3]
	VPXORQ     zneg4L13<>(SB), Z2, Z2 // Z2 = [X2, -X2, X3, -X3]
	VADDPD     Z2, Z1, Z3         // Z3 = t = [t0, t1, t2, t3]

	// Stage 2 with the conjugate twist: y = [t0+t2, t1+i*t3, t0-t2, t1-i*t3]
	VSHUFF64X2 $0x44, Z3, Z3, Z4  // Z4 = P = [t0, t1, t0, t1]
	VSHUFF64X2 $0xEE, Z3, Z3, Z5  // Z5 = [t2, t3, t2, t3]
	VPERMILPD  $0x66, Z5, Z5      // Z5 = [t2, swap(t3), t2, swap(t3)]
	VPXORQ     zsign4I<>(SB), Z5, Z5 // Z5 = Q = [t2, +i*t3, -t2, -i*t3]
	VADDPD     Z5, Z4, Z6         // Z6 = y = P + Q

	VMULPD.BCST ·quarter64(SB), Z6, Z6 // Z6 = y/4 (1/N normalisation, exact)

	VMOVUPD Z6, 0(R8) // store x[0..3]

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size4_512_r4_inv_return_false:
	MOVB $0, ret+96(FP)
	RET
