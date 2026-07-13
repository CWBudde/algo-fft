//go:build amd64 && !purego

// ===========================================================================
// AVX-512 optimized FFT Assembly for AMD64 - complex128 (float64)
// ===========================================================================
//
// Generic radix-2 DIT FFT using AVX-512F for complex128 data. Mirrors
// avx512_f32_generic.s with double-precision lane recipes:
//   - ZMM (512-bit) holds 4 complex128 values -> 4 butterflies/iteration
//     (vs 2 with the AVX2 kernel).
//   - Element size is 16 bytes, so data offsets shift by 4 (not 3).
//
// COMPLEX MULTIPLICATION (per 512-bit vector, 4 complex128)
// --------------------------------------------------------
// Forward t = w * b:
//   VMOVDDUP       w -> [w.r, w.r, ...] (duplicate even doubles)
//   VSHUFPD $0xFF  w -> [w.i, w.i, ...] (duplicate odd doubles)
//   VSHUFPD $0x55  b -> [b.i, b.r, ...] (swap within each pair)
//   VMULPD then VFMADDSUB231PD: even lanes b.r*w.r - b.i*w.i (real),
//   odd lanes b.i*w.r + b.r*w.i (imag)
// Inverse t = conj(w) * b: identical, with VFMSUBADD231PD.
//
// TWIDDLE ACCESS
// --------------
//   - Contiguous (step == 1): single VMOVUPD of 4 twiddles.
//   - Strided (step > 1): 4 XMM loads packed via VINSERTF128 into two YMM
//     halves, merged with VINSERTF64X4.
//
// PERFORMANCE NOTES
// -----------------
//   - Minimum size: n >= 16; vector paths require half >= 4, so stages with
//     half < 4 run the scalar path and vector loops leave no remainder.
//   - Scalar butterflies use VEX-encoded 128-bit FMA.
//   - VZEROUPPER before every RET; in-place transforms work in scratch.
//
// ===========================================================================

#include "textflag.h"

// Function signature:
//   func ForwardAVX512Complex128Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
//
// Frame layout identical to the complex64 variant (bitrev at FP+96,
// ret at FP+120). Register allocation identical to avx512_f32_generic.s.
// The bit-reversal permutation reads the precomputed bitrev table, like the
// AVX2 complex128 kernel.

// ===========================================================================
// ForwardAVX512Complex128Asm - Forward FFT for complex128 using AVX-512
// ===========================================================================
TEXT ·ForwardAVX512Complex128Asm(SB), NOSPLIT, $0-121
	// PHASE 1: Load parameters and validate inputs
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src_len+32(FP), R13 // R13 = n = len(src)

	TESTQ R13, R13
	JZ    return_true

	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, R13
	JL   return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   return_false

	MOVQ bitrev_len+104(FP), AX
	CMPQ AX, R13
	JL   return_false        // bitrev table too short

	CMPQ R13, $1
	JNE  check_power_of_2
	MOVUPS (R9), X0          // n == 1: copy single complex128 (16 bytes)
	MOVUPS X0, (R8)
	JMP  return_true

check_power_of_2:
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  return_false

	CMPQ R13, $16
	JL   return_false        // too small for the AVX-512 path

	// PHASE 2: Select working buffer (in-place uses scratch)
	CMPQ R8, R9
	JNE  do_bit_reversal
	MOVQ R11, R8             // in-place: work = scratch

do_bit_reversal:
	// PHASE 3: Bit-reversal permutation work[i] = src[bitrev[i]]
	MOVQ bitrev+96(FP), R12  // R12 = bit-reversal table pointer
	XORQ CX, CX              // i = 0
	XORQ SI, SI              // SI = i * 16 (destination byte offset)

bitrev_loop:
	CMPQ CX, R13
	JGE  bitrev_done
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	SHLQ $4, DX              // DX = bitrev[i] * 16 bytes
	MOVUPS (R9)(DX*1), X0    // X0 = src[bitrev[i]]
	MOVUPS X0, (R8)(SI*1)    // work[i] = src[bitrev[i]]
	ADDQ $16, SI
	INCQ CX
	JMP  bitrev_loop

bitrev_done:
	// PHASE 4: DIT butterfly stages
	MOVQ $2, R14             // size = 2

size_loop:
	CMPQ R14, R13
	JG   transform_done

	MOVQ R14, R15
	SHRQ $1, R15             // half = size / 2

	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX              // step = n / size

	XORQ CX, CX              // base = 0

base_loop:
	CMPQ CX, R13
	JGE  next_size

	CMPQ R15, $4
	JL   scalar_butterflies  // not enough butterflies for a ZMM vector

	CMPQ BX, $1
	JE   avx512_contiguous
	JMP  avx512_strided

avx512_contiguous:
	// Contiguous twiddles (step == 1): 4 butterflies per iteration
	XORQ DX, DX              // j = 0

avx512_loop:
	MOVQ R15, AX
	SUBQ DX, AX              // remaining = half - j
	CMPQ AX, $4
	JL   scalar_remainder

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI              // SI = (base + j) * 16 bytes

	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI              // DI = (base + j + half) * 16 bytes

	VMOVUPD (R8)(SI*1), Z0   // Z0 = a[j .. j+3]
	VMOVUPD (R8)(DI*1), Z1   // Z1 = b[j .. j+3]

	MOVQ DX, AX
	SHLQ $4, AX              // AX = j * 16 bytes
	VMOVUPD (R10)(AX*1), Z2  // Z2 = w[j .. j+3]

	// Butterfly: t = w * b, a' = a + t, b' = a - t
	VMOVDDUP Z2, Z3           // Z3 = [w.r, w.r, ...]
	VSHUFPD $0xFF, Z2, Z2, Z4 // Z4 = [w.i, w.i, ...]
	VSHUFPD $0x55, Z1, Z1, Z6 // Z6 = [b.i, b.r, ...] (swap pairs)
	VMULPD Z4, Z6, Z6         // Z6 = [b.i*w.i, b.r*w.i, ...]
	VFMADDSUB231PD Z3, Z1, Z6 // Z6 = t = w * b (even -, odd +)
	VADDPD Z6, Z0, Z3         // Z3 = a' = a + t
	VSUBPD Z6, Z0, Z4         // Z4 = b' = a - t
	VMOVUPD Z3, (R8)(SI*1)    // store a'
	VMOVUPD Z4, (R8)(DI*1)    // store b'

	ADDQ $4, DX               // j += 4
	JMP  avx512_loop

avx512_strided:
	// Strided twiddles (step > 1): gather 4 twiddles per iteration
	MOVQ BX, R12
	SHLQ $4, R12             // R12 = step * 16 (stride in bytes)
	XORQ R11, R11            // R11 = twiddle byte offset
	XORQ DX, DX              // j = 0

avx512_strided_loop:
	MOVQ R15, AX
	SUBQ DX, AX              // remaining = half - j
	CMPQ AX, $4
	JL   scalar_remainder

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI              // SI = (base + j) * 16

	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI              // DI = (base + j + half) * 16

	VMOVUPD (R8)(SI*1), Z0   // Z0 = a[j .. j+3]
	VMOVUPD (R8)(DI*1), Z1   // Z1 = b[j .. j+3]

	// Gather w[j*step .. (j+3)*step] into Z2
	VMOVUPD (R10)(R11*1), X2     // w0
	LEAQ (R11)(R12*1), AX        // AX = offset + stride
	VMOVUPD (R10)(AX*1), X3      // w1
	VINSERTF128 $1, X3, Y2, Y2   // Y2 = [w0, w1]
	ADDQ R12, AX
	VMOVUPD (R10)(AX*1), X4      // w2
	ADDQ R12, AX
	VMOVUPD (R10)(AX*1), X5      // w3
	VINSERTF128 $1, X5, Y4, Y4   // Y4 = [w2, w3]
	VINSERTF64X4 $1, Y4, Z2, Z2  // Z2 = [w0, w1, w2, w3]

	// Butterfly (same as contiguous path)
	VMOVDDUP Z2, Z3
	VSHUFPD $0xFF, Z2, Z2, Z4
	VSHUFPD $0x55, Z1, Z1, Z6
	VMULPD Z4, Z6, Z6
	VFMADDSUB231PD Z3, Z1, Z6    // t = w * b
	VADDPD Z6, Z0, Z3
	VSUBPD Z6, Z0, Z4
	VMOVUPD Z3, (R8)(SI*1)
	VMOVUPD Z4, (R8)(DI*1)

	LEAQ (R11)(R12*4), R11       // twiddle offset += 4 * stride
	ADDQ $4, DX                  // j += 4
	JMP  avx512_strided_loop

scalar_remainder:
	// Leftover butterflies after a vector loop (never taken when half >= 4,
	// kept for structural parity with the AVX2 kernel).
	CMPQ DX, R15
	JGE  next_base

scalar_remainder_loop:
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI              // SI = (base + j) * 16

	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI              // DI = (base + j + half) * 16

	MOVUPD (R8)(SI*1), X0    // a
	MOVUPD (R8)(DI*1), X1    // b

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2   // w = twiddle[j*step]

	// t = w * b via 128-bit VEX FMA
	VMOVDDUP X2, X3           // [w.r, w.r]
	VSHUFPD $0x3, X2, X2, X4  // [w.i, w.i]
	VSHUFPD $0x1, X1, X1, X6  // [b.i, b.r]
	VMULPD X4, X6, X6         // [b.i*w.i, b.r*w.i]
	VFMADDSUB231PD X3, X1, X6 // t = w * b
	VADDPD X6, X0, X3         // a'
	VSUBPD X6, X0, X4         // b'

	MOVUPD X3, (R8)(SI*1)
	MOVUPD X4, (R8)(DI*1)

	INCQ DX
	CMPQ DX, R15
	JL   scalar_remainder_loop

next_base:
	ADDQ R14, CX             // base += size
	JMP  base_loop

scalar_butterflies:
	// Pure scalar path for stages with half < 4
	XORQ DX, DX              // j = 0

scalar_loop:
	CMPQ DX, R15
	JGE  next_base

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI

	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI

	MOVUPD (R8)(SI*1), X0    // a
	MOVUPD (R8)(DI*1), X1    // b

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2   // w

	VMOVDDUP X2, X3
	VSHUFPD $0x3, X2, X2, X4
	VSHUFPD $0x1, X1, X1, X6
	VMULPD X4, X6, X6
	VFMADDSUB231PD X3, X1, X6 // t = w * b
	VADDPD X6, X0, X3         // a'
	VSUBPD X6, X0, X4         // b'

	MOVUPD X3, (R8)(SI*1)
	MOVUPD X4, (R8)(DI*1)

	INCQ DX
	JMP  scalar_loop

next_size:
	SHLQ $1, R14             // size *= 2
	JMP  size_loop

transform_done:
	// PHASE 5: Copy back to dst if we worked in scratch (in-place case)
	VZEROUPPER

	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   return_true

	XORQ CX, CX              // CX = byte offset
	MOVQ R13, DX
	SHLQ $4, DX              // DX = n * 16 = end byte offset

copy_loop:
	CMPQ CX, DX
	JGE  return_true
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (AX)(CX*1)
	ADDQ $16, CX
	JMP  copy_loop

return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// InverseAVX512Complex128Asm - Inverse FFT for complex128 using AVX-512
// ===========================================================================
// Same structure as the forward transform with two differences:
//   1. Conjugate twiddles: VFMSUBADD231PD instead of VFMADDSUB231PD
//   2. Output scaled by 1/n
// ===========================================================================
TEXT ·InverseAVX512Complex128Asm(SB), NOSPLIT, $0-121
	// PHASE 1: Load parameters and validate inputs
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src_len+32(FP), R13

	TESTQ R13, R13
	JZ    inv_return_true

	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	MOVQ bitrev_len+104(FP), AX
	CMPQ AX, R13
	JL   inv_return_false    // bitrev table too short

	CMPQ R13, $1
	JNE  inv_check_power_of_2
	MOVUPS (R9), X0
	MOVUPS X0, (R8)
	JMP  inv_return_true

inv_check_power_of_2:
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  inv_return_false

	CMPQ R13, $16
	JL   inv_return_false

	// PHASE 2: Select working buffer
	CMPQ R8, R9
	JNE  inv_do_bit_reversal
	MOVQ R11, R8             // in-place: work = scratch

inv_do_bit_reversal:
	// PHASE 3: Bit-reversal permutation work[i] = src[bitrev[i]]
	MOVQ bitrev+96(FP), R12  // R12 = bit-reversal table pointer
	XORQ CX, CX              // i = 0
	XORQ SI, SI              // SI = i * 16 (destination byte offset)

inv_bitrev_loop:
	CMPQ CX, R13
	JGE  inv_bitrev_done
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	SHLQ $4, DX              // DX = bitrev[i] * 16 bytes
	MOVUPS (R9)(DX*1), X0    // X0 = src[bitrev[i]]
	MOVUPS X0, (R8)(SI*1)    // work[i] = src[bitrev[i]]
	ADDQ $16, SI
	INCQ CX
	JMP  inv_bitrev_loop

inv_bitrev_done:
	// PHASE 4: DIT butterfly stages with conjugate twiddles
	MOVQ $2, R14

inv_size_loop:
	CMPQ R14, R13
	JG   inv_transform_done

	MOVQ R14, R15
	SHRQ $1, R15             // half = size / 2

	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14
	MOVQ AX, BX              // step = n / size

	XORQ CX, CX              // base = 0

inv_base_loop:
	CMPQ CX, R13
	JGE  inv_next_size

	CMPQ R15, $4
	JL   inv_scalar_butterflies

	CMPQ BX, $1
	JE   inv_avx512_contiguous
	JMP  inv_avx512_strided

inv_avx512_contiguous:
	XORQ DX, DX

inv_avx512_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   inv_scalar_remainder

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI

	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI

	VMOVUPD (R8)(SI*1), Z0   // a
	VMOVUPD (R8)(DI*1), Z1   // b

	MOVQ DX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), Z2  // w

	// Conjugate butterfly: t = conj(w) * b
	VMOVDDUP Z2, Z3
	VSHUFPD $0xFF, Z2, Z2, Z4
	VSHUFPD $0x55, Z1, Z1, Z6
	VMULPD Z4, Z6, Z6
	VFMSUBADD231PD Z3, Z1, Z6 // t = conj(w) * b (even +, odd -)
	VADDPD Z6, Z0, Z3         // a'
	VSUBPD Z6, Z0, Z4         // b'
	VMOVUPD Z3, (R8)(SI*1)
	VMOVUPD Z4, (R8)(DI*1)

	ADDQ $4, DX
	JMP  inv_avx512_loop

inv_avx512_strided:
	MOVQ BX, R12
	SHLQ $4, R12             // stride bytes = step * 16
	XORQ R11, R11
	XORQ DX, DX

inv_avx512_strided_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   inv_scalar_remainder

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI

	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI

	VMOVUPD (R8)(SI*1), Z0
	VMOVUPD (R8)(DI*1), Z1

	// Gather 4 strided twiddles (same recipe as forward)
	VMOVUPD (R10)(R11*1), X2
	LEAQ (R11)(R12*1), AX
	VMOVUPD (R10)(AX*1), X3
	VINSERTF128 $1, X3, Y2, Y2
	ADDQ R12, AX
	VMOVUPD (R10)(AX*1), X4
	ADDQ R12, AX
	VMOVUPD (R10)(AX*1), X5
	VINSERTF128 $1, X5, Y4, Y4
	VINSERTF64X4 $1, Y4, Z2, Z2

	VMOVDDUP Z2, Z3
	VSHUFPD $0xFF, Z2, Z2, Z4
	VSHUFPD $0x55, Z1, Z1, Z6
	VMULPD Z4, Z6, Z6
	VFMSUBADD231PD Z3, Z1, Z6 // t = conj(w) * b
	VADDPD Z6, Z0, Z3
	VSUBPD Z6, Z0, Z4
	VMOVUPD Z3, (R8)(SI*1)
	VMOVUPD Z4, (R8)(DI*1)

	LEAQ (R11)(R12*4), R11
	ADDQ $4, DX
	JMP  inv_avx512_strided_loop

inv_scalar_remainder:
	CMPQ DX, R15
	JGE  inv_next_base

inv_scalar_remainder_loop:
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI

	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI

	MOVUPD (R8)(SI*1), X0    // a
	MOVUPD (R8)(DI*1), X1    // b

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2   // w

	// t = conj(w) * b via 128-bit VEX FMA
	VMOVDDUP X2, X3
	VSHUFPD $0x3, X2, X2, X4
	VSHUFPD $0x1, X1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6 // t = conj(w) * b
	VADDPD X6, X0, X3         // a'
	VSUBPD X6, X0, X4         // b'

	MOVUPD X3, (R8)(SI*1)
	MOVUPD X4, (R8)(DI*1)

	INCQ DX
	CMPQ DX, R15
	JL   inv_scalar_remainder_loop

inv_next_base:
	ADDQ R14, CX
	JMP  inv_base_loop

inv_scalar_butterflies:
	XORQ DX, DX

inv_scalar_loop:
	CMPQ DX, R15
	JGE  inv_next_base

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI

	MOVQ R15, DI
	SHLQ $4, DI
	ADDQ SI, DI

	MOVUPD (R8)(SI*1), X0
	MOVUPD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	MOVUPD (R10)(AX*1), X2

	VMOVDDUP X2, X3
	VSHUFPD $0x3, X2, X2, X4
	VSHUFPD $0x1, X1, X1, X6
	VMULPD X4, X6, X6
	VFMSUBADD231PD X3, X1, X6 // t = conj(w) * b
	VADDPD X6, X0, X3
	VSUBPD X6, X0, X4

	MOVUPD X3, (R8)(SI*1)
	MOVUPD X4, (R8)(DI*1)

	INCQ DX
	JMP  inv_scalar_loop

inv_next_size:
	SHLQ $1, R14
	JMP  inv_size_loop

inv_transform_done:
	// PHASE 5: Copy back (if needed), then scale by 1/n
	VZEROUPPER

	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   inv_scale

	XORQ CX, CX              // CX = byte offset
	MOVQ R13, DX
	SHLQ $4, DX              // DX = n * 16 = end byte offset

inv_copy_loop:
	CMPQ CX, DX
	JGE  inv_scale
	MOVUPS (R8)(CX*1), X0
	MOVUPS X0, (AX)(CX*1)
	ADDQ $16, CX
	JMP  inv_copy_loop

inv_scale:
	// PHASE 6: Scale output by 1/n (4 complex128 per iteration; n is a
	// power of two >= 16, so there is never a remainder)
	MOVQ dst+0(FP), R8

	CVTSQ2SD R13, X0         // X0 = (float64)n
	MOVSD ·one64(SB), X1     // X1 = 1.0
	DIVSD X0, X1             // X1 = 1.0 / n
	VBROADCASTSD X1, Z1      // Z1 = [scale x8]

	XORQ CX, CX              // CX = byte offset
	MOVQ R13, DX
	SHLQ $4, DX              // DX = n * 16 = end byte offset

inv_scale_loop:
	CMPQ CX, DX
	JGE  inv_return_true
	VMOVUPD (R8)(CX*1), Z0   // load 4 complex128
	VMULPD Z1, Z0, Z0        // scale real and imag lanes
	VMOVUPD Z0, (R8)(CX*1)
	ADDQ $64, CX             // offset += 64 bytes
	JMP  inv_scale_loop

inv_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_return_false:
	MOVB $0, ret+120(FP)
	RET
