//go:build amd64 && !purego

// ===========================================================================
// AVX-512 optimized FFT Assembly for AMD64 - complex64 (float32)
// ===========================================================================
//
// This file implements the generic radix-2 DIT FFT using AVX-512F
// instructions for complex64 (single-precision) data.
//
// The structure mirrors avx2_f32_generic.s, widened from YMM to ZMM:
//   - ZMM (512-bit) holds 8 complex64 values, so the vector paths process
//     8 butterflies per iteration (vs 4 with AVX2).
//   - All EVEX instructions used here require only AVX512F, which is the
//     feature gate (cpu.Features.HasAVX512) checked by the dispatcher.
//
// ALGORITHM
// ---------
//   1. Bit-reversal permutation: work[i] = src[bitrev(i)]
//   2. Butterfly stages for size = 2, 4, ..., n:
//        a' = a + w*b, b' = a - w*b
//   3. Inverse uses conjugate twiddles (VFMSUBADD instead of VFMADDSUB)
//      and scales the output by 1/n.
//
// COMPLEX MULTIPLICATION (per 512-bit vector, 8 complex64)
// --------------------------------------------------------
// Forward t = w * b:
//   VMOVSLDUP  w -> [w.r, w.r, ...]
//   VMOVSHDUP  w -> [w.i, w.i, ...]
//   VSHUFPS $0xB1 b -> [b.i, b.r, ...] (swap within each pair)
//   VMULPS then VFMADDSUB231PS: even lanes b.r*w.r - b.i*w.i (real),
//   odd lanes b.i*w.r + b.r*w.i (imag)
// Inverse t = conj(w) * b: identical, with VFMSUBADD231PS.
//
// TWIDDLE ACCESS
// --------------
//   - Contiguous (step == 1): single VMOVUPS of 8 twiddles.
//   - Strided (step > 1): 8 scalar VMOVSD loads packed via VPUNPCKLQDQ +
//     VINSERTF128 into two YMM halves, merged with VINSERTF64X4.
//
// PERFORMANCE NOTES
// -----------------
//   - Minimum size: n >= 16 (smaller sizes return false -> Go/AVX2 fallback)
//   - Vector paths require half >= 8; since half is a power of two, the
//     vector loops never leave a remainder (half % 8 == 0 when half >= 8).
//     Stages with half < 8 run the scalar path.
//   - Scalar butterflies use VEX-encoded 128-bit FMA (any AVX-512 CPU has
//     FMA3), avoiding the SSE blend dance of the AVX2 kernel.
//   - VZEROUPPER before every RET to avoid AVX-SSE transition penalties.
//   - In-place transforms (dst == src) use scratch as the working buffer.
//
// ===========================================================================

#include "textflag.h"

// Function signature:
//   func ForwardAVX512Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
//
// Stack frame layout (offsets from FP):
//   dst:     FP+0  (ptr), FP+8   (len), FP+16  (cap)
//   src:     FP+24 (ptr), FP+32  (len), FP+40  (cap)
//   twiddle: FP+48 (ptr), FP+56  (len), FP+64  (cap)
//   scratch: FP+72 (ptr), FP+80  (len), FP+88  (cap)
//   bitrev:  FP+96 (ptr), FP+104 (len), FP+112 (cap)
//   return:  FP+120 (bool)
//
// Register allocation (identical to avx2_f32_generic.s):
//   R8  work buffer (dst or scratch)   R14 size (2, 4, ..., n)
//   R9  src pointer                    R15 half = size/2
//   R10 twiddle pointer                BX  step = n/size
//   R11 twiddle byte offset (strided)  CX  base
//   R12 stride bytes (strided)         DX  j
//   R13 n                              SI/DI data byte offsets, AX scratch
//
// The bit-reversal permutation reads the precomputed bitrev table (shared
// per-size via the internal/fft cache); computing it on the fly like the
// AVX2 complex64 kernel costs ~30% at n=1024.

// ===========================================================================
// ForwardAVX512Complex64Asm - Forward FFT for complex64 using AVX-512
// ===========================================================================
TEXT ·ForwardAVX512Complex64Asm(SB), NOSPLIT, $0-121
	// PHASE 1: Load parameters and validate inputs
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src_len+32(FP), R13 // R13 = n = len(src)

	TESTQ R13, R13
	JZ    return_true        // empty input is a valid no-op

	MOVQ dst_len+8(FP), AX
	CMPQ AX, R13
	JL   return_false        // dst too short

	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, R13
	JL   return_false        // twiddle too short

	MOVQ scratch_len+80(FP), AX
	CMPQ AX, R13
	JL   return_false        // scratch too short

	MOVQ bitrev_len+104(FP), AX
	CMPQ AX, R13
	JL   return_false        // bitrev table too short

	CMPQ R13, $1
	JNE  check_power_of_2
	MOVQ (R9), AX            // n == 1: copy single element
	MOVQ AX, (R8)
	JMP  return_true

check_power_of_2:
	MOVQ R13, AX
	LEAQ -1(AX), BX          // BX = n - 1
	TESTQ AX, BX             // ZF set iff n is a power of 2
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
	XORQ CX, CX              // CX = i = 0

bitrev_loop:
	CMPQ CX, R13
	JGE  bitrev_done
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	JMP  bitrev_loop

bitrev_done:
	// PHASE 4: DIT butterfly stages
	MOVQ $2, R14             // size = 2

size_loop:
	CMPQ R14, R13
	JG   transform_done      // done when size > n

	MOVQ R14, R15
	SHRQ $1, R15             // half = size / 2

	MOVQ R13, AX
	XORQ DX, DX
	DIVQ R14                 // AX = n / size
	MOVQ AX, BX              // BX = step

	XORQ CX, CX              // base = 0

base_loop:
	CMPQ CX, R13
	JGE  next_size

	CMPQ R15, $8
	JL   scalar_butterflies  // not enough butterflies for a ZMM vector

	CMPQ BX, $1
	JE   avx512_contiguous
	JMP  avx512_strided

avx512_contiguous:
	// Contiguous twiddles (step == 1): 8 butterflies per iteration
	XORQ DX, DX              // j = 0

avx512_loop:
	MOVQ R15, AX
	SUBQ DX, AX              // remaining = half - j
	CMPQ AX, $8
	JL   scalar_remainder

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI              // SI = (base + j) * 8 bytes

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI              // DI = (base + j + half) * 8 bytes

	VMOVUPS (R8)(SI*1), Z0   // Z0 = a[j .. j+7]
	VMOVUPS (R8)(DI*1), Z1   // Z1 = b[j .. j+7]

	MOVQ DX, AX
	SHLQ $3, AX              // AX = j * 8 bytes
	VMOVUPS (R10)(AX*1), Z2  // Z2 = w[j .. j+7]

	// Butterfly: t = w * b, a' = a + t, b' = a - t
	VMOVSLDUP Z2, Z3          // Z3 = [w.r, w.r, ...]
	VMOVSHDUP Z2, Z4          // Z4 = [w.i, w.i, ...]
	VSHUFPS $0xB1, Z1, Z1, Z6 // Z6 = [b.i, b.r, ...] (swap pairs)
	VMULPS Z4, Z6, Z6         // Z6 = [b.i*w.i, b.r*w.i, ...]
	VFMADDSUB231PS Z3, Z1, Z6 // Z6 = t = w * b (even -, odd +)
	VADDPS Z6, Z0, Z3         // Z3 = a' = a + t
	VSUBPS Z6, Z0, Z4         // Z4 = b' = a - t
	VMOVUPS Z3, (R8)(SI*1)    // store a'
	VMOVUPS Z4, (R8)(DI*1)    // store b'

	ADDQ $8, DX               // j += 8
	JMP  avx512_loop

avx512_strided:
	// Strided twiddles (step > 1): gather 8 twiddles per iteration
	MOVQ BX, R12
	SHLQ $3, R12             // R12 = step * 8 (stride in bytes)
	XORQ R11, R11            // R11 = twiddle byte offset
	XORQ DX, DX              // j = 0

avx512_strided_loop:
	MOVQ R15, AX
	SUBQ DX, AX              // remaining = half - j
	CMPQ AX, $8
	JL   scalar_remainder

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI              // SI = (base + j) * 8

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI              // DI = (base + j + half) * 8

	VMOVUPS (R8)(SI*1), Z0   // Z0 = a[j .. j+7]
	VMOVUPS (R8)(DI*1), Z1   // Z1 = b[j .. j+7]

	// Gather w[j*step .. (j+7)*step] into Z2:
	// low half w0..w3 packed into Y2, high half w4..w7 into Y7,
	// merged with VINSERTF64X4.
	VMOVSD (R10)(R11*1), X2      // w0
	LEAQ (R11)(R12*1), AX        // AX = offset + stride
	VMOVSD (R10)(AX*1), X3       // w1
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X4       // w2
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X5       // w3
	VPUNPCKLQDQ X3, X2, X2       // X2 = [w0, w1]
	VPUNPCKLQDQ X5, X4, X4       // X4 = [w2, w3]
	VINSERTF128 $1, X4, Y2, Y2   // Y2 = [w0, w1, w2, w3]

	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X7       // w4
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X8       // w5
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X9       // w6
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X10      // w7
	VPUNPCKLQDQ X8, X7, X7       // X7 = [w4, w5]
	VPUNPCKLQDQ X10, X9, X9      // X9 = [w6, w7]
	VINSERTF128 $1, X9, Y7, Y7   // Y7 = [w4, w5, w6, w7]

	VINSERTF64X4 $1, Y7, Z2, Z2  // Z2 = [w0 .. w7]

	// Butterfly (same as contiguous path)
	VMOVSLDUP Z2, Z3
	VMOVSHDUP Z2, Z4
	VSHUFPS $0xB1, Z1, Z1, Z6
	VMULPS Z4, Z6, Z6
	VFMADDSUB231PS Z3, Z1, Z6    // t = w * b
	VADDPS Z6, Z0, Z3
	VSUBPS Z6, Z0, Z4
	VMOVUPS Z3, (R8)(SI*1)
	VMOVUPS Z4, (R8)(DI*1)

	LEAQ (R11)(R12*8), R11       // twiddle offset += 8 * stride
	ADDQ $8, DX                  // j += 8
	JMP  avx512_strided_loop

scalar_remainder:
	// Leftover butterflies after a vector loop (never taken when half >= 8,
	// kept for structural parity with the AVX2 kernel).
	CMPQ DX, R15
	JGE  next_base

scalar_remainder_loop:
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI              // SI = (base + j) * 8

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI              // DI = (base + j + half) * 8

	VMOVSD (R8)(SI*1), X0     // a
	VMOVSD (R8)(DI*1), X1     // b

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	VMOVSD (R10)(AX*1), X2    // w = twiddle[j*step]

	// t = w * b via 128-bit VEX FMA (same lane recipe as the vector path)
	VMOVSLDUP X2, X3          // [w.r, w.r]
	VMOVSHDUP X2, X4          // [w.i, w.i]
	VSHUFPS $0xB1, X1, X1, X6 // [b.i, b.r]
	VMULPS X4, X6, X6         // [b.i*w.i, b.r*w.i]
	VFMADDSUB231PS X3, X1, X6 // t = w * b
	VADDPS X6, X0, X3         // a'
	VSUBPS X6, X0, X4         // b'

	VMOVSD X3, (R8)(SI*1)
	VMOVSD X4, (R8)(DI*1)

	INCQ DX
	CMPQ DX, R15
	JL   scalar_remainder_loop

next_base:
	ADDQ R14, CX             // base += size
	JMP  base_loop

scalar_butterflies:
	// Pure scalar path for stages with half < 8
	XORQ DX, DX              // j = 0

scalar_loop:
	CMPQ DX, R15
	JGE  next_base

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	VMOVSD (R8)(SI*1), X0     // a
	VMOVSD (R8)(DI*1), X1     // b

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	VMOVSD (R10)(AX*1), X2    // w

	VMOVSLDUP X2, X3
	VMOVSHDUP X2, X4
	VSHUFPS $0xB1, X1, X1, X6
	VMULPS X4, X6, X6
	VFMADDSUB231PS X3, X1, X6 // t = w * b
	VADDPS X6, X0, X3         // a'
	VSUBPS X6, X0, X4         // b'

	VMOVSD X3, (R8)(SI*1)
	VMOVSD X4, (R8)(DI*1)

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

	XORQ CX, CX

copy_loop:
	CMPQ CX, R13
	JGE  return_true
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  copy_loop

return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

return_false:
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// InverseAVX512Complex64Asm - Inverse FFT for complex64 using AVX-512
// ===========================================================================
// Same structure as the forward transform with two differences:
//   1. Conjugate twiddles: VFMSUBADD231PS instead of VFMADDSUB231PS
//   2. Output scaled by 1/n
// ===========================================================================
TEXT ·InverseAVX512Complex64Asm(SB), NOSPLIT, $0-121
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
	MOVQ (R9), AX
	MOVQ AX, (R8)
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
	XORQ CX, CX

inv_bitrev_loop:
	CMPQ CX, R13
	JGE  inv_bitrev_done
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
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

	CMPQ R15, $8
	JL   inv_scalar_butterflies

	CMPQ BX, $1
	JE   inv_avx512_contiguous
	JMP  inv_avx512_strided

inv_avx512_contiguous:
	XORQ DX, DX

inv_avx512_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $8
	JL   inv_scalar_remainder

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	VMOVUPS (R8)(SI*1), Z0   // a
	VMOVUPS (R8)(DI*1), Z1   // b

	MOVQ DX, AX
	SHLQ $3, AX
	VMOVUPS (R10)(AX*1), Z2  // w

	// Conjugate butterfly: t = conj(w) * b
	VMOVSLDUP Z2, Z3
	VMOVSHDUP Z2, Z4
	VSHUFPS $0xB1, Z1, Z1, Z6
	VMULPS Z4, Z6, Z6
	VFMSUBADD231PS Z3, Z1, Z6 // t = conj(w) * b (even +, odd -)
	VADDPS Z6, Z0, Z3         // a'
	VSUBPS Z6, Z0, Z4         // b'
	VMOVUPS Z3, (R8)(SI*1)
	VMOVUPS Z4, (R8)(DI*1)

	ADDQ $8, DX
	JMP  inv_avx512_loop

inv_avx512_strided:
	MOVQ BX, R12
	SHLQ $3, R12             // stride bytes = step * 8
	XORQ R11, R11
	XORQ DX, DX

inv_avx512_strided_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $8
	JL   inv_scalar_remainder

	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	VMOVUPS (R8)(SI*1), Z0
	VMOVUPS (R8)(DI*1), Z1

	// Gather 8 strided twiddles (same recipe as forward)
	VMOVSD (R10)(R11*1), X2
	LEAQ (R11)(R12*1), AX
	VMOVSD (R10)(AX*1), X3
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X4
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X5
	VPUNPCKLQDQ X3, X2, X2
	VPUNPCKLQDQ X5, X4, X4
	VINSERTF128 $1, X4, Y2, Y2

	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X7
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X8
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X9
	ADDQ R12, AX
	VMOVSD (R10)(AX*1), X10
	VPUNPCKLQDQ X8, X7, X7
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y7, Y7

	VINSERTF64X4 $1, Y7, Z2, Z2

	VMOVSLDUP Z2, Z3
	VMOVSHDUP Z2, Z4
	VSHUFPS $0xB1, Z1, Z1, Z6
	VMULPS Z4, Z6, Z6
	VFMSUBADD231PS Z3, Z1, Z6 // t = conj(w) * b
	VADDPS Z6, Z0, Z3
	VSUBPS Z6, Z0, Z4
	VMOVUPS Z3, (R8)(SI*1)
	VMOVUPS Z4, (R8)(DI*1)

	LEAQ (R11)(R12*8), R11
	ADDQ $8, DX
	JMP  inv_avx512_strided_loop

inv_scalar_remainder:
	CMPQ DX, R15
	JGE  inv_next_base

inv_scalar_remainder_loop:
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	VMOVSD (R8)(SI*1), X0     // a
	VMOVSD (R8)(DI*1), X1     // b

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	VMOVSD (R10)(AX*1), X2    // w

	// t = conj(w) * b via 128-bit VEX FMA
	VMOVSLDUP X2, X3
	VMOVSHDUP X2, X4
	VSHUFPS $0xB1, X1, X1, X6
	VMULPS X4, X6, X6
	VFMSUBADD231PS X3, X1, X6 // t = conj(w) * b
	VADDPS X6, X0, X3         // a'
	VSUBPS X6, X0, X4         // b'

	VMOVSD X3, (R8)(SI*1)
	VMOVSD X4, (R8)(DI*1)

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
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	VMOVSD (R8)(SI*1), X0
	VMOVSD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	VMOVSD (R10)(AX*1), X2

	VMOVSLDUP X2, X3
	VMOVSHDUP X2, X4
	VSHUFPS $0xB1, X1, X1, X6
	VMULPS X4, X6, X6
	VFMSUBADD231PS X3, X1, X6 // t = conj(w) * b
	VADDPS X6, X0, X3
	VSUBPS X6, X0, X4

	VMOVSD X3, (R8)(SI*1)
	VMOVSD X4, (R8)(DI*1)

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

	XORQ CX, CX

inv_copy_loop:
	CMPQ CX, R13
	JGE  inv_scale
	MOVQ (R8)(CX*8), DX
	MOVQ DX, (AX)(CX*8)
	INCQ CX
	JMP  inv_copy_loop

inv_scale:
	// PHASE 6: Scale output by 1/n (8 complex64 per iteration; n is a
	// power of two >= 16, so there is never a remainder)
	MOVQ dst+0(FP), R8

	VCVTSI2SSQ R13, X0, X0         // X0 = (float32)n
	VMOVSS   ·one32(SB), X1  // X1 = 1.0f
	VDIVSS   X0, X1, X1          // X1 = 1.0f / n
	VBROADCASTSS X1, Z1      // Z1 = [scale x16]

	XORQ CX, CX              // CX = byte offset
	MOVQ R13, DX
	SHLQ $3, DX              // DX = n * 8 = end byte offset

inv_scale_loop:
	CMPQ CX, DX
	JGE  inv_return_true
	VMOVUPS (R8)(CX*1), Z0   // load 8 complex64
	VMULPS Z1, Z0, Z0        // scale real and imag lanes
	VMOVUPS Z0, (R8)(CX*1)
	ADDQ $64, CX             // offset += 64 bytes
	JMP  inv_scale_loop

inv_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_return_false:
	MOVB $0, ret+120(FP)
	RET
