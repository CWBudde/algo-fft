//go:build amd64 && fft_asm && !purego

#include "textflag.h"

// Slice layout in Go calling convention:
// Each []T is: ptr (8 bytes) + len (8 bytes) + cap (8 bytes) = 24 bytes
// func forwardAVX2Complex64Asm(dst, src, twiddle, scratch []complex64, bitrev []int) bool
//   dst:     FP+0 (ptr), FP+8 (len), FP+16 (cap)
//   src:     FP+24 (ptr), FP+32 (len), FP+40 (cap)
//   twiddle: FP+48 (ptr), FP+56 (len), FP+64 (cap)
//   scratch: FP+72 (ptr), FP+80 (len), FP+88 (cap)
//   bitrev:  FP+96 (ptr), FP+104 (len), FP+112 (cap)
//   return:  FP+120

// Constants for complex64 layout:
// complex64 = 8 bytes (4 bytes real + 4 bytes imag)
// YMM register (256 bits) = 32 bytes = 4 complex64

// Register allocation:
// R8:  dst pointer (work pointer during transform)
// R9:  src pointer
// R10: twiddle pointer
// R11: scratch pointer
// R12: bitrev pointer
// R13: n (length)
// R14: size (outer loop)
// R15: half (size >> 1)
// BX:  step (n / size)
// CX:  base (middle loop)
// DX:  j (inner loop)
// SI:  index1 offset (in bytes)
// DI:  index2 offset (in bytes)

TEXT ·forwardAVX2Complex64Asm(SB), NOSPLIT, $0-121
	// Load slice pointers and lengths
	MOVQ dst+0(FP), R8       // dst pointer
	MOVQ src+24(FP), R9      // src pointer
	MOVQ twiddle+48(FP), R10 // twiddle pointer
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ bitrev+96(FP), R12  // bitrev pointer
	MOVQ src+32(FP), R13     // n = len(src)

	// Handle n == 0: return true
	TESTQ R13, R13
	JZ    return_true

	// Check slice lengths: dst, twiddle, scratch, bitrev all need len >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, R13
	JL   return_false

	// Handle n == 1: dst[0] = src[0]; return true
	CMPQ R13, $1
	JNE  check_power_of_2
	MOVQ (R9), AX          // Load 8 bytes (complex64)
	MOVQ AX, (R8)          // Store to dst
	JMP  return_true

check_power_of_2:
	// Check if n is power of 2: n > 0 && (n & (n-1)) == 0
	MOVQ R13, AX
	LEAQ -1(AX), BX        // BX = n - 1
	TESTQ AX, BX           // n & (n-1)
	JNZ  return_false      // Not power of 2

	// Require n >= 16 for AVX2 (fall back to Go for smaller sizes)
	CMPQ R13, $16
	JL   return_false

	// Check if dst == src (same slice)
	// If same, use scratch as work buffer
	CMPQ R8, R9
	JNE  use_dst_as_work

	// dst == src: use scratch as work buffer
	MOVQ R11, R8           // work = scratch
	MOVL $0, AX            // workIsDst = false
	JMP  do_bit_reversal

use_dst_as_work:
	MOVL $1, AX            // workIsDst = true

do_bit_reversal:
	// Save workIsDst flag on stack (we'll use it later)
	// Actually, we'll use a callee-saved register approach
	// For now, we'll just always use dst if dst != src

	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// for i := 0; i < n; i++ { work[i] = src[bitrev[i]] }
	XORQ CX, CX            // i = 0

bitrev_loop:
	CMPQ CX, R13
	JGE  bitrev_done

	// Load bitrev[i] - int is 8 bytes on amd64
	MOVQ (R12)(CX*8), DX   // DX = bitrev[i]

	// Load src[bitrev[i]] - complex64 is 8 bytes
	MOVQ (R9)(DX*8), AX    // AX = src[bitrev[i]]

	// Store to work[i]
	MOVQ AX, (R8)(CX*8)    // work[i] = src[bitrev[i]]

	INCQ CX
	JMP  bitrev_loop

bitrev_done:
	// Main DIT butterfly loops
	// for size := 2; size <= n; size <<= 1
	MOVQ $2, R14           // size = 2

size_loop:
	CMPQ R14, R13
	JG   transform_done

	// half = size >> 1
	MOVQ R14, R15
	SHRQ $1, R15           // half = size / 2

	// step = n / size
	MOVQ R13, AX           // AX = n (dividend low)
	XORQ DX, DX            // DX = 0 (dividend high)
	DIVQ R14               // AX = n / size (step), DX = remainder
	MOVQ AX, BX            // BX = step

	// for base := 0; base < n; base += size
	XORQ CX, CX            // base = 0

base_loop:
	CMPQ CX, R13
	JGE  next_size

	// Check if we can use AVX2 (half >= 4 and step == 1)
	CMPQ R15, $4
	JL   scalar_butterflies
	CMPQ BX, $1
	JNE  scalar_butterflies

	// AVX2 vectorized path: process 4 butterflies at a time
	XORQ DX, DX            // j = 0

avx2_loop:
	// Check if we have at least 4 more butterflies
	MOVQ R15, AX
	SUBQ DX, AX            // remaining = half - j
	CMPQ AX, $4
	JL   scalar_remainder

	// index1 = base + j (byte offset = index * 8)
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI            // SI = (base + j) * 8

	// index2 = index1 + half*8
	MOVQ R15, DI
	SHLQ $3, DI            // half * 8
	ADDQ SI, DI            // DI = (base + j + half) * 8

	// Load 4 complex64 from work[index1:index1+4]
	// Y0 = [a0.r, a0.i, a1.r, a1.i, a2.r, a2.i, a3.r, a3.i]
	VMOVUPS (R8)(SI*1), Y0

	// Load 4 complex64 from work[index2:index2+4]
	// Y1 = [b0.r, b0.i, b1.r, b1.i, b2.r, b2.i, b3.r, b3.i]
	VMOVUPS (R8)(DI*1), Y1

	// Load 4 twiddle factors (step == 1, so contiguous)
	// twiddle[j*step] ... twiddle[(j+3)*step] = twiddle[j:j+4]
	MOVQ DX, AX
	SHLQ $3, AX            // j * 8 (byte offset)
	VMOVUPS (R10)(AX*1), Y2

	// Complex multiply: t = w * b
	// (wr + wi*i) * (br + bi*i) = (wr*br - wi*bi) + (wr*bi + wi*br)*i
	//
	// Input layout: Y1 = [b0.r, b0.i, b1.r, b1.i, b2.r, b2.i, b3.r, b3.i]
	//               Y2 = [w0.r, w0.i, w1.r, w1.i, w2.r, w2.i, w3.r, w3.i]
	//
	// Algorithm using MOVELDUP/MOVEHDUP pattern:
	// 1. Duplicate real parts:     [r, r, r, r, ...] for each complex
	// 2. Duplicate imag parts:     [i, i, i, i, ...] for each complex
	// 3. Multiply, addsub for real/imag placement

	// Create [w.r, w.r, w.r, w.r, ...] by duplicating low floats of each pair
	VMOVSLDUP Y2, Y3         // Y3 = [w0.r, w0.r, w1.r, w1.r, w2.r, w2.r, w3.r, w3.r]

	// Create [w.i, w.i, w.i, w.i, ...] by duplicating high floats of each pair
	VMOVSHDUP Y2, Y4         // Y4 = [w0.i, w0.i, w1.i, w1.i, w2.i, w2.i, w3.i, w3.i]

	// Y5 = b * w.r = [b.r*w.r, b.i*w.r, ...]
	VMULPS Y3, Y1, Y5

	// Y6 = b_swapped * w.i where b_swapped = [b.i, b.r, b.i, b.r, ...]
	// First swap real/imag pairs in Y1
	VSHUFPS $0xB1, Y1, Y1, Y6  // 0xB1 = 10 11 00 01 => swap pairs: [b.i, b.r, ...]
	VMULPS Y4, Y6, Y6          // Y6 = [b.i*w.i, b.r*w.i, ...]

	// Result: real = b.r*w.r - b.i*w.i (even positions)
	//         imag = b.i*w.r + b.r*w.i (odd positions)
	// Use VADDSUBPS: Y5[even] - Y6[even], Y5[odd] + Y6[odd]
	VADDSUBPS Y6, Y5, Y2       // Y2 = t = w * b

	// Butterfly: a' = a + t, b' = a - t
	VADDPS Y2, Y0, Y3       // Y3 = a + t
	VSUBPS Y2, Y0, Y4       // Y4 = a - t

	// Store results
	VMOVUPS Y3, (R8)(SI*1)  // work[index1:index1+4] = a + t
	VMOVUPS Y4, (R8)(DI*1)  // work[index2:index2+4] = a - t

	ADDQ $4, DX             // j += 4
	JMP  avx2_loop

scalar_remainder:
	// Handle remaining butterflies (0-3) with scalar code
	CMPQ DX, R15
	JGE  next_base

scalar_remainder_loop:
	// index1 = base + j
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI            // byte offset

	// index2 = index1 + half*8
	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load a = work[index1]
	MOVSD (R8)(SI*1), X0   // a.real, a.imag

	// Load b = work[index2]
	MOVSD (R8)(DI*1), X1   // b.real, b.imag

	// Load twiddle[j * step]
	MOVQ DX, AX
	IMULQ BX, AX           // j * step
	SHLQ $3, AX            // byte offset
	MOVSD (R10)(AX*1), X2  // w.real, w.imag

	// Complex multiply: t = w * b using SSE MOVSLDUP/MOVSHDUP pattern
	// X1 = [b.r, b.i, ?, ?], X2 = [w.r, w.i, ?, ?]

	// Duplicate w.r and w.i
	MOVSLDUP X2, X3        // X3 = [w.r, w.r, ...]
	MOVSHDUP X2, X4        // X4 = [w.i, w.i, ...]

	// X5 = b * w.r
	MOVAPS X1, X5
	MULPS  X3, X5          // X5 = [b.r*w.r, b.i*w.r, ...]

	// Swap b's real and imag
	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6   // X6 = [b.i, b.r, ...]

	// X6 = b_swapped * w.i
	MULPS  X4, X6          // X6 = [b.i*w.i, b.r*w.i, ...]

	// Use ADDSUBPS: result = [b.r*w.r - b.i*w.i, b.i*w.r + b.r*w.i, ...]
	ADDSUBPS X6, X5        // X5 = t = w * b

	// Butterfly: a' = a + t, b' = a - t
	MOVAPS X0, X3
	ADDPS  X5, X3          // a + t
	MOVAPS X0, X4
	SUBPS  X5, X4          // a - t

	// Store (only lower 64 bits = 1 complex64)
	MOVSD X3, (R8)(SI*1)   // work[index1] = a + t
	MOVSD X4, (R8)(DI*1)   // work[index2] = a - t

	INCQ DX                // j++
	CMPQ DX, R15
	JL   scalar_remainder_loop

next_base:
	ADDQ R14, CX           // base += size
	JMP  base_loop

scalar_butterflies:
	// All butterflies scalar (step > 1 or half < 4)
	XORQ DX, DX            // j = 0

scalar_loop:
	CMPQ DX, R15
	JGE  next_base

	// index1 = base + j
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI            // byte offset

	// index2 = index1 + half*8
	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load a = work[index1]
	MOVSD (R8)(SI*1), X0

	// Load b = work[index2]
	MOVSD (R8)(DI*1), X1

	// Load twiddle[j * step]
	MOVQ DX, AX
	IMULQ BX, AX           // j * step
	SHLQ $3, AX            // byte offset
	MOVSD (R10)(AX*1), X2

	// Complex multiply: t = w * b using MOVSLDUP/MOVSHDUP pattern
	MOVSLDUP X2, X3        // X3 = [w.r, w.r, ...]
	MOVSHDUP X2, X4        // X4 = [w.i, w.i, ...]

	MOVAPS X1, X5
	MULPS  X3, X5          // X5 = [b.r*w.r, b.i*w.r, ...]

	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6   // X6 = [b.i, b.r, ...]
	MULPS  X4, X6          // X6 = [b.i*w.i, b.r*w.i, ...]

	ADDSUBPS X6, X5        // X5 = t = w * b

	// Butterfly: a' = a + t, b' = a - t
	MOVAPS X0, X3
	ADDPS  X5, X3
	MOVAPS X0, X4
	SUBPS  X5, X4

	MOVSD X3, (R8)(SI*1)
	MOVSD X4, (R8)(DI*1)

	INCQ DX
	JMP  scalar_loop

next_size:
	SHLQ $1, R14           // size <<= 1
	JMP  size_loop

transform_done:
	// VZEROUPPER to avoid AVX-SSE transition penalty
	VZEROUPPER

	// If work != dst, copy work to dst
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   return_true

	// work is in scratch, need to copy to dst
	// Copy n complex64 values (n * 8 bytes)
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

// Inverse FFT: same as forward but conjugate twiddles and scale by 1/n
TEXT ·inverseAVX2Complex64Asm(SB), NOSPLIT, $0-121
	// Load slice pointers and lengths
	MOVQ dst+0(FP), R8       // dst pointer
	MOVQ src+24(FP), R9      // src pointer
	MOVQ twiddle+48(FP), R10 // twiddle pointer
	MOVQ scratch+72(FP), R11 // scratch pointer
	MOVQ bitrev+96(FP), R12  // bitrev pointer
	MOVQ src+32(FP), R13     // n = len(src)

	// Handle n == 0: return true
	TESTQ R13, R13
	JZ    inv_return_true

	// Check slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, R13
	JL   inv_return_false

	// Handle n == 1
	CMPQ R13, $1
	JNE  inv_check_power_of_2
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  inv_return_true

inv_check_power_of_2:
	// Check if n is power of 2
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  inv_return_false

	// Require n >= 16 for AVX2
	CMPQ R13, $16
	JL   inv_return_false

	// Check if dst == src
	CMPQ R8, R9
	JNE  inv_use_dst_as_work
	MOVQ R11, R8             // work = scratch
	JMP  inv_do_bit_reversal

inv_use_dst_as_work:

inv_do_bit_reversal:
	// Bit-reversal
	XORQ CX, CX

inv_bitrev_loop:
	CMPQ CX, R13
	JGE  inv_bitrev_done
	MOVQ (R12)(CX*8), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)(CX*8)
	INCQ CX
	JMP  inv_bitrev_loop

inv_bitrev_done:
	// DIT butterfly loops (same as forward, but conjugate twiddles)
	MOVQ $2, R14

inv_size_loop:
	CMPQ R14, R13
	JG   inv_transform_done

	MOVQ R14, R15
	SHRQ $1, R15             // half

	MOVQ R13, AX             // AX = n (dividend low)
	XORQ DX, DX              // DX = 0 (dividend high)
	DIVQ R14                 // AX = n / size (step)
	MOVQ AX, BX              // step

	XORQ CX, CX              // base

inv_base_loop:
	CMPQ CX, R13
	JGE  inv_next_size

	// Check for AVX2 path
	CMPQ R15, $4
	JL   inv_scalar_butterflies
	CMPQ BX, $1
	JNE  inv_scalar_butterflies

	XORQ DX, DX

inv_avx2_loop:
	MOVQ R15, AX
	SUBQ DX, AX
	CMPQ AX, $4
	JL   inv_scalar_remainder

	// Compute offsets
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $3, SI

	MOVQ R15, DI
	SHLQ $3, DI
	ADDQ SI, DI

	// Load a and b
	VMOVUPS (R8)(SI*1), Y0
	VMOVUPS (R8)(DI*1), Y1

	// Load twiddle
	MOVQ DX, AX
	SHLQ $3, AX
	VMOVUPS (R10)(AX*1), Y2

	// Conjugate multiply: t = conj(w) * b
	// conj(w) * b = (w.r - w.i*i) * (b.r + b.i*i)
	//             = (w.r*b.r + w.i*b.i) + (w.r*b.i - w.i*b.r)*i
	//
	// With MOVSLDUP/MOVSHDUP pattern:
	// Y3 = [w.r, w.r, w.r, w.r, ...]
	// Y4 = [w.i, w.i, w.i, w.i, ...]
	// Y5 = [b.r*w.r, b.i*w.r, ...]
	// Y6 = [b.i*w.i, b.r*w.i, ...] (after swap)
	//
	// We need: real = Y5[0] + Y6[0], imag = Y5[1] - Y6[1]
	// ADDSUBPS does: even = a - b, odd = a + b
	// So VADDSUBPS Y6, Y5 gives: Y5[0]-Y6[0], Y5[1]+Y6[1] (wrong)
	// We need the opposite: Y5[0]+Y6[0], Y5[1]-Y6[1]
	//
	// Solution: Negate Y6, then VADDSUBPS gives correct result
	// -Y6 = [-b.i*w.i, -b.r*w.i, ...]
	// VADDSUBPS (-Y6), Y5: Y5[0]-(-Y6[0]), Y5[1]+(-Y6[1])
	//                    = Y5[0]+Y6[0], Y5[1]-Y6[1] ✓

	VMOVSLDUP Y2, Y3         // Y3 = [w.r, w.r, ...]
	VMOVSHDUP Y2, Y4         // Y4 = [w.i, w.i, ...]
	VMULPS Y3, Y1, Y5        // Y5 = [b.r*w.r, b.i*w.r, ...]
	VSHUFPS $0xB1, Y1, Y1, Y6  // Y6 = [b.i, b.r, ...]
	VMULPS Y4, Y6, Y6          // Y6 = [b.i*w.i, b.r*w.i, ...]

	// Negate Y6: use VXORPS with sign bit mask
	// Create mask with all sign bits set: 0x80000000 repeated
	VXORPS Y7, Y7, Y7          // Y7 = 0
	VPCMPEQD Y8, Y8, Y8        // Y8 = all 1s
	VPSLLD $31, Y8, Y8         // Y8 = [0x80000000, ...] sign bit mask
	VXORPS Y8, Y6, Y6          // Y6 = -Y6 (negate)

	// Now VADDSUBPS gives conjugate multiply result
	VADDSUBPS Y6, Y5, Y2       // Y2 = t = conj(w) * b

	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4

	VMOVUPS Y3, (R8)(SI*1)
	VMOVUPS Y4, (R8)(DI*1)

	ADDQ $4, DX
	JMP  inv_avx2_loop

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

	MOVSD (R8)(SI*1), X0
	MOVSD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	MOVSD (R10)(AX*1), X2

	// Conjugate twiddle and multiply: t = conj(w) * b
	// For conj(w) * b = (w.r - w.i*i) * (b.r + b.i*i)
	//                 = (w.r*b.r + w.i*b.i) + (w.r*b.i - w.i*b.r)*i
	//
	// We use the same MOVSLDUP/MOVSHDUP pattern, but swap the addsub operands
	// to get the correct sign pattern for conjugate multiply

	MOVSLDUP X2, X3        // X3 = [w.r, w.r, ...]
	MOVSHDUP X2, X4        // X4 = [w.i, w.i, ...]

	MOVAPS X1, X5
	MULPS  X3, X5          // X5 = [b.r*w.r, b.i*w.r, ...]

	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6   // X6 = [b.i, b.r, ...]
	MULPS  X4, X6          // X6 = [b.i*w.i, b.r*w.i, ...]

	// For conjugate: real = b.r*w.r + b.i*w.i, imag = b.i*w.r - b.r*w.i
	// X5 = [b.r*w.r, b.i*w.r, ...]
	// X6 = [b.i*w.i, b.r*w.i, ...]
	// We need: real = X5[0] + X6[0], imag = X5[1] - X6[1]
	//
	// Use UNPCKLPS to interleave, then add/sub appropriately
	// Simpler: use BLENDPS-style approach with SHUFPS
	MOVAPS X5, X7          // X7 = [b.r*w.r, b.i*w.r, ...]
	ADDPS  X6, X7          // X7[0] = b.r*w.r + b.i*w.i (real - correct)
	                       // X7[1] = b.i*w.r + b.r*w.i (wrong imag)
	MOVAPS X5, X3          // X3 = [b.r*w.r, b.i*w.r, ...]
	SUBPS  X6, X3          // X3[0] = b.r*w.r - b.i*w.i (wrong real)
	                       // X3[1] = b.i*w.r - b.r*w.i (imag - correct)
	// Blend: X7[0] and X3[1] -> use UNPCKLPS then shuffle
	// X7 = [real_ok, imag_wrong, ...], X3 = [real_wrong, imag_ok, ...]
	// UNPCKLPS X3, X7: X7 = [X7[0], X3[0], X7[1], X3[1]] = [real_ok, real_wrong, imag_wrong, imag_ok]
	// Then SHUFPS to select [0] and [3]: not directly possible
	// Use MOVSS to move X3[0..1] keeping X7[0]? MOVSS copies only [0]
	// Use SHUFPS $0xD8 to reorder? Let's use a simpler approach:
	// MOVHLPS X3, X7: X7[0:1] = X3[2:3] - wrong
	// Use INSERTPS (SSE4.1) or manual blend with masks
	// Simplest: copy low 32 bits from X3[1] to X7[1] via SHUFPS
	// SHUFPS $imm, src, dst: dst[0,1] from dst, dst[2,3] from src
	// We want dst = [X7[0], X3[1], ?, ?]
	// SHUFPS with imm = (dst[0] idx) | (src[1] idx << 2) | ...
	//   = 0 | (1 << 2) | ... = 0x04 for [X7[0], X3[0], ...]
	// Actually SHUFPS selects pairs: lower pair from first operand, upper from second
	// dst[0] = dst[imm[0:1]], dst[1] = dst[imm[2:3]], dst[2] = src[imm[4:5]], dst[3] = src[imm[6:7]]
	// We need [X7[0], X3[1], X7[2], X3[3]] = impossible with single SHUFPS
	// Use UNPCKLPS then pick elements:
	UNPCKLPS X3, X7        // X7 = [X7[0], X3[0], X7[1], X3[1]]
	                       //    = [real_ok, wrong, wrong, imag_ok]
	SHUFPS $0x0C, X7, X7   // imm = 0 | (3<<2) = 0x0C
	                       // X7 = [X7[0], X7[3], X7[0], X7[3]] = [real_ok, imag_ok, ...]
	MOVAPS X7, X5

	// Butterfly: a' = a + t, b' = a - t
	MOVAPS X0, X3
	ADDPS  X5, X3
	MOVAPS X0, X4
	SUBPS  X5, X4

	MOVSD X3, (R8)(SI*1)
	MOVSD X4, (R8)(DI*1)

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

	MOVSD (R8)(SI*1), X0
	MOVSD (R8)(DI*1), X1

	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $3, AX
	MOVSD (R10)(AX*1), X2

	// Conjugate multiply: t = conj(w) * b (same as inv_scalar_remainder_loop)
	MOVSLDUP X2, X3
	MOVSHDUP X2, X4

	MOVAPS X1, X5
	MULPS  X3, X5

	MOVAPS X1, X6
	SHUFPS $0xB1, X6, X6
	MULPS  X4, X6

	// Blend for conjugate multiply: X5[0]+X6[0], X5[1]-X6[1]
	MOVAPS X5, X7
	ADDPS  X6, X7          // X7 = [...+..., ...+...]
	MOVAPS X5, X3
	SUBPS  X6, X3          // X3 = [...-..., ...-...]
	// X7[0] = correct real, X3[1] = correct imag
	UNPCKLPS X3, X7        // X7 = [X7[0], X3[0], X7[1], X3[1]]
	SHUFPS $0x0C, X7, X7   // X7 = [X7[0], X7[3], ...] = [real, imag, ...]
	MOVAPS X7, X5

	MOVAPS X0, X3
	ADDPS  X5, X3
	MOVAPS X0, X4
	SUBPS  X5, X4

	MOVSD X3, (R8)(SI*1)
	MOVSD X4, (R8)(DI*1)

	INCQ DX
	JMP  inv_scalar_loop

inv_next_size:
	SHLQ $1, R14
	JMP  inv_size_loop

inv_transform_done:
	VZEROUPPER

	// Copy if needed
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
	// Scale by 1/n
	// Compute 1/n as float32
	MOVQ dst+0(FP), R8

	// Convert n to float and compute 1/n
	CVTSQ2SS R13, X0         // X0 = (float32)n
	MOVSS    ·one32(SB), X1  // X1 = 1.0f
	DIVSS    X0, X1          // X1 = 1.0f / n

	// Broadcast scale to all lanes
	SHUFPS   $0x00, X1, X1   // X1 = [scale, scale, scale, scale]

	// Scale each element: dst[i] *= scale (both real and imag)
	XORQ CX, CX

inv_scale_loop:
	CMPQ CX, R13
	JGE  inv_return_true

	// Load complex64
	MOVSD (R8)(CX*8), X0

	// Multiply both components by scale
	MULPS X1, X0

	// Store back
	MOVSD X0, (R8)(CX*8)

	INCQ CX
	JMP  inv_scale_loop

inv_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_return_false:
	MOVB $0, ret+120(FP)
	RET

// SSE2 stubs (delegate to pure Go for now)
TEXT ·forwardSSE2Complex64Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)
	RET

TEXT ·inverseSSE2Complex64Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)
	RET

TEXT ·forwardAVX2Complex128Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)
	RET

TEXT ·inverseAVX2Complex128Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)
	RET

TEXT ·forwardSSE2Complex128Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)
	RET

TEXT ·inverseSSE2Complex128Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)
	RET

TEXT ·asmCopyComplex64(SB), NOSPLIT|NOFRAME, $0-16
	MOVQ (BX), CX
	MOVQ CX, (AX)
	RET

// Constants
DATA ·half32+0(SB)/4, $0x3f000000  // 0.5f
GLOBL ·half32(SB), RODATA|NOPTR, $4

DATA ·one32+0(SB)/4, $0x3f800000   // 1.0f
GLOBL ·one32(SB), RODATA|NOPTR, $4

TEXT ·asmForward2Complex64(SB), NOSPLIT|NOFRAME, $0-16
	MOVSS (BX), X0
	MOVSS 4(BX), X1
	MOVSS 8(BX), X2
	MOVSS 12(BX), X3

	MOVSS X0, X4
	ADDSS X2, X4
	MOVSS X1, X5
	ADDSS X3, X5

	MOVSS X4, (AX)
	MOVSS X5, 4(AX)

	MOVSS X0, X6
	SUBSS X2, X6
	MOVSS X1, X7
	SUBSS X3, X7

	MOVSS X6, 8(AX)
	MOVSS X7, 12(AX)

	MOVL $1, AX
	RET

TEXT ·asmInverse2Complex64(SB), NOSPLIT|NOFRAME, $0-16
	MOVSS (BX), X0
	MOVSS 4(BX), X1
	MOVSS 8(BX), X2
	MOVSS 12(BX), X3

	MOVSS X0, X4
	ADDSS X2, X4
	MOVSS X1, X5
	ADDSS X3, X5

	MULSS ·half32(SB), X4
	MULSS ·half32(SB), X5

	MOVSS X4, (AX)
	MOVSS X5, 4(AX)

	MOVSS X0, X6
	SUBSS X2, X6
	MOVSS X1, X7
	SUBSS X3, X7

	MULSS ·half32(SB), X6
	MULSS ·half32(SB), X7

	MOVSS X6, 8(AX)
	MOVSS X7, 12(AX)

	MOVL $1, AX
	RET
