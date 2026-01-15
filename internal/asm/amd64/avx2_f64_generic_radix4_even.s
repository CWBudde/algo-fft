//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2/FMA-optimized FFT Assembly for AMD64 - complex128 (float64)
// Radix-4 generic DIT path for power-of-4 sizes (64, 256, 1024, ...).
// ===========================================================================
//
// Data layout: complex128 = 16 bytes (re: float64, im: float64)
// YMM register (256-bit) holds 2 complex128 values (4 float64)
// XMM register (128-bit) holds 1 complex128 value (2 float64)
//
// Radix-4 butterfly:
//   t0 = a0 + a2, t1 = a0 - a2
//   t2 = a1 + a3, t3 = a1 - a3
//   y0 = t0 + t2
//   y1 = t1 - i*t3  (forward: -i, inverse: +i)
//   y2 = t0 - t2
//   y3 = t1 + i*t3  (forward: +i, inverse: -i)
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// ForwardAVX2Complex128Radix4Asm - Forward FFT for complex128 using radix-4 DIT
// ===========================================================================
// func ForwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
// Frame size: 0, Args size: 121 bytes
//   dst:     FP+0  (ptr), FP+8  (len), FP+16 (cap)
//   src:     FP+24 (ptr), FP+32 (len), FP+40 (cap)
//   twiddle: FP+48 (ptr), FP+56 (len), FP+64 (cap)
//   scratch: FP+72 (ptr), FP+80 (len), FP+88 (cap)
//   bitrev:  FP+96 (ptr), FP+104(len), FP+112(cap)
//   ret:     FP+120
// ===========================================================================
TEXT ·ForwardAVX2Complex128Radix4Asm(SB), NOSPLIT, $0-121
	// ===================================================================
	// Parameter loading
	// ===================================================================
	MOVQ dst+0(FP), R8          // R8 = dst pointer
	MOVQ src+24(FP), R9         // R9 = src pointer
	MOVQ twiddle+48(FP), R10    // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11    // R11 = scratch pointer
	MOVQ src+32(FP), R13        // R13 = n (element count from src.len)

	// ===================================================================
	// Empty input check - empty is valid (no-op)
	// ===================================================================
	TESTQ R13, R13              // test if n == 0
	JZ    fwd_r4_c128_return_true // return true for empty input

	// ===================================================================
	// Validate all slice lengths are >= n
	// ===================================================================
	MOVQ dst+8(FP), AX          // AX = dst.len
	CMPQ AX, R13                // compare dst.len with n
	JL   fwd_r4_c128_return_false // fail if dst.len < n

	MOVQ twiddle+56(FP), AX     // AX = twiddle.len
	CMPQ AX, R13                // compare twiddle.len with n
	JL   fwd_r4_c128_return_false // fail if twiddle.len < n

	MOVQ scratch+80(FP), AX     // AX = scratch.len
	CMPQ AX, R13                // compare scratch.len with n
	JL   fwd_r4_c128_return_false // fail if scratch.len < n

	// ===================================================================
	// Trivial case: n=1, just copy single element
	// ===================================================================
	CMPQ R13, $1                // check if n == 1
	JNE  fwd_r4_c128_check_power_of_2 // continue if n != 1
	VMOVUPD (R9), X0            // load 16 bytes (1 complex128) from src
	VMOVUPD X0, (R8)            // store to dst
	JMP  fwd_r4_c128_return_true // return true

fwd_r4_c128_check_power_of_2:
	// ===================================================================
	// Verify n is power of 2: (n & (n-1)) == 0
	// ===================================================================
	MOVQ R13, AX                // AX = n
	LEAQ -1(AX), BX             // BX = n - 1
	TESTQ AX, BX                // test n & (n-1)
	JNZ  fwd_r4_c128_return_false // fail if not power of 2

	// ===================================================================
	// Require even log2 (power-of-4) and minimum size of 64
	// ===================================================================
	MOVQ R13, AX                // AX = n
	BSRQ AX, R14                // R14 = log2(n) = bit scan reverse
	TESTQ $1, R14               // check if log2(n) is odd
	JNZ  fwd_r4_c128_return_false // fail if odd (not power-of-4)

	CMPQ R13, $64               // check minimum size
	JL   fwd_r4_c128_return_false // fail if n < 64

	// ===================================================================
	// Select working buffer (use scratch for in-place transforms)
	// ===================================================================
	CMPQ R8, R9                 // check if dst == src
	JNE  fwd_r4_c128_use_dst    // use dst if different
	MOVQ R11, R8                // in-place: use scratch as working buffer

fwd_r4_c128_use_dst:
	// ===================================================================
	// Bit-reversal permutation (base-4 digit reversal)
	// For n = 4^k, reverse k base-4 digits
	// ===================================================================
	XORQ CX, CX                 // CX = i = 0 (source index)

fwd_r4_c128_bitrev_loop:
	CMPQ CX, R13                // compare i with n
	JGE  fwd_r4_c128_stage_init // done with bit-reversal if i >= n

	// Compute bit-reversed index in BX
	MOVQ CX, DX                 // DX = i (value to reverse)
	XORQ BX, BX                 // BX = reversed index = 0
	MOVQ R14, SI                // SI = log2(n) (bits remaining)

fwd_r4_c128_bitrev_inner:
	CMPQ SI, $0                 // check if all bits processed
	JE   fwd_r4_c128_bitrev_store // done reversing
	MOVQ DX, AX                 // AX = remaining value
	ANDQ $3, AX                 // AX = lowest 2 bits (base-4 digit)
	SHLQ $2, BX                 // shift reversed index left by 2
	ORQ  AX, BX                 // add new digit to reversed index
	SHRQ $2, DX                 // remove processed digit from value
	SUBQ $2, SI                 // decrement bits remaining by 2
	JMP  fwd_r4_c128_bitrev_inner // process next digit

fwd_r4_c128_bitrev_store:
	// Copy element from src[BX] to work[CX]
	// complex128 = 16 bytes, so offset = index * 16
	MOVQ BX, SI                 // SI = source index (bit-reversed)
	SHLQ $4, SI                 // SI = source byte offset (index * 16)
	VMOVUPD (R9)(SI*1), X0      // load src[bitrev[i]] (16 bytes)
	MOVQ CX, DI                 // DI = destination index
	SHLQ $4, DI                 // DI = dest byte offset (index * 16)
	VMOVUPD X0, (R8)(DI*1)      // store to work[i]
	INCQ CX                     // i++
	JMP  fwd_r4_c128_bitrev_loop // next element

fwd_r4_c128_stage_init:
	// ===================================================================
	// Radix-4 stages (size = 4, 16, 64, 256, ...)
	// For each stage, butterflies are size/4 apart
	// ===================================================================
	MOVQ $2, R12                // R12 = log2(size), starting at 2 (size=4)
	MOVQ $4, R14                // R14 = size = 4

fwd_r4_c128_stage_loop:
	CMPQ R14, R13               // compare size with n
	JG   fwd_r4_c128_copy_back  // done if size > n

	MOVQ R14, R15               // R15 = size
	SHRQ $2, R15                // R15 = quarter = size/4

	// step = n >> log2(size) = twiddle stride for this stage
	MOVQ R13, BX                // BX = n
	MOVQ R12, CX                // CX = log2(size)
	SHRQ CL, BX                 // BX = step = n >> log2(size)

	XORQ CX, CX                 // CX = base = 0 (start of current block)

fwd_r4_c128_stage_base:
	CMPQ CX, R13                // compare base with n
	JGE  fwd_r4_c128_stage_next // done with this stage if base >= n

	XORQ DX, DX                 // DX = j = 0 (element within quarter)

	// ===================================================================
	// Check for fast path: contiguous twiddles (step == 1)
	// This occurs on the final stage
	// ===================================================================
	CMPQ BX, $1                 // check if step == 1
	JNE  fwd_r4_c128_stepn_prep // use strided path if step != 1
	CMPQ R15, $2                // check if quarter >= 2 for YMM path
	JL   fwd_r4_c128_stage_scalar // fall back to scalar if quarter < 2

	// ===================================================================
	// YMM fast path: process 2 elements per iteration (step=1)
	// Uses YMM registers (256-bit) for 2 complex128 values
	// ===================================================================
	MOVQ R15, R11               // R11 = quarter
	SHLQ $4, R11                // R11 = quarter_bytes = quarter * 16

fwd_r4_c128_step1_loop:
	MOVQ R15, AX                // AX = quarter
	SUBQ DX, AX                 // AX = remaining elements in quarter
	CMPQ AX, $2                 // check if at least 2 elements remain
	JL   fwd_r4_c128_stage_scalar // fall back to scalar for remainder

	// ---------------------------------------------------------------
	// Compute element offsets
	// Base offset = (base + j) * 16
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	LEAQ (SI)(R11*1), DI        // DI = offset for quarter 1
	LEAQ (DI)(R11*1), AX        // AX = offset for quarter 2
	LEAQ (AX)(R11*1), BP        // BP = offset for quarter 3

	// ---------------------------------------------------------------
	// Load 2 complex128 from each of the 4 quarters
	// Y0 = [a0[0], a0[1]], Y1 = [a1[0], a1[1]], etc.
	// ---------------------------------------------------------------
	VMOVUPD (R8)(SI*1), Y0      // Y0 = work[base+j : base+j+2]
	VMOVUPD (R8)(DI*1), Y1      // Y1 = work[base+j+quarter : ...]
	VMOVUPD (R8)(AX*1), Y2      // Y2 = work[base+j+2*quarter : ...]
	VMOVUPD (R8)(BP*1), Y3      // Y3 = work[base+j+3*quarter : ...]

	// ---------------------------------------------------------------
	// Load twiddle factors (step=1)
	// w1 = twiddle[j], twiddle[j+1] (contiguous)
	// ---------------------------------------------------------------
	MOVQ DX, AX                 // AX = j
	SHLQ $4, AX                 // AX = j * 16 (byte offset)
	VMOVUPD (R10)(AX*1), Y4     // Y4 = w1[0:2] = twiddle[j:j+2]

	// ---------------------------------------------------------------
	// w2 = twiddle[2*j], twiddle[2*(j+1)] = twiddle[2*j], twiddle[2*j+2]
	// Stride between elements: 2 complex128 = 32 bytes
	// ---------------------------------------------------------------
	MOVQ DX, AX                 // AX = j
	SHLQ $1, AX                 // AX = 2*j
	SHLQ $4, AX                 // AX = 2*j * 16 = byte offset for twiddle[2*j]
	VMOVUPD (R10)(AX*1), X5     // X5 = twiddle[2*j]
	ADDQ $32, AX                // AX = byte offset for twiddle[2*j + 2]
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[2*(j+1)]
	VINSERTF128 $1, X6, Y5, Y5  // Y5 = [w2[0], w2[1]]

	// ---------------------------------------------------------------
	// w3 = twiddle[3*j], twiddle[3*(j+1)] = twiddle[3*j], twiddle[3*j+3]
	// Stride between elements: 3 complex128 = 48 bytes
	// ---------------------------------------------------------------
	LEAQ (DX)(DX*2), AX         // AX = 3*j
	SHLQ $4, AX                 // AX = 3*j * 16 = byte offset for twiddle[3*j]
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[3*j]
	ADDQ $48, AX                // AX = byte offset for twiddle[3*j + 3]
	VMOVUPD (R10)(AX*1), X7     // X7 = twiddle[3*(j+1)]
	VINSERTF128 $1, X7, Y6, Y6  // Y6 = [w3[0], w3[1]]

	// ---------------------------------------------------------------
	// Complex multiply a1 * w1
	// For (a + bi)(c + di) = (ac - bd) + (ad + bc)i
	// Using broadcast pattern for YMM with 2 complex values
	// ---------------------------------------------------------------
	VUNPCKLPD Y4, Y4, Y7        // Y7 = [w1[0].re, w1[0].re, w1[1].re, w1[1].re]
	VUNPCKHPD Y4, Y4, Y8        // Y8 = [w1[0].im, w1[0].im, w1[1].im, w1[1].im]
	VPERMILPD $0x05, Y1, Y9     // Y9 = [a1.im, a1.re, ...] swapped
	VMULPD Y8, Y9, Y9           // Y9 = [a1.im*w1.im, a1.re*w1.im, ...]
	VFMADDSUB231PD Y7, Y1, Y9   // Y9 = a1*w1.re +/- a1_swap*w1.im
	VMOVAPD Y9, Y1              // Y1 = a1 * w1

	// Complex multiply a2 * w2
	VUNPCKLPD Y5, Y5, Y7        // Y7 = w2.re broadcast
	VUNPCKHPD Y5, Y5, Y8        // Y8 = w2.im broadcast
	VPERMILPD $0x05, Y2, Y9     // Y9 = a2 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a2_swap * w2.im
	VFMADDSUB231PD Y7, Y2, Y9   // Y9 = a2 * w2
	VMOVAPD Y9, Y2              // Y2 = a2 * w2

	// Complex multiply a3 * w3
	VUNPCKLPD Y6, Y6, Y7        // Y7 = w3.re broadcast
	VUNPCKHPD Y6, Y6, Y8        // Y8 = w3.im broadcast
	VPERMILPD $0x05, Y3, Y9     // Y9 = a3 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a3_swap * w3.im
	VFMADDSUB231PD Y7, Y3, Y9   // Y9 = a3 * w3
	VMOVAPD Y9, Y3              // Y3 = a3 * w3

	// ---------------------------------------------------------------
	// Radix-4 butterfly computation
	// t0 = a0 + a2, t1 = a0 - a2
	// t2 = a1 + a3, t3 = a1 - a3
	// ---------------------------------------------------------------
	VADDPD Y2, Y0, Y10          // Y10 = t0 = a0 + a2
	VSUBPD Y2, Y0, Y11          // Y11 = t1 = a0 - a2
	VADDPD Y3, Y1, Y12          // Y12 = t2 = a1 + a3
	VSUBPD Y3, Y1, Y13          // Y13 = t3 = a1 - a3

	// ---------------------------------------------------------------
	// Compute -i*t3: multiply by -i = (0, -1)
	// -i * (a + bi) = (b, -a) = (im, -re)
	// Swap re<->im, then negate the original real (now at odd positions)
	// ---------------------------------------------------------------
	VPERMILPD $0x05, Y13, Y14   // Y14 = [t3.im, t3.re, ...] swapped
	// Create sign mask [0, signbit, 0, signbit] to negate odd positions
	MOVQ ·signbit64(SB), AX     // AX = signbit for float64
	VPXOR X15, X15, X15         // X15 = zero
	VMOVQ AX, X7                // X7 = [signbit, 0]
	VPUNPCKLQDQ X7, X15, X7     // X7 = [0, signbit] - mask for -i in low lane
	VINSERTI128 $1, X7, Y7, Y7  // Y7 high = X7, low = X7 (broadcast pattern)
	VINSERTI128 $0, X7, Y7, Y7  // Y7 = [0, signbit, 0, signbit]
	VXORPD Y7, Y14, Y14         // Y14 = -i*t3 = [t3.im, -t3.re, ...]

	// ---------------------------------------------------------------
	// Compute +i*t3: multiply by +i = (0, 1)
	// i * (a + bi) = (-b, a) = (-im, re)
	// Swap re<->im, then negate the original imag (now at even positions)
	// ---------------------------------------------------------------
	VPERMILPD $0x05, Y13, Y8    // Y8 = [t3.im, t3.re, ...] swapped
	VMOVQ AX, X9                // X9 = [signbit, 0]
	VPUNPCKLQDQ X15, X9, X9     // X9 = [signbit, 0] - mask for +i in low lane
	VINSERTI128 $1, X9, Y9, Y9  // Y9 high = X9
	VINSERTI128 $0, X9, Y9, Y9  // Y9 = [signbit, 0, signbit, 0]
	VXORPD Y9, Y8, Y8           // Y8 = +i*t3 = [-t3.im, t3.re, ...]

	// ---------------------------------------------------------------
	// Final butterfly outputs
	// y0 = t0 + t2
	// y1 = t1 - i*t3 = t1 + (-i*t3)
	// y2 = t0 - t2
	// y3 = t1 + i*t3
	// ---------------------------------------------------------------
	VADDPD Y12, Y10, Y0         // Y0 = y0 = t0 + t2
	VADDPD Y14, Y11, Y1         // Y1 = y1 = t1 + (-i*t3)
	VSUBPD Y12, Y10, Y2         // Y2 = y2 = t0 - t2
	VADDPD Y8, Y11, Y3          // Y3 = y3 = t1 + i*t3

	// ---------------------------------------------------------------
	// Recompute store offsets and store results
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16
	LEAQ (SI)(R11*1), DI        // DI = offset for quarter 1
	LEAQ (DI)(R11*1), AX        // AX = offset for quarter 2
	LEAQ (AX)(R11*1), BP        // BP = offset for quarter 3

	VMOVUPD Y0, (R8)(SI*1)      // store y0
	VMOVUPD Y1, (R8)(DI*1)      // store y1
	VMOVUPD Y2, (R8)(AX*1)      // store y2
	VMOVUPD Y3, (R8)(BP*1)      // store y3

	ADDQ $2, DX                 // j += 2 (processed 2 elements)
	JMP  fwd_r4_c128_step1_loop // continue loop

fwd_r4_c128_stepn_prep:
	// ===================================================================
	// YMM path for strided twiddles (step > 1)
	// ===================================================================
	CMPQ R15, $2                // check if quarter >= 2
	JL   fwd_r4_c128_stage_scalar // fall back to scalar

	MOVQ R15, R11               // R11 = quarter
	SHLQ $4, R11                // R11 = quarter_bytes = quarter * 16

	MOVQ BX, R9                 // R9 = step
	SHLQ $4, R9                 // R9 = stride1_bytes = step * 16

	MOVQ DX, DI                 // DI = j (starting at 0)
	IMULQ BX, DI                // DI = j * step
	SHLQ $4, DI                 // DI = j * step * 16 (twiddle byte offset)

fwd_r4_c128_stepn_loop:
	MOVQ R15, AX                // AX = quarter
	SUBQ DX, AX                 // AX = remaining elements
	CMPQ AX, $2                 // check if at least 2 elements remain
	JL   fwd_r4_c128_stage_scalar // fall back to scalar

	// ---------------------------------------------------------------
	// Compute element offsets
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	LEAQ (SI)(R11*1), R14       // R14 = offset for quarter 1
	LEAQ (R14)(R11*1), AX       // AX = offset for quarter 2
	LEAQ (AX)(R11*1), BP        // BP = offset for quarter 3

	// ---------------------------------------------------------------
	// Load data from 4 quarters
	// ---------------------------------------------------------------
	VMOVUPD (R8)(SI*1), Y0      // Y0 = work[base+j : base+j+2]
	VMOVUPD (R8)(R14*1), Y1     // Y1 = work[base+j+quarter : ...]
	VMOVUPD (R8)(AX*1), Y2      // Y2 = work[base+j+2*quarter : ...]
	VMOVUPD (R8)(BP*1), Y3      // Y3 = work[base+j+3*quarter : ...]

	// ---------------------------------------------------------------
	// Compute twiddle base offsets
	// For strided access: w1[k] = twiddle[(j+k)*step]
	// ---------------------------------------------------------------
	MOVQ DI, AX                 // AX = j*step*16 (w1 base offset)
	MOVQ AX, BP                 // BP = w1 offset

	// w1: gather twiddle[(j+0)*step], twiddle[(j+1)*step]
	VMOVUPD (R10)(BP*1), X4     // X4 = twiddle[j*step]
	ADDQ R9, BP                 // BP += step*16
	VMOVUPD (R10)(BP*1), X5     // X5 = twiddle[(j+1)*step]
	VINSERTF128 $1, X5, Y4, Y4  // Y4 = [w1[0], w1[1]]

	// w2: gather twiddle[2*j*step + k*2*step]
	MOVQ DI, BP                 // BP = j*step*16
	SHLQ $1, BP                 // BP = 2*j*step*16
	MOVQ R9, R14                // R14 = step*16
	SHLQ $1, R14                // R14 = 2*step*16 (stride for w2)
	VMOVUPD (R10)(BP*1), X5     // X5 = twiddle[2*j*step]
	ADDQ R14, BP                // BP += 2*step*16
	VMOVUPD (R10)(BP*1), X6     // X6 = twiddle[2*(j+1)*step]
	VINSERTF128 $1, X6, Y5, Y5  // Y5 = [w2[0], w2[1]]

	// w3: gather twiddle[3*j*step + k*3*step]
	MOVQ DI, BP                 // BP = j*step*16
	LEAQ (BP)(BP*2), BP         // BP = 3*j*step*16
	LEAQ (R9)(R9*2), R14        // R14 = 3*step*16 (stride for w3)
	VMOVUPD (R10)(BP*1), X6     // X6 = twiddle[3*j*step]
	ADDQ R14, BP                // BP += 3*step*16
	VMOVUPD (R10)(BP*1), X7     // X7 = twiddle[3*(j+1)*step]
	VINSERTF128 $1, X7, Y6, Y6  // Y6 = [w3[0], w3[1]]

	// Advance twiddle offset for next iteration
	ADDQ R9, DI                 // DI += step*16
	ADDQ R9, DI                 // DI += step*16 (total: 2*step*16)

	// ---------------------------------------------------------------
	// Complex multiply a1 * w1
	// ---------------------------------------------------------------
	VUNPCKLPD Y4, Y4, Y7        // Y7 = w1.re broadcast
	VUNPCKHPD Y4, Y4, Y8        // Y8 = w1.im broadcast
	VPERMILPD $0x05, Y1, Y9     // Y9 = a1 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a1_swap * w1.im
	VFMADDSUB231PD Y7, Y1, Y9   // Y9 = a1*w1.re +/- a1_swap*w1.im
	VMOVAPD Y9, Y1              // Y1 = a1 * w1

	// Complex multiply a2 * w2
	VUNPCKLPD Y5, Y5, Y7        // Y7 = w2.re broadcast
	VUNPCKHPD Y5, Y5, Y8        // Y8 = w2.im broadcast
	VPERMILPD $0x05, Y2, Y9     // Y9 = a2 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a2_swap * w2.im
	VFMADDSUB231PD Y7, Y2, Y9   // Y9 = a2 * w2
	VMOVAPD Y9, Y2              // Y2 = a2 * w2

	// Complex multiply a3 * w3
	VUNPCKLPD Y6, Y6, Y7        // Y7 = w3.re broadcast
	VUNPCKHPD Y6, Y6, Y8        // Y8 = w3.im broadcast
	VPERMILPD $0x05, Y3, Y9     // Y9 = a3 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a3_swap * w3.im
	VFMADDSUB231PD Y7, Y3, Y9   // Y9 = a3 * w3
	VMOVAPD Y9, Y3              // Y3 = a3 * w3

	// ---------------------------------------------------------------
	// Radix-4 butterfly
	// ---------------------------------------------------------------
	VADDPD Y2, Y0, Y10          // Y10 = t0 = a0 + a2
	VSUBPD Y2, Y0, Y11          // Y11 = t1 = a0 - a2
	VADDPD Y3, Y1, Y12          // Y12 = t2 = a1 + a3
	VSUBPD Y3, Y1, Y13          // Y13 = t3 = a1 - a3

	// -i*t3: swap re<->im, negate original real (now at odd positions)
	VPERMILPD $0x05, Y13, Y14   // Y14 = t3 swapped
	MOVQ ·signbit64(SB), AX     // AX = signbit
	VPXOR X15, X15, X15         // X15 = zero
	VMOVQ AX, X7                // X7 = [signbit, 0]
	VPUNPCKLQDQ X7, X15, X7     // X7 = [0, signbit] - mask for -i
	VINSERTI128 $1, X7, Y7, Y7  // Y7 high = X7
	VINSERTI128 $0, X7, Y7, Y7  // Y7 = [0, signbit, 0, signbit]
	VXORPD Y7, Y14, Y14         // Y14 = -i*t3

	// +i*t3: swap re<->im, negate original imag (now at even positions)
	VPERMILPD $0x05, Y13, Y8    // Y8 = t3 swapped
	VMOVQ AX, X9                // X9 = [signbit, 0]
	VPUNPCKLQDQ X15, X9, X9     // X9 = [signbit, 0] - mask for +i
	VINSERTI128 $1, X9, Y9, Y9  // Y9 high = X9
	VINSERTI128 $0, X9, Y9, Y9  // Y9 = [signbit, 0, signbit, 0]
	VXORPD Y9, Y8, Y8           // Y8 = +i*t3

	// Final outputs
	VADDPD Y12, Y10, Y0         // Y0 = y0 = t0 + t2
	VADDPD Y14, Y11, Y1         // Y1 = y1 = t1 + (-i*t3)
	VSUBPD Y12, Y10, Y2         // Y2 = y2 = t0 - t2
	VADDPD Y8, Y11, Y3          // Y3 = y3 = t1 + i*t3

	// ---------------------------------------------------------------
	// Recompute store offsets and store results
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16
	LEAQ (SI)(R11*1), R14       // R14 = offset for quarter 1
	LEAQ (R14)(R11*1), AX       // AX = offset for quarter 2
	LEAQ (AX)(R11*1), BP        // BP = offset for quarter 3

	VMOVUPD Y0, (R8)(SI*1)      // store y0
	VMOVUPD Y1, (R8)(R14*1)     // store y1
	VMOVUPD Y2, (R8)(AX*1)      // store y2
	VMOVUPD Y3, (R8)(BP*1)      // store y3

	ADDQ $2, DX                 // j += 2
	JMP  fwd_r4_c128_stepn_loop // continue loop

fwd_r4_c128_stage_scalar:
	// ===================================================================
	// Scalar fallback: process one element at a time using XMM
	// ===================================================================
	CMPQ DX, R15                // compare j with quarter
	JGE  fwd_r4_c128_stage_base_next // done with this block

	// ---------------------------------------------------------------
	// Compute indices for 4 elements in the butterfly
	// idx0 = base + j
	// idx1 = base + j + quarter
	// idx2 = base + j + 2*quarter
	// idx3 = base + j + 3*quarter
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = idx0 = base + j
	MOVQ SI, DI                 // DI = idx0
	ADDQ R15, DI                // DI = idx1 = base + j + quarter
	MOVQ DI, R11                // R11 = idx1
	ADDQ R15, R11               // R11 = idx2 = base + j + 2*quarter
	MOVQ R11, R9                // R9 = idx2
	ADDQ R15, R9                // R9 = idx3 = base + j + 3*quarter

	// ---------------------------------------------------------------
	// Compute twiddle indices
	// w1 = twiddle[j * step]
	// w2 = twiddle[2 * j * step]
	// w3 = twiddle[3 * j * step]
	// ---------------------------------------------------------------
	MOVQ DX, AX                 // AX = j
	IMULQ BX, AX                // AX = j * step
	MOVQ AX, BP                 // BP = w1 index = j * step
	SHLQ $4, BP                 // BP = w1 byte offset
	VMOVUPD (R10)(BP*1), X8     // X8 = w1 = twiddle[j*step]

	MOVQ AX, BP                 // BP = j * step
	SHLQ $1, BP                 // BP = 2 * j * step
	SHLQ $4, BP                 // BP = w2 byte offset
	VMOVUPD (R10)(BP*1), X9     // X9 = w2 = twiddle[2*j*step]

	LEAQ (AX)(AX*2), BP         // BP = 3 * j * step
	SHLQ $4, BP                 // BP = w3 byte offset
	VMOVUPD (R10)(BP*1), X10    // X10 = w3 = twiddle[3*j*step]

	// ---------------------------------------------------------------
	// Load 4 input elements
	// ---------------------------------------------------------------
	MOVQ SI, AX                 // AX = idx0
	SHLQ $4, AX                 // AX = idx0 * 16
	VMOVUPD (R8)(AX*1), X0      // X0 = a0 = work[idx0]

	MOVQ DI, AX                 // AX = idx1
	SHLQ $4, AX                 // AX = idx1 * 16
	VMOVUPD (R8)(AX*1), X1      // X1 = a1 = work[idx1]

	MOVQ R11, AX                // AX = idx2
	SHLQ $4, AX                 // AX = idx2 * 16
	VMOVUPD (R8)(AX*1), X2      // X2 = a2 = work[idx2]

	MOVQ R9, AX                 // AX = idx3
	SHLQ $4, AX                 // AX = idx3 * 16
	VMOVUPD (R8)(AX*1), X3      // X3 = a3 = work[idx3]

	// ---------------------------------------------------------------
	// Complex multiply a1 * w1 (XMM scalar version)
	// ---------------------------------------------------------------
	VMOVDDUP X8, X11            // X11 = [w1.re, w1.re]
	VPERMILPD $1, X8, X12       // X12 = [w1.im, w1.re] -> want [w1.im, w1.im]
	VMOVDDUP X12, X12           // X12 = [w1.im, w1.im]
	VPERMILPD $1, X1, X13       // X13 = [a1.im, a1.re]
	VMULPD X12, X13, X13        // X13 = [a1.im*w1.im, a1.re*w1.im]
	VFMADDSUB231PD X11, X1, X13 // X13 = a1*w1
	VMOVAPD X13, X1             // X1 = a1 * w1

	// Complex multiply a2 * w2
	VMOVDDUP X9, X11            // X11 = [w2.re, w2.re]
	VPERMILPD $1, X9, X12       // X12 -> [w2.im, ...]
	VMOVDDUP X12, X12           // X12 = [w2.im, w2.im]
	VPERMILPD $1, X2, X13       // X13 = a2 swapped
	VMULPD X12, X13, X13        // X13 = a2_swap * w2.im
	VFMADDSUB231PD X11, X2, X13 // X13 = a2 * w2
	VMOVAPD X13, X2             // X2 = a2 * w2

	// Complex multiply a3 * w3
	VMOVDDUP X10, X11           // X11 = [w3.re, w3.re]
	VPERMILPD $1, X10, X12      // X12 -> [w3.im, ...]
	VMOVDDUP X12, X12           // X12 = [w3.im, w3.im]
	VPERMILPD $1, X3, X13       // X13 = a3 swapped
	VMULPD X12, X13, X13        // X13 = a3_swap * w3.im
	VFMADDSUB231PD X11, X3, X13 // X13 = a3 * w3
	VMOVAPD X13, X3             // X3 = a3 * w3

	// ---------------------------------------------------------------
	// Radix-4 butterfly (XMM)
	// ---------------------------------------------------------------
	VADDPD X2, X0, X4           // X4 = t0 = a0 + a2
	VSUBPD X2, X0, X5           // X5 = t1 = a0 - a2
	VADDPD X3, X1, X6           // X6 = t2 = a1 + a3
	VSUBPD X3, X1, X7           // X7 = t3 = a1 - a3

	// ---------------------------------------------------------------
	// Compute -i*t3 for XMM
	// -i * (a + bi) = (b, -a)
	// ---------------------------------------------------------------
	VPERMILPD $1, X7, X8        // X8 = [t3.im, t3.re]
	MOVQ ·signbit64(SB), AX     // AX = signbit
	VMOVQ AX, X10               // X10 = [signbit, 0]
	VXORPD X11, X11, X11        // X11 = zero
	VUNPCKLPD X10, X11, X10     // X10 = [0, signbit] - negate imag position
	VXORPD X10, X8, X8          // X8 = -i*t3 = [t3.im, -t3.re]

	// ---------------------------------------------------------------
	// Compute +i*t3 for XMM
	// i * (a + bi) = (-b, a)
	// ---------------------------------------------------------------
	VPERMILPD $1, X7, X9        // X9 = [t3.im, t3.re]
	MOVQ ·signbit64(SB), AX     // AX = signbit
	VMOVQ AX, X12               // X12 = [signbit, 0]
	VUNPCKLPD X11, X12, X12     // X12 = [signbit, 0] - negate real position
	VXORPD X12, X9, X9          // X9 = +i*t3 = [-t3.im, t3.re]

	// ---------------------------------------------------------------
	// Final butterfly outputs
	// ---------------------------------------------------------------
	VADDPD X6, X4, X0           // X0 = y0 = t0 + t2
	VADDPD X8, X5, X1           // X1 = y1 = t1 + (-i*t3)
	VSUBPD X6, X4, X2           // X2 = y2 = t0 - t2
	VADDPD X9, X5, X3           // X3 = y3 = t1 + i*t3

	// ---------------------------------------------------------------
	// Store results
	// ---------------------------------------------------------------
	MOVQ CX, AX                 // AX = base
	ADDQ DX, AX                 // AX = base + j = idx0
	SHLQ $4, AX                 // AX = idx0 * 16
	VMOVUPD X0, (R8)(AX*1)      // store y0

	MOVQ CX, AX                 // AX = base
	ADDQ DX, AX                 // AX = base + j
	ADDQ R15, AX                // AX = idx1
	SHLQ $4, AX                 // AX = idx1 * 16
	VMOVUPD X1, (R8)(AX*1)      // store y1

	MOVQ CX, AX                 // AX = base
	ADDQ DX, AX                 // AX = base + j
	ADDQ R15, AX                // AX = base + j + quarter
	ADDQ R15, AX                // AX = idx2
	SHLQ $4, AX                 // AX = idx2 * 16
	VMOVUPD X2, (R8)(AX*1)      // store y2

	MOVQ CX, AX                 // AX = base
	ADDQ DX, AX                 // AX = base + j
	ADDQ R15, AX                // AX = base + j + quarter
	ADDQ R15, AX                // AX = base + j + 2*quarter
	ADDQ R15, AX                // AX = idx3
	SHLQ $4, AX                 // AX = idx3 * 16
	VMOVUPD X3, (R8)(AX*1)      // store y3

	INCQ DX                     // j++
	JMP  fwd_r4_c128_stage_scalar // continue scalar loop

fwd_r4_c128_stage_base_next:
	// Advance to next block within stage
	MOVQ R15, R14               // R14 = quarter
	SHLQ $2, R14                // R14 = quarter * 4 = size
	ADDQ R14, CX                // base += size
	JMP  fwd_r4_c128_stage_base // process next block

fwd_r4_c128_stage_next:
	// Advance to next stage (size *= 4)
	ADDQ $2, R12                // log2(size) += 2
	SHLQ $2, R14                // size *= 4
	JMP  fwd_r4_c128_stage_loop // process next stage

fwd_r4_c128_copy_back:
	// ===================================================================
	// Copy results back to dst if we used scratch buffer
	// ===================================================================
	MOVQ dst+0(FP), AX          // AX = original dst pointer
	CMPQ R8, AX                 // check if work buffer == dst
	JE   fwd_r4_c128_return_true // no copy needed if same

	// Copy using YMM registers (32 bytes at a time)
	XORQ CX, CX                 // CX = byte offset = 0
	MOVQ R13, DX                // DX = n
	SHLQ $4, DX                 // DX = total bytes = n * 16

fwd_r4_c128_copy_loop:
	CMPQ CX, DX                 // compare offset with total bytes
	JGE  fwd_r4_c128_return_true // done if offset >= total
	VMOVUPD (R8)(CX*1), Y0      // load 32 bytes from work buffer
	VMOVUPD Y0, (AX)(CX*1)      // store to dst
	ADDQ $32, CX                // offset += 32 bytes
	JMP  fwd_r4_c128_copy_loop  // continue copy

fwd_r4_c128_return_true:
	VZEROUPPER                  // clear upper YMM bits for SSE transition
	MOVB $1, ret+120(FP)        // return true
	RET

fwd_r4_c128_return_false:
	MOVB $0, ret+120(FP)        // return false
	RET

// ===========================================================================
// InverseAVX2Complex128Radix4Asm - Inverse FFT for complex128 using radix-4 DIT
// ===========================================================================
// func InverseAVX2Complex128Radix4Asm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
// Frame size: 0, Args size: 121 bytes
//   dst:     FP+0  (ptr), FP+8  (len), FP+16 (cap)
//   src:     FP+24 (ptr), FP+32 (len), FP+40 (cap)
//   twiddle: FP+48 (ptr), FP+56 (len), FP+64 (cap)
//   scratch: FP+72 (ptr), FP+80 (len), FP+88 (cap)
//   bitrev:  FP+96 (ptr), FP+104(len), FP+112(cap)
//   ret:     FP+120
//
// Inverse radix-4 butterfly (swaps +i/-i from forward):
//   t0 = a0 + a2, t1 = a0 - a2
//   t2 = a1 + a3, t3 = a1 - a3
//   y0 = t0 + t2
//   y1 = t1 + i*t3  (inverse: +i, forward was -i)
//   y2 = t0 - t2
//   y3 = t1 - i*t3  (inverse: -i, forward was +i)
// ===========================================================================
TEXT ·InverseAVX2Complex128Radix4Asm(SB), NOSPLIT, $0-121
	// ===================================================================
	// Parameter loading
	// ===================================================================
	MOVQ dst+0(FP), R8          // R8 = dst pointer
	MOVQ src+24(FP), R9         // R9 = src pointer
	MOVQ twiddle+48(FP), R10    // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11    // R11 = scratch pointer
	MOVQ src+32(FP), R13        // R13 = n (element count from src.len)

	// ===================================================================
	// Empty input check - empty is valid (no-op)
	// ===================================================================
	TESTQ R13, R13              // test if n == 0
	JZ    inv_r4_c128_return_true // return true for empty input

	// ===================================================================
	// Validate all slice lengths are >= n
	// ===================================================================
	MOVQ dst+8(FP), AX          // AX = dst.len
	CMPQ AX, R13                // compare dst.len with n
	JL   inv_r4_c128_return_false // fail if dst.len < n

	MOVQ twiddle+56(FP), AX     // AX = twiddle.len
	CMPQ AX, R13                // compare twiddle.len with n
	JL   inv_r4_c128_return_false // fail if twiddle.len < n

	MOVQ scratch+80(FP), AX     // AX = scratch.len
	CMPQ AX, R13                // compare scratch.len with n
	JL   inv_r4_c128_return_false // fail if scratch.len < n

	// ===================================================================
	// Trivial case: n=1, just copy single element
	// ===================================================================
	CMPQ R13, $1                // check if n == 1
	JNE  inv_r4_c128_check_power_of_2 // continue if n != 1
	VMOVUPD (R9), X0            // load 16 bytes (1 complex128) from src
	VMOVUPD X0, (R8)            // store to dst
	JMP  inv_r4_c128_return_true // return true

inv_r4_c128_check_power_of_2:
	// ===================================================================
	// Verify n is power of 2: (n & (n-1)) == 0
	// ===================================================================
	MOVQ R13, AX                // AX = n
	LEAQ -1(AX), BX             // BX = n - 1
	TESTQ AX, BX                // test n & (n-1)
	JNZ  inv_r4_c128_return_false // fail if not power of 2

	// ===================================================================
	// Require even log2 (power-of-4) and minimum size of 64
	// ===================================================================
	MOVQ R13, AX                // AX = n
	BSRQ AX, R14                // R14 = log2(n) = bit scan reverse
	TESTQ $1, R14               // check if log2(n) is odd
	JNZ  inv_r4_c128_return_false // fail if odd (not power-of-4)

	CMPQ R13, $64               // check minimum size
	JL   inv_r4_c128_return_false // fail if n < 64

	// ===================================================================
	// Select working buffer (use scratch for in-place transforms)
	// ===================================================================
	CMPQ R8, R9                 // check if dst == src
	JNE  inv_r4_c128_use_dst    // use dst if different
	MOVQ R11, R8                // in-place: use scratch as working buffer

inv_r4_c128_use_dst:
	// ===================================================================
	// Bit-reversal permutation (base-4 digit reversal)
	// For n = 4^k, reverse k base-4 digits
	// ===================================================================
	XORQ CX, CX                 // CX = i = 0 (source index)

inv_r4_c128_bitrev_loop:
	CMPQ CX, R13                // compare i with n
	JGE  inv_r4_c128_stage_init // done with bit-reversal if i >= n

	// Compute bit-reversed index in BX
	MOVQ CX, DX                 // DX = i (value to reverse)
	XORQ BX, BX                 // BX = reversed index = 0
	MOVQ R14, SI                // SI = log2(n) (bits remaining)

inv_r4_c128_bitrev_inner:
	CMPQ SI, $0                 // check if all bits processed
	JE   inv_r4_c128_bitrev_store // done reversing
	MOVQ DX, AX                 // AX = remaining value
	ANDQ $3, AX                 // AX = lowest 2 bits (base-4 digit)
	SHLQ $2, BX                 // shift reversed index left by 2
	ORQ  AX, BX                 // add new digit to reversed index
	SHRQ $2, DX                 // remove processed digit from value
	SUBQ $2, SI                 // decrement bits remaining by 2
	JMP  inv_r4_c128_bitrev_inner // process next digit

inv_r4_c128_bitrev_store:
	// Copy element from src[BX] to work[CX]
	// complex128 = 16 bytes, so offset = index * 16
	MOVQ BX, SI                 // SI = source index (bit-reversed)
	SHLQ $4, SI                 // SI = source byte offset (index * 16)
	VMOVUPD (R9)(SI*1), X0      // load src[bitrev[i]] (16 bytes)
	MOVQ CX, DI                 // DI = destination index
	SHLQ $4, DI                 // DI = dest byte offset (index * 16)
	VMOVUPD X0, (R8)(DI*1)      // store to work[i]
	INCQ CX                     // i++
	JMP  inv_r4_c128_bitrev_loop // next element

inv_r4_c128_stage_init:
	// ===================================================================
	// Radix-4 stages (size = 4, 16, 64, 256, ...)
	// For each stage, butterflies are size/4 apart
	// ===================================================================
	MOVQ $2, R12                // R12 = log2(size), starting at 2 (size=4)
	MOVQ $4, R14                // R14 = size = 4

inv_r4_c128_stage_loop:
	CMPQ R14, R13               // compare size with n
	JG   inv_r4_c128_copy_back  // done if size > n

	MOVQ R14, R15               // R15 = size
	SHRQ $2, R15                // R15 = quarter = size/4

	// step = n >> log2(size) = twiddle stride for this stage
	MOVQ R13, BX                // BX = n
	MOVQ R12, CX                // CX = log2(size)
	SHRQ CL, BX                 // BX = step = n >> log2(size)

	XORQ CX, CX                 // CX = base = 0 (start of current block)

inv_r4_c128_stage_base:
	CMPQ CX, R13                // compare base with n
	JGE  inv_r4_c128_stage_next // done with this stage if base >= n

	XORQ DX, DX                 // DX = j = 0 (element within quarter)

	// ===================================================================
	// Check for fast path: contiguous twiddles (step == 1)
	// This occurs on the final stage
	// ===================================================================
	CMPQ BX, $1                 // check if step == 1
	JNE  inv_r4_c128_stepn_prep // use strided path if step != 1
	CMPQ R15, $2                // check if quarter >= 2 for YMM path
	JL   inv_r4_c128_stage_scalar // fall back to scalar if quarter < 2

	// ===================================================================
	// YMM fast path: process 2 elements per iteration (step=1)
	// Uses YMM registers (256-bit) for 2 complex128 values
	// ===================================================================
	MOVQ R15, R11               // R11 = quarter
	SHLQ $4, R11                // R11 = quarter_bytes = quarter * 16

inv_r4_c128_step1_loop:
	MOVQ R15, AX                // AX = quarter
	SUBQ DX, AX                 // AX = remaining elements in quarter
	CMPQ AX, $2                 // check if at least 2 elements remain
	JL   inv_r4_c128_stage_scalar // fall back to scalar for remainder

	// ---------------------------------------------------------------
	// Compute element offsets
	// Base offset = (base + j) * 16
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	LEAQ (SI)(R11*1), DI        // DI = offset for quarter 1
	LEAQ (DI)(R11*1), AX        // AX = offset for quarter 2
	LEAQ (AX)(R11*1), BP        // BP = offset for quarter 3

	// ---------------------------------------------------------------
	// Load 2 complex128 from each of the 4 quarters
	// Y0 = [a0[0], a0[1]], Y1 = [a1[0], a1[1]], etc.
	// ---------------------------------------------------------------
	VMOVUPD (R8)(SI*1), Y0      // Y0 = work[base+j : base+j+2]
	VMOVUPD (R8)(DI*1), Y1      // Y1 = work[base+j+quarter : ...]
	VMOVUPD (R8)(AX*1), Y2      // Y2 = work[base+j+2*quarter : ...]
	VMOVUPD (R8)(BP*1), Y3      // Y3 = work[base+j+3*quarter : ...]

	// ---------------------------------------------------------------
	// Load twiddle factors (step=1)
	// w1 = twiddle[j], twiddle[j+1] (contiguous)
	// ---------------------------------------------------------------
	MOVQ DX, AX                 // AX = j
	SHLQ $4, AX                 // AX = j * 16 (byte offset)
	VMOVUPD (R10)(AX*1), Y4     // Y4 = w1[0:2] = twiddle[j:j+2]

	// ---------------------------------------------------------------
	// w2 = twiddle[2*j], twiddle[2*(j+1)] = twiddle[2*j], twiddle[2*j+2]
	// Stride between elements: 2 complex128 = 32 bytes
	// ---------------------------------------------------------------
	MOVQ DX, AX                 // AX = j
	SHLQ $1, AX                 // AX = 2*j
	SHLQ $4, AX                 // AX = 2*j * 16 = byte offset for twiddle[2*j]
	VMOVUPD (R10)(AX*1), X5     // X5 = twiddle[2*j]
	ADDQ $32, AX                // AX = byte offset for twiddle[2*j + 2]
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[2*(j+1)]
	VINSERTF128 $1, X6, Y5, Y5  // Y5 = [w2[0], w2[1]]

	// ---------------------------------------------------------------
	// w3 = twiddle[3*j], twiddle[3*(j+1)] = twiddle[3*j], twiddle[3*j+3]
	// Stride between elements: 3 complex128 = 48 bytes
	// ---------------------------------------------------------------
	LEAQ (DX)(DX*2), AX         // AX = 3*j
	SHLQ $4, AX                 // AX = 3*j * 16 = byte offset for twiddle[3*j]
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[3*j]
	ADDQ $48, AX                // AX = byte offset for twiddle[3*j + 3]
	VMOVUPD (R10)(AX*1), X7     // X7 = twiddle[3*(j+1)]
	VINSERTF128 $1, X7, Y6, Y6  // Y6 = [w3[0], w3[1]]

	// ---------------------------------------------------------------
	// Conjugate complex multiply a1 * conj(w1)
	// For (a + bi)(c - di) = (ac + bd) + (-ad + bc)i
	// Using VFMSUBADD instead of VFMADDSUB to negate the ad term
	// ---------------------------------------------------------------
	VUNPCKLPD Y4, Y4, Y7        // Y7 = [w1[0].re, w1[0].re, w1[1].re, w1[1].re]
	VUNPCKHPD Y4, Y4, Y8        // Y8 = [w1[0].im, w1[0].im, w1[1].im, w1[1].im]
	VPERMILPD $0x05, Y1, Y9     // Y9 = [a1.im, a1.re, ...] swapped
	VMULPD Y8, Y9, Y9           // Y9 = [a1.im*w1.im, a1.re*w1.im, ...]
	VFMSUBADD231PD Y7, Y1, Y9   // Y9 = a1*w1.re -/+ a1_swap*w1.im (conjugate)
	VMOVAPD Y9, Y1              // Y1 = a1 * conj(w1)

	// Conjugate complex multiply a2 * conj(w2)
	VUNPCKLPD Y5, Y5, Y7        // Y7 = w2.re broadcast
	VUNPCKHPD Y5, Y5, Y8        // Y8 = w2.im broadcast
	VPERMILPD $0x05, Y2, Y9     // Y9 = a2 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a2_swap * w2.im
	VFMSUBADD231PD Y7, Y2, Y9   // Y9 = a2 * conj(w2)
	VMOVAPD Y9, Y2              // Y2 = a2 * conj(w2)

	// Conjugate complex multiply a3 * conj(w3)
	VUNPCKLPD Y6, Y6, Y7        // Y7 = w3.re broadcast
	VUNPCKHPD Y6, Y6, Y8        // Y8 = w3.im broadcast
	VPERMILPD $0x05, Y3, Y9     // Y9 = a3 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a3_swap * w3.im
	VFMSUBADD231PD Y7, Y3, Y9   // Y9 = a3 * conj(w3)
	VMOVAPD Y9, Y3              // Y3 = a3 * conj(w3)

	// ---------------------------------------------------------------
	// Radix-4 butterfly computation
	// t0 = a0 + a2, t1 = a0 - a2
	// t2 = a1 + a3, t3 = a1 - a3
	// ---------------------------------------------------------------
	VADDPD Y2, Y0, Y10          // Y10 = t0 = a0 + a2
	VSUBPD Y2, Y0, Y11          // Y11 = t1 = a0 - a2
	VADDPD Y3, Y1, Y12          // Y12 = t2 = a1 + a3
	VSUBPD Y3, Y1, Y13          // Y13 = t3 = a1 - a3

	// ---------------------------------------------------------------
	// Compute +i*t3: multiply by +i = (0, 1)
	// i * (a + bi) = (-b, a) = (-im, re)
	// Swap re<->im, then negate the original imag (now at even positions)
	// (INVERSE: uses +i for y1, opposite of forward)
	// ---------------------------------------------------------------
	VPERMILPD $0x05, Y13, Y14   // Y14 = [t3.im, t3.re, ...] swapped
	// Create sign mask [signbit, 0, signbit, 0] to negate even positions
	MOVQ ·signbit64(SB), AX     // AX = signbit for float64
	VPXOR X15, X15, X15         // X15 = zero
	VMOVQ AX, X7                // X7 = [signbit, 0]
	VPUNPCKLQDQ X15, X7, X7     // X7 = [signbit, 0] - mask for +i in low lane
	VINSERTI128 $1, X7, Y7, Y7  // Y7 high = X7
	VINSERTI128 $0, X7, Y7, Y7  // Y7 = [signbit, 0, signbit, 0]
	VXORPD Y7, Y14, Y14         // Y14 = +i*t3 = [-t3.im, t3.re, ...]

	// ---------------------------------------------------------------
	// Compute -i*t3: multiply by -i = (0, -1)
	// -i * (a + bi) = (b, -a) = (im, -re)
	// Swap re<->im, then negate the original real (now at odd positions)
	// (INVERSE: uses -i for y3, opposite of forward)
	// ---------------------------------------------------------------
	VPERMILPD $0x05, Y13, Y8    // Y8 = [t3.im, t3.re, ...] swapped
	VMOVQ AX, X9                // X9 = [signbit, 0]
	VPUNPCKLQDQ X9, X15, X9     // X9 = [0, signbit] - mask for -i in low lane
	VINSERTI128 $1, X9, Y9, Y9  // Y9 high = X9
	VINSERTI128 $0, X9, Y9, Y9  // Y9 = [0, signbit, 0, signbit]
	VXORPD Y9, Y8, Y8           // Y8 = -i*t3 = [t3.im, -t3.re, ...]

	// ---------------------------------------------------------------
	// Final butterfly outputs (INVERSE formulas)
	// y0 = t0 + t2
	// y1 = t1 + i*t3  (inverse: +i*t3)
	// y2 = t0 - t2
	// y3 = t1 - i*t3 = t1 + (-i*t3)  (inverse: -i*t3)
	// ---------------------------------------------------------------
	VADDPD Y12, Y10, Y0         // Y0 = y0 = t0 + t2
	VADDPD Y14, Y11, Y1         // Y1 = y1 = t1 + (+i*t3)
	VSUBPD Y12, Y10, Y2         // Y2 = y2 = t0 - t2
	VADDPD Y8, Y11, Y3          // Y3 = y3 = t1 + (-i*t3)

	// ---------------------------------------------------------------
	// Recompute store offsets and store results
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16
	LEAQ (SI)(R11*1), DI        // DI = offset for quarter 1
	LEAQ (DI)(R11*1), AX        // AX = offset for quarter 2
	LEAQ (AX)(R11*1), BP        // BP = offset for quarter 3

	VMOVUPD Y0, (R8)(SI*1)      // store y0
	VMOVUPD Y1, (R8)(DI*1)      // store y1
	VMOVUPD Y2, (R8)(AX*1)      // store y2
	VMOVUPD Y3, (R8)(BP*1)      // store y3

	ADDQ $2, DX                 // j += 2 (processed 2 elements)
	JMP  inv_r4_c128_step1_loop // continue loop

inv_r4_c128_stepn_prep:
	// ===================================================================
	// YMM path for strided twiddles (step > 1)
	// ===================================================================
	CMPQ R15, $2                // check if quarter >= 2
	JL   inv_r4_c128_stage_scalar // fall back to scalar

	MOVQ R15, R11               // R11 = quarter
	SHLQ $4, R11                // R11 = quarter_bytes = quarter * 16

	MOVQ BX, R9                 // R9 = step
	SHLQ $4, R9                 // R9 = stride1_bytes = step * 16

	MOVQ DX, DI                 // DI = j (starting at 0)
	IMULQ BX, DI                // DI = j * step
	SHLQ $4, DI                 // DI = j * step * 16 (twiddle byte offset)

inv_r4_c128_stepn_loop:
	MOVQ R15, AX                // AX = quarter
	SUBQ DX, AX                 // AX = remaining elements
	CMPQ AX, $2                 // check if at least 2 elements remain
	JL   inv_r4_c128_stage_scalar // fall back to scalar

	// ---------------------------------------------------------------
	// Compute element offsets
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	LEAQ (SI)(R11*1), R14       // R14 = offset for quarter 1
	LEAQ (R14)(R11*1), AX       // AX = offset for quarter 2
	LEAQ (AX)(R11*1), BP        // BP = offset for quarter 3

	// ---------------------------------------------------------------
	// Load data from 4 quarters
	// ---------------------------------------------------------------
	VMOVUPD (R8)(SI*1), Y0      // Y0 = work[base+j : base+j+2]
	VMOVUPD (R8)(R14*1), Y1     // Y1 = work[base+j+quarter : ...]
	VMOVUPD (R8)(AX*1), Y2      // Y2 = work[base+j+2*quarter : ...]
	VMOVUPD (R8)(BP*1), Y3      // Y3 = work[base+j+3*quarter : ...]

	// ---------------------------------------------------------------
	// Compute twiddle base offsets
	// For strided access: w1[k] = twiddle[(j+k)*step]
	// ---------------------------------------------------------------
	MOVQ DI, AX                 // AX = j*step*16 (w1 base offset)
	MOVQ AX, BP                 // BP = w1 offset

	// w1: gather twiddle[(j+0)*step], twiddle[(j+1)*step]
	VMOVUPD (R10)(BP*1), X4     // X4 = twiddle[j*step]
	ADDQ R9, BP                 // BP += step*16
	VMOVUPD (R10)(BP*1), X5     // X5 = twiddle[(j+1)*step]
	VINSERTF128 $1, X5, Y4, Y4  // Y4 = [w1[0], w1[1]]

	// w2: gather twiddle[2*j*step + k*2*step]
	MOVQ DI, BP                 // BP = j*step*16
	SHLQ $1, BP                 // BP = 2*j*step*16
	MOVQ R9, R14                // R14 = step*16
	SHLQ $1, R14                // R14 = 2*step*16 (stride for w2)
	VMOVUPD (R10)(BP*1), X5     // X5 = twiddle[2*j*step]
	ADDQ R14, BP                // BP += 2*step*16
	VMOVUPD (R10)(BP*1), X6     // X6 = twiddle[2*(j+1)*step]
	VINSERTF128 $1, X6, Y5, Y5  // Y5 = [w2[0], w2[1]]

	// w3: gather twiddle[3*j*step + k*3*step]
	MOVQ DI, BP                 // BP = j*step*16
	LEAQ (BP)(BP*2), BP         // BP = 3*j*step*16
	LEAQ (R9)(R9*2), R14        // R14 = 3*step*16 (stride for w3)
	VMOVUPD (R10)(BP*1), X6     // X6 = twiddle[3*j*step]
	ADDQ R14, BP                // BP += 3*step*16
	VMOVUPD (R10)(BP*1), X7     // X7 = twiddle[3*(j+1)*step]
	VINSERTF128 $1, X7, Y6, Y6  // Y6 = [w3[0], w3[1]]

	// Advance twiddle offset for next iteration
	ADDQ R9, DI                 // DI += step*16
	ADDQ R9, DI                 // DI += step*16 (total: 2*step*16)

	// ---------------------------------------------------------------
	// Conjugate complex multiply a1 * conj(w1)
	// ---------------------------------------------------------------
	VUNPCKLPD Y4, Y4, Y7        // Y7 = w1.re broadcast
	VUNPCKHPD Y4, Y4, Y8        // Y8 = w1.im broadcast
	VPERMILPD $0x05, Y1, Y9     // Y9 = a1 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a1_swap * w1.im
	VFMSUBADD231PD Y7, Y1, Y9   // Y9 = a1*w1.re -/+ a1_swap*w1.im (conjugate)
	VMOVAPD Y9, Y1              // Y1 = a1 * conj(w1)

	// Conjugate complex multiply a2 * conj(w2)
	VUNPCKLPD Y5, Y5, Y7        // Y7 = w2.re broadcast
	VUNPCKHPD Y5, Y5, Y8        // Y8 = w2.im broadcast
	VPERMILPD $0x05, Y2, Y9     // Y9 = a2 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a2_swap * w2.im
	VFMSUBADD231PD Y7, Y2, Y9   // Y9 = a2 * conj(w2)
	VMOVAPD Y9, Y2              // Y2 = a2 * conj(w2)

	// Conjugate complex multiply a3 * conj(w3)
	VUNPCKLPD Y6, Y6, Y7        // Y7 = w3.re broadcast
	VUNPCKHPD Y6, Y6, Y8        // Y8 = w3.im broadcast
	VPERMILPD $0x05, Y3, Y9     // Y9 = a3 swapped
	VMULPD Y8, Y9, Y9           // Y9 = a3_swap * w3.im
	VFMSUBADD231PD Y7, Y3, Y9   // Y9 = a3 * conj(w3)
	VMOVAPD Y9, Y3              // Y3 = a3 * conj(w3)

	// ---------------------------------------------------------------
	// Radix-4 butterfly
	// ---------------------------------------------------------------
	VADDPD Y2, Y0, Y10          // Y10 = t0 = a0 + a2
	VSUBPD Y2, Y0, Y11          // Y11 = t1 = a0 - a2
	VADDPD Y3, Y1, Y12          // Y12 = t2 = a1 + a3
	VSUBPD Y3, Y1, Y13          // Y13 = t3 = a1 - a3

	// +i*t3: swap re<->im, negate original imag (now at even positions)
	// (INVERSE: uses +i for y1)
	VPERMILPD $0x05, Y13, Y14   // Y14 = t3 swapped
	MOVQ ·signbit64(SB), AX     // AX = signbit
	VPXOR X15, X15, X15         // X15 = zero
	VMOVQ AX, X7                // X7 = [signbit, 0]
	VPUNPCKLQDQ X15, X7, X7     // X7 = [signbit, 0] - mask for +i
	VINSERTI128 $1, X7, Y7, Y7  // Y7 high = X7
	VINSERTI128 $0, X7, Y7, Y7  // Y7 = [signbit, 0, signbit, 0]
	VXORPD Y7, Y14, Y14         // Y14 = +i*t3

	// -i*t3: swap re<->im, negate original real (now at odd positions)
	// (INVERSE: uses -i for y3)
	VPERMILPD $0x05, Y13, Y8    // Y8 = t3 swapped
	VMOVQ AX, X9                // X9 = [signbit, 0]
	VPUNPCKLQDQ X9, X15, X9     // X9 = [0, signbit] - mask for -i
	VINSERTI128 $1, X9, Y9, Y9  // Y9 high = X9
	VINSERTI128 $0, X9, Y9, Y9  // Y9 = [0, signbit, 0, signbit]
	VXORPD Y9, Y8, Y8           // Y8 = -i*t3

	// Final outputs (INVERSE formulas)
	VADDPD Y12, Y10, Y0         // Y0 = y0 = t0 + t2
	VADDPD Y14, Y11, Y1         // Y1 = y1 = t1 + (+i*t3)
	VSUBPD Y12, Y10, Y2         // Y2 = y2 = t0 - t2
	VADDPD Y8, Y11, Y3          // Y3 = y3 = t1 + (-i*t3)

	// ---------------------------------------------------------------
	// Recompute store offsets and store results
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16
	LEAQ (SI)(R11*1), R14       // R14 = offset for quarter 1
	LEAQ (R14)(R11*1), AX       // AX = offset for quarter 2
	LEAQ (AX)(R11*1), BP        // BP = offset for quarter 3

	VMOVUPD Y0, (R8)(SI*1)      // store y0
	VMOVUPD Y1, (R8)(R14*1)     // store y1
	VMOVUPD Y2, (R8)(AX*1)      // store y2
	VMOVUPD Y3, (R8)(BP*1)      // store y3

	ADDQ $2, DX                 // j += 2
	JMP  inv_r4_c128_stepn_loop // continue loop

inv_r4_c128_stage_scalar:
	// ===================================================================
	// Scalar fallback: process one element at a time using XMM
	// ===================================================================
	CMPQ DX, R15                // compare j with quarter
	JGE  inv_r4_c128_stage_base_next // done with this block

	// ---------------------------------------------------------------
	// Compute indices for 4 elements in the butterfly
	// idx0 = base + j
	// idx1 = base + j + quarter
	// idx2 = base + j + 2*quarter
	// idx3 = base + j + 3*quarter
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = idx0 = base + j
	MOVQ SI, DI                 // DI = idx0
	ADDQ R15, DI                // DI = idx1 = base + j + quarter
	MOVQ DI, R11                // R11 = idx1
	ADDQ R15, R11               // R11 = idx2 = base + j + 2*quarter
	MOVQ R11, R9                // R9 = idx2
	ADDQ R15, R9                // R9 = idx3 = base + j + 3*quarter

	// ---------------------------------------------------------------
	// Compute twiddle indices
	// w1 = twiddle[j * step]
	// w2 = twiddle[2 * j * step]
	// w3 = twiddle[3 * j * step]
	// ---------------------------------------------------------------
	MOVQ DX, AX                 // AX = j
	IMULQ BX, AX                // AX = j * step
	MOVQ AX, BP                 // BP = w1 index = j * step
	SHLQ $4, BP                 // BP = w1 byte offset
	VMOVUPD (R10)(BP*1), X8     // X8 = w1 = twiddle[j*step]

	MOVQ AX, BP                 // BP = j * step
	SHLQ $1, BP                 // BP = 2 * j * step
	SHLQ $4, BP                 // BP = w2 byte offset
	VMOVUPD (R10)(BP*1), X9     // X9 = w2 = twiddle[2*j*step]

	LEAQ (AX)(AX*2), BP         // BP = 3 * j * step
	SHLQ $4, BP                 // BP = w3 byte offset
	VMOVUPD (R10)(BP*1), X10    // X10 = w3 = twiddle[3*j*step]

	// ---------------------------------------------------------------
	// Load 4 input elements
	// ---------------------------------------------------------------
	MOVQ SI, AX                 // AX = idx0
	SHLQ $4, AX                 // AX = idx0 * 16
	VMOVUPD (R8)(AX*1), X0      // X0 = a0 = work[idx0]

	MOVQ DI, AX                 // AX = idx1
	SHLQ $4, AX                 // AX = idx1 * 16
	VMOVUPD (R8)(AX*1), X1      // X1 = a1 = work[idx1]

	MOVQ R11, AX                // AX = idx2
	SHLQ $4, AX                 // AX = idx2 * 16
	VMOVUPD (R8)(AX*1), X2      // X2 = a2 = work[idx2]

	MOVQ R9, AX                 // AX = idx3
	SHLQ $4, AX                 // AX = idx3 * 16
	VMOVUPD (R8)(AX*1), X3      // X3 = a3 = work[idx3]

	// ---------------------------------------------------------------
	// Conjugate complex multiply a1 * conj(w1) (XMM scalar version)
	// ---------------------------------------------------------------
	VMOVDDUP X8, X11            // X11 = [w1.re, w1.re]
	VPERMILPD $1, X8, X12       // X12 = [w1.im, w1.re] -> want [w1.im, w1.im]
	VMOVDDUP X12, X12           // X12 = [w1.im, w1.im]
	VPERMILPD $1, X1, X13       // X13 = [a1.im, a1.re]
	VMULPD X12, X13, X13        // X13 = [a1.im*w1.im, a1.re*w1.im]
	VFMSUBADD231PD X11, X1, X13 // X13 = a1*conj(w1)
	VMOVAPD X13, X1             // X1 = a1 * conj(w1)

	// Conjugate complex multiply a2 * conj(w2)
	VMOVDDUP X9, X11            // X11 = [w2.re, w2.re]
	VPERMILPD $1, X9, X12       // X12 -> [w2.im, ...]
	VMOVDDUP X12, X12           // X12 = [w2.im, w2.im]
	VPERMILPD $1, X2, X13       // X13 = a2 swapped
	VMULPD X12, X13, X13        // X13 = a2_swap * w2.im
	VFMSUBADD231PD X11, X2, X13 // X13 = a2 * conj(w2)
	VMOVAPD X13, X2             // X2 = a2 * conj(w2)

	// Conjugate complex multiply a3 * conj(w3)
	VMOVDDUP X10, X11           // X11 = [w3.re, w3.re]
	VPERMILPD $1, X10, X12      // X12 -> [w3.im, ...]
	VMOVDDUP X12, X12           // X12 = [w3.im, w3.im]
	VPERMILPD $1, X3, X13       // X13 = a3 swapped
	VMULPD X12, X13, X13        // X13 = a3_swap * w3.im
	VFMSUBADD231PD X11, X3, X13 // X13 = a3 * conj(w3)
	VMOVAPD X13, X3             // X3 = a3 * conj(w3)

	// ---------------------------------------------------------------
	// Radix-4 butterfly (XMM)
	// ---------------------------------------------------------------
	VADDPD X2, X0, X4           // X4 = t0 = a0 + a2
	VSUBPD X2, X0, X5           // X5 = t1 = a0 - a2
	VADDPD X3, X1, X6           // X6 = t2 = a1 + a3
	VSUBPD X3, X1, X7           // X7 = t3 = a1 - a3

	// ---------------------------------------------------------------
	// Compute +i*t3 for XMM (INVERSE: uses +i for y1)
	// i * (a + bi) = (-b, a)
	// ---------------------------------------------------------------
	VPERMILPD $1, X7, X8        // X8 = [t3.im, t3.re]
	MOVQ ·signbit64(SB), AX     // AX = signbit
	VMOVQ AX, X10               // X10 = [signbit, 0]
	VXORPD X11, X11, X11        // X11 = zero
	VUNPCKLPD X11, X10, X10     // X10 = [signbit, 0] - negate real position
	VXORPD X10, X8, X8          // X8 = +i*t3 = [-t3.im, t3.re]

	// ---------------------------------------------------------------
	// Compute -i*t3 for XMM (INVERSE: uses -i for y3)
	// -i * (a + bi) = (b, -a)
	// ---------------------------------------------------------------
	VPERMILPD $1, X7, X9        // X9 = [t3.im, t3.re]
	MOVQ ·signbit64(SB), AX     // AX = signbit
	VMOVQ AX, X12               // X12 = [signbit, 0]
	VUNPCKLPD X12, X11, X12     // X12 = [0, signbit] - negate imag position
	VXORPD X12, X9, X9          // X9 = -i*t3 = [t3.im, -t3.re]

	// ---------------------------------------------------------------
	// Final butterfly outputs (INVERSE formulas)
	// ---------------------------------------------------------------
	VADDPD X6, X4, X0           // X0 = y0 = t0 + t2
	VADDPD X8, X5, X1           // X1 = y1 = t1 + (+i*t3)
	VSUBPD X6, X4, X2           // X2 = y2 = t0 - t2
	VADDPD X9, X5, X3           // X3 = y3 = t1 + (-i*t3)

	// ---------------------------------------------------------------
	// Store results
	// ---------------------------------------------------------------
	MOVQ CX, AX                 // AX = base
	ADDQ DX, AX                 // AX = base + j = idx0
	SHLQ $4, AX                 // AX = idx0 * 16
	VMOVUPD X0, (R8)(AX*1)      // store y0

	MOVQ CX, AX                 // AX = base
	ADDQ DX, AX                 // AX = base + j
	ADDQ R15, AX                // AX = idx1
	SHLQ $4, AX                 // AX = idx1 * 16
	VMOVUPD X1, (R8)(AX*1)      // store y1

	MOVQ CX, AX                 // AX = base
	ADDQ DX, AX                 // AX = base + j
	ADDQ R15, AX                // AX = base + j + quarter
	ADDQ R15, AX                // AX = idx2
	SHLQ $4, AX                 // AX = idx2 * 16
	VMOVUPD X2, (R8)(AX*1)      // store y2

	MOVQ CX, AX                 // AX = base
	ADDQ DX, AX                 // AX = base + j
	ADDQ R15, AX                // AX = base + j + quarter
	ADDQ R15, AX                // AX = base + j + 2*quarter
	ADDQ R15, AX                // AX = idx3
	SHLQ $4, AX                 // AX = idx3 * 16
	VMOVUPD X3, (R8)(AX*1)      // store y3

	INCQ DX                     // j++
	JMP  inv_r4_c128_stage_scalar // continue scalar loop

inv_r4_c128_stage_base_next:
	// Advance to next block within stage
	MOVQ R15, R14               // R14 = quarter
	SHLQ $2, R14                // R14 = quarter * 4 = size
	ADDQ R14, CX                // base += size
	JMP  inv_r4_c128_stage_base // process next block

inv_r4_c128_stage_next:
	// Advance to next stage (size *= 4)
	ADDQ $2, R12                // log2(size) += 2
	SHLQ $2, R14                // size *= 4
	JMP  inv_r4_c128_stage_loop // process next stage

inv_r4_c128_copy_back:
	// ===================================================================
	// Copy results back to dst if we used scratch buffer
	// ===================================================================
	MOVQ dst+0(FP), AX          // AX = original dst pointer
	CMPQ R8, AX                 // check if work buffer == dst
	JE   inv_r4_c128_return_true // no copy needed if same

	// Copy using YMM registers (32 bytes at a time)
	XORQ CX, CX                 // CX = byte offset = 0
	MOVQ R13, DX                // DX = n
	SHLQ $4, DX                 // DX = total bytes = n * 16

inv_r4_c128_copy_loop:
	CMPQ CX, DX                 // compare offset with total bytes
	JGE  inv_r4_c128_return_true // done if offset >= total
	VMOVUPD (R8)(CX*1), Y0      // load 32 bytes from work buffer
	VMOVUPD Y0, (AX)(CX*1)      // store to dst
	ADDQ $32, CX                // offset += 32 bytes
	JMP  inv_r4_c128_copy_loop  // continue copy

inv_r4_c128_return_true:
	VZEROUPPER                  // clear upper YMM bits for SSE transition
	MOVB $1, ret+120(FP)        // return true
	RET

inv_r4_c128_return_false:
	MOVB $0, ret+120(FP)        // return false
	RET
