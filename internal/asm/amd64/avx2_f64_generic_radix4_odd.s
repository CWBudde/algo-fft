//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2/FMA-optimized FFT Assembly for AMD64 - complex128 (float64)
// Mixed radix-4 + final radix-2 DIT path for odd log2 sizes (n=32, 128, 512, ...).
// ===========================================================================
//
// Data layout: complex128 = 16 bytes (re: float64, im: float64)
// YMM register (256-bit) holds 2 complex128 values (4 float64)
// XMM register (128-bit) holds 1 complex128 value (2 float64)
//
// For odd log2(n), we perform:
//   - (log2(n)-1)/2 radix-4 stages
//   - 1 final radix-2 stage
//
// Radix-4 butterfly:
//   t0 = a0 + a2, t1 = a0 - a2
//   t2 = a1 + a3, t3 = a1 - a3
//   y0 = t0 + t2
//   y1 = t1 - i*t3  (forward: -i, inverse: +i)
//   y2 = t0 - t2
//   y3 = t1 + i*t3  (forward: +i, inverse: -i)
//
// Radix-2 butterfly:
//   y0 = a + w*b
//   y1 = a - w*b
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// ForwardAVX2Complex128Radix4MixedAsm - Forward FFT for complex128 using
// radix-4 stages followed by a final radix-2 stage (for odd log2 sizes).
// ===========================================================================
// func ForwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
// Frame size: 0, Args size: 121 bytes
//   dst:     FP+0  (ptr), FP+8  (len), FP+16 (cap)
//   src:     FP+24 (ptr), FP+32 (len), FP+40 (cap)
//   twiddle: FP+48 (ptr), FP+56 (len), FP+64 (cap)
//   scratch: FP+72 (ptr), FP+80 (len), FP+88 (cap)
//   bitrev:  FP+96 (ptr), FP+104(len), FP+112(cap)
//   ret:     FP+120
// ===========================================================================
TEXT ·ForwardAVX2Complex128Radix4MixedAsm(SB), NOSPLIT, $0-121
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
	JZ    fwd_r4m_c128_return_true // return true for empty input

	// ===================================================================
	// Validate all slice lengths are >= n
	// ===================================================================
	MOVQ dst+8(FP), AX          // AX = dst.len
	CMPQ AX, R13                // compare dst.len with n
	JL   fwd_r4m_c128_return_false // fail if dst.len < n

	MOVQ twiddle+56(FP), AX     // AX = twiddle.len
	CMPQ AX, R13                // compare twiddle.len with n
	JL   fwd_r4m_c128_return_false // fail if twiddle.len < n

	MOVQ scratch+80(FP), AX     // AX = scratch.len
	CMPQ AX, R13                // compare scratch.len with n
	JL   fwd_r4m_c128_return_false // fail if scratch.len < n

	// ===================================================================
	// Trivial case: n=1, just copy single element
	// ===================================================================
	CMPQ R13, $1                // check if n == 1
	JNE  fwd_r4m_c128_check_power_of_2 // continue if n != 1
	VMOVUPD (R9), X0            // load 16 bytes (1 complex128) from src
	VMOVUPD X0, (R8)            // store to dst
	JMP  fwd_r4m_c128_return_true // return true

fwd_r4m_c128_check_power_of_2:
	// ===================================================================
	// Verify n is power of 2: (n & (n-1)) == 0
	// ===================================================================
	MOVQ R13, AX                // AX = n
	LEAQ -1(AX), BX             // BX = n - 1
	TESTQ AX, BX                // test n & (n-1)
	JNZ  fwd_r4m_c128_return_false // fail if not power of 2

	// ===================================================================
	// Require odd log2 and minimum size of 32
	// ===================================================================
	MOVQ R13, AX                // AX = n
	BSRQ AX, R12                // R12 = log2(n) = bit scan reverse
	TESTQ $1, R12               // check if log2(n) is odd
	JZ   fwd_r4m_c128_return_false // fail if even (use radix-4 even path)

	CMPQ R13, $32               // check minimum size
	JL   fwd_r4m_c128_return_false // fail if n < 32

	// ===================================================================
	// Number of radix-4 stages: k = (log2(n)-1)/2
	// ===================================================================
	MOVQ R12, R14               // R14 = log2(n)
	SUBQ $1, R14                // R14 = log2(n) - 1
	SHRQ $1, R14                // R14 = k (radix-4 stage count)

	// ===================================================================
	// Select working buffer (use scratch for in-place transforms)
	// ===================================================================
	CMPQ R8, R9                 // check if dst == src
	JNE  fwd_r4m_c128_use_dst   // use dst if different
	MOVQ R11, R8                // in-place: use scratch as working buffer

fwd_r4m_c128_use_dst:
	// ===================================================================
	// Bit-reversal permutation (mixed radix: base-4 digits, then top bit)
	// For n = 2 * 4^k, reverse k base-4 digits, then append top bit
	// ===================================================================
	XORQ CX, CX                 // CX = i = 0 (source index)

fwd_r4m_c128_bitrev_loop:
	CMPQ CX, R13                // compare i with n
	JGE  fwd_r4m_c128_stage_init // done with bit-reversal if i >= n

	// Compute bit-reversed index in BX
	MOVQ CX, DX                 // DX = i (value to reverse)
	XORQ BX, BX                 // BX = reversed index = 0
	MOVQ R14, SI                // SI = k (radix-4 stages)

fwd_r4m_c128_bitrev_inner:
	CMPQ SI, $0                 // check if all base-4 digits processed
	JE   fwd_r4m_c128_bitrev_store // done with base-4 reversal
	MOVQ DX, AX                 // AX = remaining value
	ANDQ $3, AX                 // AX = lowest 2 bits (base-4 digit)
	SHLQ $2, BX                 // shift reversed index left by 2
	ORQ  AX, BX                 // add new digit to reversed index
	SHRQ $2, DX                 // remove processed digit from value
	DECQ SI                     // decrement stage counter
	JMP  fwd_r4m_c128_bitrev_inner // process next digit

fwd_r4m_c128_bitrev_store:
	// Append top bit: rev = (rev << 1) | (remaining top bit)
	SHLQ $1, BX                 // shift reversed index left by 1
	ORQ  DX, BX                 // append remaining top bit

	// Copy element from src[BX] to work[CX]
	// complex128 = 16 bytes, so offset = index * 16
	MOVQ BX, SI                 // SI = source index (bit-reversed)
	SHLQ $4, SI                 // SI = source byte offset (index * 16)
	VMOVUPD (R9)(SI*1), X0      // load src[bitrev[i]] (16 bytes)
	MOVQ CX, DI                 // DI = destination index
	SHLQ $4, DI                 // DI = dest byte offset (index * 16)
	VMOVUPD X0, (R8)(DI*1)      // store to work[i]
	INCQ CX                     // i++
	JMP  fwd_r4m_c128_bitrev_loop // next element

fwd_r4m_c128_stage_init:
	// ===================================================================
	// Radix-4 stages (size = 4, 16, 64, ...)
	// For each stage, butterflies are size/4 apart
	// ===================================================================
	MOVQ $2, R12                // R12 = log2(size), starting at 2 (size=4)
	MOVQ $4, R15                // R15 = size = 4

fwd_r4m_c128_stage_loop:
	CMPQ R14, $0                // check if radix-4 stages remaining
	JE   fwd_r4m_c128_radix2_stage // no more radix-4, do final radix-2

	MOVQ R15, R11               // R11 = size
	SHRQ $2, R11                // R11 = quarter = size/4

	// step = n >> log2(size) = twiddle stride for this stage
	MOVQ R13, BX                // BX = n
	MOVQ R12, CX                // CX = log2(size)
	SHRQ CL, BX                 // BX = step = n >> log2(size)

	XORQ CX, CX                 // CX = base = 0 (start of current block)

fwd_r4m_c128_stage_base:
	CMPQ CX, R13                // compare base with n
	JGE  fwd_r4m_c128_stage_next // done with this stage if base >= n

	XORQ DX, DX                 // DX = j = 0 (element within quarter)

	// ===================================================================
	// Check for fast path: contiguous twiddles (step == 1)
	// ===================================================================
	CMPQ BX, $1                 // check if step == 1
	JNE  fwd_r4m_c128_stepn_prep // use strided path if step != 1
	CMPQ R11, $2                // check if quarter >= 2 for YMM path
	JL   fwd_r4m_c128_stage_scalar // fall back to scalar if quarter < 2

	// ===================================================================
	// YMM fast path: process 2 elements per iteration (step=1)
	// ===================================================================
	MOVQ R11, R9                // R9 = quarter
	SHLQ $4, R9                 // R9 = quarter_bytes = quarter * 16

fwd_r4m_c128_step1_loop:
	MOVQ R11, AX                // AX = quarter
	SUBQ DX, AX                 // AX = remaining elements in quarter
	CMPQ AX, $2                 // check if at least 2 elements remain
	JL   fwd_r4m_c128_stage_scalar // fall back to scalar for remainder

	// ---------------------------------------------------------------
	// Compute element offsets
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	LEAQ (SI)(R9*1), DI         // DI = offset for quarter 1
	LEAQ (DI)(R9*1), AX         // AX = offset for quarter 2
	LEAQ (AX)(R9*1), BP         // BP = offset for quarter 3

	// ---------------------------------------------------------------
	// Load 2 complex128 from each of the 4 quarters
	// ---------------------------------------------------------------
	VMOVUPD (R8)(SI*1), Y0      // Y0 = work[base+j : base+j+2]
	VMOVUPD (R8)(DI*1), Y1      // Y1 = work[base+j+quarter : ...]
	VMOVUPD (R8)(AX*1), Y2      // Y2 = work[base+j+2*quarter : ...]
	VMOVUPD (R8)(BP*1), Y3      // Y3 = work[base+j+3*quarter : ...]

	// ---------------------------------------------------------------
	// Load twiddle factors (step=1, contiguous)
	// ---------------------------------------------------------------
	MOVQ DX, AX                 // AX = j
	SHLQ $4, AX                 // AX = j * 16 (byte offset)
	VMOVUPD (R10)(AX*1), Y4     // Y4 = w1[0:2] = twiddle[j:j+2]

	// w2 = twiddle[2*j], twiddle[2*(j+1)]
	MOVQ DX, AX                 // AX = j
	SHLQ $1, AX                 // AX = 2*j
	SHLQ $4, AX                 // AX = 2*j * 16
	VMOVUPD (R10)(AX*1), X5     // X5 = twiddle[2*j]
	ADDQ $32, AX                // skip 2 complex128
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[2*(j+1)]
	VINSERTF128 $1, X6, Y5, Y5  // Y5 = [w2[0], w2[1]]

	// w3 = twiddle[3*j], twiddle[3*(j+1)]
	LEAQ (DX)(DX*2), AX         // AX = 3*j
	SHLQ $4, AX                 // AX = 3*j * 16
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[3*j]
	ADDQ $48, AX                // skip 3 complex128
	VMOVUPD (R10)(AX*1), X7     // X7 = twiddle[3*(j+1)]
	VINSERTF128 $1, X7, Y6, Y6  // Y6 = [w3[0], w3[1]]

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

	// Compute -i*t3: swap re<->im, negate original real (odd positions)
	VPERMILPD $0x05, Y13, Y14   // Y14 = [t3.im, t3.re, ...] swapped
	MOVQ ·signbit64(SB), AX     // AX = signbit for float64
	VPXOR X15, X15, X15         // X15 = zero
	VMOVQ AX, X7                // X7 = [signbit, 0]
	VPUNPCKLQDQ X7, X15, X7     // X7 = [0, signbit]
	VINSERTI128 $1, X7, Y7, Y7  // broadcast to Y7
	VINSERTI128 $0, X7, Y7, Y7  // Y7 = [0, signbit, 0, signbit]
	VXORPD Y7, Y14, Y14         // Y14 = -i*t3 = [t3.im, -t3.re, ...]

	// Compute +i*t3: swap re<->im, negate original imag (even positions)
	VPERMILPD $0x05, Y13, Y8    // Y8 = [t3.im, t3.re, ...] swapped
	VMOVQ AX, X9                // X9 = [signbit, 0]
	VPUNPCKLQDQ X15, X9, X9     // X9 = [signbit, 0]
	VINSERTI128 $1, X9, Y9, Y9  // broadcast to Y9
	VINSERTI128 $0, X9, Y9, Y9  // Y9 = [signbit, 0, signbit, 0]
	VXORPD Y9, Y8, Y8           // Y8 = +i*t3 = [-t3.im, t3.re, ...]

	// Final butterfly outputs
	VADDPD Y12, Y10, Y0         // Y0 = y0 = t0 + t2
	VADDPD Y14, Y11, Y1         // Y1 = y1 = t1 + (-i*t3)
	VSUBPD Y12, Y10, Y2         // Y2 = y2 = t0 - t2
	VADDPD Y8, Y11, Y3          // Y3 = y3 = t1 + (+i*t3)

	// Store results
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16
	LEAQ (SI)(R9*1), DI         // DI = offset for quarter 1
	LEAQ (DI)(R9*1), AX         // AX = offset for quarter 2
	LEAQ (AX)(R9*1), BP         // BP = offset for quarter 3

	VMOVUPD Y0, (R8)(SI*1)      // store y0
	VMOVUPD Y1, (R8)(DI*1)      // store y1
	VMOVUPD Y2, (R8)(AX*1)      // store y2
	VMOVUPD Y3, (R8)(BP*1)      // store y3

	ADDQ $2, DX                 // j += 2
	JMP  fwd_r4m_c128_step1_loop

fwd_r4m_c128_stepn_prep:
	// ===================================================================
	// YMM path for strided twiddles (step > 1)
	// ===================================================================
	CMPQ R11, $2                // check if quarter >= 2
	JL   fwd_r4m_c128_stage_scalar // fall back to scalar

	MOVQ R11, R9                // R9 = quarter
	SHLQ $4, R9                 // R9 = quarter_bytes = quarter * 16

fwd_r4m_c128_stepn_loop:
	MOVQ R11, AX                // AX = quarter
	SUBQ DX, AX                 // AX = remaining elements
	CMPQ AX, $2                 // check if at least 2 elements remain
	JL   fwd_r4m_c128_stage_scalar // fall back to scalar

	// ---------------------------------------------------------------
	// Compute element offsets
	// ---------------------------------------------------------------
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	LEAQ (SI)(R9*1), DI         // DI = offset for quarter 1
	LEAQ (DI)(R9*1), AX         // AX = offset for quarter 2
	LEAQ (AX)(R9*1), BP         // BP = offset for quarter 3

	// Load data from 4 quarters
	VMOVUPD (R8)(SI*1), Y0      // Y0 = work[base+j : base+j+2]
	VMOVUPD (R8)(DI*1), Y1      // Y1 = work[base+j+quarter : ...]
	VMOVUPD (R8)(AX*1), Y2      // Y2 = work[base+j+2*quarter : ...]
	VMOVUPD (R8)(BP*1), Y3      // Y3 = work[base+j+3*quarter : ...]

	// ---------------------------------------------------------------
	// Load twiddle factors with stride
	// w1[k] = twiddle[(j+k)*step]
	// ---------------------------------------------------------------
	MOVQ DX, DI                 // DI = j
	IMULQ BX, DI                // DI = j * step
	SHLQ $4, DI                 // DI = j * step * 16 (byte offset)

	MOVQ BX, BP                 // BP = step
	SHLQ $4, BP                 // BP = step * 16 (stride in bytes)

	// w1: gather twiddle[(j+0)*step], twiddle[(j+1)*step]
	VMOVUPD (R10)(DI*1), X4     // X4 = twiddle[j*step]
	LEAQ (DI)(BP*1), AX         // AX = (j+1)*step*16
	VMOVUPD (R10)(AX*1), X5     // X5 = twiddle[(j+1)*step]
	VINSERTF128 $1, X5, Y4, Y4  // Y4 = [w1[0], w1[1]]

	// w2: gather twiddle[2*j*step], twiddle[2*(j+1)*step]
	SHLQ $1, DI                 // DI = 2*j*step*16
	SHLQ $1, BP                 // BP = 2*step*16
	VMOVUPD (R10)(DI*1), X5     // X5 = twiddle[2*j*step]
	LEAQ (DI)(BP*1), AX         // AX = 2*(j+1)*step*16
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[2*(j+1)*step]
	VINSERTF128 $1, X6, Y5, Y5  // Y5 = [w2[0], w2[1]]

	// w3: gather twiddle[3*j*step], twiddle[3*(j+1)*step]
	// 3*j*step = j*step + 2*j*step, and stride = 3*step
	MOVQ DX, AX                 // AX = j
	IMULQ BX, AX                // AX = j*step
	LEAQ (AX)(AX*2), AX         // AX = 3*j*step
	SHLQ $4, AX                 // AX = 3*j*step*16
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[3*j*step]
	MOVQ BX, BP                 // BP = step
	LEAQ (BP)(BP*2), BP         // BP = 3*step
	SHLQ $4, BP                 // BP = 3*step*16
	LEAQ (AX)(BP*1), AX         // AX = 3*(j+1)*step*16
	VMOVUPD (R10)(AX*1), X7     // X7 = twiddle[3*(j+1)*step]
	VINSERTF128 $1, X7, Y6, Y6  // Y6 = [w3[0], w3[1]]

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

	// Compute -i*t3
	VPERMILPD $0x05, Y13, Y14   // Y14 = [t3.im, t3.re, ...] swapped
	MOVQ ·signbit64(SB), AX     // AX = signbit for float64
	VPXOR X15, X15, X15         // X15 = zero
	VMOVQ AX, X7                // X7 = [signbit, 0]
	VPUNPCKLQDQ X7, X15, X7     // X7 = [0, signbit]
	VINSERTI128 $1, X7, Y7, Y7
	VINSERTI128 $0, X7, Y7, Y7  // Y7 = [0, signbit, 0, signbit]
	VXORPD Y7, Y14, Y14         // Y14 = -i*t3

	// Compute +i*t3
	VPERMILPD $0x05, Y13, Y8    // Y8 = [t3.im, t3.re, ...] swapped
	VMOVQ AX, X9                // X9 = [signbit, 0]
	VPUNPCKLQDQ X15, X9, X9     // X9 = [signbit, 0]
	VINSERTI128 $1, X9, Y9, Y9
	VINSERTI128 $0, X9, Y9, Y9  // Y9 = [signbit, 0, signbit, 0]
	VXORPD Y9, Y8, Y8           // Y8 = +i*t3

	// Final butterfly outputs
	VADDPD Y12, Y10, Y0         // Y0 = y0 = t0 + t2
	VADDPD Y14, Y11, Y1         // Y1 = y1 = t1 + (-i*t3)
	VSUBPD Y12, Y10, Y2         // Y2 = y2 = t0 - t2
	VADDPD Y8, Y11, Y3          // Y3 = y3 = t1 + (+i*t3)

	// Store results
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16
	LEAQ (SI)(R9*1), DI         // DI = offset for quarter 1
	LEAQ (DI)(R9*1), AX         // AX = offset for quarter 2
	LEAQ (AX)(R9*1), BP         // BP = offset for quarter 3

	VMOVUPD Y0, (R8)(SI*1)      // store y0
	VMOVUPD Y1, (R8)(DI*1)      // store y1
	VMOVUPD Y2, (R8)(AX*1)      // store y2
	VMOVUPD Y3, (R8)(BP*1)      // store y3

	ADDQ $2, DX                 // j += 2
	JMP  fwd_r4m_c128_stepn_loop

fwd_r4m_c128_stage_scalar:
	// ===================================================================
	// XMM scalar fallback (1 complex128 per iteration)
	// ===================================================================
	CMPQ DX, R11                // compare j with quarter
	JGE  fwd_r4m_c128_stage_base_next // done with this base

	// twiddle indices: w1 = twiddle[j*step], w2 = twiddle[2*j*step], w3 = twiddle[3*j*step]
	MOVQ DX, AX                 // AX = j
	IMULQ BX, AX                // AX = j * step
	SHLQ $4, AX                 // AX = j * step * 16
	VMOVUPD (R10)(AX*1), X8     // X8 = w1 = twiddle[j*step]

	MOVQ DX, SI                 // SI = j
	IMULQ BX, SI                // SI = j * step
	SHLQ $1, SI                 // SI = 2*j*step
	SHLQ $4, SI                 // SI = 2*j*step * 16
	VMOVUPD (R10)(SI*1), X9     // X9 = w2 = twiddle[2*j*step]

	LEAQ (AX)(SI*1), SI         // SI = j*step*16 + 2*j*step*16 = 3*j*step*16
	VMOVUPD (R10)(SI*1), X10    // X10 = w3 = twiddle[3*j*step]

	// Compute element indices
	MOVQ R11, SI                // SI = quarter
	SHLQ $4, SI                 // SI = quarter * 16

	MOVQ CX, DI                 // DI = base
	ADDQ DX, DI                 // DI = base + j
	SHLQ $4, DI                 // DI = (base + j) * 16

	LEAQ (DI)(SI*1), AX         // AX = offset for quarter 1
	LEAQ (AX)(SI*1), BP         // BP = offset for quarter 2
	LEAQ (BP)(SI*1), R9         // R9 = offset for quarter 3

	// Load inputs
	VMOVUPD (R8)(DI*1), X0      // X0 = a0
	VMOVUPD (R8)(AX*1), X1      // X1 = a1
	VMOVUPD (R8)(BP*1), X2      // X2 = a2
	VMOVUPD (R8)(R9*1), X3      // X3 = a3

	// Complex multiply a1 * w1
	VUNPCKLPD X8, X8, X11       // X11 = w1.re broadcast
	VUNPCKHPD X8, X8, X12       // X12 = w1.im broadcast
	VPERMILPD $0x01, X1, X13    // X13 = a1 swapped
	VMULPD X12, X13, X13        // X13 = a1_swap * w1.im
	VFMADDSUB231PD X11, X1, X13 // X13 = a1 * w1
	VMOVAPD X13, X1             // X1 = a1 * w1

	// Complex multiply a2 * w2
	VUNPCKLPD X9, X9, X11       // X11 = w2.re broadcast
	VUNPCKHPD X9, X9, X12       // X12 = w2.im broadcast
	VPERMILPD $0x01, X2, X13    // X13 = a2 swapped
	VMULPD X12, X13, X13        // X13 = a2_swap * w2.im
	VFMADDSUB231PD X11, X2, X13 // X13 = a2 * w2
	VMOVAPD X13, X2             // X2 = a2 * w2

	// Complex multiply a3 * w3
	VUNPCKLPD X10, X10, X11     // X11 = w3.re broadcast
	VUNPCKHPD X10, X10, X12     // X12 = w3.im broadcast
	VPERMILPD $0x01, X3, X13    // X13 = a3 swapped
	VMULPD X12, X13, X13        // X13 = a3_swap * w3.im
	VFMADDSUB231PD X11, X3, X13 // X13 = a3 * w3
	VMOVAPD X13, X3             // X3 = a3 * w3

	// Radix-4 butterfly
	VADDPD X2, X0, X4           // X4 = t0 = a0 + a2
	VSUBPD X2, X0, X5           // X5 = t1 = a0 - a2
	VADDPD X3, X1, X6           // X6 = t2 = a1 + a3
	VSUBPD X3, X1, X7           // X7 = t3 = a1 - a3

	// Compute -i*t3
	VPERMILPD $0x01, X7, X14    // X14 = [t3.im, t3.re] swapped
	MOVQ ·signbit64(SB), AX     // AX = signbit
	VPXOR X15, X15, X15         // X15 = zero
	VMOVQ AX, X11               // X11 = [signbit, 0]
	VPUNPCKLQDQ X11, X15, X11   // X11 = [0, signbit]
	VXORPD X11, X14, X14        // X14 = -i*t3 = [t3.im, -t3.re]

	// Compute +i*t3
	VPERMILPD $0x01, X7, X8     // X8 = [t3.im, t3.re] swapped
	VMOVQ AX, X12               // X12 = [signbit, 0]
	VPUNPCKLQDQ X15, X12, X12   // X12 = [signbit, 0]
	VXORPD X12, X8, X8          // X8 = +i*t3 = [-t3.im, t3.re]

	// Final butterfly outputs
	VADDPD X6, X4, X0           // X0 = y0 = t0 + t2
	VADDPD X14, X5, X1          // X1 = y1 = t1 + (-i*t3)
	VSUBPD X6, X4, X2           // X2 = y2 = t0 - t2
	VADDPD X8, X5, X3           // X3 = y3 = t1 + (+i*t3)

	// Store results
	MOVQ R11, SI                // SI = quarter
	SHLQ $4, SI                 // SI = quarter * 16

	MOVQ CX, DI                 // DI = base
	ADDQ DX, DI                 // DI = base + j
	SHLQ $4, DI                 // DI = (base + j) * 16

	LEAQ (DI)(SI*1), AX         // AX = offset for quarter 1
	LEAQ (AX)(SI*1), BP         // BP = offset for quarter 2
	LEAQ (BP)(SI*1), R9         // R9 = offset for quarter 3

	VMOVUPD X0, (R8)(DI*1)      // store y0
	VMOVUPD X1, (R8)(AX*1)      // store y1
	VMOVUPD X2, (R8)(BP*1)      // store y2
	VMOVUPD X3, (R8)(R9*1)      // store y3

	INCQ DX                     // j++
	JMP  fwd_r4m_c128_stage_scalar

fwd_r4m_c128_stage_base_next:
	MOVQ R15, AX                // AX = size
	ADDQ AX, CX                 // base += size
	JMP  fwd_r4m_c128_stage_base

fwd_r4m_c128_stage_next:
	ADDQ $2, R12                // log2(size) += 2
	SHLQ $2, R15                // size *= 4
	DECQ R14                    // stages remaining--
	JMP  fwd_r4m_c128_stage_loop

fwd_r4m_c128_radix2_stage:
	// ===================================================================
	// Final radix-2 stage (size = n, step = 1)
	// ===================================================================
	MOVQ R13, R11               // R11 = n
	SHRQ $1, R11                // R11 = half = n/2
	XORQ CX, CX                 // CX = base = 0

fwd_r4m_c128_r2_base:
	CMPQ CX, R13                // compare base with n
	JGE  fwd_r4m_c128_copy_back // done with radix-2 stage

	XORQ DX, DX                 // DX = j = 0

fwd_r4m_c128_r2_inner:
	CMPQ DX, R11                // compare j with half
	JGE  fwd_r4m_c128_r2_next   // done with this block

	// Check for YMM path (2 elements)
	MOVQ R11, AX                // AX = half
	SUBQ DX, AX                 // AX = remaining
	CMPQ AX, $2                 // check if >= 2
	JL   fwd_r4m_c128_r2_scalar // fall back to scalar

	// YMM path: process 2 elements
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	MOVQ R11, DI                // DI = half
	SHLQ $4, DI                 // DI = half * 16

	VMOVUPD (R8)(SI*1), Y0      // Y0 = a[base+j : base+j+2]
	LEAQ (SI)(DI*1), BP         // BP = offset for second half
	VMOVUPD (R8)(BP*1), Y1      // Y1 = b[base+j+half : ...]

	// Load twiddle factors (contiguous, step=1)
	MOVQ DX, AX                 // AX = j
	SHLQ $4, AX                 // AX = j * 16
	VMOVUPD (R10)(AX*1), Y2     // Y2 = twiddle[j:j+2]

	// Complex multiply b * w
	VUNPCKLPD Y2, Y2, Y3        // Y3 = w.re broadcast
	VUNPCKHPD Y2, Y2, Y4        // Y4 = w.im broadcast
	VPERMILPD $0x05, Y1, Y5     // Y5 = b swapped
	VMULPD Y4, Y5, Y5           // Y5 = b_swap * w.im
	VFMADDSUB231PD Y3, Y1, Y5   // Y5 = t = b * w

	// Radix-2 butterfly
	VADDPD Y5, Y0, Y6           // Y6 = a + t
	VSUBPD Y5, Y0, Y7           // Y7 = a - t

	// Store results
	VMOVUPD Y6, (R8)(SI*1)      // store a + t
	VMOVUPD Y7, (R8)(BP*1)      // store a - t

	ADDQ $2, DX                 // j += 2
	JMP  fwd_r4m_c128_r2_inner

fwd_r4m_c128_r2_scalar:
	// XMM scalar path (1 element)
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	MOVQ R11, DI                // DI = half
	SHLQ $4, DI                 // DI = half * 16

	VMOVUPD (R8)(SI*1), X0      // X0 = a
	LEAQ (SI)(DI*1), BP         // BP = offset for second half
	VMOVUPD (R8)(BP*1), X1      // X1 = b

	// Load twiddle factor
	MOVQ DX, AX                 // AX = j
	SHLQ $4, AX                 // AX = j * 16
	VMOVUPD (R10)(AX*1), X2     // X2 = twiddle[j]

	// Complex multiply b * w
	VUNPCKLPD X2, X2, X3        // X3 = w.re broadcast
	VUNPCKHPD X2, X2, X4        // X4 = w.im broadcast
	VPERMILPD $0x01, X1, X5     // X5 = b swapped
	VMULPD X4, X5, X5           // X5 = b_swap * w.im
	VFMADDSUB231PD X3, X1, X5   // X5 = t = b * w

	// Radix-2 butterfly
	VADDPD X5, X0, X6           // X6 = a + t
	VSUBPD X5, X0, X7           // X7 = a - t

	// Store results
	VMOVUPD X6, (R8)(SI*1)      // store a + t
	VMOVUPD X7, (R8)(BP*1)      // store a - t

	INCQ DX                     // j++
	JMP  fwd_r4m_c128_r2_inner

fwd_r4m_c128_r2_next:
	ADDQ R13, CX                // base += n (only one block for radix-2)
	JMP  fwd_r4m_c128_r2_base

fwd_r4m_c128_copy_back:
	// ===================================================================
	// Copy result back to dst if we used scratch
	// ===================================================================
	MOVQ dst+0(FP), AX          // AX = original dst
	CMPQ R8, AX                 // check if working buffer == dst
	JE   fwd_r4m_c128_return_true // no copy needed

	XORQ CX, CX                 // CX = i = 0

fwd_r4m_c128_copy_loop:
	CMPQ CX, R13                // compare i with n
	JGE  fwd_r4m_c128_return_true // done copying

	MOVQ CX, SI                 // SI = i
	SHLQ $4, SI                 // SI = i * 16
	VMOVUPD (R8)(SI*1), X0      // load from working buffer
	VMOVUPD X0, (AX)(SI*1)      // store to dst
	INCQ CX                     // i++
	JMP  fwd_r4m_c128_copy_loop

fwd_r4m_c128_return_true:
	VZEROUPPER                  // clear upper YMM to avoid AVX-SSE penalty
	MOVB $1, ret+120(FP)        // return true
	RET

fwd_r4m_c128_return_false:
	MOVB $0, ret+120(FP)        // return false
	RET

// ===========================================================================
// InverseAVX2Complex128Radix4MixedAsm - Inverse FFT for complex128 using
// radix-4 stages followed by a final radix-2 stage (for odd log2 sizes).
// Uses conjugate twiddle multiplication and swapped +i/-i butterfly.
// ===========================================================================
// func InverseAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch []complex128, bitrev []int) bool
// Frame size: 0, Args size: 121 bytes
// ===========================================================================
TEXT ·InverseAVX2Complex128Radix4MixedAsm(SB), NOSPLIT, $0-121
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
	JZ    inv_r4m_c128_return_true // return true for empty input

	// ===================================================================
	// Validate all slice lengths are >= n
	// ===================================================================
	MOVQ dst+8(FP), AX          // AX = dst.len
	CMPQ AX, R13                // compare dst.len with n
	JL   inv_r4m_c128_return_false // fail if dst.len < n

	MOVQ twiddle+56(FP), AX     // AX = twiddle.len
	CMPQ AX, R13                // compare twiddle.len with n
	JL   inv_r4m_c128_return_false // fail if twiddle.len < n

	MOVQ scratch+80(FP), AX     // AX = scratch.len
	CMPQ AX, R13                // compare scratch.len with n
	JL   inv_r4m_c128_return_false // fail if scratch.len < n

	// ===================================================================
	// Trivial case: n=1, just copy single element
	// ===================================================================
	CMPQ R13, $1                // check if n == 1
	JNE  inv_r4m_c128_check_power_of_2 // continue if n != 1
	VMOVUPD (R9), X0            // load 16 bytes (1 complex128) from src
	VMOVUPD X0, (R8)            // store to dst
	JMP  inv_r4m_c128_return_true // return true

inv_r4m_c128_check_power_of_2:
	// ===================================================================
	// Verify n is power of 2: (n & (n-1)) == 0
	// ===================================================================
	MOVQ R13, AX                // AX = n
	LEAQ -1(AX), BX             // BX = n - 1
	TESTQ AX, BX                // test n & (n-1)
	JNZ  inv_r4m_c128_return_false // fail if not power of 2

	// ===================================================================
	// Require odd log2 and minimum size of 32
	// ===================================================================
	MOVQ R13, AX                // AX = n
	BSRQ AX, R12                // R12 = log2(n) = bit scan reverse
	TESTQ $1, R12               // check if log2(n) is odd
	JZ   inv_r4m_c128_return_false // fail if even (use radix-4 even path)

	CMPQ R13, $32               // check minimum size
	JL   inv_r4m_c128_return_false // fail if n < 32

	// ===================================================================
	// Number of radix-4 stages: k = (log2(n)-1)/2
	// ===================================================================
	MOVQ R12, R14               // R14 = log2(n)
	SUBQ $1, R14                // R14 = log2(n) - 1
	SHRQ $1, R14                // R14 = k (radix-4 stage count)

	// ===================================================================
	// Select working buffer (use scratch for in-place transforms)
	// ===================================================================
	CMPQ R8, R9                 // check if dst == src
	JNE  inv_r4m_c128_use_dst   // use dst if different
	MOVQ R11, R8                // in-place: use scratch as working buffer

inv_r4m_c128_use_dst:
	// ===================================================================
	// Bit-reversal permutation (mixed radix: base-4 digits, then top bit)
	// ===================================================================
	XORQ CX, CX                 // CX = i = 0

inv_r4m_c128_bitrev_loop:
	CMPQ CX, R13                // compare i with n
	JGE  inv_r4m_c128_stage_init // done with bit-reversal if i >= n

	MOVQ CX, DX                 // DX = i
	XORQ BX, BX                 // BX = reversed index = 0
	MOVQ R14, SI                // SI = k (radix-4 stages)

inv_r4m_c128_bitrev_inner:
	CMPQ SI, $0                 // check if all base-4 digits processed
	JE   inv_r4m_c128_bitrev_store // done with base-4 reversal
	MOVQ DX, AX                 // AX = remaining value
	ANDQ $3, AX                 // AX = lowest 2 bits (base-4 digit)
	SHLQ $2, BX                 // shift reversed index left by 2
	ORQ  AX, BX                 // add new digit to reversed index
	SHRQ $2, DX                 // remove processed digit from value
	DECQ SI                     // decrement stage counter
	JMP  inv_r4m_c128_bitrev_inner // process next digit

inv_r4m_c128_bitrev_store:
	SHLQ $1, BX                 // shift reversed index left by 1
	ORQ  DX, BX                 // append remaining top bit

	MOVQ BX, SI                 // SI = source index (bit-reversed)
	SHLQ $4, SI                 // SI = source byte offset (index * 16)
	VMOVUPD (R9)(SI*1), X0      // load src[bitrev[i]] (16 bytes)
	MOVQ CX, DI                 // DI = destination index
	SHLQ $4, DI                 // DI = dest byte offset (index * 16)
	VMOVUPD X0, (R8)(DI*1)      // store to work[i]
	INCQ CX                     // i++
	JMP  inv_r4m_c128_bitrev_loop // next element

inv_r4m_c128_stage_init:
	// ===================================================================
	// Radix-4 stages (size = 4, 16, 64, ...)
	// ===================================================================
	MOVQ $2, R12                // R12 = log2(size), starting at 2 (size=4)
	MOVQ $4, R15                // R15 = size = 4

inv_r4m_c128_stage_loop:
	CMPQ R14, $0                // check if radix-4 stages remaining
	JE   inv_r4m_c128_radix2_stage // no more radix-4, do final radix-2

	MOVQ R15, R11               // R11 = size
	SHRQ $2, R11                // R11 = quarter = size/4

	// step = n >> log2(size)
	MOVQ R13, BX                // BX = n
	MOVQ R12, CX                // CX = log2(size)
	SHRQ CL, BX                 // BX = step = n >> log2(size)

	XORQ CX, CX                 // CX = base = 0

inv_r4m_c128_stage_base:
	CMPQ CX, R13                // compare base with n
	JGE  inv_r4m_c128_stage_next // done with this stage if base >= n

	XORQ DX, DX                 // DX = j = 0

	// ===================================================================
	// Check for fast path: contiguous twiddles (step == 1)
	// ===================================================================
	CMPQ BX, $1                 // check if step == 1
	JNE  inv_r4m_c128_stepn_prep // use strided path if step != 1
	CMPQ R11, $2                // check if quarter >= 2 for YMM path
	JL   inv_r4m_c128_stage_scalar // fall back to scalar if quarter < 2

	// ===================================================================
	// YMM fast path: process 2 elements per iteration (step=1)
	// Uses conjugate complex multiplication (VFMSUBADD instead of VFMADDSUB)
	// ===================================================================
	MOVQ R11, R9                // R9 = quarter
	SHLQ $4, R9                 // R9 = quarter_bytes = quarter * 16

inv_r4m_c128_step1_loop:
	MOVQ R11, AX                // AX = quarter
	SUBQ DX, AX                 // AX = remaining elements in quarter
	CMPQ AX, $2                 // check if at least 2 elements remain
	JL   inv_r4m_c128_stage_scalar // fall back to scalar for remainder

	// Compute element offsets
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	LEAQ (SI)(R9*1), DI         // DI = offset for quarter 1
	LEAQ (DI)(R9*1), AX         // AX = offset for quarter 2
	LEAQ (AX)(R9*1), BP         // BP = offset for quarter 3

	// Load 2 complex128 from each of the 4 quarters
	VMOVUPD (R8)(SI*1), Y0      // Y0 = work[base+j : base+j+2]
	VMOVUPD (R8)(DI*1), Y1      // Y1 = work[base+j+quarter : ...]
	VMOVUPD (R8)(AX*1), Y2      // Y2 = work[base+j+2*quarter : ...]
	VMOVUPD (R8)(BP*1), Y3      // Y3 = work[base+j+3*quarter : ...]

	// Load twiddle factors (step=1, contiguous)
	MOVQ DX, AX                 // AX = j
	SHLQ $4, AX                 // AX = j * 16 (byte offset)
	VMOVUPD (R10)(AX*1), Y4     // Y4 = w1[0:2] = twiddle[j:j+2]

	// w2 = twiddle[2*j], twiddle[2*(j+1)]
	MOVQ DX, AX                 // AX = j
	SHLQ $1, AX                 // AX = 2*j
	SHLQ $4, AX                 // AX = 2*j * 16
	VMOVUPD (R10)(AX*1), X5     // X5 = twiddle[2*j]
	ADDQ $32, AX                // skip 2 complex128
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[2*(j+1)]
	VINSERTF128 $1, X6, Y5, Y5  // Y5 = [w2[0], w2[1]]

	// w3 = twiddle[3*j], twiddle[3*(j+1)]
	LEAQ (DX)(DX*2), AX         // AX = 3*j
	SHLQ $4, AX                 // AX = 3*j * 16
	VMOVUPD (R10)(AX*1), X6     // X6 = twiddle[3*j]
	ADDQ $48, AX                // skip 3 complex128
	VMOVUPD (R10)(AX*1), X7     // X7 = twiddle[3*(j+1)]
	VINSERTF128 $1, X7, Y6, Y6  // Y6 = [w3[0], w3[1]]

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
	// Radix-4 butterfly (INVERSE: +i/-i swapped from forward)
	// ---------------------------------------------------------------
	VADDPD Y2, Y0, Y10          // Y10 = t0 = a0 + a2
	VSUBPD Y2, Y0, Y11          // Y11 = t1 = a0 - a2
	VADDPD Y3, Y1, Y12          // Y12 = t2 = a1 + a3
	VSUBPD Y3, Y1, Y13          // Y13 = t3 = a1 - a3

	// Compute +i*t3 for y1 (INVERSE uses +i here)
	VPERMILPD $0x05, Y13, Y14   // Y14 = [t3.im, t3.re, ...] swapped
	MOVQ ·signbit64(SB), AX     // AX = signbit for float64
	VPXOR X15, X15, X15         // X15 = zero
	VMOVQ AX, X7                // X7 = [signbit, 0]
	VPUNPCKLQDQ X15, X7, X7     // X7 = [signbit, 0] - mask for +i
	VINSERTI128 $1, X7, Y7, Y7
	VINSERTI128 $0, X7, Y7, Y7  // Y7 = [signbit, 0, signbit, 0]
	VXORPD Y7, Y14, Y14         // Y14 = +i*t3 = [-t3.im, t3.re, ...]

	// Compute -i*t3 for y3 (INVERSE uses -i here)
	VPERMILPD $0x05, Y13, Y8    // Y8 = [t3.im, t3.re, ...] swapped
	VMOVQ AX, X9                // X9 = [signbit, 0]
	VPUNPCKLQDQ X9, X15, X9     // X9 = [0, signbit] - mask for -i
	VINSERTI128 $1, X9, Y9, Y9
	VINSERTI128 $0, X9, Y9, Y9  // Y9 = [0, signbit, 0, signbit]
	VXORPD Y9, Y8, Y8           // Y8 = -i*t3 = [t3.im, -t3.re, ...]

	// Final butterfly outputs (INVERSE formulas)
	// y0 = t0 + t2
	// y1 = t1 + i*t3  (inverse: +i)
	// y2 = t0 - t2
	// y3 = t1 - i*t3 = t1 + (-i*t3)  (inverse: -i)
	VADDPD Y12, Y10, Y0         // Y0 = y0 = t0 + t2
	VADDPD Y14, Y11, Y1         // Y1 = y1 = t1 + (+i*t3)
	VSUBPD Y12, Y10, Y2         // Y2 = y2 = t0 - t2
	VADDPD Y8, Y11, Y3          // Y3 = y3 = t1 + (-i*t3)

	// Store results
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16
	LEAQ (SI)(R9*1), DI         // DI = offset for quarter 1
	LEAQ (DI)(R9*1), AX         // AX = offset for quarter 2
	LEAQ (AX)(R9*1), BP         // BP = offset for quarter 3

	VMOVUPD Y0, (R8)(SI*1)      // store y0
	VMOVUPD Y1, (R8)(DI*1)      // store y1
	VMOVUPD Y2, (R8)(AX*1)      // store y2
	VMOVUPD Y3, (R8)(BP*1)      // store y3

	ADDQ $2, DX                 // j += 2
	JMP  inv_r4m_c128_step1_loop

inv_r4m_c128_stepn_prep:
	// ===================================================================
	// YMM path for strided twiddles (step > 1)
	// ===================================================================
	CMPQ R11, $2                // check if quarter >= 2
	JL   inv_r4m_c128_stage_scalar // fall back to scalar

	MOVQ R11, R9                // R9 = quarter
	SHLQ $4, R9                 // R9 = quarter_bytes = quarter * 16

inv_r4m_c128_stepn_loop:
	MOVQ R11, AX                // AX = quarter
	SUBQ DX, AX                 // AX = remaining elements
	CMPQ AX, $2                 // check if at least 2 elements remain
	JL   inv_r4m_c128_stage_scalar // fall back to scalar

	// Compute element offsets
	MOVQ CX, SI                 // SI = base
	ADDQ DX, SI                 // SI = base + j
	SHLQ $4, SI                 // SI = (base + j) * 16

	LEAQ (SI)(R9*1), DI         // DI = offset for quarter 1
	LEAQ (DI)(R9*1), AX         // AX = offset for quarter 2
	LEAQ (AX)(R9*1), BP         // BP = offset for quarter 3

	// Load data from 4 quarters
	VMOVUPD (R8)(SI*1), Y0
	VMOVUPD (R8)(DI*1), Y1
	VMOVUPD (R8)(AX*1), Y2
	VMOVUPD (R8)(BP*1), Y3

	// Load twiddle factors with stride
	MOVQ DX, DI                 // DI = j
	IMULQ BX, DI                // DI = j * step
	SHLQ $4, DI                 // DI = j * step * 16

	MOVQ BX, BP                 // BP = step
	SHLQ $4, BP                 // BP = step * 16

	// w1: gather twiddles
	VMOVUPD (R10)(DI*1), X4
	LEAQ (DI)(BP*1), AX
	VMOVUPD (R10)(AX*1), X5
	VINSERTF128 $1, X5, Y4, Y4

	// w2: gather twiddles
	SHLQ $1, DI                 // DI = 2*j*step*16
	SHLQ $1, BP                 // BP = 2*step*16
	VMOVUPD (R10)(DI*1), X5
	LEAQ (DI)(BP*1), AX
	VMOVUPD (R10)(AX*1), X6
	VINSERTF128 $1, X6, Y5, Y5

	// w3: gather twiddles
	MOVQ DX, AX
	IMULQ BX, AX
	LEAQ (AX)(AX*2), AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X6
	MOVQ BX, BP
	LEAQ (BP)(BP*2), BP
	SHLQ $4, BP
	LEAQ (AX)(BP*1), AX
	VMOVUPD (R10)(AX*1), X7
	VINSERTF128 $1, X7, Y6, Y6

	// Conjugate complex multiply a1 * conj(w1)
	VUNPCKLPD Y4, Y4, Y7
	VUNPCKHPD Y4, Y4, Y8
	VPERMILPD $0x05, Y1, Y9
	VMULPD Y8, Y9, Y9
	VFMSUBADD231PD Y7, Y1, Y9   // conjugate
	VMOVAPD Y9, Y1

	// Conjugate complex multiply a2 * conj(w2)
	VUNPCKLPD Y5, Y5, Y7
	VUNPCKHPD Y5, Y5, Y8
	VPERMILPD $0x05, Y2, Y9
	VMULPD Y8, Y9, Y9
	VFMSUBADD231PD Y7, Y2, Y9   // conjugate
	VMOVAPD Y9, Y2

	// Conjugate complex multiply a3 * conj(w3)
	VUNPCKLPD Y6, Y6, Y7
	VUNPCKHPD Y6, Y6, Y8
	VPERMILPD $0x05, Y3, Y9
	VMULPD Y8, Y9, Y9
	VFMSUBADD231PD Y7, Y3, Y9   // conjugate
	VMOVAPD Y9, Y3

	// Radix-4 butterfly (INVERSE)
	VADDPD Y2, Y0, Y10
	VSUBPD Y2, Y0, Y11
	VADDPD Y3, Y1, Y12
	VSUBPD Y3, Y1, Y13

	// Compute +i*t3 for y1
	VPERMILPD $0x05, Y13, Y14
	MOVQ ·signbit64(SB), AX
	VPXOR X15, X15, X15
	VMOVQ AX, X7
	VPUNPCKLQDQ X15, X7, X7     // [signbit, 0] for +i
	VINSERTI128 $1, X7, Y7, Y7
	VINSERTI128 $0, X7, Y7, Y7
	VXORPD Y7, Y14, Y14         // +i*t3

	// Compute -i*t3 for y3
	VPERMILPD $0x05, Y13, Y8
	VMOVQ AX, X9
	VPUNPCKLQDQ X9, X15, X9     // [0, signbit] for -i
	VINSERTI128 $1, X9, Y9, Y9
	VINSERTI128 $0, X9, Y9, Y9
	VXORPD Y9, Y8, Y8           // -i*t3

	// Final butterfly (INVERSE)
	VADDPD Y12, Y10, Y0
	VADDPD Y14, Y11, Y1         // t1 + (+i*t3)
	VSUBPD Y12, Y10, Y2
	VADDPD Y8, Y11, Y3          // t1 + (-i*t3)

	// Store results
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI
	LEAQ (SI)(R9*1), DI
	LEAQ (DI)(R9*1), AX
	LEAQ (AX)(R9*1), BP

	VMOVUPD Y0, (R8)(SI*1)
	VMOVUPD Y1, (R8)(DI*1)
	VMOVUPD Y2, (R8)(AX*1)
	VMOVUPD Y3, (R8)(BP*1)

	ADDQ $2, DX
	JMP  inv_r4m_c128_stepn_loop

inv_r4m_c128_stage_scalar:
	// ===================================================================
	// XMM scalar fallback (1 complex128 per iteration)
	// ===================================================================
	CMPQ DX, R11                // compare j with quarter
	JGE  inv_r4m_c128_stage_base_next // done with this base

	// twiddle indices
	MOVQ DX, AX
	IMULQ BX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X8     // X8 = w1

	MOVQ DX, SI
	IMULQ BX, SI
	SHLQ $1, SI
	SHLQ $4, SI
	VMOVUPD (R10)(SI*1), X9     // X9 = w2

	LEAQ (AX)(SI*1), SI
	VMOVUPD (R10)(SI*1), X10    // X10 = w3

	// Compute element indices
	MOVQ R11, SI
	SHLQ $4, SI

	MOVQ CX, DI
	ADDQ DX, DI
	SHLQ $4, DI

	LEAQ (DI)(SI*1), AX
	LEAQ (AX)(SI*1), BP
	LEAQ (BP)(SI*1), R9

	// Load inputs
	VMOVUPD (R8)(DI*1), X0
	VMOVUPD (R8)(AX*1), X1
	VMOVUPD (R8)(BP*1), X2
	VMOVUPD (R8)(R9*1), X3

	// Conjugate complex multiply a1 * conj(w1)
	VUNPCKLPD X8, X8, X11
	VUNPCKHPD X8, X8, X12
	VPERMILPD $0x01, X1, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X1, X13 // conjugate
	VMOVAPD X13, X1

	// Conjugate complex multiply a2 * conj(w2)
	VUNPCKLPD X9, X9, X11
	VUNPCKHPD X9, X9, X12
	VPERMILPD $0x01, X2, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X2, X13 // conjugate
	VMOVAPD X13, X2

	// Conjugate complex multiply a3 * conj(w3)
	VUNPCKLPD X10, X10, X11
	VUNPCKHPD X10, X10, X12
	VPERMILPD $0x01, X3, X13
	VMULPD X12, X13, X13
	VFMSUBADD231PD X11, X3, X13 // conjugate
	VMOVAPD X13, X3

	// Radix-4 butterfly
	VADDPD X2, X0, X4
	VSUBPD X2, X0, X5
	VADDPD X3, X1, X6
	VSUBPD X3, X1, X7

	// Compute +i*t3
	VPERMILPD $0x01, X7, X14
	MOVQ ·signbit64(SB), AX
	VPXOR X15, X15, X15
	VMOVQ AX, X11
	VPUNPCKLQDQ X15, X11, X11   // [signbit, 0]
	VXORPD X11, X14, X14        // +i*t3

	// Compute -i*t3
	VPERMILPD $0x01, X7, X8
	VMOVQ AX, X12
	VPUNPCKLQDQ X12, X15, X12   // [0, signbit]
	VXORPD X12, X8, X8          // -i*t3

	// Final butterfly (INVERSE)
	VADDPD X6, X4, X0
	VADDPD X14, X5, X1          // t1 + (+i*t3)
	VSUBPD X6, X4, X2
	VADDPD X8, X5, X3           // t1 + (-i*t3)

	// Store results
	MOVQ R11, SI
	SHLQ $4, SI

	MOVQ CX, DI
	ADDQ DX, DI
	SHLQ $4, DI

	LEAQ (DI)(SI*1), AX
	LEAQ (AX)(SI*1), BP
	LEAQ (BP)(SI*1), R9

	VMOVUPD X0, (R8)(DI*1)
	VMOVUPD X1, (R8)(AX*1)
	VMOVUPD X2, (R8)(BP*1)
	VMOVUPD X3, (R8)(R9*1)

	INCQ DX
	JMP  inv_r4m_c128_stage_scalar

inv_r4m_c128_stage_base_next:
	MOVQ R15, AX
	ADDQ AX, CX
	JMP  inv_r4m_c128_stage_base

inv_r4m_c128_stage_next:
	ADDQ $2, R12
	SHLQ $2, R15
	DECQ R14
	JMP  inv_r4m_c128_stage_loop

inv_r4m_c128_radix2_stage:
	// ===================================================================
	// Final radix-2 stage (size = n, step = 1)
	// Uses conjugate twiddle multiplication
	// ===================================================================
	MOVQ R13, R11
	SHRQ $1, R11                // R11 = half = n/2
	XORQ CX, CX                 // CX = base = 0

inv_r4m_c128_r2_base:
	CMPQ CX, R13
	JGE  inv_r4m_c128_copy_back

	XORQ DX, DX                 // DX = j = 0

inv_r4m_c128_r2_inner:
	CMPQ DX, R11
	JGE  inv_r4m_c128_r2_next

	// Check for YMM path (2 elements)
	MOVQ R11, AX
	SUBQ DX, AX
	CMPQ AX, $2
	JL   inv_r4m_c128_r2_scalar

	// YMM path: process 2 elements
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI

	MOVQ R11, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), Y0      // Y0 = a
	LEAQ (SI)(DI*1), BP
	VMOVUPD (R8)(BP*1), Y1      // Y1 = b

	// Load twiddle factors
	MOVQ DX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), Y2     // Y2 = twiddle[j:j+2]

	// Conjugate complex multiply b * conj(w)
	VUNPCKLPD Y2, Y2, Y3
	VUNPCKHPD Y2, Y2, Y4
	VPERMILPD $0x05, Y1, Y5
	VMULPD Y4, Y5, Y5
	VFMSUBADD231PD Y3, Y1, Y5   // conjugate: t = b * conj(w)

	// Radix-2 butterfly
	VADDPD Y5, Y0, Y6           // Y6 = a + t
	VSUBPD Y5, Y0, Y7           // Y7 = a - t

	// Store results
	VMOVUPD Y6, (R8)(SI*1)
	VMOVUPD Y7, (R8)(BP*1)

	ADDQ $2, DX
	JMP  inv_r4m_c128_r2_inner

inv_r4m_c128_r2_scalar:
	// XMM scalar path (1 element)
	MOVQ CX, SI
	ADDQ DX, SI
	SHLQ $4, SI

	MOVQ R11, DI
	SHLQ $4, DI

	VMOVUPD (R8)(SI*1), X0      // X0 = a
	LEAQ (SI)(DI*1), BP
	VMOVUPD (R8)(BP*1), X1      // X1 = b

	// Load twiddle factor
	MOVQ DX, AX
	SHLQ $4, AX
	VMOVUPD (R10)(AX*1), X2     // X2 = twiddle[j]

	// Conjugate complex multiply b * conj(w)
	VUNPCKLPD X2, X2, X3
	VUNPCKHPD X2, X2, X4
	VPERMILPD $0x01, X1, X5
	VMULPD X4, X5, X5
	VFMSUBADD231PD X3, X1, X5   // conjugate: t = b * conj(w)

	// Radix-2 butterfly
	VADDPD X5, X0, X6           // X6 = a + t
	VSUBPD X5, X0, X7           // X7 = a - t

	// Store results
	VMOVUPD X6, (R8)(SI*1)
	VMOVUPD X7, (R8)(BP*1)

	INCQ DX
	JMP  inv_r4m_c128_r2_inner

inv_r4m_c128_r2_next:
	ADDQ R13, CX
	JMP  inv_r4m_c128_r2_base

inv_r4m_c128_copy_back:
	// ===================================================================
	// Copy result back to dst if we used scratch
	// ===================================================================
	MOVQ dst+0(FP), AX
	CMPQ R8, AX
	JE   inv_r4m_c128_return_true

	XORQ CX, CX

inv_r4m_c128_copy_loop:
	CMPQ CX, R13
	JGE  inv_r4m_c128_return_true

	MOVQ CX, SI
	SHLQ $4, SI
	VMOVUPD (R8)(SI*1), X0
	VMOVUPD X0, (AX)(SI*1)
	INCQ CX
	JMP  inv_r4m_c128_copy_loop

inv_r4m_c128_return_true:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_r4m_c128_return_false:
	MOVB $0, ret+120(FP)
	RET
