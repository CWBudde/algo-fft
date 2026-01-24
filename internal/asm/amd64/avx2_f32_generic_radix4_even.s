//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2/FMA-optimized FFT Assembly for AMD64 - complex64 (float32)
// Radix-4 generic DIT path for power-of-4 sizes.
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// forwardAVX2Complex64Radix4Asm - Forward FFT for complex64 using radix-4 DIT
// ===========================================================================
TEXT ·ForwardAVX2Complex64Radix4Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    fwd_r4_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   fwd_r4_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   fwd_r4_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   fwd_r4_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  fwd_r4_check_power_of_2
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  fwd_r4_return_true

fwd_r4_check_power_of_2:
	// Verify n is power of 2: (n & (n-1)) == 0
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  fwd_r4_return_false

	// Require even log2 (power-of-4) and minimum size
	MOVQ R13, AX
	BSRQ AX, R14
	TESTQ $1, R14
	JNZ  fwd_r4_return_false

	CMPQ R13, $64
	JL   fwd_r4_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  fwd_r4_use_dst
	MOVQ R11, R8

fwd_r4_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation (base-4)
	// -----------------------------------------------------------------------
	XORQ CX, CX

fwd_r4_bitrev_loop:
	CMPQ CX, R13              // check if index CX >= n (done?)
	JGE  fwd_r4_stage_init    // if done, jump to stage init

	MOVQ CX, DX               // DX = current index to bit-reverse
	XORQ BX, BX               // BX = 0 (accumulator for reversed index)
	MOVQ R14, SI              // SI = log2(n) (loop counter in bits)

fwd_r4_bitrev_inner:
	CMPQ SI, $0               // check if all bits processed
	JE   fwd_r4_bitrev_store  // if done, store result
	MOVQ DX, AX               // AX = copy of current index
	ANDQ $3, AX               // AX = lowest 2 bits (base-4 digit)
	SHLQ $2, BX               // BX <<= 2 (shift reversed index left by 2 bits)
	ORQ  AX, BX               // BX |= AX (append base-4 digit to reversed index)
	SHRQ $2, DX               // DX >>= 2 (shift right by 2 bits for next digit)
	SUBQ $2, SI               // SI -= 2 (processed 2 bits)
	JMP  fwd_r4_bitrev_inner  // continue inner loop

fwd_r4_bitrev_store:
	MOVQ (R9)(BX*8), AX       // load complex value from src[reversed_index]
	MOVQ AX, (R8)(CX*8)       // store to dst[sequential_index]
	INCQ CX                   // increment sequential index
	JMP  fwd_r4_bitrev_loop   // continue outer loop

fwd_r4_stage_init:
	// -----------------------------------------------------------------------
	// Radix-4 stages (size = 4, 16, 64, ...)
	// -----------------------------------------------------------------------
	MOVQ $2, R12              // R12 = 2 (log2(size) starting at 2, since first stage is size=4)
	MOVQ $4, R14              // R14 = 4 (initial size for first radix-4 stage)

fwd_r4_stage_loop:
	CMPQ R14, R13             // compare current size with n
	JG   fwd_r4_copy_back     // if size > n, all stages done, copy back if needed

	MOVQ R14, R15             // R15 = current stage size
	SHRQ $2, R15              // R15 = size/4 (quarter, number of butterflies per group)

	// step = n >> log2(size)
	MOVQ R13, BX              // BX = n
	MOVQ R12, CX              // CX = log2(size)
	SHRQ CL, BX               // BX = n >> log2(size) (twiddle stride)

	XORQ CX, CX               // CX = 0 (base index for current group)

fwd_r4_stage_base:
	CMPQ CX, R13              // check if base >= n (all groups processed?)
	JGE  fwd_r4_stage_next    // if done, advance to next stage

	XORQ DX, DX               // DX = 0 (j, butterfly index within group)

	// Fast path for contiguous twiddles on the final stage (step == 1).
	CMPQ BX, $1               // check if step == 1 (final stage)
	JNE  fwd_r4_stepn_prep    // if step != 1, use strided twiddle path
	CMPQ R15, $4              // check if quarter >= 4 (enough for vectorization)
	JL   fwd_r4_stage_scalar  // if quarter < 4, use scalar path

	MOVQ R15, R11             // R11 = quarter
	SHLQ $3, R11              // R11 = quarter * 8 (quarter_bytes, offset between a0/a1/a2/a3)

fwd_r4_step1_loop:
	MOVQ R15, AX              // AX = quarter
	SUBQ DX, AX               // AX = quarter - j (remaining butterflies)
	CMPQ AX, $4               // check if remaining >= 4 (can process 4 complex values)
	JL   fwd_r4_stage_scalar  // if remaining < 4, switch to scalar

	// base offset
	MOVQ CX, SI               // SI = base (group start index)
	ADDQ DX, SI               // SI = base + j (element index)
	SHLQ $3, SI               // SI = (base + j) * 8 (byte offset for complex64)

	LEAQ (SI)(R11*1), DI      // DI = SI + quarter_bytes (offset to a1)
	LEAQ (DI)(R11*1), AX      // AX = DI + quarter_bytes (offset to a2)
	LEAQ (AX)(R11*1), BP      // BP = AX + quarter_bytes (offset to a3)

	VMOVUPS (R8)(SI*1), Y0    // Y0 = load 4 complex from a0[j:j+4]
	VMOVUPS (R8)(DI*1), Y1    // Y1 = load 4 complex from a1[j:j+4]
	VMOVUPS (R8)(AX*1), Y2    // Y2 = load 4 complex from a2[j:j+4]
	VMOVUPS (R8)(BP*1), Y3    // Y3 = load 4 complex from a3[j:j+4]

	// w1 = twiddle[j : j+4] (contiguous)
	MOVQ DX, AX               // AX = j
	SHLQ $3, AX               // AX = j * 8 (byte offset)
	VMOVUPS (R10)(AX*1), Y4   // Y4 = load 4 contiguous twiddles w1[j:j+4]

	// w2 = twiddle[2*j, 2*j+2, 2*j+4, 2*j+6]
	MOVQ DX, AX               // AX = j
	SHLQ $1, AX               // AX = 2*j
	SHLQ $3, AX               // AX = 2*j * 8 (byte offset)
	VMOVSD (R10)(AX*1), X5    // X5 = twiddle[2*j] (one complex64)
	LEAQ 16(AX), DI           // DI = offset to twiddle[2*j+2] (+16 bytes = +2 complex64)
	VMOVSD (R10)(DI*1), X6    // X6 = twiddle[2*j+2]
	LEAQ 16(DI), DI           // DI = offset to twiddle[2*j+4]
	VMOVSD (R10)(DI*1), X7    // X7 = twiddle[2*j+4]
	LEAQ 16(DI), DI           // DI = offset to twiddle[2*j+6]
	VMOVSD (R10)(DI*1), X8    // X8 = twiddle[2*j+6]
	VPUNPCKLQDQ X6, X5, X5    // X5 = [twiddle[2*j], twiddle[2*j+2]]
	VPUNPCKLQDQ X8, X7, X7    // X7 = [twiddle[2*j+4], twiddle[2*j+6]]
	VINSERTF128 $1, X7, Y5, Y5 // Y5 = combine into YMM (4 strided w2 twiddles)

	// w3 = twiddle[3*j, 3*j+3, 3*j+6, 3*j+9]
	LEAQ (DX)(DX*2), AX       // AX = 3*j (using LEA for 3x multiplication)
	SHLQ $3, AX               // AX = 3*j * 8 (byte offset)
	VMOVSD (R10)(AX*1), X6    // X6 = twiddle[3*j]
	LEAQ 24(AX), DI           // DI = offset to twiddle[3*j+3] (+24 bytes = +3 complex64)
	VMOVSD (R10)(DI*1), X7    // X7 = twiddle[3*j+3]
	LEAQ 24(DI), DI           // DI = offset to twiddle[3*j+6]
	VMOVSD (R10)(DI*1), X8    // X8 = twiddle[3*j+6]
	LEAQ 24(DI), DI           // DI = offset to twiddle[3*j+9]
	VMOVSD (R10)(DI*1), X9    // X9 = twiddle[3*j+9]
	VPUNPCKLQDQ X7, X6, X6    // X6 = [twiddle[3*j], twiddle[3*j+3]]
	VPUNPCKLQDQ X9, X8, X8    // X8 = [twiddle[3*j+6], twiddle[3*j+9]]
	VINSERTF128 $1, X8, Y6, Y6 // Y6 = combine into YMM (4 strided w3 twiddles)

	// Complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP Y4, Y7          // Y7 = [w1.re, w1.re, w1.re, w1.re, ...] (duplicate reals)
	VMOVSHDUP Y4, Y8          // Y8 = [w1.im, w1.im, w1.im, w1.im, ...] (duplicate imags)
	VSHUFPS $0xB1, Y1, Y1, Y9 // Y9 = [a1.im, a1.re, ...] (swap re/im pairs)
	VMULPS Y8, Y9, Y9         // Y9 = [w1.im*a1.im, w1.im*a1.re, ...]
	VFMADDSUB231PS Y7, Y1, Y9 // Y9 = Y9 +/- [w1.re*a1.re, w1.re*a1.im, ...] = a1*w1
	VMOVAPS Y9, Y1            // Y1 = a1*w1 (result)

	VMOVSLDUP Y5, Y7          // Y7 = duplicate w2 reals
	VMOVSHDUP Y5, Y8          // Y8 = duplicate w2 imags
	VSHUFPS $0xB1, Y2, Y2, Y9 // Y9 = swap a2 re/im pairs
	VMULPS Y8, Y9, Y9         // Y9 = w2.im * a2 (swapped)
	VFMADDSUB231PS Y7, Y2, Y9 // Y9 = Y9 +/- w2.re * a2 = a2*w2
	VMOVAPS Y9, Y2            // Y2 = a2*w2 (result)

	VMOVSLDUP Y6, Y7          // Y7 = duplicate w3 reals
	VMOVSHDUP Y6, Y8          // Y8 = duplicate w3 imags
	VSHUFPS $0xB1, Y3, Y3, Y9 // Y9 = swap a3 re/im pairs
	VMULPS Y8, Y9, Y9         // Y9 = w3.im * a3 (swapped)
	VFMADDSUB231PS Y7, Y3, Y9 // Y9 = Y9 +/- w3.re * a3 = a3*w3
	VMOVAPS Y9, Y3            // Y3 = a3*w3 (result)

	VADDPS Y0, Y2, Y10        // Y10 = a0 + a2 (t0)
	VSUBPS Y2, Y0, Y11        // Y11 = a0 - a2 (t1)
	VADDPS Y1, Y3, Y12        // Y12 = a1 + a3 (t2)
	VSUBPS Y3, Y1, Y13        // Y13 = a1 - a3 (t3)

	// (-i)*t3
	VPERMILPS $0xB1, Y13, Y14 // Y14 = [t3.im, t3.re, ...] (swap re/im)
	VXORPS Y15, Y15, Y15      // Y15 = 0
	VSUBPS Y14, Y15, Y7       // Y7 = -Y14 = [-t3.im, -t3.re, ...]
	VBLENDPS $0xAA, Y7, Y14, Y14 // Y14 = [t3.im, -t3.re, ...] = (-i)*t3

	// i*t3
	VPERMILPS $0xB1, Y13, Y7  // Y7 = [t3.im, t3.re, ...] (swap re/im)
	VSUBPS Y7, Y15, Y8        // Y8 = -Y7 = [-t3.im, -t3.re, ...]
	VBLENDPS $0x55, Y8, Y7, Y7 // Y7 = [-t3.im, t3.re, ...] = i*t3

	VADDPS Y10, Y12, Y0       // Y0 = t0 + t2 (output to a0)
	VADDPS Y11, Y14, Y1       // Y1 = t1 + (-i)*t3 (output to a1)
	VSUBPS Y12, Y10, Y2       // Y2 = t0 - t2 (output to a2)
	VADDPS Y11, Y7, Y3        // Y3 = t1 + i*t3 (output to a3)

	LEAQ (SI)(R11*1), DI      // DI = SI + quarter_bytes (store offset for a1)
	LEAQ (DI)(R11*1), AX      // AX = DI + quarter_bytes (store offset for a2)
	LEAQ (AX)(R11*1), BP      // BP = AX + quarter_bytes (store offset for a3)

	VMOVUPS Y0, (R8)(SI*1)    // store 4 complex to a0[j:j+4]
	VMOVUPS Y1, (R8)(DI*1)    // store 4 complex to a1[j:j+4]
	VMOVUPS Y2, (R8)(AX*1)    // store 4 complex to a2[j:j+4]
	VMOVUPS Y3, (R8)(BP*1)    // store 4 complex to a3[j:j+4]

	ADDQ $4, DX               // j += 4 (processed 4 butterflies)
	JMP  fwd_r4_step1_loop    // continue loop

fwd_r4_stepn_prep:
	// Fast path for strided twiddles (step > 1).
	CMPQ R15, $4              // check if quarter >= 4
	JL   fwd_r4_stage_scalar  // if quarter < 4, use scalar path

	MOVQ R15, R11             // R11 = quarter
	SHLQ $3, R11              // R11 = quarter * 8 (quarter_bytes)

	MOVQ BX, R9               // R9 = step
	SHLQ $3, R9               // R9 = step * 8 (stride1_bytes, byte stride for w1)

	MOVQ DX, DI               // DI = j
	IMULQ BX, DI              // DI = j * step
	SHLQ $3, DI               // DI = j * step * 8 (twiddle base offset in bytes)

fwd_r4_stepn_loop:
	MOVQ R15, AX              // AX = quarter
	SUBQ DX, AX               // AX = quarter - j (remaining butterflies)
	CMPQ AX, $4               // check if remaining >= 4
	JL   fwd_r4_stage_scalar  // if remaining < 4, switch to scalar

	// base offset
	MOVQ CX, SI               // SI = base
	ADDQ DX, SI               // SI = base + j
	SHLQ $3, SI               // SI = (base + j) * 8 (byte offset)

	LEAQ (SI)(R11*1), R14     // R14 = SI + quarter_bytes (offset to a1)
	LEAQ (R14)(R11*1), AX     // AX = R14 + quarter_bytes (offset to a2)
	LEAQ (AX)(R11*1), BP      // BP = AX + quarter_bytes (offset to a3)

	VMOVUPS (R8)(SI*1), Y0    // Y0 = load 4 complex from a0[j:j+4]
	VMOVUPS (R8)(R14*1), Y1   // Y1 = load 4 complex from a1[j:j+4]
	VMOVUPS (R8)(AX*1), Y2    // Y2 = load 4 complex from a2[j:j+4]
	VMOVUPS (R8)(BP*1), Y3    // Y3 = load 4 complex from a3[j:j+4]

	// twiddle base offsets
	MOVQ DI, AX               // AX = j*step*8
	SHLQ $1, AX               // AX = 2*j*step*8 (w2 base offset)
	MOVQ DI, BP               // BP = j*step*8
	ADDQ AX, BP               // BP = 3*j*step*8 (w3 base offset)

	// w1 = twiddle[j*step + i*step] for i=0..3
	VMOVSD (R10)(DI*1), X4    // X4 = twiddle[j*step + 0*step]
	ADDQ R9, DI               // DI += stride1_bytes
	VMOVSD (R10)(DI*1), X5    // X5 = twiddle[j*step + 1*step]
	ADDQ R9, DI               // DI += stride1_bytes
	VMOVSD (R10)(DI*1), X6    // X6 = twiddle[j*step + 2*step]
	ADDQ R9, DI               // DI += stride1_bytes
	VMOVSD (R10)(DI*1), X7    // X7 = twiddle[j*step + 3*step]
	ADDQ R9, DI               // DI += stride1_bytes (advance for next iteration)
	VPUNPCKLQDQ X5, X4, X4    // X4 = [w1[0], w1[1]]
	VPUNPCKLQDQ X7, X6, X6    // X6 = [w1[2], w1[3]]
	VINSERTF128 $1, X6, Y4, Y4 // Y4 = 4 strided w1 twiddles

	// w2 = twiddle[2*j*step + i*2*step]
	MOVQ R9, R14              // R14 = stride1_bytes
	SHLQ $1, R14              // R14 = 2 * stride1_bytes (stride2_bytes)
	VMOVSD (R10)(AX*1), X5    // X5 = twiddle[2*j*step + 0*2*step]
	ADDQ R14, AX              // AX += stride2_bytes
	VMOVSD (R10)(AX*1), X6    // X6 = twiddle[2*j*step + 1*2*step]
	ADDQ R14, AX              // AX += stride2_bytes
	VMOVSD (R10)(AX*1), X7    // X7 = twiddle[2*j*step + 2*2*step]
	ADDQ R14, AX              // AX += stride2_bytes
	VMOVSD (R10)(AX*1), X8    // X8 = twiddle[2*j*step + 3*2*step]
	VPUNPCKLQDQ X6, X5, X5    // X5 = [w2[0], w2[1]]
	VPUNPCKLQDQ X8, X7, X7    // X7 = [w2[2], w2[3]]
	VINSERTF128 $1, X7, Y5, Y5 // Y5 = 4 strided w2 twiddles

	// w3 = twiddle[3*j*step + i*3*step]
	LEAQ (R9)(R14*1), R14     // R14 = stride1_bytes + stride2_bytes (stride3_bytes)
	VMOVSD (R10)(BP*1), X6    // X6 = twiddle[3*j*step + 0*3*step]
	ADDQ R14, BP              // BP += stride3_bytes
	VMOVSD (R10)(BP*1), X7    // X7 = twiddle[3*j*step + 1*3*step]
	ADDQ R14, BP              // BP += stride3_bytes
	VMOVSD (R10)(BP*1), X8    // X8 = twiddle[3*j*step + 2*3*step]
	ADDQ R14, BP              // BP += stride3_bytes
	VMOVSD (R10)(BP*1), X9    // X9 = twiddle[3*j*step + 3*3*step]
	VPUNPCKLQDQ X7, X6, X6    // X6 = [w3[0], w3[1]]
	VPUNPCKLQDQ X9, X8, X8    // X8 = [w3[2], w3[3]]
	VINSERTF128 $1, X8, Y6, Y6 // Y6 = 4 strided w3 twiddles

	// Complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP Y4, Y7          // Y7 = [w1.re, w1.re, w1.re, w1.re, ...] (duplicate reals)
	VMOVSHDUP Y4, Y8          // Y8 = [w1.im, w1.im, w1.im, w1.im, ...] (duplicate imags)
	VSHUFPS $0xB1, Y1, Y1, Y9 // Y9 = [a1.im, a1.re, ...] (swap re/im pairs)
	VMULPS Y8, Y9, Y9         // Y9 = [w1.im*a1.im, w1.im*a1.re, ...]
	VFMADDSUB231PS Y7, Y1, Y9 // Y9 = Y9 +/- [w1.re*a1.re, w1.re*a1.im, ...] = a1*w1
	VMOVAPS Y9, Y1            // Y1 = a1*w1 (result)

	VMOVSLDUP Y5, Y7          // Y7 = duplicate w2 reals
	VMOVSHDUP Y5, Y8          // Y8 = duplicate w2 imags
	VSHUFPS $0xB1, Y2, Y2, Y9 // Y9 = swap a2 re/im pairs
	VMULPS Y8, Y9, Y9         // Y9 = w2.im * a2 (swapped)
	VFMADDSUB231PS Y7, Y2, Y9 // Y9 = Y9 +/- w2.re * a2 = a2*w2
	VMOVAPS Y9, Y2            // Y2 = a2*w2 (result)

	VMOVSLDUP Y6, Y7          // Y7 = duplicate w3 reals
	VMOVSHDUP Y6, Y8          // Y8 = duplicate w3 imags
	VSHUFPS $0xB1, Y3, Y3, Y9 // Y9 = swap a3 re/im pairs
	VMULPS Y8, Y9, Y9         // Y9 = w3.im * a3 (swapped)
	VFMADDSUB231PS Y7, Y3, Y9 // Y9 = Y9 +/- w3.re * a3 = a3*w3
	VMOVAPS Y9, Y3            // Y3 = a3*w3 (result)

	VADDPS Y0, Y2, Y10        // Y10 = a0 + a2 (t0)
	VSUBPS Y2, Y0, Y11        // Y11 = a0 - a2 (t1)
	VADDPS Y1, Y3, Y12        // Y12 = a1 + a3 (t2)
	VSUBPS Y3, Y1, Y13        // Y13 = a1 - a3 (t3)

	// (-i)*t3
	VPERMILPS $0xB1, Y13, Y14    // Y14 = [t3.im, t3.re, ...] (swap re/im)
	VXORPS Y15, Y15, Y15         // Y15 = 0
	VSUBPS Y14, Y15, Y7          // Y7 = -Y14 = [-t3.im, -t3.re, ...]
	VBLENDPS $0xAA, Y7, Y14, Y14 // Y14 = [t3.im, -t3.re, ...] = (-i)*t3

	// i*t3
	VPERMILPS $0xB1, Y13, Y7    // Y7 = [t3.im, t3.re, ...] (swap re/im)
	VSUBPS Y7, Y15, Y8          // Y8 = -Y7 = [-t3.im, -t3.re, ...]
	VBLENDPS $0x55, Y8, Y7, Y7  // Y7 = [-t3.im, t3.re, ...] = i*t3

	VADDPS Y10, Y12, Y0       // Y0 = t0 + t2 (output to a0)
	VADDPS Y11, Y14, Y1       // Y1 = t1 + (-i)*t3 (output to a1)
	VSUBPS Y12, Y10, Y2       // Y2 = t0 - t2 (output to a2)
	VADDPS Y11, Y7, Y3        // Y3 = t1 + i*t3 (output to a3)

	LEAQ (SI)(R11*1), R14     // R14 = SI + quarter_bytes (store offset for a1)
	LEAQ (R14)(R11*1), AX     // AX = R14 + quarter_bytes (store offset for a2)
	LEAQ (AX)(R11*1), BP      // BP = AX + quarter_bytes (store offset for a3)

	VMOVUPS Y0, (R8)(SI*1)    // store 4 complex to a0[j:j+4]
	VMOVUPS Y1, (R8)(R14*1)   // store 4 complex to a1[j:j+4]
	VMOVUPS Y2, (R8)(AX*1)    // store 4 complex to a2[j:j+4]
	VMOVUPS Y3, (R8)(BP*1)    // store 4 complex to a3[j:j+4]

	ADDQ $4, DX               // j += 4 (processed 4 butterflies)
	JMP  fwd_r4_stepn_loop    // continue loop

fwd_r4_stage_scalar:
	CMPQ DX, R15                // check if j >= quarter (all butterflies done?)
	JGE  fwd_r4_stage_base_next // if done, advance to next group

	// indices
	MOVQ CX, SI               // SI = base + j (index of a0)
	ADDQ DX, SI               // SI = base + j
	MOVQ SI, DI               // DI = SI
	ADDQ R15, DI              // DI = SI + quarter (index of a1)
	MOVQ DI, R11              // R11 = DI
	ADDQ R15, R11             // R11 = DI + quarter (index of a2)
	MOVQ R11, R9              // R9 = R11
	ADDQ R15, R9              // R9 = R11 + quarter (index of a3)

	// twiddle indices: w1 = twiddle[j*step], w2 = twiddle[2*j*step], w3 = twiddle[3*j*step]
	MOVQ DX, AX               // AX = j
	IMULQ BX, AX              // AX = j * step
	VMOVSD (R10)(AX*8), X8    // X8 = w1 = twiddle[j*step]

	MOVQ AX, BP               // BP = j*step
	SHLQ $1, BP               // BP = 2*j*step
	VMOVSD (R10)(BP*8), X9    // X9 = w2 = twiddle[2*j*step]

	ADDQ AX, BP               // BP = 3*j*step
	VMOVSD (R10)(BP*8), X10   // X10 = w3 = twiddle[3*j*step]

	// load inputs
	VMOVSD (R8)(SI*8), X0     // X0 = a0
	VMOVSD (R8)(DI*8), X1     // X1 = a1
	VMOVSD (R8)(R11*8), X2    // X2 = a2
	VMOVSD (R8)(R9*8), X3     // X3 = a3

	// Complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP X8, X11           // X11 = [w1.re, w1.re] (duplicate reals)
	VMOVSHDUP X8, X12           // X12 = [w1.im, w1.im] (duplicate imags)
	VSHUFPS $0xB1, X1, X1, X13  // X13 = [a1.im, a1.re] (swap)
	VMULPS X12, X13, X13        // X13 = [w1.im*a1.im, w1.im*a1.re]
	VFMADDSUB231PS X11, X1, X13 // X13 = X13 +/- [w1.re*a1.re, w1.re*a1.im] = a1*w1
	VMOVAPS X13, X1             // X1 = a1*w1

	VMOVSLDUP X9, X11           // X11 = [w2.re, w2.re]
	VMOVSHDUP X9, X12           // X12 = [w2.im, w2.im]
	VSHUFPS $0xB1, X2, X2, X13  // X13 = [a2.im, a2.re]
	VMULPS X12, X13, X13        // X13 = [w2.im*a2.im, w2.im*a2.re]
	VFMADDSUB231PS X11, X2, X13 // X13 = a2*w2
	VMOVAPS X13, X2             // X2 = a2*w2

	VMOVSLDUP X10, X11          // X11 = [w3.re, w3.re]
	VMOVSHDUP X10, X12          // X12 = [w3.im, w3.im]
	VSHUFPS $0xB1, X3, X3, X13  // X13 = [a3.im, a3.re]
	VMULPS X12, X13, X13        // X13 = [w3.im*a3.im, w3.im*a3.re]
	VFMADDSUB231PS X11, X3, X13 // X13 = a3*w3
	VMOVAPS X13, X3             // X3 = a3*w3

	VADDPS X0, X2, X4         // X4 = a0 + a2 (t0)
	VSUBPS X2, X0, X5         // X5 = a0 - a2 (t1)
	VADDPS X1, X3, X6         // X6 = a1 + a3 (t2)
	VSUBPS X3, X1, X7         // X7 = a1 - a3 (t3)

	// (-i)*t3
	VPERMILPS $0xB1, X7, X8     // X8 = [t3.im, t3.re] (swap)
	VXORPS X9, X9, X9           // X9 = 0
	VSUBPS X8, X9, X10          // X10 = -X8 = [-t3.im, -t3.re]
	VBLENDPS $0x02, X10, X8, X8 // X8 = [t3.im, -t3.re] = (-i)*t3

	// i*t3
	VPERMILPS $0xB1, X7, X11      // X11 = [t3.im, t3.re] (swap)
	VSUBPS X11, X9, X10           // X10 = -X11 = [-t3.im, -t3.re]
	VBLENDPS $0x01, X10, X11, X11 // X11 = [-t3.im, t3.re] = i*t3

	VADDPS X4, X6, X0         // X0 = t0 + t2 (output to a0)
	VADDPS X5, X8, X1         // X1 = t1 + (-i)*t3 (output to a1)
	VSUBPS X6, X4, X2         // X2 = t0 - t2 (output to a2)
	VADDPS X5, X11, X3        // X3 = t1 + i*t3 (output to a3)

	VMOVSD X0, (R8)(SI*8)     // store result to a0
	VMOVSD X1, (R8)(DI*8)     // store result to a1
	VMOVSD X2, (R8)(R11*8)    // store result to a2
	VMOVSD X3, (R8)(R9*8)     // store result to a3

	INCQ DX                   // j++ (next butterfly)
	JMP  fwd_r4_stage_scalar  // continue scalar loop

fwd_r4_stage_base_next:
	MOVQ R15, R14             // R14 = quarter
	SHLQ $2, R14              // R14 = quarter * 4 = size
	ADDQ R14, CX              // CX = base + size (advance to next group)
	JMP  fwd_r4_stage_base    // continue base loop

fwd_r4_stage_next:
	ADDQ $2, R12              // R12 += 2 (log2(size) += 2)
	SHLQ $2, R14              // R14 *= 4 (size *= 4, next radix-4 stage)
	JMP  fwd_r4_stage_loop    // continue stage loop

fwd_r4_copy_back:
	MOVQ dst+0(FP), AX        // AX = original dst pointer
	CMPQ R8, AX               // check if working buffer == dst
	JE   fwd_r4_return_true   // if equal, no copy needed (already in dst)

	XORQ CX, CX               // CX = 0 (copy index)

fwd_r4_copy_loop:
	CMPQ CX, R13              // check if CX >= n (all elements copied?)
	JGE  fwd_r4_return_true   // if done, return success
	MOVQ (R8)(CX*8), DX       // DX = load element from working buffer
	MOVQ DX, (AX)(CX*8)       // store element to dst
	INCQ CX                   // CX++ (next element)
	JMP  fwd_r4_copy_loop     // continue copy loop

fwd_r4_return_true:
	VZEROUPPER                // clear upper YMM registers (required after AVX2)
	MOVB $1, ret+120(FP)      // return true
	RET

fwd_r4_return_false:
	MOVB $0, ret+120(FP)      // return false
	RET

// ===========================================================================
// inverseAVX2Complex64Radix4Asm - Inverse FFT for complex64 using radix-4 DIT
// ===========================================================================
//
// Inverse FFT follows the same structure as forward FFT with two key differences:
// 1. Uses conjugate complex multiplication (VFMSUBADD231PS instead of VFMADDSUB231PS)
//    This implements: a*conj(w) instead of a*w
// 2. Swaps i and -i butterfly operations in the final stage
//    Forward: y1 = t1 + (-i)*t3, y3 = t1 + i*t3
//    Inverse: y1 = t1 + i*t3,    y3 = t1 + (-i)*t3
// 3. Scales all output by 1/n at the end for proper normalization
//
// The algorithm structure is otherwise identical to forward transform:
// - Base-4 bit-reversal permutation
// - Radix-4 DIT butterfly stages
// - Vectorized (AVX2) and scalar paths
// - Step-1 (contiguous twiddles) and step-N (strided twiddles) optimization
//
TEXT ·InverseAVX2Complex64Radix4Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Empty input is valid (no-op)
	TESTQ R13, R13
	JZ    inv_r4_return_true

	// Validate all slice lengths are >= n
	MOVQ dst+8(FP), AX
	CMPQ AX, R13
	JL   inv_r4_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, R13
	JL   inv_r4_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, R13
	JL   inv_r4_return_false

	// Trivial case: n=1, just copy
	CMPQ R13, $1
	JNE  inv_r4_check_power_of_2
	MOVQ (R9), AX
	MOVQ AX, (R8)
	JMP  inv_r4_return_true

inv_r4_check_power_of_2:
	// Verify n is power of 2: (n & (n-1)) == 0
	MOVQ R13, AX
	LEAQ -1(AX), BX
	TESTQ AX, BX
	JNZ  inv_r4_return_false

	// Require even log2 (power-of-4) and minimum size
	MOVQ R13, AX
	BSRQ AX, R14
	TESTQ $1, R14
	JNZ  inv_r4_return_false

	CMPQ R13, $64
	JL   inv_r4_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  inv_r4_use_dst
	MOVQ R11, R8

inv_r4_use_dst:
	// -----------------------------------------------------------------------
	// Bit-reversal permutation (base-4)
	// -----------------------------------------------------------------------
	XORQ CX, CX               // CX = 0 (sequential index)

inv_r4_bitrev_loop:
	CMPQ CX, R13              // check if index CX >= n (done?)
	JGE  inv_r4_stage_init    // if done, jump to stage init

	MOVQ CX, DX               // DX = current index to bit-reverse
	XORQ BX, BX               // BX = 0 (accumulator for reversed index)
	MOVQ R14, SI              // SI = log2(n) (loop counter in bits)

inv_r4_bitrev_inner:
	CMPQ SI, $0               // check if all bits processed
	JE   inv_r4_bitrev_store  // if done, store result
	MOVQ DX, AX               // AX = copy of current index
	ANDQ $3, AX               // AX = lowest 2 bits (base-4 digit)
	SHLQ $2, BX               // BX <<= 2 (shift reversed index left by 2 bits)
	ORQ  AX, BX               // BX |= AX (append base-4 digit to reversed index)
	SHRQ $2, DX               // DX >>= 2 (shift right by 2 bits for next digit)
	SUBQ $2, SI               // SI -= 2 (processed 2 bits)
	JMP  inv_r4_bitrev_inner  // continue inner loop

inv_r4_bitrev_store:
	MOVQ (R9)(BX*8), AX       // load complex value from src[reversed_index]
	MOVQ AX, (R8)(CX*8)       // store to dst[sequential_index]
	INCQ CX                   // increment sequential index
	JMP  inv_r4_bitrev_loop   // continue outer loop

inv_r4_stage_init:
	// -----------------------------------------------------------------------
	// Radix-4 stages (size = 4, 16, 64, ...)
	// -----------------------------------------------------------------------
	MOVQ $2, R12              // R12 = 2 (log2(size) starting at 2, since first stage is size=4)
	MOVQ $4, R14              // R14 = 4 (initial size for first radix-4 stage)

inv_r4_stage_loop:
	CMPQ R14, R13             // compare current size with n
	JG   inv_r4_copy_back     // if size > n, all stages done, copy back if needed

	MOVQ R14, R15             // R15 = current stage size
	SHRQ $2, R15              // R15 = size/4 (quarter, number of butterflies per group)

	// step = n >> log2(size)
	MOVQ R13, BX              // BX = n
	MOVQ R12, CX              // CX = log2(size)
	SHRQ CL, BX               // BX = n >> log2(size) (twiddle stride)

	XORQ CX, CX               // CX = 0 (base index for current group)

inv_r4_stage_base:
	CMPQ CX, R13              // check if base >= n (all groups processed?)
	JGE  inv_r4_stage_next    // if done, advance to next stage

	XORQ DX, DX               // DX = 0 (j, butterfly index within group)

	// Fast path for contiguous twiddles on the final stage (step == 1).
	CMPQ BX, $1               // check if step == 1 (final stage)
	JNE  inv_r4_stepn_prep    // if step != 1, use strided twiddle path
	CMPQ R15, $4              // check if quarter >= 4 (enough for vectorization)
	JL   inv_r4_stage_scalar  // if quarter < 4, use scalar path

	MOVQ R15, R11             // R11 = quarter
	SHLQ $3, R11              // R11 = quarter * 8 (quarter_bytes, offset between a0/a1/a2/a3)

inv_r4_step1_loop:
	MOVQ R15, AX              // AX = quarter
	SUBQ DX, AX               // AX = quarter - j (remaining butterflies)
	CMPQ AX, $4               // check if remaining >= 4 (can process 4 complex values)
	JL   inv_r4_stage_scalar  // if remaining < 4, switch to scalar

	// base offset
	MOVQ CX, SI               // SI = base (group start index)
	ADDQ DX, SI               // SI = base + j (element index)
	SHLQ $3, SI               // SI = (base + j) * 8 (byte offset for complex64)

	LEAQ (SI)(R11*1), DI      // DI = SI + quarter_bytes (offset to a1)
	LEAQ (DI)(R11*1), AX      // AX = DI + quarter_bytes (offset to a2)
	LEAQ (AX)(R11*1), BP      // BP = AX + quarter_bytes (offset to a3)

	VMOVUPS (R8)(SI*1), Y0    // Y0 = load 4 complex from a0[j:j+4]
	VMOVUPS (R8)(DI*1), Y1    // Y1 = load 4 complex from a1[j:j+4]
	VMOVUPS (R8)(AX*1), Y2    // Y2 = load 4 complex from a2[j:j+4]
	VMOVUPS (R8)(BP*1), Y3    // Y3 = load 4 complex from a3[j:j+4]

	// w1 = twiddle[j : j+4] (contiguous)
	MOVQ DX, AX               // AX = j
	SHLQ $3, AX               // AX = j * 8 (byte offset)
	VMOVUPS (R10)(AX*1), Y4   // Y4 = load 4 contiguous twiddles w1[j:j+4]

	// w2 = twiddle[2*j, 2*j+2, 2*j+4, 2*j+6]
	MOVQ DX, AX               // AX = j
	SHLQ $1, AX               // AX = 2*j
	SHLQ $3, AX               // AX = 2*j * 8 (byte offset)
	VMOVSD (R10)(AX*1), X5    // X5 = twiddle[2*j] (one complex64)
	LEAQ 16(AX), DI           // DI = offset to twiddle[2*j+2] (+16 bytes = +2 complex64)
	VMOVSD (R10)(DI*1), X6    // X6 = twiddle[2*j+2]
	LEAQ 16(DI), DI           // DI = offset to twiddle[2*j+4]
	VMOVSD (R10)(DI*1), X7    // X7 = twiddle[2*j+4]
	LEAQ 16(DI), DI           // DI = offset to twiddle[2*j+6]
	VMOVSD (R10)(DI*1), X8    // X8 = twiddle[2*j+6]
	VPUNPCKLQDQ X6, X5, X5    // X5 = [twiddle[2*j], twiddle[2*j+2]]
	VPUNPCKLQDQ X8, X7, X7    // X7 = [twiddle[2*j+4], twiddle[2*j+6]]
	VINSERTF128 $1, X7, Y5, Y5 // Y5 = combine into YMM (4 strided w2 twiddles)

	// w3 = twiddle[3*j, 3*j+3, 3*j+6, 3*j+9]
	LEAQ (DX)(DX*2), AX       // AX = 3*j (using LEA for 3x multiplication)
	SHLQ $3, AX               // AX = 3*j * 8 (byte offset)
	VMOVSD (R10)(AX*1), X6    // X6 = twiddle[3*j]
	LEAQ 24(AX), DI           // DI = offset to twiddle[3*j+3] (+24 bytes = +3 complex64)
	VMOVSD (R10)(DI*1), X7    // X7 = twiddle[3*j+3]
	LEAQ 24(DI), DI           // DI = offset to twiddle[3*j+6]
	VMOVSD (R10)(DI*1), X8    // X8 = twiddle[3*j+6]
	LEAQ 24(DI), DI           // DI = offset to twiddle[3*j+9]
	VMOVSD (R10)(DI*1), X9    // X9 = twiddle[3*j+9]
	VPUNPCKLQDQ X7, X6, X6    // X6 = [twiddle[3*j], twiddle[3*j+3]]
	VPUNPCKLQDQ X9, X8, X8    // X8 = [twiddle[3*j+6], twiddle[3*j+9]]
	VINSERTF128 $1, X8, Y6, Y6 // Y6 = combine into YMM (4 strided w3 twiddles)

	// Conjugate complex multiply a1*conj(w1), a2*conj(w2), a3*conj(w3)
	VMOVSLDUP Y4, Y7          // Y7 = [w1.re, w1.re, w1.re, w1.re, ...] (duplicate reals)
	VMOVSHDUP Y4, Y8          // Y8 = [w1.im, w1.im, w1.im, w1.im, ...] (duplicate imags)
	VSHUFPS $0xB1, Y1, Y1, Y9 // Y9 = [a1.im, a1.re, ...] (swap re/im pairs)
	VMULPS Y8, Y9, Y9         // Y9 = [w1.im*a1.im, w1.im*a1.re, ...]
	VFMSUBADD231PS Y7, Y1, Y9 // Y9 = Y9 -/+ [w1.re*a1.re, w1.re*a1.im, ...] = a1*conj(w1)
	VMOVAPS Y9, Y1            // Y1 = a1*conj(w1) (result)

	VMOVSLDUP Y5, Y7          // Y7 = duplicate w2 reals
	VMOVSHDUP Y5, Y8          // Y8 = duplicate w2 imags
	VSHUFPS $0xB1, Y2, Y2, Y9 // Y9 = swap a2 re/im pairs
	VMULPS Y8, Y9, Y9         // Y9 = w2.im * a2 (swapped)
	VFMSUBADD231PS Y7, Y2, Y9 // Y9 = Y9 -/+ w2.re * a2 = a2*conj(w2)
	VMOVAPS Y9, Y2            // Y2 = a2*conj(w2) (result)

	VMOVSLDUP Y6, Y7          // Y7 = duplicate w3 reals
	VMOVSHDUP Y6, Y8          // Y8 = duplicate w3 imags
	VSHUFPS $0xB1, Y3, Y3, Y9 // Y9 = swap a3 re/im pairs
	VMULPS Y8, Y9, Y9         // Y9 = w3.im * a3 (swapped)
	VFMSUBADD231PS Y7, Y3, Y9 // Y9 = Y9 -/+ w3.re * a3 = a3*conj(w3)
	VMOVAPS Y9, Y3            // Y3 = a3*conj(w3) (result)

	VADDPS Y0, Y2, Y10        // Y10 = a0 + a2 (t0)
	VSUBPS Y2, Y0, Y11        // Y11 = a0 - a2 (t1)
	VADDPS Y1, Y3, Y12        // Y12 = a1 + a3 (t2)
	VSUBPS Y3, Y1, Y13        // Y13 = a1 - a3 (t3)

	// (-i)*t3
	VPERMILPS $0xB1, Y13, Y14 // Y14 = [t3.im, t3.re, ...] (swap re/im)
	VXORPS Y15, Y15, Y15      // Y15 = 0
	VSUBPS Y14, Y15, Y7       // Y7 = -Y14 = [-t3.im, -t3.re, ...]
	VBLENDPS $0xAA, Y7, Y14, Y14 // Y14 = [t3.im, -t3.re, ...] = (-i)*t3

	// i*t3
	VPERMILPS $0xB1, Y13, Y7  // Y7 = [t3.im, t3.re, ...] (swap re/im)
	VSUBPS Y7, Y15, Y8        // Y8 = -Y7 = [-t3.im, -t3.re, ...]
	VBLENDPS $0x55, Y8, Y7, Y7 // Y7 = [-t3.im, t3.re, ...] = i*t3

	VADDPS Y10, Y12, Y0       // Y0 = t0 + t2 (output to a0)
	VADDPS Y11, Y7, Y1        // Y1 = t1 + i*t3 (output to a1, note: swapped for inverse)
	VSUBPS Y12, Y10, Y2       // Y2 = t0 - t2 (output to a2)
	VADDPS Y11, Y14, Y3       // Y3 = t1 + (-i)*t3 (output to a3, note: swapped for inverse)

	LEAQ (SI)(R11*1), DI      // DI = SI + R11 (offset to a1 = base + 1*stride)
	LEAQ (DI)(R11*1), AX      // AX = DI + R11 (offset to a2 = base + 2*stride)
	LEAQ (AX)(R11*1), BP      // BP = AX + R11 (offset to a3 = base + 3*stride)

	VMOVUPS Y0, (R8)(SI*1)    // Store Y0 to dst[a0] (t0 + t2)
	VMOVUPS Y1, (R8)(DI*1)    // Store Y1 to dst[a1] (t1 + i*t3, swapped for inverse)
	VMOVUPS Y2, (R8)(AX*1)    // Store Y2 to dst[a2] (t0 - t2)
	VMOVUPS Y3, (R8)(BP*1)    // Store Y3 to dst[a3] (t1 + (-i)*t3, swapped for inverse)

	ADDQ $4, DX               // j += 4 (advance to next group of 4 elements)
	JMP  inv_r4_step1_loop    // Continue loop for next butterfly

inv_r4_stepn_prep:
	// Fast path for strided twiddles (step > 1).
	CMPQ R15, $4              // Check if quarter >= 4
	JL   inv_r4_stage_scalar  // If quarter < 4, use scalar fallback

	MOVQ R15, R11             // R11 = quarter
	SHLQ $3, R11              // quarter_bytes = quarter * 8 (complex64 = 8 bytes)

	MOVQ BX, R9               // R9 = step
	SHLQ $3, R9               // stride1_bytes = step * 8 (stride in bytes)

	MOVQ DX, DI               // DI = j
	IMULQ BX, DI              // DI = j * step
	SHLQ $3, DI               // twiddle offset bytes = (j * step) * 8

inv_r4_stepn_loop:
	MOVQ R15, AX              // AX = quarter
	SUBQ DX, AX               // AX = quarter - j (remaining elements)
	CMPQ AX, $4               // Check if remaining >= 4
	JL   inv_r4_stage_scalar  // If remaining < 4, use scalar fallback

	// base offset
	MOVQ CX, SI               // SI = base
	ADDQ DX, SI               // SI = base + j
	SHLQ $3, SI               // SI = (base + j) * 8 (byte offset)

	LEAQ (SI)(R11*1), R14     // R14 = SI + quarter_bytes (offset to a1)
	LEAQ (R14)(R11*1), AX     // AX = R14 + quarter_bytes (offset to a2)
	LEAQ (AX)(R11*1), BP      // BP = AX + quarter_bytes (offset to a3)

	VMOVUPS (R8)(SI*1), Y0    // Y0 = load a0 from dst
	VMOVUPS (R8)(R14*1), Y1   // Y1 = load a1 from dst
	VMOVUPS (R8)(AX*1), Y2    // Y2 = load a2 from dst
	VMOVUPS (R8)(BP*1), Y3    // Y3 = load a3 from dst

	// twiddle base offsets
	MOVQ DI, AX               // AX = DI (twiddle offset)
	SHLQ $1, AX               // AX = 2 * DI
	MOVQ DI, BP               // BP = DI
	ADDQ AX, BP               // BP = 3 * DI

	// w1 = twiddle[j*step + i*step] for i=0..3
	VMOVSD (R10)(DI*1), X4    // X4 = w1[0] (load 64 bits)
	ADDQ R9, DI               // DI += stride1_bytes
	VMOVSD (R10)(DI*1), X5    // X5 = w1[1] (load 64 bits)
	ADDQ R9, DI               // DI += stride1_bytes
	VMOVSD (R10)(DI*1), X6    // X6 = w1[2] (load 64 bits)
	ADDQ R9, DI               // DI += stride1_bytes
	VMOVSD (R10)(DI*1), X7    // X7 = w1[3] (load 64 bits)
	ADDQ R9, DI               // DI += stride1_bytes (advance to next block)
	VPUNPCKLQDQ X5, X4, X4    // X4 = [w1[0], w1[1]] (pack 2 complex64)
	VPUNPCKLQDQ X7, X6, X6    // X6 = [w1[2], w1[3]] (pack 2 complex64)
	VINSERTF128 $1, X6, Y4, Y4 // Y4 = [w1[0], w1[1], w1[2], w1[3]] (4 complex64)

	// w2 = twiddle[2*j*step + i*2*step]
	MOVQ R9, R14              // R14 = stride1_bytes
	SHLQ $1, R14              // stride2_bytes = 2 * stride1_bytes
	VMOVSD (R10)(AX*1), X5    // X5 = w2[0] (load 64 bits)
	ADDQ R14, AX              // AX += stride2_bytes
	VMOVSD (R10)(AX*1), X6    // X6 = w2[1] (load 64 bits)
	ADDQ R14, AX              // AX += stride2_bytes
	VMOVSD (R10)(AX*1), X7    // X7 = w2[2] (load 64 bits)
	ADDQ R14, AX              // AX += stride2_bytes
	VMOVSD (R10)(AX*1), X8    // X8 = w2[3] (load 64 bits)
	VPUNPCKLQDQ X6, X5, X5    // X5 = [w2[0], w2[1]] (pack 2 complex64)
	VPUNPCKLQDQ X8, X7, X7    // X7 = [w2[2], w2[3]] (pack 2 complex64)
	VINSERTF128 $1, X7, Y5, Y5 // Y5 = [w2[0], w2[1], w2[2], w2[3]] (4 complex64)

	// w3 = twiddle[3*j*step + i*3*step]
	LEAQ (R9)(R14*1), R14     // stride3_bytes = stride1_bytes + stride2_bytes
	VMOVSD (R10)(BP*1), X6    // X6 = w3[0] (load 64 bits)
	ADDQ R14, BP              // BP += stride3_bytes
	VMOVSD (R10)(BP*1), X7    // X7 = w3[1] (load 64 bits)
	ADDQ R14, BP              // BP += stride3_bytes
	VMOVSD (R10)(BP*1), X8    // X8 = w3[2] (load 64 bits)
	ADDQ R14, BP              // BP += stride3_bytes
	VMOVSD (R10)(BP*1), X9    // X9 = w3[3] (load 64 bits)
	VPUNPCKLQDQ X7, X6, X6    // X6 = [w3[0], w3[1]] (pack 2 complex64)
	VPUNPCKLQDQ X9, X8, X8    // X8 = [w3[2], w3[3]] (pack 2 complex64)
	VINSERTF128 $1, X8, Y6, Y6 // Y6 = [w3[0], w3[1], w3[2], w3[3]] (4 complex64)

	// Conjugate complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP Y4, Y7          // Y7 = [w1.re, w1.re, ...] (duplicate real parts)
	VMOVSHDUP Y4, Y8          // Y8 = [w1.im, w1.im, ...] (duplicate imag parts)
	VSHUFPS $0xB1, Y1, Y1, Y9 // Y9 = [a1.im, a1.re, ...] (swap re/im)
	VMULPS Y8, Y9, Y9         // Y9 = w1.im * [a1.im, a1.re, ...]
	VFMSUBADD231PS Y7, Y1, Y9 // Y9 = conj(a1) * w1 (conjugate multiply for inverse)
	VMOVAPS Y9, Y1            // Y1 = result of a1 * w1

	VMOVSLDUP Y5, Y7          // Y7 = [w2.re, w2.re, ...] (duplicate real parts)
	VMOVSHDUP Y5, Y8          // Y8 = [w2.im, w2.im, ...] (duplicate imag parts)
	VSHUFPS $0xB1, Y2, Y2, Y9 // Y9 = [a2.im, a2.re, ...] (swap re/im)
	VMULPS Y8, Y9, Y9         // Y9 = w2.im * [a2.im, a2.re, ...]
	VFMSUBADD231PS Y7, Y2, Y9 // Y9 = conj(a2) * w2 (conjugate multiply for inverse)
	VMOVAPS Y9, Y2            // Y2 = result of a2 * w2

	VMOVSLDUP Y6, Y7          // Y7 = [w3.re, w3.re, ...] (duplicate real parts)
	VMOVSHDUP Y6, Y8          // Y8 = [w3.im, w3.im, ...] (duplicate imag parts)
	VSHUFPS $0xB1, Y3, Y3, Y9 // Y9 = [a3.im, a3.re, ...] (swap re/im)
	VMULPS Y8, Y9, Y9         // Y9 = w3.im * [a3.im, a3.re, ...]
	VFMSUBADD231PS Y7, Y3, Y9 // Y9 = conj(a3) * w3 (conjugate multiply for inverse)
	VMOVAPS Y9, Y3            // Y3 = result of a3 * w3

	VADDPS Y0, Y2, Y10        // Y10 = a0 + a2 = t0
	VSUBPS Y2, Y0, Y11        // Y11 = a0 - a2 = t1
	VADDPS Y1, Y3, Y12        // Y12 = a1 + a3 = t2
	VSUBPS Y3, Y1, Y13        // Y13 = a1 - a3 = t3

	// (-i)*t3
	VPERMILPS $0xB1, Y13, Y14 // Y14 = [t3.im, t3.re, ...] (swap re/im)
	VXORPS Y15, Y15, Y15      // Y15 = 0 (zero register)
	VSUBPS Y14, Y15, Y7       // Y7 = -Y14 = [-t3.im, -t3.re, ...]
	VBLENDPS $0xAA, Y7, Y14, Y14 // Y14 = [t3.im, -t3.re, ...] = (-i)*t3

	// i*t3
	VPERMILPS $0xB1, Y13, Y7  // Y7 = [t3.im, t3.re, ...] (swap re/im)
	VSUBPS Y7, Y15, Y8        // Y8 = -Y7 = [-t3.im, -t3.re, ...]
	VBLENDPS $0x55, Y8, Y7, Y7 // Y7 = [-t3.im, t3.re, ...] = i*t3

	VADDPS Y10, Y12, Y0       // Y0 = t0 + t2 (output to a0)
	VADDPS Y11, Y7, Y1        // Y1 = t1 + i*t3 (output to a1, swapped for inverse)
	VSUBPS Y12, Y10, Y2       // Y2 = t0 - t2 (output to a2)
	VADDPS Y11, Y14, Y3       // Y3 = t1 + (-i)*t3 (output to a3, swapped for inverse)

	LEAQ (SI)(R11*1), R14     // R14 = SI + quarter_bytes (offset to a1)
	LEAQ (R14)(R11*1), AX     // AX = R14 + quarter_bytes (offset to a2)
	LEAQ (AX)(R11*1), BP      // BP = AX + quarter_bytes (offset to a3)

	VMOVUPS Y0, (R8)(SI*1)    // Store Y0 to dst[a0] (t0 + t2)
	VMOVUPS Y1, (R8)(R14*1)   // Store Y1 to dst[a1] (t1 + i*t3)
	VMOVUPS Y2, (R8)(AX*1)    // Store Y2 to dst[a2] (t0 - t2)
	VMOVUPS Y3, (R8)(BP*1)    // Store Y3 to dst[a3] (t1 + (-i)*t3)

	ADDQ $4, DX               // j += 4 (advance to next group of 4 elements)
	JMP  inv_r4_stepn_loop    // Continue loop for next butterfly

inv_r4_stage_scalar:
	CMPQ DX, R15              // Check if j >= quarter
	JGE  inv_r4_stage_base_next // If j >= quarter, done with this stage

	// indices
	MOVQ CX, SI               // SI = base
	ADDQ DX, SI               // SI = base + j (index a0)
	MOVQ SI, DI               // DI = base + j
	ADDQ R15, DI              // DI = base + j + quarter (index a1)
	MOVQ DI, R11              // R11 = base + j + quarter
	ADDQ R15, R11             // R11 = base + j + 2*quarter (index a2)
	MOVQ R11, R9              // R9 = base + j + 2*quarter
	ADDQ R15, R9              // R9 = base + j + 3*quarter (index a3)

	// twiddle indices: w1 = twiddle[j*step], w2 = twiddle[2*j*step], w3 = twiddle[3*j*step]
	MOVQ DX, AX               // AX = j
	IMULQ BX, AX              // AX = j * step
	VMOVSD (R10)(AX*8), X8    // X8 = w1 (load complex64 at index j*step)

	MOVQ AX, BP               // BP = j * step
	SHLQ $1, BP               // BP = 2 * j * step
	VMOVSD (R10)(BP*8), X9    // X9 = w2 (load complex64 at index 2*j*step)

	ADDQ AX, BP               // BP = 3 * j * step
	VMOVSD (R10)(BP*8), X10   // X10 = w3 (load complex64 at index 3*j*step)

	// load inputs
	VMOVSD (R8)(SI*8), X0     // X0 = a0 (load from dst[base+j])
	VMOVSD (R8)(DI*8), X1     // X1 = a1 (load from dst[base+j+quarter])
	VMOVSD (R8)(R11*8), X2    // X2 = a2 (load from dst[base+j+2*quarter])
	VMOVSD (R8)(R9*8), X3     // X3 = a3 (load from dst[base+j+3*quarter])

	// Conjugate complex multiply a1*w1, a2*w2, a3*w3
	VMOVSLDUP X8, X11         // X11 = [w1.re, w1.re] (duplicate real part)
	VMOVSHDUP X8, X12         // X12 = [w1.im, w1.im] (duplicate imag part)
	VSHUFPS $0xB1, X1, X1, X13 // X13 = [a1.im, a1.re] (swap re/im)
	VMULPS X12, X13, X13      // X13 = w1.im * [a1.im, a1.re]
	VFMSUBADD231PS X11, X1, X13 // X13 = conj(a1) * w1 (conjugate multiply)
	VMOVAPS X13, X1           // X1 = result of a1 * w1

	VMOVSLDUP X9, X11         // X11 = [w2.re, w2.re] (duplicate real part)
	VMOVSHDUP X9, X12         // X12 = [w2.im, w2.im] (duplicate imag part)
	VSHUFPS $0xB1, X2, X2, X13 // X13 = [a2.im, a2.re] (swap re/im)
	VMULPS X12, X13, X13      // X13 = w2.im * [a2.im, a2.re]
	VFMSUBADD231PS X11, X2, X13 // X13 = conj(a2) * w2 (conjugate multiply)
	VMOVAPS X13, X2           // X2 = result of a2 * w2

	VMOVSLDUP X10, X11        // X11 = [w3.re, w3.re] (duplicate real part)
	VMOVSHDUP X10, X12        // X12 = [w3.im, w3.im] (duplicate imag part)
	VSHUFPS $0xB1, X3, X3, X13 // X13 = [a3.im, a3.re] (swap re/im)
	VMULPS X12, X13, X13      // X13 = w3.im * [a3.im, a3.re]
	VFMSUBADD231PS X11, X3, X13 // X13 = conj(a3) * w3 (conjugate multiply)
	VMOVAPS X13, X3           // X3 = result of a3 * w3

	VADDPS X0, X2, X4         // X4 = a0 + a2 = t0
	VSUBPS X2, X0, X5         // X5 = a0 - a2 = t1
	VADDPS X1, X3, X6         // X6 = a1 + a3 = t2
	VSUBPS X3, X1, X7         // X7 = a1 - a3 = t3

	// (-i)*t3
	VPERMILPS $0xB1, X7, X8   // X8 = [t3.im, t3.re] (swap re/im)
	VXORPS X9, X9, X9         // X9 = 0 (zero register)
	VSUBPS X8, X9, X10        // X10 = -X8 = [-t3.im, -t3.re]
	VBLENDPS $0x02, X10, X8, X8 // X8 = [t3.im, -t3.re] = (-i)*t3

	// i*t3
	VPERMILPS $0xB1, X7, X11  // X11 = [t3.im, t3.re] (swap re/im)
	VSUBPS X11, X9, X10       // X10 = -X11 = [-t3.im, -t3.re]
	VBLENDPS $0x01, X10, X11, X11 // X11 = [-t3.im, t3.re] = i*t3

	VADDPS X4, X6, X0         // X0 = t0 + t2 (output to a0)
	VADDPS X5, X11, X1        // X1 = t1 + i*t3 (output to a1)
	VSUBPS X6, X4, X2         // X2 = t0 - t2 (output to a2)
	VADDPS X5, X8, X3         // X3 = t1 + (-i)*t3 (output to a3)

	VMOVSD X0, (R8)(SI*8)     // Store X0 to dst[a0]
	VMOVSD X1, (R8)(DI*8)     // Store X1 to dst[a1]
	VMOVSD X2, (R8)(R11*8)    // Store X2 to dst[a2]
	VMOVSD X3, (R8)(R9*8)     // Store X3 to dst[a3]

	INCQ DX                   // j++ (advance to next element)
	JMP  inv_r4_stage_scalar  // Continue scalar loop

inv_r4_stage_base_next:
	MOVQ R15, R14             // R14 = quarter
	SHLQ $2, R14              // R14 = quarter * 4 (next base offset)
	ADDQ R14, CX              // CX = base + 4*quarter (advance to next group)
	JMP  inv_r4_stage_base    // Continue with next base group

inv_r4_stage_next:
	ADDQ $2, R12              // stage += 2 (next stage of FFT)
	SHLQ $2, R14              // R14 *= 4 (update size for next stage)
	JMP  inv_r4_stage_loop    // Continue with next stage

inv_r4_copy_back:
	MOVQ dst+0(FP), AX        // AX = original dst pointer
	CMPQ R8, AX               // Check if working buffer == dst
	JE   inv_r4_scale         // If equal, skip copy (already in place)

	XORQ CX, CX               // CX = 0 (copy index)

inv_r4_copy_loop:
	CMPQ CX, R13              // Check if CX >= n (all elements copied?)
	JGE  inv_r4_scale         // If done copying, go to scaling
	MOVQ (R8)(CX*8), DX       // DX = load element from working buffer
	MOVQ DX, (AX)(CX*8)       // Store element to dst
	INCQ CX                   // CX++ (next element)
	JMP  inv_r4_copy_loop     // Continue copy loop

inv_r4_scale:
	// scale by 1/n (inverse FFT normalization)
	CVTSQ2SS R13, X0          // X0 = float32(n) (convert n to float)
	MOVSS    ·one32(SB), X1   // X1 = 1.0 (load constant)
	DIVSS    X0, X1           // X1 = 1.0 / n (compute scale factor)
	SHUFPS   $0x00, X1, X1    // X1 = [1/n, 1/n, 1/n, 1/n] (broadcast to all lanes)

	XORQ CX, CX               // CX = 0 (scale index)

inv_r4_scale_loop:
	CMPQ CX, R13              // Check if CX >= n (all elements scaled?)
	JGE  inv_r4_return_true   // If done, return success
	MOVSD (AX)(CX*8), X0      // X0 = load one complex64 value (8 bytes)
	MULPS X1, X0              // X0 *= [1/n, 1/n] (scale both real and imaginary parts)
	MOVSD X0, (AX)(CX*8)      // Store scaled value back to dst
	INCQ CX                   // CX++ (next element)
	JMP  inv_r4_scale_loop    // Continue scaling loop

inv_r4_return_true:
	VZEROUPPER                // Clear upper YMM registers (required after AVX2 usage)
	MOVB $1, ret+120(FP)      // Return true (success)
	RET

inv_r4_return_false:
	MOVB $0, ret+120(FP)      // Return false (failure)
	RET
