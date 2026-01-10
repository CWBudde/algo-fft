//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-256 Radix-2 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 256.
// These kernels provide better performance than the generic implementation by:
//   - Eliminating loop overhead
//   - Using hardcoded twiddle factor indices
//   - Optimal register allocation for this size
//   - Processing 4 complex64 values (32 bytes) per YMM register
//
// Memory layout: 256 complex64 = 2048 bytes = 64 YMM registers worth
// We use a hybrid approach: keep some data in registers, spill to memory as needed.
//
// Algorithm: 8-stage radix-2 DIT FFT
//   Stage 1 (size=2):   128 butterflies, step=128, twiddle[0] for all
//   Stage 2 (size=4):   128 butterflies, step=64,  twiddle indices [0,64]
//   Stage 3 (size=8):   128 butterflies, step=32,  twiddle indices [0,32,64,96]
//   Stage 4 (size=16):  128 butterflies, step=16,  twiddle indices [0,16,32,...,112]
//   Stage 5 (size=32):  128 butterflies, step=8,   twiddle indices [0,8,16,...,120]
//   Stage 6 (size=64):  128 butterflies, step=4,   twiddle indices [0,4,8,...,124]
//   Stage 7 (size=128): 128 butterflies, step=2,   twiddle indices [0,2,4,...,126]
//   Stage 8 (size=256): 128 butterflies, step=1,   twiddle indices [0,1,2,...,127]
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 256, complex64
// ===========================================================================
// Uses looped structure for stages 5-8 to balance code size with performance.
// Stages 1-4 are more unrolled for critical early stages.
//
TEXT ·ForwardAVX2Size256Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 256)
	LEAQ ·bitrev256_r2(SB), R12

	// Verify n == 256
	CMPQ R13, $256
	JNE  size256_r2_return_false

	// Validate all slice lengths >= 256
	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   size256_r2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   size256_r2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   size256_r2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size256_r2_use_dst

	// In-place: use scratch
	MOVQ R11, R8

size256_r2_use_dst:
	// Out-of-place: use dst

	// =======================================================================
	// Bit-reversal permutation (radix-2) into work buffer
	// =======================================================================
	XORQ CX, CX

size256_r2_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $256
	JL   size256_r2_bitrev_loop

	// =======================================================================
	// STAGE 1: size=2, half=1, step=128
	// =======================================================================
	// 128 butterflies with pairs: (0,1), (2,3), (4,5), ..., (254,255)
	// All use twiddle[0] = (1, 0) which is identity multiplication.
	// Process 4 pairs at a time using YMM registers.

	XORQ CX, CX              // CX = base offset in bytes

size256_r2_stage1_loop:
	// Load 8 complex64 values (4 pairs) = 2 YMM registers
	// Each complex64 is 8 bytes (2 x float32)
	// Y0 = [a0.re, a0.im, b0.re, b0.im, a1.re, a1.im, b1.re, b1.im]
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1

	// Stage 1 butterfly: pairs (a0,b0), (a1,b1)
	// Want output: [a0+b0, a0-b0, a1+b1, a1-b1] = [(a0.re+b0.re, a0.im+b0.im), (a0.re-b0.re, a0.im-b0.im), ...]
	//
	// Use VSHUFPS to create:
	//   Y2 = [a0.re, a0.im, a0.re, a0.im, a1.re, a1.im, a1.re, a1.im] (a duplicated)
	//   Y3 = [b0.re, b0.im, b0.re, b0.im, b1.re, b1.im, b1.re, b1.im] (b duplicated)
	// Then: sum = Y2 + Y3, diff = Y2 - Y3
	// Finally blend to get [sum0, diff0, sum1, diff1]

	// Y0 = [a0.re, a0.im, b0.re, b0.im | a1.re, a1.im, b1.re, b1.im]
	// Duplicate 'a' positions to all slots, then 'b' positions
	VPERMILPS $0x44, Y0, Y2      // Y2 = [a0.re, a0.im, a0.re, a0.im | a1.re, a1.im, a1.re, a1.im]
	VPERMILPS $0xEE, Y0, Y3      // Y3 = [b0.re, b0.im, b0.re, b0.im | b1.re, b1.im, b1.re, b1.im]
	VADDPS Y3, Y2, Y4            // Y4 = [sum.re, sum.im, sum.re, sum.im | ...]
	VSUBPS Y3, Y2, Y5            // Y5 = [diff.re, diff.im, diff.re, diff.im | ...]
	VBLENDPS $0xCC, Y5, Y4, Y0   // Y0 = [sum0.re, sum0.im, diff0.re, diff0.im | sum1.re, sum1.im, diff1.re, diff1.im]

	// Same for Y1
	VPERMILPS $0x44, Y1, Y2
	VPERMILPS $0xEE, Y1, Y3
	VADDPS Y3, Y2, Y4
	VSUBPS Y3, Y2, Y5
	VBLENDPS $0xCC, Y5, Y4, Y1

	// Store results
	VMOVUPS Y0, (R8)(CX*1)
	VMOVUPS Y1, 32(R8)(CX*1)

	ADDQ $64, CX             // Move to next 8 complex64 (64 bytes)
	CMPQ CX, $2048           // 256 * 8 bytes = 2048
	JL   size256_r2_stage1_loop

	// =======================================================================
	// STAGE 2: size=4, half=2, step=64
	// =======================================================================
	// For each group of 4: [d0, d1, d2, d3]
	// Butterfly (d0, d2) with tw[0]=1: output d0' = d0+d2, d2' = d0-d2
	// Butterfly (d1, d3) with tw[64]:  output d1' = d1+tw*d3, d3' = d1-tw*d3
	// Final layout: [d0', d1', d2', d3']

	// Load twiddle[64] for j=1 butterflies
	VBROADCASTSD 512(R10), Y8    // Y8 = [tw64, tw64, tw64, tw64]

	XORQ CX, CX

size256_r2_stage2_loop:
	// Load 4 complex64 values
	VMOVUPS (R8)(CX*1), Y0       // Y0 = [d0, d1, d2, d3]

	// Butterfly (d0, d2) with twiddle=1:
	// Create [d0, d0, d0, d0] and [d2, d2, d2, d2] via permutation
	// d0 is at positions 0-1 (floats 0,1), d2 is at positions 4-5 (floats 4,5)
	// Within each 128-bit lane, d0 at 0-1, d1 at 2-3 in low lane; d2 at 0-1, d3 at 2-3 in high lane

	// Extract d0, d1 by duplicating low lane
	VPERM2F128 $0x00, Y0, Y0, Y1 // Y1 = [d0, d1, d0, d1]
	// Extract d2, d3 by duplicating high lane
	VPERM2F128 $0x11, Y0, Y0, Y2 // Y2 = [d2, d3, d2, d3]

	// For butterflies: (d0,d2) and (d1,d3)
	// j=0: tw=1, so t = 1*d2 = d2. Result: d0+d2, d0-d2
	// j=1: tw=tw64, so t = tw64*d3. Result: d1+t, d1-t

	// Since tw[0]=1, the j=0 butterfly is simple addition/subtraction
	// Y1 low lane = [d0, d1], Y2 low lane = [d2, d3]
	// For j=0: use d0 and d2 directly (they're in positions 0-1 of each lane)
	// For j=1: multiply d3 by tw64

	// Actually, let's be more explicit about the data layout
	// Y1 = [d0.re, d0.im, d1.re, d1.im | d0.re, d0.im, d1.re, d1.im]
	// Y2 = [d2.re, d2.im, d3.re, d3.im | d2.re, d2.im, d3.re, d3.im]

	// For j=0 butterfly: need d0 (pos 0) and d2 (pos 0 of Y2)
	// For j=1 butterfly: need d1 (pos 1) and d3 (pos 1 of Y2)

	// Complex multiply tw64 * d3
	// tw64 is in Y8 (broadcasted)
	// d3 is at positions 2-3 in each 128-bit lane of Y2

	// Extract d3 into its own location
	VPERMILPS $0xEE, Y2, Y3      // Y3 = [d3.re, d3.im, d3.re, d3.im | d3.re, d3.im, d3.re, d3.im]

	// Complex multiply: tw64 * d3
	VMOVSLDUP Y8, Y4             // Y4 = [tw.re, tw.re, tw.re, tw.re, ...]
	VMOVSHDUP Y8, Y5             // Y5 = [tw.im, tw.im, tw.im, tw.im, ...]
	VSHUFPS $0xB1, Y3, Y3, Y6    // Y6 = [d3.im, d3.re, d3.im, d3.re, ...]
	VMULPS Y5, Y6, Y6            // Y6 = [d3.im*tw.im, d3.re*tw.im, ...]
	VFMADDSUB231PS Y4, Y3, Y6    // Y6 = tw64 * d3 (complex product)

	// Now we need to construct the output
	// Output should be: [d0+d2, d1+tw*d3, d0-d2, d1-tw*d3]

	// Extract individual components
	VPERMILPS $0x44, Y1, Y7      // Y7 = [d0, d0 | d0, d0] (d0 duplicated)
	VPERMILPS $0xEE, Y1, Y9      // Y9 = [d1, d1 | d1, d1] (d1 duplicated)
	VPERMILPS $0x44, Y2, Y10     // Y10 = [d2, d2 | d2, d2] (d2 duplicated)
	// Y6 already has tw*d3

	// j=0: d0' = d0 + d2, d2' = d0 - d2
	VADDPS Y10, Y7, Y11          // Y11 = d0 + d2
	VSUBPS Y10, Y7, Y12          // Y12 = d0 - d2

	// j=1: d1' = d1 + t, d3' = d1 - t
	VADDPS Y6, Y9, Y13           // Y13 = d1 + tw*d3
	VSUBPS Y6, Y9, Y14           // Y14 = d1 - tw*d3

	// Combine: [d0', d1', d2', d3']
	// d0' is at Y11 low 64 bits, d1' at Y13 low 64 bits
	// d2' is at Y12 low 64 bits, d3' at Y14 low 64 bits
	VBLENDPS $0x0C, Y13, Y11, Y0 // Y0 low lane = [d0', d1'] (blend positions 2-3 from Y13)
	VBLENDPS $0x0C, Y14, Y12, Y1 // Y1 low lane = [d2', d3']
	VINSERTF128 $1, X1, Y0, Y0   // Y0 = [d0', d1', d2', d3']

	VMOVUPS Y0, (R8)(CX*1)

	ADDQ $32, CX                 // Next group of 4
	CMPQ CX, $2048
	JL   size256_r2_stage2_loop

	// =======================================================================
	// STAGE 3: size=8, half=4, step=32
	// =======================================================================
	// Groups of 8: process indices 0-3 with 4-7
	// Twiddles: tw[0], tw[32], tw[64], tw[96]

	VMOVSD (R10), X8             // tw[0]
	VMOVSD 256(R10), X9          // tw[32] ; 32 * 8 = 256
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 512(R10), X9          // tw[64]
	VMOVSD 768(R10), X10         // tw[96] ; 96 * 8 = 768
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw0, tw32, tw64, tw96]

	VMOVSLDUP Y8, Y14            // Pre-split for all iterations
	VMOVSHDUP Y8, Y15

	XORQ CX, CX

size256_r2_stage3_loop:
	// Load group of 8
	VMOVUPS (R8)(CX*1), Y0       // Y0 = indices 0-3
	VMOVUPS 32(R8)(CX*1), Y1     // Y1 = indices 4-7

	// Complex multiply: t = tw * b (Y8 * Y1)
	VSHUFPS $0xB1, Y1, Y1, Y2    // Y2 = b_swapped
	VMULPS Y15, Y2, Y2           // Y2 = b_swap * tw.i
	VFMADDSUB231PS Y14, Y1, Y2   // Y2 = t = tw * b

	// Butterfly
	VADDPS Y2, Y0, Y3            // Y3 = a + t (new indices 0-3)
	VSUBPS Y2, Y0, Y4            // Y4 = a - t (new indices 4-7)

	VMOVUPS Y3, (R8)(CX*1)
	VMOVUPS Y4, 32(R8)(CX*1)

	ADDQ $64, CX                 // Next group of 8 (64 bytes)
	CMPQ CX, $2048
	JL   size256_r2_stage3_loop

	// =======================================================================
	// STAGE 4: size=16, half=8, step=16
	// =======================================================================
	// Groups of 16: process indices 0-7 with 8-15
	// Twiddles: tw[0], tw[16], tw[32], tw[48], tw[64], tw[80], tw[96], tw[112]

	// Load first 4 twiddles into Y8
	VMOVSD (R10), X8             // tw[0]
	VMOVSD 128(R10), X9          // tw[16]
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 256(R10), X9          // tw[32]
	VMOVSD 384(R10), X10         // tw[48]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw0, tw16, tw32, tw48]

	// Load next 4 twiddles into Y9
	VMOVSD 512(R10), X9          // tw[64]
	VMOVSD 640(R10), X10         // tw[80]
	VPUNPCKLQDQ X10, X9, X9
	VMOVSD 768(R10), X10         // tw[96]
	VMOVSD 896(R10), X11         // tw[112]
	VPUNPCKLQDQ X11, X10, X10
	VINSERTF128 $1, X10, Y9, Y9  // Y9 = [tw64, tw80, tw96, tw112]

	XORQ CX, CX

size256_r2_stage4_loop:
	// Load group of 16 (4 YMM registers)
	VMOVUPS (R8)(CX*1), Y0       // Y0 = indices 0-3
	VMOVUPS 32(R8)(CX*1), Y1     // Y1 = indices 4-7
	VMOVUPS 64(R8)(CX*1), Y2     // Y2 = indices 8-11
	VMOVUPS 96(R8)(CX*1), Y3     // Y3 = indices 12-15

	// Complex multiply for indices 8-11 with tw[0,16,32,48]
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y2, Y2, Y4
	VMULPS Y11, Y4, Y4
	VFMADDSUB231PS Y10, Y2, Y4   // Y4 = tw * b (for 0-3 vs 8-11)

	// Complex multiply for indices 12-15 with tw[64,80,96,112]
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y3, Y3, Y5
	VMULPS Y11, Y5, Y5
	VFMADDSUB231PS Y10, Y3, Y5   // Y5 = tw * b (for 4-7 vs 12-15)

	// Butterflies
	VADDPS Y4, Y0, Y6            // Y6 = new 0-3
	VSUBPS Y4, Y0, Y2            // Y2 = new 8-11
	VADDPS Y5, Y1, Y7            // Y7 = new 4-7
	VSUBPS Y5, Y1, Y3            // Y3 = new 12-15

	VMOVUPS Y6, (R8)(CX*1)
	VMOVUPS Y7, 32(R8)(CX*1)
	VMOVUPS Y2, 64(R8)(CX*1)
	VMOVUPS Y3, 96(R8)(CX*1)

	ADDQ $128, CX                // Next group of 16 (128 bytes)
	CMPQ CX, $2048
	JL   size256_r2_stage4_loop

	// =======================================================================
	// STAGE 5: size=32, half=16, step=8
	// =======================================================================
	// Groups of 32: process indices 0-15 with 16-31
	// Twiddles: tw[0], tw[8], tw[16], ..., tw[120]

	XORQ CX, CX

size256_r2_stage5_loop:
	// Process 4 pairs at a time within the group
	XORQ DX, DX                  // DX = j offset

size256_r2_stage5_inner:
	// Load 4 twiddle factors for indices j, j+1, j+2, j+3
	// Each twiddle is at tw[k*8] where k = DX+0, DX+1, DX+2, DX+3
	// Byte offset = k * 8 * 8 = k * 64
	MOVQ DX, AX
	SHLQ $6, AX                  // AX = j * 64 (byte offset for tw[j*8])
	VMOVSD (R10)(AX*1), X8       // tw[(j+0)*8]
	ADDQ $64, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+1)*8]
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $64, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+2)*8]
	ADDQ $64, AX
	VMOVSD (R10)(AX*1), X10      // tw[(j+3)*8]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw[j*8], tw[(j+1)*8], tw[(j+2)*8], tw[(j+3)*8]]

	// Calculate data offsets
	MOVQ CX, AX                  // AX = base offset
	MOVQ DX, BX
	SHLQ $3, BX                  // BX = j * 8 bytes
	ADDQ BX, AX                  // AX = base + j*8

	// Load a (indices j..j+3) and b (indices j+16..j+19)
	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 128(R8)(AX*1), Y1    // 16 * 8 = 128 bytes ahead

	// Complex multiply
	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMADDSUB231PS Y2, Y1, Y4

	// Butterfly
	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 128(R8)(AX*1)

	ADDQ $4, DX                  // j += 4
	CMPQ DX, $16
	JL   size256_r2_stage5_inner

	ADDQ $256, CX                // Next group of 32 (256 bytes)
	CMPQ CX, $2048
	JL   size256_r2_stage5_loop

	// =======================================================================
	// STAGE 6: size=64, half=32, step=4
	// =======================================================================
	XORQ CX, CX

size256_r2_stage6_loop:
	XORQ DX, DX

size256_r2_stage6_inner:
	// Load 4 twiddle factors for indices j, j+1, j+2, j+3
	// Each twiddle is at tw[k*4] where k = DX+0, DX+1, DX+2, DX+3
	// Byte offset = k * 4 * 8 = k * 32
	MOVQ DX, AX
	SHLQ $5, AX                  // AX = j * 32 (byte offset for tw[j*4])
	VMOVSD (R10)(AX*1), X8       // tw[(j+0)*4]
	ADDQ $32, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+1)*4]
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $32, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+2)*4]
	ADDQ $32, AX
	VMOVSD (R10)(AX*1), X10      // tw[(j+3)*4]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw[j*4], tw[(j+1)*4], tw[(j+2)*4], tw[(j+3)*4]]

	MOVQ CX, AX
	MOVQ DX, BX
	SHLQ $3, BX
	ADDQ BX, AX

	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 256(R8)(AX*1), Y1    // 32 * 8 = 256 bytes

	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMADDSUB231PS Y2, Y1, Y4

	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 256(R8)(AX*1)

	ADDQ $4, DX
	CMPQ DX, $32
	JL   size256_r2_stage6_inner

	ADDQ $512, CX                // Next group of 64 (512 bytes)
	CMPQ CX, $2048
	JL   size256_r2_stage6_loop

	// =======================================================================
	// STAGE 7: size=128, half=64, step=2
	// =======================================================================
	XORQ CX, CX

size256_r2_stage7_loop:
	XORQ DX, DX

size256_r2_stage7_inner:
	// Load 4 twiddle factors for indices j, j+1, j+2, j+3
	// Each twiddle is at tw[k*2] where k = DX+0, DX+1, DX+2, DX+3
	// Byte offset = k * 2 * 8 = k * 16
	MOVQ DX, AX
	SHLQ $4, AX                  // AX = j * 16 (byte offset for tw[j*2])
	VMOVSD (R10)(AX*1), X8       // tw[(j+0)*2]
	ADDQ $16, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+1)*2]
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $16, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+2)*2]
	ADDQ $16, AX
	VMOVSD (R10)(AX*1), X10      // tw[(j+3)*2]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw[j*2], tw[(j+1)*2], tw[(j+2)*2], tw[(j+3)*2]]

	MOVQ CX, AX
	MOVQ DX, BX
	SHLQ $3, BX
	ADDQ BX, AX

	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 512(R8)(AX*1), Y1    // 64 * 8 = 512 bytes

	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMADDSUB231PS Y2, Y1, Y4

	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 512(R8)(AX*1)

	ADDQ $4, DX
	CMPQ DX, $64
	JL   size256_r2_stage7_inner

	ADDQ $1024, CX               // Next group of 128 (1024 bytes)
	CMPQ CX, $2048
	JL   size256_r2_stage7_loop

	// =======================================================================
	// STAGE 8: size=256, half=128, step=1
	// =======================================================================
	// Single group: indices 0-127 with 128-255
	// Twiddles: tw[0], tw[1], tw[2], ..., tw[127]

	XORQ DX, DX

size256_r2_stage8_loop:
	// Load 4 consecutive twiddle factors
	VMOVUPS (R10)(DX*8), Y8      // 4 twiddles at once

	MOVQ DX, AX
	SHLQ $3, AX                  // AX = j * 8 bytes

	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 1024(R8)(AX*1), Y1   // 128 * 8 = 1024 bytes

	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMADDSUB231PS Y2, Y1, Y4

	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 1024(R8)(AX*1)

	ADDQ $4, DX
	CMPQ DX, $128
	JL   size256_r2_stage8_loop

	// =======================================================================
	// Copy results to dst if we used scratch buffer
	// =======================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size256_r2_done

	// Copy 2048 bytes from scratch to dst
	XORQ CX, CX

size256_r2_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $2048
	JL   size256_r2_copy_loop

size256_r2_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size256_r2_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 256, complex64
// ===========================================================================
// Same as forward but uses conjugated twiddles via VFMSUBADD instead of VFMADDSUB
//
TEXT ·InverseAVX2Size256Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13
	LEAQ ·bitrev256_r2(SB), R12

	// Verify n == 256
	CMPQ R13, $256
	JNE  size256_r2_inv_return_false

	// Validate slices
	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   size256_r2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   size256_r2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   size256_r2_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size256_r2_inv_use_dst
	MOVQ R11, R8

size256_r2_inv_use_dst:

	// =======================================================================
	// Bit-reversal permutation (radix-2) into work buffer
	// =======================================================================
	XORQ CX, CX

size256_r2_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $256
	JL   size256_r2_inv_bitrev_loop

	// =======================================================================
	// STAGE 1: same as forward (tw[0] = 1+0i, conjugate has no effect)
	// =======================================================================
	XORQ CX, CX

size256_r2_inv_stage1_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1

	// Stage 1 butterfly using correct permutation
	VPERMILPS $0x44, Y0, Y2
	VPERMILPS $0xEE, Y0, Y3
	VADDPS Y3, Y2, Y4
	VSUBPS Y3, Y2, Y5
	VBLENDPS $0xCC, Y5, Y4, Y0

	VPERMILPS $0x44, Y1, Y2
	VPERMILPS $0xEE, Y1, Y3
	VADDPS Y3, Y2, Y4
	VSUBPS Y3, Y2, Y5
	VBLENDPS $0xCC, Y5, Y4, Y1

	VMOVUPS Y0, (R8)(CX*1)
	VMOVUPS Y1, 32(R8)(CX*1)

	ADDQ $64, CX
	CMPQ CX, $2048
	JL   size256_r2_inv_stage1_loop

	// =======================================================================
	// STAGE 2: conjugate multiply via VFMSUBADD
	// =======================================================================
	VMOVSD (R10), X8
	VMOVSD 512(R10), X9
	VPUNPCKLQDQ X9, X8, X8
	VINSERTF128 $1, X8, Y8, Y8

	XORQ CX, CX

size256_r2_inv_stage2_loop:
	VMOVUPS (R8)(CX*1), Y0

	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2

	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMSUBADD231PS Y3, Y2, Y5    // Conjugate multiply

	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0

	VMOVUPS Y0, (R8)(CX*1)

	ADDQ $32, CX
	CMPQ CX, $2048
	JL   size256_r2_inv_stage2_loop

	// =======================================================================
	// STAGE 3: conjugate multiply
	// =======================================================================
	VMOVSD (R10), X8
	VMOVSD 256(R10), X9
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 512(R10), X9
	VMOVSD 768(R10), X10
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8

	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15

	XORQ CX, CX

size256_r2_inv_stage3_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1

	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2

	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4

	VMOVUPS Y3, (R8)(CX*1)
	VMOVUPS Y4, 32(R8)(CX*1)

	ADDQ $64, CX
	CMPQ CX, $2048
	JL   size256_r2_inv_stage3_loop

	// =======================================================================
	// STAGE 4: conjugate multiply
	// =======================================================================
	VMOVSD (R10), X8
	VMOVSD 128(R10), X9
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 256(R10), X9
	VMOVSD 384(R10), X10
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8

	VMOVSD 512(R10), X9
	VMOVSD 640(R10), X10
	VPUNPCKLQDQ X10, X9, X9
	VMOVSD 768(R10), X10
	VMOVSD 896(R10), X11
	VPUNPCKLQDQ X11, X10, X10
	VINSERTF128 $1, X10, Y9, Y9

	XORQ CX, CX

size256_r2_inv_stage4_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS 64(R8)(CX*1), Y2
	VMOVUPS 96(R8)(CX*1), Y3

	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y2, Y2, Y4
	VMULPS Y11, Y4, Y4
	VFMSUBADD231PS Y10, Y2, Y4

	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y3, Y3, Y5
	VMULPS Y11, Y5, Y5
	VFMSUBADD231PS Y10, Y3, Y5

	VADDPS Y4, Y0, Y6
	VSUBPS Y4, Y0, Y2
	VADDPS Y5, Y1, Y7
	VSUBPS Y5, Y1, Y3

	VMOVUPS Y6, (R8)(CX*1)
	VMOVUPS Y7, 32(R8)(CX*1)
	VMOVUPS Y2, 64(R8)(CX*1)
	VMOVUPS Y3, 96(R8)(CX*1)

	ADDQ $128, CX
	CMPQ CX, $2048
	JL   size256_r2_inv_stage4_loop

	// =======================================================================
	// STAGE 5: conjugate multiply
	// =======================================================================
	XORQ CX, CX

size256_r2_inv_stage5_loop:
	XORQ DX, DX

size256_r2_inv_stage5_inner:
	// Load 4 twiddle factors for indices j, j+1, j+2, j+3
	// Each twiddle is at tw[k*8] where k = DX+0, DX+1, DX+2, DX+3
	// Byte offset = k * 8 * 8 = k * 64
	MOVQ DX, AX
	SHLQ $6, AX                  // AX = j * 64 (byte offset for tw[j*8])
	VMOVSD (R10)(AX*1), X8       // tw[(j+0)*8]
	ADDQ $64, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+1)*8]
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $64, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+2)*8]
	ADDQ $64, AX
	VMOVSD (R10)(AX*1), X10      // tw[(j+3)*8]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw[j*8], tw[(j+1)*8], tw[(j+2)*8], tw[(j+3)*8]]

	MOVQ CX, AX
	MOVQ DX, BX
	SHLQ $3, BX
	ADDQ BX, AX

	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 128(R8)(AX*1), Y1

	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMSUBADD231PS Y2, Y1, Y4

	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 128(R8)(AX*1)

	ADDQ $4, DX
	CMPQ DX, $16
	JL   size256_r2_inv_stage5_inner

	ADDQ $256, CX
	CMPQ CX, $2048
	JL   size256_r2_inv_stage5_loop

	// =======================================================================
	// STAGE 6: conjugate multiply
	// =======================================================================
	XORQ CX, CX

size256_r2_inv_stage6_loop:
	XORQ DX, DX

size256_r2_inv_stage6_inner:
	// Load 4 twiddle factors for indices j, j+1, j+2, j+3
	// Each twiddle is at tw[k*4] where k = DX+0, DX+1, DX+2, DX+3
	// Byte offset = k * 4 * 8 = k * 32
	MOVQ DX, AX
	SHLQ $5, AX                  // AX = j * 32 (byte offset for tw[j*4])
	VMOVSD (R10)(AX*1), X8       // tw[(j+0)*4]
	ADDQ $32, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+1)*4]
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $32, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+2)*4]
	ADDQ $32, AX
	VMOVSD (R10)(AX*1), X10      // tw[(j+3)*4]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw[j*4], tw[(j+1)*4], tw[(j+2)*4], tw[(j+3)*4]]

	MOVQ CX, AX
	MOVQ DX, BX
	SHLQ $3, BX
	ADDQ BX, AX

	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 256(R8)(AX*1), Y1

	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMSUBADD231PS Y2, Y1, Y4

	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 256(R8)(AX*1)

	ADDQ $4, DX
	CMPQ DX, $32
	JL   size256_r2_inv_stage6_inner

	ADDQ $512, CX
	CMPQ CX, $2048
	JL   size256_r2_inv_stage6_loop

	// =======================================================================
	// STAGE 7: conjugate multiply
	// =======================================================================
	XORQ CX, CX

size256_r2_inv_stage7_loop:
	XORQ DX, DX

size256_r2_inv_stage7_inner:
	// Load 4 twiddle factors for indices j, j+1, j+2, j+3
	// Each twiddle is at tw[k*2] where k = DX+0, DX+1, DX+2, DX+3
	// Byte offset = k * 2 * 8 = k * 16
	MOVQ DX, AX
	SHLQ $4, AX                  // AX = j * 16 (byte offset for tw[j*2])
	VMOVSD (R10)(AX*1), X8       // tw[(j+0)*2]
	ADDQ $16, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+1)*2]
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $16, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+2)*2]
	ADDQ $16, AX
	VMOVSD (R10)(AX*1), X10      // tw[(j+3)*2]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw[j*2], tw[(j+1)*2], tw[(j+2)*2], tw[(j+3)*2]]

	MOVQ CX, AX
	MOVQ DX, BX
	SHLQ $3, BX
	ADDQ BX, AX

	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 512(R8)(AX*1), Y1

	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMSUBADD231PS Y2, Y1, Y4

	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 512(R8)(AX*1)

	ADDQ $4, DX
	CMPQ DX, $64
	JL   size256_r2_inv_stage7_inner

	ADDQ $1024, CX
	CMPQ CX, $2048
	JL   size256_r2_inv_stage7_loop

	// =======================================================================
	// STAGE 8: conjugate multiply
	// =======================================================================
	XORQ DX, DX

size256_r2_inv_stage8_loop:
	VMOVUPS (R10)(DX*8), Y8

	MOVQ DX, AX
	SHLQ $3, AX

	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 1024(R8)(AX*1), Y1

	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMSUBADD231PS Y2, Y1, Y4

	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 1024(R8)(AX*1)

	ADDQ $4, DX
	CMPQ DX, $128
	JL   size256_r2_inv_stage8_loop

	// =======================================================================
	// Apply 1/N scaling for inverse transform
	// =======================================================================
	// scale = 1/256 = 0.00390625
	// Broadcast scale factor to all 8 float32 positions in Y8
	MOVL ·twoFiftySixth32(SB), AX         // IEEE 754 representation of 1/256 = 0.00390625
	MOVD AX, X8
	VBROADCASTSS X8, Y8          // Y8 = [scale, scale, scale, scale, scale, scale, scale, scale]

	XORQ CX, CX

size256_r2_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y0       // Load 4 complex64 values
	VMOVUPS 32(R8)(CX*1), Y1
	VMULPS Y8, Y0, Y0            // Multiply by scale
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, (R8)(CX*1)
	VMOVUPS Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $2048
	JL   size256_r2_inv_scale_loop

	// =======================================================================
	// Copy results to dst if needed
	// =======================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size256_r2_inv_done

	XORQ CX, CX

size256_r2_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $2048
	JL   size256_r2_inv_copy_loop

size256_r2_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size256_r2_inv_return_false:
	MOVB $0, ret+96(FP)
	RET


DATA ·bitrev256_r2+0(SB)/8, $0
DATA ·bitrev256_r2+8(SB)/8, $128
DATA ·bitrev256_r2+16(SB)/8, $64
DATA ·bitrev256_r2+24(SB)/8, $192
DATA ·bitrev256_r2+32(SB)/8, $32
DATA ·bitrev256_r2+40(SB)/8, $160
DATA ·bitrev256_r2+48(SB)/8, $96
DATA ·bitrev256_r2+56(SB)/8, $224
DATA ·bitrev256_r2+64(SB)/8, $16
DATA ·bitrev256_r2+72(SB)/8, $144
DATA ·bitrev256_r2+80(SB)/8, $80
DATA ·bitrev256_r2+88(SB)/8, $208
DATA ·bitrev256_r2+96(SB)/8, $48
DATA ·bitrev256_r2+104(SB)/8, $176
DATA ·bitrev256_r2+112(SB)/8, $112
DATA ·bitrev256_r2+120(SB)/8, $240
DATA ·bitrev256_r2+128(SB)/8, $8
DATA ·bitrev256_r2+136(SB)/8, $136
DATA ·bitrev256_r2+144(SB)/8, $72
DATA ·bitrev256_r2+152(SB)/8, $200
DATA ·bitrev256_r2+160(SB)/8, $40
DATA ·bitrev256_r2+168(SB)/8, $168
DATA ·bitrev256_r2+176(SB)/8, $104
DATA ·bitrev256_r2+184(SB)/8, $232
DATA ·bitrev256_r2+192(SB)/8, $24
DATA ·bitrev256_r2+200(SB)/8, $152
DATA ·bitrev256_r2+208(SB)/8, $88
DATA ·bitrev256_r2+216(SB)/8, $216
DATA ·bitrev256_r2+224(SB)/8, $56
DATA ·bitrev256_r2+232(SB)/8, $184
DATA ·bitrev256_r2+240(SB)/8, $120
DATA ·bitrev256_r2+248(SB)/8, $248
DATA ·bitrev256_r2+256(SB)/8, $4
DATA ·bitrev256_r2+264(SB)/8, $132
DATA ·bitrev256_r2+272(SB)/8, $68
DATA ·bitrev256_r2+280(SB)/8, $196
DATA ·bitrev256_r2+288(SB)/8, $36
DATA ·bitrev256_r2+296(SB)/8, $164
DATA ·bitrev256_r2+304(SB)/8, $100
DATA ·bitrev256_r2+312(SB)/8, $228
DATA ·bitrev256_r2+320(SB)/8, $20
DATA ·bitrev256_r2+328(SB)/8, $148
DATA ·bitrev256_r2+336(SB)/8, $84
DATA ·bitrev256_r2+344(SB)/8, $212
DATA ·bitrev256_r2+352(SB)/8, $52
DATA ·bitrev256_r2+360(SB)/8, $180
DATA ·bitrev256_r2+368(SB)/8, $116
DATA ·bitrev256_r2+376(SB)/8, $244
DATA ·bitrev256_r2+384(SB)/8, $12
DATA ·bitrev256_r2+392(SB)/8, $140
DATA ·bitrev256_r2+400(SB)/8, $76
DATA ·bitrev256_r2+408(SB)/8, $204
DATA ·bitrev256_r2+416(SB)/8, $44
DATA ·bitrev256_r2+424(SB)/8, $172
DATA ·bitrev256_r2+432(SB)/8, $108
DATA ·bitrev256_r2+440(SB)/8, $236
DATA ·bitrev256_r2+448(SB)/8, $28
DATA ·bitrev256_r2+456(SB)/8, $156
DATA ·bitrev256_r2+464(SB)/8, $92
DATA ·bitrev256_r2+472(SB)/8, $220
DATA ·bitrev256_r2+480(SB)/8, $60
DATA ·bitrev256_r2+488(SB)/8, $188
DATA ·bitrev256_r2+496(SB)/8, $124
DATA ·bitrev256_r2+504(SB)/8, $252
DATA ·bitrev256_r2+512(SB)/8, $2
DATA ·bitrev256_r2+520(SB)/8, $130
DATA ·bitrev256_r2+528(SB)/8, $66
DATA ·bitrev256_r2+536(SB)/8, $194
DATA ·bitrev256_r2+544(SB)/8, $34
DATA ·bitrev256_r2+552(SB)/8, $162
DATA ·bitrev256_r2+560(SB)/8, $98
DATA ·bitrev256_r2+568(SB)/8, $226
DATA ·bitrev256_r2+576(SB)/8, $18
DATA ·bitrev256_r2+584(SB)/8, $146
DATA ·bitrev256_r2+592(SB)/8, $82
DATA ·bitrev256_r2+600(SB)/8, $210
DATA ·bitrev256_r2+608(SB)/8, $50
DATA ·bitrev256_r2+616(SB)/8, $178
DATA ·bitrev256_r2+624(SB)/8, $114
DATA ·bitrev256_r2+632(SB)/8, $242
DATA ·bitrev256_r2+640(SB)/8, $10
DATA ·bitrev256_r2+648(SB)/8, $138
DATA ·bitrev256_r2+656(SB)/8, $74
DATA ·bitrev256_r2+664(SB)/8, $202
DATA ·bitrev256_r2+672(SB)/8, $42
DATA ·bitrev256_r2+680(SB)/8, $170
DATA ·bitrev256_r2+688(SB)/8, $106
DATA ·bitrev256_r2+696(SB)/8, $234
DATA ·bitrev256_r2+704(SB)/8, $26
DATA ·bitrev256_r2+712(SB)/8, $154
DATA ·bitrev256_r2+720(SB)/8, $90
DATA ·bitrev256_r2+728(SB)/8, $218
DATA ·bitrev256_r2+736(SB)/8, $58
DATA ·bitrev256_r2+744(SB)/8, $186
DATA ·bitrev256_r2+752(SB)/8, $122
DATA ·bitrev256_r2+760(SB)/8, $250
DATA ·bitrev256_r2+768(SB)/8, $6
DATA ·bitrev256_r2+776(SB)/8, $134
DATA ·bitrev256_r2+784(SB)/8, $70
DATA ·bitrev256_r2+792(SB)/8, $198
DATA ·bitrev256_r2+800(SB)/8, $38
DATA ·bitrev256_r2+808(SB)/8, $166
DATA ·bitrev256_r2+816(SB)/8, $102
DATA ·bitrev256_r2+824(SB)/8, $230
DATA ·bitrev256_r2+832(SB)/8, $22
DATA ·bitrev256_r2+840(SB)/8, $150
DATA ·bitrev256_r2+848(SB)/8, $86
DATA ·bitrev256_r2+856(SB)/8, $214
DATA ·bitrev256_r2+864(SB)/8, $54
DATA ·bitrev256_r2+872(SB)/8, $182
DATA ·bitrev256_r2+880(SB)/8, $118
DATA ·bitrev256_r2+888(SB)/8, $246
DATA ·bitrev256_r2+896(SB)/8, $14
DATA ·bitrev256_r2+904(SB)/8, $142
DATA ·bitrev256_r2+912(SB)/8, $78
DATA ·bitrev256_r2+920(SB)/8, $206
DATA ·bitrev256_r2+928(SB)/8, $46
DATA ·bitrev256_r2+936(SB)/8, $174
DATA ·bitrev256_r2+944(SB)/8, $110
DATA ·bitrev256_r2+952(SB)/8, $238
DATA ·bitrev256_r2+960(SB)/8, $30
DATA ·bitrev256_r2+968(SB)/8, $158
DATA ·bitrev256_r2+976(SB)/8, $94
DATA ·bitrev256_r2+984(SB)/8, $222
DATA ·bitrev256_r2+992(SB)/8, $62
DATA ·bitrev256_r2+1000(SB)/8, $190
DATA ·bitrev256_r2+1008(SB)/8, $126
DATA ·bitrev256_r2+1016(SB)/8, $254
DATA ·bitrev256_r2+1024(SB)/8, $1
DATA ·bitrev256_r2+1032(SB)/8, $129
DATA ·bitrev256_r2+1040(SB)/8, $65
DATA ·bitrev256_r2+1048(SB)/8, $193
DATA ·bitrev256_r2+1056(SB)/8, $33
DATA ·bitrev256_r2+1064(SB)/8, $161
DATA ·bitrev256_r2+1072(SB)/8, $97
DATA ·bitrev256_r2+1080(SB)/8, $225
DATA ·bitrev256_r2+1088(SB)/8, $17
DATA ·bitrev256_r2+1096(SB)/8, $145
DATA ·bitrev256_r2+1104(SB)/8, $81
DATA ·bitrev256_r2+1112(SB)/8, $209
DATA ·bitrev256_r2+1120(SB)/8, $49
DATA ·bitrev256_r2+1128(SB)/8, $177
DATA ·bitrev256_r2+1136(SB)/8, $113
DATA ·bitrev256_r2+1144(SB)/8, $241
DATA ·bitrev256_r2+1152(SB)/8, $9
DATA ·bitrev256_r2+1160(SB)/8, $137
DATA ·bitrev256_r2+1168(SB)/8, $73
DATA ·bitrev256_r2+1176(SB)/8, $201
DATA ·bitrev256_r2+1184(SB)/8, $41
DATA ·bitrev256_r2+1192(SB)/8, $169
DATA ·bitrev256_r2+1200(SB)/8, $105
DATA ·bitrev256_r2+1208(SB)/8, $233
DATA ·bitrev256_r2+1216(SB)/8, $25
DATA ·bitrev256_r2+1224(SB)/8, $153
DATA ·bitrev256_r2+1232(SB)/8, $89
DATA ·bitrev256_r2+1240(SB)/8, $217
DATA ·bitrev256_r2+1248(SB)/8, $57
DATA ·bitrev256_r2+1256(SB)/8, $185
DATA ·bitrev256_r2+1264(SB)/8, $121
DATA ·bitrev256_r2+1272(SB)/8, $249
DATA ·bitrev256_r2+1280(SB)/8, $5
DATA ·bitrev256_r2+1288(SB)/8, $133
DATA ·bitrev256_r2+1296(SB)/8, $69
DATA ·bitrev256_r2+1304(SB)/8, $197
DATA ·bitrev256_r2+1312(SB)/8, $37
DATA ·bitrev256_r2+1320(SB)/8, $165
DATA ·bitrev256_r2+1328(SB)/8, $101
DATA ·bitrev256_r2+1336(SB)/8, $229
DATA ·bitrev256_r2+1344(SB)/8, $21
DATA ·bitrev256_r2+1352(SB)/8, $149
DATA ·bitrev256_r2+1360(SB)/8, $85
DATA ·bitrev256_r2+1368(SB)/8, $213
DATA ·bitrev256_r2+1376(SB)/8, $53
DATA ·bitrev256_r2+1384(SB)/8, $181
DATA ·bitrev256_r2+1392(SB)/8, $117
DATA ·bitrev256_r2+1400(SB)/8, $245
DATA ·bitrev256_r2+1408(SB)/8, $13
DATA ·bitrev256_r2+1416(SB)/8, $141
DATA ·bitrev256_r2+1424(SB)/8, $77
DATA ·bitrev256_r2+1432(SB)/8, $205
DATA ·bitrev256_r2+1440(SB)/8, $45
DATA ·bitrev256_r2+1448(SB)/8, $173
DATA ·bitrev256_r2+1456(SB)/8, $109
DATA ·bitrev256_r2+1464(SB)/8, $237
DATA ·bitrev256_r2+1472(SB)/8, $29
DATA ·bitrev256_r2+1480(SB)/8, $157
DATA ·bitrev256_r2+1488(SB)/8, $93
DATA ·bitrev256_r2+1496(SB)/8, $221
DATA ·bitrev256_r2+1504(SB)/8, $61
DATA ·bitrev256_r2+1512(SB)/8, $189
DATA ·bitrev256_r2+1520(SB)/8, $125
DATA ·bitrev256_r2+1528(SB)/8, $253
DATA ·bitrev256_r2+1536(SB)/8, $3
DATA ·bitrev256_r2+1544(SB)/8, $131
DATA ·bitrev256_r2+1552(SB)/8, $67
DATA ·bitrev256_r2+1560(SB)/8, $195
DATA ·bitrev256_r2+1568(SB)/8, $35
DATA ·bitrev256_r2+1576(SB)/8, $163
DATA ·bitrev256_r2+1584(SB)/8, $99
DATA ·bitrev256_r2+1592(SB)/8, $227
DATA ·bitrev256_r2+1600(SB)/8, $19
DATA ·bitrev256_r2+1608(SB)/8, $147
DATA ·bitrev256_r2+1616(SB)/8, $83
DATA ·bitrev256_r2+1624(SB)/8, $211
DATA ·bitrev256_r2+1632(SB)/8, $51
DATA ·bitrev256_r2+1640(SB)/8, $179
DATA ·bitrev256_r2+1648(SB)/8, $115
DATA ·bitrev256_r2+1656(SB)/8, $243
DATA ·bitrev256_r2+1664(SB)/8, $11
DATA ·bitrev256_r2+1672(SB)/8, $139
DATA ·bitrev256_r2+1680(SB)/8, $75
DATA ·bitrev256_r2+1688(SB)/8, $203
DATA ·bitrev256_r2+1696(SB)/8, $43
DATA ·bitrev256_r2+1704(SB)/8, $171
DATA ·bitrev256_r2+1712(SB)/8, $107
DATA ·bitrev256_r2+1720(SB)/8, $235
DATA ·bitrev256_r2+1728(SB)/8, $27
DATA ·bitrev256_r2+1736(SB)/8, $155
DATA ·bitrev256_r2+1744(SB)/8, $91
DATA ·bitrev256_r2+1752(SB)/8, $219
DATA ·bitrev256_r2+1760(SB)/8, $59
DATA ·bitrev256_r2+1768(SB)/8, $187
DATA ·bitrev256_r2+1776(SB)/8, $123
DATA ·bitrev256_r2+1784(SB)/8, $251
DATA ·bitrev256_r2+1792(SB)/8, $7
DATA ·bitrev256_r2+1800(SB)/8, $135
DATA ·bitrev256_r2+1808(SB)/8, $71
DATA ·bitrev256_r2+1816(SB)/8, $199
DATA ·bitrev256_r2+1824(SB)/8, $39
DATA ·bitrev256_r2+1832(SB)/8, $167
DATA ·bitrev256_r2+1840(SB)/8, $103
DATA ·bitrev256_r2+1848(SB)/8, $231
DATA ·bitrev256_r2+1856(SB)/8, $23
DATA ·bitrev256_r2+1864(SB)/8, $151
DATA ·bitrev256_r2+1872(SB)/8, $87
DATA ·bitrev256_r2+1880(SB)/8, $215
DATA ·bitrev256_r2+1888(SB)/8, $55
DATA ·bitrev256_r2+1896(SB)/8, $183
DATA ·bitrev256_r2+1904(SB)/8, $119
DATA ·bitrev256_r2+1912(SB)/8, $247
DATA ·bitrev256_r2+1920(SB)/8, $15
DATA ·bitrev256_r2+1928(SB)/8, $143
DATA ·bitrev256_r2+1936(SB)/8, $79
DATA ·bitrev256_r2+1944(SB)/8, $207
DATA ·bitrev256_r2+1952(SB)/8, $47
DATA ·bitrev256_r2+1960(SB)/8, $175
DATA ·bitrev256_r2+1968(SB)/8, $111
DATA ·bitrev256_r2+1976(SB)/8, $239
DATA ·bitrev256_r2+1984(SB)/8, $31
DATA ·bitrev256_r2+1992(SB)/8, $159
DATA ·bitrev256_r2+2000(SB)/8, $95
DATA ·bitrev256_r2+2008(SB)/8, $223
DATA ·bitrev256_r2+2016(SB)/8, $63
DATA ·bitrev256_r2+2024(SB)/8, $191
DATA ·bitrev256_r2+2032(SB)/8, $127
DATA ·bitrev256_r2+2040(SB)/8, $255
GLOBL ·bitrev256_r2(SB), RODATA, $2048
