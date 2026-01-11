//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-512 Radix-2 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 512.
// These kernels provide better performance than the generic implementation by:
//   - Eliminating loop overhead
//   - Using hardcoded twiddle factor indices
//   - Optimal register allocation for this size
//   - Processing 4 complex64 values (32 bytes) per YMM register
//
// Memory layout: 512 complex64 = 4096 bytes = 128 YMM registers worth
// We use a hybrid approach: keep some data in registers, spill to memory as needed.
//
// Algorithm: 9-stage radix-2 DIT FFT
//   Stage 1 (size=2):   256 butterflies, step=256, twiddle[0] for all
//   Stage 2 (size=4):   256 butterflies, step=128, twiddle indices [0,128]
//   Stage 3 (size=8):   256 butterflies, step=64,  twiddle indices [0,64,128,192]
//   Stage 4 (size=16):  256 butterflies, step=32,  twiddle indices [0,32,64,...,224]
//   Stage 5 (size=32):  256 butterflies, step=16,  twiddle indices [0,16,32,...,240]
//   Stage 6 (size=64):  256 butterflies, step=8,   twiddle indices [0,8,16,...,248]
//   Stage 7 (size=128): 256 butterflies, step=4,   twiddle indices [0,4,8,...,252]
//   Stage 8 (size=256): 256 butterflies, step=2,   twiddle indices [0,2,4,...,254]
//   Stage 9 (size=512): 256 butterflies, step=1,   twiddle indices [0,1,2,...,255]
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// Forward transform, size 512, complex64
// ===========================================================================
// Uses looped structure for stages 5-9 to balance code size with performance.
// Stages 1-4 are more unrolled for critical early stages.
//
TEXT ·ForwardAVX2Size512Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 512)
	LEAQ ·bitrev512_r2(SB), R12

	// Verify n == 512
	CMPQ R13, $512
	JNE  size512_r2_return_false

	// Validate all slice lengths >= 512
	MOVQ dst+8(FP), AX
	CMPQ AX, $512
	JL   size512_r2_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $512
	JL   size512_r2_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $512
	JL   size512_r2_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size512_r2_use_dst

	// In-place: use scratch
	MOVQ R11, R8

size512_r2_use_dst:
	// Out-of-place: use dst

	// =======================================================================
	// Bit-reversal permutation (radix-2) into work buffer
	// =======================================================================
	XORQ CX, CX

size512_r2_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $512
	JL   size512_r2_bitrev_loop

	// =======================================================================
	// STAGE 1: size=2, half=1, step=256
	// =======================================================================
	// 256 butterflies with pairs: (0,1), (2,3), (4,5), ..., (510,511)
	// All use twiddle[0] = (1, 0) which is identity multiplication.
	// Process 4 pairs at a time using YMM registers.

	XORQ CX, CX              // CX = base offset in bytes

size512_r2_stage1_loop:
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
	CMPQ CX, $4096           // 512 * 8 bytes = 4096
	JL   size512_r2_stage1_loop

	// =======================================================================
	// STAGE 2: size=4, half=2, step=128
	// =======================================================================
	// For each group of 4: [d0, d1, d2, d3]
	// Butterfly (d0, d2) with tw[0]=1: output d0' = d0+d2, d2' = d0-d2
	// Butterfly (d1, d3) with tw[128]: output d1' = d1+tw*d3, d3' = d1-tw*d3
	// Final layout: [d0', d1', d2', d3']

	// Load twiddle[128] for j=1 butterflies
	VBROADCASTSD 1024(R10), Y8   // Y8 = [tw128, tw128, tw128, tw128] ; 128 * 8 = 1024

	XORQ CX, CX

size512_r2_stage2_loop:
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
	// j=1: tw=tw128, so t = tw128*d3. Result: d1+t, d1-t

	// Since tw[0]=1, the j=0 butterfly is simple addition/subtraction
	// Y1 low lane = [d0, d1], Y2 low lane = [d2, d3]
	// For j=0: use d0 and d2 directly (they're in positions 0-1 of each lane)
	// For j=1: multiply d3 by tw128

	// Actually, let's be more explicit about the data layout
	// Y1 = [d0.re, d0.im, d1.re, d1.im | d0.re, d0.im, d1.re, d1.im]
	// Y2 = [d2.re, d2.im, d3.re, d3.im | d2.re, d2.im, d3.re, d3.im]

	// For j=0 butterfly: need d0 (pos 0) and d2 (pos 0 of Y2)
	// For j=1 butterfly: need d1 (pos 1) and d3 (pos 1 of Y2)

	// Complex multiply tw128 * d3
	// tw128 is in Y8 (broadcasted)
	// d3 is at positions 2-3 in each 128-bit lane of Y2

	// Extract d3 into its own location
	VPERMILPS $0xEE, Y2, Y3      // Y3 = [d3.re, d3.im, d3.re, d3.im | d3.re, d3.im, d3.re, d3.im]

	// Complex multiply: tw128 * d3
	VMOVSLDUP Y8, Y4             // Y4 = [tw.re, tw.re, tw.re, tw.re, ...]
	VMOVSHDUP Y8, Y5             // Y5 = [tw.im, tw.im, tw.im, tw.im, ...]
	VSHUFPS $0xB1, Y3, Y3, Y6    // Y6 = [d3.im, d3.re, d3.im, d3.re, ...]
	VMULPS Y5, Y6, Y6            // Y6 = [d3.im*tw.im, d3.re*tw.im, ...]
	VFMADDSUB231PS Y4, Y3, Y6    // Y6 = tw128 * d3 (complex product)

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
	CMPQ CX, $4096
	JL   size512_r2_stage2_loop

	// =======================================================================
	// STAGE 3: size=8, half=4, step=64
	// =======================================================================
	// Groups of 8: process indices 0-3 with 4-7
	// Twiddles: tw[0], tw[64], tw[128], tw[192]

	VMOVSD (R10), X8             // tw[0]
	VMOVSD 512(R10), X9          // tw[64] ; 64 * 8 = 512
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 1024(R10), X9         // tw[128]
	VMOVSD 1536(R10), X10        // tw[192] ; 192 * 8 = 1536
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw0, tw64, tw128, tw192]

	VMOVSLDUP Y8, Y14            // Pre-split for all iterations
	VMOVSHDUP Y8, Y15

	XORQ CX, CX

size512_r2_stage3_loop:
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
	CMPQ CX, $4096
	JL   size512_r2_stage3_loop

	// =======================================================================
	// STAGE 4: size=16, half=8, step=32
	// =======================================================================
	// Groups of 16: process indices 0-7 with 8-15
	// Twiddles: tw[0], tw[32], tw[64], tw[96], tw[128], tw[160], tw[192], tw[224]

	// Load first 4 twiddles into Y8
	VMOVSD (R10), X8             // tw[0]
	VMOVSD 256(R10), X9          // tw[32] ; 32 * 8 = 256
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 512(R10), X9          // tw[64]
	VMOVSD 768(R10), X10         // tw[96]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw0, tw32, tw64, tw96]

	// Load next 4 twiddles into Y9
	VMOVSD 1024(R10), X9         // tw[128]
	VMOVSD 1280(R10), X10        // tw[160]
	VPUNPCKLQDQ X10, X9, X9
	VMOVSD 1536(R10), X10        // tw[192]
	VMOVSD 1792(R10), X11        // tw[224]
	VPUNPCKLQDQ X11, X10, X10
	VINSERTF128 $1, X10, Y9, Y9  // Y9 = [tw128, tw160, tw192, tw224]

	XORQ CX, CX

size512_r2_stage4_loop:
	// Load group of 16 (4 YMM registers)
	VMOVUPS (R8)(CX*1), Y0       // Y0 = indices 0-3
	VMOVUPS 32(R8)(CX*1), Y1     // Y1 = indices 4-7
	VMOVUPS 64(R8)(CX*1), Y2     // Y2 = indices 8-11
	VMOVUPS 96(R8)(CX*1), Y3     // Y3 = indices 12-15

	// Complex multiply for indices 8-11 with tw[0,32,64,96]
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y2, Y2, Y4
	VMULPS Y11, Y4, Y4
	VFMADDSUB231PS Y10, Y2, Y4   // Y4 = tw * b (for 0-3 vs 8-11)

	// Complex multiply for indices 12-15 with tw[128,160,192,224]
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
	CMPQ CX, $4096
	JL   size512_r2_stage4_loop

	// =======================================================================
	// STAGE 5: size=32, half=16, step=16
	// =======================================================================
	// Groups of 32: process indices 0-15 with 16-31
	// Twiddles: tw[0], tw[16], tw[32], ..., tw[240]

	XORQ CX, CX

size512_r2_stage5_loop:
	// Process 4 pairs at a time within the group
	XORQ DX, DX                  // DX = j offset

size512_r2_stage5_inner:
	// Load 4 twiddle factors for indices j, j+1, j+2, j+3
	// Each twiddle is at tw[k*16] where k = DX+0, DX+1, DX+2, DX+3
	// Byte offset = k * 16 * 8 = k * 128
	MOVQ DX, AX
	SHLQ $7, AX                  // AX = j * 128 (byte offset for tw[j*16])
	VMOVSD (R10)(AX*1), X8       // tw[(j+0)*16]
	ADDQ $128, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+1)*16]
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $128, AX
	VMOVSD (R10)(AX*1), X9       // tw[(j+2)*16]
	ADDQ $128, AX
	VMOVSD (R10)(AX*1), X10      // tw[(j+3)*16]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw[j*16], tw[(j+1)*16], tw[(j+2)*16], tw[(j+3)*16]]

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
	JL   size512_r2_stage5_inner

	ADDQ $256, CX                // Next group of 32 (256 bytes)
	CMPQ CX, $4096
	JL   size512_r2_stage5_loop

	// =======================================================================
	// STAGE 6: size=64, half=32, step=8
	// =======================================================================
	XORQ CX, CX

size512_r2_stage6_loop:
	XORQ DX, DX

size512_r2_stage6_inner:
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
	JL   size512_r2_stage6_inner

	ADDQ $512, CX                // Next group of 64 (512 bytes)
	CMPQ CX, $4096
	JL   size512_r2_stage6_loop

	// =======================================================================
	// STAGE 7: size=128, half=64, step=4
	// =======================================================================
	XORQ CX, CX

size512_r2_stage7_loop:
	XORQ DX, DX

size512_r2_stage7_inner:
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
	JL   size512_r2_stage7_inner

	ADDQ $1024, CX               // Next group of 128 (1024 bytes)
	CMPQ CX, $4096
	JL   size512_r2_stage7_loop

	// =======================================================================
	// STAGE 8: size=256, half=128, step=2
	// =======================================================================
	XORQ CX, CX

size512_r2_stage8_loop:
	XORQ DX, DX

size512_r2_stage8_inner:
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
	JL   size512_r2_stage8_inner

	ADDQ $2048, CX               // Next group of 256 (2048 bytes)
	CMPQ CX, $4096
	JL   size512_r2_stage8_loop

	// =======================================================================
	// STAGE 9: size=512, half=256, step=1
	// =======================================================================
	// Single group: indices 0-255 with 256-511
	// Twiddles: tw[0], tw[1], tw[2], ..., tw[255]

	XORQ DX, DX

size512_r2_stage9_loop:
	// Load 4 consecutive twiddle factors
	VMOVUPS (R10)(DX*8), Y8      // 4 twiddles at once

	MOVQ DX, AX
	SHLQ $3, AX                  // AX = j * 8 bytes

	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 2048(R8)(AX*1), Y1   // 256 * 8 = 2048 bytes

	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMADDSUB231PS Y2, Y1, Y4

	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 2048(R8)(AX*1)

	ADDQ $4, DX
	CMPQ DX, $256
	JL   size512_r2_stage9_loop

	// =======================================================================
	// Copy results to dst if we used scratch buffer
	// =======================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size512_r2_done

	// Copy 4096 bytes from scratch to dst
	XORQ CX, CX

size512_r2_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $4096
	JL   size512_r2_copy_loop

size512_r2_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size512_r2_return_false:
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 512, complex64
// ===========================================================================
// Same as forward but uses conjugated twiddles via VFMSUBADD instead of VFMADDSUB
// and applies 1/N scaling at the end.
//
TEXT ·InverseAVX2Size512Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 512)
	LEAQ ·bitrev512_r2(SB), R12

	// Verify n == 512
	CMPQ R13, $512
	JNE  size512_r2_inv_return_false

	// Validate all slice lengths >= 512
	MOVQ dst+8(FP), AX
	CMPQ AX, $512
	JL   size512_r2_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $512
	JL   size512_r2_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $512
	JL   size512_r2_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size512_r2_inv_use_dst

	// In-place: use scratch
	MOVQ R11, R8

size512_r2_inv_use_dst:
	// Out-of-place: use dst

	// =======================================================================
	// Bit-reversal permutation (radix-2) into work buffer
	// =======================================================================
	XORQ CX, CX

size512_r2_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[i]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[i]]
	MOVQ AX, (R8)(CX*8)      // work[i] = src[bitrev[i]]
	INCQ CX
	CMPQ CX, $512
	JL   size512_r2_inv_bitrev_loop

	// =======================================================================
	// STAGE 1: size=2, half=1, step=256
	// =======================================================================
	// Same as forward (tw[0] = 1+0i, conjugate has no effect)
	XORQ CX, CX

size512_r2_inv_stage1_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1

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
	CMPQ CX, $4096
	JL   size512_r2_inv_stage1_loop

	// =======================================================================
	// STAGE 2: size=4, half=2, step=128 (conjugate multiply via VFMSUBADD)
	// =======================================================================
	VBROADCASTSD 1024(R10), Y8   // Y8 = [tw128, tw128, tw128, tw128]

	XORQ CX, CX

size512_r2_inv_stage2_loop:
	VMOVUPS (R8)(CX*1), Y0

	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2

	VPERMILPS $0xEE, Y2, Y3

	VMOVSLDUP Y8, Y4
	VMOVSHDUP Y8, Y5
	VSHUFPS $0xB1, Y3, Y3, Y6
	VMULPS Y5, Y6, Y6
	VFMSUBADD231PS Y4, Y3, Y6    // Conjugate multiply

	VPERMILPS $0x44, Y1, Y7
	VPERMILPS $0xEE, Y1, Y9
	VPERMILPS $0x44, Y2, Y10

	VADDPS Y10, Y7, Y11
	VSUBPS Y10, Y7, Y12

	VADDPS Y6, Y9, Y13
	VSUBPS Y6, Y9, Y14

	VBLENDPS $0x0C, Y13, Y11, Y0
	VBLENDPS $0x0C, Y14, Y12, Y1
	VINSERTF128 $1, X1, Y0, Y0

	VMOVUPS Y0, (R8)(CX*1)

	ADDQ $32, CX
	CMPQ CX, $4096
	JL   size512_r2_inv_stage2_loop

	// =======================================================================
	// STAGE 3: size=8, half=4, step=64 (conjugate multiply)
	// =======================================================================
	VMOVSD (R10), X8
	VMOVSD 512(R10), X9
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 1024(R10), X9
	VMOVSD 1536(R10), X10
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8

	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15

	XORQ CX, CX

size512_r2_inv_stage3_loop:
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
	CMPQ CX, $4096
	JL   size512_r2_inv_stage3_loop

	// =======================================================================
	// STAGE 4: size=16, half=8, step=32 (conjugate multiply)
	// =======================================================================
	VMOVSD (R10), X8
	VMOVSD 256(R10), X9
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 512(R10), X9
	VMOVSD 768(R10), X10
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8

	VMOVSD 1024(R10), X9
	VMOVSD 1280(R10), X10
	VPUNPCKLQDQ X10, X9, X9
	VMOVSD 1536(R10), X10
	VMOVSD 1792(R10), X11
	VPUNPCKLQDQ X11, X10, X10
	VINSERTF128 $1, X10, Y9, Y9

	XORQ CX, CX

size512_r2_inv_stage4_loop:
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
	CMPQ CX, $4096
	JL   size512_r2_inv_stage4_loop

	// =======================================================================
	// STAGE 5: size=32, half=16, step=16 (conjugate multiply)
	// =======================================================================
	XORQ CX, CX

size512_r2_inv_stage5_loop:
	XORQ DX, DX

size512_r2_inv_stage5_inner:
	MOVQ DX, AX
	SHLQ $7, AX
	VMOVSD (R10)(AX*1), X8
	ADDQ $128, AX
	VMOVSD (R10)(AX*1), X9
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $128, AX
	VMOVSD (R10)(AX*1), X9
	ADDQ $128, AX
	VMOVSD (R10)(AX*1), X10
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8

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
	JL   size512_r2_inv_stage5_inner

	ADDQ $256, CX
	CMPQ CX, $4096
	JL   size512_r2_inv_stage5_loop

	// =======================================================================
	// STAGE 6: size=64, half=32, step=8 (conjugate multiply)
	// =======================================================================
	XORQ CX, CX

size512_r2_inv_stage6_loop:
	XORQ DX, DX

size512_r2_inv_stage6_inner:
	MOVQ DX, AX
	SHLQ $6, AX
	VMOVSD (R10)(AX*1), X8
	ADDQ $64, AX
	VMOVSD (R10)(AX*1), X9
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $64, AX
	VMOVSD (R10)(AX*1), X9
	ADDQ $64, AX
	VMOVSD (R10)(AX*1), X10
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8

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
	JL   size512_r2_inv_stage6_inner

	ADDQ $512, CX
	CMPQ CX, $4096
	JL   size512_r2_inv_stage6_loop

	// =======================================================================
	// STAGE 7: size=128, half=64, step=4 (conjugate multiply)
	// =======================================================================
	XORQ CX, CX

size512_r2_inv_stage7_loop:
	XORQ DX, DX

size512_r2_inv_stage7_inner:
	MOVQ DX, AX
	SHLQ $5, AX
	VMOVSD (R10)(AX*1), X8
	ADDQ $32, AX
	VMOVSD (R10)(AX*1), X9
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $32, AX
	VMOVSD (R10)(AX*1), X9
	ADDQ $32, AX
	VMOVSD (R10)(AX*1), X10
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8

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
	JL   size512_r2_inv_stage7_inner

	ADDQ $1024, CX
	CMPQ CX, $4096
	JL   size512_r2_inv_stage7_loop

	// =======================================================================
	// STAGE 8: size=256, half=128, step=2 (conjugate multiply)
	// =======================================================================
	XORQ CX, CX

size512_r2_inv_stage8_loop:
	XORQ DX, DX

size512_r2_inv_stage8_inner:
	MOVQ DX, AX
	SHLQ $4, AX
	VMOVSD (R10)(AX*1), X8
	ADDQ $16, AX
	VMOVSD (R10)(AX*1), X9
	VPUNPCKLQDQ X9, X8, X8
	ADDQ $16, AX
	VMOVSD (R10)(AX*1), X9
	ADDQ $16, AX
	VMOVSD (R10)(AX*1), X10
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8

	MOVQ CX, AX
	MOVQ DX, BX
	SHLQ $3, BX
	ADDQ BX, AX

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
	JL   size512_r2_inv_stage8_inner

	ADDQ $2048, CX
	CMPQ CX, $4096
	JL   size512_r2_inv_stage8_loop

	// =======================================================================
	// STAGE 9: size=512, half=256, step=1 (conjugate multiply)
	// =======================================================================
	XORQ DX, DX

size512_r2_inv_stage9_loop:
	VMOVUPS (R10)(DX*8), Y8

	MOVQ DX, AX
	SHLQ $3, AX

	VMOVUPS (R8)(AX*1), Y0
	VMOVUPS 2048(R8)(AX*1), Y1

	VMOVSLDUP Y8, Y2
	VMOVSHDUP Y8, Y3
	VSHUFPS $0xB1, Y1, Y1, Y4
	VMULPS Y3, Y4, Y4
	VFMSUBADD231PS Y2, Y1, Y4

	VADDPS Y4, Y0, Y5
	VSUBPS Y4, Y0, Y6

	VMOVUPS Y5, (R8)(AX*1)
	VMOVUPS Y6, 2048(R8)(AX*1)

	ADDQ $4, DX
	CMPQ DX, $256
	JL   size512_r2_inv_stage9_loop

	// =======================================================================
	// Apply 1/N scaling for inverse transform
	// =======================================================================
	// scale = 1/512 = 0.001953125
	MOVL ·fiveHundredTwelfth32(SB), AX
	MOVD AX, X8
	VBROADCASTSS X8, Y8

	XORQ CX, CX

size512_r2_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, (R8)(CX*1)
	VMOVUPS Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $4096
	JL   size512_r2_inv_scale_loop

	// =======================================================================
	// Copy results to dst if we used scratch buffer
	// =======================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size512_r2_inv_done

	XORQ CX, CX

size512_r2_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $4096
	JL   size512_r2_inv_copy_loop

size512_r2_inv_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size512_r2_inv_return_false:
	MOVB $0, ret+96(FP)
	RET
