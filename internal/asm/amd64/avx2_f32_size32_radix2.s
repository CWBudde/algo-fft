//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-32 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 32.
// These kernels provide better performance than the generic implementation by:
//   - Eliminating loop overhead
//   - Using hardcoded twiddle factor indices
//   - Optimal register allocation for this size
//
// See asm_amd64_avx2_generic.s for algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

// ===========================================================================
// SIZE 32 KERNELS
// ===========================================================================
// Forward transform, size 32, complex64
// Fully unrolled 5-stage FFT with AVX2 vectorization
//
// This kernel implements a complete radix-2 DIT FFT for exactly 32 complex64 values.
// All 5 stages are fully unrolled with hardcoded twiddle factor indices:
//   Stage 1 (size=2):  16 butterflies, step=16, twiddle index 0 for all
//   Stage 2 (size=4):  16 butterflies in 4 groups, step=8, twiddle indices [0,8]
//   Stage 3 (size=8):  16 butterflies in 2 groups, step=4, twiddle indices [0,4,8,12]
//   Stage 4 (size=16): 16 butterflies in 1 group, step=2, twiddle indices [0,2,4,6,8,10,12,14]
//   Stage 5 (size=32): 16 butterflies, step=1, twiddle indices [0,1,2,...,15]
//
// Bit-reversal permutation indices for n=32:
//   [0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31]
//
// Register allocation:
//   R8:  work buffer (dst or scratch for in-place)
//   R9:  src pointer
//   R10: twiddle pointer
//   R11: scratch pointer
//   Y0-Y7: data registers for butterflies (32 complex64 = 8 YMM registers)
//   Y8-Y13: twiddle and intermediate values
//
TEXT ·ForwardAVX2Size32Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 32)

	// Verify n == 32
	CMPQ R13, $32            // Check size == 32
	JNE  size32_return_false // Abort if not exactly 32

	// Validate all slice lengths >= 32
	MOVQ dst+8(FP), AX       // Get dst length
	CMPQ AX, $32             // Verify length >= 32
	JL   size32_return_false

	MOVQ twiddle+56(FP), AX  // Get twiddle length
	CMPQ AX, $32
	JL   size32_return_false

	MOVQ scratch+80(FP), AX  // Get scratch length
	CMPQ AX, $32
	JL   size32_return_false

	MOVQ bitrev+104(FP), AX  // Get bitrev length
	CMPQ AX, $32
	JL   size32_return_false

	// Select working buffer
	CMPQ R8, R9              // Compare dst and src pointers
	JNE  size32_use_dst      // If different, use dst for out-of-place transform

	// In-place: use scratch buffer to avoid overwriting input
	MOVQ R11, R8             // R8 = R11 (scratch buffer)
	JMP  size32_bitrev

size32_use_dst:
	// Out-of-place: R8 already points to dst

size32_bitrev:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// For size 32, bitrev = [0,16,8,24,4,20,12,28,2,18,10,26,6,22,14,30,
	//                        1,17,9,25,5,21,13,29,3,19,11,27,7,23,15,31]
	// We use precomputed indices from the bitrev slice for correctness.
	// Unrolled into 8 groups of 4 for efficiency.

	// Group 0: indices 0-3 (offset 0-24 bytes, each complex64 is 8 bytes)
	MOVQ (R12), DX           // DX = bitrev[0] (source index for work[0])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[0]] (load complex64 from source)
	MOVQ AX, (R8)            // work[0] = src[bitrev[0]]

	MOVQ 8(R12), DX          // DX = bitrev[1] (source index for work[1])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[1]]
	MOVQ AX, 8(R8)           // work[1] = src[bitrev[1]]

	MOVQ 16(R12), DX         // DX = bitrev[2] (source index for work[2])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[2]]
	MOVQ AX, 16(R8)          // work[2] = src[bitrev[2]]

	MOVQ 24(R12), DX         // DX = bitrev[3] (source index for work[3])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[3]]
	MOVQ AX, 24(R8)          // work[3] = src[bitrev[3]]

	// Group 1: indices 4-7 (offset 32-56 bytes)
	MOVQ 32(R12), DX         // DX = bitrev[4]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[4]]
	MOVQ AX, 32(R8)          // work[4] = src[bitrev[4]]

	MOVQ 40(R12), DX         // DX = bitrev[5]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[5]]
	MOVQ AX, 40(R8)          // work[5] = src[bitrev[5]]

	MOVQ 48(R12), DX         // DX = bitrev[6]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[6]]
	MOVQ AX, 48(R8)          // work[6] = src[bitrev[6]]

	MOVQ 56(R12), DX         // DX = bitrev[7]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[7]]
	MOVQ AX, 56(R8)          // work[7] = src[bitrev[7]]

	// Group 2: indices 8-11 (offset 64-88 bytes)
	MOVQ 64(R12), DX         // DX = bitrev[8]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[8]]
	MOVQ AX, 64(R8)          // work[8] = src[bitrev[8]]

	MOVQ 72(R12), DX         // DX = bitrev[9]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[9]]
	MOVQ AX, 72(R8)          // work[9] = src[bitrev[9]]

	MOVQ 80(R12), DX         // DX = bitrev[10]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[10]]
	MOVQ AX, 80(R8)          // work[10] = src[bitrev[10]]

	MOVQ 88(R12), DX         // DX = bitrev[11]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[11]]
	MOVQ AX, 88(R8)          // work[11] = src[bitrev[11]]

	// Group 3: indices 12-15 (offset 96-120 bytes)
	MOVQ 96(R12), DX         // DX = bitrev[12]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[12]]
	MOVQ AX, 96(R8)          // work[12] = src[bitrev[12]]

	MOVQ 104(R12), DX        // DX = bitrev[13]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[13]]
	MOVQ AX, 104(R8)         // work[13] = src[bitrev[13]]

	MOVQ 112(R12), DX        // DX = bitrev[14]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[14]]
	MOVQ AX, 112(R8)         // work[14] = src[bitrev[14]]

	MOVQ 120(R12), DX        // DX = bitrev[15]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[15]]
	MOVQ AX, 120(R8)         // work[15] = src[bitrev[15]]

	// Group 4: indices 16-19 (offset 128-152 bytes)
	MOVQ 128(R12), DX        // DX = bitrev[16]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[16]]
	MOVQ AX, 128(R8)         // work[16] = src[bitrev[16]]

	MOVQ 136(R12), DX        // DX = bitrev[17]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[17]]
	MOVQ AX, 136(R8)         // work[17] = src[bitrev[17]]

	MOVQ 144(R12), DX        // DX = bitrev[18]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[18]]
	MOVQ AX, 144(R8)         // work[18] = src[bitrev[18]]

	MOVQ 152(R12), DX        // DX = bitrev[19]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[19]]
	MOVQ AX, 152(R8)         // work[19] = src[bitrev[19]]

	// Group 5: indices 20-23 (offset 160-184 bytes)
	MOVQ 160(R12), DX        // DX = bitrev[20]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[20]]
	MOVQ AX, 160(R8)         // work[20] = src[bitrev[20]]

	MOVQ 168(R12), DX        // DX = bitrev[21]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[21]]
	MOVQ AX, 168(R8)         // work[21] = src[bitrev[21]]

	MOVQ 176(R12), DX        // DX = bitrev[22]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[22]]
	MOVQ AX, 176(R8)         // work[22] = src[bitrev[22]]

	MOVQ 184(R12), DX        // DX = bitrev[23]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[23]]
	MOVQ AX, 184(R8)         // work[23] = src[bitrev[23]]

	// Group 6: indices 24-27 (offset 192-216 bytes)
	MOVQ 192(R12), DX        // DX = bitrev[24]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[24]]
	MOVQ AX, 192(R8)         // work[24] = src[bitrev[24]]

	MOVQ 200(R12), DX        // DX = bitrev[25]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[25]]
	MOVQ AX, 200(R8)         // work[25] = src[bitrev[25]]

	MOVQ 208(R12), DX        // DX = bitrev[26]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[26]]
	MOVQ AX, 208(R8)         // work[26] = src[bitrev[26]]

	MOVQ 216(R12), DX        // DX = bitrev[27]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[27]]
	MOVQ AX, 216(R8)         // work[27] = src[bitrev[27]]

	// Group 7: indices 28-31 (offset 224-248 bytes)
	MOVQ 224(R12), DX        // DX = bitrev[28]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[28]]
	MOVQ AX, 224(R8)         // work[28] = src[bitrev[28]]

	MOVQ 232(R12), DX        // DX = bitrev[29]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[29]]
	MOVQ AX, 232(R8)         // work[29] = src[bitrev[29]]

	MOVQ 240(R12), DX        // DX = bitrev[30]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[30]]
	MOVQ AX, 240(R8)         // work[30] = src[bitrev[30]]

	MOVQ 248(R12), DX        // DX = bitrev[31]
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[31]]
	MOVQ AX, 248(R8)         // work[31] = src[bitrev[31]]

	// =======================================================================
	// STAGE 1: size=2, half=1, step=16
	// =======================================================================
	// 16 independent butterflies with pairs: (0,1), (2,3), (4,5), ..., (30,31)
	// All use twiddle[0] = (1, 0) which is identity multiplication.
	// So: a' = a + b, b' = a - b (no complex multiply needed)

	// Load all 32 complex64 values into 8 YMM registers
	// Y0 = [work[0], work[1], work[2], work[3]]
	// Y1 = [work[4], work[5], work[6], work[7]]
	// ...
	// Y7 = [work[28], work[29], work[30], work[31]]
	VMOVUPS (R8), Y0         // Y0 = [work[0:4]]
	VMOVUPS 32(R8), Y1       // Y1 = [work[4:8]]
	VMOVUPS 64(R8), Y2       // Y2 = [work[8:12]]
	VMOVUPS 96(R8), Y3       // Y3 = [work[12:16]]
	VMOVUPS 128(R8), Y4      // Y4 = [work[16:20]]
	VMOVUPS 160(R8), Y5      // Y5 = [work[20:24]]
	VMOVUPS 192(R8), Y6      // Y6 = [work[24:28]]
	VMOVUPS 224(R8), Y7      // Y7 = [work[28:32]]

	// Stage 1: Butterflies on adjacent pairs within each 128-bit lane
	// For size=2 FFT: out[0] = in[0] + in[1], out[1] = in[0] - in[1]
	// Using twiddle[0] = 1+0i means t = b * 1 = b
	//
	// Y0 = [a0, b0, a1, b1] where a0=work[0], b0=work[1], a1=work[2], b1=work[3]
	// We want: [a0+b0, a0-b0, a1+b1, a1-b1]

	// Y0: [w0, w1, w2, w3] -> pairs (w0,w1), (w2,w3)
	// For size-2 butterfly: out[0] = in[0] + in[1], out[1] = in[0] - in[1]
	VPERMILPD $0x05, Y0, Y8  // Y8 = [w1, w0, w3, w2] (swap within 128-bit lanes)
	VADDPS Y8, Y0, Y9        // Y9 = [w0+w1, w1+w0, w2+w3, w3+w2]
	VSUBPS Y0, Y8, Y10       // Y10 = [w1-w0, w0-w1, w3-w2, w2-w3] (Y8-Y0)
	VBLENDPD $0x0A, Y10, Y9, Y0  // Y0 = [w0+w1, w0-w1, w2+w3, w2-w3]

	// Same for Y1: Butterfly pairs (w4,w5) and (w6,w7)
	VPERMILPD $0x05, Y1, Y8
	VADDPS Y8, Y1, Y9
	VSUBPS Y1, Y8, Y10
	VBLENDPD $0x0A, Y10, Y9, Y1

	// Same for Y2: Butterfly pairs (w8,w9) and (w10,w11)
	VPERMILPD $0x05, Y2, Y8
	VADDPS Y8, Y2, Y9
	VSUBPS Y2, Y8, Y10
	VBLENDPD $0x0A, Y10, Y9, Y2

	// Same for Y3: Butterfly pairs (w12,w13) and (w14,w15)
	VPERMILPD $0x05, Y3, Y8
	VADDPS Y8, Y3, Y9
	VSUBPS Y3, Y8, Y10
	VBLENDPD $0x0A, Y10, Y9, Y3

	// Same for Y4: Butterfly pairs (w16,w17) and (w18,w19)
	VPERMILPD $0x05, Y4, Y8
	VADDPS Y8, Y4, Y9
	VSUBPS Y4, Y8, Y10
	VBLENDPD $0x0A, Y10, Y9, Y4

	// Same for Y5: Butterfly pairs (w20,w21) and (w22,w23)
	VPERMILPD $0x05, Y5, Y8
	VADDPS Y8, Y5, Y9
	VSUBPS Y5, Y8, Y10
	VBLENDPD $0x0A, Y10, Y9, Y5

	// Same for Y6: Butterfly pairs (w24,w25) and (w26,w27)
	VPERMILPD $0x05, Y6, Y8
	VADDPS Y8, Y6, Y9
	VSUBPS Y6, Y8, Y10
	VBLENDPD $0x0A, Y10, Y9, Y6

	// Same for Y7: Butterfly pairs (w28,w29) and (w30,w31)
	VPERMILPD $0x05, Y7, Y8
	VADDPS Y8, Y7, Y9
	VSUBPS Y7, Y8, Y10
	VBLENDPD $0x0A, Y10, Y9, Y7

	// =======================================================================
	// STAGE 2: size=4, half=2, step=8
	// =======================================================================
	// 8 groups of 2 butterflies: (0,2), (1,3), (4,6), (5,7), ...
	// Twiddle factors: j=0 uses twiddle[0], j=1 uses twiddle[8]
	// twiddle[0] = (1, 0), twiddle[8] = (0, -1) for n=32

	// Load twiddle factors for stage 2
	// twiddle[0] = exp(0) = (1, 0)
	// twiddle[8] = exp(-2*pi*i*8/32) = exp(-pi*i/2) = (0, -1)
	VMOVSD (R10), X8         // X8 = twiddle[0] = (1, 0)
	VMOVSD 64(R10), X9       // X9 = twiddle[8] = (0, -1) ; offset = 8 * 8 bytes
	VPUNPCKLQDQ X9, X8, X8   // X8 = [tw0, tw8]
	VINSERTF128 $1, X8, Y8, Y8  // Y8 = [tw0, tw8, tw0, tw8]

	// Y0 = [d0, d1, d2, d3]
	// Extract a = [d0, d1, d0, d1] (low 128 bits duplicated)
	// Extract b = [d2, d3, d2, d3] (high 128 bits duplicated)
	VPERM2F128 $0x00, Y0, Y0, Y9   // Y9 = [d0, d1, d0, d1] (low lane to both)
	VPERM2F128 $0x11, Y0, Y0, Y10  // Y10 = [d2, d3, d2, d3] (high lane to both)

	// Complex multiply: t = w * b (Y8 * Y10)
	VMOVSLDUP Y8, Y11        // Y11 = [w.r, w.r, ...] (broadcast real parts)
	VMOVSHDUP Y8, Y12        // Y12 = [w.i, w.i, ...] (broadcast imag parts)
	VSHUFPS $0xB1, Y10, Y10, Y13  // Y13 = b_swapped = [b.i, b.r, ...]
	VMULPS Y12, Y13, Y13     // Y13 = [b.i*w.i, b.r*w.i, ...]
	VFMADDSUB231PS Y11, Y10, Y13  // Y13 = t = w * b

	// Butterfly: a' = a + t, b' = a - t
	VADDPS Y13, Y9, Y11      // Y11 = a + t
	VSUBPS Y13, Y9, Y12      // Y12 = a - t

	// Recombine: Y0 = [a'0, a'1, b'0, b'1]
	VINSERTF128 $1, X12, Y11, Y0

	// Same for Y1: Apply stage 2 twiddle to Y1 = [d4, d5, d6, d7]
	VPERM2F128 $0x00, Y1, Y1, Y9      // Y9 = [d4, d5, d4, d5] (low lane duplicated)
	VPERM2F128 $0x11, Y1, Y1, Y10     // Y10 = [d6, d7, d6, d7] (high lane duplicated)
	VMOVSLDUP Y8, Y11                 // Y11 = [w.r, w.r, ...] (real parts of twiddles)
	VMOVSHDUP Y8, Y12                 // Y12 = [w.i, w.i, ...] (imag parts of twiddles)
	VSHUFPS $0xB1, Y10, Y10, Y13      // Y13 = b_swapped (swap element pairs)
	VMULPS Y12, Y13, Y13              // Y13 = b_swapped * w.i
	VFMADDSUB231PS Y11, Y10, Y13      // Y13 = t = w * b (complex multiply)
	VADDPS Y13, Y9, Y11               // Y11 = a + t (butterfly add)
	VSUBPS Y13, Y9, Y12               // Y12 = a - t (butterfly subtract)
	VINSERTF128 $1, X12, Y11, Y1      // Y1 = [a'4, a'5, b'4, b'5]

	// Same for Y2: Apply stage 2 twiddle to Y2 = [d8, d9, d10, d11]
	VPERM2F128 $0x00, Y2, Y2, Y9      // Y9 = [d8, d9, d8, d9]
	VPERM2F128 $0x11, Y2, Y2, Y10     // Y10 = [d10, d11, d10, d11]
	VMOVSLDUP Y8, Y11                 // Y11 = real parts broadcast
	VMOVSHDUP Y8, Y12                 // Y12 = imag parts broadcast
	VSHUFPS $0xB1, Y10, Y10, Y13      // Y13 = b_swapped
	VMULPS Y12, Y13, Y13              // Y13 = b_swapped * w.i
	VFMADDSUB231PS Y11, Y10, Y13      // Y13 = t = w * b
	VADDPS Y13, Y9, Y11               // Y11 = a + t
	VSUBPS Y13, Y9, Y12               // Y12 = a - t
	VINSERTF128 $1, X12, Y11, Y2      // Y2 = [a'8, a'9, b'8, b'9]

	// Same for Y3: Apply stage 2 twiddle to Y3 = [d12, d13, d14, d15]
	VPERM2F128 $0x00, Y3, Y3, Y9      // Y9 = [d12, d13, d12, d13]
	VPERM2F128 $0x11, Y3, Y3, Y10     // Y10 = [d14, d15, d14, d15]
	VMOVSLDUP Y8, Y11                 // Y11 = real parts broadcast
	VMOVSHDUP Y8, Y12                 // Y12 = imag parts broadcast
	VSHUFPS $0xB1, Y10, Y10, Y13      // Y13 = b_swapped
	VMULPS Y12, Y13, Y13              // Y13 = b_swapped * w.i
	VFMADDSUB231PS Y11, Y10, Y13      // Y13 = t = w * b
	VADDPS Y13, Y9, Y11               // Y11 = a + t
	VSUBPS Y13, Y9, Y12               // Y12 = a - t
	VINSERTF128 $1, X12, Y11, Y3      // Y3 = [a'12, a'13, b'12, b'13]

	// Same for Y4: Apply stage 2 twiddle to Y4 = [d16, d17, d18, d19]
	VPERM2F128 $0x00, Y4, Y4, Y9      // Y9 = [d16, d17, d16, d17]
	VPERM2F128 $0x11, Y4, Y4, Y10     // Y10 = [d18, d19, d18, d19]
	VMOVSLDUP Y8, Y11                 // Y11 = real parts broadcast
	VMOVSHDUP Y8, Y12                 // Y12 = imag parts broadcast
	VSHUFPS $0xB1, Y10, Y10, Y13      // Y13 = b_swapped
	VMULPS Y12, Y13, Y13              // Y13 = b_swapped * w.i
	VFMADDSUB231PS Y11, Y10, Y13      // Y13 = t = w * b
	VADDPS Y13, Y9, Y11               // Y11 = a + t
	VSUBPS Y13, Y9, Y12               // Y12 = a - t
	VINSERTF128 $1, X12, Y11, Y4      // Y4 = [a'16, a'17, b'16, b'17]

	// Same for Y5: Apply stage 2 twiddle to Y5 = [d20, d21, d22, d23]
	VPERM2F128 $0x00, Y5, Y5, Y9      // Y9 = [d20, d21, d20, d21]
	VPERM2F128 $0x11, Y5, Y5, Y10     // Y10 = [d22, d23, d22, d23]
	VMOVSLDUP Y8, Y11                 // Y11 = real parts broadcast
	VMOVSHDUP Y8, Y12                 // Y12 = imag parts broadcast
	VSHUFPS $0xB1, Y10, Y10, Y13      // Y13 = b_swapped
	VMULPS Y12, Y13, Y13              // Y13 = b_swapped * w.i
	VFMADDSUB231PS Y11, Y10, Y13      // Y13 = t = w * b
	VADDPS Y13, Y9, Y11               // Y11 = a + t
	VSUBPS Y13, Y9, Y12               // Y12 = a - t
	VINSERTF128 $1, X12, Y11, Y5      // Y5 = [a'20, a'21, b'20, b'21]

	// Same for Y6: Apply stage 2 twiddle to Y6 = [d24, d25, d26, d27]
	VPERM2F128 $0x00, Y6, Y6, Y9      // Y9 = [d24, d25, d24, d25]
	VPERM2F128 $0x11, Y6, Y6, Y10     // Y10 = [d26, d27, d26, d27]
	VMOVSLDUP Y8, Y11                 // Y11 = real parts broadcast
	VMOVSHDUP Y8, Y12                 // Y12 = imag parts broadcast
	VSHUFPS $0xB1, Y10, Y10, Y13      // Y13 = b_swapped
	VMULPS Y12, Y13, Y13              // Y13 = b_swapped * w.i
	VFMADDSUB231PS Y11, Y10, Y13      // Y13 = t = w * b
	VADDPS Y13, Y9, Y11               // Y11 = a + t
	VSUBPS Y13, Y9, Y12               // Y12 = a - t
	VINSERTF128 $1, X12, Y11, Y6      // Y6 = [a'24, a'25, b'24, b'25]

	// Same for Y7: Apply stage 2 twiddle to Y7 = [d28, d29, d30, d31]
	VPERM2F128 $0x00, Y7, Y7, Y9      // Y9 = [d28, d29, d28, d29]
	VPERM2F128 $0x11, Y7, Y7, Y10     // Y10 = [d30, d31, d30, d31]
	VMOVSLDUP Y8, Y11                 // Y11 = real parts broadcast
	VMOVSHDUP Y8, Y12                 // Y12 = imag parts broadcast
	VSHUFPS $0xB1, Y10, Y10, Y13      // Y13 = b_swapped
	VMULPS Y12, Y13, Y13              // Y13 = b_swapped * w.i
	VFMADDSUB231PS Y11, Y10, Y13      // Y13 = t = w * b
	VADDPS Y13, Y9, Y11               // Y11 = a + t
	VSUBPS Y13, Y9, Y12               // Y12 = a - t
	VINSERTF128 $1, X12, Y11, Y7      // Y7 = [a'28, a'29, b'28, b'29]

	// =======================================================================
	// STAGE 3: size=8, half=4, step=4
	// =======================================================================
	// 4 groups of 4 butterflies: indices 0-3 with 4-7, indices 8-11 with 12-15, etc.
	// Twiddle factors: twiddle[0], twiddle[4], twiddle[8], twiddle[12]

	// Load twiddle factors for stage 3
	VMOVSD (R10), X8         // twiddle[0]
	VMOVSD 32(R10), X9       // twiddle[4]
	VPUNPCKLQDQ X9, X8, X8   // X8 = [tw0, tw4]
	VMOVSD 64(R10), X9       // twiddle[8]
	VMOVSD 96(R10), X10      // twiddle[12]
	VPUNPCKLQDQ X10, X9, X9  // X9 = [tw8, tw12]
	VINSERTF128 $1, X9, Y8, Y8  // Y8 = [tw0, tw4, tw8, tw12]

	// Group 1: Y0 (indices 0-3) with Y1 (indices 4-7)
	VMOVSLDUP Y8, Y9         // Y9 = [w.r, w.r, ...]
	VMOVSHDUP Y8, Y10        // Y10 = [w.i, w.i, ...]
	VSHUFPS $0xB1, Y1, Y1, Y11  // Y11 = b_swapped
	VMULPS Y10, Y11, Y11     // Y11 = b_swap * w.i
	VFMADDSUB231PS Y9, Y1, Y11  // Y11 = t = w * b

	VADDPS Y11, Y0, Y12      // Y12 = a + t = new indices 0-3
	VSUBPS Y11, Y0, Y13      // Y13 = a - t = new indices 4-7

	// Group 2: Y2 (indices 8-11) with Y3 (indices 12-15)
	VSHUFPS $0xB1, Y3, Y3, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y3, Y11

	VADDPS Y11, Y2, Y0       // Y0 = new indices 8-11
	VSUBPS Y11, Y2, Y1       // Y1 = new indices 12-15

	// Group 3: Y4 (indices 16-19) with Y5 (indices 20-23)
	VSHUFPS $0xB1, Y5, Y5, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y5, Y11

	VADDPS Y11, Y4, Y2       // Y2 = new indices 16-19
	VSUBPS Y11, Y4, Y3       // Y3 = new indices 20-23

	// Group 4: Y6 (indices 24-27) with Y7 (indices 28-31)
	VSHUFPS $0xB1, Y7, Y7, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y7, Y11

	VADDPS Y11, Y6, Y4       // Y4 = new indices 24-27
	VSUBPS Y11, Y6, Y5       // Y5 = new indices 28-31

	// Reorder: move results to Y0-Y7 in order
	// Currently: Y12=0-3, Y13=4-7, Y0=8-11, Y1=12-15, Y2=16-19, Y3=20-23, Y4=24-27, Y5=28-31
	// Use the work buffer to shuffle without clobbering registers.
	VMOVUPS Y0, (R8)         // temp store 8-11
	VMOVUPS Y1, 32(R8)       // temp store 12-15
	VMOVUPS Y2, 64(R8)       // temp store 16-19
	VMOVUPS Y3, 96(R8)       // temp store 20-23
	VMOVUPS Y4, 128(R8)      // temp store 24-27
	VMOVUPS Y5, 160(R8)      // temp store 28-31
	VMOVAPS Y12, Y0          // Y0 = 0-3
	VMOVAPS Y13, Y1          // Y1 = 4-7
	VMOVUPS (R8), Y2         // Y2 = 8-11
	VMOVUPS 32(R8), Y3       // Y3 = 12-15
	VMOVUPS 64(R8), Y4       // Y4 = 16-19
	VMOVUPS 96(R8), Y5       // Y5 = 20-23
	VMOVUPS 128(R8), Y6      // Y6 = 24-27
	VMOVUPS 160(R8), Y7      // Y7 = 28-31

	// =======================================================================
	// STAGE 4: size=16, half=8, step=2 - use non-conjugated twiddles via VFMADDSUB
	// =======================================================================
	// For n=32, Stage 4 implements size-16 sub-FFTs with step=2.
	// Butterfly k pairs index k with k+8, using twiddle[2*k]:
	//   k=0: tw[0],  k=1: tw[2],  k=2: tw[4],  k=3: tw[6]
	//   k=4: tw[8],  k=5: tw[10], k=6: tw[12], k=7: tw[14]
	//
	// Data layout after Stage 3:
	//   Y0 = indices 0-3,   Y1 = indices 4-7
	//   Y2 = indices 8-11,  Y3 = indices 12-15
	//   Y4 = indices 16-19, Y5 = indices 20-23
	//   Y6 = indices 24-27, Y7 = indices 28-31
	//
	// Stage 4 butterflies:
	//   Group 1: (0-3) ↔ (8-11)   using tw[0,2,4,6]
	//   Group 2: (4-7) ↔ (12-15)  using tw[8,10,12,14]  ← KEY: different twiddles!
	//   Group 3: (16-19) ↔ (24-27) using tw[0,2,4,6]
	//   Group 4: (20-23) ↔ (28-31) using tw[8,10,12,14]

	// Load twiddle factors for stage 4:
	// Y8 = [tw0, tw2, tw4, tw6] for groups 1 and 3 (positions 0-3 and 16-19)
	VMOVSD (R10), X8         // twiddle[0]
	VMOVSD 16(R10), X9       // twiddle[2]
	VPUNPCKLQDQ X9, X8, X8   // X8 = [tw0, tw2]
	VMOVSD 32(R10), X9       // twiddle[4]
	VMOVSD 48(R10), X10      // twiddle[6]
	VPUNPCKLQDQ X10, X9, X9  // X9 = [tw4, tw6]
	VINSERTF128 $1, X9, Y8, Y8  // Y8 = [tw0, tw2, tw4, tw6]

	// Group 1: Y0 (indices 0-3) with Y2 (indices 8-11) using Y8 [tw0,tw2,tw4,tw6]
	VMOVSLDUP Y8, Y9         // Y9 = [w.r broadcast]
	VMOVSHDUP Y8, Y10        // Y10 = [w.i broadcast]
	VSHUFPS $0xB1, Y2, Y2, Y11  // Y11 = b_swapped
	VMULPS Y10, Y11, Y11     // Y11 = b_swap * w.i
	VFMADDSUB231PS Y9, Y2, Y11  // Y11 = t = w * b

	VADDPS Y11, Y0, Y12      // Y12 = a' (new indices 0-3)
	VSUBPS Y11, Y0, Y13      // Y13 = b' (new indices 8-11)

	// Load twiddle factors for groups 2 and 4:
	// Y14 = [tw8, tw10, tw12, tw14] for positions 4-7 and 20-23
	VMOVSD 64(R10), X14      // twiddle[8]
	VMOVSD 80(R10), X15      // twiddle[10]
	VPUNPCKLQDQ X15, X14, X14  // X14 = [tw8, tw10]
	VMOVSD 96(R10), X15      // twiddle[12]
	VMOVSD 112(R10), X9      // twiddle[14] - reuse X9
	VPUNPCKLQDQ X9, X15, X15   // X15 = [tw12, tw14]
	VINSERTF128 $1, X15, Y14, Y14  // Y14 = [tw8, tw10, tw12, tw14]

	// Group 2: Y1 (indices 4-7) with Y3 (indices 12-15) using Y14 [tw8,tw10,tw12,tw14]
	VMOVSLDUP Y14, Y9        // Y9 = [w.r broadcast]
	VMOVSHDUP Y14, Y10       // Y10 = [w.i broadcast]
	VSHUFPS $0xB1, Y3, Y3, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y3, Y11

	VADDPS Y11, Y1, Y15      // Y15 = a' (new indices 4-7)
	VSUBPS Y11, Y1, Y11      // Y11 = b' (new indices 12-15) - reuse Y11

	// Save group 1 and 2 results - reorder to Y0-Y3
	VMOVAPS Y12, Y0          // Y0 = 0-3
	VMOVAPS Y15, Y1          // Y1 = 4-7
	VMOVAPS Y13, Y2          // Y2 = 8-11
	VMOVAPS Y11, Y3          // Y3 = 12-15

	// Group 3: Y4 (indices 16-19) with Y6 (indices 24-27) using Y8 [tw0,tw2,tw4,tw6]
	VMOVSLDUP Y8, Y9         // Y9 = [w.r broadcast]
	VMOVSHDUP Y8, Y10        // Y10 = [w.i broadcast]
	VSHUFPS $0xB1, Y6, Y6, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y6, Y11

	VADDPS Y11, Y4, Y12      // Y12 = a' (new indices 16-19)
	VSUBPS Y11, Y4, Y13      // Y13 = b' (new indices 24-27)

	// Group 4: Y5 (indices 20-23) with Y7 (indices 28-31) using Y14 [tw8,tw10,tw12,tw14]
	VMOVSLDUP Y14, Y9        // Y9 = [w.r broadcast]
	VMOVSHDUP Y14, Y10       // Y10 = [w.i broadcast]
	VSHUFPS $0xB1, Y7, Y7, Y11
	VMULPS Y10, Y11, Y11
	VFMADDSUB231PS Y9, Y7, Y11

	VADDPS Y11, Y5, Y15      // Y15 = a' (new indices 20-23)
	VSUBPS Y11, Y5, Y11      // Y11 = b' (new indices 28-31) - reuse Y11

	// Save group 3 and 4 results - reorder to Y4-Y7
	VMOVAPS Y12, Y4          // Y4 = 16-19
	VMOVAPS Y15, Y5          // Y5 = 20-23
	VMOVAPS Y13, Y6          // Y6 = 24-27
	VMOVAPS Y11, Y7          // Y7 = 28-31

	// =======================================================================
	// STAGE 5: size=32, half=16, step=1
	// =======================================================================
	// 16 butterflies: index i with index i+16, for i=0..15
	// Twiddle factors: twiddle[0..15]
	// Need 4 YMM registers for twiddles: [tw0-3], [tw4-7], [tw8-11], [tw12-15]

	// Load twiddle factors for stage 5
	VMOVUPS (R10), Y8        // Y8 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y9      // Y9 = [tw4, tw5, tw6, tw7]
	VMOVUPS 64(R10), Y10     // Y10 = [tw8, tw9, tw10, tw11]
	VMOVUPS 96(R10), Y11     // Y11 = [tw12, tw13, tw14, tw15]

	// Group 1: Y0 (indices 0-3) with Y4 (indices 16-19) using Y8 (tw0-3)
	VMOVSLDUP Y8, Y12        // Y12 = [w.r broadcast]
	VMOVSHDUP Y8, Y13        // Y13 = [w.i broadcast]
	VSHUFPS $0xB1, Y4, Y4, Y14   // Y14 = b_swapped
	VMULPS Y13, Y14, Y14     // Y14 = b_swap * w.i
	VFMADDSUB231PS Y12, Y4, Y14  // Y14 = t = w * b

	// Store to memory since we need all 16 registers
	VMOVUPS Y14, (R8)        // Temp store t0-3

	VADDPS Y14, Y0, Y14      // Y14 = a' (new indices 0-3)
	VMOVUPS (R8), Y15        // Reload t0-3
	VSUBPS Y15, Y0, Y15      // Y15 = b' (new indices 16-19)
	VMOVUPS Y14, (R8)        // Store new 0-3 in dst
	VMOVUPS Y15, 128(R8)     // Store new 16-19 in dst

	// Group 2: Y1 (indices 4-7) with Y5 (indices 20-23) using Y9 (tw4-7)
	VMOVSLDUP Y9, Y12                 // Y12 = real parts of twiddles (tw4, tw5, tw6, tw7)
	VMOVSHDUP Y9, Y13                 // Y13 = imag parts of twiddles
	VSHUFPS $0xB1, Y5, Y5, Y14        // Y14 = Y5 swapped (b_swapped for Y5)
	VMULPS Y13, Y14, Y14              // Y14 = b_swapped * tw.i
	VFMADDSUB231PS Y12, Y5, Y14       // Y14 = t = tw * b (complex multiply)

	VADDPS Y14, Y1, Y15               // Y15 = a + t (butterfly add for Y1)
	VSUBPS Y14, Y1, Y14               // Y14 = a - t (butterfly subtract for Y1)
	VMOVUPS Y15, 32(R8)               // Store a' to indices 4-7
	VMOVUPS Y14, 160(R8)              // Store b' to indices 20-23

	// Group 3: Y2 (indices 8-11) with Y6 (indices 24-27) using Y10 (tw8-11)
	VMOVSLDUP Y10, Y12                // Y12 = real parts of twiddles (tw8, tw9, tw10, tw11)
	VMOVSHDUP Y10, Y13                // Y13 = imag parts of twiddles
	VSHUFPS $0xB1, Y6, Y6, Y14        // Y14 = Y6 swapped (b_swapped for Y6)
	VMULPS Y13, Y14, Y14              // Y14 = b_swapped * tw.i
	VFMADDSUB231PS Y12, Y6, Y14       // Y14 = t = tw * b (complex multiply)

	VADDPS Y14, Y2, Y15               // Y15 = a + t (butterfly add for Y2)
	VSUBPS Y14, Y2, Y14               // Y14 = a - t (butterfly subtract for Y2)
	VMOVUPS Y15, 64(R8)               // Store a' to indices 8-11
	VMOVUPS Y14, 192(R8)              // Store b' to indices 24-27

	// Group 4: Y3 (indices 12-15) with Y7 (indices 28-31) using Y11 (tw12-15)
	VMOVSLDUP Y11, Y12                // Y12 = real parts of twiddles (tw12, tw13, tw14, tw15)
	VMOVSHDUP Y11, Y13                // Y13 = imag parts of twiddles
	VSHUFPS $0xB1, Y7, Y7, Y14        // Y14 = Y7 swapped (b_swapped for Y7)
	VMULPS Y13, Y14, Y14              // Y14 = b_swapped * tw.i
	VFMADDSUB231PS Y12, Y7, Y14       // Y14 = t = tw * b (complex multiply)

	VADDPS Y14, Y3, Y15               // Y15 = a + t (butterfly add for Y3)
	VSUBPS Y14, Y3, Y14               // Y14 = a - t (butterfly subtract for Y3)
	VMOVUPS Y15, 96(R8)               // Store a' to indices 12-15
	VMOVUPS Y14, 224(R8)              // Store b' to indices 28-31

	// =======================================================================
	// Copy results to dst if we used scratch buffer
	// =======================================================================
	MOVQ dst+0(FP), R9       // R9 = dst pointer
	CMPQ R8, R9              // Compare working buffer and dst
	JE   size32_done         // If same, already in dst

	// Copy from scratch to dst (8 registers = 256 bytes total)
	VMOVUPS (R8), Y0         // Load results from scratch[0:32] (elements 0-3)
	VMOVUPS 32(R8), Y1       // Load scratch[32:64] (elements 4-7)
	VMOVUPS 64(R8), Y2       // Load scratch[64:96] (elements 8-11)
	VMOVUPS 96(R8), Y3       // Load scratch[96:128] (elements 12-15)
	VMOVUPS 128(R8), Y4      // Load scratch[128:160] (elements 16-19)
	VMOVUPS 160(R8), Y5      // Load scratch[160:192] (elements 20-23)
	VMOVUPS 192(R8), Y6      // Load scratch[192:224] (elements 24-27)
	VMOVUPS 224(R8), Y7      // Load scratch[224:256] (elements 28-31)
	// Write to dst
	VMOVUPS Y0, (R9)         // Store results to dst[0:32] (elements 0-3)
	VMOVUPS Y1, 32(R9)       // Store to dst[32:64] (elements 4-7)
	VMOVUPS Y2, 64(R9)       // Store to dst[64:96] (elements 8-11)
	VMOVUPS Y3, 96(R9)       // Store to dst[96:128] (elements 12-15)
	VMOVUPS Y4, 128(R9)      // Store to dst[128:160] (elements 16-19)
	VMOVUPS Y5, 160(R9)      // Store to dst[160:192] (elements 20-23)
	VMOVUPS Y6, 192(R9)      // Store to dst[192:224] (elements 24-27)
	VMOVUPS Y7, 224(R9)      // Store to dst[224:256] (elements 28-31)

size32_done:
	VZEROUPPER               // Clear SIMD state
	MOVB $1, ret+120(FP)     // Return true (success)
	RET

size32_return_false:
	MOVB $0, ret+120(FP)     // Return false (validation failed)
	RET

// ===========================================================================
// Inverse transform, size 32, complex64
// ===========================================================================
// Fully unrolled 5-stage inverse FFT with AVX2 vectorization
//
// This kernel implements a complete radix-2 DIT inverse FFT for exactly 32 complex64 values.
// The only difference from forward transform is using conjugated twiddle factors,
// achieved by using VFMSUBADD231PS instead of VFMADDSUB231PS for complex multiplication.
//
TEXT ·InverseAVX2Size32Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer (conjugated for inverse)
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 32)

	// Verify n == 32
	CMPQ R13, $32            // Check size == 32
	JNE  size32_inv_return_false

	// Validate all slice lengths >= 32
	MOVQ dst+8(FP), AX       // Get dst length
	CMPQ AX, $32
	JL   size32_inv_return_false

	MOVQ twiddle+56(FP), AX  // Get twiddle length
	CMPQ AX, $32
	JL   size32_inv_return_false

	MOVQ scratch+80(FP), AX  // Get scratch length
	CMPQ AX, $32
	JL   size32_inv_return_false

	MOVQ bitrev+104(FP), AX  // Get bitrev length
	CMPQ AX, $32
	JL   size32_inv_return_false

	// Select working buffer
	CMPQ R8, R9              // Compare dst and src pointers
	JNE  size32_inv_use_dst  // If different, use dst for out-of-place

	// In-place: use scratch buffer to avoid overwriting input
	MOVQ R11, R8             // R8 = scratch buffer
	JMP  size32_inv_bitrev

size32_inv_use_dst:
	// Out-of-place: R8 already points to dst

size32_inv_bitrev:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// R12 points to bitrev indices array, R9 points to src, R8 points to work
	// Each bitrev[i] is an 8-byte offset (complex64 is 8 bytes)
	// We load 32 elements total in 8 groups of 4
	// =======================================================================
	// Group 0: indices 0-3 (offset 0-24 bytes from R12, work offsets 0-24 from R8)
	MOVQ (R12), DX           // DX = bitrev[0] (source index for work[0])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[0]] (load complex64 from source)
	MOVQ AX, (R8)            // work[0] = src[bitrev[0]]

	MOVQ 8(R12), DX          // DX = bitrev[1] (source index for work[1])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[1]]
	MOVQ AX, 8(R8)           // work[1] = src[bitrev[1]]

	MOVQ 16(R12), DX         // DX = bitrev[2] (source index for work[2])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[2]]
	MOVQ AX, 16(R8)          // work[2] = src[bitrev[2]]

	MOVQ 24(R12), DX         // DX = bitrev[3] (source index for work[3])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[3]]
	MOVQ AX, 24(R8)          // work[3] = src[bitrev[3]]

	// Group 1: indices 4-7 (offset 32-56 bytes from R12, work offsets 32-56 from R8)
	MOVQ 32(R12), DX         // DX = bitrev[4] (source index for work[4])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[4]]
	MOVQ AX, 32(R8)          // work[4] = src[bitrev[4]]

	MOVQ 40(R12), DX         // DX = bitrev[5] (source index for work[5])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[5]]
	MOVQ AX, 40(R8)          // work[5] = src[bitrev[5]]

	MOVQ 48(R12), DX         // DX = bitrev[6] (source index for work[6])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[6]]
	MOVQ AX, 48(R8)          // work[6] = src[bitrev[6]]

	MOVQ 56(R12), DX         // DX = bitrev[7] (source index for work[7])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[7]]
	MOVQ AX, 56(R8)          // work[7] = src[bitrev[7]]

	// Group 2: indices 8-11 (offset 64-88 bytes from R12, work offsets 64-88 from R8)
	MOVQ 64(R12), DX         // DX = bitrev[8] (source index for work[8])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[8]]
	MOVQ AX, 64(R8)          // work[8] = src[bitrev[8]]

	MOVQ 72(R12), DX         // DX = bitrev[9] (source index for work[9])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[9]]
	MOVQ AX, 72(R8)          // work[9] = src[bitrev[9]]

	MOVQ 80(R12), DX         // DX = bitrev[10] (source index for work[10])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[10]]
	MOVQ AX, 80(R8)          // work[10] = src[bitrev[10]]

	MOVQ 88(R12), DX         // DX = bitrev[11] (source index for work[11])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[11]]
	MOVQ AX, 88(R8)          // work[11] = src[bitrev[11]]

	// Group 3: indices 12-15 (offset 96-120 bytes from R12, work offsets 96-120 from R8)
	MOVQ 96(R12), DX         // DX = bitrev[12] (source index for work[12])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[12]]
	MOVQ AX, 96(R8)          // work[12] = src[bitrev[12]]

	MOVQ 104(R12), DX        // DX = bitrev[13] (source index for work[13])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[13]]
	MOVQ AX, 104(R8)         // work[13] = src[bitrev[13]]

	MOVQ 112(R12), DX        // DX = bitrev[14] (source index for work[14])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[14]]
	MOVQ AX, 112(R8)         // work[14] = src[bitrev[14]]

	MOVQ 120(R12), DX        // DX = bitrev[15] (source index for work[15])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[15]]
	MOVQ AX, 120(R8)         // work[15] = src[bitrev[15]]

	// Group 4: indices 16-19 (offset 128-152 bytes from R12, work offsets 128-152 from R8)
	MOVQ 128(R12), DX        // DX = bitrev[16] (source index for work[16])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[16]]
	MOVQ AX, 128(R8)         // work[16] = src[bitrev[16]]

	MOVQ 136(R12), DX        // DX = bitrev[17] (source index for work[17])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[17]]
	MOVQ AX, 136(R8)         // work[17] = src[bitrev[17]]

	MOVQ 144(R12), DX        // DX = bitrev[18] (source index for work[18])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[18]]
	MOVQ AX, 144(R8)         // work[18] = src[bitrev[18]]

	MOVQ 152(R12), DX        // DX = bitrev[19] (source index for work[19])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[19]]
	MOVQ AX, 152(R8)         // work[19] = src[bitrev[19]]

	// Group 5: indices 20-23 (offset 160-184 bytes from R12, work offsets 160-184 from R8)
	MOVQ 160(R12), DX        // DX = bitrev[20] (source index for work[20])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[20]]
	MOVQ AX, 160(R8)         // work[20] = src[bitrev[20]]

	MOVQ 168(R12), DX        // DX = bitrev[21] (source index for work[21])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[21]]
	MOVQ AX, 168(R8)         // work[21] = src[bitrev[21]]

	MOVQ 176(R12), DX        // DX = bitrev[22] (source index for work[22])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[22]]
	MOVQ AX, 176(R8)         // work[22] = src[bitrev[22]]

	MOVQ 184(R12), DX        // DX = bitrev[23] (source index for work[23])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[23]]
	MOVQ AX, 184(R8)         // work[23] = src[bitrev[23]]

	// Group 6: indices 24-27 (offset 192-216 bytes from R12, work offsets 192-216 from R8)
	MOVQ 192(R12), DX        // DX = bitrev[24] (source index for work[24])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[24]]
	MOVQ AX, 192(R8)         // work[24] = src[bitrev[24]]

	MOVQ 200(R12), DX        // DX = bitrev[25] (source index for work[25])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[25]]
	MOVQ AX, 200(R8)         // work[25] = src[bitrev[25]]

	MOVQ 208(R12), DX        // DX = bitrev[26] (source index for work[26])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[26]]
	MOVQ AX, 208(R8)         // work[26] = src[bitrev[26]]

	MOVQ 216(R12), DX        // DX = bitrev[27] (source index for work[27])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[27]]
	MOVQ AX, 216(R8)         // work[27] = src[bitrev[27]]

	// Group 7: indices 28-31 (offset 224-248 bytes from R12, work offsets 224-248 from R8)
	MOVQ 224(R12), DX        // DX = bitrev[28] (source index for work[28])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[28]]
	MOVQ AX, 224(R8)         // work[28] = src[bitrev[28]]

	MOVQ 232(R12), DX        // DX = bitrev[29] (source index for work[29])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[29]]
	MOVQ AX, 232(R8)         // work[29] = src[bitrev[29]]

	MOVQ 240(R12), DX        // DX = bitrev[30] (source index for work[30])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[30]]
	MOVQ AX, 240(R8)         // work[30] = src[bitrev[30]]

	MOVQ 248(R12), DX        // DX = bitrev[31] (source index for work[31])
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[31]]
	MOVQ AX, 248(R8)         // work[31] = src[bitrev[31]]

	// =======================================================================
	// STAGE 1: size=2, half=1, step=16 (same as forward - tw[0]=1+0i)
	// =======================================================================
	// Conjugation has no effect on identity twiddle

	VMOVUPS (R8), Y0          // Load elements 0-3 (work[0-3]) into Y0
	VMOVUPS 32(R8), Y1        // Load elements 4-7 (work[4-7]) into Y1
	VMOVUPS 64(R8), Y2        // Load elements 8-11 (work[8-11]) into Y2
	VMOVUPS 96(R8), Y3        // Load elements 12-15 (work[12-15]) into Y3
	VMOVUPS 128(R8), Y4       // Load elements 16-19 (work[16-19]) into Y4
	VMOVUPS 160(R8), Y5       // Load elements 20-23 (work[20-23]) into Y5
	VMOVUPS 192(R8), Y6       // Load elements 24-27 (work[24-27]) into Y6
	VMOVUPS 224(R8), Y7       // Load elements 28-31 (work[28-31]) into Y7

	// Process 16 pairs using identity twiddle (w[0] = 1+0i)
	// For size-2 butterfly: out[0] = in[0] + in[1], out[1] = in[0] - in[1]
	// Y0: pairs (w0,w1), (w2,w3)
	VPERMILPD $0x05, Y0, Y8  // Swap elements within 64-bit lanes: (a.r,a.i,b.r,b.i) -> (a.i,a.r,b.i,b.r)
	VADDPS Y8, Y0, Y9        // Y9 = Y0 + Y8 (element-wise add: butterfly add)
	VSUBPS Y0, Y8, Y10       // Y8-Y0, not Y0-Y8! (element-wise sub: butterfly subtract)
	VBLENDPD $0x0A, Y10, Y9, Y0  // 64-bit blend: select alternating 64-bit values to recombine butterfly results

	// Y1: pairs (w4,w5), (w6,w7)
	VPERMILPD $0x05, Y1, Y8  // Swap elements within 64-bit lanes
	VADDPS Y8, Y1, Y9        // Butterfly add: (Y1 + swapped Y1)
	VSUBPS Y1, Y8, Y10       // Butterfly subtract: (swapped Y1 - Y1)
	VBLENDPD $0x0A, Y10, Y9, Y1  // Blend to recombine butterfly results

	// Y2: pairs (w8,w9), (w10,w11)
	VPERMILPD $0x05, Y2, Y8  // Swap elements within 64-bit lanes
	VADDPS Y8, Y2, Y9        // Butterfly add
	VSUBPS Y2, Y8, Y10       // Butterfly subtract
	VBLENDPD $0x0A, Y10, Y9, Y2  // Blend to recombine results

	// Y3: pairs (w12,w13), (w14,w15)
	VPERMILPD $0x05, Y3, Y8  // Swap elements within 64-bit lanes
	VADDPS Y8, Y3, Y9        // Butterfly add
	VSUBPS Y3, Y8, Y10       // Butterfly subtract
	VBLENDPD $0x0A, Y10, Y9, Y3  // Blend to recombine results

	// Y4: pairs (w16,w17), (w18,w19)
	VPERMILPD $0x05, Y4, Y8  // Swap elements within 64-bit lanes
	VADDPS Y8, Y4, Y9        // Butterfly add
	VSUBPS Y4, Y8, Y10       // Butterfly subtract
	VBLENDPD $0x0A, Y10, Y9, Y4  // Blend to recombine results

	// Y5: pairs (w20,w21), (w22,w23)
	VPERMILPD $0x05, Y5, Y8  // Swap elements within 64-bit lanes
	VADDPS Y8, Y5, Y9        // Butterfly add
	VSUBPS Y5, Y8, Y10       // Butterfly subtract
	VBLENDPD $0x0A, Y10, Y9, Y5  // Blend to recombine results

	// Y6: pairs (w24,w25), (w26,w27)
	VPERMILPD $0x05, Y6, Y8  // Swap elements within 64-bit lanes
	VADDPS Y8, Y6, Y9        // Butterfly add
	VSUBPS Y6, Y8, Y10       // Butterfly subtract
	VBLENDPD $0x0A, Y10, Y9, Y6  // Blend to recombine results

	// Y7: pairs (w28,w29), (w30,w31)
	VPERMILPD $0x05, Y7, Y8  // Swap elements within 64-bit lanes
	VADDPS Y8, Y7, Y9        // Butterfly add
	VSUBPS Y7, Y8, Y10       // Butterfly subtract
	VBLENDPD $0x0A, Y10, Y9, Y7  // Blend to recombine results

	// =======================================================================
	// STAGE 2: size=4 - use conjugated twiddles via VFMSUBADD
	// =======================================================================
	// VFMSUBADD gives: even=a*b+c, odd=a*b-c -> conjugate multiply result

	// Load twiddle factors for stage 2
	VMOVSD (R10), X8         // twiddle[0]: Load tw0 (real+imag pair) into low 64 bits of X8
	VMOVSD 64(R10), X9       // twiddle[8]: Load tw8 (real+imag pair) into low 64 bits of X9
	VPUNPCKLQDQ X9, X8, X8   // Unpack: X8 = [tw8.r, tw8.i, tw0.r, tw0.i]
	VINSERTF128 $1, X8, Y8, Y8   // Y8 = [tw0, tw8, tw0, tw8] (replicate pattern across 256-bit register)

	// Pre-split twiddle (reused for all 8 registers)
	VMOVSLDUP Y8, Y14        // Y14 = [w.r, w.r, ...] (broadcast real parts to all lanes)
	VMOVSHDUP Y8, Y15        // Y15 = [w.i, w.i, ...] (broadcast imaginary parts to all lanes)

	// Y0: Process elements 0-3 (d0-d3)
	VPERM2F128 $0x00, Y0, Y0, Y9  // Y9 = low 128-bit lanes (d0-d1)
	VPERM2F128 $0x11, Y0, Y0, Y10 // Y10 = high 128-bit lanes (d2-d3)
	VSHUFPS $0xB1, Y10, Y10, Y11  // Y11: swap elements in Y10 for complex multiply (d2.i,d2.r,d3.i,d3.r)
	VMULPS Y15, Y11, Y11          // Y11 *= imaginary parts (Y15 = [w.i, w.i, ...])
	VFMSUBADD231PS Y14, Y10, Y11  // Conjugate multiply: Y11 = Y10*Y14 ± Y11 (complex conjugate multiply)
	VADDPS Y11, Y9, Y12           // Y12 = Y9 + Y11 (butterfly add: upper + twiddle*lower)
	VSUBPS Y11, Y9, Y13           // Y13 = Y9 - Y11 (butterfly subtract: upper - twiddle*lower)
	VINSERTF128 $1, X13, Y12, Y0  // Y0 = [Y12_low128, Y13_low128] (recombine butterfly results)

	// Y1: Process elements 4-7 (d4-d7)
	VPERM2F128 $0x00, Y1, Y1, Y9  // Y9 = low 128-bit lanes (d4-d5)
	VPERM2F128 $0x11, Y1, Y1, Y10 // Y10 = high 128-bit lanes (d6-d7)
	VSHUFPS $0xB1, Y10, Y10, Y11  // Swap elements in Y10 for complex multiply
	VMULPS Y15, Y11, Y11          // Multiply by imaginary parts
	VFMSUBADD231PS Y14, Y10, Y11  // Conjugate multiply with real parts
	VADDPS Y11, Y9, Y12           // Butterfly add
	VSUBPS Y11, Y9, Y13           // Butterfly subtract
	VINSERTF128 $1, X13, Y12, Y1  // Recombine butterfly results

	// Y2: Process elements 8-11 (d8-d11)
	VPERM2F128 $0x00, Y2, Y2, Y9  // Y9 = low 128-bit lanes
	VPERM2F128 $0x11, Y2, Y2, Y10 // Y10 = high 128-bit lanes
	VSHUFPS $0xB1, Y10, Y10, Y11  // Swap elements for complex multiply
	VMULPS Y15, Y11, Y11          // Multiply by imaginary parts
	VFMSUBADD231PS Y14, Y10, Y11  // Conjugate multiply
	VADDPS Y11, Y9, Y12           // Butterfly add
	VSUBPS Y11, Y9, Y13           // Butterfly subtract
	VINSERTF128 $1, X13, Y12, Y2  // Recombine results

	// Y3: Process elements 12-15 (d12-d15)
	VPERM2F128 $0x00, Y3, Y3, Y9  // Y9 = low 128-bit lanes
	VPERM2F128 $0x11, Y3, Y3, Y10 // Y10 = high 128-bit lanes
	VSHUFPS $0xB1, Y10, Y10, Y11  // Swap elements for complex multiply
	VMULPS Y15, Y11, Y11          // Multiply by imaginary parts
	VFMSUBADD231PS Y14, Y10, Y11  // Conjugate multiply
	VADDPS Y11, Y9, Y12           // Butterfly add
	VSUBPS Y11, Y9, Y13           // Butterfly subtract
	VINSERTF128 $1, X13, Y12, Y3  // Recombine results

	// Y4: Process elements 16-19 (d16-d19)
	VPERM2F128 $0x00, Y4, Y4, Y9  // Y9 = low 128-bit lanes
	VPERM2F128 $0x11, Y4, Y4, Y10 // Y10 = high 128-bit lanes
	VSHUFPS $0xB1, Y10, Y10, Y11  // Swap elements for complex multiply
	VMULPS Y15, Y11, Y11          // Multiply by imaginary parts
	VFMSUBADD231PS Y14, Y10, Y11  // Conjugate multiply
	VADDPS Y11, Y9, Y12           // Butterfly add
	VSUBPS Y11, Y9, Y13           // Butterfly subtract
	VINSERTF128 $1, X13, Y12, Y4  // Recombine results

	// Y5: Process elements 20-23 (d20-d23)
	VPERM2F128 $0x00, Y5, Y5, Y9  // Y9 = low 128-bit lanes
	VPERM2F128 $0x11, Y5, Y5, Y10 // Y10 = high 128-bit lanes
	VSHUFPS $0xB1, Y10, Y10, Y11  // Swap elements for complex multiply
	VMULPS Y15, Y11, Y11          // Multiply by imaginary parts
	VFMSUBADD231PS Y14, Y10, Y11  // Conjugate multiply
	VADDPS Y11, Y9, Y12           // Butterfly add
	VSUBPS Y11, Y9, Y13           // Butterfly subtract
	VINSERTF128 $1, X13, Y12, Y5  // Recombine results

	// Y6: Process elements 24-27 (d24-d27)
	VPERM2F128 $0x00, Y6, Y6, Y9  // Y9 = low 128-bit lanes
	VPERM2F128 $0x11, Y6, Y6, Y10 // Y10 = high 128-bit lanes
	VSHUFPS $0xB1, Y10, Y10, Y11  // Swap elements for complex multiply
	VMULPS Y15, Y11, Y11          // Multiply by imaginary parts
	VFMSUBADD231PS Y14, Y10, Y11  // Conjugate multiply
	VADDPS Y11, Y9, Y12           // Butterfly add
	VSUBPS Y11, Y9, Y13           // Butterfly subtract
	VINSERTF128 $1, X13, Y12, Y6  // Recombine results

	// Y7: Process elements 28-31 (d28-d31)
	VPERM2F128 $0x00, Y7, Y7, Y9  // Y9 = low 128-bit lanes
	VPERM2F128 $0x11, Y7, Y7, Y10 // Y10 = high 128-bit lanes
	VSHUFPS $0xB1, Y10, Y10, Y11  // Swap elements for complex multiply
	VMULPS Y15, Y11, Y11          // Multiply by imaginary parts
	VFMSUBADD231PS Y14, Y10, Y11  // Conjugate multiply
	VADDPS Y11, Y9, Y12           // Butterfly add
	VSUBPS Y11, Y9, Y13           // Butterfly subtract
	VINSERTF128 $1, X13, Y12, Y7  // Recombine results

	// =======================================================================
	// STAGE 3: size=8 - use conjugated twiddles via VFMSUBADD
	// =======================================================================

	// Load twiddle factors for stage 3
	VMOVSD (R10), X8         // twiddle[0]
	VMOVSD 32(R10), X9       // twiddle[4]
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 64(R10), X9       // twiddle[8]
	VMOVSD 96(R10), X10      // twiddle[12]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8   // Y8 = [tw0, tw4, tw8, tw12]

	// Pre-split twiddle
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15

	// Group 1: Y0 with Y1 using conjugated twiddles
	VSHUFPS $0xB1, Y1, Y1, Y9
	VMULPS Y15, Y9, Y9
	VFMSUBADD231PS Y14, Y1, Y9  // Conjugate multiply

	VADDPS Y9, Y0, Y10       // Y10 = new indices 0-3
	VSUBPS Y9, Y0, Y11       // Y11 = new indices 4-7

	// Group 2: Y2 with Y3
	VSHUFPS $0xB1, Y3, Y3, Y9
	VMULPS Y15, Y9, Y9
	VFMSUBADD231PS Y14, Y3, Y9

	VADDPS Y9, Y2, Y12       // Y12 = new indices 8-11
	VSUBPS Y9, Y2, Y13       // Y13 = new indices 12-15

	// Group 3: Y4 with Y5
	VSHUFPS $0xB1, Y5, Y5, Y9
	VMULPS Y15, Y9, Y9
	VFMSUBADD231PS Y14, Y5, Y9

	VADDPS Y9, Y4, Y0        // Y0 = new indices 16-19
	VSUBPS Y9, Y4, Y1        // Y1 = new indices 20-23

	// Group 4: Y6 with Y7
	VSHUFPS $0xB1, Y7, Y7, Y9
	VMULPS Y15, Y9, Y9
	VFMSUBADD231PS Y14, Y7, Y9

	VADDPS Y9, Y6, Y2        // Y2 = new indices 24-27
	VSUBPS Y9, Y6, Y3        // Y3 = new indices 28-31

	// Move results to correct registers
	// Y10->Y0, Y11->Y1, Y12->Y2, Y13->Y3 are already in Y0-Y3
	VMOVAPS Y10, Y4          // Y4 = indices 0-3
	VMOVAPS Y11, Y5          // Y5 = indices 4-7
	VMOVAPS Y12, Y6          // Y6 = indices 8-11
	VMOVAPS Y13, Y7          // Y7 = indices 12-15
	// Y0-Y3 already have indices 16-31

	// =======================================================================
	// STAGE 4: size=16, half=8, step=2 - use conjugated twiddles via VFMSUBADD
	// =======================================================================
	// For n=32, Stage 4 implements size-16 sub-FFTs with step=2.
	// Butterfly k pairs index k with k+8, using twiddle[2*k]:
	//   k=0: tw[0],  k=1: tw[2],  k=2: tw[4],  k=3: tw[6]
	//   k=4: tw[8],  k=5: tw[10], k=6: tw[12], k=7: tw[14]
	//
	// Register layout after Stage 3:
	//   Y4=indices[0-3], Y5=indices[4-7], Y6=indices[8-11], Y7=indices[12-15]
	//   Y0=indices[16-19], Y1=indices[20-23], Y2=indices[24-27], Y3=indices[28-31]
	//
	// Butterfly pairs for first half (0-15):
	//   Y4[0-3] with Y6[8-11] using [tw0, tw2, tw4, tw6]
	//   Y5[4-7] with Y7[12-15] using [tw8, tw10, tw12, tw14]
	//
	// Butterfly pairs for second half (16-31):
	//   Y0[16-19] with Y2[24-27] using [tw0, tw2, tw4, tw6]
	//   Y1[20-23] with Y3[28-31] using [tw8, tw10, tw12, tw14]

	// Load twiddle factors for stage 4:
	// Y8 = [tw0, tw2, tw4, tw6] for positions 0-3 and 16-19
	VMOVSD (R10), X8         // twiddle[0]
	VMOVSD 16(R10), X9       // twiddle[2]
	VPUNPCKLQDQ X9, X8, X8   // X8 = [tw0, tw2]
	VMOVSD 32(R10), X9       // twiddle[4]
	VMOVSD 48(R10), X10      // twiddle[6]
	VPUNPCKLQDQ X10, X9, X9  // X9 = [tw4, tw6]
	VINSERTF128 $1, X9, Y8, Y8  // Y8 = [tw0, tw2, tw4, tw6]

	// Y9 = [tw8, tw10, tw12, tw14] for positions 4-7 and 20-23
	VMOVSD 64(R10), X9       // twiddle[8]
	VMOVSD 80(R10), X10      // twiddle[10]
	VPUNPCKLQDQ X10, X9, X9  // X9 = [tw8, tw10]
	VMOVSD 96(R10), X10      // twiddle[12]
	VMOVSD 112(R10), X11     // twiddle[14]
	VPUNPCKLQDQ X11, X10, X10  // X10 = [tw12, tw14]
	VINSERTF128 $1, X10, Y9, Y9  // Y9 = [tw8, tw10, tw12, tw14]

	// Group 1: Y4 (indices 0-3) with Y6 (indices 8-11) using Y8 (tw0,tw2,tw4,tw6)
	VMOVSLDUP Y8, Y10        // Y10 = [w.r broadcast]
	VMOVSHDUP Y8, Y11        // Y11 = [w.i broadcast]
	VSHUFPS $0xB1, Y6, Y6, Y12  // Y12 = b_swapped
	VMULPS Y11, Y12, Y12     // Y12 = b_swap * w.i
	VFMSUBADD231PS Y10, Y6, Y12  // Y12 = conj(w) * b

	VADDPS Y12, Y4, Y13      // Y13 = new indices 0-3
	VSUBPS Y12, Y4, Y14      // Y14 = new indices 8-11

	// Group 2: Y5 (indices 4-7) with Y7 (indices 12-15) using Y9 (tw8,tw10,tw12,tw14)
	VMOVSLDUP Y9, Y10        // Y10 = [w.r broadcast]
	VMOVSHDUP Y9, Y11        // Y11 = [w.i broadcast]
	VSHUFPS $0xB1, Y7, Y7, Y12
	VMULPS Y11, Y12, Y12
	VFMSUBADD231PS Y10, Y7, Y12

	VADDPS Y12, Y5, Y15      // Y15 = new indices 4-7
	VSUBPS Y12, Y5, Y6       // Y6 = new indices 12-15

	// Group 3: Y0 (indices 16-19) with Y2 (indices 24-27) using Y8 (tw0,tw2,tw4,tw6)
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y2, Y2, Y12
	VMULPS Y11, Y12, Y12
	VFMSUBADD231PS Y10, Y2, Y12

	VADDPS Y12, Y0, Y4       // Y4 = new indices 16-19
	VSUBPS Y12, Y0, Y7       // Y7 = new indices 24-27

	// Group 4: Y1 (indices 20-23) with Y3 (indices 28-31) using Y9 (tw8,tw10,tw12,tw14)
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y3, Y3, Y12
	VMULPS Y11, Y12, Y12
	VFMSUBADD231PS Y10, Y3, Y12

	VADDPS Y12, Y1, Y5       // Y5 = new indices 20-23
	VSUBPS Y12, Y1, Y3       // Y3 = new indices 28-31

	// Reorder results to sequential registers
	// Current: Y13=0-3, Y15=4-7, Y14=8-11, Y6=12-15, Y4=16-19, Y5=20-23, Y7=24-27, Y3=28-31
	// Need:    Y0=0-3, Y1=4-7, Y2=8-11, Y3=12-15, Y4=16-19, Y5=20-23, Y6=24-27, Y7=28-31
	VMOVAPS Y3, Y8           // Save Y3 (28-31) temporarily
	VMOVAPS Y6, Y3           // Y3 = indices 12-15 (from Y6)
	VMOVAPS Y7, Y6           // Y6 = indices 24-27 (from Y7)
	VMOVAPS Y8, Y7           // Y7 = indices 28-31 (from saved Y3)
	VMOVAPS Y13, Y0          // Y0 = indices 0-3
	VMOVAPS Y15, Y1          // Y1 = indices 4-7
	VMOVAPS Y14, Y2          // Y2 = indices 8-11
	// Y3 = indices 12-15 (set above)
	// Y4 = indices 16-19 (already correct)
	// Y5 = indices 20-23 (already correct)
	// Y6 = indices 24-27 (set above)
	// Y7 = indices 28-31 (set above)

	// =======================================================================
	// STAGE 5: size=32 - use conjugated twiddles via VFMSUBADD
	// =======================================================================

	// Load twiddle factors for stage 5
	VMOVUPS (R10), Y8        // Y8 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y9      // Y9 = [tw4, tw5, tw6, tw7]
	VMOVUPS 64(R10), Y10     // Y10 = [tw8, tw9, tw10, tw11]
	VMOVUPS 96(R10), Y11     // Y11 = [tw12, tw13, tw14, tw15]

	// Group 1: Y0 (indices 0-3) with Y4 (indices 16-19) using Y8 (tw0-3)
	VMOVSLDUP Y8, Y12
	VMOVSHDUP Y8, Y13
	VSHUFPS $0xB1, Y4, Y4, Y14
	VMULPS Y13, Y14, Y14
	VFMSUBADD231PS Y12, Y4, Y14  // Conjugate multiply

	VADDPS Y14, Y0, Y15      // Y15 = a' (new indices 0-3)
	VSUBPS Y14, Y0, Y4       // Y4 = b' (new indices 16-19)
	VMOVUPS Y15, (R8)        // Store new 0-3 in dst
	VMOVUPS Y4, 128(R8)      // Store new 16-19 in dst

	// Group 2: Y1 (indices 4-7) with Y5 (indices 20-23) using Y9 (tw4-7)
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y5, Y5, Y14
	VMULPS Y13, Y14, Y14
	VFMSUBADD231PS Y12, Y5, Y14

	VADDPS Y14, Y1, Y15      // Y15 = a' (new indices 4-7)
	VSUBPS Y14, Y1, Y4       // Y4 = b' (new indices 20-23)
	VMOVUPS Y15, 32(R8)      // Store new 4-7
	VMOVUPS Y4, 160(R8)      // Store new 20-23

	// Group 3: Y2 (indices 8-11) with Y6 (indices 24-27) using Y10 (tw8-11)
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y6, Y6, Y14
	VMULPS Y13, Y14, Y14
	VFMSUBADD231PS Y12, Y6, Y14

	VADDPS Y14, Y2, Y15      // Y15 = a' (new indices 8-11)
	VSUBPS Y14, Y2, Y4       // Y4 = b' (new indices 24-27)
	VMOVUPS Y15, 64(R8)      // Store new 8-11
	VMOVUPS Y4, 192(R8)      // Store new 24-27

	// Group 4: Y3 (indices 12-15) with Y7 (indices 28-31) using Y11 (tw12-15)
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y7, Y7, Y14
	VMULPS Y13, Y14, Y14
	VFMSUBADD231PS Y12, Y7, Y14

	VADDPS Y14, Y3, Y15      // Y15 = a' (new indices 12-15)
	VSUBPS Y14, Y3, Y4       // Y4 = b' (new indices 28-31)
	VMOVUPS Y15, 96(R8)      // Store new 12-15
	VMOVUPS Y4, 224(R8)      // Store new 28-31

	// =======================================================================
	// Apply 1/n scaling (1/32 = 0.03125) for inverse normalization
	// =======================================================================
	// Load all 8 registers from work buffer
	VMOVUPS (R8), Y0         // Load stage 5 results from working buffer
	VMOVUPS 32(R8), Y1
	VMOVUPS 64(R8), Y2
	VMOVUPS 96(R8), Y3
	VMOVUPS 128(R8), Y4
	VMOVUPS 160(R8), Y5
	VMOVUPS 192(R8), Y6
	VMOVUPS 224(R8), Y7

	// Create scale factor: 1/32 = 0.03125 = 0x3D000000 in IEEE-754
	MOVL ·thirtySecond32(SB), AX     // Load 0.03125 constant
	MOVD AX, X8              // Move to X8
	VBROADCASTSS X8, Y8      // Y8 = [0.03125, 0.03125, ...] (broadcast to all 8 elements)

	// Scale all 8 registers by 1/32
	VMULPS Y8, Y0, Y0        // Y0 *= 1/32
	VMULPS Y8, Y1, Y1        // Y1 *= 1/32
	VMULPS Y8, Y2, Y2        // Y2 *= 1/32
	VMULPS Y8, Y3, Y3        // Y3 *= 1/32
	VMULPS Y8, Y4, Y4        // Y4 *= 1/32
	VMULPS Y8, Y5, Y5        // Y5 *= 1/32
	VMULPS Y8, Y6, Y6        // Y6 *= 1/32
	VMULPS Y8, Y7, Y7        // Y7 *= 1/32

	// =======================================================================
	// Store results to dst
	// =======================================================================
	MOVQ dst+0(FP), R9       // R9 = dst pointer
	VMOVUPS Y0, (R9)         // Store scaled results to dst[0:32]
	VMOVUPS Y1, 32(R9)
	VMOVUPS Y2, 64(R9)
	VMOVUPS Y3, 96(R9)
	VMOVUPS Y4, 128(R9)
	VMOVUPS Y5, 160(R9)
	VMOVUPS Y6, 192(R9)
	VMOVUPS Y7, 224(R9)

size32_inv_done:
	VZEROUPPER               // Clear SIMD state
	MOVB $1, ret+120(FP)     // Return true (success)
	RET

size32_inv_return_false:
	MOVB $0, ret+120(FP)     // Return false (validation failed)
	RET
