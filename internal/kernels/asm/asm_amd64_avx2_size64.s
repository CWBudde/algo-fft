//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-64 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains fully-unrolled FFT kernels optimized for size 64.
// These kernels provide better performance than the generic implementation by:
//   - Eliminating loop overhead
//   - Using hardcoded twiddle factor indices
//   - Optimal register allocation for size 64
//
// See asm_amd64_avx2_generic.s for algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 64, complex64
// Fully unrolled 6-stage FFT with AVX2 vectorization
//
// This kernel implements a complete radix-2 DIT FFT for exactly 64 complex64 values.
// All 6 stages are fully unrolled with hardcoded twiddle factor indices:
//   Stage 1 (size=2):  32 butterflies, step=32, twiddle index 0 for all
//   Stage 2 (size=4):  32 butterflies in 8 groups, step=16, twiddle indices [0,16]
//   Stage 3 (size=8):  32 butterflies in 4 groups, step=8, twiddle indices [0,8,16,24]
//   Stage 4 (size=16): 32 butterflies in 2 groups, step=4, twiddle indices [0,4,8,12,16,20,24,28]
//   Stage 5 (size=32): 32 butterflies in 1 group, step=2, twiddle indices [0,2,...,30]
//   Stage 6 (size=64): 32 butterflies, step=1, twiddle indices [0,1,2,...,31]
//
// Register allocation:
//   R8:  work buffer (dst or scratch for in-place)
//   R9:  src pointer
//   R10: twiddle pointer
//   R11: scratch pointer
//   R12: bitrev pointer
//   Data stored in memory (R8), processed in groups of 4 YMM registers
//
TEXT ·forwardAVX2Size64Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 64)

	// Verify n == 64
	CMPQ R13, $64
	JNE  size64_return_false

	// Validate all slice lengths >= 64
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   size64_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   size64_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   size64_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $64
	JL   size64_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  size64_use_dst

	// In-place: use scratch
	MOVQ R11, R8
	JMP  size64_bitrev

size64_use_dst:
	// Out-of-place: use dst

size64_bitrev:
	// =======================================================================
	// Bit-reversal permutation: work[i] = src[bitrev[i]]
	// =======================================================================
	// Unrolled loop for 64 elements

	// Indices 0-7
	MOVQ (R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, (R8)

	MOVQ 8(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 8(R8)

	MOVQ 16(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 16(R8)

	MOVQ 24(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 24(R8)

	MOVQ 32(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 32(R8)

	MOVQ 40(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 40(R8)

	MOVQ 48(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 48(R8)

	MOVQ 56(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 56(R8)

	// Indices 8-15
	MOVQ 64(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 64(R8)

	MOVQ 72(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 72(R8)

	MOVQ 80(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 80(R8)

	MOVQ 88(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 88(R8)

	MOVQ 96(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 96(R8)

	MOVQ 104(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 104(R8)

	MOVQ 112(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 112(R8)

	MOVQ 120(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 120(R8)

	// Indices 16-23
	MOVQ 128(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 128(R8)

	MOVQ 136(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 136(R8)

	MOVQ 144(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 144(R8)

	MOVQ 152(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 152(R8)

	MOVQ 160(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 160(R8)

	MOVQ 168(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 168(R8)

	MOVQ 176(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 176(R8)

	MOVQ 184(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 184(R8)

	// Indices 24-31
	MOVQ 192(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 192(R8)

	MOVQ 200(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 200(R8)

	MOVQ 208(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 208(R8)

	MOVQ 216(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 216(R8)

	MOVQ 224(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 224(R8)

	MOVQ 232(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 232(R8)

	MOVQ 240(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 240(R8)

	MOVQ 248(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 248(R8)

	// Indices 32-39
	MOVQ 256(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 256(R8)

	MOVQ 264(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 264(R8)

	MOVQ 272(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 272(R8)

	MOVQ 280(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 280(R8)

	MOVQ 288(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 288(R8)

	MOVQ 296(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 296(R8)

	MOVQ 304(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 304(R8)

	MOVQ 312(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 312(R8)

	// Indices 40-47
	MOVQ 320(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 320(R8)

	MOVQ 328(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 328(R8)

	MOVQ 336(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 336(R8)

	MOVQ 344(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 344(R8)

	MOVQ 352(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 352(R8)

	MOVQ 360(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 360(R8)

	MOVQ 368(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 368(R8)

	MOVQ 376(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 376(R8)

	// Indices 48-55
	MOVQ 384(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 384(R8)

	MOVQ 392(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 392(R8)

	MOVQ 400(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 400(R8)

	MOVQ 408(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 408(R8)

	MOVQ 416(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 416(R8)

	MOVQ 424(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 424(R8)

	MOVQ 432(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 432(R8)

	MOVQ 440(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 440(R8)

	// Indices 56-63
	MOVQ 448(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 448(R8)

	MOVQ 456(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 456(R8)

	MOVQ 464(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 464(R8)

	MOVQ 472(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 472(R8)

	MOVQ 480(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 480(R8)

	MOVQ 488(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 488(R8)

	MOVQ 496(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 496(R8)

	MOVQ 504(R12), DX
	MOVQ (R9)(DX*8), AX
	MOVQ AX, 504(R8)

	// =======================================================================
	// STAGE 1: size=2, half=1, step=32
	// =======================================================================
	// 32 independent butterflies with twiddle[0] = (1,0) = identity
	// Process in groups of 4 YMM registers (16 complex values at a time)

	// Group 0: indices 0-15
	// For size-2 butterfly: out[0] = in[0] + in[1], out[1] = in[0] - in[1]
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VMOVUPS 64(R8), Y2
	VMOVUPS 96(R8), Y3

	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6        // Y4-Y0, not Y0-Y4!
	VBLENDPD $0x0A, Y6, Y5, Y0  // 64-bit blend

	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1

	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2

	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3

	VMOVUPS Y0, (R8)
	VMOVUPS Y1, 32(R8)
	VMOVUPS Y2, 64(R8)
	VMOVUPS Y3, 96(R8)

	// Group 1: indices 16-31
	VMOVUPS 128(R8), Y0
	VMOVUPS 160(R8), Y1
	VMOVUPS 192(R8), Y2
	VMOVUPS 224(R8), Y3

	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0

	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1

	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2

	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3

	VMOVUPS Y0, 128(R8)
	VMOVUPS Y1, 160(R8)
	VMOVUPS Y2, 192(R8)
	VMOVUPS Y3, 224(R8)

	// Group 2: indices 32-47
	VMOVUPS 256(R8), Y0
	VMOVUPS 288(R8), Y1
	VMOVUPS 320(R8), Y2
	VMOVUPS 352(R8), Y3

	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0

	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1

	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2

	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3

	VMOVUPS Y0, 256(R8)
	VMOVUPS Y1, 288(R8)
	VMOVUPS Y2, 320(R8)
	VMOVUPS Y3, 352(R8)

	// Group 3: indices 48-63
	VMOVUPS 384(R8), Y0
	VMOVUPS 416(R8), Y1
	VMOVUPS 448(R8), Y2
	VMOVUPS 480(R8), Y3

	VPERMILPD $0x05, Y0, Y4
	VADDPS Y4, Y0, Y5
	VSUBPS Y0, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y0

	VPERMILPD $0x05, Y1, Y4
	VADDPS Y4, Y1, Y5
	VSUBPS Y1, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y1

	VPERMILPD $0x05, Y2, Y4
	VADDPS Y4, Y2, Y5
	VSUBPS Y2, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y2

	VPERMILPD $0x05, Y3, Y4
	VADDPS Y4, Y3, Y5
	VSUBPS Y3, Y4, Y6
	VBLENDPD $0x0A, Y6, Y5, Y3

	VMOVUPS Y0, 384(R8)
	VMOVUPS Y1, 416(R8)
	VMOVUPS Y2, 448(R8)
	VMOVUPS Y3, 480(R8)

	// =======================================================================
	// STAGE 2: size=4, half=2, step=16
	// =======================================================================
	// Twiddle factors: twiddle[0], twiddle[16]
	// twiddle[0] = (1, 0), twiddle[16] = (0, -1) for n=64

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 128(R10), X9        // twiddle[16] (16 * 8 bytes)
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw16]
	VINSERTF128 $1, X8, Y8, Y8 // Y8 = [tw0, tw16, tw0, tw16]

	// Process all YMM registers with stage 2 butterflies
	// Each YMM has 4 complex values [d0, d1, d2, d3]
	// Butterfly pairs: (d0, d2) and (d1, d3)

	// Helper macro pattern for stage 2 (process one YMM at offset)
	// Load, extract halves, multiply, butterfly, store

	// Indices 0-3
	VMOVUPS (R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1    // Y1 = [d0, d1, d0, d1]
	VPERM2F128 $0x11, Y0, Y0, Y2    // Y2 = [d2, d3, d2, d3]
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, (R8)

	// Indices 4-7
	VMOVUPS 32(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 32(R8)

	// Indices 8-11
	VMOVUPS 64(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 64(R8)

	// Indices 12-15
	VMOVUPS 96(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 96(R8)

	// Indices 16-19
	VMOVUPS 128(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 128(R8)

	// Indices 20-23
	VMOVUPS 160(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 160(R8)

	// Indices 24-27
	VMOVUPS 192(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 192(R8)

	// Indices 28-31
	VMOVUPS 224(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 224(R8)

	// Indices 32-35
	VMOVUPS 256(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 256(R8)

	// Indices 36-39
	VMOVUPS 288(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 288(R8)

	// Indices 40-43
	VMOVUPS 320(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 320(R8)

	// Indices 44-47
	VMOVUPS 352(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 352(R8)

	// Indices 48-51
	VMOVUPS 384(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 384(R8)

	// Indices 52-55
	VMOVUPS 416(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 416(R8)

	// Indices 56-59
	VMOVUPS 448(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 448(R8)

	// Indices 60-63
	VMOVUPS 480(R8), Y0
	VPERM2F128 $0x00, Y0, Y0, Y1
	VPERM2F128 $0x11, Y0, Y0, Y2
	VMOVSLDUP Y8, Y3
	VMOVSHDUP Y8, Y4
	VSHUFPS $0xB1, Y2, Y2, Y5
	VMULPS Y4, Y5, Y5
	VFMADDSUB231PS Y3, Y2, Y5
	VADDPS Y5, Y1, Y6
	VSUBPS Y5, Y1, Y7
	VINSERTF128 $1, X7, Y6, Y0
	VMOVUPS Y0, 480(R8)

	// =======================================================================
	// STAGE 3: size=8, half=4, step=8
	// =======================================================================
	// Pairs: indices 0-3 with 4-7, 8-11 with 12-15, etc.
	// Twiddle factors: twiddle[0], twiddle[8], twiddle[16], twiddle[24]

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 64(R10), X9         // twiddle[8]
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw8]
	VMOVSD 128(R10), X9        // twiddle[16]
	VMOVSD 192(R10), X10       // twiddle[24]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw16, tw24]
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw8, tw16, tw24]

	// Extract twiddle components once
	VMOVSLDUP Y8, Y14          // Y14 = [w.r, w.r, ...]
	VMOVSHDUP Y8, Y15          // Y15 = [w.i, w.i, ...]

	// Group: indices 0-3 with 4-7
	VMOVUPS (R8), Y0           // a = indices 0-3
	VMOVUPS 32(R8), Y1         // b = indices 4-7
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2 // Y2 = t = w * b
	VADDPS Y2, Y0, Y3          // Y3 = a + t
	VSUBPS Y2, Y0, Y4          // Y4 = a - t
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 32(R8)

	// Group: indices 8-11 with 12-15
	VMOVUPS 64(R8), Y0
	VMOVUPS 96(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 96(R8)

	// Group: indices 16-19 with 20-23
	VMOVUPS 128(R8), Y0
	VMOVUPS 160(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 160(R8)

	// Group: indices 24-27 with 28-31
	VMOVUPS 192(R8), Y0
	VMOVUPS 224(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 192(R8)
	VMOVUPS Y4, 224(R8)

	// Group: indices 32-35 with 36-39
	VMOVUPS 256(R8), Y0
	VMOVUPS 288(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 288(R8)

	// Group: indices 40-43 with 44-47
	VMOVUPS 320(R8), Y0
	VMOVUPS 352(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y4, 352(R8)

	// Group: indices 48-51 with 52-55
	VMOVUPS 384(R8), Y0
	VMOVUPS 416(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y4, 416(R8)

	// Group: indices 56-59 with 60-63
	VMOVUPS 448(R8), Y0
	VMOVUPS 480(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 448(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 4: size=16, half=8, step=4
	// =======================================================================
	// Pairs: indices 0-7 with 8-15, 16-23 with 24-31, 32-39 with 40-47, 48-55 with 56-63
	// Twiddle factors: twiddle[0,4,8,12] for first 4 elements, same for next 4

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 32(R10), X9         // twiddle[4]
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw4]
	VMOVSD 64(R10), X9         // twiddle[8]
	VMOVSD 96(R10), X10        // twiddle[12]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw8, tw12]
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw4, tw8, tw12]

	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15

	// Group 1: indices 0-3 with 8-11 (first half of 16-point group)
	VMOVUPS (R8), Y0
	VMOVUPS 64(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 64(R8)

	// Group 1: indices 4-7 with 12-15
	VMOVUPS 32(R8), Y0
	VMOVUPS 96(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 96(R8)

	// Group 2: indices 16-19 with 24-27
	VMOVUPS 128(R8), Y0
	VMOVUPS 192(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 192(R8)

	// Group 2: indices 20-23 with 28-31
	VMOVUPS 160(R8), Y0
	VMOVUPS 224(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 160(R8)
	VMOVUPS Y4, 224(R8)

	// Group 3: indices 32-35 with 40-43
	VMOVUPS 256(R8), Y0
	VMOVUPS 320(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 320(R8)

	// Group 3: indices 36-39 with 44-47
	VMOVUPS 288(R8), Y0
	VMOVUPS 352(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y4, 352(R8)

	// Group 4: indices 48-51 with 56-59
	VMOVUPS 384(R8), Y0
	VMOVUPS 448(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y4, 448(R8)

	// Group 4: indices 52-55 with 60-63
	VMOVUPS 416(R8), Y0
	VMOVUPS 480(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMADDSUB231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 416(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 5: size=32, half=16, step=2
	// =======================================================================
	// Pairs: indices 0-15 with 16-31, 32-47 with 48-63
	// Twiddle factors: twiddle[0,2,4,6,8,10,12,14] for each 4-element chunk

	// Load twiddles for indices 0-3: tw[0,2,4,6]
	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 16(R10), X9         // twiddle[2]
	VPUNPCKLQDQ X9, X8, X8
	VMOVSD 32(R10), X9         // twiddle[4]
	VMOVSD 48(R10), X10        // twiddle[6]
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw2, tw4, tw6]

	// Load twiddles for indices 4-7: tw[8,10,12,14]
	VMOVSD 64(R10), X9         // twiddle[8]
	VMOVSD 80(R10), X10        // twiddle[10]
	VPUNPCKLQDQ X10, X9, X9
	VMOVSD 96(R10), X10        // twiddle[12]
	VMOVSD 112(R10), X11       // twiddle[14]
	VPUNPCKLQDQ X11, X10, X10
	VINSERTF128 $1, X10, Y9, Y9 // Y9 = [tw8, tw10, tw12, tw14]

	// Group 1: indices 0-3 with 16-19
	VMOVUPS (R8), Y0
	VMOVUPS 128(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 128(R8)

	// Group 1: indices 4-7 with 20-23
	VMOVUPS 32(R8), Y0
	VMOVUPS 160(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 160(R8)

	// Group 1: indices 8-11 with 24-27
	VMOVUPS 64(R8), Y0
	VMOVUPS 192(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 192(R8)

	// Group 1: indices 12-15 with 28-31
	VMOVUPS 96(R8), Y0
	VMOVUPS 224(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y4, 224(R8)

	// Group 2: indices 32-35 with 48-51
	VMOVUPS 256(R8), Y0
	VMOVUPS 384(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 384(R8)

	// Group 2: indices 36-39 with 52-55
	VMOVUPS 288(R8), Y0
	VMOVUPS 416(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y4, 416(R8)

	// Group 2: indices 40-43 with 56-59
	VMOVUPS 320(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y4, 448(R8)

	// Group 2: indices 44-47 with 60-63
	VMOVUPS 352(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 352(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 6: size=64, half=32, step=1
	// =======================================================================
	// Pairs: indices 0-31 with 32-63
	// Twiddle factors: twiddle[0,1,2,...,31]

	// Load twiddles for indices 0-3
	VMOVUPS (R10), Y8          // Y8 = [tw0, tw1, tw2, tw3]
	VMOVUPS 32(R10), Y9        // Y9 = [tw4, tw5, tw6, tw7]

	// Group: indices 0-3 with 32-35
	VMOVUPS (R8), Y0
	VMOVUPS 256(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 256(R8)

	// Group: indices 4-7 with 36-39
	VMOVUPS 32(R8), Y0
	VMOVUPS 288(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 288(R8)

	// Load twiddles for indices 8-15
	VMOVUPS 64(R10), Y8        // Y8 = [tw8, tw9, tw10, tw11]
	VMOVUPS 96(R10), Y9        // Y9 = [tw12, tw13, tw14, tw15]

	// Group: indices 8-11 with 40-43
	VMOVUPS 64(R8), Y0
	VMOVUPS 320(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 320(R8)

	// Group: indices 12-15 with 44-47
	VMOVUPS 96(R8), Y0
	VMOVUPS 352(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y4, 352(R8)

	// Load twiddles for indices 16-23
	VMOVUPS 128(R10), Y8       // Y8 = [tw16, tw17, tw18, tw19]
	VMOVUPS 160(R10), Y9       // Y9 = [tw20, tw21, tw22, tw23]

	// Group: indices 16-19 with 48-51
	VMOVUPS 128(R8), Y0
	VMOVUPS 384(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 384(R8)

	// Group: indices 20-23 with 52-55
	VMOVUPS 160(R8), Y0
	VMOVUPS 416(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 160(R8)
	VMOVUPS Y4, 416(R8)

	// Load twiddles for indices 24-31
	VMOVUPS 192(R10), Y8       // Y8 = [tw24, tw25, tw26, tw27]
	VMOVUPS 224(R10), Y9       // Y9 = [tw28, tw29, tw30, tw31]

	// Group: indices 24-27 with 56-59
	VMOVUPS 192(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 192(R8)
	VMOVUPS Y4, 448(R8)

	// Group: indices 28-31 with 60-63
	VMOVUPS 224(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y9, Y10
	VMOVSHDUP Y9, Y11
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y11, Y2, Y2
	VFMADDSUB231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 224(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// Copy results to dst if we used scratch buffer
	// =======================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size64_done

	// Copy from scratch to dst (512 bytes = 16 YMM registers)
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VMOVUPS 64(R8), Y2
	VMOVUPS 96(R8), Y3
	VMOVUPS 128(R8), Y4
	VMOVUPS 160(R8), Y5
	VMOVUPS 192(R8), Y6
	VMOVUPS 224(R8), Y7
	VMOVUPS Y0, (R9)
	VMOVUPS Y1, 32(R9)
	VMOVUPS Y2, 64(R9)
	VMOVUPS Y3, 96(R9)
	VMOVUPS Y4, 128(R9)
	VMOVUPS Y5, 160(R9)
	VMOVUPS Y6, 192(R9)
	VMOVUPS Y7, 224(R9)

	VMOVUPS 256(R8), Y0
	VMOVUPS 288(R8), Y1
	VMOVUPS 320(R8), Y2
	VMOVUPS 352(R8), Y3
	VMOVUPS 384(R8), Y4
	VMOVUPS 416(R8), Y5
	VMOVUPS 448(R8), Y6
	VMOVUPS 480(R8), Y7
	VMOVUPS Y0, 256(R9)
	VMOVUPS Y1, 288(R9)
	VMOVUPS Y2, 320(R9)
	VMOVUPS Y3, 352(R9)
	VMOVUPS Y4, 384(R9)
	VMOVUPS Y5, 416(R9)
	VMOVUPS Y6, 448(R9)
	VMOVUPS Y7, 480(R9)

size64_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size64_return_false:
	MOVB $0, ret+120(FP)
	RET

// Inverse transform, size 64, complex64
// Fully unrolled 6-stage IFFT with AVX2 vectorization
//
// This kernel implements a complete radix-2 DIF (Decimation-in-Frequency) IFFT
// for exactly 64 complex64 values. The inverse uses conjugate twiddles and
// reversed stage order compared to forward transform.
//
// Stage order (reversed from forward):
//   Stage 1 (size=64): 32 butterflies, step=1, twiddle indices [0,1,...,31] (conjugated)
//   Stage 2 (size=32): 32 butterflies, step=2, twiddle indices [0,2,...,30] (conjugated)
//   Stage 3 (size=16): 32 butterflies, step=4, twiddle indices [0,4,8,12,...] (conjugated)
//   Stage 4 (size=8):  32 butterflies, step=8, twiddle indices [0,8,16,24] (conjugated)
//   Stage 5 (size=4):  32 butterflies, step=16, twiddle indices [0,16] (conjugated)
//   Stage 6 (size=2):  32 butterflies, identity twiddle (w=1)
//
// For inverse FFT butterfly: out[i] = in[i] + in[j], out[j] = (in[i] - in[j]) * conj(w)
// Conjugate of twiddle is applied by negating the imaginary part during multiply.
//
TEXT ·inverseAVX2Size64Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 64)

	// Verify n == 64
	CMPQ R13, $64
	JNE  inv_size64_return_false

	// Validate all slice lengths >= 64
	MOVQ dst+8(FP), AX
	CMPQ AX, $64
	JL   inv_size64_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $64
	JL   inv_size64_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $64
	JL   inv_size64_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $64
	JL   inv_size64_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  inv_size64_use_dst

	// In-place: use scratch as work buffer, copy src to it first
	MOVQ R11, R8
	JMP  inv_size64_copy_input

inv_size64_use_dst:
	// Out-of-place: copy src to dst, work in dst
	// Fall through to copy

inv_size64_copy_input:
	// Copy src to work buffer (512 bytes = 64 complex64)
	VMOVUPS (R9), Y0
	VMOVUPS 32(R9), Y1
	VMOVUPS 64(R9), Y2
	VMOVUPS 96(R9), Y3
	VMOVUPS 128(R9), Y4
	VMOVUPS 160(R9), Y5
	VMOVUPS 192(R9), Y6
	VMOVUPS 224(R9), Y7
	VMOVUPS Y0, (R8)
	VMOVUPS Y1, 32(R8)
	VMOVUPS Y2, 64(R8)
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y4, 128(R8)
	VMOVUPS Y5, 160(R8)
	VMOVUPS Y6, 192(R8)
	VMOVUPS Y7, 224(R8)

	VMOVUPS 256(R9), Y0
	VMOVUPS 288(R9), Y1
	VMOVUPS 320(R9), Y2
	VMOVUPS 352(R9), Y3
	VMOVUPS 384(R9), Y4
	VMOVUPS 416(R9), Y5
	VMOVUPS 448(R9), Y6
	VMOVUPS 480(R9), Y7
	VMOVUPS Y0, 256(R8)
	VMOVUPS Y1, 288(R8)
	VMOVUPS Y2, 320(R8)
	VMOVUPS Y3, 352(R8)
	VMOVUPS Y4, 384(R8)
	VMOVUPS Y5, 416(R8)
	VMOVUPS Y6, 448(R8)
	VMOVUPS Y7, 480(R8)

	// =======================================================================
	// STAGE 1 (IFFT): size=64, step=1
	// 32 butterflies, twiddle indices [0,1,...,31] (conjugated)
	// For IFFT butterfly: out[i] = in[i] + in[j], out[j] = (in[i] - in[j]) * conj(w)
	// conj(w) means we negate the imaginary part: use VFMSUBADD instead of VFMADDSUB
	// =======================================================================

	// Load twiddles for indices 0-3
	VMOVUPS (R10), Y8          // Y8 = [tw0, tw1, tw2, tw3]

	// Group: indices 0-3 with 32-35
	VMOVUPS (R8), Y0           // Y0 = work[0:3]
	VMOVUPS 256(R8), Y1        // Y1 = work[32:35]
	VADDPS Y1, Y0, Y3          // Y3 = Y0 + Y1
	VSUBPS Y1, Y0, Y4          // Y4 = Y0 - Y1
	// Multiply Y4 by conj(tw): conj multiply = (a-bi)(c-di) = (ac-bd) + i(-ad-bc)
	// Using VFMSUBADD: real = a*c - b*d (sub), imag = a*d + b*c (add) -> then negate imag
	// Actually for conj(w), we negate imag of w before multiply:
	// (a+bi)(c-di) = (ac+bd) + i(bc-ad) = VFMSUBADD pattern
	VMOVSLDUP Y8, Y10          // Y10 = [re, re, re, re, ...] of twiddle
	VMOVSHDUP Y8, Y11          // Y11 = [im, im, im, im, ...] of twiddle
	VSHUFPS $0xB1, Y4, Y4, Y2  // Y2 = [im, re, im, re, ...] of (in[i]-in[j])
	VMULPS Y11, Y2, Y2         // Y2 = im_tw * [im, re, ...]
	VFMSUBADD231PS Y10, Y4, Y2 // Y2 = re_tw * (in[i]-in[j]) -/+ Y2 = conj multiply result
	VMOVUPS Y3, (R8)
	VMOVUPS Y2, 256(R8)

	// Load twiddles for indices 4-7
	VMOVUPS 32(R10), Y8

	// Group: indices 4-7 with 36-39
	VMOVUPS 32(R8), Y0
	VMOVUPS 288(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y2, 288(R8)

	// Load twiddles for indices 8-11
	VMOVUPS 64(R10), Y8

	// Group: indices 8-11 with 40-43
	VMOVUPS 64(R8), Y0
	VMOVUPS 320(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y2, 320(R8)

	// Load twiddles for indices 12-15
	VMOVUPS 96(R10), Y8

	// Group: indices 12-15 with 44-47
	VMOVUPS 96(R8), Y0
	VMOVUPS 352(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y2, 352(R8)

	// Load twiddles for indices 16-19
	VMOVUPS 128(R10), Y8

	// Group: indices 16-19 with 48-51
	VMOVUPS 128(R8), Y0
	VMOVUPS 384(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y2, 384(R8)

	// Load twiddles for indices 20-23
	VMOVUPS 160(R10), Y8

	// Group: indices 20-23 with 52-55
	VMOVUPS 160(R8), Y0
	VMOVUPS 416(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 160(R8)
	VMOVUPS Y2, 416(R8)

	// Load twiddles for indices 24-27
	VMOVUPS 192(R10), Y8

	// Group: indices 24-27 with 56-59
	VMOVUPS 192(R8), Y0
	VMOVUPS 448(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 192(R8)
	VMOVUPS Y2, 448(R8)

	// Load twiddles for indices 28-31
	VMOVUPS 224(R10), Y8

	// Group: indices 28-31 with 60-63
	VMOVUPS 224(R8), Y0
	VMOVUPS 480(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 224(R8)
	VMOVUPS Y2, 480(R8)

	// =======================================================================
	// STAGE 2 (IFFT): size=32, step=2
	// Twiddle indices: 0,2,4,...,30 (every 2nd twiddle, conjugated)
	// Pairs: (0,16), (1,17), ..., (15,31) and (32,48), (33,49), ..., (47,63)
	// =======================================================================

	// Twiddles for this stage: indices 0,2,4,6 -> offsets 0,16,32,48
	VMOVUPS (R10), Y8          // tw[0] at offset 0
	VPERM2F128 $0x00, Y8, Y8, Y8  // Broadcast tw[0] to both lanes (not quite right)
	// Actually we need tw[0,2,4,6] which are at byte offsets 0,16,32,48
	// Load them individually and pack

	// For stage 2, twiddles needed per 4-element group:
	// Group 0-3 pairs with 16-19: twiddles [0,2,4,6] at offsets 0,16,32,48
	// Group 4-7 pairs with 20-23: twiddles [8,10,12,14] at offsets 64,80,96,112
	// etc for bottom half

	// Load tw[0], tw[2], tw[4], tw[6] and pack into Y8
	VMOVSD (R10), X8           // tw[0] (8 bytes)
	VMOVSD 16(R10), X9         // tw[2]
	VUNPCKLPD X9, X8, X8       // X8 = [tw[0], tw[2]]
	VMOVSD 32(R10), X9         // tw[4]
	VMOVSD 48(R10), X10        // tw[6]
	VUNPCKLPD X10, X9, X9      // X9 = [tw[4], tw[6]]
	VINSERTI128 $1, X9, Y8, Y8 // Y8 = [tw[0], tw[2], tw[4], tw[6]]

	// Group: indices 0-3 with 16-19
	VMOVUPS (R8), Y0
	VMOVUPS 128(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, (R8)
	VMOVUPS Y2, 128(R8)

	// Group: indices 4-7 with 20-23, twiddles [8,10,12,14]
	VMOVSD 64(R10), X8
	VMOVSD 80(R10), X9
	VUNPCKLPD X9, X8, X8
	VMOVSD 96(R10), X9
	VMOVSD 112(R10), X10
	VUNPCKLPD X10, X9, X9
	VINSERTI128 $1, X9, Y8, Y8

	VMOVUPS 32(R8), Y0
	VMOVUPS 160(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y2, 160(R8)

	// Group: indices 8-11 with 24-27, twiddles [16,18,20,22]
	VMOVSD 128(R10), X8
	VMOVSD 144(R10), X9
	VUNPCKLPD X9, X8, X8
	VMOVSD 160(R10), X9
	VMOVSD 176(R10), X10
	VUNPCKLPD X10, X9, X9
	VINSERTI128 $1, X9, Y8, Y8

	VMOVUPS 64(R8), Y0
	VMOVUPS 192(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y2, 192(R8)

	// Group: indices 12-15 with 28-31, twiddles [24,26,28,30]
	VMOVSD 192(R10), X8
	VMOVSD 208(R10), X9
	VUNPCKLPD X9, X8, X8
	VMOVSD 224(R10), X9
	VMOVSD 240(R10), X10
	VUNPCKLPD X10, X9, X9
	VINSERTI128 $1, X9, Y8, Y8

	VMOVUPS 96(R8), Y0
	VMOVUPS 224(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y2, 224(R8)

	// Bottom half: indices 32-47 with 48-63 (same twiddle pattern)
	// Group: indices 32-35 with 48-51, twiddles [0,2,4,6]
	VMOVSD (R10), X8
	VMOVSD 16(R10), X9
	VUNPCKLPD X9, X8, X8
	VMOVSD 32(R10), X9
	VMOVSD 48(R10), X10
	VUNPCKLPD X10, X9, X9
	VINSERTI128 $1, X9, Y8, Y8

	VMOVUPS 256(R8), Y0
	VMOVUPS 384(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y2, 384(R8)

	// Group: indices 36-39 with 52-55, twiddles [8,10,12,14]
	VMOVSD 64(R10), X8
	VMOVSD 80(R10), X9
	VUNPCKLPD X9, X8, X8
	VMOVSD 96(R10), X9
	VMOVSD 112(R10), X10
	VUNPCKLPD X10, X9, X9
	VINSERTI128 $1, X9, Y8, Y8

	VMOVUPS 288(R8), Y0
	VMOVUPS 416(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y2, 416(R8)

	// Group: indices 40-43 with 56-59, twiddles [16,18,20,22]
	VMOVSD 128(R10), X8
	VMOVSD 144(R10), X9
	VUNPCKLPD X9, X8, X8
	VMOVSD 160(R10), X9
	VMOVSD 176(R10), X10
	VUNPCKLPD X10, X9, X9
	VINSERTI128 $1, X9, Y8, Y8

	VMOVUPS 320(R8), Y0
	VMOVUPS 448(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y2, 448(R8)

	// Group: indices 44-47 with 60-63, twiddles [24,26,28,30]
	VMOVSD 192(R10), X8
	VMOVSD 208(R10), X9
	VUNPCKLPD X9, X8, X8
	VMOVSD 224(R10), X9
	VMOVSD 240(R10), X10
	VUNPCKLPD X10, X9, X9
	VINSERTI128 $1, X9, Y8, Y8

	VMOVUPS 352(R8), Y0
	VMOVUPS 480(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 352(R8)
	VMOVUPS Y2, 480(R8)

	// =======================================================================
	// STAGE 3 (IFFT): size=16, step=4
	// Twiddle indices: 0,4,8,12 (every 4th twiddle, conjugated)
	// =======================================================================

	// Load twiddles [0,4,8,12] -> byte offsets 0,32,64,96
	VMOVSD (R10), X8
	VMOVSD 32(R10), X9
	VUNPCKLPD X9, X8, X8
	VMOVSD 64(R10), X9
	VMOVSD 96(R10), X10
	VUNPCKLPD X10, X9, X9
	VINSERTI128 $1, X9, Y8, Y8

	// 4 groups of 16 elements each, pairs at distance 8
	// Group 0: indices 0-3 with 8-11
	VMOVUPS (R8), Y0
	VMOVUPS 64(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, (R8)
	VMOVUPS Y2, 64(R8)

	// Group 0 continued: indices 4-7 with 12-15 (same twiddles)
	VMOVUPS 32(R8), Y0
	VMOVUPS 96(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y2, 96(R8)

	// Group 1: indices 16-19 with 24-27
	VMOVUPS 128(R8), Y0
	VMOVUPS 192(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y2, 192(R8)

	// Group 1 continued: indices 20-23 with 28-31
	VMOVUPS 160(R8), Y0
	VMOVUPS 224(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 160(R8)
	VMOVUPS Y2, 224(R8)

	// Group 2: indices 32-35 with 40-43
	VMOVUPS 256(R8), Y0
	VMOVUPS 320(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y2, 320(R8)

	// Group 2 continued: indices 36-39 with 44-47
	VMOVUPS 288(R8), Y0
	VMOVUPS 352(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y2, 352(R8)

	// Group 3: indices 48-51 with 56-59
	VMOVUPS 384(R8), Y0
	VMOVUPS 448(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y2, 448(R8)

	// Group 3 continued: indices 52-55 with 60-63
	VMOVUPS 416(R8), Y0
	VMOVUPS 480(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 416(R8)
	VMOVUPS Y2, 480(R8)

	// =======================================================================
	// STAGE 4 (IFFT): size=8, step=8
	// Twiddle indices: 0,8,16,24 (every 8th twiddle, conjugated)
	// =======================================================================

	// Load twiddles [0,8,16,24] -> byte offsets 0,64,128,192
	VMOVSD (R10), X8
	VMOVSD 64(R10), X9
	VUNPCKLPD X9, X8, X8
	VMOVSD 128(R10), X9
	VMOVSD 192(R10), X10
	VUNPCKLPD X10, X9, X9
	VINSERTI128 $1, X9, Y8, Y8

	// 8 groups of 8 elements, pairs at distance 4
	// Group 0: indices 0-3 with 4-7
	VMOVUPS (R8), Y0
	VMOVUPS 32(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, (R8)
	VMOVUPS Y2, 32(R8)

	// Group 1: indices 8-11 with 12-15
	VMOVUPS 64(R8), Y0
	VMOVUPS 96(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y2, 96(R8)

	// Group 2: indices 16-19 with 20-23
	VMOVUPS 128(R8), Y0
	VMOVUPS 160(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y2, 160(R8)

	// Group 3: indices 24-27 with 28-31
	VMOVUPS 192(R8), Y0
	VMOVUPS 224(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 192(R8)
	VMOVUPS Y2, 224(R8)

	// Group 4: indices 32-35 with 36-39
	VMOVUPS 256(R8), Y0
	VMOVUPS 288(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y2, 288(R8)

	// Group 5: indices 40-43 with 44-47
	VMOVUPS 320(R8), Y0
	VMOVUPS 352(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y2, 352(R8)

	// Group 6: indices 48-51 with 52-55
	VMOVUPS 384(R8), Y0
	VMOVUPS 416(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y2, 416(R8)

	// Group 7: indices 56-59 with 60-63
	VMOVUPS 448(R8), Y0
	VMOVUPS 480(R8), Y1
	VADDPS Y1, Y0, Y3
	VSUBPS Y1, Y0, Y4
	VMOVSLDUP Y8, Y10
	VMOVSHDUP Y8, Y11
	VSHUFPS $0xB1, Y4, Y4, Y2
	VMULPS Y11, Y2, Y2
	VFMSUBADD231PS Y10, Y4, Y2
	VMOVUPS Y3, 448(R8)
	VMOVUPS Y2, 480(R8)

	// =======================================================================
	// STAGE 5 (IFFT): size=4, step=16
	// Twiddle indices: 0,16 -> twiddles at offsets 0, 128
	// Within-register butterflies: pairs at distance 2 in each YMM
	// =======================================================================

	// Load tw[0] and tw[16], we need them duplicated for within-lane ops
	// tw[0] = 1+0i (identity), tw[16] = -i for standard FFT
	VMOVSD (R10), X8           // tw[0]
	VMOVSD 128(R10), X9        // tw[16]
	VUNPCKLPD X9, X8, X8       // X8 = [tw[0], tw[16]]
	VINSERTI128 $1, X8, Y8, Y8 // Y8 = [tw[0], tw[16], tw[0], tw[16]]

	// For size-4 butterflies within YMM register:
	// Y0 = [a0, a1, a2, a3] pairs (a0,a2) and (a1,a3)
	// Use VPERM2F128 and VPERMILPD to reorganize

	// Process all 16 groups (each YMM holds 4 complex = 1 size-4 block)
	// Group 0: elements 0-3
	VMOVUPS (R8), Y0
	VPERM2F128 $0x01, Y0, Y0, Y1  // Y1 = [a2,a3, a0,a1] - swap 128-bit lanes
	VADDPS Y1, Y0, Y3              // Y3 = [a0+a2, a1+a3, a2+a0, a3+a1]
	VSUBPS Y1, Y0, Y4              // Y4 = [a0-a2, a1-a3, ...]
	// Now apply twiddles: top lane gets tw[0], bottom gets tw[16] (conjugated)
	// Actually we need to select which elements get which twiddle...
	// For size-4: pairs (0,2) use tw[0], pairs (1,3) use tw[16]
	// After VPERM2F128, low lane has (a0+a2,a1+a3), high has (a2+a0,a3+a1)
	// We want output[0]=a0+a2, output[1]=a1+a3*conj(tw16), output[2]=a0-a2, output[3]=(a1-a3)*conj(tw16)
	// This is tricky with the permutation... Let me reconsider

	// Actually for within-register size-4: use VSHUFPS/VBLENDPS patterns
	// Elements: [e0, e1, e2, e3] where e0 pairs with e2, e1 pairs with e3
	VMOVUPS (R8), Y0              // [e0, e1, e2, e3] (each is complex64)
	VSHUFPS $0x4E, Y0, Y0, Y1     // Y1 = [e2, e3, e0, e1] (swap pairs within 128-bit)
	VADDPS Y1, Y0, Y2             // Y2 = [e0+e2, e1+e3, e2+e0, e3+e1]
	VSUBPS Y1, Y0, Y3             // Y3 = [e0-e2, e1-e3, -(e0-e2), -(e1-e3)]
	// Blend: take adds from low positions, subs*twiddle from high
	// For tw[0]=1, no multiply needed. For tw[16], multiply (e1-e3) by conj(tw[16])
	// Build result: [e0+e2, (e1-e3)*conj(tw16), e0-e2, (e1+e3)*conj(tw0)=e1+e3]
	// Wait, this doesn't match standard butterfly pattern...

	// Let me use a simpler approach: explicit index-based butterflies
	// For size-4 stage in IFFT:
	// out[0] = in[0] + in[2]
	// out[2] = (in[0] - in[2]) * conj(tw[0]) = in[0] - in[2]  (tw[0]=1)
	// out[1] = in[1] + in[3]
	// out[3] = (in[1] - in[3]) * conj(tw[16])

	// Using blend approach:
	VMOVUPS (R8), Y0
	VPERMILPS $0x4E, Y0, Y1       // Swap within 64-bit pairs: [e1,e0,e3,e2] wait no
	// VPERMILPS $0x4E = 01 00 11 10 -> indices 1,0,3,2 for floats
	// For complex64 (2 floats each), we want to swap complex pairs
	// VSHUFPS $0x4E for 128-bit: swaps 64-bit halves

	// Actually let me try VSHUFPD which works on 64-bit granularity
	VMOVUPS (R8), Y0              // Y0 = [c0, c1, c2, c3] (complex64)
	VSHUFPD $0x05, Y0, Y0, Y1     // Swap within 128-bit lanes: Y1 = [c1, c0, c3, c2]

	// Hmm, this is getting complex. Let me use a cleaner approach with explicit loads
	// Process pairs individually then combine

	// Alternative: use the same gather/scatter pattern as other stages
	// For size-4, step=16 means pairs at distance 2 complex64 = 16 bytes
	// But within a single YMM (4 complex64), that's elements 0,2 and 1,3

	// Simplest correct approach for remaining stages:
	// Load 2 elements, butterfly, store. Repeat.

	// Elements 0,2
	MOVQ (R8), AX
	MOVQ 16(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2    // e0 + e2
	VSUBPS X1, X0, X3    // e0 - e2 (multiply by tw[0]=1, no-op)
	MOVQ X2, (R8)
	MOVQ X3, 16(R8)

	// Elements 1,3 with tw[16]
	MOVQ 8(R8), AX
	MOVQ 24(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	// Multiply X3 by conj(tw[16])
	VMOVSD 128(R10), X4    // tw[16]
	VMOVSLDUP X4, X5       // re
	VMOVSHDUP X4, X6       // im
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 8(R8)
	MOVQ X7, 24(R8)

	// Elements 4,6
	MOVQ 32(R8), AX
	MOVQ 48(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 32(R8)
	MOVQ X3, 48(R8)

	// Elements 5,7 with tw[16]
	MOVQ 40(R8), AX
	MOVQ 56(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 40(R8)
	MOVQ X7, 56(R8)

	// Elements 8,10
	MOVQ 64(R8), AX
	MOVQ 80(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 64(R8)
	MOVQ X3, 80(R8)

	// Elements 9,11 with tw[16]
	MOVQ 72(R8), AX
	MOVQ 88(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 72(R8)
	MOVQ X7, 88(R8)

	// Elements 12,14
	MOVQ 96(R8), AX
	MOVQ 112(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 96(R8)
	MOVQ X3, 112(R8)

	// Elements 13,15 with tw[16]
	MOVQ 104(R8), AX
	MOVQ 120(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 104(R8)
	MOVQ X7, 120(R8)

	// Continue for remaining 48 elements (12 more pairs of each type)
	// Elements 16,18
	MOVQ 128(R8), AX
	MOVQ 144(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 128(R8)
	MOVQ X3, 144(R8)

	// Elements 17,19 with tw[16]
	MOVQ 136(R8), AX
	MOVQ 152(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 136(R8)
	MOVQ X7, 152(R8)

	// Elements 20,22
	MOVQ 160(R8), AX
	MOVQ 176(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 160(R8)
	MOVQ X3, 176(R8)

	// Elements 21,23 with tw[16]
	MOVQ 168(R8), AX
	MOVQ 184(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 168(R8)
	MOVQ X7, 184(R8)

	// Elements 24,26
	MOVQ 192(R8), AX
	MOVQ 208(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 192(R8)
	MOVQ X3, 208(R8)

	// Elements 25,27 with tw[16]
	MOVQ 200(R8), AX
	MOVQ 216(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 200(R8)
	MOVQ X7, 216(R8)

	// Elements 28,30
	MOVQ 224(R8), AX
	MOVQ 240(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 224(R8)
	MOVQ X3, 240(R8)

	// Elements 29,31 with tw[16]
	MOVQ 232(R8), AX
	MOVQ 248(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 232(R8)
	MOVQ X7, 248(R8)

	// Elements 32,34
	MOVQ 256(R8), AX
	MOVQ 272(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 256(R8)
	MOVQ X3, 272(R8)

	// Elements 33,35 with tw[16]
	MOVQ 264(R8), AX
	MOVQ 280(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 264(R8)
	MOVQ X7, 280(R8)

	// Elements 36,38
	MOVQ 288(R8), AX
	MOVQ 304(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 288(R8)
	MOVQ X3, 304(R8)

	// Elements 37,39 with tw[16]
	MOVQ 296(R8), AX
	MOVQ 312(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 296(R8)
	MOVQ X7, 312(R8)

	// Elements 40,42
	MOVQ 320(R8), AX
	MOVQ 336(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 320(R8)
	MOVQ X3, 336(R8)

	// Elements 41,43 with tw[16]
	MOVQ 328(R8), AX
	MOVQ 344(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 328(R8)
	MOVQ X7, 344(R8)

	// Elements 44,46
	MOVQ 352(R8), AX
	MOVQ 368(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 352(R8)
	MOVQ X3, 368(R8)

	// Elements 45,47 with tw[16]
	MOVQ 360(R8), AX
	MOVQ 376(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 360(R8)
	MOVQ X7, 376(R8)

	// Elements 48,50
	MOVQ 384(R8), AX
	MOVQ 400(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 384(R8)
	MOVQ X3, 400(R8)

	// Elements 49,51 with tw[16]
	MOVQ 392(R8), AX
	MOVQ 408(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 392(R8)
	MOVQ X7, 408(R8)

	// Elements 52,54
	MOVQ 416(R8), AX
	MOVQ 432(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 416(R8)
	MOVQ X3, 432(R8)

	// Elements 53,55 with tw[16]
	MOVQ 424(R8), AX
	MOVQ 440(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 424(R8)
	MOVQ X7, 440(R8)

	// Elements 56,58
	MOVQ 448(R8), AX
	MOVQ 464(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 448(R8)
	MOVQ X3, 464(R8)

	// Elements 57,59 with tw[16]
	MOVQ 456(R8), AX
	MOVQ 472(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 456(R8)
	MOVQ X7, 472(R8)

	// Elements 60,62
	MOVQ 480(R8), AX
	MOVQ 496(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 480(R8)
	MOVQ X3, 496(R8)

	// Elements 61,63 with tw[16]
	MOVQ 488(R8), AX
	MOVQ 504(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	VMOVSD 128(R10), X4
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VSHUFPS $0xB1, X3, X3, X7
	VMULPS X6, X7, X7
	VFMSUBADD231PS X5, X3, X7
	MOVQ X2, 488(R8)
	MOVQ X7, 504(R8)

	// =======================================================================
	// STAGE 6 (IFFT): size=2, identity twiddle
	// Pairs at distance 1: (0,1), (2,3), (4,5), ..., (62,63)
	// Just add/sub, no twiddle multiply
	// =======================================================================

	// Process 32 pairs using scalar operations (could vectorize but simple is safer)
	// Pair 0,1
	MOVQ (R8), AX
	MOVQ 8(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, (R8)
	MOVQ X3, 8(R8)

	// Pair 2,3
	MOVQ 16(R8), AX
	MOVQ 24(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 16(R8)
	MOVQ X3, 24(R8)

	// Pair 4,5
	MOVQ 32(R8), AX
	MOVQ 40(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 32(R8)
	MOVQ X3, 40(R8)

	// Pair 6,7
	MOVQ 48(R8), AX
	MOVQ 56(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 48(R8)
	MOVQ X3, 56(R8)

	// Pair 8,9
	MOVQ 64(R8), AX
	MOVQ 72(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 64(R8)
	MOVQ X3, 72(R8)

	// Pair 10,11
	MOVQ 80(R8), AX
	MOVQ 88(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 80(R8)
	MOVQ X3, 88(R8)

	// Pair 12,13
	MOVQ 96(R8), AX
	MOVQ 104(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 96(R8)
	MOVQ X3, 104(R8)

	// Pair 14,15
	MOVQ 112(R8), AX
	MOVQ 120(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 112(R8)
	MOVQ X3, 120(R8)

	// Pair 16,17
	MOVQ 128(R8), AX
	MOVQ 136(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 128(R8)
	MOVQ X3, 136(R8)

	// Pair 18,19
	MOVQ 144(R8), AX
	MOVQ 152(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 144(R8)
	MOVQ X3, 152(R8)

	// Pair 20,21
	MOVQ 160(R8), AX
	MOVQ 168(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 160(R8)
	MOVQ X3, 168(R8)

	// Pair 22,23
	MOVQ 176(R8), AX
	MOVQ 184(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 176(R8)
	MOVQ X3, 184(R8)

	// Pair 24,25
	MOVQ 192(R8), AX
	MOVQ 200(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 192(R8)
	MOVQ X3, 200(R8)

	// Pair 26,27
	MOVQ 208(R8), AX
	MOVQ 216(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 208(R8)
	MOVQ X3, 216(R8)

	// Pair 28,29
	MOVQ 224(R8), AX
	MOVQ 232(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 224(R8)
	MOVQ X3, 232(R8)

	// Pair 30,31
	MOVQ 240(R8), AX
	MOVQ 248(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 240(R8)
	MOVQ X3, 248(R8)

	// Pair 32,33
	MOVQ 256(R8), AX
	MOVQ 264(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 256(R8)
	MOVQ X3, 264(R8)

	// Pair 34,35
	MOVQ 272(R8), AX
	MOVQ 280(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 272(R8)
	MOVQ X3, 280(R8)

	// Pair 36,37
	MOVQ 288(R8), AX
	MOVQ 296(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 288(R8)
	MOVQ X3, 296(R8)

	// Pair 38,39
	MOVQ 304(R8), AX
	MOVQ 312(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 304(R8)
	MOVQ X3, 312(R8)

	// Pair 40,41
	MOVQ 320(R8), AX
	MOVQ 328(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 320(R8)
	MOVQ X3, 328(R8)

	// Pair 42,43
	MOVQ 336(R8), AX
	MOVQ 344(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 336(R8)
	MOVQ X3, 344(R8)

	// Pair 44,45
	MOVQ 352(R8), AX
	MOVQ 360(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 352(R8)
	MOVQ X3, 360(R8)

	// Pair 46,47
	MOVQ 368(R8), AX
	MOVQ 376(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 368(R8)
	MOVQ X3, 376(R8)

	// Pair 48,49
	MOVQ 384(R8), AX
	MOVQ 392(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 384(R8)
	MOVQ X3, 392(R8)

	// Pair 50,51
	MOVQ 400(R8), AX
	MOVQ 408(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 400(R8)
	MOVQ X3, 408(R8)

	// Pair 52,53
	MOVQ 416(R8), AX
	MOVQ 424(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 416(R8)
	MOVQ X3, 424(R8)

	// Pair 54,55
	MOVQ 432(R8), AX
	MOVQ 440(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 432(R8)
	MOVQ X3, 440(R8)

	// Pair 56,57
	MOVQ 448(R8), AX
	MOVQ 456(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 448(R8)
	MOVQ X3, 456(R8)

	// Pair 58,59
	MOVQ 464(R8), AX
	MOVQ 472(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 464(R8)
	MOVQ X3, 472(R8)

	// Pair 60,61
	MOVQ 480(R8), AX
	MOVQ 488(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 480(R8)
	MOVQ X3, 488(R8)

	// Pair 62,63
	MOVQ 496(R8), AX
	MOVQ 504(R8), BX
	MOVQ AX, X0
	MOVQ BX, X1
	VADDPS X1, X0, X2
	VSUBPS X1, X0, X3
	MOVQ X2, 496(R8)
	MOVQ X3, 504(R8)

	// =======================================================================
	// Bit-reversal permutation to dst
	// For IFFT, we apply bit-reversal at the END (opposite of forward)
	// =======================================================================
	MOVQ dst+0(FP), R9       // R9 = dst

	// work[bitrev[i]] -> dst[i]
	// Actually for DIF IFFT, output is bit-reversed, need to unscramble
	// dst[i] = work[bitrev[i]]

	// Unrolled for 64 elements
	MOVQ (R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, (R9)

	MOVQ 8(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 8(R9)

	MOVQ 16(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 16(R9)

	MOVQ 24(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 24(R9)

	MOVQ 32(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 32(R9)

	MOVQ 40(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 40(R9)

	MOVQ 48(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 48(R9)

	MOVQ 56(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 56(R9)

	MOVQ 64(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 64(R9)

	MOVQ 72(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 72(R9)

	MOVQ 80(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 80(R9)

	MOVQ 88(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 88(R9)

	MOVQ 96(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 96(R9)

	MOVQ 104(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 104(R9)

	MOVQ 112(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 112(R9)

	MOVQ 120(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 120(R9)

	MOVQ 128(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 128(R9)

	MOVQ 136(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 136(R9)

	MOVQ 144(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 144(R9)

	MOVQ 152(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 152(R9)

	MOVQ 160(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 160(R9)

	MOVQ 168(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 168(R9)

	MOVQ 176(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 176(R9)

	MOVQ 184(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 184(R9)

	MOVQ 192(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 192(R9)

	MOVQ 200(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 200(R9)

	MOVQ 208(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 208(R9)

	MOVQ 216(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 216(R9)

	MOVQ 224(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 224(R9)

	MOVQ 232(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 232(R9)

	MOVQ 240(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 240(R9)

	MOVQ 248(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 248(R9)

	MOVQ 256(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 256(R9)

	MOVQ 264(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 264(R9)

	MOVQ 272(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 272(R9)

	MOVQ 280(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 280(R9)

	MOVQ 288(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 288(R9)

	MOVQ 296(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 296(R9)

	MOVQ 304(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 304(R9)

	MOVQ 312(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 312(R9)

	MOVQ 320(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 320(R9)

	MOVQ 328(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 328(R9)

	MOVQ 336(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 336(R9)

	MOVQ 344(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 344(R9)

	MOVQ 352(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 352(R9)

	MOVQ 360(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 360(R9)

	MOVQ 368(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 368(R9)

	MOVQ 376(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 376(R9)

	MOVQ 384(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 384(R9)

	MOVQ 392(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 392(R9)

	MOVQ 400(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 400(R9)

	MOVQ 408(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 408(R9)

	MOVQ 416(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 416(R9)

	MOVQ 424(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 424(R9)

	MOVQ 432(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 432(R9)

	MOVQ 440(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 440(R9)

	MOVQ 448(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 448(R9)

	MOVQ 456(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 456(R9)

	MOVQ 464(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 464(R9)

	MOVQ 472(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 472(R9)

	MOVQ 480(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 480(R9)

	MOVQ 488(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 488(R9)

	MOVQ 496(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 496(R9)

	MOVQ 504(R12), DX
	MOVQ (R8)(DX*8), AX
	MOVQ AX, 504(R9)

inv_size64_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_size64_return_false:
	MOVB $0, ret+120(FP)
	RET
