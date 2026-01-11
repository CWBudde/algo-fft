//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-64 Radix-2 FFT Kernels for AMD64
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
TEXT ·ForwardAVX2Size64Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	LEAQ ·bitrev64_r2(SB), R12 // R12 = internal bitrev table
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
	// Twiddle factors: j=0..3 use [0,4,8,12], j=4..7 use [16,20,24,28]

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 32(R10), X9         // twiddle[4]
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw4]
	VMOVSD 64(R10), X9         // twiddle[8]
	VMOVSD 96(R10), X10        // twiddle[12]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw8, tw12]
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw4, tw8, tw12]

	VMOVSD 128(R10), X9        // twiddle[16]
	VMOVSD 160(R10), X10       // twiddle[20]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw16, tw20]
	VMOVSD 192(R10), X10       // twiddle[24]
	VMOVSD 224(R10), X11       // twiddle[28]
	VPUNPCKLQDQ X11, X10, X10  // X10 = [tw24, tw28]
	VINSERTF128 $1, X10, Y9, Y9 // Y9 = [tw16, tw20, tw24, tw28]

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
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
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
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
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
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
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
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
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
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
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
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
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
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
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
	// Twiddle factors: j=0..3 -> [0,2,4,6], j=4..7 -> [8,10,12,14],
	//                   j=8..11 -> [16,18,20,22], j=12..15 -> [24,26,28,30]

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

	// Load twiddles for indices 8-11: tw[16,18,20,22]
	VMOVSD 128(R10), X10       // twiddle[16]
	VMOVSD 144(R10), X11       // twiddle[18]
	VPUNPCKLQDQ X11, X10, X10
	VMOVSD 160(R10), X11       // twiddle[20]
	VMOVSD 176(R10), X12       // twiddle[22]
	VPUNPCKLQDQ X12, X11, X11
	VINSERTF128 $1, X11, Y10, Y10 // Y10 = [tw16, tw18, tw20, tw22]

	// Load twiddles for indices 12-15: tw[24,26,28,30]
	VMOVSD 192(R10), X11       // twiddle[24]
	VMOVSD 208(R10), X12       // twiddle[26]
	VPUNPCKLQDQ X12, X11, X11
	VMOVSD 224(R10), X12       // twiddle[28]
	VMOVSD 240(R10), X13       // twiddle[30]
	VPUNPCKLQDQ X13, X12, X12
	VINSERTF128 $1, X12, Y11, Y11 // Y11 = [tw24, tw26, tw28, tw30]

	// Group 1: indices 0-3 with 16-19
	VMOVUPS (R8), Y0
	VMOVUPS 128(R8), Y1
	VMOVSLDUP Y8, Y12
	VMOVSHDUP Y8, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 128(R8)

	// Group 1: indices 4-7 with 20-23
	VMOVUPS 32(R8), Y0
	VMOVUPS 160(R8), Y1
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 160(R8)

	// Group 1: indices 8-11 with 24-27
	VMOVUPS 64(R8), Y0
	VMOVUPS 192(R8), Y1
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 192(R8)

	// Group 1: indices 12-15 with 28-31
	VMOVUPS 96(R8), Y0
	VMOVUPS 224(R8), Y1
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y4, 224(R8)

	// Group 2: indices 32-35 with 48-51
	VMOVUPS 256(R8), Y0
	VMOVUPS 384(R8), Y1
	VMOVSLDUP Y8, Y12
	VMOVSHDUP Y8, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 384(R8)

	// Group 2: indices 36-39 with 52-55
	VMOVUPS 288(R8), Y0
	VMOVUPS 416(R8), Y1
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y4, 416(R8)

	// Group 2: indices 40-43 with 56-59
	VMOVUPS 320(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y4, 448(R8)

	// Group 2: indices 44-47 with 60-63
	VMOVUPS 352(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMADDSUB231PS Y12, Y1, Y2
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
	MOVB $1, ret+96(FP)
	RET

size64_return_false:
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 64, complex64
// Fully unrolled 6-stage IFFT with AVX2/FMA vectorization (DIT).
//
// This kernel mirrors the forward DIT schedule and applies conjugated
// twiddle factors during each butterfly. Inputs are bit-reversed at the start,
// and the output is scaled by 1/64.
//
TEXT ·InverseAVX2Size64Radix2Complex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	LEAQ ·bitrev64_r2(SB), R12 // R12 = internal bitrev table
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

	// Select working buffer
	CMPQ R8, R9
	JNE  inv_size64_use_dst

	// In-place: use scratch
	MOVQ R11, R8
	JMP  inv_size64_bitrev

inv_size64_use_dst:
	// Out-of-place: use dst

inv_size64_bitrev:
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y3, Y2, Y5
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
	VFMSUBADD231PS Y14, Y1, Y2 // Y2 = t = w * b
	VADDPS Y2, Y0, Y3          // Y3 = a + t
	VSUBPS Y2, Y0, Y4          // Y4 = a - t
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 32(R8)

	// Group: indices 8-11 with 12-15
	VMOVUPS 64(R8), Y0
	VMOVUPS 96(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 96(R8)

	// Group: indices 16-19 with 20-23
	VMOVUPS 128(R8), Y0
	VMOVUPS 160(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 160(R8)

	// Group: indices 24-27 with 28-31
	VMOVUPS 192(R8), Y0
	VMOVUPS 224(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 192(R8)
	VMOVUPS Y4, 224(R8)

	// Group: indices 32-35 with 36-39
	VMOVUPS 256(R8), Y0
	VMOVUPS 288(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 288(R8)

	// Group: indices 40-43 with 44-47
	VMOVUPS 320(R8), Y0
	VMOVUPS 352(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y4, 352(R8)

	// Group: indices 48-51 with 52-55
	VMOVUPS 384(R8), Y0
	VMOVUPS 416(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y4, 416(R8)

	// Group: indices 56-59 with 60-63
	VMOVUPS 448(R8), Y0
	VMOVUPS 480(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 448(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 4: size=16, half=8, step=4
	// =======================================================================
	// Pairs: indices 0-7 with 8-15, 16-23 with 24-31, 32-39 with 40-47, 48-55 with 56-63
	// Twiddle factors: j=0..3 use [0,4,8,12], j=4..7 use [16,20,24,28]

	VMOVSD (R10), X8           // twiddle[0]
	VMOVSD 32(R10), X9         // twiddle[4]
	VPUNPCKLQDQ X9, X8, X8     // X8 = [tw0, tw4]
	VMOVSD 64(R10), X9         // twiddle[8]
	VMOVSD 96(R10), X10        // twiddle[12]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw8, tw12]
	VINSERTF128 $1, X9, Y8, Y8 // Y8 = [tw0, tw4, tw8, tw12]

	VMOVSD 128(R10), X9        // twiddle[16]
	VMOVSD 160(R10), X10       // twiddle[20]
	VPUNPCKLQDQ X10, X9, X9    // X9 = [tw16, tw20]
	VMOVSD 192(R10), X10       // twiddle[24]
	VMOVSD 224(R10), X11       // twiddle[28]
	VPUNPCKLQDQ X11, X10, X10  // X10 = [tw24, tw28]
	VINSERTF128 $1, X10, Y9, Y9 // Y9 = [tw16, tw20, tw24, tw28]

	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15

	// Group 1: indices 0-3 with 8-11 (first half of 16-point group)
	VMOVUPS (R8), Y0
	VMOVUPS 64(R8), Y1
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 64(R8)

	// Group 1: indices 4-7 with 12-15
	VMOVUPS 32(R8), Y0
	VMOVUPS 96(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 96(R8)

	// Group 2: indices 16-19 with 24-27
	VMOVUPS 128(R8), Y0
	VMOVUPS 192(R8), Y1
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 128(R8)
	VMOVUPS Y4, 192(R8)

	// Group 2: indices 20-23 with 28-31
	VMOVUPS 160(R8), Y0
	VMOVUPS 224(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 160(R8)
	VMOVUPS Y4, 224(R8)

	// Group 3: indices 32-35 with 40-43
	VMOVUPS 256(R8), Y0
	VMOVUPS 320(R8), Y1
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 320(R8)

	// Group 3: indices 36-39 with 44-47
	VMOVUPS 288(R8), Y0
	VMOVUPS 352(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y4, 352(R8)

	// Group 4: indices 48-51 with 56-59
	VMOVUPS 384(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y8, Y14
	VMOVSHDUP Y8, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 384(R8)
	VMOVUPS Y4, 448(R8)

	// Group 4: indices 52-55 with 60-63
	VMOVUPS 416(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y9, Y14
	VMOVSHDUP Y9, Y15
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y15, Y2, Y2
	VFMSUBADD231PS Y14, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 416(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// STAGE 5: size=32, half=16, step=2
	// =======================================================================
	// Pairs: indices 0-15 with 16-31, 32-47 with 48-63
	// Twiddle factors: j=0..3 -> [0,2,4,6], j=4..7 -> [8,10,12,14],
	//                   j=8..11 -> [16,18,20,22], j=12..15 -> [24,26,28,30]

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

	// Load twiddles for indices 8-11: tw[16,18,20,22]
	VMOVSD 128(R10), X10       // twiddle[16]
	VMOVSD 144(R10), X11       // twiddle[18]
	VPUNPCKLQDQ X11, X10, X10
	VMOVSD 160(R10), X11       // twiddle[20]
	VMOVSD 176(R10), X12       // twiddle[22]
	VPUNPCKLQDQ X12, X11, X11
	VINSERTF128 $1, X11, Y10, Y10 // Y10 = [tw16, tw18, tw20, tw22]

	// Load twiddles for indices 12-15: tw[24,26,28,30]
	VMOVSD 192(R10), X11       // twiddle[24]
	VMOVSD 208(R10), X12       // twiddle[26]
	VPUNPCKLQDQ X12, X11, X11
	VMOVSD 224(R10), X12       // twiddle[28]
	VMOVSD 240(R10), X13       // twiddle[30]
	VPUNPCKLQDQ X13, X12, X12
	VINSERTF128 $1, X12, Y11, Y11 // Y11 = [tw24, tw26, tw28, tw30]

	// Group 1: indices 0-3 with 16-19
	VMOVUPS (R8), Y0
	VMOVUPS 128(R8), Y1
	VMOVSLDUP Y8, Y12
	VMOVSHDUP Y8, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, (R8)
	VMOVUPS Y4, 128(R8)

	// Group 1: indices 4-7 with 20-23
	VMOVUPS 32(R8), Y0
	VMOVUPS 160(R8), Y1
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 32(R8)
	VMOVUPS Y4, 160(R8)

	// Group 1: indices 8-11 with 24-27
	VMOVUPS 64(R8), Y0
	VMOVUPS 192(R8), Y1
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 64(R8)
	VMOVUPS Y4, 192(R8)

	// Group 1: indices 12-15 with 28-31
	VMOVUPS 96(R8), Y0
	VMOVUPS 224(R8), Y1
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 96(R8)
	VMOVUPS Y4, 224(R8)

	// Group 2: indices 32-35 with 48-51
	VMOVUPS 256(R8), Y0
	VMOVUPS 384(R8), Y1
	VMOVSLDUP Y8, Y12
	VMOVSHDUP Y8, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 256(R8)
	VMOVUPS Y4, 384(R8)

	// Group 2: indices 36-39 with 52-55
	VMOVUPS 288(R8), Y0
	VMOVUPS 416(R8), Y1
	VMOVSLDUP Y9, Y12
	VMOVSHDUP Y9, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 288(R8)
	VMOVUPS Y4, 416(R8)

	// Group 2: indices 40-43 with 56-59
	VMOVUPS 320(R8), Y0
	VMOVUPS 448(R8), Y1
	VMOVSLDUP Y10, Y12
	VMOVSHDUP Y10, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 320(R8)
	VMOVUPS Y4, 448(R8)

	// Group 2: indices 44-47 with 60-63
	VMOVUPS 352(R8), Y0
	VMOVUPS 480(R8), Y1
	VMOVSLDUP Y11, Y12
	VMOVSHDUP Y11, Y13
	VSHUFPS $0xB1, Y1, Y1, Y2
	VMULPS Y13, Y2, Y2
	VFMSUBADD231PS Y12, Y1, Y2
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
	VFMSUBADD231PS Y10, Y1, Y2
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
	VFMSUBADD231PS Y10, Y1, Y2
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
	VFMSUBADD231PS Y10, Y1, Y2
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
	VFMSUBADD231PS Y10, Y1, Y2
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
	VFMSUBADD231PS Y10, Y1, Y2
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
	VFMSUBADD231PS Y10, Y1, Y2
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
	VFMSUBADD231PS Y10, Y1, Y2
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
	VFMSUBADD231PS Y10, Y1, Y2
	VADDPS Y2, Y0, Y3
	VSUBPS Y2, Y0, Y4
	VMOVUPS Y3, 224(R8)
	VMOVUPS Y4, 480(R8)

	// =======================================================================
	// Apply 1/N scaling for inverse transform (1/64)
	// =======================================================================
	MOVL ·sixtyFourth32(SB), AX
	MOVD AX, X8
	VBROADCASTSS X8, Y8

	XORQ CX, CX
inv_size64_scale_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMULPS Y8, Y0, Y0
	VMOVUPS Y0, (R8)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $512
	JL   inv_size64_scale_loop

	// =======================================================================
	// Copy results to dst if we used scratch buffer
	// =======================================================================
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   inv_size64_done

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

inv_size64_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

inv_size64_return_false:
	MOVB $0, ret+96(FP)
	RET
