//go:build amd64 && asm && !purego

#include "textflag.h"

// ===========================================================================
// AVX2 Size-512 Radix-8 FFT (complex64) Kernels for AMD64
// ===========================================================================
//
// 512 = 8 * 8 * 8 (3 stages).
// Each stage consists of 64 groups of 8-point DFTs.
//
// Register usage strategy:
// Y0-Y7:  8 complex64 vectors (each 4 values) representing 4 parallel 8-pt butterflies.
// Y8-Y15: Scratch and twiddles.
// ===========================================================================

// Forward transform, size 512, complex64, radix-8 DIT
TEXT ·ForwardAVX2Size512Radix8Complex64Asm(SB), $4096-97
	MOVQ dst+0(FP), R8               // R8 = dst pointer
	MOVQ src+24(FP), R9              // R9 = src pointer
	MOVQ twiddle+48(FP), R10         // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11         // R11 = scratch pointer
	MOVQ src+32(FP), R13             // R13 = len(src)
	LEAQ ·bitrev512_r8(SB), R12      // R12 = bitrev table

	CMPQ R13, $512                   // len(src) < 512?
	JL   size512_r8_fwd_return_false // return false
	MOVQ dst+8(FP), AX               // AX = len(dst)
	CMPQ AX, $512                    // len(dst) < 512?
	JL   size512_r8_fwd_return_false // return false
	MOVQ twiddle+56(FP), AX          // AX = len(twiddle)
	CMPQ AX, $512                    // len(twiddle) < 512?
	JL   size512_r8_fwd_return_false // return false
	MOVQ scratch+80(FP), AX          // AX = len(scratch)
	CMPQ AX, $512                    // len(scratch) < 512?
	JL   size512_r8_fwd_return_false // return false

	CMPQ R8, R9                      // dst == src?
	JNE  size512_r8_fwd_use_dst      // out-of-place
	MOVQ R11, R8                     // work = scratch

size512_r8_fwd_use_dst:
	LEAQ 0(SP), R14                  // R14 = stage1 base pointer

	MOVL ·signbit32(SB), AX          // AX = float32 sign bit
	MOVD AX, X12                     // X12 = sign bit scalar
	VBROADCASTSS X12, X12            // X12 = [sign, sign, sign, sign]
	VXORPS X0, X0, X0                // X0 = 0
	VBLENDPS $0xAA, X12, X0, X12     // X12 = [0, sign, 0, sign]

	VMOVSD 512(R10), X13             // X13 = w1_8
	VMOVSD 1024(R10), X14            // X14 = w2_8
	VMOVSD 1536(R10), X15            // X15 = w3_8

	XORQ CX, CX                      // base = 0

size512_r8_fwd_stage1_loop:
	CMPQ CX, $512                    // base >= 512?
	JGE  size512_r8_fwd_stage2       // next stage
	LEAQ (R12)(CX*8), R15            // R15 = &bitrev[base]

	MOVQ 0(R15), AX                  // AX = bitrev[base]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X0            // X0 = x0
	MOVQ 8(R15), AX                  // AX = bitrev[base+1]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X1            // X1 = x1
	MOVQ 16(R15), AX                 // AX = bitrev[base+2]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X2            // X2 = x2
	MOVQ 24(R15), AX                 // AX = bitrev[base+3]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X3            // X3 = x3
	MOVQ 32(R15), AX                 // AX = bitrev[base+4]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X4            // X4 = x4
	MOVQ 40(R15), AX                 // AX = bitrev[base+5]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X5            // X5 = x5
	MOVQ 48(R15), AX                 // AX = bitrev[base+6]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X6            // X6 = x6
	MOVQ 56(R15), AX                 // AX = bitrev[base+7]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X7            // X7 = x7

	VADDPS X4, X0, X8                // X8 = a0 = x0 + x4
	VSUBPS X4, X0, X9                // X9 = a1 = x0 - x4
	VADDPS X6, X2, X10               // X10 = a2 = x2 + x6
	VSUBPS X6, X2, X2                // X2 = a3 = x2 - x6
	VADDPS X5, X1, X11               // X11 = a4 = x1 + x5
	VSUBPS X5, X1, X4                // X4 = a5 = x1 - x5
	VADDPS X7, X3, X1                // X1 = a6 = x3 + x7
	VSUBPS X7, X3, X5                // X5 = a7 = x3 - x7

	VADDPS X10, X8, X0               // X0 = e0 = a0 + a2
	VSUBPS X10, X8, X6               // X6 = e2 = a0 - a2
	VSHUFPS $0xB1, X2, X2, X7        // X7 = swap(a3)
	VXORPS X12, X7, X7               // X7 = a3 * (-i)
	VADDPS X7, X9, X10               // X10 = e1 = a1 + (-i)*a3
	VSUBPS X7, X9, X2                // X2 = e3 = a1 - (-i)*a3
	VADDPS X1, X11, X9               // X9 = o0 = a4 + a6
	VSUBPS X1, X11, X11              // X11 = o2 = a4 - a6
	VSHUFPS $0xB1, X5, X5, X7        // X7 = swap(a7)
	VXORPS X12, X7, X7               // X7 = a7 * (-i)
	VADDPS X7, X4, X1                // X1 = o1 = a5 + (-i)*a7
	VSUBPS X7, X4, X4                // X4 = o3 = a5 - (-i)*a7

	VADDPS X9, X0, X5                // X5 = y0 = e0 + o0
	VSUBPS X9, X0, X7                // X7 = y4 = e0 - o0

	VMOVSLDUP X13, X8                // X8 = w1_8.real
	VMOVSHDUP X13, X9                // X9 = w1_8.imag
	VSHUFPS $0xB1, X1, X1, X3        // X3 = swap(o1)
	VMULPS X9, X3, X3                // X3 = o1.im * w1_8.imag
	VFMADDSUB231PS X8, X1, X3        // X3 = w1_8 * o1
	VADDPS X3, X10, X8               // X8 = y1 = e1 + w1_8*o1
	VSUBPS X3, X10, X9               // X9 = y5 = e1 - w1_8*o1

	VMOVSLDUP X14, X0                // X0 = w2_8.real
	VMOVSHDUP X14, X10               // X10 = w2_8.imag
	VSHUFPS $0xB1, X11, X11, X3      // X3 = swap(o2)
	VMULPS X10, X3, X3               // X3 = o2.im * w2_8.imag
	VFMADDSUB231PS X0, X11, X3       // X3 = w2_8 * o2
	VADDPS X3, X6, X1                // X1 = y2 = e2 + w2_8*o2
	VSUBPS X3, X6, X11               // X11 = y6 = e2 - w2_8*o2

	VMOVSLDUP X15, X0                // X0 = w3_8.real
	VMOVSHDUP X15, X10               // X10 = w3_8.imag
	VSHUFPS $0xB1, X4, X4, X3        // X3 = swap(o3)
	VMULPS X10, X3, X3               // X3 = o3.im * w3_8.imag
	VFMADDSUB231PS X0, X4, X3        // X3 = w3_8 * o3
	VADDPS X3, X2, X6                // X6 = y3 = e3 + w3_8*o3
	VSUBPS X3, X2, X4                // X4 = y7 = e3 - w3_8*o3

	MOVQ CX, BX                      // BX = base
	SHLQ $3, BX                      // BX = base*8
	VMOVSD X5, 0(R14)(BX*1)          // stage1[base] = y0
	VMOVSD X7, 32(R14)(BX*1)         // stage1[base+4] = y4
	VMOVSD X8, 8(R14)(BX*1)          // stage1[base+1] = y1
	VMOVSD X9, 40(R14)(BX*1)         // stage1[base+5] = y5
	VMOVSD X1, 16(R14)(BX*1)         // stage1[base+2] = y2
	VMOVSD X11, 48(R14)(BX*1)        // stage1[base+6] = y6
	VMOVSD X6, 24(R14)(BX*1)         // stage1[base+3] = y3
	VMOVSD X4, 56(R14)(BX*1)         // stage1[base+7] = y7

	ADDQ $8, CX                      // base += 8
	JMP  size512_r8_fwd_stage1_loop  // continue stage1

size512_r8_fwd_stage2:
	XORQ CX, CX                      // base = 0

size512_r8_fwd_stage2_outer:
	CMPQ CX, $512                    // base >= 512?
	JGE  size512_r8_fwd_stage3       // next stage
	XORQ DX, DX                      // j = 0

size512_r8_fwd_stage2_inner:
	CMPQ DX, $8                      // j >= 8?
	JGE  size512_r8_fwd_stage2_next  // next base
	MOVQ CX, BX                      // BX = base
	ADDQ DX, BX                      // BX = base + j
	MOVQ BX, SI                      // SI = base + j
	SHLQ $3, SI                      // SI = (base+j)*8

	VMOVSD 0(R14)(SI*1), X0          // X0 = x0
	VMOVSD 64(R14)(SI*1), X1         // X1 = x1
	MOVQ DX, AX                      // AX = j
	IMULQ $64, AX                    // AX = j*64 bytes
	VMOVSD (R10)(AX*1), X3           // X3 = tw1
	VMOVSLDUP X3, X8                 // X8 = tw1.real
	VMOVSHDUP X3, X9                 // X9 = tw1.imag
	VSHUFPS $0xB1, X1, X1, X11       // X11 = swap(x1)
	VMULPS X9, X11, X11              // X11 = x1.im * tw1.imag
	VFMADDSUB231PS X8, X1, X11       // X11 = tw1 * x1
	VMOVAPS X11, X1                  // X1 = tw1 * x1

	VMOVSD 128(R14)(SI*1), X2        // X2 = x2
	MOVQ DX, AX                      // AX = j
	IMULQ $128, AX                   // AX = j*128 bytes
	VMOVSD (R10)(AX*1), X3           // X3 = tw2
	VMOVSLDUP X3, X8                 // X8 = tw2.real
	VMOVSHDUP X3, X9                 // X9 = tw2.imag
	VSHUFPS $0xB1, X2, X2, X11       // X11 = swap(x2)
	VMULPS X9, X11, X11              // X11 = x2.im * tw2.imag
	VFMADDSUB231PS X8, X2, X11       // X11 = tw2 * x2
	VMOVAPS X11, X2                  // X2 = tw2 * x2

	VMOVSD 192(R14)(SI*1), X3        // X3 = x3
	MOVQ DX, AX                      // AX = j
	IMULQ $192, AX                   // AX = j*192 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw3
	VMOVSLDUP X8, X9                 // X9 = tw3.real
	VMOVSHDUP X8, X10                // X10 = tw3.imag
	VSHUFPS $0xB1, X3, X3, X11       // X11 = swap(x3)
	VMULPS X10, X11, X11             // X11 = x3.im * tw3.imag
	VFMADDSUB231PS X9, X3, X11       // X11 = tw3 * x3
	VMOVAPS X11, X3                  // X3 = tw3 * x3

	VMOVSD 256(R14)(SI*1), X4        // X4 = x4
	MOVQ DX, AX                      // AX = j
	IMULQ $256, AX                   // AX = j*256 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw4
	VMOVSLDUP X8, X9                 // X9 = tw4.real
	VMOVSHDUP X8, X10                // X10 = tw4.imag
	VSHUFPS $0xB1, X4, X4, X11       // X11 = swap(x4)
	VMULPS X10, X11, X11             // X11 = x4.im * tw4.imag
	VFMADDSUB231PS X9, X4, X11       // X11 = tw4 * x4
	VMOVAPS X11, X4                  // X4 = tw4 * x4

	VMOVSD 320(R14)(SI*1), X5        // X5 = x5
	MOVQ DX, AX                      // AX = j
	IMULQ $320, AX                   // AX = j*320 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw5
	VMOVSLDUP X8, X9                 // X9 = tw5.real
	VMOVSHDUP X8, X10                // X10 = tw5.imag
	VSHUFPS $0xB1, X5, X5, X11       // X11 = swap(x5)
	VMULPS X10, X11, X11             // X11 = x5.im * tw5.imag
	VFMADDSUB231PS X9, X5, X11       // X11 = tw5 * x5
	VMOVAPS X11, X5                  // X5 = tw5 * x5

	VMOVSD 384(R14)(SI*1), X6        // X6 = x6
	MOVQ DX, AX                      // AX = j
	IMULQ $384, AX                   // AX = j*384 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw6
	VMOVSLDUP X8, X9                 // X9 = tw6.real
	VMOVSHDUP X8, X10                // X10 = tw6.imag
	VSHUFPS $0xB1, X6, X6, X11       // X11 = swap(x6)
	VMULPS X10, X11, X11             // X11 = x6.im * tw6.imag
	VFMADDSUB231PS X9, X6, X11       // X11 = tw6 * x6
	VMOVAPS X11, X6                  // X6 = tw6 * x6

	VMOVSD 448(R14)(SI*1), X7        // X7 = x7
	MOVQ DX, AX                      // AX = j
	IMULQ $448, AX                   // AX = j*448 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw7
	VMOVSLDUP X8, X9                 // X9 = tw7.real
	VMOVSHDUP X8, X10                // X10 = tw7.imag
	VSHUFPS $0xB1, X7, X7, X11       // X11 = swap(x7)
	VMULPS X10, X11, X11             // X11 = x7.im * tw7.imag
	VFMADDSUB231PS X9, X7, X11       // X11 = tw7 * x7
	VMOVAPS X11, X7                  // X7 = tw7 * x7

	VADDPS X4, X0, X8                // X8 = a0 = x0 + x4
	VSUBPS X4, X0, X9                // X9 = a1 = x0 - x4
	VADDPS X6, X2, X10               // X10 = a2 = x2 + x6
	VSUBPS X6, X2, X2                // X2 = a3 = x2 - x6
	VADDPS X5, X1, X11               // X11 = a4 = x1 + x5
	VSUBPS X5, X1, X4                // X4 = a5 = x1 - x5
	VADDPS X7, X3, X1                // X1 = a6 = x3 + x7
	VSUBPS X7, X3, X5                // X5 = a7 = x3 - x7

	VADDPS X10, X8, X0               // X0 = e0 = a0 + a2
	VSUBPS X10, X8, X6               // X6 = e2 = a0 - a2
	VSHUFPS $0xB1, X2, X2, X7        // X7 = swap(a3)
	VXORPS X12, X7, X7               // X7 = a3 * (-i)
	VADDPS X7, X9, X10               // X10 = e1 = a1 + (-i)*a3
	VSUBPS X7, X9, X2                // X2 = e3 = a1 - (-i)*a3
	VADDPS X1, X11, X9               // X9 = o0 = a4 + a6
	VSUBPS X1, X11, X11              // X11 = o2 = a4 - a6
	VSHUFPS $0xB1, X5, X5, X7        // X7 = swap(a7)
	VXORPS X12, X7, X7               // X7 = a7 * (-i)
	VADDPS X7, X4, X1                // X1 = o1 = a5 + (-i)*a7
	VSUBPS X7, X4, X4                // X4 = o3 = a5 - (-i)*a7

	VADDPS X9, X0, X5                // X5 = y0 = e0 + o0
	VSUBPS X9, X0, X7                // X7 = y4 = e0 - o0

	VMOVSLDUP X13, X8                // X8 = w1_8.real
	VMOVSHDUP X13, X9                // X9 = w1_8.imag
	VSHUFPS $0xB1, X1, X1, X3        // X3 = swap(o1)
	VMULPS X9, X3, X3                // X3 = o1.im * w1_8.imag
	VFMADDSUB231PS X8, X1, X3        // X3 = w1_8 * o1
	VADDPS X3, X10, X8               // X8 = y1 = e1 + w1_8*o1
	VSUBPS X3, X10, X9               // X9 = y5 = e1 - w1_8*o1

	VMOVSLDUP X14, X0                // X0 = w2_8.real
	VMOVSHDUP X14, X10               // X10 = w2_8.imag
	VSHUFPS $0xB1, X11, X11, X3      // X3 = swap(o2)
	VMULPS X10, X3, X3               // X3 = o2.im * w2_8.imag
	VFMADDSUB231PS X0, X11, X3       // X3 = w2_8 * o2
	VADDPS X3, X6, X1                // X1 = y2 = e2 + w2_8*o2
	VSUBPS X3, X6, X11               // X11 = y6 = e2 - w2_8*o2

	VMOVSLDUP X15, X0                // X0 = w3_8.real
	VMOVSHDUP X15, X10               // X10 = w3_8.imag
	VSHUFPS $0xB1, X4, X4, X3        // X3 = swap(o3)
	VMULPS X10, X3, X3               // X3 = o3.im * w3_8.imag
	VFMADDSUB231PS X0, X4, X3        // X3 = w3_8 * o3
	VADDPS X3, X2, X6                // X6 = y3 = e3 + w3_8*o3
	VSUBPS X3, X2, X4                // X4 = y7 = e3 - w3_8*o3

	MOVQ BX, DI                      // DI = base + j
	SHLQ $3, DI                      // DI = (base+j)*8
	VMOVSD X5, 0(R8)(DI*1)           // stage2[base+j] = y0
	VMOVSD X7, 256(R8)(DI*1)         // stage2[base+j+32] = y4
	VMOVSD X8, 64(R8)(DI*1)          // stage2[base+j+8] = y1
	VMOVSD X9, 320(R8)(DI*1)         // stage2[base+j+40] = y5
	VMOVSD X1, 128(R8)(DI*1)         // stage2[base+j+16] = y2
	VMOVSD X11, 384(R8)(DI*1)        // stage2[base+j+48] = y6
	VMOVSD X6, 192(R8)(DI*1)         // stage2[base+j+24] = y3
	VMOVSD X4, 448(R8)(DI*1)         // stage2[base+j+56] = y7

	INCQ DX                          // j++
	JMP  size512_r8_fwd_stage2_inner // continue inner

size512_r8_fwd_stage2_next:
	ADDQ $64, CX                     // base += 64
	JMP  size512_r8_fwd_stage2_outer // continue outer

size512_r8_fwd_stage3:
	XORQ CX, CX                      // j = 0

size512_r8_fwd_stage3_loop:
	CMPQ CX, $64                     // j >= 64?
	JGE  size512_r8_fwd_copy         // copy if needed
	MOVQ CX, SI                      // SI = j
	SHLQ $3, SI                      // SI = j*8

	VMOVSD 0(R8)(SI*1), X0           // X0 = x0
	VMOVSD 512(R8)(SI*1), X1         // X1 = x1
	MOVQ CX, AX                      // AX = j
	IMULQ $8, AX                     // AX = j*8 bytes
	VMOVSD (R10)(AX*1), X3           // X3 = tw1
	VMOVSLDUP X3, X8                 // X8 = tw1.real
	VMOVSHDUP X3, X9                 // X9 = tw1.imag
	VSHUFPS $0xB1, X1, X1, X11       // X11 = swap(x1)
	VMULPS X9, X11, X11              // X11 = x1.im * tw1.imag
	VFMADDSUB231PS X8, X1, X11       // X11 = tw1 * x1
	VMOVAPS X11, X1                  // X1 = tw1 * x1

	VMOVSD 1024(R8)(SI*1), X2        // X2 = x2
	MOVQ CX, AX                      // AX = j
	IMULQ $16, AX                    // AX = j*16 bytes
	VMOVSD (R10)(AX*1), X3           // X3 = tw2
	VMOVSLDUP X3, X8                 // X8 = tw2.real
	VMOVSHDUP X3, X9                 // X9 = tw2.imag
	VSHUFPS $0xB1, X2, X2, X11       // X11 = swap(x2)
	VMULPS X9, X11, X11              // X11 = x2.im * tw2.imag
	VFMADDSUB231PS X8, X2, X11       // X11 = tw2 * x2
	VMOVAPS X11, X2                  // X2 = tw2 * x2

	VMOVSD 1536(R8)(SI*1), X3        // X3 = x3
	MOVQ CX, AX                      // AX = j
	IMULQ $24, AX                    // AX = j*24 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw3
	VMOVSLDUP X8, X9                 // X9 = tw3.real
	VMOVSHDUP X8, X10                // X10 = tw3.imag
	VSHUFPS $0xB1, X3, X3, X11       // X11 = swap(x3)
	VMULPS X10, X11, X11             // X11 = x3.im * tw3.imag
	VFMADDSUB231PS X9, X3, X11       // X11 = tw3 * x3
	VMOVAPS X11, X3                  // X3 = tw3 * x3

	VMOVSD 2048(R8)(SI*1), X4        // X4 = x4
	MOVQ CX, AX                      // AX = j
	IMULQ $32, AX                    // AX = j*32 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw4
	VMOVSLDUP X8, X9                 // X9 = tw4.real
	VMOVSHDUP X8, X10                // X10 = tw4.imag
	VSHUFPS $0xB1, X4, X4, X11       // X11 = swap(x4)
	VMULPS X10, X11, X11             // X11 = x4.im * tw4.imag
	VFMADDSUB231PS X9, X4, X11       // X11 = tw4 * x4
	VMOVAPS X11, X4                  // X4 = tw4 * x4

	VMOVSD 2560(R8)(SI*1), X5        // X5 = x5
	MOVQ CX, AX                      // AX = j
	IMULQ $40, AX                    // AX = j*40 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw5
	VMOVSLDUP X8, X9                 // X9 = tw5.real
	VMOVSHDUP X8, X10                // X10 = tw5.imag
	VSHUFPS $0xB1, X5, X5, X11       // X11 = swap(x5)
	VMULPS X10, X11, X11             // X11 = x5.im * tw5.imag
	VFMADDSUB231PS X9, X5, X11       // X11 = tw5 * x5
	VMOVAPS X11, X5                  // X5 = tw5 * x5

	VMOVSD 3072(R8)(SI*1), X6        // X6 = x6
	MOVQ CX, AX                      // AX = j
	IMULQ $48, AX                    // AX = j*48 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw6
	VMOVSLDUP X8, X9                 // X9 = tw6.real
	VMOVSHDUP X8, X10                // X10 = tw6.imag
	VSHUFPS $0xB1, X6, X6, X11       // X11 = swap(x6)
	VMULPS X10, X11, X11             // X11 = x6.im * tw6.imag
	VFMADDSUB231PS X9, X6, X11       // X11 = tw6 * x6
	VMOVAPS X11, X6                  // X6 = tw6 * x6

	VMOVSD 3584(R8)(SI*1), X7        // X7 = x7
	MOVQ CX, AX                      // AX = j
	IMULQ $56, AX                    // AX = j*56 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw7
	VMOVSLDUP X8, X9                 // X9 = tw7.real
	VMOVSHDUP X8, X10                // X10 = tw7.imag
	VSHUFPS $0xB1, X7, X7, X11       // X11 = swap(x7)
	VMULPS X10, X11, X11             // X11 = x7.im * tw7.imag
	VFMADDSUB231PS X9, X7, X11       // X11 = tw7 * x7
	VMOVAPS X11, X7                  // X7 = tw7 * x7

	VADDPS X4, X0, X8                // X8 = a0 = x0 + x4
	VSUBPS X4, X0, X9                // X9 = a1 = x0 - x4
	VADDPS X6, X2, X10               // X10 = a2 = x2 + x6
	VSUBPS X6, X2, X2                // X2 = a3 = x2 - x6
	VADDPS X5, X1, X11               // X11 = a4 = x1 + x5
	VSUBPS X5, X1, X4                // X4 = a5 = x1 - x5
	VADDPS X7, X3, X1                // X1 = a6 = x3 + x7
	VSUBPS X7, X3, X5                // X5 = a7 = x3 - x7

	VADDPS X10, X8, X0               // X0 = e0 = a0 + a2
	VSUBPS X10, X8, X6               // X6 = e2 = a0 - a2
	VSHUFPS $0xB1, X2, X2, X7        // X7 = swap(a3)
	VXORPS X12, X7, X7               // X7 = a3 * (-i)
	VADDPS X7, X9, X10               // X10 = e1 = a1 + (-i)*a3
	VSUBPS X7, X9, X2                // X2 = e3 = a1 - (-i)*a3
	VADDPS X1, X11, X9               // X9 = o0 = a4 + a6
	VSUBPS X1, X11, X11              // X11 = o2 = a4 - a6
	VSHUFPS $0xB1, X5, X5, X7        // X7 = swap(a7)
	VXORPS X12, X7, X7               // X7 = a7 * (-i)
	VADDPS X7, X4, X1                // X1 = o1 = a5 + (-i)*a7
	VSUBPS X7, X4, X4                // X4 = o3 = a5 - (-i)*a7

	VADDPS X9, X0, X5                // X5 = y0 = e0 + o0
	VSUBPS X9, X0, X7                // X7 = y4 = e0 - o0

	VMOVSLDUP X13, X8                // X8 = w1_8.real
	VMOVSHDUP X13, X9                // X9 = w1_8.imag
	VSHUFPS $0xB1, X1, X1, X3        // X3 = swap(o1)
	VMULPS X9, X3, X3                // X3 = o1.im * w1_8.imag
	VFMADDSUB231PS X8, X1, X3        // X3 = w1_8 * o1
	VADDPS X3, X10, X8               // X8 = y1 = e1 + w1_8*o1
	VSUBPS X3, X10, X9               // X9 = y5 = e1 - w1_8*o1

	VMOVSLDUP X14, X0                // X0 = w2_8.real
	VMOVSHDUP X14, X10               // X10 = w2_8.imag
	VSHUFPS $0xB1, X11, X11, X3      // X3 = swap(o2)
	VMULPS X10, X3, X3               // X3 = o2.im * w2_8.imag
	VFMADDSUB231PS X0, X11, X3       // X3 = w2_8 * o2
	VADDPS X3, X6, X1                // X1 = y2 = e2 + w2_8*o2
	VSUBPS X3, X6, X11               // X11 = y6 = e2 - w2_8*o2

	VMOVSLDUP X15, X0                // X0 = w3_8.real
	VMOVSHDUP X15, X10               // X10 = w3_8.imag
	VSHUFPS $0xB1, X4, X4, X3        // X3 = swap(o3)
	VMULPS X10, X3, X3               // X3 = o3.im * w3_8.imag
	VFMADDSUB231PS X0, X4, X3        // X3 = w3_8 * o3
	VADDPS X3, X2, X6                // X6 = y3 = e3 + w3_8*o3
	VSUBPS X3, X2, X4                // X4 = y7 = e3 - w3_8*o3

	VMOVSD X5, 0(R8)(SI*1)           // work[j] = y0
	VMOVSD X7, 2048(R8)(SI*1)        // work[j+256] = y4
	VMOVSD X8, 512(R8)(SI*1)         // work[j+64] = y1
	VMOVSD X9, 2560(R8)(SI*1)        // work[j+320] = y5
	VMOVSD X1, 1024(R8)(SI*1)        // work[j+128] = y2
	VMOVSD X11, 3072(R8)(SI*1)       // work[j+384] = y6
	VMOVSD X6, 1536(R8)(SI*1)        // work[j+192] = y3
	VMOVSD X4, 3584(R8)(SI*1)        // work[j+448] = y7

	INCQ CX                          // j++
	JMP  size512_r8_fwd_stage3_loop  // continue stage3

size512_r8_fwd_copy:
	MOVQ dst+0(FP), R9               // R9 = dst pointer
	CMPQ R8, R9                      // work == dst?
	JE   size512_r8_fwd_done         // skip copy
	MOVQ R8, SI                      // SI = src (work)
	MOVQ R9, DI                      // DI = dst
	MOVQ $512, CX                    // CX = count (qwords)

size512_r8_fwd_copy_loop:
	MOVQ (SI), AX                    // AX = *src
	MOVQ AX, (DI)                    // *dst = AX
	ADDQ $8, SI                      // src++
	ADDQ $8, DI                      // dst++
	DECQ CX                          // count--
	JNZ  size512_r8_fwd_copy_loop    // loop

size512_r8_fwd_done:
	VZEROUPPER                       // clear upper YMM state
	MOVB $1, ret+96(FP)              // return true
	RET                              // done

size512_r8_fwd_return_false:
	MOVB $0, ret+96(FP)              // return false
	RET                              // done

// Inverse transform, size 512, complex64, radix-8 DIT
TEXT ·InverseAVX2Size512Radix8Complex64Asm(SB), $4096-97
	MOVQ dst+0(FP), R8               // R8 = dst pointer
	MOVQ src+24(FP), R9              // R9 = src pointer
	MOVQ twiddle+48(FP), R10         // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11         // R11 = scratch pointer
	MOVQ src+32(FP), R13             // R13 = len(src)
	LEAQ ·bitrev512_r8(SB), R12      // R12 = bitrev table

	CMPQ R13, $512                   // len(src) < 512?
	JL   size512_r8_inv_return_false // return false
	MOVQ dst+8(FP), AX               // AX = len(dst)
	CMPQ AX, $512                    // len(dst) < 512?
	JL   size512_r8_inv_return_false // return false
	MOVQ twiddle+56(FP), AX          // AX = len(twiddle)
	CMPQ AX, $512                    // len(twiddle) < 512?
	JL   size512_r8_inv_return_false // return false
	MOVQ scratch+80(FP), AX          // AX = len(scratch)
	CMPQ AX, $512                    // len(scratch) < 512?
	JL   size512_r8_inv_return_false // return false

	CMPQ R8, R9                      // dst == src?
	JNE  size512_r8_inv_use_dst      // out-of-place
	MOVQ R11, R8                     // work = scratch

size512_r8_inv_use_dst:
	LEAQ 0(SP), R14                  // R14 = stage1 base pointer

	MOVL ·signbit32(SB), AX          // AX = float32 sign bit
	MOVD AX, X12                     // X12 = sign bit scalar
	VBROADCASTSS X12, X12            // X12 = [sign, sign, sign, sign]
	VXORPS X0, X0, X0                // X0 = 0
	VBLENDPS $0xAA, X12, X0, X12     // X12 = [0, sign, 0, sign]

	VMOVSD 512(R10), X13             // X13 = tw[64]
	VXORPS X12, X13, X13             // X13 = conj(tw[64])
	VMOVSD 1024(R10), X14            // X14 = tw[128]
	VXORPS X12, X14, X14             // X14 = conj(tw[128])
	VMOVSD 1536(R10), X15            // X15 = tw[192]
	VXORPS X12, X15, X15             // X15 = conj(tw[192])

	XORQ CX, CX                      // base = 0

size512_r8_inv_stage1_loop:
	CMPQ CX, $512                    // base >= 512?
	JGE  size512_r8_inv_stage2       // next stage
	LEAQ (R12)(CX*8), R15            // R15 = &bitrev[base]

	MOVQ 0(R15), AX                  // AX = bitrev[base]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X0            // X0 = x0
	MOVQ 8(R15), AX                  // AX = bitrev[base+1]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X1            // X1 = x1
	MOVQ 16(R15), AX                 // AX = bitrev[base+2]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X2            // X2 = x2
	MOVQ 24(R15), AX                 // AX = bitrev[base+3]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X3            // X3 = x3
	MOVQ 32(R15), AX                 // AX = bitrev[base+4]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X4            // X4 = x4
	MOVQ 40(R15), AX                 // AX = bitrev[base+5]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X5            // X5 = x5
	MOVQ 48(R15), AX                 // AX = bitrev[base+6]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X6            // X6 = x6
	MOVQ 56(R15), AX                 // AX = bitrev[base+7]
	SHLQ $3, AX                      // AX = byte offset
	VMOVSD (R9)(AX*1), X7            // X7 = x7

	VADDPS X4, X0, X8                // X8 = a0 = x0 + x4
	VSUBPS X4, X0, X9                // X9 = a1 = x0 - x4
	VADDPS X6, X2, X10               // X10 = a2 = x2 + x6
	VSUBPS X6, X2, X2                // X2 = a3 = x2 - x6
	VADDPS X5, X1, X11               // X11 = a4 = x1 + x5
	VSUBPS X5, X1, X4                // X4 = a5 = x1 - x5
	VADDPS X7, X3, X1                // X1 = a6 = x3 + x7
	VSUBPS X7, X3, X5                // X5 = a7 = x3 - x7

	VADDPS X10, X8, X0               // X0 = e0 = a0 + a2
	VSUBPS X10, X8, X6               // X6 = e2 = a0 - a2
	VSHUFPS $0xB1, X2, X2, X7        // X7 = swap(a3)
	VXORPS X12, X7, X7               // X7 = a3 * (-i)
	VSUBPS X7, X9, X10               // X10 = e1 = a1 - (-i)*a3
	VADDPS X7, X9, X2                // X2 = e3 = a1 + (-i)*a3
	VADDPS X1, X11, X9               // X9 = o0 = a4 + a6
	VSUBPS X1, X11, X11              // X11 = o2 = a4 - a6
	VSHUFPS $0xB1, X5, X5, X7        // X7 = swap(a7)
	VXORPS X12, X7, X7               // X7 = a7 * (-i)
	VSUBPS X7, X4, X1                // X1 = o1 = a5 - (-i)*a7
	VADDPS X7, X4, X4                // X4 = o3 = a5 + (-i)*a7

	VADDPS X9, X0, X5                // X5 = y0 = e0 + o0
	VSUBPS X9, X0, X7                // X7 = y4 = e0 - o0

	VMOVSLDUP X13, X8                // X8 = w1_8.real
	VMOVSHDUP X13, X9                // X9 = w1_8.imag
	VSHUFPS $0xB1, X1, X1, X3        // X3 = swap(o1)
	VMULPS X9, X3, X3                // X3 = o1.im * w1_8.imag
	VFMADDSUB231PS X8, X1, X3        // X3 = w1_8 * o1
	VADDPS X3, X10, X8               // X8 = y1 = e1 + w1_8*o1
	VSUBPS X3, X10, X9               // X9 = y5 = e1 - w1_8*o1

	VMOVSLDUP X14, X0                // X0 = w2_8.real
	VMOVSHDUP X14, X10               // X10 = w2_8.imag
	VSHUFPS $0xB1, X11, X11, X3      // X3 = swap(o2)
	VMULPS X10, X3, X3               // X3 = o2.im * w2_8.imag
	VFMADDSUB231PS X0, X11, X3       // X3 = w2_8 * o2
	VADDPS X3, X6, X1                // X1 = y2 = e2 + w2_8*o2
	VSUBPS X3, X6, X11               // X11 = y6 = e2 - w2_8*o2

	VMOVSLDUP X15, X0                // X0 = w3_8.real
	VMOVSHDUP X15, X10               // X10 = w3_8.imag
	VSHUFPS $0xB1, X4, X4, X3        // X3 = swap(o3)
	VMULPS X10, X3, X3               // X3 = o3.im * w3_8.imag
	VFMADDSUB231PS X0, X4, X3        // X3 = w3_8 * o3
	VADDPS X3, X2, X6                // X6 = y3 = e3 + w3_8*o3
	VSUBPS X3, X2, X4                // X4 = y7 = e3 - w3_8*o3

	MOVQ CX, BX                      // BX = base
	SHLQ $3, BX                      // BX = base*8
	VMOVSD X5, 0(R14)(BX*1)          // stage1[base] = y0
	VMOVSD X7, 32(R14)(BX*1)         // stage1[base+4] = y4
	VMOVSD X8, 8(R14)(BX*1)          // stage1[base+1] = y1
	VMOVSD X9, 40(R14)(BX*1)         // stage1[base+5] = y5
	VMOVSD X1, 16(R14)(BX*1)         // stage1[base+2] = y2
	VMOVSD X11, 48(R14)(BX*1)        // stage1[base+6] = y6
	VMOVSD X6, 24(R14)(BX*1)         // stage1[base+3] = y3
	VMOVSD X4, 56(R14)(BX*1)         // stage1[base+7] = y7

	ADDQ $8, CX                      // base += 8
	JMP  size512_r8_inv_stage1_loop  // continue stage1

size512_r8_inv_stage2:
	XORQ CX, CX                      // base = 0

size512_r8_inv_stage2_outer:
	CMPQ CX, $512                    // base >= 512?
	JGE  size512_r8_inv_stage3       // next stage
	XORQ DX, DX                      // j = 0

size512_r8_inv_stage2_inner:
	CMPQ DX, $8                      // j >= 8?
	JGE  size512_r8_inv_stage2_next  // next base
	MOVQ CX, BX                      // BX = base
	ADDQ DX, BX                      // BX = base + j
	MOVQ BX, SI                      // SI = base + j
	SHLQ $3, SI                      // SI = (base+j)*8

	VMOVSD 0(R14)(SI*1), X0          // X0 = x0
	VMOVSD 64(R14)(SI*1), X1         // X1 = x1
	MOVQ DX, AX                      // AX = j
	IMULQ $64, AX                    // AX = j*64 bytes
	VMOVSD (R10)(AX*1), X3           // X3 = tw1
	VXORPS X12, X3, X3               // X3 = conj(tw1)
	VMOVSLDUP X3, X8                 // X8 = tw1.real
	VMOVSHDUP X3, X9                 // X9 = tw1.imag
	VSHUFPS $0xB1, X1, X1, X11       // X11 = swap(x1)
	VMULPS X9, X11, X11              // X11 = x1.im * tw1.imag
	VFMADDSUB231PS X8, X1, X11       // X11 = tw1 * x1
	VMOVAPS X11, X1                  // X1 = tw1 * x1

	VMOVSD 128(R14)(SI*1), X2        // X2 = x2
	MOVQ DX, AX                      // AX = j
	IMULQ $128, AX                   // AX = j*128 bytes
	VMOVSD (R10)(AX*1), X3           // X3 = tw2
	VXORPS X12, X3, X3               // X3 = conj(tw2)
	VMOVSLDUP X3, X8                 // X8 = tw2.real
	VMOVSHDUP X3, X9                 // X9 = tw2.imag
	VSHUFPS $0xB1, X2, X2, X11       // X11 = swap(x2)
	VMULPS X9, X11, X11              // X11 = x2.im * tw2.imag
	VFMADDSUB231PS X8, X2, X11       // X11 = tw2 * x2
	VMOVAPS X11, X2                  // X2 = tw2 * x2

	VMOVSD 192(R14)(SI*1), X3        // X3 = x3
	MOVQ DX, AX                      // AX = j
	IMULQ $192, AX                   // AX = j*192 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw3
	VXORPS X12, X8, X8               // X8 = conj(tw3)
	VMOVSLDUP X8, X9                 // X9 = tw3.real
	VMOVSHDUP X8, X10                // X10 = tw3.imag
	VSHUFPS $0xB1, X3, X3, X11       // X11 = swap(x3)
	VMULPS X10, X11, X11             // X11 = x3.im * tw3.imag
	VFMADDSUB231PS X9, X3, X11       // X11 = tw3 * x3
	VMOVAPS X11, X3                  // X3 = tw3 * x3

	VMOVSD 256(R14)(SI*1), X4        // X4 = x4
	MOVQ DX, AX                      // AX = j
	IMULQ $256, AX                   // AX = j*256 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw4
	VXORPS X12, X8, X8               // X8 = conj(tw4)
	VMOVSLDUP X8, X9                 // X9 = tw4.real
	VMOVSHDUP X8, X10                // X10 = tw4.imag
	VSHUFPS $0xB1, X4, X4, X11       // X11 = swap(x4)
	VMULPS X10, X11, X11             // X11 = x4.im * tw4.imag
	VFMADDSUB231PS X9, X4, X11       // X11 = tw4 * x4
	VMOVAPS X11, X4                  // X4 = tw4 * x4

	VMOVSD 320(R14)(SI*1), X5        // X5 = x5
	MOVQ DX, AX                      // AX = j
	IMULQ $320, AX                   // AX = j*320 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw5
	VXORPS X12, X8, X8               // X8 = conj(tw5)
	VMOVSLDUP X8, X9                 // X9 = tw5.real
	VMOVSHDUP X8, X10                // X10 = tw5.imag
	VSHUFPS $0xB1, X5, X5, X11       // X11 = swap(x5)
	VMULPS X10, X11, X11             // X11 = x5.im * tw5.imag
	VFMADDSUB231PS X9, X5, X11       // X11 = tw5 * x5
	VMOVAPS X11, X5                  // X5 = tw5 * x5

	VMOVSD 384(R14)(SI*1), X6        // X6 = x6
	MOVQ DX, AX                      // AX = j
	IMULQ $384, AX                   // AX = j*384 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw6
	VXORPS X12, X8, X8               // X8 = conj(tw6)
	VMOVSLDUP X8, X9                 // X9 = tw6.real
	VMOVSHDUP X8, X10                // X10 = tw6.imag
	VSHUFPS $0xB1, X6, X6, X11       // X11 = swap(x6)
	VMULPS X10, X11, X11             // X11 = x6.im * tw6.imag
	VFMADDSUB231PS X9, X6, X11       // X11 = tw6 * x6
	VMOVAPS X11, X6                  // X6 = tw6 * x6

	VMOVSD 448(R14)(SI*1), X7        // X7 = x7
	MOVQ DX, AX                      // AX = j
	IMULQ $448, AX                   // AX = j*448 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw7
	VXORPS X12, X8, X8               // X8 = conj(tw7)
	VMOVSLDUP X8, X9                 // X9 = tw7.real
	VMOVSHDUP X8, X10                // X10 = tw7.imag
	VSHUFPS $0xB1, X7, X7, X11       // X11 = swap(x7)
	VMULPS X10, X11, X11             // X11 = x7.im * tw7.imag
	VFMADDSUB231PS X9, X7, X11       // X11 = tw7 * x7
	VMOVAPS X11, X7                  // X7 = tw7 * x7

	VADDPS X4, X0, X8                // X8 = a0 = x0 + x4
	VSUBPS X4, X0, X9                // X9 = a1 = x0 - x4
	VADDPS X6, X2, X10               // X10 = a2 = x2 + x6
	VSUBPS X6, X2, X2                // X2 = a3 = x2 - x6
	VADDPS X5, X1, X11               // X11 = a4 = x1 + x5
	VSUBPS X5, X1, X4                // X4 = a5 = x1 - x5
	VADDPS X7, X3, X1                // X1 = a6 = x3 + x7
	VSUBPS X7, X3, X5                // X5 = a7 = x3 - x7

	VADDPS X10, X8, X0               // X0 = e0 = a0 + a2
	VSUBPS X10, X8, X6               // X6 = e2 = a0 - a2
	VSHUFPS $0xB1, X2, X2, X7        // X7 = swap(a3)
	VXORPS X12, X7, X7               // X7 = a3 * (-i)
	VSUBPS X7, X9, X10               // X10 = e1 = a1 - (-i)*a3
	VADDPS X7, X9, X2                // X2 = e3 = a1 + (-i)*a3
	VADDPS X1, X11, X9               // X9 = o0 = a4 + a6
	VSUBPS X1, X11, X11              // X11 = o2 = a4 - a6
	VSHUFPS $0xB1, X5, X5, X7        // X7 = swap(a7)
	VXORPS X12, X7, X7               // X7 = a7 * (-i)
	VSUBPS X7, X4, X1                // X1 = o1 = a5 - (-i)*a7
	VADDPS X7, X4, X4                // X4 = o3 = a5 + (-i)*a7

	VADDPS X9, X0, X5                // X5 = y0 = e0 + o0
	VSUBPS X9, X0, X7                // X7 = y4 = e0 - o0

	VMOVSLDUP X13, X8                // X8 = w1_8.real
	VMOVSHDUP X13, X9                // X9 = w1_8.imag
	VSHUFPS $0xB1, X1, X1, X3        // X3 = swap(o1)
	VMULPS X9, X3, X3                // X3 = o1.im * w1_8.imag
	VFMADDSUB231PS X8, X1, X3        // X3 = w1_8 * o1
	VADDPS X3, X10, X8               // X8 = y1 = e1 + w1_8*o1
	VSUBPS X3, X10, X9               // X9 = y5 = e1 - w1_8*o1

	VMOVSLDUP X14, X0                // X0 = w2_8.real
	VMOVSHDUP X14, X10               // X10 = w2_8.imag
	VSHUFPS $0xB1, X11, X11, X3      // X3 = swap(o2)
	VMULPS X10, X3, X3               // X3 = o2.im * w2_8.imag
	VFMADDSUB231PS X0, X11, X3       // X3 = w2_8 * o2
	VADDPS X3, X6, X1                // X1 = y2 = e2 + w2_8*o2
	VSUBPS X3, X6, X11               // X11 = y6 = e2 - w2_8*o2

	VMOVSLDUP X15, X0                // X0 = w3_8.real
	VMOVSHDUP X15, X10               // X10 = w3_8.imag
	VSHUFPS $0xB1, X4, X4, X3        // X3 = swap(o3)
	VMULPS X10, X3, X3               // X3 = o3.im * w3_8.imag
	VFMADDSUB231PS X0, X4, X3        // X3 = w3_8 * o3
	VADDPS X3, X2, X6                // X6 = y3 = e3 + w3_8*o3
	VSUBPS X3, X2, X4                // X4 = y7 = e3 - w3_8*o3

	MOVQ BX, DI                      // DI = base + j
	SHLQ $3, DI                      // DI = (base+j)*8
	VMOVSD X5, 0(R8)(DI*1)           // stage2[base+j] = y0
	VMOVSD X7, 256(R8)(DI*1)         // stage2[base+j+32] = y4
	VMOVSD X8, 64(R8)(DI*1)          // stage2[base+j+8] = y1
	VMOVSD X9, 320(R8)(DI*1)         // stage2[base+j+40] = y5
	VMOVSD X1, 128(R8)(DI*1)         // stage2[base+j+16] = y2
	VMOVSD X11, 384(R8)(DI*1)        // stage2[base+j+48] = y6
	VMOVSD X6, 192(R8)(DI*1)         // stage2[base+j+24] = y3
	VMOVSD X4, 448(R8)(DI*1)         // stage2[base+j+56] = y7

	INCQ DX                          // j++
	JMP  size512_r8_inv_stage2_inner // continue inner

size512_r8_inv_stage2_next:
	ADDQ $64, CX                     // base += 64
	JMP  size512_r8_inv_stage2_outer // continue outer

size512_r8_inv_stage3:
	XORQ CX, CX                      // j = 0

size512_r8_inv_stage3_loop:
	CMPQ CX, $64                     // j >= 64?
	JGE  size512_r8_inv_copy         // copy if needed
	MOVQ CX, SI                      // SI = j
	SHLQ $3, SI                      // SI = j*8

	VMOVSD 0(R8)(SI*1), X0           // X0 = x0
	VMOVSD 512(R8)(SI*1), X1         // X1 = x1
	MOVQ CX, AX                      // AX = j
	IMULQ $8, AX                     // AX = j*8 bytes
	VMOVSD (R10)(AX*1), X3           // X3 = tw1
	VXORPS X12, X3, X3               // X3 = conj(tw1)
	VMOVSLDUP X3, X8                 // X8 = tw1.real
	VMOVSHDUP X3, X9                 // X9 = tw1.imag
	VSHUFPS $0xB1, X1, X1, X11       // X11 = swap(x1)
	VMULPS X9, X11, X11              // X11 = x1.im * tw1.imag
	VFMADDSUB231PS X8, X1, X11       // X11 = tw1 * x1
	VMOVAPS X11, X1                  // X1 = tw1 * x1

	VMOVSD 1024(R8)(SI*1), X2        // X2 = x2
	MOVQ CX, AX                      // AX = j
	IMULQ $16, AX                    // AX = j*16 bytes
	VMOVSD (R10)(AX*1), X3           // X3 = tw2
	VXORPS X12, X3, X3               // X3 = conj(tw2)
	VMOVSLDUP X3, X8                 // X8 = tw2.real
	VMOVSHDUP X3, X9                 // X9 = tw2.imag
	VSHUFPS $0xB1, X2, X2, X11       // X11 = swap(x2)
	VMULPS X9, X11, X11              // X11 = x2.im * tw2.imag
	VFMADDSUB231PS X8, X2, X11       // X11 = tw2 * x2
	VMOVAPS X11, X2                  // X2 = tw2 * x2

	VMOVSD 1536(R8)(SI*1), X3        // X3 = x3
	MOVQ CX, AX                      // AX = j
	IMULQ $24, AX                    // AX = j*24 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw3
	VXORPS X12, X8, X8               // X8 = conj(tw3)
	VMOVSLDUP X8, X9                 // X9 = tw3.real
	VMOVSHDUP X8, X10                // X10 = tw3.imag
	VSHUFPS $0xB1, X3, X3, X11       // X11 = swap(x3)
	VMULPS X10, X11, X11             // X11 = x3.im * tw3.imag
	VFMADDSUB231PS X9, X3, X11       // X11 = tw3 * x3
	VMOVAPS X11, X3                  // X3 = tw3 * x3

	VMOVSD 2048(R8)(SI*1), X4        // X4 = x4
	MOVQ CX, AX                      // AX = j
	IMULQ $32, AX                    // AX = j*32 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw4
	VXORPS X12, X8, X8               // X8 = conj(tw4)
	VMOVSLDUP X8, X9                 // X9 = tw4.real
	VMOVSHDUP X8, X10                // X10 = tw4.imag
	VSHUFPS $0xB1, X4, X4, X11       // X11 = swap(x4)
	VMULPS X10, X11, X11             // X11 = x4.im * tw4.imag
	VFMADDSUB231PS X9, X4, X11       // X11 = tw4 * x4
	VMOVAPS X11, X4                  // X4 = tw4 * x4

	VMOVSD 2560(R8)(SI*1), X5        // X5 = x5
	MOVQ CX, AX                      // AX = j
	IMULQ $40, AX                    // AX = j*40 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw5
	VXORPS X12, X8, X8               // X8 = conj(tw5)
	VMOVSLDUP X8, X9                 // X9 = tw5.real
	VMOVSHDUP X8, X10                // X10 = tw5.imag
	VSHUFPS $0xB1, X5, X5, X11       // X11 = swap(x5)
	VMULPS X10, X11, X11             // X11 = x5.im * tw5.imag
	VFMADDSUB231PS X9, X5, X11       // X11 = tw5 * x5
	VMOVAPS X11, X5                  // X5 = tw5 * x5

	VMOVSD 3072(R8)(SI*1), X6        // X6 = x6
	MOVQ CX, AX                      // AX = j
	IMULQ $48, AX                    // AX = j*48 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw6
	VXORPS X12, X8, X8               // X8 = conj(tw6)
	VMOVSLDUP X8, X9                 // X9 = tw6.real
	VMOVSHDUP X8, X10                // X10 = tw6.imag
	VSHUFPS $0xB1, X6, X6, X11       // X11 = swap(x6)
	VMULPS X10, X11, X11             // X11 = x6.im * tw6.imag
	VFMADDSUB231PS X9, X6, X11       // X11 = tw6 * x6
	VMOVAPS X11, X6                  // X6 = tw6 * x6

	VMOVSD 3584(R8)(SI*1), X7        // X7 = x7
	MOVQ CX, AX                      // AX = j
	IMULQ $56, AX                    // AX = j*56 bytes
	VMOVSD (R10)(AX*1), X8           // X8 = tw7
	VXORPS X12, X8, X8               // X8 = conj(tw7)
	VMOVSLDUP X8, X9                 // X9 = tw7.real
	VMOVSHDUP X8, X10                // X10 = tw7.imag
	VSHUFPS $0xB1, X7, X7, X11       // X11 = swap(x7)
	VMULPS X10, X11, X11             // X11 = x7.im * tw7.imag
	VFMADDSUB231PS X9, X7, X11       // X11 = tw7 * x7
	VMOVAPS X11, X7                  // X7 = tw7 * x7

	VADDPS X4, X0, X8                // X8 = a0 = x0 + x4
	VSUBPS X4, X0, X9                // X9 = a1 = x0 - x4
	VADDPS X6, X2, X10               // X10 = a2 = x2 + x6
	VSUBPS X6, X2, X2                // X2 = a3 = x2 - x6
	VADDPS X5, X1, X11               // X11 = a4 = x1 + x5
	VSUBPS X5, X1, X4                // X4 = a5 = x1 - x5
	VADDPS X7, X3, X1                // X1 = a6 = x3 + x7
	VSUBPS X7, X3, X5                // X5 = a7 = x3 - x7

	VADDPS X10, X8, X0               // X0 = e0 = a0 + a2
	VSUBPS X10, X8, X6               // X6 = e2 = a0 - a2
	VSHUFPS $0xB1, X2, X2, X7        // X7 = swap(a3)
	VXORPS X12, X7, X7               // X7 = a3 * (-i)
	VSUBPS X7, X9, X10               // X10 = e1 = a1 - (-i)*a3
	VADDPS X7, X9, X2                // X2 = e3 = a1 + (-i)*a3
	VADDPS X1, X11, X9               // X9 = o0 = a4 + a6
	VSUBPS X1, X11, X11              // X11 = o2 = a4 - a6
	VSHUFPS $0xB1, X5, X5, X7        // X7 = swap(a7)
	VXORPS X12, X7, X7               // X7 = a7 * (-i)
	VSUBPS X7, X4, X1                // X1 = o1 = a5 - (-i)*a7
	VADDPS X7, X4, X4                // X4 = o3 = a5 + (-i)*a7

	VADDPS X9, X0, X5                // X5 = y0 = e0 + o0
	VSUBPS X9, X0, X7                // X7 = y4 = e0 - o0

	VMOVSLDUP X13, X8                // X8 = w1_8.real
	VMOVSHDUP X13, X9                // X9 = w1_8.imag
	VSHUFPS $0xB1, X1, X1, X3        // X3 = swap(o1)
	VMULPS X9, X3, X3                // X3 = o1.im * w1_8.imag
	VFMADDSUB231PS X8, X1, X3        // X3 = w1_8 * o1
	VADDPS X3, X10, X8               // X8 = y1 = e1 + w1_8*o1
	VSUBPS X3, X10, X9               // X9 = y5 = e1 - w1_8*o1

	VMOVSLDUP X14, X0                // X0 = w2_8.real
	VMOVSHDUP X14, X10               // X10 = w2_8.imag
	VSHUFPS $0xB1, X11, X11, X3      // X3 = swap(o2)
	VMULPS X10, X3, X3               // X3 = o2.im * w2_8.imag
	VFMADDSUB231PS X0, X11, X3       // X3 = w2_8 * o2
	VADDPS X3, X6, X1                // X1 = y2 = e2 + w2_8*o2
	VSUBPS X3, X6, X11               // X11 = y6 = e2 - w2_8*o2

	VMOVSLDUP X15, X0                // X0 = w3_8.real
	VMOVSHDUP X15, X10               // X10 = w3_8.imag
	VSHUFPS $0xB1, X4, X4, X3        // X3 = swap(o3)
	VMULPS X10, X3, X3               // X3 = o3.im * w3_8.imag
	VFMADDSUB231PS X0, X4, X3        // X3 = w3_8 * o3
	VADDPS X3, X2, X6                // X6 = y3 = e3 + w3_8*o3
	VSUBPS X3, X2, X4                // X4 = y7 = e3 - w3_8*o3

	VMOVSD X5, 0(R8)(SI*1)           // work[j] = y0
	VMOVSD X7, 2048(R8)(SI*1)        // work[j+256] = y4
	VMOVSD X8, 512(R8)(SI*1)         // work[j+64] = y1
	VMOVSD X9, 2560(R8)(SI*1)        // work[j+320] = y5
	VMOVSD X1, 1024(R8)(SI*1)        // work[j+128] = y2
	VMOVSD X11, 3072(R8)(SI*1)       // work[j+384] = y6
	VMOVSD X6, 1536(R8)(SI*1)        // work[j+192] = y3
	VMOVSD X4, 3584(R8)(SI*1)        // work[j+448] = y7

	INCQ CX                          // j++
	JMP  size512_r8_inv_stage3_loop  // continue stage3

size512_r8_inv_copy:
	MOVQ dst+0(FP), R9               // R9 = dst pointer
	CMPQ R8, R9                      // work == dst?
	JE   size512_r8_inv_scale        // skip copy
	MOVQ R8, SI                      // SI = src (work)
	MOVQ R9, DI                      // DI = dst
	MOVQ $512, CX                    // CX = count (qwords)

size512_r8_inv_copy_loop:
	MOVQ (SI), AX                    // AX = *src
	MOVQ AX, (DI)                    // *dst = AX
	ADDQ $8, SI                      // src++
	ADDQ $8, DI                      // dst++
	DECQ CX                          // count--
	JNZ  size512_r8_inv_copy_loop    // loop

size512_r8_inv_scale:
	MOVL $0x3B000000, AX             // AX = 1/512 as float32
	MOVD AX, X0                      // X0 = scale scalar
	VBROADCASTSS X0, X0              // X0 = [scale, scale, scale, scale]
	MOVQ dst+0(FP), R9               // R9 = dst pointer
	XORQ CX, CX                      // i = 0

size512_r8_inv_scale_loop:
	CMPQ CX, $512                    // i >= 512?
	JGE  size512_r8_inv_done         // done
	MOVQ CX, AX                      // AX = i
	SHLQ $3, AX                      // AX = i*8
	VMOVSD (R9)(AX*1), X1            // X1 = dst[i]
	VMULPS X0, X1, X1                // X1 = dst[i] * scale
	VMOVSD X1, (R9)(AX*1)            // dst[i] = scaled
	INCQ CX                          // i++
	JMP  size512_r8_inv_scale_loop   // continue scaling

size512_r8_inv_done:
	VZEROUPPER                       // clear upper YMM state
	MOVB $1, ret+96(FP)              // return true
	RET                              // done

size512_r8_inv_return_false:
	MOVB $0, ret+96(FP)              // return false
	RET                              // done


DATA ·bitrev512_r8+0(SB)/8, $0
DATA ·bitrev512_r8+8(SB)/8, $64
DATA ·bitrev512_r8+16(SB)/8, $128
DATA ·bitrev512_r8+24(SB)/8, $192
DATA ·bitrev512_r8+32(SB)/8, $256
DATA ·bitrev512_r8+40(SB)/8, $320
DATA ·bitrev512_r8+48(SB)/8, $384
DATA ·bitrev512_r8+56(SB)/8, $448
DATA ·bitrev512_r8+64(SB)/8, $8
DATA ·bitrev512_r8+72(SB)/8, $72
DATA ·bitrev512_r8+80(SB)/8, $136
DATA ·bitrev512_r8+88(SB)/8, $200
DATA ·bitrev512_r8+96(SB)/8, $264
DATA ·bitrev512_r8+104(SB)/8, $328
DATA ·bitrev512_r8+112(SB)/8, $392
DATA ·bitrev512_r8+120(SB)/8, $456
DATA ·bitrev512_r8+128(SB)/8, $16
DATA ·bitrev512_r8+136(SB)/8, $80
DATA ·bitrev512_r8+144(SB)/8, $144
DATA ·bitrev512_r8+152(SB)/8, $208
DATA ·bitrev512_r8+160(SB)/8, $272
DATA ·bitrev512_r8+168(SB)/8, $336
DATA ·bitrev512_r8+176(SB)/8, $400
DATA ·bitrev512_r8+184(SB)/8, $464
DATA ·bitrev512_r8+192(SB)/8, $24
DATA ·bitrev512_r8+200(SB)/8, $88
DATA ·bitrev512_r8+208(SB)/8, $152
DATA ·bitrev512_r8+216(SB)/8, $216
DATA ·bitrev512_r8+224(SB)/8, $280
DATA ·bitrev512_r8+232(SB)/8, $344
DATA ·bitrev512_r8+240(SB)/8, $408
DATA ·bitrev512_r8+248(SB)/8, $472
DATA ·bitrev512_r8+256(SB)/8, $32
DATA ·bitrev512_r8+264(SB)/8, $96
DATA ·bitrev512_r8+272(SB)/8, $160
DATA ·bitrev512_r8+280(SB)/8, $224
DATA ·bitrev512_r8+288(SB)/8, $288
DATA ·bitrev512_r8+296(SB)/8, $352
DATA ·bitrev512_r8+304(SB)/8, $416
DATA ·bitrev512_r8+312(SB)/8, $480
DATA ·bitrev512_r8+320(SB)/8, $40
DATA ·bitrev512_r8+328(SB)/8, $104
DATA ·bitrev512_r8+336(SB)/8, $168
DATA ·bitrev512_r8+344(SB)/8, $232
DATA ·bitrev512_r8+352(SB)/8, $296
DATA ·bitrev512_r8+360(SB)/8, $360
DATA ·bitrev512_r8+368(SB)/8, $424
DATA ·bitrev512_r8+376(SB)/8, $488
DATA ·bitrev512_r8+384(SB)/8, $48
DATA ·bitrev512_r8+392(SB)/8, $112
DATA ·bitrev512_r8+400(SB)/8, $176
DATA ·bitrev512_r8+408(SB)/8, $240
DATA ·bitrev512_r8+416(SB)/8, $304
DATA ·bitrev512_r8+424(SB)/8, $368
DATA ·bitrev512_r8+432(SB)/8, $432
DATA ·bitrev512_r8+440(SB)/8, $496
DATA ·bitrev512_r8+448(SB)/8, $56
DATA ·bitrev512_r8+456(SB)/8, $120
DATA ·bitrev512_r8+464(SB)/8, $184
DATA ·bitrev512_r8+472(SB)/8, $248
DATA ·bitrev512_r8+480(SB)/8, $312
DATA ·bitrev512_r8+488(SB)/8, $376
DATA ·bitrev512_r8+496(SB)/8, $440
DATA ·bitrev512_r8+504(SB)/8, $504
DATA ·bitrev512_r8+512(SB)/8, $1
DATA ·bitrev512_r8+520(SB)/8, $65
DATA ·bitrev512_r8+528(SB)/8, $129
DATA ·bitrev512_r8+536(SB)/8, $193
DATA ·bitrev512_r8+544(SB)/8, $257
DATA ·bitrev512_r8+552(SB)/8, $321
DATA ·bitrev512_r8+560(SB)/8, $385
DATA ·bitrev512_r8+568(SB)/8, $449
DATA ·bitrev512_r8+576(SB)/8, $9
DATA ·bitrev512_r8+584(SB)/8, $73
DATA ·bitrev512_r8+592(SB)/8, $137
DATA ·bitrev512_r8+600(SB)/8, $201
DATA ·bitrev512_r8+608(SB)/8, $265
DATA ·bitrev512_r8+616(SB)/8, $329
DATA ·bitrev512_r8+624(SB)/8, $393
DATA ·bitrev512_r8+632(SB)/8, $457
DATA ·bitrev512_r8+640(SB)/8, $17
DATA ·bitrev512_r8+648(SB)/8, $81
DATA ·bitrev512_r8+656(SB)/8, $145
DATA ·bitrev512_r8+664(SB)/8, $209
DATA ·bitrev512_r8+672(SB)/8, $273
DATA ·bitrev512_r8+680(SB)/8, $337
DATA ·bitrev512_r8+688(SB)/8, $401
DATA ·bitrev512_r8+696(SB)/8, $465
DATA ·bitrev512_r8+704(SB)/8, $25
DATA ·bitrev512_r8+712(SB)/8, $89
DATA ·bitrev512_r8+720(SB)/8, $153
DATA ·bitrev512_r8+728(SB)/8, $217
DATA ·bitrev512_r8+736(SB)/8, $281
DATA ·bitrev512_r8+744(SB)/8, $345
DATA ·bitrev512_r8+752(SB)/8, $409
DATA ·bitrev512_r8+760(SB)/8, $473
DATA ·bitrev512_r8+768(SB)/8, $33
DATA ·bitrev512_r8+776(SB)/8, $97
DATA ·bitrev512_r8+784(SB)/8, $161
DATA ·bitrev512_r8+792(SB)/8, $225
DATA ·bitrev512_r8+800(SB)/8, $289
DATA ·bitrev512_r8+808(SB)/8, $353
DATA ·bitrev512_r8+816(SB)/8, $417
DATA ·bitrev512_r8+824(SB)/8, $481
DATA ·bitrev512_r8+832(SB)/8, $41
DATA ·bitrev512_r8+840(SB)/8, $105
DATA ·bitrev512_r8+848(SB)/8, $169
DATA ·bitrev512_r8+856(SB)/8, $233
DATA ·bitrev512_r8+864(SB)/8, $297
DATA ·bitrev512_r8+872(SB)/8, $361
DATA ·bitrev512_r8+880(SB)/8, $425
DATA ·bitrev512_r8+888(SB)/8, $489
DATA ·bitrev512_r8+896(SB)/8, $49
DATA ·bitrev512_r8+904(SB)/8, $113
DATA ·bitrev512_r8+912(SB)/8, $177
DATA ·bitrev512_r8+920(SB)/8, $241
DATA ·bitrev512_r8+928(SB)/8, $305
DATA ·bitrev512_r8+936(SB)/8, $369
DATA ·bitrev512_r8+944(SB)/8, $433
DATA ·bitrev512_r8+952(SB)/8, $497
DATA ·bitrev512_r8+960(SB)/8, $57
DATA ·bitrev512_r8+968(SB)/8, $121
DATA ·bitrev512_r8+976(SB)/8, $185
DATA ·bitrev512_r8+984(SB)/8, $249
DATA ·bitrev512_r8+992(SB)/8, $313
DATA ·bitrev512_r8+1000(SB)/8, $377
DATA ·bitrev512_r8+1008(SB)/8, $441
DATA ·bitrev512_r8+1016(SB)/8, $505
DATA ·bitrev512_r8+1024(SB)/8, $2
DATA ·bitrev512_r8+1032(SB)/8, $66
DATA ·bitrev512_r8+1040(SB)/8, $130
DATA ·bitrev512_r8+1048(SB)/8, $194
DATA ·bitrev512_r8+1056(SB)/8, $258
DATA ·bitrev512_r8+1064(SB)/8, $322
DATA ·bitrev512_r8+1072(SB)/8, $386
DATA ·bitrev512_r8+1080(SB)/8, $450
DATA ·bitrev512_r8+1088(SB)/8, $10
DATA ·bitrev512_r8+1096(SB)/8, $74
DATA ·bitrev512_r8+1104(SB)/8, $138
DATA ·bitrev512_r8+1112(SB)/8, $202
DATA ·bitrev512_r8+1120(SB)/8, $266
DATA ·bitrev512_r8+1128(SB)/8, $330
DATA ·bitrev512_r8+1136(SB)/8, $394
DATA ·bitrev512_r8+1144(SB)/8, $458
DATA ·bitrev512_r8+1152(SB)/8, $18
DATA ·bitrev512_r8+1160(SB)/8, $82
DATA ·bitrev512_r8+1168(SB)/8, $146
DATA ·bitrev512_r8+1176(SB)/8, $210
DATA ·bitrev512_r8+1184(SB)/8, $274
DATA ·bitrev512_r8+1192(SB)/8, $338
DATA ·bitrev512_r8+1200(SB)/8, $402
DATA ·bitrev512_r8+1208(SB)/8, $466
DATA ·bitrev512_r8+1216(SB)/8, $26
DATA ·bitrev512_r8+1224(SB)/8, $90
DATA ·bitrev512_r8+1232(SB)/8, $154
DATA ·bitrev512_r8+1240(SB)/8, $218
DATA ·bitrev512_r8+1248(SB)/8, $282
DATA ·bitrev512_r8+1256(SB)/8, $346
DATA ·bitrev512_r8+1264(SB)/8, $410
DATA ·bitrev512_r8+1272(SB)/8, $474
DATA ·bitrev512_r8+1280(SB)/8, $34
DATA ·bitrev512_r8+1288(SB)/8, $98
DATA ·bitrev512_r8+1296(SB)/8, $162
DATA ·bitrev512_r8+1304(SB)/8, $226
DATA ·bitrev512_r8+1312(SB)/8, $290
DATA ·bitrev512_r8+1320(SB)/8, $354
DATA ·bitrev512_r8+1328(SB)/8, $418
DATA ·bitrev512_r8+1336(SB)/8, $482
DATA ·bitrev512_r8+1344(SB)/8, $42
DATA ·bitrev512_r8+1352(SB)/8, $106
DATA ·bitrev512_r8+1360(SB)/8, $170
DATA ·bitrev512_r8+1368(SB)/8, $234
DATA ·bitrev512_r8+1376(SB)/8, $298
DATA ·bitrev512_r8+1384(SB)/8, $362
DATA ·bitrev512_r8+1392(SB)/8, $426
DATA ·bitrev512_r8+1400(SB)/8, $490
DATA ·bitrev512_r8+1408(SB)/8, $50
DATA ·bitrev512_r8+1416(SB)/8, $114
DATA ·bitrev512_r8+1424(SB)/8, $178
DATA ·bitrev512_r8+1432(SB)/8, $242
DATA ·bitrev512_r8+1440(SB)/8, $306
DATA ·bitrev512_r8+1448(SB)/8, $370
DATA ·bitrev512_r8+1456(SB)/8, $434
DATA ·bitrev512_r8+1464(SB)/8, $498
DATA ·bitrev512_r8+1472(SB)/8, $58
DATA ·bitrev512_r8+1480(SB)/8, $122
DATA ·bitrev512_r8+1488(SB)/8, $186
DATA ·bitrev512_r8+1496(SB)/8, $250
DATA ·bitrev512_r8+1504(SB)/8, $314
DATA ·bitrev512_r8+1512(SB)/8, $378
DATA ·bitrev512_r8+1520(SB)/8, $442
DATA ·bitrev512_r8+1528(SB)/8, $506
DATA ·bitrev512_r8+1536(SB)/8, $3
DATA ·bitrev512_r8+1544(SB)/8, $67
DATA ·bitrev512_r8+1552(SB)/8, $131
DATA ·bitrev512_r8+1560(SB)/8, $195
DATA ·bitrev512_r8+1568(SB)/8, $259
DATA ·bitrev512_r8+1576(SB)/8, $323
DATA ·bitrev512_r8+1584(SB)/8, $387
DATA ·bitrev512_r8+1592(SB)/8, $451
DATA ·bitrev512_r8+1600(SB)/8, $11
DATA ·bitrev512_r8+1608(SB)/8, $75
DATA ·bitrev512_r8+1616(SB)/8, $139
DATA ·bitrev512_r8+1624(SB)/8, $203
DATA ·bitrev512_r8+1632(SB)/8, $267
DATA ·bitrev512_r8+1640(SB)/8, $331
DATA ·bitrev512_r8+1648(SB)/8, $395
DATA ·bitrev512_r8+1656(SB)/8, $459
DATA ·bitrev512_r8+1664(SB)/8, $19
DATA ·bitrev512_r8+1672(SB)/8, $83
DATA ·bitrev512_r8+1680(SB)/8, $147
DATA ·bitrev512_r8+1688(SB)/8, $211
DATA ·bitrev512_r8+1696(SB)/8, $275
DATA ·bitrev512_r8+1704(SB)/8, $339
DATA ·bitrev512_r8+1712(SB)/8, $403
DATA ·bitrev512_r8+1720(SB)/8, $467
DATA ·bitrev512_r8+1728(SB)/8, $27
DATA ·bitrev512_r8+1736(SB)/8, $91
DATA ·bitrev512_r8+1744(SB)/8, $155
DATA ·bitrev512_r8+1752(SB)/8, $219
DATA ·bitrev512_r8+1760(SB)/8, $283
DATA ·bitrev512_r8+1768(SB)/8, $347
DATA ·bitrev512_r8+1776(SB)/8, $411
DATA ·bitrev512_r8+1784(SB)/8, $475
DATA ·bitrev512_r8+1792(SB)/8, $35
DATA ·bitrev512_r8+1800(SB)/8, $99
DATA ·bitrev512_r8+1808(SB)/8, $163
DATA ·bitrev512_r8+1816(SB)/8, $227
DATA ·bitrev512_r8+1824(SB)/8, $291
DATA ·bitrev512_r8+1832(SB)/8, $355
DATA ·bitrev512_r8+1840(SB)/8, $419
DATA ·bitrev512_r8+1848(SB)/8, $483
DATA ·bitrev512_r8+1856(SB)/8, $43
DATA ·bitrev512_r8+1864(SB)/8, $107
DATA ·bitrev512_r8+1872(SB)/8, $171
DATA ·bitrev512_r8+1880(SB)/8, $235
DATA ·bitrev512_r8+1888(SB)/8, $299
DATA ·bitrev512_r8+1896(SB)/8, $363
DATA ·bitrev512_r8+1904(SB)/8, $427
DATA ·bitrev512_r8+1912(SB)/8, $491
DATA ·bitrev512_r8+1920(SB)/8, $51
DATA ·bitrev512_r8+1928(SB)/8, $115
DATA ·bitrev512_r8+1936(SB)/8, $179
DATA ·bitrev512_r8+1944(SB)/8, $243
DATA ·bitrev512_r8+1952(SB)/8, $307
DATA ·bitrev512_r8+1960(SB)/8, $371
DATA ·bitrev512_r8+1968(SB)/8, $435
DATA ·bitrev512_r8+1976(SB)/8, $499
DATA ·bitrev512_r8+1984(SB)/8, $59
DATA ·bitrev512_r8+1992(SB)/8, $123
DATA ·bitrev512_r8+2000(SB)/8, $187
DATA ·bitrev512_r8+2008(SB)/8, $251
DATA ·bitrev512_r8+2016(SB)/8, $315
DATA ·bitrev512_r8+2024(SB)/8, $379
DATA ·bitrev512_r8+2032(SB)/8, $443
DATA ·bitrev512_r8+2040(SB)/8, $507
DATA ·bitrev512_r8+2048(SB)/8, $4
DATA ·bitrev512_r8+2056(SB)/8, $68
DATA ·bitrev512_r8+2064(SB)/8, $132
DATA ·bitrev512_r8+2072(SB)/8, $196
DATA ·bitrev512_r8+2080(SB)/8, $260
DATA ·bitrev512_r8+2088(SB)/8, $324
DATA ·bitrev512_r8+2096(SB)/8, $388
DATA ·bitrev512_r8+2104(SB)/8, $452
DATA ·bitrev512_r8+2112(SB)/8, $12
DATA ·bitrev512_r8+2120(SB)/8, $76
DATA ·bitrev512_r8+2128(SB)/8, $140
DATA ·bitrev512_r8+2136(SB)/8, $204
DATA ·bitrev512_r8+2144(SB)/8, $268
DATA ·bitrev512_r8+2152(SB)/8, $332
DATA ·bitrev512_r8+2160(SB)/8, $396
DATA ·bitrev512_r8+2168(SB)/8, $460
DATA ·bitrev512_r8+2176(SB)/8, $20
DATA ·bitrev512_r8+2184(SB)/8, $84
DATA ·bitrev512_r8+2192(SB)/8, $148
DATA ·bitrev512_r8+2200(SB)/8, $212
DATA ·bitrev512_r8+2208(SB)/8, $276
DATA ·bitrev512_r8+2216(SB)/8, $340
DATA ·bitrev512_r8+2224(SB)/8, $404
DATA ·bitrev512_r8+2232(SB)/8, $468
DATA ·bitrev512_r8+2240(SB)/8, $28
DATA ·bitrev512_r8+2248(SB)/8, $92
DATA ·bitrev512_r8+2256(SB)/8, $156
DATA ·bitrev512_r8+2264(SB)/8, $220
DATA ·bitrev512_r8+2272(SB)/8, $284
DATA ·bitrev512_r8+2280(SB)/8, $348
DATA ·bitrev512_r8+2288(SB)/8, $412
DATA ·bitrev512_r8+2296(SB)/8, $476
DATA ·bitrev512_r8+2304(SB)/8, $36
DATA ·bitrev512_r8+2312(SB)/8, $100
DATA ·bitrev512_r8+2320(SB)/8, $164
DATA ·bitrev512_r8+2328(SB)/8, $228
DATA ·bitrev512_r8+2336(SB)/8, $292
DATA ·bitrev512_r8+2344(SB)/8, $356
DATA ·bitrev512_r8+2352(SB)/8, $420
DATA ·bitrev512_r8+2360(SB)/8, $484
DATA ·bitrev512_r8+2368(SB)/8, $44
DATA ·bitrev512_r8+2376(SB)/8, $108
DATA ·bitrev512_r8+2384(SB)/8, $172
DATA ·bitrev512_r8+2392(SB)/8, $236
DATA ·bitrev512_r8+2400(SB)/8, $300
DATA ·bitrev512_r8+2408(SB)/8, $364
DATA ·bitrev512_r8+2416(SB)/8, $428
DATA ·bitrev512_r8+2424(SB)/8, $492
DATA ·bitrev512_r8+2432(SB)/8, $52
DATA ·bitrev512_r8+2440(SB)/8, $116
DATA ·bitrev512_r8+2448(SB)/8, $180
DATA ·bitrev512_r8+2456(SB)/8, $244
DATA ·bitrev512_r8+2464(SB)/8, $308
DATA ·bitrev512_r8+2472(SB)/8, $372
DATA ·bitrev512_r8+2480(SB)/8, $436
DATA ·bitrev512_r8+2488(SB)/8, $500
DATA ·bitrev512_r8+2496(SB)/8, $60
DATA ·bitrev512_r8+2504(SB)/8, $124
DATA ·bitrev512_r8+2512(SB)/8, $188
DATA ·bitrev512_r8+2520(SB)/8, $252
DATA ·bitrev512_r8+2528(SB)/8, $316
DATA ·bitrev512_r8+2536(SB)/8, $380
DATA ·bitrev512_r8+2544(SB)/8, $444
DATA ·bitrev512_r8+2552(SB)/8, $508
DATA ·bitrev512_r8+2560(SB)/8, $5
DATA ·bitrev512_r8+2568(SB)/8, $69
DATA ·bitrev512_r8+2576(SB)/8, $133
DATA ·bitrev512_r8+2584(SB)/8, $197
DATA ·bitrev512_r8+2592(SB)/8, $261
DATA ·bitrev512_r8+2600(SB)/8, $325
DATA ·bitrev512_r8+2608(SB)/8, $389
DATA ·bitrev512_r8+2616(SB)/8, $453
DATA ·bitrev512_r8+2624(SB)/8, $13
DATA ·bitrev512_r8+2632(SB)/8, $77
DATA ·bitrev512_r8+2640(SB)/8, $141
DATA ·bitrev512_r8+2648(SB)/8, $205
DATA ·bitrev512_r8+2656(SB)/8, $269
DATA ·bitrev512_r8+2664(SB)/8, $333
DATA ·bitrev512_r8+2672(SB)/8, $397
DATA ·bitrev512_r8+2680(SB)/8, $461
DATA ·bitrev512_r8+2688(SB)/8, $21
DATA ·bitrev512_r8+2696(SB)/8, $85
DATA ·bitrev512_r8+2704(SB)/8, $149
DATA ·bitrev512_r8+2712(SB)/8, $213
DATA ·bitrev512_r8+2720(SB)/8, $277
DATA ·bitrev512_r8+2728(SB)/8, $341
DATA ·bitrev512_r8+2736(SB)/8, $405
DATA ·bitrev512_r8+2744(SB)/8, $469
DATA ·bitrev512_r8+2752(SB)/8, $29
DATA ·bitrev512_r8+2760(SB)/8, $93
DATA ·bitrev512_r8+2768(SB)/8, $157
DATA ·bitrev512_r8+2776(SB)/8, $221
DATA ·bitrev512_r8+2784(SB)/8, $285
DATA ·bitrev512_r8+2792(SB)/8, $349
DATA ·bitrev512_r8+2800(SB)/8, $413
DATA ·bitrev512_r8+2808(SB)/8, $477
DATA ·bitrev512_r8+2816(SB)/8, $37
DATA ·bitrev512_r8+2824(SB)/8, $101
DATA ·bitrev512_r8+2832(SB)/8, $165
DATA ·bitrev512_r8+2840(SB)/8, $229
DATA ·bitrev512_r8+2848(SB)/8, $293
DATA ·bitrev512_r8+2856(SB)/8, $357
DATA ·bitrev512_r8+2864(SB)/8, $421
DATA ·bitrev512_r8+2872(SB)/8, $485
DATA ·bitrev512_r8+2880(SB)/8, $45
DATA ·bitrev512_r8+2888(SB)/8, $109
DATA ·bitrev512_r8+2896(SB)/8, $173
DATA ·bitrev512_r8+2904(SB)/8, $237
DATA ·bitrev512_r8+2912(SB)/8, $301
DATA ·bitrev512_r8+2920(SB)/8, $365
DATA ·bitrev512_r8+2928(SB)/8, $429
DATA ·bitrev512_r8+2936(SB)/8, $493
DATA ·bitrev512_r8+2944(SB)/8, $53
DATA ·bitrev512_r8+2952(SB)/8, $117
DATA ·bitrev512_r8+2960(SB)/8, $181
DATA ·bitrev512_r8+2968(SB)/8, $245
DATA ·bitrev512_r8+2976(SB)/8, $309
DATA ·bitrev512_r8+2984(SB)/8, $373
DATA ·bitrev512_r8+2992(SB)/8, $437
DATA ·bitrev512_r8+3000(SB)/8, $501
DATA ·bitrev512_r8+3008(SB)/8, $61
DATA ·bitrev512_r8+3016(SB)/8, $125
DATA ·bitrev512_r8+3024(SB)/8, $189
DATA ·bitrev512_r8+3032(SB)/8, $253
DATA ·bitrev512_r8+3040(SB)/8, $317
DATA ·bitrev512_r8+3048(SB)/8, $381
DATA ·bitrev512_r8+3056(SB)/8, $445
DATA ·bitrev512_r8+3064(SB)/8, $509
DATA ·bitrev512_r8+3072(SB)/8, $6
DATA ·bitrev512_r8+3080(SB)/8, $70
DATA ·bitrev512_r8+3088(SB)/8, $134
DATA ·bitrev512_r8+3096(SB)/8, $198
DATA ·bitrev512_r8+3104(SB)/8, $262
DATA ·bitrev512_r8+3112(SB)/8, $326
DATA ·bitrev512_r8+3120(SB)/8, $390
DATA ·bitrev512_r8+3128(SB)/8, $454
DATA ·bitrev512_r8+3136(SB)/8, $14
DATA ·bitrev512_r8+3144(SB)/8, $78
DATA ·bitrev512_r8+3152(SB)/8, $142
DATA ·bitrev512_r8+3160(SB)/8, $206
DATA ·bitrev512_r8+3168(SB)/8, $270
DATA ·bitrev512_r8+3176(SB)/8, $334
DATA ·bitrev512_r8+3184(SB)/8, $398
DATA ·bitrev512_r8+3192(SB)/8, $462
DATA ·bitrev512_r8+3200(SB)/8, $22
DATA ·bitrev512_r8+3208(SB)/8, $86
DATA ·bitrev512_r8+3216(SB)/8, $150
DATA ·bitrev512_r8+3224(SB)/8, $214
DATA ·bitrev512_r8+3232(SB)/8, $278
DATA ·bitrev512_r8+3240(SB)/8, $342
DATA ·bitrev512_r8+3248(SB)/8, $406
DATA ·bitrev512_r8+3256(SB)/8, $470
DATA ·bitrev512_r8+3264(SB)/8, $30
DATA ·bitrev512_r8+3272(SB)/8, $94
DATA ·bitrev512_r8+3280(SB)/8, $158
DATA ·bitrev512_r8+3288(SB)/8, $222
DATA ·bitrev512_r8+3296(SB)/8, $286
DATA ·bitrev512_r8+3304(SB)/8, $350
DATA ·bitrev512_r8+3312(SB)/8, $414
DATA ·bitrev512_r8+3320(SB)/8, $478
DATA ·bitrev512_r8+3328(SB)/8, $38
DATA ·bitrev512_r8+3336(SB)/8, $102
DATA ·bitrev512_r8+3344(SB)/8, $166
DATA ·bitrev512_r8+3352(SB)/8, $230
DATA ·bitrev512_r8+3360(SB)/8, $294
DATA ·bitrev512_r8+3368(SB)/8, $358
DATA ·bitrev512_r8+3376(SB)/8, $422
DATA ·bitrev512_r8+3384(SB)/8, $486
DATA ·bitrev512_r8+3392(SB)/8, $46
DATA ·bitrev512_r8+3400(SB)/8, $110
DATA ·bitrev512_r8+3408(SB)/8, $174
DATA ·bitrev512_r8+3416(SB)/8, $238
DATA ·bitrev512_r8+3424(SB)/8, $302
DATA ·bitrev512_r8+3432(SB)/8, $366
DATA ·bitrev512_r8+3440(SB)/8, $430
DATA ·bitrev512_r8+3448(SB)/8, $494
DATA ·bitrev512_r8+3456(SB)/8, $54
DATA ·bitrev512_r8+3464(SB)/8, $118
DATA ·bitrev512_r8+3472(SB)/8, $182
DATA ·bitrev512_r8+3480(SB)/8, $246
DATA ·bitrev512_r8+3488(SB)/8, $310
DATA ·bitrev512_r8+3496(SB)/8, $374
DATA ·bitrev512_r8+3504(SB)/8, $438
DATA ·bitrev512_r8+3512(SB)/8, $502
DATA ·bitrev512_r8+3520(SB)/8, $62
DATA ·bitrev512_r8+3528(SB)/8, $126
DATA ·bitrev512_r8+3536(SB)/8, $190
DATA ·bitrev512_r8+3544(SB)/8, $254
DATA ·bitrev512_r8+3552(SB)/8, $318
DATA ·bitrev512_r8+3560(SB)/8, $382
DATA ·bitrev512_r8+3568(SB)/8, $446
DATA ·bitrev512_r8+3576(SB)/8, $510
DATA ·bitrev512_r8+3584(SB)/8, $7
DATA ·bitrev512_r8+3592(SB)/8, $71
DATA ·bitrev512_r8+3600(SB)/8, $135
DATA ·bitrev512_r8+3608(SB)/8, $199
DATA ·bitrev512_r8+3616(SB)/8, $263
DATA ·bitrev512_r8+3624(SB)/8, $327
DATA ·bitrev512_r8+3632(SB)/8, $391
DATA ·bitrev512_r8+3640(SB)/8, $455
DATA ·bitrev512_r8+3648(SB)/8, $15
DATA ·bitrev512_r8+3656(SB)/8, $79
DATA ·bitrev512_r8+3664(SB)/8, $143
DATA ·bitrev512_r8+3672(SB)/8, $207
DATA ·bitrev512_r8+3680(SB)/8, $271
DATA ·bitrev512_r8+3688(SB)/8, $335
DATA ·bitrev512_r8+3696(SB)/8, $399
DATA ·bitrev512_r8+3704(SB)/8, $463
DATA ·bitrev512_r8+3712(SB)/8, $23
DATA ·bitrev512_r8+3720(SB)/8, $87
DATA ·bitrev512_r8+3728(SB)/8, $151
DATA ·bitrev512_r8+3736(SB)/8, $215
DATA ·bitrev512_r8+3744(SB)/8, $279
DATA ·bitrev512_r8+3752(SB)/8, $343
DATA ·bitrev512_r8+3760(SB)/8, $407
DATA ·bitrev512_r8+3768(SB)/8, $471
DATA ·bitrev512_r8+3776(SB)/8, $31
DATA ·bitrev512_r8+3784(SB)/8, $95
DATA ·bitrev512_r8+3792(SB)/8, $159
DATA ·bitrev512_r8+3800(SB)/8, $223
DATA ·bitrev512_r8+3808(SB)/8, $287
DATA ·bitrev512_r8+3816(SB)/8, $351
DATA ·bitrev512_r8+3824(SB)/8, $415
DATA ·bitrev512_r8+3832(SB)/8, $479
DATA ·bitrev512_r8+3840(SB)/8, $39
DATA ·bitrev512_r8+3848(SB)/8, $103
DATA ·bitrev512_r8+3856(SB)/8, $167
DATA ·bitrev512_r8+3864(SB)/8, $231
DATA ·bitrev512_r8+3872(SB)/8, $295
DATA ·bitrev512_r8+3880(SB)/8, $359
DATA ·bitrev512_r8+3888(SB)/8, $423
DATA ·bitrev512_r8+3896(SB)/8, $487
DATA ·bitrev512_r8+3904(SB)/8, $47
DATA ·bitrev512_r8+3912(SB)/8, $111
DATA ·bitrev512_r8+3920(SB)/8, $175
DATA ·bitrev512_r8+3928(SB)/8, $239
DATA ·bitrev512_r8+3936(SB)/8, $303
DATA ·bitrev512_r8+3944(SB)/8, $367
DATA ·bitrev512_r8+3952(SB)/8, $431
DATA ·bitrev512_r8+3960(SB)/8, $495
DATA ·bitrev512_r8+3968(SB)/8, $55
DATA ·bitrev512_r8+3976(SB)/8, $119
DATA ·bitrev512_r8+3984(SB)/8, $183
DATA ·bitrev512_r8+3992(SB)/8, $247
DATA ·bitrev512_r8+4000(SB)/8, $311
DATA ·bitrev512_r8+4008(SB)/8, $375
DATA ·bitrev512_r8+4016(SB)/8, $439
DATA ·bitrev512_r8+4024(SB)/8, $503
DATA ·bitrev512_r8+4032(SB)/8, $63
DATA ·bitrev512_r8+4040(SB)/8, $127
DATA ·bitrev512_r8+4048(SB)/8, $191
DATA ·bitrev512_r8+4056(SB)/8, $255
DATA ·bitrev512_r8+4064(SB)/8, $319
DATA ·bitrev512_r8+4072(SB)/8, $383
DATA ·bitrev512_r8+4080(SB)/8, $447
DATA ·bitrev512_r8+4088(SB)/8, $511
GLOBL ·bitrev512_r8(SB), RODATA, $4096
