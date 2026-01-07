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
TEXT ·ForwardAVX2Size512Radix8Complex64Asm(SB), $4096-121
	MOVQ dst+0(FP), R8               // R8 = dst pointer
	MOVQ src+24(FP), R9              // R9 = src pointer
	MOVQ twiddle+48(FP), R10         // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11         // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12          // R12 = bitrev pointer
	MOVQ src+32(FP), R13             // R13 = len(src)

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
	MOVQ bitrev+104(FP), AX          // AX = len(bitrev)
	CMPQ AX, $512                    // len(bitrev) < 512?
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
	MOVB $1, ret+120(FP)             // return true
	RET                              // done

size512_r8_fwd_return_false:
	MOVB $0, ret+120(FP)             // return false
	RET                              // done

// Inverse transform, size 512, complex64, radix-8 DIT
TEXT ·InverseAVX2Size512Radix8Complex64Asm(SB), $4096-121
	MOVQ dst+0(FP), R8               // R8 = dst pointer
	MOVQ src+24(FP), R9              // R9 = src pointer
	MOVQ twiddle+48(FP), R10         // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11         // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12          // R12 = bitrev pointer
	MOVQ src+32(FP), R13             // R13 = len(src)

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
	MOVQ bitrev+104(FP), AX          // AX = len(bitrev)
	CMPQ AX, $512                    // len(bitrev) < 512?
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
	MOVB $1, ret+120(FP)             // return true
	RET                              // done

size512_r8_inv_return_false:
	MOVB $0, ret+120(FP)             // return false
	RET                              // done
