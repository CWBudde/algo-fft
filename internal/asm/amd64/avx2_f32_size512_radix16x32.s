//go:build amd64 && asm && !purego

#include "textflag.h"

// ===========================================================================
// AVX2 Size-512 Mixed Radix-16×32 FFT (complex64) Kernels for AMD64
// ===========================================================================
//
// Algorithm: Six-step FFT with 512 = 16 × 32 decomposition
//
// Forward transform:
//   Stage 1: 16 column FFT-32s (each via 2×FFT-16 + W_32 twiddles)
//   Stage 2: 32 row FFT-16s
//
// FFT-16 DIT structure (bit-reversed input -> natural output):
//   Stage 1: 8 radix-2 butterflies on pairs (0,1), (2,3), ..., (14,15)
//   Stage 2: 4 radix-4 butterflies with -i twiddles on v3, v7, v11, v15
//   Stage 3: 2 radix-8 butterflies with W_8 twiddles
//   Stage 4: 1 radix-16 butterfly with W_16 twiddles
//
// Memory layout (complex64 = 8 bytes = real32 + imag32):
//   Each complex number: [real, imag] as two float32
//
// ===========================================================================

// ===== CONSTANTS =====
// All constants are 32-byte aligned for YMM register loads

// isq2 = 1/sqrt(2) = 0.70710678 (IEEE 754: 0x3F3504F3)
DATA const_isq2<>+0x00(SB)/4, $0x3F3504F3
DATA const_isq2<>+0x04(SB)/4, $0x3F3504F3
DATA const_isq2<>+0x08(SB)/4, $0x3F3504F3
DATA const_isq2<>+0x0C(SB)/4, $0x3F3504F3
DATA const_isq2<>+0x10(SB)/4, $0x3F3504F3
DATA const_isq2<>+0x14(SB)/4, $0x3F3504F3
DATA const_isq2<>+0x18(SB)/4, $0x3F3504F3
DATA const_isq2<>+0x1C(SB)/4, $0x3F3504F3
GLOBL const_isq2<>(SB), RODATA|NOPTR, $32

// sin1 = sin(π/8) = 0.38268343 (IEEE 754: 0x3EC3EF15)
DATA const_sin1<>+0x00(SB)/4, $0x3EC3EF15
DATA const_sin1<>+0x04(SB)/4, $0x3EC3EF15
DATA const_sin1<>+0x08(SB)/4, $0x3EC3EF15
DATA const_sin1<>+0x0C(SB)/4, $0x3EC3EF15
DATA const_sin1<>+0x10(SB)/4, $0x3EC3EF15
DATA const_sin1<>+0x14(SB)/4, $0x3EC3EF15
DATA const_sin1<>+0x18(SB)/4, $0x3EC3EF15
DATA const_sin1<>+0x1C(SB)/4, $0x3EC3EF15
GLOBL const_sin1<>(SB), RODATA|NOPTR, $32

// cos1 = cos(π/8) = 0.92387953 (IEEE 754: 0x3F6C835E)
DATA const_cos1<>+0x00(SB)/4, $0x3F6C835E
DATA const_cos1<>+0x04(SB)/4, $0x3F6C835E
DATA const_cos1<>+0x08(SB)/4, $0x3F6C835E
DATA const_cos1<>+0x0C(SB)/4, $0x3F6C835E
DATA const_cos1<>+0x10(SB)/4, $0x3F6C835E
DATA const_cos1<>+0x14(SB)/4, $0x3F6C835E
DATA const_cos1<>+0x18(SB)/4, $0x3F6C835E
DATA const_cos1<>+0x1C(SB)/4, $0x3F6C835E
GLOBL const_cos1<>(SB), RODATA|NOPTR, $32

// scale = 1/512 = 0.001953125 (IEEE 754: 0x3B000000)
DATA const_scale<>+0x00(SB)/4, $0x3B000000
DATA const_scale<>+0x04(SB)/4, $0x3B000000
DATA const_scale<>+0x08(SB)/4, $0x3B000000
DATA const_scale<>+0x0C(SB)/4, $0x3B000000
DATA const_scale<>+0x10(SB)/4, $0x3B000000
DATA const_scale<>+0x14(SB)/4, $0x3B000000
DATA const_scale<>+0x18(SB)/4, $0x3B000000
DATA const_scale<>+0x1C(SB)/4, $0x3B000000
GLOBL const_scale<>(SB), RODATA|NOPTR, $32

// Sign mask for -i rotation: negate imaginary parts [0, -0, 0, -0, ...]
// Used for: (a+bi) * (-i) = (b, -a) with sign flip on second component
DATA const_signmask_imag<>+0x00(SB)/4, $0x00000000
DATA const_signmask_imag<>+0x04(SB)/4, $0x80000000
DATA const_signmask_imag<>+0x08(SB)/4, $0x00000000
DATA const_signmask_imag<>+0x0C(SB)/4, $0x80000000
DATA const_signmask_imag<>+0x10(SB)/4, $0x00000000
DATA const_signmask_imag<>+0x14(SB)/4, $0x80000000
DATA const_signmask_imag<>+0x18(SB)/4, $0x00000000
DATA const_signmask_imag<>+0x1C(SB)/4, $0x80000000
GLOBL const_signmask_imag<>(SB), RODATA|NOPTR, $32

// Sign mask for full negation [-, -, -, -, ...]
DATA const_signmask_full<>+0x00(SB)/4, $0x80000000
DATA const_signmask_full<>+0x04(SB)/4, $0x80000000
DATA const_signmask_full<>+0x08(SB)/4, $0x80000000
DATA const_signmask_full<>+0x0C(SB)/4, $0x80000000
DATA const_signmask_full<>+0x10(SB)/4, $0x80000000
DATA const_signmask_full<>+0x14(SB)/4, $0x80000000
DATA const_signmask_full<>+0x18(SB)/4, $0x80000000
DATA const_signmask_full<>+0x1C(SB)/4, $0x80000000
GLOBL const_signmask_full<>(SB), RODATA|NOPTR, $32

// ===========================================================================
// Forward transform, size 512, complex64, radix-16x32
// ===========================================================================
//
// func ForwardAVX2Size512Radix16x32Complex64Asm(
//     dst []complex64,      // 0(FP): ptr, 8(FP): len, 16(FP): cap
//     src []complex64,      // 24(FP): ptr, 32(FP): len, 40(FP): cap
//     twiddle []complex64,  // 48(FP): ptr, 56(FP): len, 64(FP): cap
//     scratch []complex64,  // 72(FP): ptr, 80(FP): len, 88(FP): cap
//     bitrev []int,         // 96(FP): ptr, 104(FP): len, 112(FP): cap
// ) bool                    // 120(FP): return value
//
// Stack frame: 8192 bytes for intermediate storage
//   0-4095:    Stage 1 output buffer (512 complex64 = 4096 bytes)
//   4096-4223: FFT-16 workspace for even elements (16 complex64 = 128 bytes)
//   4224-4351: FFT-16 workspace for odd elements (16 complex64 = 128 bytes)
//   4352-4479: FFT-16 result E (16 complex64 = 128 bytes)
//   4480-4607: FFT-16 result O (16 complex64 = 128 bytes)
//
// Register allocation:
//   R8  = dst pointer
//   R9  = src pointer
//   R10 = twiddle pointer
//   R11 = scratch pointer (stage 1 output on stack)
//   R12 = loop counter (column/row index)
//   R13 = temporary
//   R14 = stack frame base
//   R15 = temporary
//
// ===========================================================================
TEXT ·ForwardAVX2Size512Radix16x32Complex64Asm(SB), $8192-121
	// ===== VALIDATION =====
	MOVQ src+32(FP), AX
	CMPQ AX, $512
	JL   fwd_fail
	MOVQ dst+8(FP), AX
	CMPQ AX, $512
	JL   fwd_fail
	MOVQ twiddle+56(FP), AX
	CMPQ AX, $512
	JL   fwd_fail
	MOVQ scratch+80(FP), AX
	CMPQ AX, $512
	JL   fwd_fail

	// ===== SETUP =====
	MOVQ dst+0(FP), R8          // R8 = dst
	MOVQ src+24(FP), R9         // R9 = src
	MOVQ twiddle+48(FP), R10    // R10 = twiddle table
	LEAQ 0(SP), R14             // R14 = stack frame base
	LEAQ 0(R14), R11            // R11 = stage 1 output buffer (on stack)

	// =======================================================================
	// STAGE 1: 16 column FFT-32s
	// =======================================================================
	// For each column n1 = 0..15:
	//   - Load 16 even elements e[k] = src[(2k)*16 + n1] for k=0..15
	//   - Load 16 odd elements  o[k] = src[(2k+1)*16 + n1] for k=0..15
	//   - E = FFT-16(e[bitrev16])
	//   - O = FFT-16(o[bitrev16])
	//   - Combine: out[k2*16+n1] = (E[k2] + W_32^k2 * O[k2]) * W_512^(k2*n1)
	//            out[(k2+16)*16+n1] = (E[k2] - W_32^k2 * O[k2]) * W_512^((k2+16)*n1)
	// =======================================================================

	XORQ R12, R12               // R12 = n1 (column 0..15)

fwd_stage1_col_loop:
	CMPQ R12, $16
	JGE  fwd_stage2_start

	// Calculate source column offset: n1 * 8 bytes
	MOVQ R12, DI
	SHLQ $3, DI                 // DI = n1 * 8 (byte offset for column)
	LEAQ (R9)(DI*1), SI         // SI = &src[n1]

	// ----- Load 16 even elements in bit-reversed order for FFT-16 -----
	// e[k] = src[32*k + n1], stride = 256 bytes per element
	// FFT-16 expects bit-reversed input: v0=e[0], v1=e[8], v2=e[4], v3=e[12], ...
	// Bit-reversal mapping: [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
	//
	// Load directly to registers/stack in bit-reversed order:
	// v0  = e[0]  at offset 0*256 = 0
	// v1  = e[8]  at offset 8*256 = 2048
	// v2  = e[4]  at offset 4*256 = 1024
	// v3  = e[12] at offset 12*256 = 3072
	// v4  = e[2]  at offset 2*256 = 512
	// v5  = e[10] at offset 10*256 = 2560
	// v6  = e[6]  at offset 6*256 = 1536
	// v7  = e[14] at offset 14*256 = 3584
	// v8  = e[1]  at offset 1*256 = 256
	// v9  = e[9]  at offset 9*256 = 2304
	// v10 = e[5]  at offset 5*256 = 1280
	// v11 = e[13] at offset 13*256 = 3328
	// v12 = e[3]  at offset 3*256 = 768
	// v13 = e[11] at offset 11*256 = 2816
	// v14 = e[7]  at offset 7*256 = 1792
	// v15 = e[15] at offset 15*256 = 3840

	// Load v0-v11 directly into X0-X11, v12-v15 to stack
	VMOVSD 0(SI), X0           // v0 = e[0]
	VMOVSD 2048(SI), X1        // v1 = e[8]
	VMOVSD 1024(SI), X2        // v2 = e[4]
	VMOVSD 3072(SI), X3        // v3 = e[12]
	VMOVSD 512(SI), X4         // v4 = e[2]
	VMOVSD 2560(SI), X5        // v5 = e[10]
	VMOVSD 1536(SI), X6        // v6 = e[6]
	VMOVSD 3584(SI), X7        // v7 = e[14]
	VMOVSD 256(SI), X8         // v8 = e[1]
	VMOVSD 2304(SI), X9        // v9 = e[9]
	VMOVSD 1280(SI), X10       // v10 = e[5]
	VMOVSD 3328(SI), X11       // v11 = e[13]
	// v12-v15 to stack (we'll load when needed)
	VMOVSD 768(SI), X12        // v12 = e[3]
	VMOVSD X12, 4192(R14)
	VMOVSD 2816(SI), X12       // v13 = e[11]
	VMOVSD X12, 4200(R14)
	VMOVSD 1792(SI), X12       // v14 = e[7]
	VMOVSD X12, 4208(R14)
	VMOVSD 3840(SI), X12       // v15 = e[15]
	VMOVSD X12, 4216(R14)

	// ----- Load 16 odd elements in bit-reversed order -----
	// o[k] = src[32*k + 16 + n1], stride = 256 bytes, base offset = 128
	// Same bit-reversal: v0=o[0], v1=o[8], v2=o[4], etc.
	// Store to 4224(R14) for later processing
	VMOVSD 128(SI), X12        // v0 = o[0]
	VMOVSD X12, 4224(R14)
	VMOVSD 2176(SI), X12       // v1 = o[8]
	VMOVSD X12, 4232(R14)
	VMOVSD 1152(SI), X12       // v2 = o[4]
	VMOVSD X12, 4240(R14)
	VMOVSD 3200(SI), X12       // v3 = o[12]
	VMOVSD X12, 4248(R14)
	VMOVSD 640(SI), X12        // v4 = o[2]
	VMOVSD X12, 4256(R14)
	VMOVSD 2688(SI), X12       // v5 = o[10]
	VMOVSD X12, 4264(R14)
	VMOVSD 1664(SI), X12       // v6 = o[6]
	VMOVSD X12, 4272(R14)
	VMOVSD 3712(SI), X12       // v7 = o[14]
	VMOVSD X12, 4280(R14)
	VMOVSD 384(SI), X12        // v8 = o[1]
	VMOVSD X12, 4288(R14)
	VMOVSD 2432(SI), X12       // v9 = o[9]
	VMOVSD X12, 4296(R14)
	VMOVSD 1408(SI), X12       // v10 = o[5]
	VMOVSD X12, 4304(R14)
	VMOVSD 3456(SI), X12       // v11 = o[13]
	VMOVSD X12, 4312(R14)
	VMOVSD 896(SI), X12        // v12 = o[3]
	VMOVSD X12, 4320(R14)
	VMOVSD 2944(SI), X12       // v13 = o[11]
	VMOVSD X12, 4328(R14)
	VMOVSD 1920(SI), X12       // v14 = o[7]
	VMOVSD X12, 4336(R14)
	VMOVSD 3968(SI), X12       // v15 = o[15]
	VMOVSD X12, 4344(R14)

	// ===== FFT-16 on even elements =====
	// X0-X11 already contain v0-v11, v12-v15 at 4192, 4200, 4208, 4216

	// ----- FFT-16 Stage 1: 8 radix-2 butterflies -----
	// (v0,v1), (v2,v3), (v4,v5), (v6,v7), (v8,v9), (v10,v11), (v12,v13), (v14,v15)

	VADDPS X1, X0, X12         // t = v0 + v1
	VSUBPS X1, X0, X1          // v1 = v0 - v1
	VMOVAPS X12, X0            // v0 = t

	VADDPS X3, X2, X12
	VSUBPS X3, X2, X3
	VMOVAPS X12, X2

	VADDPS X5, X4, X12
	VSUBPS X5, X4, X5
	VMOVAPS X12, X4

	VADDPS X7, X6, X12
	VSUBPS X7, X6, X7
	VMOVAPS X12, X6

	VADDPS X9, X8, X12
	VSUBPS X9, X8, X9
	VMOVAPS X12, X8

	VADDPS X11, X10, X12
	VSUBPS X11, X10, X11
	VMOVAPS X12, X10

	// v12, v13 butterfly (from memory)
	VMOVSD 4192(R14), X12      // v12
	VMOVSD 4200(R14), X13      // v13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4192(R14)      // new v12
	VMOVSD X13, 4200(R14)      // new v13

	// v14, v15 butterfly (from memory)
	VMOVSD 4208(R14), X12      // v14
	VMOVSD 4216(R14), X13      // v15
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4208(R14)      // new v14
	VMOVSD X13, 4216(R14)      // new v15

	// ----- FFT-16 Stage 2: radix-4 with -i rotation -----
	// Apply -i to v3, v7, v11, v15 (swap re/im, negate new imag)
	// -i * (a+bi) = (b - ai) = (b, -a)

	// Load sign mask for -i rotation
	VMOVUPS const_signmask_imag<>(SB), X15

	// v0, v2 butterfly (no twiddle)
	VADDPS X2, X0, X12
	VSUBPS X2, X0, X2
	VMOVAPS X12, X0

	// v1, v3 with -i*v3
	VSHUFPS $0xB1, X3, X3, X12 // swap re/im: (im, re)
	VXORPS X15, X12, X12       // negate im: (im, -re) = -i * v3
	VADDPS X12, X1, X13        // v1 + (-i)*v3
	VSUBPS X12, X1, X3         // v1 - (-i)*v3 -> new v3
	VMOVAPS X13, X1            // new v1

	// v4, v6 butterfly
	VADDPS X6, X4, X12
	VSUBPS X6, X4, X6
	VMOVAPS X12, X4

	// v5, v7 with -i*v7
	VSHUFPS $0xB1, X7, X7, X12
	VXORPS X15, X12, X12
	VADDPS X12, X5, X13
	VSUBPS X12, X5, X7
	VMOVAPS X13, X5

	// v8, v10 butterfly
	VADDPS X10, X8, X12
	VSUBPS X10, X8, X10
	VMOVAPS X12, X8

	// v9, v11 with -i*v11
	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X15, X12, X12
	VADDPS X12, X9, X13
	VSUBPS X12, X9, X11
	VMOVAPS X13, X9

	// v12, v14 butterfly (from memory)
	VMOVSD 4192(R14), X12      // v12
	VMOVSD 4208(R14), X13      // v14
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4192(R14)      // new v12
	VMOVSD X13, 4208(R14)      // new v14

	// v13, v15 with -i*v15 (from memory)
	VMOVSD 4200(R14), X12      // v13
	VMOVSD 4216(R14), X13      // v15
	VSHUFPS $0xB1, X13, X13, X14
	VXORPS X15, X14, X14       // -i * v15
	VADDPS X14, X12, X13       // v13 + (-i)*v15
	VSUBPS X14, X12, X12       // v13 - (-i)*v15 -> new v15
	VMOVSD X13, 4200(R14)      // new v13
	VMOVSD X12, 4216(R14)      // new v15

	// ----- FFT-16 Stage 3: W_8 twiddles -----
	// W_8^0 = 1, W_8^1 = isq2*(1-i), W_8^2 = -i, W_8^3 = isq2*(-1-i)
	// Pairs: (v0,v4), (v1,v5), (v2,v6), (v3,v7)
	//        (v8,v12), (v9,v13), (v10,v14), (v11,v15)

	// Load isq2 constant
	VBROADCASTSS const_isq2<>(SB), X15

	// v0, v4: W_8^0 = 1 (no twiddle)
	VADDPS X4, X0, X12
	VSUBPS X4, X0, X4
	VMOVAPS X12, X0

	// v1, v5: W_8^1 = isq2*(1-i)
	// (a+bi) * isq2*(1-i) = isq2*((a+b) + (b-a)i)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VSHUFPS $0xB1, X5, X5, X12 // X12 = (im, re)
	VADDPS X5, X12, X13        // X13 = (re+im, im+re)
	VSUBPS X5, X12, X12        // X12 = (im-re, re-im)
	// We need (re+im, im-re). Use VUNPCKLPS to interleave low floats.
	// X13[0] = re+im, X12[0] = im-re
	// VUNPCKLPS X12, X13, result -> (X13[0], X12[0], X13[1], X12[1]) = (re+im, im-re, ...)
	VUNPCKLPS X12, X13, X12    // X12 = (re+im, im-re, im+re, re-im)
	VMULPS X15, X12, X12       // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X5
	VMOVAPS X13, X1

	// v2, v6: W_8^2 = -i
	VSHUFPS $0xB1, X6, X6, X12
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X6
	VMOVAPS X13, X2

	// v3, v7: W_8^3 = isq2*(-1-i)
	// (a+bi) * isq2*(-1-i) = isq2*((b-a) + (-(a+b))i)
	// t3.re = isq2*(im - re), t3.im = -isq2*(re + im) = -isq2*(im + re)
	VSHUFPS $0xB1, X7, X7, X12 // X12 = (im, re)
	VSUBPS X7, X12, X13        // X13 = (im-re, re-im), X13[0] = im-re ✓
	VADDPS X7, X12, X12        // X12 = (im+re, re+im), X12[0] = im+re
	// We need (im-re, -(im+re)). Use VUNPCKLPS then negate the im part.
	VUNPCKLPS X12, X13, X12    // X12 = (im-re, im+re, ...)
	VMULPS X15, X12, X12       // X12 = isq2*(im-re, im+re, ...)
	// Negate just the imaginary component (position 1)
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12       // X12 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X12, X3, X13
	VSUBPS X12, X3, X7
	VMOVAPS X13, X3

	// v8, v12: W_8^0 = 1 (from memory for v12)
	VMOVSD 4192(R14), X12      // v12
	VADDPS X12, X8, X13
	VSUBPS X12, X8, X12
	VMOVAPS X13, X8
	VMOVSD X12, 4192(R14)      // new v12

	// v9, v13: W_8^1 = isq2*(1-i) (v13 from memory)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VMOVSD 4200(R14), X12      // v13 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VADDPS X12, X13, X14       // X14 = (re+im, im+re)
	VSUBPS X12, X13, X13       // X13 = (im-re, re-im)
	VUNPCKLPS X13, X14, X13    // X13 = (re+im, im-re, ...)
	VMULPS X15, X13, X13       // X13 = isq2 * (re+im, im-re, ...)
	VADDPS X13, X9, X14
	VSUBPS X13, X9, X13
	VMOVAPS X14, X9
	VMOVSD X13, 4200(R14)      // new v13

	// v10, v14: W_8^2 = -i (v14 from memory)
	VMOVSD 4208(R14), X12      // v14
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X13, X13
	VADDPS X13, X10, X14
	VSUBPS X13, X10, X13
	VMOVAPS X14, X10
	VMOVSD X13, 4208(R14)      // new v14

	// v11, v15: W_8^3 = isq2*(-1-i) (v15 from memory)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VMOVSD 4216(R14), X12      // v15 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X11, X14
	VSUBPS X13, X11, X13
	VMOVAPS X14, X11
	VMOVSD X13, 4216(R14)      // new v15

	// ----- FFT-16 Stage 4: W_16 twiddles -----
	// Pairs: (v0,v8), (v1,v9), (v2,v10), (v3,v11), (v4,v12), (v5,v13), (v6,v14), (v7,v15)
	// Twiddles: W_16^0=1, W_16^1=(cos1,-sin1), W_16^2=isq2*(1-i), W_16^3=(sin1,-cos1)
	//           W_16^4=-i, W_16^5=(cos1*(-1)-sin1*i)=(-cos1,-sin1), etc.
	// Actually W_16^k = exp(-2πik/16) = cos(2πk/16) - i*sin(2πk/16)

	// Load constants
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v0, v8: W_16^0 = 1
	VADDPS X8, X0, X12
	VSUBPS X8, X0, X8
	VMOVAPS X12, X0

	// v1, v9: W_16^1 = (cos1, -sin1)
	// (a+bi) * (cos1 - sin1*i) = (a*cos1 + b*sin1) + (b*cos1 - a*sin1)i
	// t3.re = re*cos1 + im*sin1, t3.im = im*cos1 - re*sin1
	VSHUFPS $0xB1, X9, X9, X12 // (im, re)
	VMULPS X14, X9, X15        // (re*cos1, im*cos1)
	VMULPS X13, X12, X12       // (im*sin1, re*sin1)
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12    // (re*cos1+im*sin1, im*cos1-re*sin1)
	VADDPS X12, X1, X15
	VSUBPS X12, X1, X9
	VMOVAPS X15, X1

	// v2, v10: W_16^2 = isq2*(1-i) (same as W_8^1)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VBROADCASTSS const_isq2<>(SB), X15
	VSHUFPS $0xB1, X10, X10, X12 // X12 = (im, re)
	VADDPS X10, X12, X13         // X13 = (re+im, im+re)
	VSUBPS X10, X12, X12         // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12      // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12         // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X10
	VMOVAPS X13, X2

	// Reload sin1, cos1
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v3, v11: W_16^3 = (sin1, -cos1)
	// (a+bi) * (sin1 - cos1*i) = (a*sin1 + b*cos1) + (b*sin1 - a*cos1)i
	VSHUFPS $0xB1, X11, X11, X12
	VMULPS X13, X11, X15       // (re*sin1, im*sin1)
	VMULPS X14, X12, X12       // (im*cos1, re*cos1)
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12    // (re*sin1+im*cos1, im*sin1-re*cos1)
	VADDPS X12, X3, X15
	VSUBPS X12, X3, X11
	VMOVAPS X15, X3

	// v4, v12: W_16^4 = -i (v12 from memory)
	VMOVSD 4192(R14), X12      // v12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X15
	VXORPS X15, X13, X13
	VADDPS X13, X4, X15
	VSUBPS X13, X4, X12
	VMOVAPS X15, X4
	VMOVSD X12, 4192(R14)      // new v12

	// Reload constants
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v5, v13: W_16^5 = (-sin1, -cos1) = -(sin1, cos1)
	// Actually W_16^5 = cos(5π/8) - i*sin(5π/8) = -sin1 - i*cos1
	// (a+bi) * (-sin1 - cos1*i) = (-a*sin1 + b*cos1) + (-b*sin1 - a*cos1)i
	VMOVSD 4200(R14), X12      // v13
	VSHUFPS $0xB1, X12, X12, X15
	VMULPS X13, X12, X12       // (re*sin1, im*sin1)
	VMULPS X14, X15, X15       // (im*cos1, re*cos1)
	VSUBPS X12, X15, X15       // (im*cos1 - re*sin1, re*cos1 - im*sin1)
	VMOVUPS const_signmask_full<>(SB), X12
	// Actually let me recalculate: (-sin1 - i*cos1) * (a+bi)
	// = -a*sin1 - a*i*cos1 - i*b*sin1 - i²*b*cos1
	// = -a*sin1 + b*cos1 + i*(-a*cos1 - b*sin1)
	// t3.re = -re*sin1 + im*cos1 = im*cos1 - re*sin1
	// t3.im = -re*cos1 - im*sin1 = -(re*cos1 + im*sin1)
	VMOVSD 4200(R14), X12      // v13 again
	VSHUFPS $0xB1, X12, X12, X15 // (im, re)
	VMULPS X14, X15, X15       // (im*cos1, re*cos1)
	VMULPS X13, X12, X12       // (re*sin1, im*sin1)
	VSUBPS X12, X15, X15       // (im*cos1-re*sin1, re*cos1-im*sin1)
	// Now need to negate the imag part: result.im = -(re*cos1 + im*sin1)
	// But we have (re*cos1 - im*sin1), need -(re*cos1 + im*sin1)
	// Let me redo: we have (im*cos1-re*sin1, re*cos1-im*sin1)
	// We want (im*cos1-re*sin1, -(re*cos1+im*sin1))
	// Hmm, this isn't matching. Let me recalculate W_16^5.
	// W_16^5 = exp(-2πi*5/16) = cos(5π/8) - i*sin(5π/8)
	// cos(5π/8) = cos(π - 3π/8) = -cos(3π/8) = -sin(π/8) = -sin1
	// sin(5π/8) = sin(π - 3π/8) = sin(3π/8) = cos(π/8) = cos1
	// So W_16^5 = -sin1 - i*cos1
	// (a+bi)*(-sin1 - i*cos1) = -a*sin1 - ai*cos1 - bi*sin1 + b*cos1
	//                        = (b*cos1 - a*sin1) + i*(-a*cos1 - b*sin1)
	// t3.re = im*cos1 - re*sin1
	// t3.im = -(re*cos1 + im*sin1)
	VMOVSD 4200(R14), X12      // v13
	VMULPS X14, X12, X15       // (re*cos1, im*cos1)
	VSHUFPS $0xB1, X12, X12, X12 // (im, re)
	VMULPS X13, X12, X12       // (im*sin1, re*sin1)
	VSHUFPS $0xB1, X12, X12, X12 // (re*sin1, im*sin1)
	// Now X15 = (re*cos1, im*cos1), X12 = (re*sin1, im*sin1)
	// We want: re part = im*cos1 - re*sin1, im part = -(re*cos1 + im*sin1)
	// X15[1] - X12[0] for re, -(X15[0] + X12[1]) for im
	// This is getting complicated. Let me use a cleaner approach.
	VMOVSD 4200(R14), X12      // v13 = (re, im)
	VSHUFPS $0xB1, X12, X12, X15 // (im, re)
	// re_new = im*cos1 - re*sin1
	// im_new = -re*cos1 - im*sin1
	// Construct: (im, -re) then multiply by (cos1, cos1) -> (im*cos1, -re*cos1)
	// and (re, im) multiply by (sin1, sin1) -> (re*sin1, im*sin1)
	// then subtract: (im*cos1 - re*sin1, -re*cos1 - im*sin1)
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15       // (im, -re)
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X15, X15       // (im*cos1, -re*cos1)
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X12, X12       // (re*sin1, im*sin1)
	VSUBPS X12, X15, X12       // (im*cos1 - re*sin1, -re*cos1 - im*sin1)
	VADDPS X12, X5, X15
	VSUBPS X12, X5, X13
	VMOVAPS X15, X5
	VMOVSD X13, 4200(R14)      // new v13

	// v6, v14: W_16^6 = isq2*(-1-i) (same as W_8^3)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VBROADCASTSS const_isq2<>(SB), X15
	VMOVSD 4208(R14), X12      // v14 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X6, X14
	VSUBPS X13, X6, X13
	VMOVAPS X14, X6
	VMOVSD X13, 4208(R14)      // new v14

	// v7, v15: W_16^7 = (-cos1, -sin1)
	// W_16^7 = cos(7π/8) - i*sin(7π/8)
	// cos(7π/8) = -cos(π/8) = -cos1
	// sin(7π/8) = sin(π/8) = sin1
	// So W_16^7 = -cos1 - i*sin1
	// (a+bi)*(-cos1 - i*sin1) = -a*cos1 - ai*sin1 - bi*cos1 + b*sin1
	//                        = (b*sin1 - a*cos1) + i*(-a*sin1 - b*cos1)
	// t3.re = im*sin1 - re*cos1
	// t3.im = -(re*sin1 + im*cos1)
	VMOVSD 4216(R14), X12      // v15 = (re, im)
	VSHUFPS $0xB1, X12, X12, X15 // (im, re)
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15       // (im, -re)
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X15, X15       // (im*sin1, -re*sin1)
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X12, X12       // (re*cos1, im*cos1)
	VSUBPS X12, X15, X12       // (im*sin1 - re*cos1, -re*sin1 - im*cos1)
	VADDPS X12, X7, X15
	VSUBPS X12, X7, X13
	VMOVAPS X15, X7
	VMOVSD X13, 4216(R14)      // new v15

	// Store FFT-16 results for even elements to 4352(R14)
	VMOVSD X0, 4352(R14)       // E[0]
	VMOVSD X1, 4360(R14)       // E[1]
	VMOVSD X2, 4368(R14)       // E[2]
	VMOVSD X3, 4376(R14)       // E[3]
	VMOVSD X4, 4384(R14)       // E[4]
	VMOVSD X5, 4392(R14)       // E[5]
	VMOVSD X6, 4400(R14)       // E[6]
	VMOVSD X7, 4408(R14)       // E[7]
	VMOVSD X8, 4416(R14)       // E[8]
	VMOVSD X9, 4424(R14)       // E[9]
	VMOVSD X10, 4432(R14)      // E[10]
	VMOVSD X11, 4440(R14)      // E[11]
	VMOVSD 4192(R14), X0       // v12 -> E[12]
	VMOVSD X0, 4448(R14)
	VMOVSD 4200(R14), X0       // v13 -> E[13]
	VMOVSD X0, 4456(R14)
	VMOVSD 4208(R14), X0       // v14 -> E[14]
	VMOVSD X0, 4464(R14)
	VMOVSD 4216(R14), X0       // v15 -> E[15]
	VMOVSD X0, 4472(R14)

	// ===== FFT-16 on odd elements =====
	// Odd elements already loaded to stack at 4224(R14) in bit-reversed order
	// Load v0-v11 into X0-X11, v12-v15 stay at 4320, 4328, 4336, 4344
	VMOVSD 4224(R14), X0       // v0
	VMOVSD 4232(R14), X1       // v1
	VMOVSD 4240(R14), X2       // v2
	VMOVSD 4248(R14), X3       // v3
	VMOVSD 4256(R14), X4       // v4
	VMOVSD 4264(R14), X5       // v5
	VMOVSD 4272(R14), X6       // v6
	VMOVSD 4280(R14), X7       // v7
	VMOVSD 4288(R14), X8       // v8
	VMOVSD 4296(R14), X9       // v9
	VMOVSD 4304(R14), X10      // v10
	VMOVSD 4312(R14), X11      // v11
	// v12-v15 stay at 4320, 4328, 4336, 4344

	// ----- FFT-16 Stage 1: 8 radix-2 butterflies -----
	VADDPS X1, X0, X12
	VSUBPS X1, X0, X1
	VMOVAPS X12, X0

	VADDPS X3, X2, X12
	VSUBPS X3, X2, X3
	VMOVAPS X12, X2

	VADDPS X5, X4, X12
	VSUBPS X5, X4, X5
	VMOVAPS X12, X4

	VADDPS X7, X6, X12
	VSUBPS X7, X6, X7
	VMOVAPS X12, X6

	VADDPS X9, X8, X12
	VSUBPS X9, X8, X9
	VMOVAPS X12, X8

	VADDPS X11, X10, X12
	VSUBPS X11, X10, X11
	VMOVAPS X12, X10

	VMOVSD 4320(R14), X12
	VMOVSD 4328(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4320(R14)
	VMOVSD X13, 4328(R14)

	VMOVSD 4336(R14), X12
	VMOVSD 4344(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4336(R14)
	VMOVSD X13, 4344(R14)

	// ----- FFT-16 Stage 2: radix-4 with -i rotation -----
	VMOVUPS const_signmask_imag<>(SB), X15

	VADDPS X2, X0, X12
	VSUBPS X2, X0, X2
	VMOVAPS X12, X0

	VSHUFPS $0xB1, X3, X3, X12
	VXORPS X15, X12, X12
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X3
	VMOVAPS X13, X1

	VADDPS X6, X4, X12
	VSUBPS X6, X4, X6
	VMOVAPS X12, X4

	VSHUFPS $0xB1, X7, X7, X12
	VXORPS X15, X12, X12
	VADDPS X12, X5, X13
	VSUBPS X12, X5, X7
	VMOVAPS X13, X5

	VADDPS X10, X8, X12
	VSUBPS X10, X8, X10
	VMOVAPS X12, X8

	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X15, X12, X12
	VADDPS X12, X9, X13
	VSUBPS X12, X9, X11
	VMOVAPS X13, X9

	VMOVSD 4320(R14), X12
	VMOVSD 4336(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4320(R14)
	VMOVSD X13, 4336(R14)

	VMOVSD 4328(R14), X12
	VMOVSD 4344(R14), X13
	VSHUFPS $0xB1, X13, X13, X14
	VXORPS X15, X14, X14
	VADDPS X14, X12, X13
	VSUBPS X14, X12, X12
	VMOVSD X13, 4328(R14)
	VMOVSD X12, 4344(R14)

	// ----- FFT-16 Stage 3: W_8 twiddles -----
	VBROADCASTSS const_isq2<>(SB), X15

	VADDPS X4, X0, X12
	VSUBPS X4, X0, X4
	VMOVAPS X12, X0

	// v1, v5: W_8^1 = isq2*(1-i)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VSHUFPS $0xB1, X5, X5, X12 // X12 = (im, re)
	VADDPS X5, X12, X13        // X13 = (re+im, im+re)
	VSUBPS X5, X12, X12        // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12    // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12       // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X5
	VMOVAPS X13, X1

	// v2, v6: W_8^2 = -i
	VSHUFPS $0xB1, X6, X6, X12
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X6
	VMOVAPS X13, X2

	// v3, v7: W_8^3 = isq2*(-1-i)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VSHUFPS $0xB1, X7, X7, X12 // X12 = (im, re)
	VSUBPS X7, X12, X13        // X13 = (im-re, re-im), X13[0] = im-re ✓
	VADDPS X7, X12, X12        // X12 = (im+re, re+im), X12[0] = im+re
	VUNPCKLPS X12, X13, X12    // X12 = (im-re, im+re, ...)
	VMULPS X15, X12, X12       // X12 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12       // X12 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X12, X3, X13
	VSUBPS X12, X3, X7
	VMOVAPS X13, X3

	// v8, v12: W_8^0
	VMOVSD 4320(R14), X12
	VADDPS X12, X8, X13
	VSUBPS X12, X8, X12
	VMOVAPS X13, X8
	VMOVSD X12, 4320(R14)

	// v9, v13: W_8^1 = isq2*(1-i)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VMOVSD 4328(R14), X12      // v13 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VADDPS X12, X13, X14       // X14 = (re+im, im+re)
	VSUBPS X12, X13, X13       // X13 = (im-re, re-im)
	VUNPCKLPS X13, X14, X13    // X13 = (re+im, im-re, ...)
	VMULPS X15, X13, X13       // X13 = isq2 * (re+im, im-re, ...)
	VADDPS X13, X9, X14
	VSUBPS X13, X9, X13
	VMOVAPS X14, X9
	VMOVSD X13, 4328(R14)

	// v10, v14: W_8^2
	VMOVSD 4336(R14), X12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X13, X13
	VADDPS X13, X10, X14
	VSUBPS X13, X10, X13
	VMOVAPS X14, X10
	VMOVSD X13, 4336(R14)

	// v11, v15: W_8^3 = isq2*(-1-i)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VMOVSD 4344(R14), X12      // v15 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X11, X14
	VSUBPS X13, X11, X13
	VMOVAPS X14, X11
	VMOVSD X13, 4344(R14)

	// ----- FFT-16 Stage 4: W_16 twiddles -----
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v0, v8: W_16^0 = 1
	VADDPS X8, X0, X12
	VSUBPS X8, X0, X8
	VMOVAPS X12, X0

	// v1, v9: W_16^1
	VSHUFPS $0xB1, X9, X9, X12
	VMULPS X14, X9, X15
	VMULPS X13, X12, X12
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12
	VADDPS X12, X1, X15
	VSUBPS X12, X1, X9
	VMOVAPS X15, X1

	// v2, v10: W_16^2 = isq2*(1-i)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VBROADCASTSS const_isq2<>(SB), X15
	VSHUFPS $0xB1, X10, X10, X12 // X12 = (im, re)
	VADDPS X10, X12, X13         // X13 = (re+im, im+re)
	VSUBPS X10, X12, X12         // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12      // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12         // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X10
	VMOVAPS X13, X2

	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v3, v11: W_16^3
	VSHUFPS $0xB1, X11, X11, X12
	VMULPS X13, X11, X15
	VMULPS X14, X12, X12
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12
	VADDPS X12, X3, X15
	VSUBPS X12, X3, X11
	VMOVAPS X15, X3

	// v4, v12: W_16^4 = -i
	VMOVSD 4320(R14), X12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X15
	VXORPS X15, X13, X13
	VADDPS X13, X4, X15
	VSUBPS X13, X4, X12
	VMOVAPS X15, X4
	VMOVSD X12, 4320(R14)

	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v5, v13: W_16^5
	VMOVSD 4328(R14), X12
	VSHUFPS $0xB1, X12, X12, X15
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X15, X15
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X12, X12
	VSUBPS X12, X15, X12
	VADDPS X12, X5, X15
	VSUBPS X12, X5, X13
	VMOVAPS X15, X5
	VMOVSD X13, 4328(R14)

	// v6, v14: W_16^6 = isq2*(-1-i)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VBROADCASTSS const_isq2<>(SB), X15
	VMOVSD 4336(R14), X12      // v14 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X6, X14
	VSUBPS X13, X6, X13
	VMOVAPS X14, X6
	VMOVSD X13, 4336(R14)

	// v7, v15: W_16^7
	VMOVSD 4344(R14), X12
	VSHUFPS $0xB1, X12, X12, X15
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X15, X15
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X12, X12
	VSUBPS X12, X15, X12
	VADDPS X12, X7, X15
	VSUBPS X12, X7, X13
	VMOVAPS X15, X7
	VMOVSD X13, 4344(R14)

	// Store FFT-16 results for odd elements to 4480(R14)
	VMOVSD X0, 4480(R14)       // O[0]
	VMOVSD X1, 4488(R14)       // O[1]
	VMOVSD X2, 4496(R14)       // O[2]
	VMOVSD X3, 4504(R14)       // O[3]
	VMOVSD X4, 4512(R14)       // O[4]
	VMOVSD X5, 4520(R14)       // O[5]
	VMOVSD X6, 4528(R14)       // O[6]
	VMOVSD X7, 4536(R14)       // O[7]
	VMOVSD X8, 4544(R14)       // O[8]
	VMOVSD X9, 4552(R14)       // O[9]
	VMOVSD X10, 4560(R14)      // O[10]
	VMOVSD X11, 4568(R14)      // O[11]
	VMOVSD 4320(R14), X0
	VMOVSD X0, 4576(R14)       // O[12]
	VMOVSD 4328(R14), X0
	VMOVSD X0, 4584(R14)       // O[13]
	VMOVSD 4336(R14), X0
	VMOVSD X0, 4592(R14)       // O[14]
	VMOVSD 4344(R14), X0
	VMOVSD X0, 4600(R14)       // O[15]

	// ===== Combine E and O with W_32 and W_512 twiddles =====
	// For k2 = 0..15:
	//   t = O[k2] * tw[k2*16]  (W_32^k2 = tw[k2*16] from 512-point table)
	//   out[k2*16 + n1] = (E[k2] + t) * tw[k2*n1]
	//   out[(k2+16)*16 + n1] = (E[k2] - t) * tw[(k2+16)*n1]

	// Calculate output offset for this column
	MOVQ R12, DI               // DI = n1
	SHLQ $3, DI                // DI = n1 * 8 (byte offset)

	// k2 = 0: W_32^0 = tw[0] = 1, W_512^0 = 1, W_512^(16*n1) = tw[16*n1]
	VMOVSD 4352(R14), X0       // E[0]
	VMOVSD 4480(R14), X1       // O[0]
	VADDPS X1, X0, X2          // E[0] + O[0]
	VSUBPS X1, X0, X3          // E[0] - O[0]
	// tw[0] = 1, no multiply needed for first result
	VMOVSD X2, (R11)(DI*1)     // out[0*16 + n1]
	// tw[16*n1]: need to load from twiddle table
	MOVQ R12, AX
	SHLQ $4, AX                // AX = n1 * 16
	SHLQ $3, AX                // AX = n1 * 16 * 8 = n1 * 128
	VMOVSD (R10)(AX*1), X4     // tw[16*n1]
	// Complex multiply: X3 * X4
	VMOVSLDUP X4, X5           // (tw.re, tw.re)
	VMOVSHDUP X4, X6           // (tw.im, tw.im)
	VMULPS X5, X3, X5          // (a.re*tw.re, a.im*tw.re)
	VSHUFPS $0xB1, X3, X3, X3  // (a.im, a.re)
	VMULPS X6, X3, X3          // (a.im*tw.im, a.re*tw.im)
	VADDSUBPS X3, X5, X3       // (a.re*tw.re - a.im*tw.im, a.im*tw.re + a.re*tw.im)
	LEAQ 2048(R11), AX         // out + 16*16*8 = out + 2048
	VMOVSD X3, (AX)(DI*1)      // out[16*16 + n1]

	// k2 = 1..15: Full twiddle multiplications
	// This is getting very long. Let me create a loop.

	MOVQ $1, R13               // k2 = 1
	MOVQ R12, R15              // tw_index = n1 (k2*n1 for k2=1)
	MOVQ R12, BX
	SHLQ $4, BX                // 16*n1
	ADDQ R12, BX               // tw_index2 = (k2+16)*n1 for k2=1

fwd_s1_combine_loop:
	CMPQ R13, $16
	JGE  fwd_s1_col_next

	// Load E[k2] and O[k2]
	MOVQ R13, AX
	SHLQ $3, AX                // AX = k2 * 8
	LEAQ 4352(R14), SI         // E array
	VMOVSD (SI)(AX*1), X0      // E[k2]
	LEAQ 4480(R14), SI         // O array
	VMOVSD (SI)(AX*1), X1      // O[k2]

	// Load W_32^k2 = tw[k2*16]
	MOVQ R13, AX
	SHLQ $4, AX                // AX = k2 * 16
	SHLQ $3, AX                // AX = k2 * 16 * 8 = k2 * 128
	VMOVSD (R10)(AX*1), X2     // tw[k2*16]

	// Complex multiply: O[k2] * tw[k2*16]
	VMOVSLDUP X2, X3           // (tw.re, tw.re)
	VMOVSHDUP X2, X4           // (tw.im, tw.im)
	VMULPS X3, X1, X3          // (o.re*tw.re, o.im*tw.re)
	VSHUFPS $0xB1, X1, X1, X5  // (o.im, o.re)
	VMULPS X4, X5, X5          // (o.im*tw.im, o.re*tw.im)
	VADDSUBPS X5, X3, X1       // t = O[k2] * tw[k2*16]

	// E[k2] + t, E[k2] - t
	VADDPS X1, X0, X2          // sum = E[k2] + t
	VSUBPS X1, X0, X3          // diff = E[k2] - t

	// Apply W_512^(k2*n1) to sum: tw[k2*n1]
	MOVQ R15, AX               // tw_index = k2*n1
	SHLQ $3, AX                // (k2*n1) * 8
	VMOVSD (R10)(AX*1), X4     // tw[k2*n1]
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VMULPS X5, X2, X5
	VSHUFPS $0xB1, X2, X2, X2
	VMULPS X6, X2, X2
	VADDSUBPS X2, X5, X2       // sum * tw[k2*n1]

	// Store to out[k2*16 + n1]
	MOVQ R13, AX               // k2
	SHLQ $4, AX                // k2 * 16
	ADDQ R12, AX               // k2*16 + n1
	SHLQ $3, AX                // byte offset
	VMOVSD X2, (R11)(AX*1)

	// Apply W_512^((k2+16)*n1) to diff: tw[(k2+16)*n1]
	MOVQ BX, AX                // tw_index2 = (k2+16)*n1
	SHLQ $3, AX                // byte offset
	VMOVSD (R10)(AX*1), X4     // tw[(k2+16)*n1]
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VMULPS X5, X3, X5
	VSHUFPS $0xB1, X3, X3, X3
	VMULPS X6, X3, X3
	VADDSUBPS X3, X5, X3       // diff * tw[(k2+16)*n1]

	// Store to out[(k2+16)*16 + n1]
	MOVQ R13, AX               // k2
	ADDQ $16, AX               // k2 + 16
	SHLQ $4, AX                // (k2+16) * 16
	ADDQ R12, AX               // (k2+16)*16 + n1
	SHLQ $3, AX                // byte offset
	VMOVSD X3, (R11)(AX*1)

	ADDQ R12, R15              // tw_index += n1
	ADDQ R12, BX               // tw_index2 += n1
	INCQ R13
	JMP  fwd_s1_combine_loop

fwd_s1_col_next:
	INCQ R12
	JMP  fwd_stage1_col_loop

fwd_stage2_start:
	// =======================================================================
	// STAGE 2: 32 row FFT-16s
	// =======================================================================
	// For each row k2 = 0..31:
	//   - Load 16 elements from out[k2*16 + 0..15] in bit-reversed order
	//   - Perform FFT-16
	//   - Store to dst[32*k1 + k2] for k1 = 0..15

	XORQ R12, R12              // R12 = k2 (row 0..31)

fwd_stage2_row_loop:
	CMPQ R12, $32
	JGE  fwd_success

	// Calculate row base offset: k2 * 16 * 8 = k2 * 128
	MOVQ R12, DI
	SHLQ $7, DI                // DI = k2 * 128

	// Load 16 elements in bit-reversed order for FFT-16
	// Row base at out[k2*16], each element is 8 bytes
	// Bit-reversal: v0=out[0], v1=out[8], v2=out[4], v3=out[12], ...
	// Offsets: 0*8=0, 8*8=64, 4*8=32, 12*8=96, 2*8=16, 10*8=80, 6*8=48, 14*8=112,
	//          1*8=8, 9*8=72, 5*8=40, 13*8=104, 3*8=24, 11*8=88, 7*8=56, 15*8=120
	LEAQ (R11)(DI*1), SI       // SI = &out[k2*16]

	// Load v0-v11 in bit-reversed order, v12-v15 to stack
	VMOVSD 0(SI), X0           // v0 = out[0]
	VMOVSD 64(SI), X1          // v1 = out[8]
	VMOVSD 32(SI), X2          // v2 = out[4]
	VMOVSD 96(SI), X3          // v3 = out[12]
	VMOVSD 16(SI), X4          // v4 = out[2]
	VMOVSD 80(SI), X5          // v5 = out[10]
	VMOVSD 48(SI), X6          // v6 = out[6]
	VMOVSD 112(SI), X7         // v7 = out[14]
	VMOVSD 8(SI), X8           // v8 = out[1]
	VMOVSD 72(SI), X9          // v9 = out[9]
	VMOVSD 40(SI), X10         // v10 = out[5]
	VMOVSD 104(SI), X11        // v11 = out[13]
	// v12-v15 to stack
	VMOVSD 24(SI), X12         // v12 = out[3]
	VMOVSD X12, 4096(R14)
	VMOVSD 88(SI), X12         // v13 = out[11]
	VMOVSD X12, 4104(R14)
	VMOVSD 56(SI), X12         // v14 = out[7]
	VMOVSD X12, 4112(R14)
	VMOVSD 120(SI), X12        // v15 = out[15]
	VMOVSD X12, 4120(R14)

	// ----- FFT-16 Stage 1 -----
	VADDPS X1, X0, X12
	VSUBPS X1, X0, X1
	VMOVAPS X12, X0

	VADDPS X3, X2, X12
	VSUBPS X3, X2, X3
	VMOVAPS X12, X2

	VADDPS X5, X4, X12
	VSUBPS X5, X4, X5
	VMOVAPS X12, X4

	VADDPS X7, X6, X12
	VSUBPS X7, X6, X7
	VMOVAPS X12, X6

	VADDPS X9, X8, X12
	VSUBPS X9, X8, X9
	VMOVAPS X12, X8

	VADDPS X11, X10, X12
	VSUBPS X11, X10, X11
	VMOVAPS X12, X10

	VMOVSD 4096(R14), X12
	VMOVSD 4104(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4096(R14)
	VMOVSD X13, 4104(R14)

	VMOVSD 4112(R14), X12
	VMOVSD 4120(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4112(R14)
	VMOVSD X13, 4120(R14)

	// ----- FFT-16 Stage 2 -----
	VMOVUPS const_signmask_imag<>(SB), X15

	VADDPS X2, X0, X12
	VSUBPS X2, X0, X2
	VMOVAPS X12, X0

	VSHUFPS $0xB1, X3, X3, X12
	VXORPS X15, X12, X12
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X3
	VMOVAPS X13, X1

	VADDPS X6, X4, X12
	VSUBPS X6, X4, X6
	VMOVAPS X12, X4

	VSHUFPS $0xB1, X7, X7, X12
	VXORPS X15, X12, X12
	VADDPS X12, X5, X13
	VSUBPS X12, X5, X7
	VMOVAPS X13, X5

	VADDPS X10, X8, X12
	VSUBPS X10, X8, X10
	VMOVAPS X12, X8

	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X15, X12, X12
	VADDPS X12, X9, X13
	VSUBPS X12, X9, X11
	VMOVAPS X13, X9

	VMOVSD 4096(R14), X12
	VMOVSD 4112(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4096(R14)
	VMOVSD X13, 4112(R14)

	VMOVSD 4104(R14), X12
	VMOVSD 4120(R14), X13
	VSHUFPS $0xB1, X13, X13, X14
	VXORPS X15, X14, X14
	VADDPS X14, X12, X13
	VSUBPS X14, X12, X12
	VMOVSD X13, 4104(R14)
	VMOVSD X12, 4120(R14)

	// ----- FFT-16 Stage 3 -----
	VBROADCASTSS const_isq2<>(SB), X15

	VADDPS X4, X0, X12
	VSUBPS X4, X0, X4
	VMOVAPS X12, X0

	// W_8^1 = isq2*(1-i): t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VSHUFPS $0xB1, X5, X5, X12 // X12 = (im, re)
	VADDPS X5, X12, X13        // X13 = (re+im, im+re)
	VSUBPS X5, X12, X12        // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12    // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12       // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X5
	VMOVAPS X13, X1

	// W_8^2 = -i
	VSHUFPS $0xB1, X6, X6, X12
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X6
	VMOVAPS X13, X2

	// W_8^3 = isq2*(-1-i): t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VSHUFPS $0xB1, X7, X7, X12 // X12 = (im, re)
	VSUBPS X7, X12, X13        // X13 = (im-re, re-im), X13[0] = im-re ✓
	VADDPS X7, X12, X12        // X12 = (im+re, re+im), X12[0] = im+re
	VUNPCKLPS X12, X13, X12    // X12 = (im-re, im+re, ...)
	VMULPS X15, X12, X12       // X12 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12       // X12 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X12, X3, X13
	VSUBPS X12, X3, X7
	VMOVAPS X13, X3

	VMOVSD 4096(R14), X12
	VADDPS X12, X8, X13
	VSUBPS X12, X8, X12
	VMOVAPS X13, X8
	VMOVSD X12, 4096(R14)

	// W_8^1 = isq2*(1-i): t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VMOVSD 4104(R14), X12      // v13 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VADDPS X12, X13, X14       // X14 = (re+im, im+re)
	VSUBPS X12, X13, X13       // X13 = (im-re, re-im)
	VUNPCKLPS X13, X14, X13    // X13 = (re+im, im-re, ...)
	VMULPS X15, X13, X13       // X13 = isq2 * (re+im, im-re, ...)
	VADDPS X13, X9, X14
	VSUBPS X13, X9, X13
	VMOVAPS X14, X9
	VMOVSD X13, 4104(R14)

	// W_8^2 = -i
	VMOVSD 4112(R14), X12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X13, X13
	VADDPS X13, X10, X14
	VSUBPS X13, X10, X13
	VMOVAPS X14, X10
	VMOVSD X13, 4112(R14)

	// W_8^3 = isq2*(-1-i): t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VMOVSD 4120(R14), X12      // v15 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X11, X14
	VSUBPS X13, X11, X13
	VMOVAPS X14, X11
	VMOVSD X13, 4120(R14)

	// ----- FFT-16 Stage 4 -----
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	VADDPS X8, X0, X12
	VSUBPS X8, X0, X8
	VMOVAPS X12, X0

	VSHUFPS $0xB1, X9, X9, X12
	VMULPS X14, X9, X15
	VMULPS X13, X12, X12
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12
	VADDPS X12, X1, X15
	VSUBPS X12, X1, X9
	VMOVAPS X15, X1

	// W_16^2 = isq2*(1-i): t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VBROADCASTSS const_isq2<>(SB), X15
	VSHUFPS $0xB1, X10, X10, X12 // X12 = (im, re)
	VADDPS X10, X12, X13         // X13 = (re+im, im+re)
	VSUBPS X10, X12, X12         // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12      // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12         // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X10
	VMOVAPS X13, X2

	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// W_16^3 uses sin1/cos1
	VSHUFPS $0xB1, X11, X11, X12
	VMULPS X13, X11, X15
	VMULPS X14, X12, X12
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12
	VADDPS X12, X3, X15
	VSUBPS X12, X3, X11
	VMOVAPS X15, X3

	VMOVSD 4096(R14), X12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X15
	VXORPS X15, X13, X13
	VADDPS X13, X4, X15
	VSUBPS X13, X4, X12
	VMOVAPS X15, X4
	VMOVSD X12, 4096(R14)

	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	VMOVSD 4104(R14), X12
	VSHUFPS $0xB1, X12, X12, X15
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X15, X15
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X12, X12
	VSUBPS X12, X15, X12
	VADDPS X12, X5, X15
	VSUBPS X12, X5, X13
	VMOVAPS X15, X5
	VMOVSD X13, 4104(R14)

	// W_16^6 = isq2*(-1-i): t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VBROADCASTSS const_isq2<>(SB), X15
	VMOVSD 4112(R14), X12      // v14 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X6, X14
	VSUBPS X13, X6, X13
	VMOVAPS X14, X6
	VMOVSD X13, 4112(R14)

	VMOVSD 4120(R14), X12
	VSHUFPS $0xB1, X12, X12, X15
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X15, X15
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X12, X12
	VSUBPS X12, X15, X12
	VADDPS X12, X7, X15
	VSUBPS X12, X7, X13
	VMOVAPS X15, X7
	VMOVSD X13, 4120(R14)

	// Store to dst: dst[32*k1 + k2] for k1 = 0..15
	// k2 is in R12, stride = 32 * 8 = 256 bytes
	MOVQ R12, DI
	SHLQ $3, DI                // DI = k2 * 8

	VMOVSD X0, (R8)(DI*1)      // dst[32*0 + k2]
	ADDQ $256, DI
	VMOVSD X1, (R8)(DI*1)      // dst[32*1 + k2]
	ADDQ $256, DI
	VMOVSD X2, (R8)(DI*1)      // dst[32*2 + k2]
	ADDQ $256, DI
	VMOVSD X3, (R8)(DI*1)      // dst[32*3 + k2]
	ADDQ $256, DI
	VMOVSD X4, (R8)(DI*1)      // dst[32*4 + k2]
	ADDQ $256, DI
	VMOVSD X5, (R8)(DI*1)      // dst[32*5 + k2]
	ADDQ $256, DI
	VMOVSD X6, (R8)(DI*1)      // dst[32*6 + k2]
	ADDQ $256, DI
	VMOVSD X7, (R8)(DI*1)      // dst[32*7 + k2]
	ADDQ $256, DI
	VMOVSD X8, (R8)(DI*1)      // dst[32*8 + k2]
	ADDQ $256, DI
	VMOVSD X9, (R8)(DI*1)      // dst[32*9 + k2]
	ADDQ $256, DI
	VMOVSD X10, (R8)(DI*1)     // dst[32*10 + k2]
	ADDQ $256, DI
	VMOVSD X11, (R8)(DI*1)     // dst[32*11 + k2]
	ADDQ $256, DI
	VMOVSD 4096(R14), X0
	VMOVSD X0, (R8)(DI*1)      // dst[32*12 + k2]
	ADDQ $256, DI
	VMOVSD 4104(R14), X0
	VMOVSD X0, (R8)(DI*1)      // dst[32*13 + k2]
	ADDQ $256, DI
	VMOVSD 4112(R14), X0
	VMOVSD X0, (R8)(DI*1)      // dst[32*14 + k2]
	ADDQ $256, DI
	VMOVSD 4120(R14), X0
	VMOVSD X0, (R8)(DI*1)      // dst[32*15 + k2]

	INCQ R12
	JMP  fwd_stage2_row_loop

fwd_success:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

fwd_fail:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 512, complex64, radix-16x32
// ===========================================================================
TEXT ·InverseAVX2Size512Radix16x32Complex64Asm(SB), $8192-121
	// ===== VALIDATION =====
	MOVQ src+32(FP), AX
	CMPQ AX, $512
	JL   inv_fail
	MOVQ dst+8(FP), AX
	CMPQ AX, $512
	JL   inv_fail
	MOVQ twiddle+56(FP), AX
	CMPQ AX, $512
	JL   inv_fail
	MOVQ scratch+80(FP), AX
	CMPQ AX, $512
	JL   inv_fail

	// ===== SETUP =====
	MOVQ dst+0(FP), R8          // R8 = dst
	MOVQ src+24(FP), R9         // R9 = src
	MOVQ twiddle+48(FP), R10    // R10 = twiddle table
	LEAQ 0(SP), R14             // R14 = stack frame base
	LEAQ 0(R14), R11            // R11 = stage 1 output buffer (on stack)

	// ===== CONJUGATE INPUT TO SCRATCH =====
	MOVQ scratch+72(FP), R15    // R15 = scratch (conj input)
	MOVQ R9, SI                 // SI = src
	MOVQ R15, DI                // DI = scratch
	VMOVUPS const_signmask_imag<>(SB), X12
	MOVQ $256, CX               // 512 complex64 = 256 * 16 bytes

inv_conj_in_loop:
	VMOVUPS (SI), X0
	VXORPS X12, X0, X0
	VMOVUPS X0, (DI)
	ADDQ $16, SI
	ADDQ $16, DI
	DECQ CX
	JNZ  inv_conj_in_loop

	MOVQ R15, R9                // R9 = conjugated src

	// =======================================================================
	// STAGE 1: 16 column FFT-32s
	// =======================================================================
	// For each column n1 = 0..15:
	//   - Load 16 even elements e[k] = src[(2k)*16 + n1] for k=0..15
	//   - Load 16 odd elements  o[k] = src[(2k+1)*16 + n1] for k=0..15
	//   - E = FFT-16(e[bitrev16])
	//   - O = FFT-16(o[bitrev16])
	//   - Combine: out[k2*16+n1] = (E[k2] + W_32^k2 * O[k2]) * W_512^(k2*n1)
	//            out[(k2+16)*16+n1] = (E[k2] - W_32^k2 * O[k2]) * W_512^((k2+16)*n1)
	// =======================================================================

	XORQ R12, R12               // R12 = n1 (column 0..15)

inv_stage1_col_loop:
	CMPQ R12, $16
	JGE  inv_stage2_start

	// Calculate source column offset: n1 * 8 bytes
	MOVQ R12, DI
	SHLQ $3, DI                 // DI = n1 * 8 (byte offset for column)
	LEAQ (R9)(DI*1), SI         // SI = &src[n1]

	// ----- Load 16 even elements in bit-reversed order for FFT-16 -----
	// e[k] = src[32*k + n1], stride = 256 bytes per element
	// FFT-16 expects bit-reversed input: v0=e[0], v1=e[8], v2=e[4], v3=e[12], ...
	// Bit-reversal mapping: [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
	//
	// Load directly to registers/stack in bit-reversed order:
	// v0  = e[0]  at offset 0*256 = 0
	// v1  = e[8]  at offset 8*256 = 2048
	// v2  = e[4]  at offset 4*256 = 1024
	// v3  = e[12] at offset 12*256 = 3072
	// v4  = e[2]  at offset 2*256 = 512
	// v5  = e[10] at offset 10*256 = 2560
	// v6  = e[6]  at offset 6*256 = 1536
	// v7  = e[14] at offset 14*256 = 3584
	// v8  = e[1]  at offset 1*256 = 256
	// v9  = e[9]  at offset 9*256 = 2304
	// v10 = e[5]  at offset 5*256 = 1280
	// v11 = e[13] at offset 13*256 = 3328
	// v12 = e[3]  at offset 3*256 = 768
	// v13 = e[11] at offset 11*256 = 2816
	// v14 = e[7]  at offset 7*256 = 1792
	// v15 = e[15] at offset 15*256 = 3840

	// Load v0-v11 directly into X0-X11, v12-v15 to stack
	VMOVSD 0(SI), X0           // v0 = e[0]
	VMOVSD 2048(SI), X1        // v1 = e[8]
	VMOVSD 1024(SI), X2        // v2 = e[4]
	VMOVSD 3072(SI), X3        // v3 = e[12]
	VMOVSD 512(SI), X4         // v4 = e[2]
	VMOVSD 2560(SI), X5        // v5 = e[10]
	VMOVSD 1536(SI), X6        // v6 = e[6]
	VMOVSD 3584(SI), X7        // v7 = e[14]
	VMOVSD 256(SI), X8         // v8 = e[1]
	VMOVSD 2304(SI), X9        // v9 = e[9]
	VMOVSD 1280(SI), X10       // v10 = e[5]
	VMOVSD 3328(SI), X11       // v11 = e[13]
	// v12-v15 to stack (we'll load when needed)
	VMOVSD 768(SI), X12        // v12 = e[3]
	VMOVSD X12, 4192(R14)
	VMOVSD 2816(SI), X12       // v13 = e[11]
	VMOVSD X12, 4200(R14)
	VMOVSD 1792(SI), X12       // v14 = e[7]
	VMOVSD X12, 4208(R14)
	VMOVSD 3840(SI), X12       // v15 = e[15]
	VMOVSD X12, 4216(R14)

	// ----- Load 16 odd elements in bit-reversed order -----
	// o[k] = src[32*k + 16 + n1], stride = 256 bytes, base offset = 128
	// Same bit-reversal: v0=o[0], v1=o[8], v2=o[4], etc.
	// Store to 4224(R14) for later processing
	VMOVSD 128(SI), X12        // v0 = o[0]
	VMOVSD X12, 4224(R14)
	VMOVSD 2176(SI), X12       // v1 = o[8]
	VMOVSD X12, 4232(R14)
	VMOVSD 1152(SI), X12       // v2 = o[4]
	VMOVSD X12, 4240(R14)
	VMOVSD 3200(SI), X12       // v3 = o[12]
	VMOVSD X12, 4248(R14)
	VMOVSD 640(SI), X12        // v4 = o[2]
	VMOVSD X12, 4256(R14)
	VMOVSD 2688(SI), X12       // v5 = o[10]
	VMOVSD X12, 4264(R14)
	VMOVSD 1664(SI), X12       // v6 = o[6]
	VMOVSD X12, 4272(R14)
	VMOVSD 3712(SI), X12       // v7 = o[14]
	VMOVSD X12, 4280(R14)
	VMOVSD 384(SI), X12        // v8 = o[1]
	VMOVSD X12, 4288(R14)
	VMOVSD 2432(SI), X12       // v9 = o[9]
	VMOVSD X12, 4296(R14)
	VMOVSD 1408(SI), X12       // v10 = o[5]
	VMOVSD X12, 4304(R14)
	VMOVSD 3456(SI), X12       // v11 = o[13]
	VMOVSD X12, 4312(R14)
	VMOVSD 896(SI), X12        // v12 = o[3]
	VMOVSD X12, 4320(R14)
	VMOVSD 2944(SI), X12       // v13 = o[11]
	VMOVSD X12, 4328(R14)
	VMOVSD 1920(SI), X12       // v14 = o[7]
	VMOVSD X12, 4336(R14)
	VMOVSD 3968(SI), X12       // v15 = o[15]
	VMOVSD X12, 4344(R14)

	// ===== FFT-16 on even elements =====
	// X0-X11 already contain v0-v11, v12-v15 at 4192, 4200, 4208, 4216

	// ----- FFT-16 Stage 1: 8 radix-2 butterflies -----
	// (v0,v1), (v2,v3), (v4,v5), (v6,v7), (v8,v9), (v10,v11), (v12,v13), (v14,v15)

	VADDPS X1, X0, X12         // t = v0 + v1
	VSUBPS X1, X0, X1          // v1 = v0 - v1
	VMOVAPS X12, X0            // v0 = t

	VADDPS X3, X2, X12
	VSUBPS X3, X2, X3
	VMOVAPS X12, X2

	VADDPS X5, X4, X12
	VSUBPS X5, X4, X5
	VMOVAPS X12, X4

	VADDPS X7, X6, X12
	VSUBPS X7, X6, X7
	VMOVAPS X12, X6

	VADDPS X9, X8, X12
	VSUBPS X9, X8, X9
	VMOVAPS X12, X8

	VADDPS X11, X10, X12
	VSUBPS X11, X10, X11
	VMOVAPS X12, X10

	// v12, v13 butterfly (from memory)
	VMOVSD 4192(R14), X12      // v12
	VMOVSD 4200(R14), X13      // v13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4192(R14)      // new v12
	VMOVSD X13, 4200(R14)      // new v13

	// v14, v15 butterfly (from memory)
	VMOVSD 4208(R14), X12      // v14
	VMOVSD 4216(R14), X13      // v15
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4208(R14)      // new v14
	VMOVSD X13, 4216(R14)      // new v15

	// ----- FFT-16 Stage 2: radix-4 with -i rotation -----
	// Apply -i to v3, v7, v11, v15 (swap re/im, negate new imag)
	// -i * (a+bi) = (b - ai) = (b, -a)

	// Load sign mask for -i rotation
	VMOVUPS const_signmask_imag<>(SB), X15

	// v0, v2 butterfly (no twiddle)
	VADDPS X2, X0, X12
	VSUBPS X2, X0, X2
	VMOVAPS X12, X0

	// v1, v3 with -i*v3
	VSHUFPS $0xB1, X3, X3, X12 // swap re/im: (im, re)
	VXORPS X15, X12, X12       // negate im: (im, -re) = -i * v3
	VADDPS X12, X1, X13        // v1 + (-i)*v3
	VSUBPS X12, X1, X3         // v1 - (-i)*v3 -> new v3
	VMOVAPS X13, X1            // new v1

	// v4, v6 butterfly
	VADDPS X6, X4, X12
	VSUBPS X6, X4, X6
	VMOVAPS X12, X4

	// v5, v7 with -i*v7
	VSHUFPS $0xB1, X7, X7, X12
	VXORPS X15, X12, X12
	VADDPS X12, X5, X13
	VSUBPS X12, X5, X7
	VMOVAPS X13, X5

	// v8, v10 butterfly
	VADDPS X10, X8, X12
	VSUBPS X10, X8, X10
	VMOVAPS X12, X8

	// v9, v11 with -i*v11
	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X15, X12, X12
	VADDPS X12, X9, X13
	VSUBPS X12, X9, X11
	VMOVAPS X13, X9

	// v12, v14 butterfly (from memory)
	VMOVSD 4192(R14), X12      // v12
	VMOVSD 4208(R14), X13      // v14
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4192(R14)      // new v12
	VMOVSD X13, 4208(R14)      // new v14

	// v13, v15 with -i*v15 (from memory)
	VMOVSD 4200(R14), X12      // v13
	VMOVSD 4216(R14), X13      // v15
	VSHUFPS $0xB1, X13, X13, X14
	VXORPS X15, X14, X14       // -i * v15
	VADDPS X14, X12, X13       // v13 + (-i)*v15
	VSUBPS X14, X12, X12       // v13 - (-i)*v15 -> new v15
	VMOVSD X13, 4200(R14)      // new v13
	VMOVSD X12, 4216(R14)      // new v15

	// ----- FFT-16 Stage 3: W_8 twiddles -----
	// W_8^0 = 1, W_8^1 = isq2*(1-i), W_8^2 = -i, W_8^3 = isq2*(-1-i)
	// Pairs: (v0,v4), (v1,v5), (v2,v6), (v3,v7)
	//        (v8,v12), (v9,v13), (v10,v14), (v11,v15)

	// Load isq2 constant
	VBROADCASTSS const_isq2<>(SB), X15

	// v0, v4: W_8^0 = 1 (no twiddle)
	VADDPS X4, X0, X12
	VSUBPS X4, X0, X4
	VMOVAPS X12, X0

	// v1, v5: W_8^1 = isq2*(1-i)
	// (a+bi) * isq2*(1-i) = isq2*((a+b) + (b-a)i)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VSHUFPS $0xB1, X5, X5, X12 // X12 = (im, re)
	VADDPS X5, X12, X13        // X13 = (re+im, im+re)
	VSUBPS X5, X12, X12        // X12 = (im-re, re-im)
	// We need (re+im, im-re). Use VUNPCKLPS to interleave low floats.
	// X13[0] = re+im, X12[0] = im-re
	// VUNPCKLPS X12, X13, result -> (X13[0], X12[0], X13[1], X12[1]) = (re+im, im-re, ...)
	VUNPCKLPS X12, X13, X12    // X12 = (re+im, im-re, im+re, re-im)
	VMULPS X15, X12, X12       // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X5
	VMOVAPS X13, X1

	// v2, v6: W_8^2 = -i
	VSHUFPS $0xB1, X6, X6, X12
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X6
	VMOVAPS X13, X2

	// v3, v7: W_8^3 = isq2*(-1-i)
	// (a+bi) * isq2*(-1-i) = isq2*((b-a) + (-(a+b))i)
	// t3.re = isq2*(im - re), t3.im = -isq2*(re + im) = -isq2*(im + re)
	VSHUFPS $0xB1, X7, X7, X12 // X12 = (im, re)
	VSUBPS X7, X12, X13        // X13 = (im-re, re-im), X13[0] = im-re ✓
	VADDPS X7, X12, X12        // X12 = (im+re, re+im), X12[0] = im+re
	// We need (im-re, -(im+re)). Use VUNPCKLPS then negate the im part.
	VUNPCKLPS X12, X13, X12    // X12 = (im-re, im+re, ...)
	VMULPS X15, X12, X12       // X12 = isq2*(im-re, im+re, ...)
	// Negate just the imaginary component (position 1)
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12       // X12 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X12, X3, X13
	VSUBPS X12, X3, X7
	VMOVAPS X13, X3

	// v8, v12: W_8^0 = 1 (from memory for v12)
	VMOVSD 4192(R14), X12      // v12
	VADDPS X12, X8, X13
	VSUBPS X12, X8, X12
	VMOVAPS X13, X8
	VMOVSD X12, 4192(R14)      // new v12

	// v9, v13: W_8^1 = isq2*(1-i) (v13 from memory)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VMOVSD 4200(R14), X12      // v13 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VADDPS X12, X13, X14       // X14 = (re+im, im+re)
	VSUBPS X12, X13, X13       // X13 = (im-re, re-im)
	VUNPCKLPS X13, X14, X13    // X13 = (re+im, im-re, ...)
	VMULPS X15, X13, X13       // X13 = isq2 * (re+im, im-re, ...)
	VADDPS X13, X9, X14
	VSUBPS X13, X9, X13
	VMOVAPS X14, X9
	VMOVSD X13, 4200(R14)      // new v13

	// v10, v14: W_8^2 = -i (v14 from memory)
	VMOVSD 4208(R14), X12      // v14
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X13, X13
	VADDPS X13, X10, X14
	VSUBPS X13, X10, X13
	VMOVAPS X14, X10
	VMOVSD X13, 4208(R14)      // new v14

	// v11, v15: W_8^3 = isq2*(-1-i) (v15 from memory)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VMOVSD 4216(R14), X12      // v15 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X11, X14
	VSUBPS X13, X11, X13
	VMOVAPS X14, X11
	VMOVSD X13, 4216(R14)      // new v15

	// ----- FFT-16 Stage 4: W_16 twiddles -----
	// Pairs: (v0,v8), (v1,v9), (v2,v10), (v3,v11), (v4,v12), (v5,v13), (v6,v14), (v7,v15)
	// Twiddles: W_16^0=1, W_16^1=(cos1,-sin1), W_16^2=isq2*(1-i), W_16^3=(sin1,-cos1)
	//           W_16^4=-i, W_16^5=(cos1*(-1)-sin1*i)=(-cos1,-sin1), etc.
	// Actually W_16^k = exp(-2πik/16) = cos(2πk/16) - i*sin(2πk/16)

	// Load constants
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v0, v8: W_16^0 = 1
	VADDPS X8, X0, X12
	VSUBPS X8, X0, X8
	VMOVAPS X12, X0

	// v1, v9: W_16^1 = (cos1, -sin1)
	// (a+bi) * (cos1 - sin1*i) = (a*cos1 + b*sin1) + (b*cos1 - a*sin1)i
	// t3.re = re*cos1 + im*sin1, t3.im = im*cos1 - re*sin1
	VSHUFPS $0xB1, X9, X9, X12 // (im, re)
	VMULPS X14, X9, X15        // (re*cos1, im*cos1)
	VMULPS X13, X12, X12       // (im*sin1, re*sin1)
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12    // (re*cos1+im*sin1, im*cos1-re*sin1)
	VADDPS X12, X1, X15
	VSUBPS X12, X1, X9
	VMOVAPS X15, X1

	// v2, v10: W_16^2 = isq2*(1-i) (same as W_8^1)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VBROADCASTSS const_isq2<>(SB), X15
	VSHUFPS $0xB1, X10, X10, X12 // X12 = (im, re)
	VADDPS X10, X12, X13         // X13 = (re+im, im+re)
	VSUBPS X10, X12, X12         // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12      // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12         // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X10
	VMOVAPS X13, X2

	// Reload sin1, cos1
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v3, v11: W_16^3 = (sin1, -cos1)
	// (a+bi) * (sin1 - cos1*i) = (a*sin1 + b*cos1) + (b*sin1 - a*cos1)i
	VSHUFPS $0xB1, X11, X11, X12
	VMULPS X13, X11, X15       // (re*sin1, im*sin1)
	VMULPS X14, X12, X12       // (im*cos1, re*cos1)
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12    // (re*sin1+im*cos1, im*sin1-re*cos1)
	VADDPS X12, X3, X15
	VSUBPS X12, X3, X11
	VMOVAPS X15, X3

	// v4, v12: W_16^4 = -i (v12 from memory)
	VMOVSD 4192(R14), X12      // v12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X15
	VXORPS X15, X13, X13
	VADDPS X13, X4, X15
	VSUBPS X13, X4, X12
	VMOVAPS X15, X4
	VMOVSD X12, 4192(R14)      // new v12

	// Reload constants
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v5, v13: W_16^5 = (-sin1, -cos1) = -(sin1, cos1)
	// Actually W_16^5 = cos(5π/8) - i*sin(5π/8) = -sin1 - i*cos1
	// (a+bi) * (-sin1 - cos1*i) = (-a*sin1 + b*cos1) + (-b*sin1 - a*cos1)i
	VMOVSD 4200(R14), X12      // v13
	VSHUFPS $0xB1, X12, X12, X15
	VMULPS X13, X12, X12       // (re*sin1, im*sin1)
	VMULPS X14, X15, X15       // (im*cos1, re*cos1)
	VSUBPS X12, X15, X15       // (im*cos1 - re*sin1, re*cos1 - im*sin1)
	VMOVUPS const_signmask_full<>(SB), X12
	// Actually let me recalculate: (-sin1 - i*cos1) * (a+bi)
	// = -a*sin1 - a*i*cos1 - i*b*sin1 - i²*b*cos1
	// = -a*sin1 + b*cos1 + i*(-a*cos1 - b*sin1)
	// t3.re = -re*sin1 + im*cos1 = im*cos1 - re*sin1
	// t3.im = -re*cos1 - im*sin1 = -(re*cos1 + im*sin1)
	VMOVSD 4200(R14), X12      // v13 again
	VSHUFPS $0xB1, X12, X12, X15 // (im, re)
	VMULPS X14, X15, X15       // (im*cos1, re*cos1)
	VMULPS X13, X12, X12       // (re*sin1, im*sin1)
	VSUBPS X12, X15, X15       // (im*cos1-re*sin1, re*cos1-im*sin1)
	// Now need to negate the imag part: result.im = -(re*cos1 + im*sin1)
	// But we have (re*cos1 - im*sin1), need -(re*cos1 + im*sin1)
	// Let me redo: we have (im*cos1-re*sin1, re*cos1-im*sin1)
	// We want (im*cos1-re*sin1, -(re*cos1+im*sin1))
	// Hmm, this isn't matching. Let me recalculate W_16^5.
	// W_16^5 = exp(-2πi*5/16) = cos(5π/8) - i*sin(5π/8)
	// cos(5π/8) = cos(π - 3π/8) = -cos(3π/8) = -sin(π/8) = -sin1
	// sin(5π/8) = sin(π - 3π/8) = sin(3π/8) = cos(π/8) = cos1
	// So W_16^5 = -sin1 - i*cos1
	// (a+bi)*(-sin1 - i*cos1) = -a*sin1 - ai*cos1 - bi*sin1 + b*cos1
	//                        = (b*cos1 - a*sin1) + i*(-a*cos1 - b*sin1)
	// t3.re = im*cos1 - re*sin1
	// t3.im = -(re*cos1 + im*sin1)
	VMOVSD 4200(R14), X12      // v13
	VMULPS X14, X12, X15       // (re*cos1, im*cos1)
	VSHUFPS $0xB1, X12, X12, X12 // (im, re)
	VMULPS X13, X12, X12       // (im*sin1, re*sin1)
	VSHUFPS $0xB1, X12, X12, X12 // (re*sin1, im*sin1)
	// Now X15 = (re*cos1, im*cos1), X12 = (re*sin1, im*sin1)
	// We want: re part = im*cos1 - re*sin1, im part = -(re*cos1 + im*sin1)
	// X15[1] - X12[0] for re, -(X15[0] + X12[1]) for im
	// This is getting complicated. Let me use a cleaner approach.
	VMOVSD 4200(R14), X12      // v13 = (re, im)
	VSHUFPS $0xB1, X12, X12, X15 // (im, re)
	// re_new = im*cos1 - re*sin1
	// im_new = -re*cos1 - im*sin1
	// Construct: (im, -re) then multiply by (cos1, cos1) -> (im*cos1, -re*cos1)
	// and (re, im) multiply by (sin1, sin1) -> (re*sin1, im*sin1)
	// then subtract: (im*cos1 - re*sin1, -re*cos1 - im*sin1)
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15       // (im, -re)
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X15, X15       // (im*cos1, -re*cos1)
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X12, X12       // (re*sin1, im*sin1)
	VSUBPS X12, X15, X12       // (im*cos1 - re*sin1, -re*cos1 - im*sin1)
	VADDPS X12, X5, X15
	VSUBPS X12, X5, X13
	VMOVAPS X15, X5
	VMOVSD X13, 4200(R14)      // new v13

	// v6, v14: W_16^6 = isq2*(-1-i) (same as W_8^3)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VBROADCASTSS const_isq2<>(SB), X15
	VMOVSD 4208(R14), X12      // v14 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X6, X14
	VSUBPS X13, X6, X13
	VMOVAPS X14, X6
	VMOVSD X13, 4208(R14)      // new v14

	// v7, v15: W_16^7 = (-cos1, -sin1)
	// W_16^7 = cos(7π/8) - i*sin(7π/8)
	// cos(7π/8) = -cos(π/8) = -cos1
	// sin(7π/8) = sin(π/8) = sin1
	// So W_16^7 = -cos1 - i*sin1
	// (a+bi)*(-cos1 - i*sin1) = -a*cos1 - ai*sin1 - bi*cos1 + b*sin1
	//                        = (b*sin1 - a*cos1) + i*(-a*sin1 - b*cos1)
	// t3.re = im*sin1 - re*cos1
	// t3.im = -(re*sin1 + im*cos1)
	VMOVSD 4216(R14), X12      // v15 = (re, im)
	VSHUFPS $0xB1, X12, X12, X15 // (im, re)
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15       // (im, -re)
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X15, X15       // (im*sin1, -re*sin1)
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X12, X12       // (re*cos1, im*cos1)
	VSUBPS X12, X15, X12       // (im*sin1 - re*cos1, -re*sin1 - im*cos1)
	VADDPS X12, X7, X15
	VSUBPS X12, X7, X13
	VMOVAPS X15, X7
	VMOVSD X13, 4216(R14)      // new v15

	// Store FFT-16 results for even elements to 4352(R14)
	VMOVSD X0, 4352(R14)       // E[0]
	VMOVSD X1, 4360(R14)       // E[1]
	VMOVSD X2, 4368(R14)       // E[2]
	VMOVSD X3, 4376(R14)       // E[3]
	VMOVSD X4, 4384(R14)       // E[4]
	VMOVSD X5, 4392(R14)       // E[5]
	VMOVSD X6, 4400(R14)       // E[6]
	VMOVSD X7, 4408(R14)       // E[7]
	VMOVSD X8, 4416(R14)       // E[8]
	VMOVSD X9, 4424(R14)       // E[9]
	VMOVSD X10, 4432(R14)      // E[10]
	VMOVSD X11, 4440(R14)      // E[11]
	VMOVSD 4192(R14), X0       // v12 -> E[12]
	VMOVSD X0, 4448(R14)
	VMOVSD 4200(R14), X0       // v13 -> E[13]
	VMOVSD X0, 4456(R14)
	VMOVSD 4208(R14), X0       // v14 -> E[14]
	VMOVSD X0, 4464(R14)
	VMOVSD 4216(R14), X0       // v15 -> E[15]
	VMOVSD X0, 4472(R14)

	// ===== FFT-16 on odd elements =====
	// Odd elements already loaded to stack at 4224(R14) in bit-reversed order
	// Load v0-v11 into X0-X11, v12-v15 stay at 4320, 4328, 4336, 4344
	VMOVSD 4224(R14), X0       // v0
	VMOVSD 4232(R14), X1       // v1
	VMOVSD 4240(R14), X2       // v2
	VMOVSD 4248(R14), X3       // v3
	VMOVSD 4256(R14), X4       // v4
	VMOVSD 4264(R14), X5       // v5
	VMOVSD 4272(R14), X6       // v6
	VMOVSD 4280(R14), X7       // v7
	VMOVSD 4288(R14), X8       // v8
	VMOVSD 4296(R14), X9       // v9
	VMOVSD 4304(R14), X10      // v10
	VMOVSD 4312(R14), X11      // v11
	// v12-v15 stay at 4320, 4328, 4336, 4344

	// ----- FFT-16 Stage 1: 8 radix-2 butterflies -----
	VADDPS X1, X0, X12
	VSUBPS X1, X0, X1
	VMOVAPS X12, X0

	VADDPS X3, X2, X12
	VSUBPS X3, X2, X3
	VMOVAPS X12, X2

	VADDPS X5, X4, X12
	VSUBPS X5, X4, X5
	VMOVAPS X12, X4

	VADDPS X7, X6, X12
	VSUBPS X7, X6, X7
	VMOVAPS X12, X6

	VADDPS X9, X8, X12
	VSUBPS X9, X8, X9
	VMOVAPS X12, X8

	VADDPS X11, X10, X12
	VSUBPS X11, X10, X11
	VMOVAPS X12, X10

	VMOVSD 4320(R14), X12
	VMOVSD 4328(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4320(R14)
	VMOVSD X13, 4328(R14)

	VMOVSD 4336(R14), X12
	VMOVSD 4344(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4336(R14)
	VMOVSD X13, 4344(R14)

	// ----- FFT-16 Stage 2: radix-4 with -i rotation -----
	VMOVUPS const_signmask_imag<>(SB), X15

	VADDPS X2, X0, X12
	VSUBPS X2, X0, X2
	VMOVAPS X12, X0

	VSHUFPS $0xB1, X3, X3, X12
	VXORPS X15, X12, X12
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X3
	VMOVAPS X13, X1

	VADDPS X6, X4, X12
	VSUBPS X6, X4, X6
	VMOVAPS X12, X4

	VSHUFPS $0xB1, X7, X7, X12
	VXORPS X15, X12, X12
	VADDPS X12, X5, X13
	VSUBPS X12, X5, X7
	VMOVAPS X13, X5

	VADDPS X10, X8, X12
	VSUBPS X10, X8, X10
	VMOVAPS X12, X8

	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X15, X12, X12
	VADDPS X12, X9, X13
	VSUBPS X12, X9, X11
	VMOVAPS X13, X9

	VMOVSD 4320(R14), X12
	VMOVSD 4336(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4320(R14)
	VMOVSD X13, 4336(R14)

	VMOVSD 4328(R14), X12
	VMOVSD 4344(R14), X13
	VSHUFPS $0xB1, X13, X13, X14
	VXORPS X15, X14, X14
	VADDPS X14, X12, X13
	VSUBPS X14, X12, X12
	VMOVSD X13, 4328(R14)
	VMOVSD X12, 4344(R14)

	// ----- FFT-16 Stage 3: W_8 twiddles -----
	VBROADCASTSS const_isq2<>(SB), X15

	VADDPS X4, X0, X12
	VSUBPS X4, X0, X4
	VMOVAPS X12, X0

	// v1, v5: W_8^1 = isq2*(1-i)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VSHUFPS $0xB1, X5, X5, X12 // X12 = (im, re)
	VADDPS X5, X12, X13        // X13 = (re+im, im+re)
	VSUBPS X5, X12, X12        // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12    // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12       // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X5
	VMOVAPS X13, X1

	// v2, v6: W_8^2 = -i
	VSHUFPS $0xB1, X6, X6, X12
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X6
	VMOVAPS X13, X2

	// v3, v7: W_8^3 = isq2*(-1-i)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VSHUFPS $0xB1, X7, X7, X12 // X12 = (im, re)
	VSUBPS X7, X12, X13        // X13 = (im-re, re-im), X13[0] = im-re ✓
	VADDPS X7, X12, X12        // X12 = (im+re, re+im), X12[0] = im+re
	VUNPCKLPS X12, X13, X12    // X12 = (im-re, im+re, ...)
	VMULPS X15, X12, X12       // X12 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12       // X12 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X12, X3, X13
	VSUBPS X12, X3, X7
	VMOVAPS X13, X3

	// v8, v12: W_8^0
	VMOVSD 4320(R14), X12
	VADDPS X12, X8, X13
	VSUBPS X12, X8, X12
	VMOVAPS X13, X8
	VMOVSD X12, 4320(R14)

	// v9, v13: W_8^1 = isq2*(1-i)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VMOVSD 4328(R14), X12      // v13 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VADDPS X12, X13, X14       // X14 = (re+im, im+re)
	VSUBPS X12, X13, X13       // X13 = (im-re, re-im)
	VUNPCKLPS X13, X14, X13    // X13 = (re+im, im-re, ...)
	VMULPS X15, X13, X13       // X13 = isq2 * (re+im, im-re, ...)
	VADDPS X13, X9, X14
	VSUBPS X13, X9, X13
	VMOVAPS X14, X9
	VMOVSD X13, 4328(R14)

	// v10, v14: W_8^2
	VMOVSD 4336(R14), X12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X13, X13
	VADDPS X13, X10, X14
	VSUBPS X13, X10, X13
	VMOVAPS X14, X10
	VMOVSD X13, 4336(R14)

	// v11, v15: W_8^3 = isq2*(-1-i)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VMOVSD 4344(R14), X12      // v15 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X11, X14
	VSUBPS X13, X11, X13
	VMOVAPS X14, X11
	VMOVSD X13, 4344(R14)

	// ----- FFT-16 Stage 4: W_16 twiddles -----
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v0, v8: W_16^0 = 1
	VADDPS X8, X0, X12
	VSUBPS X8, X0, X8
	VMOVAPS X12, X0

	// v1, v9: W_16^1
	VSHUFPS $0xB1, X9, X9, X12
	VMULPS X14, X9, X15
	VMULPS X13, X12, X12
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12
	VADDPS X12, X1, X15
	VSUBPS X12, X1, X9
	VMOVAPS X15, X1

	// v2, v10: W_16^2 = isq2*(1-i)
	// t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VBROADCASTSS const_isq2<>(SB), X15
	VSHUFPS $0xB1, X10, X10, X12 // X12 = (im, re)
	VADDPS X10, X12, X13         // X13 = (re+im, im+re)
	VSUBPS X10, X12, X12         // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12      // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12         // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X10
	VMOVAPS X13, X2

	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v3, v11: W_16^3
	VSHUFPS $0xB1, X11, X11, X12
	VMULPS X13, X11, X15
	VMULPS X14, X12, X12
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12
	VADDPS X12, X3, X15
	VSUBPS X12, X3, X11
	VMOVAPS X15, X3

	// v4, v12: W_16^4 = -i
	VMOVSD 4320(R14), X12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X15
	VXORPS X15, X13, X13
	VADDPS X13, X4, X15
	VSUBPS X13, X4, X12
	VMOVAPS X15, X4
	VMOVSD X12, 4320(R14)

	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// v5, v13: W_16^5
	VMOVSD 4328(R14), X12
	VSHUFPS $0xB1, X12, X12, X15
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X15, X15
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X12, X12
	VSUBPS X12, X15, X12
	VADDPS X12, X5, X15
	VSUBPS X12, X5, X13
	VMOVAPS X15, X5
	VMOVSD X13, 4328(R14)

	// v6, v14: W_16^6 = isq2*(-1-i)
	// t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VBROADCASTSS const_isq2<>(SB), X15
	VMOVSD 4336(R14), X12      // v14 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X6, X14
	VSUBPS X13, X6, X13
	VMOVAPS X14, X6
	VMOVSD X13, 4336(R14)

	// v7, v15: W_16^7
	VMOVSD 4344(R14), X12
	VSHUFPS $0xB1, X12, X12, X15
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X15, X15
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X12, X12
	VSUBPS X12, X15, X12
	VADDPS X12, X7, X15
	VSUBPS X12, X7, X13
	VMOVAPS X15, X7
	VMOVSD X13, 4344(R14)

	// Store FFT-16 results for odd elements to 4480(R14)
	VMOVSD X0, 4480(R14)       // O[0]
	VMOVSD X1, 4488(R14)       // O[1]
	VMOVSD X2, 4496(R14)       // O[2]
	VMOVSD X3, 4504(R14)       // O[3]
	VMOVSD X4, 4512(R14)       // O[4]
	VMOVSD X5, 4520(R14)       // O[5]
	VMOVSD X6, 4528(R14)       // O[6]
	VMOVSD X7, 4536(R14)       // O[7]
	VMOVSD X8, 4544(R14)       // O[8]
	VMOVSD X9, 4552(R14)       // O[9]
	VMOVSD X10, 4560(R14)      // O[10]
	VMOVSD X11, 4568(R14)      // O[11]
	VMOVSD 4320(R14), X0
	VMOVSD X0, 4576(R14)       // O[12]
	VMOVSD 4328(R14), X0
	VMOVSD X0, 4584(R14)       // O[13]
	VMOVSD 4336(R14), X0
	VMOVSD X0, 4592(R14)       // O[14]
	VMOVSD 4344(R14), X0
	VMOVSD X0, 4600(R14)       // O[15]

	// ===== Combine E and O with W_32 and W_512 twiddles =====
	// For k2 = 0..15:
	//   t = O[k2] * tw[k2*16]  (W_32^k2 = tw[k2*16] from 512-point table)
	//   out[k2*16 + n1] = (E[k2] + t) * tw[k2*n1]
	//   out[(k2+16)*16 + n1] = (E[k2] - t) * tw[(k2+16)*n1]

	// Calculate output offset for this column
	MOVQ R12, DI               // DI = n1
	SHLQ $3, DI                // DI = n1 * 8 (byte offset)

	// k2 = 0: W_32^0 = tw[0] = 1, W_512^0 = 1, W_512^(16*n1) = tw[16*n1]
	VMOVSD 4352(R14), X0       // E[0]
	VMOVSD 4480(R14), X1       // O[0]
	VADDPS X1, X0, X2          // E[0] + O[0]
	VSUBPS X1, X0, X3          // E[0] - O[0]
	// tw[0] = 1, no multiply needed for first result
	VMOVSD X2, (R11)(DI*1)     // out[0*16 + n1]
	// tw[16*n1]: need to load from twiddle table
	MOVQ R12, AX
	SHLQ $4, AX                // AX = n1 * 16
	SHLQ $3, AX                // AX = n1 * 16 * 8 = n1 * 128
	VMOVSD (R10)(AX*1), X4     // tw[16*n1]
	// Complex multiply: X3 * X4
	VMOVSLDUP X4, X5           // (tw.re, tw.re)
	VMOVSHDUP X4, X6           // (tw.im, tw.im)
	VMULPS X5, X3, X5          // (a.re*tw.re, a.im*tw.re)
	VSHUFPS $0xB1, X3, X3, X3  // (a.im, a.re)
	VMULPS X6, X3, X3          // (a.im*tw.im, a.re*tw.im)
	VADDSUBPS X3, X5, X3       // (a.re*tw.re - a.im*tw.im, a.im*tw.re + a.re*tw.im)
	LEAQ 2048(R11), AX         // out + 16*16*8 = out + 2048
	VMOVSD X3, (AX)(DI*1)      // out[16*16 + n1]

	// k2 = 1..15: Full twiddle multiplications
	// This is getting very long. Let me create a loop.

	MOVQ $1, R13               // k2 = 1
	MOVQ R12, R15              // tw_index = n1 (k2*n1 for k2=1)
	MOVQ R12, BX
	SHLQ $4, BX                // 16*n1
	ADDQ R12, BX               // tw_index2 = (k2+16)*n1 for k2=1

inv_s1_combine_loop:
	CMPQ R13, $16
	JGE  inv_s1_col_next

	// Load E[k2] and O[k2]
	MOVQ R13, AX
	SHLQ $3, AX                // AX = k2 * 8
	LEAQ 4352(R14), SI         // E array
	VMOVSD (SI)(AX*1), X0      // E[k2]
	LEAQ 4480(R14), SI         // O array
	VMOVSD (SI)(AX*1), X1      // O[k2]

	// Load W_32^k2 = tw[k2*16]
	MOVQ R13, AX
	SHLQ $4, AX                // AX = k2 * 16
	SHLQ $3, AX                // AX = k2 * 16 * 8 = k2 * 128
	VMOVSD (R10)(AX*1), X2     // tw[k2*16]

	// Complex multiply: O[k2] * tw[k2*16]
	VMOVSLDUP X2, X3           // (tw.re, tw.re)
	VMOVSHDUP X2, X4           // (tw.im, tw.im)
	VMULPS X3, X1, X3          // (o.re*tw.re, o.im*tw.re)
	VSHUFPS $0xB1, X1, X1, X5  // (o.im, o.re)
	VMULPS X4, X5, X5          // (o.im*tw.im, o.re*tw.im)
	VADDSUBPS X5, X3, X1       // t = O[k2] * tw[k2*16]

	// E[k2] + t, E[k2] - t
	VADDPS X1, X0, X2          // sum = E[k2] + t
	VSUBPS X1, X0, X3          // diff = E[k2] - t

	// Apply W_512^(k2*n1) to sum: tw[k2*n1]
	MOVQ R15, AX               // tw_index = k2*n1
	SHLQ $3, AX                // (k2*n1) * 8
	VMOVSD (R10)(AX*1), X4     // tw[k2*n1]
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VMULPS X5, X2, X5
	VSHUFPS $0xB1, X2, X2, X2
	VMULPS X6, X2, X2
	VADDSUBPS X2, X5, X2       // sum * tw[k2*n1]

	// Store to out[k2*16 + n1]
	MOVQ R13, AX               // k2
	SHLQ $4, AX                // k2 * 16
	ADDQ R12, AX               // k2*16 + n1
	SHLQ $3, AX                // byte offset
	VMOVSD X2, (R11)(AX*1)

	// Apply W_512^((k2+16)*n1) to diff: tw[(k2+16)*n1]
	MOVQ BX, AX                // tw_index2 = (k2+16)*n1
	SHLQ $3, AX                // byte offset
	VMOVSD (R10)(AX*1), X4     // tw[(k2+16)*n1]
	VMOVSLDUP X4, X5
	VMOVSHDUP X4, X6
	VMULPS X5, X3, X5
	VSHUFPS $0xB1, X3, X3, X3
	VMULPS X6, X3, X3
	VADDSUBPS X3, X5, X3       // diff * tw[(k2+16)*n1]

	// Store to out[(k2+16)*16 + n1]
	MOVQ R13, AX               // k2
	ADDQ $16, AX               // k2 + 16
	SHLQ $4, AX                // (k2+16) * 16
	ADDQ R12, AX               // (k2+16)*16 + n1
	SHLQ $3, AX                // byte offset
	VMOVSD X3, (R11)(AX*1)

	ADDQ R12, R15              // tw_index += n1
	ADDQ R12, BX               // tw_index2 += n1
	INCQ R13
	JMP  inv_s1_combine_loop

inv_s1_col_next:
	INCQ R12
	JMP  inv_stage1_col_loop

inv_stage2_start:
	// =======================================================================
	// STAGE 2: 32 row FFT-16s
	// =======================================================================
	// For each row k2 = 0..31:
	//   - Load 16 elements from out[k2*16 + 0..15] in bit-reversed order
	//   - Perform FFT-16
	//   - Store to dst[32*k1 + k2] for k1 = 0..15

	XORQ R12, R12              // R12 = k2 (row 0..31)

inv_stage2_row_loop:
	CMPQ R12, $32
	JGE  inv_success

	// Calculate row base offset: k2 * 16 * 8 = k2 * 128
	MOVQ R12, DI
	SHLQ $7, DI                // DI = k2 * 128

	// Load 16 elements in bit-reversed order for FFT-16
	// Row base at out[k2*16], each element is 8 bytes
	// Bit-reversal: v0=out[0], v1=out[8], v2=out[4], v3=out[12], ...
	// Offsets: 0*8=0, 8*8=64, 4*8=32, 12*8=96, 2*8=16, 10*8=80, 6*8=48, 14*8=112,
	//          1*8=8, 9*8=72, 5*8=40, 13*8=104, 3*8=24, 11*8=88, 7*8=56, 15*8=120
	LEAQ (R11)(DI*1), SI       // SI = &out[k2*16]

	// Load v0-v11 in bit-reversed order, v12-v15 to stack
	VMOVSD 0(SI), X0           // v0 = out[0]
	VMOVSD 64(SI), X1          // v1 = out[8]
	VMOVSD 32(SI), X2          // v2 = out[4]
	VMOVSD 96(SI), X3          // v3 = out[12]
	VMOVSD 16(SI), X4          // v4 = out[2]
	VMOVSD 80(SI), X5          // v5 = out[10]
	VMOVSD 48(SI), X6          // v6 = out[6]
	VMOVSD 112(SI), X7         // v7 = out[14]
	VMOVSD 8(SI), X8           // v8 = out[1]
	VMOVSD 72(SI), X9          // v9 = out[9]
	VMOVSD 40(SI), X10         // v10 = out[5]
	VMOVSD 104(SI), X11        // v11 = out[13]
	// v12-v15 to stack
	VMOVSD 24(SI), X12         // v12 = out[3]
	VMOVSD X12, 4096(R14)
	VMOVSD 88(SI), X12         // v13 = out[11]
	VMOVSD X12, 4104(R14)
	VMOVSD 56(SI), X12         // v14 = out[7]
	VMOVSD X12, 4112(R14)
	VMOVSD 120(SI), X12        // v15 = out[15]
	VMOVSD X12, 4120(R14)

	// ----- FFT-16 Stage 1 -----
	VADDPS X1, X0, X12
	VSUBPS X1, X0, X1
	VMOVAPS X12, X0

	VADDPS X3, X2, X12
	VSUBPS X3, X2, X3
	VMOVAPS X12, X2

	VADDPS X5, X4, X12
	VSUBPS X5, X4, X5
	VMOVAPS X12, X4

	VADDPS X7, X6, X12
	VSUBPS X7, X6, X7
	VMOVAPS X12, X6

	VADDPS X9, X8, X12
	VSUBPS X9, X8, X9
	VMOVAPS X12, X8

	VADDPS X11, X10, X12
	VSUBPS X11, X10, X11
	VMOVAPS X12, X10

	VMOVSD 4096(R14), X12
	VMOVSD 4104(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4096(R14)
	VMOVSD X13, 4104(R14)

	VMOVSD 4112(R14), X12
	VMOVSD 4120(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4112(R14)
	VMOVSD X13, 4120(R14)

	// ----- FFT-16 Stage 2 -----
	VMOVUPS const_signmask_imag<>(SB), X15

	VADDPS X2, X0, X12
	VSUBPS X2, X0, X2
	VMOVAPS X12, X0

	VSHUFPS $0xB1, X3, X3, X12
	VXORPS X15, X12, X12
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X3
	VMOVAPS X13, X1

	VADDPS X6, X4, X12
	VSUBPS X6, X4, X6
	VMOVAPS X12, X4

	VSHUFPS $0xB1, X7, X7, X12
	VXORPS X15, X12, X12
	VADDPS X12, X5, X13
	VSUBPS X12, X5, X7
	VMOVAPS X13, X5

	VADDPS X10, X8, X12
	VSUBPS X10, X8, X10
	VMOVAPS X12, X8

	VSHUFPS $0xB1, X11, X11, X12
	VXORPS X15, X12, X12
	VADDPS X12, X9, X13
	VSUBPS X12, X9, X11
	VMOVAPS X13, X9

	VMOVSD 4096(R14), X12
	VMOVSD 4112(R14), X13
	VADDPS X13, X12, X14
	VSUBPS X13, X12, X13
	VMOVSD X14, 4096(R14)
	VMOVSD X13, 4112(R14)

	VMOVSD 4104(R14), X12
	VMOVSD 4120(R14), X13
	VSHUFPS $0xB1, X13, X13, X14
	VXORPS X15, X14, X14
	VADDPS X14, X12, X13
	VSUBPS X14, X12, X12
	VMOVSD X13, 4104(R14)
	VMOVSD X12, 4120(R14)

	// ----- FFT-16 Stage 3 -----
	VBROADCASTSS const_isq2<>(SB), X15

	VADDPS X4, X0, X12
	VSUBPS X4, X0, X4
	VMOVAPS X12, X0

	// W_8^1 = isq2*(1-i): t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VSHUFPS $0xB1, X5, X5, X12 // X12 = (im, re)
	VADDPS X5, X12, X13        // X13 = (re+im, im+re)
	VSUBPS X5, X12, X12        // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12    // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12       // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X1, X13
	VSUBPS X12, X1, X5
	VMOVAPS X13, X1

	// W_8^2 = -i
	VSHUFPS $0xB1, X6, X6, X12
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X6
	VMOVAPS X13, X2

	// W_8^3 = isq2*(-1-i): t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VSHUFPS $0xB1, X7, X7, X12 // X12 = (im, re)
	VSUBPS X7, X12, X13        // X13 = (im-re, re-im), X13[0] = im-re ✓
	VADDPS X7, X12, X12        // X12 = (im+re, re+im), X12[0] = im+re
	VUNPCKLPS X12, X13, X12    // X12 = (im-re, im+re, ...)
	VMULPS X15, X12, X12       // X12 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X12, X12       // X12 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X12, X3, X13
	VSUBPS X12, X3, X7
	VMOVAPS X13, X3

	VMOVSD 4096(R14), X12
	VADDPS X12, X8, X13
	VSUBPS X12, X8, X12
	VMOVAPS X13, X8
	VMOVSD X12, 4096(R14)

	// W_8^1 = isq2*(1-i): t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VMOVSD 4104(R14), X12      // v13 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VADDPS X12, X13, X14       // X14 = (re+im, im+re)
	VSUBPS X12, X13, X13       // X13 = (im-re, re-im)
	VUNPCKLPS X13, X14, X13    // X13 = (re+im, im-re, ...)
	VMULPS X15, X13, X13       // X13 = isq2 * (re+im, im-re, ...)
	VADDPS X13, X9, X14
	VSUBPS X13, X9, X13
	VMOVAPS X14, X9
	VMOVSD X13, 4104(R14)

	// W_8^2 = -i
	VMOVSD 4112(R14), X12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X14
	VXORPS X14, X13, X13
	VADDPS X13, X10, X14
	VSUBPS X13, X10, X13
	VMOVAPS X14, X10
	VMOVSD X13, 4112(R14)

	// W_8^3 = isq2*(-1-i): t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VMOVSD 4120(R14), X12      // v15 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X11, X14
	VSUBPS X13, X11, X13
	VMOVAPS X14, X11
	VMOVSD X13, 4120(R14)

	// ----- FFT-16 Stage 4 -----
	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	VADDPS X8, X0, X12
	VSUBPS X8, X0, X8
	VMOVAPS X12, X0

	VSHUFPS $0xB1, X9, X9, X12
	VMULPS X14, X9, X15
	VMULPS X13, X12, X12
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12
	VADDPS X12, X1, X15
	VSUBPS X12, X1, X9
	VMOVAPS X15, X1

	// W_16^2 = isq2*(1-i): t3.re = isq2*(re + im), t3.im = isq2*(im - re)
	VBROADCASTSS const_isq2<>(SB), X15
	VSHUFPS $0xB1, X10, X10, X12 // X12 = (im, re)
	VADDPS X10, X12, X13         // X13 = (re+im, im+re)
	VSUBPS X10, X12, X12         // X12 = (im-re, re-im)
	VUNPCKLPS X12, X13, X12      // X12 = (re+im, im-re, ...)
	VMULPS X15, X12, X12         // X12 = isq2 * (re+im, im-re, ...)
	VADDPS X12, X2, X13
	VSUBPS X12, X2, X10
	VMOVAPS X13, X2

	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	// W_16^3 uses sin1/cos1
	VSHUFPS $0xB1, X11, X11, X12
	VMULPS X13, X11, X15
	VMULPS X14, X12, X12
	VXORPS const_signmask_full<>(SB), X12, X12
	VADDSUBPS X12, X15, X12
	VADDPS X12, X3, X15
	VSUBPS X12, X3, X11
	VMOVAPS X15, X3

	VMOVSD 4096(R14), X12
	VSHUFPS $0xB1, X12, X12, X13
	VMOVUPS const_signmask_imag<>(SB), X15
	VXORPS X15, X13, X13
	VADDPS X13, X4, X15
	VSUBPS X13, X4, X12
	VMOVAPS X15, X4
	VMOVSD X12, 4096(R14)

	VBROADCASTSS const_cos1<>(SB), X14
	VBROADCASTSS const_sin1<>(SB), X13

	VMOVSD 4104(R14), X12
	VSHUFPS $0xB1, X12, X12, X15
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X15, X15
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X12, X12
	VSUBPS X12, X15, X12
	VADDPS X12, X5, X15
	VSUBPS X12, X5, X13
	VMOVAPS X15, X5
	VMOVSD X13, 4104(R14)

	// W_16^6 = isq2*(-1-i): t3.re = isq2*(im - re), t3.im = -isq2*(im + re)
	VBROADCASTSS const_isq2<>(SB), X15
	VMOVSD 4112(R14), X12      // v14 = (re, im)
	VSHUFPS $0xB1, X12, X12, X13 // X13 = (im, re)
	VSUBPS X12, X13, X14       // X14 = (im-re, re-im), X14[0] = im-re ✓
	VADDPS X12, X13, X13       // X13 = (im+re, re+im), X13[0] = im+re
	VUNPCKLPS X13, X14, X13    // X13 = (im-re, im+re, ...)
	VMULPS X15, X13, X13       // X13 = isq2*(im-re, im+re, ...)
	VMOVUPS const_signmask_imag<>(SB), X12
	VXORPS X12, X13, X13       // X13 = (isq2*(im-re), -isq2*(im+re), ...)
	VADDPS X13, X6, X14
	VSUBPS X13, X6, X13
	VMOVAPS X14, X6
	VMOVSD X13, 4112(R14)

	VMOVSD 4120(R14), X12
	VSHUFPS $0xB1, X12, X12, X15
	VMOVUPS const_signmask_imag<>(SB), X13
	VXORPS X13, X15, X15
	VBROADCASTSS const_sin1<>(SB), X13
	VMULPS X13, X15, X15
	VBROADCASTSS const_cos1<>(SB), X14
	VMULPS X14, X12, X12
	VSUBPS X12, X15, X12
	VADDPS X12, X7, X15
	VSUBPS X12, X7, X13
	VMOVAPS X15, X7
	VMOVSD X13, 4120(R14)

	// Store to dst: dst[32*k1 + k2] for k1 = 0..15
	VMOVUPS const_signmask_imag<>(SB), X12
	VBROADCASTSS const_scale<>(SB), X13
	VXORPS X12, X0, X0
	VMULPS X13, X0, X0
	VXORPS X12, X1, X1
	VMULPS X13, X1, X1
	VXORPS X12, X2, X2
	VMULPS X13, X2, X2
	VXORPS X12, X3, X3
	VMULPS X13, X3, X3
	VXORPS X12, X4, X4
	VMULPS X13, X4, X4
	VXORPS X12, X5, X5
	VMULPS X13, X5, X5
	VXORPS X12, X6, X6
	VMULPS X13, X6, X6
	VXORPS X12, X7, X7
	VMULPS X13, X7, X7
	VXORPS X12, X8, X8
	VMULPS X13, X8, X8
	VXORPS X12, X9, X9
	VMULPS X13, X9, X9
	VXORPS X12, X10, X10
	VMULPS X13, X10, X10
	VXORPS X12, X11, X11
	VMULPS X13, X11, X11
	VMOVSD 4096(R14), X14
	VXORPS X12, X14, X14
	VMULPS X13, X14, X14
	VMOVSD X14, 4096(R14)
	VMOVSD 4104(R14), X14
	VXORPS X12, X14, X14
	VMULPS X13, X14, X14
	VMOVSD X14, 4104(R14)
	VMOVSD 4112(R14), X14
	VXORPS X12, X14, X14
	VMULPS X13, X14, X14
	VMOVSD X14, 4112(R14)
	VMOVSD 4120(R14), X14
	VXORPS X12, X14, X14
	VMULPS X13, X14, X14
	VMOVSD X14, 4120(R14)

	// k2 is in R12, stride = 32 * 8 = 256 bytes
	MOVQ R12, DI
	SHLQ $3, DI                // DI = k2 * 8

	VMOVSD X0, (R8)(DI*1)      // dst[32*0 + k2]
	ADDQ $256, DI
	VMOVSD X1, (R8)(DI*1)      // dst[32*1 + k2]
	ADDQ $256, DI
	VMOVSD X2, (R8)(DI*1)      // dst[32*2 + k2]
	ADDQ $256, DI
	VMOVSD X3, (R8)(DI*1)      // dst[32*3 + k2]
	ADDQ $256, DI
	VMOVSD X4, (R8)(DI*1)      // dst[32*4 + k2]
	ADDQ $256, DI
	VMOVSD X5, (R8)(DI*1)      // dst[32*5 + k2]
	ADDQ $256, DI
	VMOVSD X6, (R8)(DI*1)      // dst[32*6 + k2]
	ADDQ $256, DI
	VMOVSD X7, (R8)(DI*1)      // dst[32*7 + k2]
	ADDQ $256, DI
	VMOVSD X8, (R8)(DI*1)      // dst[32*8 + k2]
	ADDQ $256, DI
	VMOVSD X9, (R8)(DI*1)      // dst[32*9 + k2]
	ADDQ $256, DI
	VMOVSD X10, (R8)(DI*1)     // dst[32*10 + k2]
	ADDQ $256, DI
	VMOVSD X11, (R8)(DI*1)     // dst[32*11 + k2]
	ADDQ $256, DI
	VMOVSD 4096(R14), X0
	VMOVSD X0, (R8)(DI*1)      // dst[32*12 + k2]
	ADDQ $256, DI
	VMOVSD 4104(R14), X0
	VMOVSD X0, (R8)(DI*1)      // dst[32*13 + k2]
	ADDQ $256, DI
	VMOVSD 4112(R14), X0
	VMOVSD X0, (R8)(DI*1)      // dst[32*14 + k2]
	ADDQ $256, DI
	VMOVSD 4120(R14), X0
	VMOVSD X0, (R8)(DI*1)      // dst[32*15 + k2]

	INCQ R12
	JMP  inv_stage2_row_loop

inv_success:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_fail:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET
