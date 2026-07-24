//go:build amd64 && !purego

// ===========================================================================
// AVX2 Size-512 Radix-8 FFT Kernel for AMD64 (complex128)
// ===========================================================================
//
// Algorithm: Radix-8 Decimation-in-Time (DIT) FFT
// 512 = 8 * 8 * 8 (three radix-8 stages).
//
// Stage structure (each stage = 64 radix-8 butterflies):
//   Stage 1: bit-reversed (base-8 digit-reversal) load from src, no input
//            twiddles; write reordered into an on-stack stage buffer.
//   Stage 2: 8 groups x 8 butterflies, 7 input twiddles per butterfly.
//   Stage 3: 64 butterflies (span 64), 7 input twiddles per butterfly.
//
// Every radix-8 butterfly additionally multiplies its odd outputs by the
// three fixed W8 = W_512^{64,128,192} factors, loaded from the twiddle table.
//
// Performance idioms (mirroring avx2_f64_size1024_radix4.s):
//   * Byte-pointer addressing: one SHLQ $4 per butterfly, fixed byte
//     displacements for the eight lane accesses (no per-access shift).
//   * FMA complex multiply (VFMADDSUB231PD) with the single maskNegHi
//     [0, signbit] mask for the (-i) multiplies.
//   * complex128 = 16 bytes; one XMM holds one complex128 ([re, im]).
//
// The bit-reversal table (·bitrev512_r8) and the W8/conjugation idiom are
// shared with the complex64 kernel in avx2_f32_size512_radix8.s and the
// complex128 radix-8 butterfly in avx2_f64_size8_radix8.s.  The table stores
// element indices, so the same table serves both precisions (shift by 4).
//
// ===========================================================================

#include "textflag.h"

// Forward radix-8 butterfly.
// Inputs  : X0..X7 = x0..x7 (already twiddled).  X15 = maskNegHi [0, signbit].
// Outputs : y0=X5 y1=X8 y2=X1 y3=X6 y4=X7 y5=X9 y6=X11 y7=X4.
// Clobbers: X0..X14.  W8 factors loaded from 1024/2048/3072(R10).
#define FWDBFLY \
	VADDPD X4, X0, X8       \ // a0 = x0 + x4
	VSUBPD X4, X0, X9       \ // a1 = x0 - x4
	VADDPD X6, X2, X10      \ // a2 = x2 + x6
	VSUBPD X6, X2, X2       \ // a3 = x2 - x6
	VADDPD X5, X1, X11      \ // a4 = x1 + x5
	VSUBPD X5, X1, X4       \ // a5 = x1 - x5
	VADDPD X7, X3, X1       \ // a6 = x3 + x7
	VSUBPD X7, X3, X5       \ // a7 = x3 - x7
	VADDPD X10, X8, X0      \ // e0 = a0 + a2
	VSUBPD X10, X8, X6      \ // e2 = a0 - a2
	VPERMILPD $1, X2, X7    \ // swap(a3)
	VXORPD X15, X7, X7      \ // (-i)*a3
	VADDPD X7, X9, X10      \ // e1 = a1 + (-i)*a3
	VSUBPD X7, X9, X2       \ // e3 = a1 - (-i)*a3
	VADDPD X1, X11, X9      \ // o0 = a4 + a6
	VSUBPD X1, X11, X11     \ // o2 = a4 - a6
	VPERMILPD $1, X5, X7    \ // swap(a7)
	VXORPD X15, X7, X7      \ // (-i)*a7
	VADDPD X7, X4, X1       \ // o1 = a5 + (-i)*a7
	VSUBPD X7, X4, X4       \ // o3 = a5 - (-i)*a7
	VADDPD X9, X0, X5       \ // y0 = e0 + o0
	VSUBPD X9, X0, X7       \ // y4 = e0 - o0
	MOVUPD 1024(R10), X8    \ // w1_8
	VMOVDDUP X8, X3         \ // w1_8.re
	VPERMILPD $1, X8, X9    \ // [im, re]
	VMOVDDUP X9, X9         \ // w1_8.im
	VPERMILPD $1, X1, X0    \ // swap(o1)
	VMULPD X9, X0, X0       \ // o1.im * w1_8.im
	VFMADDSUB231PD X3, X1, X0 \ // t1 = w1_8 * o1
	VADDPD X0, X10, X8      \ // y1 = e1 + t1
	VSUBPD X0, X10, X9      \ // y5 = e1 - t1
	MOVUPD 2048(R10), X0    \ // w2_8
	VMOVDDUP X0, X3         \ // w2_8.re
	VPERMILPD $1, X0, X1    \ // [im, re]
	VMOVDDUP X1, X1         \ // w2_8.im
	VPERMILPD $1, X11, X10  \ // swap(o2)
	VMULPD X1, X10, X10     \ // o2.im * w2_8.im
	VFMADDSUB231PD X3, X11, X10 \ // t2 = w2_8 * o2
	VADDPD X10, X6, X1      \ // y2 = e2 + t2
	VSUBPD X10, X6, X11     \ // y6 = e2 - t2
	MOVUPD 3072(R10), X0    \ // w3_8
	VMOVDDUP X0, X3         \ // w3_8.re
	VPERMILPD $1, X0, X6    \ // [im, re]
	VMOVDDUP X6, X6         \ // w3_8.im
	VPERMILPD $1, X4, X10   \ // swap(o3)
	VMULPD X6, X10, X10     \ // o3.im * w3_8.im
	VFMADDSUB231PD X3, X4, X10 \ // t3 = w3_8 * o3
	VADDPD X10, X2, X6      \ // y3 = e3 + t3
	VSUBPD X10, X2, X4        // y7 = e3 - t3

// Inverse radix-8 butterfly: swap the (-i) add/sub directions and conjugate
// the W8 factors (VXORPD maskNegHi flips the imaginary lane).
#define INVBFLY \
	VADDPD X4, X0, X8       \ // a0 = x0 + x4
	VSUBPD X4, X0, X9       \ // a1 = x0 - x4
	VADDPD X6, X2, X10      \ // a2 = x2 + x6
	VSUBPD X6, X2, X2       \ // a3 = x2 - x6
	VADDPD X5, X1, X11      \ // a4 = x1 + x5
	VSUBPD X5, X1, X4       \ // a5 = x1 - x5
	VADDPD X7, X3, X1       \ // a6 = x3 + x7
	VSUBPD X7, X3, X5       \ // a7 = x3 - x7
	VADDPD X10, X8, X0      \ // e0 = a0 + a2
	VSUBPD X10, X8, X6      \ // e2 = a0 - a2
	VPERMILPD $1, X2, X7    \ // swap(a3)
	VXORPD X15, X7, X7      \ // (-i)*a3
	VSUBPD X7, X9, X10      \ // e1 = a1 - (-i)*a3
	VADDPD X7, X9, X2       \ // e3 = a1 + (-i)*a3
	VADDPD X1, X11, X9      \ // o0 = a4 + a6
	VSUBPD X1, X11, X11     \ // o2 = a4 - a6
	VPERMILPD $1, X5, X7    \ // swap(a7)
	VXORPD X15, X7, X7      \ // (-i)*a7
	VSUBPD X7, X4, X1       \ // o1 = a5 - (-i)*a7
	VADDPD X7, X4, X4       \ // o3 = a5 + (-i)*a7
	VADDPD X9, X0, X5       \ // y0 = e0 + o0
	VSUBPD X9, X0, X7       \ // y4 = e0 - o0
	MOVUPD 1024(R10), X8    \ // w1_8
	VXORPD X15, X8, X8      \ // conj(w1_8)
	VMOVDDUP X8, X3         \ // re
	VPERMILPD $1, X8, X9    \ // [im, re]
	VMOVDDUP X9, X9         \ // im
	VPERMILPD $1, X1, X0    \ // swap(o1)
	VMULPD X9, X0, X0       \ // o1.im * im
	VFMADDSUB231PD X3, X1, X0 \ // t1 = conj(w1_8) * o1
	VADDPD X0, X10, X8      \ // y1 = e1 + t1
	VSUBPD X0, X10, X9      \ // y5 = e1 - t1
	MOVUPD 2048(R10), X0    \ // w2_8
	VXORPD X15, X0, X0      \ // conj(w2_8)
	VMOVDDUP X0, X3         \ // re
	VPERMILPD $1, X0, X1    \ // [im, re]
	VMOVDDUP X1, X1         \ // im
	VPERMILPD $1, X11, X10  \ // swap(o2)
	VMULPD X1, X10, X10     \ // o2.im * im
	VFMADDSUB231PD X3, X11, X10 \ // t2 = conj(w2_8) * o2
	VADDPD X10, X6, X1      \ // y2 = e2 + t2
	VSUBPD X10, X6, X11     \ // y6 = e2 - t2
	MOVUPD 3072(R10), X0    \ // w3_8
	VXORPD X15, X0, X0      \ // conj(w3_8)
	VMOVDDUP X0, X3         \ // re
	VPERMILPD $1, X0, X6    \ // [im, re]
	VMOVDDUP X6, X6         \ // im
	VPERMILPD $1, X4, X10   \ // swap(o3)
	VMULPD X6, X10, X10     \ // o3.im * im
	VFMADDSUB231PD X3, X4, X10 \ // t3 = conj(w3_8) * o3
	VADDPD X10, X2, X6      \ // y3 = e3 + t3
	VSUBPD X10, X2, X4        // y7 = e3 - t3

// Forward input twiddle: reg = tw[R15] * reg.  Temps X8..X11.
#define FTWID(reg) \
	MOVUPD (R10)(R15*1), X8 \ // tw
	VMOVDDUP X8, X9         \ // tw.re
	VPERMILPD $1, X8, X10   \ // [im, re]
	VMOVDDUP X10, X10       \ // tw.im
	VPERMILPD $1, reg, X11  \ // swap(x)
	VMULPD X10, X11, X11    \ // x.im * tw.im
	VFMADDSUB231PD X9, reg, X11 \ // tw * x
	VMOVAPD X11, reg

// Inverse input twiddle: conjugate the twiddle first.
#define ITWID(reg) \
	MOVUPD (R10)(R15*1), X8 \ // tw
	VXORPD X15, X8, X8      \ // conj(tw)
	VMOVDDUP X8, X9         \ // re
	VPERMILPD $1, X8, X10   \ // [im, re]
	VMOVDDUP X10, X10       \ // im
	VPERMILPD $1, reg, X11  \ // swap(x)
	VMULPD X10, X11, X11    \ // x.im * im
	VFMADDSUB231PD X9, reg, X11 \ // conj(tw) * x
	VMOVAPD X11, reg

// ===========================================================================
// Forward transform, size 512, complex128, radix-8 DIT
// ===========================================================================
TEXT ·ForwardAVX2Size512Radix8Complex128Asm(SB), $8192-97
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src_len+32(FP), R13 // R13 = len(src)
	LEAQ ·bitrev512_r8(SB), R12

	CMPQ R13, $512
	JL   size512_r8_128_fwd_return_false
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $512
	JL   size512_r8_128_fwd_return_false
	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $512
	JL   size512_r8_128_fwd_return_false
	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $512
	JL   size512_r8_128_fwd_return_false

	CMPQ R8, R9              // dst == src?
	JNE  size512_r8_128_fwd_use_dst
	MOVQ R11, R8             // work = scratch (in-place)

size512_r8_128_fwd_use_dst:
	LEAQ 0(SP), R14         // R14 = stage-1 buffer base

	// maskNegHi = [0, signbit]
	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X15
	VPERMILPD $1, X15, X15

	// -------------------------------------------------------------------
	// Stage 1: bit-reversed load from src, radix-8 butterfly, reorder into
	// the on-stack buffer (R14).
	// -------------------------------------------------------------------
	XORQ CX, CX             // base = 0

size512_r8_128_fwd_stage1_loop:
	CMPQ CX, $512
	JGE  size512_r8_128_fwd_stage2
	LEAQ (R12)(CX*8), R15   // &bitrev[base]

	MOVQ 0(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X0
	MOVQ 8(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X1
	MOVQ 16(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X2
	MOVQ 24(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X3
	MOVQ 32(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X4
	MOVQ 40(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X5
	MOVQ 48(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X6
	MOVQ 56(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X7

	FWDBFLY

	MOVQ CX, BX
	SHLQ $4, BX             // BX = base*16
	MOVUPD X5, 0(R14)(BX*1)   // y0
	MOVUPD X7, 64(R14)(BX*1)  // y4
	MOVUPD X8, 16(R14)(BX*1)  // y1
	MOVUPD X9, 80(R14)(BX*1)  // y5
	MOVUPD X1, 32(R14)(BX*1)  // y2
	MOVUPD X11, 96(R14)(BX*1) // y6
	MOVUPD X6, 48(R14)(BX*1)  // y3
	MOVUPD X4, 112(R14)(BX*1) // y7

	ADDQ $8, CX
	JMP  size512_r8_128_fwd_stage1_loop

size512_r8_128_fwd_stage2:
	// -------------------------------------------------------------------
	// Stage 2: 8 groups x 8 butterflies, span 8, twiddle stride 128*j bytes.
	// -------------------------------------------------------------------
	XORQ CX, CX             // base = 0

size512_r8_128_fwd_stage2_outer:
	CMPQ CX, $512
	JGE  size512_r8_128_fwd_stage3
	XORQ DX, DX             // j = 0

size512_r8_128_fwd_stage2_inner:
	CMPQ DX, $8
	JGE  size512_r8_128_fwd_stage2_next
	MOVQ CX, BX
	ADDQ DX, BX             // base + j
	MOVQ BX, SI
	SHLQ $4, SI             // (base+j)*16

	MOVUPD 0(R14)(SI*1), X0
	MOVUPD 128(R14)(SI*1), X1
	MOVUPD 256(R14)(SI*1), X2
	MOVUPD 384(R14)(SI*1), X3
	MOVUPD 512(R14)(SI*1), X4
	MOVUPD 640(R14)(SI*1), X5
	MOVUPD 768(R14)(SI*1), X6
	MOVUPD 896(R14)(SI*1), X7

	MOVQ DX, AX
	IMULQ $128, AX          // T = 128*j
	MOVQ AX, R15
	FTWID(X1)
	ADDQ AX, R15
	FTWID(X2)
	ADDQ AX, R15
	FTWID(X3)
	ADDQ AX, R15
	FTWID(X4)
	ADDQ AX, R15
	FTWID(X5)
	ADDQ AX, R15
	FTWID(X6)
	ADDQ AX, R15
	FTWID(X7)

	FWDBFLY

	MOVUPD X5, 0(R8)(SI*1)    // y0
	MOVUPD X7, 512(R8)(SI*1)  // y4
	MOVUPD X8, 128(R8)(SI*1)  // y1
	MOVUPD X9, 640(R8)(SI*1)  // y5
	MOVUPD X1, 256(R8)(SI*1)  // y2
	MOVUPD X11, 768(R8)(SI*1) // y6
	MOVUPD X6, 384(R8)(SI*1)  // y3
	MOVUPD X4, 896(R8)(SI*1)  // y7

	INCQ DX
	JMP  size512_r8_128_fwd_stage2_inner

size512_r8_128_fwd_stage2_next:
	ADDQ $64, CX
	JMP  size512_r8_128_fwd_stage2_outer

size512_r8_128_fwd_stage3:
	// -------------------------------------------------------------------
	// Stage 3: 64 butterflies, span 64, twiddle stride 16*j bytes. In place.
	// -------------------------------------------------------------------
	XORQ CX, CX             // j = 0

size512_r8_128_fwd_stage3_loop:
	CMPQ CX, $64
	JGE  size512_r8_128_fwd_copy
	MOVQ CX, SI
	SHLQ $4, SI             // j*16

	MOVUPD 0(R8)(SI*1), X0
	MOVUPD 1024(R8)(SI*1), X1
	MOVUPD 2048(R8)(SI*1), X2
	MOVUPD 3072(R8)(SI*1), X3
	MOVUPD 4096(R8)(SI*1), X4
	MOVUPD 5120(R8)(SI*1), X5
	MOVUPD 6144(R8)(SI*1), X6
	MOVUPD 7168(R8)(SI*1), X7

	MOVQ CX, AX
	IMULQ $16, AX           // T = 16*j
	MOVQ AX, R15
	FTWID(X1)
	ADDQ AX, R15
	FTWID(X2)
	ADDQ AX, R15
	FTWID(X3)
	ADDQ AX, R15
	FTWID(X4)
	ADDQ AX, R15
	FTWID(X5)
	ADDQ AX, R15
	FTWID(X6)
	ADDQ AX, R15
	FTWID(X7)

	FWDBFLY

	MOVUPD X5, 0(R8)(SI*1)     // y0
	MOVUPD X7, 4096(R8)(SI*1)  // y4
	MOVUPD X8, 1024(R8)(SI*1)  // y1
	MOVUPD X9, 5120(R8)(SI*1)  // y5
	MOVUPD X1, 2048(R8)(SI*1)  // y2
	MOVUPD X11, 6144(R8)(SI*1) // y6
	MOVUPD X6, 3072(R8)(SI*1)  // y3
	MOVUPD X4, 7168(R8)(SI*1)  // y7

	INCQ CX
	JMP  size512_r8_128_fwd_stage3_loop

size512_r8_128_fwd_copy:
	MOVQ dst+0(FP), R9      // R9 = dst pointer
	CMPQ R8, R9             // work == dst?
	JE   size512_r8_128_fwd_done
	XORQ CX, CX

size512_r8_128_fwd_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $8192          // 512 * 16 bytes
	JL   size512_r8_128_fwd_copy_loop

size512_r8_128_fwd_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size512_r8_128_fwd_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 512, complex128, radix-8 DIT
// ===========================================================================
TEXT ·InverseAVX2Size512Radix8Complex128Asm(SB), $8192-97
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src_len+32(FP), R13
	LEAQ ·bitrev512_r8(SB), R12

	CMPQ R13, $512
	JL   size512_r8_128_inv_return_false
	MOVQ dst_len+8(FP), AX
	CMPQ AX, $512
	JL   size512_r8_128_inv_return_false
	MOVQ twiddle_len+56(FP), AX
	CMPQ AX, $512
	JL   size512_r8_128_inv_return_false
	MOVQ scratch_len+80(FP), AX
	CMPQ AX, $512
	JL   size512_r8_128_inv_return_false

	CMPQ R8, R9
	JNE  size512_r8_128_inv_use_dst
	MOVQ R11, R8

size512_r8_128_inv_use_dst:
	LEAQ 0(SP), R14

	MOVQ ·signbit64(SB), AX
	VMOVQ AX, X15
	VPERMILPD $1, X15, X15   // maskNegHi = [0, signbit]

	XORQ CX, CX

size512_r8_128_inv_stage1_loop:
	CMPQ CX, $512
	JGE  size512_r8_128_inv_stage2
	LEAQ (R12)(CX*8), R15

	MOVQ 0(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X0
	MOVQ 8(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X1
	MOVQ 16(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X2
	MOVQ 24(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X3
	MOVQ 32(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X4
	MOVQ 40(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X5
	MOVQ 48(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X6
	MOVQ 56(R15), AX
	SHLQ $4, AX
	MOVUPD (R9)(AX*1), X7

	INVBFLY

	MOVQ CX, BX
	SHLQ $4, BX
	MOVUPD X5, 0(R14)(BX*1)
	MOVUPD X7, 64(R14)(BX*1)
	MOVUPD X8, 16(R14)(BX*1)
	MOVUPD X9, 80(R14)(BX*1)
	MOVUPD X1, 32(R14)(BX*1)
	MOVUPD X11, 96(R14)(BX*1)
	MOVUPD X6, 48(R14)(BX*1)
	MOVUPD X4, 112(R14)(BX*1)

	ADDQ $8, CX
	JMP  size512_r8_128_inv_stage1_loop

size512_r8_128_inv_stage2:
	XORQ CX, CX

size512_r8_128_inv_stage2_outer:
	CMPQ CX, $512
	JGE  size512_r8_128_inv_stage3
	XORQ DX, DX

size512_r8_128_inv_stage2_inner:
	CMPQ DX, $8
	JGE  size512_r8_128_inv_stage2_next
	MOVQ CX, BX
	ADDQ DX, BX
	MOVQ BX, SI
	SHLQ $4, SI

	MOVUPD 0(R14)(SI*1), X0
	MOVUPD 128(R14)(SI*1), X1
	MOVUPD 256(R14)(SI*1), X2
	MOVUPD 384(R14)(SI*1), X3
	MOVUPD 512(R14)(SI*1), X4
	MOVUPD 640(R14)(SI*1), X5
	MOVUPD 768(R14)(SI*1), X6
	MOVUPD 896(R14)(SI*1), X7

	MOVQ DX, AX
	IMULQ $128, AX
	MOVQ AX, R15
	ITWID(X1)
	ADDQ AX, R15
	ITWID(X2)
	ADDQ AX, R15
	ITWID(X3)
	ADDQ AX, R15
	ITWID(X4)
	ADDQ AX, R15
	ITWID(X5)
	ADDQ AX, R15
	ITWID(X6)
	ADDQ AX, R15
	ITWID(X7)

	INVBFLY

	MOVUPD X5, 0(R8)(SI*1)
	MOVUPD X7, 512(R8)(SI*1)
	MOVUPD X8, 128(R8)(SI*1)
	MOVUPD X9, 640(R8)(SI*1)
	MOVUPD X1, 256(R8)(SI*1)
	MOVUPD X11, 768(R8)(SI*1)
	MOVUPD X6, 384(R8)(SI*1)
	MOVUPD X4, 896(R8)(SI*1)

	INCQ DX
	JMP  size512_r8_128_inv_stage2_inner

size512_r8_128_inv_stage2_next:
	ADDQ $64, CX
	JMP  size512_r8_128_inv_stage2_outer

size512_r8_128_inv_stage3:
	XORQ CX, CX

size512_r8_128_inv_stage3_loop:
	CMPQ CX, $64
	JGE  size512_r8_128_inv_copy
	MOVQ CX, SI
	SHLQ $4, SI

	MOVUPD 0(R8)(SI*1), X0
	MOVUPD 1024(R8)(SI*1), X1
	MOVUPD 2048(R8)(SI*1), X2
	MOVUPD 3072(R8)(SI*1), X3
	MOVUPD 4096(R8)(SI*1), X4
	MOVUPD 5120(R8)(SI*1), X5
	MOVUPD 6144(R8)(SI*1), X6
	MOVUPD 7168(R8)(SI*1), X7

	MOVQ CX, AX
	IMULQ $16, AX
	MOVQ AX, R15
	ITWID(X1)
	ADDQ AX, R15
	ITWID(X2)
	ADDQ AX, R15
	ITWID(X3)
	ADDQ AX, R15
	ITWID(X4)
	ADDQ AX, R15
	ITWID(X5)
	ADDQ AX, R15
	ITWID(X6)
	ADDQ AX, R15
	ITWID(X7)

	INVBFLY

	MOVUPD X5, 0(R8)(SI*1)
	MOVUPD X7, 4096(R8)(SI*1)
	MOVUPD X8, 1024(R8)(SI*1)
	MOVUPD X9, 5120(R8)(SI*1)
	MOVUPD X1, 2048(R8)(SI*1)
	MOVUPD X11, 6144(R8)(SI*1)
	MOVUPD X6, 3072(R8)(SI*1)
	MOVUPD X4, 7168(R8)(SI*1)

	INCQ CX
	JMP  size512_r8_128_inv_stage3_loop

size512_r8_128_inv_copy:
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size512_r8_128_inv_scale
	XORQ CX, CX

size512_r8_128_inv_copy_loop:
	VMOVUPD (R8)(CX*1), Y0
	VMOVUPD 32(R8)(CX*1), Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $8192
	JL   size512_r8_128_inv_copy_loop

size512_r8_128_inv_scale:
	// 1/512 scaling on dst.
	MOVSD ·fiveHundredTwelfth64(SB), X8
	VBROADCASTSD X8, Y8
	MOVQ dst+0(FP), R9
	XORQ CX, CX

size512_r8_128_inv_scale_loop:
	VMOVUPD (R9)(CX*1), Y0
	VMOVUPD 32(R9)(CX*1), Y1
	VMULPD Y8, Y0, Y0
	VMULPD Y8, Y1, Y1
	VMOVUPD Y0, (R9)(CX*1)
	VMOVUPD Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $8192
	JL   size512_r8_128_inv_scale_loop

	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

size512_r8_128_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET
