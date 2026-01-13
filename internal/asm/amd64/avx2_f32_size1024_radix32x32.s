//go:build amd64 && asm && !purego

// ===========================================================================
// Size-1024 Radix-32×32 FFT Kernel (complex64)
// ===========================================================================
//
// This is a from-scratch, building-blocks implementation with minimal unrolling.
// It matches the reference decomposition in:
//   internal/kernels/dit_size1024_radix32x32.go
//
// Indexing conventions:
//   time index  n = n2*32 + n1
//   freq index  k = k1*32 + k2
//
// Forward (mixed radix 32×32):
//   Stage 1 (columns): for each n1, FFT32 over n2 (DIT with bit-reversed load)
//   Twiddle: multiply by W_1024^(k2*n1)
//   Stage 2 (rows): for each k2, FFT32 over n1 (DIT with bit-reversed load)
//   Store: dst[k1*32 + k2]
//
// Inverse:
//   Stage 1 (rows): for each k2, IFFT32 over k1 (DIT with bit-reversed load)
//   Twiddle: multiply by conj(W_1024^(k2*n1))
//   Stage 2 (cols): for each n1, IFFT32 over k2
//   Scale by 1/1024 and store dst[n2*32 + n1]
//
// FFT32 building block:
//   iterative radix-2 DIT using W_32 twiddles.
//   W_32^p is loaded from the provided W_1024 table as W_1024^(p*32).
//
// Notes:
// - Column FFTs (forward stage 1) and column IFFTs (inverse stage 2) are
//   vectorized by processing 4 columns at once using AVX2 YMM lanes.
// - Row FFTs/IFFTs and irregular twiddle handling remain scalar for now.
// - We still do VZEROUPPER.
// ===========================================================================

#include "textflag.h"

// bit-reversal permutation for N=32 (5 bits): rev[i] = bitreverse5(i)
// [0 16 8 24 4 20 12 28 2 18 10 26 6 22 14 30 1 17 9 25 5 21 13 29 3 19 11 27 7 23 15 31]
GLOBL ·bitrev32<>(SB), RODATA|NOPTR, $32
DATA ·bitrev32<>+0(SB)/8,  $0x1c0c140418081000
DATA ·bitrev32<>+8(SB)/8,  $0x1e0e16061a0a1202
DATA ·bitrev32<>+16(SB)/8, $0x1d0d150519091101
DATA ·bitrev32<>+24(SB)/8, $0x1f0f17071b0b1303

// float32(1/1024) = 2^-10
GLOBL ·scale1024f32<>(SB), RODATA|NOPTR, $4
DATA ·scale1024f32<>(SB)/4, $0x3a800000

// float32 sign bit mask (0x80000000)
GLOBL ·signMaskF32<>(SB), RODATA|NOPTR, $4
DATA ·signMaskF32<>(SB)/4, $0x80000000

// ---------------------------------------------------------------------------
// ForwardAVX2Size1024Radix32x32Complex64Asm
// Signature: func(dst, src, twiddle, scratch []complex64) bool
// ---------------------------------------------------------------------------
TEXT ·ForwardAVX2Size1024Radix32x32Complex64Asm(SB), $1280-97
	// ---- len checks ----
	MOVQ src+32(FP), AX
	CMPQ AX, $1024
	JL   fwd_fail
	MOVQ dst+8(FP), AX
	CMPQ AX, $1024
	JL   fwd_fail
	MOVQ twiddle+56(FP), AX
	CMPQ AX, $1024
	JL   fwd_fail
	MOVQ scratch+80(FP), AX
	CMPQ AX, $1024
	JL   fwd_fail

	// ---- args ----
	MOVQ dst+0(FP), R8       // dst base
	MOVQ src+24(FP), R9      // src base
	MOVQ twiddle+48(FP), R10 // twiddle base (W_1024)
	MOVQ scratch+72(FP), R11 // work base
	LEAQ 0(SP), R12          // local bufV[32]YMM (4 columns packed)

	// =====================================================================
	// Stage 1: for each n1, FFT32 over n2 (column) then apply W_1024^(k2*n1)
	// work[k2*32 + n1]
	// =====================================================================
	XORQ R13, R13 // n1=0 (processed in blocks of 4)
fwd_stage1_n1_loop:
	CMPQ R13, $32
	JGE  fwd_stage2

	// gather 4 columns src[n2*32+n1..n1+3] into bufV in bit-reversed order over n2
	LEAQ ·bitrev32<>(SB), R15
	XORQ AX, AX // i=0
fwd_gather_col_loop:
	CMPQ AX, $32
	JGE  fwd_do_fft_col
	MOVBLZX (R15)(AX*1), BX
	SHLQ $8, BX               // rev*256
	MOVQ R13, CX
	SHLQ $3, CX               // n1*8
	ADDQ CX, BX
	LEAQ (R9)(BX*1), SI
	VMOVUPS (SI), Y0
	MOVQ AX, DX
	SHLQ $5, DX
	VMOVUPS Y0, (R12)(DX*1)
	INCQ AX
	JMP  fwd_gather_col_loop

fwd_do_fft_col:
	JMP  fwd_fft32x4
fwd_fft32x4_return_col:

fwd_fft32_return_col:

	// apply inter-stage twiddle and store 4 columns to work
	XORQ R14, R14 // k2=0
fwd_store_work_loop:
	CMPQ R14, $32
	JGE  fwd_stage1_next_n1

	// Load packed FFT output for this k2: 4 complex64 lanes
	MOVQ R14, DX
	SHLQ $5, DX
	VMOVUPS (R12)(DX*1), Y0

	// Build packed twiddle vector [W(k2*(n1+0)), W(k2*(n1+1)), W(k2*(n1+2)), W(k2*(n1+3))]
	// baseIdx = k2*n1
	MOVQ R14, AX
	IMULQ R13, AX
	SHLQ $3, AX
	LEAQ (R10)(AX*1), DI      // DI = &twiddle[baseIdx]
	MOVQ R14, BX
	SHLQ $3, BX               // strideBytes = k2*8
	VMOVSD (DI), X8
	LEAQ (DI)(BX*1), SI
	VMOVSD (SI), X9
	VPUNPCKLQDQ X9, X8, X8
	LEAQ (SI)(BX*1), SI
	VMOVSD (SI), X9
	LEAQ (SI)(BX*1), SI
	VMOVSD (SI), X10
	VPUNPCKLQDQ X10, X9, X9
	VINSERTF128 $0, X8, Y8, Y8
	VINSERTF128 $1, X9, Y8, Y8
	VMOVSLDUP Y8, Y9
	VMOVSHDUP Y8, Y10

	// Y0 *= Y8 (complex multiply)
	VMOVAPS Y0, Y1
	VSHUFPS $0xB1, Y1, Y1, Y1
	VMULPS Y10, Y1, Y1
	VMULPS Y9, Y0, Y0
	VADDSUBPS Y1, Y0, Y0

	// Store to work[k2*32 + n1..n1+3]
	MOVQ R14, AX
	SHLQ $8, AX               // k2*256
	MOVQ R13, CX
	SHLQ $3, CX               // n1*8
	ADDQ CX, AX
	LEAQ (R11)(AX*1), DI
	VMOVUPS Y0, (DI)

	INCQ R14
	JMP  fwd_store_work_loop

fwd_stage1_next_n1:
	ADDQ $4, R13
	JMP  fwd_stage1_n1_loop

	// =====================================================================
	// Stage 2: for each k2, FFT32 over n1 (row) and store dst[k1*32+k2]
	// =====================================================================
fwd_stage2:
	LEAQ 1024(SP), R12        // R12 = scalar buf base
	XORQ R14, R14 // k2=0
fwd_stage2_k2_loop:
	CMPQ R14, $32
	JGE  fwd_ok

	MOVQ R14, DX
	SHLQ $8, DX               // base=k2*256
	LEAQ ·bitrev32<>(SB), R15
	XORQ AX, AX               // i=0
fwd_gather_row_loop:
	CMPQ AX, $32
	JGE  fwd_do_fft_row
	MOVBLZX (R15)(AX*1), BX
	MOVQ BX, CX
	SHLQ $3, CX               // rev*8
	ADDQ DX, CX
	MOVQ (R11)(CX*1), SI
	MOVQ SI, (R12)(AX*8)
	INCQ AX
	JMP  fwd_gather_row_loop

fwd_do_fft_row:
	MOVQ $1, BP               // return selector 1
	JMP  fwd_fft32
fwd_fft32_return_row:

	XORQ R13, R13 // k1=0
fwd_store_dst_loop:
	CMPQ R13, $32
	JGE  fwd_stage2_next_k2
	MOVQ (R12)(R13*8), AX
	MOVQ R13, BX
	SHLQ $8, BX               // k1*256
	MOVQ R14, CX
	SHLQ $3, CX               // k2*8
	ADDQ CX, BX
	MOVQ AX, (R8)(BX*1)
	INCQ R13
	JMP  fwd_store_dst_loop

fwd_stage2_next_k2:
	INCQ R14
	JMP  fwd_stage2_k2_loop

	// =======================================================================
	// Local helper: iterative radix-2 DIT FFT-32 on bufV (forward), 4 columns packed
	// Expects:
	//   R12 = &bufV[0] (32 YMM values, each holds 4 complex64 lanes)
	//   R10 = twiddle base (W_1024) where W_32^p == W_1024^(p*32)
	// Returns to fwd_fft32x4_return_col
	// =======================================================================

fwd_fft32x4:
	// Stage m=2 (tw=1)
	XORQ AX, AX
fwdx4_s1_kloop:
	CMPQ AX, $32
	JGE  fwdx4_s2
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 32(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	VMOVAPS Y0, Y2
	VADDPS Y1, Y2, Y2
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y2, (SI)
	VMOVUPS Y0, (DI)
	ADDQ $2, AX
	JMP  fwdx4_s1_kloop

fwdx4_s2:
	// Stage m=4: tw offset = j*2048 bytes (W_32^{j*8})
	XORQ CX, CX
fwdx4_s2_jloop:
	CMPQ CX, $2
	JGE  fwdx4_s3
	CMPQ CX, $0
	JEQ  fwdx4_s2_kloop_init
	MOVQ CX, DX
	SHLQ $11, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSS 0(R15), Y8
	VBROADCASTSS 4(R15), Y9

fwdx4_s2_kloop_init:
	MOVQ CX, AX
fwdx4_s2_kloop:
	CMPQ AX, $32
	JGE  fwdx4_s2_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 64(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	CMPQ CX, $0
	JEQ  fwdx4_s2_bfly
	// Y1 *= (Y8 + i*Y9)
	VMOVAPS Y1, Y2
	VSHUFPS $0xB1, Y2, Y2, Y2
	VMULPS Y9, Y2, Y2
	VMULPS Y8, Y1, Y1
	VADDSUBPS Y2, Y1, Y1
fwdx4_s2_bfly:
	VMOVAPS Y0, Y3
	VADDPS Y1, Y3, Y3
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y3, (SI)
	VMOVUPS Y0, (DI)
	ADDQ $4, AX
	JMP  fwdx4_s2_kloop
fwdx4_s2_nextj:
	INCQ CX
	JMP  fwdx4_s2_jloop

fwdx4_s3:
	// Stage m=8: tw offset = j*1024 bytes (W_32^{j*4})
	XORQ CX, CX
fwdx4_s3_jloop:
	CMPQ CX, $4
	JGE  fwdx4_s4
	CMPQ CX, $0
	JEQ  fwdx4_s3_kloop_init
	MOVQ CX, DX
	SHLQ $10, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSS 0(R15), Y8
	VBROADCASTSS 4(R15), Y9

fwdx4_s3_kloop_init:
	MOVQ CX, AX
fwdx4_s3_kloop:
	CMPQ AX, $32
	JGE  fwdx4_s3_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 128(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	CMPQ CX, $0
	JEQ  fwdx4_s3_bfly
	VMOVAPS Y1, Y2
	VSHUFPS $0xB1, Y2, Y2, Y2
	VMULPS Y9, Y2, Y2
	VMULPS Y8, Y1, Y1
	VADDSUBPS Y2, Y1, Y1
fwdx4_s3_bfly:
	VMOVAPS Y0, Y3
	VADDPS Y1, Y3, Y3
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y3, (SI)
	VMOVUPS Y0, (DI)
	ADDQ $8, AX
	JMP  fwdx4_s3_kloop
fwdx4_s3_nextj:
	INCQ CX
	JMP  fwdx4_s3_jloop

fwdx4_s4:
	// Stage m=16: tw offset = j*512 bytes (W_32^{j*2})
	XORQ CX, CX
fwdx4_s4_jloop:
	CMPQ CX, $8
	JGE  fwdx4_s5
	CMPQ CX, $0
	JEQ  fwdx4_s4_kloop_init
	MOVQ CX, DX
	SHLQ $9, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSS 0(R15), Y8
	VBROADCASTSS 4(R15), Y9

fwdx4_s4_kloop_init:
	MOVQ CX, AX
fwdx4_s4_kloop:
	CMPQ AX, $32
	JGE  fwdx4_s4_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 256(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	CMPQ CX, $0
	JEQ  fwdx4_s4_bfly
	VMOVAPS Y1, Y2
	VSHUFPS $0xB1, Y2, Y2, Y2
	VMULPS Y9, Y2, Y2
	VMULPS Y8, Y1, Y1
	VADDSUBPS Y2, Y1, Y1
fwdx4_s4_bfly:
	VMOVAPS Y0, Y3
	VADDPS Y1, Y3, Y3
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y3, (SI)
	VMOVUPS Y0, (DI)
	ADDQ $16, AX
	JMP  fwdx4_s4_kloop
fwdx4_s4_nextj:
	INCQ CX
	JMP  fwdx4_s4_jloop

fwdx4_s5:
	// Stage m=32: tw offset = j*256 bytes (W_32^{j})
	XORQ CX, CX
fwdx4_s5_jloop:
	CMPQ CX, $16
	JGE  fwd_fft32x4_ret
	CMPQ CX, $0
	JEQ  fwdx4_s5_do
	MOVQ CX, DX
	SHLQ $8, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSS 0(R15), Y8
	VBROADCASTSS 4(R15), Y9

fwdx4_s5_do:
	MOVQ CX, AX
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 512(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	CMPQ CX, $0
	JEQ  fwdx4_s5_bfly
	VMOVAPS Y1, Y2
	VSHUFPS $0xB1, Y2, Y2, Y2
	VMULPS Y9, Y2, Y2
	VMULPS Y8, Y1, Y1
	VADDSUBPS Y2, Y1, Y1
fwdx4_s5_bfly:
	VMOVAPS Y0, Y3
	VADDPS Y1, Y3, Y3
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y3, (SI)
	VMOVUPS Y0, (DI)
	INCQ CX
	JMP  fwdx4_s5_jloop

fwd_fft32x4_ret:
	JMP  fwd_fft32x4_return_col

	// =======================================================================
	// Local helper: iterative radix-2 DIT FFT-32 on buf (forward)
	// Expects:
	//   R12 = &buf[0]
	//   R10 = twiddle base (W_1024) where W_32^p == W_1024^(p*32)
	//   BP  = selector (0->return_col, 1->return_row)
	// =======================================================================

	// ---- forward FFT32 (DIT, twiddle W_32) ----
fwd_fft32:
	// Stage m=2 (tw=1)
	XORQ AX, AX
fwd_s1_kloop:
	CMPQ AX, $32
	JGE  fwd_s2
	LEAQ (R12)(AX*8), SI
	LEAQ 8(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	ADDSS X2, X0
	ADDSS X3, X1
	MOVSS 0(SI), X4
	MOVSS 4(SI), X5
	SUBSS X2, X4
	SUBSS X3, X5
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X4, 0(DI)
	MOVSS X5, 4(DI)
	ADDQ $2, AX
	JMP  fwd_s1_kloop

fwd_s2:
	// Stage m=4: tw offset = j*2048 bytes (W_32^{j*8})
	XORQ CX, CX
fwd_s2_jloop:
	CMPQ CX, $2
	JGE  fwd_s3
	MOVQ CX, DX
	SHLQ $11, DX
	LEAQ (R10)(DX*1), R15
	MOVSS 0(R15), X8
	MOVSS 4(R15), X9
	MOVQ CX, AX
fwd_s2_kloop:
	CMPQ AX, $32
	JGE  fwd_s2_nextj
	LEAQ (R12)(AX*8), SI
	LEAQ 16(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	MOVSS X2, X4
	MULSS X8, X4
	MOVSS X3, X5
	MULSS X9, X5
	SUBSS X5, X4
	MOVSS X2, X6
	MULSS X9, X6
	MOVSS X3, X7
	MULSS X8, X7
	ADDSS X7, X6
	ADDSS X4, X0
	ADDSS X6, X1
	MOVSS 0(SI), X2
	MOVSS 4(SI), X3
	SUBSS X4, X2
	SUBSS X6, X3
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X2, 0(DI)
	MOVSS X3, 4(DI)
	ADDQ $4, AX
	JMP  fwd_s2_kloop
fwd_s2_nextj:
	INCQ CX
	JMP  fwd_s2_jloop

fwd_s3:
	// Stage m=8: tw offset = j*1024 bytes (W_32^{j*4})
	XORQ CX, CX
fwd_s3_jloop:
	CMPQ CX, $4
	JGE  fwd_s4
	MOVQ CX, DX
	SHLQ $10, DX
	LEAQ (R10)(DX*1), R15
	MOVSS 0(R15), X8
	MOVSS 4(R15), X9
	MOVQ CX, AX
fwd_s3_kloop:
	CMPQ AX, $32
	JGE  fwd_s3_nextj
	LEAQ (R12)(AX*8), SI
	LEAQ 32(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	MOVSS X2, X4
	MULSS X8, X4
	MOVSS X3, X5
	MULSS X9, X5
	SUBSS X5, X4
	MOVSS X2, X6
	MULSS X9, X6
	MOVSS X3, X7
	MULSS X8, X7
	ADDSS X7, X6
	ADDSS X4, X0
	ADDSS X6, X1
	MOVSS 0(SI), X2
	MOVSS 4(SI), X3
	SUBSS X4, X2
	SUBSS X6, X3
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X2, 0(DI)
	MOVSS X3, 4(DI)
	ADDQ $8, AX
	JMP  fwd_s3_kloop
fwd_s3_nextj:
	INCQ CX
	JMP  fwd_s3_jloop

fwd_s4:
	// Stage m=16: tw offset = j*512 bytes (W_32^{j*2})
	XORQ CX, CX
fwd_s4_jloop:
	CMPQ CX, $8
	JGE  fwd_s5
	MOVQ CX, DX
	SHLQ $9, DX
	LEAQ (R10)(DX*1), R15
	MOVSS 0(R15), X8
	MOVSS 4(R15), X9
	MOVQ CX, AX
fwd_s4_kloop:
	CMPQ AX, $32
	JGE  fwd_s4_nextj
	LEAQ (R12)(AX*8), SI
	LEAQ 64(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	MOVSS X2, X4
	MULSS X8, X4
	MOVSS X3, X5
	MULSS X9, X5
	SUBSS X5, X4
	MOVSS X2, X6
	MULSS X9, X6
	MOVSS X3, X7
	MULSS X8, X7
	ADDSS X7, X6
	ADDSS X4, X0
	ADDSS X6, X1
	MOVSS 0(SI), X2
	MOVSS 4(SI), X3
	SUBSS X4, X2
	SUBSS X6, X3
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X2, 0(DI)
	MOVSS X3, 4(DI)
	ADDQ $16, AX
	JMP  fwd_s4_kloop
fwd_s4_nextj:
	INCQ CX
	JMP  fwd_s4_jloop

fwd_s5:
	// Stage m=32: tw offset = j*256 bytes (W_32^{j})
	XORQ CX, CX
fwd_s5_jloop:
	CMPQ CX, $16
	JGE  fwd_fft32_ret
	MOVQ CX, DX
	SHLQ $8, DX
	LEAQ (R10)(DX*1), R15
	MOVSS 0(R15), X8
	MOVSS 4(R15), X9
	MOVQ CX, AX
	LEAQ (R12)(AX*8), SI
	LEAQ 128(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	MOVSS X2, X4
	MULSS X8, X4
	MOVSS X3, X5
	MULSS X9, X5
	SUBSS X5, X4
	MOVSS X2, X6
	MULSS X9, X6
	MOVSS X3, X7
	MULSS X8, X7
	ADDSS X7, X6
	ADDSS X4, X0
	ADDSS X6, X1
	MOVSS 0(SI), X2
	MOVSS 4(SI), X3
	SUBSS X4, X2
	SUBSS X6, X3
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X2, 0(DI)
	MOVSS X3, 4(DI)
	INCQ CX
	JMP  fwd_s5_jloop

fwd_fft32_ret:
	CMPQ BP, $1
	JEQ  fwd_fft32_return_row
	JMP  fwd_fft32_return_col

fwd_ok:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

fwd_fail:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ---------------------------------------------------------------------------
// InverseAVX2Size1024Radix32x32Complex64Asm
// Signature: func(dst, src, twiddle, scratch []complex64) bool
// ---------------------------------------------------------------------------
TEXT ·InverseAVX2Size1024Radix32x32Complex64Asm(SB), $1280-97
	// ---- len checks ----
	MOVQ src+32(FP), AX
	CMPQ AX, $1024
	JL   inv_fail
	MOVQ dst+8(FP), AX
	CMPQ AX, $1024
	JL   inv_fail
	MOVQ twiddle+56(FP), AX
	CMPQ AX, $1024
	JL   inv_fail
	MOVQ scratch+80(FP), AX
	CMPQ AX, $1024
	JL   inv_fail

	// ---- args ----
	MOVQ dst+0(FP), R8       // dst base (time)
	MOVQ src+24(FP), R9      // src base (freq)
	MOVQ twiddle+48(FP), R10 // twiddle base (W_1024)
	MOVQ scratch+72(FP), R11 // work base
	LEAQ 0(SP), R12          // local bufV[32]YMM (4 columns packed)
	MOVSS ·scale1024f32<>(SB), X14

	// =====================================================================
	// Stage 1: for each k2, IFFT32 over k1 then apply conj(W_1024^(k2*n1))
	// work[k2*32 + n1]
	// =====================================================================
	XORQ R14, R14 // k2=0
inv_stage1_k2_loop:
	CMPQ R14, $32
	JGE  inv_stage2

	LEAQ ·bitrev32<>(SB), R15
	LEAQ 1024(SP), R12        // scalar buf base
	XORQ AX, AX // i=0
inv_gather_row_loop:
	CMPQ AX, $32
	JGE  inv_do_ifft_row
	MOVBLZX (R15)(AX*1), BX
	MOVQ BX, CX
	SHLQ $8, CX               // rev*256
	MOVQ R14, DX
	SHLQ $3, DX               // k2*8
	ADDQ DX, CX
	MOVQ (R9)(CX*1), SI
	MOVQ SI, (R12)(AX*8)
	INCQ AX
	JMP  inv_gather_row_loop

inv_do_ifft_row:
	XORQ BP, BP               // return selector 0
	JMP  inv_fft32
inv_fft32_return_row:

	XORQ R13, R13 // n1=0
inv_store_work_loop:
	CMPQ R13, $32
	JGE  inv_stage1_next_k2

	LEAQ (R12)(R13*8), SI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1

	MOVQ R14, AX
	IMULQ R13, AX
	SHLQ $3, AX
	LEAQ (R10)(AX*1), DI
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3

	// (a+bi)*(c-di)
	MOVSS X0, X4
	MULSS X2, X4
	MOVSS X1, X5
	MULSS X3, X5
	ADDSS X5, X4              // real
	MOVSS X1, X6
	MULSS X2, X6
	MOVSS X0, X7
	MULSS X3, X7
	SUBSS X7, X6              // imag

	MOVQ R14, AX
	SHLQ $8, AX               // k2*256
	MOVQ R13, BX
	SHLQ $3, BX               // n1*8
	ADDQ BX, AX
	LEAQ (R11)(AX*1), DI
	MOVSS X4, 0(DI)
	MOVSS X6, 4(DI)

	INCQ R13
	JMP  inv_store_work_loop

inv_stage1_next_k2:
	INCQ R14
	JMP  inv_stage1_k2_loop

	// =====================================================================
	// Stage 2: for each n1, IFFT32 over k2, scale by 1/1024, store dst[n2*32+n1]
	// =====================================================================
inv_stage2:
	LEAQ 0(SP), R12           // bufV base
	XORQ R13, R13 // n1=0 (processed in blocks of 4)
inv_stage2_n1_loop:
	CMPQ R13, $32
	JGE  inv_ok

	// gather 4 columns work[rev*32+n1..n1+3] into bufV
	LEAQ ·bitrev32<>(SB), R15
	XORQ AX, AX // i=0
inv_gather_col_loop:
	CMPQ AX, $32
	JGE  inv_do_ifft_col
	MOVBLZX (R15)(AX*1), BX
	MOVQ BX, CX
	SHLQ $8, CX               // rev*256
	MOVQ R13, DX
	SHLQ $3, DX               // n1*8
	ADDQ DX, CX
	LEAQ (R11)(CX*1), SI
	VMOVUPS (SI), Y0
	MOVQ AX, DI
	SHLQ $5, DI
	VMOVUPS Y0, (R12)(DI*1)
	INCQ AX
	JMP  inv_gather_col_loop

inv_do_ifft_col:
	JMP  inv_fft32x4
inv_fft32x4_return_col:

inv_fft32_return_col:

	VBROADCASTSS ·scale1024f32<>(SB), Y15
	XORQ R14, R14 // n2=0
inv_store_dst_loop:
	CMPQ R14, $32
	JGE  inv_stage2_next_n1

	MOVQ R14, DX
	SHLQ $5, DX
	VMOVUPS (R12)(DX*1), Y0
	VMULPS Y15, Y0, Y0
	MOVQ R14, AX
	SHLQ $8, AX               // n2*256
	MOVQ R13, BX
	SHLQ $3, BX               // n1*8
	ADDQ BX, AX
	LEAQ (R8)(AX*1), DI
	VMOVUPS Y0, (DI)

	INCQ R14
	JMP  inv_store_dst_loop

inv_stage2_next_n1:
	ADDQ $4, R13
	JMP  inv_stage2_n1_loop

	// =======================================================================
	// Local helper: iterative radix-2 DIT IFFT-32 on bufV (unscaled), 4 columns packed
	// Expects:
	//   R12 = &bufV[0]
	//   R10 = twiddle base (W_1024)
	// Returns to inv_fft32x4_return_col
	// =======================================================================

inv_fft32x4:
	VBROADCASTSS ·signMaskF32<>(SB), Y14
	// Stage m=2
	XORQ AX, AX
invx4_s1_kloop:
	CMPQ AX, $32
	JGE  invx4_s2
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 32(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	VMOVAPS Y0, Y2
	VADDPS Y1, Y2, Y2
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y2, (SI)
	VMOVUPS Y0, (DI)
	ADDQ $2, AX
	JMP  invx4_s1_kloop

invx4_s2:
	// Stage m=4 (conjugated twiddle: use imag = -d)
	XORQ CX, CX
invx4_s2_jloop:
	CMPQ CX, $2
	JGE  invx4_s3
	CMPQ CX, $0
	JEQ  invx4_s2_kloop_init
	MOVQ CX, DX
	SHLQ $11, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSS 0(R15), Y8
	VBROADCASTSS 4(R15), Y9
	VXORPS Y14, Y9, Y9
invx4_s2_kloop_init:
	MOVQ CX, AX
invx4_s2_kloop:
	CMPQ AX, $32
	JGE  invx4_s2_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 64(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	CMPQ CX, $0
	JEQ  invx4_s2_bfly
	VMOVAPS Y1, Y2
	VSHUFPS $0xB1, Y2, Y2, Y2
	VMULPS Y9, Y2, Y2
	VMULPS Y8, Y1, Y1
	VADDSUBPS Y2, Y1, Y1
invx4_s2_bfly:
	VMOVAPS Y0, Y3
	VADDPS Y1, Y3, Y3
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y3, (SI)
	VMOVUPS Y0, (DI)
	ADDQ $4, AX
	JMP  invx4_s2_kloop
invx4_s2_nextj:
	INCQ CX
	JMP  invx4_s2_jloop

invx4_s3:
	// Stage m=8
	XORQ CX, CX
invx4_s3_jloop:
	CMPQ CX, $4
	JGE  invx4_s4
	CMPQ CX, $0
	JEQ  invx4_s3_kloop_init
	MOVQ CX, DX
	SHLQ $10, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSS 0(R15), Y8
	VBROADCASTSS 4(R15), Y9
	VXORPS Y14, Y9, Y9
invx4_s3_kloop_init:
	MOVQ CX, AX
invx4_s3_kloop:
	CMPQ AX, $32
	JGE  invx4_s3_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 128(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	CMPQ CX, $0
	JEQ  invx4_s3_bfly
	VMOVAPS Y1, Y2
	VSHUFPS $0xB1, Y2, Y2, Y2
	VMULPS Y9, Y2, Y2
	VMULPS Y8, Y1, Y1
	VADDSUBPS Y2, Y1, Y1
invx4_s3_bfly:
	VMOVAPS Y0, Y3
	VADDPS Y1, Y3, Y3
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y3, (SI)
	VMOVUPS Y0, (DI)
	ADDQ $8, AX
	JMP  invx4_s3_kloop
invx4_s3_nextj:
	INCQ CX
	JMP  invx4_s3_jloop

invx4_s4:
	// Stage m=16
	XORQ CX, CX
invx4_s4_jloop:
	CMPQ CX, $8
	JGE  invx4_s5
	CMPQ CX, $0
	JEQ  invx4_s4_kloop_init
	MOVQ CX, DX
	SHLQ $9, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSS 0(R15), Y8
	VBROADCASTSS 4(R15), Y9
	VXORPS Y14, Y9, Y9
invx4_s4_kloop_init:
	MOVQ CX, AX
invx4_s4_kloop:
	CMPQ AX, $32
	JGE  invx4_s4_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 256(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	CMPQ CX, $0
	JEQ  invx4_s4_bfly
	VMOVAPS Y1, Y2
	VSHUFPS $0xB1, Y2, Y2, Y2
	VMULPS Y9, Y2, Y2
	VMULPS Y8, Y1, Y1
	VADDSUBPS Y2, Y1, Y1
invx4_s4_bfly:
	VMOVAPS Y0, Y3
	VADDPS Y1, Y3, Y3
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y3, (SI)
	VMOVUPS Y0, (DI)
	ADDQ $16, AX
	JMP  invx4_s4_kloop
invx4_s4_nextj:
	INCQ CX
	JMP  invx4_s4_jloop

invx4_s5:
	// Stage m=32
	XORQ CX, CX
invx4_s5_jloop:
	CMPQ CX, $16
	JGE  inv_fft32x4_ret
	CMPQ CX, $0
	JEQ  invx4_s5_do
	MOVQ CX, DX
	SHLQ $8, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSS 0(R15), Y8
	VBROADCASTSS 4(R15), Y9
	VXORPS Y14, Y9, Y9
invx4_s5_do:
	MOVQ CX, AX
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 512(SI), DI
	VMOVUPS (SI), Y0
	VMOVUPS (DI), Y1
	CMPQ CX, $0
	JEQ  invx4_s5_bfly
	VMOVAPS Y1, Y2
	VSHUFPS $0xB1, Y2, Y2, Y2
	VMULPS Y9, Y2, Y2
	VMULPS Y8, Y1, Y1
	VADDSUBPS Y2, Y1, Y1
invx4_s5_bfly:
	VMOVAPS Y0, Y3
	VADDPS Y1, Y3, Y3
	VSUBPS Y1, Y0, Y0
	VMOVUPS Y3, (SI)
	VMOVUPS Y0, (DI)
	INCQ CX
	JMP  invx4_s5_jloop

inv_fft32x4_ret:
	JMP  inv_fft32x4_return_col

	// =======================================================================
	// Local helper: iterative radix-2 DIT IFFT-32 on buf (unscaled)
	// Expects:
	//   R12 = &buf[0]
	//   R10 = twiddle base (W_1024) where W_32^p == W_1024^(p*32)
	//   BP  = selector (0->return_row, 1->return_col)
	// =======================================================================

	// ---- inverse IFFT32 (DIT, conjugated W_32, unscaled) ----
inv_fft32:
	// Stage m=2
	XORQ AX, AX
inv_s1_kloop:
	CMPQ AX, $32
	JGE  inv_s2
	LEAQ (R12)(AX*8), SI
	LEAQ 8(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	ADDSS X2, X0
	ADDSS X3, X1
	MOVSS 0(SI), X4
	MOVSS 4(SI), X5
	SUBSS X2, X4
	SUBSS X3, X5
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X4, 0(DI)
	MOVSS X5, 4(DI)
	ADDQ $2, AX
	JMP  inv_s1_kloop

inv_s2:
	// Stage m=4 (conj twiddle)
	XORQ CX, CX
inv_s2_jloop:
	CMPQ CX, $2
	JGE  inv_s3
	MOVQ CX, DX
	SHLQ $11, DX
	LEAQ (R10)(DX*1), R15
	MOVSS 0(R15), X8
	MOVSS 4(R15), X9
	MOVQ CX, AX
inv_s2_kloop:
	CMPQ AX, $32
	JGE  inv_s2_nextj
	LEAQ (R12)(AX*8), SI
	LEAQ 16(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	MOVSS X2, X4
	MULSS X8, X4
	MOVSS X3, X5
	MULSS X9, X5
	ADDSS X5, X4
	MOVSS X3, X6
	MULSS X8, X6
	MOVSS X2, X7
	MULSS X9, X7
	SUBSS X7, X6
	ADDSS X4, X0
	ADDSS X6, X1
	MOVSS 0(SI), X2
	MOVSS 4(SI), X3
	SUBSS X4, X2
	SUBSS X6, X3
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X2, 0(DI)
	MOVSS X3, 4(DI)
	ADDQ $4, AX
	JMP  inv_s2_kloop
inv_s2_nextj:
	INCQ CX
	JMP  inv_s2_jloop

inv_s3:
	// Stage m=8
	XORQ CX, CX
inv_s3_jloop:
	CMPQ CX, $4
	JGE  inv_s4
	MOVQ CX, DX
	SHLQ $10, DX
	LEAQ (R10)(DX*1), R15
	MOVSS 0(R15), X8
	MOVSS 4(R15), X9
	MOVQ CX, AX
inv_s3_kloop:
	CMPQ AX, $32
	JGE  inv_s3_nextj
	LEAQ (R12)(AX*8), SI
	LEAQ 32(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	MOVSS X2, X4
	MULSS X8, X4
	MOVSS X3, X5
	MULSS X9, X5
	ADDSS X5, X4
	MOVSS X3, X6
	MULSS X8, X6
	MOVSS X2, X7
	MULSS X9, X7
	SUBSS X7, X6
	ADDSS X4, X0
	ADDSS X6, X1
	MOVSS 0(SI), X2
	MOVSS 4(SI), X3
	SUBSS X4, X2
	SUBSS X6, X3
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X2, 0(DI)
	MOVSS X3, 4(DI)
	ADDQ $8, AX
	JMP  inv_s3_kloop
inv_s3_nextj:
	INCQ CX
	JMP  inv_s3_jloop

inv_s4:
	// Stage m=16
	XORQ CX, CX
inv_s4_jloop:
	CMPQ CX, $8
	JGE  inv_s5
	MOVQ CX, DX
	SHLQ $9, DX
	LEAQ (R10)(DX*1), R15
	MOVSS 0(R15), X8
	MOVSS 4(R15), X9
	MOVQ CX, AX
inv_s4_kloop:
	CMPQ AX, $32
	JGE  inv_s4_nextj
	LEAQ (R12)(AX*8), SI
	LEAQ 64(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	MOVSS X2, X4
	MULSS X8, X4
	MOVSS X3, X5
	MULSS X9, X5
	ADDSS X5, X4
	MOVSS X3, X6
	MULSS X8, X6
	MOVSS X2, X7
	MULSS X9, X7
	SUBSS X7, X6
	ADDSS X4, X0
	ADDSS X6, X1
	MOVSS 0(SI), X2
	MOVSS 4(SI), X3
	SUBSS X4, X2
	SUBSS X6, X3
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X2, 0(DI)
	MOVSS X3, 4(DI)
	ADDQ $16, AX
	JMP  inv_s4_kloop
inv_s4_nextj:
	INCQ CX
	JMP  inv_s4_jloop

inv_s5:
	// Stage m=32
	XORQ CX, CX
inv_s5_jloop:
	CMPQ CX, $16
	JGE  inv_fft32_ret
	MOVQ CX, DX
	SHLQ $8, DX
	LEAQ (R10)(DX*1), R15
	MOVSS 0(R15), X8
	MOVSS 4(R15), X9
	MOVQ CX, AX
	LEAQ (R12)(AX*8), SI
	LEAQ 128(SI), DI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3
	MOVSS X2, X4
	MULSS X8, X4
	MOVSS X3, X5
	MULSS X9, X5
	ADDSS X5, X4
	MOVSS X3, X6
	MULSS X8, X6
	MOVSS X2, X7
	MULSS X9, X7
	SUBSS X7, X6
	ADDSS X4, X0
	ADDSS X6, X1
	MOVSS 0(SI), X2
	MOVSS 4(SI), X3
	SUBSS X4, X2
	SUBSS X6, X3
	MOVSS X0, 0(SI)
	MOVSS X1, 4(SI)
	MOVSS X2, 0(DI)
	MOVSS X3, 4(DI)
	INCQ CX
	JMP  inv_s5_jloop

inv_fft32_ret:
	CMPQ BP, $1
	JEQ  inv_fft32_return_col
	JMP  inv_fft32_return_row

inv_ok:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

inv_fail:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET
