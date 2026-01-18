//go:build amd64 && asm && !purego

// ===========================================================================
// Size-1024 Radix-32×32 FFT Kernel (complex128)
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
//   vectorized by processing 2 columns at once using AVX2 YMM lanes.
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

// float64(1/1024) = 2^-10
GLOBL ·scale1024f64<>(SB), RODATA|NOPTR, $8
DATA ·scale1024f64<>(SB)/8, $0x3f50000000000000

// float64 sign bit mask (0x8000000000000000)
GLOBL ·signMaskF64<>(SB), RODATA|NOPTR, $8
DATA ·signMaskF64<>(SB)/8, $0x8000000000000000

// ---------------------------------------------------------------------------
// ForwardAVX2Size1024Radix32x32Complex128Asm
// Signature: func(dst, src, twiddle, scratch []complex128) bool
// ---------------------------------------------------------------------------
TEXT ·ForwardAVX2Size1024Radix32x32Complex128Asm(SB), $1536-97
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
	LEAQ 0(SP), R12          // local bufV[32]YMM (2 columns packed)

	// =====================================================================
	// Stage 1: for each n1, FFT32 over n2 (column) then apply W_1024^(k2*n1)
	// work[k2*32 + n1]
	// =====================================================================
	XORQ R13, R13 // n1=0 (processed in blocks of 2)
fwd_stage1_n1_loop:
	CMPQ R13, $32
	JGE  fwd_stage2

	// gather 2 columns src[n2*32+n1..n1+1] into bufV in bit-reversed order over n2
	LEAQ ·bitrev32<>(SB), R15
	XORQ AX, AX // i=0
fwd_gather_col_loop:
	CMPQ AX, $32
	JGE  fwd_do_fft_col
	MOVBLZX (R15)(AX*1), BX
	SHLQ $9, BX               // rev*512
	MOVQ R13, CX
	SHLQ $4, CX               // n1*16
	ADDQ CX, BX
	LEAQ (R9)(BX*1), SI
	VMOVUPD (SI), Y0
	MOVQ AX, DX
	SHLQ $5, DX
	VMOVUPD Y0, (R12)(DX*1)
	INCQ AX
	JMP  fwd_gather_col_loop

fwd_do_fft_col:
	JMP  fwd_fft32x4
fwd_fft32x4_return_col:

fwd_fft32_return_col:

	// apply inter-stage twiddle and store 2 columns to work
	XORQ R14, R14 // k2=0
fwd_store_work_loop:
	CMPQ R14, $32
	JGE  fwd_stage1_next_n1

	// Load packed FFT output for this k2: 2 complex128 lanes
	MOVQ R14, DX
	SHLQ $5, DX
	VMOVUPD (R12)(DX*1), Y0

	// Build packed twiddle vector [W(k2*(n1+0)), W(k2*(n1+1))]
	// baseIdx = k2*n1
	MOVQ R14, AX
	IMULQ R13, AX
	SHLQ $4, AX
	LEAQ (R10)(AX*1), DI      // DI = &twiddle[baseIdx]
	MOVQ R14, BX
	SHLQ $4, BX               // strideBytes = k2*16
	VMOVUPD (DI), X8
	LEAQ (DI)(BX*1), SI
	VMOVUPD (SI), X9
	VINSERTF128 $0, X8, Y8, Y8
	VINSERTF128 $1, X9, Y8, Y8
	VMOVDDUP Y8, Y9
	VPERMILPD $0x0F, Y8, Y10

	// Y0 *= Y8 (complex multiply)
	VMOVAPD Y0, Y1
	VPERMILPD $0x05, Y1, Y1
	VMULPD Y10, Y1, Y1
	VMULPD Y9, Y0, Y0
	VADDSUBPD Y1, Y0, Y0

	// Store to work[k2*32 + n1..n1+1]
	MOVQ R14, AX
	SHLQ $9, AX               // k2*512
	MOVQ R13, CX
	SHLQ $4, CX               // n1*16
	ADDQ CX, AX
	LEAQ (R11)(AX*1), DI
	VMOVUPD Y0, (DI)

	INCQ R14
	JMP  fwd_store_work_loop

fwd_stage1_next_n1:
	ADDQ $2, R13
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
	SHLQ $9, DX               // base=k2*512
	LEAQ ·bitrev32<>(SB), R15
	XORQ AX, AX               // i=0
fwd_gather_row_loop:
	CMPQ AX, $32
	JGE  fwd_do_fft_row
	MOVBLZX (R15)(AX*1), BX
	MOVQ BX, CX
	SHLQ $4, CX               // rev*16
	ADDQ DX, CX
	VMOVUPD (R11)(CX*1), X0
	VMOVUPD X0, (R12)(AX*16)
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
	VMOVUPD (R12)(R13*16), X0
	MOVQ R13, BX
	SHLQ $9, BX               // k1*512
	MOVQ R14, CX
	SHLQ $4, CX               // k2*16
	ADDQ CX, BX
	VMOVUPD X0, (R8)(BX*1)
	INCQ R13
	JMP  fwd_store_dst_loop

fwd_stage2_next_k2:
	INCQ R14
	JMP  fwd_stage2_k2_loop

	// =======================================================================
	// Local helper: iterative radix-2 DIT FFT-32 on bufV (forward), 2 columns packed
	// Expects:
	//   R12 = &bufV[0] (32 YMM values, each holds 2 complex128 lanes)
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
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	VMOVAPD Y0, Y2
	VADDPD Y1, Y2, Y2
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y2, (SI)
	VMOVUPD Y0, (DI)
	ADDQ $2, AX
	JMP  fwdx4_s1_kloop

fwdx4_s2:
	// Stage m=4: tw offset = j*4096 bytes (W_32^{j*8})
	XORQ CX, CX
fwdx4_s2_jloop:
	CMPQ CX, $2
	JGE  fwdx4_s3
	CMPQ CX, $0
	JEQ  fwdx4_s2_kloop_init
	MOVQ CX, DX
	SHLQ $12, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSD 0(R15), Y8
	VBROADCASTSD 8(R15), Y9

fwdx4_s2_kloop_init:
	MOVQ CX, AX
fwdx4_s2_kloop:
	CMPQ AX, $32
	JGE  fwdx4_s2_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 64(SI), DI
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	CMPQ CX, $0
	JEQ  fwdx4_s2_bfly
	// Y1 *= (Y8 + i*Y9)
	VMOVAPD Y1, Y2
	VPERMILPD $0x05, Y2, Y2
	VMULPD Y9, Y2, Y2
	VMULPD Y8, Y1, Y1
	VADDSUBPD Y2, Y1, Y1
fwdx4_s2_bfly:
	VMOVAPD Y0, Y3
	VADDPD Y1, Y3, Y3
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y3, (SI)
	VMOVUPD Y0, (DI)
	ADDQ $4, AX
	JMP  fwdx4_s2_kloop
fwdx4_s2_nextj:
	INCQ CX
	JMP  fwdx4_s2_jloop

fwdx4_s3:
	// Stage m=8: tw offset = j*2048 bytes (W_32^{j*4})
	XORQ CX, CX
fwdx4_s3_jloop:
	CMPQ CX, $4
	JGE  fwdx4_s4
	CMPQ CX, $0
	JEQ  fwdx4_s3_kloop_init
	MOVQ CX, DX
	SHLQ $11, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSD 0(R15), Y8
	VBROADCASTSD 8(R15), Y9

fwdx4_s3_kloop_init:
	MOVQ CX, AX
fwdx4_s3_kloop:
	CMPQ AX, $32
	JGE  fwdx4_s3_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 128(SI), DI
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	CMPQ CX, $0
	JEQ  fwdx4_s3_bfly
	VMOVAPD Y1, Y2
	VPERMILPD $0x05, Y2, Y2
	VMULPD Y9, Y2, Y2
	VMULPD Y8, Y1, Y1
	VADDSUBPD Y2, Y1, Y1
fwdx4_s3_bfly:
	VMOVAPD Y0, Y3
	VADDPD Y1, Y3, Y3
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y3, (SI)
	VMOVUPD Y0, (DI)
	ADDQ $8, AX
	JMP  fwdx4_s3_kloop
fwdx4_s3_nextj:
	INCQ CX
	JMP  fwdx4_s3_jloop

fwdx4_s4:
	// Stage m=16: tw offset = j*1024 bytes (W_32^{j*2})
	XORQ CX, CX
fwdx4_s4_jloop:
	CMPQ CX, $8
	JGE  fwdx4_s5
	CMPQ CX, $0
	JEQ  fwdx4_s4_kloop_init
	MOVQ CX, DX
	SHLQ $10, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSD 0(R15), Y8
	VBROADCASTSD 8(R15), Y9

fwdx4_s4_kloop_init:
	MOVQ CX, AX
fwdx4_s4_kloop:
	CMPQ AX, $32
	JGE  fwdx4_s4_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 256(SI), DI
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	CMPQ CX, $0
	JEQ  fwdx4_s4_bfly
	VMOVAPD Y1, Y2
	VPERMILPD $0x05, Y2, Y2
	VMULPD Y9, Y2, Y2
	VMULPD Y8, Y1, Y1
	VADDSUBPD Y2, Y1, Y1
fwdx4_s4_bfly:
	VMOVAPD Y0, Y3
	VADDPD Y1, Y3, Y3
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y3, (SI)
	VMOVUPD Y0, (DI)
	ADDQ $16, AX
	JMP  fwdx4_s4_kloop
fwdx4_s4_nextj:
	INCQ CX
	JMP  fwdx4_s4_jloop

fwdx4_s5:
	// Stage m=32: tw offset = j*512 bytes (W_32^{j})
	XORQ CX, CX
fwdx4_s5_jloop:
	CMPQ CX, $16
	JGE  fwd_fft32x4_ret
	CMPQ CX, $0
	JEQ  fwdx4_s5_do
	MOVQ CX, DX
	SHLQ $9, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSD 0(R15), Y8
	VBROADCASTSD 8(R15), Y9

fwdx4_s5_do:
	MOVQ CX, AX
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 512(SI), DI
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	CMPQ CX, $0
	JEQ  fwdx4_s5_bfly
	VMOVAPD Y1, Y2
	VPERMILPD $0x05, Y2, Y2
	VMULPD Y9, Y2, Y2
	VMULPD Y8, Y1, Y1
	VADDSUBPD Y2, Y1, Y1
fwdx4_s5_bfly:
	VMOVAPD Y0, Y3
	VADDPD Y1, Y3, Y3
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y3, (SI)
	VMOVUPD Y0, (DI)
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
	LEAQ (R12)(AX*16), SI
	LEAQ 16(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	ADDSD X2, X0
	ADDSD X3, X1
	MOVSD 0(SI), X4
	MOVSD 8(SI), X5
	SUBSD X2, X4
	SUBSD X3, X5
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X4, 0(DI)
	MOVSD X5, 8(DI)
	ADDQ $2, AX
	JMP  fwd_s1_kloop

fwd_s2:
	// Stage m=4: tw offset = j*4096 bytes (W_32^{j*8})
	XORQ CX, CX
fwd_s2_jloop:
	CMPQ CX, $2
	JGE  fwd_s3
	MOVQ CX, DX
	SHLQ $12, DX
	LEAQ (R10)(DX*1), R15
	MOVSD 0(R15), X8
	MOVSD 8(R15), X9
	MOVQ CX, AX
fwd_s2_kloop:
	CMPQ AX, $32
	JGE  fwd_s2_nextj
	LEAQ (R12)(AX*16), SI
	LEAQ 32(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	MOVSD X2, X4
	MULSD X8, X4
	MOVSD X3, X5
	MULSD X9, X5
	SUBSD X5, X4
	MOVSD X2, X6
	MULSD X9, X6
	MOVSD X3, X7
	MULSD X8, X7
	ADDSD X7, X6
	ADDSD X4, X0
	ADDSD X6, X1
	MOVSD 0(SI), X2
	MOVSD 8(SI), X3
	SUBSD X4, X2
	SUBSD X6, X3
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 0(DI)
	MOVSD X3, 8(DI)
	ADDQ $4, AX
	JMP  fwd_s2_kloop
fwd_s2_nextj:
	INCQ CX
	JMP  fwd_s2_jloop

fwd_s3:
	// Stage m=8: tw offset = j*2048 bytes (W_32^{j*4})
	XORQ CX, CX
fwd_s3_jloop:
	CMPQ CX, $4
	JGE  fwd_s4
	MOVQ CX, DX
	SHLQ $11, DX
	LEAQ (R10)(DX*1), R15
	MOVSD 0(R15), X8
	MOVSD 8(R15), X9
	MOVQ CX, AX
fwd_s3_kloop:
	CMPQ AX, $32
	JGE  fwd_s3_nextj
	LEAQ (R12)(AX*16), SI
	LEAQ 64(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	MOVSD X2, X4
	MULSD X8, X4
	MOVSD X3, X5
	MULSD X9, X5
	SUBSD X5, X4
	MOVSD X2, X6
	MULSD X9, X6
	MOVSD X3, X7
	MULSD X8, X7
	ADDSD X7, X6
	ADDSD X4, X0
	ADDSD X6, X1
	MOVSD 0(SI), X2
	MOVSD 8(SI), X3
	SUBSD X4, X2
	SUBSD X6, X3
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 0(DI)
	MOVSD X3, 8(DI)
	ADDQ $8, AX
	JMP  fwd_s3_kloop
fwd_s3_nextj:
	INCQ CX
	JMP  fwd_s3_jloop

fwd_s4:
	// Stage m=16: tw offset = j*1024 bytes (W_32^{j*2})
	XORQ CX, CX
fwd_s4_jloop:
	CMPQ CX, $8
	JGE  fwd_s5
	MOVQ CX, DX
	SHLQ $10, DX
	LEAQ (R10)(DX*1), R15
	MOVSD 0(R15), X8
	MOVSD 8(R15), X9
	MOVQ CX, AX
fwd_s4_kloop:
	CMPQ AX, $32
	JGE  fwd_s4_nextj
	LEAQ (R12)(AX*16), SI
	LEAQ 128(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	MOVSD X2, X4
	MULSD X8, X4
	MOVSD X3, X5
	MULSD X9, X5
	SUBSD X5, X4
	MOVSD X2, X6
	MULSD X9, X6
	MOVSD X3, X7
	MULSD X8, X7
	ADDSD X7, X6
	ADDSD X4, X0
	ADDSD X6, X1
	MOVSD 0(SI), X2
	MOVSD 8(SI), X3
	SUBSD X4, X2
	SUBSD X6, X3
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 0(DI)
	MOVSD X3, 8(DI)
	ADDQ $16, AX
	JMP  fwd_s4_kloop
fwd_s4_nextj:
	INCQ CX
	JMP  fwd_s4_jloop

fwd_s5:
	// Stage m=32: tw offset = j*512 bytes (W_32^{j})
	XORQ CX, CX
fwd_s5_jloop:
	CMPQ CX, $16
	JGE  fwd_fft32_ret
	MOVQ CX, DX
	SHLQ $9, DX
	LEAQ (R10)(DX*1), R15
	MOVSD 0(R15), X8
	MOVSD 8(R15), X9
	MOVQ CX, AX
	LEAQ (R12)(AX*16), SI
	LEAQ 256(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	MOVSD X2, X4
	MULSD X8, X4
	MOVSD X3, X5
	MULSD X9, X5
	SUBSD X5, X4
	MOVSD X2, X6
	MULSD X9, X6
	MOVSD X3, X7
	MULSD X8, X7
	ADDSD X7, X6
	ADDSD X4, X0
	ADDSD X6, X1
	MOVSD 0(SI), X2
	MOVSD 8(SI), X3
	SUBSD X4, X2
	SUBSD X6, X3
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 0(DI)
	MOVSD X3, 8(DI)
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
// InverseAVX2Size1024Radix32x32Complex128Asm
// Signature: func(dst, src, twiddle, scratch []complex128) bool
// ---------------------------------------------------------------------------
TEXT ·InverseAVX2Size1024Radix32x32Complex128Asm(SB), $1536-97
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
	LEAQ 0(SP), R12          // local bufV[32]YMM (2 columns packed)
	MOVSD ·scale1024f64<>(SB), X14

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
	SHLQ $9, CX               // rev*512
	MOVQ R14, DX
	SHLQ $4, DX               // k2*16
	ADDQ DX, CX
	VMOVUPD (R9)(CX*1), X0
	VMOVUPD X0, (R12)(AX*16)
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

	LEAQ (R12)(R13*16), SI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1

	MOVQ R14, AX
	IMULQ R13, AX
	SHLQ $4, AX
	LEAQ (R10)(AX*1), DI
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3

	// (a+bi)*(c-di)
	MOVSD X0, X4
	MULSD X2, X4
	MOVSD X1, X5
	MULSD X3, X5
	ADDSD X5, X4              // real
	MOVSD X1, X6
	MULSD X2, X6
	MOVSD X0, X7
	MULSD X3, X7
	SUBSD X7, X6              // imag

	MOVQ R14, AX
	SHLQ $9, AX               // k2*512
	MOVQ R13, BX
	SHLQ $4, BX               // n1*16
	ADDQ BX, AX
	LEAQ (R11)(AX*1), DI
	MOVSD X4, 0(DI)
	MOVSD X6, 8(DI)

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
	XORQ R13, R13 // n1=0 (processed in blocks of 2)
inv_stage2_n1_loop:
	CMPQ R13, $32
	JGE  inv_ok

	// gather 2 columns work[rev*32+n1..n1+1] into bufV
	LEAQ ·bitrev32<>(SB), R15
	XORQ AX, AX // i=0
inv_gather_col_loop:
	CMPQ AX, $32
	JGE  inv_do_ifft_col
	MOVBLZX (R15)(AX*1), BX
	MOVQ BX, CX
	SHLQ $9, CX               // rev*512
	MOVQ R13, DX
	SHLQ $4, DX               // n1*16
	ADDQ DX, CX
	LEAQ (R11)(CX*1), SI
	VMOVUPD (SI), Y0
	MOVQ AX, DI
	SHLQ $5, DI
	VMOVUPD Y0, (R12)(DI*1)
	INCQ AX
	JMP  inv_gather_col_loop

inv_do_ifft_col:
	JMP  inv_fft32x4
inv_fft32x4_return_col:

inv_fft32_return_col:

	VBROADCASTSD ·scale1024f64<>(SB), Y15
	XORQ R14, R14 // n2=0
inv_store_dst_loop:
	CMPQ R14, $32
	JGE  inv_stage2_next_n1

	MOVQ R14, DX
	SHLQ $5, DX
	VMOVUPD (R12)(DX*1), Y0
	VMULPD Y15, Y0, Y0
	MOVQ R14, AX
	SHLQ $9, AX               // n2*512
	MOVQ R13, BX
	SHLQ $4, BX               // n1*16
	ADDQ BX, AX
	LEAQ (R8)(AX*1), DI
	VMOVUPD Y0, (DI)

	INCQ R14
	JMP  inv_store_dst_loop

inv_stage2_next_n1:
	ADDQ $2, R13
	JMP  inv_stage2_n1_loop

	// =======================================================================
	// Local helper: iterative radix-2 DIT IFFT-32 on bufV (unscaled), 2 columns packed
	// Expects:
	//   R12 = &bufV[0]
	//   R10 = twiddle base (W_1024)
	// Returns to inv_fft32x4_return_col
	// =======================================================================

inv_fft32x4:
	VBROADCASTSD ·signMaskF64<>(SB), Y14
	// Stage m=2
	XORQ AX, AX
invx4_s1_kloop:
	CMPQ AX, $32
	JGE  invx4_s2
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 32(SI), DI
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	VMOVAPD Y0, Y2
	VADDPD Y1, Y2, Y2
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y2, (SI)
	VMOVUPD Y0, (DI)
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
	SHLQ $12, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSD 0(R15), Y8
	VBROADCASTSD 8(R15), Y9
	VXORPD Y14, Y9, Y9
invx4_s2_kloop_init:
	MOVQ CX, AX
invx4_s2_kloop:
	CMPQ AX, $32
	JGE  invx4_s2_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 64(SI), DI
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	CMPQ CX, $0
	JEQ  invx4_s2_bfly
	VMOVAPD Y1, Y2
	VPERMILPD $0x05, Y2, Y2
	VMULPD Y9, Y2, Y2
	VMULPD Y8, Y1, Y1
	VADDSUBPD Y2, Y1, Y1
invx4_s2_bfly:
	VMOVAPD Y0, Y3
	VADDPD Y1, Y3, Y3
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y3, (SI)
	VMOVUPD Y0, (DI)
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
	SHLQ $11, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSD 0(R15), Y8
	VBROADCASTSD 8(R15), Y9
	VXORPD Y14, Y9, Y9
invx4_s3_kloop_init:
	MOVQ CX, AX
invx4_s3_kloop:
	CMPQ AX, $32
	JGE  invx4_s3_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 128(SI), DI
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	CMPQ CX, $0
	JEQ  invx4_s3_bfly
	VMOVAPD Y1, Y2
	VPERMILPD $0x05, Y2, Y2
	VMULPD Y9, Y2, Y2
	VMULPD Y8, Y1, Y1
	VADDSUBPD Y2, Y1, Y1
invx4_s3_bfly:
	VMOVAPD Y0, Y3
	VADDPD Y1, Y3, Y3
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y3, (SI)
	VMOVUPD Y0, (DI)
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
	SHLQ $10, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSD 0(R15), Y8
	VBROADCASTSD 8(R15), Y9
	VXORPD Y14, Y9, Y9
invx4_s4_kloop_init:
	MOVQ CX, AX
invx4_s4_kloop:
	CMPQ AX, $32
	JGE  invx4_s4_nextj
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 256(SI), DI
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	CMPQ CX, $0
	JEQ  invx4_s4_bfly
	VMOVAPD Y1, Y2
	VPERMILPD $0x05, Y2, Y2
	VMULPD Y9, Y2, Y2
	VMULPD Y8, Y1, Y1
	VADDSUBPD Y2, Y1, Y1
invx4_s4_bfly:
	VMOVAPD Y0, Y3
	VADDPD Y1, Y3, Y3
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y3, (SI)
	VMOVUPD Y0, (DI)
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
	SHLQ $9, DX
	LEAQ (R10)(DX*1), R15
	VBROADCASTSD 0(R15), Y8
	VBROADCASTSD 8(R15), Y9
	VXORPD Y14, Y9, Y9
invx4_s5_do:
	MOVQ CX, AX
	MOVQ AX, DX
	SHLQ $5, DX
	LEAQ (R12)(DX*1), SI
	LEAQ 512(SI), DI
	VMOVUPD (SI), Y0
	VMOVUPD (DI), Y1
	CMPQ CX, $0
	JEQ  invx4_s5_bfly
	VMOVAPD Y1, Y2
	VPERMILPD $0x05, Y2, Y2
	VMULPD Y9, Y2, Y2
	VMULPD Y8, Y1, Y1
	VADDSUBPD Y2, Y1, Y1
invx4_s5_bfly:
	VMOVAPD Y0, Y3
	VADDPD Y1, Y3, Y3
	VSUBPD Y1, Y0, Y0
	VMOVUPD Y3, (SI)
	VMOVUPD Y0, (DI)
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
	LEAQ (R12)(AX*16), SI
	LEAQ 16(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	ADDSD X2, X0
	ADDSD X3, X1
	MOVSD 0(SI), X4
	MOVSD 8(SI), X5
	SUBSD X2, X4
	SUBSD X3, X5
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X4, 0(DI)
	MOVSD X5, 8(DI)
	ADDQ $2, AX
	JMP  inv_s1_kloop

inv_s2:
	// Stage m=4 (conj twiddle)
	XORQ CX, CX
inv_s2_jloop:
	CMPQ CX, $2
	JGE  inv_s3
	MOVQ CX, DX
	SHLQ $12, DX
	LEAQ (R10)(DX*1), R15
	MOVSD 0(R15), X8
	MOVSD 8(R15), X9
	MOVQ CX, AX
inv_s2_kloop:
	CMPQ AX, $32
	JGE  inv_s2_nextj
	LEAQ (R12)(AX*16), SI
	LEAQ 32(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	MOVSD X2, X4
	MULSD X8, X4
	MOVSD X3, X5
	MULSD X9, X5
	ADDSD X5, X4
	MOVSD X3, X6
	MULSD X8, X6
	MOVSD X2, X7
	MULSD X9, X7
	SUBSD X7, X6
	ADDSD X4, X0
	ADDSD X6, X1
	MOVSD 0(SI), X2
	MOVSD 8(SI), X3
	SUBSD X4, X2
	SUBSD X6, X3
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 0(DI)
	MOVSD X3, 8(DI)
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
	SHLQ $11, DX
	LEAQ (R10)(DX*1), R15
	MOVSD 0(R15), X8
	MOVSD 8(R15), X9
	MOVQ CX, AX
inv_s3_kloop:
	CMPQ AX, $32
	JGE  inv_s3_nextj
	LEAQ (R12)(AX*16), SI
	LEAQ 64(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	MOVSD X2, X4
	MULSD X8, X4
	MOVSD X3, X5
	MULSD X9, X5
	ADDSD X5, X4
	MOVSD X3, X6
	MULSD X8, X6
	MOVSD X2, X7
	MULSD X9, X7
	SUBSD X7, X6
	ADDSD X4, X0
	ADDSD X6, X1
	MOVSD 0(SI), X2
	MOVSD 8(SI), X3
	SUBSD X4, X2
	SUBSD X6, X3
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 0(DI)
	MOVSD X3, 8(DI)
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
	SHLQ $10, DX
	LEAQ (R10)(DX*1), R15
	MOVSD 0(R15), X8
	MOVSD 8(R15), X9
	MOVQ CX, AX
inv_s4_kloop:
	CMPQ AX, $32
	JGE  inv_s4_nextj
	LEAQ (R12)(AX*16), SI
	LEAQ 128(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	MOVSD X2, X4
	MULSD X8, X4
	MOVSD X3, X5
	MULSD X9, X5
	ADDSD X5, X4
	MOVSD X3, X6
	MULSD X8, X6
	MOVSD X2, X7
	MULSD X9, X7
	SUBSD X7, X6
	ADDSD X4, X0
	ADDSD X6, X1
	MOVSD 0(SI), X2
	MOVSD 8(SI), X3
	SUBSD X4, X2
	SUBSD X6, X3
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 0(DI)
	MOVSD X3, 8(DI)
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
	SHLQ $9, DX
	LEAQ (R10)(DX*1), R15
	MOVSD 0(R15), X8
	MOVSD 8(R15), X9
	MOVQ CX, AX
	LEAQ (R12)(AX*16), SI
	LEAQ 256(SI), DI
	MOVSD 0(SI), X0
	MOVSD 8(SI), X1
	MOVSD 0(DI), X2
	MOVSD 8(DI), X3
	MOVSD X2, X4
	MULSD X8, X4
	MOVSD X3, X5
	MULSD X9, X5
	ADDSD X5, X4
	MOVSD X3, X6
	MULSD X8, X6
	MOVSD X2, X7
	MULSD X9, X7
	SUBSD X7, X6
	ADDSD X4, X0
	ADDSD X6, X1
	MOVSD 0(SI), X2
	MOVSD 8(SI), X3
	SUBSD X4, X2
	SUBSD X6, X3
	MOVSD X0, 0(SI)
	MOVSD X1, 8(SI)
	MOVSD X2, 0(DI)
	MOVSD X3, 8(DI)
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
