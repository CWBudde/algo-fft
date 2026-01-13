//go:build amd64 && asm && !purego

// ===========================================================================
// Size-1024 Radix-32×32 FFT Kernel (complex64) — correctness-first
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
// - Uses scalar SSE (MOVSS/ADDSS/MULSS/...) on float32 lanes.
// - “AVX2” naming is for dispatch consistency; we still do VZEROUPPER.
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

// ---------------------------------------------------------------------------
// ForwardAVX2Size1024Radix32x32Complex64Asm
// Signature: func(dst, src, twiddle, scratch []complex64) bool
// ---------------------------------------------------------------------------
TEXT ·ForwardAVX2Size1024Radix32x32Complex64Asm(SB), NOSPLIT, $256-97
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
	LEAQ 0(SP), R12          // local buf[32]complex64

	// =====================================================================
	// Stage 1: for each n1, FFT32 over n2 (column) then apply W_1024^(k2*n1)
	// work[k2*32 + n1]
	// =====================================================================
	XORQ R13, R13 // n1=0
fwd_stage1_n1_loop:
	CMPQ R13, $32
	JGE  fwd_stage2

	// gather src[n2*32+n1] into buf in bit-reversed order over n2
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
	MOVQ (R9)(BX*1), DX
	MOVQ DX, (R12)(AX*8)
	INCQ AX
	JMP  fwd_gather_col_loop

fwd_do_fft_col:
	XORQ BP, BP               // return selector 0
	JMP  fwd_fft32
fwd_fft32_return_col:

	// apply inter-stage twiddle and store to work
	XORQ R14, R14 // k2=0
fwd_store_work_loop:
	CMPQ R14, $32
	JGE  fwd_stage1_next_n1

	LEAQ (R12)(R14*8), SI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1

	MOVQ R14, AX
	IMULQ R13, AX
	SHLQ $3, AX
	LEAQ (R10)(AX*1), DI
	MOVSS 0(DI), X2
	MOVSS 4(DI), X3

	// (a+bi)*(c+di)
	MOVSS X0, X4
	MULSS X2, X4
	MOVSS X1, X5
	MULSS X3, X5
	SUBSS X5, X4              // real
	MOVSS X0, X6
	MULSS X3, X6
	MOVSS X1, X7
	MULSS X2, X7
	ADDSS X7, X6              // imag

	MOVQ R14, AX
	SHLQ $8, AX               // k2*256
	MOVQ R13, BX
	SHLQ $3, BX               // n1*8
	ADDQ BX, AX
	LEAQ (R11)(AX*1), DI
	MOVSS X4, 0(DI)
	MOVSS X6, 4(DI)

	INCQ R14
	JMP  fwd_store_work_loop

fwd_stage1_next_n1:
	INCQ R13
	JMP  fwd_stage1_n1_loop

	// =====================================================================
	// Stage 2: for each k2, FFT32 over n1 (row) and store dst[k1*32+k2]
	// =====================================================================
fwd_stage2:
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
TEXT ·InverseAVX2Size1024Radix32x32Complex64Asm(SB), NOSPLIT, $256-97
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
	LEAQ 0(SP), R12          // local buf[32]complex64
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
	XORQ R13, R13 // n1=0
inv_stage2_n1_loop:
	CMPQ R13, $32
	JGE  inv_ok

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
	MOVQ (R11)(CX*1), SI
	MOVQ SI, (R12)(AX*8)
	INCQ AX
	JMP  inv_gather_col_loop

inv_do_ifft_col:
	MOVQ $1, BP               // return selector 1
	JMP  inv_fft32
inv_fft32_return_col:

	XORQ R14, R14 // n2=0
inv_store_dst_loop:
	CMPQ R14, $32
	JGE  inv_stage2_next_n1

	LEAQ (R12)(R14*8), SI
	MOVSS 0(SI), X0
	MOVSS 4(SI), X1
	MULSS X14, X0
	MULSS X14, X1

	MOVQ R14, AX
	SHLQ $8, AX               // n2*256
	MOVQ R13, BX
	SHLQ $3, BX               // n1*8
	ADDQ BX, AX
	LEAQ (R8)(AX*1), DI
	MOVSS X0, 0(DI)
	MOVSS X1, 4(DI)

	INCQ R14
	JMP  inv_store_dst_loop

inv_stage2_next_n1:
	INCQ R13
	JMP  inv_stage2_n1_loop

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
