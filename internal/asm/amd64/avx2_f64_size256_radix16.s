//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-256 Radix-16 FFT Kernels for AMD64 (complex128)
// ===========================================================================
// Correctness-first implementation that mirrors the 16x16 factorization:
// 1. Transpose input to column-major (scratch).
// 2. Stage 1: 16 FFT-16 passes over columns.
// 3. Twiddle multiply W_256^(row*col).
// 4. Transpose back to row-major (dst).
// 5. Stage 2: 16 FFT-16 passes over rows.
// 6. Final transpose to natural order and copy to dst.
// ===========================================================================

#include "textflag.h"

// Forward transform, size 256, complex128, radix-16 variant
TEXT ·ForwardAVX2Size256Radix16Complex128Asm(SB), NOSPLIT, $256-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Verify n == 256
	CMPQ R13, $256
	JNE  fwd_r16_256_return_false

	// Validate all slice lengths >= 256
	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   fwd_r16_256_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   fwd_r16_256_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   fwd_r16_256_return_false

	// ==================================================================
	// STEP 0: Build W_16 table in local stack (from W_256 with stride 16)
	// ==================================================================
	LEAQ 0(SP), R15
	XORQ CX, CX

fwd_r16_w16_loop:
	MOVQ CX, AX
	SHLQ $8, AX              // AX = k * 256 bytes
	VMOVUPD (R10)(AX*1), X0  // W_256[k*16]
	MOVQ CX, DX
	SHLQ $4, DX              // DX = k * 16 bytes
	VMOVUPD X0, (R15)(DX*1)
	INCQ CX
	CMPQ CX, $16
	JL   fwd_r16_w16_loop

	// ==================================================================
	// STEP 1: Transpose input (row-major) -> scratch (column-major)
	// ==================================================================
	XORQ R12, R12             // row

fwd_r16_transpose_in_row:
	CMPQ R12, $16
	JGE  fwd_r16_stage1
	MOVQ R12, R13
	SHLQ $4, R13              // row*16
	XORQ R14, R14             // col

fwd_r16_transpose_in_col:
	CMPQ R14, $16
	JGE  fwd_r16_transpose_in_row_next
	MOVQ R13, AX
	ADDQ R14, AX              // idx = row*16 + col
	MOVQ AX, BX
	SHLQ $4, BX               // byte offset
	VMOVUPD (R9)(BX*1), X0
	MOVQ R14, CX
	SHLQ $4, CX               // col*16
	ADDQ R12, CX              // col*16 + row
	SHLQ $4, CX               // byte offset
	VMOVUPD X0, (R11)(CX*1)
	INCQ R14
	JMP  fwd_r16_transpose_in_col

fwd_r16_transpose_in_row_next:
	INCQ R12
	JMP  fwd_r16_transpose_in_row

fwd_r16_stage1:
	// ==================================================================
	// STEP 2: Stage 1 - FFT-16 on each column (contiguous blocks)
	// ==================================================================
	XORQ R12, R12             // col

fwd_r16_stage1_col:
	CMPQ R12, $16
	JGE  fwd_r16_twiddle
	MOVQ R12, AX
	SHLQ $8, AX               // col*256 bytes
	LEAQ (R11)(AX*1), SI
	MOVQ R12, R14             // save col
	MOVQ $2, R12              // size
	MOVQ $1, R13              // half
	MOVQ $8, R11              // step

	// Bit-reversal permutation for size-16 (radix-2)
	// Swaps: (1,8), (2,4), (3,12), (5,10), (7,14), (11,13)
	VMOVUPD 16(SI), X0
	VMOVUPD 128(SI), X1
	VMOVUPD X1, 16(SI)
	VMOVUPD X0, 128(SI)

	VMOVUPD 32(SI), X0
	VMOVUPD 64(SI), X1
	VMOVUPD X1, 32(SI)
	VMOVUPD X0, 64(SI)

	VMOVUPD 48(SI), X0
	VMOVUPD 192(SI), X1
	VMOVUPD X1, 48(SI)
	VMOVUPD X0, 192(SI)

	VMOVUPD 80(SI), X0
	VMOVUPD 160(SI), X1
	VMOVUPD X1, 80(SI)
	VMOVUPD X0, 160(SI)

	VMOVUPD 112(SI), X0
	VMOVUPD 224(SI), X1
	VMOVUPD X1, 112(SI)
	VMOVUPD X0, 224(SI)

	VMOVUPD 176(SI), X0
	VMOVUPD 208(SI), X1
	VMOVUPD X1, 176(SI)
	VMOVUPD X0, 208(SI)

fwd_r16_fft16_s1_stage_loop:
	CMPQ R12, $17
	JGE  fwd_r16_fft16_s1_done
	XORQ DX, DX               // j

fwd_r16_fft16_s1_j_loop:
	CMPQ DX, R13
	JGE  fwd_r16_fft16_s1_next_stage
	MOVQ DX, AX
	IMULQ R11, AX             // j*step
	SHLQ $4, AX               // twiddle byte offset
	LEAQ (R15)(AX*1), DI      // twiddle pointer
	MOVQ DX, R9              // k = j

fwd_r16_fft16_s1_k_loop:
	CMPQ R9, $16
	JGE  fwd_r16_fft16_s1_j_next
	// a = data[k]
	MOVQ R9, AX
	SHLQ $4, AX
	LEAQ (SI)(AX*1), BX
	VMOVUPD (BX), X0
	// b = data[k+half]
	MOVQ R9, AX
	ADDQ R13, AX
	SHLQ $4, AX
	LEAQ (SI)(AX*1), CX
	VMOVUPD (CX), X1
	// twiddle
	VMOVUPD (DI), X2
	// t = b * twiddle
	VPERMILPD $0, X2, X3
	VPERMILPD $3, X2, X4
	VMULPD X3, X1, X5
	VSHUFPD $1, X1, X1, X6
	VMULPD X4, X6, X6
	VADDSUBPD X6, X5, X5
	// a+t, a-t
	VADDPD X5, X0, X7
	VSUBPD X5, X0, X8
	VMOVUPD X7, (BX)
	VMOVUPD X8, (CX)
	ADDQ R12, R9
	JMP  fwd_r16_fft16_s1_k_loop

fwd_r16_fft16_s1_j_next:
	INCQ DX
	JMP  fwd_r16_fft16_s1_j_loop

fwd_r16_fft16_s1_next_stage:
	SHLQ $1, R12              // size *= 2
	SHLQ $1, R13              // half *= 2
	SHRQ $1, R11              // step /= 2
	JMP  fwd_r16_fft16_s1_stage_loop

fwd_r16_fft16_s1_done:
	MOVQ R14, R12             // restore col
	MOVQ scratch+72(FP), R11  // restore scratch pointer
	INCQ R12
	JMP  fwd_r16_stage1_col

fwd_r16_twiddle:
	// ==================================================================
	// STEP 3: Twiddle multiply W_256^(row*col)
	// ==================================================================
	MOVQ $1, R12              // col

fwd_r16_twiddle_col:
	CMPQ R12, $16
	JGE  fwd_r16_transpose_out
	MOVQ R12, AX
	SHLQ $8, AX
	LEAQ (R11)(AX*1), SI
	XORQ R13, R13             // row

fwd_r16_twiddle_row:
	CMPQ R13, $16
	JGE  fwd_r16_twiddle_next_col
	MOVQ R13, DX
	SHLQ $4, DX               // row*16 bytes
	VMOVUPD (SI)(DX*1), X0
	MOVQ R13, AX
	IMULQ R12, AX             // row*col
	SHLQ $4, AX               // byte offset
	VMOVUPD (R10)(AX*1), X1   // twiddle
	// X0 = X0 * X1
	VPERMILPD $0, X1, X2      // re
	VPERMILPD $3, X1, X3      // im
	VMULPD X2, X0, X4         // b*re
	VSHUFPD $1, X0, X0, X5    // swap b
	VMULPD X3, X5, X5         // swap(b)*im
	VADDSUBPD X5, X4, X4      // complex multiply
	VMOVUPD X4, (SI)(DX*1)
	INCQ R13
	JMP  fwd_r16_twiddle_row

fwd_r16_twiddle_next_col:
	INCQ R12
	JMP  fwd_r16_twiddle_col

fwd_r16_transpose_out:
	// ==================================================================
	// STEP 4: Transpose scratch (column-major) -> dst (row-major)
	// ==================================================================
	XORQ R12, R12             // row

fwd_r16_transpose_out_row:
	CMPQ R12, $16
	JGE  fwd_r16_stage2
	MOVQ R12, R13
	SHLQ $4, R13              // row*16
	XORQ R14, R14             // col

fwd_r16_transpose_out_col:
	CMPQ R14, $16
	JGE  fwd_r16_transpose_out_row_next
	MOVQ R14, AX
	SHLQ $4, AX               // col*16
	ADDQ R12, AX              // col*16 + row
	SHLQ $4, AX               // byte offset
	VMOVUPD (R11)(AX*1), X0
	MOVQ R13, BX
	ADDQ R14, BX              // idx = row*16 + col
	SHLQ $4, BX
	VMOVUPD X0, (R8)(BX*1)
	INCQ R14
	JMP  fwd_r16_transpose_out_col

fwd_r16_transpose_out_row_next:
	INCQ R12
	JMP  fwd_r16_transpose_out_row

fwd_r16_stage2:
	// ==================================================================
	// STEP 5: Stage 2 - FFT-16 on each row (contiguous blocks)
	// ==================================================================
	XORQ R12, R12             // row

fwd_r16_stage2_row:
	CMPQ R12, $16
	JGE  fwd_r16_final_transpose
	MOVQ R12, AX
	SHLQ $8, AX               // row*256 bytes
	LEAQ (R8)(AX*1), SI
	MOVQ R12, R14             // save row
	MOVQ $2, R12              // size
	MOVQ $1, R13              // half
	MOVQ $8, R11              // step

	// Bit-reversal permutation for size-16 (radix-2)
	// Swaps: (1,8), (2,4), (3,12), (5,10), (7,14), (11,13)
	VMOVUPD 16(SI), X0
	VMOVUPD 128(SI), X1
	VMOVUPD X1, 16(SI)
	VMOVUPD X0, 128(SI)

	VMOVUPD 32(SI), X0
	VMOVUPD 64(SI), X1
	VMOVUPD X1, 32(SI)
	VMOVUPD X0, 64(SI)

	VMOVUPD 48(SI), X0
	VMOVUPD 192(SI), X1
	VMOVUPD X1, 48(SI)
	VMOVUPD X0, 192(SI)

	VMOVUPD 80(SI), X0
	VMOVUPD 160(SI), X1
	VMOVUPD X1, 80(SI)
	VMOVUPD X0, 160(SI)

	VMOVUPD 112(SI), X0
	VMOVUPD 224(SI), X1
	VMOVUPD X1, 112(SI)
	VMOVUPD X0, 224(SI)

	VMOVUPD 176(SI), X0
	VMOVUPD 208(SI), X1
	VMOVUPD X1, 176(SI)
	VMOVUPD X0, 208(SI)

fwd_r16_fft16_s2_stage_loop:
	CMPQ R12, $17
	JGE  fwd_r16_fft16_s2_done
	XORQ DX, DX               // j

fwd_r16_fft16_s2_j_loop:
	CMPQ DX, R13
	JGE  fwd_r16_fft16_s2_next_stage
	MOVQ DX, AX
	IMULQ R11, AX             // j*step
	SHLQ $4, AX               // twiddle byte offset
	LEAQ (R15)(AX*1), DI      // twiddle pointer
	MOVQ DX, R9              // k = j

fwd_r16_fft16_s2_k_loop:
	CMPQ R9, $16
	JGE  fwd_r16_fft16_s2_j_next
	// a = data[k]
	MOVQ R9, AX
	SHLQ $4, AX
	LEAQ (SI)(AX*1), BX
	VMOVUPD (BX), X0
	// b = data[k+half]
	MOVQ R9, AX
	ADDQ R13, AX
	SHLQ $4, AX
	LEAQ (SI)(AX*1), CX
	VMOVUPD (CX), X1
	// twiddle
	VMOVUPD (DI), X2
	// t = b * twiddle
	VPERMILPD $0, X2, X3
	VPERMILPD $3, X2, X4
	VMULPD X3, X1, X5
	VSHUFPD $1, X1, X1, X6
	VMULPD X4, X6, X6
	VADDSUBPD X6, X5, X5
	// a+t, a-t
	VADDPD X5, X0, X7
	VSUBPD X5, X0, X8
	VMOVUPD X7, (BX)
	VMOVUPD X8, (CX)
	ADDQ R12, R9
	JMP  fwd_r16_fft16_s2_k_loop

fwd_r16_fft16_s2_j_next:
	INCQ DX
	JMP  fwd_r16_fft16_s2_j_loop

fwd_r16_fft16_s2_next_stage:
	SHLQ $1, R12              // size *= 2
	SHLQ $1, R13              // half *= 2
	SHRQ $1, R11              // step /= 2
	JMP  fwd_r16_fft16_s2_stage_loop

fwd_r16_fft16_s2_done:
	MOVQ R14, R12             // restore row
	MOVQ scratch+72(FP), R11  // restore scratch pointer
	INCQ R12
	JMP  fwd_r16_stage2_row

fwd_r16_final_transpose:
	// ==================================================================
	// STEP 6: Final transpose (dst -> scratch) and copy to dst
	// ==================================================================
	XORQ R12, R12             // row

fwd_r16_final_row:
	CMPQ R12, $16
	JGE  fwd_r16_final_copy
	MOVQ R12, R13
	SHLQ $4, R13              // row*16
	XORQ R14, R14             // col

fwd_r16_final_col:
	CMPQ R14, $16
	JGE  fwd_r16_final_row_next
	MOVQ R13, AX
	ADDQ R14, AX              // idx = row*16 + col
	SHLQ $4, AX
	VMOVUPD (R8)(AX*1), X0
	MOVQ R14, BX
	SHLQ $4, BX               // col*16
	ADDQ R12, BX              // col*16 + row
	SHLQ $4, BX
	VMOVUPD X0, (R11)(BX*1)
	INCQ R14
	JMP  fwd_r16_final_col

fwd_r16_final_row_next:
	INCQ R12
	JMP  fwd_r16_final_row

fwd_r16_final_copy:
	XORQ CX, CX

fwd_r16_final_copy_loop:
	CMPQ CX, $4096
	JGE  fwd_r16_done
	VMOVUPD (R11)(CX*1), X0
	VMOVUPD X0, (R8)(CX*1)
	ADDQ $16, CX
	JMP  fwd_r16_final_copy_loop

fwd_r16_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

fwd_r16_256_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// Inverse transform, size 256, complex128, radix-16 variant
TEXT ·InverseAVX2Size256Radix16Complex128Asm(SB), NOSPLIT, $256-97
	// Load parameters
	MOVQ dst+0(FP), R8
	MOVQ src+24(FP), R9
	MOVQ twiddle+48(FP), R10
	MOVQ scratch+72(FP), R11
	MOVQ src+32(FP), R13

	// Verify n == 256
	CMPQ R13, $256
	JNE  inv_r16_256_return_false

	// Validate all slice lengths >= 256
	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   inv_r16_256_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   inv_r16_256_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   inv_r16_256_return_false

	// ==================================================================
	// STEP 0: Build conjugate W_16 table in local stack
	// ==================================================================
	LEAQ 0(SP), R15
	XORQ CX, CX

inv_r16_w16_loop:
	MOVQ CX, AX
	SHLQ $8, AX              // AX = k * 256 bytes
	VMOVUPD (R10)(AX*1), X0  // W_256[k*16]
	// conjugate
	VXORPD ·maskNegHiPD(SB), X0, X0
	MOVQ CX, DX
	SHLQ $4, DX              // DX = k * 16 bytes
	VMOVUPD X0, (R15)(DX*1)
	INCQ CX
	CMPQ CX, $16
	JL   inv_r16_w16_loop

	// ==================================================================
	// STEP 1: Transpose input (row-major) -> scratch (column-major)
	// ==================================================================
	XORQ R12, R12             // row

inv_r16_transpose_in_row:
	CMPQ R12, $16
	JGE  inv_r16_stage1
	MOVQ R12, R13
	SHLQ $4, R13              // row*16
	XORQ R14, R14             // col

inv_r16_transpose_in_col:
	CMPQ R14, $16
	JGE  inv_r16_transpose_in_row_next
	MOVQ R13, AX
	ADDQ R14, AX              // idx = row*16 + col
	MOVQ AX, BX
	SHLQ $4, BX               // byte offset
	VMOVUPD (R9)(BX*1), X0
	MOVQ R14, CX
	SHLQ $4, CX               // col*16
	ADDQ R12, CX              // col*16 + row
	SHLQ $4, CX               // byte offset
	VMOVUPD X0, (R11)(CX*1)
	INCQ R14
	JMP  inv_r16_transpose_in_col

inv_r16_transpose_in_row_next:
	INCQ R12
	JMP  inv_r16_transpose_in_row

inv_r16_stage1:
	// ==================================================================
	// STEP 2: Stage 1 - FFT-16 on each column (contiguous blocks)
	// ==================================================================
	XORQ R12, R12             // col

inv_r16_stage1_col:
	CMPQ R12, $16
	JGE  inv_r16_twiddle
	MOVQ R12, AX
	SHLQ $8, AX               // col*256 bytes
	LEAQ (R11)(AX*1), SI
	MOVQ R12, R14             // save col
	MOVQ $2, R12              // size
	MOVQ $1, R13              // half
	MOVQ $8, R11              // step

	// Bit-reversal permutation for size-16 (radix-2)
	// Swaps: (1,8), (2,4), (3,12), (5,10), (7,14), (11,13)
	VMOVUPD 16(SI), X0
	VMOVUPD 128(SI), X1
	VMOVUPD X1, 16(SI)
	VMOVUPD X0, 128(SI)

	VMOVUPD 32(SI), X0
	VMOVUPD 64(SI), X1
	VMOVUPD X1, 32(SI)
	VMOVUPD X0, 64(SI)

	VMOVUPD 48(SI), X0
	VMOVUPD 192(SI), X1
	VMOVUPD X1, 48(SI)
	VMOVUPD X0, 192(SI)

	VMOVUPD 80(SI), X0
	VMOVUPD 160(SI), X1
	VMOVUPD X1, 80(SI)
	VMOVUPD X0, 160(SI)

	VMOVUPD 112(SI), X0
	VMOVUPD 224(SI), X1
	VMOVUPD X1, 112(SI)
	VMOVUPD X0, 224(SI)

	VMOVUPD 176(SI), X0
	VMOVUPD 208(SI), X1
	VMOVUPD X1, 176(SI)
	VMOVUPD X0, 208(SI)

inv_r16_fft16_s1_stage_loop:
	CMPQ R12, $17
	JGE  inv_r16_fft16_s1_done
	XORQ DX, DX               // j

inv_r16_fft16_s1_j_loop:
	CMPQ DX, R13
	JGE  inv_r16_fft16_s1_next_stage
	MOVQ DX, AX
	IMULQ R11, AX             // j*step
	SHLQ $4, AX               // twiddle byte offset
	LEAQ (R15)(AX*1), DI      // twiddle pointer
	MOVQ DX, R9              // k = j

inv_r16_fft16_s1_k_loop:
	CMPQ R9, $16
	JGE  inv_r16_fft16_s1_j_next
	// a = data[k]
	MOVQ R9, AX
	SHLQ $4, AX
	LEAQ (SI)(AX*1), BX
	VMOVUPD (BX), X0
	// b = data[k+half]
	MOVQ R9, AX
	ADDQ R13, AX
	SHLQ $4, AX
	LEAQ (SI)(AX*1), CX
	VMOVUPD (CX), X1
	// twiddle
	VMOVUPD (DI), X2
	// t = b * twiddle
	VPERMILPD $0, X2, X3
	VPERMILPD $3, X2, X4
	VMULPD X3, X1, X5
	VSHUFPD $1, X1, X1, X6
	VMULPD X4, X6, X6
	VADDSUBPD X6, X5, X5
	// a+t, a-t
	VADDPD X5, X0, X7
	VSUBPD X5, X0, X8
	VMOVUPD X7, (BX)
	VMOVUPD X8, (CX)
	ADDQ R12, R9
	JMP  inv_r16_fft16_s1_k_loop

inv_r16_fft16_s1_j_next:
	INCQ DX
	JMP  inv_r16_fft16_s1_j_loop

inv_r16_fft16_s1_next_stage:
	SHLQ $1, R12              // size *= 2
	SHLQ $1, R13              // half *= 2
	SHRQ $1, R11              // step /= 2
	JMP  inv_r16_fft16_s1_stage_loop

inv_r16_fft16_s1_done:
	MOVQ R14, R12             // restore col
	MOVQ scratch+72(FP), R11  // restore scratch pointer
	INCQ R12
	JMP  inv_r16_stage1_col

inv_r16_twiddle:
	// ==================================================================
	// STEP 3: Twiddle multiply W_256^(-row*col)
	// ==================================================================
	MOVQ $1, R12              // col

inv_r16_twiddle_col:
	CMPQ R12, $16
	JGE  inv_r16_transpose_out
	MOVQ R12, AX
	SHLQ $8, AX
	LEAQ (R11)(AX*1), SI
	XORQ R13, R13             // row

inv_r16_twiddle_row:
	CMPQ R13, $16
	JGE  inv_r16_twiddle_next_col
	MOVQ R13, DX
	SHLQ $4, DX               // row*16 bytes
	VMOVUPD (SI)(DX*1), X0
	MOVQ R13, AX
	IMULQ R12, AX             // row*col
	SHLQ $4, AX               // byte offset
	VMOVUPD (R10)(AX*1), X1   // twiddle
	// conjugate twiddle
	VXORPD ·maskNegHiPD(SB), X1, X1
	// X0 = X0 * X1
	VPERMILPD $0, X1, X2
	VPERMILPD $3, X1, X3
	VMULPD X2, X0, X4
	VSHUFPD $1, X0, X0, X5
	VMULPD X3, X5, X5
	VADDSUBPD X5, X4, X4
	VMOVUPD X4, (SI)(DX*1)
	INCQ R13
	JMP  inv_r16_twiddle_row

inv_r16_twiddle_next_col:
	INCQ R12
	JMP  inv_r16_twiddle_col

inv_r16_transpose_out:
	// ==================================================================
	// STEP 4: Transpose scratch (column-major) -> dst (row-major)
	// ==================================================================
	XORQ R12, R12             // row

inv_r16_transpose_out_row:
	CMPQ R12, $16
	JGE  inv_r16_stage2
	MOVQ R12, R13
	SHLQ $4, R13              // row*16
	XORQ R14, R14             // col

inv_r16_transpose_out_col:
	CMPQ R14, $16
	JGE  inv_r16_transpose_out_row_next
	MOVQ R14, AX
	SHLQ $4, AX               // col*16
	ADDQ R12, AX              // col*16 + row
	SHLQ $4, AX               // byte offset
	VMOVUPD (R11)(AX*1), X0
	MOVQ R13, BX
	ADDQ R14, BX              // idx = row*16 + col
	SHLQ $4, BX
	VMOVUPD X0, (R8)(BX*1)
	INCQ R14
	JMP  inv_r16_transpose_out_col

inv_r16_transpose_out_row_next:
	INCQ R12
	JMP  inv_r16_transpose_out_row

inv_r16_stage2:
	// ==================================================================
	// STEP 5: Stage 2 - FFT-16 on each row (contiguous blocks)
	// ==================================================================
	XORQ R12, R12             // row

inv_r16_stage2_row:
	CMPQ R12, $16
	JGE  inv_r16_final_transpose
	MOVQ R12, AX
	SHLQ $8, AX               // row*256 bytes
	LEAQ (R8)(AX*1), SI
	MOVQ R12, R14             // save row
	MOVQ $2, R12              // size
	MOVQ $1, R13              // half
	MOVQ $8, R11              // step

	// Bit-reversal permutation for size-16 (radix-2)
	// Swaps: (1,8), (2,4), (3,12), (5,10), (7,14), (11,13)
	VMOVUPD 16(SI), X0
	VMOVUPD 128(SI), X1
	VMOVUPD X1, 16(SI)
	VMOVUPD X0, 128(SI)

	VMOVUPD 32(SI), X0
	VMOVUPD 64(SI), X1
	VMOVUPD X1, 32(SI)
	VMOVUPD X0, 64(SI)

	VMOVUPD 48(SI), X0
	VMOVUPD 192(SI), X1
	VMOVUPD X1, 48(SI)
	VMOVUPD X0, 192(SI)

	VMOVUPD 80(SI), X0
	VMOVUPD 160(SI), X1
	VMOVUPD X1, 80(SI)
	VMOVUPD X0, 160(SI)

	VMOVUPD 112(SI), X0
	VMOVUPD 224(SI), X1
	VMOVUPD X1, 112(SI)
	VMOVUPD X0, 224(SI)

	VMOVUPD 176(SI), X0
	VMOVUPD 208(SI), X1
	VMOVUPD X1, 176(SI)
	VMOVUPD X0, 208(SI)

inv_r16_fft16_s2_stage_loop:
	CMPQ R12, $17
	JGE  inv_r16_fft16_s2_done
	XORQ DX, DX               // j

inv_r16_fft16_s2_j_loop:
	CMPQ DX, R13
	JGE  inv_r16_fft16_s2_next_stage
	MOVQ DX, AX
	IMULQ R11, AX             // j*step
	SHLQ $4, AX               // twiddle byte offset
	LEAQ (R15)(AX*1), DI      // twiddle pointer
	MOVQ DX, R9              // k = j

inv_r16_fft16_s2_k_loop:
	CMPQ R9, $16
	JGE  inv_r16_fft16_s2_j_next
	// a = data[k]
	MOVQ R9, AX
	SHLQ $4, AX
	LEAQ (SI)(AX*1), BX
	VMOVUPD (BX), X0
	// b = data[k+half]
	MOVQ R9, AX
	ADDQ R13, AX
	SHLQ $4, AX
	LEAQ (SI)(AX*1), CX
	VMOVUPD (CX), X1
	// twiddle
	VMOVUPD (DI), X2
	// t = b * twiddle
	VPERMILPD $0, X2, X3
	VPERMILPD $3, X2, X4
	VMULPD X3, X1, X5
	VSHUFPD $1, X1, X1, X6
	VMULPD X4, X6, X6
	VADDSUBPD X6, X5, X5
	// a+t, a-t
	VADDPD X5, X0, X7
	VSUBPD X5, X0, X8
	VMOVUPD X7, (BX)
	VMOVUPD X8, (CX)
	ADDQ R12, R9
	JMP  inv_r16_fft16_s2_k_loop

inv_r16_fft16_s2_j_next:
	INCQ DX
	JMP  inv_r16_fft16_s2_j_loop

inv_r16_fft16_s2_next_stage:
	SHLQ $1, R12              // size *= 2
	SHLQ $1, R13              // half *= 2
	SHRQ $1, R11              // step /= 2
	JMP  inv_r16_fft16_s2_stage_loop

inv_r16_fft16_s2_done:
	MOVQ R14, R12             // restore row
	MOVQ scratch+72(FP), R11  // restore scratch pointer
	INCQ R12
	JMP  inv_r16_stage2_row

inv_r16_final_transpose:
	// ==================================================================
	// STEP 6: Final transpose (dst -> scratch), scale, and copy to dst
	// ==================================================================
	XORQ R12, R12             // row

inv_r16_final_row:
	CMPQ R12, $16
	JGE  inv_r16_scale_copy
	MOVQ R12, R13
	SHLQ $4, R13              // row*16
	XORQ R14, R14             // col

inv_r16_final_col:
	CMPQ R14, $16
	JGE  inv_r16_final_row_next
	MOVQ R13, AX
	ADDQ R14, AX              // idx = row*16 + col
	SHLQ $4, AX
	VMOVUPD (R8)(AX*1), X0
	MOVQ R14, BX
	SHLQ $4, BX               // col*16
	ADDQ R12, BX              // col*16 + row
	SHLQ $4, BX
	VMOVUPD X0, (R11)(BX*1)
	INCQ R14
	JMP  inv_r16_final_col

inv_r16_final_row_next:
	INCQ R12
	JMP  inv_r16_final_row

inv_r16_scale_copy:
	MOVQ ·twoFiftySixth64(SB), AX
	VMOVQ AX, X8
	VMOVDDUP X8, X8
	XORQ CX, CX

inv_r16_scale_copy_loop:
	CMPQ CX, $4096
	JGE  inv_r16_done
	VMOVUPD (R11)(CX*1), X0
	VMULPD X8, X0, X0
	VMOVUPD X0, (R8)(CX*1)
	ADDQ $16, CX
	JMP  inv_r16_scale_copy_loop

inv_r16_done:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

inv_r16_256_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET
