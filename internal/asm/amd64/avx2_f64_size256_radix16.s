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
//
// Parallelization notes (future work, not implemented here):
// - Stage 1 (columns) and Stage 2 (rows) are 16 independent FFT-16 passes.
//   These loops can be split across workers by column/row index.
// - Transpose steps are 4x4 tiles; each tile is independent and can be
//   distributed across workers or vectorized further without cross-tile sync.
// - Twiddle multiply is independent per (row, col) cell; it can be fused into
//   the transpose-out step once the mapping is verified.
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
	// rb = row block index (0..3), cb = col block index (0..3)
	// Each block is a 4x4 tile of complex128 values (32 bytes per row).
	XORQ R12, R12             // row block (0..3)

fwd_r16_transpose_in_rb:
	CMPQ R12, $4
	JGE  fwd_r16_stage1
	MOVQ R12, R14
	SHLQ $10, R14             // row_block_bytes = rb * 1024
	MOVQ R12, DX
	SHLQ $6, DX               // row_base_bytes = rb * 64
	XORQ R13, R13             // col block (0..3)

fwd_r16_transpose_in_cb:
	CMPQ R13, $4
	JGE  fwd_r16_transpose_in_rb_next
	MOVQ R13, AX
	SHLQ $6, AX               // col_block_bytes = cb * 64
	LEAQ (R9)(R14*1), SI
	ADDQ AX, SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), BX
	LEAQ 768(SI), CX

	VMOVUPD 0(SI), Y0
	VMOVUPD 32(SI), Y1
	VMOVUPD 0(DI), Y2
	VMOVUPD 32(DI), Y3
	VMOVUPD 0(BX), Y4
	VMOVUPD 32(BX), Y5
	VMOVUPD 0(CX), Y6
	VMOVUPD 32(CX), Y7

	// Transpose 4x4 complex128 tile:
	// rows are contiguous in src; columns become contiguous in scratch.
	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	MOVQ R13, AX
	SHLQ $10, AX              // cb * 1024
	LEAQ (R11)(AX*1), DI
	ADDQ DX, DI               // add row_base_bytes

	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)

	INCQ R13
	JMP  fwd_r16_transpose_in_cb

fwd_r16_transpose_in_rb_next:
	INCQ R12
	JMP  fwd_r16_transpose_in_rb

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
	// Twiddle table prepends base twiddles (0..255), then packed pairs per column:
	//   [re0,re0,re1,re1] then [im0,im0,im1,im1] for rows (2*pair,2*pair+1).
	// Indexing: packed_base = 4096 + (col-1)*512 + pair*64.
	MOVQ $1, R12              // col

fwd_r16_twiddle_col:
	CMPQ R12, $16
	JGE  fwd_r16_transpose_out
	MOVQ R12, AX
	SHLQ $8, AX               // col*256 bytes
	LEAQ (R11)(AX*1), SI
	MOVQ R12, AX
	DECQ AX                   // col-1
	SHLQ $9, AX               // (col-1)*512 bytes (8 pairs * 64 bytes)
	LEAQ 4096(R10)(AX*1), R14 // twiddle base for col (skip base twiddle[0:256])
	XORQ R13, R13             // pair (0..7)

fwd_r16_twiddle_row:
	CMPQ R13, $8
	JGE  fwd_r16_twiddle_next_col
	MOVQ R13, DX
	SHLQ $5, DX               // pair*32 bytes (2 complex128)
	VMOVUPD (SI)(DX*1), Y0

	MOVQ R13, BX
	SHLQ $6, BX               // pair*64 bytes (4 complex128)
	LEAQ (R14)(BX*1), DI
	VMOVUPD (DI), Y1          // reDup
	VMOVUPD 32(DI), Y2        // imDup

	// Complex multiply: Y0 *= twiddle
	VPERMILPD $0x05, Y0, Y3   // [i0, r0, i1, r1]
	VMULPD Y1, Y0, Y4
	VMULPD Y2, Y3, Y5
	VADDSUBPD Y5, Y4, Y0
	VMOVUPD Y0, (SI)(DX*1)

	INCQ R13
	JMP  fwd_r16_twiddle_row

fwd_r16_twiddle_next_col:
	INCQ R12
	JMP  fwd_r16_twiddle_col

fwd_r16_transpose_out:
	// ==================================================================
	// STEP 4: Transpose scratch (column-major) -> dst (row-major)
	// ==================================================================
	// Mirror of transpose-in: tiles are independent 4x4 blocks.
	XORQ R12, R12             // row block (0..3)

fwd_r16_transpose_out_rb:
	CMPQ R12, $4
	JGE  fwd_r16_stage2
	MOVQ R12, R14
	SHLQ $10, R14             // row_block_bytes = rb * 1024
	MOVQ R12, DX
	SHLQ $6, DX               // row_base_bytes = rb * 64
	XORQ R13, R13             // col block (0..3)

fwd_r16_transpose_out_cb:
	CMPQ R13, $4
	JGE  fwd_r16_transpose_out_rb_next
	MOVQ R13, AX
	SHLQ $6, AX               // col_block_bytes = cb * 64
	LEAQ (R11)(R14*1), SI
	ADDQ AX, SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), BX
	LEAQ 768(SI), CX

	VMOVUPD 0(SI), Y0
	VMOVUPD 32(SI), Y1
	VMOVUPD 0(DI), Y2
	VMOVUPD 32(DI), Y3
	VMOVUPD 0(BX), Y4
	VMOVUPD 32(BX), Y5
	VMOVUPD 0(CX), Y6
	VMOVUPD 32(CX), Y7

	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	MOVQ R13, AX
	SHLQ $10, AX              // cb * 1024
	LEAQ (R8)(AX*1), DI
	ADDQ DX, DI               // add row_base_bytes

	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)

	INCQ R13
	JMP  fwd_r16_transpose_out_cb

fwd_r16_transpose_out_rb_next:
	INCQ R12
	JMP  fwd_r16_transpose_out_rb

fwd_r16_stage2:
	// ==================================================================
	// STEP 5: Stage 2 - FFT-16 on each row (contiguous blocks)
	// ==================================================================
	// Rows are contiguous in dst after transpose-out; each row is independent.
	XORQ R12, R12             // row

// ---------------------------------------------------------------------------
// Diagnostic fused-twiddle mapping (disabled; not referenced).
// Use to validate the transpose-out twiddle mapping on a single tile:
// - Assumes scratch base in R11, twiddle base in R10, dst base in R8.
// - Operates on rb=0, cb=0 (rows 0..3, cols 0..3) and returns.
// To enable for debugging, redirect the JGE at fwd_r16_stage1_col to this label.
// ---------------------------------------------------------------------------
fwd_r16_twiddle_fused_diag:
	// Load scratch[0:4,0:4] (column-major tile)
	LEAQ (R11), SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), BX
	LEAQ 768(SI), CX
	VMOVUPD 0(SI), Y0
	VMOVUPD 32(SI), Y1
	VMOVUPD 0(DI), Y2
	VMOVUPD 32(DI), Y3
	VMOVUPD 0(BX), Y4
	VMOVUPD 32(BX), Y5
	VMOVUPD 0(CX), Y6
	VMOVUPD 32(CX), Y7

	// Apply twiddles for col=1..3 (col=0 is identity)
	LEAQ 4096(R10), DI        // col=1 base
	// pair0 (rows 0..1)
	VMOVUPD (DI), Y8
	VMOVUPD 32(DI), Y9
	VPERMILPD $0x05, Y2, Y10
	VMULPD Y8, Y2, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y2
	// pair1 (rows 2..3)
	VMOVUPD 64(DI), Y8
	VMOVUPD 96(DI), Y9
	VPERMILPD $0x05, Y3, Y10
	VMULPD Y8, Y3, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y3

	ADDQ $512, DI             // col=2 base
	VMOVUPD (DI), Y8
	VMOVUPD 32(DI), Y9
	VPERMILPD $0x05, Y4, Y10
	VMULPD Y8, Y4, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y4
	VMOVUPD 64(DI), Y8
	VMOVUPD 96(DI), Y9
	VPERMILPD $0x05, Y5, Y10
	VMULPD Y8, Y5, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y5

	ADDQ $512, DI             // col=3 base
	VMOVUPD (DI), Y8
	VMOVUPD 32(DI), Y9
	VPERMILPD $0x05, Y6, Y10
	VMULPD Y8, Y6, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y6
	VMOVUPD 64(DI), Y8
	VMOVUPD 96(DI), Y9
	VPERMILPD $0x05, Y7, Y10
	VMULPD Y8, Y7, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y7

	// Store to dst as row-major tile
	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	VMOVUPD Y8, 0(R8)
	VMOVUPD Y10, 32(R8)
	VMOVUPD Y9, 256(R8)
	VMOVUPD Y11, 288(R8)
	VMOVUPD Y12, 512(R8)
	VMOVUPD Y14, 544(R8)
	VMOVUPD Y13, 768(R8)
	VMOVUPD Y15, 800(R8)
	RET

// Diagnostic fused-twiddle mapping for rb=0, cb=1 (cols 4..7), disabled.
fwd_r16_twiddle_fused_diag_cb1:
	// Load scratch[0:4,4:8] (column-major tile, cb=1)
	LEAQ 64(R11), SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), BX
	LEAQ 768(SI), CX
	VMOVUPD 0(SI), Y0
	VMOVUPD 32(SI), Y1
	VMOVUPD 0(DI), Y2
	VMOVUPD 32(DI), Y3
	VMOVUPD 0(BX), Y4
	VMOVUPD 32(BX), Y5
	VMOVUPD 0(CX), Y6
	VMOVUPD 32(CX), Y7

	// Apply twiddles for col_in=1..3 (col_in=0 uses identity).
	// cb=1 means row_in=4..7, so start at pair=2 (rows 4..5) and pair=3 (rows 6..7).
	LEAQ 4096(R10), DI        // col=1 base
	VMOVUPD 128(DI), Y8       // pair=2 (rows 4..5) reDup
	VMOVUPD 160(DI), Y9       // pair=2 (rows 4..5) imDup
	VPERMILPD $0x05, Y2, Y10
	VMULPD Y8, Y2, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y2
	VMOVUPD 192(DI), Y8       // pair=3 (rows 6..7) reDup
	VMOVUPD 224(DI), Y9       // pair=3 (rows 6..7) imDup
	VPERMILPD $0x05, Y3, Y10
	VMULPD Y8, Y3, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y3

	ADDQ $512, DI             // col=2 base
	VMOVUPD 128(DI), Y8       // pair=2 (rows 4..5) reDup
	VMOVUPD 160(DI), Y9       // pair=2 (rows 4..5) imDup
	VPERMILPD $0x05, Y4, Y10
	VMULPD Y8, Y4, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y4
	VMOVUPD 192(DI), Y8       // pair=3 (rows 6..7) reDup
	VMOVUPD 224(DI), Y9       // pair=3 (rows 6..7) imDup
	VPERMILPD $0x05, Y5, Y10
	VMULPD Y8, Y5, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y5

	ADDQ $512, DI             // col=3 base
	VMOVUPD 128(DI), Y8       // pair=2 (rows 4..5) reDup
	VMOVUPD 160(DI), Y9       // pair=2 (rows 4..5) imDup
	VPERMILPD $0x05, Y6, Y10
	VMULPD Y8, Y6, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y6
	VMOVUPD 192(DI), Y8       // pair=3 (rows 6..7) reDup
	VMOVUPD 224(DI), Y9       // pair=3 (rows 6..7) imDup
	VPERMILPD $0x05, Y7, Y10
	VMULPD Y8, Y7, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y7

	// Store to dst as row-major tile (cols 4..7)
	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	LEAQ 64(R8), DI
	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)
	RET

// Diagnostic transpose mapping only for rb=0, cb=1 (cols 4..7), no twiddle.
fwd_r16_twiddle_fused_diag_cb1_notw:
	LEAQ 64(R11), SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), BX
	LEAQ 768(SI), CX
	VMOVUPD 0(SI), Y0
	VMOVUPD 32(SI), Y1
	VMOVUPD 0(DI), Y2
	VMOVUPD 32(DI), Y3
	VMOVUPD 0(BX), Y4
	VMOVUPD 32(BX), Y5
	VMOVUPD 0(CX), Y6
	VMOVUPD 32(CX), Y7

	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	LEAQ 64(R8), DI
	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)
	RET

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
	XORQ R12, R12             // row block (0..3)

fwd_r16_final_rb:
	CMPQ R12, $4
	JGE  fwd_r16_final_copy
	MOVQ R12, R14
	SHLQ $10, R14             // row_block_bytes = rb * 1024
	MOVQ R12, DX
	SHLQ $6, DX               // row_base_bytes = rb * 64
	XORQ R13, R13             // col block (0..3)

fwd_r16_final_cb:
	CMPQ R13, $4
	JGE  fwd_r16_final_rb_next
	MOVQ R13, AX
	SHLQ $6, AX               // col_block_bytes = cb * 64
	LEAQ (R8)(R14*1), SI
	ADDQ AX, SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), BX
	LEAQ 768(SI), CX

	VMOVUPD 0(SI), Y0
	VMOVUPD 32(SI), Y1
	VMOVUPD 0(DI), Y2
	VMOVUPD 32(DI), Y3
	VMOVUPD 0(BX), Y4
	VMOVUPD 32(BX), Y5
	VMOVUPD 0(CX), Y6
	VMOVUPD 32(CX), Y7

	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	MOVQ R13, AX
	SHLQ $10, AX              // cb * 1024
	LEAQ (R11)(AX*1), DI
	ADDQ DX, DI               // add row_base_bytes

	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)

	INCQ R13
	JMP  fwd_r16_final_cb

fwd_r16_final_rb_next:
	INCQ R12
	JMP  fwd_r16_final_rb

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
	// rb/cb tile layout matches forward; columns become contiguous in scratch.
	XORQ R12, R12             // row block (0..3)

inv_r16_transpose_in_rb:
	CMPQ R12, $4
	JGE  inv_r16_stage1
	MOVQ R12, R14
	SHLQ $10, R14             // row_block_bytes = rb * 1024
	MOVQ R12, DX
	SHLQ $6, DX               // row_base_bytes = rb * 64
	XORQ R13, R13             // col block (0..3)

inv_r16_transpose_in_cb:
	CMPQ R13, $4
	JGE  inv_r16_transpose_in_rb_next
	MOVQ R13, AX
	SHLQ $6, AX               // col_block_bytes = cb * 64
	LEAQ (R9)(R14*1), SI
	ADDQ AX, SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), BX
	LEAQ 768(SI), CX

	VMOVUPD 0(SI), Y0
	VMOVUPD 32(SI), Y1
	VMOVUPD 0(DI), Y2
	VMOVUPD 32(DI), Y3
	VMOVUPD 0(BX), Y4
	VMOVUPD 32(BX), Y5
	VMOVUPD 0(CX), Y6
	VMOVUPD 32(CX), Y7

	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	MOVQ R13, AX
	SHLQ $10, AX              // cb * 1024
	LEAQ (R11)(AX*1), DI
	ADDQ DX, DI               // add row_base_bytes

	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)

	INCQ R13
	JMP  inv_r16_transpose_in_cb

inv_r16_transpose_in_rb_next:
	INCQ R12
	JMP  inv_r16_transpose_in_rb

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
	// Twiddle table prepends base twiddles (0..255), then packed pairs per column
	// (conjugated for inverse).
	MOVQ $1, R12              // col

inv_r16_twiddle_col:
	CMPQ R12, $16
	JGE  inv_r16_transpose_out
	MOVQ R12, AX
	SHLQ $8, AX               // col*256 bytes
	LEAQ (R11)(AX*1), SI
	MOVQ R12, AX
	DECQ AX                   // col-1
	SHLQ $9, AX               // (col-1)*512 bytes
	LEAQ 4096(R10)(AX*1), R14 // twiddle base for col (skip base twiddle[0:256])
	XORQ R13, R13             // pair (0..7)

inv_r16_twiddle_row:
	CMPQ R13, $8
	JGE  inv_r16_twiddle_next_col
	MOVQ R13, DX
	SHLQ $5, DX               // pair*32 bytes (2 complex128)
	VMOVUPD (SI)(DX*1), Y0

	MOVQ R13, BX
	SHLQ $6, BX               // pair*64 bytes (4 complex128)
	LEAQ (R14)(BX*1), DI
	VMOVUPD (DI), Y1          // reDup
	VMOVUPD 32(DI), Y2        // imDup

	// Complex multiply: Y0 *= conj(twiddle)
	VPERMILPD $0x05, Y0, Y3   // [i0, r0, i1, r1]
	VMULPD Y1, Y0, Y4
	VMULPD Y2, Y3, Y5
	VADDSUBPD Y5, Y4, Y0
	VMOVUPD Y0, (SI)(DX*1)

	INCQ R13
	JMP  inv_r16_twiddle_row

inv_r16_twiddle_next_col:
	INCQ R12
	JMP  inv_r16_twiddle_col

inv_r16_transpose_out:
	// ==================================================================
	// STEP 4: Transpose scratch (column-major) -> dst (row-major)
	// ==================================================================
	// Mirror of transpose-in; tiles are independent 4x4 blocks.
	XORQ R12, R12             // row block (0..3)

inv_r16_transpose_out_rb:
	CMPQ R12, $4
	JGE  inv_r16_stage2
	MOVQ R12, R14
	SHLQ $10, R14             // row_block_bytes = rb * 1024
	MOVQ R12, DX
	SHLQ $6, DX               // row_base_bytes = rb * 64
	XORQ R13, R13             // col block (0..3)

inv_r16_transpose_out_cb:
	CMPQ R13, $4
	JGE  inv_r16_transpose_out_rb_next
	MOVQ R13, AX
	SHLQ $6, AX               // col_block_bytes = cb * 64
	LEAQ (R11)(R14*1), SI
	ADDQ AX, SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), BX
	LEAQ 768(SI), CX

	VMOVUPD 0(SI), Y0
	VMOVUPD 32(SI), Y1
	VMOVUPD 0(DI), Y2
	VMOVUPD 32(DI), Y3
	VMOVUPD 0(BX), Y4
	VMOVUPD 32(BX), Y5
	VMOVUPD 0(CX), Y6
	VMOVUPD 32(CX), Y7

	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	MOVQ R13, AX
	SHLQ $10, AX              // cb * 1024
	LEAQ (R8)(AX*1), DI
	ADDQ DX, DI               // add row_base_bytes

	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)

	INCQ R13
	JMP  inv_r16_transpose_out_cb

inv_r16_transpose_out_rb_next:
	INCQ R12
	JMP  inv_r16_transpose_out_rb

inv_r16_stage2:
	// ==================================================================
	// STEP 5: Stage 2 - FFT-16 on each row (contiguous blocks)
	// ==================================================================
	// Rows are contiguous in dst after transpose-out; each row is independent.
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
	XORQ R12, R12             // row block (0..3)

inv_r16_final_rb:
	CMPQ R12, $4
	JGE  inv_r16_scale_copy
	MOVQ R12, R14
	SHLQ $10, R14             // row_block_bytes = rb * 1024
	MOVQ R12, DX
	SHLQ $6, DX               // row_base_bytes = rb * 64
	XORQ R13, R13             // col block (0..3)

inv_r16_final_cb:
	CMPQ R13, $4
	JGE  inv_r16_final_rb_next
	MOVQ R13, AX
	SHLQ $6, AX               // col_block_bytes = cb * 64
	LEAQ (R8)(R14*1), SI
	ADDQ AX, SI
	LEAQ 256(SI), DI
	LEAQ 512(SI), BX
	LEAQ 768(SI), CX

	VMOVUPD 0(SI), Y0
	VMOVUPD 32(SI), Y1
	VMOVUPD 0(DI), Y2
	VMOVUPD 32(DI), Y3
	VMOVUPD 0(BX), Y4
	VMOVUPD 32(BX), Y5
	VMOVUPD 0(CX), Y6
	VMOVUPD 32(CX), Y7

	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	MOVQ R13, AX
	SHLQ $10, AX              // cb * 1024
	LEAQ (R11)(AX*1), DI
	ADDQ DX, DI               // add row_base_bytes

	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)

	INCQ R13
	JMP  inv_r16_final_cb

inv_r16_final_rb_next:
	INCQ R12
	JMP  inv_r16_final_rb

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
