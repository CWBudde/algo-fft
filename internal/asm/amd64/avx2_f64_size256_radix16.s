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
	JGE  fwd_r16_twiddle_transpose_fused
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

fwd_r16_twiddle_transpose_fused:
	// ==================================================================
	// STEP 3+4 FUSED: Twiddle multiply + transpose in single pass
	// ==================================================================
	// Process all 16 tiles (4x4 complex128 blocks) in nested loops.
	// For each tile: load from scratch (column-major), apply twiddles,
	// transpose in-register, write to dst (row-major).
	// This eliminates one full memory round-trip over the 4KB scratch buffer.
	//
	// Memory layout:
	// - Scratch: column-major, 256-byte column stride (16 complex128 per column)
	// - Dst: row-major, 256-byte row stride (16 complex128 per row)
	// - Twiddles: 4096 + (global_col-1)*512 + pair*64
	//
	// Tile addressing:
	// - scratch_tile_base = cb*1024 + rb*64
	// - dst_tile_base = rb*1024 + cb*64
	// - global_row = rb*4 + row_in_tile
	// - global_col = cb*4 + col_in_tile
	XORQ R12, R12             // rb = row block (0..3)

fwd_r16_fused_rb:
	CMPQ R12, $4
	JGE  fwd_r16_stage2
	XORQ R13, R13             // cb = col block (0..3)

fwd_r16_fused_cb:
	CMPQ R13, $4
	JGE  fwd_r16_fused_rb_next

	// ----------------------------------------------------------------
	// Calculate scratch source address (column-major layout)
	// scratch_base = cb*1024 + rb*64
	// ----------------------------------------------------------------
	MOVQ R13, AX
	SHLQ $10, AX              // AX = cb*1024
	MOVQ R12, DX
	SHLQ $6, DX               // DX = rb*64
	ADDQ DX, AX               // AX = cb*1024 + rb*64
	LEAQ (R11)(AX*1), SI      // SI = scratch + tile_base

	// ----------------------------------------------------------------
	// Load 4x4 tile (8 Y-registers, column-major order)
	// Y0/Y1: col_in_tile=0 (global_col = cb*4+0)
	// Y2/Y3: col_in_tile=1 (global_col = cb*4+1)
	// Y4/Y5: col_in_tile=2 (global_col = cb*4+2)
	// Y6/Y7: col_in_tile=3 (global_col = cb*4+3)
	// Y0,Y2,Y4,Y6: rows 0-1 in tile (global rows rb*4+0, rb*4+1)
	// Y1,Y3,Y5,Y7: rows 2-3 in tile (global rows rb*4+2, rb*4+3)
	// ----------------------------------------------------------------
	VMOVUPD 0(SI), Y0         // col 0, rows 0-1
	VMOVUPD 32(SI), Y1        // col 0, rows 2-3
	VMOVUPD 256(SI), Y2       // col 1, rows 0-1
	VMOVUPD 288(SI), Y3       // col 1, rows 2-3
	VMOVUPD 512(SI), Y4       // col 2, rows 0-1
	VMOVUPD 544(SI), Y5       // col 2, rows 2-3
	VMOVUPD 768(SI), Y6       // col 3, rows 0-1
	VMOVUPD 800(SI), Y7       // col 3, rows 2-3

	// ----------------------------------------------------------------
	// Apply twiddles for W_256^(global_row * global_col)
	// For tile (rb, cb):
	// - global_row_base = rb*4, global_col_base = cb*4
	// - pair_base = rb*2 (each rb covers 4 rows = 2 pairs)
	// - Columns 1,2,3 in tile need twiddles if cb>0; else only cols 1,2,3
	// - Column 0 gets twiddles if cb>0 (global_col = cb*4 >= 4)
	// ----------------------------------------------------------------

	// Calculate pair base offset for this tile: pair = rb*2
	MOVQ R12, BX
	SHLQ $7, BX               // BX = rb*2*64 = rb*128 (pair offset in bytes)

	// Check if cb == 0 (special case: col 0 has identity twiddle)
	TESTQ R13, R13
	JZ   fwd_r16_fused_cb0_twiddle

	// ----------------------------------------------------------------
	// cb > 0: Apply twiddles to all 4 columns
	// ----------------------------------------------------------------
	// Column 0 in tile (global_col = cb*4)
	MOVQ R13, AX
	SHLQ $2, AX               // AX = cb*4 = global_col
	DECQ AX                   // AX = global_col - 1
	SHLQ $9, AX               // AX = (global_col-1)*512
	LEAQ 4096(R10)(AX*1), R14 // R14 = twiddle base for this column

	// Y0: rows 0-1 (pair = rb*2)
	VMOVUPD (R14)(BX*1), Y8   // reDup for pair rb*2
	VMOVUPD 32(R14)(BX*1), Y9 // imDup for pair rb*2
	VPERMILPD $0x05, Y0, Y10
	VMULPD Y8, Y0, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y0

	// Y1: rows 2-3 (pair = rb*2+1)
	VMOVUPD 64(R14)(BX*1), Y8 // reDup for pair rb*2+1
	VMOVUPD 96(R14)(BX*1), Y9 // imDup for pair rb*2+1
	VPERMILPD $0x05, Y1, Y10
	VMULPD Y8, Y1, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y1

	// Column 1 in tile (global_col = cb*4+1)
	ADDQ $512, R14            // next column

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y2, Y10
	VMULPD Y8, Y2, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y2

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y3, Y10
	VMULPD Y8, Y3, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y3

	// Column 2 in tile (global_col = cb*4+2)
	ADDQ $512, R14

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y4, Y10
	VMULPD Y8, Y4, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y4

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y5, Y10
	VMULPD Y8, Y5, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y5

	// Column 3 in tile (global_col = cb*4+3)
	ADDQ $512, R14

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y6, Y10
	VMULPD Y8, Y6, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y6

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y7, Y10
	VMULPD Y8, Y7, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y7

	JMP fwd_r16_fused_transpose

fwd_r16_fused_cb0_twiddle:
	// ----------------------------------------------------------------
	// cb == 0: Column 0 has identity twiddle (W_256^0 = 1)
	// Only apply twiddles to columns 1, 2, 3 (global_col = 1, 2, 3)
	// ----------------------------------------------------------------
	// Column 1 in tile (global_col = 1)
	LEAQ 4096(R10), R14       // twiddle base for global_col=1 (col-1=0)

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y2, Y10
	VMULPD Y8, Y2, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y2

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y3, Y10
	VMULPD Y8, Y3, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y3

	// Column 2 in tile (global_col = 2)
	ADDQ $512, R14

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y4, Y10
	VMULPD Y8, Y4, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y4

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y5, Y10
	VMULPD Y8, Y5, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y5

	// Column 3 in tile (global_col = 3)
	ADDQ $512, R14

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y6, Y10
	VMULPD Y8, Y6, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y6

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y7, Y10
	VMULPD Y8, Y7, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y7

fwd_r16_fused_transpose:
	// ----------------------------------------------------------------
	// Transpose 4x4 tile via VPERM2F128
	// Input (column-major in Y0-Y7):
	//   Y0/Y1 = col0, Y2/Y3 = col1, Y4/Y5 = col2, Y6/Y7 = col3
	// Output (row-major in Y8-Y15):
	//   Y8/Y10 = row0, Y9/Y11 = row1, Y12/Y14 = row2, Y13/Y15 = row3
	// ----------------------------------------------------------------
	VPERM2F128 $0x20, Y2, Y0, Y8   // low halves of Y0,Y2
	VPERM2F128 $0x31, Y2, Y0, Y9   // high halves of Y0,Y2
	VPERM2F128 $0x20, Y6, Y4, Y10  // low halves of Y4,Y6
	VPERM2F128 $0x31, Y6, Y4, Y11  // high halves of Y4,Y6
	VPERM2F128 $0x20, Y3, Y1, Y12  // low halves of Y1,Y3
	VPERM2F128 $0x31, Y3, Y1, Y13  // high halves of Y1,Y3
	VPERM2F128 $0x20, Y7, Y5, Y14  // low halves of Y5,Y7
	VPERM2F128 $0x31, Y7, Y5, Y15  // high halves of Y5,Y7

	// ----------------------------------------------------------------
	// Calculate dst address (row-major layout)
	// dst_base = rb*1024 + cb*64
	// ----------------------------------------------------------------
	MOVQ R12, AX
	SHLQ $10, AX              // AX = rb*1024
	MOVQ R13, DX
	SHLQ $6, DX               // DX = cb*64
	ADDQ DX, AX               // AX = rb*1024 + cb*64
	LEAQ (R8)(AX*1), DI       // DI = dst + tile_base

	// ----------------------------------------------------------------
	// Store transposed tile to dst
	// ----------------------------------------------------------------
	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)

	INCQ R13
	JMP  fwd_r16_fused_cb

fwd_r16_fused_rb_next:
	INCQ R12
	JMP  fwd_r16_fused_rb

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
	JGE  inv_r16_twiddle_transpose_fused
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

inv_r16_twiddle_transpose_fused:
	// ==================================================================
	// STEP 3+4 FUSED: Twiddle multiply + transpose in single pass
	// ==================================================================
	// Same structure as forward. The packed twiddle table for inverse
	// already has conjugated imaginary parts (prepared by prepareTwiddle256Radix16AVX2),
	// so we use the same multiply code as forward.
	XORQ R12, R12             // rb = row block (0..3)

inv_r16_fused_rb:
	CMPQ R12, $4
	JGE  inv_r16_stage2
	XORQ R13, R13             // cb = col block (0..3)

inv_r16_fused_cb:
	CMPQ R13, $4
	JGE  inv_r16_fused_rb_next

	// Calculate scratch source address (column-major layout)
	MOVQ R13, AX
	SHLQ $10, AX              // AX = cb*1024
	MOVQ R12, DX
	SHLQ $6, DX               // DX = rb*64
	ADDQ DX, AX               // AX = cb*1024 + rb*64
	LEAQ (R11)(AX*1), SI      // SI = scratch + tile_base

	// Load 4x4 tile (8 Y-registers, column-major order)
	VMOVUPD 0(SI), Y0         // col 0, rows 0-1
	VMOVUPD 32(SI), Y1        // col 0, rows 2-3
	VMOVUPD 256(SI), Y2       // col 1, rows 0-1
	VMOVUPD 288(SI), Y3       // col 1, rows 2-3
	VMOVUPD 512(SI), Y4       // col 2, rows 0-1
	VMOVUPD 544(SI), Y5       // col 2, rows 2-3
	VMOVUPD 768(SI), Y6       // col 3, rows 0-1
	VMOVUPD 800(SI), Y7       // col 3, rows 2-3

	// Calculate pair base offset for this tile: pair = rb*2
	MOVQ R12, BX
	SHLQ $7, BX               // BX = rb*2*64 = rb*128 (pair offset in bytes)

	// Check if cb == 0 (special case: col 0 has identity twiddle)
	TESTQ R13, R13
	JZ   inv_r16_fused_cb0_twiddle

	// ----------------------------------------------------------------
	// cb > 0: Apply twiddles to all 4 columns
	// (Twiddle table already pre-conjugated for inverse)
	// ----------------------------------------------------------------
	// Column 0 in tile (global_col = cb*4)
	MOVQ R13, AX
	SHLQ $2, AX               // AX = cb*4 = global_col
	DECQ AX                   // AX = global_col - 1
	SHLQ $9, AX               // AX = (global_col-1)*512
	LEAQ 4096(R10)(AX*1), R14 // R14 = twiddle base for this column

	// Y0: rows 0-1 (pair = rb*2)
	VMOVUPD (R14)(BX*1), Y8   // reDup for pair rb*2
	VMOVUPD 32(R14)(BX*1), Y9 // imDup for pair rb*2
	VPERMILPD $0x05, Y0, Y10
	VMULPD Y8, Y0, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y0

	// Y1: rows 2-3 (pair = rb*2+1)
	VMOVUPD 64(R14)(BX*1), Y8 // reDup for pair rb*2+1
	VMOVUPD 96(R14)(BX*1), Y9 // imDup for pair rb*2+1
	VPERMILPD $0x05, Y1, Y10
	VMULPD Y8, Y1, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y1

	// Column 1 in tile (global_col = cb*4+1)
	ADDQ $512, R14

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y2, Y10
	VMULPD Y8, Y2, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y2

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y3, Y10
	VMULPD Y8, Y3, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y3

	// Column 2 in tile (global_col = cb*4+2)
	ADDQ $512, R14

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y4, Y10
	VMULPD Y8, Y4, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y4

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y5, Y10
	VMULPD Y8, Y5, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y5

	// Column 3 in tile (global_col = cb*4+3)
	ADDQ $512, R14

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y6, Y10
	VMULPD Y8, Y6, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y6

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y7, Y10
	VMULPD Y8, Y7, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y7

	JMP inv_r16_fused_transpose

inv_r16_fused_cb0_twiddle:
	// ----------------------------------------------------------------
	// cb == 0: Column 0 has identity twiddle (W_256^0 = 1)
	// Only apply twiddles to columns 1, 2, 3 (global_col = 1, 2, 3)
	// ----------------------------------------------------------------
	// Column 1 in tile (global_col = 1)
	LEAQ 4096(R10), R14       // twiddle base for global_col=1 (col-1=0)

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y2, Y10
	VMULPD Y8, Y2, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y2

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y3, Y10
	VMULPD Y8, Y3, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y3

	// Column 2 in tile (global_col = 2)
	ADDQ $512, R14

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y4, Y10
	VMULPD Y8, Y4, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y4

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y5, Y10
	VMULPD Y8, Y5, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y5

	// Column 3 in tile (global_col = 3)
	ADDQ $512, R14

	VMOVUPD (R14)(BX*1), Y8
	VMOVUPD 32(R14)(BX*1), Y9
	VPERMILPD $0x05, Y6, Y10
	VMULPD Y8, Y6, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y6

	VMOVUPD 64(R14)(BX*1), Y8
	VMOVUPD 96(R14)(BX*1), Y9
	VPERMILPD $0x05, Y7, Y10
	VMULPD Y8, Y7, Y11
	VMULPD Y9, Y10, Y12
	VADDSUBPD Y12, Y11, Y7

inv_r16_fused_transpose:
	// Transpose 4x4 tile via VPERM2F128
	VPERM2F128 $0x20, Y2, Y0, Y8
	VPERM2F128 $0x31, Y2, Y0, Y9
	VPERM2F128 $0x20, Y6, Y4, Y10
	VPERM2F128 $0x31, Y6, Y4, Y11
	VPERM2F128 $0x20, Y3, Y1, Y12
	VPERM2F128 $0x31, Y3, Y1, Y13
	VPERM2F128 $0x20, Y7, Y5, Y14
	VPERM2F128 $0x31, Y7, Y5, Y15

	// Calculate dst address (row-major layout)
	MOVQ R12, AX
	SHLQ $10, AX              // AX = rb*1024
	MOVQ R13, DX
	SHLQ $6, DX               // DX = cb*64
	ADDQ DX, AX               // AX = rb*1024 + cb*64
	LEAQ (R8)(AX*1), DI       // DI = dst + tile_base

	// Store transposed tile to dst
	VMOVUPD Y8, 0(DI)
	VMOVUPD Y10, 32(DI)
	VMOVUPD Y9, 256(DI)
	VMOVUPD Y11, 288(DI)
	VMOVUPD Y12, 512(DI)
	VMOVUPD Y14, 544(DI)
	VMOVUPD Y13, 768(DI)
	VMOVUPD Y15, 800(DI)

	INCQ R13
	JMP  inv_r16_fused_cb

inv_r16_fused_rb_next:
	INCQ R12
	JMP  inv_r16_fused_rb

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
