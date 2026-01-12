//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-256 Radix-16 FFT Kernel for AMD64 (complex64)
// ===========================================================================
// Algorithm outline (16x16 matrix factorization):
// 1. Transpose input to column-major (scratch) so columns are contiguous.
// 2. Stage 1: 16 FFT-16 passes over columns (contiguous blocks).
// 3. Twiddle multiply: W_256^(row*col) applied element-wise.
// 4. Transpose back to row-major (dst) for contiguous row FFTs.
// 5. Stage 2: 16 FFT-16 passes over rows (contiguous blocks).
//
// Note: Final output is in natural order after Stage 2 (row-major layout).
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardAVX2Size256Radix16Complex64Asm(SB), NOSPLIT, $128-97
	// --- Argument Loading ---
	MOVQ dst+0(FP), R8           // R8 = Destination pointer
	MOVQ src+24(FP), R9          // R9 = Source pointer
	MOVQ twiddle+48(FP), R10     // R10 = Twiddle factors pointer (W_256)
	MOVQ scratch+72(FP), R11     // R11 = Scratch pointer (size 256)
	MOVQ src+32(FP), R13         // R13 = Length of source slice

	// --- Input Validation ---
	CMPQ R13, $256               // Verify length is exactly 256
	JNE  fwd_ret_false           // Return false if validation fails

	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   fwd_ret_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   fwd_ret_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   fwd_ret_false

	// =======================================================================
	// STEP 0: Build W_16 table in local stack (from W_256 with stride 16)
	// =======================================================================
	LEAQ 0(SP), R15              // R15 = local W_16 pointer
	XORQ CX, CX                  // CX = k

w16_copy_loop:
	MOVQ CX, DX                  // DX = k
	SHLQ $7, DX                  // DX = k * 128 (16 * 8 bytes)
	MOVQ (R10)(DX*1), AX         // AX = W_256[k*16]
	MOVQ AX, (R15)(CX*8)         // W_16[k] = W_256[k*16]
	INCQ CX
	CMPQ CX, $16
	JL   w16_copy_loop

	// =======================================================================
	// STEP 1: Transpose input (row-major) -> scratch (column-major)
	// =======================================================================
	// Vectorized 16x16 transpose using 4x4 blocks (each element is complex64 = 8B).
	// Operates on 4 rows x 4 cols at a time:
	// - Load 4 contiguous complex64 from 4 rows (row-major src)
	// - Transpose in registers (64-bit lanes)
	// - Store 4 contiguous complex64 into 4 columns (col-major scratch)
	// Fully unrolled rowBlock loop (0..3)
	// rowBlock 0
	MOVQ $0, R12                 // rowStartBytes
	MOVQ $0, R14                 // rowOffScratchBytes
	// Unrolled colBlock loop (0..3)
	// colBlock 0 (colOffBytes=0, scratch+0)
	LEAQ (R9)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1 (colOffBytes=32, scratch+512)
	LEAQ (R9)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2 (colOffBytes=64, scratch+1024)
	LEAQ (R9)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3 (colOffBytes=96, scratch+1536)
	LEAQ (R9)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 1
	MOVQ $512, R12
	MOVQ $32, R14
	// colBlock 0
	LEAQ (R9)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R9)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R9)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R9)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 2
	MOVQ $1024, R12
	MOVQ $64, R14
	// colBlock 0
	LEAQ (R9)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R9)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R9)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R9)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 3
	MOVQ $1536, R12
	MOVQ $96, R14
	// colBlock 0
	LEAQ (R9)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R9)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R9)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R9)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// =======================================================================
	// STEP 2: Stage 1 - FFT-16 on each column (contiguous blocks in scratch)
	// =======================================================================
	XORQ CX, CX                  // CX = column index

stage1_fft_loop:
	MOVQ CX, AX                  // AX = col
	SHLQ $7, AX                  // AX = col * 128
	LEAQ (R11)(AX*1), SI         // SI = scratch + col*128

	// --- FFT-16 kernel (in-place, SI points to 16 complex64) ---
	VMOVUPS 0(SI), Y0            // Y0 = Row 0: elements [0, 1, 2, 3]
	VMOVUPS 32(SI), Y1           // Y1 = Row 1: elements [4, 5, 6, 7]
	VMOVUPS 64(SI), Y2           // Y2 = Row 2: elements [8, 9, 10, 11]
	VMOVUPS 96(SI), Y3           // Y3 = Row 3: elements [12, 13, 14, 15]

	VADDPS Y2, Y0, Y4            // T0 = Row0 + Row2
	VADDPS Y3, Y1, Y5            // T1 = Row1 + Row3
	VSUBPS Y2, Y0, Y6            // T2 = Row0 - Row2
	VSUBPS Y3, Y1, Y7            // T3 = Row1 - Row3

	VADDPS Y5, Y4, Y0            // F0 = T0 + T1 (Final Row 0 DC)
	VSUBPS Y5, Y4, Y2            // F2 = T0 - T1 (Final Row 2 harmonic)

	VPERMILPS $0xB1, Y7, Y7      // Swap Real/Imag
	VMOVUPS ·maskNegHiPS(SB), X8 // Load 128-bit sign mask for imaginary parts
	VINSERTF128 $0x01, X8, Y8, Y8 // Broadcast mask to 256-bit Y8
	VXORPS Y8, Y7, Y7            // Apply mask to get (y - ix) = T3 * -i

	VADDPS Y7, Y6, Y1            // F1 = T2 + (T3 * -i)
	VSUBPS Y7, Y6, Y3            // F3 = T2 - (T3 * -i)

	// --- Internal Twiddle Factor Multiplication (W_16^{row * col}) ---
	VMOVUPS 0(R15), Y4           // Load contiguous twiddles W^0..W^3
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y1, Y7            // Y7 = Row1 * Re(W)
	VPERMILPS $0xB1, Y1, Y8      // Y8 = Row1 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row1_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y1         // Row 1 = (ac-bd, ad+bc)

	VMOVSD 0(R15), X4            // Load W^0
	VMOVSD 16(R15), X5           // Load W^2
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^2)
	VMOVSD 32(R15), X5           // Load W^4
	VMOVSD 48(R15), X6           // Load W^6
	VUNPCKLPD X6, X5, X5         // X5 = (W^4, W^6)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^2, W^4, W^6)
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y2, Y7            // Y7 = Row2 * Re(W)
	VPERMILPS $0xB1, Y2, Y8      // Y8 = Row2 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row2_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y2         // Row 2 = (ac-bd, ad+bc)

	VMOVSD 0(R15), X4            // Load W^0
	VMOVSD 24(R15), X5           // Load W^3
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^3)
	VMOVSD 48(R15), X5           // Load W^6
	VMOVSD 8(R15), X6            // Load W^1
	VXORPS X7, X7, X7            // Zero X7
	VSUBPS X6, X7, X6            // X6 = -W^1 = W^9
	VUNPCKLPD X6, X5, X5         // X5 = (W^6, W^9)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^3, W^6, W^9)
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y3, Y7            // Y7 = Row3 * Re(W)
	VPERMILPS $0xB1, Y3, Y8      // Y8 = Row3 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row3_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y3         // Row 3 = (ac-bd, ad+bc)

	// --- Horizontal FFT4 (Transform within each YMM Row) ---
	VMOVUPS ·maskNegLoPS(SB), X15 // Sign mask for rotation by i

	VEXTRACTF128 $0x01, Y0, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X0, X5            // X5 = Sum
	VSUBPS X4, X0, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y0 // Y0 = Row 0 Final

	VEXTRACTF128 $0x01, Y1, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X1, X5            // X5 = Sum
	VSUBPS X4, X1, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y1 // Y1 = Row 1 Final

	VEXTRACTF128 $0x01, Y2, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X2, X5            // X5 = Sum
	VSUBPS X4, X2, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y2 // Y2 = Row 2 Final

	VEXTRACTF128 $0x01, Y3, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X3, X5            // X5 = Sum
	VSUBPS X4, X3, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y3 // Y3 = Row 3 Final

	// --- Matrix Transposition (4x4) ---
	VUNPCKLPD Y1, Y0, Y4         // Y4 = (y00, y10, y02, y12)
	VUNPCKHPD Y1, Y0, Y5         // Y5 = (y01, y11, y03, y13)
	VUNPCKLPD Y3, Y2, Y6         // Y6 = (y20, y30, y22, y32)
	VUNPCKHPD Y3, Y2, Y7         // Y7 = (y21, y31, y23, y33)

	VPERM2F128 $0x20, Y6, Y4, Y0 // Y0 = (y00, y10, y20, y30)
	VPERM2F128 $0x20, Y7, Y5, Y1 // Y1 = (y01, y11, y21, y31)
	VPERM2F128 $0x31, Y6, Y4, Y2 // Y2 = (y02, y12, y22, y32)
	VPERM2F128 $0x31, Y7, Y5, Y3 // Y3 = (y03, y13, y23, y33)

	VMOVUPS Y0, 0(SI)            // Store row 0
	VMOVUPS Y1, 32(SI)           // Store row 1
	VMOVUPS Y2, 64(SI)           // Store row 2
	VMOVUPS Y3, 96(SI)           // Store row 3

	INCQ CX
	CMPQ CX, $16
	JL   stage1_fft_loop

	// =======================================================================
	// STEP 3: Twiddle multiplication W_256^(row*col)
	// =======================================================================
	XORQ CX, CX                  // CX = col

stage1_twiddle_col_loop:
	MOVQ CX, AX                  // AX = col
	SHLQ $7, AX                  // AX = col * 128
	LEAQ (R11)(AX*1), SI         // SI = scratch + col*128
	CMPQ CX, $0
	JE   stage1_twiddle_next_col // col=0 => twiddle=1, no-op

	XORQ R14, R14                // R14 = baseIdx

	// Unrolled pair loop (pair 0..7), 2 complex64 per pair
	// pair 0 (offset 0)
	VMOVUPS 0(SI), X0            // X0 = 2 complex64
	LEAQ (R14)(CX*1), R13        // R13 = baseIdx + col
	VMOVSD (R10)(R14*8), X1      // twiddle0
	VMOVSD (R10)(R13*8), X2      // twiddle1
	VUNPCKLPD X2, X1, X1         // X1 = (tw0, tw1)
	VPERMILPS $0xA0, X1, X2      // X2 = Re(W)
	VPERMILPS $0xF5, X1, X3      // X3 = Im(W)
	VMULPS X2, X0, X4            // X4 = data * Re(W)
	VPERMILPS $0xB1, X0, X5      // X5 = swap(data)
	VMULPS X3, X5, X5            // X5 = swap(data) * Im(W)
	VADDSUBPS X5, X4, X0         // X0 = (ac-bd, ad+bc)
	VMOVUPS X0, 0(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 1 (offset 16)
	VMOVUPS 16(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 16(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 2 (offset 32)
	VMOVUPS 32(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 32(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 3 (offset 48)
	VMOVUPS 48(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 48(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 4 (offset 64)
	VMOVUPS 64(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 64(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 5 (offset 80)
	VMOVUPS 80(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 80(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 6 (offset 96)
	VMOVUPS 96(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 96(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 7 (offset 112)
	VMOVUPS 112(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 112(SI)

stage1_twiddle_next_col:
	INCQ CX
	CMPQ CX, $16
	JL   stage1_twiddle_col_loop

	// =======================================================================
	// STEP 4: Transpose scratch (column-major) -> dst (row-major)
	// =======================================================================
	// Vectorized 16x16 transpose using 4x4 blocks.
	// Fully unrolled rowBlock loop (0..3)
	// rowBlock 0
	MOVQ $0, R12                 // rowStartBytes
	MOVQ $0, R14                 // rowOffScratchBytes
	// Unrolled colBlock loop (0..3)
	// colBlock 0 (scratch+0, colOffBytes=0)
	LEAQ 0(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 1 (scratch+512, colOffBytes=32)
	LEAQ 512(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 32(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 2 (scratch+1024, colOffBytes=64)
	LEAQ 1024(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 64(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 3 (scratch+1536, colOffBytes=96)
	LEAQ 1536(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 96(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// rowBlock 1
	MOVQ $512, R12
	MOVQ $32, R14
	// colBlock 0
	LEAQ 0(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 1
	LEAQ 512(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 32(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 2
	LEAQ 1024(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 64(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 3
	LEAQ 1536(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 96(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// rowBlock 2
	MOVQ $1024, R12
	MOVQ $64, R14
	// colBlock 0
	LEAQ 0(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 1
	LEAQ 512(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 32(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 2
	LEAQ 1024(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 64(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 3
	LEAQ 1536(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 96(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// rowBlock 3
	MOVQ $1536, R12
	MOVQ $96, R14
	// colBlock 0
	LEAQ 0(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 1
	LEAQ 512(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 32(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 2
	LEAQ 1024(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 64(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 3
	LEAQ 1536(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 96(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// =======================================================================
	// STEP 5: Stage 2 - FFT-16 on each row (contiguous blocks in dst)
	// =======================================================================
	XORQ CX, CX                  // CX = row index

stage2_fft_loop:
	MOVQ CX, AX                  // AX = row
	SHLQ $7, AX                  // AX = row * 128
	LEAQ (R8)(AX*1), SI          // SI = dst + row*128

	// --- FFT-16 kernel (in-place, SI points to 16 complex64) ---
	VMOVUPS 0(SI), Y0            // Y0 = Row 0: elements [0, 1, 2, 3]
	VMOVUPS 32(SI), Y1           // Y1 = Row 1: elements [4, 5, 6, 7]
	VMOVUPS 64(SI), Y2           // Y2 = Row 2: elements [8, 9, 10, 11]
	VMOVUPS 96(SI), Y3           // Y3 = Row 3: elements [12, 13, 14, 15]

	VADDPS Y2, Y0, Y4            // T0 = Row0 + Row2
	VADDPS Y3, Y1, Y5            // T1 = Row1 + Row3
	VSUBPS Y2, Y0, Y6            // T2 = Row0 - Row2
	VSUBPS Y3, Y1, Y7            // T3 = Row1 - Row3

	VADDPS Y5, Y4, Y0            // F0 = T0 + T1
	VSUBPS Y5, Y4, Y2            // F2 = T0 - T1

	VPERMILPS $0xB1, Y7, Y7      // Swap Real/Imag
	VMOVUPS ·maskNegHiPS(SB), X8 // Load 128-bit sign mask for imaginary parts
	VINSERTF128 $0x01, X8, Y8, Y8 // Broadcast mask to 256-bit Y8
	VXORPS Y8, Y7, Y7            // Apply mask to get (y - ix) = T3 * -i

	VADDPS Y7, Y6, Y1            // F1 = T2 + (T3 * -i)
	VSUBPS Y7, Y6, Y3            // F3 = T2 - (T3 * -i)

	// --- Internal Twiddle Factor Multiplication (W_16^{row * col}) ---
	VMOVUPS 0(R15), Y4           // Load contiguous twiddles W^0..W^3
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y1, Y7            // Y7 = Row1 * Re(W)
	VPERMILPS $0xB1, Y1, Y8      // Y8 = Row1 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row1_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y1         // Row 1 = (ac-bd, ad+bc)

	VMOVSD 0(R15), X4            // Load W^0
	VMOVSD 16(R15), X5           // Load W^2
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^2)
	VMOVSD 32(R15), X5           // Load W^4
	VMOVSD 48(R15), X6           // Load W^6
	VUNPCKLPD X6, X5, X5         // X5 = (W^4, W^6)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^2, W^4, W^6)
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y2, Y7            // Y7 = Row2 * Re(W)
	VPERMILPS $0xB1, Y2, Y8      // Y8 = Row2 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row2_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y2         // Row 2 = (ac-bd, ad+bc)

	VMOVSD 0(R15), X4            // Load W^0
	VMOVSD 24(R15), X5           // Load W^3
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^3)
	VMOVSD 48(R15), X5           // Load W^6
	VMOVSD 8(R15), X6            // Load W^1
	VXORPS X7, X7, X7            // Zero X7
	VSUBPS X6, X7, X6            // X6 = -W^1 = W^9
	VUNPCKLPD X6, X5, X5         // X5 = (W^6, W^9)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^3, W^6, W^9)
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y3, Y7            // Y7 = Row3 * Re(W)
	VPERMILPS $0xB1, Y3, Y8      // Y8 = Row3 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row3_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y3         // Row 3 = (ac-bd, ad+bc)

	// --- Horizontal FFT4 (Transform within each YMM Row) ---
	VMOVUPS ·maskNegLoPS(SB), X15 // Sign mask for rotation by i

	VEXTRACTF128 $0x01, Y0, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X0, X5            // X5 = Sum
	VSUBPS X4, X0, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y0 // Y0 = Row 0 Final

	VEXTRACTF128 $0x01, Y1, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X1, X5            // X5 = Sum
	VSUBPS X4, X1, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y1 // Y1 = Row 1 Final

	VEXTRACTF128 $0x01, Y2, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X2, X5            // X5 = Sum
	VSUBPS X4, X2, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y2 // Y2 = Row 2 Final

	VEXTRACTF128 $0x01, Y3, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X3, X5            // X5 = Sum
	VSUBPS X4, X3, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y3 // Y3 = Row 3 Final

	// --- Matrix Transposition (4x4) ---
	VUNPCKLPD Y1, Y0, Y4         // Y4 = (y00, y10, y02, y12)
	VUNPCKHPD Y1, Y0, Y5         // Y5 = (y01, y11, y03, y13)
	VUNPCKLPD Y3, Y2, Y6         // Y6 = (y20, y30, y22, y32)
	VUNPCKHPD Y3, Y2, Y7         // Y7 = (y21, y31, y23, y33)

	VPERM2F128 $0x20, Y6, Y4, Y0 // Y0 = (y00, y10, y20, y30)
	VPERM2F128 $0x20, Y7, Y5, Y1 // Y1 = (y01, y11, y21, y31)
	VPERM2F128 $0x31, Y6, Y4, Y2 // Y2 = (y02, y12, y22, y32)
	VPERM2F128 $0x31, Y7, Y5, Y3 // Y3 = (y03, y13, y23, y33)

	VMOVUPS Y0, 0(SI)            // Store row 0
	VMOVUPS Y1, 32(SI)           // Store row 1
	VMOVUPS Y2, 64(SI)           // Store row 2
	VMOVUPS Y3, 96(SI)           // Store row 3

	INCQ CX
	CMPQ CX, $16
	JL   stage2_fft_loop

	// =======================================================================
	// STEP 6: Final transposition to natural order (dst -> scratch)
	// =======================================================================
	// Vectorized 16x16 transpose using 4x4 blocks.
	// Fully unrolled rowBlock loop (0..3)
	// rowBlock 0
	MOVQ $0, R12                 // rowStartBytes
	MOVQ $0, R14                 // rowOffScratchBytes
	// Unrolled colBlock loop (0..3)
	// colBlock 0 (colOffBytes=0, scratch+0)
	LEAQ (R8)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1 (colOffBytes=32, scratch+512)
	LEAQ (R8)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2 (colOffBytes=64, scratch+1024)
	LEAQ (R8)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3 (colOffBytes=96, scratch+1536)
	LEAQ (R8)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 1
	MOVQ $512, R12
	MOVQ $32, R14
	// colBlock 0
	LEAQ (R8)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R8)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R8)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R8)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 2
	MOVQ $1024, R12
	MOVQ $64, R14
	// colBlock 0
	LEAQ (R8)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R8)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R8)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R8)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 3
	MOVQ $1536, R12
	MOVQ $96, R14
	// colBlock 0
	LEAQ (R8)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R8)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R8)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R8)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// Copy scratch -> dst
	XORQ CX, CX

fwd_final_copy_loop:
	VMOVUPS (R11)(CX*1), Y0      // Load 4 complex64 values
	VMOVUPS 32(R11)(CX*1), Y1
	VMOVUPS Y0, (R8)(CX*1)
	VMOVUPS Y1, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $2048
	JL   fwd_final_copy_loop

	VZEROUPPER                   // Reset SIMD state
	MOVB $1, ret+96(FP)         // Signal success
	RET

fwd_ret_false:
	MOVB $0, ret+96(FP)         // Signal failure
	RET

TEXT ·InverseAVX2Size256Radix16Complex64Asm(SB), NOSPLIT, $128-97
	// --- Argument Loading ---
	MOVQ dst+0(FP), R8           // R8 = Destination pointer
	MOVQ src+24(FP), R9          // R9 = Source pointer
	MOVQ twiddle+48(FP), R10     // R10 = Twiddle factors pointer (W_256)
	MOVQ scratch+72(FP), R11     // R11 = Scratch pointer (size 256)
	MOVQ src+32(FP), R13         // R13 = Length of source slice

	// --- Input Validation ---
	CMPQ R13, $256               // Verify length is exactly 256
	JNE  inv_ret_false           // Return false if validation fails

	MOVQ dst+8(FP), AX
	CMPQ AX, $256
	JL   inv_ret_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $256
	JL   inv_ret_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $256
	JL   inv_ret_false

	// =======================================================================
	// STEP 0: Build W_16 table in local stack (from W_256 with stride 16)
	// =======================================================================
	LEAQ 0(SP), R15              // R15 = local W_16 pointer
	XORQ CX, CX                  // CX = k

inv_w16_copy_loop:
	MOVQ CX, DX                  // DX = k
	SHLQ $7, DX                  // DX = k * 128 (16 * 8 bytes)
	MOVQ (R10)(DX*1), AX         // AX = W_256[k*16]
	MOVQ AX, (R15)(CX*8)         // W_16[k] = W_256[k*16]
	INCQ CX
	CMPQ CX, $16
	JL   inv_w16_copy_loop

	// =======================================================================
	// STEP 1: Conjugate + transpose input (row-major) -> scratch (column-major)
	// =======================================================================
	// Vectorized conjugate + transpose using 4x4 blocks.
	// Conjugation is done by flipping the sign bit of the imaginary float32 lanes.
	VXORPS Y12, Y12, Y12
	VMOVUPS ·maskNegHiPS(SB), X12
	VINSERTF128 $0x01, X12, Y12, Y12 // Y12 = conjugation mask in both lanes

	// Fully unrolled rowBlock loop (0..3)
	// rowBlock 0
	MOVQ $0, R12                 // rowStartBytes
	MOVQ $0, R14                 // rowOffScratchBytes
	// Unrolled colBlock loop (0..3)
	// colBlock 0 (colOffBytes=0, scratch+0)
	LEAQ (R9)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0           // conjugate
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1 (colOffBytes=32, scratch+512)
	LEAQ (R9)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0           // conjugate
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2 (colOffBytes=64, scratch+1024)
	LEAQ (R9)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0           // conjugate
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3 (colOffBytes=96, scratch+1536)
	LEAQ (R9)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0           // conjugate
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 1
	MOVQ $512, R12
	MOVQ $32, R14
	// colBlock 0
	LEAQ (R9)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R9)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R9)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R9)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 2
	MOVQ $1024, R12
	MOVQ $64, R14
	// colBlock 0
	LEAQ (R9)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R9)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R9)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R9)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 3
	MOVQ $1536, R12
	MOVQ $96, R14
	// colBlock 0
	LEAQ (R9)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R9)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R9)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R9)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VXORPS Y12, Y0, Y0
	VXORPS Y12, Y1, Y1
	VXORPS Y12, Y2, Y2
	VXORPS Y12, Y3, Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// =======================================================================
	// STEP 2: Stage 1 - FFT-16 on each column (contiguous blocks in scratch)
	// =======================================================================
	XORQ CX, CX                  // CX = column index

inv_stage1_fft_loop:
	MOVQ CX, AX                  // AX = col
	SHLQ $7, AX                  // AX = col * 128
	LEAQ (R11)(AX*1), SI         // SI = scratch + col*128

	// --- FFT-16 kernel (in-place, SI points to 16 complex64) ---
	VMOVUPS 0(SI), Y0            // Y0 = Row 0: elements [0, 1, 2, 3]
	VMOVUPS 32(SI), Y1           // Y1 = Row 1: elements [4, 5, 6, 7]
	VMOVUPS 64(SI), Y2           // Y2 = Row 2: elements [8, 9, 10, 11]
	VMOVUPS 96(SI), Y3           // Y3 = Row 3: elements [12, 13, 14, 15]

	VADDPS Y2, Y0, Y4            // T0 = Row0 + Row2
	VADDPS Y3, Y1, Y5            // T1 = Row1 + Row3
	VSUBPS Y2, Y0, Y6            // T2 = Row0 - Row2
	VSUBPS Y3, Y1, Y7            // T3 = Row1 - Row3

	VADDPS Y5, Y4, Y0            // F0 = T0 + T1 (Final Row 0 DC)
	VSUBPS Y5, Y4, Y2            // F2 = T0 - T1 (Final Row 2 harmonic)

	VPERMILPS $0xB1, Y7, Y7      // Swap Real/Imag
	VMOVUPS ·maskNegHiPS(SB), X8 // Load 128-bit sign mask for imaginary parts
	VINSERTF128 $0x01, X8, Y8, Y8 // Broadcast mask to 256-bit Y8
	VXORPS Y8, Y7, Y7            // Apply mask to get (y - ix) = T3 * -i

	VADDPS Y7, Y6, Y1            // F1 = T2 + (T3 * -i)
	VSUBPS Y7, Y6, Y3            // F3 = T2 - (T3 * -i)

	// --- Internal Twiddle Factor Multiplication (W_16^{row * col}) ---
	VMOVUPS 0(R15), Y4           // Load contiguous twiddles W^0..W^3
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y1, Y7            // Y7 = Row1 * Re(W)
	VPERMILPS $0xB1, Y1, Y8      // Y8 = Row1 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row1_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y1         // Row 1 = (ac-bd, ad+bc)

	VMOVSD 0(R15), X4            // Load W^0
	VMOVSD 16(R15), X5           // Load W^2
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^2)
	VMOVSD 32(R15), X5           // Load W^4
	VMOVSD 48(R15), X6           // Load W^6
	VUNPCKLPD X6, X5, X5         // X5 = (W^4, W^6)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^2, W^4, W^6)
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y2, Y7            // Y7 = Row2 * Re(W)
	VPERMILPS $0xB1, Y2, Y8      // Y8 = Row2 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row2_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y2         // Row 2 = (ac-bd, ad+bc)

	VMOVSD 0(R15), X4            // Load W^0
	VMOVSD 24(R15), X5           // Load W^3
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^3)
	VMOVSD 48(R15), X5           // Load W^6
	VMOVSD 8(R15), X6            // Load W^1
	VXORPS X7, X7, X7            // Zero X7
	VSUBPS X6, X7, X6            // X6 = -W^1 = W^9
	VUNPCKLPD X6, X5, X5         // X5 = (W^6, W^9)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^3, W^6, W^9)
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y3, Y7            // Y7 = Row3 * Re(W)
	VPERMILPS $0xB1, Y3, Y8      // Y8 = Row3 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row3_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y3         // Row 3 = (ac-bd, ad+bc)

	// --- Horizontal FFT4 (Transform within each YMM Row) ---
	VMOVUPS ·maskNegLoPS(SB), X15 // Sign mask for rotation by i

	VEXTRACTF128 $0x01, Y0, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X0, X5            // X5 = Sum
	VSUBPS X4, X0, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y0 // Y0 = Row 0 Final

	VEXTRACTF128 $0x01, Y1, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X1, X5            // X5 = Sum
	VSUBPS X4, X1, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y1 // Y1 = Row 1 Final

	VEXTRACTF128 $0x01, Y2, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X2, X5            // X5 = Sum
	VSUBPS X4, X2, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y2 // Y2 = Row 2 Final

	VEXTRACTF128 $0x01, Y3, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X3, X5            // X5 = Sum
	VSUBPS X4, X3, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y3 // Y3 = Row 3 Final

	// --- Matrix Transposition (4x4) ---
	VUNPCKLPD Y1, Y0, Y4         // Y4 = (y00, y10, y02, y12)
	VUNPCKHPD Y1, Y0, Y5         // Y5 = (y01, y11, y03, y13)
	VUNPCKLPD Y3, Y2, Y6         // Y6 = (y20, y30, y22, y32)
	VUNPCKHPD Y3, Y2, Y7         // Y7 = (y21, y31, y23, y33)

	VPERM2F128 $0x20, Y6, Y4, Y0 // Y0 = (y00, y10, y20, y30)
	VPERM2F128 $0x20, Y7, Y5, Y1 // Y1 = (y01, y11, y21, y31)
	VPERM2F128 $0x31, Y6, Y4, Y2 // Y2 = (y02, y12, y22, y32)
	VPERM2F128 $0x31, Y7, Y5, Y3 // Y3 = (y03, y13, y23, y33)

	VMOVUPS Y0, 0(SI)            // Store row 0
	VMOVUPS Y1, 32(SI)           // Store row 1
	VMOVUPS Y2, 64(SI)           // Store row 2
	VMOVUPS Y3, 96(SI)           // Store row 3

	INCQ CX
	CMPQ CX, $16
	JL   inv_stage1_fft_loop

	// =======================================================================
	// STEP 3: Twiddle multiplication W_256^(row*col)
	// =======================================================================
	XORQ CX, CX                  // CX = col

inv_stage1_twiddle_col_loop:
	MOVQ CX, AX                  // AX = col
	SHLQ $7, AX                  // AX = col * 128
	LEAQ (R11)(AX*1), SI         // SI = scratch + col*128
	CMPQ CX, $0
	JE   inv_stage1_twiddle_next_col

	XORQ R14, R14                // R14 = baseIdx

	// Unrolled pair loop (pair 0..7), 2 complex64 per pair
	// pair 0 (offset 0)
	VMOVUPS 0(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 0(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 1 (offset 16)
	VMOVUPS 16(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 16(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 2 (offset 32)
	VMOVUPS 32(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 32(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 3 (offset 48)
	VMOVUPS 48(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 48(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 4 (offset 64)
	VMOVUPS 64(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 64(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 5 (offset 80)
	VMOVUPS 80(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 80(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 6 (offset 96)
	VMOVUPS 96(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 96(SI)
	ADDQ CX, R14
	ADDQ CX, R14

	// pair 7 (offset 112)
	VMOVUPS 112(SI), X0
	LEAQ (R14)(CX*1), R13
	VMOVSD (R10)(R14*8), X1
	VMOVSD (R10)(R13*8), X2
	VUNPCKLPD X2, X1, X1
	VPERMILPS $0xA0, X1, X2
	VPERMILPS $0xF5, X1, X3
	VMULPS X2, X0, X4
	VPERMILPS $0xB1, X0, X5
	VMULPS X3, X5, X5
	VADDSUBPS X5, X4, X0
	VMOVUPS X0, 112(SI)

inv_stage1_twiddle_next_col:
	INCQ CX
	CMPQ CX, $16
	JL   inv_stage1_twiddle_col_loop

	// =======================================================================
	// STEP 4: Transpose scratch (column-major) -> dst (row-major)
	// =======================================================================
	// Vectorized 16x16 transpose using 4x4 blocks.
	// Fully unrolled rowBlock loop (0..3)
	// rowBlock 0
	MOVQ $0, R12                 // rowStartBytes
	MOVQ $0, R14                 // rowOffScratchBytes
	// Unrolled colBlock loop (0..3)
	// colBlock 0 (scratch+0, colOffBytes=0)
	LEAQ 0(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 1 (scratch+512, colOffBytes=32)
	LEAQ 512(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 32(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 2 (scratch+1024, colOffBytes=64)
	LEAQ 1024(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 64(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 3 (scratch+1536, colOffBytes=96)
	LEAQ 1536(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 96(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// rowBlock 1
	MOVQ $512, R12
	MOVQ $32, R14
	// colBlock 0
	LEAQ 0(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 1
	LEAQ 512(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 32(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 2
	LEAQ 1024(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 64(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 3
	LEAQ 1536(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 96(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// rowBlock 2
	MOVQ $1024, R12
	MOVQ $64, R14
	// colBlock 0
	LEAQ 0(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 1
	LEAQ 512(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 32(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 2
	LEAQ 1024(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 64(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 3
	LEAQ 1536(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 96(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// rowBlock 3
	MOVQ $1536, R12
	MOVQ $96, R14
	// colBlock 0
	LEAQ 0(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 1
	LEAQ 512(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 32(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 2
	LEAQ 1024(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 64(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)

	// colBlock 3
	LEAQ 1536(R11), SI
	VMOVUPS (SI)(R14*1), Y0
	VMOVUPS 128(SI)(R14*1), Y1
	VMOVUPS 256(SI)(R14*1), Y2
	VMOVUPS 384(SI)(R14*1), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ (R8)(R12*1), DI
	LEAQ 96(DI), DI
	VMOVUPS Y0, 0(DI)
	VMOVUPS Y1, 128(DI)
	VMOVUPS Y2, 256(DI)
	VMOVUPS Y3, 384(DI)
	XORQ CX, CX                  // CX = row index

inv_stage2_fft_loop:
	MOVQ CX, AX                  // AX = row
	SHLQ $7, AX                  // AX = row * 128
	LEAQ (R8)(AX*1), SI          // SI = dst + row*128

	// --- FFT-16 kernel (in-place, SI points to 16 complex64) ---
	VMOVUPS 0(SI), Y0            // Y0 = Row 0: elements [0, 1, 2, 3]
	VMOVUPS 32(SI), Y1           // Y1 = Row 1: elements [4, 5, 6, 7]
	VMOVUPS 64(SI), Y2           // Y2 = Row 2: elements [8, 9, 10, 11]
	VMOVUPS 96(SI), Y3           // Y3 = Row 3: elements [12, 13, 14, 15]

	VADDPS Y2, Y0, Y4            // T0 = Row0 + Row2
	VADDPS Y3, Y1, Y5            // T1 = Row1 + Row3
	VSUBPS Y2, Y0, Y6            // T2 = Row0 - Row2
	VSUBPS Y3, Y1, Y7            // T3 = Row1 - Row3

	VADDPS Y5, Y4, Y0            // F0 = T0 + T1
	VSUBPS Y5, Y4, Y2            // F2 = T0 - T1

	VPERMILPS $0xB1, Y7, Y7      // Swap Real/Imag
	VMOVUPS ·maskNegHiPS(SB), X8 // Load 128-bit sign mask for imaginary parts
	VINSERTF128 $0x01, X8, Y8, Y8 // Broadcast mask to 256-bit Y8
	VXORPS Y8, Y7, Y7            // Apply mask to get (y - ix) = T3 * -i

	VADDPS Y7, Y6, Y1            // F1 = T2 + (T3 * -i)
	VSUBPS Y7, Y6, Y3            // F3 = T2 - (T3 * -i)

	// --- Internal Twiddle Factor Multiplication (W_16^{row * col}) ---
	VMOVUPS 0(R15), Y4           // Load contiguous twiddles W^0..W^3
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y1, Y7            // Y7 = Row1 * Re(W)
	VPERMILPS $0xB1, Y1, Y8      // Y8 = Row1 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row1_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y1         // Row 1 = (ac-bd, ad+bc)

	VMOVSD 0(R15), X4            // Load W^0
	VMOVSD 16(R15), X5           // Load W^2
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^2)
	VMOVSD 32(R15), X5           // Load W^4
	VMOVSD 48(R15), X6           // Load W^6
	VUNPCKLPD X6, X5, X5         // X5 = (W^4, W^6)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^2, W^4, W^6)
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y2, Y7            // Y7 = Row2 * Re(W)
	VPERMILPS $0xB1, Y2, Y8      // Y8 = Row2 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row2_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y2         // Row 2 = (ac-bd, ad+bc)

	VMOVSD 0(R15), X4            // Load W^0
	VMOVSD 24(R15), X5           // Load W^3
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^3)
	VMOVSD 48(R15), X5           // Load W^6
	VMOVSD 8(R15), X6            // Load W^1
	VXORPS X7, X7, X7            // Zero X7
	VSUBPS X6, X7, X6            // X6 = -W^1 = W^9
	VUNPCKLPD X6, X5, X5         // X5 = (W^6, W^9)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^3, W^6, W^9)
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y3, Y7            // Y7 = Row3 * Re(W)
	VPERMILPS $0xB1, Y3, Y8      // Y8 = Row3 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row3_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y3         // Row 3 = (ac-bd, ad+bc)

	// --- Horizontal FFT4 (Transform within each YMM Row) ---
	VMOVUPS ·maskNegLoPS(SB), X15 // Sign mask for rotation by i

	VEXTRACTF128 $0x01, Y0, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X0, X5            // X5 = Sum
	VSUBPS X4, X0, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y0 // Y0 = Row 0 Final

	VEXTRACTF128 $0x01, Y1, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X1, X5            // X5 = Sum
	VSUBPS X4, X1, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y1 // Y1 = Row 1 Final

	VEXTRACTF128 $0x01, Y2, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X2, X5            // X5 = Sum
	VSUBPS X4, X2, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y2 // Y2 = Row 2 Final

	VEXTRACTF128 $0x01, Y3, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X3, X5            // X5 = Sum
	VSUBPS X4, X3, X6            // X6 = Diff
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = i*D1
	VSUBPS X10, X6, X12          // X12 = y1
	VADDPS X10, X6, X13          // X13 = y3
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y3 // Y3 = Row 3 Final

	// --- Matrix Transposition (4x4) ---
	VUNPCKLPD Y1, Y0, Y4         // Y4 = (y00, y10, y02, y12)
	VUNPCKHPD Y1, Y0, Y5         // Y5 = (y01, y11, y03, y13)
	VUNPCKLPD Y3, Y2, Y6         // Y6 = (y20, y30, y22, y32)
	VUNPCKHPD Y3, Y2, Y7         // Y7 = (y21, y31, y23, y33)

	VPERM2F128 $0x20, Y6, Y4, Y0 // Y0 = (y00, y10, y20, y30)
	VPERM2F128 $0x20, Y7, Y5, Y1 // Y1 = (y01, y11, y21, y31)
	VPERM2F128 $0x31, Y6, Y4, Y2 // Y2 = (y02, y12, y22, y32)
	VPERM2F128 $0x31, Y7, Y5, Y3 // Y3 = (y03, y13, y23, y33)

	VMOVUPS Y0, 0(SI)            // Store row 0
	VMOVUPS Y1, 32(SI)           // Store row 1
	VMOVUPS Y2, 64(SI)           // Store row 2
	VMOVUPS Y3, 96(SI)           // Store row 3

	INCQ CX
	CMPQ CX, $16
	JL   inv_stage2_fft_loop

	// =======================================================================
	// STEP 6: Final transposition to natural order (dst -> scratch)
	// =======================================================================
	// Vectorized 16x16 transpose using 4x4 blocks.
	// Fully unrolled rowBlock loop (0..3)
	// rowBlock 0
	MOVQ $0, R12                 // rowStartBytes
	MOVQ $0, R14                 // rowOffScratchBytes
	// Unrolled colBlock loop (0..3)
	// colBlock 0 (colOffBytes=0, scratch+0)
	LEAQ (R8)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1 (colOffBytes=32, scratch+512)
	LEAQ (R8)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2 (colOffBytes=64, scratch+1024)
	LEAQ (R8)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3 (colOffBytes=96, scratch+1536)
	LEAQ (R8)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 1
	MOVQ $512, R12
	MOVQ $32, R14
	// colBlock 0
	LEAQ (R8)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R8)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R8)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R8)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 2
	MOVQ $1024, R12
	MOVQ $64, R14
	// colBlock 0
	LEAQ (R8)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R8)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R8)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R8)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// rowBlock 3
	MOVQ $1536, R12
	MOVQ $96, R14
	// colBlock 0
	LEAQ (R8)(R12*1), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 0(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 1
	LEAQ (R8)(R12*1), SI
	LEAQ 32(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 512(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 2
	LEAQ (R8)(R12*1), SI
	LEAQ 64(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1024(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)

	// colBlock 3
	LEAQ (R8)(R12*1), SI
	LEAQ 96(SI), SI
	VMOVUPS 0(SI), Y0
	VMOVUPS 128(SI), Y1
	VMOVUPS 256(SI), Y2
	VMOVUPS 384(SI), Y3
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3
	LEAQ 1536(R11), DI
	VMOVUPS Y0, (DI)(R14*1)
	VMOVUPS Y1, 128(DI)(R14*1)
	VMOVUPS Y2, 256(DI)(R14*1)
	VMOVUPS Y3, 384(DI)(R14*1)
	MOVL ·twoFiftySixth32(SB), AX // 1/256 = 0.00390625
	MOVD AX, X8
	VBROADCASTSS X8, Y8         // Y8 = [scale,...]
	VXORPS Y9, Y9, Y9
	VMOVUPS ·maskNegHiPS(SB), X9 // Conjugation mask
	VINSERTF128 $0x01, X9, Y9, Y9 // Broadcast mask to 256-bit Y9

	// Unrolled scaling/conjugation: 2048 bytes total, 64 bytes per iteration => 32 iters
	// Each iter handles 8 complex64 values (2x YMM).
	VMOVUPS 0(R11), Y0           // Load
	VMOVUPS 32(R11), Y1
	VXORPS Y9, Y0, Y0            // Conjugate
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0            // Scale
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 0(R8)            // Store
	VMOVUPS Y1, 32(R8)

	VMOVUPS 64(R11), Y0
	VMOVUPS 96(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 64(R8)
	VMOVUPS Y1, 96(R8)

	VMOVUPS 128(R11), Y0
	VMOVUPS 160(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 128(R8)
	VMOVUPS Y1, 160(R8)

	VMOVUPS 192(R11), Y0
	VMOVUPS 224(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 192(R8)
	VMOVUPS Y1, 224(R8)

	VMOVUPS 256(R11), Y0
	VMOVUPS 288(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 256(R8)
	VMOVUPS Y1, 288(R8)

	VMOVUPS 320(R11), Y0
	VMOVUPS 352(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 320(R8)
	VMOVUPS Y1, 352(R8)

	VMOVUPS 384(R11), Y0
	VMOVUPS 416(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 384(R8)
	VMOVUPS Y1, 416(R8)

	VMOVUPS 448(R11), Y0
	VMOVUPS 480(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 448(R8)
	VMOVUPS Y1, 480(R8)

	VMOVUPS 512(R11), Y0
	VMOVUPS 544(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 512(R8)
	VMOVUPS Y1, 544(R8)

	VMOVUPS 576(R11), Y0
	VMOVUPS 608(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 576(R8)
	VMOVUPS Y1, 608(R8)

	VMOVUPS 640(R11), Y0
	VMOVUPS 672(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 640(R8)
	VMOVUPS Y1, 672(R8)

	VMOVUPS 704(R11), Y0
	VMOVUPS 736(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 704(R8)
	VMOVUPS Y1, 736(R8)

	VMOVUPS 768(R11), Y0
	VMOVUPS 800(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 768(R8)
	VMOVUPS Y1, 800(R8)

	VMOVUPS 832(R11), Y0
	VMOVUPS 864(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 832(R8)
	VMOVUPS Y1, 864(R8)

	VMOVUPS 896(R11), Y0
	VMOVUPS 928(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 896(R8)
	VMOVUPS Y1, 928(R8)

	VMOVUPS 960(R11), Y0
	VMOVUPS 992(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 960(R8)
	VMOVUPS Y1, 992(R8)

	VMOVUPS 1024(R11), Y0
	VMOVUPS 1056(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1024(R8)
	VMOVUPS Y1, 1056(R8)

	VMOVUPS 1088(R11), Y0
	VMOVUPS 1120(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1088(R8)
	VMOVUPS Y1, 1120(R8)

	VMOVUPS 1152(R11), Y0
	VMOVUPS 1184(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1152(R8)
	VMOVUPS Y1, 1184(R8)

	VMOVUPS 1216(R11), Y0
	VMOVUPS 1248(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1216(R8)
	VMOVUPS Y1, 1248(R8)

	VMOVUPS 1280(R11), Y0
	VMOVUPS 1312(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1280(R8)
	VMOVUPS Y1, 1312(R8)

	VMOVUPS 1344(R11), Y0
	VMOVUPS 1376(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1344(R8)
	VMOVUPS Y1, 1376(R8)

	VMOVUPS 1408(R11), Y0
	VMOVUPS 1440(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1408(R8)
	VMOVUPS Y1, 1440(R8)

	VMOVUPS 1472(R11), Y0
	VMOVUPS 1504(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1472(R8)
	VMOVUPS Y1, 1504(R8)

	VMOVUPS 1536(R11), Y0
	VMOVUPS 1568(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1536(R8)
	VMOVUPS Y1, 1568(R8)

	VMOVUPS 1600(R11), Y0
	VMOVUPS 1632(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1600(R8)
	VMOVUPS Y1, 1632(R8)

	VMOVUPS 1664(R11), Y0
	VMOVUPS 1696(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1664(R8)
	VMOVUPS Y1, 1696(R8)

	VMOVUPS 1728(R11), Y0
	VMOVUPS 1760(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1728(R8)
	VMOVUPS Y1, 1760(R8)

	VMOVUPS 1792(R11), Y0
	VMOVUPS 1824(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1792(R8)
	VMOVUPS Y1, 1824(R8)

	VMOVUPS 1856(R11), Y0
	VMOVUPS 1888(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1856(R8)
	VMOVUPS Y1, 1888(R8)

	VMOVUPS 1920(R11), Y0
	VMOVUPS 1952(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1920(R8)
	VMOVUPS Y1, 1952(R8)

	VMOVUPS 1984(R11), Y0
	VMOVUPS 2016(R11), Y1
	VXORPS Y9, Y0, Y0
	VXORPS Y9, Y1, Y1
	VMULPS Y8, Y0, Y0
	VMULPS Y8, Y1, Y1
	VMOVUPS Y0, 1984(R8)
	VMOVUPS Y1, 2016(R8)

	VZEROUPPER                   // Reset SIMD state
	MOVB $1, ret+96(FP)         // Signal success
	RET

inv_ret_false:
	MOVB $0, ret+96(FP)         // Signal failure
	RET
