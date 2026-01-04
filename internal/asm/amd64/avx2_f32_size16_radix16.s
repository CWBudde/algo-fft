//go:build amd64 && asm && !purego

#include "textflag.h"

// ===========================================================================
// Forward transform, size 16, complex64, AVX2 radix-16 (4x4) variant
// ===========================================================================
// This implementation uses a 4x4 Cooley-Tukey decomposition:
// 1. Vertical FFT4: 4 parallel FFTs on columns (using YMM registers as rows).
// 2. Internal Twiddles: Element-wise multiplication by 4x4 twiddle matrix.
// 3. Horizontal FFT4: FFT4 applied within each YMM register (on rows).
// 4. Matrix Transpose: Final 4x4 transposition to restore Natural Order.
// ===========================================================================
TEXT ·ForwardAVX2Size16Radix16Complex64Asm(SB), NOSPLIT, $0-121
	// --- Argument Loading ---
	MOVQ dst+0(FP), R8           // R8 = Destination pointer
	MOVQ src+24(FP), R9          // R9 = Source pointer
	MOVQ twiddle+48(FP), R10     // R10 = Twiddle factors pointer
	MOVQ scratch+72(FP), R11     // R11 = Scratch pointer (not used)
	MOVQ bitrev+96(FP), R12      // R12 = Bit-reversal pointer (identity expected)
	MOVQ src+32(FP), R13         // R13 = Length of source slice

	// --- Input Validation ---
	CMPQ R13, $16                // Verify length is exactly 16
	JNE  fwd_ret_false           // Return false if validation fails

	// =======================================================================
	// STEP 0: Load 4x4 Matrix into YMM Registers
	// =======================================================================
	VMOVUPS 0(R9), Y0            // Y0 = Row 0: elements [0, 1, 2, 3]
	VMOVUPS 32(R9), Y1           // Y1 = Row 1: elements [4, 5, 6, 7]
	VMOVUPS 64(R9), Y2           // Y2 = Row 2: elements [8, 9, 10, 11]
	VMOVUPS 96(R9), Y3           // Y3 = Row 3: elements [12, 13, 14, 15]


	// =======================================================================
	// STEP 1: Vertical FFT4 (Column-wise transformation)
	// =======================================================================
	VADDPS Y2, Y0, Y4            // T0 = Row0 + Row2
	VADDPS Y3, Y1, Y5            // T1 = Row1 + Row3
	VSUBPS Y2, Y0, Y6            // T2 = Row0 - Row2
	VSUBPS Y3, Y1, Y7            // T3 = Row1 - Row3

	VADDPS Y5, Y4, Y0            // F0 = T0 + T1 (Final Row 0 DC)
	VSUBPS Y5, Y4, Y2            // F2 = T0 - T1 (Final Row 2 harmonic)

	// Rotation by -i for DIF structure: (x+iy)*(-i) = y - ix
	VPERMILPS $0xB1, Y7, Y7      // Swap Real/Imag: (iy + x) -> (y + ix)
	VMOVUPS ·maskNegHiPS(SB), X8 // Load 128-bit sign mask for imaginary parts
	VINSERTF128 $0x01, X8, Y8, Y8 // Broadcast mask to 256-bit Y8
	VXORPS Y8, Y7, Y7            // Apply mask to get (y - ix) = T3 * -i

	VADDPS Y7, Y6, Y1            // F1 = T2 + (T3 * -i) = T2 - i*T3 (Result Row 1)
	VSUBPS Y7, Y6, Y3            // F3 = T2 - (T3 * -i) = T2 + i*T3 (Result Row 3)


	// =======================================================================
	// STEP 2: Internal Twiddle Factor Multiplication (W_16^{row * col})
	// =======================================================================
	
	// --- Row 1 Multiplication ---
	VMOVUPS 0(R10), Y4           // Load contiguous twiddles W^0..W^3
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y1, Y7            // Y7 = Row1 * Re(W)
	VPERMILPS $0xB1, Y1, Y8      // Y8 = Row1 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row1_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y1         // Row 1 = (ac-bd, ad+bc)

	// --- Row 2 Multiplication ---
	VMOVSD 0(R10), X4            // Load W^0
	VMOVSD 16(R10), X5           // Load W^2
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^2)
	VMOVSD 32(R10), X5           // Load W^4
	VMOVSD 48(R10), X6           // Load W^6
	VUNPCKLPD X6, X5, X5         // X5 = (W^4, W^6)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^2, W^4, W^6)
	VPERMILPS $0xA0, Y4, Y5      // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6      // Y6 = Im(W)
	VMULPS Y5, Y2, Y7            // Y7 = Row2 * Re(W)
	VPERMILPS $0xB1, Y2, Y8      // Y8 = Row2 swapped
	VMULPS Y6, Y8, Y8            // Y8 = Row2_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y2         // Row 2 = (ac-bd, ad+bc)

	// --- Row 3 Multiplication ---
	VMOVSD 0(R10), X4            // Load W^0
	VMOVSD 24(R10), X5           // Load W^3
	VUNPCKLPD X5, X4, X4         // X4 = (W^0, W^3)
	VMOVSD 48(R10), X5           // Load W^6
	VMOVSD 8(R10), X6            // Load W^1
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


	// =======================================================================
	// STEP 3: Horizontal FFT4 (Transform within each YMM Row)
	// =======================================================================
	VMOVUPS ·maskNegLoPS(SB), X15 // Sign mask for rotation by i

	// --- Process Row 0 ---
	VEXTRACTF128 $0x01, Y0, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X0, X5            // X5 = Sum (x0+x2, x1+x3)
	VSUBPS X4, X0, X6            // X6 = Diff (x0-x2, x1-x3)
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
	VPERM2F128 $0x20, Y9, Y8, Y0 // Y0 = Row 0 Final (y0, y1, y2, y3)

	// --- Process Row 1 ---
	VEXTRACTF128 $0x01, Y1, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X1, X5            // X5 = Sum (x0+x2, x1+x3)
	VSUBPS X4, X1, X6            // X6 = Diff (x0-x2, x1-x3)
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
	VPERM2F128 $0x20, Y9, Y8, Y1 // Y1 = Row 1 Final (y0, y1, y2, y3)

	// --- Process Row 2 ---
	VEXTRACTF128 $0x01, Y2, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X2, X5            // X5 = Sum (x0+x2, x1+x3)
	VSUBPS X4, X2, X6            // X6 = Diff (x0-x2, x1-x3)
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
	VPERM2F128 $0x20, Y9, Y8, Y2 // Y2 = Row 2 Final (y0, y1, y2, y3)

	// --- Process Row 3 ---
	VEXTRACTF128 $0x01, Y3, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X3, X5            // X5 = Sum (x0+x2, x1+x3)
	VSUBPS X4, X3, X6            // X6 = Diff (x0-x2, x1-x3)
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
	VPERM2F128 $0x20, Y9, Y8, Y3 // Y3 = Row 3 Final (y0, y1, y2, y3)


	// =======================================================================
	// STEP 4: Matrix Transposition (To restore 1D Natural Order)
	// =======================================================================
	VUNPCKLPD Y1, Y0, Y4         // Y4 = (y00, y10, y02, y12)
	VUNPCKHPD Y1, Y0, Y5         // Y5 = (y01, y11, y03, y13)
	VUNPCKLPD Y3, Y2, Y6         // Y6 = (y20, y30, y22, y32)
	VUNPCKHPD Y3, Y2, Y7         // Y7 = (y21, y31, y23, y33)
	
	VPERM2F128 $0x20, Y6, Y4, Y0 // Y0 = (y00, y10, y20, y30) -> elements 0, 1, 2, 3
	VPERM2F128 $0x20, Y7, Y5, Y1 // Y1 = (y01, y11, y21, y31) -> elements 4, 5, 6, 7
	VPERM2F128 $0x31, Y6, Y4, Y2 // Y2 = (y02, y12, y22, y32) -> elements 8, 9, 10, 11
	VPERM2F128 $0x31, Y7, Y5, Y3 // Y3 = (y03, y13, y23, y33) -> elements 12, 13, 14, 15


	// =======================================================================
	// STEP 5: Store Results and Return
	// =======================================================================
	VMOVUPS Y0, 0(R8)            // Store matrix Row 0
	VMOVUPS Y1, 32(R8)           // Store matrix Row 1
	VMOVUPS Y2, 64(R8)           // Store matrix Row 2
	VMOVUPS Y3, 96(R8)           // Store matrix Row 3

	VZEROUPPER                   // Reset SIMD state
	MOVB $1, ret+120(FP)         // Signal success
	RET

fwd_ret_false:
	MOVB $0, ret+120(FP)         // Signal failure
	RET


// ===========================================================================
// Inverse transform, size 16, complex64, AVX2 radix-16 (4x4) variant
// ===========================================================================
TEXT ·InverseAVX2Size16Radix16Complex64Asm(SB), NOSPLIT, $0-121
	// --- Argument Loading ---
	MOVQ dst+0(FP), R8           // R8 = Destination pointer
	MOVQ src+24(FP), R9          // R9 = Source pointer
	MOVQ twiddle+48(FP), R10     // R10 = Twiddle factors pointer
	MOVQ scratch+72(FP), R11     // R11 = Scratch pointer
	MOVQ bitrev+96(FP), R12      // R12 = Bit-reversal pointer
	MOVQ src+32(FP), R13         // R13 = Length

	// --- Input Validation ---
	CMPQ R13, $16                // Verify length is 16
	JNE  inv_ret_false

	// =======================================================================
	// STEP 0: Load 4x4 Matrix
	// =======================================================================
	VMOVUPS 0(R9), Y0             // Y0 = Row 0: elements [0, 1, 2, 3]
	VMOVUPS 32(R9), Y1            // Y1 = Row 1: elements [4, 5, 6, 7]
	VMOVUPS 64(R9), Y2            // Y2 = Row 2: elements [8, 9, 10, 11]
	VMOVUPS 96(R9), Y3            // Y3 = Row 3: elements [12, 13, 14, 15]

	// =======================================================================
	// STEP 1: Vertical IFFT4 (Column-wise transformation)
	// =======================================================================
	VADDPS Y2, Y0, Y4            // T0 = Row0 + Row2
	VADDPS Y3, Y1, Y5            // T1 = Row1 + Row3
	VSUBPS Y2, Y0, Y6            // T2 = Row0 - Row2
	VSUBPS Y3, Y1, Y7            // T3 = Row1 - Row3

	VADDPS Y5, Y4, Y0            // F0 = T0 + T1
	VSUBPS Y5, Y4, Y2            // F2 = T0 - T1

	// Rotation by +i for Inverse structure: (x+iy)*(+i) = -y + ix
	VPERMILPS $0xB1, Y7, Y7      // Swap Real/Imag -> (iy + x) -> (y + ix)
	VMOVUPS ·maskNegLoPS(SB), X8 // Load mask to negate real parts (index 0, 2)
	VINSERTF128 $0x01, X8, Y8, Y8 // Broadcast mask
	VXORPS Y8, Y7, Y7            // Apply mask -> (-y + ix) = T3 * +i

	VADDPS Y7, Y6, Y1            // F1 = T2 + (T3 * +i) = T2 + i*T3
	VSUBPS Y7, Y6, Y3            // F3 = T2 - (T3 * +i) = T2 - i*T3

	// =======================================================================
	// STEP 2: Internal Twiddle Multiplication (Conjugated)
	// =======================================================================
	VMOVUPS ·maskNegHiPS(SB), X15 // Mask to negate imaginary parts for conjugation
	VINSERTF128 $0x01, X15, Y15, Y15 // Broadcast conjugation mask

	// --- Row 1 ---
	VMOVUPS 0(R10), Y4            // Load contiguous twiddles W^0..W^3
	VXORPS Y15, Y4, Y4            // Conjugate twiddles: W -> conj(W)
	VPERMILPS $0xA0, Y4, Y5       // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6       // Y6 = Im(W)
	VMULPS Y5, Y1, Y7             // Y7 = Row1 * Re(W)
	VPERMILPS $0xB1, Y1, Y8       // Y8 = Row1 swapped
	VMULPS Y6, Y8, Y8             // Y8 = Row1_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y1          // Row 1 = (ac-bd, ad+bc)

	// --- Row 2 ---
	VMOVSD 0(R10), X4             // Load W^0
	VMOVSD 16(R10), X5            // Load W^2
	VUNPCKLPD X5, X4, X4          // X4 = (W^0, W^2)
	VMOVSD 32(R10), X5            // Load W^4
	VMOVSD 48(R10), X6            // Load W^6
	VUNPCKLPD X6, X5, X5          // X5 = (W^4, W^6)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^2, W^4, W^6)
	VXORPS Y15, Y4, Y4            // Conjugate
	VPERMILPS $0xA0, Y4, Y5       // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6       // Y6 = Im(W)
	VMULPS Y5, Y2, Y7             // Y7 = Row2 * Re(W)
	VPERMILPS $0xB1, Y2, Y8       // Y8 = Row2 swapped
	VMULPS Y6, Y8, Y8             // Y8 = Row2_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y2          // Row 2 = (ac-bd, ad+bc)

	// --- Row 3 ---
	VMOVSD 0(R10), X4             // Load W^0
	VMOVSD 24(R10), X5            // Load W^3
	VUNPCKLPD X5, X4, X4          // X4 = (W^0, W^3)
	VMOVSD 48(R10), X5            // Load W^6
	VMOVSD 8(R10), X6             // Load W^1
	VXORPS X7, X7, X7             // Zero X7
	VSUBPS X6, X7, X6             // X6 = -W^1 = W^9
	VUNPCKLPD X6, X5, X5          // X5 = (W^6, W^9)
	VINSERTF128 $0x01, X5, Y4, Y4 // Y4 = (W^0, W^3, W^6, W^9)
	VXORPS Y15, Y4, Y4            // Conjugate
	VPERMILPS $0xA0, Y4, Y5       // Y5 = Re(W)
	VPERMILPS $0xF5, Y4, Y6       // Y6 = Im(W)
	VMULPS Y5, Y3, Y7             // Y7 = Row3 * Re(W)
	VPERMILPS $0xB1, Y3, Y8       // Y8 = Row3 swapped
	VMULPS Y6, Y8, Y8             // Y8 = Row3_swapped * Im(W)
	VADDSUBPS Y8, Y7, Y3          // Row 3 = (ac-bd, ad+bc)

	// =======================================================================
	// STEP 3: Horizontal IFFT4 (Transform within each YMM Row)
	// =======================================================================
	VMOVUPS ·maskNegLoPS(SB), X15 // Mask for +i rotation

	// --- Row 0 ---
	VEXTRACTF128 $0x01, Y0, X4 
	VADDPS X4, X0, X5          
	VSUBPS X4, X0, X6          
	VPERMILPS $0x4E, X5, X7    
	VADDPS X7, X5, X8          
	VSUBPS X7, X5, X9          
	VPERMILPS $0x4E, X6, X10   
	VPERMILPS $0xB1, X10, X10  
	VXORPS X15, X10, X10       // +i rotation
	VADDPS X10, X6, X12        // y1 = D0 + i*D1
	VSUBPS X10, X6, X13        // y3 = D0 - i*D1
	VUNPCKLPD X12, X8, X8      
	VUNPCKLPD X13, X9, X9      
	VPERM2F128 $0x20, Y9, Y8, Y0 

	// --- Row 1 ---
	VEXTRACTF128 $0x01, Y1, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X1, X5            // X5 = Sum (x0+x2, x1+x3)
	VSUBPS X4, X1, X6            // X6 = Diff (x0-x2, x1-x3)
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = +i*D1
	VADDPS X10, X6, X12          // y1 = D0 + i*D1
	VSUBPS X10, X6, X13          // y3 = D0 - i*D1
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y1 // Y1 = Row 1 Final (y0, y1, y2, y3)

	// --- Row 2 ---
	VEXTRACTF128 $0x01, Y2, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X2, X5            // X5 = Sum (x0+x2, x1+x3)
	VSUBPS X4, X2, X6            // X6 = Diff (x0-x2, x1-x3)
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = +i*D1
	VADDPS X10, X6, X12          // y1 = D0 + i*D1
	VSUBPS X10, X6, X13          // y3 = D0 - i*D1
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y2 // Y2 = Row 2 Final (y0, y1, y2, y3)

	// --- Row 3 ---
	VEXTRACTF128 $0x01, Y3, X4   // X4 = Lane 1 (x2, x3)
	VADDPS X4, X3, X5            // X5 = Sum (x0+x2, x1+x3)
	VSUBPS X4, X3, X6            // X6 = Diff (x0-x2, x1-x3)
	VPERMILPS $0x4E, X5, X7      // X7 = Swap halves of S
	VADDPS X7, X5, X8            // X8 = (y0, y0)
	VSUBPS X7, X5, X9            // X9 = (y2, y2)
	VPERMILPS $0x4E, X6, X10     // X10 = Swap halves of D
	VPERMILPS $0xB1, X10, X10    // X10 = (Im, Re) for rotation
	VXORPS X15, X10, X10         // X10 = +i*D1
	VADDPS X10, X6, X12          // y1 = D0 + i*D1
	VSUBPS X10, X6, X13          // y3 = D0 - i*D1
	VUNPCKLPD X12, X8, X8        // X8 = (y0, y1)
	VUNPCKLPD X13, X9, X9        // X9 = (y2, y3)
	VPERM2F128 $0x20, Y9, Y8, Y3 // Y3 = Row 3 Final (y0, y1, y2, y3)

	// =======================================================================
	// STEP 4: Matrix Transposition
	// =======================================================================
	VUNPCKLPD Y1, Y0, Y4
	VUNPCKHPD Y1, Y0, Y5
	VUNPCKLPD Y3, Y2, Y6
	VUNPCKHPD Y3, Y2, Y7
	VPERM2F128 $0x20, Y6, Y4, Y0
	VPERM2F128 $0x20, Y7, Y5, Y1
	VPERM2F128 $0x31, Y6, Y4, Y2
	VPERM2F128 $0x31, Y7, Y5, Y3

	// =======================================================================
	// STEP 5: Scaling (1/16)
	// =======================================================================
	VMOVSS ·sixteenth32(SB), X15  // Load 1/16 constant
	VBROADCASTSS X15, Y15         // Broadcast to all elements
	VMULPS Y15, Y0, Y0
	VMULPS Y15, Y1, Y1
	VMULPS Y15, Y2, Y2
	VMULPS Y15, Y3, Y3

	// =======================================================================
	// STEP 6: Store Results and Return
	// =======================================================================
	VMOVUPS Y0, 0(R8)
	VMOVUPS Y1, 32(R8)
	VMOVUPS Y2, 64(R8)
	VMOVUPS Y3, 96(R8)

	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

inv_ret_false:
	MOVB $0, ret+120(FP)
	RET


