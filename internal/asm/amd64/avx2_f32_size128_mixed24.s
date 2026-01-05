//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-128 Mixed-Radix (2×4) FFT Kernels for AMD64 (complex64)
// ===========================================================================
//
// Size 128 = 2 × 4³, implemented as mixed-radix decomposition:
//   Stage 1: 32 radix-4 butterflies, stride=4, twiddle=1 (no multiply)
//   Stage 2: 8 groups × 4 butterflies, stride=16
//   Stage 3: 2 groups × 16 butterflies, stride=64
//   Stage 4: 32 radix-2 butterflies, stride=128
//
// This reduces from 7 stages (radix-2) to 4 stages (mixed radix).
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 128, complex64, mixed-radix 2×4 (AVX2)
TEXT ·ForwardAVX2Size128Mixed24Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R10 // R10 = twiddle pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ bitrev+96(FP), R12  // R12 = bitrev pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 128)

	// Verify n == 128
	CMPQ R13, $128
	JNE  size128_r4_return_false

	// Validate slice lengths (all must be >= 128)
	MOVQ dst+8(FP), AX
	CMPQ AX, $128
	JL   size128_r4_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $128
	JL   size128_r4_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $128
	JL   size128_r4_return_false

	MOVQ bitrev+104(FP), AX
	CMPQ AX, $128
	JL   size128_r4_return_false

	// Select working buffer: use scratch for in-place, dst for out-of-place
	CMPQ R8, R9
	JNE  size128_r4_use_dst
	MOVQ R11, R8             // In-place: use scratch buffer

size128_r4_use_dst:
	// ==================================================================
	// Bit-reversal permutation (copy src[bitrev[i]] -> work[i])
	// Size 128 requires bit-reversal for mixed-radix ordering
	// ==================================================================
	XORQ CX, CX              // CX = index 0..127

size128_r4_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[CX] (bit-reversed index)
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[CX]] (load 8 bytes = complex64)
	MOVQ AX, (R8)(CX*8)      // work[CX] = src[bitrev[CX]]
	INCQ CX                  // CX++
	CMPQ CX, $128            // Loop 128 times
	JL   size128_r4_bitrev_loop

size128_r4_stage1:
	// ==================================================================
	// Stage 1: 32 radix-4 butterflies, stride=4, twiddle=1 (no multiply)
	// Process groups of 4 elements: [0,1,2,3], [4,5,6,7], ..., [124,125,126,127]
	// Each group: 4-point FFT with twiddle factor W^0 = 1
	// ==================================================================
	XORQ CX, CX              // CX = base index (0, 4, 8, ..., 124)

size128_r4_stage1_loop:
	CMPQ CX, $128
	JGE  size128_r4_stage2

	// Load 4 complex64 values from consecutive memory locations
	LEAQ (R8)(CX*8), SI      // SI = base address for this group
	VMOVSD (SI), X0          // X0 = a0 (first element)
	VMOVSD 8(SI), X1         // X1 = a1 (second element)
	VMOVSD 16(SI), X2        // X2 = a2 (third element)
	VMOVSD 24(SI), X3        // X3 = a3 (fourth element)

	// Radix-4 butterfly (twiddle = 1, so no complex multiplies needed)
	// Compute intermediates: t0 = a0 + a2, t1 = a0 - a2, t2 = a1 + a3, t3 = a1 - a3
	VADDPS X0, X2, X4        // X4 = t0 = a0 + a2
	VSUBPS X2, X0, X5        // X5 = t1 = a0 - a2 (note: VSUBPS b,a → dst = a - b)
	VADDPS X1, X3, X6        // X6 = t2 = a1 + a3
	VSUBPS X3, X1, X7        // X7 = t3 = a1 - a3

	// Compute (-i)*t3 for y1 output: (-i)*t3 = (imag(t3), -real(t3))
	VPERMILPS $0xB1, X7, X8  // X8 = (imag(t3), real(t3), imag(t3), real(t3))
	VXORPS X9, X9, X9        // X9 = 0 (zero register)
	VSUBPS X8, X9, X10       // X10 = -X8 = (-imag(t3), -real(t3), -imag(t3), -real(t3))
	VBLENDPS $0x02, X10, X8, X8 // X8 = (imag(t3), -real(t3), imag(t3), -real(t3)) = (-i)*t3

	// Compute i*t3 for y3 output: i*t3 = (-imag(t3), real(t3))
	VPERMILPS $0xB1, X7, X11 // X11 = (imag(t3), real(t3), imag(t3), real(t3))
	VSUBPS X11, X9, X10      // X10 = -X11 = (-imag(t3), -real(t3), -imag(t3), -real(t3))
	VBLENDPS $0x01, X10, X11, X11 // X11 = (-imag(t3), real(t3), -imag(t3), real(t3)) = i*t3

	// Final radix-4 outputs
	VADDPS X4, X6, X0        // X0 = y0 = t0 + t2
	VADDPS X5, X8, X1        // X1 = y1 = t1 + (-i)*t3
	VSUBPS X6, X4, X2        // X2 = y2 = t0 - t2 (note: VSUBPS b,a → dst = a - b)
	VADDPS X5, X11, X3       // X3 = y3 = t1 + i*t3

	// Store results back to the same locations
	VMOVSD X0, (SI)          // Store y0
	VMOVSD X1, 8(SI)         // Store y1
	VMOVSD X2, 16(SI)        // Store y2
	VMOVSD X3, 24(SI)        // Store y3

	ADDQ $4, CX              // Advance to next group (stride = 4)
	JMP  size128_r4_stage1_loop

size128_r4_stage2:
	// ==================================================================
	// Stage 2: 8 groups × 4 butterflies each, stride=16, twiddle step=8
	// Group g processes: indices [g*16 + j], [g*16 + j + 4], [g*16 + j + 8], [g*16 + j + 12]
	// for j = 0,1,2,3 within each group
	// Twiddles: W^{j*8}, W^{j*16}, W^{j*24} for each butterfly j
	// ==================================================================
	XORQ BX, BX              // BX = group index (0..7)

size128_r4_stage2_outer:
	CMPQ BX, $8
	JGE  size128_r4_stage3

	XORQ DX, DX              // DX = butterfly index within group (0..3)

size128_r4_stage2_inner:
	CMPQ DX, $4
	JGE  size128_r4_stage2_next_group

	// Calculate base index: BX*16 + DX
	MOVQ BX, SI
	SHLQ $4, SI              // SI = BX * 16
	ADDQ DX, SI              // SI = base index

	// Load indices for 4-point butterfly
	// idx0 = SI, idx1 = SI+4, idx2 = SI+8, idx3 = SI+12
	MOVQ SI, DI
	ADDQ $4, DI              // idx1
	MOVQ SI, R14
	ADDQ $8, R14             // idx2
	MOVQ SI, R15
	ADDQ $12, R15            // idx3

	// Load twiddle factors: twiddle[DX*8], twiddle[DX*16], twiddle[DX*24]
	// For stage 2 with twiddle step=8: indices are j*8, 2*j*8, 3*j*8 where j=DX
	MOVQ DX, CX
	SHLQ $3, CX              // CX = DX * 8
	VMOVSD (R10)(CX*8), X8   // X8 = w1 = W^{j*8}

	MOVQ DX, CX
	SHLQ $4, CX              // CX = DX * 16
	VMOVSD (R10)(CX*8), X9   // X9 = w2 = W^{j*16}

	MOVQ DX, CX
	IMULQ $24, CX            // CX = DX * 24
	VMOVSD (R10)(CX*8), X10  // X10 = w3 = W^{j*24}

	// Load input data for this butterfly
	VMOVSD (R8)(SI*8), X0    // X0 = a0
	VMOVSD (R8)(DI*8), X1    // X1 = a1
	VMOVSD (R8)(R14*8), X2   // X2 = a2
	VMOVSD (R8)(R15*8), X3   // X3 = a3

	// Complex multiply X1 *= W^{j*8}
	VMOVSLDUP X8, X11        // X11 = broadcast real parts of w1
	VMOVSHDUP X8, X12        // X12 = broadcast imag parts of w1
	VSHUFPS $0xB1, X1, X1, X13 // X13 = swap real/imag of X1
	VMULPS X12, X13, X13     // X13 = imag_w1 * swapped_X1
	VFMADDSUB231PS X11, X1, X13 // X13 ± X11*X1 (fused multiply-add-subtract)
	VMOVAPS X13, X1          // X1 = a1 * w1

	// Complex multiply X2 *= W^{j*16}
	VMOVSLDUP X9, X11        // X11 = broadcast real parts of w2
	VMOVSHDUP X9, X12        // X12 = broadcast imag parts of w2
	VSHUFPS $0xB1, X2, X2, X13 // X13 = swap real/imag of X2
	VMULPS X12, X13, X13     // X13 = imag_w2 * swapped_X2
	VFMADDSUB231PS X11, X2, X13 // Fused
	VMOVAPS X13, X2          // X2 = a2 * w2

	// Complex multiply X3 *= W^{j*24}
	VMOVSLDUP X10, X11       // X11 = broadcast real parts of w3
	VMOVSHDUP X10, X12       // X12 = broadcast imag parts of w3
	VSHUFPS $0xB1, X3, X3, X13 // X13 = swap real/imag of X3
	VMULPS X12, X13, X13     // X13 = imag_w3 * swapped_X3
	VFMADDSUB231PS X11, X3, X13 // Fused
	VMOVAPS X13, X3          // X3 = a3 * w3

	// Radix-4 butterfly
	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// Compute (-i)*t3 for y1 output
	VPERMILPS $0xB1, X7, X14 // X14 = permute t3 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x02, X11, X14, X14 // X14 = (imag, -real, imag, -real) = (-i)*t3

	// Compute i*t3 for y3 output
	VPERMILPS $0xB1, X7, X12 // X12 = permute t3
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x01, X11, X12, X12 // X12 = (-imag, real, -imag, real) = i*t3

	// Final outputs
	VADDPS X4, X6, X0        // X0 = y0 = t0 + t2
	VADDPS X5, X14, X1       // X1 = y1 = t1 + (-i)*t3
	VSUBPS X6, X4, X2        // X2 = y2 = t0 - t2
	VADDPS X5, X12, X3       // X3 = y3 = t1 + i*t3

	// Store results
	VMOVSD X0, (R8)(SI*8)    // Store y0
	VMOVSD X1, (R8)(DI*8)    // Store y1
	VMOVSD X2, (R8)(R14*8)   // Store y2
	VMOVSD X3, (R8)(R15*8)   // Store y3

	INCQ DX
	JMP  size128_r4_stage2_inner

size128_r4_stage2_next_group:
	INCQ BX
	JMP  size128_r4_stage2_outer

size128_r4_stage3:
	// ==================================================================
	// Stage 3: 2 groups × 16 butterflies each, stride=64, twiddle step=2
	// Group g processes: [g*64 + j], [g*64 + j + 16], [g*64 + j + 32], [g*64 + j + 48]
	// for j = 0..15 within each group
	// Twiddles: W^{j*2}, W^{j*4}, W^{j*6} for each butterfly j
	// ==================================================================
	XORQ BX, BX              // BX = group index (0..1)

size128_r4_stage3_outer:
	CMPQ BX, $2
	JGE  size128_r4_stage4

	XORQ DX, DX              // DX = butterfly index (0..15)

size128_r4_stage3_inner:
	CMPQ DX, $16
	JGE  size128_r4_stage3_next_group

	// Calculate base index: BX*64 + DX
	MOVQ BX, SI
	SHLQ $6, SI              // SI = BX * 64
	ADDQ DX, SI              // SI = base index for this butterfly

	// Load indices for 4-point butterfly: SI, SI+16, SI+32, SI+48
	MOVQ SI, DI
	ADDQ $16, DI             // DI = idx1 = SI + 16
	MOVQ SI, R14
	ADDQ $32, R14            // R14 = idx2 = SI + 32
	MOVQ SI, R15
	ADDQ $48, R15            // R15 = idx3 = SI + 48

	// Load twiddle factors: twiddle[DX*2], twiddle[DX*4], twiddle[DX*6]
	// For stage 3 with twiddle step=2: indices are j*2, 2*j*2, 3*j*2 where j=DX
	MOVQ DX, CX
	SHLQ $1, CX              // CX = DX * 2
	VMOVSD (R10)(CX*8), X8   // X8 = w1 = W^{j*2}

	MOVQ DX, CX
	SHLQ $2, CX              // CX = DX * 4
	VMOVSD (R10)(CX*8), X9   // X9 = w2 = W^{j*4}

	MOVQ DX, CX
	IMULQ $6, CX             // CX = DX * 6
	VMOVSD (R10)(CX*8), X10  // X10 = w3 = W^{j*6}

	// Load input data for this butterfly
	VMOVSD (R8)(SI*8), X0    // X0 = a0
	VMOVSD (R8)(DI*8), X1    // X1 = a1
	VMOVSD (R8)(R14*8), X2   // X2 = a2
	VMOVSD (R8)(R15*8), X3   // X3 = a3

	// Complex multiply X1 *= W^{j*2}
	VMOVSLDUP X8, X11        // X11 = broadcast real parts of w1
	VMOVSHDUP X8, X12        // X12 = broadcast imag parts of w1
	VSHUFPS $0xB1, X1, X1, X13 // X13 = swap real/imag of X1
	VMULPS X12, X13, X13     // X13 = imag_w1 * swapped_X1
	VFMADDSUB231PS X11, X1, X13 // X13 ± X11*X1 (fused multiply-add-subtract)
	VMOVAPS X13, X1          // X1 = a1 * w1

	// Complex multiply X2 *= W^{j*4}
	VMOVSLDUP X9, X11        // X11 = broadcast real parts of w2
	VMOVSHDUP X9, X12        // X12 = broadcast imag parts of w2
	VSHUFPS $0xB1, X2, X2, X13 // X13 = swap real/imag of X2
	VMULPS X12, X13, X13     // X13 = imag_w2 * swapped_X2
	VFMADDSUB231PS X11, X2, X13 // Fused
	VMOVAPS X13, X2          // X2 = a2 * w2

	// Complex multiply X3 *= W^{j*6}
	VMOVSLDUP X10, X11       // X11 = broadcast real parts of w3
	VMOVSHDUP X10, X12       // X12 = broadcast imag parts of w3
	VSHUFPS $0xB1, X3, X3, X13 // X13 = swap real/imag of X3
	VMULPS X12, X13, X13     // X13 = imag_w3 * swapped_X3
	VFMADDSUB231PS X11, X3, X13 // Fused
	VMOVAPS X13, X3          // X3 = a3 * w3

	// Radix-4 butterfly intermediates
	VADDPS X0, X2, X4        // X4 = t0 = a0 + a2
	VSUBPS X2, X0, X5        // X5 = t1 = a0 - a2
	VADDPS X1, X3, X6        // X6 = t2 = a1 + a3
	VSUBPS X3, X1, X7        // X7 = t3 = a1 - a3

	// Compute (-i)*t3 for y1 output
	VPERMILPS $0xB1, X7, X14 // X14 = permute t3 (imag, real, imag, real)
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x02, X11, X14, X14 // X14 = (imag, -real, imag, -real) = (-i)*t3

	// Compute i*t3 for y3 output
	VPERMILPS $0xB1, X7, X12 // X12 = permute t3
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x01, X11, X12, X12 // X12 = (-imag, real, -imag, real) = i*t3

	// Final outputs
	VADDPS X4, X6, X0        // X0 = y0 = t0 + t2
	VADDPS X5, X14, X1       // X1 = y1 = t1 + (-i)*t3
	VSUBPS X6, X4, X2        // X2 = y2 = t0 - t2
	VADDPS X5, X12, X3       // X3 = y3 = t1 + i*t3

	// Store results
	VMOVSD X0, (R8)(SI*8)    // Store y0
	VMOVSD X1, (R8)(DI*8)    // Store y1
	VMOVSD X2, (R8)(R14*8)   // Store y2
	VMOVSD X3, (R8)(R15*8)   // Store y3

	INCQ DX
	JMP  size128_r4_stage3_inner

size128_r4_stage3_next_group:
	INCQ BX
	JMP  size128_r4_stage3_outer

size128_r4_stage4:
	// ==================================================================
	// Stage 4: 64 radix-2 butterflies, stride=128, twiddle step=1
	// Each butterfly processes pair: [j, j+64] for j = 0..63
	// Twiddle: W^{j} for each pair j
	// ==================================================================
	XORQ DX, DX              // DX = butterfly index (0..63)

size128_r4_stage4_loop:
	CMPQ DX, $64
	JGE  size128_r4_done

	// Indices for radix-2 pair: DX and DX+64
	MOVQ DX, SI              // SI = idx0 = j
	MOVQ DX, DI
	ADDQ $64, DI             // DI = idx1 = j + 64

	// Load twiddle factor: twiddle[DX] = W^{j}
	VMOVSD (R10)(DX*8), X8   // X8 = w = W^{j}

	// Load input data for this butterfly pair
	VMOVSD (R8)(SI*8), X0    // X0 = a = data[j]
	VMOVSD (R8)(DI*8), X1    // X1 = b = data[j+64]

	// Complex multiply X1 *= W^{j}
	VMOVSLDUP X8, X11        // X11 = broadcast real parts of w
	VMOVSHDUP X8, X12        // X12 = broadcast imag parts of w
	VSHUFPS $0xB1, X1, X1, X13 // X13 = swap real/imag of X1
	VMULPS X12, X13, X13     // X13 = imag_w * swapped_X1
	VFMADDSUB231PS X11, X1, X13 // X13 ± X11*X1 (fused multiply-add-subtract)
	VMOVAPS X13, X1          // X1 = b' = b * w

	// Radix-2 butterfly: (a, b') -> (a + b', a - b')
	VADDPS X0, X1, X2        // X2 = y0 = a + b'
	VSUBPS X1, X0, X3        // X3 = y1 = a - b'  (note: VSUBPS src, dst -> dst = dst - src)

	// Store results
	VMOVSD X2, (R8)(SI*8)    // Store y0 at index j
	VMOVSD X3, (R8)(DI*8)    // Store y1 at index j+64

	INCQ DX
	JMP  size128_r4_stage4_loop

size128_r4_done:
	// Copy results to dst if we used scratch buffer
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size128_r4_done_direct

	XORQ CX, CX

size128_r4_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $1024           // 128 * 8 bytes
	JL   size128_r4_copy_loop

size128_r4_done_direct:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size128_r4_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET

// ===========================================================================
// Inverse transform, size 128, complex64, mixed-radix 2×4 (AVX2)
// Inverse FFT: Forward FFT with conjugated twiddles + 1/N scaling
// ===========================================================================
TEXT ·InverseAVX2Size128Mixed24Complex64Asm(SB), NOSPLIT, $0-121
	// Load parameters: dst, src, twiddle, scratch, bitrev, n
	MOVQ dst+0(FP), R8       // R8 = dst slice data
	MOVQ src+24(FP), R9      // R9 = src slice data
	MOVQ twiddle+48(FP), R10 // R10 = twiddle factors
	MOVQ scratch+72(FP), R11 // R11 = scratch buffer
	MOVQ bitrev+96(FP), R12  // R12 = bit-reversal indices
	MOVQ src+32(FP), R13     // R13 = n (size)

	// Validate n == 128
	CMPQ R13, $128
	JNE  size128_r4_inv_return_false

	// Validate slice lengths (must be >= 128 complex64 elements)
	MOVQ dst+8(FP), AX       // dst len
	CMPQ AX, $128
	JL   size128_r4_inv_return_false

	MOVQ twiddle+56(FP), AX  // twiddle len
	CMPQ AX, $128
	JL   size128_r4_inv_return_false

	MOVQ scratch+80(FP), AX  // scratch len
	CMPQ AX, $128
	JL   size128_r4_inv_return_false

	MOVQ bitrev+104(FP), AX  // bitrev len
	CMPQ AX, $128
	JL   size128_r4_inv_return_false

	// Select working buffer: use dst if in-place, scratch if out-of-place
	CMPQ R8, R9
	JNE  size128_r4_inv_use_dst
	MOVQ R11, R8             // Use scratch buffer for out-of-place transform

size128_r4_inv_use_dst:
	// Bit-reversal permutation: rearrange input data for in-order output
	XORQ CX, CX              // CX = index 0..127

size128_r4_inv_bitrev_loop:
	MOVQ (R12)(CX*8), DX     // DX = bitrev[CX] (permuted index)
	MOVQ (R9)(DX*8), AX      // AX = src[bitrev[CX]]
	MOVQ AX, (R8)(CX*8)      // dst[CX] = src[bitrev[CX]]
	INCQ CX
	CMPQ CX, $128
	JL   size128_r4_inv_bitrev_loop

size128_r4_inv_stage1:
	// ==================================================================
	// Stage 1: 32 radix-4 butterflies, twiddle=1 (inverse: swap i/-i rotations)
	// Inverse FFT uses same butterfly structure but with i/-i rotations swapped
	// ==================================================================
	XORQ CX, CX              // CX = group index (0..31, step by 4)

size128_r4_inv_stage1_loop:
	CMPQ CX, $128
	JGE  size128_r4_inv_stage2

	LEAQ (R8)(CX*8), SI      // SI = base address for this group
	VMOVSD (SI), X0          // X0 = a0
	VMOVSD 8(SI), X1         // X1 = a1
	VMOVSD 16(SI), X2        // X2 = a2
	VMOVSD 24(SI), X3        // X3 = a3

	// Radix-4 butterfly intermediates (same as forward)
	VADDPS X0, X2, X4        // X4 = t0 = a0 + a2
	VSUBPS X2, X0, X5        // X5 = t1 = a0 - a2
	VADDPS X1, X3, X6        // X6 = t2 = a1 + a3
	VSUBPS X3, X1, X7        // X7 = t3 = a1 - a3

	// For inverse: use i*t3 for y1, (-i)*t3 for y3 (swapped from forward)
	// i*t3 = (-imag, real) permutation
	VPERMILPS $0xB1, X7, X11 // X11 = permute t3 (imag, real, imag, real)
	VXORPS X9, X9, X9        // X9 = 0
	VSUBPS X11, X9, X10      // X10 = -X11
	VBLENDPS $0x01, X10, X11, X11 // X11 = (-imag, real, -imag, real) = i*t3

	// (-i)*t3 = (imag, -real) permutation
	VPERMILPS $0xB1, X7, X8  // X8 = permute t3
	VSUBPS X8, X9, X10       // X10 = -X8
	VBLENDPS $0x02, X10, X8, X8 // X8 = (imag, -real, imag, -real) = (-i)*t3

	// Final outputs
	VADDPS X4, X6, X0        // X0 = y0 = t0 + t2
	VADDPS X5, X11, X1       // X1 = y1 = t1 + i*t3
	VSUBPS X6, X4, X2        // X2 = y2 = t0 - t2
	VADDPS X5, X8, X3        // X3 = y3 = t1 + (-i)*t3

	// Store results
	VMOVSD X0, (SI)          // Store y0
	VMOVSD X1, 8(SI)         // Store y1
	VMOVSD X2, 16(SI)        // Store y2
	VMOVSD X3, 24(SI)        // Store y3

	ADDQ $4, CX              // Next group (stride = 4)
	JMP  size128_r4_inv_stage1_loop

size128_r4_inv_stage2:
	// ==================================================================
	// Stage 2: 8 groups × 4 butterflies each, stride=16 (conjugated twiddles)
	// Inverse FFT uses conjugate of forward twiddles: W^{-k} = conjugate(W^{k})
	// ==================================================================
	XORQ BX, BX

size128_r4_inv_stage2_outer:
	CMPQ BX, $8
	JGE  size128_r4_inv_stage3

	XORQ DX, DX

size128_r4_inv_stage2_inner:
	CMPQ DX, $4
	JGE  size128_r4_inv_stage2_next

	MOVQ BX, SI
	SHLQ $4, SI
	ADDQ DX, SI

	MOVQ SI, DI
	ADDQ $4, DI
	MOVQ SI, R14
	ADDQ $8, R14
	MOVQ SI, R15
	ADDQ $12, R15

	// Load twiddle factors (same as forward, but will conjugate in multiply)
	MOVQ DX, CX
	SHLQ $3, CX
	VMOVSD (R10)(CX*8), X8   // X8 = w1 = W^{j*8}

	MOVQ DX, CX
	SHLQ $4, CX
	VMOVSD (R10)(CX*8), X9   // X9 = w2 = W^{j*16}

	MOVQ DX, CX
	IMULQ $24, CX
	VMOVSD (R10)(CX*8), X10  // X10 = w3 = W^{j*24}

	// Load input data for this butterfly
	VMOVSD (R8)(SI*8), X0    // X0 = a0
	VMOVSD (R8)(DI*8), X1    // X1 = a1
	VMOVSD (R8)(R14*8), X2   // X2 = a2
	VMOVSD (R8)(R15*8), X3   // X3 = a3

	// Conjugate complex multiply: X1 *= conjugate(W^{j*8})
	// For inverse FFT: multiply by conjugate of twiddle
	VMOVSLDUP X8, X11        // X11 = broadcast real(w1)
	VMOVSHDUP X8, X12        // X12 = broadcast imag(w1)
	VSHUFPS $0xB1, X1, X1, X13 // X13 = swap real/imag of X1
	VMULPS X12, X13, X13     // X13 = imag(w1) * swapped_X1
	VFMSUBADD231PS X11, X1, X13 // X13 = X11*X1 - X13 (conjugate multiply)
	VMOVAPS X13, X1          // X1 = a1 * conjugate(w1)

	// Conjugate multiply X2 *= conjugate(W^{j*16})
	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X2, X13
	VMOVAPS X13, X2

	// Conjugate multiply X3 *= conjugate(W^{j*24})
	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X3, X13
	VMOVAPS X13, X3

	// Radix-4 butterfly (inverse: same structure as forward)
	VADDPS X0, X2, X4        // X4 = t0 = a0 + a2
	VSUBPS X2, X0, X5        // X5 = t1 = a0 - a2
	VADDPS X1, X3, X6        // X6 = t2 = a1 + a3
	VSUBPS X3, X1, X7        // X7 = t3 = a1 - a3

	// Inverse butterfly rotations: i*t3 for y1, (-i)*t3 for y3
	VPERMILPS $0xB1, X7, X12 // X12 = permute t3
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x01, X11, X12, X12 // X12 = (-imag, real, -imag, real) = i*t3

	// (-i)*t3 for y3
	VPERMILPS $0xB1, X7, X14 // X14 = permute t3
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x02, X11, X14, X14 // X14 = (imag, -real, imag, -real) = (-i)*t3

	// Final outputs
	VADDPS X4, X6, X0        // X0 = y0 = t0 + t2
	VADDPS X5, X12, X1       // X1 = y1 = t1 + i*t3
	VSUBPS X6, X4, X2        // X2 = y2 = t0 - t2
	VADDPS X5, X14, X3       // X3 = y3 = t1 + (-i)*t3

	// Store results
	VMOVSD X0, (R8)(SI*8)    // Store y0
	VMOVSD X1, (R8)(DI*8)    // Store y1
	VMOVSD X2, (R8)(R14*8)   // Store y2
	VMOVSD X3, (R8)(R15*8)   // Store y3

	INCQ DX
	JMP  size128_r4_inv_stage2_inner

size128_r4_inv_stage2_next:
	INCQ BX
	JMP  size128_r4_inv_stage2_outer

size128_r4_inv_stage3:
	// ==================================================================
	// Stage 3: 2 groups × 16 butterflies each, stride=64 (conjugated twiddles)
	// Inverse FFT uses conjugate of forward twiddles: W^{-k} = conjugate(W^{k})
	// ==================================================================
	XORQ BX, BX

size128_r4_inv_stage3_outer:
	CMPQ BX, $2
	JGE  size128_r4_inv_stage4

	XORQ DX, DX

size128_r4_inv_stage3_inner:
	CMPQ DX, $16
	JGE  size128_r4_inv_stage3_next

	// Calculate base index: BX*64 + DX
	MOVQ BX, SI
	SHLQ $6, SI              // SI = BX * 64
	ADDQ DX, SI              // SI = base index for this butterfly

	// Load indices for 4-point butterfly: SI, SI+16, SI+32, SI+48
	MOVQ SI, DI
	ADDQ $16, DI             // DI = idx1 = SI + 16
	MOVQ SI, R14
	ADDQ $32, R14            // R14 = idx2 = SI + 32
	MOVQ SI, R15
	ADDQ $48, R15            // R15 = idx3 = SI + 48

	// Load twiddle factors: twiddle[DX*2], twiddle[DX*4], twiddle[DX*6]
	// For inverse stage 3 with twiddle step=2: indices are j*2, 2*j*2, 3*j*2 where j=DX
	MOVQ DX, CX
	SHLQ $1, CX              // CX = DX * 2
	VMOVSD (R10)(CX*8), X8   // X8 = w1 = W^{j*2}

	MOVQ DX, CX
	SHLQ $2, CX              // CX = DX * 4
	VMOVSD (R10)(CX*8), X9   // X9 = w2 = W^{j*4}

	MOVQ DX, CX
	IMULQ $6, CX             // CX = DX * 6
	VMOVSD (R10)(CX*8), X10  // X10 = w3 = W^{j*6}

	// Load input data for this butterfly
	VMOVSD (R8)(SI*8), X0    // X0 = a0
	VMOVSD (R8)(DI*8), X1    // X1 = a1
	VMOVSD (R8)(R14*8), X2   // X2 = a2
	VMOVSD (R8)(R15*8), X3   // X3 = a3

	// Conjugate complex multiply X1 *= conjugate(W^{j*2})
	VMOVSLDUP X8, X11        // X11 = broadcast real(w1)
	VMOVSHDUP X8, X12        // X12 = broadcast imag(w1)
	VSHUFPS $0xB1, X1, X1, X13 // X13 = swap real/imag of X1
	VMULPS X12, X13, X13     // X13 = imag(w1) * swapped_X1
	VFMSUBADD231PS X11, X1, X13 // X13 = X11*X1 - X13 (conjugate multiply)
	VMOVAPS X13, X1          // X1 = a1 * conjugate(w1)

	// Conjugate multiply X2 *= conjugate(W^{j*4})
	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X2, X13
	VMOVAPS X13, X2

	// Conjugate multiply X3 *= conjugate(W^{j*6})
	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMSUBADD231PS X11, X3, X13
	VMOVAPS X13, X3

	// Radix-4 butterfly (inverse: same structure as forward)
	VADDPS X0, X2, X4        // X4 = t0 = a0 + a2
	VSUBPS X2, X0, X5        // X5 = t1 = a0 - a2
	VADDPS X1, X3, X6        // X6 = t2 = a1 + a3
	VSUBPS X3, X1, X7        // X7 = t3 = a1 - a3

	// Inverse butterfly rotations: i*t3 for y1, (-i)*t3 for y3
	VPERMILPS $0xB1, X7, X12 // X12 = permute t3
	VXORPS X15, X15, X15     // X15 = 0
	VSUBPS X12, X15, X11     // X11 = -X12
	VBLENDPS $0x01, X11, X12, X12 // X12 = (-imag, real, -imag, real) = i*t3

	// (-i)*t3 for y3
	VPERMILPS $0xB1, X7, X14 // X14 = permute t3
	VSUBPS X14, X15, X11     // X11 = -X14
	VBLENDPS $0x02, X11, X14, X14 // X14 = (imag, -real, imag, -real) = (-i)*t3

	// Final outputs
	VADDPS X4, X6, X0        // X0 = y0 = t0 + t2
	VADDPS X5, X12, X1       // X1 = y1 = t1 + i*t3
	VSUBPS X6, X4, X2        // X2 = y2 = t0 - t2
	VADDPS X5, X14, X3       // X3 = y3 = t1 + (-i)*t3

	// Store results
	VMOVSD X0, (R8)(SI*8)    // Store y0
	VMOVSD X1, (R8)(DI*8)    // Store y1
	VMOVSD X2, (R8)(R14*8)   // Store y2
	VMOVSD X3, (R8)(R15*8)   // Store y3

	INCQ DX
	JMP  size128_r4_inv_stage3_inner

size128_r4_inv_stage3_next:
	INCQ BX
	JMP  size128_r4_inv_stage3_outer

size128_r4_inv_stage4:
	// ==================================================================
	// Stage 4: 64 radix-2 butterflies, stride=128 (conjugated twiddles)
	// Inverse FFT uses conjugate of forward twiddles: W^{-k} = conjugate(W^{k})
	// ==================================================================
	XORQ DX, DX

size128_r4_inv_stage4_loop:
	CMPQ DX, $64
	JGE  size128_r4_inv_scale

	// Indices for radix-2 pair: DX and DX+64
	MOVQ DX, SI              // SI = idx0 = j
	MOVQ DX, DI
	ADDQ $64, DI             // DI = idx1 = j + 64

	// Load twiddle factor: W^{j} (will be conjugated in multiply)
	VMOVSD (R10)(DX*8), X8   // X8 = w = W^{j}

	// Load input data for this butterfly pair
	VMOVSD (R8)(SI*8), X0    // X0 = a = data[j]
	VMOVSD (R8)(DI*8), X1    // X1 = b = data[j+64]

	// Conjugate complex multiply: X1 *= conjugate(W^{j})
	VMOVSLDUP X8, X11        // X11 = broadcast real(w)
	VMOVSHDUP X8, X12        // X12 = broadcast imag(w)
	VSHUFPS $0xB1, X1, X1, X13 // X13 = swap real/imag of X1
	VMULPS X12, X13, X13     // X13 = imag(w) * swapped_X1
	VFMSUBADD231PS X11, X1, X13 // X13 = X11*X1 - X13 (conjugate multiply)
	VMOVAPS X13, X1          // X1 = b' = b * conjugate(w)

	// Radix-2 butterfly: (a, b') -> (a + b', a - b')
	VADDPS X0, X1, X2        // X2 = y0 = a + b'
	VSUBPS X1, X0, X3        // X3 = y1 = a - b'  (note: VSUBPS src, dst -> dst = dst - src)

	// Store results
	VMOVSD X2, (R8)(SI*8)    // Store y0 at index j
	VMOVSD X3, (R8)(DI*8)    // Store y1 at index j+64

	INCQ DX
	JMP  size128_r4_inv_stage4_loop

size128_r4_inv_scale:
	// ==================================================================
	// Apply 1/N scaling for inverse FFT: multiply all outputs by 1/128 = 0.0078125
	// This completes the inverse FFT: IFFT(x) = conjugate(FFT(conjugate(x))) / N
	// ==================================================================
	MOVL ·oneTwentyEighth32(SB), AX     // Load 1/128 constant (0.0078125 in float32)
	MOVD AX, X8               // Move to XMM register
	VBROADCASTSS X8, Y8       // Broadcast to YMM register for vector scaling

	XORQ CX, CX              // CX = byte offset (0, 32, 64, ...)

size128_r4_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y0   // Load 8 complex64 values (256 bits)
	VMULPS Y8, Y0, Y0        // Scale by 1/128
	VMOVUPS Y0, (R8)(CX*1)   // Store scaled values
	ADDQ $32, CX             // Advance 32 bytes (8 complex64 elements)
	CMPQ CX, $1024           // 128 complex64 * 8 bytes = 1024 bytes total
	JL   size128_r4_inv_scale_loop

	// Copy to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   size128_r4_inv_done

	XORQ CX, CX

size128_r4_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS Y0, (R9)(CX*1)
	ADDQ $32, CX
	CMPQ CX, $1024
	JL   size128_r4_inv_copy_loop

size128_r4_inv_done:
	VZEROUPPER
	MOVB $1, ret+120(FP)
	RET

size128_r4_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+120(FP)
	RET
