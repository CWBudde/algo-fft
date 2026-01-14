//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Size-8192 Mixed-Radix-2/4 FFT Kernels with Pre-Prepared Twiddle Data
// ===========================================================================
//
// This file contains the twiddle-extra variant of the mixed-radix-2/4 DIT FFT
// for size 8192. The twiddle buffer contains pre-broadcast twiddle factors,
// eliminating runtime index computation and scalar-to-vector broadcasts.
//
// Twiddle-extra layout (131,008 bytes total):
//   Stage 2: 4 butterflies × 48 bytes = 192 bytes (offsets 0-191)
//   Stage 3: 16 butterflies × 48 bytes = 768 bytes (offsets 192-959)
//   Stage 4: 64 butterflies × 48 bytes = 3,072 bytes (offsets 960-4,031)
//   Stage 5: 256 butterflies × 48 bytes = 12,288 bytes (offsets 4,032-16,319)
//   Stage 6: 1024 butterflies × 48 bytes = 49,152 bytes (offsets 16,320-65,471)
//   Stage 7: 4096 butterflies × 16 bytes = 65,536 bytes (offsets 65,472-131,007)
//
// Per radix-4 butterfly (48 bytes):
//   [w1.r, w1.r, w1.i, w1.i] 16 bytes
//   [w2.r, w2.r, w2.i, w2.i] 16 bytes
//   [w3.r, w3.r, w3.i, w3.i] 16 bytes
//
// Per radix-2 butterfly (16 bytes):
//   [w.r, w.r, w.i, w.i] 16 bytes
//
// ===========================================================================

#include "textflag.h"

// Stage offsets within twiddle-extra buffer
#define PARAMS_STAGE2_OFFSET 0
#define PARAMS_STAGE3_OFFSET 192
#define PARAMS_STAGE4_OFFSET 960
#define PARAMS_STAGE5_OFFSET 4032
#define PARAMS_STAGE6_OFFSET 16320
#define PARAMS_STAGE7_OFFSET 65472

// Bytes per butterfly
#define BYTES_PER_RADIX4 48
#define BYTES_PER_RADIX2 16

TEXT ·ForwardAVX2Size8192Mixed24ParamsComplex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R14 // R14 = twiddle-extra pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8192)
	LEAQ ·bitrev8192_m24(SB), R12

	// Verify n == 8192
	CMPQ R13, $8192
	JNE  m24p_8192_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $8192
	JL   m24p_8192_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16376
	JL   m24p_8192_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8192
	JL   m24p_8192_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24p_8192_use_dst
	MOVQ R11, R8             // In-place: use scratch

m24p_8192_use_dst:
	// ==================================================================
	// Stage 1: 2048 radix-4 butterflies with mixed-radix bit-reversal
	// No extra twiddles needed - all twiddles are 1
	// ==================================================================
	XORQ CX, CX              // CX = base offset

m24p_8192_stage1_loop:
	CMPQ CX, $8192
	JGE  m24p_8192_stage2

	// Load bit-reversed indices
	MOVQ (R12)(CX*8), DX
	MOVQ 8(R12)(CX*8), SI
	MOVQ 16(R12)(CX*8), DI
	MOVQ 24(R12)(CX*8), R15

	// Load input values
	VMOVSD (R9)(DX*8), X0
	VMOVSD (R9)(SI*8), X1
	VMOVSD (R9)(DI*8), X2
	VMOVSD (R9)(R15*8), X3

	// Radix-4 butterfly (twiddle=1)
	VADDPS X0, X2, X4        // t0 = a0 + a2
	VSUBPS X2, X0, X5        // t1 = a0 - a2
	VADDPS X1, X3, X6        // t2 = a1 + a3
	VSUBPS X3, X1, X7        // t3 = a1 - a3

	// (-i)*t3 for y1: swap and negate real
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x02, X10, X8, X8

	// i*t3 for y3: swap and negate imag
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x01, X10, X11, X11

	VADDPS X4, X6, X0        // y0 = t0 + t2
	VADDPS X5, X8, X1        // y1 = t1 + (-i)*t3
	VSUBPS X6, X4, X2        // y2 = t0 - t2
	VADDPS X5, X11, X3       // y3 = t1 + i*t3

	// Store outputs to work buffer
	VMOVSD X0, (R8)(CX*8)
	VMOVSD X1, 8(R8)(CX*8)
	VMOVSD X2, 16(R8)(CX*8)
	VMOVSD X3, 24(R8)(CX*8)

	ADDQ $4, CX
	JMP  m24p_8192_stage1_loop

m24p_8192_stage2:
	// ==================================================================
	// Stage 2: 512 groups, each with 4 butterflies
	// Twiddle-extra: 4 butterflies × 48 bytes = 192 bytes at offset 0
	// ==================================================================
	XORQ CX, CX              // CX = outer loop counter (data offset)
	XORQ R15, R15            // R15 = twiddle-extra offset for stage 2

m24p_8192_stage2_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_stage3

	XORQ DX, DX              // DX = inner loop (j = 0..3)
	XORQ R15, R15            // Reset twiddle-extra offset for each group

m24p_8192_stage2_inner:
	CMPQ DX, $4
	JGE  m24p_8192_stage2_next

	// Calculate data indices
	MOVQ CX, BX
	ADDQ DX, BX              // BX = base index
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R13

	// Load data
	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	// Load pre-broadcast twiddles from twiddle-extra (eliminates index computation + broadcast)
	// w1: twiddle[R15+0..15], w2: twiddle[R15+16..31], w3: twiddle[R15+32..47]
	VMOVAPS (R14)(R15*1), X8       // w1 = [r, r, i, i]
	VMOVAPS 16(R14)(R15*1), X9     // w2 = [r, r, i, i]
	VMOVAPS 32(R14)(R15*1), X10    // w3 = [r, r, i, i]

	// Complex multiply a1*w1: X1 = X1 * X8
	// X8 = [w1.r, w1.r, w1.i, w1.i]
	VMOVSLDUP X8, X11        // [w1.r, w1.r, w1.r, w1.r]
	VMOVSHDUP X8, X12        // [w1.i, w1.i, w1.i, w1.i]
	VSHUFPS $0xB1, X1, X1, X13  // [b.i, b.r, ...]
	VMULPS X12, X13, X13     // [b.i*w.i, b.r*w.i, ...]
	VFMADDSUB231PS X11, X1, X13  // [b.r*w.r ± b.i*w.i, ...]
	VMOVAPS X13, X1

	// Complex multiply a2*w2: X2 = X2 * X9
	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	// Complex multiply a3*w3: X3 = X3 * X10
	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	// Radix-4 butterfly
	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// (-i)*t3 for y1
	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	// i*t3 for y3
	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	// Store results
	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15  // Advance twiddle-extra offset
	INCQ DX
	JMP  m24p_8192_stage2_inner

m24p_8192_stage2_next:
	ADDQ $16, CX
	JMP  m24p_8192_stage2_outer

m24p_8192_stage3:
	// ==================================================================
	// Stage 3: 128 groups, each with 16 butterflies
	// Twiddle-extra: 16 butterflies × 48 bytes = 768 bytes at offset 192
	// ==================================================================
	XORQ CX, CX

m24p_8192_stage3_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_stage4

	XORQ DX, DX
	MOVQ $PARAMS_STAGE3_OFFSET, R15  // Reset to stage 3 twiddle-extra

m24p_8192_stage3_inner:
	CMPQ DX, $16
	JGE  m24p_8192_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R13

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	// Load pre-broadcast twiddles
	VMOVAPS (R14)(R15*1), X8
	VMOVAPS 16(R14)(R15*1), X9
	VMOVAPS 32(R14)(R15*1), X10

	// Complex multiply a1*w1
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	// Complex multiply a2*w2
	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	// Complex multiply a3*w3
	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15
	INCQ DX
	JMP  m24p_8192_stage3_inner

m24p_8192_stage3_next:
	ADDQ $64, CX
	JMP  m24p_8192_stage3_outer

m24p_8192_stage4:
	// ==================================================================
	// Stage 4: 32 groups, each with 64 butterflies
	// Twiddle-extra: 64 butterflies × 48 bytes = 3072 bytes at offset 960
	// ==================================================================
	XORQ CX, CX

m24p_8192_stage4_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_stage5

	XORQ DX, DX
	MOVQ $PARAMS_STAGE4_OFFSET, R15

m24p_8192_stage4_inner:
	CMPQ DX, $64
	JGE  m24p_8192_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R13

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	VMOVAPS (R14)(R15*1), X8
	VMOVAPS 16(R14)(R15*1), X9
	VMOVAPS 32(R14)(R15*1), X10

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15
	INCQ DX
	JMP  m24p_8192_stage4_inner

m24p_8192_stage4_next:
	ADDQ $256, CX
	JMP  m24p_8192_stage4_outer

m24p_8192_stage5:
	// ==================================================================
	// Stage 5: 8 groups, each with 256 butterflies
	// Twiddle-extra: 256 butterflies × 48 bytes = 12288 bytes at offset 4032
	// ==================================================================
	XORQ CX, CX

m24p_8192_stage5_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_stage6

	XORQ DX, DX
	MOVQ $PARAMS_STAGE5_OFFSET, R15

m24p_8192_stage5_inner:
	CMPQ DX, $256
	JGE  m24p_8192_stage5_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 256(BX), SI
	LEAQ 512(BX), DI
	LEAQ 768(BX), R13

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	VMOVAPS (R14)(R15*1), X8
	VMOVAPS 16(R14)(R15*1), X9
	VMOVAPS 32(R14)(R15*1), X10

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15
	INCQ DX
	JMP  m24p_8192_stage5_inner

m24p_8192_stage5_next:
	ADDQ $1024, CX
	JMP  m24p_8192_stage5_outer

m24p_8192_stage6:
	// ==================================================================
	// Stage 6: 2 groups, each with 1024 butterflies
	// Twiddle-extra: 1024 butterflies × 48 bytes = 49152 bytes at offset 16320
	// ==================================================================
	XORQ CX, CX

m24p_8192_stage6_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_stage7

	XORQ DX, DX
	MOVQ $PARAMS_STAGE6_OFFSET, R15

m24p_8192_stage6_inner:
	CMPQ DX, $1024
	JGE  m24p_8192_stage6_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 1024(BX), SI
	LEAQ 2048(BX), DI
	LEAQ 3072(BX), R13

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	VMOVAPS (R14)(R15*1), X8
	VMOVAPS 16(R14)(R15*1), X9
	VMOVAPS 32(R14)(R15*1), X10

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x02, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x01, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15
	INCQ DX
	JMP  m24p_8192_stage6_inner

m24p_8192_stage6_next:
	ADDQ $4096, CX
	JMP  m24p_8192_stage6_outer

m24p_8192_stage7:
	// ==================================================================
	// Stage 7: radix-2 final stage, 4096 butterflies
	// Twiddle-extra: 4096 butterflies × 16 bytes = 65536 bytes at offset 65472
	// ==================================================================
	XORQ CX, CX
	MOVQ $PARAMS_STAGE7_OFFSET, R15

m24p_8192_stage7_loop:
	CMPQ CX, $4096
	JGE  m24p_8192_forward_done

	MOVQ CX, BX
	LEAQ 4096(BX), SI

	VMOVSD (R8)(BX*8), X0    // a
	VMOVSD (R8)(SI*8), X1    // b

	// Load pre-broadcast twiddle
	VMOVAPS (R14)(R15*1), X8

	// b = b * w
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VADDPS X0, X1, X2        // a + b*w
	VSUBPS X1, X0, X3        // a - b*w

	VMOVSD X2, (R8)(BX*8)
	VMOVSD X3, (R8)(SI*8)

	ADDQ $BYTES_PER_RADIX2, R15
	INCQ CX
	JMP  m24p_8192_stage7_loop

m24p_8192_forward_done:
	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   m24p_8192_forward_ret

	XORQ CX, CX

m24p_8192_forward_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $65536
	JL   m24p_8192_forward_copy_loop

m24p_8192_forward_ret:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

m24p_8192_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET

// ===========================================================================
// Inverse transform, size 8192, complex64, mixed-radix-2/4 with twiddle-extra
// ===========================================================================
TEXT ·InverseAVX2Size8192Mixed24ParamsComplex64Asm(SB), NOSPLIT, $0-97
	// Load parameters
	MOVQ dst+0(FP), R8       // R8  = dst pointer
	MOVQ src+24(FP), R9      // R9  = src pointer
	MOVQ twiddle+48(FP), R14 // R14 = twiddle-extra pointer
	MOVQ scratch+72(FP), R11 // R11 = scratch pointer
	MOVQ src+32(FP), R13     // R13 = n (should be 8192)
	LEAQ ·bitrev8192_m24(SB), R12

	// Verify n == 8192
	CMPQ R13, $8192
	JNE  m24p_8192_inv_return_false

	// Validate slice lengths
	MOVQ dst+8(FP), AX
	CMPQ AX, $8192
	JL   m24p_8192_inv_return_false

	MOVQ twiddle+56(FP), AX
	CMPQ AX, $16376
	JL   m24p_8192_inv_return_false

	MOVQ scratch+80(FP), AX
	CMPQ AX, $8192
	JL   m24p_8192_inv_return_false

	// Select working buffer
	CMPQ R8, R9
	JNE  m24p_8192_inv_use_dst
	MOVQ R11, R8             // In-place: use scratch

m24p_8192_inv_use_dst:
	// ==================================================================
	// Stage 1: 2048 radix-4 butterflies with mixed-radix bit-reversal
	// Inverse uses i*t3 for y1, (-i)*t3 for y3 (swapped from forward)
	// ==================================================================
	XORQ CX, CX

m24p_8192_inv_stage1_loop:
	CMPQ CX, $8192
	JGE  m24p_8192_inv_stage2

	MOVQ (R12)(CX*8), DX
	MOVQ 8(R12)(CX*8), SI
	MOVQ 16(R12)(CX*8), DI
	MOVQ 24(R12)(CX*8), R15

	VMOVSD (R9)(DX*8), X0
	VMOVSD (R9)(SI*8), X1
	VMOVSD (R9)(DI*8), X2
	VMOVSD (R9)(R15*8), X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// i*t3 for y1 (inverse)
	VPERMILPS $0xB1, X7, X8
	VXORPS X9, X9, X9
	VSUBPS X8, X9, X10
	VBLENDPS $0x01, X10, X8, X8

	// (-i)*t3 for y3 (inverse)
	VPERMILPS $0xB1, X7, X11
	VSUBPS X11, X9, X10
	VBLENDPS $0x02, X10, X11, X11

	VADDPS X4, X6, X0
	VADDPS X5, X8, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X11, X3

	VMOVSD X0, (R8)(CX*8)
	VMOVSD X1, 8(R8)(CX*8)
	VMOVSD X2, 16(R8)(CX*8)
	VMOVSD X3, 24(R8)(CX*8)

	ADDQ $4, CX
	JMP  m24p_8192_inv_stage1_loop

m24p_8192_inv_stage2:
	// ==================================================================
	// Stage 2: 512 groups, each with 4 butterflies (inverse)
	// Uses VFMSUBADD for conjugate twiddle multiplication
	// ==================================================================
	XORQ CX, CX

m24p_8192_inv_stage2_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_inv_stage3

	XORQ DX, DX
	XORQ R15, R15

m24p_8192_inv_stage2_inner:
	CMPQ DX, $4
	JGE  m24p_8192_inv_stage2_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 4(BX), SI
	LEAQ 8(BX), DI
	LEAQ 12(BX), R13

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	VMOVAPS (R14)(R15*1), X8
	VMOVAPS 16(R14)(R15*1), X9
	VMOVAPS 32(R14)(R15*1), X10

	// Complex multiply with pre-negated twiddles (use VFMADDSUB, same as forward)
	// Twiddle-extra already has imaginary parts negated for inverse
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	// i*t3 for y1 (inverse)
	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x01, X11, X14, X14

	// (-i)*t3 for y3 (inverse)
	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x02, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15
	INCQ DX
	JMP  m24p_8192_inv_stage2_inner

m24p_8192_inv_stage2_next:
	ADDQ $16, CX
	JMP  m24p_8192_inv_stage2_outer

m24p_8192_inv_stage3:
	// ==================================================================
	// Stage 3: 128 groups, each with 16 butterflies (inverse)
	// ==================================================================
	XORQ CX, CX

m24p_8192_inv_stage3_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_inv_stage4

	XORQ DX, DX
	MOVQ $PARAMS_STAGE3_OFFSET, R15

m24p_8192_inv_stage3_inner:
	CMPQ DX, $16
	JGE  m24p_8192_inv_stage3_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 16(BX), SI
	LEAQ 32(BX), DI
	LEAQ 48(BX), R13

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	VMOVAPS (R14)(R15*1), X8
	VMOVAPS 16(R14)(R15*1), X9
	VMOVAPS 32(R14)(R15*1), X10

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x01, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x02, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15
	INCQ DX
	JMP  m24p_8192_inv_stage3_inner

m24p_8192_inv_stage3_next:
	ADDQ $64, CX
	JMP  m24p_8192_inv_stage3_outer

m24p_8192_inv_stage4:
	// ==================================================================
	// Stage 4: 32 groups, each with 64 butterflies (inverse)
	// ==================================================================
	XORQ CX, CX

m24p_8192_inv_stage4_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_inv_stage5

	XORQ DX, DX
	MOVQ $PARAMS_STAGE4_OFFSET, R15

m24p_8192_inv_stage4_inner:
	CMPQ DX, $64
	JGE  m24p_8192_inv_stage4_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 64(BX), SI
	LEAQ 128(BX), DI
	LEAQ 192(BX), R13

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	VMOVAPS (R14)(R15*1), X8
	VMOVAPS 16(R14)(R15*1), X9
	VMOVAPS 32(R14)(R15*1), X10

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x01, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x02, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15
	INCQ DX
	JMP  m24p_8192_inv_stage4_inner

m24p_8192_inv_stage4_next:
	ADDQ $256, CX
	JMP  m24p_8192_inv_stage4_outer

m24p_8192_inv_stage5:
	// ==================================================================
	// Stage 5: 8 groups, each with 256 butterflies (inverse)
	// ==================================================================
	XORQ CX, CX

m24p_8192_inv_stage5_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_inv_stage6

	XORQ DX, DX
	MOVQ $PARAMS_STAGE5_OFFSET, R15

m24p_8192_inv_stage5_inner:
	CMPQ DX, $256
	JGE  m24p_8192_inv_stage5_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 256(BX), SI
	LEAQ 512(BX), DI
	LEAQ 768(BX), R13

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	VMOVAPS (R14)(R15*1), X8
	VMOVAPS 16(R14)(R15*1), X9
	VMOVAPS 32(R14)(R15*1), X10

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x01, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x02, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15
	INCQ DX
	JMP  m24p_8192_inv_stage5_inner

m24p_8192_inv_stage5_next:
	ADDQ $1024, CX
	JMP  m24p_8192_inv_stage5_outer

m24p_8192_inv_stage6:
	// ==================================================================
	// Stage 6: 2 groups, each with 1024 butterflies (inverse)
	// ==================================================================
	XORQ CX, CX

m24p_8192_inv_stage6_outer:
	CMPQ CX, $8192
	JGE  m24p_8192_inv_stage7

	XORQ DX, DX
	MOVQ $PARAMS_STAGE6_OFFSET, R15

m24p_8192_inv_stage6_inner:
	CMPQ DX, $1024
	JGE  m24p_8192_inv_stage6_next

	MOVQ CX, BX
	ADDQ DX, BX
	LEAQ 1024(BX), SI
	LEAQ 2048(BX), DI
	LEAQ 3072(BX), R13

	VMOVSD (R8)(BX*8), X0
	VMOVSD (R8)(SI*8), X1
	VMOVSD (R8)(DI*8), X2
	VMOVSD (R8)(R13*8), X3

	VMOVAPS (R14)(R15*1), X8
	VMOVAPS 16(R14)(R15*1), X9
	VMOVAPS 32(R14)(R15*1), X10

	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VMOVSLDUP X9, X11
	VMOVSHDUP X9, X12
	VSHUFPS $0xB1, X2, X2, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X2, X13
	VMOVAPS X13, X2

	VMOVSLDUP X10, X11
	VMOVSHDUP X10, X12
	VSHUFPS $0xB1, X3, X3, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X3, X13
	VMOVAPS X13, X3

	VADDPS X0, X2, X4
	VSUBPS X2, X0, X5
	VADDPS X1, X3, X6
	VSUBPS X3, X1, X7

	VPERMILPS $0xB1, X7, X14
	VXORPS X15, X15, X15
	VSUBPS X14, X15, X11
	VBLENDPS $0x01, X11, X14, X14

	VPERMILPS $0xB1, X7, X12
	VSUBPS X12, X15, X11
	VBLENDPS $0x02, X11, X12, X12

	VADDPS X4, X6, X0
	VADDPS X5, X14, X1
	VSUBPS X6, X4, X2
	VADDPS X5, X12, X3

	VMOVSD X0, (R8)(BX*8)
	VMOVSD X1, (R8)(SI*8)
	VMOVSD X2, (R8)(DI*8)
	VMOVSD X3, (R8)(R13*8)

	ADDQ $BYTES_PER_RADIX4, R15
	INCQ DX
	JMP  m24p_8192_inv_stage6_inner

m24p_8192_inv_stage6_next:
	ADDQ $4096, CX
	JMP  m24p_8192_inv_stage6_outer

m24p_8192_inv_stage7:
	// ==================================================================
	// Stage 7: radix-2 final stage (inverse), 4096 butterflies
	// ==================================================================
	XORQ CX, CX
	MOVQ $PARAMS_STAGE7_OFFSET, R15

m24p_8192_inv_stage7_loop:
	CMPQ CX, $4096
	JGE  m24p_8192_inv_scale

	MOVQ CX, BX
	LEAQ 4096(BX), SI

	VMOVSD (R8)(BX*8), X0    // a
	VMOVSD (R8)(SI*8), X1    // b

	// Load pre-broadcast twiddle
	VMOVAPS (R14)(R15*1), X8

	// b = b * conj(w) using VFMSUBADD
	VMOVSLDUP X8, X11
	VMOVSHDUP X8, X12
	VSHUFPS $0xB1, X1, X1, X13
	VMULPS X12, X13, X13
	VFMADDSUB231PS X11, X1, X13
	VMOVAPS X13, X1

	VADDPS X0, X1, X2        // a + b*conj(w)
	VSUBPS X1, X0, X3        // a - b*conj(w)

	VMOVSD X2, (R8)(BX*8)
	VMOVSD X3, (R8)(SI*8)

	ADDQ $BYTES_PER_RADIX2, R15
	INCQ CX
	JMP  m24p_8192_inv_stage7_loop

m24p_8192_inv_scale:
	// ==================================================================
	// Scale by 1/8192
	// ==================================================================
	VBROADCASTSS ·eightThousandOneHundredThirtySecond32(SB), Y0
	XORQ CX, CX

m24p_8192_inv_scale_loop:
	VMOVUPS (R8)(CX*1), Y1
	VMOVUPS 32(R8)(CX*1), Y2
	VMULPS Y0, Y1, Y1
	VMULPS Y0, Y2, Y2
	VMOVUPS Y1, (R8)(CX*1)
	VMOVUPS Y2, 32(R8)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $65536
	JL   m24p_8192_inv_scale_loop

	// Copy results to dst if needed
	MOVQ dst+0(FP), R9
	CMPQ R8, R9
	JE   m24p_8192_inv_ret

	XORQ CX, CX

m24p_8192_inv_copy_loop:
	VMOVUPS (R8)(CX*1), Y0
	VMOVUPS 32(R8)(CX*1), Y1
	VMOVUPS Y0, (R9)(CX*1)
	VMOVUPS Y1, 32(R9)(CX*1)
	ADDQ $64, CX
	CMPQ CX, $65536
	JL   m24p_8192_inv_copy_loop

m24p_8192_inv_ret:
	VZEROUPPER
	MOVB $1, ret+96(FP)
	RET

m24p_8192_inv_return_false:
	VZEROUPPER
	MOVB $0, ret+96(FP)
	RET
