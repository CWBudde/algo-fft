//go:build amd64 && asm && !purego

// ===========================================================================
// AVX2 Radix-5 Butterfly (complex64) for AMD64
// ===========================================================================
// Processes 2 radix-5 butterflies in parallel using XMM registers.
// Each XMM holds 2 complex64 values: [re0, im0, re1, im1].
// ===========================================================================

#include "textflag.h"

// Radix-5 twiddle constants (exp(-2πi*k/5) for k=1..4), packed as:
// [re, im, re, im] so both butterflies share the same constants.
DATA radix5_w1<>+0x00(SB)/4, $0x3E9E377A  //  0.30901699
DATA radix5_w1<>+0x04(SB)/4, $0xBF737871  // -0.95105654
DATA radix5_w1<>+0x08(SB)/4, $0x3E9E377A
DATA radix5_w1<>+0x0C(SB)/4, $0xBF737871
GLOBL radix5_w1<>(SB), RODATA|NOPTR, $16

DATA radix5_w2<>+0x00(SB)/4, $0xBF4F1BBD  // -0.80901699
DATA radix5_w2<>+0x04(SB)/4, $0xBF167918  // -0.58778524
DATA radix5_w2<>+0x08(SB)/4, $0xBF4F1BBD
DATA radix5_w2<>+0x0C(SB)/4, $0xBF167918
GLOBL radix5_w2<>(SB), RODATA|NOPTR, $16

DATA radix5_w3<>+0x00(SB)/4, $0xBF4F1BBD  // -0.80901699
DATA radix5_w3<>+0x04(SB)/4, $0x3F167918  //  0.58778524
DATA radix5_w3<>+0x08(SB)/4, $0xBF4F1BBD
DATA radix5_w3<>+0x0C(SB)/4, $0x3F167918
GLOBL radix5_w3<>(SB), RODATA|NOPTR, $16

DATA radix5_w4<>+0x00(SB)/4, $0x3E9E377A  //  0.30901699
DATA radix5_w4<>+0x04(SB)/4, $0x3F737871  //  0.95105654
DATA radix5_w4<>+0x08(SB)/4, $0x3E9E377A
DATA radix5_w4<>+0x0C(SB)/4, $0x3F737871
GLOBL radix5_w4<>(SB), RODATA|NOPTR, $16

// Conjugated twiddles for inverse butterfly.
DATA radix5_w1_inv<>+0x00(SB)/4, $0x3E9E377A  //  0.30901699
DATA radix5_w1_inv<>+0x04(SB)/4, $0x3F737871  //  0.95105654
DATA radix5_w1_inv<>+0x08(SB)/4, $0x3E9E377A
DATA radix5_w1_inv<>+0x0C(SB)/4, $0x3F737871
GLOBL radix5_w1_inv<>(SB), RODATA|NOPTR, $16

DATA radix5_w2_inv<>+0x00(SB)/4, $0xBF4F1BBD  // -0.80901699
DATA radix5_w2_inv<>+0x04(SB)/4, $0x3F167918  //  0.58778524
DATA radix5_w2_inv<>+0x08(SB)/4, $0xBF4F1BBD
DATA radix5_w2_inv<>+0x0C(SB)/4, $0x3F167918
GLOBL radix5_w2_inv<>(SB), RODATA|NOPTR, $16

DATA radix5_w3_inv<>+0x00(SB)/4, $0xBF4F1BBD  // -0.80901699
DATA radix5_w3_inv<>+0x04(SB)/4, $0xBF167918  // -0.58778524
DATA radix5_w3_inv<>+0x08(SB)/4, $0xBF4F1BBD
DATA radix5_w3_inv<>+0x0C(SB)/4, $0xBF167918
GLOBL radix5_w3_inv<>(SB), RODATA|NOPTR, $16

DATA radix5_w4_inv<>+0x00(SB)/4, $0x3E9E377A  //  0.30901699
DATA radix5_w4_inv<>+0x04(SB)/4, $0xBF737871  // -0.95105654
DATA radix5_w4_inv<>+0x08(SB)/4, $0x3E9E377A
DATA radix5_w4_inv<>+0x0C(SB)/4, $0xBF737871
GLOBL radix5_w4_inv<>(SB), RODATA|NOPTR, $16

// Complex multiply: dst = src * w
// Uses VFMADDSUB231PS: even lanes subtract, odd lanes add.
#define CMUL_CONST_XMM(src, wmem, dst, tmp1, tmp2, tmp3) \
	VMOVAPS wmem, tmp1; \
	VMOVSLDUP tmp1, tmp2; \
	VMOVSHDUP tmp1, tmp3; \
	VSHUFPS $0xB1, src, src, dst; \
	VMULPS tmp3, dst, dst; \
	VFMADDSUB231PS tmp2, src, dst

// ===========================================================================
// Function: Butterfly5ForwardAVX2Complex64
// ===========================================================================
// Processes 2 radix-5 forward butterflies in parallel.
//
// func Butterfly5ForwardAVX2Complex64(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4 []complex64)
TEXT ·Butterfly5ForwardAVX2Complex64(SB), NOSPLIT, $0-240
	// Load input pointers
	MOVQ y0+0(FP), R8
	MOVQ y1+24(FP), R9
	MOVQ y2+48(FP), R10
	MOVQ y3+72(FP), R11
	MOVQ y4+96(FP), R12
	MOVQ a0+120(FP), R13
	MOVQ a1+144(FP), R14
	MOVQ a2+168(FP), R15
	MOVQ a3+192(FP), BX
	MOVQ a4+216(FP), CX

	// Verify all slices have length >= 2
	MOVQ y0+8(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	MOVQ y1+32(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	MOVQ y2+56(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	MOVQ y3+80(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	MOVQ y4+104(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	MOVQ a0+128(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	MOVQ a1+152(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	MOVQ a2+176(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	MOVQ a3+200(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	MOVQ a4+224(FP), AX
	CMPQ AX, $2
	JL   butterfly5_fwd_return

	// Load inputs: 2 complex64 = 16 bytes per register
	VMOVUPS (R13), X0  // a0
	VMOVUPS (R14), X1  // a1
	VMOVUPS (R15), X2  // a2
	VMOVUPS (BX), X3   // a3
	VMOVUPS (CX), X4   // a4

	// y0 = a0 + a1 + a2 + a3 + a4
	VMOVAPS X0, X5
	VADDPS X1, X5, X5
	VADDPS X2, X5, X5
	VADDPS X3, X5, X5
	VADDPS X4, X5, X5

	// y1 = a0 + a1*w1 + a2*w2 + a3*w3 + a4*w4
	CMUL_CONST_XMM(X1, radix5_w1<>(SB), X6, X12, X13, X14)
	CMUL_CONST_XMM(X2, radix5_w2<>(SB), X7, X12, X13, X14)
	CMUL_CONST_XMM(X3, radix5_w3<>(SB), X8, X12, X13, X14)
	CMUL_CONST_XMM(X4, radix5_w4<>(SB), X9, X12, X13, X14)
	VMOVAPS X0, X10
	VADDPS X6, X10, X10
	VADDPS X7, X10, X10
	VADDPS X8, X10, X10
	VADDPS X9, X10, X10

	// y2 = a0 + a1*w2 + a2*w4 + a3*w1 + a4*w3
	CMUL_CONST_XMM(X1, radix5_w2<>(SB), X6, X12, X13, X14)
	CMUL_CONST_XMM(X2, radix5_w4<>(SB), X7, X12, X13, X14)
	CMUL_CONST_XMM(X3, radix5_w1<>(SB), X8, X12, X13, X14)
	CMUL_CONST_XMM(X4, radix5_w3<>(SB), X9, X12, X13, X14)
	VMOVAPS X0, X11
	VADDPS X6, X11, X11
	VADDPS X7, X11, X11
	VADDPS X8, X11, X11
	VADDPS X9, X11, X11

	// y3 = a0 + a1*w3 + a2*w1 + a3*w4 + a4*w2
	CMUL_CONST_XMM(X1, radix5_w3<>(SB), X6, X12, X13, X14)
	CMUL_CONST_XMM(X2, radix5_w1<>(SB), X7, X12, X13, X14)
	CMUL_CONST_XMM(X3, radix5_w4<>(SB), X8, X12, X13, X14)
	CMUL_CONST_XMM(X4, radix5_w2<>(SB), X9, X12, X13, X14)
	VMOVAPS X0, X15
	VADDPS X6, X15, X15
	VADDPS X7, X15, X15
	VADDPS X8, X15, X15
	VADDPS X9, X15, X15

	// y4 = a0 + a1*w4 + a2*w3 + a3*w2 + a4*w1
	CMUL_CONST_XMM(X1, radix5_w4<>(SB), X6, X12, X13, X14)
	CMUL_CONST_XMM(X2, radix5_w3<>(SB), X7, X12, X13, X14)
	CMUL_CONST_XMM(X3, radix5_w2<>(SB), X8, X12, X13, X14)
	CMUL_CONST_XMM(X4, radix5_w1<>(SB), X9, X12, X13, X14)
	VMOVAPS X0, X12
	VADDPS X6, X12, X12
	VADDPS X7, X12, X12
	VADDPS X8, X12, X12
	VADDPS X9, X12, X12

	// Store results
	VMOVUPS X5, (R8)
	VMOVUPS X10, (R9)
	VMOVUPS X11, (R10)
	VMOVUPS X15, (R11)
	VMOVUPS X12, (R12)

butterfly5_fwd_return:
	RET

// ===========================================================================
// Function: Butterfly5InverseAVX2Complex64
// ===========================================================================
// Processes 2 radix-5 inverse butterflies in parallel.
//
// func Butterfly5InverseAVX2Complex64(y0, y1, y2, y3, y4, a0, a1, a2, a3, a4 []complex64)
TEXT ·Butterfly5InverseAVX2Complex64(SB), NOSPLIT, $0-240
	// Load input pointers
	MOVQ y0+0(FP), R8
	MOVQ y1+24(FP), R9
	MOVQ y2+48(FP), R10
	MOVQ y3+72(FP), R11
	MOVQ y4+96(FP), R12
	MOVQ a0+120(FP), R13
	MOVQ a1+144(FP), R14
	MOVQ a2+168(FP), R15
	MOVQ a3+192(FP), BX
	MOVQ a4+216(FP), CX

	// Verify all slices have length >= 2
	MOVQ y0+8(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	MOVQ y1+32(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	MOVQ y2+56(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	MOVQ y3+80(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	MOVQ y4+104(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	MOVQ a0+128(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	MOVQ a1+152(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	MOVQ a2+176(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	MOVQ a3+200(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	MOVQ a4+224(FP), AX
	CMPQ AX, $2
	JL   butterfly5_inv_return

	// Load inputs: 2 complex64 = 16 bytes per register
	VMOVUPS (R13), X0  // a0
	VMOVUPS (R14), X1  // a1
	VMOVUPS (R15), X2  // a2
	VMOVUPS (BX), X3   // a3
	VMOVUPS (CX), X4   // a4

	// y0 = a0 + a1 + a2 + a3 + a4
	VMOVAPS X0, X5
	VADDPS X1, X5, X5
	VADDPS X2, X5, X5
	VADDPS X3, X5, X5
	VADDPS X4, X5, X5

	// y1 = a0 + a1*w1 + a2*w2 + a3*w3 + a4*w4 (conjugated)
	CMUL_CONST_XMM(X1, radix5_w1_inv<>(SB), X6, X12, X13, X14)
	CMUL_CONST_XMM(X2, radix5_w2_inv<>(SB), X7, X12, X13, X14)
	CMUL_CONST_XMM(X3, radix5_w3_inv<>(SB), X8, X12, X13, X14)
	CMUL_CONST_XMM(X4, radix5_w4_inv<>(SB), X9, X12, X13, X14)
	VMOVAPS X0, X10
	VADDPS X6, X10, X10
	VADDPS X7, X10, X10
	VADDPS X8, X10, X10
	VADDPS X9, X10, X10

	// y2 = a0 + a1*w2 + a2*w4 + a3*w1 + a4*w3 (conjugated)
	CMUL_CONST_XMM(X1, radix5_w2_inv<>(SB), X6, X12, X13, X14)
	CMUL_CONST_XMM(X2, radix5_w4_inv<>(SB), X7, X12, X13, X14)
	CMUL_CONST_XMM(X3, radix5_w1_inv<>(SB), X8, X12, X13, X14)
	CMUL_CONST_XMM(X4, radix5_w3_inv<>(SB), X9, X12, X13, X14)
	VMOVAPS X0, X11
	VADDPS X6, X11, X11
	VADDPS X7, X11, X11
	VADDPS X8, X11, X11
	VADDPS X9, X11, X11

	// y3 = a0 + a1*w3 + a2*w1 + a3*w4 + a4*w2 (conjugated)
	CMUL_CONST_XMM(X1, radix5_w3_inv<>(SB), X6, X12, X13, X14)
	CMUL_CONST_XMM(X2, radix5_w1_inv<>(SB), X7, X12, X13, X14)
	CMUL_CONST_XMM(X3, radix5_w4_inv<>(SB), X8, X12, X13, X14)
	CMUL_CONST_XMM(X4, radix5_w2_inv<>(SB), X9, X12, X13, X14)
	VMOVAPS X0, X15
	VADDPS X6, X15, X15
	VADDPS X7, X15, X15
	VADDPS X8, X15, X15
	VADDPS X9, X15, X15

	// y4 = a0 + a1*w4 + a2*w3 + a3*w2 + a4*w1 (conjugated)
	CMUL_CONST_XMM(X1, radix5_w4_inv<>(SB), X6, X12, X13, X14)
	CMUL_CONST_XMM(X2, radix5_w3_inv<>(SB), X7, X12, X13, X14)
	CMUL_CONST_XMM(X3, radix5_w2_inv<>(SB), X8, X12, X13, X14)
	CMUL_CONST_XMM(X4, radix5_w1_inv<>(SB), X9, X12, X13, X14)
	VMOVAPS X0, X12
	VADDPS X6, X12, X12
	VADDPS X7, X12, X12
	VADDPS X8, X12, X12
	VADDPS X9, X12, X12

	// Store results
	VMOVUPS X5, (R8)
	VMOVUPS X10, (R9)
	VMOVUPS X11, (R10)
	VMOVUPS X15, (R11)
	VMOVUPS X12, (R12)

butterfly5_inv_return:
	RET
