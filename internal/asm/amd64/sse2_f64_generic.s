//go:build amd64 && asm && !purego

// ===========================================================================
// SSE2-optimized FFT Assembly for AMD64 (complex128/float64)
// ===========================================================================
//
// This file implements FFT transforms using SSE2 instructions for complex128
// (double-precision) data types as a fallback for systems without AVX2 support.
// SSE2 is available on all x86-64 CPUs.
//
// SIMD STRATEGY
// -------------
// - complex128: Process 1 butterfly per iteration using XMM (128-bit) registers
//               Each complex128 = 16 bytes, so XMM holds 1 complex number
//
// NOTE: Current implementation uses Go fallback for complex128.
//       Future optimization opportunity for SSE2 complex128 kernels.
//
// See avx2_f64_generic.s for detailed algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

TEXT ·ForwardSSE2Complex128Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)        // Return false (use Go fallback)
	RET

TEXT ·InverseSSE2Complex128Asm(SB), NOSPLIT|NOFRAME, $0-121
	MOVB $0, ret+120(FP)        // Return false (use Go fallback)
	RET
