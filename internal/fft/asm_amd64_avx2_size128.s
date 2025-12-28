//go:build amd64 && fft_asm && !purego

// ===========================================================================
// AVX2 Size-128 FFT Kernels for AMD64
// ===========================================================================
//
// This file contains FFT kernel stubs for size 128.
// These stubs return false to fall back to generic implementation.
//
// TODO: Implement fully-unrolled 7-stage FFT for size 128.
//
// See asm_amd64_avx2_generic.s for algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

// Forward transform, size 128, complex64
TEXT ·forwardAVX2Size128Complex64Asm(SB), NOSPLIT, $0-121
	MOVB $0, ret+120(FP)        // Return false
	RET

// Inverse transform, size 128, complex64
TEXT ·inverseAVX2Size128Complex64Asm(SB), NOSPLIT, $0-121
	MOVB $0, ret+120(FP)        // Return false
	RET
