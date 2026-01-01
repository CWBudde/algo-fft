//go:build 386 && fft_asm && !purego

// =====================================================================
// 386 FFT Assembly - Core Constants
// =====================================================================
//
// This file contains shared constants used by the SSE2 FFT implementations
// for 32-bit x86 (GOARCH=386).
//
// See asm_386_sse2.s for the FFT implementation.
//
// =====================================================================

#include "textflag.h"

// half32: Single-precision 0.5 for inverse FFT scaling
DATA ·half32(SB)/4, $0x3F000000 // 0.5f
GLOBL ·half32(SB), RODATA, $4

// one32: Single-precision 1.0 for scale factor computation
DATA ·one32(SB)/4, $0x3F800000 // 1.0f
GLOBL ·one32(SB), RODATA, $4
