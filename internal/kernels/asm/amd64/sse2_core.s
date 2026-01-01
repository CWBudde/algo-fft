//go:build amd64 && fft_asm && !purego

#include "textflag.h"

// SSE2-specific constants and utilities
// Note: Lane negation masks (maskNegLoPS, maskNegHiPS, maskNegLoPD, maskNegHiPD)
// are now defined in core.s and shared across SSE2 and AVX2 implementations.
