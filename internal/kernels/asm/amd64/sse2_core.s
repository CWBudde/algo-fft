//go:build amd64 && fft_asm && !purego

#include "textflag.h"

// SSE2 sign masks for complex64 lane negation.
// Layout is 4x float32 in an XMM register.
// We only use the low 2 lanes (re, im); high lanes are kept 0.

DATA ·sse2MaskNegLoPS+0(SB)/4, $0x80000000 // negate lane0 (re)
DATA ·sse2MaskNegLoPS+4(SB)/4, $0x00000000
DATA ·sse2MaskNegLoPS+8(SB)/4, $0x00000000
DATA ·sse2MaskNegLoPS+12(SB)/4, $0x00000000
GLOBL ·sse2MaskNegLoPS(SB), RODATA|NOPTR, $16

DATA ·sse2MaskNegHiPS+0(SB)/4, $0x00000000
DATA ·sse2MaskNegHiPS+4(SB)/4, $0x80000000 // negate lane1 (im)
DATA ·sse2MaskNegHiPS+8(SB)/4, $0x00000000
DATA ·sse2MaskNegHiPS+12(SB)/4, $0x00000000
GLOBL ·sse2MaskNegHiPS(SB), RODATA|NOPTR, $16
