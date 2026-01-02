//go:build arm64 && fft_asm && !purego && neon256_radix4_fallback

#include "textflag.h"

// The repository currently ships a size-256 radix-2 NEON kernel, but no dedicated
// size-256 radix-4 NEON kernel. Some Go declarations/wrappers expect the radix-4
// symbols to exist. Provide thin fallbacks that forward to the radix-2 kernel.

TEXT ·ForwardNEONSize256Radix4Complex64Asm(SB), NOSPLIT, $0-121
	B ·ForwardNEONSize256Radix2Complex64Asm(SB)

TEXT ·InverseNEONSize256Radix4Complex64Asm(SB), NOSPLIT, $0-121
	B ·InverseNEONSize256Radix2Complex64Asm(SB)
