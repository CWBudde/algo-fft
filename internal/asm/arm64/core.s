//go:build arm64 && asm && !purego

// ===========================================================================
// ARM64 NEON FFT Assembly - Core Utilities and Constants
// ===========================================================================
//
// This file contains shared constants and helper functions used by the
// NEON FFT implementations.
//
// See asm_arm64_neon_generic.s for the main FFT algorithm documentation.
//
// ===========================================================================

#include "textflag.h"

DATA ·neonOnes+0(SB)/4, $0x3f800000
DATA ·neonOnes+4(SB)/4, $0x3f800000
DATA ·neonOnes+8(SB)/4, $0x3f800000
DATA ·neonOnes+12(SB)/4, $0x3f800000
GLOBL ·neonOnes(SB), RODATA, $16

DATA ·neonSignImag+0(SB)/4, $0x00000000
DATA ·neonSignImag+4(SB)/4, $0x80000000
DATA ·neonSignImag+8(SB)/4, $0x00000000
DATA ·neonSignImag+12(SB)/4, $0x80000000
GLOBL ·neonSignImag(SB), RODATA, $16

DATA ·neonOne64+0(SB)/8, $0x3ff0000000000000
GLOBL ·neonOne64(SB), RODATA, $8

DATA ·neonInv8+0(SB)/4, $0x3e000000
GLOBL ·neonInv8(SB), RODATA, $4

DATA ·neonInv4+0(SB)/4, $0x3e800000
GLOBL ·neonInv4(SB), RODATA, $4
