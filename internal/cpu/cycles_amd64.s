//go:build amd64

#include "textflag.h"

// func readCycleCounter() int64
TEXT ·readCycleCounter(SB), NOSPLIT, $0-8
	// LFENCE ensures all previous instructions complete before reading TSC
	// This prevents out-of-order execution from skewing measurements
	BYTE $0x0F; BYTE $0xAE; BYTE $0xE8  // LFENCE

	// RDTSC reads the 64-bit time-stamp counter into EDX:EAX
	RDTSC

	// Combine EDX:EAX into a single 64-bit value in AX
	SHLQ $32, DX         // DX = high 32 bits << 32
	ORQ  DX, AX          // AX = (high << 32) | low

	MOVQ AX, ret+0(FP)   // Return 64-bit cycle count
	RET
