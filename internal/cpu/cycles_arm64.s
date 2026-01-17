//go:build arm64

#include "textflag.h"

// func readCycleCounter() int64
TEXT ·readCycleCounter(SB), NOSPLIT, $0-8
	// Read the virtual counter (CNTVCT_EL0)
	// This is a 64-bit counter that increments at a fixed frequency
	// ISB ensures ordering of instructions
	WORD $0xD5033FDF  // ISB (Instruction Synchronization Barrier)
	WORD $0xD53BE040  // MRS CNTVCT_EL0, X0

	MOVD R0, ret+0(FP)  // Return 64-bit cycle count
	RET

// func readCounterFrequency() int64
TEXT ·readCounterFrequency(SB), NOSPLIT, $0-8
	// Read the counter frequency (CNTFRQ_EL0)
	// This tells us the frequency at which CNTVCT_EL0 increments
	WORD $0xD53BE000  // MRS CNTFRQ_EL0, X0

	MOVD R0, ret+0(FP)  // Return counter frequency in Hz
	RET
