//go:build amd64 && !purego

#include "textflag.h"

// func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
TEXT ·cpuid(SB), NOSPLIT, $0-24
	MOVL eaxArg+0(FP), AX  // Select the CPUID leaf.
	MOVL ecxArg+4(FP), CX  // Select the CPUID subleaf.
	CPUID                  // Query the processor.
	MOVL AX, eax+8(FP)     // Return EAX.
	MOVL BX, ebx+12(FP)    // Return EBX.
	MOVL CX, ecx+16(FP)    // Return ECX.
	MOVL DX, edx+20(FP)    // Return EDX.
	RET
