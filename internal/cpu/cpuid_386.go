//go:build 386 && !purego

package cpu

// cpuid executes the CPUID instruction with the given EAX and ECX inputs.
// Returns EAX, EBX, ECX, EDX outputs.
// Defined in cpuid_386.s
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32)
