//go:build amd64 && purego

package cpu

func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32) {
	return 0, 0, 0, 0
}
