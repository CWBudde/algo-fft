//go:build 386 && purego

package cpu

// cpuid fallback for purego builds (no assembly allowed).
// We cannot execute CPUID instruction directly.
func cpuid(eaxArg, ecxArg uint32) (eax, ebx, ecx, edx uint32) {
	return 0, 0, 0, 0
}
