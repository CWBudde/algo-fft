//go:build !amd64 || !asm || purego

package kernels

// registerAVX2DITCodelets64 is a no-op when AVX2 assembly is not available.
func registerAVX2DITCodelets64() {
	// No AVX2 codelets to register on this platform
}

// registerAVX2DITCodelets128 is a no-op when AVX2 assembly is not available.
func registerAVX2DITCodelets128() {
	// No AVX2 codelets to register on this platform
}
