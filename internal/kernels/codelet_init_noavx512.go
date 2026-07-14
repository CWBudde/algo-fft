//go:build !amd64 || purego

package kernels

// registerAVX512DITCodelets64 is a no-op when AVX-512 assembly is not available.
func registerAVX512DITCodelets64() {
	// No AVX-512 codelets to register on this platform
}

// registerAVX512DITCodelets128 is a no-op when AVX-512 assembly is not available.
func registerAVX512DITCodelets128() {
	// No AVX-512 codelets to register on this platform
}
