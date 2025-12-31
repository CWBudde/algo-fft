//go:build !amd64 || !fft_asm || purego

package kernels

// registerSSE2DITCodelets64 is a no-op when SSE2 assembly is not available.
func registerSSE2DITCodelets64() {
	// No SSE2 codelets to register
}

// registerSSE2DITCodelets128 is a no-op when SSE2 assembly is not available.
func registerSSE2DITCodelets128() {
	// No SSE2 codelets to register
}
