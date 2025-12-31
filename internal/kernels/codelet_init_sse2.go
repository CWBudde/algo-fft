//go:build amd64 && fft_asm && !purego

package kernels

// registerSSE2DITCodelets64 registers SSE2-optimized complex64 DIT codelets.
// These registrations are conditional on the fft_asm build tag and amd64 architecture.
// SSE2 provides a fallback for systems without AVX2 support.
func registerSSE2DITCodelets64() {
	// Size 16: Radix-4 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(forwardSSE2Size16Radix4Complex64Asm),
		Inverse:    wrapCodelet64(inverseSSE2Size16Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix4_sse2",
		Priority:   18, // Between generic (15) and AVX2 (20-25)
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})
}

// registerSSE2DITCodelets128 registers SSE2-optimized complex128 DIT codelets.
// Currently no size-specific SSE2 implementations for complex128.
func registerSSE2DITCodelets128() {
	// No size-specific SSE2 codelets for complex128 yet
}
