//go:build amd64 && fft_asm && !purego

package kernels

// registerSSE2DITCodelets64 registers SSE2-optimized complex64 DIT codelets.
// These registrations are conditional on the fft_asm build tag and amd64 architecture.
// SSE2 provides a fallback for systems without AVX2 support.
func registerSSE2DITCodelets64() {
	// Size 4: Radix-4 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCodelet64(forwardSSE2Size4Radix4Complex64Asm),
		Inverse:    wrapCodelet64(inverseSSE2Size4Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit4_radix4_sse2",
		Priority:   5, // Lower priority - scalar ops may not beat generic
		BitrevFunc: nil,
	})

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

	// Size 64: Radix-4 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(forwardSSE2Size64Radix4Complex64Asm),
		Inverse:    wrapCodelet64(inverseSSE2Size64Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit64_radix4_sse2",
		Priority:   18, // Between generic (15) and AVX2 (20-25)
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 128: Mixed Radix-2/4 SSE2 variant (3 radix-4 + 1 radix-2 stages)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(forwardSSE2Size128Radix4Complex64Asm),
		Inverse:    wrapCodelet64(inverseSSE2Size128Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit128_radix4_sse2",
		Priority:   18, // Between generic (15) and AVX2 (20-25)
		BitrevFunc: ComputeBitReversalIndices,
	})
}

// registerSSE2DITCodelets128 registers SSE2-optimized complex128 DIT codelets.
func registerSSE2DITCodelets128() {
	// Size 4: Radix-4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       4,
		Forward:    wrapCodelet128(forwardSSE2Size4Radix4Complex128Asm),
		Inverse:    wrapCodelet128(inverseSSE2Size4Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit4_radix4_sse2",
		Priority:   5, // Lower priority - scalar ops may not beat generic
		BitrevFunc: nil,
	})
}
