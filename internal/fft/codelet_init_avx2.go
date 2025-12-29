//go:build amd64 && fft_asm && !purego

package fft

// registerAVX2DITCodelets64 registers AVX2-optimized complex64 DIT codelets.
// These registrations are conditional on the fft_asm build tag and amd64 architecture.
func registerAVX2DITCodelets64() {
	// Size 4: Radix-4 AVX2 variant
	// Note: This implementation exists but may not provide speedup over generic
	// due to scalar operations. Registered with low priority for benchmarking.
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCodelet64(forwardAVX2Size4Radix4Complex64Asm),
		Inverse:    wrapCodelet64(inverseAVX2Size4Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit4_radix4_avx2",
		Priority:   5, // Lower priority - scalar ops may not beat generic
		BitrevFunc: nil,
	})

	// Size 8: AVX2 variants are currently slower than the Go radix-8 codelet,
	// so we skip registration until tuned.

	// Size 16: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(forwardAVX2Size16Complex64Asm),
		Inverse:    wrapCodelet64(inverseAVX2Size16Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix2_avx2",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 32: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(forwardAVX2Size32Complex64Asm),
		Inverse:    wrapCodelet64(inverseAVX2Size32Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit32_radix2_avx2",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 64: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(forwardAVX2Size64Complex64Asm),
		Inverse:    wrapCodelet64(inverseAVX2Size64Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix2_avx2",
		Priority:   15,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 64: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(forwardAVX2Size64Radix4Complex64Asm),
		Inverse:    wrapCodelet64(inverseAVX2Size64Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit64_radix4_avx2",
		Priority:   25, // Prefer radix-4 for size 64
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 128: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(forwardAVX2Size128Complex64Asm),
		Inverse:    wrapCodelet64(inverseAVX2Size128Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit128_radix2_avx2",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 256: Radix-2 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCodelet64(forwardAVX2Size256Radix2Complex64Asm),
		Inverse:    wrapCodelet64(inverseAVX2Size256Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit256_radix2_avx2",
		Priority:   15, // Lower than radix-4 variant
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 256: Radix-4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCodelet64(forwardAVX2Size256Radix4Complex64Asm),
		Inverse:    wrapCodelet64(inverseAVX2Size256Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit256_radix4_avx2",
		Priority:   20, // Higher priority - potentially faster
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})
}

// registerAVX2DITCodelets128 registers AVX2-optimized complex128 DIT codelets.
// Note: These assembly functions may not exist yet. This is a placeholder for future implementation.
func registerAVX2DITCodelets128() {
	// TODO: Implement AVX2 complex128 variants
	// For now, complex128 will use generic codelets or fallback kernels
}
