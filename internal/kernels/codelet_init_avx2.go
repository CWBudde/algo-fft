//go:build amd64 && fft_asm && !purego

package kernels

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

	// Size 32: Mixed-radix-2/4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(forwardDIT32MixedRadix24Complex64AVX2),
		Inverse:    wrapCodelet64(inverseDIT32MixedRadix24Complex64AVX2),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit32_mixed24_avx2",
		Priority:   25, // Higher than radix-2 to prefer mixed radix
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

	// Size 128: Mixed-radix-2/4 AVX2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(forwardDIT128MixedRadix24Complex64AVX2),
		Inverse:    wrapCodelet64(inverseDIT128MixedRadix24Complex64AVX2),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit128_mixed24_avx2",
		Priority:   25, // Higher than radix-2 to prefer mixed radix
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
func registerAVX2DITCodelets128() {
	// Size 8: Radix-8 AVX2 variant (single-stage butterfly)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(forwardAVX2Size8Radix8Complex128Asm),
		Inverse:    wrapCodelet128(inverseAVX2Size8Radix8Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix8_avx2",
		Priority:   9, // Keep below Go radix-8 unless proven faster
		BitrevFunc: nil,
	})

	// Size 8: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(forwardAVX2Size8Radix2Complex128Asm),
		Inverse:    wrapCodelet128(inverseAVX2Size8Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit8_radix2_avx2",
		Priority:   10, // Lower priority than Go radix-8 for now
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 16: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCodelet128(forwardAVX2Size16Complex128Asm),
		Inverse:    wrapCodelet128(inverseAVX2Size16Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit16_radix2_avx2",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 32: Radix-2 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCodelet128(forwardAVX2Size32Complex128Asm),
		Inverse:    wrapCodelet128(inverseAVX2Size32Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit32_radix2_avx2",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 32: Mixed-radix-2/4 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCodelet128(forwardDIT32MixedRadix24Complex128AVX2),
		Inverse:    wrapCodelet128(inverseDIT32MixedRadix24Complex128AVX2),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit32_mixed24_avx2",
		Priority:   25, // Higher than radix-2 to prefer mixed radix
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 128: Mixed-radix-2/4 AVX2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCodelet128(forwardDIT128MixedRadix24Complex128AVX2),
		Inverse:    wrapCodelet128(inverseDIT128MixedRadix24Complex128AVX2),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDAVX2,
		Signature:  "dit128_mixed24_avx2",
		Priority:   25, // Higher than radix-2 to prefer mixed radix
		BitrevFunc: ComputeBitReversalIndices,
	})
}
