//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// registerSSE2DITCodelets64 registers SSE2-optimized complex64 DIT codelets.
// These registrations are conditional on the asm build tag and amd64 architecture.
// SSE2 provides a fallback for systems without AVX2 support.
func registerSSE2DITCodelets64() {
	// Size 4: Radix-4 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size4Radix4Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size4Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit4_radix4_sse2",
		Priority:   5, // Lower priority - scalar ops may not beat generic
		BitrevFunc: nil,
	})

	// Size 8: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size8Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size8Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix2_sse2",
		Priority:   18, // Between generic (15) and AVX2 (20-25)
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 8: Radix-8 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size8Radix8Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size8Radix8Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix8_sse2",
		Priority:   30, // Higher priority than radix-2 (18) and mixed-radix (??)
		BitrevFunc: mathpkg.ComputeIdentityIndices,
	})

	// Size 16: Radix-16 SSE2 variant (4x4)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size16Radix16Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size16Radix16Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix16_sse2",
		Priority:   40, // Highest priority for size 16
		BitrevFunc: mathpkg.ComputeIdentityIndices,
	})

	// Size 16: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size16Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size16Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix2_sse2",
		Priority:   17, // Lower priority than radix-4 (18)
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 16: Radix-4 SSE2 variant
	// FIXED: Corrected butterfly operations in both stages
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size16Radix4Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size16Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix4_sse2",
		Priority:   18, // Re-enabled: bugs fixed in butterfly operations
		BitrevFunc: mathpkg.ComputeBitReversalIndicesRadix4,
	})

	// Size 32: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size32Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size32Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit32_radix2_sse2",
		Priority:   17, // Lower priority than radix-32 (??)
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 32: Mixed-radix-2/4 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size32Mixed24Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size32Mixed24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit32_mixed24_sse2",
		Priority:   19, // Higher priority than radix-2 (17)
		BitrevFunc: mathpkg.ComputeBitReversalIndicesMixed24,
	})

	// Size 64: Radix-4 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size64Radix4Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size64Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit64_radix4_sse2",
		Priority:   18, // Between generic (15) and AVX2 (20-25)
		BitrevFunc: mathpkg.ComputeBitReversalIndicesRadix4,
	})

	// Size 64: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size64Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size64Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit64_radix2_sse2",
		Priority:   17, // Lower priority than radix-4
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 128: Mixed Radix-2/4 SSE2 variant (3 radix-4 + 1 radix-2 stages)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size128Mixed24Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size128Mixed24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit128_mixed24_sse2",
		Priority:   17,
		BitrevFunc: mathpkg.ComputeBitReversalIndicesMixed24,
	})

	// Size 128: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(amd64.ForwardSSE2Size128Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE2Size128Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit128_radix2_sse2",
		Priority:   17,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})
}

// registerSSE2DITCodelets128 registers SSE2-optimized complex128 DIT codelets.
func registerSSE2DITCodelets128() {
	// Size 4: Radix-4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       4,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size4Radix4Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size4Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit4_radix4_sse2",
		Priority:   5, // Lower priority - scalar ops may not beat generic
		BitrevFunc: nil,
	})

	// Size 8: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size8Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size8Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix2_sse2",
		Priority:   17,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 8: Radix-8 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size8Radix8Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size8Radix8Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix8_sse2",
		Priority:   30,
		BitrevFunc: mathpkg.ComputeIdentityIndices,
	})

	// Size 8: Radix-4 (Mixed-radix) SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size8Radix4Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size8Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix4_sse2",
		Priority:   18,
		BitrevFunc: mathpkg.ComputeBitReversalIndicesMixed24,
	})

	// Size 16: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size16Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size16Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix2_sse2",
		Priority:   17,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 16: Radix-4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size16Radix4Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size16Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix4_sse2",
		Priority:   18,
		BitrevFunc: mathpkg.ComputeBitReversalIndicesRadix4,
	})

	// Size 32: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size32Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size32Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit32_radix2_sse2",
		Priority:   17,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 32: Mixed-radix-2/4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size32Mixed24Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size32Mixed24Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit32_mixed24_sse2",
		Priority:   19,
		BitrevFunc: mathpkg.ComputeBitReversalIndicesMixed24,
	})

	// Size 64: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size64Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size64Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit64_radix2_sse2",
		Priority:   18,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 64: Radix-4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size64Radix4Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size64Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit64_radix4_sse2",
		Priority:   18,
		BitrevFunc: mathpkg.ComputeBitReversalIndicesRadix4,
	})

	// Size 128: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size128Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size128Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit128_radix2_sse2",
		Priority:   17,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})

	// Size 128: Mixed-radix-2/4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size128Mixed24Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size128Mixed24Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit128_mixed24_sse2",
		Priority:   18,
		BitrevFunc: mathpkg.ComputeBitReversalIndices,
	})
}
