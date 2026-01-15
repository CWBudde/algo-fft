//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

// registerSSE2DITCodelets64 registers SSE2/SSE3-optimized complex64 DIT codelets.
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
		Priority:   5,             // Lower priority - scalar ops may not beat generic
		KernelType: KernelTypeDIT, // Self-contained, no external bitrev
	})

	// Size 8: Radix-2 SSE3 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size8Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size8Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit8_radix2_sse3",
		Priority:   18, // Between generic (15) and AVX2 (20-25)
		KernelType: KernelTypeDIT,
	})

	// Size 8: Radix-8 SSE3 variant (single-stage butterfly, identity permutation)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size8Radix8Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size8Radix8Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit8_radix8_sse3",
		Priority:   30,             // Higher priority than radix-2 (18) and mixed-radix (??)
		KernelType: KernelTypeCore, // Identity permutation, no bitrev needed
	})

	// Size 16: Radix-16 SSE3 variant (4x4)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size16Radix16Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size16Radix16Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit16_radix16_sse3",
		Priority:   40, // Highest priority for size 16
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-2 SSE3 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size16Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size16Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit16_radix2_sse3",
		Priority:   17, // Lower priority than radix-4 (18)
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-4 SSE3 variant
	// FIXED: Corrected butterfly operations in both stages
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size16Radix4Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size16Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit16_radix4_sse3",
		Priority:   18, // Re-enabled: bugs fixed in butterfly operations
		KernelType: KernelTypeDIT,
	})

	// Size 32: Radix-2 SSE3 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size32Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size32Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit32_radix2_sse3",
		Priority:   17, // Lower priority than radix-32 (??)
		KernelType: KernelTypeDIT,
	})

	// Size 32: Mixed-radix-2/4 SSE3 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size32Mixed24Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size32Mixed24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit32_mixed24_sse3",
		Priority:   19, // Higher priority than radix-2 (17)
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-4 SSE3 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size64Radix4Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size64Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit64_radix4_sse3",
		Priority:   18, // Between generic (15) and AVX2 (20-25)
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-2 SSE3 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size64Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size64Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit64_radix2_sse3",
		Priority:   17, // Lower priority than radix-4
		KernelType: KernelTypeDIT,
	})

	// Size 128: Mixed Radix-2/4 SSE3 variant (3 radix-4 + 1 radix-2 stages)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size128Mixed24Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size128Mixed24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit128_mixed24_sse3",
		Priority:   17,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Radix-2 SSE3 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size128Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size128Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit128_radix2_sse3",
		Priority:   17,
		KernelType: KernelTypeDIT,
	})

	// Size 512: Radix-2 SSE3 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCodelet64(amd64.ForwardSSE3Size512Radix2Complex64Asm),
		Inverse:    wrapCodelet64(amd64.InverseSSE3Size512Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE3,
		Signature:  "dit512_radix2_sse3",
		Priority:   10,
		KernelType: KernelTypeDIT,
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
		Priority:   5,             // Lower priority - scalar ops may not beat generic
		KernelType: KernelTypeDIT, // Self-contained, no external bitrev
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
		KernelType: KernelTypeDIT,
	})

	// Size 8: Radix-8 SSE2 variant (single-stage butterfly, identity permutation)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size8Radix8Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size8Radix8Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix8_sse2",
		Priority:   30,
		KernelType: KernelTypeCore, // Identity permutation, no bitrev needed
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
		KernelType: KernelTypeDIT,
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
		KernelType: KernelTypeDIT,
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
		KernelType: KernelTypeDIT,
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
		KernelType: KernelTypeDIT,
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
		KernelType: KernelTypeDIT,
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
		KernelType: KernelTypeDIT,
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
		KernelType: KernelTypeDIT,
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
		KernelType: KernelTypeDIT,
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
		KernelType: KernelTypeDIT,
	})

	// Size 256: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size256Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size256Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit256_radix2_sse2",
		Priority:   10,
		KernelType: KernelTypeDIT,
	})

	// Size 256: Radix-4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size256Radix4Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size256Radix4Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit256_radix4_sse2",
		Priority:   15,
		KernelType: KernelTypeDIT,
	})

	// Size 512: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCodelet128(amd64.ForwardSSE2Size512Radix2Complex128Asm),
		Inverse:    wrapCodelet128(amd64.InverseSSE2Size512Radix2Complex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit512_radix2_sse2",
		Priority:   10,
		KernelType: KernelTypeDIT,
	})
}
