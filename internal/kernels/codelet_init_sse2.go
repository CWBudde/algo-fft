//go:build amd64 && asm && !purego

package kernels

import (
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

// registerSSE2DITCodelets64 registers SSE2-optimized complex64 DIT codelets.
// These registrations are conditional on the asm build tag and amd64 architecture.
// SSE2 provides a fallback for systems without AVX2 support.
func registerSSE2DITCodelets64() {
	// Size 4: Radix-4 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size4Radix4Complex64Asm, nil)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size4Radix4Complex64Asm, nil)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit4_radix4_sse2",
		Priority:   5, // Lower priority - scalar ops may not beat generic
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained, no external bitrev
	})

	// Size 8: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size8Radix2Complex64Asm, bitrevSize8Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size8Radix2Complex64Asm, bitrevSize8Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix2_sse2",
		Priority:   18, // Between generic (15) and AVX2 (20-25)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 8: Radix-8 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size8Radix8Complex64Asm, nil)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size8Radix8Complex64Asm, nil)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix8_sse2",
		Priority:   30, // Higher priority than radix-2 (18) and mixed-radix (??)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-16 SSE2 variant (4x4)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size16Radix16Complex64Asm, nil)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size16Radix16Complex64Asm, nil)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix16_sse2",
		Priority:   40, // Highest priority for size 16
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size16Radix2Complex64Asm, bitrevSize16Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size16Radix2Complex64Asm, bitrevSize16Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix2_sse2",
		Priority:   17, // Lower priority than radix-4 (18)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-4 SSE2 variant
	// FIXED: Corrected butterfly operations in both stages
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size16Radix4Complex64Asm, bitrevSize16Radix4)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size16Radix4Complex64Asm, bitrevSize16Radix4)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix4_sse2",
		Priority:   18, // Re-enabled: bugs fixed in butterfly operations
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 32: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size32Radix2Complex64Asm, bitrevSize32Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size32Radix2Complex64Asm, bitrevSize32Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit32_radix2_sse2",
		Priority:   17, // Lower priority than radix-32 (??)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 32: Mixed-radix-2/4 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size32Mixed24Complex64Asm, bitrevSize32Radix2)), // Mixed 2/4 for size 32 is radix-2 reversal
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size32Mixed24Complex64Asm, bitrevSize32Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit32_mixed24_sse2",
		Priority:   19, // Higher priority than radix-2 (17)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-4 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size64Radix4Complex64Asm, bitrevSize64Radix4)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size64Radix4Complex64Asm, bitrevSize64Radix4)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit64_radix4_sse2",
		Priority:   18, // Between generic (15) and AVX2 (20-25)
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size64Radix2Complex64Asm, bitrevSize64Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size64Radix2Complex64Asm, bitrevSize64Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit64_radix2_sse2",
		Priority:   17, // Lower priority than radix-4
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Mixed Radix-2/4 SSE2 variant (3 radix-4 + 1 radix-2 stages)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size128Mixed24Complex64Asm, bitrevSize128Radix4)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size128Mixed24Complex64Asm, bitrevSize128Radix4)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit128_mixed24_sse2",
		Priority:   17,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Radix-2 SSE2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCore64(wrapAsmDIT64(amd64.ForwardSSE2Size128Radix2Complex64Asm, bitrevSize128Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(amd64.InverseSSE2Size128Radix2Complex64Asm, bitrevSize128Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit128_radix2_sse2",
		Priority:   17,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
}

// registerSSE2DITCodelets128 registers SSE2-optimized complex128 DIT codelets.
func registerSSE2DITCodelets128() {
	// Size 4: Radix-4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       4,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size4Radix4Complex128Asm, nil)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size4Radix4Complex128Asm, nil)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit4_radix4_sse2",
		Priority:   5, // Lower priority - scalar ops may not beat generic
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained, no external bitrev
	})

	// Size 8: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size8Radix2Complex128Asm, bitrevSize8Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size8Radix2Complex128Asm, bitrevSize8Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix2_sse2",
		Priority:   17,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 8: Radix-8 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size8Radix8Complex128Asm, nil)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size8Radix8Complex128Asm, nil)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix8_sse2",
		Priority:   30,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 8: Radix-4 (Mixed-radix) SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size8Radix4Complex128Asm, bitrevSize8Radix4)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size8Radix4Complex128Asm, bitrevSize8Radix4)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit8_radix4_sse2",
		Priority:   18,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size16Radix2Complex128Asm, bitrevSize16Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size16Radix2Complex128Asm, bitrevSize16Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix2_sse2",
		Priority:   17,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 16: Radix-4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size16Radix4Complex128Asm, bitrevSize16Radix4)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size16Radix4Complex128Asm, bitrevSize16Radix4)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit16_radix4_sse2",
		Priority:   18,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 32: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size32Radix2Complex128Asm, bitrevSize32Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size32Radix2Complex128Asm, bitrevSize32Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit32_radix2_sse2",
		Priority:   17,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 32: Mixed-radix-2/4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size32Mixed24Complex128Asm, bitrevSize32Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size32Mixed24Complex128Asm, bitrevSize32Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit32_mixed24_sse2",
		Priority:   19,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size64Radix2Complex128Asm, bitrevSize64Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size64Radix2Complex128Asm, bitrevSize64Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit64_radix2_sse2",
		Priority:   18,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: Radix-4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size64Radix4Complex128Asm, bitrevSize64Radix4)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size64Radix4Complex128Asm, bitrevSize64Radix4)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit64_radix4_sse2",
		Priority:   18,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Radix-2 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size128Radix2Complex128Asm, bitrevSize128Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size128Radix2Complex128Asm, bitrevSize128Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit128_radix2_sse2",
		Priority:   17,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: Mixed-radix-2/4 SSE2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCore128(wrapAsmDIT128(amd64.ForwardSSE2Size128Mixed24Complex128Asm, bitrevSize128Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(amd64.InverseSSE2Size128Mixed24Complex128Asm, bitrevSize128Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDSSE2,
		Signature:  "dit128_mixed24_sse2",
		Priority:   18,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
}
