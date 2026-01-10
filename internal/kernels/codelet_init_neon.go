//go:build arm64 && asm && !purego

package kernels

import (
	arm64 "github.com/MeKo-Christian/algo-fft/internal/asm/arm64"
)

// registerNEONDITCodelets64 registers NEON-optimized complex64 DIT codelets.
func registerNEONDITCodelets64() {
	// Size 4: Radix-4 NEON variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONSize4Radix4Complex64Asm, bitrevSize4Identity)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONSize4Radix4Complex64Asm, bitrevSize4Identity)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit4_radix4_neon",
		Priority:   10,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT, // Self-contained, no external bitrev
	})

	// Size 8: prefer radix-8 NEON, then radix-4, then radix-2
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(arm64.ForwardNEONSize8Radix2Complex64Asm),
		Inverse:    wrapCore64(arm64.InverseNEONSize8Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit8_radix2_neon",
		Priority:   20,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONSize8Radix4Complex64Asm, bitrevSize8Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONSize8Radix4Complex64Asm, bitrevSize8Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit8_radix4_neon",
		Priority:   25,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	// Size 8: Radix-8 NEON variant (single-stage butterfly, identity permutation)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONSize8Radix8Complex64Asm, bitrevSize8Identity)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONSize8Radix8Complex64Asm, bitrevSize8Identity)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit8_radix8_neon",
		Priority:   30,
		BitrevFunc: nil,
		KernelType: KernelTypeCore, // Identity permutation, no bitrev needed
	})

	// Size 16: radix-4 NEON beats radix-2 NEON
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(arm64.ForwardNEONSize16Radix2Complex64Asm),
		Inverse:    wrapCore64(arm64.InverseNEONSize16Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit16_radix2_neon",
		Priority:   22,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONSize16Radix4Complex64Asm, bitrevSize16Radix4)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONSize16Radix4Complex64Asm, bitrevSize16Radix4)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit16_radix4_neon",
		Priority:   28,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 32: mixed-24 preferred over radix-2
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCore64(arm64.ForwardNEONSize32Radix2Complex64Asm),
		Inverse:    wrapCore64(arm64.InverseNEONSize32Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit32_radix2_neon",
		Priority:   20,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONSize32MixedRadix24Complex64Asm, bitrevSize32Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONSize32MixedRadix24Complex64Asm, bitrevSize32Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit32_mixed24_neon",
		Priority:   24,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 64: radix-4 NEON beats radix-2 NEON
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCore64(arm64.ForwardNEONSize64Radix2Complex64Asm),
		Inverse:    wrapCore64(arm64.InverseNEONSize64Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit64_radix2_neon",
		Priority:   22,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONSize64Radix4Complex64Asm, bitrevSize64Radix4)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONSize64Radix4Complex64Asm, bitrevSize64Radix4)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit64_radix4_neon",
		Priority:   28,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 128: mixed-24 preferred over radix-2
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONSize128Complex64Asm, bitrevSize128Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONSize128Complex64Asm, bitrevSize128Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit128_radix2_neon",
		Priority:   20,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONSize128MixedRadix24Complex64Asm, bitrevSize128Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONSize128MixedRadix24Complex64Asm, bitrevSize128Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit128_mixed24_neon",
		Priority:   24,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 256: radix-2 NEON variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONSize256Radix2Complex64Asm, bitrevSize256Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONSize256Radix2Complex64Asm, bitrevSize256Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit256_radix2_neon",
		Priority:   18,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 512: generic NEON radix-2 kernel
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONComplex64Asm, bitrevSize512Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONComplex64Asm, bitrevSize512Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit512_radix2_neon",
		Priority:   1,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})

	// Size 1024: generic NEON radix-2 kernel
	Registry64.Register(CodeletEntry[complex64]{
		Size:       1024,
		Forward:    wrapCore64(wrapAsmDIT64(arm64.ForwardNEONComplex64Asm, bitrevSize1024Radix2)),
		Inverse:    wrapCore64(wrapAsmDIT64(arm64.InverseNEONComplex64Asm, bitrevSize1024Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit1024_radix2_neon",
		Priority:   1,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
}

// registerNEONDITCodelets128 registers NEON-optimized complex128 DIT codelets.
//
// Note: At the moment these entries route to the generic NEON complex128 asm kernel
// (radix-2 DIT) rather than fully-unrolled, size-specific kernels.
func registerNEONDITCodelets128() {
	// Start at 32 to avoid overriding the tiny / radix-8 Go codelets.
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCore128(wrapAsmDIT128(arm64.ForwardNEONComplex128Asm, bitrevSize32Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(arm64.InverseNEONComplex128Asm, bitrevSize32Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit32_radix2_neon",
		Priority:   1,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCore128(wrapAsmDIT128(arm64.ForwardNEONComplex128Asm, bitrevSize64Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(arm64.InverseNEONComplex128Asm, bitrevSize64Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit64_radix2_neon",
		Priority:   1,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCore128(wrapAsmDIT128(arm64.ForwardNEONComplex128Asm, bitrevSize128Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(arm64.InverseNEONComplex128Asm, bitrevSize128Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit128_radix2_neon",
		Priority:   1,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCore128(wrapAsmDIT128(arm64.ForwardNEONComplex128Asm, bitrevSize256Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(arm64.InverseNEONComplex128Asm, bitrevSize256Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit256_radix2_neon",
		Priority:   1,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCore128(wrapAsmDIT128(arm64.ForwardNEONComplex128Asm, bitrevSize512Radix2)),
		Inverse:    wrapCore128(wrapAsmDIT128(arm64.InverseNEONComplex128Asm, bitrevSize512Radix2)),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit512_radix2_neon",
		Priority:   1,
		BitrevFunc: nil,
		KernelType: KernelTypeDIT,
	})
}
