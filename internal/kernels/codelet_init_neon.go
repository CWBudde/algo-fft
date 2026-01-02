//go:build arm64 && asm && !purego

package kernels

import arm64 "github.com/MeKo-Christian/algo-fft/internal/asm/arm64"

// registerNEONDITCodelets64 registers NEON-optimized complex64 DIT codelets.
func registerNEONDITCodelets64() {
	// Size 4: Radix-4 NEON variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize4Radix4Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize4Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit4_radix4_neon",
		Priority:   10,
		BitrevFunc: nil,
	})

	// Size 8: prefer radix-8 NEON, then radix-4, then radix-2
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize8Radix2Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize8Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit8_radix2_neon",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize8Radix4Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize8Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit8_radix4_neon",
		Priority:   25,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize8Radix8Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize8Radix8Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit8_radix8_neon",
		Priority:   30,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 16: radix-4 NEON beats radix-2 NEON
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize16Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize16Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit16_radix2_neon",
		Priority:   22,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize16Radix4Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize16Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit16_radix4_neon",
		Priority:   28,
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 32: mixed-24 preferred over radix-2
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize32Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize32Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit32_radix2_neon",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize32MixedRadix24Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize32MixedRadix24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit32_mixed24_neon",
		Priority:   24,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 64: radix-4 NEON beats radix-2 NEON
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize64Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize64Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit64_radix2_neon",
		Priority:   22,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize64Radix4Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize64Radix4Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit64_radix4_neon",
		Priority:   28,
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 128: mixed-24 preferred over radix-2
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize128Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize128Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit128_radix2_neon",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize128MixedRadix24Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize128MixedRadix24Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit128_mixed24_neon",
		Priority:   24,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 256: radix-2 NEON variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCodelet64(arm64.ForwardNEONSize256Radix2Complex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONSize256Radix2Complex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit256_radix2_neon",
		Priority:   18,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 512: generic NEON radix-2 kernel
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCodelet64(arm64.ForwardNEONComplex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONComplex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit512_radix2_neon",
		Priority:   1,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 1024: generic NEON radix-2 kernel
	Registry64.Register(CodeletEntry[complex64]{
		Size:       1024,
		Forward:    wrapCodelet64(arm64.ForwardNEONComplex64Asm),
		Inverse:    wrapCodelet64(arm64.InverseNEONComplex64Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit1024_radix2_neon",
		Priority:   1,
		BitrevFunc: ComputeBitReversalIndices,
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
		Forward:    wrapCodelet128(arm64.ForwardNEONComplex128Asm),
		Inverse:    wrapCodelet128(arm64.InverseNEONComplex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit32_radix2_neon",
		Priority:   1,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCodelet128(arm64.ForwardNEONComplex128Asm),
		Inverse:    wrapCodelet128(arm64.InverseNEONComplex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit64_radix2_neon",
		Priority:   1,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCodelet128(arm64.ForwardNEONComplex128Asm),
		Inverse:    wrapCodelet128(arm64.InverseNEONComplex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit128_radix2_neon",
		Priority:   1,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCodelet128(arm64.ForwardNEONComplex128Asm),
		Inverse:    wrapCodelet128(arm64.InverseNEONComplex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit256_radix2_neon",
		Priority:   1,
		BitrevFunc: ComputeBitReversalIndices,
	})
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCodelet128(arm64.ForwardNEONComplex128Asm),
		Inverse:    wrapCodelet128(arm64.InverseNEONComplex128Asm),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNEON,
		Signature:  "dit512_radix2_neon",
		Priority:   1,
		BitrevFunc: ComputeBitReversalIndices,
	})
}
