package kernels


// This file registers all built-in codelets with the global registries.
// Registration happens at init time so codelets are available when plans are created.

//nolint:gochecknoinits
func init() {
	// Register complex64 DIT codelets
	registerDITCodelets64()

	// Register complex128 DIT codelets
	registerDITCodelets128()

	// Register AVX2 codelets (conditional on build tags)
	registerAVX2DITCodelets64()
	registerAVX2DITCodelets128()
}

// registerDITCodelets64 registers all complex64 DIT codelets with multiple radix variants.
func registerDITCodelets64() {
	// Size 4: Only radix-4 (no bit-reversal needed)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       4,
		Forward:    wrapCodelet64(forwardDIT4Radix4Complex64),
		Inverse:    wrapCodelet64(inverseDIT4Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit4_radix4_generic",
		Priority:   0,
		BitrevFunc: nil, // No bit-reversal for size 4
	})

	// Size 8: Radix-2 variant (default for now)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(forwardDIT8Radix2Complex64),
		Inverse:    wrapCodelet64(inverseDIT8Radix2Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_radix2_generic",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 8: Radix-8 variant (single butterfly)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(forwardDIT8Radix8Complex64),
		Inverse:    wrapCodelet64(inverseDIT8Radix8Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_radix8_generic",
		Priority:   30, // Highest priority among generic size-8 codelets
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 8: Mixed-radix variant (1x radix-4 + 1x radix-2) - TODO: investigate correctness
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8,
		Forward:    wrapCodelet64(forwardDIT8Radix4Complex64),
		Inverse:    wrapCodelet64(inverseDIT8Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_mixedradix_generic",
		Priority:   10,                        // Lower priority until correctness verified
		BitrevFunc: ComputeBitReversalIndices, // Still uses binary reversal (8 is not a power of 4)
	})

	// Size 16: Radix-2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(forwardDIT16Complex64),
		Inverse:    wrapCodelet64(inverseDIT16Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16_radix2_generic",
		Priority:   10,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 16: Radix-4 variant (12-15% faster per benchmarks)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       16,
		Forward:    wrapCodelet64(forwardDIT16Radix4Complex64),
		Inverse:    wrapCodelet64(inverseDIT16Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16_radix4_generic",
		Priority:   20, // Higher priority (empirically faster)
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 32: Radix-2 only
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(forwardDIT32Complex64),
		Inverse:    wrapCodelet64(inverseDIT32Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit32_radix2_generic",
		Priority:   0,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 32: Mixed-radix-2/4 variant (3 stages: 2 radix-4 + 1 radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       32,
		Forward:    wrapCodelet64(forwardDIT32MixedRadix24Complex64),
		Inverse:    wrapCodelet64(inverseDIT32MixedRadix24Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit32_mixed24_generic",
		Priority:   15, // Higher than base radix-2, lower than potential radix-8
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 64: Radix-2 only
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(forwardDIT64Complex64),
		Inverse:    wrapCodelet64(inverseDIT64Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit64_radix2_generic",
		Priority:   0,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 64: Radix-4 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       64,
		Forward:    wrapCodelet64(forwardDIT64Radix4Complex64),
		Inverse:    wrapCodelet64(inverseDIT64Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit64_radix4_generic",
		Priority:   20, // Higher priority (expected faster)
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 128: Radix-2 only
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(forwardDIT128Complex64),
		Inverse:    wrapCodelet64(inverseDIT128Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit128_radix2_generic",
		Priority:   0,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 128: Mixed-radix-2/4 variant (3 stages: 2 radix-4 + 1 radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       128,
		Forward:    wrapCodelet64(forwardDIT128MixedRadix24Complex64),
		Inverse:    wrapCodelet64(inverseDIT128MixedRadix24Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit128_mixed24_generic",
		Priority:   15, // Higher than base radix-2, lower than potential radix-8
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 256: Radix-2 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCodelet64(forwardDIT256Complex64),
		Inverse:    wrapCodelet64(inverseDIT256Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix2_generic",
		Priority:   10,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 256: Radix-4 variant
	Registry64.Register(CodeletEntry[complex64]{
		Size:       256,
		Forward:    wrapCodelet64(forwardDIT256Radix4Complex64),
		Inverse:    wrapCodelet64(inverseDIT256Radix4Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix4_generic",
		Priority:   20, // Higher priority (potentially faster)
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 512: Radix-2 only
	Registry64.Register(CodeletEntry[complex64]{
		Size:       512,
		Forward:    wrapCodelet64(forwardDIT512Complex64),
		Inverse:    wrapCodelet64(inverseDIT512Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_radix2_generic",
		Priority:   0,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 2048: Mixed-radix-2/4 variant (odd log2, faster than pure radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       2048,
		Forward:    wrapCodelet64(forwardMixedRadix24Complex64),
		Inverse:    wrapCodelet64(inverseMixedRadix24Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit2048_mixedradix24_generic",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 8192: Mixed-radix-2/4 variant (odd log2, faster than pure radix-2)
	Registry64.Register(CodeletEntry[complex64]{
		Size:       8192,
		Forward:    wrapCodelet64(forwardMixedRadix24Complex64),
		Inverse:    wrapCodelet64(inverseMixedRadix24Complex64),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8192_mixedradix24_generic",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})
}

// registerDITCodelets128 registers all complex128 DIT codelets with multiple radix variants.
func registerDITCodelets128() {
	// Size 4: Only radix-4 (no bit-reversal needed)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       4,
		Forward:    wrapCodelet128(forwardDIT4Radix4Complex128),
		Inverse:    wrapCodelet128(inverseDIT4Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit4_radix4_generic",
		Priority:   0,
		BitrevFunc: nil, // No bit-reversal for size 4
	})

	// Size 8: Radix-2 variant (default for now)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(forwardDIT8Radix2Complex128),
		Inverse:    wrapCodelet128(inverseDIT8Radix2Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_radix2_generic",
		Priority:   20,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 8: Radix-8 variant (single butterfly)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(forwardDIT8Radix8Complex128),
		Inverse:    wrapCodelet128(inverseDIT8Radix8Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_radix8_generic",
		Priority:   30, // Highest priority among generic size-8 codelets
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 8: Mixed-radix variant (1x radix-4 + 1x radix-2) - TODO: investigate correctness
	Registry128.Register(CodeletEntry[complex128]{
		Size:       8,
		Forward:    wrapCodelet128(forwardDIT8Radix4Complex128),
		Inverse:    wrapCodelet128(inverseDIT8Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit8_mixedradix_generic",
		Priority:   10,                        // Lower priority until correctness verified
		BitrevFunc: ComputeBitReversalIndices, // Still uses binary reversal (8 is not a power of 4)
	})

	// Size 16: Radix-2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCodelet128(forwardDIT16Complex128),
		Inverse:    wrapCodelet128(inverseDIT16Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16_radix2_generic",
		Priority:   10,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 16: Radix-4 variant (12-15% faster per benchmarks)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       16,
		Forward:    wrapCodelet128(forwardDIT16Radix4Complex128),
		Inverse:    wrapCodelet128(inverseDIT16Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit16_radix4_generic",
		Priority:   20, // Higher priority (empirically faster)
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 32: Radix-2 only
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCodelet128(forwardDIT32Complex128),
		Inverse:    wrapCodelet128(inverseDIT32Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit32_radix2_generic",
		Priority:   0,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 32: Mixed-radix-2/4 variant (3 stages: 2 radix-4 + 1 radix-2)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       32,
		Forward:    wrapCodelet128(forwardDIT32MixedRadix24Complex128),
		Inverse:    wrapCodelet128(inverseDIT32MixedRadix24Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit32_mixed24_generic",
		Priority:   15, // Higher than base radix-2, lower than potential radix-8
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 64: Radix-2 only
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCodelet128(forwardDIT64Complex128),
		Inverse:    wrapCodelet128(inverseDIT64Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit64_radix2_generic",
		Priority:   0,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 64: Radix-4 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       64,
		Forward:    wrapCodelet128(forwardDIT64Radix4Complex128),
		Inverse:    wrapCodelet128(inverseDIT64Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit64_radix4_generic",
		Priority:   20, // Higher priority (expected faster)
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 128: Radix-2 only
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCodelet128(forwardDIT128Complex128),
		Inverse:    wrapCodelet128(inverseDIT128Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit128_radix2_generic",
		Priority:   0,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 128: Mixed-radix-2/4 variant (3 stages: 2 radix-4 + 1 radix-2)
	Registry128.Register(CodeletEntry[complex128]{
		Size:       128,
		Forward:    wrapCodelet128(forwardDIT128MixedRadix24Complex128),
		Inverse:    wrapCodelet128(inverseDIT128MixedRadix24Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit128_mixed24_generic",
		Priority:   15, // Higher than base radix-2, lower than potential radix-8
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 256: Radix-2 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCodelet128(forwardDIT256Complex128),
		Inverse:    wrapCodelet128(inverseDIT256Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix2_generic",
		Priority:   10,
		BitrevFunc: ComputeBitReversalIndices,
	})

	// Size 256: Radix-4 variant
	Registry128.Register(CodeletEntry[complex128]{
		Size:       256,
		Forward:    wrapCodelet128(forwardDIT256Radix4Complex128),
		Inverse:    wrapCodelet128(inverseDIT256Radix4Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit256_radix4_generic",
		Priority:   20, // Higher priority (potentially faster)
		BitrevFunc: ComputeBitReversalIndicesRadix4,
	})

	// Size 512: Radix-2 only
	Registry128.Register(CodeletEntry[complex128]{
		Size:       512,
		Forward:    wrapCodelet128(forwardDIT512Complex128),
		Inverse:    wrapCodelet128(inverseDIT512Complex128),
		Algorithm:  KernelDIT,
		SIMDLevel:  SIMDNone,
		Signature:  "dit512_radix2_generic",
		Priority:   0,
		BitrevFunc: ComputeBitReversalIndices,
	})
}

// KernelFunc64 is the signature of existing complex64 kernels that return bool.
type KernelFunc64 func(dst, src, twiddle, scratch []complex64, bitrev []int) bool

// KernelFunc128 is the signature of existing complex128 kernels that return bool.
type KernelFunc128 func(dst, src, twiddle, scratch []complex128, bitrev []int) bool

// wrapCodelet64 adapts a bool-returning kernel to the CodeletFunc signature.
// The bool return is ignored because codelets trust their inputs are pre-validated.
func wrapCodelet64(fn KernelFunc64) CodeletFunc[complex64] {
	return func(dst, src, twiddle, scratch []complex64, bitrev []int) {
		fn(dst, src, twiddle, scratch, bitrev)
	}
}

// wrapCodelet128 adapts a bool-returning kernel to the CodeletFunc signature.
// The bool return is ignored because codelets trust their inputs are pre-validated.
func wrapCodelet128(fn KernelFunc128) CodeletFunc[complex128] {
	return func(dst, src, twiddle, scratch []complex128, bitrev []int) {
		fn(dst, src, twiddle, scratch, bitrev)
	}
}
