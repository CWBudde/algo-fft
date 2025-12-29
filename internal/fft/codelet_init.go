package fft

// This file registers all built-in codelets with the global registries.
// Registration happens at init time so codelets are available when plans are created.

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
		Priority:   20, // Higher priority - default choice
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
		Priority:   20, // Higher priority - default choice
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
