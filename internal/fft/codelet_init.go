package fft

// This file registers all built-in codelets with the global registries.
// Registration happens at init time so codelets are available when plans are created.

func init() {
	// Register complex64 DIT codelets
	registerDITCodelets64()

	// Register complex128 DIT codelets
	registerDITCodelets128()
}

// registerDITCodelets64 registers all complex64 DIT codelets.
func registerDITCodelets64() {
	// Size 8
	Registry64.Register(CodeletEntry[complex64]{
		Size:      8,
		Forward:   wrapCodelet64(forwardDIT8Complex64),
		Inverse:   wrapCodelet64(inverseDIT8Complex64),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit8_generic",
		Priority:  0,
	})

	// Size 16
	Registry64.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   wrapCodelet64(forwardDIT16Complex64),
		Inverse:   wrapCodelet64(inverseDIT16Complex64),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit16_generic",
		Priority:  0,
	})

	// Size 32
	Registry64.Register(CodeletEntry[complex64]{
		Size:      32,
		Forward:   wrapCodelet64(forwardDIT32Complex64),
		Inverse:   wrapCodelet64(inverseDIT32Complex64),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit32_generic",
		Priority:  0,
	})

	// Size 64
	Registry64.Register(CodeletEntry[complex64]{
		Size:      64,
		Forward:   wrapCodelet64(forwardDIT64Complex64),
		Inverse:   wrapCodelet64(inverseDIT64Complex64),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit64_generic",
		Priority:  0,
	})

	// Size 128
	Registry64.Register(CodeletEntry[complex64]{
		Size:      128,
		Forward:   wrapCodelet64(forwardDIT128Complex64),
		Inverse:   wrapCodelet64(inverseDIT128Complex64),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit128_generic",
		Priority:  0,
	})
}

// registerDITCodelets128 registers all complex128 DIT codelets.
func registerDITCodelets128() {
	// Size 8
	Registry128.Register(CodeletEntry[complex128]{
		Size:      8,
		Forward:   wrapCodelet128(forwardDIT8Complex128),
		Inverse:   wrapCodelet128(inverseDIT8Complex128),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit8_generic",
		Priority:  0,
	})

	// Size 16
	Registry128.Register(CodeletEntry[complex128]{
		Size:      16,
		Forward:   wrapCodelet128(forwardDIT16Complex128),
		Inverse:   wrapCodelet128(inverseDIT16Complex128),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit16_generic",
		Priority:  0,
	})

	// Size 32
	Registry128.Register(CodeletEntry[complex128]{
		Size:      32,
		Forward:   wrapCodelet128(forwardDIT32Complex128),
		Inverse:   wrapCodelet128(inverseDIT32Complex128),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit32_generic",
		Priority:  0,
	})

	// Size 64
	Registry128.Register(CodeletEntry[complex128]{
		Size:      64,
		Forward:   wrapCodelet128(forwardDIT64Complex128),
		Inverse:   wrapCodelet128(inverseDIT64Complex128),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit64_generic",
		Priority:  0,
	})

	// Size 128
	Registry128.Register(CodeletEntry[complex128]{
		Size:      128,
		Forward:   wrapCodelet128(forwardDIT128Complex128),
		Inverse:   wrapCodelet128(inverseDIT128Complex128),
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit128_generic",
		Priority:  0,
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
