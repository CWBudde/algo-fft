package fft

import "fmt"

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
	type funcs struct {
		forward KernelFunc64
		inverse KernelFunc64
	}

	codeletFuncs := map[int]funcs{
		8:   {forwardDIT8Complex64, inverseDIT8Complex64},
		16:  {forwardDIT16Complex64, inverseDIT16Complex64},
		32:  {forwardDIT32Complex64, inverseDIT32Complex64},
		64:  {forwardDIT64Complex64, inverseDIT64Complex64},
		128: {forwardDIT128Complex64, inverseDIT128Complex64},
	}

	for size, f := range codeletFuncs {
		Registry64.Register(CodeletEntry[complex64]{
			Size:      size,
			Forward:   wrapCodelet64(f.forward),
			Inverse:   wrapCodelet64(f.inverse),
			Algorithm: KernelDIT,
			SIMDLevel: SIMDNone,
			Signature: fmt.Sprintf("dit%d_generic", size),
			Priority:  0,
		})
	}
}

// registerDITCodelets128 registers all complex128 DIT codelets.
func registerDITCodelets128() {
	type funcs struct {
		forward KernelFunc128
		inverse KernelFunc128
	}

	codeletFuncs := map[int]funcs{
		8:   {forwardDIT8Complex128, inverseDIT8Complex128},
		16:  {forwardDIT16Complex128, inverseDIT16Complex128},
		32:  {forwardDIT32Complex128, inverseDIT32Complex128},
		64:  {forwardDIT64Complex128, inverseDIT64Complex128},
		128: {forwardDIT128Complex128, inverseDIT128Complex128},
	}

	for size, f := range codeletFuncs {
		Registry128.Register(CodeletEntry[complex128]{
			Size:      size,
			Forward:   wrapCodelet128(f.forward),
			Inverse:   wrapCodelet128(f.inverse),
			Algorithm: KernelDIT,
			SIMDLevel: SIMDNone,
			Signature: fmt.Sprintf("dit%d_generic", size),
			Priority:  0,
		})
	}
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
