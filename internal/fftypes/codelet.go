package fftypes

// KernelType classifies how a kernel handles input permutation (bit-reversal).
// This enables the transition from caller-controlled to kernel-controlled permutation.
type KernelType int

const (
	// KernelTypeCore is a self-contained kernel with no external bitrev.
	// The kernel handles all permutation internally or doesn't need any.
	// Examples: Stockham autosort, single-stage radix-8/16/32 butterflies.
	KernelTypeCore KernelType = iota

	// KernelTypeDIT is a DIT kernel with hardcoded internal bit-reversal.
	// The kernel precomputes and applies its own permutation.
	// Examples: Multi-stage DIT kernels with fixed radix.
	KernelTypeDIT
)

// String returns a human-readable name for the kernel type.
func (kt KernelType) String() string {
	switch kt {
	case KernelTypeCore:
		return "core"
	case KernelTypeDIT:
		return "dit"
	default:
		return "unknown"
	}
}

// CodeletFunc is a kernel function for a specific fixed size.
// Unlike Kernel[T], codelets have a hardcoded size and perform no runtime checks.
// The caller guarantees that all slices have the required length.
//
// Parameters:
//   - dst, src: Input/output data (len >= codelet size)
//   - twiddle: Codelet-specific twiddle data (may be standard twiddles or SIMD-friendly layout)
//   - scratch: Working buffer for intermediate results
//
// Kernels using this signature handle permutation internally.
type CodeletFunc[T Complex] func(dst, src, twiddle, scratch []T)
