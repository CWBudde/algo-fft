package fftypes

// KernelType classifies how a kernel handles input permutation (bit-reversal).
// This enables the transition from caller-controlled to kernel-controlled permutation.
type KernelType int

const (
	// KernelTypeLegacy uses the old caller-controlled bitrev parameter.
	// Used during migration - kernels still accept bitrev []int parameter.
	KernelTypeLegacy KernelType = iota

	// KernelTypeCore is a self-contained kernel with no external bitrev.
	// The kernel handles all permutation internally or doesn't need any.
	// Examples: Stockham autosort, single-stage radix-8/16/32 butterflies.
	KernelTypeCore

	// KernelTypeDIT is a DIT kernel with hardcoded internal bit-reversal.
	// The kernel precomputes and applies its own permutation.
	// Examples: Multi-stage DIT kernels with fixed radix.
	KernelTypeDIT
)

// String returns a human-readable name for the kernel type.
func (kt KernelType) String() string {
	switch kt {
	case KernelTypeLegacy:
		return "legacy"
	case KernelTypeCore:
		return "core"
	case KernelTypeDIT:
		return "dit"
	default:
		return "unknown"
	}
}

// CodeletFunc is a kernel function for a specific fixed size (legacy signature).
// Unlike Kernel[T], codelets have a hardcoded size and perform no runtime checks.
// The caller guarantees that all slices have the required length.
//
// Deprecated: Use CoreCodeletFunc for new kernels. This signature is kept
// for backward compatibility during the Phase 11 migration.
type CodeletFunc[T Complex] func(dst, src, twiddle, scratch []T, bitrev []int)

// CoreCodeletFunc is the new kernel function signature without bitrev parameter.
// Kernels using this signature handle permutation internally.
// This is the target signature for all kernels after Phase 11 migration.
type CoreCodeletFunc[T Complex] func(dst, src, twiddle, scratch []T)

// BitrevFunc generates bit-reversal indices for a given size.
// Returns nil if no bit-reversal is needed (e.g., size 4 radix-4).
//
// Deprecated: Will be removed after Phase 11 migration when all kernels
// handle permutation internally.
type BitrevFunc func(n int) []int
