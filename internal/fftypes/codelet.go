package fftypes

// CodeletFunc is a kernel function for a specific fixed size.
// Unlike Kernel[T], codelets have a hardcoded size and perform no runtime checks.
// The caller guarantees that all slices have the required length.
type CodeletFunc[T Complex] func(dst, src, twiddle, scratch []T, bitrev []int)

// BitrevFunc generates bit-reversal indices for a given size.
// Returns nil if no bit-reversal is needed (e.g., size 4 radix-4).
type BitrevFunc func(n int) []int
