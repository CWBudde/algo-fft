package fft

// RadixKernel describes a pluggable radix implementation for mixed-radix FFTs.
// Implementations should return false when they do not support the given length.
type RadixKernel[T Complex] struct {
	Name    string
	Radix   int
	Forward Kernel[T]
	Inverse Kernel[T]
}
