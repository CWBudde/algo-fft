package fft

// Note: Split-radix FFT implementation is not yet implemented.
// Split-radix typically combines one radix-2 FFT and two radix-4 FFTs per stage:
// N -> N/2 (even indices) and N/4 (odd indices), reducing multiplications.
//
//nolint:unused
func splitRadixFFT[T Complex](dst, src, twiddle, scratch []T, bitrev []int, inverse bool) bool {
	_ = dst
	_ = src
	_ = twiddle
	_ = scratch
	_ = bitrev
	_ = inverse

	return false
}
