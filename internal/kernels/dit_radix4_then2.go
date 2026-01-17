package kernels

// forwardRadix4Then2Complex64 computes a forward FFT using radix-4-then-2
// Decimation-in-Time (DIT) algorithm for complex64 data.
//
// This function is optimized for power-of-2 sizes with ODD log2 exponents
// (e.g., 8, 32, 128, 512, 2048, 8192) which cannot use pure radix-4.
//
// Algorithm:
//   - Stage 1: ONE radix-2 stage (standard DIT butterfly)
//   - Stages 2+: Pure radix-4 stages (reusing butterfly4Forward)
//
// This reduces the total number of stages significantly:
//   - Size 512 (2^9): 9 radix-2 stages → 1 radix-2 + 4 radix-4 = 5 total
//   - Size 2048 (2^11): 11 radix-2 → 1 radix-2 + 5 radix-4 = 6 total
//
// Expected speedup: 30-40% over pure radix-2 for affected sizes.
//
// Returns false if any slice is too small or if size is not power-of-2.
//

func forwardRadix4Then2Complex64(dst, src, twiddle, scratch []complex64) bool {
	// NOTE: The previous radix-4-then-2 implementation for odd log2 sizes was
	// incorrect and produced wrong results (e.g., impulse DC bin doubled for n=2048).
	// For correctness, delegate to the proven radix-2 DIT implementation.
	return ditForward[complex64](dst, src, twiddle, scratch)
}

// inverseRadix4Then2Complex64 computes an inverse FFT using radix-4-then-2
// Decimation-in-Time (DIT) algorithm for complex64 data.
//
// This function is optimized for power-of-2 sizes with ODD log2 exponents
// (e.g., 8, 32, 128, 512, 2048, 8192) which cannot use pure radix-4.
//
// Algorithm:
//   - Stage 1: ONE radix-2 stage (standard DIT butterfly)
//   - Stages 2+: Pure radix-4 stages (reusing butterfly4Inverse)
//   - Final: 1/N scaling
//
// This reduces the total number of stages significantly:
//   - Size 512 (2^9): 9 radix-2 stages → 1 radix-2 + 4 radix-4 = 5 total
//   - Size 2048 (2^11): 11 radix-2 → 1 radix-2 + 5 radix-4 = 6 total
//
// Expected speedup: 30-40% over pure radix-2 for affected sizes.
//
// Returns false if any slice is too small or if size is not power-of-2.
//

func inverseRadix4Then2Complex64(dst, src, twiddle, scratch []complex64) bool {
	// See forwardRadix4Then2Complex64: delegate to the proven radix-2 inverse.
	return ditInverseComplex64(dst, src, twiddle, scratch)
}
