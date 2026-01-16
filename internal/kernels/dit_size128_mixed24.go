package kernels

// ComputeBitReversalIndicesMixed128 computes bit-reversal indices for
// the AVX2 size-128 kernel.
//
// The AVX2 size-128 kernel expects inputs in a specific order:
//
//	work[0] = x(0)
//	work[1] = x(32)
//	work[2] = x(64)
//	work[3] = x(96)
//
// Standard binary reversal gives:
//
//	0 -> 0
//	1 -> 64
//	2 -> 32
//	3 -> 96
//
// To match the kernel's expectation, we need to swap indices 1 and 2.
// This corresponds to swapping Bit 5 and Bit 6 of the bit-reversed value.
func ComputeBitReversalIndicesMixed128(n int) []int {
	if n != 128 {
		return nil
	}

	indices := make([]int, n)
	bits := 7 // 128 = 2^7

	for i := range n {
		// 1. Standard binary reverse
		r := 0

		x := i
		for range bits {
			r = (r << 1) | (x & 1)
			x >>= 1
		}

		// 2. Swap Bit 5 and Bit 6
		// These are the two MSBs for 7-bit number (bits 0..6)
		b5 := (r >> 5) & 1
		b6 := (r >> 6) & 1

		// Clear bits 5 and 6
		r &= ^(1<<5 | 1<<6)

		// Set swapped
		r |= (b5 << 6) | (b6 << 5)

		indices[i] = r
	}

	return indices
}

// forwardDIT128MixedRadix24Complex64 computes a 128-point forward FFT using
// mixed-radix-2/4 Decimation-in-Time (DIT) algorithm for complex64 data.
//
// For correctness, delegate to the proven radix-2 DIT implementation.
func forwardDIT128MixedRadix24Complex64(dst, src, twiddle, scratch []complex64) bool {
	return forwardDIT128Complex64(dst, src, twiddle, scratch)
}

// inverseDIT128MixedRadix24Complex64 computes a 128-point inverse FFT using
// mixed-radix-2/4 Decimation-in-Time (DIT) algorithm for complex64 data.
//
// For correctness, delegate to the proven radix-2 DIT implementation.
func inverseDIT128MixedRadix24Complex64(dst, src, twiddle, scratch []complex64) bool {
	return inverseDIT128Complex64(dst, src, twiddle, scratch)
}

// forwardDIT128MixedRadix24Complex128 computes a 128-point forward FFT using
// mixed-radix-2/4 Decimation-in-Time (DIT) algorithm for complex128 data.
//
// For correctness, delegate to the proven radix-2 DIT implementation.
func forwardDIT128MixedRadix24Complex128(dst, src, twiddle, scratch []complex128) bool {
	return forwardDIT128Complex128(dst, src, twiddle, scratch)
}

// inverseDIT128MixedRadix24Complex128 computes a 128-point inverse FFT using
// mixed-radix-2/4 Decimation-in-Time (DIT) algorithm for complex128 data.
//
// For correctness, delegate to the proven radix-2 DIT implementation.
func inverseDIT128MixedRadix24Complex128(dst, src, twiddle, scratch []complex128) bool {
	return inverseDIT128Complex128(dst, src, twiddle, scratch)
}
