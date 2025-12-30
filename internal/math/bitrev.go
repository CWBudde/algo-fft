package math

// ComputeBitReversalIndices returns the bit-reversal permutation indices
// for a size-n radix-2 FFT.
func ComputeBitReversalIndices(n int) []int {
	if n <= 0 {
		return nil
	}

	bitrev := make([]int, n)
	bits := Log2(n)

	for i := range n {
		bitrev[i] = ReverseBits(i, bits)
	}

	return bitrev
}

// Log2 returns the base-2 logarithm of n (assuming n is a power of 2).
func Log2(n int) int {
	result := 0

	for n > 1 {
		n >>= 1
		result++
	}

	return result
}

// log2 is a private alias for Log2.
func log2(n int) int {
	return Log2(n)
}

// ReverseBits reverses the lower 'bits' bits of x.
// Example: ReverseBits(6, 3) = ReverseBits(0b110, 3) = 0b011 = 3.
func ReverseBits(x, bits int) int {
	result := 0
	for range bits {
		result = (result << 1) | (x & 1)
		x >>= 1
	}

	return result
}

// reverseBits is a private alias for ReverseBits.
func reverseBits(x, bits int) int {
	return ReverseBits(x, bits)
}
