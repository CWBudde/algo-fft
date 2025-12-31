package math

import "math/bits"

// ComputeBitReversalIndices returns the bit-reversal permutation indices
// for a size-n radix-2 FFT.
func ComputeBitReversalIndices(n int) []int {
	if n <= 0 {
		return nil
	}

	bitrev := make([]int, n)
	nbits := bits.Len(uint(n)) - 1

	for i := range n {
		bitrev[i] = ReverseBits(i, nbits)
	}

	return bitrev
}

// log2 returns the base-2 logarithm of n (assuming n is a power of 2).
// Uses bits.Len() for efficiency. This is internal to the math package
// and should not be exported. Other packages should use bits.Len() directly.
func log2(n int) int {
	return bits.Len(uint(n)) - 1
}

// ReverseBits reverses the lower 'nbits' bits of x using hardware bit reversal.
// Example: ReverseBits(6, 3) = ReverseBits(0b110, 3) = 0b011 = 3.
func ReverseBits(x, nbits int) int {
	if nbits <= 0 {
		return 0
	}
	// Mask to keep only the lower nbits bits, then reverse using hardware instruction,
	// then shift right to position the reversed bits at the lower end.
	mask := uint64((1 << uint(nbits)) - 1)
	masked := uint64(x) & mask
	reversed := bits.Reverse64(masked)
	return int(reversed >> uint(64-nbits))
}
