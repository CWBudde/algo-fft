package math

import "math/bits"

// ComputePermutationIndices computes index permutation for FFT algorithms.
// It supports:
//   - radix = 0: Identity permutation (no reordering, e.g., radix-8, radix-16)
//   - radix = 2: Bit-reversal permutation (standard radix-2 FFT)
//   - radix = 4: Digit-reversal in base-4 (radix-4 FFT)
//   - radix = mixed24: Mixed radix-2/4 (one binary bit + quaternary digits)
//
// For radix = 0, returns [0, 1, 2, ..., n-1].
// For radix > 0, returns digit-reversed permutation in the given radix.
// For radix = mixed24, n must be 2 * 4^k for some k >= 1.
//
// Returns nil if n is invalid for the given radix.
func ComputePermutationIndices(n int, radix int) []int {
	if n <= 0 {
		return nil
	}

	switch radix {
	case 0:
		// Identity permutation
		indices := make([]int, n)
		for i := range n {
			indices[i] = i
		}

		return indices

	case 2:
		// Bit-reversal (radix-2) - uses hardware bit reversal
		bitrev := make([]int, n)

		nbits := bits.Len(uint(n)) - 1
		for i := range n {
			bitrev[i] = ReverseBits(i, nbits)
		}

		return bitrev

	case 4:
		// Digit-reversal in base-4
		if n&(n-1) != 0 {
			return nil // Not a power of 2
		}

		return computeDigitReversal(n, 4)

	case -24: // Special marker for mixed radix-2/4
		// Mixed radix: n = 2 * 4^k
		return computeMixedRadix24(n)

	default:
		// General radix digit reversal
		return computeDigitReversal(n, radix)
	}
}

// computeDigitReversal computes digit-reversal permutation in the given radix.
// n must be a power of radix.
func computeDigitReversal(n int, radix int) []int {
	if radix <= 1 {
		return nil
	}

	// Calculate number of digits needed
	digits := 0

	temp := n
	for temp > 1 {
		if temp%radix != 0 {
			return nil // Not a power of radix
		}

		digits++
		temp /= radix
	}

	indices := make([]int, n)
	bitsPerDigit := bits.Len(uint(radix - 1))

	for i := range n {
		indices[i] = reverseDigits(i, digits, radix, bitsPerDigit)
	}

	return indices
}

// reverseDigits reverses the digits of x in the given radix.
func reverseDigits(x, digits, radix, bitsPerDigit int) int {
	result := 0
	mask := radix - 1

	for range digits {
		result = (result * radix) | (x & mask)
		x >>= bitsPerDigit
	}

	return result
}

// computeMixedRadix24 computes bit-reversal for mixed radix-2/4 FFT.
// n must be 2 * 4^k for some k >= 1.
func computeMixedRadix24(n int) []int {
	if n <= 0 || n%2 != 0 {
		return nil
	}

	m := n / 2 // m should be a power of 4
	if m < 4 || (m&(m-1)) != 0 {
		return nil
	}

	// Check that m is a power of 4
	temp := m
	for temp > 1 {
		if temp%4 != 0 {
			return nil
		}

		temp /= 4
	}

	// Count quaternary digits
	quatDigits := 0

	temp = m
	for temp > 1 {
		quatDigits++
		temp /= 4
	}

	indices := make([]int, n)
	half := n / 2

	for i := range n {
		// Split i into: binary_bit (MSB) and quaternary part (k digits)
		binaryBit := i / half // 0 or 1
		quatIndex := i % half // 0 to half-1

		// Reverse the quaternary digits
		revQuat := 0

		q := quatIndex
		for range quatDigits {
			revQuat = (revQuat << 2) | (q & 0x3)
			q >>= 2
		}

		// Reconstruct with binary bit in LSB position
		indices[i] = (revQuat << 1) | binaryBit
	}

	return indices
}

// ComputeBitReversalIndices returns the bit-reversal permutation indices
// for a size-n radix-2 FFT.
// This is a convenience wrapper around ComputePermutationIndices(n, 2).
func ComputeBitReversalIndices(n int) []int {
	return ComputePermutationIndices(n, 2)
}

// ComputeIdentityIndices returns identity permutation [0, 1, ..., n-1].
// This is a convenience wrapper around ComputePermutationIndices(n, 0).
func ComputeIdentityIndices(n int) []int {
	return ComputePermutationIndices(n, 0)
}

// ComputeBitReversalIndicesRadix4 returns digit-reversal permutation for radix-4 FFT.
// This is a convenience wrapper around ComputePermutationIndices(n, 4).
func ComputeBitReversalIndicesRadix4(n int) []int {
	return ComputePermutationIndices(n, 4)
}

// ComputeBitReversalIndicesRadix8 returns digit-reversal permutation for radix-8 FFT.
// This is a convenience wrapper around ComputePermutationIndices(n, 8).
func ComputeBitReversalIndicesRadix8(n int) []int {
	return ComputePermutationIndices(n, 8)
}

// ComputeBitReversalIndicesMixed24 returns permutation for mixed radix-2/4 FFT.
// This is a convenience wrapper around ComputePermutationIndices(n, -24).
func ComputeBitReversalIndicesMixed24(n int) []int {
	return ComputePermutationIndices(n, -24)
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
