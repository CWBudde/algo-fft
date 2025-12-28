package fft

import "math/bits"

// IsPowerOf2 reports whether n is a positive power of two.
func IsPowerOf2(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// NextPowerOfTwo returns the smallest power of two greater than or equal to n.
// For n <= 1, returns 1.
func NextPowerOfTwo(n int) int {
	if n <= 1 {
		return 1
	}

	if IsPowerOf2(n) {
		return n
	}

	x := uint(n - 1)
	x |= x >> 1
	x |= x >> 2
	x |= x >> 4
	x |= x >> 8

	x |= x >> 16
	if bits.UintSize == 64 {
		x |= x >> 32
	}

	return int(x + 1)
}

// isPowerOf reports whether n is a positive integer power of the given base.
// For example, isPowerOf(125, 5) returns true because 125 = 5^3.
func isPowerOf(n, base int) bool {
	if n < 1 || base < 2 {
		return false
	}

	for n%base == 0 {
		n /= base
	}

	return n == 1
}

// isPowerOf3 reports whether n is a positive power of three (3^k for some k >= 0).
func isPowerOf3(n int) bool {
	return isPowerOf(n, 3)
}

// isPowerOf4 reports whether n is a positive power of four (4^k for some k >= 0).
// This is equivalent to n being a power of 2 with an even log2.
func isPowerOf4(n int) bool {
	if !IsPowerOf2(n) {
		return false
	}

	return log2(n)%2 == 0
}

// isPowerOf5 reports whether n is a positive power of five (5^k for some k >= 0).
func isPowerOf5(n int) bool {
	return isPowerOf(n, 5)
}
