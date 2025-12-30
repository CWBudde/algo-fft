package math

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

// IsPowerOf reports whether n is a positive integer power of the given base.
// For example, IsPowerOf(125, 5) returns true because 125 = 5^3.
func IsPowerOf(n, base int) bool {
	if n < 1 || base < 2 {
		return false
	}

	for n%base == 0 {
		n /= base
	}

	return n == 1
}

// isPowerOf is a private alias for IsPowerOf.
func isPowerOf(n, base int) bool {
	return IsPowerOf(n, base)
}

// IsPowerOf3 reports whether n is a positive power of three (3^k for some k >= 0).
func IsPowerOf3(n int) bool {
	return IsPowerOf(n, 3)
}

// isPowerOf3 is a private alias for IsPowerOf3.
func isPowerOf3(n int) bool {
	return IsPowerOf3(n)
}

// IsPowerOf4 reports whether n is a positive power of four (4^k for some k >= 0).
// This is equivalent to n being a power of 2 with an even log2.
func IsPowerOf4(n int) bool {
	if !IsPowerOf2(n) {
		return false
	}

	return Log2(n)%2 == 0
}

// isPowerOf4 is a private alias for IsPowerOf4.
func isPowerOf4(n int) bool {
	return IsPowerOf4(n)
}

// IsPowerOf5 reports whether n is a positive power of five (5^k for some k >= 0).
func IsPowerOf5(n int) bool {
	return IsPowerOf(n, 5)
}

// isPowerOf5 is a private alias for IsPowerOf5.
func isPowerOf5(n int) bool {
	return IsPowerOf5(n)
}
