package fft

import "math"

// Complex is a type constraint for complex number types supported by the FFT internals.
type Complex interface {
	complex64 | complex128
}

// IsPowerOfTwo reports whether n is a positive power of two.
func IsPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// ComputeTwiddleFactors returns the precomputed twiddle factors (roots of unity)
// for a size-n FFT: W_n^k = exp(-2πik/n) for k = 0..n-1.
func ComputeTwiddleFactors[T Complex](n int) []T {
	if n <= 0 {
		return nil
	}

	twiddle := make([]T, n)
	for k := range n {
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		re := math.Cos(angle)
		im := math.Sin(angle)
		twiddle[k] = complexFromFloat64[T](re, im)
	}

	return twiddle
}

// ComputeBitReversalIndices returns the bit-reversal permutation indices
// for a size-n radix-2 FFT.
func ComputeBitReversalIndices(n int) []int {
	if n <= 0 {
		return nil
	}

	bitrev := make([]int, n)
	bits := log2(n)

	for i := range n {
		bitrev[i] = reverseBits(i, bits)
	}

	return bitrev
}

// log2 returns the base-2 logarithm of n (assuming n is a power of 2).
func log2(n int) int {
	result := 0

	for n > 1 {
		n >>= 1
		result++
	}

	return result
}

// reverseBits reverses the lower 'bits' bits of x.
// Example: reverseBits(6, 3) = reverseBits(0b110, 3) = 0b011 = 3.
func reverseBits(x, bits int) int {
	result := 0
	for range bits {
		result = (result << 1) | (x & 1)
		x >>= 1
	}

	return result
}

// complexFromFloat64 creates a complex number of type T from float64 components.
func complexFromFloat64[T Complex](re, im float64) T {
	var zero T

	switch any(zero).(type) {
	case complex64:
		result, _ := any(complex(float32(re), float32(im))).(T)
		return result
	case complex128:
		result, _ := any(complex(re, im)).(T)
		return result
	default:
		panic("unsupported complex type")
	}
}
