package math

// Factorize performs prime factorization of n, returning factors in ascending order.
func Factorize(n int) []int {
	if n <= 1 {
		return nil
	}

	factors := make([]int, 0, 8)

	for n%2 == 0 {
		factors = append(factors, 2)
		n /= 2
	}

	for p := 3; p*p <= n; p += 2 {
		for n%p == 0 {
			factors = append(factors, p)
			n /= p
		}
	}

	if n > 1 {
		factors = append(factors, n)
	}

	return factors
}

// IsHighlyComposite reports whether n only contains 2, 3, or 5 factors.
//
// It divides out all factors of 2, 3, and 5 in place rather than materializing
// the full factor list from Factorize, so it is allocation-free. This matters
// because it runs on the per-transform dispatch hot path (see the mixed-radix
// selection in internal/fft): a size like 768 = 2^8·3 would otherwise allocate
// a growing factor slice on every transform.
func IsHighlyComposite(n int) bool {
	if n <= 0 {
		return false
	}

	for n%2 == 0 {
		n /= 2
	}

	for n%3 == 0 {
		n /= 3
	}

	for n%5 == 0 {
		n /= 5
	}

	return n == 1
}
