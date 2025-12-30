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

// factorize is a private alias for Factorize.
func factorize(n int) []int {
	return Factorize(n)
}

// IsHighlyComposite reports whether n only contains 2, 3, or 5 factors.
func IsHighlyComposite(n int) bool {
	if n <= 0 {
		return false
	}

	for _, factor := range Factorize(n) {
		if factor != 2 && factor != 3 && factor != 5 {
			return false
		}
	}

	return true
}

// isHighlyComposite is a private alias for IsHighlyComposite.
func isHighlyComposite(n int) bool {
	return IsHighlyComposite(n)
}
