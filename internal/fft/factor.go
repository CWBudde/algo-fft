package fft

func factorize(n int) []int {
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

func isHighlyComposite(n int) bool {
	if n <= 0 {
		return false
	}

	for _, factor := range factorize(n) {
		if factor != 2 && factor != 3 && factor != 5 {
			return false
		}
	}

	return true
}

// IsHighlyComposite reports whether n only contains 2, 3, or 5 factors.
func IsHighlyComposite(n int) bool {
	return isHighlyComposite(n)
}
