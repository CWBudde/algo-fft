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

// NextHighlyComposite returns the smallest 5-smooth number (of the form
// 2^a·3^b·5^c) greater than or equal to n. For n <= 1, returns 1. For n so
// large that no power-of-two search bound is representable in int
// (n > 2^(bits.UintSize-2)), it returns 0.
//
// A non-zero result is never larger than NextPowerOfTwo(n), since powers of
// two are themselves 5-smooth. Callers use this to pick padded FFT lengths
// that the mixed-radix engine can execute exactly (e.g. Bluestein sub-FFT
// sizes), which are frequently much smaller than the next power of two.
func NextHighlyComposite(n int) int {
	if n <= 1 {
		return 1
	}

	// The next power of two is always a valid 5-smooth candidate and bounds
	// the search: only products of powers of 3 and 5 below it can improve on
	// it. If that bound is not representable, NextPowerOfTwo wraps; report
	// "no result" rather than searching against a corrupted bound.
	best := NextPowerOfTwo(n)
	if best < n {
		return 0
	}

	// Enumerate the 3^b·5^c products below best. All loop steps and the
	// candidate comparison are guarded by division, so no intermediate
	// product can overflow.
	for p5 := 1; p5 < best; {
		for p35 := p5; p35 < best; {
			// Smallest power of two lifting p35 to >= n, computed from
			// ceil(n/p35) so the intermediate never exceeds n.
			needed := n / p35
			if n%p35 != 0 {
				needed++
			}

			p2 := NextPowerOfTwo(needed)

			// p35*p2 < best, checked without multiplying.
			if p2 <= (best-1)/p35 {
				best = p35 * p2
			}

			if p35 > (best-1)/3 {
				break
			}

			p35 *= 3
		}

		if p5 > (best-1)/5 {
			break
		}

		p5 *= 5
	}

	return best
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
