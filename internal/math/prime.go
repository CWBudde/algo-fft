package math

import "math/bits"

// MulMod returns (a * b) mod m without overflow using a 128-bit intermediate
// product. m must be non-zero.
func MulMod(a, b, m uint64) uint64 {
	a %= m
	b %= m

	// The quotient of the 128-bit product by m fits in 64 bits because
	// a, b < m implies hi < m, which is the precondition of bits.Div64.
	hi, lo := bits.Mul64(a, b)
	_, rem := bits.Div64(hi, lo, m)

	return rem
}

// PowMod returns base^exp mod m by square-and-multiply. m must be non-zero.
func PowMod(base, exp, m uint64) uint64 {
	if m == 1 {
		return 0
	}

	result := uint64(1)
	base %= m

	for exp > 0 {
		if exp&1 == 1 {
			result = MulMod(result, base, m)
		}

		base = MulMod(base, base, m)
		exp >>= 1
	}

	return result
}

// millerRabinBases is a base set that makes the Miller-Rabin test
// deterministic for every value below ~3.3e24 — far beyond the int range
// IsPrime accepts on any platform, so the answer is always exact.
//
//nolint:gochecknoglobals // immutable lookup table
var millerRabinBases = [...]uint64{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

// IsPrime reports whether n is prime. It runs a Miller-Rabin test with a
// fixed base set that is deterministic for the full int range, so the answer
// is exact (no probabilistic error).
func IsPrime(n int) bool {
	if n < 2 {
		return false
	}

	u := uint64(n)
	if u%2 == 0 {
		return u == 2
	}

	// Write u-1 = d * 2^s with d odd.
	d := u - 1
	s := bits.TrailingZeros64(d)
	d >>= s

	for _, a := range millerRabinBases {
		if a%u == 0 {
			continue
		}

		x := PowMod(a, d, u)
		if x == 1 || x == u-1 {
			continue
		}

		witness := true

		for range s - 1 {
			x = MulMod(x, x, u)
			if x == u-1 {
				witness = false
				break
			}
		}

		if witness {
			return false
		}
	}

	return true
}

// PrimitiveRoot returns the smallest primitive root modulo the prime p >= 3,
// i.e. a generator of the multiplicative group (Z/pZ)*. The cost is dominated
// by factoring p-1 with trial division, so callers on hot paths should
// restrict themselves to p whose p-1 is smooth (as Rader plan construction
// does). Calling it with a non-prime p does not terminate reliably; callers
// must check IsPrime first.
func PrimitiveRoot(p int) int {
	order := uint64(p) - 1

	// Distinct prime factors of p-1.
	factors := Factorize(p - 1)
	distinct := factors[:0]

	last := 0
	for _, f := range factors {
		if f != last {
			distinct = append(distinct, f)
			last = f
		}
	}

	// g generates the group iff g^((p-1)/q) != 1 for every prime factor q of
	// p-1. A generator always exists for prime p, so the loop terminates.
	for g := 2; ; g++ {
		isRoot := true

		for _, q := range distinct {
			if PowMod(uint64(g), order/uint64(q), uint64(p)) == 1 {
				isRoot = false
				break
			}
		}

		if isRoot {
			return g
		}
	}
}
