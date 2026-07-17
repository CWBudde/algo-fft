package math

import (
	"math/big"
	"testing"
)

func TestIsPrime(t *testing.T) {
	t.Parallel()

	primes := []int{
		2, 3, 5, 7, 11, 13, 17, 19, 23, 97, 101, 257, 641, 769,
		65537, 2147483647, // 2^31 - 1 (Mersenne)
		1000000007, 67280421310721,
	}
	for _, n := range primes {
		if !IsPrime(n) {
			t.Errorf("IsPrime(%d) = false, want true", n)
		}
	}

	composites := []int{
		-7, 0, 1, 4, 6, 9, 15, 25, 121, 1024,
		561, 41041, 825265, // Carmichael numbers
		2047, 3277, 4033, // base-2 Fermat pseudoprimes
		2147483649, 1000000005,
	}
	for _, n := range composites {
		if IsPrime(n) {
			t.Errorf("IsPrime(%d) = true, want false", n)
		}
	}
}

func TestIsPrime_MatchesTrialDivision(t *testing.T) {
	t.Parallel()

	trialDivision := func(n int) bool {
		if n < 2 {
			return false
		}

		for d := 2; d*d <= n; d++ {
			if n%d == 0 {
				return false
			}
		}

		return true
	}

	for n := range 5000 {
		if IsPrime(n) != trialDivision(n) {
			t.Fatalf("IsPrime(%d) = %v, trial division says %v", n, IsPrime(n), trialDivision(n))
		}
	}
}

func TestMulMod_Overflow(t *testing.T) {
	t.Parallel()

	// Values large enough that a*b overflows uint64; validate against big.Int.
	cases := [][3]uint64{
		{1 << 63, 1 << 63, 2147483647},
		{18446744073709551610, 18446744073709551611, 18446744073709551557},
		{12345678901234567, 98765432109876543, 1000000007},
	}

	for _, c := range cases {
		a, b, m := c[0], c[1], c[2]

		var want big.Int

		want.Mul(new(big.Int).SetUint64(a), new(big.Int).SetUint64(b))
		want.Mod(&want, new(big.Int).SetUint64(m))

		if got := MulMod(a, b, m); got != want.Uint64() {
			t.Errorf("MulMod(%d, %d, %d) = %d, want %d", a, b, m, got, want.Uint64())
		}
	}
}

func TestPowMod(t *testing.T) {
	t.Parallel()

	cases := [][4]uint64{
		{2, 10, 1000000007, 1024},
		{3, 0, 7, 1},
		{5, 3, 13, 8},
		{2, 64, 18446744073709551557, 59}, // 2^64 mod p = 2^64 - p·1 = 59
		{7, 100, 1, 0},
	}

	for _, c := range cases {
		if got := PowMod(c[0], c[1], c[2]); got != c[3] {
			t.Errorf("PowMod(%d, %d, %d) = %d, want %d", c[0], c[1], c[2], got, c[3])
		}
	}
}

func TestPrimitiveRoot(t *testing.T) {
	t.Parallel()

	primes := []int{3, 5, 7, 11, 13, 17, 41, 97, 101, 257, 641, 65537}

	for _, p := range primes {
		g := PrimitiveRoot(p)

		if g < 2 || g >= p {
			t.Fatalf("PrimitiveRoot(%d) = %d out of range", p, g)
		}

		// g must have multiplicative order exactly p-1: walking its powers
		// visits every nonzero residue exactly once before returning to 1.
		seen := make([]bool, p)
		x := 1

		for range p - 1 {
			x = int(MulMod(uint64(x), uint64(g), uint64(p)))
			if seen[x] {
				t.Fatalf("PrimitiveRoot(%d) = %d revisits residue %d before covering the group", p, g, x)
			}

			seen[x] = true
		}

		if x != 1 {
			t.Fatalf("PrimitiveRoot(%d) = %d: g^(p-1) = %d, want 1", p, g, x)
		}
	}
}
