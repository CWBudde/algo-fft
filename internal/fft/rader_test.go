package fft

import "testing"

func TestRaderEligible(t *testing.T) {
	t.Parallel()

	eligible := []int{
		17, 257, 65537, // n-1 a power of two
		97, 769, 1153, 3001, // n-1 5-smooth, power-of-two part >= 8
		401, 641, 1601, 4001, 12289, 18433, 40961,
		// n-1 needs a radix-7/11 stage, n-1 < 2048 with power-of-two part
		// >= 16 and a single odd stage (see rader7Or11Wins).
		113, 353, 449, 673, 1409,
		// n-1 needs a radix-7/11 stage, n-1 >= 2048 with power-of-two
		// part >= 4.
		2113, 2269, 2689, 2801, 4201, 4481, 9901, 14081, 30241,
	}
	for _, n := range eligible {
		if !RaderEligible(n) {
			t.Errorf("RaderEligible(%d) = false, want true", n)
		}
	}

	ineligible := []int{
		1, 2, 3, 4, 5, 6, 8, 16, // too small / not on the Bluestein path
		47, 59, 83, // prime but n-1 has a factor > 11
		25, 121, 256, 1000, // not prime
		7, 11, 13, 31, 41, 61, // n-1 too small; measured slower than Bluestein
		101, 151, 251, // n-1 power-of-two part <= 4: measured slower
		// n-1 needs a radix-7/11 stage but the shape measured slower:
		// power-of-two part <= 2 (23, 127, 463, 2311, 22051),
		// <= 4 below 2048 (29, 197, 701), 8 below 2048 (89, 281, 1321),
		// or an odd part above 33 below 2048 (881, 1009, 2017).
		23, 29, 89, 127, 197, 281, 463, 701, 881, 1009, 1321, 2017, 2311, 22051,
	}
	for _, n := range ineligible {
		if RaderEligible(n) {
			t.Errorf("RaderEligible(%d) = true, want false", n)
		}
	}
}

func TestComputeRaderTables_Permutations(t *testing.T) {
	t.Parallel()

	for _, p := range []int{17, 257, 401, 641} {
		l := p - 1
		scratch := make([]complex128, l)
		permIn, permOut, filter, filterInv, twiddle, _ := ComputeRaderTables[complex128](p, scratch)

		if len(permIn) != l || len(permOut) != l || len(filter) != l ||
			len(filterInv) != l || len(twiddle) != l {
			t.Fatalf("p=%d: table lengths %d/%d/%d/%d/%d, want all %d",
				p, len(permIn), len(permOut), len(filter), len(filterInv), len(twiddle), l)
		}

		// Both permutations must be bijections over the nonzero residues 1..p-1.
		for name, perm := range map[string][]int{"permIn": permIn, "permOut": permOut} {
			seen := make([]bool, p)
			for _, j := range perm {
				if j < 1 || j >= p || seen[j] {
					t.Fatalf("p=%d: %s is not a bijection over 1..p-1 (index %d)", p, name, j)
				}

				seen[j] = true
			}
		}

		// permIn[q] must be the group inverse walk of permOut: permIn[q] = permOut[(l-q) mod l].
		for q := range l {
			if permIn[q] != permOut[(l-q)%l] {
				t.Fatalf("p=%d: permIn[%d] = %d, want permOut[%d] = %d",
					p, q, permIn[q], (l-q)%l, permOut[(l-q)%l])
			}
		}
	}
}
