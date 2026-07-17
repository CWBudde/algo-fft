package fft

import "testing"

func TestRaderEligible(t *testing.T) {
	t.Parallel()

	eligible := []int{
		17, 257, 65537, // n-1 a power of two
		401, 641, 1601, 4001, // n-1 = 2^a*5^b, a >= 4, >= 400
		12289, 18433, 40961, // n-1 5-smooth >= 4096
	}
	for _, n := range eligible {
		if !RaderEligible(n) {
			t.Errorf("RaderEligible(%d) = false, want true", n)
		}
	}

	ineligible := []int{
		1, 2, 3, 4, 5, 6, 8, 16, // too small / not on the Bluestein path
		23, 29, 47, 59, 1009, // prime but n-1 not 5-smooth
		25, 121, 256, 1000, // not prime
		7, 11, 13, 31, 61, 97, 101, 151, 251, // measured slower than Bluestein
		769, 1153, 3001, // n-1 has a factor of 3 below the 4096 cutoff
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
