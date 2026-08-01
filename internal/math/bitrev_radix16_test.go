package math_test

import (
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// The radix-16 ladder compresses its stage-1 permutation to one entry per group
// of sixteen, relying on p[16g+d] = p[16g] + d*(n/16). A table can be a perfectly
// valid bijection and still lack that stride property, in which case a full-table
// kernel round-trips and only the compressed one is wrong — so both halves are
// checked here rather than left to the kernel's own tests.
func checkRadix16Permutation(t *testing.T, name string, n int, p []int) {
	t.Helper()

	if len(p) != n {
		t.Fatalf("%s(%d): length %d, want %d", name, n, len(p), n)
	}

	seen := make([]bool, n)

	for i, v := range p {
		if v < 0 || v >= n {
			t.Fatalf("%s(%d): p[%d] = %d out of range", name, n, i, v)
		}

		if seen[v] {
			t.Fatalf("%s(%d): value %d appears twice, not a bijection", name, n, v)
		}

		seen[v] = true
	}

	stride := n / 16

	for g := range n / 16 {
		for d := 1; d < 16; d++ {
			if got, want := p[16*g+d], p[16*g]+d*stride; got != want {
				t.Fatalf("%s(%d): p[%d] = %d, want %d (stride property)", name, n, 16*g+d, got, want)
			}
		}
	}
}

func TestComputeBitReversalIndicesRadix16(t *testing.T) {
	t.Parallel()

	for _, n := range []int{16, 256, 4096, 65536} {
		checkRadix16Permutation(t, "Radix16", n, mathpkg.ComputeBitReversalIndicesRadix16(n))
	}
}

func TestComputeBitReversalIndicesRadix16Then2(t *testing.T) {
	t.Parallel()

	for _, n := range []int{32, 512, 8192} {
		checkRadix16Permutation(t, "Radix16Then2", n, mathpkg.ComputeBitReversalIndicesRadix16Then2(n))
	}
}

func TestComputeBitReversalIndicesRadix16Then4(t *testing.T) {
	t.Parallel()

	for _, n := range []int{64, 1024, 16384} {
		checkRadix16Permutation(t, "Radix16Then4", n, mathpkg.ComputeBitReversalIndicesRadix16Then4(n))
	}
}

func TestComputeBitReversalIndicesRadix16Then8(t *testing.T) {
	t.Parallel()

	for _, n := range []int{128, 2048, 32768} {
		checkRadix16Permutation(t, "Radix16Then8", n, mathpkg.ComputeBitReversalIndicesRadix16Then8(n))
	}
}

// The four constructors must disagree only about the split: a length that is
// both 16^k and (say) 4*16^j cannot exist, but a caller picking the wrong one
// for its length should get nil rather than a plausible-looking table.
func TestRadix16PermutationsRejectWrongShape(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		fn   func(int) []int
		bad  []int
	}{
		{"Radix16Then2", mathpkg.ComputeBitReversalIndicesRadix16Then2, []int{16, 64, 128}},
		{"Radix16Then4", mathpkg.ComputeBitReversalIndicesRadix16Then4, []int{16, 32, 128}},
		{"Radix16Then8", mathpkg.ComputeBitReversalIndicesRadix16Then8, []int{16, 32, 64}},
	}

	for _, tc := range cases {
		for _, n := range tc.bad {
			if got := tc.fn(n); got != nil {
				t.Errorf("%s(%d) = table of len %d, want nil", tc.name, n, len(got))
			}
		}
	}
}
