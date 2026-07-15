package math

import (
	"math/bits"
	"reflect"
	"testing"
)

func TestFactorize(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want []int
	}{
		{n: -5, want: nil},
		{n: 0, want: nil},
		{n: 1, want: nil},
		{n: 2, want: []int{2}},
		{n: 3, want: []int{3}},
		{n: 4, want: []int{2, 2}},
		{n: 6, want: []int{2, 3}},
		{n: 12, want: []int{2, 2, 3}},
		{n: 45, want: []int{3, 3, 5}},
		{n: 97, want: []int{97}},
		{n: 100, want: []int{2, 2, 5, 5}},
		{n: 1024, want: []int{2, 2, 2, 2, 2, 2, 2, 2, 2, 2}},
	}

	for _, tt := range tests {
		got := Factorize(tt.n)
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("Factorize(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}

func TestNextHighlyComposite(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want int
	}{
		{n: -5, want: 1},
		{n: 0, want: 1},
		{n: 1, want: 1},
		{n: 2, want: 2},
		{n: 5, want: 5},
		{n: 7, want: 8},
		{n: 11, want: 12},
		{n: 13, want: 15},
		{n: 25, want: 25},
		{n: 97, want: 100},
		{n: 121, want: 125},
		{n: 513, want: 540},
		{n: 997, want: 1000},
		{n: 1001, want: 1024},
		{n: 1993, want: 2000},
		{n: 2017, want: 2025},
		{n: 6001, want: 6075},
		{n: 46656, want: 46656}, // 2^6·3^6
	}

	for _, tt := range tests {
		got := NextHighlyComposite(tt.n)
		if got != tt.want {
			t.Errorf("NextHighlyComposite(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

// TestNextHighlyComposite_HugeInputs pins the overflow behavior: the largest
// n with a representable power-of-two search bound still resolves, and inputs
// beyond it report 0 instead of searching against wrapped arithmetic.
func TestNextHighlyComposite_HugeInputs(t *testing.T) {
	t.Parallel()

	// 2^(bits.UintSize-2) is itself a power of two, hence 5-smooth.
	huge := 1 << (bits.UintSize - 2)
	if got := NextHighlyComposite(huge); got != huge {
		t.Errorf("NextHighlyComposite(2^%d) = %d, want %d", bits.UintSize-2, got, huge)
	}

	// One past it, NextPowerOfTwo(n) is no longer representable.
	if got := NextHighlyComposite(huge + 1); got != 0 {
		t.Errorf("NextHighlyComposite(2^%d+1) = %d, want 0", bits.UintSize-2, got)
	}

	maxInt := 1<<(bits.UintSize-1) - 1
	if got := NextHighlyComposite(maxInt); got != 0 {
		t.Errorf("NextHighlyComposite(MaxInt) = %d, want 0", got)
	}

	// Near-bound composite input: the search must stay overflow-free and
	// return a representable 5-smooth result >= n.
	n := huge - 12345
	got := NextHighlyComposite(n)

	if got < n || got > huge || !IsHighlyComposite(got) {
		t.Errorf("NextHighlyComposite(%d) = %d, want 5-smooth in [n, 2^%d]", n, got, bits.UintSize-2)
	}
}

// TestNextHighlyComposite_Minimality brute-forces the smallest 5-smooth number
// >= n for every n up to the sweep bound and checks the closed-form search
// agrees, along with the structural invariants callers rely on.
func TestNextHighlyComposite_Minimality(t *testing.T) {
	t.Parallel()

	next := 1

	for n := 1; n <= 20000; n++ {
		for next < n || !IsHighlyComposite(next) {
			next++
		}

		got := NextHighlyComposite(n)
		if got != next {
			t.Fatalf("NextHighlyComposite(%d) = %d, want %d", n, got, next)
		}

		if got > NextPowerOfTwo(n) {
			t.Fatalf("NextHighlyComposite(%d) = %d exceeds NextPowerOfTwo = %d", n, got, NextPowerOfTwo(n))
		}
	}
}

func TestIsHighlyComposite(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want bool
	}{
		{n: -5, want: false},
		{n: 0, want: false},
		{n: 1, want: true},
		{n: 2, want: true},
		{n: 3, want: true},
		{n: 4, want: true},
		{n: 5, want: true},
		{n: 6, want: true},
		{n: 8, want: true},
		{n: 9, want: true},
		{n: 10, want: true},
		{n: 12, want: true},
		{n: 15, want: true},
		{n: 18, want: true},
		{n: 25, want: true},
		{n: 30, want: true},
		{n: 16, want: true},
		{n: 14, want: false},
		{n: 49, want: false},
		{n: 11, want: false},
	}

	for _, tt := range tests {
		got := IsHighlyComposite(tt.n)
		if got != tt.want {
			t.Errorf("IsHighlyComposite(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}
