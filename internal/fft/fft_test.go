package fft

import "testing"

func TestIsPowerOfTwo(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want bool
	}{
		{0, false},
		{1, true},
		{2, true},
		{3, false},
		{4, true},
		{5, false},
		{6, false},
		{7, false},
		{8, true},
		{15, false},
		{16, true},
		{17, false},
		{256, true},
		{1024, true},
		{1000, false},
		{-1, false},
		{-2, false},
	}

	for _, tt := range tests {
		got := IsPowerOfTwo(tt.n)
		if got != tt.want {
			t.Errorf("IsPowerOfTwo(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}

func TestLog2(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n    int
		want int
	}{
		{1, 0},
		{2, 1},
		{4, 2},
		{8, 3},
		{16, 4},
		{32, 5},
		{64, 6},
		{128, 7},
		{256, 8},
		{512, 9},
		{1024, 10},
	}

	for _, tt := range tests {
		got := log2(tt.n)
		if got != tt.want {
			t.Errorf("log2(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

func TestReverseBits(t *testing.T) {
	t.Parallel()

	tests := []struct {
		x    int
		bits int
		want int
	}{
		{0, 3, 0},      // 000 -> 000
		{1, 3, 4},      // 001 -> 100
		{2, 3, 2},      // 010 -> 010
		{3, 3, 6},      // 011 -> 110
		{4, 3, 1},      // 100 -> 001
		{5, 3, 5},      // 101 -> 101
		{6, 3, 3},      // 110 -> 011
		{7, 3, 7},      // 111 -> 111
		{0, 4, 0},      // 0000 -> 0000
		{1, 4, 8},      // 0001 -> 1000
		{15, 4, 15},    // 1111 -> 1111
		{0b1010, 4, 5}, // 1010 -> 0101
		{0b1100, 4, 3}, // 1100 -> 0011
	}

	for _, tt := range tests {
		got := reverseBits(tt.x, tt.bits)
		if got != tt.want {
			t.Errorf("reverseBits(%d, %d) = %d, want %d", tt.x, tt.bits, got, tt.want)
		}
	}
}

func TestComplexFromFloat64_Complex64(t *testing.T) {
	t.Parallel()

	c := complexFromFloat64[complex64](3.0, 4.0)
	expected := complex(float32(3.0), float32(4.0))

	if c != expected {
		t.Errorf("complexFromFloat64[complex64](3.0, 4.0) = %v, want %v", c, expected)
	}
}

func TestComplexFromFloat64_Complex128(t *testing.T) {
	t.Parallel()

	c := complexFromFloat64[complex128](3.0, 4.0)
	expected := complex(3.0, 4.0)

	if c != expected {
		t.Errorf("complexFromFloat64[complex128](3.0, 4.0) = %v, want %v", c, expected)
	}
}
