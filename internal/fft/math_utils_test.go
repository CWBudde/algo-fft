package fft

import "testing"

func TestIsPowerOf2(t *testing.T) {
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
		{7, false},
		{8, true},
		{15, false},
		{16, true},
		{31, false},
		{32, true},
		{63, false},
		{64, true},
		{100, false},
		{128, true},
		{256, true},
		{1024, true},
		{1025, false},
		{-1, false},
		{-2, false},
	}

	for _, tt := range tests {
		got := IsPowerOf2(tt.n)
		if got != tt.want {
			t.Errorf("IsPowerOf2(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}

func TestNextPowerOfTwo(t *testing.T) {
	t.Parallel()
	tests := []struct {
		n    int
		want int
	}{
		{-1, 1},
		{0, 1},
		{1, 1},
		{2, 2},
		{3, 4},
		{4, 4},
		{5, 8},
		{7, 8},
		{8, 8},
		{9, 16},
		{15, 16},
		{16, 16},
		{17, 32},
		{31, 32},
		{32, 32},
		{33, 64},
		{63, 64},
		{64, 64},
		{100, 128},
		{1000, 1024},
		{1024, 1024},
		{1025, 2048},
	}

	for _, tt := range tests {
		got := NextPowerOfTwo(tt.n)
		if got != tt.want {
			t.Errorf("NextPowerOfTwo(%d) = %d, want %d", tt.n, got, tt.want)
		}
	}
}

func TestIsPowerOf(t *testing.T) {
	t.Parallel()
	tests := []struct {
		n    int
		base int
		want bool
	}{
		{1, 2, true},   // 2^0 = 1
		{1, 3, true},   // 3^0 = 1
		{1, 5, true},   // 5^0 = 1
		{2, 2, true},   // 2^1 = 2
		{3, 3, true},   // 3^1 = 3
		{4, 2, true},   // 2^2 = 4
		{8, 2, true},   // 2^3 = 8
		{9, 3, true},   // 3^2 = 9
		{27, 3, true},  // 3^3 = 27
		{125, 5, true}, // 5^3 = 125
		{6, 2, false},  // not a power of 2
		{10, 3, false}, // not a power of 3
		{24, 5, false}, // not a power of 5
		{0, 2, false},  // invalid n
		{-1, 2, false}, // invalid n
		{2, 1, false},  // invalid base
		{2, 0, false},  // invalid base
		{2, -1, false}, // invalid base
	}

	for _, tt := range tests {
		got := isPowerOf(tt.n, tt.base)
		if got != tt.want {
			t.Errorf("isPowerOf(%d, %d) = %v, want %v", tt.n, tt.base, got, tt.want)
		}
	}
}

func TestIsPowerOf3(t *testing.T) {
	t.Parallel()
	tests := []struct {
		n    int
		want bool
	}{
		{1, true},   // 3^0
		{3, true},   // 3^1
		{9, true},   // 3^2
		{27, true},  // 3^3
		{81, true},  // 3^4
		{243, true}, // 3^5
		{2, false},
		{4, false},
		{6, false},
		{10, false},
		{0, false},
		{-3, false},
	}

	for _, tt := range tests {
		got := isPowerOf3(tt.n)
		if got != tt.want {
			t.Errorf("isPowerOf3(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}

func TestIsPowerOf4(t *testing.T) {
	t.Parallel()
	tests := []struct {
		n    int
		want bool
	}{
		{1, true},    // 4^0 = 2^0
		{4, true},    // 4^1 = 2^2
		{16, true},   // 4^2 = 2^4
		{64, true},   // 4^3 = 2^6
		{256, true},  // 4^4 = 2^8
		{1024, true}, // 4^5 = 2^10
		{2, false},   // 2^1 (odd power of 2)
		{8, false},   // 2^3 (odd power of 2)
		{32, false},  // 2^5 (odd power of 2)
		{128, false}, // 2^7 (odd power of 2)
		{3, false},
		{5, false},
		{0, false},
		{-4, false},
	}

	for _, tt := range tests {
		got := isPowerOf4(tt.n)
		if got != tt.want {
			t.Errorf("isPowerOf4(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}

func TestIsPowerOf5(t *testing.T) {
	t.Parallel()
	tests := []struct {
		n    int
		want bool
	}{
		{1, true},    // 5^0
		{5, true},    // 5^1
		{25, true},   // 5^2
		{125, true},  // 5^3
		{625, true},  // 5^4
		{3125, true}, // 5^5
		{2, false},
		{4, false},
		{10, false},
		{20, false},
		{0, false},
		{-5, false},
	}

	for _, tt := range tests {
		got := isPowerOf5(tt.n)
		if got != tt.want {
			t.Errorf("isPowerOf5(%d) = %v, want %v", tt.n, got, tt.want)
		}
	}
}
