package math

import (
	"testing"
)

func TestReverseBits(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		x      int
		nbits  int
		expect int
	}{
		// Edge cases
		{"zero value", 0, 3, 0},
		{"zero nbits", 6, 0, 0},
		{"negative nbits", 6, -1, 0},

		// Small bit counts
		{"1 bit: 0", 0, 1, 0},
		{"1 bit: 1", 1, 1, 1},

		{"2 bits: 0b00", 0b00, 2, 0b00},
		{"2 bits: 0b01", 0b01, 2, 0b10},
		{"2 bits: 0b10", 0b10, 2, 0b01},
		{"2 bits: 0b11", 0b11, 2, 0b11},

		// 3 bits (example from docstring)
		{"3 bits: 0b000", 0b000, 3, 0b000},
		{"3 bits: 0b001", 0b001, 3, 0b100},
		{"3 bits: 0b010", 0b010, 3, 0b010},
		{"3 bits: 0b011", 0b011, 3, 0b110},
		{"3 bits: 0b100", 0b100, 3, 0b001},
		{"3 bits: 0b101", 0b101, 3, 0b101},
		{"3 bits: 0b110 (docstring example)", 0b110, 3, 0b011},
		{"3 bits: 0b111", 0b111, 3, 0b111},

		// 4 bits
		{"4 bits: 0b0001", 0b0001, 4, 0b1000},
		{"4 bits: 0b0010", 0b0010, 4, 0b0100},
		{"4 bits: 0b0011", 0b0011, 4, 0b1100},
		{"4 bits: 0b0101", 0b0101, 4, 0b1010},
		{"4 bits: 0b1111", 0b1111, 4, 0b1111},

		// Larger bit counts
		{"8 bits: 0x12", 0x12, 8, 0x48},
		{"8 bits: 0xFF", 0xFF, 8, 0xFF},
		{"10 bits: 0x123", 0x123, 10, 0x312},
		{"16 bits: 0x1234", 0x1234, 16, 0x2C48},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := ReverseBits(tt.x, tt.nbits)
			if got != tt.expect {
				t.Errorf("ReverseBits(%#b, %d) = %#b, want %#b (decimal: got %d, want %d)",
					tt.x, tt.nbits, got, tt.expect, got, tt.expect)
			}
		})
	}
}

func TestReverseBitsSymmetry(t *testing.T) {
	t.Parallel()
	// Property: reversing twice should return the original value
	for nbits := 1; nbits <= 16; nbits++ {
		maxVal := 1 << uint(nbits)
		for x := range maxVal {
			reversed := ReverseBits(x, nbits)

			doubleReversed := ReverseBits(reversed, nbits)
			if doubleReversed != x {
				t.Errorf("ReverseBits(ReverseBits(%d, %d), %d) = %d, want %d",
					x, nbits, nbits, doubleReversed, x)
			}
		}
	}
}

func TestComputeBitReversalIndices(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		n      int
		expect []int
	}{
		// Edge cases
		{"zero", 0, nil},
		{"negative", -1, nil},

		// Powers of 2
		{"n=1", 1, []int{0}},
		{"n=2", 2, []int{0, 1}},
		{"n=4", 4, []int{0, 2, 1, 3}},
		{"n=8", 8, []int{0, 4, 2, 6, 1, 5, 3, 7}},
		{"n=16", 16, []int{0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := ComputeBitReversalIndices(tt.n)
			if len(got) != len(tt.expect) {
				t.Fatalf("ComputeBitReversalIndices(%d) returned length %d, want %d",
					tt.n, len(got), len(tt.expect))
			}

			for i := range got {
				if got[i] != tt.expect[i] {
					t.Errorf("ComputeBitReversalIndices(%d)[%d] = %d, want %d",
						tt.n, i, got[i], tt.expect[i])
				}
			}
		})
	}
}

func TestComputeBitReversalIndicesProperties(t *testing.T) {
	t.Parallel()

	sizes := []int{1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}

	for _, n := range sizes {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			indices := ComputeBitReversalIndices(n)

			// Property 1: Length should equal n
			if len(indices) != n {
				t.Errorf("length = %d, want %d", len(indices), n)
			}

			// Property 2: All indices should be in range [0, n)
			for i, idx := range indices {
				if idx < 0 || idx >= n {
					t.Errorf("indices[%d] = %d, out of range [0, %d)", i, idx, n)
				}
			}

			// Property 3: Should be a permutation (all indices unique)
			seen := make(map[int]bool)
			for i, idx := range indices {
				if seen[idx] {
					t.Errorf("duplicate index %d at position %d", idx, i)
				}

				seen[idx] = true
			}

			// Property 4: First element should always be 0
			if indices[0] != 0 {
				t.Errorf("indices[0] = %d, want 0", indices[0])
			}

			// Property 5: If n is power of 2, last element should be n-1
			if isPowerOfTwo(n) && indices[n-1] != n-1 {
				t.Errorf("indices[%d] = %d, want %d", n-1, indices[n-1], n-1)
			}

			// Property 6: Applying permutation twice should give identity
			// i.e., indices[indices[i]] == i for all i
			for i := range n {
				if indices[indices[i]] != i {
					t.Errorf("indices[indices[%d]] = %d, want %d (not a symmetric permutation)",
						i, indices[indices[i]], i)
				}
			}
		})
	}
}

func TestComputeBitReversalIndicesNonPowerOfTwo(t *testing.T) {
	t.Parallel()
	// Test that the function still works for non-power-of-2 sizes
	// even though FFT typically uses power-of-2 sizes
	sizes := []int{3, 5, 6, 7, 9, 10, 12, 15}

	for _, n := range sizes {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			indices := ComputeBitReversalIndices(n)

			// Should still return valid indices
			if len(indices) != n {
				t.Errorf("length = %d, want %d", len(indices), n)
			}

			// All indices should be in valid range
			for i, idx := range indices {
				if idx < 0 || idx >= n {
					t.Errorf("indices[%d] = %d, out of range [0, %d)", i, idx, n)
				}
			}
		})
	}
}

// Helper functions

func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

func formatSize(n int) string {
	if n < 1000 {
		return formatInt(n)
	}

	return formatInt(n/1000) + "k"
}

func formatInt(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}

	return string(rune('0'+n/10)) + string(rune('0'+n%10))
}

// Benchmarks

func BenchmarkReverseBits(b *testing.B) {
	nbits := 10

	b.ResetTimer()

	for i := range b.N {
		ReverseBits(i&1023, nbits)
	}
}

func BenchmarkComputeBitReversalIndices(b *testing.B) {
	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}

	for _, size := range sizes {
		b.Run(formatSize(size), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_ = ComputeBitReversalIndices(size)
			}
		})
	}
}
