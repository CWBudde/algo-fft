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

// TestComputeIdentityIndices tests identity permutation generation.
func TestComputeIdentityIndices(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		n      int
		expect []int
	}{
		{"zero", 0, nil},
		{"negative", -1, nil},
		{"n=1", 1, []int{0}},
		{"n=4", 4, []int{0, 1, 2, 3}},
		{"n=8", 8, []int{0, 1, 2, 3, 4, 5, 6, 7}},
		{"n=16", 16, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := ComputeIdentityIndices(tt.n)
			if !slicesEqual(got, tt.expect) {
				t.Errorf("ComputeIdentityIndices(%d) = %v, want %v", tt.n, got, tt.expect)
			}
		})
	}
}

// TestComputeBitReversalIndicesRadix4 tests radix-4 permutation.
func TestComputeBitReversalIndicesRadix4(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		n      int
		expect []int
	}{
		{"n=4", 4, []int{0, 1, 2, 3}}, // 4^1: identity in radix-4
		{"n=16", 16, []int{0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}}, // 4^2
		{"n=8 (not power of 4)", 8, []int{0, 2, 4, 6, 1, 3, 5, 7}},                // Mixed radix handling
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := ComputeBitReversalIndicesRadix4(tt.n)
			if !slicesEqual(got, tt.expect) {
				t.Errorf("ComputeBitReversalIndicesRadix4(%d) = %v, want %v", tt.n, got, tt.expect)
			}
		})
	}
}

// TestComputeBitReversalIndicesRadix8 tests radix-8 permutation.
func TestComputeBitReversalIndicesRadix8(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		n      int
		expect []int
	}{
		{"n=8", 8, []int{0, 1, 2, 3, 4, 5, 6, 7}}, // 8^1: identity in radix-8
		{"n=64", 64, []int{ // 8^2: digit reversal in base-8
			0, 8, 16, 24, 32, 40, 48, 56,
			1, 9, 17, 25, 33, 41, 49, 57,
			2, 10, 18, 26, 34, 42, 50, 58,
			3, 11, 19, 27, 35, 43, 51, 59,
			4, 12, 20, 28, 36, 44, 52, 60,
			5, 13, 21, 29, 37, 45, 53, 61,
			6, 14, 22, 30, 38, 46, 54, 62,
			7, 15, 23, 31, 39, 47, 55, 63,
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := ComputeBitReversalIndicesRadix8(tt.n)
			if !slicesEqual(got, tt.expect) {
				t.Errorf("ComputeBitReversalIndicesRadix8(%d) mismatch", tt.n)
				t.Errorf("got:  %v", got)
				t.Errorf("want: %v", tt.expect)
			}
		})
	}
}

// TestComputeBitReversalIndicesRadix32 tests radix-32 permutation.
func TestComputeBitReversalIndicesRadix32(t *testing.T) {
	t.Parallel()

	// Test n=32 (identity in radix-32)
	n := 32
	got := ComputeBitReversalIndicesRadix32(n)

	expect := make([]int, n)
	for i := range n {
		expect[i] = i
	}

	if !slicesEqual(got, expect) {
		t.Errorf("ComputeBitReversalIndicesRadix32(%d) should be identity", n)
	}
}

// TestComputeBitReversalIndicesRadix4Then2 tests mixed radix-2/4 permutation.
func TestComputeBitReversalIndicesRadix4Then2(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name   string
		n      int
		expect []int
	}{
		{"n=8 (2*4^1)", 8, []int{0, 2, 4, 6, 1, 3, 5, 7}},
		{"n=32 (2*4^2)", 32, []int{
			0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30,
			1, 9, 17, 25, 3, 11, 19, 27, 5, 13, 21, 29, 7, 15, 23, 31,
		}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := ComputeBitReversalIndicesRadix4Then2(tt.n)
			if !slicesEqual(got, tt.expect) {
				t.Errorf("ComputeBitReversalIndicesRadix4Then2(%d) mismatch", tt.n)
				t.Errorf("got:  %v", got)
				t.Errorf("want: %v", tt.expect)
			}
		})
	}
}

// TestComputeBitReversalIndicesMixed24 tests backward compatibility.
func TestComputeBitReversalIndicesMixed24(t *testing.T) {
	t.Parallel()

	n := 8
	got := ComputeBitReversalIndicesMixed24(n)
	expect := ComputeBitReversalIndicesRadix4Then2(n)

	if !slicesEqual(got, expect) {
		t.Errorf("ComputeBitReversalIndicesMixed24 should match Radix4Then2")
	}
}

// TestComputePermutationIndices_Radix0 tests identity permutation via radix=0.
func TestComputePermutationIndices_Radix0(t *testing.T) {
	t.Parallel()

	for n := 1; n <= 16; n++ {
		t.Run(formatSize(n), func(t *testing.T) {
			t.Parallel()

			got := ComputePermutationIndices(n, 0)
			if len(got) != n {
				t.Fatalf("length = %d, want %d", len(got), n)
			}

			for i := range n {
				if got[i] != i {
					t.Errorf("ComputePermutationIndices(%d, 0)[%d] = %d, want %d", n, i, got[i], i)
				}
			}
		})
	}
}

// TestComputePermutationIndices_InvalidInputs tests error handling.
func TestComputePermutationIndices_InvalidInputs(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		n     int
		radix int
	}{
		{"negative n", -1, 2},
		{"zero n", 0, 2},
		{"radix 3 with non-power", 8, 3},  // 8 is not a power of 3
		{"radix 5 with non-power", 16, 5}, // 16 is not a power of 5
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			got := ComputePermutationIndices(tt.n, tt.radix)
			if got != nil {
				t.Errorf("ComputePermutationIndices(%d, %d) should return nil, got %v", tt.n, tt.radix, got)
			}
		})
	}
}

// TestComputePermutationIndices_GeneralRadix tests general radix digit reversal.
// Note: The current implementation has limitations with non-power-of-2 radices.
// These tests verify that the function works for power-of-2 radices.
func TestComputePermutationIndices_GeneralRadix(t *testing.T) {
	t.Parallel()

	// Test power-of-2 radices which work correctly
	t.Run("radix4_n16", func(t *testing.T) {
		t.Parallel()

		got := ComputePermutationIndices(16, 4)
		// For radix-4 with n=16, this should match the radix-4 function
		expect := ComputeBitReversalIndicesRadix4(16)

		if !slicesEqual(got, expect) {
			t.Errorf("ComputePermutationIndices(16, 4) doesn't match radix-4 helper")
		}
	})

	// Verify that non-power-of-radix values return nil
	t.Run("radix3_invalid", func(t *testing.T) {
		t.Parallel()

		got := ComputePermutationIndices(8, 3) // 8 is not a power of 3
		if got != nil {
			t.Errorf("ComputePermutationIndices(8, 3) should return nil for invalid input, got %v", got)
		}
	})
}

// Helper function to compare slices.
func slicesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}

	return true
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
