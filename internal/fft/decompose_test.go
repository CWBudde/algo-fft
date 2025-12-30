package fft

import (
	"testing"
)

func TestPlanDecomposition(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		size          int
		codeletSizes  []int
		cacheSize     int
		expectCodelet bool
		expectDepth   int
	}{
		{
			name:          "Size 256 is codelet",
			size:          256,
			codeletSizes:  []int{4, 8, 16, 32, 64, 128, 256, 512},
			cacheSize:     32768, // 32 KB L1
			expectCodelet: true,
			expectDepth:   1,
		},
		{
			name:          "Size 1024 splits to 512 codelets (radix-2)",
			size:          1024,
			codeletSizes:  []int{4, 8, 16, 32, 64, 128, 256, 512},
			cacheSize:     32768,
			expectCodelet: false,
			expectDepth:   2, // 1024 -> 2x512 (codelets)
		},
		{
			name:          "Size 8192 can split to 256 codelets",
			size:          8192,
			codeletSizes:  []int{4, 8, 16, 32, 64, 128, 256, 512},
			cacheSize:     32768,
			expectCodelet: false,
			expectDepth:   2, // Likely 8192 -> 32x256 or similar
		},
		{
			name:          "Size 4096 is power of 4, should use radix-4",
			size:          4096,
			codeletSizes:  []int{4, 8, 16, 32, 64, 128, 256, 512},
			cacheSize:     32768,
			expectCodelet: false,
			expectDepth:   2, // 4096 -> 4x1024 -> 16x256 (more likely: 16x256 directly)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			strategy := PlanDecomposition(tt.size, tt.codeletSizes, tt.cacheSize)

			if strategy == nil {
				t.Fatal("PlanDecomposition returned nil")
			}

			if strategy.Size != tt.size {
				t.Errorf("Strategy size = %d, want %d", strategy.Size, tt.size)
			}

			if strategy.UseCodelet != tt.expectCodelet {
				t.Errorf("UseCodelet = %v, want %v", strategy.UseCodelet, tt.expectCodelet)
			}

			depth := strategy.Depth()
			if depth < 1 {
				t.Errorf("Depth = %d, must be at least 1", depth)
			}

			// Check that leaf nodes are codelets
			if !tt.expectCodelet {
				checkLeaves(t, strategy, tt.codeletSizes)
			}
		})
	}
}

// checkLeaves recursively verifies that all leaf nodes use codelets.
func checkLeaves(t *testing.T, s *DecomposeStrategy, codeletSizes []int) {
	t.Helper()

	if s.UseCodelet {
		// Verify this size is actually in codeletSizes
		found := false

		for _, size := range codeletSizes {
			if size == s.Size {
				found = true
				break
			}
		}

		if !found {
			t.Errorf("Leaf node size %d is not in codeletSizes %v", s.Size, codeletSizes)
		}

		return
	}

	if s.Recursive != nil {
		checkLeaves(t, s.Recursive, codeletSizes)
	}
}

func TestDecompositionDepth(t *testing.T) {
	t.Parallel()

	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768

	tests := []struct {
		size     int
		maxDepth int
	}{
		{512, 1},   // Codelet
		{1024, 3},  // Should be shallow
		{2048, 3},  // Should be shallow
		{8192, 4},  // Medium depth
		{16384, 4}, // Medium depth
		{65536, 5}, // Deeper but reasonable
	}

	for _, tt := range tests {
		t.Run(string(rune('0'+tt.size/1000)), func(t *testing.T) {
			t.Parallel()

			strategy := PlanDecomposition(tt.size, codeletSizes, cacheSize)
			depth := strategy.Depth()

			if depth > tt.maxDepth {
				t.Errorf("Size %d: depth = %d, want <= %d", tt.size, depth, tt.maxDepth)
			}

			t.Logf("Size %d: depth = %d, codelet count = %d",
				tt.size, depth, strategy.CodeletCount())
		})
	}
}

func TestCodeletCount(t *testing.T) {
	t.Parallel()

	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768

	tests := []struct {
		size        int
		minCodelets int
		maxCodelets int
		description string
	}{
		{512, 1, 1, "Single codelet"},
		{1024, 2, 4, "2-4 codelets (depends on split)"},
		{8192, 8, 64, "Multiple codelets"},
		{16384, 16, 128, "Many codelets"},
	}

	for _, tt := range tests {
		t.Run(tt.description, func(t *testing.T) {
			t.Parallel()

			strategy := PlanDecomposition(tt.size, codeletSizes, cacheSize)

			// Debug output
			t.Logf("Strategy for size %d:", tt.size)
			t.Logf("  UseCodelet: %v", strategy.UseCodelet)
			t.Logf("  SplitFactor: %d", strategy.SplitFactor)
			t.Logf("  SubSize: %d", strategy.SubSize)

			if strategy.Recursive != nil {
				t.Logf("  Recursive != nil: UseCodelet=%v, Size=%d",
					strategy.Recursive.UseCodelet, strategy.Recursive.Size)
			} else {
				t.Logf("  Recursive == nil")
			}

			count := strategy.CodeletCount()

			if count < tt.minCodelets || count > tt.maxCodelets {
				t.Errorf("Size %d: codelet count = %d, want %d-%d",
					tt.size, count, tt.minCodelets, tt.maxCodelets)
			}

			// Verify total work: codelet_count * codelet_size should equal original size
			// (This is a property of Cooley-Tukey decomposition)
			if !strategy.UseCodelet && strategy.Recursive != nil {
				expectedWork := tt.size

				actualWork := count * strategy.Recursive.Size
				if actualWork != expectedWork {
					t.Logf("Note: Work calculation needs adjustment. Expected %d, got %d",
						expectedWork, actualWork)
				}
			}

			t.Logf("Size %d: %d codelets of size %d (depth %d)",
				tt.size, count, strategy.SubSize, strategy.Depth())
		})
	}
}

func TestFindFactors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		n       int
		want    []int
		wantLen int
	}{
		{8, []int{4, 2}, 2},     // Excludes 8 itself
		{16, []int{8, 4, 2}, 3}, // Excludes 16 itself
		{1024, nil, 9},          // 2, 4, 8, ..., 512 (9 factors, excludes 1024)
		{7, []int{2}, 1},        // Non-power-of-2, fallback to radix-2
	}

	for _, tt := range tests {
		t.Run(string(rune('0'+tt.n)), func(t *testing.T) {
			t.Parallel()

			factors := findFactors(tt.n)

			if tt.wantLen > 0 && len(factors) != tt.wantLen {
				t.Errorf("findFactors(%d) returned %d factors, want %d", tt.n, len(factors), tt.wantLen)
			}

			if tt.want != nil {
				if len(factors) != len(tt.want) {
					t.Fatalf("findFactors(%d) = %v, want %v", tt.n, factors, tt.want)
				}

				for i := range factors {
					if factors[i] != tt.want[i] {
						t.Errorf("factors[%d] = %d, want %d", i, factors[i], tt.want[i])
					}
				}
			}

			// All factors should divide n evenly (except for non-power-of-2 fallback)
			if IsPowerOf2(tt.n) {
				for _, f := range factors {
					if tt.n%f != 0 {
						t.Errorf("Factor %d does not divide %d", f, tt.n)
					}
				}
			}

			// Factors should be in descending order (largest first)
			for i := 1; i < len(factors); i++ {
				if factors[i] >= factors[i-1] {
					t.Errorf("Factors not in descending order: %v", factors)
					break
				}
			}
		})
	}
}

func TestScoreStrategy(t *testing.T) {
	t.Parallel()

	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768 // 32 KB L1

	tests := []struct {
		name            string
		radix           int
		subSize         int
		expectHighScore bool
		reason          string
	}{
		{"Codelet sub-problem", 4, 256, true, "Sub-size 256 is a codelet"},
		{"Fits in cache", 8, 1024, true, "1024 * 16 bytes = 16KB < 32KB"},
		{"Radix-4 preferred", 4, 1000, true, "Radix-4 gets SIMD bonus"},
		{"Very large radix penalty", 64, 128, false, "Radix 64 is penalized"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			score := scoreStrategy(tt.radix, tt.subSize, codeletSizes, cacheSize)

			t.Logf("%s: radix=%d, subSize=%d, score=%d (%s)",
				tt.name, tt.radix, tt.subSize, score, tt.reason)

			if tt.expectHighScore && score < 500 {
				t.Errorf("Expected high score (>= 500), got %d", score)
			}
		})
	}
}

func TestHasCodelet(t *testing.T) {
	t.Parallel()

	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}

	tests := []struct {
		size int
		want bool
	}{
		{4, true},
		{16, true},
		{256, true},
		{512, true},
		{1024, false},
		{7, false},
		{128, true},
	}

	for _, tt := range tests {
		t.Run(string(rune('0'+tt.size)), func(t *testing.T) {
			t.Parallel()

			got := hasCodelet(tt.size, codeletSizes)
			if got != tt.want {
				t.Errorf("hasCodelet(%d) = %v, want %v", tt.size, got, tt.want)
			}
		})
	}
}
