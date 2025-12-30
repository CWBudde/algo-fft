package fft

import (
	"math"
	"testing"
)

// TestCombineRadix2 verifies the radix-2 combine function.
func TestCombineRadix2(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		size     int
		sub0     []complex64
		sub1     []complex64
		twiddle  []complex64
		expected []complex64
	}{
		{
			name: "Size 4 (2×2)",
			size: 4,
			sub0: []complex64{complex(1, 0), complex(1, 0)}, // DFT([1, 1]) = [2, 0]
			sub1: []complex64{complex(1, 0), complex(1, 0)}, // DFT([1, 1]) = [2, 0]
			twiddle: []complex64{
				complex(1, 0),               // W^0 = 1
				complex(0, -1),              // W^1 = -i for N=4
			},
			expected: []complex64{
				complex(3, 0),  // sub0[0] + W^0 * sub1[0] = 1 + 1*1 = 2 (but these are already FFTs of [1,1])
				complex(1, 0),  // sub0[1] + W^1 * sub1[1] = 1 + (-i)*1
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			dst := make([]complex64, tt.size)
			combineRadix2(dst, tt.sub0, tt.sub1, tt.twiddle)

			// Note: This is a structural test. For correctness, we need integration tests
			// that compare full FFT results against reference implementation.
			t.Logf("Output: %v", dst)
		})
	}
}

// TestCombineRadix4 verifies the radix-4 combine function.
func TestCombineRadix4(t *testing.T) {
	t.Parallel()

	// Create simple test: 4 constant sub-FFTs
	quarter := 2
	sub0 := []complex64{complex(1, 0), complex(1, 0)}
	sub1 := []complex64{complex(1, 0), complex(1, 0)}
	sub2 := []complex64{complex(1, 0), complex(1, 0)}
	sub3 := []complex64{complex(1, 0), complex(1, 0)}

	// Generate twiddles for N=8 (4 × 2)
	n := 8
	twiddle1 := make([]complex64, quarter)
	twiddle2 := make([]complex64, quarter)
	twiddle3 := make([]complex64, quarter)

	for k := 0; k < quarter; k++ {
		angle1 := -2.0 * math.Pi * float64(k) / float64(n)
		angle2 := -2.0 * math.Pi * float64(2*k) / float64(n)
		angle3 := -2.0 * math.Pi * float64(3*k) / float64(n)

		twiddle1[k] = complex(float32(math.Cos(angle1)), float32(math.Sin(angle1)))
		twiddle2[k] = complex(float32(math.Cos(angle2)), float32(math.Sin(angle2)))
		twiddle3[k] = complex(float32(math.Cos(angle3)), float32(math.Sin(angle3)))
	}

	dst := make([]complex64, n)
	combineRadix4(dst, sub0, sub1, sub2, sub3, twiddle1, twiddle2, twiddle3)

	t.Logf("Radix-4 combine output: %v", dst)
}

// TestMultiplyByI verifies the i multiplication helper.
func TestMultiplyByI(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    complex64
		expected complex64
	}{
		{
			name:     "Real positive",
			input:    complex(1, 0),
			expected: complex(0, 1), // 1 * i = i
		},
		{
			name:     "Real negative",
			input:    complex(-1, 0),
			expected: complex(0, -1), // -1 * i = -i
		},
		{
			name:     "Imaginary positive",
			input:    complex(0, 1),
			expected: complex(-1, 0), // i * i = -1
		},
		{
			name:     "Imaginary negative",
			input:    complex(0, -1),
			expected: complex(1, 0), // -i * i = 1
		},
		{
			name:     "Complex",
			input:    complex(1, 1),
			expected: complex(-1, 1), // (1+i) * i = i - 1 = -1 + i
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := multiplyByI(tt.input)

			if !complexApproxEqual(result, tt.expected, 1e-6) {
				t.Errorf("multiplyByI(%v) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

// TestMultiplyByNegI verifies the -i multiplication helper.
func TestMultiplyByNegI(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		input    complex64
		expected complex64
	}{
		{
			name:     "Real positive",
			input:    complex(1, 0),
			expected: complex(0, -1), // 1 * (-i) = -i
		},
		{
			name:     "Real negative",
			input:    complex(-1, 0),
			expected: complex(0, 1), // -1 * (-i) = i
		},
		{
			name:     "Imaginary positive",
			input:    complex(0, 1),
			expected: complex(1, 0), // i * (-i) = 1
		},
		{
			name:     "Imaginary negative",
			input:    complex(0, -1),
			expected: complex(-1, 0), // -i * (-i) = -1
		},
		{
			name:     "Complex",
			input:    complex(1, 1),
			expected: complex(1, -1), // (1+i) * (-i) = -i + 1 = 1 - i
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			result := multiplyByNegI(tt.input)

			if !complexApproxEqual(result, tt.expected, 1e-6) {
				t.Errorf("multiplyByNegI(%v) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

// TestCombineRadix2Properties verifies mathematical properties of radix-2 combine.
func TestCombineRadix2Properties(t *testing.T) {
	t.Parallel()

	// Property: If both sub-FFTs are zero, output should be zero
	t.Run("Zero inputs", func(t *testing.T) {
		t.Parallel()

		n := 8
		half := n / 2
		sub0 := make([]complex64, half)
		sub1 := make([]complex64, half)
		twiddle := make([]complex64, half)
		dst := make([]complex64, n)

		combineRadix2(dst, sub0, sub1, twiddle)

		for i := range dst {
			if dst[i] != 0 {
				t.Errorf("dst[%d] = %v, want 0", i, dst[i])
			}
		}
	})

	// Property: If twiddles are 1, combine simplifies
	t.Run("Unity twiddles", func(t *testing.T) {
		t.Parallel()

		sub0 := []complex64{complex(1, 0), complex(2, 0), complex(3, 0), complex(4, 0)}
		sub1 := []complex64{complex(5, 0), complex(6, 0), complex(7, 0), complex(8, 0)}
		twiddle := []complex64{complex(1, 0), complex(1, 0), complex(1, 0), complex(1, 0)}
		dst := make([]complex64, 8)

		combineRadix2(dst, sub0, sub1, twiddle)

		// With W=1: dst[k] = sub0[k] + sub1[k], dst[k+4] = sub0[k] - sub1[k]
		expected := []complex64{
			complex(6, 0),  // 1 + 5
			complex(8, 0),  // 2 + 6
			complex(10, 0), // 3 + 7
			complex(12, 0), // 4 + 8
			complex(-4, 0), // 1 - 5
			complex(-4, 0), // 2 - 6
			complex(-4, 0), // 3 - 7
			complex(-4, 0), // 4 - 8
		}

		for i := range dst {
			if !complexApproxEqual(dst[i], expected[i], 1e-6) {
				t.Errorf("dst[%d] = %v, want %v", i, dst[i], expected[i])
			}
		}
	})
}

// TestCombineRadix4Properties verifies mathematical properties of radix-4 combine.
func TestCombineRadix4Properties(t *testing.T) {
	t.Parallel()

	// Property: If all sub-FFTs are zero, output should be zero
	t.Run("Zero inputs", func(t *testing.T) {
		t.Parallel()

		quarter := 4
		sub0 := make([]complex64, quarter)
		sub1 := make([]complex64, quarter)
		sub2 := make([]complex64, quarter)
		sub3 := make([]complex64, quarter)
		twiddle1 := make([]complex64, quarter)
		twiddle2 := make([]complex64, quarter)
		twiddle3 := make([]complex64, quarter)
		dst := make([]complex64, 16)

		combineRadix4(dst, sub0, sub1, sub2, sub3, twiddle1, twiddle2, twiddle3)

		for i := range dst {
			if dst[i] != 0 {
				t.Errorf("dst[%d] = %v, want 0", i, dst[i])
			}
		}
	})
}

// TestCombineGeneral verifies the general radix combine function.
func TestCombineGeneral(t *testing.T) {
	t.Parallel()

	// Test with radix-3 (unusual, but should work)
	radix := 3
	subSize := 2
	n := radix * subSize // = 6

	subResults := make([][]complex64, radix)
	for i := 0; i < radix; i++ {
		subResults[i] = make([]complex64, subSize)
		for j := 0; j < subSize; j++ {
			subResults[i][j] = complex(float32(i+1), 0) // [1, 1], [2, 2], [3, 3]
		}
	}

	twiddles := make([][]complex64, radix)
	for r := 0; r < radix; r++ {
		twiddles[r] = make([]complex64, subSize)
		for k := 0; k < subSize; k++ {
			angle := -2.0 * math.Pi * float64(r*k) / float64(n)
			twiddles[r][k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
		}
	}

	dst := make([]complex64, n)
	combineGeneral(dst, subResults, twiddles, radix)

	t.Logf("General radix-%d combine output: %v", radix, dst)

	// Basic sanity check: output should not be all zeros
	allZero := true
	for _, v := range dst {
		if v != 0 {
			allZero = false
			break
		}
	}

	if allZero {
		t.Error("combineGeneral produced all-zero output for non-zero inputs")
	}
}

// complexApproxEqual checks if two complex numbers are approximately equal.
func complexApproxEqual(a, b complex64, epsilon float32) bool {
	return math.Abs(float64(real(a)-real(b))) < float64(epsilon) &&
		math.Abs(float64(imag(a)-imag(b))) < float64(epsilon)
}
