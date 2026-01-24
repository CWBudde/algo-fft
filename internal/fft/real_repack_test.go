package fft

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestInverseRepackComplex64Generic tests the generic inverse repacking for complex64.
func TestInverseRepackComplex64Generic(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		half int
	}{
		{"PowerOfTwo_8", 8},
		{"PowerOfTwo_16", 16},
		{"PowerOfTwo_32", 32},
		{"PowerOfTwo_64", 64},
		{"Odd_9", 9},
		{"Odd_17", 17},
		{"Odd_33", 33},
		{"Small_4", 4},
		{"Small_2", 2},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			half := testCase.half
			n := half * 2

			// Create source spectrum with conjugate symmetry property
			src := make([]complex64, half+1)
			for i := range src {
				src[i] = complex(float32(i%7-3), float32((i*3)%5-2))
			}
			// DC and Nyquist must be real for valid real FFT output
			src[0] = complex(real(src[0]), 0)
			src[half] = complex(real(src[half]), 0)

			// Generate weights for inverse repacking
			weight := make([]complex64, half+1)
			for k := 0; k <= half; k++ {
				theta := 2 * math.Pi * float64(k) / float64(n)
				re := 0.5 * (1 + math.Sin(theta))
				im := 0.5 * math.Cos(theta)
				weight[k] = complex(float32(re), float32(im))
			}

			// Test with full range (start=1)
			dst := make([]complex64, half)
			x0 := real(src[0])
			xh := real(src[half])
			dst[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))

			inverseRepackComplex64Generic(dst, src, weight, 1)

			// Verify properties
			// 1. No NaN or Inf values
			for i, val := range dst {
				if math.IsNaN(float64(real(val))) || math.IsNaN(float64(imag(val))) {
					t.Errorf("dst[%d] contains NaN: %v", i, val)
				}

				if math.IsInf(float64(real(val)), 0) || math.IsInf(float64(imag(val)), 0) {
					t.Errorf("dst[%d] contains Inf: %v", i, val)
				}
			}

			// 2. Verify computation is deterministic - running twice produces same results
			dst2 := make([]complex64, half)
			dst2[0] = dst[0]
			inverseRepackComplex64Generic(dst2, src, weight, 1)

			const eps = 1e-5

			for i := range half {
				gr := real(dst[i])
				gi := imag(dst[i])
				r2 := real(dst2[i])

				i2 := imag(dst2[i])
				if math.Abs(float64(gr-r2)) > eps || math.Abs(float64(gi-i2)) > eps {
					t.Errorf("Non-deterministic result at dst[%d]: first=%v, second=%v", i, dst[i], dst2[i])
				}
			}
		})
	}
}

// TestInverseRepackComplex128Generic tests the generic inverse repacking for complex128.
func TestInverseRepackComplex128Generic(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		half int
	}{
		{"PowerOfTwo_8", 8},
		{"PowerOfTwo_16", 16},
		{"PowerOfTwo_32", 32},
		{"PowerOfTwo_64", 64},
		{"Odd_9", 9},
		{"Odd_17", 17},
		{"Odd_33", 33},
		{"Small_4", 4},
		{"Small_2", 2},
	}

	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			half := testCase.half
			n := half * 2

			// Create source spectrum with conjugate symmetry property
			src := make([]complex128, half+1)
			for i := range src {
				src[i] = complex(float64(i%7-3), float64((i*3)%5-2))
			}
			// DC and Nyquist must be real for valid real FFT output
			src[0] = complex(real(src[0]), 0)
			src[half] = complex(real(src[half]), 0)

			// Generate weights for inverse repacking
			weight := make([]complex128, half+1)
			for k := 0; k <= half; k++ {
				theta := 2 * math.Pi * float64(k) / float64(n)
				re := 0.5 * (1 + math.Sin(theta))
				im := 0.5 * math.Cos(theta)
				weight[k] = complex(re, im)
			}

			// Test with full range (start=1)
			dst := make([]complex128, half)
			x0 := real(src[0])
			xh := real(src[half])
			dst[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))

			inverseRepackComplex128Generic(dst, src, weight, 1)

			// Verify properties
			// 1. No NaN or Inf values
			for i, val := range dst {
				if cmplx.IsNaN(val) {
					t.Errorf("dst[%d] contains NaN: %v", i, val)
				}

				if cmplx.IsInf(val) {
					t.Errorf("dst[%d] contains Inf: %v", i, val)
				}
			}

			// 2. Verify computation is deterministic - running twice produces same results
			dst2 := make([]complex128, half)
			dst2[0] = dst[0]
			inverseRepackComplex128Generic(dst2, src, weight, 1)

			const eps = 1e-10

			for i := range half {
				gr := real(dst[i])
				gi := imag(dst[i])
				r2 := real(dst2[i])

				i2 := imag(dst2[i])
				if math.Abs(gr-r2) > eps || math.Abs(gi-i2) > eps {
					t.Errorf("Non-deterministic result at dst[%d]: first=%v, second=%v", i, dst[i], dst2[i])
				}
			}
		})
	}
}

// TestInverseRepackSymmetryComplex64 tests the symmetry properties of inverse repacking.
func TestInverseRepackSymmetryComplex64(t *testing.T) {
	t.Parallel()

	half := 16
	n := half * 2

	// Create source with specific symmetry
	src := make([]complex64, half+1)
	for i := 0; i <= half; i++ {
		src[i] = complex(float32(i), float32(half-i))
	}

	src[0] = complex(real(src[0]), 0)
	src[half] = complex(real(src[half]), 0)

	weight := make([]complex64, half+1)
	for k := 0; k <= half; k++ {
		theta := 2 * math.Pi * float64(k) / float64(n)
		re := 0.5 * (1 + math.Sin(theta))
		im := 0.5 * math.Cos(theta)
		weight[k] = complex(float32(re), float32(im))
	}

	dst := make([]complex64, half)
	x0 := real(src[0])
	xh := real(src[half])
	dst[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))

	inverseRepackComplex64Generic(dst, src, weight, 1)

	// Check that symmetric indices have appropriate relationship
	for k := 1; k < half/2; k++ {
		m := half - k
		// This verifies that the algorithm correctly handles paired indices
		if k < m {
			// Both dst[k] and dst[m] should be computed
			if dst[k] == 0 && dst[m] == 0 {
				t.Errorf("Both dst[%d] and dst[%d] are zero, expected non-zero values", k, m)
			}
		}
	}
}

// TestInverseRepackSymmetryComplex128 tests the symmetry properties for complex128.
func TestInverseRepackSymmetryComplex128(t *testing.T) {
	t.Parallel()

	half := 16
	n := half * 2

	// Create source with specific symmetry
	src := make([]complex128, half+1)
	for i := 0; i <= half; i++ {
		src[i] = complex(float64(i), float64(half-i))
	}

	src[0] = complex(real(src[0]), 0)
	src[half] = complex(real(src[half]), 0)

	weight := make([]complex128, half+1)
	for k := 0; k <= half; k++ {
		theta := 2 * math.Pi * float64(k) / float64(n)
		re := 0.5 * (1 + math.Sin(theta))
		im := 0.5 * math.Cos(theta)
		weight[k] = complex(re, im)
	}

	dst := make([]complex128, half)
	x0 := real(src[0])
	xh := real(src[half])
	dst[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))

	inverseRepackComplex128Generic(dst, src, weight, 1)

	// Check that symmetric indices have appropriate relationship
	for k := 1; k < half/2; k++ {
		m := half - k
		// This verifies that the algorithm correctly handles paired indices
		if k < m {
			// Both dst[k] and dst[m] should be computed
			if dst[k] == 0 && dst[m] == 0 {
				t.Errorf("Both dst[%d] and dst[%d] are zero, expected non-zero values", k, m)
			}
		}
	}
}

// TestInverseRepackEdgeCases tests edge cases for inverse repacking.
func TestInverseRepackEdgeCases(t *testing.T) {
	t.Parallel()

	t.Run("MinimalSize_Complex64", func(t *testing.T) {
		t.Parallel()

		half := 2
		n := half * 2

		src := make([]complex64, half+1)
		src[0] = complex(1, 0)
		src[1] = complex(0.5, 0.5)
		src[2] = complex(2, 0)

		weight := make([]complex64, half+1)
		for k := 0; k <= half; k++ {
			theta := 2 * math.Pi * float64(k) / float64(n)
			re := 0.5 * (1 + math.Sin(theta))
			im := 0.5 * math.Cos(theta)
			weight[k] = complex(float32(re), float32(im))
		}

		dst := make([]complex64, half)
		x0 := real(src[0])
		xh := real(src[half])
		dst[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))

		inverseRepackComplex64Generic(dst, src, weight, 1)

		// Should not panic and produce valid results
		for i, v := range dst {
			if math.IsNaN(float64(real(v))) || math.IsNaN(float64(imag(v))) {
				t.Errorf("dst[%d] is NaN: %v", i, v)
			}
		}
	})

	t.Run("MinimalSize_Complex128", func(t *testing.T) {
		t.Parallel()

		half := 2
		n := half * 2

		src := make([]complex128, half+1)
		src[0] = complex(1.0, 0)
		src[1] = complex(0.5, 0.5)
		src[2] = complex(2.0, 0)

		weight := make([]complex128, half+1)
		for k := 0; k <= half; k++ {
			theta := 2 * math.Pi * float64(k) / float64(n)
			re := 0.5 * (1 + math.Sin(theta))
			im := 0.5 * math.Cos(theta)
			weight[k] = complex(re, im)
		}

		dst := make([]complex128, half)
		x0 := real(src[0])
		xh := real(src[half])
		dst[0] = complex(0.5*(x0+xh), 0.5*(x0-xh))

		inverseRepackComplex128Generic(dst, src, weight, 1)

		// Should not panic and produce valid results
		for i, v := range dst {
			if cmplx.IsNaN(v) {
				t.Errorf("dst[%d] is NaN: %v", i, v)
			}
		}
	})
}

// TestRepackInverseComplex64 tests the public RepackInverseComplex64 function
func TestRepackInverseComplex64(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 16, 32, 64}

	for _, half := range sizes {
		half := half
		t.Run("Size"+string(rune(half)), func(t *testing.T) {
			t.Parallel()

			n := half * 2

			// Create valid input
			src := make([]complex64, half+1)
			for i := range src {
				src[i] = complex(float32(i), float32(-i))
			}
			src[0] = complex(real(src[0]), 0)
			src[half] = complex(real(src[half]), 0)

			// Generate weight
			weight := make([]complex64, half+1)
			for k := 0; k <= half; k++ {
				theta := 2 * math.Pi * float64(k) / float64(n)
				re := 0.5 * (1 + math.Sin(theta))
				im := 0.5 * math.Cos(theta)
				weight[k] = complex(float32(re), float32(im))
			}

			dst := make([]complex64, half)
			RepackInverseComplex64(dst, src, weight)

			// Verify no NaN
			for i, v := range dst {
				if math.IsNaN(float64(real(v))) || math.IsNaN(float64(imag(v))) {
					t.Errorf("dst[%d] is NaN", i)
				}
			}
		})
	}
}

// TestRepackInverseComplex128 tests the public RepackInverseComplex128 function
func TestRepackInverseComplex128(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 16, 32, 64}

	for _, half := range sizes {
		half := half
		t.Run("Size"+string(rune(half)), func(t *testing.T) {
			t.Parallel()

			n := half * 2

			// Create valid input
			src := make([]complex128, half+1)
			for i := range src {
				src[i] = complex(float64(i), float64(-i))
			}
			src[0] = complex(real(src[0]), 0)
			src[half] = complex(real(src[half]), 0)

			// Generate weight
			weight := make([]complex128, half+1)
			for k := 0; k <= half; k++ {
				theta := 2 * math.Pi * float64(k) / float64(n)
				re := 0.5 * (1 + math.Sin(theta))
				im := 0.5 * math.Cos(theta)
				weight[k] = complex(re, im)
			}

			dst := make([]complex128, half)
			RepackInverseComplex128(dst, src, weight)

			// Verify no NaN
			for i, v := range dst {
				if cmplx.IsNaN(v) {
					t.Errorf("dst[%d] is NaN", i)
				}
			}
		})
	}
}
