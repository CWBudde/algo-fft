package math

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestComputeTwiddleFactors(t *testing.T) {
	t.Parallel()
	t.Run("edge cases", func(t *testing.T) {
		t.Parallel()
		// Zero size
		result := ComputeTwiddleFactors[complex64](0)
		if result != nil {
			t.Errorf("ComputeTwiddleFactors(0) = %v, want nil", result)
		}

		// Negative size
		result = ComputeTwiddleFactors[complex64](-1)
		if result != nil {
			t.Errorf("ComputeTwiddleFactors(-1) = %v, want nil", result)
		}
	})

	t.Run("size 1", func(t *testing.T) {
		t.Parallel()
		// n=1: W_1^0 = exp(-2πi*0/1) = 1
		twiddle := ComputeTwiddleFactors[complex64](1)
		if len(twiddle) != 1 {
			t.Fatalf("len = %d, want 1", len(twiddle))
		}

		if twiddle[0] != 1 {
			t.Errorf("twiddle[0] = %v, want 1", twiddle[0])
		}
	})

	t.Run("size 2", func(t *testing.T) {
		t.Parallel()
		// n=2: W_2^k = exp(-2πik/2) for k=0,1
		// W_2^0 = 1, W_2^1 = -1
		twiddle := ComputeTwiddleFactors[complex64](2)
		if len(twiddle) != 2 {
			t.Fatalf("len = %d, want 2", len(twiddle))
		}

		const eps = 1e-6

		if !approxEqual64(twiddle[0], 1+0i) {
			t.Errorf("twiddle[0] = %v, want 1", twiddle[0])
		}

		if !approxEqual64(twiddle[1], -1+0i) {
			t.Errorf("twiddle[1] = %v, want -1", twiddle[1])
		}
	})

	t.Run("size 4", func(t *testing.T) {
		t.Parallel()
		// n=4: W_4^k = exp(-2πik/4) for k=0,1,2,3
		// W_4^0 = 1, W_4^1 = -i, W_4^2 = -1, W_4^3 = i
		twiddle := ComputeTwiddleFactors[complex64](4)
		if len(twiddle) != 4 {
			t.Fatalf("len = %d, want 4", len(twiddle))
		}

		const eps = 1e-6

		expected := []complex64{1 + 0i, 0 - 1i, -1 + 0i, 0 + 1i}
		for i, exp := range expected {
			if !approxEqual64(twiddle[i], exp) {
				t.Errorf("twiddle[%d] = %v, want %v", i, twiddle[i], exp)
			}
		}
	})

	t.Run("size 8", func(t *testing.T) {
		t.Parallel()

		twiddle := ComputeTwiddleFactors[complex64](8)
		if len(twiddle) != 8 {
			t.Fatalf("len = %d, want 8", len(twiddle))
		}

		const eps = 1e-6
		// W_8^k = exp(-2πik/8) for k=0..7
		sqrt2 := float32(math.Sqrt(2) / 2)
		expected := []complex64{
			complex(1, 0),           // k=0
			complex(sqrt2, -sqrt2),  // k=1
			complex(0, -1),          // k=2
			complex(-sqrt2, -sqrt2), // k=3
			complex(-1, 0),          // k=4
			complex(-sqrt2, sqrt2),  // k=5
			complex(0, 1),           // k=6
			complex(sqrt2, sqrt2),   // k=7
		}

		for i, exp := range expected {
			if !approxEqual64(twiddle[i], exp) {
				t.Errorf("twiddle[%d] = %v, want %v", i, twiddle[i], exp)
			}
		}
	})
}

func TestComputeTwiddleFactorsComplex128(t *testing.T) {
	t.Parallel()
	t.Run("size 4 complex128", func(t *testing.T) {
		t.Parallel()

		twiddle := ComputeTwiddleFactors[complex128](4)
		if len(twiddle) != 4 {
			t.Fatalf("len = %d, want 4", len(twiddle))
		}

		const eps = 1e-14

		expected := []complex128{1 + 0i, 0 - 1i, -1 + 0i, 0 + 1i}
		for i, exp := range expected {
			if !approxEqual128(twiddle[i], exp) {
				t.Errorf("twiddle[%d] = %v, want %v", i, twiddle[i], exp)
			}
		}
	})
}

func TestComputeTwiddleFactorsProperties(t *testing.T) {
	t.Parallel()

	sizes := []int{2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}

	for _, n := range sizes {
		t.Run(formatSizeTwiddle(n), func(t *testing.T) {
			t.Parallel()

			twiddle := ComputeTwiddleFactors[complex128](n)

			// Property 1: Length should equal n
			if len(twiddle) != n {
				t.Errorf("len = %d, want %d", len(twiddle), n)
			}

			// Property 2: First element should be 1
			const eps = 1e-14

			if !approxEqual128(twiddle[0], 1+0i) {
				t.Errorf("twiddle[0] = %v, want 1", twiddle[0])
			}

			// Property 3: All twiddle factors should have magnitude 1 (roots of unity)
			for k, w := range twiddle {
				mag := cmplx.Abs(complex128(w))
				if math.Abs(mag-1.0) > eps {
					t.Errorf("twiddle[%d] magnitude = %v, want 1.0", k, mag)
				}
			}

			// Property 4: W_n^(n/2) should be -1 (for even n)
			if n >= 2 {
				halfN := n / 2

				expected := complex128(-1 + 0i)
				if !approxEqual128(twiddle[halfN], expected) {
					t.Errorf("twiddle[%d] = %v, want -1", halfN, twiddle[halfN])
				}
			}

			// Property 5: W_n^n should equal W_n^0 = 1 (periodicity)
			// We can verify: angle(W_n^(n-1)) + angle(W_n^1) ≈ 2π (mod 2π)
			if n >= 2 {
				angle1 := math.Atan2(imag(complex128(twiddle[1])), real(complex128(twiddle[1])))
				angleNm1 := math.Atan2(imag(complex128(twiddle[n-1])), real(complex128(twiddle[n-1])))
				sumAngle := angle1 + angleNm1
				// Should be close to 0 (mod 2π), accounting for -2π ≡ 0 (mod 2π)
				if math.Abs(sumAngle) > eps && math.Abs(sumAngle-2*math.Pi) > eps && math.Abs(sumAngle+2*math.Pi) > eps {
					t.Errorf("angle sum = %v, expected 0, 2π, or -2π", sumAngle)
				}
			}
		})
	}
}

func TestComplexFromFloat64(t *testing.T) {
	t.Parallel()
	t.Run("complex64", func(t *testing.T) {
		t.Parallel()

		re, im := 3.14, 2.71
		result := ComplexFromFloat64[complex64](re, im)

		const eps = 1e-6

		expected := complex64(complex(float32(re), float32(im)))
		if !approxEqual64(result, expected) {
			t.Errorf("ComplexFromFloat64[complex64](%v, %v) = %v, want %v",
				re, im, result, expected)
		}
	})

	t.Run("complex128", func(t *testing.T) {
		t.Parallel()

		re, im := 3.141592653589793, 2.718281828459045
		result := ComplexFromFloat64[complex128](re, im)

		const eps = 1e-14

		expected := complex(re, im)
		if !approxEqual128(result, expected) {
			t.Errorf("ComplexFromFloat64[complex128](%v, %v) = %v, want %v",
				re, im, result, expected)
		}
	})

	t.Run("zero values", func(t *testing.T) {
		t.Parallel()

		result64 := ComplexFromFloat64[complex64](0, 0)
		if result64 != 0 {
			t.Errorf("ComplexFromFloat64[complex64](0, 0) = %v, want 0", result64)
		}

		result128 := ComplexFromFloat64[complex128](0, 0)
		if result128 != 0 {
			t.Errorf("ComplexFromFloat64[complex128](0, 0) = %v, want 0", result128)
		}
	})

	t.Run("negative values", func(t *testing.T) {
		t.Parallel()

		result := ComplexFromFloat64[complex64](-1.5, -2.5)
		expected := complex64(complex(float32(-1.5), float32(-2.5)))

		const eps = 1e-6

		if !approxEqual64(result, expected) {
			t.Errorf("ComplexFromFloat64[complex64](-1.5, -2.5) = %v, want %v",
				result, expected)
		}
	})
}

func TestConj(t *testing.T) {
	t.Parallel()
	t.Run("complex64", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name     string
			input    complex64
			expected complex64
		}{
			{"zero", 0, 0},
			{"real only", 3 + 0i, 3 + 0i},
			{"imaginary only", 0 + 4i, 0 - 4i},
			{"positive both", 3 + 4i, 3 - 4i},
			{"negative real", -3 + 4i, -3 - 4i},
			{"negative imag", 3 - 4i, 3 + 4i},
			{"negative both", -3 - 4i, -3 + 4i},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				t.Parallel()

				result := Conj(tt.input)
				if result != tt.expected {
					t.Errorf("Conj(%v) = %v, want %v", tt.input, result, tt.expected)
				}
			})
		}
	})

	t.Run("complex128", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name     string
			input    complex128
			expected complex128
		}{
			{"zero", 0, 0},
			{"positive both", 3 + 4i, 3 - 4i},
			{"negative imag", 3 - 4i, 3 + 4i},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				t.Parallel()

				result := Conj(tt.input)
				if result != tt.expected {
					t.Errorf("Conj(%v) = %v, want %v", tt.input, result, tt.expected)
				}
			})
		}
	})

	t.Run("double conjugate", func(t *testing.T) {
		t.Parallel()
		// Property: Conj(Conj(z)) = z
		val := complex64(3 + 4i)

		result := Conj(Conj(val))
		if result != val {
			t.Errorf("Conj(Conj(%v)) = %v, want %v", val, result, val)
		}
	})
}

func TestConjugateOf(t *testing.T) {
	t.Parallel()
	// ConjugateOf is just a wrapper for Conj
	t.Run("basic", func(t *testing.T) {
		t.Parallel()

		input := complex64(3 + 4i)
		expected := complex64(3 - 4i)

		result := ConjugateOf(input)
		if result != expected {
			t.Errorf("ConjugateOf(%v) = %v, want %v", input, result, expected)
		}
	})

	t.Run("matches Conj", func(t *testing.T) {
		t.Parallel()

		input := complex128(1.5 + 2.5i)
		resultConj := Conj(input)

		resultConjugateOf := ConjugateOf(input)
		if resultConj != resultConjugateOf {
			t.Errorf("Conj(%v) = %v, ConjugateOf(%v) = %v, should be equal",
				input, resultConj, input, resultConjugateOf)
		}
	})
}

func TestTwiddleFactorSymmetry(t *testing.T) {
	t.Parallel()
	// Property: W_n^k and W_n^(n-k) are complex conjugates
	sizes := []int{4, 8, 16, 32}

	for _, n := range sizes {
		t.Run(formatSizeTwiddle(n), func(t *testing.T) {
			t.Parallel()

			twiddle := ComputeTwiddleFactors[complex128](n)

			const eps = 1e-14

			for k := 1; k < n/2; k++ {
				wk := twiddle[k]
				wnk := twiddle[n-k]
				expectedConj := Conj(wk)

				if !approxEqual128(wnk, expectedConj) {
					t.Errorf("W_%d^%d = %v, W_%d^%d = %v, expected conjugates",
						n, k, wk, n, n-k, wnk)
				}
			}
		})
	}
}

// Helper functions

func approxEqual64(a, b complex64) bool {
	const eps = 1e-6

	return math.Abs(float64(real(a)-real(b))) < float64(eps) &&
		math.Abs(float64(imag(a)-imag(b))) < float64(eps)
}

func approxEqual128(a, b complex128) bool {
	const eps = 1e-14

	return math.Abs(real(a)-real(b)) < eps &&
		math.Abs(imag(a)-imag(b)) < eps
}

// formatSizeTwiddle is a local copy to avoid redeclaration with bitrev_test.go.
func formatSizeTwiddle(n int) string {
	if n < 1000 {
		return formatIntTwiddle(n)
	}

	return formatIntTwiddle(n/1000) + "k"
}

func formatIntTwiddle(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}

	if n < 100 {
		return string(rune('0'+n/10)) + string(rune('0'+n%10))
	}
	// For n >= 100
	hundreds := n / 100
	tens := (n % 100) / 10
	ones := n % 10

	return string(rune('0'+hundreds)) + string(rune('0'+tens)) + string(rune('0'+ones))
}

// Benchmarks

func BenchmarkComputeTwiddleFactors(b *testing.B) {
	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}

	for _, size := range sizes {
		b.Run("complex64/"+formatSizeTwiddle(size), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_ = ComputeTwiddleFactors[complex64](size)
			}
		})

		b.Run("complex128/"+formatSizeTwiddle(size), func(b *testing.B) {
			b.ReportAllocs()

			for range b.N {
				_ = ComputeTwiddleFactors[complex128](size)
			}
		})
	}
}

func BenchmarkComplexFromFloat64(b *testing.B) {
	b.Run("complex64", func(b *testing.B) {
		b.ReportAllocs()

		for range b.N {
			_ = ComplexFromFloat64[complex64](3.14, 2.71)
		}
	})

	b.Run("complex128", func(b *testing.B) {
		b.ReportAllocs()

		for range b.N {
			_ = ComplexFromFloat64[complex128](3.14, 2.71)
		}
	})
}

func BenchmarkConj(b *testing.B) {
	b.Run("complex64", func(b *testing.B) {
		val := complex64(3 + 4i)

		b.ReportAllocs()

		for range b.N {
			_ = Conj(val)
		}
	})

	b.Run("complex128", func(b *testing.B) {
		val := complex128(3 + 4i)

		b.ReportAllocs()

		for range b.N {
			_ = Conj(val)
		}
	})
}
