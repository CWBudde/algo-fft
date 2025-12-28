package fft

import "testing"

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

// Benchmarks for twiddle factor generation (Phase 3.1)

func BenchmarkComputeTwiddleFactors64_16(b *testing.B) {
	for b.Loop() {
		_ = ComputeTwiddleFactors[complex64](16)
	}
}

func BenchmarkComputeTwiddleFactors64_256(b *testing.B) {
	for b.Loop() {
		_ = ComputeTwiddleFactors[complex64](256)
	}
}

func BenchmarkComputeTwiddleFactors64_1024(b *testing.B) {
	for b.Loop() {
		_ = ComputeTwiddleFactors[complex64](1024)
	}
}

func BenchmarkComputeTwiddleFactors64_4096(b *testing.B) {
	for b.Loop() {
		_ = ComputeTwiddleFactors[complex64](4096)
	}
}

func BenchmarkComputeTwiddleFactors64_65536(b *testing.B) {
	for b.Loop() {
		_ = ComputeTwiddleFactors[complex64](65536)
	}
}

func BenchmarkComputeTwiddleFactors128_1024(b *testing.B) {
	for b.Loop() {
		_ = ComputeTwiddleFactors[complex128](1024)
	}
}

// Benchmarks for bit-reversal index computation (Phase 3.2)

func BenchmarkComputeBitReversalIndices_16(b *testing.B) {
	for b.Loop() {
		_ = ComputeBitReversalIndices(16)
	}
}

func BenchmarkComputeBitReversalIndices_256(b *testing.B) {
	for b.Loop() {
		_ = ComputeBitReversalIndices(256)
	}
}

func BenchmarkComputeBitReversalIndices_1024(b *testing.B) {
	for b.Loop() {
		_ = ComputeBitReversalIndices(1024)
	}
}

func BenchmarkComputeBitReversalIndices_4096(b *testing.B) {
	for b.Loop() {
		_ = ComputeBitReversalIndices(4096)
	}
}

func BenchmarkComputeBitReversalIndices_65536(b *testing.B) {
	for b.Loop() {
		_ = ComputeBitReversalIndices(65536)
	}
}

func TestComputePackedTwiddles_Radix4(t *testing.T) {
	t.Parallel()

	n := 16
	twiddle := ComputeTwiddleFactors[complex64](n)
	packed := ComputePackedTwiddles[complex64](n, 4, twiddle)

	if packed == nil {
		t.Fatalf("ComputePackedTwiddles returned nil")
	}

	if packed.Radix != 4 {
		t.Fatalf("Radix = %d, want 4", packed.Radix)
	}

	if len(packed.StageOffsets) != 2 {
		t.Fatalf("StageOffsets length = %d, want 2", len(packed.StageOffsets))
	}

	if packed.StageOffsets[0] != 0 || packed.StageOffsets[1] != 3 {
		t.Fatalf("StageOffsets = %v, want [0 3]", packed.StageOffsets)
	}

	expect := make([]complex64, 0, 15)

	for _, size := range []int{4, 16} {
		step := n / size

		span := size / 4
		for j := range span {
			base := j * step
			for k := 1; k < 4; k++ {
				expect = append(expect, twiddle[(k*base)%n])
			}
		}
	}

	if len(packed.Values) != len(expect) {
		t.Fatalf("Values length = %d, want %d", len(packed.Values), len(expect))
	}

	for i := range expect {
		if packed.Values[i] != expect[i] {
			t.Fatalf("Values[%d] = %v, want %v", i, packed.Values[i], expect[i])
		}
	}
}

// Precision tests comparing complex64 to complex128 reference (Phase 3.3)

func TestTwiddleFactorPrecision(t *testing.T) {
	t.Parallel()

	sizes := []int{8, 16, 64, 256, 1024}
	for _, n := range sizes {
		twiddle64 := ComputeTwiddleFactors[complex64](n)
		twiddle128 := ComputeTwiddleFactors[complex128](n)

		for k := range n {
			// Convert complex64 to complex128 for comparison
			got := complex128(twiddle64[k])
			want := twiddle128[k]

			// Calculate relative error
			diff := got - want
			magnitude := abs128(want)

			var relError float64
			if magnitude > 1e-15 {
				relError = abs128(diff) / magnitude
			} else {
				relError = abs128(diff)
			}

			// float32 has ~7 decimal digits of precision
			// Allow for ~1e-6 relative error due to float32 truncation
			if relError > 1e-6 {
				t.Errorf("n=%d, k=%d: twiddle64=%v, twiddle128=%v, relError=%e",
					n, k, got, want, relError)
			}
		}
	}
}

func TestComplexMultiplicationPrecision(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		a, b complex128
	}{
		{1 + 2i, 3 + 4i},
		{0.1 + 0.2i, 0.3 + 0.4i},
		{1e6 + 2e6i, 3e6 + 4e6i},
		{1e-6 + 2e-6i, 3e-6 + 4e-6i},
		{0.707106781186547 + 0.707106781186547i, 0.5 + 0.866025403784438i}, // unit circle values
	}

	for _, testCase := range testCases {
		// Reference calculation in complex128
		want := testCase.a * testCase.b

		// Calculate in complex64
		a64 := complex64(testCase.a)
		b64 := complex64(testCase.b)
		got64 := a64 * b64
		got := complex128(got64)

		// Calculate relative error
		diff := got - want
		magnitude := abs128(want)

		var relError float64
		if magnitude > 1e-15 {
			relError = abs128(diff) / magnitude
		} else {
			relError = abs128(diff)
		}

		// float32 multiplication should maintain ~1e-6 relative precision
		if relError > 1e-5 {
			t.Errorf("(%v)*(%v): got=%v, want=%v, relError=%e",
				testCase.a, testCase.b, got, want, relError)
		}
	}
}

func TestComplexAdditionPrecision(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		a, b complex128
	}{
		{1 + 2i, 3 + 4i},
		{1e10 + 2e10i, 1e-10 + 1e-10i}, // Large + small (potential precision loss)
		{-1e6 + 1e6i, 1e6 - 1e6i},      // Cancellation case
	}

	for _, testCase := range testCases {
		want := testCase.a + testCase.b

		a64 := complex64(testCase.a)
		b64 := complex64(testCase.b)
		got64 := a64 + b64
		got := complex128(got64)

		diff := got - want
		magnitude := abs128(want)

		var relError float64
		if magnitude > 1e-15 {
			relError = abs128(diff) / magnitude
		} else {
			relError = abs128(diff)
		}

		// Addition should maintain precision except in cancellation cases
		if relError > 1e-5 && magnitude > 1e-6 {
			t.Errorf("(%v)+(%v): got=%v, want=%v, relError=%e",
				testCase.a, testCase.b, got, want, relError)
		}
	}
}

func TestComplexSubtractionPrecision(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		a, b complex128
	}{
		{5 + 6i, 3 + 4i},
		{1.000001 + 0i, 1.0 + 0i}, // Subtraction of nearly equal numbers
	}

	for _, testCase := range testCases {
		want := testCase.a - testCase.b

		a64 := complex64(testCase.a)
		b64 := complex64(testCase.b)
		got64 := a64 - b64
		got := complex128(got64)

		diff := got - want
		magnitude := abs128(want)

		var relError float64
		if magnitude > 1e-15 {
			relError = abs128(diff) / magnitude
		} else {
			relError = abs128(diff)
		}

		// Subtraction of nearly equal numbers can have high relative error
		// but the absolute error should still be bounded
		absError := abs128(diff)
		if relError > 1e-4 && absError > 1e-6 {
			t.Errorf("(%v)-(%v): got=%v, want=%v, relError=%e, absError=%e",
				testCase.a, testCase.b, got, want, relError, absError)
		}
	}
}

// abs128 computes the absolute value of a complex128 number.
func abs128(c complex128) float64 {
	re := real(c)
	im := imag(c)

	return sqrt64(re*re + im*im)
}

// sqrt64 computes the square root using Newton-Raphson iteration.
func sqrt64(x float64) float64 {
	if x < 0 {
		return 0 // NaN case, but we only use positive values
	}

	if x == 0 {
		return 0
	}

	// Newton-Raphson: x_{n+1} = 0.5 * (x_n + x/x_n)
	guess := x
	for range 100 {
		next := 0.5 * (guess + x/guess)
		if next == guess {
			break
		}

		guess = next
	}

	return guess
}
