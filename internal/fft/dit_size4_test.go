package fft

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestDIT4Radix4Complex64 tests the 4-point radix-4 FFT for complex64
func TestDIT4Radix4Complex64(t *testing.T) {
	const n = 4

	// Generate twiddle factors
	twiddle := make([]complex64, n)
	for k := 0; k < n; k++ {
		angle := -2 * math.Pi * float64(k) / float64(n)
		twiddle[k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
	}

	// Bit-reversal indices (not used for radix-4 size 4, but required by signature)
	bitrev := make([]int, n)
	for i := 0; i < n; i++ {
		bitrev[i] = i
	}

	scratch := make([]complex64, n)

	tests := []struct {
		name  string
		input []complex64
	}{
		{
			name:  "zeros",
			input: []complex64{0, 0, 0, 0},
		},
		{
			name:  "ones",
			input: []complex64{1, 1, 1, 1},
		},
		{
			name:  "impulse",
			input: []complex64{1, 0, 0, 0},
		},
		{
			name:  "alternating",
			input: []complex64{1, -1, 1, -1},
		},
		{
			name:  "complex",
			input: []complex64{1 + 2i, 3 - 4i, -5 + 6i, 7 - 8i},
		},
		{
			name:  "random",
			input: []complex64{0.5 + 0.3i, -0.2 + 0.8i, 1.1 - 0.6i, -0.7 - 0.4i},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test forward transform
			dst := make([]complex64, n)
			src := make([]complex64, n)
			copy(src, tt.input)

			ok := forwardDIT4Radix4Complex64(dst, src, twiddle, scratch, bitrev)
			if !ok {
				t.Fatal("forwardDIT4Radix4Complex64 returned false")
			}

			// Compare with naive DFT
			expected64 := reference.NaiveDFT(src)
			expected := make([]complex128, n)
			for i := range expected64 {
				expected[i] = complex128(expected64[i])
			}

			for i := 0; i < n; i++ {
				got := complex128(dst[i])
				want := expected[i]
				diff := cmplx.Abs(got - want)
				if diff > 1e-5 {
					t.Errorf("dst[%d] = %v, want %v (diff = %v)", i, got, want, diff)
				}
			}

			// Test inverse transform (round-trip)
			invDst := make([]complex64, n)
			ok = inverseDIT4Radix4Complex64(invDst, dst, twiddle, scratch, bitrev)
			if !ok {
				t.Fatal("inverseDIT4Radix4Complex64 returned false")
			}

			for i := 0; i < n; i++ {
				got := complex128(invDst[i])
				want := complex128(tt.input[i])
				diff := cmplx.Abs(got - want)
				if diff > 1e-5 {
					t.Errorf("round-trip: invDst[%d] = %v, want %v (diff = %v)", i, got, want, diff)
				}
			}
		})
	}
}

// TestDIT4Radix4Complex128 tests the 4-point radix-4 FFT for complex128
func TestDIT4Radix4Complex128(t *testing.T) {
	const n = 4

	twiddle := make([]complex128, n)
	for k := 0; k < n; k++ {
		angle := -2 * math.Pi * float64(k) / float64(n)
		twiddle[k] = complex(math.Cos(angle), math.Sin(angle))
	}

	bitrev := make([]int, n)
	for i := 0; i < n; i++ {
		bitrev[i] = i
	}

	scratch := make([]complex128, n)

	tests := []struct {
		name  string
		input []complex128
	}{
		{
			name:  "zeros",
			input: []complex128{0, 0, 0, 0},
		},
		{
			name:  "ones",
			input: []complex128{1, 1, 1, 1},
		},
		{
			name:  "impulse",
			input: []complex128{1, 0, 0, 0},
		},
		{
			name:  "alternating",
			input: []complex128{1, -1, 1, -1},
		},
		{
			name:  "complex",
			input: []complex128{1 + 2i, 3 - 4i, -5 + 6i, 7 - 8i},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := make([]complex128, n)
			src := make([]complex128, n)
			copy(src, tt.input)

			ok := forwardDIT4Radix4Complex128(dst, src, twiddle, scratch, bitrev)
			if !ok {
				t.Fatal("forwardDIT4Radix4Complex128 returned false")
			}

			// Compare with naive DFT
			expected := reference.NaiveDFT128(src)

			for i := 0; i < n; i++ {
				diff := cmplx.Abs(dst[i] - expected[i])
				if diff > 1e-12 {
					t.Errorf("dst[%d] = %v, want %v (diff = %v)", i, dst[i], expected[i], diff)
				}
			}

			// Test round-trip
			invDst := make([]complex128, n)
			ok = inverseDIT4Radix4Complex128(invDst, dst, twiddle, scratch, bitrev)
			if !ok {
				t.Fatal("inverseDIT4Radix4Complex128 returned false")
			}

			for i := 0; i < n; i++ {
				diff := cmplx.Abs(invDst[i] - tt.input[i])
				if diff > 1e-12 {
					t.Errorf("round-trip: invDst[%d] = %v, want %v (diff = %v)", i, invDst[i], tt.input[i], diff)
				}
			}
		})
	}
}

// TestDIT4Radix4InPlace tests in-place transforms
func TestDIT4Radix4InPlace(t *testing.T) {
	const n = 4

	twiddle := make([]complex64, n)
	for k := 0; k < n; k++ {
		angle := -2 * math.Pi * float64(k) / float64(n)
		twiddle[k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
	}

	bitrev := make([]int, n)
	for i := 0; i < n; i++ {
		bitrev[i] = i
	}
	scratch := make([]complex64, n)

	input := []complex64{1 + 2i, 3 - 4i, -5 + 6i, 7 - 8i}
	data := make([]complex64, n)
	copy(data, input)

	// Forward in-place
	ok := forwardDIT4Radix4Complex64(data, data, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("in-place forward failed")
	}

	// Verify against reference
	expected64 := reference.NaiveDFT(input)
	expected := make([]complex128, n)
	for i := range expected64 {
		expected[i] = complex128(expected64[i])
	}

	for i := 0; i < n; i++ {
		got := complex128(data[i])
		diff := cmplx.Abs(got - expected[i])
		if diff > 1e-5 {
			t.Errorf("in-place forward: data[%d] = %v, want %v (diff = %v)", i, got, expected[i], diff)
		}
	}

	// Inverse in-place (round-trip)
	ok = inverseDIT4Radix4Complex64(data, data, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("in-place inverse failed")
	}

	for i := 0; i < n; i++ {
		got := complex128(data[i])
		want := complex128(input[i])
		diff := cmplx.Abs(got - want)
		if diff > 1e-5 {
			t.Errorf("in-place round-trip: data[%d] = %v, want %v (diff = %v)", i, got, want, diff)
		}
	}
}

// BenchmarkDIT4Radix4Complex64 benchmarks the 4-point radix-4 FFT
func BenchmarkDIT4Radix4Complex64(b *testing.B) {
	const n = 4

	twiddle := make([]complex64, n)
	for k := 0; k < n; k++ {
		angle := -2 * math.Pi * float64(k) / float64(n)
		twiddle[k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
	}

	bitrev := make([]int, n)
	for i := 0; i < n; i++ {
		bitrev[i] = i
	}

	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)

	for i := 0; i < n; i++ {
		src[i] = complex(float32(i), float32(i)*0.5)
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(int64(n * 8)) // complex64 = 8 bytes

	for i := 0; i < b.N; i++ {
		forwardDIT4Radix4Complex64(dst, src, twiddle, scratch, bitrev)
	}
}
