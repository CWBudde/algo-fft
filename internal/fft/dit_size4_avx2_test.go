//go:build amd64 && fft_asm && !purego

package fft

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestAVX2Size4Radix4Complex64 tests the AVX2 assembly implementation for size 4
func TestAVX2Size4Radix4Complex64(t *testing.T) {
	const n = 4

	// Generate twiddle factors
	twiddle := make([]complex64, n)
	for k := 0; k < n; k++ {
		angle := -2 * math.Pi * float64(k) / float64(n)
		twiddle[k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
	}

	// Bit-reversal indices (not used for radix-4 size 4)
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
			// Test AVX2 forward transform
			dst := make([]complex64, n)
			src := make([]complex64, n)
			copy(src, tt.input)

			ok := forwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
			if !ok {
				t.Fatal("forwardAVX2Size4Radix4Complex64Asm returned false")
			}

			// Compare with naive DFT
			expected64 := reference.NaiveDFT(tt.input)
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

			// Test AVX2 inverse transform (round-trip)
			invDst := make([]complex64, n)
			ok = inverseAVX2Size4Radix4Complex64Asm(invDst, dst, twiddle, scratch, bitrev)
			if !ok {
				t.Fatal("inverseAVX2Size4Radix4Complex64Asm returned false")
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

// TestAVX2Size4Radix4InPlace tests in-place AVX2 transforms
func TestAVX2Size4Radix4InPlace(t *testing.T) {
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
	ok := forwardAVX2Size4Radix4Complex64Asm(data, data, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("AVX2 in-place forward failed")
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
			t.Errorf("AVX2 in-place forward: data[%d] = %v, want %v (diff = %v)", i, got, expected[i], diff)
		}
	}

	// Inverse in-place (round-trip)
	ok = inverseAVX2Size4Radix4Complex64Asm(data, data, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("AVX2 in-place inverse failed")
	}

	for i := 0; i < n; i++ {
		got := complex128(data[i])
		want := complex128(input[i])
		diff := cmplx.Abs(got - want)
		if diff > 1e-5 {
			t.Errorf("AVX2 in-place round-trip: data[%d] = %v, want %v (diff = %v)", i, got, want, diff)
		}
	}
}

// TestAVX2Size4MatchesGo tests that AVX2 and Go implementations produce identical results
func TestAVX2Size4MatchesGo(t *testing.T) {
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

	// Go implementation
	dstGo := make([]complex64, n)
	srcGo := make([]complex64, n)
	copy(srcGo, input)
	forwardDIT4Radix4Complex64(dstGo, srcGo, twiddle, scratch, bitrev)

	// AVX2 implementation
	dstAVX2 := make([]complex64, n)
	srcAVX2 := make([]complex64, n)
	copy(srcAVX2, input)
	forwardAVX2Size4Radix4Complex64Asm(dstAVX2, srcAVX2, twiddle, scratch, bitrev)

	// Compare
	for i := 0; i < n; i++ {
		diff := cmplx.Abs(complex128(dstGo[i]) - complex128(dstAVX2[i]))
		if diff > 1e-6 {
			t.Errorf("Forward: Go[%d] = %v, AVX2[%d] = %v (diff = %v)", i, dstGo[i], i, dstAVX2[i], diff)
		}
	}

	// Test inverse as well
	invGo := make([]complex64, n)
	inverseDIT4Radix4Complex64(invGo, dstGo, twiddle, scratch, bitrev)

	invAVX2 := make([]complex64, n)
	inverseAVX2Size4Radix4Complex64Asm(invAVX2, dstAVX2, twiddle, scratch, bitrev)

	for i := 0; i < n; i++ {
		diff := cmplx.Abs(complex128(invGo[i]) - complex128(invAVX2[i]))
		if diff > 1e-6 {
			t.Errorf("Inverse: Go[%d] = %v, AVX2[%d] = %v (diff = %v)", i, invGo[i], i, invAVX2[i], diff)
		}
	}
}

// BenchmarkAVX2Size4Radix4Complex64 benchmarks the AVX2 implementation
func BenchmarkAVX2Size4Radix4Complex64(b *testing.B) {
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
		forwardAVX2Size4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}
