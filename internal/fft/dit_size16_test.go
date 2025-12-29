package fft

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	testSize16      = 16
	testTolerance64 = 1e-5
	testTolerance32 = 1e-4
)

// TestDIT16Radix2ForwardComplex64MatchesReference verifies the radix-2 forward transform
// against a naive DFT reference implementation for complex64.
func TestDIT16Radix2ForwardComplex64MatchesReference(t *testing.T) {
	data := make([]complex64, testSize16)
	for i := range data {
		data[i] = complex(float32(i), float32(i*2))
	}

	// Compute using radix-2 DIT
	dst := make([]complex64, testSize16)
	twiddle := ComputeTwiddleFactors[complex64](testSize16)
	scratch := make([]complex64, testSize16)
	bitrev := ComputeBitReversalIndices(testSize16)

	if !forwardDIT16Complex64(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Complex64 failed")
	}

	// Compute reference
	ref := reference.NaiveDFT(data)

	// Compare
	for i := range dst {
		if diff := cmplx.Abs(complex128(dst[i] - ref[i])); diff > testTolerance32 {
			t.Errorf("Mismatch at index %d: got %v, want %v (diff: %v)", i, dst[i], ref[i], diff)
		}
	}
}

// TestDIT16Radix2InverseComplex64MatchesReference verifies the radix-2 inverse transform
// against a naive IDFT reference implementation for complex64.
func TestDIT16Radix2InverseComplex64MatchesReference(t *testing.T) {
	data := make([]complex64, testSize16)
	for i := range data {
		data[i] = complex(float32(i+1), float32(i*3))
	}

	// Compute using radix-2 DIT
	dst := make([]complex64, testSize16)
	twiddle := ComputeTwiddleFactors[complex64](testSize16)
	scratch := make([]complex64, testSize16)
	bitrev := ComputeBitReversalIndices(testSize16)

	if !inverseDIT16Complex64(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Complex64 failed")
	}

	// Compute reference
	ref := reference.NaiveIDFT(data)

	// Compare
	for i := range dst {
		if diff := cmplx.Abs(complex128(dst[i] - ref[i])); diff > testTolerance32 {
			t.Errorf("Mismatch at index %d: got %v, want %v (diff: %v)", i, dst[i], ref[i], diff)
		}
	}
}

// TestDIT16Radix2RoundTripComplex64 verifies that Forward(Inverse(x)) ≈ x for radix-2.
func TestDIT16Radix2RoundTripComplex64(t *testing.T) {
	data := make([]complex64, testSize16)
	for i := range data {
		data[i] = complex(float32(i), float32(i*2))
	}
	original := make([]complex64, testSize16)
	copy(original, data)

	twiddle := ComputeTwiddleFactors[complex64](testSize16)
	scratch := make([]complex64, testSize16)
	bitrev := ComputeBitReversalIndices(testSize16)

	// Forward
	dst := make([]complex64, testSize16)
	if !forwardDIT16Complex64(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Complex64 failed")
	}

	// Inverse
	if !inverseDIT16Complex64(data, dst, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Complex64 failed")
	}

	// Compare
	for i := range data {
		if diff := cmplx.Abs(complex128(data[i] - original[i])); diff > testTolerance32 {
			t.Errorf("Round-trip mismatch at index %d: got %v, want %v (diff: %v)", i, data[i], original[i], diff)
		}
	}
}

// TestDIT16Radix4ForwardComplex64MatchesReference verifies the radix-4 forward transform
// against a naive DFT reference implementation for complex64.
func TestDIT16Radix4ForwardComplex64MatchesReference(t *testing.T) {
	data := make([]complex64, testSize16)
	for i := range data {
		data[i] = complex(float32(i), float32(i*2))
	}

	// Compute using radix-4 DIT
	dst := make([]complex64, testSize16)
	twiddle := ComputeTwiddleFactors[complex64](testSize16)
	scratch := make([]complex64, testSize16)
	bitrev := ComputeBitReversalIndicesRadix4(testSize16)

	if !forwardDIT16Radix4Complex64(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Radix4Complex64 failed")
	}

	// Compute reference
	ref := reference.NaiveDFT(data)

	// Compare
	for i := range dst {
		if diff := cmplx.Abs(complex128(dst[i] - ref[i])); diff > testTolerance32 {
			t.Errorf("Mismatch at index %d: got %v, want %v (diff: %v)", i, dst[i], ref[i], diff)
		}
	}
}

// TestDIT16Radix4InverseComplex64MatchesReference verifies the radix-4 inverse transform
// against a naive IDFT reference implementation for complex64.
func TestDIT16Radix4InverseComplex64MatchesReference(t *testing.T) {
	data := make([]complex64, testSize16)
	for i := range data {
		data[i] = complex(float32(i+1), float32(i*3))
	}

	// Compute using radix-4 DIT
	dst := make([]complex64, testSize16)
	twiddle := ComputeTwiddleFactors[complex64](testSize16)
	scratch := make([]complex64, testSize16)
	bitrev := ComputeBitReversalIndicesRadix4(testSize16)

	if !inverseDIT16Radix4Complex64(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Radix4Complex64 failed")
	}

	// Compute reference
	ref := reference.NaiveIDFT(data)

	// Compare
	for i := range dst {
		if diff := cmplx.Abs(complex128(dst[i] - ref[i])); diff > testTolerance32 {
			t.Errorf("Mismatch at index %d: got %v, want %v (diff: %v)", i, dst[i], ref[i], diff)
		}
	}
}

// TestDIT16Radix4RoundTripComplex64 verifies that Forward(Inverse(x)) ≈ x for radix-4.
func TestDIT16Radix4RoundTripComplex64(t *testing.T) {
	data := make([]complex64, testSize16)
	for i := range data {
		data[i] = complex(float32(i), float32(i*2))
	}
	original := make([]complex64, testSize16)
	copy(original, data)

	twiddle := ComputeTwiddleFactors[complex64](testSize16)
	scratch := make([]complex64, testSize16)
	bitrev := ComputeBitReversalIndicesRadix4(testSize16)

	// Forward
	dst := make([]complex64, testSize16)
	if !forwardDIT16Radix4Complex64(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Radix4Complex64 failed")
	}

	// Inverse
	if !inverseDIT16Radix4Complex64(data, dst, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Radix4Complex64 failed")
	}

	// Compare
	for i := range data {
		if diff := cmplx.Abs(complex128(data[i] - original[i])); diff > testTolerance32 {
			t.Errorf("Round-trip mismatch at index %d: got %v, want %v (diff: %v)", i, data[i], original[i], diff)
		}
	}
}

// TestDIT16CompareRadix2AndRadix4 verifies that radix-2 and radix-4 produce identical results.
func TestDIT16CompareRadix2AndRadix4(t *testing.T) {
	data := make([]complex64, testSize16)
	for i := range data {
		data[i] = complex(float32(math.Cos(float64(i))), float32(math.Sin(float64(i))))
	}

	// Radix-2 forward
	dst2 := make([]complex64, testSize16)
	twiddle := ComputeTwiddleFactors[complex64](testSize16)
	scratch := make([]complex64, testSize16)
	bitrev2 := ComputeBitReversalIndices(testSize16)
	if !forwardDIT16Complex64(dst2, data, twiddle, scratch, bitrev2) {
		t.Fatal("forwardDIT16Complex64 (radix-2) failed")
	}

	// Radix-4 forward
	dst4 := make([]complex64, testSize16)
	bitrev4 := ComputeBitReversalIndicesRadix4(testSize16)
	if !forwardDIT16Radix4Complex64(dst4, data, twiddle, scratch, bitrev4) {
		t.Fatal("forwardDIT16Radix4Complex64 (radix-4) failed")
	}

	// Compare
	for i := range dst2 {
		if diff := cmplx.Abs(complex128(dst2[i] - dst4[i])); diff > testTolerance32 {
			t.Errorf("Radix mismatch at index %d: radix-2=%v, radix-4=%v (diff: %v)", i, dst2[i], dst4[i], diff)
		}
	}
}

// TestDIT16Radix2ForwardComplex128MatchesReference verifies the radix-2 forward transform
// against a naive DFT reference implementation for complex128.
func TestDIT16Radix2ForwardComplex128MatchesReference(t *testing.T) {
	data := make([]complex128, testSize16)
	for i := range data {
		data[i] = complex(float64(i), float64(i*2))
	}

	// Compute using radix-2 DIT
	dst := make([]complex128, testSize16)
	twiddle := ComputeTwiddleFactors[complex128](testSize16)
	scratch := make([]complex128, testSize16)
	bitrev := ComputeBitReversalIndices(testSize16)

	if !forwardDIT16Complex128(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Complex128 failed")
	}

	// Compute reference
	ref := reference.NaiveDFT128(data)

	// Compare
	for i := range dst {
		if diff := cmplx.Abs(dst[i] - ref[i]); diff > testTolerance64 {
			t.Errorf("Mismatch at index %d: got %v, want %v (diff: %v)", i, dst[i], ref[i], diff)
		}
	}
}

// TestDIT16Radix2InverseComplex128MatchesReference verifies the radix-2 inverse transform
// against a naive IDFT reference implementation for complex128.
func TestDIT16Radix2InverseComplex128MatchesReference(t *testing.T) {
	data := make([]complex128, testSize16)
	for i := range data {
		data[i] = complex(float64(i+1), float64(i*3))
	}

	// Compute using radix-2 DIT
	dst := make([]complex128, testSize16)
	twiddle := ComputeTwiddleFactors[complex128](testSize16)
	scratch := make([]complex128, testSize16)
	bitrev := ComputeBitReversalIndices(testSize16)

	if !inverseDIT16Complex128(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Complex128 failed")
	}

	// Compute reference
	ref := reference.NaiveIDFT128(data)

	// Compare
	for i := range dst {
		if diff := cmplx.Abs(dst[i] - ref[i]); diff > testTolerance64 {
			t.Errorf("Mismatch at index %d: got %v, want %v (diff: %v)", i, dst[i], ref[i], diff)
		}
	}
}

// TestDIT16Radix4ForwardComplex128MatchesReference verifies the radix-4 forward transform
// against a naive DFT reference implementation for complex128.
func TestDIT16Radix4ForwardComplex128MatchesReference(t *testing.T) {
	data := make([]complex128, testSize16)
	for i := range data {
		data[i] = complex(float64(i), float64(i*2))
	}

	// Compute using radix-4 DIT
	dst := make([]complex128, testSize16)
	twiddle := ComputeTwiddleFactors[complex128](testSize16)
	scratch := make([]complex128, testSize16)
	bitrev := ComputeBitReversalIndicesRadix4(testSize16)

	if !forwardDIT16Radix4Complex128(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Radix4Complex128 failed")
	}

	// Compute reference
	ref := reference.NaiveDFT128(data)

	// Compare
	for i := range dst {
		if diff := cmplx.Abs(dst[i] - ref[i]); diff > testTolerance64 {
			t.Errorf("Mismatch at index %d: got %v, want %v (diff: %v)", i, dst[i], ref[i], diff)
		}
	}
}

// TestDIT16Radix4InverseComplex128MatchesReference verifies the radix-4 inverse transform
// against a naive IDFT reference implementation for complex128.
func TestDIT16Radix4InverseComplex128MatchesReference(t *testing.T) {
	data := make([]complex128, testSize16)
	for i := range data {
		data[i] = complex(float64(i+1), float64(i*3))
	}

	// Compute using radix-4 DIT
	dst := make([]complex128, testSize16)
	twiddle := ComputeTwiddleFactors[complex128](testSize16)
	scratch := make([]complex128, testSize16)
	bitrev := ComputeBitReversalIndicesRadix4(testSize16)

	if !inverseDIT16Radix4Complex128(dst, data, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Radix4Complex128 failed")
	}

	// Compute reference
	ref := reference.NaiveIDFT128(data)

	// Compare
	for i := range dst {
		if diff := cmplx.Abs(dst[i] - ref[i]); diff > testTolerance64 {
			t.Errorf("Mismatch at index %d: got %v, want %v (diff: %v)", i, dst[i], ref[i], diff)
		}
	}
}
