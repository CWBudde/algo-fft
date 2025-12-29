package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// Tests for size-256 FFT implementations (both radix-2 and radix-4)

// TestDIT256Radix2ForwardMatchesReference tests radix-2 forward transform.
func TestDIT256Radix2ForwardMatchesReference(t *testing.T) {
	t.Parallel()
	const n = 256

	src := randomComplex64(n, 0xBAD14+n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT256Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT256Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

// TestDIT256Radix4ForwardMatchesReference tests radix-4 forward transform.
func TestDIT256Radix4ForwardMatchesReference(t *testing.T) {
	t.Parallel()
	const n = 256

	src := randomComplex64(n, 0xBAD14+n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT256Radix4Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

// TestDIT256Radix4MatchesRadix2 ensures radix-4 and radix-2 produce identical results.
func TestDIT256Radix4MatchesRadix2(t *testing.T) {
	t.Parallel()
	const n = 256

	src := randomComplex64(n, 0xFACE+n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	// Test radix-4 implementation
	dst4 := make([]complex64, n)
	scratch4 := make([]complex64, n)
	bitrev4 := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT256Radix4Complex64(dst4, src, twiddle, scratch4, bitrev4) {
		t.Fatalf("forwardDIT256Radix4Complex64 failed")
	}

	// Test radix-2 implementation
	dst2 := make([]complex64, n)
	scratch2 := make([]complex64, n)
	bitrev2 := ComputeBitReversalIndices(n)

	if !forwardDIT256Complex64(dst2, src, twiddle, scratch2, bitrev2) {
		t.Fatalf("forwardDIT256Complex64 failed")
	}

	// Both should produce identical results
	assertComplex64SliceClose(t, dst4, dst2, n)
}

// TestDIT256Radix4InverseComplex64MatchesReference tests radix-4 inverse transform for complex64.
func TestDIT256Radix4InverseComplex64MatchesReference(t *testing.T) {
	t.Parallel()
	const n = 256

	src := randomComplex64(n, 0xDEAD+n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !inverseDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT256Radix4Complex64 failed")
	}

	want := reference.NaiveIDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

// TestDIT256Radix4RoundTripComplex64 tests forward then inverse with radix-4 for complex64.
func TestDIT256Radix4RoundTripComplex64(t *testing.T) {
	t.Parallel()
	const n = 256

	src := randomComplex64(n, 0xBEEF+n)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	// Forward transform
	if !forwardDIT256Radix4Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT256Radix4Complex64 failed")
	}

	// Inverse transform
	if !inverseDIT256Radix4Complex64(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT256Radix4Complex64 failed")
	}

	// Should recover original
	assertComplex64SliceClose(t, inv, src, n)
}

// TestDIT256Radix4ForwardComplex128MatchesReference tests radix-4 forward transform for complex128.
func TestDIT256Radix4ForwardComplex128MatchesReference(t *testing.T) {
	t.Parallel()
	const n = 256

	src := randomComplex128(n, 0xCAFE+n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT256Radix4Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT256Radix4Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}

// TestDIT256Radix4InverseComplex128MatchesReference tests radix-4 inverse transform for complex128.
func TestDIT256Radix4InverseComplex128MatchesReference(t *testing.T) {
	t.Parallel()
	const n = 256

	src := randomComplex128(n, 0xFEED+n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !inverseDIT256Radix4Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT256Radix4Complex128 failed")
	}

	want := reference.NaiveIDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}

// TestDIT256Radix4RoundTripComplex128 tests forward then inverse with radix-4 for complex128.
func TestDIT256Radix4RoundTripComplex128(t *testing.T) {
	t.Parallel()
	const n = 256

	src := randomComplex128(n, 0xC0DE+n)
	fwd := make([]complex128, n)
	inv := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	// Forward transform
	if !forwardDIT256Radix4Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT256Radix4Complex128 failed")
	}

	// Inverse transform
	if !inverseDIT256Radix4Complex128(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT256Radix4Complex128 failed")
	}

	// Should recover original
	assertComplex128SliceClose(t, inv, src, n)
}
