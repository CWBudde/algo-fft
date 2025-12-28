package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// Tests for size-256 FFT implementations (both radix-2 and radix-4)

// TestDIT256Radix2ForwardMatchesReference tests radix-2 forward transform
func TestDIT256Radix2ForwardMatchesReference(t *testing.T) {
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

// TestDIT256Radix4ForwardMatchesReference tests radix-4 forward transform
func TestDIT256Radix4ForwardMatchesReference(t *testing.T) {
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

// TestDIT256Radix4MatchesRadix2 ensures radix-4 and radix-2 produce identical results
func TestDIT256Radix4MatchesRadix2(t *testing.T) {
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

// TestDIT256Radix4InverseMatchesReference is a placeholder for inverse radix-4
func TestDIT256Radix4InverseMatchesReference(t *testing.T) {
	t.Skip("Inverse radix-4 implementation not yet created")
}

// TestDIT256Radix4RoundTrip is a placeholder for radix-4 round-trip test
func TestDIT256Radix4RoundTrip(t *testing.T) {
	t.Skip("Inverse radix-4 implementation not yet created")
}
