package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestDIT256Radix4ForwardMatchesReference(t *testing.T) {
	const n = 256
	src := randomComplex64(n, 0xBAD14+n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT256Radix4Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestDIT256Radix4InverseMatchesReference(t *testing.T) {
	const n = 256
	src := randomComplex64(n, 0xDEC0DE+n)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT256Radix4Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT256Radix4Complex64 failed")
	}

	if !inverseDIT256Radix4Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT256Radix4Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestDIT256Radix4RoundTrip(t *testing.T) {
	const n = 256
	src := randomComplex64(n, 0xBEEF+n)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	// Forward transform
	if !forwardDIT256Radix4Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT256Radix4Complex64 failed")
	}

	// Inverse transform
	if !inverseDIT256Radix4Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT256Radix4Complex64 failed")
	}

	// Should recover original signal
	assertComplex64SliceClose(t, dst, src, n)
}

func TestDIT256Radix4MatchesRadix2(t *testing.T) {
	const n = 256
	src := randomComplex64(n, 0xFACE+n)

	// Test radix-4 implementation
	dst4 := make([]complex64, n)
	scratch4 := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT256Radix4Complex64(dst4, src, twiddle, scratch4, bitrev) {
		t.Fatalf("forwardDIT256Radix4Complex64 failed")
	}

	// Test radix-2 implementation
	dst2 := make([]complex64, n)
	scratch2 := make([]complex64, n)

	if !forwardDIT256Complex64(dst2, src, twiddle, scratch2, bitrev) {
		t.Fatalf("forwardDIT256Complex64 failed")
	}

	// Both should produce identical results
	assertComplex64SliceClose(t, dst4, dst2, n)
}
