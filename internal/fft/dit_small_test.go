package fft

import (
	"testing"

	"github.com/MeKo-Christian/algoforge/internal/reference"
)

func TestDITSmallForwardMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{8, 16, 32, 64, 128} {
		src := randomComplex64(n, 0xD17D17+uint64(n))
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := ComputeBitReversalIndices(n)

		if !forwardDITComplex64(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardDITComplex64 failed for n=%d", n)
		}

		want := reference.NaiveDFT(src)
		assertComplex64SliceClose(t, dst, want, n)
	}
}

func TestDITSmallInverseMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

	for _, n := range []int{8, 16, 32, 64, 128} {
		src := randomComplex64(n, 0x1A2B3C+uint64(n))
		fwd := make([]complex64, n)
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)
		bitrev := ComputeBitReversalIndices(n)

		if !forwardDITComplex64(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardDITComplex64 failed for n=%d", n)
		}

		if !inverseDITComplex64(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseDITComplex64 failed for n=%d", n)
		}

		want := reference.NaiveIDFT(fwd)
		assertComplex64SliceClose(t, dst, want, n)
	}
}

func TestDITSmallForwardMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{8, 16, 32, 64, 128} {
		src := randomComplex128(n, 0xC0FFEE+uint64(n))
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := ComputeBitReversalIndices(n)

		if !forwardDITComplex128(dst, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardDITComplex128 failed for n=%d", n)
		}

		want := reference.NaiveDFT128(src)
		assertComplex128SliceClose(t, dst, want, n)
	}
}

func TestDITSmallInverseMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

	for _, n := range []int{8, 16, 32, 64, 128} {
		src := randomComplex128(n, 0xF00DBA+uint64(n))
		fwd := make([]complex128, n)
		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		twiddle := ComputeTwiddleFactors[complex128](n)
		bitrev := ComputeBitReversalIndices(n)

		if !forwardDITComplex128(fwd, src, twiddle, scratch, bitrev) {
			t.Fatalf("forwardDITComplex128 failed for n=%d", n)
		}

		if !inverseDITComplex128(dst, fwd, twiddle, scratch, bitrev) {
			t.Fatalf("inverseDITComplex128 failed for n=%d", n)
		}

		want := reference.NaiveIDFT128(fwd)
		assertComplex128SliceClose(t, dst, want, n)
	}
}
