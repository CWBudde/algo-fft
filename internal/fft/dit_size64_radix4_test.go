package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// Tests for size-64 radix-4 FFT implementations

func TestDIT64Radix4ForwardMatchesReference(t *testing.T) {
	t.Parallel()
	const n = 64

	src := randomComplex64(n, 0xD164+n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT64Radix4Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT64Radix4Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestDIT64Radix4MatchesRadix2(t *testing.T) {
	t.Parallel()
	const n = 64

	src := randomComplex64(n, 0xD164+0x10+n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	dst4 := make([]complex64, n)
	scratch4 := make([]complex64, n)
	bitrev4 := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT64Radix4Complex64(dst4, src, twiddle, scratch4, bitrev4) {
		t.Fatalf("forwardDIT64Radix4Complex64 failed")
	}

	dst2 := make([]complex64, n)
	scratch2 := make([]complex64, n)
	bitrev2 := ComputeBitReversalIndices(n)

	if !forwardDIT64Complex64(dst2, src, twiddle, scratch2, bitrev2) {
		t.Fatalf("forwardDIT64Complex64 failed")
	}

	assertComplex64SliceClose(t, dst4, dst2, n)
}

func TestDIT64Radix4InverseComplex64MatchesReference(t *testing.T) {
	t.Parallel()
	const n = 64

	src := randomComplex64(n, 0xD164+0x20+n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !inverseDIT64Radix4Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT64Radix4Complex64 failed")
	}

	want := reference.NaiveIDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

func TestDIT64Radix4RoundTripComplex64(t *testing.T) {
	t.Parallel()
	const n = 64

	src := randomComplex64(n, 0xD164+0x30+n)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT64Radix4Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT64Radix4Complex64 failed")
	}

	if !inverseDIT64Radix4Complex64(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT64Radix4Complex64 failed")
	}

	assertComplex64SliceClose(t, inv, src, n)
}

func TestDIT64Radix4ForwardComplex128MatchesReference(t *testing.T) {
	t.Parallel()
	const n = 64

	src := randomComplex128(n, 0xD164+0x40+n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT64Radix4Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT64Radix4Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}

func TestDIT64Radix4InverseComplex128MatchesReference(t *testing.T) {
	t.Parallel()
	const n = 64

	src := randomComplex128(n, 0xD164+0x50+n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !inverseDIT64Radix4Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT64Radix4Complex128 failed")
	}

	want := reference.NaiveIDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}

func TestDIT64Radix4RoundTripComplex128(t *testing.T) {
	t.Parallel()
	const n = 64

	src := randomComplex128(n, 0xD164+0x60+n)
	fwd := make([]complex128, n)
	inv := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT64Radix4Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT64Radix4Complex128 failed")
	}

	if !inverseDIT64Radix4Complex128(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT64Radix4Complex128 failed")
	}

	assertComplex128SliceClose(t, inv, src, n)
}
