package kernels

import (
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestDIT8Radix4Complex64MatchesReference(t *testing.T) {
	t.Parallel()

	n := 8
	src := randomComplex64(n, 0xABCDEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT8Radix4Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT8Radix4Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size8Tol64)
}

func TestDIT8Radix4Complex128MatchesReference(t *testing.T) {
	t.Parallel()

	n := 8
	src := randomComplex128(n, 0x1234)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT8Radix4Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT8Radix4Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size8Tol128)
}
