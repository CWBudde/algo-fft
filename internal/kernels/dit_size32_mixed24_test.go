package kernels

import (
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestForwardDIT32MixedRadix24Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0x1234BEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT32MixedRadix24Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT32MixedRadix24Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, mixedRadix24Tol64)
}

func TestInverseDIT32MixedRadix24Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0xABCD1234)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT32MixedRadix24Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT32MixedRadix24Complex64 failed")
	}

	if !inverseDIT32MixedRadix24Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT32MixedRadix24Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, mixedRadix24Tol64)
}

func TestForwardDIT32MixedRadix24Complex128(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex128(n, 0x1111AAAA)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT32MixedRadix24Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT32MixedRadix24Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, 1e-10)
}

func TestInverseDIT32MixedRadix24Complex128(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex128(n, 0x2222BBBB)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeBitReversalIndicesMixed24(n)

	if !forwardDIT32MixedRadix24Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT32MixedRadix24Complex128 failed")
	}

	if !inverseDIT32MixedRadix24Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT32MixedRadix24Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, 1e-10)
}
