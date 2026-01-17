package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestForwardDIT32Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0x1234BEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32Radix4Then2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix4Then2Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, radix4Then2Tol64)
}

func TestInverseDIT32Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0xABCD1234)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix4Then2Complex64 failed")
	}

	if !inverseDIT32Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32Radix4Then2Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, radix4Then2Tol64)
}

func TestForwardDIT32Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex128(n, 0x1111AAAA)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32Radix4Then2Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix4Then2Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, 1e-10)
}

func TestInverseDIT32Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex128(n, 0x2222BBBB)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix4Then2Complex128 failed")
	}

	if !inverseDIT32Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32Radix4Then2Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, 1e-10)
}

func TestRoundTripDIT32Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix4Then2Complex64 failed")
	}

	if !inverseDIT32Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32Radix4Then2Complex64 failed")
	}

	assertComplex64Close(t, dst, src, radix4Then2Tol64)
}

func TestRoundTripDIT32Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 32

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32Radix4Then2Complex128 failed")
	}

	if !inverseDIT32Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32Radix4Then2Complex128 failed")
	}

	assertComplex128Close(t, dst, src, 1e-10)
}
