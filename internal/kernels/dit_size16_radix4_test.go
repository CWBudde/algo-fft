package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardDIT16Radix4Complex64 tests the size-16 radix-4 forward kernel
func TestForwardDIT16Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT16Radix4Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Radix4Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size16Tol64)
}

// TestInverseDIT16Radix4Complex64 tests the size-16 radix-4 inverse kernel
func TestInverseDIT16Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT16Radix4Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Radix4Complex64 failed")
	}

	if !inverseDIT16Radix4Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Radix4Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size16Tol64)
}

// TestForwardDIT16Radix4Complex128 tests the size-16 radix-4 forward kernel (complex128)
func TestForwardDIT16Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT16Radix4Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Radix4Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size16Tol128)
}

// TestInverseDIT16Radix4Complex128 tests the size-16 radix-4 inverse kernel (complex128)
func TestInverseDIT16Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardDIT16Radix4Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT16Radix4Complex128 failed")
	}

	if !inverseDIT16Radix4Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT16Radix4Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size16Tol128)
}
