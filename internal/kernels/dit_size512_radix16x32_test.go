package kernels

import (
	"math"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestForwardDIT512Mixed16x32Complex64(t *testing.T) {
	t.Parallel()

	const n = 512

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i), float32(n-i))
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n) // Mixed radix operates on natural-order input.
	refBitrev := mathpkg.ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)
	dst := make([]complex64, n)

	if !forwardDIT512Mixed16x32Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT512Mixed16x32Complex64 failed")
	}

	// Compare with reference (or another known good implementation)
	refDst := make([]complex64, n)
	forwardDIT512Complex64(refDst, src, twiddle, scratch, refBitrev)

	// Use relative tolerance for float32 (explicit DFT accumulates more FP error)
	const relTol = 1e-5

	for i := range dst {
		diff := dst[i] - refDst[i]
		errMag := math.Hypot(float64(real(diff)), float64(imag(diff)))
		refMag := math.Hypot(float64(real(refDst[i])), float64(imag(refDst[i])))
		// Use relative tolerance, with minimum absolute tolerance for small values
		tol := relTol * refMag
		if tol < 1e-3 {
			tol = 1e-3
		}

		if errMag > tol {
			t.Errorf("Mismatch at index %d: got %v, want %v (err=%v, tol=%v)", i, dst[i], refDst[i], errMag, tol)
		}
	}
}

func TestInverseDIT512Mixed16x32Complex64(t *testing.T) {
	t.Parallel()

	const n = 512

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	if !forwardDIT512Mixed16x32Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT512Mixed16x32Complex64 failed")
	}

	if !inverseDIT512Mixed16x32Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT512Mixed16x32Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	// Larger tolerance because 16x32 might have more accumulated error or the reference naive IDFT is precise
	// but the kernel uses mixed radix. 1e-3 is generally safe for complex64 FFTs of this size.
	assertComplex64Close(t, dst, want, 1e-3)
}

func TestRoundTripDIT512Mixed16x32Complex64(t *testing.T) {
	t.Parallel()

	const n = 512

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i), float32(n-i))
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n) // Mixed radix operates on natural-order input.
	scratch := make([]complex64, n)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)

	forwardDIT512Mixed16x32Complex64(fwd, src, twiddle, scratch, bitrev)

	if !inverseDIT512Mixed16x32Complex64(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT512Mixed16x32Complex64 failed")
	}

	for i := range src {
		diff := inv[i] - src[i]
		if math.Hypot(float64(real(diff)), float64(imag(diff))) > 1e-3 {
			t.Errorf("Mismatch at index %d: got %v, want %v", i, inv[i], src[i])
		}
	}
}

func TestForwardDIT512Mixed16x32Complex128(t *testing.T) {
	t.Parallel()

	const n = 512

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i), float64(n-i))
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeIdentityIndices(n) // Mixed radix operates on natural-order input.
	refBitrev := mathpkg.ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)

	if !forwardDIT512Mixed16x32Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT512Mixed16x32Complex128 failed")
	}

	refDst := make([]complex128, n)
	forwardDIT512Complex128(refDst, src, twiddle, scratch, refBitrev)

	for i := range dst {
		diff := dst[i] - refDst[i]
		if math.Hypot(real(diff), imag(diff)) > 1e-9 {
			t.Errorf("Mismatch at index %d: got %v, want %v", i, dst[i], refDst[i])
		}
	}
}

func TestInverseDIT512Mixed16x32Complex128(t *testing.T) {
	t.Parallel()

	const n = 512

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	if !forwardDIT512Mixed16x32Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT512Mixed16x32Complex128 failed")
	}

	if !inverseDIT512Mixed16x32Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT512Mixed16x32Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, 1e-9)
}

func TestRoundTripDIT512Mixed16x32Complex128(t *testing.T) {
	t.Parallel()

	const n = 512

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i), float64(n-i))
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)
	scratch := make([]complex128, n)
	fwd := make([]complex128, n)
	inv := make([]complex128, n)

	forwardDIT512Mixed16x32Complex128(fwd, src, twiddle, scratch, bitrev)

	if !inverseDIT512Mixed16x32Complex128(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT512Mixed16x32Complex128 failed")
	}

	for i := range src {
		diff := inv[i] - src[i]
		if math.Hypot(real(diff), imag(diff)) > 1e-9 {
			t.Errorf("Mismatch at index %d: got %v, want %v", i, inv[i], src[i])
		}
	}
}
