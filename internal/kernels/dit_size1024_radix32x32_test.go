package kernels

import (
	"math"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestForwardDIT1024Mixed32x32Complex64(t *testing.T) {
	t.Parallel()

	const n = 1024

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i), float32(n-i))
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)
	scratch := make([]complex64, n)
	dst := make([]complex64, n)

	if !forwardDIT1024Mixed32x32Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT1024Mixed32x32Complex64 failed")
	}

	refDst := reference.NaiveDFT(src)

	const relTol = 2e-5

	for i := range dst {
		diff := dst[i] - refDst[i]
		errMag := math.Hypot(float64(real(diff)), float64(imag(diff)))
		refMag := math.Hypot(float64(real(refDst[i])), float64(imag(refDst[i])))

		tol := relTol * refMag
		if tol < 1e-3 {
			tol = 1e-3
		}

		if errMag > tol {
			t.Errorf("Mismatch at index %d: got %v, want %v (err=%v, tol=%v)", i, dst[i], refDst[i], errMag, tol)
		}
	}
}

func TestInverseDIT1024Mixed32x32Complex64(t *testing.T) {
	t.Parallel()

	const n = 1024

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	if !forwardDIT1024Mixed32x32Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT1024Mixed32x32Complex64 failed")
	}

	if !inverseDIT1024Mixed32x32Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT1024Mixed32x32Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	// Using the same tolerance logic as in Forward
	assertComplex64Close(t, dst, want, 2e-3)
}

func TestRoundTripDIT1024Mixed32x32Complex64(t *testing.T) {
	t.Parallel()

	const n = 1024

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i), float32(n-i))
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)
	scratch := make([]complex64, n)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)

	forwardDIT1024Mixed32x32Complex64(fwd, src, twiddle, scratch, bitrev)

	if !inverseDIT1024Mixed32x32Complex64(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT1024Mixed32x32Complex64 failed")
	}

	for i := range src {
		diff := inv[i] - src[i]
		if math.Hypot(float64(real(diff)), float64(imag(diff))) > 1e-3 {
			t.Errorf("Mismatch at index %d: got %v, want %v", i, inv[i], src[i])
		}
	}
}

func TestForwardDIT1024Mixed32x32Complex128(t *testing.T) {
	t.Parallel()

	const n = 1024

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i), float64(n-i))
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)

	if !forwardDIT1024Mixed32x32Complex128(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT1024Mixed32x32Complex128 failed")
	}

	refDst := reference.NaiveDFT128(src)

	for i := range dst {
		diff := dst[i] - refDst[i]
		if math.Hypot(real(diff), imag(diff)) > 1e-7 {
			t.Errorf("Mismatch at index %d: got %v, want %v", i, dst[i], refDst[i])
		}
	}
}

func TestInverseDIT1024Mixed32x32Complex128(t *testing.T) {
	t.Parallel()

	const n = 1024

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	if !forwardDIT1024Mixed32x32Complex128(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT1024Mixed32x32Complex128 failed")
	}

	if !inverseDIT1024Mixed32x32Complex128(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT1024Mixed32x32Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, 1e-9)
}

func TestRoundTripDIT1024Mixed32x32Complex128(t *testing.T) {
	t.Parallel()

	const n = 1024

	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i), float64(n-i))
	}

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)
	scratch := make([]complex128, n)
	fwd := make([]complex128, n)
	inv := make([]complex128, n)

	forwardDIT1024Mixed32x32Complex128(fwd, src, twiddle, scratch, bitrev)

	if !inverseDIT1024Mixed32x32Complex128(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT1024Mixed32x32Complex128 failed")
	}

	for i := range src {
		diff := inv[i] - src[i]
		if math.Hypot(real(diff), imag(diff)) > 1e-9 {
			t.Errorf("Mismatch at index %d: got %v, want %v", i, inv[i], src[i])
		}
	}
}

func BenchmarkRadix32x32Forward_1024(b *testing.B) {
	const n = 1024

	src := randomComplex64(n, 0x1234+uint64(n))
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	b.SetBytes(int64(n * 8))
	b.ReportAllocs()
	b.ResetTimer()

	for range b.N {
		if !forwardDIT1024Mixed32x32Complex64(dst, src, twiddle, scratch, bitrev) {
			b.Fatalf("kernel failed for n=%d", n)
		}
	}
}
