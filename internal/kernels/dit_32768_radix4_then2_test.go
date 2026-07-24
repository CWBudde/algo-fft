package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

const (
	size32768Tol64  = 2e-3
	size32768Tol128 = 1e-8 // Relaxed for accumulated floating-point error in 8 stages (and in the naive reference) at n=32768
)

// TestForwardDIT32768Radix4Then2Complex64 tests the size-32768 forward
// radix-4-then-2 kernel. Size 32768 = 2 x 4^7, so this uses 7 radix-4 stages
// plus a final radix-2 stage.
func TestForwardDIT32768Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 32768

	skipNaiveReferenceIfSlow(t)

	if raceDetectorEnabled {
		t.Skip("naive DFT at n=32768 is too slow under the race detector")
	}

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32768Radix4Then2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT32768Radix4Then2Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size32768Tol64)
}

// TestInverseDIT32768Radix4Then2Complex64 tests the size-32768 inverse kernel.
func TestInverseDIT32768Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 32768

	skipNaiveReferenceIfSlow(t)

	if raceDetectorEnabled {
		t.Skip("naive DFT at n=32768 is too slow under the race detector")
	}

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32768Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32768Radix4Then2Complex64 failed")
	}

	if !inverseDIT32768Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32768Radix4Then2Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size32768Tol64)
}

// TestRoundTripDIT32768Radix4Then2Complex64 tests forward then inverse returns original.
func TestRoundTripDIT32768Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 32768

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32768Radix4Then2Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32768Radix4Then2Complex64 failed")
	}

	if !inverseDIT32768Radix4Then2Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32768Radix4Then2Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size32768Tol64)
}

// TestInPlaceDIT32768Radix4Then2Complex64 tests the dst==src aliased case:
// the ping-pong buffer scheme relies on src being fully consumed by stage 1.
func TestInPlaceDIT32768Radix4Then2Complex64(t *testing.T) {
	t.Parallel()

	const n = 32768

	src := randomComplex64(n, 0x0DDBA11)
	data := make([]complex64, n)
	copy(data, src)

	want := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32768Radix4Then2Complex64(want, src, twiddle, scratch) {
		t.Fatal("forwardDIT32768Radix4Then2Complex64 out-of-place failed")
	}

	if !forwardDIT32768Radix4Then2Complex64(data, data, twiddle, scratch) {
		t.Fatal("forwardDIT32768Radix4Then2Complex64 in-place failed")
	}

	assertComplex64Close(t, data, want, size32768Tol64)
}

// TestForwardDIT32768Radix4Then2Complex128 tests the size-32768 forward kernel (complex128).
func TestForwardDIT32768Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 32768

	skipNaiveReferenceIfSlow(t)

	if raceDetectorEnabled {
		t.Skip("naive DFT at n=32768 is too slow under the race detector")
	}

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32768Radix4Then2Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT32768Radix4Then2Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size32768Tol128)
}

// TestInverseDIT32768Radix4Then2Complex128 tests the size-32768 inverse kernel (complex128).
func TestInverseDIT32768Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 32768

	skipNaiveReferenceIfSlow(t)

	if raceDetectorEnabled {
		t.Skip("naive DFT at n=32768 is too slow under the race detector")
	}

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32768Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32768Radix4Then2Complex128 failed")
	}

	if !inverseDIT32768Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32768Radix4Then2Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size32768Tol128)
}

// TestRoundTripDIT32768Radix4Then2Complex128 tests forward then inverse returns original.
func TestRoundTripDIT32768Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	const n = 32768

	src := randomComplex128(n, 0xC0DEC0DE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32768Radix4Then2Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT32768Radix4Then2Complex128 failed")
	}

	if !inverseDIT32768Radix4Then2Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT32768Radix4Then2Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size32768Tol128)
}
