package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

const (
	size4096Tol64  = 1e-3
	size4096Tol128 = 1e-9
)

// TestForwardDIT4096Radix4Complex64 tests the size-4096 forward radix-4 kernel
// Size 4096 = 4^6, so this uses 6 radix-4 stages instead of 12 radix-2 stages.
func TestForwardDIT4096Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 4096

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT4096Radix4Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT4096Radix4Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size4096Tol64)
}

// TestInverseDIT4096Radix4Complex64 tests the size-4096 inverse radix-4 kernel.
func TestInverseDIT4096Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 4096

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT4096Radix4Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT4096Radix4Complex64 failed")
	}

	if !inverseDIT4096Radix4Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT4096Radix4Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size4096Tol64)
}

// TestRoundTripDIT4096Radix4Complex64 tests forward then inverse returns original.
func TestRoundTripDIT4096Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 4096

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT4096Radix4Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT4096Radix4Complex64 failed")
	}

	if !inverseDIT4096Radix4Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT4096Radix4Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size4096Tol64)
}

// TestForwardDIT4096Radix4Complex128 tests the size-4096 forward radix-4 kernel (complex128).
func TestForwardDIT4096Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 4096

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT4096Radix4Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT4096Radix4Complex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size4096Tol128)
}

// TestInverseDIT4096Radix4Complex128 tests the size-4096 inverse radix-4 kernel (complex128).
func TestInverseDIT4096Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 4096

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT4096Radix4Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT4096Radix4Complex128 failed")
	}

	if !inverseDIT4096Radix4Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT4096Radix4Complex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size4096Tol128)
}

// TestRoundTripDIT4096Radix4Complex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT4096Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 4096

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT4096Radix4Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT4096Radix4Complex128 failed")
	}

	if !inverseDIT4096Radix4Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT4096Radix4Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size4096Tol128)
}
