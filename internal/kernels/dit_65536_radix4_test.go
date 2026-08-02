package kernels

import (
	"testing"
)

// A naive O(n^2) DFT at n=65536 is ~4.3 billion multiply-adds — far too slow
// for a unit test (the 16384 template's naive reference is already the
// largest one this package runs). Instead this validates against
// forwardRadix4Complex64/forwardRadix4Complex128 (internal/kernels/radix4.go
// and its generated twin): an independently implemented, loop-based,
// bit-reversal-driven radix-4 engine that is not specialized per size and
// shares no code with the unrolled ping-pong stages here. It is exercised
// elsewhere (radix4_test.go) at 4, 16, 64, 256, 1024 against the naive DFT,
// and radix4Transform's algorithm accepts any power of 4, so it is a valid
// oracle at 65536 too.
const (
	size65536Tol64  = 2e-3
	size65536Tol128 = 4e-9 // Relaxed vs. 16384's 2e-9: one more accumulation stage.
)

// TestForwardDIT65536Radix4Complex64 tests the size-65536 forward radix-4
// kernel. 65536 = 4^8, so this uses 8 radix-4 stages, ping-ponging between
// dst and scratch instead of holding each stage in its own stack array (see
// the header comment on forwardDIT65536Radix4Complex64).
func TestForwardDIT65536Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 65536

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT65536Radix4Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT65536Radix4Complex64 failed")
	}

	want := make([]complex64, n)
	wantScratch := make([]complex64, n)

	if !forwardRadix4Complex64(want, src, twiddle, wantScratch) {
		t.Fatal("forwardRadix4Complex64 reference failed")
	}

	assertComplex64Close(t, dst, want, size65536Tol64)
}

// TestInverseDIT65536Radix4Complex64 tests the size-65536 inverse radix-4 kernel.
func TestInverseDIT65536Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 65536

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT65536Radix4Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT65536Radix4Complex64 failed")
	}

	if !inverseDIT65536Radix4Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT65536Radix4Complex64 failed")
	}

	wantScratch := make([]complex64, n)
	want := make([]complex64, n)

	if !inverseRadix4Complex64(want, fwd, twiddle, wantScratch) {
		t.Fatal("inverseRadix4Complex64 reference failed")
	}

	assertComplex64Close(t, dst, want, size65536Tol64)
}

// TestRoundTripDIT65536Radix4Complex64 tests forward then inverse returns original.
func TestRoundTripDIT65536Radix4Complex64(t *testing.T) {
	t.Parallel()

	const n = 65536

	src := randomComplex64(n, 0xBADC0FFE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT65536Radix4Complex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT65536Radix4Complex64 failed")
	}

	if !inverseDIT65536Radix4Complex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT65536Radix4Complex64 failed")
	}

	assertComplex64Close(t, dst, src, size65536Tol64)
}

// TestForwardDIT65536Radix4Complex128 tests the size-65536 forward radix-4
// kernel (complex128).
func TestForwardDIT65536Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 65536

	src := randomComplex128(n, 0xBEEFCAFE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT65536Radix4Complex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT65536Radix4Complex128 failed")
	}

	want := make([]complex128, n)
	wantScratch := make([]complex128, n)

	if !forwardRadix4Complex128(want, src, twiddle, wantScratch) {
		t.Fatal("forwardRadix4Complex128 reference failed")
	}

	assertComplex128Close(t, dst, want, size65536Tol128)
}

// TestInverseDIT65536Radix4Complex128 tests the size-65536 inverse radix-4
// kernel (complex128).
func TestInverseDIT65536Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 65536

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT65536Radix4Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT65536Radix4Complex128 failed")
	}

	if !inverseDIT65536Radix4Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT65536Radix4Complex128 failed")
	}

	wantScratch := make([]complex128, n)
	want := make([]complex128, n)

	if !inverseRadix4Complex128(want, fwd, twiddle, wantScratch) {
		t.Fatal("inverseRadix4Complex128 reference failed")
	}

	assertComplex128Close(t, dst, want, size65536Tol128)
}

// TestRoundTripDIT65536Radix4Complex128 tests forward then inverse returns
// original (complex128).
func TestRoundTripDIT65536Radix4Complex128(t *testing.T) {
	t.Parallel()

	const n = 65536

	src := randomComplex128(n, 0xC0FFEE42)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT65536Radix4Complex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT65536Radix4Complex128 failed")
	}

	if !inverseDIT65536Radix4Complex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT65536Radix4Complex128 failed")
	}

	assertComplex128Close(t, dst, src, size65536Tol128)
}
