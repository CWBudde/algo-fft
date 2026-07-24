//go:build amd64 && !purego

package kernels

import (
	"testing"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
)

// The SSE size-16384 and size-32768 kernels are cross-checked against the
// generic Go codelets, which are themselves validated against the naive
// reference DFT in their own tests.

func TestForwardSSE3Size16384Radix4Complex64VsGeneric(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 16384

	src := randomComplex64(n, 0xA1B2C3D4)
	want := make([]complex64, n)
	got := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT16384Radix4Complex64(want, src, twiddle, scratch) {
		t.Fatal("generic forward failed")
	}

	if !amd64.ForwardSSE3Size16384Radix4Complex64Asm(got, src, twiddle, scratch) {
		t.Fatal("SSE3 forward failed")
	}

	assertComplex64Close(t, got, want, 1e-3)
}

func TestInverseSSE3Size16384Radix4Complex64VsGeneric(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 16384

	src := randomComplex64(n, 0x4D3C2B1A)
	want := make([]complex64, n)
	got := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !inverseDIT16384Radix4Complex64(want, src, twiddle, scratch) {
		t.Fatal("generic inverse failed")
	}

	if !amd64.InverseSSE3Size16384Radix4Complex64Asm(got, src, twiddle, scratch) {
		t.Fatal("SSE3 inverse failed")
	}

	assertComplex64Close(t, got, want, 1e-6)
}

func TestInPlaceSSE3Size16384Radix4Complex64(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 16384

	src := randomComplex64(n, 0x600DCAFE)
	data := make([]complex64, n)
	copy(data, src)

	want := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !amd64.ForwardSSE3Size16384Radix4Complex64Asm(want, src, twiddle, scratch) {
		t.Fatal("SSE3 forward out-of-place failed")
	}

	if !amd64.ForwardSSE3Size16384Radix4Complex64Asm(data, data, twiddle, scratch) {
		t.Fatal("SSE3 forward in-place failed")
	}

	assertComplex64Close(t, data, want, 0)
}

func TestForwardSSE2Size16384Radix4Complex128VsGeneric(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 16384

	src := randomComplex128(n, 0xD4C3B2A1)
	want := make([]complex128, n)
	got := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT16384Radix4Complex128(want, src, twiddle, scratch) {
		t.Fatal("generic forward failed")
	}

	if !amd64.ForwardSSE2Size16384Radix4Complex128Asm(got, src, twiddle, scratch) {
		t.Fatal("SSE2 forward failed")
	}

	assertComplex128Close(t, got, want, 1e-9)
}

func TestInverseSSE2Size16384Radix4Complex128VsGeneric(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 16384

	src := randomComplex128(n, 0x1A2B3C4D)
	want := make([]complex128, n)
	got := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !inverseDIT16384Radix4Complex128(want, src, twiddle, scratch) {
		t.Fatal("generic inverse failed")
	}

	if !amd64.InverseSSE2Size16384Radix4Complex128Asm(got, src, twiddle, scratch) {
		t.Fatal("SSE2 inverse failed")
	}

	assertComplex128Close(t, got, want, 1e-12)
}

func TestInPlaceSSE2Size16384Radix4Complex128(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 16384

	src := randomComplex128(n, 0xFEEDFACE)
	data := make([]complex128, n)
	copy(data, src)

	want := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !amd64.ForwardSSE2Size16384Radix4Complex128Asm(want, src, twiddle, scratch) {
		t.Fatal("SSE2 forward out-of-place failed")
	}

	if !amd64.ForwardSSE2Size16384Radix4Complex128Asm(data, data, twiddle, scratch) {
		t.Fatal("SSE2 forward in-place failed")
	}

	assertComplex128Close(t, data, want, 0)
}

func TestForwardDIT32768Radix4Then2SSE3Complex64VsGeneric(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 32768

	src := randomComplex64(n, 0xB16B00B5)
	want := make([]complex64, n)
	got := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32768Radix4Then2Complex64(want, src, twiddle, scratch) {
		t.Fatal("generic forward failed")
	}

	if !forwardDIT32768Radix4Then2SSE3Complex64(got, src, twiddle, scratch) {
		t.Fatal("SSE3 forward failed")
	}

	assertComplex64Close(t, got, want, 1e-3)
}

func TestInverseDIT32768Radix4Then2SSE3Complex64VsGeneric(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 32768

	src := randomComplex64(n, 0x5B00B1B6)
	want := make([]complex64, n)
	got := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !inverseDIT32768Radix4Then2Complex64(want, src, twiddle, scratch) {
		t.Fatal("generic inverse failed")
	}

	if !inverseDIT32768Radix4Then2SSE3Complex64(got, src, twiddle, scratch) {
		t.Fatal("SSE3 inverse failed")
	}

	assertComplex64Close(t, got, want, 1e-6)
}

func TestInPlaceDIT32768Radix4Then2SSE3Complex64(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 32768

	src := randomComplex64(n, 0xDEADBEA7)
	data := make([]complex64, n)
	copy(data, src)

	want := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT32768Radix4Then2SSE3Complex64(want, src, twiddle, scratch) {
		t.Fatal("SSE3 forward out-of-place failed")
	}

	if !forwardDIT32768Radix4Then2SSE3Complex64(data, data, twiddle, scratch) {
		t.Fatal("SSE3 forward in-place failed")
	}

	assertComplex64Close(t, data, want, 0)
}

func TestForwardDIT32768Radix4Then2SSE2Complex128VsGeneric(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 32768

	src := randomComplex128(n, 0xCAFED00D)
	want := make([]complex128, n)
	got := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32768Radix4Then2Complex128(want, src, twiddle, scratch) {
		t.Fatal("generic forward failed")
	}

	if !forwardDIT32768Radix4Then2SSE2Complex128(got, src, twiddle, scratch) {
		t.Fatal("SSE2 forward failed")
	}

	assertComplex128Close(t, got, want, 1e-9)
}

func TestInverseDIT32768Radix4Then2SSE2Complex128VsGeneric(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 32768

	src := randomComplex128(n, 0xD00DCAFE)
	want := make([]complex128, n)
	got := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !inverseDIT32768Radix4Then2Complex128(want, src, twiddle, scratch) {
		t.Fatal("generic inverse failed")
	}

	if !inverseDIT32768Radix4Then2SSE2Complex128(got, src, twiddle, scratch) {
		t.Fatal("SSE2 inverse failed")
	}

	assertComplex128Close(t, got, want, 1e-12)
}

func TestInPlaceDIT32768Radix4Then2SSE2Complex128(t *testing.T) {
	t.Parallel()
	requireSSE3(t)

	const n = 32768

	src := randomComplex128(n, 0xBAADF00D)
	data := make([]complex128, n)
	copy(data, src)

	want := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32768Radix4Then2SSE2Complex128(want, src, twiddle, scratch) {
		t.Fatal("SSE2 forward out-of-place failed")
	}

	if !forwardDIT32768Radix4Then2SSE2Complex128(data, data, twiddle, scratch) {
		t.Fatal("SSE2 forward in-place failed")
	}

	assertComplex128Close(t, data, want, 0)
}
