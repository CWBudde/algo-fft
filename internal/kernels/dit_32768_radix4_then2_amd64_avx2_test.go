//go:build amd64 && !purego

package kernels

import (
	"testing"
)

// The AVX2 size-32768 kernel is cross-checked against the generic Go codelet,
// which is itself validated against the naive reference DFT in
// dit_32768_radix4_then2_test.go.
//
// complex128 only: the complex64 side of this kernel was replaced by the
// size-generic 256-bit dit32768_radix4_avx2 (see radix4_ymm_amd64_test.go).

func TestForwardDIT32768Radix4Then2AVX2Complex128VsGeneric(t *testing.T) {
	t.Parallel()
	requireAVX2(t)

	const n = 32768

	src := randomComplex128(n, 0xC3C3C3C3)
	want := make([]complex128, n)
	got := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32768Radix4Then2Complex128(want, src, twiddle, scratch) {
		t.Fatal("generic forward failed")
	}

	if !forwardDIT32768Radix4Then2AVX2Complex128(got, src, twiddle, scratch) {
		t.Fatal("AVX2 forward failed")
	}

	assertComplex128Close(t, got, want, 1e-9)
}

func TestInverseDIT32768Radix4Then2AVX2Complex128VsGeneric(t *testing.T) {
	t.Parallel()
	requireAVX2(t)

	const n = 32768

	src := randomComplex128(n, 0x3C3C3C3C)
	want := make([]complex128, n)
	got := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !inverseDIT32768Radix4Then2Complex128(want, src, twiddle, scratch) {
		t.Fatal("generic inverse failed")
	}

	if !inverseDIT32768Radix4Then2AVX2Complex128(got, src, twiddle, scratch) {
		t.Fatal("AVX2 inverse failed")
	}

	assertComplex128Close(t, got, want, 1e-12)
}

func TestInPlaceDIT32768Radix4Then2AVX2Complex128(t *testing.T) {
	t.Parallel()
	requireAVX2(t)

	const n = 32768

	src := randomComplex128(n, 0xF00DF00D)
	data := make([]complex128, n)
	copy(data, src)

	want := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT32768Radix4Then2AVX2Complex128(want, src, twiddle, scratch) {
		t.Fatal("AVX2 forward out-of-place failed")
	}

	if !forwardDIT32768Radix4Then2AVX2Complex128(data, data, twiddle, scratch) {
		t.Fatal("AVX2 forward in-place failed")
	}

	assertComplex128Close(t, data, want, 0)
}
