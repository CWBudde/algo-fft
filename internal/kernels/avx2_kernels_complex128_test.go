//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
)

func TestForwardAVX2Size16384Radix4Complex128_VsGo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping size-16384 AVX2 complex128 test in short mode")
	}

	const n = 16384

	src := randomComplex128(n, 0xA1B2C3D4)
	dstASM := make([]complex128, n)
	dstGo := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !amd64.ForwardAVX2Size16384Radix4Complex128Asm(dstASM, src, twiddle, scratch) {
		t.Fatal("ForwardAVX2Size16384Radix4Complex128Asm failed")
	}

	if !forwardDIT16384Radix4Complex128(dstGo, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384Radix4Complex128 failed")
	}

	assertComplex128Close(t, dstASM, dstGo, size16384Tol128)
}

func TestRoundTripAVX2Size16384Radix4Complex128(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping size-16384 AVX2 complex128 test in short mode")
	}

	const n = 16384

	src := randomComplex128(n, 0x11223344)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !amd64.ForwardAVX2Size16384Radix4Complex128Asm(fwd, src, twiddle, scratch) {
		t.Fatal("ForwardAVX2Size16384Radix4Complex128Asm failed")
	}

	if !amd64.InverseAVX2Size16384Radix4Complex128Asm(dst, fwd, twiddle, scratch) {
		t.Fatal("InverseAVX2Size16384Radix4Complex128Asm failed")
	}

	assertComplex128Close(t, dst, src, size16384Tol128)
}
