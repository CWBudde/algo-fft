package kernels

import (
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func TestForwardAVX2Size512Radix8Complex64(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputePermutationIndices(n, 8)

	if !amd64.ForwardAVX2Size512Radix8Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardAVX2Size512Radix8Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-4)
}

func TestInverseAVX2Size512Radix8Complex64(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputePermutationIndices(n, 8)

	if !amd64.ForwardAVX2Size512Radix8Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardAVX2Size512Radix8Complex64Asm failed")
	}

	if !amd64.InverseAVX2Size512Radix8Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("InverseAVX2Size512Radix8Complex64Asm failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, 1e-4)
}

func TestRoundTripAVX2Size512Radix8Complex64(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xFEEDFACE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputePermutationIndices(n, 8)

	if !amd64.ForwardAVX2Size512Radix8Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardAVX2Size512Radix8Complex64Asm failed")
	}

	if !amd64.InverseAVX2Size512Radix8Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatal("InverseAVX2Size512Radix8Complex64Asm failed")
	}

	assertComplex64Close(t, dst, src, 1e-4)
}
