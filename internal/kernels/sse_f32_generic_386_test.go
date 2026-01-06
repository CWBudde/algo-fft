//go:build 386 && asm && !purego

package kernels

import (
	"testing"

	x86 "github.com/MeKo-Christian/algo-fft/internal/asm/x86"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestForwardSSEGenericComplex64_386(t *testing.T) {
	const n = 32 // Power of 2, not 4
	src := randomComplex64(n, 0x11111111)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	// Call the SSE assembly implementation
	if !x86.ForwardSSEComplex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSEComplex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-4)
}

func TestInverseSSEGenericComplex64_386(t *testing.T) {
	const n = 32
	src := randomComplex64(n, 0x22222222)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	// We need input in frequency domain for Inverse test to get back to time domain?
	// Or just test IFFT(src) vs NaiveIDFT(src).
	
	if !x86.InverseSSEComplex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSEComplex64Asm failed")
	}

	want := reference.NaiveIDFT(src)
	assertComplex64Close(t, dst, want, 1e-4)
}