//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardSSE2Size16Radix16Complex64 tests the SSE2 size-16 radix-16 forward kernel
func TestForwardSSE2Size16Radix16Complex64(t *testing.T) {
	t.Parallel()

	const n = 16
	src := randomComplex64(n, 0xABCDEFFF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	// Radix-16 kernel expects natural order input
	bitrev := mathpkg.ComputeIdentityIndices(n)

	if !amd64.ForwardSSE2Size16Radix16Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSE2Size16Radix16Complex64Asm failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-6)
}
