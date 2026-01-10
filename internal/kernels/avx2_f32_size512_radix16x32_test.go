//go:build amd64 && asm && !purego

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestForwardAVX2Size512Radix16x32Complex64(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	if !amd64.ForwardAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		// AVX2 kernel not implemented, use Go fallback
		if !forwardDIT512Mixed16x32Complex64(dst, src, twiddle, scratch) {
			t.Fatal("forwardDIT512Mixed16x32Complex64 failed")
		}
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-4)
}

func TestInverseAVX2Size512Radix16x32Complex64(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	// Forward transform
	if !amd64.ForwardAVX2Size512Radix16x32Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		if !forwardDIT512Mixed16x32Complex64(fwd, src, twiddle, scratch) {
			t.Fatal("forwardDIT512Mixed16x32Complex64 failed")
		}
	}

	// Inverse transform
	if !amd64.InverseAVX2Size512Radix16x32Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		if !inverseDIT512Mixed16x32Complex64(dst, fwd, twiddle, scratch) {
			t.Fatal("inverseDIT512Mixed16x32Complex64 failed")
		}
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, 1e-4)
}

func TestRoundTripAVX2Size512Radix16x32Complex64(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xFEEDFACE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	// Forward transform
	if !amd64.ForwardAVX2Size512Radix16x32Complex64Asm(fwd, src, twiddle, scratch, bitrev) {
		if !forwardDIT512Mixed16x32Complex64(fwd, src, twiddle, scratch) {
			t.Fatal("forwardDIT512Mixed16x32Complex64 failed")
		}
	}

	// Inverse transform
	if !amd64.InverseAVX2Size512Radix16x32Complex64Asm(dst, fwd, twiddle, scratch, bitrev) {
		if !inverseDIT512Mixed16x32Complex64(dst, fwd, twiddle, scratch) {
			t.Fatal("inverseDIT512Mixed16x32Complex64 failed")
		}
	}

	assertComplex64Close(t, dst, src, 1e-4)
}

func BenchmarkForwardAVX2Size512Radix16x32Complex64(b *testing.B) {
	if !cpu.HasAVX2() {
		b.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xBEEFCAFE)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	b.SetBytes(int64(n * 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !amd64.ForwardAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			forwardDIT512Mixed16x32Complex64(dst, src, twiddle, scratch)
		}
	}
}

func BenchmarkInverseAVX2Size512Radix16x32Complex64(b *testing.B) {
	if !cpu.HasAVX2() {
		b.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xCAFEBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	b.SetBytes(int64(n * 8))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !amd64.InverseAVX2Size512Radix16x32Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			inverseDIT512Mixed16x32Complex64(dst, src, twiddle, scratch)
		}
	}
}
