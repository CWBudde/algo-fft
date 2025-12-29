//go:build amd64 && fft_asm && !purego

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func requireAVX2(t *testing.T) {
	if !cpu.DetectFeatures().HasAVX2 {
		t.Skip("AVX2 not available on this machine")
	}
}

func TestAVX2Size16Complex128ForwardMatchesReference(t *testing.T) {
	requireAVX2(t)

	const n = 16
	src := randomComplex128(n, 0x16C128)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardAVX2Size16Complex128Asm failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}

func TestAVX2Size16Complex128InverseMatchesReference(t *testing.T) {
	requireAVX2(t)

	const n = 16
	src := randomComplex128(n, 0x161128)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)

	if !inverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseAVX2Size16Complex128Asm failed")
	}

	want := reference.NaiveIDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}

func TestAVX2Size16Radix4Complex128ForwardMatchesReference(t *testing.T) {
	requireAVX2(t)

	const n = 16
	src := randomComplex128(n, 0x164C0FEE)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !forwardAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardAVX2Size16Radix4Complex128Asm failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}

func TestAVX2Size16Radix4Complex128InverseMatchesReference(t *testing.T) {
	requireAVX2(t)

	const n = 16
	src := randomComplex128(n, 0x1641D1F7)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	if !inverseAVX2Size16Radix4Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseAVX2Size16Radix4Complex128Asm failed")
	}

	want := reference.NaiveIDFT128(src)
	assertComplex128SliceClose(t, dst, want, n)
}
