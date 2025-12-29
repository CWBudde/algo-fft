package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

func TestStockhamPackedForwardMatchesReferenceComplex64(t *testing.T) {
	if !StockhamPackedAvailable() {
		t.Skip("packed stockham disabled in this build")
	}

	sizes := []int{4, 8, 16, 32, 64}
	for _, n := range sizes {
		src := randomComplex64(n, 0xA11CE+uint64(n))
		twiddle := ComputeTwiddleFactors[complex64](n)
		packed := ComputePackedTwiddles[complex64](n, 4, twiddle)
		if packed == nil {
			t.Fatalf("ComputePackedTwiddles(%d) returned nil", n)
		}

		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		if !ForwardStockhamPacked(dst, src, twiddle, scratch, packed) {
			t.Fatalf("ForwardStockhamPacked(%d) returned false", n)
		}

		want := reference.NaiveDFT(src)
		assertComplex64SliceClose(t, dst, want, n)
	}
}

func TestStockhamPackedInverseMatchesReferenceComplex64(t *testing.T) {
	if !StockhamPackedAvailable() {
		t.Skip("packed stockham disabled in this build")
	}

	sizes := []int{4, 8, 16, 32, 64}
	for _, n := range sizes {
		src := randomComplex64(n, 0xBADC0DE+uint64(n))
		twiddle := ComputeTwiddleFactors[complex64](n)
		packed := ComputePackedTwiddles[complex64](n, 4, twiddle)
		if packed == nil {
			t.Fatalf("ComputePackedTwiddles(%d) returned nil", n)
		}

		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		if !InverseStockhamPacked(dst, src, twiddle, scratch, packed) {
			t.Fatalf("InverseStockhamPacked(%d) returned false", n)
		}

		want := reference.NaiveIDFT(src)
		assertComplex64SliceClose(t, dst, want, n)
	}
}

func TestStockhamPackedForwardMatchesReferenceComplex128(t *testing.T) {
	if !StockhamPackedAvailable() {
		t.Skip("packed stockham disabled in this build")
	}

	sizes := []int{4, 8, 16, 32}
	for _, n := range sizes {
		src := randomComplex128(n, 0xC001D00D+uint64(n))
		twiddle := ComputeTwiddleFactors[complex128](n)
		packed := ComputePackedTwiddles[complex128](n, 4, twiddle)
		if packed == nil {
			t.Fatalf("ComputePackedTwiddles(%d) returned nil", n)
		}

		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		if !ForwardStockhamPacked(dst, src, twiddle, scratch, packed) {
			t.Fatalf("ForwardStockhamPacked(%d) returned false", n)
		}

		want := reference.NaiveDFT128(src)
		assertComplex128SliceClose(t, dst, want, n)
	}
}

func TestStockhamPackedInverseMatchesReferenceComplex128(t *testing.T) {
	if !StockhamPackedAvailable() {
		t.Skip("packed stockham disabled in this build")
	}

	sizes := []int{4, 8, 16, 32}
	for _, n := range sizes {
		src := randomComplex128(n, 0xDEADBEEF+uint64(n))
		twiddle := ComputeTwiddleFactors[complex128](n)
		packed := ComputePackedTwiddles[complex128](n, 4, twiddle)
		if packed == nil {
			t.Fatalf("ComputePackedTwiddles(%d) returned nil", n)
		}

		dst := make([]complex128, n)
		scratch := make([]complex128, n)
		if !InverseStockhamPacked(dst, src, twiddle, scratch, packed) {
			t.Fatalf("InverseStockhamPacked(%d) returned false", n)
		}

		want := reference.NaiveIDFT128(src)
		assertComplex128SliceClose(t, dst, want, n)
	}
}

func TestStockhamPackedMatchesStockhamComplex64(t *testing.T) {
	if !StockhamPackedAvailable() {
		t.Skip("packed stockham disabled in this build")
	}

	sizes := []int{256, 1024, 2048}
	for _, n := range sizes {
		src := randomComplex64(n, 0xFEEDFACE+uint64(n))
		twiddle := ComputeTwiddleFactors[complex64](n)
		packed := ComputePackedTwiddles[complex64](n, 4, twiddle)
		if packed == nil {
			t.Fatalf("ComputePackedTwiddles(%d) returned nil", n)
		}

		dstPacked := make([]complex64, n)
		dstGo := make([]complex64, n)
		scratch := make([]complex64, n)
		scratchGo := make([]complex64, n)
		bitrev := make([]int, n)

		if !ForwardStockhamPacked(dstPacked, src, twiddle, scratch, packed) {
			t.Fatalf("ForwardStockhamPacked(%d) returned false", n)
		}
		if !forwardStockhamComplex64(dstGo, src, twiddle, scratchGo, bitrev) {
			t.Fatalf("forwardStockhamComplex64(%d) returned false", n)
		}

		assertComplex64SliceClose(t, dstPacked, dstGo, n)
	}
}

func TestStockhamPackedMatchesStockhamComplex128(t *testing.T) {
	if !StockhamPackedAvailable() {
		t.Skip("packed stockham disabled in this build")
	}

	sizes := []int{256, 1024, 2048}
	for _, n := range sizes {
		src := randomComplex128(n, 0xF00DBAAD+uint64(n))
		twiddle := ComputeTwiddleFactors[complex128](n)
		packed := ComputePackedTwiddles[complex128](n, 4, twiddle)
		if packed == nil {
			t.Fatalf("ComputePackedTwiddles(%d) returned nil", n)
		}

		dstPacked := make([]complex128, n)
		dstGo := make([]complex128, n)
		scratch := make([]complex128, n)
		scratchGo := make([]complex128, n)
		bitrev := make([]int, n)

		if !ForwardStockhamPacked(dstPacked, src, twiddle, scratch, packed) {
			t.Fatalf("ForwardStockhamPacked(%d) returned false", n)
		}
		if !forwardStockhamComplex128(dstGo, src, twiddle, scratchGo, bitrev) {
			t.Fatalf("forwardStockhamComplex128(%d) returned false", n)
		}

		assertComplex128SliceClose(t, dstPacked, dstGo, n)
	}
}
