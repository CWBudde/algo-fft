package transform

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// The correctness tests below call stockhamPackedRun directly so the packed
// implementation is verified on every build; the public wrappers gate on the
// per-build dispatch toggle (see TestStockhamPackedToggleGatesPublicAPI).

func TestStockhamPackedForwardMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

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
		if !stockhamPackedRun(dst, src, twiddle, scratch, packed, false) {
			t.Fatalf("stockhamPackedRun(%d) returned false", n)
		}

		want := reference.NaiveDFT(src)
		assertComplex64Close(t, dst, want, 1e-4)
	}
}

func TestStockhamPackedInverseMatchesReferenceComplex64(t *testing.T) {
	t.Parallel()

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
		if !stockhamPackedRun(dst, src, twiddle, scratch, packed, true) {
			t.Fatalf("stockhamPackedRun(%d) returned false", n)
		}

		want := reference.NaiveIDFT(src)
		assertComplex64Close(t, dst, want, 1e-4)
	}
}

func TestStockhamPackedForwardMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

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
		if !stockhamPackedRun(dst, src, twiddle, scratch, packed, false) {
			t.Fatalf("stockhamPackedRun(%d) returned false", n)
		}

		want := reference.NaiveDFT128(src)
		assertComplex128Close(t, dst, want, 1e-10)
	}
}

func TestStockhamPackedInverseMatchesReferenceComplex128(t *testing.T) {
	t.Parallel()

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
		if !stockhamPackedRun(dst, src, twiddle, scratch, packed, true) {
			t.Fatalf("stockhamPackedRun(%d) returned false", n)
		}

		want := reference.NaiveIDFT128(src)
		assertComplex128Close(t, dst, want, 1e-10)
	}
}

func TestStockhamPackedMatchesStockhamComplex64(t *testing.T) {
	t.Parallel()

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

		if !stockhamPackedRun(dstPacked, src, twiddle, scratch, packed, false) {
			t.Fatalf("stockhamPackedRun(%d) returned false", n)
		}

		if !forwardStockhamComplex64(dstGo, src, twiddle, scratchGo) {
			t.Fatalf("forwardStockhamComplex64(%d) returned false", n)
		}

		assertComplex64Close(t, dstPacked, dstGo, 1e-4)
	}
}

func TestStockhamPackedMatchesStockhamComplex128(t *testing.T) {
	t.Parallel()

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

		if !stockhamPackedRun(dstPacked, src, twiddle, scratch, packed, false) {
			t.Fatalf("stockhamPackedRun(%d) returned false", n)
		}

		if !forwardStockhamComplex128(dstGo, src, twiddle, scratchGo) {
			t.Fatalf("forwardStockhamComplex128(%d) returned false", n)
		}

		assertComplex128Close(t, dstPacked, dstGo, 1e-10)
	}
}

// TestStockhamPackedHandlesEveryBuild locks in the contract that replaced the
// per-build toggle: the exported entry points always run a valid call. Whether
// a *plan* takes this route is now decided at plan construction by
// PackedStockhamEnabled, not by the engine refusing to execute.
func TestStockhamPackedHandlesEveryBuild(t *testing.T) {
	t.Parallel()

	const n = 16

	src := randomComplex64(n, 0x70661E)
	twiddle := ComputeTwiddleFactors[complex64](n)

	packed := ComputePackedTwiddles[complex64](n, 4, twiddle)
	if packed == nil {
		t.Fatalf("ComputePackedTwiddles(%d) returned nil", n)
	}

	dst := make([]complex64, n)
	scratch := make([]complex64, n)

	if !ForwardStockhamPacked(dst, src, twiddle, scratch, packed) {
		t.Error("ForwardStockhamPacked declined a valid call")
	}

	if !InverseStockhamPacked(dst, src, twiddle, scratch, packed) {
		t.Error("InverseStockhamPacked declined a valid call")
	}
}
