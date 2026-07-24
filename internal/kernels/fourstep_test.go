package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestFourStepComplex64(t *testing.T) {
	t.Parallel()

	// Four-step works for any power-of-two size >= 4, including the
	// non-square sizes six-step declines (8, 32, 128, ...).
	sizes := []int{4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}

	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x12345678+uint64(n))
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !ForwardFourStepComplex64(dst, src, twiddle, scratch) {
				t.Fatalf("ForwardFourStepComplex64 failed for n=%d", n)
			}

			want := reference.NaiveDFT(src)
			assertComplex64Close(t, dst, want, 1e-4)
		})

		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x87654321+uint64(n))
			fwd := make([]complex64, n)
			dst := make([]complex64, n)
			scratch := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !ForwardFourStepComplex64(fwd, src, twiddle, scratch) {
				t.Fatalf("ForwardFourStepComplex64 failed for n=%d", n)
			}

			if !InverseFourStepComplex64(dst, fwd, twiddle, scratch) {
				t.Fatalf("InverseFourStepComplex64 failed for n=%d", n)
			}

			want := reference.NaiveIDFT(fwd)
			assertComplex64Close(t, dst, want, 1e-4)
		})
	}
}

func TestFourStepComplex128(t *testing.T) {
	t.Parallel()

	sizes := []int{4, 8, 32, 128, 512, 2048}

	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x12345678+uint64(n))
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			if !ForwardFourStepComplex128(dst, src, twiddle, scratch) {
				t.Fatalf("ForwardFourStepComplex128 failed for n=%d", n)
			}

			want := reference.NaiveDFT128(src)
			assertComplex128Close(t, dst, want, 1e-10)
		})

		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x87654321+uint64(n))
			fwd := make([]complex128, n)
			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			if !ForwardFourStepComplex128(fwd, src, twiddle, scratch) {
				t.Fatalf("ForwardFourStepComplex128 failed for n=%d", n)
			}

			if !InverseFourStepComplex128(dst, fwd, twiddle, scratch) {
				t.Fatalf("InverseFourStepComplex128 failed for n=%d", n)
			}

			want := reference.NaiveIDFT128(fwd)
			assertComplex128Close(t, dst, want, 1e-10)
		})
	}
}

func TestFourStepComplex64InPlace(t *testing.T) {
	t.Parallel()

	for _, n := range []int{64, 128, 2048} {
		src := randomComplex64(n, 0xABCDEF+uint64(n))
		data := make([]complex64, n)
		copy(data, src)

		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)

		if !ForwardFourStepComplex64(data, data, twiddle, scratch) {
			t.Fatalf("in-place forward failed for n=%d", n)
		}

		want := reference.NaiveDFT(src)
		assertComplex64Close(t, data, want, 1e-4)
	}
}

func TestFourStepComplex64RoundTripLarge(t *testing.T) {
	t.Parallel()

	// Non-square power-of-two sizes above the reference-DFT range.
	for _, n := range []int{8192, 32768} {
		src := randomComplex64(n, 0x5EED+uint64(n))
		fwd := make([]complex64, n)
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)

		if !ForwardFourStepComplex64(fwd, src, twiddle, scratch) {
			t.Fatalf("forward failed for n=%d", n)
		}

		if !InverseFourStepComplex64(dst, fwd, twiddle, scratch) {
			t.Fatalf("inverse failed for n=%d", n)
		}

		assertComplex64Close(t, dst, src, 1e-4)
	}
}

func TestFourStepDeclinesUnsupportedSizes(t *testing.T) {
	t.Parallel()

	for _, n := range []int{2, 3, 12, 100, 768} {
		src := randomComplex64(n, uint64(n))
		dst := make([]complex64, n)
		scratch := make([]complex64, n)
		twiddle := ComputeTwiddleFactors[complex64](n)

		if ForwardFourStepComplex64(dst, src, twiddle, scratch) {
			t.Errorf("expected four-step to decline n=%d", n)
		}
	}
}

func TestFourStepSplit(t *testing.T) {
	caches := cpu.CacheInfo{L1DataBytes: 32 * 1024, L2Bytes: 256 * 1024}

	for _, n := range []int{4, 8, 64, 1024, 1 << 18, 1 << 19, 1 << 22, 1 << 23} {
		n1, n2 := fourStepSplit(n, 8, caches)

		if n1*n2 != n {
			t.Errorf("n=%d: n1*n2 = %d*%d = %d, want %d", n, n1, n2, n1*n2, n)
		}

		if n1 < 2 || n2 < 2 {
			t.Errorf("n=%d: factors %d, %d below minimum 2", n, n1, n2)
		}

		if n1&(n1-1) != 0 || n2&(n2-1) != 0 {
			t.Errorf("n=%d: factors %d, %d not powers of two", n, n1, n2)
		}
	}
}

func TestFourStepSplitBalancedWhenCacheIsLarge(t *testing.T) {
	// With both row passes fitting L1 comfortably, the split should be
	// (near-)balanced: that minimizes the longer row length.
	caches := cpu.CacheInfo{L1DataBytes: 1 << 20, L2Bytes: 1 << 24}

	n1, n2 := fourStepSplit(1<<16, 8, caches)
	if n1 != 256 || n2 != 256 {
		t.Errorf("n=2^16: got %dx%d, want balanced 256x256", n1, n2)
	}

	n1, n2 = fourStepSplit(1<<17, 8, caches)
	if n1*n2 != 1<<17 || max(n1, n2) != 2*min(n1, n2) {
		t.Errorf("n=2^17: got %dx%d, want a 1:2 split", n1, n2)
	}
}

func TestFourStepSplitIndependentOfChoice(t *testing.T) {
	t.Parallel()

	// The transform must be correct for every valid split, since the split
	// choice is a performance decision tuned per machine.
	const n = 1 << 12

	src := randomComplex64(n, 0xF00D)
	twiddle := ComputeTwiddleFactors[complex64](n)
	want := reference.NaiveDFT(src)

	for n1 := 2; n1 <= n/2; n1 *= 2 {
		dst := make([]complex64, n)

		scratch := make([]complex64, n)
		if !fourStepTransform(dst, src, twiddle, scratch, n1, false) {
			t.Fatalf("fourStepTransform failed for split %dx%d", n1, n/n1)
		}

		assertComplex64Close(t, dst, want, 1e-4)
	}
}

func TestFourStepZeroAlloc(t *testing.T) {
	const n = 1 << 14

	src := randomComplex64(n, 0xA110C)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	allocs := testing.AllocsPerRun(5, func() {
		if !ForwardFourStepComplex64(dst, src, twiddle, scratch) {
			t.Fatal("forward failed")
		}

		if !InverseFourStepComplex64(dst, dst, twiddle, scratch) {
			t.Fatal("inverse failed")
		}
	})
	if allocs != 0 {
		t.Errorf("transform allocated %v times, want 0", allocs)
	}
}
