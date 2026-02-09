package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestSixStepComplex64(t *testing.T) {
	t.Parallel()

	// Six-step works for any square number size
	sizes := []int{16, 64, 256, 1024, 4096}

	for _, n := range sizes {
		t.Run(testName("forward", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x12345678+uint64(n))
			dst := make([]complex64, n)
			scratch := make([]complex64, n+intSqrt(n)*2) // Six-step needs scratch
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !ForwardSixStepComplex64(dst, src, twiddle, scratch) {
				t.Fatalf("ForwardSixStepComplex64 failed for n=%d", n)
			}

			want := reference.NaiveDFT(src)
			assertComplex64Close(t, dst, want, 1e-4)
		})

		t.Run(testName("inverse", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x87654321+uint64(n))
			fwd := make([]complex64, n)
			dst := make([]complex64, n)
			scratch := make([]complex64, n+intSqrt(n)*2)
			twiddle := ComputeTwiddleFactors[complex64](n)

			if !ForwardSixStepComplex64(fwd, src, twiddle, scratch) {
				t.Fatalf("ForwardSixStepComplex64 failed for n=%d", n)
			}

			if !InverseSixStepComplex64(dst, fwd, twiddle, scratch) {
				t.Fatalf("InverseSixStepComplex64 failed for n=%d", n)
			}

			want := reference.NaiveIDFT(fwd)
			assertComplex64Close(t, dst, want, 1e-4)
		})
	}
}
