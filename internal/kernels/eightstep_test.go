package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestEightStepForwardInverse(t *testing.T) {
	t.Parallel()

	n := 64

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n+intSqrt(n)*2)

	dst := make([]complex64, n)
	if !ForwardEightStepComplex64(dst, src, twiddle, scratch) {
		t.Fatalf("ForwardEightStepComplex64 returned false")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, 1e-4)

	roundTrip := make([]complex64, n)
	if !InverseEightStepComplex64(roundTrip, dst, twiddle, scratch) {
		t.Fatalf("InverseEightStepComplex64 returned false")
	}

	// Verify round trip
	// Note: Inverse does not scale, so we expect N * srcOriginal if we didn't scale manually.
	// But `eightStepInverse` logic (in `eightstep.go`) does NOT scale?
	// `eightStepInverse` calls `stockhamInverse`. `stockhamInverse` DOES scale.
	// So `roundTrip` should be equal to `src`.

	assertComplex64Close(t, roundTrip, src, 1e-4)
}
