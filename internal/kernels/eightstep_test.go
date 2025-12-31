package kernels

import "testing"

func TestEightStepForwardInverse(t *testing.T) {
	t.Parallel()

	n := 64

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndices(n)

	dst := make([]complex64, n)
	if !ForwardEightStepComplex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("ForwardEightStepComplex64 returned false")
	}

	roundTrip := make([]complex64, n)
	if !InverseEightStepComplex64(roundTrip, dst, twiddle, scratch, bitrev) {
		t.Fatalf("InverseEightStepComplex64 returned false")
	}

	for i := range src {
		if absDiffComplex64(roundTrip[i], src[i]) > 1e-4 {
			t.Fatalf("roundTrip[%d] = %v, want %v", i, roundTrip[i], src[i])
		}
	}
}
