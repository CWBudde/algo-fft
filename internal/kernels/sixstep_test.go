package kernels

import "testing"

func TestSixStepForwardInverse(t *testing.T) {
	t.Parallel()

	n := 16

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndices(n)

	dst := make([]complex64, n)
	if !ForwardSixStepComplex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("ForwardSixStepComplex64 returned false")
	}

	roundTrip := make([]complex64, n)
	if !InverseSixStepComplex64(roundTrip, dst, twiddle, scratch, bitrev) {
		t.Fatalf("InverseSixStepComplex64 returned false")
	}

	for i := range src {
		if absDiffComplex64(roundTrip[i], src[i]) > 1e-4 {
			t.Fatalf("roundTrip[%d] = %v, want %v", i, roundTrip[i], src[i])
		}
	}
}

func absDiffComplex64(a, b complex64) float64 {
	d := complex128(a - b)
	return real(d)*real(d) + imag(d)*imag(d)
}
