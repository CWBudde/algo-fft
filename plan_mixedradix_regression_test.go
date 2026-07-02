package algofft

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestPlan_MixedRadixCompositeSizes pins the mixed-radix fallback against a
// naive DFT for composite sizes whose factorization reduces to a registered
// codelet size (regression: sizes 2^k*5^m with 2^k >= 8 returned garbage when
// the schedule emitted a composite radix the recursion driver could not
// execute).
func TestPlan_MixedRadixCompositeSizes(t *testing.T) {
	t.Parallel()

	sizes := []int{20, 40, 48, 80, 96, 120, 160, 200, 320, 640, 1280}

	for _, n := range sizes {
		plan, err := NewPlan64(n)
		if err != nil {
			t.Fatalf("NewPlan64(%d): %v", n, err)
		}

		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(math.Sin(float64(i)*0.7)+1, math.Cos(float64(i)*0.3))
		}

		want := naiveDFTComplex128(src)
		dst := make([]complex128, n)

		err = plan.Forward(dst, src)
		if err != nil {
			t.Fatalf("Forward(%d): %v", n, err)
		}

		if maxDiff, tol := maxAbsDiffComplex128(dst, want), 1e-9*float64(n); maxDiff > tol {
			t.Errorf("n=%d (algo=%s): forward max error %g exceeds tol %g", n, plan.Algorithm(), maxDiff, tol)
		}

		roundTrip := make([]complex128, n)

		err = plan.Inverse(roundTrip, dst)
		if err != nil {
			t.Fatalf("Inverse(%d): %v", n, err)
		}

		if maxDiff, tol := maxAbsDiffComplex128(roundTrip, src), 1e-11*float64(n); maxDiff > tol {
			t.Errorf("n=%d (algo=%s): round-trip max error %g exceeds tol %g", n, plan.Algorithm(), maxDiff, tol)
		}
	}
}

func naiveDFTComplex128(src []complex128) []complex128 {
	n := len(src)
	dst := make([]complex128, n)

	for k := range n {
		var sum complex128
		for j := range n {
			sum += src[j] * cmplx.Exp(complex(0, -2*math.Pi*float64(k*j)/float64(n)))
		}

		dst[k] = sum
	}

	return dst
}

func maxAbsDiffComplex128(a, b []complex128) float64 {
	maxDiff := 0.0

	for i := range a {
		if diff := cmplx.Abs(a[i] - b[i]); diff > maxDiff {
			maxDiff = diff
		}
	}

	return maxDiff
}
