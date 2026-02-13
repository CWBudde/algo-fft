package algofft

import (
	"math/cmplx"
	"testing"
)

func TestSystematicSizes(t *testing.T) {
	sizes := []int{8, 10, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 64}
	const tol = 1e-10

	for _, size := range sizes {
		t.Run(sprintf("size_%d", size), func(t *testing.T) {
			data := make([]complex128, size)
			for i := range data {
				data[i] = complex(float64(i), 0)
			}

			orig := make([]complex128, size)
			copy(orig, data)

			plan, err := NewPlanT[complex128](size)
			if err != nil {
				t.Fatalf("Failed to create plan: %v", err)
			}

			err = plan.Forward(data, data)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			err = plan.Inverse(data, data)
			if err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			maxErr := 0.0
			for i := range data {
				err := cmplx.Abs(data[i] - orig[i])
				if err > maxErr {
					maxErr = err
				}
			}

			if maxErr > tol {
				t.Errorf("Round-trip error: %e (algorithm: %s)", maxErr, plan.Algorithm())
			}
		})
	}
}
