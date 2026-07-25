//go:build !race

package algofft

import (
	"strconv"
	"testing"
)

// TestRecursiveTransformsNoAllocs pins the zero-allocation guarantee for the
// recursive strategy. The decomposition allocated per call — sub-input
// buffers at every tree node, plus one temporary per output element in the
// generic combine — which is the "and allocates" half of the PLAN.md P5.0
// recursive defect.
//
//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestRecursiveTransformsNoAllocs(t *testing.T) {
	for _, n := range recursiveLargeSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: KernelRecursive})
			if err != nil {
				t.Fatalf("NewPlanWithOptions(%d, recursive): %v", n, err)
			}

			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i%13)-6, float32(i%5)-2)
			}

			dst := make([]complex64, n)
			freq := make([]complex64, n)

			// Warm any lazily-populated shared tables before measuring.
			if err := plan.Forward(freq, src); err != nil {
				t.Fatalf("Forward: %v", err)
			}

			assertNoAllocs(t, "Forward", func() error {
				return plan.Forward(dst, src)
			})
			assertNoAllocs(t, "Inverse", func() error {
				return plan.Inverse(dst, freq)
			})
		})
	}
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestRecursiveTransformsNoAllocsComplex128(t *testing.T) {
	for _, n := range recursiveLargeSizes {
		t.Run(strconv.Itoa(n), func(t *testing.T) {
			plan, err := NewPlanWithOptions[complex128](n, PlanOptions{Strategy: KernelRecursive})
			if err != nil {
				t.Fatalf("NewPlanWithOptions(%d, recursive): %v", n, err)
			}

			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i%13)-6, float64(i%5)-2)
			}

			dst := make([]complex128, n)
			freq := make([]complex128, n)

			if err := plan.Forward(freq, src); err != nil {
				t.Fatalf("Forward: %v", err)
			}

			assertNoAllocs(t, "Forward", func() error {
				return plan.Forward(dst, src)
			})
			assertNoAllocs(t, "Inverse", func() error {
				return plan.Inverse(dst, freq)
			})
		})
	}
}
