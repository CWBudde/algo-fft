package algofft

import (
	"fmt"
	"math"
	"math/cmplx"
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

// TestSIMDVsGeneric verifies that SIMD-optimized implementations produce
// identical results to pure-Go fallback implementations.
func TestSIMDVsGeneric(t *testing.T) {
	t.Parallel()

	// Skip if not on SIMD-capable architecture
	arch := runtime.GOARCH
	if arch != "amd64" && arch != "arm64" {
		t.Skipf("SIMD verification only on amd64/arm64, got %s", arch)
	}

	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, n := range sizes {
		n := n
		t.Run(fmt.Sprintf("size_%d_complex64", n), func(t *testing.T) {
			t.Parallel()
			testSIMDvsGeneric64(t, n)
		})
		t.Run(fmt.Sprintf("size_%d_complex128", n), func(t *testing.T) {
			t.Parallel()
			testSIMDvsGeneric128(t, n)
		})
	}
}

func testSIMDvsGeneric64(t *testing.T, n int) {
	t.Helper()
	// Generate test input
	input := make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(i)*0.1, float32(n-i)*0.1)
	}

	// Test with SIMD enabled
	f := cpu.DetectFeatures()
	t.Logf("Detected features: AVX2=%v, SSE2=%v", f.HasAVX2, f.HasSSE2)

	plan, err := newPlanWithFeatures[complex64](n, f, PlanOptions{})
	if err != nil {
		t.Fatalf("failed to create SIMD plan: %v", err)
	}

	simdOut := make([]complex64, n)
	if err := plan.Forward(simdOut, input); err != nil {
		t.Fatalf("SIMD Forward failed: %v", err)
	}

	// Test with forced generic features (no global override)
	genericFeatures := cpu.Features{
		ForceGeneric: true,
		Architecture: runtime.GOARCH,
	}

	planGeneric, err := newPlanWithFeatures[complex64](n, genericFeatures, PlanOptions{})
	if err != nil {
		t.Fatalf("failed to create generic plan: %v", err)
	}

	genericOut := make([]complex64, n)
	if err := planGeneric.Forward(genericOut, input); err != nil {
		t.Fatalf("Generic Forward failed: %v", err)
	}

	t.Logf("SIMD Plan using algorithm: %s", plan.Algorithm())
	t.Logf("Generic Plan using algorithm: %s", planGeneric.Algorithm())

	// Compare
	var maxRelErr float32

	for i := range simdOut {
		diff := cmplx64abs(simdOut[i] - genericOut[i])

		maxMag := math.Max(float64(cmplx64abs(simdOut[i])), float64(cmplx64abs(genericOut[i])))
		if maxMag > 1e-10 {
			relErr := float32(float64(diff) / maxMag)
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
	}

	threshold := float32(1e-6)
	if n == 64 {
		threshold = 2e-6
	}

	if n >= 256 {
		threshold = 5e-6
	}

	if n >= 1024 {
		threshold = 5e-5
	}

	if n >= 4096 {
		// Six-step algorithm for size 4096 has slightly more accumulated error
		// due to additional transpose and twiddle multiply operations
		threshold = 1e-4
	}

	if n >= 16384 {
		threshold = 4e-4
	}

	if maxRelErr > threshold {
		t.Errorf("SIMD vs Generic: max relative error %e exceeds %e", maxRelErr, threshold)
	}
}

func testSIMDvsGeneric128(t *testing.T, n int) {
	t.Helper()

	input := make([]complex128, n)
	for i := range input {
		input[i] = complex(float64(i)*0.1, float64(n-i)*0.1)
	}

	f := cpu.DetectFeatures()

	plan, err := newPlanWithFeatures[complex128](n, f, PlanOptions{})
	if err != nil {
		t.Fatalf("failed to create SIMD plan: %v", err)
	}

	simdOut := make([]complex128, n)
	if err := plan.Forward(simdOut, input); err != nil {
		t.Fatalf("SIMD Forward failed: %v", err)
	}

	genericFeatures := cpu.Features{
		ForceGeneric: true,
		Architecture: runtime.GOARCH,
	}

	planGeneric, err := newPlanWithFeatures[complex128](n, genericFeatures, PlanOptions{})
	if err != nil {
		t.Fatalf("failed to create generic plan: %v", err)
	}

	genericOut := make([]complex128, n)
	if err := planGeneric.Forward(genericOut, input); err != nil {
		t.Fatalf("Generic Forward failed: %v", err)
	}

	var (
		maxRelErr   float64
		firstBadIdx = -1
	)

	for i := range simdOut {
		diff := cmplx.Abs(simdOut[i] - genericOut[i])

		maxMag := math.Max(cmplx.Abs(simdOut[i]), cmplx.Abs(genericOut[i]))
		if maxMag > 1e-14 {
			relErr := diff / maxMag
			if relErr > maxRelErr {
				maxRelErr = relErr
			}

			if relErr > 1e-6 && firstBadIdx == -1 {
				firstBadIdx = i
			}
		}
	}

	if maxRelErr > 1e-14 {
		t.Errorf("SIMD vs Generic: max relative error %e exceeds 1e-14", maxRelErr)

		if n == 64 && firstBadIdx >= 0 {
			t.Logf("First bad index: %d", firstBadIdx)

			badCount := 0
			for i := 0; i < len(simdOut) && badCount < 20; i++ {
				diff := cmplx.Abs(simdOut[i] - genericOut[i])

				maxMag := math.Max(cmplx.Abs(simdOut[i]), cmplx.Abs(genericOut[i]))
				if maxMag > 1e-14 {
					relErr := diff / maxMag
					if relErr > 1e-6 {
						t.Logf("  BAD[%d] SIMD=%v Generic=%v (relErr=%.2e)", i, simdOut[i], genericOut[i], relErr)

						badCount++
					}
				}
			}
		}
	}
}
