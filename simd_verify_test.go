package algofft

import (
	"fmt"
	"math"
	"math/cmplx"
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algofft/internal/cpu"
)

// TestSIMDVsGeneric verifies that SIMD-optimized implementations produce
// identical results to pure-Go fallback implementations.
func TestSIMDVsGeneric(t *testing.T) {
	// Skip if not on SIMD-capable architecture
	arch := runtime.GOARCH
	if arch != "amd64" && arch != "arm64" {
		t.Skipf("SIMD verification only on amd64/arm64, got %s", arch)
	}

	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("size_%d_complex64", n), func(t *testing.T) {
			testSIMDvsGeneric64(t, n)
		})
		t.Run(fmt.Sprintf("size_%d_complex128", n), func(t *testing.T) {
			testSIMDvsGeneric128(t, n)
		})
	}
}

func testSIMDvsGeneric64(t *testing.T, n int) {
	// Generate test input
	input := make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(i)*0.1, float32(n-i)*0.1)
	}

	// Test with SIMD enabled
	cpu.ResetDetection()
	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create SIMD plan: %v", err)
	}

	simdOut := make([]complex64, n)
	if err := plan.Forward(simdOut, input); err != nil {
		t.Fatalf("SIMD Forward failed: %v", err)
	}

	// Test with forced generic
	cpu.SetForcedFeatures(cpu.Features{
		ForceGeneric: true,
		Architecture: runtime.GOARCH,
	})
	defer cpu.ResetDetection()

	planGeneric, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create generic plan: %v", err)
	}

	genericOut := make([]complex64, n)
	if err := planGeneric.Forward(genericOut, input); err != nil {
		t.Fatalf("Generic Forward failed: %v", err)
	}

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

	if maxRelErr > 1e-6 {
		t.Errorf("SIMD vs Generic: max relative error %e exceeds 1e-6", maxRelErr)
	}
}

func testSIMDvsGeneric128(t *testing.T, n int) {
	input := make([]complex128, n)
	for i := range input {
		input[i] = complex(float64(i)*0.1, float64(n-i)*0.1)
	}

	cpu.ResetDetection()
	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create SIMD plan: %v", err)
	}

	simdOut := make([]complex128, n)
	if err := plan.Forward(simdOut, input); err != nil {
		t.Fatalf("SIMD Forward failed: %v", err)
	}

	cpu.SetForcedFeatures(cpu.Features{
		ForceGeneric: true,
		Architecture: runtime.GOARCH,
	})
	defer cpu.ResetDetection()

	planGeneric, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create generic plan: %v", err)
	}

	genericOut := make([]complex128, n)
	if err := planGeneric.Forward(genericOut, input); err != nil {
		t.Fatalf("Generic Forward failed: %v", err)
	}

	var maxRelErr float64
	for i := range simdOut {
		diff := cmplx.Abs(simdOut[i] - genericOut[i])
		maxMag := math.Max(cmplx.Abs(simdOut[i]), cmplx.Abs(genericOut[i]))
		if maxMag > 1e-14 {
			relErr := diff / maxMag
			if relErr > maxRelErr {
				maxRelErr = relErr
			}
		}
	}

	if maxRelErr > 1e-14 {
		t.Errorf("SIMD vs Generic: max relative error %e exceeds 1e-14", maxRelErr)
	}
}
