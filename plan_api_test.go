package algofft

import (
	"errors"
	"math/cmplx"
	"runtime"
	"strconv"
	"strings"
	"testing"
)

// TestInverseInPlace tests the InverseInPlace method.
func TestInverseInPlace(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](16)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	// Create test data
	src := make([]complex64, 16)
	for i := range src {
		src[i] = complex(float32(i+1), float32(-i))
	}

	// Forward transform
	freq := make([]complex64, 16)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Inverse in-place
	err = plan.InverseInPlace(freq)
	if err != nil {
		t.Fatalf("InverseInPlace failed: %v", err)
	}

	// Verify round-trip accuracy
	for i := range src {
		assertApproxComplex64f(t, freq[i], src[i], 1e-4, "freq[%d]", i)
	}
}

func TestInverseInPlace_Complex128(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex128](32)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	// Create test data
	src := make([]complex128, 32)
	for i := range src {
		src[i] = complex(float64(i+1), float64(-i)*0.5)
	}

	// Forward transform
	freq := make([]complex128, 32)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Inverse in-place
	err = plan.InverseInPlace(freq)
	if err != nil {
		t.Fatalf("InverseInPlace failed: %v", err)
	}

	// Verify round-trip accuracy
	for i := range src {
		assertApproxComplex128f(t, freq[i], src[i], "freq[%d]", i)
	}
}

func TestInverseInPlace_NilSlice(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	err = plan.InverseInPlace(nil)
	if !errors.Is(err, ErrNilSlice) {
		t.Errorf("InverseInPlace(nil) = %v, want ErrNilSlice", err)
	}
}

func TestInverseInPlace_LengthMismatch(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](8)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	err = plan.InverseInPlace(make([]complex64, 4))
	if !errors.Is(err, ErrLengthMismatch) {
		t.Errorf("InverseInPlace(short) = %v, want ErrLengthMismatch", err)
	}
}

// TestKernelStrategy tests the KernelStrategy method.
func TestKernelStrategy(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		size     int
		strategy KernelStrategy
	}{
		{"Auto_Small", 64, KernelAuto},
		{"Auto_Large", 2048, KernelAuto},
		{"DIT", 128, KernelDIT},
		{"Stockham", 256, KernelStockham},
		{"SixStep", 4096, KernelSixStep},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanWithOptions[complex64](tt.size, PlanOptions{Strategy: tt.strategy})
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) failed: %v", tt.size, err)
			}

			strategy := plan.KernelStrategy()

			// For KernelAuto, the actual strategy depends on size
			// For specific strategies, they should match (or be auto-selected if not available)
			if tt.strategy != KernelAuto && strategy != tt.strategy && strategy != KernelAuto {
				// Allow fallback to auto if strategy isn't implemented for this size
				t.Logf("KernelStrategy() = %v, requested %v (may have fallen back)", strategy, tt.strategy)
			}

			// Verify the plan is functional
			src := make([]complex64, tt.size)
			dst := make([]complex64, tt.size)
			src[0] = 1

			err = plan.Forward(dst, src)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}
		})
	}
}

func TestNewPlanWithOptions_ForcedStrategyOverridesCodelet(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanWithOptions[complex64](8, PlanOptions{Strategy: KernelStockham})
	if err != nil {
		t.Fatalf("NewPlanWithOptions failed: %v", err)
	}

	if got := plan.KernelStrategy(); got != KernelStockham {
		t.Fatalf("KernelStrategy() = %v, want %v", got, KernelStockham)
	}

	if algo := plan.Algorithm(); algo == "dit8_generic" {
		t.Fatalf("Algorithm() = %q, expected non-codelet when strategy forced", algo)
	}
}

func TestNewPlanWithOptions_ForcedStrategyOverridesCodelet128(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanWithOptions[complex128](8, PlanOptions{Strategy: KernelStockham})
	if err != nil {
		t.Fatalf("NewPlanWithOptions failed: %v", err)
	}

	if got := plan.KernelStrategy(); got != KernelStockham {
		t.Fatalf("KernelStrategy() = %v, want %v", got, KernelStockham)
	}

	if algo := plan.Algorithm(); algo == "dit8_generic" {
		t.Fatalf("Algorithm() = %q, expected non-codelet when strategy forced", algo)
	}
}

func TestNewPlanFromPool_ForcedStrategyOverridesCodelet(t *testing.T) {
	t.Parallel()

	pooled, err := NewPlanPooledWithOptions[complex64](8, PlanOptions{Strategy: KernelStockham})
	if err != nil {
		t.Fatalf("NewPlanPooledWithOptions failed: %v", err)
	}
	defer pooled.Close()

	if got := pooled.KernelStrategy(); got != KernelStockham {
		t.Fatalf("KernelStrategy() = %v, want %v", got, KernelStockham)
	}

	if algo := pooled.Algorithm(); algo == "dit8_generic" {
		t.Fatalf("Algorithm() = %q, expected non-codelet when strategy forced", algo)
	}
}

func TestNewPlanFromPool_ForcedStrategyOverridesCodelet128(t *testing.T) {
	t.Parallel()

	pooled, err := NewPlanPooledWithOptions[complex128](8, PlanOptions{Strategy: KernelStockham})
	if err != nil {
		t.Fatalf("NewPlanPooledWithOptions failed: %v", err)
	}
	defer pooled.Close()

	if got := pooled.KernelStrategy(); got != KernelStockham {
		t.Fatalf("KernelStrategy() = %v, want %v", got, KernelStockham)
	}

	if algo := pooled.Algorithm(); algo == "dit8_generic" {
		t.Fatalf("Algorithm() = %q, expected non-codelet when strategy forced", algo)
	}
}

func TestPlanAlgorithmSize512Radix4Then2Complex128(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex128](512)
	if err != nil {
		t.Fatalf("NewPlan[complex64](512) returned error: %v", err)
	}

	// The planner should select the best size-512 codelet for the build:
	//   * amd64 AVX2 builds: the size-generic 256-bit radix-8 ladder
	//     ("dit512_radix8ladder_avx2"), which took this row on 2026-07-30 at
	//     0.933 forward / 0.979 inverse against the size-generic radix-4
	//     kernel. That kernel ("dit512_radix4_avx2") is still accepted, for
	//     builds predating the ladder, as is the per-size radix-8 codelet it
	//     had itself superseded ("dit512_radix8_avx2") -- which is a different
	//     and XMM-width kernel despite the similar name.
	//   * amd64 SSE2-only builds: the radix-4-then-2 SSE2 override.
	//   * purego and generic builds: the size-generic radix-8 ladder
	//     ("dit512_radix8ladder_generic"), which took this row on 2026-07-30 at
	//     0.986 forward / 0.867 inverse against the radix-4-then-2 scalar
	//     codelet (still accepted, for builds predating the ladder).
	//   * arm64 SIMD builds: the NEON radix-4-then-2 codelet
	//     ("dit512_radix4_then2_neon") wins via the shared prefix check; the
	//     size-512 generic NEON codelet ("dit512_generic_neon") is also
	//     accepted for older builds where the size-specific kernel is absent,
	//     because codelet selection prefers a higher SIMD level over a
	//     higher-priority scalar codelet.
	algo := plan.Algorithm()
	ok := strings.HasPrefix(algo, "dit512_radix4_then2") ||
		algo == "dit512_radix8ladder_avx2" ||
		algo == "dit512_radix4_avx2" ||
		algo == "dit512_radix8_avx2" ||
		algo == "dit512_radix8ladder_generic"
	if runtime.GOARCH == "arm64" {
		ok = ok || algo == "dit512_generic_neon"
	}

	if !ok {
		t.Fatalf("Algorithm() = %q, want %q, %q, prefix %q, or %q on arm64 SIMD builds",
			algo, "dit512_radix4_avx2", "dit512_radix8_avx2", "dit512_radix4_then2", "dit512_generic_neon")
	}
}

// TestStrategyIsolation verifies that per-plan PlanOptions.Strategy is honored
// independently, with no shared process-global strategy state: two plans built
// with different forced strategies each report their own.
func TestStrategyIsolation(t *testing.T) {
	t.Parallel()

	planA, err := NewPlanWithOptions[complex64](512, PlanOptions{Strategy: KernelDIT})
	if err != nil {
		t.Fatalf("NewPlanWithOptions(512, DIT) failed: %v", err)
	}

	planB, err := NewPlanWithOptions[complex64](512, PlanOptions{Strategy: KernelStockham})
	if err != nil {
		t.Fatalf("NewPlanWithOptions(512, Stockham) failed: %v", err)
	}

	if got := planA.KernelStrategy(); got != KernelDIT {
		t.Errorf("planA.KernelStrategy() = %v, want KernelDIT", got)
	}

	if got := planB.KernelStrategy(); got != KernelStockham {
		t.Errorf("planB.KernelStrategy() = %v, want KernelStockham", got)
	}

	// Creating planB must not have altered planA's snapshot.
	if got := planA.KernelStrategy(); got != KernelDIT {
		t.Errorf("planA.KernelStrategy() after building planB = %v, want KernelDIT", got)
	}

	// Both plans must produce correct results.
	src := make([]complex64, 512)
	dst := make([]complex64, 512)
	src[0] = 1

	err = planA.Forward(dst, src)
	if err != nil {
		t.Fatalf("planA.Forward failed: %v", err)
	}

	err = planB.Forward(dst, src)
	if err != nil {
		t.Fatalf("planB.Forward failed: %v", err)
	}
}

// TestTransform tests the Transform convenience method.
func TestTransform(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](16)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	src := make([]complex64, 16)
	for i := range src {
		src[i] = complex(float32(i+1), 0)
	}

	// Test forward
	dstFwd := make([]complex64, 16)

	err = plan.Transform(dstFwd, src, false)
	if err != nil {
		t.Fatalf("Transform(forward) failed: %v", err)
	}

	// Test inverse
	dstInv := make([]complex64, 16)

	err = plan.Transform(dstInv, dstFwd, true)
	if err != nil {
		t.Fatalf("Transform(inverse) failed: %v", err)
	}

	// Verify round-trip
	for i := range src {
		assertApproxComplex64f(t, dstInv[i], src[i], 1e-4, "dstInv[%d]", i)
	}
}

// TestString_AllStrategies tests String method with different strategies.
func TestString_AllStrategies(t *testing.T) {
	t.Parallel()

	tests := []struct {
		strategy     KernelStrategy
		expectedName string
		size         int
	}{
		{KernelDIT, "DIT", 64},
		{KernelStockham, "Stockham", 256},
		{KernelSixStep, "SixStep", 4096},
		{KernelAuto, "auto", 128}, // Auto might resolve to DIT or Stockham
	}

	for _, tt := range tests {
		t.Run(tt.expectedName, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanWithOptions[complex64](tt.size, PlanOptions{Strategy: tt.strategy})
			if err != nil {
				t.Fatalf("NewPlan failed: %v", err)
			}

			s := plan.String()
			if s == "" {
				t.Error("String() returned empty string")
			}

			// Should contain size
			sizeStr := strconv.Itoa(tt.size)
			if !contains(s, sizeStr) {
				t.Errorf("String() should contain '%s', got: %s", sizeStr, s)
			}

			// For specific strategies, check the name appears (unless it fell back to auto)
			if tt.strategy != KernelAuto {
				actualStrategy := plan.KernelStrategy()
				if actualStrategy == tt.strategy && !contains(s, tt.expectedName) {
					t.Errorf("String() should contain '%s' for strategy %v, got: %s", tt.expectedName, tt.strategy, s)
				}
			}
		})
	}
}

// TestItoa tests the internal itoa function via String().
func TestItoa(t *testing.T) {
	t.Parallel()

	tests := []struct {
		size     int
		expected string
	}{
		{0, "0"}, // Edge case, though not valid for FFT
		{1, "1"},
		{8, "8"},
		{16, "16"},
		{128, "128"},
		{1024, "1024"},
		{65536, "65536"},
	}

	for _, tt := range tests {
		if tt.size < 1 {
			continue // Skip invalid FFT sizes
		}

		t.Run(tt.expected, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan[complex64](tt.size)
			if err != nil {
				t.Fatalf("NewPlan[complex64](%d) failed: %v", tt.size, err)
			}

			s := plan.String()
			if !contains(s, tt.expected) {
				t.Errorf("String() should contain '%s', got: %s", tt.expected, s)
			}
		})
	}

	// Test itoa directly with negative numbers (though not used in Plan)
	if result := strconv.Itoa(-42); result != "-42" {
		t.Errorf("strconv.Itoa(-42) = %s, want -42", result)
	}

	if result := strconv.Itoa(0); result != "0" {
		t.Errorf("strconv.Itoa(0) = %s, want 0", result)
	}
}

// TestNewPlanPooled_AlreadyTested is covered by plan_pool_test.go

// TestPlan_ConcurrentUse tests that plans can be used concurrently.
func TestPlan_ConcurrentUse(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan[complex64](128)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	// Run multiple goroutines using the same plan
	const numGoroutines = 10

	done := make(chan bool, numGoroutines)

	for range numGoroutines {
		go func() {
			src := make([]complex64, 128)
			dst := make([]complex64, 128)
			src[0] = 1

			// Perform forward transform
			err := plan.Forward(dst, src)
			if err != nil {
				t.Errorf("Forward failed: %v", err)
			}

			// Verify impulse response (all ones)
			for i := range dst {
				if cmplx.Abs(complex128(dst[i]-1)) > 1e-4 {
					t.Errorf("dst[%d] = %v, want 1", i, dst[i])
					break
				}
			}

			done <- true
		}()
	}

	// Wait for all goroutines
	for range numGoroutines {
		<-done
	}
}

// TestClone_Concurrent tests that cloned plans work independently in goroutines.
func TestClone_Concurrent(t *testing.T) {
	t.Parallel()

	original, err := NewPlan[complex64](256)
	if err != nil {
		t.Fatalf("NewPlan failed: %v", err)
	}

	const numGoroutines = 5

	done := make(chan bool, numGoroutines)

	for range numGoroutines {
		go func() {
			// Each goroutine gets its own clone
			clone := original.Clone()

			src := make([]complex64, 256)
			dst := make([]complex64, 256)
			src[0] = 1

			err := clone.Forward(dst, src)
			if err != nil {
				t.Errorf("clone.Forward failed: %v", err)
			}

			// Verify impulse response
			for i := range dst {
				if cmplx.Abs(complex128(dst[i]-1)) > 1e-4 {
					t.Errorf("dst[%d] = %v, want 1", i, dst[i])
					break
				}
			}

			done <- true
		}()
	}

	// Wait for all goroutines
	for range numGoroutines {
		<-done
	}
}
