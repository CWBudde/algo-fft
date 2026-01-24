package fft

import (
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

// TestSelectKernels tests the generic kernel selection.
func TestSelectKernels(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	// Test complex64 kernel selection
	kernels64 := SelectKernels[complex64](features)
	if kernels64.Forward == nil {
		t.Error("SelectKernels[complex64] returned nil Forward kernel")
	}

	if kernels64.Inverse == nil {
		t.Error("SelectKernels[complex64] returned nil Inverse kernel")
	}

	// Test complex128 kernel selection
	kernels128 := SelectKernels[complex128](features)
	if kernels128.Forward == nil {
		t.Error("SelectKernels[complex128] returned nil Forward kernel")
	}

	if kernels128.Inverse == nil {
		t.Error("SelectKernels[complex128] returned nil Inverse kernel")
	}
}

// TestSelectKernelsWithStrategy tests kernel selection with specific strategies.
func TestSelectKernelsWithStrategy(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	strategies := []KernelStrategy{
		KernelAuto,
		KernelDIT,
		KernelStockham,
		KernelSixStep,
		KernelEightStep,
	}

	strategyNames := []string{
		"Auto",
		"DIT",
		"Stockham",
		"SixStep",
		"EightStep",
	}

	for i, strategy := range strategies {
		t.Run(strategyNames[i], func(t *testing.T) {
			t.Parallel()

			// Test complex64
			kernels64 := SelectKernelsWithStrategy[complex64](features, strategy)
			if kernels64.Forward == nil {
				t.Errorf("SelectKernelsWithStrategy[complex64](%v) returned nil Forward kernel", strategy)
			}

			if kernels64.Inverse == nil {
				t.Errorf("SelectKernelsWithStrategy[complex64](%v) returned nil Inverse kernel", strategy)
			}

			// Test complex128
			kernels128 := SelectKernelsWithStrategy[complex128](features, strategy)
			if kernels128.Forward == nil {
				t.Errorf("SelectKernelsWithStrategy[complex128](%v) returned nil Forward kernel", strategy)
			}

			if kernels128.Inverse == nil {
				t.Errorf("SelectKernelsWithStrategy[complex128](%v) returned nil Inverse kernel", strategy)
			}
		})
	}
}

// TestStubKernel tests the stub kernel fallback.
func TestStubKernel(t *testing.T) {
	t.Parallel()

	dst := make([]complex64, 8)
	src := make([]complex64, 8)
	twiddle := make([]complex64, 8)
	scratch := make([]complex64, 8)

	// Stub kernel should return false (indicating it didn't handle the transform)
	handled := stubKernel(dst, src, twiddle, scratch)
	if handled {
		t.Error("stubKernel should return false")
	}
}

// TestDetectFeatures tests CPU feature detection.
func TestDetectFeatures(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	// Architecture should always be set
	if features.Architecture == "" {
		t.Error("Architecture should be set")
	}

	// Architecture should match runtime.GOARCH
	if features.Architecture != runtime.GOARCH {
		t.Errorf("Architecture mismatch: got %q, want %q", features.Architecture, runtime.GOARCH)
	}

	// On amd64, SSE2 should always be available
	if runtime.GOARCH == "amd64" && !features.HasSSE2 {
		t.Error("SSE2 should be available on amd64")
	}

	// On arm64, NEON should always be available
	if runtime.GOARCH == "arm64" && !features.HasNEON {
		t.Error("NEON should be available on arm64")
	}

	t.Logf("Detected features: %+v", features)
}

// TestKernelSelectionWithForcedFeatures tests kernel selection with mocked CPU features.
func TestKernelSelectionWithForcedFeatures(t *testing.T) {
	t.Parallel()

	// Test SSE2-only system (no AVX2)
	t.Run("SSE2Only", func(t *testing.T) {
		t.Parallel()

		features := cpu.Features{
			HasSSE2:      true,
			Architecture: "amd64",
		}

		kernels := SelectKernels[complex64](features)
		if kernels.Forward == nil || kernels.Inverse == nil {
			t.Error("Should have valid kernels even with SSE2 only")
		}
	})

	// Test AVX2 system
	t.Run("AVX2System", func(t *testing.T) {
		t.Parallel()

		features := cpu.Features{
			HasSSE2:      true,
			HasSSE3:      true,
			HasSSSE3:     true,
			HasSSE41:     true,
			HasAVX:       true,
			HasAVX2:      true,
			Architecture: "amd64",
		}

		kernels := SelectKernels[complex64](features)
		if kernels.Forward == nil || kernels.Inverse == nil {
			t.Error("Should have valid kernels with AVX2")
		}
	})

	// Test ForceGeneric flag disables SIMD
	t.Run("ForceGeneric", func(t *testing.T) {
		t.Parallel()

		features := cpu.Features{
			HasAVX2:      true,
			ForceGeneric: true,
			Architecture: "amd64",
		}

		// Kernels should still be selected (ForceGeneric is a hint, not a hard requirement)
		kernels := SelectKernels[complex64](features)
		if kernels.Forward == nil || kernels.Inverse == nil {
			t.Error("Should have valid kernels even with ForceGeneric")
		}
	})

	// Test ARM NEON system
	t.Run("NEONSystem", func(t *testing.T) {
		t.Parallel()

		features := cpu.Features{
			HasNEON:      true,
			Architecture: "arm64",
		}

		kernels := SelectKernels[complex64](features)
		if kernels.Forward == nil || kernels.Inverse == nil {
			t.Error("Should have valid kernels with NEON")
		}
	})
}

// TestKernelsFunctional tests that selected kernels actually work.
func TestKernelsFunctional_Complex64(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex64](features)

	n := 8
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	// Test forward kernel
	src := make([]complex64, n)
	dst := make([]complex64, n)
	src[0] = 1 // impulse

	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Fatal("Forward kernel returned false")
	}

	// Impulse should transform to all ones (DC component)
	for i := range dst {
		if real(dst[i]) < 0.99 || real(dst[i]) > 1.01 {
			t.Errorf("dst[%d] = %v, expected ~1", i, dst[i])
		}

		if imag(dst[i]) < -0.01 || imag(dst[i]) > 0.01 {
			t.Errorf("dst[%d] = %v, expected imaginary part ~0", i, dst[i])
		}
	}

	// Test inverse kernel
	roundTrip := make([]complex64, n)
	if !kernels.Inverse(roundTrip, dst, twiddle, scratch) {
		t.Fatal("Inverse kernel returned false")
	}

	// Should get back original impulse
	if real(roundTrip[0]) < 0.99 || real(roundTrip[0]) > 1.01 {
		t.Errorf("roundTrip[0] = %v, expected ~1", roundTrip[0])
	}

	for i := 1; i < n; i++ {
		if real(roundTrip[i]) < -0.01 || real(roundTrip[i]) > 0.01 {
			t.Errorf("roundTrip[%d] = %v, expected ~0", i, roundTrip[i])
		}
	}
}

// TestKernelsFunctional_Complex128 tests complex128 kernels.
func TestKernelsFunctional_Complex128(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	kernels := SelectKernels[complex128](features)

	n := 16
	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)

	// Test forward kernel
	src := make([]complex128, n)
	dst := make([]complex128, n)
	src[0] = 1 // impulse

	if !kernels.Forward(dst, src, twiddle, scratch) {
		t.Fatal("Forward kernel returned false")
	}

	// Impulse should transform to all ones
	for i := range dst {
		if real(dst[i]) < 0.999 || real(dst[i]) > 1.001 {
			t.Errorf("dst[%d] = %v, expected ~1", i, dst[i])
		}
	}

	// Test inverse kernel
	roundTrip := make([]complex128, n)
	if !kernels.Inverse(roundTrip, dst, twiddle, scratch) {
		t.Fatal("Inverse kernel returned false")
	}

	// Should get back original
	if real(roundTrip[0]) < 0.999 || real(roundTrip[0]) > 1.001 {
		t.Errorf("roundTrip[0] = %v, expected ~1", roundTrip[0])
	}

	for i := 1; i < n; i++ {
		if abs128(roundTrip[i]) > 0.001 {
			t.Errorf("roundTrip[%d] = %v, expected ~0", i, roundTrip[i])
		}
	}
}

// TestAVX2KernelStrategyDispatch tests AVX2 strategy kernel dispatch functions.
func TestAVX2KernelStrategyDispatch(t *testing.T) {
	t.Parallel()

	t.Run("Complex64_DIT_Strategy", func(t *testing.T) {
		t.Parallel()

		n := 8
		ditCalled := false
		stockhamCalled := false

		// Mock DIT and Stockham kernels
		ditKernel := func(dst, src, twiddle, scratch []complex64) bool {
			ditCalled = true
			return true
		}

		stockhamKernel := func(dst, src, twiddle, scratch []complex64) bool {
			stockhamCalled = true
			return true
		}

		// Create AVX2 strategy kernel
		kernel := avx2KernelComplex64(KernelDIT, ditKernel, stockhamKernel)

		src := make([]complex64, n)
		dst := make([]complex64, n)
		twiddle := make([]complex64, n)
		scratch := make([]complex64, n)

		// Call kernel with DIT strategy
		handled := kernel(dst, src, twiddle, scratch)

		if !handled {
			t.Error("Kernel should have handled the transform")
		}

		if !ditCalled {
			t.Error("DIT kernel should have been called")
		}

		if stockhamCalled {
			t.Error("Stockham kernel should not have been called")
		}
	})

	t.Run("Complex64_Stockham_Strategy", func(t *testing.T) {
		t.Parallel()

		n := 1024
		ditCalled := false
		stockhamCalled := false

		ditKernel := func(dst, src, twiddle, scratch []complex64) bool {
			ditCalled = true
			return true
		}

		stockhamKernel := func(dst, src, twiddle, scratch []complex64) bool {
			stockhamCalled = true
			return true
		}

		kernel := avx2KernelComplex64(KernelStockham, ditKernel, stockhamKernel)

		src := make([]complex64, n)
		dst := make([]complex64, n)
		twiddle := make([]complex64, n)
		scratch := make([]complex64, n)

		handled := kernel(dst, src, twiddle, scratch)

		if !handled {
			t.Error("Kernel should have handled the transform")
		}

		if ditCalled {
			t.Error("DIT kernel should not have been called")
		}

		if !stockhamCalled {
			t.Error("Stockham kernel should have been called")
		}
	})

	t.Run("Complex128_DIT_Strategy", func(t *testing.T) {
		t.Parallel()

		n := 8
		ditCalled := false

		ditKernel := func(dst, src, twiddle, scratch []complex128) bool {
			ditCalled = true
			return true
		}

		stockhamKernel := func(dst, src, twiddle, scratch []complex128) bool {
			return true
		}

		kernel := avx2KernelComplex128(KernelDIT, ditKernel, stockhamKernel)

		src := make([]complex128, n)
		dst := make([]complex128, n)
		twiddle := make([]complex128, n)
		scratch := make([]complex128, n)

		handled := kernel(dst, src, twiddle, scratch)

		if !handled {
			t.Error("Kernel should have handled the transform")
		}

		if !ditCalled {
			t.Error("DIT kernel should have been called")
		}
	})

	t.Run("Complex128_Stockham_Strategy", func(t *testing.T) {
		t.Parallel()

		n := 1024
		stockhamCalled := false

		ditKernel := func(dst, src, twiddle, scratch []complex128) bool {
			return true
		}

		stockhamKernel := func(dst, src, twiddle, scratch []complex128) bool {
			stockhamCalled = true
			return true
		}

		kernel := avx2KernelComplex128(KernelStockham, ditKernel, stockhamKernel)

		src := make([]complex128, n)
		dst := make([]complex128, n)
		twiddle := make([]complex128, n)
		scratch := make([]complex128, n)

		handled := kernel(dst, src, twiddle, scratch)

		if !handled {
			t.Error("Kernel should have handled the transform")
		}

		if !stockhamCalled {
			t.Error("Stockham kernel should have been called")
		}
	})
}

// TestEstimatePlan tests plan estimation functionality
func TestEstimatePlan(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()
	wisdom := NewWisdom()

	testCases := []struct {
		name     string
		n        int
		strategy KernelStrategy
	}{
		{"Small_Auto", 16, KernelAuto},
		{"Medium_DIT", 256, KernelDIT},
		{"Large_Stockham", 4096, KernelStockham},
		{"PowerOf2_Auto", 1024, KernelAuto},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Test complex64
			est64 := EstimatePlan[complex64](tc.n, features, wisdom, tc.strategy)
			if est64.Algorithm == "" && est64.Strategy == 0 {
				t.Errorf("complex64: empty plan estimate for n=%d", tc.n)
			}

			// Test complex128
			est128 := EstimatePlan[complex128](tc.n, features, wisdom, tc.strategy)
			if est128.Algorithm == "" && est128.Strategy == 0 {
				t.Errorf("complex128: empty plan estimate for n=%d", tc.n)
			}
		})
	}
}

// TestHasCodelet tests codelet availability checking
func TestHasCodelet(t *testing.T) {
	t.Parallel()

	features := cpu.DetectFeatures()

	// Common sizes that should have codelets
	sizes := []int{4, 8, 16, 32, 64}

	for _, n := range sizes {
		has64 := HasCodelet[complex64](n, features)
		has128 := HasCodelet[complex128](n, features)

		// Just verify the function runs without error
		// Actual availability depends on build configuration
		t.Logf("n=%d: complex64=%v, complex128=%v", n, has64, has128)
	}
}

// TestConjugatePackedTwiddles tests packed twiddle conjugation
func TestConjugatePackedTwiddles(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		n := 16
		twiddle := ComputeTwiddleFactors[complex64](n)
		packed := ComputePackedTwiddles[complex64](n, 4, twiddle)

		if packed == nil {
			t.Fatal("ComputePackedTwiddles returned nil")
		}

		conjugated := ConjugatePackedTwiddles(packed)

		if conjugated == nil {
			t.Fatal("ConjugatePackedTwiddles returned nil")
		}

		// Verify conjugation
		if len(conjugated.Values) != len(packed.Values) {
			t.Fatalf("Length mismatch: got %d, want %d", len(conjugated.Values), len(packed.Values))
		}

		for i, v := range packed.Values {
			expected := complex(real(v), -imag(v))
			if conjugated.Values[i] != expected {
				t.Errorf("index %d: got %v, want %v", i, conjugated.Values[i], expected)
			}
		}
	})

	t.Run("complex128", func(t *testing.T) {
		n := 16
		twiddle := ComputeTwiddleFactors[complex128](n)
		packed := ComputePackedTwiddles[complex128](n, 4, twiddle)

		if packed == nil {
			t.Fatal("ComputePackedTwiddles returned nil")
		}

		conjugated := ConjugatePackedTwiddles(packed)

		if conjugated == nil {
			t.Fatal("ConjugatePackedTwiddles returned nil")
		}

		for i, v := range packed.Values {
			expected := complex(real(v), -imag(v))
			if conjugated.Values[i] != expected {
				t.Errorf("index %d: got %v, want %v", i, conjugated.Values[i], expected)
			}
		}
	})
}

// TestComputeSquareTransposePairs tests transpose pair computation
func TestComputeSquareTransposePairs(t *testing.T) {
	t.Parallel()

	sizes := []int{2, 4, 8, 16}

	for _, n := range sizes {
		pairs := ComputeSquareTransposePairs(n)

		// For n×n matrix, we expect at most (n²-n)/2 swaps
		maxPairs := (n*n - n) / 2
		if len(pairs) > maxPairs {
			t.Errorf("n=%d: too many pairs: got %d, max %d", n, len(pairs), maxPairs)
		}

		// Verify no duplicate pairs
		seen := make(map[int]bool)
		for _, pair := range pairs {
			if pair.I == pair.J {
				t.Errorf("n=%d: self-swap at index %d", n, pair.I)
			}
			if seen[pair.I] && seen[pair.J] {
				t.Errorf("n=%d: duplicate pair (%d, %d)", n, pair.I, pair.J)
			}
			seen[pair.I] = true
			seen[pair.J] = true
		}
	}
}

// TestApplyTransposePairs tests transpose application
func TestApplyTransposePairs(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		n := 4
		data := make([]complex64, n*n)
		for i := range data {
			data[i] = complex(float32(i), 0)
		}

		// Create a copy for reference
		original := make([]complex64, len(data))
		copy(original, data)

		pairs := ComputeSquareTransposePairs(n)
		ApplyTransposePairs(data, pairs)

		// Verify transpose: data[i*n+j] should equal original[j*n+i]
		for i := range n {
			for j := range n {
				idx := i*n + j
				transIdx := j*n + i
				if data[idx] != original[transIdx] {
					t.Errorf("Transpose mismatch at (%d,%d): got %v, want %v",
						i, j, data[idx], original[transIdx])
				}
			}
		}
	})

	t.Run("complex128", func(t *testing.T) {
		n := 4
		data := make([]complex128, n*n)
		for i := range data {
			data[i] = complex(float64(i), 0)
		}

		original := make([]complex128, len(data))
		copy(original, data)

		pairs := ComputeSquareTransposePairs(n)
		ApplyTransposePairs(data, pairs)

		for i := range n {
			for j := range n {
				idx := i*n + j
				transIdx := j*n + i
				if data[idx] != original[transIdx] {
					t.Errorf("Transpose mismatch at (%d,%d): got %v, want %v",
						i, j, data[idx], original[transIdx])
				}
			}
		}
	})
}
