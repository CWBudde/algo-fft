package fft

import (
	"testing"
)

// TestSelectKernels tests the generic kernel selection.
func TestSelectKernels(t *testing.T) {
	t.Parallel()

	features := DetectFeatures()

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

	features := DetectFeatures()

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
	bitrev := make([]int, 8)

	// Stub kernel should return false (indicating it didn't handle the transform)
	handled := stubKernel(dst, src, twiddle, scratch, bitrev)
	if handled {
		t.Error("stubKernel should return false")
	}
}

// TestDetectFeatures tests CPU feature detection.
func TestDetectFeatures(t *testing.T) {
	t.Parallel()

	features := DetectFeatures()

	// At a minimum, the generic fallback should always be available
	// We can't test for specific SIMD features as it depends on the test machine
	t.Logf("Detected features: %+v", features)
}

// TestKernelsFunctional tests that selected kernels actually work.
func TestKernelsFunctional_Complex64(t *testing.T) {
	t.Parallel()

	features := DetectFeatures()
	kernels := SelectKernels[complex64](features)

	n := 8
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)

	// Test forward kernel
	src := make([]complex64, n)
	dst := make([]complex64, n)
	src[0] = 1 // impulse

	if !kernels.Forward(dst, src, twiddle, scratch, bitrev) {
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
	if !kernels.Inverse(roundTrip, dst, twiddle, scratch, bitrev) {
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

	features := DetectFeatures()
	kernels := SelectKernels[complex128](features)

	n := 16
	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)

	// Test forward kernel
	src := make([]complex128, n)
	dst := make([]complex128, n)
	src[0] = 1 // impulse

	if !kernels.Forward(dst, src, twiddle, scratch, bitrev) {
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
	if !kernels.Inverse(roundTrip, dst, twiddle, scratch, bitrev) {
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
