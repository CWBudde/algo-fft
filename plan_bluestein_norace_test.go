//go:build !race

// The zero-allocation assertions here are excluded from the race build for the
// same reason as the Rader and radix-7/11 ones: they exercise the mixed-radix
// engine, whose pooled scratch does not survive race instrumentation, so
// AllocsPerRun reports a handful of allocations that the normal build does not
// make. The power-of-two padded paths are covered in the race build by the
// untagged tests in convolve_fastlen_test.go.

package algofft

import "testing"

// TestBluestein_MixedRadixPadZeroAlloc checks that the transform hot path stays
// allocation-free when the shape-aware pad model picks a mixed-radix padded
// sub-FFT instead of a power of two. That path runs through the mixed-radix
// engine, whose scratch pools are separate from the DIT sub-FFT's, so the
// power-of-two case (1009 -> 2048) alone would not cover it.
func TestBluestein_MixedRadixPadZeroAlloc(t *testing.T) {
	tests := []struct{ n, pad int }{
		{n: 1009, pad: 2048},  // power-of-two pad: control
		{n: 2531, pad: 6144},  // 2^11·3
		{n: 3079, pad: 7680},  // 2^9·15
		{n: 4099, pad: 12288}, // 2^12·3
	}

	for _, tt := range tests {
		if got := bluesteinPadSize(tt.n); got != tt.pad {
			t.Fatalf("test premise broken: bluesteinPadSize(%d) = %d, want %d", tt.n, got, tt.pad)
		}

		plan, err := NewPlan[complex64](tt.n)
		if err != nil {
			t.Fatalf("NewPlan[complex64](%d) failed: %v", tt.n, err)
		}

		if plan.Algorithm() != "bluestein" {
			t.Fatalf("test premise broken: n=%d uses %q, want bluestein", tt.n, plan.Algorithm())
		}

		src := randomComplex64(tt.n, 5)
		dst := make([]complex64, tt.n)

		// Warm up pooled scratch and the mixed-radix schedule pool.
		_ = plan.Forward(dst, src)
		_ = plan.Inverse(dst, src)

		if allocs := testing.AllocsPerRun(100, func() {
			_ = plan.Forward(dst, src)
			_ = plan.Inverse(dst, src)
		}); allocs != 0 {
			t.Errorf("n=%d (pad %d): transforms allocate %v times per run, want 0", tt.n, tt.pad, allocs)
		}
	}
}

// TestConvolver_ZeroAllocSteadyStateMixedRadixPad is the convolution
// counterpart: convLen 2531 pads to 3072 = 2^10·3, so the steady state runs
// through the mixed-radix engine rather than the DIT kernels. The power-of-two
// pad is covered by TestConvolver_ZeroAllocSteadyStatePadded.
func TestConvolver_ZeroAllocSteadyStateMixedRadixPad(t *testing.T) {
	const lenA, lenB = 1266, 1266 // convLen 2531 (prime), padded to 3072

	if got := fastConvolutionLength(lenA + lenB - 1); got != 3072 {
		t.Fatalf("test premise broken: fastConvolutionLength(%d) = %d, want 3072", lenA+lenB-1, got)
	}

	conv, err := NewConvolver[complex64](lenA, lenB)
	if err != nil {
		t.Fatalf("NewConvolver failed: %v", err)
	}

	a := make([]complex64, lenA)
	b := make([]complex64, lenB)
	dst := make([]complex64, conv.Len())

	a[0], b[0] = 1, 1

	if err := conv.Convolve(dst, a, b); err != nil {
		t.Fatalf("warm-up Convolve failed: %v", err)
	}

	allocs := testing.AllocsPerRun(10, func() {
		if err := conv.Convolve(dst, a, b); err != nil {
			t.Errorf("Convolve failed: %v", err)
		}
	})

	if allocs != 0 {
		t.Errorf("Convolver.Convolve allocates %.1f per call, want 0", allocs)
	}
}
