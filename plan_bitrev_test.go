package algofft

import (
	"testing"
)

// TestPlanBitReversal_PowerOfTwo verifies that non-pooled power-of-two plans
// precompute the bit-reversal table so the strided DIT fast path is reachable.
func TestPlanBitReversal_PowerOfTwo(t *testing.T) {
	t.Parallel()

	for _, n := range []int{2, 4, 8, 64, 1024} {
		plan, err := NewPlan32(n)
		if err != nil {
			t.Fatalf("NewPlan32(%d) failed: %v", n, err)
		}

		if plan.bitrev == nil {
			t.Fatalf("NewPlan32(%d): bitrev not precomputed", n)
		}

		if len(plan.bitrev) != n {
			t.Fatalf("NewPlan32(%d): len(bitrev) = %d, want %d", n, len(plan.bitrev), n)
		}

		if !isRadix2BitRev(plan.bitrev, n) {
			t.Fatalf("NewPlan32(%d): bitrev is not a radix-2 table: %v", n, plan.bitrev)
		}
	}
}

// TestPlanBitReversal_NonPowerOfTwo verifies that non-power-of-two plans leave
// the table nil, which disables the radix-2 strided fast path.
func TestPlanBitReversal_NonPowerOfTwo(t *testing.T) {
	t.Parallel()

	for _, n := range []int{6, 12, 60, 100} {
		plan, err := NewPlan32(n)
		if err != nil {
			t.Fatalf("NewPlan32(%d) failed: %v", n, err)
		}

		if plan.bitrev != nil {
			t.Fatalf("NewPlan32(%d): expected nil bitrev for non-power-of-two size", n)
		}
	}
}

// TestPlanStrided_FastPathMatchesContiguous cross-checks the strided DIT fast
// path (active now that bitrev is precomputed) against contiguous transforms.
func TestPlanStrided_FastPathMatchesContiguous(t *testing.T) {
	t.Parallel()

	for _, n := range []int{8, 64, 256, 1024} {
		checkStridedMatchesContiguous(t, n)
	}
}

func checkStridedMatchesContiguous(t *testing.T, n int) {
	t.Helper()

	const stride = 3

	plan, err := NewPlan32(n)
	if err != nil {
		t.Fatalf("NewPlan32(%d) failed: %v", n, err)
	}

	src := make([]complex64, n*stride)
	for i := range n {
		src[i*stride] = complex(float32(i%17)-8, float32((i*5)%13)-6)
	}

	contig := make([]complex64, n)
	for i := range n {
		contig[i] = src[i*stride]
	}

	want := make([]complex64, n)

	err = plan.Forward(want, contig)
	if err != nil {
		t.Fatalf("Forward(%d) failed: %v", n, err)
	}

	dst := make([]complex64, n*stride)

	err = plan.ForwardStrided(dst, src, stride)
	if err != nil {
		t.Fatalf("ForwardStrided(%d) failed: %v", n, err)
	}

	for i := range n {
		assertApproxComplex64f(t, dst[i*stride], want[i], 1e-3, "n=%d forward[%d]", n, i)
	}

	wantInv := make([]complex64, n)

	err = plan.Inverse(wantInv, contig)
	if err != nil {
		t.Fatalf("Inverse(%d) failed: %v", n, err)
	}

	dstInv := make([]complex64, n*stride)

	err = plan.InverseStrided(dstInv, src, stride)
	if err != nil {
		t.Fatalf("InverseStrided(%d) failed: %v", n, err)
	}

	for i := range n {
		assertApproxComplex64f(t, dstInv[i*stride], wantInv[i], 1e-3, "n=%d inverse[%d]", n, i)
	}
}
