package algofft

import (
	"errors"
	"testing"
)

func TestPlanForwardStrided_Complex64(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(4)
	if err != nil {
		t.Fatalf("NewPlan32 failed: %v", err)
	}

	src := make([]complex64, 16)
	for i := range src {
		src[i] = complex(float32(i+1), float32(i)*0.25)
	}

	srcCopy := append([]complex64(nil), src...)

	dst := make([]complex64, len(src))
	stride := 4
	col := 2

	contig := make([]complex64, plan.Len())
	for i := range plan.Len() {
		contig[i] = src[col+i*stride]
	}

	want := make([]complex64, plan.Len())

	err = plan.Forward(want, contig)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	err = plan.ForwardStrided(dst[col:], src[col:], stride)
	if err != nil {
		t.Fatalf("ForwardStrided failed: %v", err)
	}

	for i := range plan.Len() {
		assertApproxComplex64f(t, dst[col+i*stride], want[i], 1e-4, "col[%d]", i)
	}

	for i := range src {
		if src[i] != srcCopy[i] {
			t.Fatalf("src mutated at %d: got %v want %v", i, src[i], srcCopy[i])
		}
	}
}

func TestPlanInverseStrided_Complex128(t *testing.T) {
	t.Parallel()

	const n = 8

	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("NewPlan64 failed: %v", err)
	}

	time := make([]complex128, n)
	for i := range time {
		time[i] = complex(float64(i+1), float64(i)*0.1)
	}

	freq := make([]complex128, n)

	err = plan.Forward(freq, time)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	stride := 5
	total := 1 + (n-1)*stride
	src := make([]complex128, total)
	dst := make([]complex128, total)

	for i := range n {
		src[i*stride] = freq[i]
	}

	err = plan.InverseStrided(dst, src, stride)
	if err != nil {
		t.Fatalf("InverseStrided failed: %v", err)
	}

	for i := range n {
		assertApproxComplex128f(t, dst[i*stride], time[i], "idx[%d]", i)
	}
}

func TestPlanStrided_Errors(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan32(4)
	if err != nil {
		t.Fatalf("NewPlan32 failed: %v", err)
	}

	data := make([]complex64, 4)

	err = plan.ForwardStrided(data, data, 0)
	if !errors.Is(err, ErrInvalidStride) {
		t.Fatalf("expected ErrInvalidStride, got %v", err)
	}

	short := make([]complex64, 5)

	err = plan.ForwardStrided(short, short, 2)
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("expected ErrLengthMismatch, got %v", err)
	}
}

// TestPlanForwardStridedRecursive_Complex64 guards against the strided DIT fast
// path running on a recursive plan. Recursive plans store recursive-layout
// twiddles in p.twiddle, which are incompatible with ForwardStridedDIT; if the
// bit-reversal table is populated for them, the strided transform silently
// produces the wrong spectrum. See planBitReversal.
//
// The mismatch only surfaces once the recursive decomposition twiddle table
// diverges from the standard DIT layout, which happens at multi-level sizes
// (e.g. n=1024, where the recursive table is twice as long), so the size here
// is deliberately large.
func TestPlanForwardStridedRecursive_Complex64(t *testing.T) {
	t.Parallel()

	const n = 1024

	plan, err := NewPlanWithOptions[complex64](n, PlanOptions{Strategy: KernelRecursive})
	if err != nil {
		t.Fatalf("NewPlanWithOptions(Recursive) failed: %v", err)
	}

	if plan.kernelStrategy != KernelRecursive {
		t.Fatalf("expected KernelRecursive strategy, got %v", plan.kernelStrategy)
	}

	stride := 3
	col := 1
	total := col + 1 + (n-1)*stride

	src := make([]complex64, total)
	contig := make([]complex64, n)

	for i := range n {
		v := complex(float32(i+1), float32(i)*0.25)
		src[col+i*stride] = v
		contig[i] = v
	}

	// The strided transform must agree with the plan's own contiguous transform
	// of the gathered data. Comparing against the plan itself keeps the check
	// robust to the recursive kernel's numerics while still catching the
	// wrong-twiddle fast path, which diverges by many orders of magnitude.
	want := make([]complex64, n)

	err = plan.Forward(want, contig)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	dst := make([]complex64, total)

	err = plan.ForwardStrided(dst[col:], src[col:], stride)
	if err != nil {
		t.Fatalf("ForwardStrided failed: %v", err)
	}

	for i := range n {
		got := dst[col+i*stride]

		diff := cabsf32(got - want[i])
		if float64(diff/(cabsf32(want[i])+1)) > 1e-4 {
			t.Fatalf("recursive strided[%d]: got %v want %v (rel diff too large)", i, got, want[i])
		}
	}
}
