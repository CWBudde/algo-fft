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
