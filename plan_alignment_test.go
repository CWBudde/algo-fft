package algofft

import (
	"testing"
	"unsafe"

	mem "github.com/MeKo-Christian/algo-fft/internal/memory"
)

func TestPlanAlignmentComplex64(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex64](1024)
	if err != nil {
		t.Fatalf("NewPlan(1024) returned error: %v", err)
	}

	checkAlignment(t, unsafe.Pointer(&plan.twiddle[0]))
	checkAlignment(t, unsafe.Pointer(&plan.scratch[0]))
}

func TestPlanAlignmentComplex128(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanT[complex128](1024)
	if err != nil {
		t.Fatalf("NewPlan(1024) returned error: %v", err)
	}

	checkAlignment(t, unsafe.Pointer(&plan.twiddle[0]))
	checkAlignment(t, unsafe.Pointer(&plan.scratch[0]))
}

func checkAlignment(t *testing.T, ptr unsafe.Pointer) {
	t.Helper()

	addr := uintptr(ptr)
	if addr%mem.AlignmentBytes != 0 {
		t.Fatalf("pointer alignment = %d, want multiple of %d", addr%mem.AlignmentBytes, mem.AlignmentBytes)
	}
}
