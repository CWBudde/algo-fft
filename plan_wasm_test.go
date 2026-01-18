//go:build js && wasm

package algofft

import (
	"math/cmplx"
	"testing"
)

func TestWASMRoundTripComplex64(t *testing.T) {
	plan, err := NewPlan32(8)
	if err != nil {
		t.Fatalf("NewPlan32 failed: %v", err)
	}

	src := []complex64{
		1 + 2i,
		-3 + 4i,
		5 - 6i,
		-7 - 8i,
		9 + 0i,
		-1 + 1i,
		2 - 2i,
		-3 + 3i,
	}

	tmp := make([]complex64, len(src))
	dst := make([]complex64, len(src))

	err = plan.Forward(tmp, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	err = plan.Inverse(dst, tmp)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	for i, want := range src {
		assertApproxComplex64Tol(t, dst[i], want, 1e-4, "round-trip mismatch at %d", i)
	}
}

func TestWASMRoundTripComplex128(t *testing.T) {
	plan, err := NewPlan64(8)
	if err != nil {
		t.Fatalf("NewPlan64 failed: %v", err)
	}

	src := []complex128{
		1 + 2i,
		-3 + 4i,
		5 - 6i,
		-7 - 8i,
		9 + 0i,
		-1 + 1i,
		2 - 2i,
		-3 + 3i,
	}

	tmp := make([]complex128, len(src))
	dst := make([]complex128, len(src))

	err = plan.Forward(tmp, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	err = plan.Inverse(dst, tmp)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	for i, want := range src {
		assertApproxComplex128Tolf(t, dst[i], want, 1e-10, "round-trip mismatch at %d", i)
	}
}

func assertApproxComplex64Tol(t *testing.T, got, want complex64, tol float64, format string, args ...any) {
	t.Helper()
	if cmplx.Abs(complex128(got-want)) > tol {
		t.Fatalf(format+": got %v want %v (diff=%v)", append(args, got, want, cmplx.Abs(complex128(got-want)))...)
	}
}
