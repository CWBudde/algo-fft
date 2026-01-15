package algofft

import (
	"math/cmplx"
	"testing"
)

// Shared test helper functions used across multiple test files

func assertApproxComplex128Tolf(t *testing.T, got, want complex128, tol float64, format string, args ...any) {
	t.Helper()

	if cmplx.Abs(got-want) > tol {
		t.Fatalf(format+": got %v want %v (diff=%v)", append(args, got, want, cmplx.Abs(got-want))...)
	}
}
