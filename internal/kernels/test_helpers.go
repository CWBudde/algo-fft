package kernels

import (
	"fmt"
	"math/cmplx"
	"math/rand/v2"
	"os"
	"testing"
)

// randomComplex64 generates deterministic random complex64 test data.
func randomComplex64(n int, seed uint64) []complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0x5A5A5A5A)) //nolint:gosec // Deterministic test data

	out := make([]complex64, n)
	for i := range out {
		re := float32(rng.Float64()*2 - 1)
		im := float32(rng.Float64()*2 - 1)
		out[i] = complex(re, im)
	}

	return out
}

// randomComplex128 generates deterministic random complex128 test data.
func randomComplex128(n int, seed uint64) []complex128 {
	rng := rand.New(rand.NewPCG(seed, seed^0xA5A5A5A5)) //nolint:gosec // Deterministic test data

	out := make([]complex128, n)
	for i := range out {
		re := rng.Float64()*2 - 1
		im := rng.Float64()*2 - 1
		out[i] = complex(re, im)
	}

	return out
}

// assertComplex64SliceClose checks that got and want are close within a default tolerance.
// The tolerance is scaled by the transform size n.
func assertComplex64SliceClose(t *testing.T, got, want []complex64, n int) {
	t.Helper()

	// Tolerance scales with size due to accumulated floating-point errors
	tol := 1e-4 * float64(n) / 8
	if tol < 1e-4 {
		tol = 1e-4
	}

	if tol > 0.1 {
		tol = 0.1
	}

	assertComplex64Close(t, got, want, tol)
}

// assertComplex64Close checks that got and want are close within tolerance.
func assertComplex64Close(t *testing.T, got, want []complex64, tol float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}

	for i := range got {
		diff := cmplx.Abs(complex128(got[i] - want[i]))
		if diff > tol {
			t.Errorf("index=%d got=%v want=%v diff=%g tol=%g", i, got[i], want[i], diff, tol)
		}
	}
}

// assertComplex128Close checks that got and want are close within tolerance.
func assertComplex128Close(t *testing.T, got, want []complex128, tol float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}

	for i := range got {
		diff := cmplx.Abs(got[i] - want[i])
		if diff > tol {
			t.Errorf("index=%d got=%v want=%v diff=%g tol=%g", i, got[i], want[i], diff, tol)
		}
	}
}

// testName creates a descriptive test name.
func testName(op string, size int) string {
	return fmt.Sprintf("%s_size_%d", op, size)
}

// benchName creates a descriptive benchmark name.
func benchName(op string, size int) string {
	return fmt.Sprintf("%s_%d", op, size)
}

func skipNaiveReferenceIfSlow(t *testing.T, n int) {
	t.Helper()

	if testing.Short() {
		t.Skip("skipping naive reference in -short mode")
	}

	if os.Getenv("ALGOFFT_QEMU") == "1" && n > 4096 {
		t.Skip("skipping naive reference on QEMU for large sizes; run without ALGOFFT_QEMU for full coverage")
	}
}
