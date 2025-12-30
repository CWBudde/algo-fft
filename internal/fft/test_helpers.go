package fft

import (
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

const (
	testTol64  = 1e-4
	testTol128 = 1e-10
)

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

func assertComplex64SliceClose(t *testing.T, got, want []complex64, n int) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}

	for i := range got {
		if cmplx.Abs(complex128(got[i]-want[i])) > testTol64 {
			t.Fatalf("n=%d index=%d got=%v want=%v", n, i, got[i], want[i])
		}
	}
}

func assertComplex128SliceClose(t *testing.T, got, want []complex128, n int) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want))
	}

	for i := range got {
		if cmplx.Abs(got[i]-want[i]) > testTol128 {
			t.Fatalf("n=%d index=%d got=%v want=%v", n, i, got[i], want[i])
		}
	}
}
