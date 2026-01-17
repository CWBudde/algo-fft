//go:build amd64 && asm && !purego

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

func TestDebugAVX2Size8Radix4(t *testing.T) {
	const n = 8

	// Simple test input
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i+1), float32(i+1)*0.5)
	}

	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeBitReversalIndicesRadix4Then2(n)

	t.Logf("=== Input ===")
	for i, v := range src {
		t.Logf("src[%d] = %v", i, v)
	}
	t.Logf("bitrev = %v", bitrev)
	t.Logf("twiddle = %v", twiddle[:4])

	// Test forward with Go reference
	goFwd := make([]complex64, n)
	goScratch := make([]complex64, n)
	forwardDIT8Radix4Complex64(goFwd, src, twiddle, goScratch)

	// Test forward with AVX2
	avxFwd := make([]complex64, n)
	avxScratch := make([]complex64, n)
	amd64.ForwardAVX2Size8Radix4Complex64Asm(avxFwd, src, twiddle, avxScratch)

	t.Logf("\n=== Forward Transform ===")
	t.Logf("Go:   %v", goFwd)
	t.Logf("AVX2: %v", avxFwd)

	// Check forward match
	fwdMatch := true
	for i := 0; i < n; i++ {
		diff := abs64(goFwd[i] - avxFwd[i])
		if diff > 1e-5 {
			t.Logf("Forward mismatch at [%d]: Go=%v AVX2=%v diff=%e", i, goFwd[i], avxFwd[i], diff)
			fwdMatch = false
		}
	}
	t.Logf("Forward match: %v", fwdMatch)

	// Test inverse with Go reference (using Go forward result)
	goInv := make([]complex64, n)
	inverseDIT8Radix4Complex64(goInv, goFwd, twiddle, goScratch)

	// Test inverse with AVX2 (using Go forward result for fair comparison)
	avxInv := make([]complex64, n)
	amd64.InverseAVX2Size8Radix4Complex64Asm(avxInv, goFwd, twiddle, avxScratch)

	t.Logf("\n=== Inverse Transform (from same forward result) ===")
	t.Logf("Go:   %v", goInv)
	t.Logf("AVX2: %v", avxInv)

	// Check inverse match
	t.Logf("\n=== Per-element inverse differences ===")
	invMatch := true
	for i := 0; i < n; i++ {
		diff := abs64(goInv[i] - avxInv[i])
		t.Logf("[%d] Go=%v AVX2=%v diff=%e", i, goInv[i], avxInv[i], diff)
		if diff > 1e-5 {
			invMatch = false
		}
	}
	t.Logf("Inverse match: %v", invMatch)

	// Check roundtrip
	t.Logf("\n=== Round-trip (should match original src) ===")
	t.Logf("Original: %v", src)
	t.Logf("Go:       %v", goInv)
	t.Logf("AVX2:     %v", avxInv)

	goRoundTrip := true
	avxRoundTrip := true
	for i := 0; i < n; i++ {
		if abs64(src[i]-goInv[i]) > 1e-5 {
			goRoundTrip = false
		}
		if abs64(src[i]-avxInv[i]) > 1e-5 {
			avxRoundTrip = false
		}
	}
	t.Logf("Go round-trip match: %v", goRoundTrip)
	t.Logf("AVX2 round-trip match: %v", avxRoundTrip)

	if !fwdMatch {
		t.Error("Forward transform mismatch between Go and AVX2")
	}
	if !invMatch {
		t.Error("Inverse transform mismatch between Go and AVX2")
	}
	if !avxRoundTrip {
		t.Error("AVX2 round-trip failed")
	}
}

func abs64(c complex64) float32 {
	r, i := real(c), imag(c)
	return float32(r*r + i*i)
}
