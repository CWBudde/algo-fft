//go:build amd64

package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func TestAVX2Size8192Mixed24(t *testing.T) {
	if !cpu.DetectFeatures().HasAVX2 {
		t.Skip("AVX2 not supported")
	}

	// Get the kernel that uses the size-specific dispatch
	kernel := avx2SizeSpecificOrGenericDITComplex64(KernelDIT)

	n := 8192
	src := generateRandomComplex64(n, 12345)
	twiddle, bitrev, scratch := prepareFFTData(n)
	dst := make([]complex64, n)

	// Test Forward
	ok := kernel(dst, src, twiddle, scratch, bitrev)
	if !ok {
		t.Fatal("AVX2 Size 8192 Forward kernel returned false")
	}

	// Verify correctness against Pure Go (which we trust as reference for now)
	goForward := forwardDITComplex64 // Use the internal pure-go wrapper
	dstGo := make([]complex64, n)
	if !goForward(dstGo, src, twiddle, scratch, bitrev) {
		t.Fatal("Pure Go Forward failed")
	}

	if !complexSliceEqual(dst, dstGo, 1e-4) {
		t.Error("AVX2 Size 8192 Forward result mismatches Pure Go")
		reportDifferences(t, dst, dstGo, 1e-4)
	}

	// Test Inverse
	kernelInv := avx2SizeSpecificOrGenericDITInverseComplex64(KernelDIT)
	
	// Inverse input (use result of forward)
	dstInv := make([]complex64, n)
	okInv := kernelInv(dstInv, dst, twiddle, scratch, bitrev)
	if !okInv {
		t.Fatal("AVX2 Size 8192 Inverse kernel returned false")
	}

	// Verify Round Trip
	if !complexSliceEqual(dstInv, src, 1e-4) {
		t.Error("AVX2 Size 8192 Round Trip failed")
		reportDifferences(t, dstInv, src, 1e-4)
	}
}
