//go:build amd64 && asm && !purego

package kernels

import (
	"fmt"
	"math/cmplx"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	"github.com/MeKo-Christian/algo-fft/internal/cpu"
)

func TestDebugAVX2vsGo(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xDEADBEEF)
	
	dstAVX2 := make([]complex64, n)
	dstGo := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	// Run Go version
	if !forwardDIT512Mixed16x32Complex64(dstGo, src, twiddle, scratch, bitrev) {
		t.Fatal("Go kernel failed")
	}
	
	// Run AVX2 version
	if !amd64.ForwardAVX2Size512Radix16x32Complex64Asm(dstAVX2, src, twiddle, scratch, bitrev) {
		t.Fatal("AVX2 kernel returned false")
	}

	// Compare and find first discrepancy
	const tol = 1e-4
	errors := 0
	for i := 0; i < n && errors < 20; i++ {
		if cmplx.Abs(complex128(dstAVX2[i])-complex128(dstGo[i])) > tol {
			k1 := i / 32
			k2 := i % 32
			fmt.Printf("idx=%3d (k1=%2d,k2=%2d): AVX2=(%+.4f%+.4fi) Go=(%+.4f%+.4fi)\n",
				i, k1, k2,
				real(dstAVX2[i]), imag(dstAVX2[i]),
				real(dstGo[i]), imag(dstGo[i]))
			errors++
		}
	}
	if errors > 0 {
		t.Fatalf("Found %d+ errors between AVX2 and Go", errors)
	}
}
