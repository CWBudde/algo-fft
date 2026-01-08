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

func TestStage1OutputCompare(t *testing.T) {
	if !cpu.HasAVX2() {
		t.Skip("AVX2 not supported")
	}

	const n = 512
	src := randomComplex64(n, 0xDEADBEEF)
	
	dstAVX2 := make([]complex64, n)
	scratchAVX2 := make([]complex64, n)
	dstGo := make([]complex64, n)
	scratchGo := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := mathpkg.ComputeIdentityIndices(n)

	// Run Go version to get its scratch (out) buffer after Stage 1
	forwardDIT512Mixed16x32Complex64(dstGo, src, twiddle, scratchGo, bitrev)
	
	// Run AVX2 version
	amd64.ForwardAVX2Size512Radix16x32Complex64Asm(dstAVX2, src, twiddle, scratchAVX2, bitrev)

	// Compare scratch buffers (out after Stage 1)
	fmt.Println("Comparing Stage 1 output (scratch buffer):")
	const tol = 1e-4
	errors := 0
	for i := 0; i < n && errors < 30; i++ {
		if cmplx.Abs(complex128(scratchAVX2[i])-complex128(scratchGo[i])) > tol {
			k2 := i / 16
			n1 := i % 16
			fmt.Printf("out[%3d] (k2=%2d,n1=%2d): AVX2=(%+.4f%+.4fi) Go=(%+.4f%+.4fi)\n",
				i, k2, n1,
				real(scratchAVX2[i]), imag(scratchAVX2[i]),
				real(scratchGo[i]), imag(scratchGo[i]))
			errors++
		}
	}
	if errors > 0 {
		t.Logf("Found %d+ Stage 1 output differences", errors)
	} else {
		fmt.Println("Stage 1 outputs MATCH!")
	}
}
