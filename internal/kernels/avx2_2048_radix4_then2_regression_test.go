//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
)

func TestAVX2Size2048Radix4Then2Regression(t *testing.T) {
	const (
		n   = 2048
		tol = 2e-5
	)

	src := randomComplex64(n, 0x2048A5A5)
	twiddle := ComputeTwiddleFactors[complex64](n)

	asmForward := make([]complex64, n)
	asmInverse := make([]complex64, n)
	asmScratch := make([]complex64, n)

	if !amd64.ForwardAVX2Size2048Radix4Then2Complex64Asm(asmForward, src, twiddle, asmScratch) {
		t.Fatal("forward AVX2 size-2048 radix4_then2 failed")
	}

	goForward := make([]complex64, n)
	goScratch := make([]complex64, n)
	if !forwardDIT2048Radix4Then2Complex64(goForward, src, twiddle, goScratch) {
		t.Fatal("forward Go size-2048 radix4_then2 failed")
	}

	assertComplex64Close(t, asmForward, goForward, tol)

	if !amd64.InverseAVX2Size2048Radix4Then2Complex64Asm(asmInverse, asmForward, twiddle, asmScratch) {
		t.Fatal("inverse AVX2 size-2048 radix4_then2 failed")
	}

	assertComplex64Close(t, asmInverse, src, tol)
}
