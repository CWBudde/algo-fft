//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
)

func TestAVX2Size32Mixed24Complex64(t *testing.T) {
	const (
		n   = 32
		tol = mixedRadix24Tol64
	)

	src := randomComplex64(n, 0x32A5BEEF)
	twiddle := ComputeTwiddleFactors[complex64](n)

	asmForward := make([]complex64, n)
	asmInverse := make([]complex64, n)
	asmScratch := make([]complex64, n)

	if !amd64.ForwardAVX2Size32Mixed24Complex64Asm(asmForward, src, twiddle, asmScratch) {
		t.Fatal("forward AVX2 size-32 mixed24 failed")
	}

	goForward := make([]complex64, n)
	goScratch := make([]complex64, n)
	if !forwardDIT32MixedRadix24Complex64(goForward, src, twiddle, goScratch) {
		t.Fatal("forward Go size-32 mixed24 failed")
	}

	assertComplex64Close(t, asmForward, goForward, tol)

	if !amd64.InverseAVX2Size32Mixed24Complex64Asm(asmInverse, asmForward, twiddle, asmScratch) {
		t.Fatal("inverse AVX2 size-32 mixed24 failed")
	}

	assertComplex64Close(t, asmInverse, src, tol)
}
