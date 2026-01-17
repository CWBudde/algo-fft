//go:build amd64 && asm

package kernels

import (
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
)

func TestAVX2Size32Radix4Then2Complex64(t *testing.T) {
	const (
		n   = 32
		tol = radix4Then2Tol64
	)

	src := randomComplex64(n, 0x32A5BEEF)
	twiddle := ComputeTwiddleFactors[complex64](n)

	asmForward := make([]complex64, n)
	asmInverse := make([]complex64, n)
	asmScratch := make([]complex64, n)

	if !amd64.ForwardAVX2Size32Radix4Then2Complex64Asm(asmForward, src, twiddle, asmScratch) {
		t.Fatal("forward AVX2 size-32 radix4_then2 failed")
	}

	goForward := make([]complex64, n)
	goScratch := make([]complex64, n)
	if !forwardDIT32Radix4Then2Complex64(goForward, src, twiddle, goScratch) {
		t.Fatal("forward Go size-32 radix4_then2 failed")
	}

	assertComplex64Close(t, asmForward, goForward, tol)

	if !amd64.InverseAVX2Size32Radix4Then2Complex64Asm(asmInverse, asmForward, twiddle, asmScratch) {
		t.Fatal("inverse AVX2 size-32 radix4_then2 failed")
	}

	assertComplex64Close(t, asmInverse, src, tol)
}

func TestAVX2Size32Radix4Then2Complex128(t *testing.T) {
	const (
		n   = 32
		tol = 1e-10
	)

	src := randomComplex128(n, 0x32C0FFEE)
	twiddle := ComputeTwiddleFactors[complex128](n)

	asmForward := make([]complex128, n)
	asmInverse := make([]complex128, n)
	asmScratch := make([]complex128, n)

	if !amd64.ForwardAVX2Size32Radix4Then2Complex128Asm(asmForward, src, twiddle, asmScratch) {
		t.Fatal("forward AVX2 size-32 radix4_then2 complex128 failed")
	}

	goForward := make([]complex128, n)
	goScratch := make([]complex128, n)
	if !forwardDIT32Radix4Then2Complex128(goForward, src, twiddle, goScratch) {
		t.Fatal("forward Go size-32 radix4_then2 complex128 failed")
	}

	assertComplex128Close(t, asmForward, goForward, tol)

	if !amd64.InverseAVX2Size32Radix4Then2Complex128Asm(asmInverse, asmForward, twiddle, asmScratch) {
		t.Fatal("inverse AVX2 size-32 radix4_then2 complex128 failed")
	}

	assertComplex128Close(t, asmInverse, src, tol)
}
