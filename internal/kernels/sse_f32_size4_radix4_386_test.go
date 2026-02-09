//go:build 386 && asm && !purego

package kernels

import (
	"testing"

	x86 "github.com/cwbudde/algo-fft/internal/asm/x86"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

func TestForwardSSESize4Radix4Complex64_386(t *testing.T) {
	const n = 4
	src := randomComplex64(n, 0x12345678)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	want := make([]complex64, n)
	copy(want, src)
	// Use the generic Go implementation as reference
	forwardDIT4Radix4Complex64(want, want, twiddle, scratch)

	// Call the SSE assembly implementation
	if !x86.ForwardSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("ForwardSSESize4Radix4Complex64Asm failed")
	}

	assertComplex64Close(t, dst, want, 1e-5)
}

func TestInverseSSESize4Radix4Complex64_386(t *testing.T) {
	const n = 4
	src := randomComplex64(n, 0x87654321)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	want := make([]complex64, n)
	copy(want, src)
	// Use the generic Go implementation as reference
	inverseDIT4Radix4Complex64(want, want, twiddle, scratch)

	// Call the SSE assembly implementation
	if !x86.InverseSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("InverseSSESize4Radix4Complex64Asm failed")
	}

	assertComplex64Close(t, dst, want, 1e-5)
}

func BenchmarkForwardSSESize4Radix4Complex64_386(b *testing.B) {
	const n = 4
	src := randomComplex64(n, 0x99999999)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8)) // 8 bytes per complex64
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		x86.ForwardSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkInverseSSESize4Radix4Complex64_386(b *testing.B) {
	const n = 4
	src := randomComplex64(n, 0x88888888)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	dst := make([]complex64, n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		x86.InverseSSESize4Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}
