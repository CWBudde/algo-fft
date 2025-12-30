//go:build amd64 && fft_asm && !purego

package fft

import (
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const tolComplex128 = 1e-12

func TestAVX2Size8Radix2Complex128(t *testing.T) {
	requireAVX2(t)

	n := 8
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1), float64(i+1)*0.5)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)

	// Forward
	if !forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardAVX2Size8Radix2Complex128Asm failed")
	}

	want := reference.NaiveDFT128(src)
	for i := range dst {
		if cmplx.Abs(dst[i]-want[i]) > tolComplex128 {
			t.Errorf("Forward[%d] = %v, want %v", i, dst[i], want[i])
		}
	}

	// Inverse
	srcInv := make([]complex128, n)
	copy(srcInv, dst)
	if !inverseAVX2Size8Radix2Complex128Asm(dst, srcInv, twiddle, scratch, bitrev) {
		t.Fatal("inverseAVX2Size8Radix2Complex128Asm failed")
	}

	for i := range dst {
		if cmplx.Abs(dst[i]-src[i]) > tolComplex128 {
			t.Errorf("Inverse[%d] = %v, want %v", i, dst[i], src[i])
		}
	}
}

func TestAVX2Size8Radix8Complex128(t *testing.T) {
	requireAVX2(t)

	n := 8
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1), float64(i+1)*0.5)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)
	dst := make([]complex128, n)

	// Forward
	if !forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardAVX2Size8Radix8Complex128Asm failed")
	}

	want := reference.NaiveDFT128(src)
	for i := range dst {
		if cmplx.Abs(dst[i]-want[i]) > tolComplex128 {
			t.Errorf("Forward[%d] = %v, want %v", i, dst[i], want[i])
		}
	}

	// Inverse
	srcInv := make([]complex128, n)
	copy(srcInv, dst)
	if !inverseAVX2Size8Radix8Complex128Asm(dst, srcInv, twiddle, scratch, bitrev) {
		t.Fatal("inverseAVX2Size8Radix8Complex128Asm failed")
	}

	for i := range dst {
		if cmplx.Abs(dst[i]-src[i]) > tolComplex128 {
			t.Errorf("Inverse[%d] = %v, want %v", i, dst[i], src[i])
		}
	}
}

func TestAVX2Size8Radix8Complex128_InPlace(t *testing.T) {
	requireAVX2(t)

	n := 8
	buf := make([]complex128, n)
	for i := range buf {
		buf[i] = complex(float64(i+1), float64(i+1)*0.25)
	}
	orig := make([]complex128, n)
	copy(orig, buf)

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)

	if !forwardAVX2Size8Radix8Complex128Asm(buf, buf, twiddle, scratch, bitrev) {
		t.Fatal("forward (in-place) failed")
	}
	if !inverseAVX2Size8Radix8Complex128Asm(buf, buf, twiddle, scratch, bitrev) {
		t.Fatal("inverse (in-place) failed")
	}

	for i := range buf {
		if cmplx.Abs(buf[i]-orig[i]) > tolComplex128 {
			t.Errorf("Roundtrip[%d] = %v, want %v", i, buf[i], orig[i])
		}
	}
}

func TestAVX2Size8Complex128_Radix2VsRadix8(t *testing.T) {
	requireAVX2(t)

	n := 8
	src := make([]complex128, n)
	for i := range src {
		// Slightly non-symmetric values to exercise twiddle multiplies
		src[i] = complex(float64(i+1)*0.75, float64(i+1)*0.33)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)

	dstR2 := make([]complex128, n)
	dstR8 := make([]complex128, n)

	if !forwardAVX2Size8Radix2Complex128Asm(dstR2, src, twiddle, scratch, bitrev) {
		t.Fatal("forward radix-2 failed")
	}
	if !forwardAVX2Size8Radix8Complex128Asm(dstR8, src, twiddle, scratch, bitrev) {
		t.Fatal("forward radix-8 failed")
	}

	for i := 0; i < n; i++ {
		if cmplx.Abs(dstR2[i]-dstR8[i]) > tolComplex128 {
			t.Fatalf("forward mismatch at %d: radix2=%v radix8=%v", i, dstR2[i], dstR8[i])
		}
	}

	invR2 := make([]complex128, n)
	invR8 := make([]complex128, n)
	copy(invR2, dstR2) // src for inverse
	copy(invR8, dstR8) // src for inverse

	dstInvR2 := make([]complex128, n)
	dstInvR8 := make([]complex128, n)

	if !inverseAVX2Size8Radix2Complex128Asm(dstInvR2, invR2, twiddle, scratch, bitrev) {
		t.Fatal("inverse radix-2 failed")
	}
	if !inverseAVX2Size8Radix8Complex128Asm(dstInvR8, invR8, twiddle, scratch, bitrev) {
		t.Fatal("inverse radix-8 failed")
	}

	for i := 0; i < n; i++ {
		if cmplx.Abs(dstInvR2[i]-dstInvR8[i]) > tolComplex128 {
			t.Fatalf("inverse mismatch at %d: radix2=%v radix8=%v", i, dstInvR2[i], dstInvR8[i])
		}
		if cmplx.Abs(dstInvR2[i]-src[i]) > tolComplex128 {
			t.Fatalf("roundtrip mismatch at %d: got=%v want=%v", i, dstInvR2[i], src[i])
		}
	}
}

func TestAVX2Size8Complex128_Radix8AsmVsGo(t *testing.T) {
	requireAVX2(t)

	const n = 8
	src := make([]complex128, n)
	for i := range src {
		src[i] = complex(float64(i+1)*0.9, float64(i+1)*0.4)
	}

	twiddle := ComputeTwiddleFactors[complex128](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex128, n)

	gotAsm := make([]complex128, n)
	gotGo := make([]complex128, n)

	if !forwardAVX2Size8Radix8Complex128Asm(gotAsm, src, twiddle, scratch, bitrev) {
		t.Fatal("forward asm radix-8 failed")
	}
	if !forwardDIT8Radix8Complex128(gotGo, src, twiddle, scratch, bitrev) {
		t.Fatal("forward go radix-8 failed")
	}

	for i := 0; i < n; i++ {
		if cmplx.Abs(gotAsm[i]-gotGo[i]) > tolComplex128 {
			t.Fatalf("forward asm vs go mismatch at %d: asm=%v go=%v", i, gotAsm[i], gotGo[i])
		}
	}

	invAsm := make([]complex128, n)
	invGo := make([]complex128, n)
	if !inverseAVX2Size8Radix8Complex128Asm(invAsm, gotAsm, twiddle, scratch, bitrev) {
		t.Fatal("inverse asm radix-8 failed")
	}
	if !inverseDIT8Radix8Complex128(invGo, gotGo, twiddle, scratch, bitrev) {
		t.Fatal("inverse go radix-8 failed")
	}

	for i := 0; i < n; i++ {
		if cmplx.Abs(invAsm[i]-invGo[i]) > tolComplex128 {
			t.Fatalf("inverse asm vs go mismatch at %d: asm=%v go=%v", i, invAsm[i], invGo[i])
		}
	}
}
