//go:build amd64 && asm && !purego

package kernels

import (
	"math"
	"math/cmplx"
	"testing"

	amd64 "github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	m "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestTwiddleSize8192Radix4Then2AVX2 verifies twiddle size calculation.
func TestTwiddleSize8192Radix4Then2AVX2(t *testing.T) {
	size := twiddleSize8192Radix4Then2AVX2(8192)
	if size != twiddleSize8192Radix4Then2Elems {
		t.Errorf("twiddleSize8192Radix4Then2AVX2(8192) = %d, want %d", size, twiddleSize8192Radix4Then2Elems)
	}
}

// TestPrepareTwiddle8192Radix4Then2AVX2 verifies twiddle preparation.
func TestPrepareTwiddle8192Radix4Then2AVX2(t *testing.T) {
	// Test forward twiddle-extra
	forwardExtra := make([]complex64, twiddleSize8192Radix4Then2Elems)
	prepareTwiddle8192Radix4Then2AVX2(8192, false, forwardExtra)

	// Test inverse twiddle-extra
	inverseExtra := make([]complex64, twiddleSize8192Radix4Then2Elems)
	prepareTwiddle8192Radix4Then2AVX2(8192, true, inverseExtra)

	// Verify first twiddle in stage 2 (j=0, w1=twiddle[0], w2=twiddle[0], w3=twiddle[0])
	if diff := cmplx.Abs(complex128(forwardExtra[0] - 1)); diff > 1e-6 {
		t.Errorf("w0 = %v, want 1+0i", forwardExtra[0])
	}

	// Verify inverse twiddle-extra negates imaginary parts
	// For j=1 in stage 2, w1=twiddle[512]
	// twiddle[512] = cos(-pi/8) + i*sin(-pi/8), imag is negative for forward
	offset := 6 // 48 bytes / 8 bytes per complex64
	fwd := forwardExtra[offset]
	inv := inverseExtra[offset]
	if math.Abs(float64(real(fwd)-real(inv))) > 1e-6 {
		t.Errorf("forward and inverse real parts should match: fwd=%f, inv=%f", real(fwd), real(inv))
	}
	if math.Abs(float64(imag(fwd)+imag(inv))) > 1e-6 {
		t.Errorf("inverse imaginary should be conjugate: fwd=%f, inv=%f", imag(fwd), imag(inv))
	}
}

// TestForwardAVX2Size8192Radix4Then2ParamsVsReference verifies forward transform against DFT.
func TestForwardAVX2Size8192Radix4Then2ParamsVsReference(t *testing.T) {
	const n = 8192
	twiddleExtra := make([]complex64, twiddleSize8192Radix4Then2Elems)
	prepareTwiddle8192Radix4Then2AVX2(n, false, twiddleExtra)

	// Generate test input
	src := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i%256), float32((n-i)%256))
	}

	dst := make([]complex64, n)
	scratch := make([]complex64, n)

	// Run twiddle-extra kernel
	ok := amd64.ForwardAVX2Size8192Radix4Then2ParamsComplex64Asm(dst, src, twiddleExtra, scratch)
	if !ok {
		t.Fatal("ForwardAVX2Size8192Radix4Then2ParamsComplex64Asm returned false")
	}

	// Compute reference DFT
	src128 := make([]complex128, n)
	for i := range n {
		src128[i] = complex128(src[i])
	}
	expected := reference.NaiveDFT128(src128)

	// Compare results (looser tolerance for larger transforms due to accumulated error)
	maxDiff := 0.0
	for i := range n {
		got := complex128(dst[i])
		diff := cmplx.Abs(got - expected[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		// Relative tolerance based on magnitude
		tolerance := 0.01 * (cmplx.Abs(expected[i]) + 1.0)
		if tolerance < 0.1 {
			tolerance = 0.1
		}
		if diff > tolerance {
			t.Errorf("dst[%d] = %v, want %v (diff=%v, tolerance=%v)", i, got, expected[i], diff, tolerance)
			if i > 10 {
				t.Fatal("Too many errors, stopping")
			}
		}
	}
	t.Logf("Forward transform max diff: %v", maxDiff)
}

// TestInverseAVX2Size8192Radix4Then2ParamsVsReference verifies inverse transform against IDFT.
func TestInverseAVX2Size8192Radix4Then2ParamsVsReference(t *testing.T) {
	const n = 8192
	twiddleExtra := make([]complex64, twiddleSize8192Radix4Then2Elems)
	prepareTwiddle8192Radix4Then2AVX2(n, true, twiddleExtra)

	// Generate test input in frequency domain
	src := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i%256), float32((n-i)%256))
	}

	dst := make([]complex64, n)
	scratch := make([]complex64, n)

	// Run twiddle-extra kernel
	ok := amd64.InverseAVX2Size8192Radix4Then2ParamsComplex64Asm(dst, src, twiddleExtra, scratch)
	if !ok {
		t.Fatal("InverseAVX2Size8192Radix4Then2ParamsComplex64Asm returned false")
	}

	// Compute reference IDFT
	src128 := make([]complex128, n)
	for i := range n {
		src128[i] = complex128(src[i])
	}
	expected := reference.NaiveIDFT128(src128)

	// Compare results (looser tolerance for larger transforms due to accumulated error)
	maxDiff := 0.0
	for i := range n {
		got := complex128(dst[i])
		diff := cmplx.Abs(got - expected[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		// Relative tolerance based on magnitude
		tolerance := 0.01 * (cmplx.Abs(expected[i]) + 1.0)
		if tolerance < 0.1 {
			tolerance = 0.1
		}
		if diff > tolerance {
			t.Errorf("dst[%d] = %v, want %v (diff=%v, tolerance=%v)", i, got, expected[i], diff, tolerance)
			if i > 10 {
				t.Fatal("Too many errors, stopping")
			}
		}
	}
	t.Logf("Inverse transform max diff: %v", maxDiff)
}

// TestTwiddleSize1024Radix32x32AVX2 verifies twiddle size calculation.
func TestTwiddleSize1024Radix32x32AVX2(t *testing.T) {
	size := twiddleSize1024Radix32x32AVX2(1024)
	if size != twiddleSize1024Radix32x32AVX2Elems {
		t.Errorf("twiddleSize1024Radix32x32AVX2(1024) = %d, want %d", size, twiddleSize1024Radix32x32AVX2Elems)
	}
}

// TestPrepareTwiddle1024Radix32x32AVX2 verifies twiddle preparation.
func TestPrepareTwiddle1024Radix32x32AVX2(t *testing.T) {
	const n = 1024
	forwardExtra := make([]complex128, twiddleSize1024Radix32x32AVX2Elems)
	prepareTwiddle1024Radix32x32AVX2(n, false, forwardExtra)

	inverseExtra := make([]complex128, twiddleSize1024Radix32x32AVX2Elems)
	prepareTwiddle1024Radix32x32AVX2(n, true, inverseExtra)

	twiddle := m.ComputeTwiddleFactors[complex128](n)
	indices := []int{0, 1, 2, 511, 1023}
	for _, idx := range indices {
		if forwardExtra[idx] != twiddle[idx] {
			t.Errorf("forward twiddle[%d] = %v, want %v", idx, forwardExtra[idx], twiddle[idx])
		}
		if inverseExtra[idx] != twiddle[idx] {
			t.Errorf("inverse twiddle[%d] = %v, want %v", idx, inverseExtra[idx], twiddle[idx])
		}
	}

	offset := twiddleStage2Offset1024
	w := twiddle[256]
	fwdRe := forwardExtra[offset]
	fwdIm := forwardExtra[offset+2]
	invIm := inverseExtra[offset+2]

	if real(fwdRe) != imag(fwdRe) || math.Abs(real(fwdRe)-real(w)) > 1e-12 {
		t.Errorf("forward packed real entry = %v, want real=%f", fwdRe, real(w))
	}
	if real(fwdIm) != imag(fwdIm) || math.Abs(real(fwdIm)-imag(w)) > 1e-12 {
		t.Errorf("forward packed imag entry = %v, want imag=%f", fwdIm, imag(w))
	}
	if math.Abs(real(invIm)+imag(w)) > 1e-12 || real(invIm) != imag(invIm) {
		t.Errorf("inverse packed imag entry = %v, want imag=%f", invIm, -imag(w))
	}
}

// TestForwardAVX2Size1024Radix32x32ParamsVsReference verifies forward transform against DFT.
func TestForwardAVX2Size1024Radix32x32ParamsVsReference(t *testing.T) {
	const n = 1024
	twiddleExtra := make([]complex128, twiddleSize1024Radix32x32AVX2Elems)
	prepareTwiddle1024Radix32x32AVX2(n, false, twiddleExtra)

	src := make([]complex128, n)
	for i := range n {
		src[i] = complex(float64(i%256), float64((n-i)%256))
	}

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	if !amd64.ForwardAVX2Size1024Radix32x32Complex128Asm(dst, src, twiddleExtra, scratch) {
		t.Fatal("ForwardAVX2Size1024Radix32x32Complex128Asm returned false")
	}

	expected := reference.NaiveDFT128(src)
	for i := range n {
		diff := cmplx.Abs(dst[i] - expected[i])
		tolerance := 1e-8 * (cmplx.Abs(expected[i]) + 1.0)
		if diff > tolerance {
			t.Errorf("dst[%d] = %v, want %v (diff=%v, tolerance=%v)", i, dst[i], expected[i], diff, tolerance)
			if i > 10 {
				t.Fatal("Too many errors, stopping")
			}
		}
	}
}

// TestInverseAVX2Size1024Radix32x32ParamsVsReference verifies inverse transform against IDFT.
func TestInverseAVX2Size1024Radix32x32ParamsVsReference(t *testing.T) {
	const n = 1024
	twiddleExtra := make([]complex128, twiddleSize1024Radix32x32AVX2Elems)
	prepareTwiddle1024Radix32x32AVX2(n, true, twiddleExtra)

	src := make([]complex128, n)
	for i := range n {
		src[i] = complex(float64(i%256), float64((n-i)%256))
	}

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	if !amd64.InverseAVX2Size1024Radix32x32Complex128Asm(dst, src, twiddleExtra, scratch) {
		t.Fatal("InverseAVX2Size1024Radix32x32Complex128Asm returned false")
	}

	expected := reference.NaiveIDFT128(src)
	for i := range n {
		diff := cmplx.Abs(dst[i] - expected[i])
		tolerance := 1e-8 * (cmplx.Abs(expected[i]) + 1.0)
		if diff > tolerance {
			t.Errorf("dst[%d] = %v, want %v (diff=%v, tolerance=%v)", i, dst[i], expected[i], diff, tolerance)
			if i > 10 {
				t.Fatal("Too many errors, stopping")
			}
		}
	}
}
