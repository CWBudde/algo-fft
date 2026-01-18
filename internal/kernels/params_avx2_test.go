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

// TestTwiddleSize256Radix16AVX2 verifies twiddle size calculation.
func TestTwiddleSize256Radix16AVX2(t *testing.T) {
	size := twiddleSize256Radix16AVX2(256)
	if size != twiddleSize256Radix16AVX2Elems {
		t.Errorf("twiddleSize256Radix16AVX2(256) = %d, want %d", size, twiddleSize256Radix16AVX2Elems)
	}
}

// TestPrepareTwiddle256Radix16AVX2 verifies twiddle preparation layout.
func TestPrepareTwiddle256Radix16AVX2(t *testing.T) {
	const n = 256
	forwardExtra := make([]complex128, twiddleSize256Radix16AVX2Elems)
	prepareTwiddle256Radix16AVX2(n, false, forwardExtra)

	inverseExtra := make([]complex128, twiddleSize256Radix16AVX2Elems)
	prepareTwiddle256Radix16AVX2(n, true, inverseExtra)

	twiddle := m.ComputeTwiddleFactors[complex128](n)

	checkPair := func(col, pair int) {
		row := pair * 2
		idx0 := row * col
		idx1 := (row + 1) * col
		offset := twiddleSize256Radix16BaseElems + ((col-1)*twiddlePairsPerCol256Radix16+pair)*twiddleElemsPerPair256Radix16

		w0 := twiddle[idx0]
		w1 := twiddle[idx1]
		fwd := forwardExtra[offset:]
		inv := inverseExtra[offset:]

		if math.Abs(real(fwd[0])-real(w0)) > 1e-12 || math.Abs(imag(fwd[0])-real(w0)) > 1e-12 {
			t.Errorf("forward re0 = %v, want %v", fwd[0], complex(real(w0), real(w0)))
		}
		if math.Abs(real(fwd[1])-real(w1)) > 1e-12 || math.Abs(imag(fwd[1])-real(w1)) > 1e-12 {
			t.Errorf("forward re1 = %v, want %v", fwd[1], complex(real(w1), real(w1)))
		}
		if math.Abs(real(fwd[2])-imag(w0)) > 1e-12 || math.Abs(imag(fwd[2])-imag(w0)) > 1e-12 {
			t.Errorf("forward im0 = %v, want %v", fwd[2], complex(imag(w0), imag(w0)))
		}
		if math.Abs(real(fwd[3])-imag(w1)) > 1e-12 || math.Abs(imag(fwd[3])-imag(w1)) > 1e-12 {
			t.Errorf("forward im1 = %v, want %v", fwd[3], complex(imag(w1), imag(w1)))
		}

		if math.Abs(real(inv[2])+imag(w0)) > 1e-12 || math.Abs(imag(inv[2])+imag(w0)) > 1e-12 {
			t.Errorf("inverse im0 = %v, want %v", inv[2], complex(-imag(w0), -imag(w0)))
		}
		if math.Abs(real(inv[3])+imag(w1)) > 1e-12 || math.Abs(imag(inv[3])+imag(w1)) > 1e-12 {
			t.Errorf("inverse im1 = %v, want %v", inv[3], complex(-imag(w1), -imag(w1)))
		}
	}

	checkPair(1, 0)
	checkPair(3, 4)
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
		tolerance := 5e-8 * (cmplx.Abs(expected[i]) + 1.0)
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

// TestInverseAVX2Size1024Radix32x32ScaleVsDIT compares AVX2 output against the
// Go DIT implementation to highlight uniform scaling issues.
func TestInverseAVX2Size1024Radix32x32ScaleVsDIT(t *testing.T) {
	const n = 1024
	src := make([]complex128, n)
	for i := range n {
		src[i] = complex(float64(i%256), float64((n-i)%256))
	}

	twiddle := m.ComputeTwiddleFactors[complex128](n)
	twiddleExtra := make([]complex128, twiddleSize1024Radix32x32AVX2Elems)
	prepareTwiddle1024Radix32x32AVX2(n, true, twiddleExtra)

	dstGo := make([]complex128, n)
	scratchGo := make([]complex128, n)
	if !inverseDIT1024Mixed32x32Complex128(dstGo, src, twiddle, scratchGo) {
		t.Fatal("inverseDIT1024Mixed32x32Complex128 returned false")
	}

	dstAVX := make([]complex128, n)
	scratchAVX := make([]complex128, n)
	if !amd64.InverseAVX2Size1024Radix32x32Complex128Asm(dstAVX, src, twiddleExtra, scratchAVX) {
		t.Fatal("InverseAVX2Size1024Radix32x32Complex128Asm returned false")
	}

	var sumGo, sumAVX float64
	var count int
	for i := range n {
		g := cmplx.Abs(dstGo[i])
		a := cmplx.Abs(dstAVX[i])
		if g > 1e-9 {
			sumGo += g
			sumAVX += a
			count++
		}
	}
	if count == 0 {
		t.Fatal("insufficient non-zero samples for scale comparison")
	}
	ratio := sumAVX / sumGo

	var maxDiff, maxScaledDiff float64
	scale := complex(1.0, 0)
	if ratio != 0 {
		scale = complex(1.0/ratio, 0)
	}
	for i := range n {
		diff := cmplx.Abs(dstAVX[i] - dstGo[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		scaledDiff := cmplx.Abs(dstAVX[i]*scale - dstGo[i])
		if scaledDiff > maxScaledDiff {
			maxScaledDiff = scaledDiff
		}
	}

	t.Logf("scale ratio avx/go=%.6f max diff=%.6e max scaled diff=%.6e", ratio, maxDiff, maxScaledDiff)
	if ratio < 0.9 || ratio > 1.1 {
		t.Fatalf("inverse scale mismatch: avx/go=%.6f (expected ~1.0); max diff=%.6e", ratio, maxDiff)
	}
}

// TestInverseAVX2Size1024Radix32x32RowIFFTIsolation uses a sparse spectrum
// (k2=0 only) so the inverse should replicate a 32-point IDFT across n2.
func TestInverseAVX2Size1024Radix32x32RowIFFTIsolation(t *testing.T) {
	const n = 1024
	const n1 = 32
	src := make([]complex128, n)

	time32 := make([]complex128, n1)
	for i := range n1 {
		time32[i] = complex(float64((i*3)%7), float64(i%5))
	}

	freq32 := reference.NaiveDFT128(time32)
	for k1 := range n1 {
		src[k1*32] = freq32[k1]
	}

	twiddle := m.ComputeTwiddleFactors[complex128](n)
	twiddleExtra := make([]complex128, twiddleSize1024Radix32x32AVX2Elems)
	prepareTwiddle1024Radix32x32AVX2(n, true, twiddleExtra)

	dstGo := make([]complex128, n)
	scratchGo := make([]complex128, n)
	if !inverseDIT1024Mixed32x32Complex128(dstGo, src, twiddle, scratchGo) {
		t.Fatal("inverseDIT1024Mixed32x32Complex128 returned false")
	}

	expected := make([]complex128, n)
	for n2 := 0; n2 < 32; n2++ {
		for n1 := 0; n1 < 32; n1++ {
			expected[n2*32+n1] = time32[n1] / complex(32, 0)
		}
	}

	for i := range n {
		diff := cmplx.Abs(dstGo[i] - expected[i])
		if diff > 1e-10 {
			t.Fatalf("Go inverse mismatch at %d: got=%v want=%v diff=%e", i, dstGo[i], expected[i], diff)
		}
	}

	dstAVX := make([]complex128, n)
	scratchAVX := make([]complex128, n)
	if !amd64.InverseAVX2Size1024Radix32x32Complex128Asm(dstAVX, src, twiddleExtra, scratchAVX) {
		t.Fatal("InverseAVX2Size1024Radix32x32Complex128Asm returned false")
	}

	var maxDiff float64
	var worst int
	for i := range n {
		diff := cmplx.Abs(dstAVX[i] - expected[i])
		if diff > maxDiff {
			maxDiff = diff
			worst = i
		}
	}

	t.Logf("max diff=%e at %d avx=%v want=%v", maxDiff, worst, dstAVX[worst], expected[worst])
	if maxDiff > 1e-8 {
		t.Fatalf("AVX2 inverse mismatch (sparse k2=0): max diff=%e", maxDiff)
	}
}

// TestInverseAVX2Size1024Radix32x32K2Isolation compares AVX2 vs Go DIT when only
// a single non-zero k2 column is present, isolating inter-stage twiddle usage.
func TestInverseAVX2Size1024Radix32x32K2Isolation(t *testing.T) {
	const n = 1024
	const n1 = 32
	const k2 = 1

	src := make([]complex128, n)
	for k1 := range n1 {
		src[k1*32+k2] = complex(float64(k1+1), float64(-(k1 + 1)))
	}

	twiddle := m.ComputeTwiddleFactors[complex128](n)
	twiddleExtra := make([]complex128, twiddleSize1024Radix32x32AVX2Elems)
	prepareTwiddle1024Radix32x32AVX2(n, true, twiddleExtra)

	dstGo := make([]complex128, n)
	scratchGo := make([]complex128, n)
	if !inverseDIT1024Mixed32x32Complex128(dstGo, src, twiddle, scratchGo) {
		t.Fatal("inverseDIT1024Mixed32x32Complex128 returned false")
	}

	dstAVX := make([]complex128, n)
	scratchAVX := make([]complex128, n)
	if !amd64.InverseAVX2Size1024Radix32x32Complex128Asm(dstAVX, src, twiddleExtra, scratchAVX) {
		t.Fatal("InverseAVX2Size1024Radix32x32Complex128Asm returned false")
	}

	var maxDiff float64
	var worst int
	for i := range n {
		diff := cmplx.Abs(dstAVX[i] - dstGo[i])
		if diff > maxDiff {
			maxDiff = diff
			worst = i
		}
	}

	t.Logf("max diff=%e at %d avx=%v go=%v", maxDiff, worst, dstAVX[worst], dstGo[worst])
	if maxDiff > 1e-8 {
		t.Fatalf("AVX2 inverse mismatch (k2=%d): max diff=%e", k2, maxDiff)
	}
}

// TestInverseAVX2Size1024Radix32x32WorkVsDITK2Isolation compares the stage-1
// work buffer between Go and AVX2 for a single non-zero k2 column.
func TestInverseAVX2Size1024Radix32x32WorkVsDITK2Isolation(t *testing.T) {
	const n = 1024
	const n1 = 32
	const k2 = 1

	src := make([]complex128, n)
	for k1 := range n1 {
		src[k1*32+k2] = complex(float64(k1+1), float64(-(k1 + 1)))
	}

	twiddle := m.ComputeTwiddleFactors[complex128](n)
	twiddleExtra := make([]complex128, twiddleSize1024Radix32x32AVX2Elems)
	prepareTwiddle1024Radix32x32AVX2(n, true, twiddleExtra)

	workGo := make([]complex128, n)
	stage1InverseDIT1024Mixed32x32Complex128(workGo, src, twiddle)

	dstAVX := make([]complex128, n)
	workAVX := make([]complex128, n)
	if !amd64.InverseAVX2Size1024Radix32x32Complex128Asm(dstAVX, src, twiddleExtra, workAVX) {
		t.Fatal("InverseAVX2Size1024Radix32x32Complex128Asm returned false")
	}

	var maxDiff float64
	var worst int
	var maxDiffK20 float64
	var maxDiffK21 float64
	var nonZeroK21 int
	for i := range n {
		diff := cmplx.Abs(workAVX[i] - workGo[i])
		if diff > maxDiff {
			maxDiff = diff
			worst = i
		}
		if i < 32 {
			if diff > maxDiffK20 {
				maxDiffK20 = diff
			}
		}
		if i >= 32 && i < 64 {
			if cmplx.Abs(workAVX[i]) > 1e-9 {
				nonZeroK21++
			}
			if diff > maxDiffK21 {
				maxDiffK21 = diff
			}
		}
	}

	t.Logf("work max diff=%e at %d avx=%v go=%v", maxDiff, worst, workAVX[worst], workGo[worst])
	t.Logf("work max diff k2=0=%e k2=1=%e", maxDiffK20, maxDiffK21)
	t.Logf("work nonzero k2=1 entries=%d", nonZeroK21)
	if maxDiff > 1e-8 {
		bitrev := []int{
			0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30,
			1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31,
		}
		var maxBitrev float64
		for n1 := 0; n1 < 32; n1++ {
			i := 32 + n1
			j := 32 + bitrev[n1]
			diff := cmplx.Abs(workAVX[i] - workGo[j])
			if diff > maxBitrev {
				maxBitrev = diff
			}
		}
		t.Logf("work max diff k2=1 bitrev=%e", maxBitrev)
		var maxSwap float64
		for k := 0; k < 32; k += 2 {
			for n1 := 0; n1 < 32; n1++ {
				i0 := k*32 + n1
				i1 := (k+1)*32 + n1
				diff0 := cmplx.Abs(workAVX[i0] - workGo[i1])
				diff1 := cmplx.Abs(workAVX[i1] - workGo[i0])
				if diff0 > maxSwap {
					maxSwap = diff0
				}
				if diff1 > maxSwap {
					maxSwap = diff1
				}
			}
		}
		t.Logf("work max diff after k2 lane swap=%e", maxSwap)
		t.Fatalf("work buffer mismatch (k2=%d): max diff=%e", k2, maxDiff)
	}
}

// TestInverseAVX2Size1024Radix32x32WorkVsDITK2Zero isolates k2=0 data to
// verify the stage-1 output for the lowest column.
func TestInverseAVX2Size1024Radix32x32WorkVsDITK2Zero(t *testing.T) {
	const n = 1024
	const n1 = 32
	const k2 = 0

	src := make([]complex128, n)
	for k1 := range n1 {
		src[k1*32+k2] = complex(float64(k1+1), float64(-(k1 + 1)))
	}

	twiddle := m.ComputeTwiddleFactors[complex128](n)
	twiddleExtra := make([]complex128, twiddleSize1024Radix32x32AVX2Elems)
	prepareTwiddle1024Radix32x32AVX2(n, true, twiddleExtra)

	workGo := make([]complex128, n)
	stage1InverseDIT1024Mixed32x32Complex128(workGo, src, twiddle)

	dstAVX := make([]complex128, n)
	workAVX := make([]complex128, n)
	if !amd64.InverseAVX2Size1024Radix32x32Complex128Asm(dstAVX, src, twiddleExtra, workAVX) {
		t.Fatal("InverseAVX2Size1024Radix32x32Complex128Asm returned false")
	}

	var maxDiff float64
	var worst int
	for i := 0; i < 32; i++ {
		diff := cmplx.Abs(workAVX[i] - workGo[i])
		if diff > maxDiff {
			maxDiff = diff
			worst = i
		}
	}

	t.Logf("work k2=0 max diff=%e at %d avx=%v go=%v", maxDiff, worst, workAVX[worst], workGo[worst])
	if maxDiff > 1e-8 {
		t.Fatalf("work buffer mismatch (k2=%d): max diff=%e", k2, maxDiff)
	}
}

func stage1InverseDIT1024Mixed32x32Complex128(work, src, twiddle []complex128) {
	for k2 := 0; k2 < 32; k2++ {
		e00, e01, e02, e03, e04, e05, e06, e07, e08, e09, e10, e11, e12, e13, e14, e15 := fft16Complex128Inverse(
			src[32*0+k2], src[32*16+k2], src[32*8+k2], src[32*24+k2], src[32*4+k2], src[32*20+k2], src[32*12+k2], src[32*28+k2],
			src[32*2+k2], src[32*18+k2], src[32*10+k2], src[32*26+k2], src[32*6+k2], src[32*22+k2], src[32*14+k2], src[32*30+k2])
		o00, o01, o02, o03, o04, o05, o06, o07, o08, o09, o10, o11, o12, o13, o14, o15 := fft16Complex128Inverse(
			src[32*1+k2], src[32*17+k2], src[32*9+k2], src[32*25+k2], src[32*5+k2], src[32*21+k2], src[32*13+k2], src[32*29+k2],
			src[32*3+k2], src[32*19+k2], src[32*11+k2], src[32*27+k2], src[32*7+k2], src[32*23+k2], src[32*15+k2], src[32*31+k2])

		r0 := e00 + o00
		r16 := e00 - o00

		t1 := o01 * complex(real(twiddle[32]), -imag(twiddle[32]))
		r1 := e01 + t1
		r17 := e01 - t1

		t2 := o02 * complex(real(twiddle[64]), -imag(twiddle[64]))
		r2 := e02 + t2
		r18 := e02 - t2

		t3 := o03 * complex(real(twiddle[96]), -imag(twiddle[96]))
		r3 := e03 + t3
		r19 := e03 - t3

		t4 := o04 * complex(real(twiddle[128]), -imag(twiddle[128]))
		r4 := e04 + t4
		r20 := e04 - t4

		t5 := o05 * complex(real(twiddle[160]), -imag(twiddle[160]))
		r5 := e05 + t5
		r21 := e05 - t5

		t6 := o06 * complex(real(twiddle[192]), -imag(twiddle[192]))
		r6 := e06 + t6
		r22 := e06 - t6

		t7 := o07 * complex(real(twiddle[224]), -imag(twiddle[224]))
		r7 := e07 + t7
		r23 := e07 - t7

		t8 := o08 * complex(real(twiddle[256]), -imag(twiddle[256]))
		r8 := e08 + t8
		r24 := e08 - t8

		t9 := o09 * complex(real(twiddle[288]), -imag(twiddle[288]))
		r9 := e09 + t9
		r25 := e09 - t9

		t10 := o10 * complex(real(twiddle[320]), -imag(twiddle[320]))
		r10 := e10 + t10
		r26 := e10 - t10

		t11 := o11 * complex(real(twiddle[352]), -imag(twiddle[352]))
		r11 := e11 + t11
		r27 := e11 - t11

		t12 := o12 * complex(real(twiddle[384]), -imag(twiddle[384]))
		r12 := e12 + t12
		r28 := e12 - t12

		t13 := o13 * complex(real(twiddle[416]), -imag(twiddle[416]))
		r13 := e13 + t13
		r29 := e13 - t13

		t14 := o14 * complex(real(twiddle[448]), -imag(twiddle[448]))
		r14 := e14 + t14
		r30 := e14 - t14

		t15 := o15 * complex(real(twiddle[480]), -imag(twiddle[480]))
		r15 := e15 + t15
		r31 := e15 - t15

		base := k2 * 32
		work[base+0] = r0 * complex(real(twiddle[k2*0]), -imag(twiddle[k2*0]))
		work[base+1] = r1 * complex(real(twiddle[k2*1]), -imag(twiddle[k2*1]))
		work[base+2] = r2 * complex(real(twiddle[k2*2]), -imag(twiddle[k2*2]))
		work[base+3] = r3 * complex(real(twiddle[k2*3]), -imag(twiddle[k2*3]))
		work[base+4] = r4 * complex(real(twiddle[k2*4]), -imag(twiddle[k2*4]))
		work[base+5] = r5 * complex(real(twiddle[k2*5]), -imag(twiddle[k2*5]))
		work[base+6] = r6 * complex(real(twiddle[k2*6]), -imag(twiddle[k2*6]))
		work[base+7] = r7 * complex(real(twiddle[k2*7]), -imag(twiddle[k2*7]))
		work[base+8] = r8 * complex(real(twiddle[k2*8]), -imag(twiddle[k2*8]))
		work[base+9] = r9 * complex(real(twiddle[k2*9]), -imag(twiddle[k2*9]))
		work[base+10] = r10 * complex(real(twiddle[k2*10]), -imag(twiddle[k2*10]))
		work[base+11] = r11 * complex(real(twiddle[k2*11]), -imag(twiddle[k2*11]))
		work[base+12] = r12 * complex(real(twiddle[k2*12]), -imag(twiddle[k2*12]))
		work[base+13] = r13 * complex(real(twiddle[k2*13]), -imag(twiddle[k2*13]))
		work[base+14] = r14 * complex(real(twiddle[k2*14]), -imag(twiddle[k2*14]))
		work[base+15] = r15 * complex(real(twiddle[k2*15]), -imag(twiddle[k2*15]))
		work[base+16] = r16 * complex(real(twiddle[k2*16]), -imag(twiddle[k2*16]))
		work[base+17] = r17 * complex(real(twiddle[k2*17]), -imag(twiddle[k2*17]))
		work[base+18] = r18 * complex(real(twiddle[k2*18]), -imag(twiddle[k2*18]))
		work[base+19] = r19 * complex(real(twiddle[k2*19]), -imag(twiddle[k2*19]))
		work[base+20] = r20 * complex(real(twiddle[k2*20]), -imag(twiddle[k2*20]))
		work[base+21] = r21 * complex(real(twiddle[k2*21]), -imag(twiddle[k2*21]))
		work[base+22] = r22 * complex(real(twiddle[k2*22]), -imag(twiddle[k2*22]))
		work[base+23] = r23 * complex(real(twiddle[k2*23]), -imag(twiddle[k2*23]))
		work[base+24] = r24 * complex(real(twiddle[k2*24]), -imag(twiddle[k2*24]))
		work[base+25] = r25 * complex(real(twiddle[k2*25]), -imag(twiddle[k2*25]))
		work[base+26] = r26 * complex(real(twiddle[k2*26]), -imag(twiddle[k2*26]))
		work[base+27] = r27 * complex(real(twiddle[k2*27]), -imag(twiddle[k2*27]))
		work[base+28] = r28 * complex(real(twiddle[k2*28]), -imag(twiddle[k2*28]))
		work[base+29] = r29 * complex(real(twiddle[k2*29]), -imag(twiddle[k2*29]))
		work[base+30] = r30 * complex(real(twiddle[k2*30]), -imag(twiddle[k2*30]))
		work[base+31] = r31 * complex(real(twiddle[k2*31]), -imag(twiddle[k2*31]))
	}
}
