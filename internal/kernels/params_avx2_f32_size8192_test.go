//go:build amd64 && asm && !purego

package kernels

import (
	"math"
	"math/cmplx"
	"testing"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// generateTwiddles8192 generates standard twiddle factors for size 8192.
func generateTwiddles8192() []complex64 {
	const n = 8192
	twiddle := make([]complex64, n)
	for k := range n {
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		twiddle[k] = complex64(cmplx.Exp(complex(0, angle)))
	}
	return twiddle
}

// TestTwiddleSize8192Mixed24AVX2 verifies twiddle size calculation.
func TestTwiddleSize8192Mixed24AVX2(t *testing.T) {
	size := twiddleSize8192Mixed24AVX2(8192)
	if size != twiddleSize8192Mixed24Elems {
		t.Errorf("twiddleSize8192Mixed24AVX2(8192) = %d, want %d", size, twiddleSize8192Mixed24Elems)
	}
}

// TestPrepareTwiddle8192Mixed24AVX2 verifies twiddle preparation.
func TestPrepareTwiddle8192Mixed24AVX2(t *testing.T) {
	// Test forward twiddle-extra
	forwardExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	prepareTwiddle8192Mixed24AVX2(8192, false, forwardExtra)

	// Test inverse twiddle-extra
	inverseExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	prepareTwiddle8192Mixed24AVX2(8192, true, inverseExtra)

	// Verify first twiddle in stage 2 (j=0, w1=twiddle[0], w2=twiddle[0], w3=twiddle[0])
	if diff := cmplx.Abs(complex128(forwardExtra[0] - 1)); diff > 1e-6 {
		t.Errorf("w0 = %v, want 1+0i", forwardExtra[0])
	}

	// Verify inverse twiddle-extra negates imaginary parts
	// For j=1 in stage 2, w1=twiddle[512]
	// twiddle[512] = cos(-π/8) + i*sin(-π/8), imag is negative for forward
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

// TestForwardAVX2Size8192Mixed24ParamsVsReference verifies forward transform against DFT.
func TestForwardAVX2Size8192Mixed24ParamsVsReference(t *testing.T) {
	const n = 8192
	twiddleExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	prepareTwiddle8192Mixed24AVX2(n, false, twiddleExtra)

	// Generate test input
	src := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i%256), float32((n-i)%256))
	}

	dst := make([]complex64, n)
	scratch := make([]complex64, n)

	// Run twiddle-extra kernel
	ok := amd64.ForwardAVX2Size8192Mixed24ParamsComplex64Asm(dst, src, twiddleExtra, scratch)
	if !ok {
		t.Fatal("ForwardAVX2Size8192Mixed24ParamsComplex64Asm returned false")
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

// TestInverseAVX2Size8192Mixed24ParamsVsReference verifies inverse transform against IDFT.
func TestInverseAVX2Size8192Mixed24ParamsVsReference(t *testing.T) {
	const n = 8192
	twiddleExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	prepareTwiddle8192Mixed24AVX2(n, true, twiddleExtra)

	// Generate test input in frequency domain
	src := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i*i%17), float32((i+5)%13))
	}

	dst := make([]complex64, n)
	scratch := make([]complex64, n)

	// Run twiddle-extra kernel
	ok := amd64.InverseAVX2Size8192Mixed24ParamsComplex64Asm(dst, src, twiddleExtra, scratch)
	if !ok {
		t.Fatal("InverseAVX2Size8192Mixed24ParamsComplex64Asm returned false")
	}

	// Compute reference IDFT
	src128 := make([]complex128, n)
	for i := range n {
		src128[i] = complex128(src[i])
	}
	expected := reference.NaiveIDFT128(src128)

	// Compare results
	maxDiff := 0.0
	for i := range n {
		got := complex128(dst[i])
		diff := cmplx.Abs(got - expected[i])
		if diff > maxDiff {
			maxDiff = diff
		}
		tolerance := 0.01 * (cmplx.Abs(expected[i]) + 1.0)
		if tolerance < 0.001 {
			tolerance = 0.001
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

// TestRoundtripAVX2Size8192Mixed24Params verifies forward+inverse = identity.
func TestRoundtripAVX2Size8192Mixed24Params(t *testing.T) {
	const n = 8192
	forwardExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	inverseExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	prepareTwiddle8192Mixed24AVX2(n, false, forwardExtra)
	prepareTwiddle8192Mixed24AVX2(n, true, inverseExtra)

	// Generate test input
	original := make([]complex64, n)
	for i := range n {
		original[i] = complex(float32(i)*0.001, float32(n-i)*0.001)
	}

	freq := make([]complex64, n)
	recovered := make([]complex64, n)
	scratch := make([]complex64, n)

	// Forward transform
	ok := amd64.ForwardAVX2Size8192Mixed24ParamsComplex64Asm(freq, original, forwardExtra, scratch)
	if !ok {
		t.Fatal("Forward transform failed")
	}

	// Inverse transform
	ok = amd64.InverseAVX2Size8192Mixed24ParamsComplex64Asm(recovered, freq, inverseExtra, scratch)
	if !ok {
		t.Fatal("Inverse transform failed")
	}

	// Verify roundtrip
	maxDiff := 0.0
	for i := range n {
		diff := cmplx.Abs(complex128(recovered[i] - original[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 0.01 {
			t.Errorf("roundtrip[%d]: got %v, want %v (diff=%v)", i, recovered[i], original[i], diff)
			if i > 10 {
				t.Fatal("Too many errors, stopping")
			}
		}
	}
	t.Logf("Roundtrip max diff: %v", maxDiff)
}

// TestParamsVsNonParamsAVX2Size8192Mixed24 verifies twiddle-extra kernel matches non-extra kernel.
func TestParamsVsNonParamsAVX2Size8192Mixed24(t *testing.T) {
	const n = 8192
	twiddle := generateTwiddles8192()
	forwardExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	inverseExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	prepareTwiddle8192Mixed24AVX2(n, false, forwardExtra)
	prepareTwiddle8192Mixed24AVX2(n, true, inverseExtra)

	// Generate test input
	src := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i%100), float32(-i%100))
	}

	// Non-extra kernel
	dstNonParams := make([]complex64, n)
	scratchNonParams := make([]complex64, n)
	ok := amd64.ForwardAVX2Size8192Mixed24Complex64Asm(dstNonParams, src, twiddle, scratchNonParams)
	if !ok {
		t.Fatal("Non-extra forward transform failed")
	}

	// Twiddle-extra kernel
	dstParams := make([]complex64, n)
	scratchParams := make([]complex64, n)
	ok = amd64.ForwardAVX2Size8192Mixed24ParamsComplex64Asm(dstParams, src, forwardExtra, scratchParams)
	if !ok {
		t.Fatal("Extra forward transform failed")
	}

	// Compare forward results
	maxDiff := 0.0
	for i := range n {
		diff := cmplx.Abs(complex128(dstParams[i] - dstNonParams[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-4 {
			t.Errorf("forward[%d]: extra=%v, non-extra=%v (diff=%v)", i, dstParams[i], dstNonParams[i], diff)
			if i > 10 {
				t.Fatal("Too many errors, stopping")
			}
		}
	}
	t.Logf("Forward extra vs non-extra max diff: %v", maxDiff)

	// Test inverse
	invSrc := dstNonParams // Use forward output as inverse input
	invDstNonParams := make([]complex64, n)
	invDstParams := make([]complex64, n)

	ok = amd64.InverseAVX2Size8192Mixed24Complex64Asm(invDstNonParams, invSrc, twiddle, scratchNonParams)
	if !ok {
		t.Fatal("Non-extra inverse transform failed")
	}

	ok = amd64.InverseAVX2Size8192Mixed24ParamsComplex64Asm(invDstParams, invSrc, inverseExtra, scratchParams)
	if !ok {
		t.Fatal("Extra inverse transform failed")
	}

	// Compare inverse results
	maxDiff = 0.0
	for i := range n {
		diff := cmplx.Abs(complex128(invDstParams[i] - invDstNonParams[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 1e-4 {
			t.Errorf("inverse[%d]: extra=%v, non-extra=%v (diff=%v)", i, invDstParams[i], invDstNonParams[i], diff)
			if i > 10 {
				t.Fatal("Too many errors, stopping")
			}
		}
	}
	t.Logf("Inverse extra vs non-extra max diff: %v", maxDiff)
}

// BenchmarkForwardAVX2Size8192Mixed24Params benchmarks the twiddle-extra kernel.
func BenchmarkForwardAVX2Size8192Mixed24Params(b *testing.B) {
	const n = 8192
	twiddleExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	prepareTwiddle8192Mixed24AVX2(n, false, twiddleExtra)

	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(n * 8) // 8 bytes per complex64

	for i := 0; i < b.N; i++ {
		amd64.ForwardAVX2Size8192Mixed24ParamsComplex64Asm(dst, src, twiddleExtra, scratch)
	}
}

// BenchmarkForwardAVX2Size8192Mixed24NonParams benchmarks the non-extra kernel for comparison.
func BenchmarkForwardAVX2Size8192Mixed24NonParams(b *testing.B) {
	const n = 8192
	twiddle := generateTwiddles8192()

	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(n * 8)

	for i := 0; i < b.N; i++ {
		amd64.ForwardAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch)
	}
}

// BenchmarkInverseAVX2Size8192Mixed24Params benchmarks the inverse twiddle-extra kernel.
func BenchmarkInverseAVX2Size8192Mixed24Params(b *testing.B) {
	const n = 8192
	twiddleExtra := make([]complex64, twiddleSize8192Mixed24Elems)
	prepareTwiddle8192Mixed24AVX2(n, true, twiddleExtra)

	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(n * 8)

	for i := 0; i < b.N; i++ {
		amd64.InverseAVX2Size8192Mixed24ParamsComplex64Asm(dst, src, twiddleExtra, scratch)
	}
}

// BenchmarkInverseAVX2Size8192Mixed24NonParams benchmarks the non-extra inverse kernel.
func BenchmarkInverseAVX2Size8192Mixed24NonParams(b *testing.B) {
	const n = 8192
	twiddle := generateTwiddles8192()

	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	for i := range n {
		src[i] = complex(float32(i), float32(-i))
	}

	b.ResetTimer()
	b.ReportAllocs()
	b.SetBytes(n * 8)

	for i := 0; i < b.N; i++ {
		amd64.InverseAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch)
	}
}
