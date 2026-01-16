//go:build amd64 && asm && !purego

package kernels

import (
	"math"
	"math/rand"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// TestForwardDIT8192SixStep64x128AVX2_Complex64 tests that the AVX2 64×128 six-step
// implementation produces the same results as the existing mixed-radix implementation.
func TestForwardDIT8192SixStep64x128AVX2_Complex64(t *testing.T) {
	const n = 8192

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstMixed := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	if !forwardDIT8192SixStep64x128AVX2Complex64(dstSixStep, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192SixStep64x128AVX2Complex64 returned false")
	}

	if !forwardDIT8192Mixed24Complex64(dstMixed, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192Mixed24Complex64 returned false")
	}

	maxErr := float32(0)
	for i := range n {
		re := real(dstSixStep[i]) - real(dstMixed[i])
		im := imag(dstSixStep[i]) - imag(dstMixed[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max error between AVX2 64×128 six-step and mixed-radix: %e", maxErr)

	const tolerance = 1e-3
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

// TestInverseDIT8192SixStep64x128AVX2_Complex64 tests the AVX2 inverse transform.
func TestInverseDIT8192SixStep64x128AVX2_Complex64(t *testing.T) {
	const n = 8192

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstMixed := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	if !inverseDIT8192SixStep64x128AVX2Complex64(dstSixStep, src, twiddle, scratch) {
		t.Fatal("inverseDIT8192SixStep64x128AVX2Complex64 returned false")
	}

	if !inverseDIT8192Mixed24Complex64(dstMixed, src, twiddle, scratch) {
		t.Fatal("inverseDIT8192Mixed24Complex64 returned false")
	}

	maxErr := float32(0)
	for i := range n {
		re := real(dstSixStep[i]) - real(dstMixed[i])
		im := imag(dstSixStep[i]) - imag(dstMixed[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max error between AVX2 inverse implementations: %e", maxErr)

	const tolerance = 1e-3
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

// TestRoundTripDIT8192SixStep64x128AVX2_Complex64 verifies IFFT(FFT(x)) ≈ x.
func TestRoundTripDIT8192SixStep64x128AVX2_Complex64(t *testing.T) {
	const n = 8192

	src := make([]complex64, n)
	freq := make([]complex64, n)
	result := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	if !forwardDIT8192SixStep64x128AVX2Complex64(freq, src, twiddle, scratch) {
		t.Fatal("forward returned false")
	}

	if !inverseDIT8192SixStep64x128AVX2Complex64(result, freq, twiddle, scratch) {
		t.Fatal("inverse returned false")
	}

	maxErr := float32(0)
	for i := range n {
		re := real(result[i]) - real(src[i])
		im := imag(result[i]) - imag(src[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max AVX2 round-trip error: %e", maxErr)

	const tolerance = 1e-4
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

// BenchmarkForwardDIT8192SixStep64x128AVX2_Complex64 benchmarks the AVX2 version.
func BenchmarkForwardDIT8192SixStep64x128AVX2_Complex64(b *testing.B) {
	const n = 8192

	src := make([]complex64, n)
	dst := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	b.ResetTimer()
	b.SetBytes(int64(n) * 8)

	for b.Loop() {
		forwardDIT8192SixStep64x128AVX2Complex64(dst, src, twiddle, scratch)
	}
}
