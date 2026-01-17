package kernels

import (
	"math"
	"math/rand"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// TestForwardDIT8192SixStep64x128_Complex64 tests that the 64×128 six-step
// implementation produces the same results as the existing mixed-radix implementation.
func TestForwardDIT8192SixStep64x128_Complex64(t *testing.T) {
	const n = 8192

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstMixed := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, 2*n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	if !forwardDIT8192SixStep64x128Complex64(dstSixStep, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192SixStep64x128Complex64 returned false")
	}

	if !forwardDIT8192Radix4Then2Complex64(dstMixed, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192Radix4Then2Complex64 returned false")
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

	t.Logf("Max error between 64×128 six-step and mixed-radix: %e", maxErr)

	const tolerance = 1e-3
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

// TestInverseDIT8192SixStep64x128_Complex64 tests the inverse transform.
func TestInverseDIT8192SixStep64x128_Complex64(t *testing.T) {
	const n = 8192

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstMixed := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, 2*n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	if !inverseDIT8192SixStep64x128Complex64(dstSixStep, src, twiddle, scratch) {
		t.Fatal("inverseDIT8192SixStep64x128Complex64 returned false")
	}

	if !inverseDIT8192Radix4Then2Complex64(dstMixed, src, twiddle, scratch) {
		t.Fatal("inverseDIT8192Radix4Then2Complex64 returned false")
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

	t.Logf("Max error between inverse implementations: %e", maxErr)

	const tolerance = 1e-3
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

// TestRoundTripDIT8192SixStep64x128_Complex64 verifies IFFT(FFT(x)) ≈ x.
func TestRoundTripDIT8192SixStep64x128_Complex64(t *testing.T) {
	const n = 8192

	src := make([]complex64, n)
	freq := make([]complex64, n)
	result := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, 2*n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	if !forwardDIT8192SixStep64x128Complex64(freq, src, twiddle, scratch) {
		t.Fatal("forward returned false")
	}

	if !inverseDIT8192SixStep64x128Complex64(result, freq, twiddle, scratch) {
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

	t.Logf("Max round-trip error: %e", maxErr)

	const tolerance = 1e-4
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

// TestForwardDIT8192SixStep64x128_Complex128 tests the complex128 implementation.
func TestForwardDIT8192SixStep64x128_Complex128(t *testing.T) {
	const n = 8192

	src := make([]complex128, n)
	dstSixStep := make([]complex128, n)
	dstMixed := make([]complex128, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, 2*n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	if !forwardDIT8192SixStep64x128Complex128(dstSixStep, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192SixStep64x128Complex128 returned false")
	}

	if !forwardDIT8192Radix4Then2Complex128(dstMixed, src, twiddle, scratch) {
		t.Fatal("forwardDIT8192Radix4Then2Complex128 returned false")
	}

	maxErr := 0.0

	for i := range n {
		re := real(dstSixStep[i]) - real(dstMixed[i])
		im := imag(dstSixStep[i]) - imag(dstMixed[i])

		err := math.Sqrt(re*re + im*im)
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max error (complex128): %e", maxErr)

	const tolerance = 1e-10
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

// TestRoundTripDIT8192SixStep64x128_Complex128 verifies round-trip for complex128.
func TestRoundTripDIT8192SixStep64x128_Complex128(t *testing.T) {
	const n = 8192

	src := make([]complex128, n)
	freq := make([]complex128, n)
	result := make([]complex128, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, 2*n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	if !forwardDIT8192SixStep64x128Complex128(freq, src, twiddle, scratch) {
		t.Fatal("forward returned false")
	}

	if !inverseDIT8192SixStep64x128Complex128(result, freq, twiddle, scratch) {
		t.Fatal("inverse returned false")
	}

	maxErr := 0.0

	for i := range n {
		re := real(result[i]) - real(src[i])
		im := imag(result[i]) - imag(src[i])

		err := math.Sqrt(re*re + im*im)
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max round-trip error (complex128): %e", maxErr)

	const tolerance = 1e-12
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

// BenchmarkForwardDIT8192SixStep64x128_Complex64 benchmarks the 64×128 six-step.
func BenchmarkForwardDIT8192SixStep64x128_Complex64(b *testing.B) {
	const n = 8192

	src := make([]complex64, n)
	dst := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, 2*n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	b.ResetTimer()
	b.SetBytes(int64(n) * 8)

	for b.Loop() {
		forwardDIT8192SixStep64x128Complex64(dst, src, twiddle, scratch)
	}
}

// BenchmarkForwardDIT8192Radix4Then2_Complex64 benchmarks the mixed-radix for comparison.
func BenchmarkForwardDIT8192Radix4Then2_Complex64(b *testing.B) {
	const n = 8192

	src := make([]complex64, n)
	dst := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, 2*n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	b.ResetTimer()
	b.SetBytes(int64(n) * 8)

	for b.Loop() {
		forwardDIT8192Radix4Then2Complex64(dst, src, twiddle, scratch)
	}
}
