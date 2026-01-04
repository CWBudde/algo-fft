package kernels

import (
	"math"
	"math/rand"
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

// Helper aliases for brevity
var ComputeBitReversalIndicesRadix4 = mathpkg.ComputeBitReversalIndicesRadix4

func TestForwardDIT4096SixStep_Complex64(t *testing.T) {
	const n = 4096

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstRadix4 := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	// Fill with test data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Run both implementations
	if !forwardDIT4096SixStepComplex64(dstSixStep, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT4096SixStepComplex64 returned false")
	}

	if !forwardDIT4096Radix4Complex64(dstRadix4, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT4096Radix4Complex64 returned false")
	}

	// Compare results
	maxErr := float32(0)
	for i := range n {
		re := real(dstSixStep[i]) - real(dstRadix4[i])
		im := imag(dstSixStep[i]) - imag(dstRadix4[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max error between six-step and radix-4: %e", maxErr)

	// Allow for floating-point accumulation errors
	const tolerance = 1e-4
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func TestInverseDIT4096SixStep_Complex64(t *testing.T) {
	const n = 4096

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstRadix4 := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	// Fill with test data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Run both implementations
	if !inverseDIT4096SixStepComplex64(dstSixStep, src, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT4096SixStepComplex64 returned false")
	}

	if !inverseDIT4096Radix4Complex64(dstRadix4, src, twiddle, scratch, bitrev) {
		t.Fatal("inverseDIT4096Radix4Complex64 returned false")
	}

	// Compare results
	maxErr := float32(0)
	for i := range n {
		re := real(dstSixStep[i]) - real(dstRadix4[i])
		im := imag(dstSixStep[i]) - imag(dstRadix4[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max error between six-step and radix-4 (inverse): %e", maxErr)

	const tolerance = 1e-4
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func TestRoundTrip4096SixStep_Complex64(t *testing.T) {
	const n = 4096

	src := make([]complex64, n)
	freq := make([]complex64, n)
	result := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	// Fill with test data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Forward then inverse
	if !forwardDIT4096SixStepComplex64(freq, src, twiddle, scratch, bitrev) {
		t.Fatal("forward returned false")
	}

	if !inverseDIT4096SixStepComplex64(result, freq, twiddle, scratch, bitrev) {
		t.Fatal("inverse returned false")
	}

	// Verify round-trip
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

	const tolerance = 1e-5
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func TestForwardDIT4096SixStep_Complex128(t *testing.T) {
	const n = 4096

	src := make([]complex128, n)
	dstSixStep := make([]complex128, n)
	dstRadix4 := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	// Fill with test data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	// Run both implementations
	if !forwardDIT4096SixStepComplex128(dstSixStep, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT4096SixStepComplex128 returned false")
	}

	if !forwardDIT4096Radix4Complex128(dstRadix4, src, twiddle, scratch, bitrev) {
		t.Fatal("forwardDIT4096Radix4Complex128 returned false")
	}

	// Compare results
	maxErr := 0.0
	for i := range n {
		re := real(dstSixStep[i]) - real(dstRadix4[i])
		im := imag(dstSixStep[i]) - imag(dstRadix4[i])
		err := math.Sqrt(re*re + im*im)
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max error between six-step and radix-4: %e", maxErr)

	const tolerance = 1e-12
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func TestRoundTrip4096SixStep_Complex128(t *testing.T) {
	const n = 4096

	src := make([]complex128, n)
	freq := make([]complex128, n)
	result := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	// Fill with test data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	// Forward then inverse
	if !forwardDIT4096SixStepComplex128(freq, src, twiddle, scratch, bitrev) {
		t.Fatal("forward returned false")
	}

	if !inverseDIT4096SixStepComplex128(result, freq, twiddle, scratch, bitrev) {
		t.Fatal("inverse returned false")
	}

	// Verify round-trip
	maxErr := 0.0
	for i := range n {
		re := real(result[i]) - real(src[i])
		im := imag(result[i]) - imag(src[i])
		err := math.Sqrt(re*re + im*im)
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max round-trip error: %e", maxErr)

	const tolerance = 1e-13
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func BenchmarkForwardDIT4096Radix4_Complex64(b *testing.B) {
	const n = 4096

	src := make([]complex64, n)
	dst := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	b.ResetTimer()
	b.SetBytes(int64(n) * 8)

	for b.Loop() {
		forwardDIT4096Radix4Complex64(dst, src, twiddle, scratch, bitrev)
	}
}

func BenchmarkForwardDIT4096SixStep_Complex64(b *testing.B) {
	const n = 4096

	src := make([]complex64, n)
	dst := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrev := ComputeBitReversalIndicesRadix4(n)

	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	b.ResetTimer()
	b.SetBytes(int64(n) * 8)

	for b.Loop() {
		forwardDIT4096SixStepComplex64(dst, src, twiddle, scratch, bitrev)
	}
}
