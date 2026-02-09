//go:build amd64 && asm && !purego

package kernels

import (
	"math"
	"math/rand"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

func TestForwardDIT4096SixStepAVX2_Complex64(t *testing.T) {
	const n = 4096

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstRadix4 := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	// Fill with test data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Run both implementations
	if !forwardDIT4096SixStepAVX2Complex64(dstSixStep, src, twiddle, scratch) {
		t.Fatal("forwardDIT4096SixStepAVX2Complex64 returned false")
	}

	if !forwardDIT4096Radix4Complex64(dstRadix4, src, twiddle, scratch) {
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

	t.Logf("Max error between AVX2 six-step and radix-4: %e", maxErr)

	const tolerance = 1e-4
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func TestRoundTripDIT4096SixStepAVX2_Complex64(t *testing.T) {
	const n = 4096

	src := make([]complex64, n)
	freq := make([]complex64, n)
	result := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	// Fill with test data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Forward then inverse
	if !forwardDIT4096SixStepAVX2Complex64(freq, src, twiddle, scratch) {
		t.Fatal("forward returned false")
	}

	if !inverseDIT4096SixStepAVX2Complex64(result, freq, twiddle, scratch) {
		t.Fatal("inverse returned false")
	}

	// Apply scaling (the AVX2 inverse doesn't scale automatically)
	// Note: We need to check if the inverse already scales or not
	// The size-64 AVX2 inverse does include 1/64 scaling

	// Verify round-trip
	// Since we do two 64-point IFFTs in series, each scales by 1/64
	// so total scaling should be 1/64 * 1/64 = 1/4096 - need to investigate

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

	// This might fail if scaling is wrong - we'll see
	const tolerance = 1e-5
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func TestInPlaceDIT4096SixStepAVX2_Complex64(t *testing.T) {
	const n = 4096

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	// Generate test data
	rng := rand.New(rand.NewSource(42))
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Out-of-place reference
	dstOOP := make([]complex64, n)
	if !forwardDIT4096SixStepAVX2Complex64(dstOOP, src, twiddle, scratch) {
		t.Fatal("out-of-place forward returned false")
	}

	// In-place test: dst == src
	dstIP := make([]complex64, n)
	copy(dstIP, src)
	scratch2 := make([]complex64, n)
	if !forwardDIT4096SixStepAVX2Complex64(dstIP, dstIP, twiddle, scratch2) {
		t.Fatal("in-place forward returned false")
	}

	// Compare in-place vs out-of-place
	maxErr := float32(0)
	for i := range n {
		re := real(dstOOP[i]) - real(dstIP[i])
		im := imag(dstOOP[i]) - imag(dstIP[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("In-place vs out-of-place max error: %e", maxErr)

	const tolerance = 1e-6
	if maxErr > tolerance {
		t.Errorf("In-place differs from out-of-place: max error %e exceeds %e", maxErr, tolerance)
	}
}

func BenchmarkForwardDIT4096SixStepAVX2_Complex64(b *testing.B) {
	const n = 4096

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
		forwardDIT4096SixStepAVX2Complex64(dst, src, twiddle, scratch)
	}
}
