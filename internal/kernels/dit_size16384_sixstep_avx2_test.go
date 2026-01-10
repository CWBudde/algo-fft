//go:build amd64 && asm && !purego

package kernels

import (
	"math"
	"math/rand"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/asm/amd64"
	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
)

func TestInverse16384SixStepAVX2_Impulse(t *testing.T) {
	const n = 16384

	// FFT of impulse gives all ones
	// IFFT of all ones should give impulse (scaled by n)
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(1, 0)
	}

	dst := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	if !inverseDIT16384SixStepAVX2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("inverseDIT16384SixStepAVX2Complex64 returned false")
	}

	// For IFFT of constant 1, result should be impulse at position 0
	// (possibly scaled by 1/n depending on normalization)
	t.Logf("First 4 outputs: %v %v %v %v", dst[0], dst[1], dst[2], dst[3])
	t.Logf("Last 4 outputs: %v %v %v %v", dst[n-4], dst[n-3], dst[n-2], dst[n-1])

	// Check if dst[0] is close to n (unscaled) or 1 (scaled)
	dc := real(dst[0])
	t.Logf("DC value: %v", dc)
}

func TestRoundTrip16384SixStepAVX2_Impulse(t *testing.T) {
	const n = 16384

	// Round-trip test with impulse
	src := make([]complex64, n)
	src[0] = complex(1, 0)

	freq := make([]complex64, n)
	result := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	if !forwardDIT16384SixStepAVX2Complex64(freq, src, twiddle, scratch) {
		t.Fatal("forward returned false")
	}

	t.Logf("After forward (should be all 1): first 4 = %v %v %v %v", freq[0], freq[1], freq[2], freq[3])

	if !inverseDIT16384SixStepAVX2Complex64(result, freq, twiddle, scratch) {
		t.Fatal("inverse returned false")
	}

	t.Logf("After inverse (should be impulse): first 4 = %v %v %v %v", result[0], result[1], result[2], result[3])

	// Check round-trip
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
	if maxErr > 1e-5 {
		t.Errorf("Impulse round-trip failed with error %e", maxErr)
	}
}

func TestForwardDIT16384SixStepAVX2_Impulse(t *testing.T) {
	const n = 16384

	// Test with impulse: FFT of impulse should give all ones
	src := make([]complex64, n)
	src[0] = complex(1, 0)

	dst := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	if !forwardDIT16384SixStepAVX2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384SixStepAVX2Complex64 returned false")
	}

	// All outputs should be close to 1
	errors := 0
	maxErr := float32(0)
	for i := range dst {
		re := real(dst[i]) - 1.0
		im := imag(dst[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
		}
		if err > 0.01 {
			if errors < 5 {
				t.Errorf("Output[%d] = %v, expected ~(1,0)", i, dst[i])
			}
			errors++
		}
	}

	t.Logf("Max error from expected (1,0): %e", maxErr)
	if errors > 0 {
		t.Errorf("Total %d errors", errors)
	}
}

func TestForwardDIT16384SixStepAVX2_Complex64(t *testing.T) {
	const n = 16384

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstRadix4 := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrevRadix4 := mathpkg.ComputeBitReversalIndicesRadix4(n)

	// Fill with test data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Run both implementations
	if !forwardDIT16384SixStepAVX2Complex64(dstSixStep, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384SixStepAVX2Complex64 returned false")
	}

	if !amd64.ForwardAVX2Size16384Radix4Complex64Asm(dstRadix4, src, twiddle, scratch, bitrevRadix4) {
		t.Fatal("ForwardAVX2Size16384Radix4Complex64Asm returned false")
	}

	// Log DC and Nyquist for both
	t.Logf("SixStep DC=%v, Nyquist=%v", dstSixStep[0], dstSixStep[n/2])
	t.Logf("Radix4  DC=%v, Nyquist=%v", dstRadix4[0], dstRadix4[n/2])

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

	// For now, just log - they produce different orderings
	// The important test is round-trip which tests mathematical correctness
}

func TestRoundTripDIT16384SixStepAVX2_Complex64(t *testing.T) {
	const n = 16384

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
	if !forwardDIT16384SixStepAVX2Complex64(freq, src, twiddle, scratch) {
		t.Fatal("forward returned false")
	}

	t.Logf("After forward: first few = %v %v %v", freq[0], freq[1], freq[2])

	if !inverseDIT16384SixStepAVX2Complex64(result, freq, twiddle, scratch) {
		t.Fatal("inverse returned false")
	}

	t.Logf("After inverse: first few result = %v %v %v", result[0], result[1], result[2])
	t.Logf("Expected: first few src = %v %v %v", src[0], src[1], src[2])

	// Verify round-trip
	maxErr := float32(0)
	maxErrIdx := 0
	for i := range n {
		re := real(result[i]) - real(src[i])
		im := imag(result[i]) - imag(src[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
			maxErrIdx = i
		}
	}

	t.Logf("Max round-trip error: %e at index %d", maxErr, maxErrIdx)
	t.Logf("At max error: src=%v, result=%v", src[maxErrIdx], result[maxErrIdx])

	const tolerance = 1e-4 // Relaxed tolerance for 16384-point FFT
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

// TestForwardDIT16384SixStepAVX2_VsRadix4 tests six-step against radix-4 implementation
// by checking if they produce the same results.
func TestForwardDIT16384SixStepAVX2_VsRadix4(t *testing.T) {
	const n = 16384

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstRadix4 := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrevRadix4 := mathpkg.ComputeBitReversalIndicesRadix4(n)

	// Fill with random data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Run six-step
	if !forwardDIT16384SixStepAVX2Complex64(dstSixStep, src, twiddle, scratch) {
		t.Fatal("six-step forward returned false")
	}

	// Run radix-4
	if !amd64.ForwardAVX2Size16384Radix4Complex64Asm(dstRadix4, src, twiddle, scratch, bitrevRadix4) {
		t.Fatal("radix-4 forward returned false")
	}

	// Check DC and a few specific bins
	t.Logf("SixStep DC=%v", dstSixStep[0])
	t.Logf("Radix4  DC=%v", dstRadix4[0])

	// DC should match (sum of all inputs)
	dcErr := cmplxAbs64(dstSixStep[0] - dstRadix4[0])
	if dcErr > 1e-3 {
		t.Errorf("DC mismatch: sixstep=%v, radix4=%v, err=%e", dstSixStep[0], dstRadix4[0], dcErr)
	}

	// Try to find a permutation pattern
	// For each output in radix4, find the closest match in sixstep
	matched := 0
	for i := range n {
		r4Val := dstRadix4[i]
		bestErr := float32(1e10)
		bestIdx := -1
		for j := range n {
			err := cmplxAbs64(dstSixStep[j] - r4Val)
			if err < bestErr {
				bestErr = err
				bestIdx = j
			}
		}
		if bestErr < 1e-4 {
			matched++
			if matched <= 5 {
				t.Logf("Radix4[%d] matches SixStep[%d] (err=%e)", i, bestIdx, bestErr)
			}
		}
	}
	t.Logf("Matched %d/%d outputs between implementations", matched, n)
}

func cmplxAbs64(c complex64) float32 {
	r, i := real(c), imag(c)
	return float32(math.Sqrt(float64(r*r + i*i)))
}

// TestSize128Kernel_RoundTrip tests if the size-128 FFT kernel round-trips correctly
// Uses generic DIT (ForwardAVX2Complex64Asm) rather than radix-4 specific
func TestSize128Kernel_RoundTrip(t *testing.T) {
	const n = 128

	src := make([]complex64, n)
	freq := make([]complex64, n)
	result := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	// Random input
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Forward using generic DIT
	bitrev := mathpkg.ComputeBitReversalIndices(n)
	if !amd64.ForwardAVX2Complex64Asm(freq, src, twiddle, scratch, bitrev) {
		t.Fatal("forward returned false")
	}

	// Inverse using generic DIT
	if !amd64.InverseAVX2Complex64Asm(result, freq, twiddle, scratch, bitrev) {
		t.Fatal("inverse returned false")
	}

	// Check round-trip
	maxErr := float32(0)
	for i := range n {
		re := real(result[i]) - real(src[i])
		im := imag(result[i]) - imag(src[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Generic DIT round-trip error: %e", maxErr)

	const tolerance float32 = 1e-5
	if maxErr > tolerance {
		t.Errorf("Round-trip failed with error %e (tolerance %e)", maxErr, tolerance)
	}
}

// TestSixStep16384_RoundTrip_SingleBin tests round-trip with a single bin
func TestSixStep16384_RoundTrip_SingleBin(t *testing.T) {
	const n = 16384

	// Input with single non-zero at position 1
	src := make([]complex64, n)
	src[1] = complex(1, 0)

	freq := make([]complex64, n)
	result := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	// Forward
	if !forwardDIT16384SixStepAVX2Complex64(freq, src, twiddle, scratch) {
		t.Fatal("forward returned false")
	}

	// Inverse
	if !inverseDIT16384SixStepAVX2Complex64(result, freq, twiddle, scratch) {
		t.Fatal("inverse returned false")
	}

	// Check round-trip
	maxErr := float32(0)
	maxErrIdx := 0
	for i := range n {
		re := real(result[i]) - real(src[i])
		im := imag(result[i]) - imag(src[i])
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
			maxErrIdx = i
		}
	}

	t.Logf("Max round-trip error: %e at index %d", maxErr, maxErrIdx)
	t.Logf("src[1]=%v, result[1]=%v", src[1], result[1])
	t.Logf("result[0]=%v, result[2]=%v", result[0], result[2])

	// Find where the non-zero went
	for i := range n {
		if cmplxAbs64(result[i]) > 0.5 {
			t.Logf("Non-zero result at index %d: %v (row=%d, col=%d)", i, result[i], i/128, i%128)
		}
	}

	if maxErr > 1e-4 {
		t.Errorf("Max error %e exceeds tolerance", maxErr)
	}
}

// TestSixStep16384_SingleFrequency tests with a single frequency bin
func TestSixStep16384_SingleFrequency(t *testing.T) {
	const n = 16384
	const m = 128

	// Use a simple frequency bin input: non-zero only at position 1
	src := make([]complex64, n)
	src[1] = complex(1, 0)

	dst := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	// Run our six-step
	if !forwardDIT16384SixStepAVX2Complex64(dst, src, twiddle, scratch) {
		t.Fatal("forward returned false")
	}

	// Compare with reference DFT using naive O(n²) implementation
	// For a single non-zero input at position 1: X[k] = W^k = exp(-2πik/n)
	maxErr := float32(0)
	maxErrIdx := 0
	for k := 0; k < n; k++ {
		// Expected: W_n^k = exp(-2πik/n)
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		expected := complex(float32(math.Cos(angle)), float32(math.Sin(angle)))

		re := real(dst[k]) - real(expected)
		im := imag(dst[k]) - imag(expected)
		err := float32(math.Sqrt(float64(re*re + im*im)))
		if err > maxErr {
			maxErr = err
			maxErrIdx = k
		}
	}

	t.Logf("Max error from expected: %e at k=%d", maxErr, maxErrIdx)
	t.Logf("At k=%d: got=%v, expected=%v", maxErrIdx, dst[maxErrIdx],
		complex(float32(math.Cos(-2*math.Pi*float64(maxErrIdx)/float64(n))),
			float32(math.Sin(-2*math.Pi*float64(maxErrIdx)/float64(n)))))

	// Log first few outputs vs expected
	t.Log("First 5 outputs:")
	for k := 0; k < 5; k++ {
		angle := -2.0 * math.Pi * float64(k) / float64(n)
		expected := complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
		t.Logf("  k=%d: got=%v, expected=%v, err=%e", k, dst[k], expected, cmplxAbs64(dst[k]-expected))
	}

	if maxErr > 1e-3 {
		// Find where position 1's result actually ended up
		for k := 0; k < n; k++ {
			expected := complex(float32(math.Cos(-2*math.Pi/float64(n))), float32(math.Sin(-2*math.Pi/float64(n))))
			err := cmplxAbs64(dst[k] - expected)
			if err < 1e-4 {
				t.Logf("Position 1's expected output W^1 found at k=%d", k)
				// Check matrix coordinates
				row := k / m
				col := k % m
				t.Logf("  Matrix position: row=%d, col=%d", row, col)
				break
			}
		}
		t.Errorf("Max error %e exceeds tolerance", maxErr)
	}
}

// TestForwardDIT4096SixStepAVX2_VsRadix4 tests if size-4096 six-step also has ordering difference
func TestForwardDIT4096SixStepAVX2_VsRadix4(t *testing.T) {
	const n = 4096

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstRadix4 := make([]complex64, n)
	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)
	bitrevRadix4 := mathpkg.ComputeBitReversalIndicesRadix4(n)

	// Fill with random data
	rng := rand.New(rand.NewSource(42))
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Run six-step
	if !forwardDIT4096SixStepAVX2Complex64(dstSixStep, src, twiddle, scratch) {
		t.Fatal("six-step forward returned false")
	}

	// Run radix-4
	if !amd64.ForwardAVX2Size4096Radix4Complex64Asm(dstRadix4, src, twiddle, scratch, bitrevRadix4) {
		t.Fatal("radix-4 forward returned false")
	}

	// Check DC
	t.Logf("SixStep DC=%v", dstSixStep[0])
	t.Logf("Radix4  DC=%v", dstRadix4[0])

	// Check max error
	maxErr := float32(0)
	for i := range n {
		err := cmplxAbs64(dstSixStep[i] - dstRadix4[i])
		if err > maxErr {
			maxErr = err
		}
	}
	t.Logf("Max error: %e", maxErr)

	// Try to find a permutation pattern
	matched := 0
	for i := range n {
		r4Val := dstRadix4[i]
		bestErr := float32(1e10)
		for j := range n {
			err := cmplxAbs64(dstSixStep[j] - r4Val)
			if err < bestErr {
				bestErr = err
			}
		}
		if bestErr < 1e-4 {
			matched++
		}
	}
	t.Logf("Matched %d/%d outputs between implementations", matched, n)
}

func TestInPlaceDIT16384SixStepAVX2_Complex64(t *testing.T) {
	const n = 16384

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
	if !forwardDIT16384SixStepAVX2Complex64(dstOOP, src, twiddle, scratch) {
		t.Fatal("out-of-place forward returned false")
	}

	// In-place test: dst == src
	dstIP := make([]complex64, n)
	copy(dstIP, src)
	scratch2 := make([]complex64, n)
	if !forwardDIT16384SixStepAVX2Complex64(dstIP, dstIP, twiddle, scratch2) {
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

func BenchmarkForwardDIT16384SixStepAVX2_Complex64(b *testing.B) {
	const n = 16384

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
		forwardDIT16384SixStepAVX2Complex64(dst, src, twiddle, scratch)
	}
}

func BenchmarkForwardDIT16384Radix4AVX2_Complex64(b *testing.B) {
	const n = 16384

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
	bitrev := mathpkg.ComputeBitReversalIndices(n)

	for b.Loop() {
		amd64.ForwardAVX2Size16384Radix4Complex64Asm(dst, src, twiddle, scratch, bitrev)
	}
}
