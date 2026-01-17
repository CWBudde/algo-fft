package algofft

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// TestPrecisionErrorAccumulation tests error accumulation over repeated FFT/IFFT cycles.
// This verifies that errors don't compound excessively with multiple transforms.
func TestPrecisionErrorAccumulation(t *testing.T) {
	t.Parallel()

	sizes := []int{256, 1024, 4096}
	cycles := []int{10, 100, 1000}

	for _, n := range sizes {
		for _, numCycles := range cycles {
			t.Run(fmt.Sprintf("size_%d_cycles_%d", n, numCycles), func(t *testing.T) {
				t.Parallel()
				testErrorAccumulation64(t, n, numCycles)
			})
		}
	}
}

func testErrorAccumulation64(t *testing.T, n, numCycles int) {
	t.Helper()
	// Generate random input
	original := make([]complex64, n)
	for i := range original {
		original[i] = complex(rand.Float32()*2-1, rand.Float32()*2-1)
	}

	// Create plan
	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	// Copy for repeated transforms
	data := make([]complex64, n)
	temp := make([]complex64, n)

	copy(data, original)

	// Perform repeated Forward->Inverse cycles
	for range numCycles {
		err := plan.Forward(temp, data)
		if err != nil {
			t.Fatalf("Forward failed: %v", err)
		}

		err = plan.Inverse(data, temp)
		if err != nil {
			t.Fatalf("Inverse failed: %v", err)
		}
	}

	// Measure error
	var maxError, sumError float64

	for i := range original {
		diff := cmplx64abs(data[i] - original[i])

		sumError += float64(diff)
		if float64(diff) > maxError {
			maxError = float64(diff)
		}
	}

	avgError := sumError / float64(n)

	// Expected error bounds (rough heuristics)
	expectedMaxError := 1e-4 * float64(numCycles) * math.Log2(float64(n))
	expectedAvgError := 1e-5 * float64(numCycles) * math.Log2(float64(n))

	if maxError > expectedMaxError {
		t.Errorf("Max error %e exceeds expected bound %e after %d cycles", maxError, expectedMaxError, numCycles)
	}

	if avgError > expectedAvgError {
		t.Errorf("Avg error %e exceeds expected bound %e after %d cycles", avgError, expectedAvgError, numCycles)
	}

	t.Logf("After %d cycles: max error = %e, avg error = %e", numCycles, maxError, avgError)
}

// TestPrecisionParseval verifies Parseval's theorem: energy is conserved in FFT.
// For a signal x and its FFT X: sum(|x|²) = sum(|X|²) / N.
func TestPrecisionParseval(t *testing.T) {
	t.Parallel()

	sizes := []int{256, 1024, 4096, 16384}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("size_%d", n), func(t *testing.T) {
			t.Parallel()
			testParseval64(t, n)
			testParseval128(t, n)
		})
	}
}

func testParseval64(t *testing.T, n int) {
	t.Helper()
	// Generate random input
	data := make([]complex64, n)

	var inputEnergy float64

	for i := range data {
		data[i] = complex(rand.Float32()*2-1, rand.Float32()*2-1)
		inputEnergy += float64(cmplx64abs(data[i]) * cmplx64abs(data[i]))
	}

	// Perform FFT
	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	output := make([]complex64, n)
	if err := plan.Forward(output, data); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Measure output energy
	var outputEnergy float64
	for i := range output {
		outputEnergy += float64(cmplx64abs(output[i]) * cmplx64abs(output[i]))
	}

	outputEnergy /= float64(n) // Parseval: divide by N

	// Check energy conservation
	relativeError := math.Abs(inputEnergy-outputEnergy) / inputEnergy
	if relativeError > 1e-5 {
		t.Errorf("Parseval's theorem violated: input energy %e, output energy %e, relative error %e",
			inputEnergy, outputEnergy, relativeError)
	}

	t.Logf("complex64: input energy = %e, output energy = %e, relative error = %e", inputEnergy, outputEnergy, relativeError)
}

func testParseval128(t *testing.T, n int) {
	t.Helper()
	// Generate random input
	data := make([]complex128, n)

	var inputEnergy float64

	for i := range data {
		data[i] = complex(rand.Float64()*2-1, rand.Float64()*2-1)
		inputEnergy += cmplx.Abs(data[i]) * cmplx.Abs(data[i])
	}

	// Perform FFT
	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	output := make([]complex128, n)
	if err := plan.Forward(output, data); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Measure output energy
	var outputEnergy float64
	for i := range output {
		outputEnergy += cmplx.Abs(output[i]) * cmplx.Abs(output[i])
	}

	outputEnergy /= float64(n) // Parseval: divide by N

	// Check energy conservation
	relativeError := math.Abs(inputEnergy-outputEnergy) / inputEnergy
	if relativeError > 1e-13 {
		t.Errorf("Parseval's theorem violated: input energy %e, output energy %e, relative error %e",
			inputEnergy, outputEnergy, relativeError)
	}

	t.Logf("complex128: input energy = %e, output energy = %e, relative error = %e", inputEnergy, outputEnergy, relativeError)
}

// TestPrecisionComplex64VsComplex128 compares precision between complex64 and complex128.
//
//nolint:paralleltest
func TestPrecisionComplex64VsComplex128(t *testing.T) {
	sizes := []int{256, 1024, 4096, 16384, 65536}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("size_%d", n), func(t *testing.T) {
			testPrecisionComparison(t, n)
		})
	}
}

// TestPrecisionRoundTripSweep measures round-trip error across sizes for both precisions.
func TestPrecisionRoundTripSweep(t *testing.T) {
	t.Parallel()

	sizes := []int{64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("size_%d", n), func(t *testing.T) {
			t.Parallel()

			rng := rand.New(rand.NewSource(int64(n)))
			input64, input128 := makePrecisionInputs(n, rng)

			err64 := roundTripMaxError64(t, input64)
			err128 := roundTripMaxError128(t, input128)

			log2n := math.Log2(float64(n))

			ratio := math.Inf(1)
			if err128 > 0 {
				ratio = err64 / err128
			}

			t.Logf("size=%d log2=%.0f err64=%e err128=%e ratio=%e", n, log2n, err64, err128, ratio)

			maxErr64 := 5e-4 * log2n
			if maxErr64 < 1e-4 {
				maxErr64 = 1e-4
			}

			if err64 > maxErr64 {
				t.Errorf("complex64 round-trip error %e exceeds bound %e", err64, maxErr64)
			}

			maxErr128 := 5e-10 * log2n
			if maxErr128 < 1e-11 {
				maxErr128 = 1e-11
			}

			if err128 > maxErr128 {
				t.Errorf("complex128 round-trip error %e exceeds bound %e", err128, maxErr128)
			}
		})
	}
}

func testPrecisionComparison(t *testing.T, n int) {
	t.Helper()
	// Generate test signals: sine wave
	input64 := make([]complex64, n)
	input128 := make([]complex128, n)

	freq := 5.0 // 5 cycles
	for i := range input64 {
		val := math.Sin(2 * math.Pi * freq * float64(i) / float64(n))
		input64[i] = complex(float32(val), 0)
		input128[i] = complex(val, 0)
	}

	// Perform FFT with both precisions
	plan64, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create complex64 plan: %v", err)
	}

	output64 := make([]complex64, n)
	if err := plan64.Forward(output64, input64); err != nil {
		t.Fatalf("Forward complex64 failed: %v", err)
	}

	plan128, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create complex128 plan: %v", err)
	}

	output128 := make([]complex128, n)
	if err := plan128.Forward(output128, input128); err != nil {
		t.Fatalf("Forward complex128 failed: %v", err)
	}

	// Compare results
	var maxAbsDiff, maxRelDiff float64

	for i := range output64 {
		diff := cmplx.Abs(complex128(output64[i]) - output128[i])
		if diff > maxAbsDiff {
			maxAbsDiff = diff
		}

		mag128 := cmplx.Abs(output128[i])
		if mag128 > 1e-10 {
			relDiff := diff / mag128
			if relDiff > maxRelDiff {
				maxRelDiff = relDiff
			}
		}
	}

	t.Logf("Size %d: max abs diff = %e, max rel diff = %e", n, maxAbsDiff, maxRelDiff)

	// complex64 should have ~6-7 decimal digits of precision (float32)
	// Expect relative error around 1e-6 to 1e-7
	if maxRelDiff > 1e-5 {
		t.Errorf("Precision difference too large: max relative diff %e", maxRelDiff)
	}
}

// TestPrecisionLargeFFT tests precision for very large FFT sizes.
func TestPrecisionLargeFFT(t *testing.T) {
	t.Parallel()

	if testing.Short() {
		t.Skip("skipping large FFT test in short mode")
	}

	sizes := []int{65536, 131072, 262144}

	for _, n := range sizes {
		t.Run(fmt.Sprintf("size_%d", n), func(t *testing.T) {
			t.Parallel()
			testLargePrecision(t, n)
		})
	}
}

func testLargePrecision(t *testing.T, n int) {
	t.Helper()
	// Generate simple signal: impulse at position 0
	data := make([]complex128, n)
	data[0] = complex(1.0, 0.0)

	// Expected FFT output: all ones (DC component)
	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	output := make([]complex128, n)
	if err := plan.Forward(output, data); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Check that all elements are approximately 1.0
	var maxError float64

	expected := complex(1.0, 0.0)
	for i, val := range output {
		err := cmplx.Abs(val - expected)
		if err > maxError {
			maxError = err
		}

		if i < 10 && err > 1e-10 {
			t.Logf("data[%d] = %v, expected %v, error = %e", i, val, expected, err)
		}
	}

	t.Logf("Size %d: max error = %e", n, maxError)

	// For complex128, expect error < 1e-12
	if maxError > 1e-11 {
		t.Errorf("Large FFT precision error %e exceeds threshold 1e-11", maxError)
	}

	// Test round-trip
	roundtrip := make([]complex128, n)
	if err := plan.Inverse(roundtrip, output); err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	data = roundtrip

	impulseError := cmplx.Abs(data[0] - complex(1.0, 0.0))
	if impulseError > 1e-10 {
		t.Errorf("Round-trip error at impulse position: %e", impulseError)
	}

	// Check other positions are near zero
	var maxNoiseError float64

	for i := 1; i < n; i++ {
		err := cmplx.Abs(data[i])
		if err > maxNoiseError {
			maxNoiseError = err
		}
	}

	if maxNoiseError > 1e-10 {
		t.Errorf("Round-trip noise error %e exceeds threshold 1e-10", maxNoiseError)
	}
}

// TestPrecisionKnownSignals tests FFT of signals with known analytical results.
func TestPrecisionKnownSignals(t *testing.T) {
	t.Parallel()
	t.Run("sine_wave", func(t *testing.T) { t.Parallel(); testSineWavePrecision(t) })
	t.Run("cosine_wave", func(t *testing.T) { t.Parallel(); testCosineWavePrecision(t) })
	t.Run("impulse", func(t *testing.T) { t.Parallel(); testImpulsePrecision(t) })
	t.Run("white_noise", func(t *testing.T) { t.Parallel(); testWhiteNoisePrecision(t) })
}

func testSineWavePrecision(t *testing.T) {
	t.Helper()

	n := 1024
	freq := 10 // 10 cycles in n samples

	// Generate sine wave
	data := make([]complex128, n)
	for i := range data {
		val := math.Sin(2 * math.Pi * float64(freq) * float64(i) / float64(n))
		data[i] = complex(val, 0)
	}

	// Perform FFT
	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	output := make([]complex128, n)
	if err := plan.Forward(output, data); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Sine wave should have peaks at +freq and -freq (or n-freq due to symmetry)
	// Peak magnitude should be n/2 * i (imaginary component)
	expectedMag := float64(n) / 2.0

	posFreq := output[freq]
	negFreq := output[n-freq]

	// Check magnitude at positive frequency
	posMag := cmplx.Abs(posFreq)
	if math.Abs(posMag-expectedMag) > 1.0 {
		t.Errorf("Positive frequency peak magnitude %f, expected ~%f", posMag, expectedMag)
	}

	// Check it's predominantly imaginary (sine)
	if math.Abs(real(posFreq)) > math.Abs(imag(posFreq))*0.01 {
		t.Errorf("Sine wave should be imaginary in frequency domain, got %v", posFreq)
	}

	t.Logf("Sine wave FFT: pos_freq=%v (mag=%f), neg_freq=%v (mag=%f), expected_mag=%f",
		posFreq, posMag, negFreq, cmplx.Abs(negFreq), expectedMag)
}

func testCosineWavePrecision(t *testing.T) {
	t.Helper()

	n := 1024
	freq := 10

	// Generate cosine wave
	data := make([]complex128, n)
	for i := range data {
		val := math.Cos(2 * math.Pi * float64(freq) * float64(i) / float64(n))
		data[i] = complex(val, 0)
	}

	// Perform FFT
	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	output := make([]complex128, n)
	if err := plan.Forward(output, data); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Cosine wave should have peaks at +freq and -freq
	// Peak magnitude should be n/2 (real component)
	expectedMag := float64(n) / 2.0

	posFreq := output[freq]
	posMag := cmplx.Abs(posFreq)

	if math.Abs(posMag-expectedMag) > 1.0 {
		t.Errorf("Positive frequency peak magnitude %f, expected ~%f", posMag, expectedMag)
	}

	// Check it's predominantly real (cosine)
	if math.Abs(imag(posFreq)) > math.Abs(real(posFreq))*0.01 {
		t.Errorf("Cosine wave should be real in frequency domain, got %v", posFreq)
	}

	t.Logf("Cosine wave FFT: pos_freq=%v (mag=%f), expected_mag=%f", posFreq, posMag, expectedMag)
}

func testImpulsePrecision(t *testing.T) {
	t.Helper()

	n := 512

	// Impulse at position 0
	data := make([]complex128, n)
	data[0] = complex(1.0, 0.0)

	// Perform FFT
	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	output := make([]complex128, n)
	if err := plan.Forward(output, data); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// FFT of impulse should be constant (all values = 1.0)
	var maxError float64

	for i, val := range output {
		err := cmplx.Abs(val - complex(1.0, 0.0))
		if err > maxError {
			maxError = err
		}

		if i < 5 {
			t.Logf("data[%d] = %v, error = %e", i, val, err)
		}
	}

	if maxError > 1e-12 {
		t.Errorf("Impulse FFT max error %e exceeds threshold 1e-12", maxError)
	}
}

func testWhiteNoisePrecision(t *testing.T) {
	t.Helper()

	n := 2048

	// Generate white noise
	original := make([]complex128, n)
	for i := range original {
		original[i] = complex(rand.Float64()*2-1, rand.Float64()*2-1)
	}

	// Round-trip test
	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	freq := make([]complex128, n)
	if err := plan.Forward(freq, original); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	roundtrip := make([]complex128, n)
	if err := plan.Inverse(roundtrip, freq); err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	// Check reconstruction
	var maxError float64

	for i := range roundtrip {
		err := cmplx.Abs(roundtrip[i] - original[i])
		if err > maxError {
			maxError = err
		}
	}

	if maxError > 1e-12 {
		t.Errorf("White noise round-trip max error %e exceeds threshold 1e-12", maxError)
	}

	t.Logf("White noise round-trip: max error = %e", maxError)
}

// cmplx64abs returns the absolute value of a complex64.
func cmplx64abs(c complex64) float32 {
	r, i := real(c), imag(c)
	return float32(math.Sqrt(float64(r*r + i*i)))
}

func makePrecisionInputs(n int, rng *rand.Rand) ([]complex64, []complex128) {
	input64 := make([]complex64, n)
	input128 := make([]complex128, n)

	for i := range n {
		re := rng.Float64()*2 - 1
		im := rng.Float64()*2 - 1
		input64[i] = complex(float32(re), float32(im))
		input128[i] = complex(re, im)
	}

	return input64, input128
}

func roundTripMaxError64(t *testing.T, input []complex64) float64 {
	t.Helper()

	n := len(input)

	plan, err := NewPlan(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	src := make([]complex64, n)
	copy(src, input)

	freq := make([]complex64, n)
	if err := plan.Forward(freq, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	roundtrip := make([]complex64, n)
	if err := plan.Inverse(roundtrip, freq); err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	var maxErr float64

	for i := range input {
		err := cmplx64abs(roundtrip[i] - input[i])
		if float64(err) > maxErr {
			maxErr = float64(err)
		}
	}

	return maxErr
}

func roundTripMaxError128(t *testing.T, input []complex128) float64 {
	t.Helper()

	n := len(input)

	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	src := make([]complex128, n)
	copy(src, input)

	freq := make([]complex128, n)
	if err := plan.Forward(freq, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	roundtrip := make([]complex128, n)
	if err := plan.Inverse(roundtrip, freq); err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	var maxErr float64

	for i := range input {
		err := cmplx.Abs(roundtrip[i] - input[i])
		if err > maxErr {
			maxErr = err
		}
	}

	return maxErr
}
