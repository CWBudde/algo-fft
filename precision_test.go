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
	plan, err := NewPlan[complex64](n)
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
	plan, err := NewPlan[complex64](n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	output := make([]complex64, n)

	err = plan.Forward(output, data)
	if err != nil {
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

	err = plan.Forward(output, data)
	if err != nil {
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

	// Broadband test signal. This used to be a real sine at bin 5 — a lattice
	// frequency, whose spectrum is two nonzero bins and n-2 zeros. Since the
	// relative-difference loop below skips bins under 1e-10, that input left
	// the comparison checking two bins out of n, and left every twiddle in the
	// transform multiplying a value that rounds away.
	input64 := make([]complex64, n)
	input128 := make([]complex128, n)

	for i := range input64 {
		f := float64(i)
		re := math.Cos(0.7*f) + 0.3*math.Sin(2.9*f) + 0.05*math.Sqrt(f)
		im := math.Sin(1.3*f) - 0.4*math.Cos(0.11*f)
		input64[i] = complex(float32(re), float32(im))
		input128[i] = complex(re, im)
	}

	// Perform FFT with both precisions
	plan64, err := NewPlan[complex64](n)
	if err != nil {
		t.Fatalf("failed to create complex64 plan: %v", err)
	}

	output64 := make([]complex64, n)

	err = plan64.Forward(output64, input64)
	if err != nil {
		t.Fatalf("Forward complex64 failed: %v", err)
	}

	plan128, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create complex128 plan: %v", err)
	}

	output128 := make([]complex128, n)

	err = plan128.Forward(output128, input128)
	if err != nil {
		t.Fatalf("Forward complex128 failed: %v", err)
	}

	// Compare results
	var maxAbsDiff, maxRelDiff, peak float64

	for i := range output64 {
		diff := cmplx.Abs(complex128(output64[i]) - output128[i])
		if diff > maxAbsDiff {
			maxAbsDiff = diff
		}

		mag128 := cmplx.Abs(output128[i])
		peak = math.Max(peak, mag128)

		if mag128 > 1e-10 {
			relDiff := diff / mag128
			if relDiff > maxRelDiff {
				maxRelDiff = relDiff
			}
		}
	}

	t.Logf("Size %d: max abs diff = %e, max rel diff = %e, peak = %e", n, maxAbsDiff, maxRelDiff, peak)

	// Bound relative to the spectrum peak. Now that every bin carries energy,
	// a per-bin relative bound would be set by whichever bin happens to be
	// smallest, which says nothing about the transform: float32 carries ~7
	// decimal digits of the largest term it accumulated, not of each output.
	if maxAbsDiff > 1e-5*peak {
		t.Errorf("Precision difference too large: max abs diff %e vs peak %e (rel %e)",
			maxAbsDiff, peak, maxAbsDiff/peak)
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

// testLargePrecision checks the spectrum at sizes the O(n²) reference cannot
// reach. The input is a sum of complex exponentials at distinct bins, whose
// forward transform is known in closed form (n·amplitude at those bins, zero
// everywhere else) and costs O(n) to construct.
//
// It used to be an impulse, asserting the spectrum was all ones. That measured
// nothing: an impulse multiplies every twiddle by zero, and every permutation
// of an all-ones vector is still all-ones, so a wrong twiddle table and a wrong
// bin order both produce a "max error" of 0.
func testLargePrecision(t *testing.T, n int) {
	t.Helper()

	tones := []struct {
		k int
		a complex128
	}{
		{1, complex(1, 0)},
		{5, complex(0, -2)},
		{n/4 + 3, complex(0.5, 0.25)},
		{n / 2, complex(-1.5, 0)},
		{n - 9, complex(0.75, 1.25)},
	}

	data := make([]complex128, n)
	expected := make([]complex128, n)

	for _, tone := range tones {
		for j := range data {
			// Reduce k*j modulo n before scaling to radians: at these sizes
			// the unreduced angle reaches ~1e6 rad, where argument reduction
			// alone costs ~1e-10 of relative phase accuracy and the "exact"
			// expected spectrum stops being exact.
			phase := 2 * math.Pi * float64((int64(tone.k)*int64(j))%int64(n)) / float64(n)
			data[j] += tone.a * cmplx.Exp(complex(0, phase))
		}

		expected[tone.k] += tone.a * complex(float64(n), 0)
	}

	src := make([]complex128, n)
	copy(src, data)

	plan, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	output := make([]complex128, n)

	err = plan.Forward(output, data)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// The spikes are n·amplitude, so the achievable absolute error scales with
	// n: float64 carries ~16 digits of the largest term accumulated. Measured
	// headroom at 262144 is ~40x.
	tol := 5e-14 * float64(n)

	var maxError float64

	for i, val := range output {
		diff := cmplx.Abs(val - expected[i])
		if diff > maxError {
			maxError = diff
		}

		if diff > tol {
			t.Errorf("size %d: bin %d = %v, want %v (error %e > %e)", n, i, val, expected[i], diff, tol)
			break
		}
	}

	t.Logf("Size %d: max error = %e (tolerance %e)", n, maxError, tol)

	// Test round-trip against the same broadband signal.
	roundtrip := make([]complex128, n)

	err = plan.Inverse(roundtrip, output)
	if err != nil {
		t.Fatalf("Inverse failed: %v", err)
	}

	var maxRoundTripError float64

	for i := range roundtrip {
		diff := cmplx.Abs(roundtrip[i] - src[i])
		if diff > maxRoundTripError {
			maxRoundTripError = diff
		}
	}

	if maxRoundTripError > 1e-10 {
		t.Errorf("Round-trip error %e exceeds threshold 1e-10", maxRoundTripError)
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

	err = plan.Forward(output, data)
	if err != nil {
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

	err = plan.Forward(output, data)
	if err != nil {
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

	err = plan.Forward(output, data)
	if err != nil {
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

	err = plan.Forward(freq, original)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	roundtrip := make([]complex128, n)

	err = plan.Inverse(roundtrip, freq)
	if err != nil {
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

	plan, err := NewPlan[complex64](n)
	if err != nil {
		t.Fatalf("failed to create plan: %v", err)
	}

	src := make([]complex64, n)
	copy(src, input)

	freq := make([]complex64, n)

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	roundtrip := make([]complex64, n)

	err = plan.Inverse(roundtrip, freq)
	if err != nil {
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

	err = plan.Forward(freq, src)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	roundtrip := make([]complex128, n)

	err = plan.Inverse(roundtrip, freq)
	if err != nil {
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
