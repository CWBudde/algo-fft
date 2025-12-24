package reference

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

const tolerance = 1e-5

// generateRandomSignal creates a random complex64 signal of the given length.
// Uses a deterministic seed for reproducibility in tests.
func generateRandomSignal(length int, seed uint64) []complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF)) //nolint:gosec // Intentionally non-crypto for reproducible tests
	signal := make([]complex64, length)

	for i := range signal {
		realPart := float32(rng.Float64()*20 - 10) // Range [-10, 10]
		imagPart := float32(rng.Float64()*20 - 10)
		signal[i] = complex(realPart, imagPart)
	}

	return signal
}

// generateRandomSignal128 creates a random complex128 signal of the given length.
func generateRandomSignal128(length int, seed uint64) []complex128 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF)) //nolint:gosec // Intentionally non-crypto for reproducible tests
	signal := make([]complex128, length)

	for i := range signal {
		realPart := rng.Float64()*20 - 10 // Range [-10, 10]
		imagPart := rng.Float64()*20 - 10
		signal[i] = complex(realPart, imagPart)
	}

	return signal
}

// complexNearlyEqual checks if two complex64 numbers are nearly equal within tolerance.
func complexNearlyEqual(first, second complex64) bool {
	return cmplx.Abs(complex128(first)-complex128(second)) < tolerance
}

// complex128NearlyEqual checks if two complex128 numbers are nearly equal within tolerance.
func complex128NearlyEqual(first, second complex128, tol float64) bool {
	return cmplx.Abs(first-second) < tol
}

func TestNaiveDFT_Empty(t *testing.T) {
	t.Parallel()

	result := NaiveDFT(nil)
	if result != nil {
		t.Errorf("NaiveDFT(nil) = %v, want nil", result)
	}

	result = NaiveDFT([]complex64{})
	if result != nil {
		t.Errorf("NaiveDFT([]) = %v, want nil", result)
	}
}

func TestNaiveIDFT_Empty(t *testing.T) {
	t.Parallel()

	result := NaiveIDFT(nil)
	if result != nil {
		t.Errorf("NaiveIDFT(nil) = %v, want nil", result)
	}

	result = NaiveIDFT([]complex64{})
	if result != nil {
		t.Errorf("NaiveIDFT([]) = %v, want nil", result)
	}
}

func TestNaiveDFT_SingleElement(t *testing.T) {
	t.Parallel()

	// DFT of a single element is the element itself
	input := []complex64{3 + 4i}
	result := NaiveDFT(input)

	if len(result) != 1 {
		t.Fatalf("NaiveDFT returned %d elements, want 1", len(result))
	}

	if !complexNearlyEqual(result[0], 3+4i) {
		t.Errorf("NaiveDFT([3+4i]) = [%v], want [3+4i]", result[0])
	}
}

func TestNaiveIDFT_SingleElement(t *testing.T) {
	t.Parallel()

	// IDFT of a single element is the element itself
	input := []complex64{3 + 4i}
	result := NaiveIDFT(input)

	if len(result) != 1 {
		t.Fatalf("NaiveIDFT returned %d elements, want 1", len(result))
	}

	if !complexNearlyEqual(result[0], 3+4i) {
		t.Errorf("NaiveIDFT([3+4i]) = [%v], want [3+4i]", result[0])
	}
}

func TestNaiveDFT_Impulse(t *testing.T) {
	t.Parallel()

	// DFT of an impulse [1, 0, 0, 0] should be [1, 1, 1, 1]
	input := []complex64{1, 0, 0, 0}
	result := NaiveDFT(input)

	expected := []complex64{1, 1, 1, 1}

	if len(result) != len(expected) {
		t.Fatalf("NaiveDFT returned %d elements, want %d", len(result), len(expected))
	}

	for i := range result {
		if !complexNearlyEqual(result[i], expected[i]) {
			t.Errorf("NaiveDFT(impulse)[%d] = %v, want %v", i, result[i], expected[i])
		}
	}
}

func TestNaiveDFT_Constant(t *testing.T) {
	t.Parallel()

	// DFT of a constant [1, 1, 1, 1] should be [4, 0, 0, 0] (DC component only)
	input := []complex64{1, 1, 1, 1}
	result := NaiveDFT(input)

	expected := []complex64{4, 0, 0, 0}

	if len(result) != len(expected) {
		t.Fatalf("NaiveDFT returned %d elements, want %d", len(result), len(expected))
	}

	for i := range result {
		if !complexNearlyEqual(result[i], expected[i]) {
			t.Errorf("NaiveDFT(constant)[%d] = %v, want %v", i, result[i], expected[i])
		}
	}
}

func TestNaiveDFT_TwoElements(t *testing.T) {
	t.Parallel()

	// DFT of [1, 1] should be [2, 0]
	input := []complex64{1, 1}
	result := NaiveDFT(input)

	expected := []complex64{2, 0}

	for i := range result {
		if !complexNearlyEqual(result[i], expected[i]) {
			t.Errorf("NaiveDFT([1,1])[%d] = %v, want %v", i, result[i], expected[i])
		}
	}

	// DFT of [1, -1] should be [0, 2]
	input = []complex64{1, -1}
	result = NaiveDFT(input)
	expected = []complex64{0, 2}

	for i := range result {
		if !complexNearlyEqual(result[i], expected[i]) {
			t.Errorf("NaiveDFT([1,-1])[%d] = %v, want %v", i, result[i], expected[i])
		}
	}
}

func TestNaiveDFT_Sinusoid(t *testing.T) {
	t.Parallel()

	// A pure cosine at frequency 1 for N=8
	// cos(2*pi*k/N) for k=0..N-1
	fftSize := 8

	input := make([]complex64, fftSize)
	for sampleIdx := range fftSize {
		input[sampleIdx] = complex(float32(math.Cos(2*math.Pi*float64(sampleIdx)/float64(fftSize))), 0)
	}

	result := NaiveDFT(input)

	// For a pure cosine at frequency 1, we expect:
	// - Peak at bin 1 with magnitude N/2 = 4
	// - Peak at bin N-1 (=7) with magnitude N/2 = 4
	// - All other bins should be 0
	for i := range result {
		mag := cmplx.Abs(complex128(result[i]))
		switch i {
		case 1, 7:
			if math.Abs(mag-4.0) > tolerance {
				t.Errorf("NaiveDFT(cos)[%d] magnitude = %v, want 4.0", i, mag)
			}
		default:
			if mag > tolerance {
				t.Errorf("NaiveDFT(cos)[%d] magnitude = %v, want 0", i, mag)
			}
		}
	}
}

func TestNaiveDFT_IDFT_RoundTrip(t *testing.T) {
	t.Parallel()

	// IDFT(DFT(x)) should equal x
	testCases := []struct {
		name  string
		input []complex64
	}{
		{"impulse", []complex64{1, 0, 0, 0}},
		{"constant", []complex64{1, 1, 1, 1}},
		{"alternating", []complex64{1, -1, 1, -1}},
		{"complex", []complex64{1 + 2i, 3 - 1i, -2 + 4i, 0 - 3i}},
		{"size8", []complex64{1, 2, 3, 4, 5, 6, 7, 8}},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			forward := NaiveDFT(testCase.input)
			inverse := NaiveIDFT(forward)

			if len(inverse) != len(testCase.input) {
				t.Fatalf("IDFT(DFT(x)) has length %d, want %d", len(inverse), len(testCase.input))
			}

			for i := range inverse {
				if !complexNearlyEqual(inverse[i], testCase.input[i]) {
					t.Errorf("IDFT(DFT(x))[%d] = %v, want %v", i, inverse[i], testCase.input[i])
				}
			}
		})
	}
}

func TestNaiveDFT128_Empty(t *testing.T) {
	t.Parallel()

	result := NaiveDFT128(nil)
	if result != nil {
		t.Errorf("NaiveDFT128(nil) = %v, want nil", result)
	}
}

func TestNaiveIDFT128_Empty(t *testing.T) {
	t.Parallel()

	result := NaiveIDFT128(nil)
	if result != nil {
		t.Errorf("NaiveIDFT128(nil) = %v, want nil", result)
	}
}

func TestNaiveDFT128_IDFT128_RoundTrip(t *testing.T) {
	t.Parallel()

	// IDFT(DFT(x)) should equal x with high precision
	input := []complex128{1 + 2i, 3 - 1i, -2 + 4i, 0 - 3i, 5.5 + 0.5i, -1.1 - 2.2i, 0, 7.7 - 3.3i}

	forward := NaiveDFT128(input)
	inverse := NaiveIDFT128(forward)

	if len(inverse) != len(input) {
		t.Fatalf("IDFT128(DFT128(x)) has length %d, want %d", len(inverse), len(input))
	}

	// Use tighter tolerance for complex128
	tol := 1e-12
	for i := range inverse {
		if !complex128NearlyEqual(inverse[i], input[i], tol) {
			t.Errorf("IDFT128(DFT128(x))[%d] = %v, want %v", i, inverse[i], input[i])
		}
	}
}

func TestNaiveDFT_Linearity(t *testing.T) {
	t.Parallel()

	// DFT(a*x + b*y) should equal a*DFT(x) + b*DFT(y)
	signalX := []complex64{1, 2, 3, 4}
	signalY := []complex64{4, 3, 2, 1}
	coeffA := complex64(2 + 1i)
	coeffB := complex64(1 - 0.5i)

	// Compute DFT(a*x + b*y)
	combined := make([]complex64, len(signalX))
	for i := range combined {
		combined[i] = coeffA*signalX[i] + coeffB*signalY[i]
	}

	dftCombined := NaiveDFT(combined)

	// Compute a*DFT(x) + b*DFT(y)
	dftX := NaiveDFT(signalX)
	dftY := NaiveDFT(signalY)
	expected := make([]complex64, len(signalX))

	for i := range expected {
		expected[i] = coeffA*dftX[i] + coeffB*dftY[i]
	}

	for i := range dftCombined {
		if !complexNearlyEqual(dftCombined[i], expected[i]) {
			t.Errorf("Linearity failed at [%d]: DFT(ax+by) = %v, a*DFT(x)+b*DFT(y) = %v",
				i, dftCombined[i], expected[i])
		}
	}
}

func TestNaiveDFT_Parseval(t *testing.T) {
	t.Parallel()

	// Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)
	input := []complex64{1 + 2i, 3 - 1i, -2 + 4i, 0 - 3i}
	signalLength := float64(len(input))

	// Time domain energy
	var timeEnergy float64

	for _, value := range input {
		timeEnergy += math.Pow(cmplx.Abs(complex128(value)), 2)
	}

	// Frequency domain energy
	dft := NaiveDFT(input)

	var freqEnergy float64

	for _, value := range dft {
		freqEnergy += math.Pow(cmplx.Abs(complex128(value)), 2)
	}

	freqEnergy /= signalLength

	if math.Abs(timeEnergy-freqEnergy) > tolerance {
		t.Errorf("Parseval's theorem failed: time energy = %v, freq energy/N = %v",
			timeEnergy, freqEnergy)
	}
}

func TestNaiveDFT_TimeShift(t *testing.T) {
	t.Parallel()

	// Time shift theorem: DFT(x[n-k]) = DFT(x[n]) * exp(-2πi*k*m/N)
	// A circular shift in time domain multiplies frequency bins by a phase factor
	fftSize := 8
	shiftAmount := 2

	// Original signal
	original := []complex64{1, 2, 3, 4, 5, 6, 7, 8}

	// Circularly shifted signal (shift right by shiftAmount)
	shifted := make([]complex64, fftSize)
	for i := range shifted {
		srcIdx := (i - shiftAmount + fftSize) % fftSize
		shifted[i] = original[srcIdx]
	}

	dftOriginal := NaiveDFT(original)
	dftShifted := NaiveDFT(shifted)

	// Verify: DFT(shifted)[k] = DFT(original)[k] * exp(-2πi*k*shiftAmount/N)
	for freqBin := range fftSize {
		phaseAngle := -2.0 * math.Pi * float64(freqBin) * float64(shiftAmount) / float64(fftSize)
		phaseFactor := complex(float32(math.Cos(phaseAngle)), float32(math.Sin(phaseAngle)))
		expected := dftOriginal[freqBin] * phaseFactor

		if !complexNearlyEqual(dftShifted[freqBin], expected) {
			t.Errorf("Time shift theorem failed at bin %d: got %v, want %v",
				freqBin, dftShifted[freqBin], expected)
		}
	}
}

func TestNaiveDFT_FrequencyShift(t *testing.T) {
	t.Parallel()

	// Frequency shift theorem: DFT(x[n] * exp(2πi*k0*n/N)) = X[k-k0]
	// Modulating by a complex exponential shifts the spectrum
	fftSize := 8
	freqShift := 2

	// Original signal
	original := []complex64{1, 2, 3, 4, 5, 6, 7, 8}

	// Modulated signal: x[n] * exp(2πi*freqShift*n/N)
	modulated := make([]complex64, fftSize)
	for sampleIdx := range fftSize {
		modAngle := 2.0 * math.Pi * float64(freqShift) * float64(sampleIdx) / float64(fftSize)
		modFactor := complex(float32(math.Cos(modAngle)), float32(math.Sin(modAngle)))
		modulated[sampleIdx] = original[sampleIdx] * modFactor
	}

	dftOriginal := NaiveDFT(original)
	dftModulated := NaiveDFT(modulated)

	// Verify: DFT(modulated)[k] = DFT(original)[k - freqShift] (circular)
	for freqBin := range fftSize {
		origBin := (freqBin - freqShift + fftSize) % fftSize
		expected := dftOriginal[origBin]

		if !complexNearlyEqual(dftModulated[freqBin], expected) {
			t.Errorf("Frequency shift theorem failed at bin %d: got %v, want %v (from bin %d)",
				freqBin, dftModulated[freqBin], expected, origBin)
		}
	}
}

func TestNaiveDFT_Convolution(t *testing.T) {
	t.Parallel()

	// Convolution theorem: DFT(x * y) = DFT(x) .* DFT(y)
	// where * is circular convolution and .* is element-wise multiplication
	fftSize := 8

	signalX := []complex64{1, 2, 3, 4, 0, 0, 0, 0}
	signalY := []complex64{1, 1, 1, 1, 0, 0, 0, 0}

	// Compute circular convolution in time domain
	convolved := make([]complex64, fftSize)
	for outputIdx := range fftSize {
		var sum complex64

		for inputIdx := range fftSize {
			yIdx := (outputIdx - inputIdx + fftSize) % fftSize
			sum += signalX[inputIdx] * signalY[yIdx]
		}

		convolved[outputIdx] = sum
	}

	// Compute via frequency domain
	dftX := NaiveDFT(signalX)
	dftY := NaiveDFT(signalY)

	dftProduct := make([]complex64, fftSize)
	for i := range dftProduct {
		dftProduct[i] = dftX[i] * dftY[i]
	}

	convolvedViaFreq := NaiveIDFT(dftProduct)

	// Compare results
	for i := range fftSize {
		if !complexNearlyEqual(convolved[i], convolvedViaFreq[i]) {
			t.Errorf("Convolution theorem failed at [%d]: time domain = %v, freq domain = %v",
				i, convolved[i], convolvedViaFreq[i])
		}
	}
}

func TestNaiveDFT_RandomSignals(t *testing.T) {
	t.Parallel()

	// Test round-trip with various random signals
	testCases := []struct {
		name   string
		length int
		seed   uint64
	}{
		{"size4_seed1", 4, 1},
		{"size8_seed42", 8, 42},
		{"size16_seed123", 16, 123},
		{"size32_seed999", 32, 999},
		{"size64_seed7777", 64, 7777},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			input := generateRandomSignal(testCase.length, testCase.seed)
			forward := NaiveDFT(input)
			inverse := NaiveIDFT(forward)

			for i := range inverse {
				if !complexNearlyEqual(inverse[i], input[i]) {
					t.Errorf("Round-trip failed at [%d]: got %v, want %v",
						i, inverse[i], input[i])
				}
			}
		})
	}
}

func TestNaiveDFT128_RandomSignals(t *testing.T) {
	t.Parallel()

	// Test round-trip with random signals using high precision
	testCases := []struct {
		name   string
		length int
		seed   uint64
	}{
		{"size4_seed1", 4, 1},
		{"size16_seed42", 16, 42},
		{"size64_seed123", 64, 123},
	}

	tol := 1e-12

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			input := generateRandomSignal128(testCase.length, testCase.seed)
			forward := NaiveDFT128(input)
			inverse := NaiveIDFT128(forward)

			for i := range inverse {
				if !complex128NearlyEqual(inverse[i], input[i], tol) {
					t.Errorf("Round-trip failed at [%d]: got %v, want %v",
						i, inverse[i], input[i])
				}
			}
		})
	}
}

func TestNaiveDFT_ParsevalRandomSignals(t *testing.T) {
	t.Parallel()

	// Test Parseval's theorem with random signals
	testCases := []struct {
		name   string
		length int
		seed   uint64
	}{
		{"size8_seed1", 8, 1},
		{"size16_seed42", 16, 42},
		{"size32_seed123", 32, 123},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			input := generateRandomSignal(testCase.length, testCase.seed)
			signalLength := float64(len(input))

			var timeEnergy float64

			for _, value := range input {
				timeEnergy += math.Pow(cmplx.Abs(complex128(value)), 2)
			}

			dft := NaiveDFT(input)

			var freqEnergy float64

			for _, value := range dft {
				freqEnergy += math.Pow(cmplx.Abs(complex128(value)), 2)
			}

			freqEnergy /= signalLength

			// Use slightly relaxed tolerance for random signals due to float32 precision
			relaxedTol := 1e-4
			if math.Abs(timeEnergy-freqEnergy) > relaxedTol {
				t.Errorf("Parseval's theorem failed: time energy = %v, freq energy/N = %v",
					timeEnergy, freqEnergy)
			}
		})
	}
}
