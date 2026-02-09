package transform

import (
	"math"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestRecursiveFFTCorrectness validates recursive FFT against reference DFT.
func TestRecursiveFFTCorrectness(t *testing.T) {
	t.Parallel()

	// Test sizes that should decompose well
	sizes := []int{
		512,   // Single codelet
		1024,  // 2 × 512 codelets
		2048,  // 4 × 512 codelets
		4096,  // 8 × 512 codelets
		8192,  // 16 × 512 codelets
		16384, // 32 × 512 codelets
	}

	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768 // 32 KB L1

	features := cpu.DetectFeatures()

	for _, size := range sizes {
		t.Run(formatSize(size), func(t *testing.T) {
			t.Parallel()

			// Plan decomposition
			strategy := PlanDecomposition(size, codeletSizes, cacheSize)

			// Generate test input: impulse at index 0
			input := make([]complex64, size)
			input[0] = complex(1, 0)

			// Allocate buffers
			output := make([]complex64, size)
			twiddle := TwiddleFactorsRecursive[complex64](strategy)
			scratch := make([]complex64, ScratchSizeRecursive(strategy))

			// Execute recursive FFT
			recursiveForward(output, input, strategy, twiddle, scratch, Registry64, features)

			// Compute reference DFT
			expected := reference.NaiveDFT(input)

			// Compare results
			err := compareComplexSlices(output, expected, 1e-5)
			if err != nil {
				t.Errorf("Size %d forward FFT mismatch: %v", size, err)
				t.Logf("Strategy: SplitFactor=%d, SubSize=%d, Depth=%d, Codelets=%d",
					strategy.SplitFactor, strategy.SubSize, strategy.Depth(), strategy.CodeletCount())
			}
		})
	}
}

// TestRecursiveIFFTCorrectness validates recursive inverse FFT.
func TestRecursiveIFFTCorrectness(t *testing.T) {
	t.Parallel()

	sizes := []int{512, 1024, 2048, 4096}

	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768

	features := cpu.DetectFeatures()

	for _, size := range sizes {
		t.Run(formatSize(size), func(t *testing.T) {
			t.Parallel()

			strategy := PlanDecomposition(size, codeletSizes, cacheSize)

			// Generate random input
			input := make([]complex64, size)
			for i := range input {
				input[i] = complex(float32(i), float32(i*2))
			}

			// Allocate buffers
			forward := make([]complex64, size)
			inverse := make([]complex64, size)
			twiddle := TwiddleFactorsRecursive[complex64](strategy)
			scratch := make([]complex64, ScratchSizeRecursive(strategy))

			// Forward transform
			recursiveForward(forward, input, strategy, twiddle, scratch, Registry64, features)

			// Inverse transform
			recursiveInverse(inverse, forward, strategy, twiddle, scratch, Registry64, features)

			// Should recover original input
			err := compareComplexSlices(inverse, input, 3e-3)
			if err != nil {
				maxDiff, maxIndex := maxDiffComplex64(inverse, input)
				t.Errorf("Size %d round-trip mismatch: %v (max diff=%v at index %d)", size, err, maxDiff, maxIndex)
			}
		})
	}
}

// TestRecursiveFFTParsevalTheorem verifies energy conservation.
func TestRecursiveFFTParsevalTheorem(t *testing.T) {
	t.Parallel()

	size := 2048
	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768

	features := cpu.DetectFeatures()
	strategy := PlanDecomposition(size, codeletSizes, cacheSize)

	// Generate random input
	input := make([]complex64, size)
	for i := range input {
		input[i] = complex(float32(i%10), float32((i*3)%7))
	}

	// Compute energy in time domain
	timeEnergy := float32(0)
	for _, v := range input {
		timeEnergy += real(v)*real(v) + imag(v)*imag(v)
	}

	// Transform
	output := make([]complex64, size)
	twiddle := TwiddleFactorsRecursive[complex64](strategy)
	scratch := make([]complex64, ScratchSizeRecursive(strategy))
	recursiveForward(output, input, strategy, twiddle, scratch, Registry64, features)

	// Compute energy in frequency domain
	freqEnergy := float32(0)
	for _, v := range output {
		freqEnergy += real(v)*real(v) + imag(v)*imag(v)
	}

	// Parseval's theorem (unnormalized FFT): Σ|x[n]|² = (1/N) * Σ|X[k]|²
	// Allow 1% relative error
	expectedFreqEnergy := timeEnergy * float32(size)

	relError := math.Abs(float64(freqEnergy-expectedFreqEnergy)) / float64(expectedFreqEnergy)
	if relError > 0.01 {
		t.Errorf("Parseval's theorem violated: time energy = %v, freq energy = %v (expected %v), rel error = %v",
			timeEnergy, freqEnergy, expectedFreqEnergy, relError)
	}
}

// TestRecursiveFFTLinearity verifies FFT(a*x + b*y) = a*FFT(x) + b*FFT(y).
func TestRecursiveFFTLinearity(t *testing.T) {
	t.Parallel()

	size := 1024
	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768

	features := cpu.DetectFeatures()
	strategy := PlanDecomposition(size, codeletSizes, cacheSize)

	// Generate two input signals
	x := make([]complex64, size)
	y := make([]complex64, size)

	for i := range x {
		x[i] = complex(float32(i), 0)
		y[i] = complex(0, float32(i))
	}

	// Scalars
	a := complex(float32(2), float32(3))
	b := complex(float32(-1), float32(4))

	// Compute FFT(x) and FFT(y)
	fftX := make([]complex64, size)
	fftY := make([]complex64, size)
	twiddle := TwiddleFactorsRecursive[complex64](strategy)
	scratchSize := ScratchSizeRecursive(strategy)
	scratch := make([]complex64, scratchSize)

	recursiveForward(fftX, x, strategy, twiddle, scratch, Registry64, features)
	recursiveForward(fftY, y, strategy, twiddle, scratch, Registry64, features)

	// Compute a*FFT(x) + b*FFT(y)
	expected := make([]complex64, size)
	for i := range expected {
		expected[i] = a*fftX[i] + b*fftY[i]
	}

	// Compute a*x + b*y
	combined := make([]complex64, size)
	for i := range combined {
		combined[i] = a*x[i] + b*y[i]
	}

	// Compute FFT(a*x + b*y)
	actual := make([]complex64, size)
	recursiveForward(actual, combined, strategy, twiddle, scratch, Registry64, features)

	// Should match
	err := compareComplexSlicesRel(actual, expected, 1e-1, 1e-7)
	if err != nil {
		maxDiff, maxIndex := maxDiffComplex64(actual, expected)
		rel := float32(0)

		den := cmplx64Abs(expected[maxIndex])
		if den > 0 {
			rel = maxDiff / den
		}

		t.Errorf("Linearity test failed: %v (max diff=%v at index %d, rel=%v)", err, maxDiff, maxIndex, rel)
	}
}

// TestRecursiveFFTComplex128 validates complex128 precision.
func TestRecursiveFFTComplex128(t *testing.T) {
	t.Parallel()

	size := 2048
	codeletSizes := []int{4, 8, 16, 32, 64, 128, 256, 512}
	cacheSize := 32768

	features := cpu.DetectFeatures()
	strategy := PlanDecomposition(size, codeletSizes, cacheSize)

	// Generate test input
	input := make([]complex128, size)
	for i := range input {
		input[i] = complex(float64(i), float64(i*2))
	}

	// Allocate buffers
	output := make([]complex128, size)
	twiddle := TwiddleFactorsRecursive[complex128](strategy)
	scratch := make([]complex128, ScratchSizeRecursive(strategy))

	// Execute FFT
	recursiveForward(output, input, strategy, twiddle, scratch, Registry128, features)

	// Compute reference DFT in complex128 to avoid float32 truncation.
	expected := naiveDFTComplex128(input)

	for i := range output {
		diff := cmplx128Abs(output[i] - expected[i])
		if diff > 1e-6 {
			t.Errorf("Index %d: got %v, want %v, diff=%v", i, output[i], expected[i], diff)
			break
		}
	}
}

// TestRecursiveFFTSmallSizes tests sizes smaller than smallest codelet.
func TestRecursiveFFTSmallSizes(t *testing.T) {
	t.Parallel()

	// These should fall back to generic DIT
	sizes := []int{2, 4, 8}
	codeletSizes := []int{16, 32, 64, 128, 256, 512} // Smallest is 16

	features := cpu.DetectFeatures()

	for _, size := range sizes {
		t.Run(formatSize(size), func(t *testing.T) {
			t.Parallel()

			strategy := PlanDecomposition(size, codeletSizes, 32768)

			// Should be marked as codelet (will use fallback)
			if !strategy.UseCodelet {
				t.Errorf("Size %d should use codelet (fallback), got UseCodelet=false", size)
			}

			input := make([]complex64, size)
			input[0] = complex(1, 0)

			output := make([]complex64, size)
			twiddle := TwiddleFactorsRecursive[complex64](strategy)
			scratch := make([]complex64, ScratchSizeRecursive(strategy))

			recursiveForward(output, input, strategy, twiddle, scratch, Registry64, features)

			// Compare against reference
			expected := reference.NaiveDFT(input)

			err := compareComplexSlices(output, expected, 1e-5)
			if err != nil {
				t.Errorf("Size %d FFT mismatch: %v", size, err)
			}
		})
	}
}

// Helper functions

func formatSize(n int) string {
	if n >= 1024 {
		return formatInt(n/1024) + "K"
	}

	return formatInt(n)
}

func formatInt(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}

	return string(rune('0'+n/10)) + string(rune('0'+n%10))
}

func compareComplexSlices(a, b []complex64, tolerance float32) error {
	if len(a) != len(b) {
		return &compareError{msg: "length mismatch"}
	}

	for i := range a {
		diff := cmplx64Abs(a[i] - b[i])
		if diff > tolerance {
			return &compareError{
				msg:   "value mismatch",
				index: i,
				got:   a[i],
				want:  b[i],
				diff:  diff,
			}
		}
	}

	return nil
}

type compareError struct {
	msg   string
	index int
	got   complex64
	want  complex64
	diff  float32
}

func (e *compareError) Error() string {
	if e.index > 0 {
		return e.msg + " at index " + formatInt(e.index)
	}

	return e.msg
}

func cmplx64Abs(x complex64) float32 {
	r := real(x)
	i := imag(x)

	return float32(math.Sqrt(float64(r*r + i*i)))
}

func cmplx128Abs(x complex128) float64 {
	r := real(x)
	i := imag(x)

	return math.Sqrt(r*r + i*i)
}

func naiveDFTComplex128(input []complex128) []complex128 {
	n := len(input)
	if n == 0 {
		return nil
	}

	output := make([]complex128, n)
	for k := range n {
		var sum complex128

		for t := range n {
			angle := -2.0 * math.Pi * float64(t*k) / float64(n)
			w := complex(math.Cos(angle), math.Sin(angle))
			sum += input[t] * w
		}

		output[k] = sum
	}

	return output
}

func maxDiffComplex64(a, b []complex64) (float32, int) {
	maxDiff := float32(0)

	maxIndex := 0
	if len(a) < len(b) {
		maxIndex = len(a) - 1
	}

	for i := 0; i < len(a) && i < len(b); i++ {
		diff := cmplx64Abs(a[i] - b[i])
		if diff > maxDiff {
			maxDiff = diff
			maxIndex = i
		}
	}

	return maxDiff, maxIndex
}

func compareComplexSlicesRel(a, b []complex64, absTol, relTol float32) error {
	if len(a) != len(b) {
		return &compareError{msg: "length mismatch"}
	}

	for i := range a {
		diff := cmplx64Abs(a[i] - b[i])
		if diff <= absTol {
			continue
		}

		den := cmplx64Abs(b[i])
		if den == 0 || diff > relTol*den {
			return &compareError{
				msg:   "value mismatch",
				index: i,
				got:   a[i],
				want:  b[i],
				diff:  diff,
			}
		}
	}

	return nil
}
