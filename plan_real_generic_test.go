package algofft

import (
	"math"
	"math/cmplx"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestPlanReal64_Correctness tests float64 real FFT correctness against naive DFT.
func TestPlanReal64_Correctness(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 32, 64, 128, 256}

	for _, n := range sizes {
		t.Run("Size"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanReal64(n)
			if err != nil {
				t.Fatalf("NewPlanReal64(%d) failed: %v", n, err)
			}

			// Test impulse
			input := make([]float64, n)
			input[0] = 1.0

			output := make([]complex128, plan.SpectrumLen())

			err = plan.Forward(output, input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Compare with reference DFT
			ref := reference.NaiveDFT128(complexify64(input))

			// Check half-spectrum (0 to N/2)
			for k := range output {
				refVal := ref[k]
				gotVal := output[k]

				if !complexNear128(refVal, gotVal, 1e-11) {
					t.Errorf("bin[%d]: got %v, want %v (diff=%g)", k, gotVal, refVal, cmplx.Abs(gotVal-refVal))
				}
			}
		})
	}
}

// TestPlanReal64_RoundTrip tests float64 inverse correctness.
func TestPlanReal64_RoundTrip(t *testing.T) {
	t.Parallel()

	sizes := []int{16, 32, 64, 128, 256, 1024}

	for _, n := range sizes {
		t.Run("Size"+itoa(n), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanReal64(n)
			if err != nil {
				t.Fatalf("NewPlanReal64(%d) failed: %v", n, err)
			}

			// Create test signal (mix of frequencies)
			input := make([]float64, n)
			for i := range input {
				t := float64(i) / float64(n)
				input[i] = math.Sin(2*math.Pi*3*t) + 0.5*math.Cos(2*math.Pi*7*t)
			}

			// Forward transform
			spectrum := make([]complex128, plan.SpectrumLen())

			err = plan.Forward(spectrum, input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Inverse transform
			recovered := make([]float64, n)

			err = plan.Inverse(recovered, spectrum)
			if err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			// Check round-trip error
			maxErr := 0.0

			for i := range input {
				err := math.Abs(recovered[i] - input[i])
				if err > maxErr {
					maxErr = err
				}
			}

			// float64 should have much better precision than float32
			tolerance := 1e-11
			if maxErr > tolerance {
				t.Errorf("Round-trip error too high: %g (max tolerance: %g)", maxErr, tolerance)
			}
		})
	}
}

// TestPlanReal64_vsFloat32_Precision compares error accumulation.
func TestPlanReal64_vsFloat32_Precision(t *testing.T) {
	t.Parallel()

	n := 1024

	// Create test signal
	input32 := make([]float32, n)

	input64 := make([]float64, n)
	for i := range n {
		val := math.Sin(2 * math.Pi * float64(i) / float64(n) * 17) // 17 cycles
		input32[i] = float32(val)
		input64[i] = val
	}

	// Float32 path
	plan32, err := NewPlanReal32(n)
	if err != nil {
		t.Fatalf("NewPlanReal32 failed: %v", err)
	}

	spectrum32 := make([]complex64, plan32.SpectrumLen())

	err = plan32.Forward(spectrum32, input32)
	if err != nil {
		t.Fatalf("Forward32 failed: %v", err)
	}

	recovered32 := make([]float32, n)

	err = plan32.Inverse(recovered32, spectrum32)
	if err != nil {
		t.Fatalf("Inverse32 failed: %v", err)
	}

	// Float64 path
	plan64, err := NewPlanReal64(n)
	if err != nil {
		t.Fatalf("NewPlanReal64 failed: %v", err)
	}

	spectrum64 := make([]complex128, plan64.SpectrumLen())

	err = plan64.Forward(spectrum64, input64)
	if err != nil {
		t.Fatalf("Forward64 failed: %v", err)
	}

	recovered64 := make([]float64, n)

	err = plan64.Inverse(recovered64, spectrum64)
	if err != nil {
		t.Fatalf("Inverse64 failed: %v", err)
	}

	// Measure round-trip errors
	maxErr32 := float32(0.0)

	for i := range n {
		err := abs32(recovered32[i] - input32[i])
		if err > maxErr32 {
			maxErr32 = err
		}
	}

	maxErr64 := 0.0

	for i := range n {
		err := math.Abs(recovered64[i] - input64[i])
		if err > maxErr64 {
			maxErr64 = err
		}
	}

	t.Logf("float32 round-trip error: %g", maxErr32)
	t.Logf("float64 round-trip error: %g", maxErr64)

	// Verify both are within expected bounds
	if maxErr32 > 1e-5 {
		t.Errorf("float32 error too high: %g", maxErr32)
	}

	if maxErr64 > 1e-11 {
		t.Errorf("float64 error too high: %g", maxErr64)
	}

	// Verify float64 is significantly more accurate
	if maxErr64 >= float64(maxErr32)*1e-4 {
		t.Errorf("float64 not significantly more accurate: f32=%g, f64=%g", maxErr32, maxErr64)
	}
}

// TestPlanReal64_ConjugateSymmetry verifies output has conjugate symmetry.
func TestPlanReal64_ConjugateSymmetry(t *testing.T) {
	t.Parallel()

	n := 256

	plan, err := NewPlanReal64(n)
	if err != nil {
		t.Fatalf("NewPlanReal64 failed: %v", err)
	}

	// Random real signal
	input := make([]float64, n)
	for i := range input {
		input[i] = math.Sin(float64(i) * 0.123)
	}

	// Get half-spectrum
	halfSpectrum := make([]complex128, plan.SpectrumLen())

	err = plan.Forward(halfSpectrum, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Get full spectrum via complex FFT for comparison
	inputComplex := make([]complex128, n)
	for i := range input {
		inputComplex[i] = complex(input[i], 0)
	}

	planComplex, err := NewPlan64(n)
	if err != nil {
		t.Fatalf("NewPlan64 failed: %v", err)
	}

	fullSpectrum := make([]complex128, n)

	err = planComplex.Forward(fullSpectrum, inputComplex)
	if err != nil {
		t.Fatalf("Complex Forward failed: %v", err)
	}

	// Verify half-spectrum matches first N/2+1 bins of full spectrum
	for k := range halfSpectrum {
		if !complexNear128(halfSpectrum[k], fullSpectrum[k], 1e-11) {
			t.Errorf("bin[%d]: half=%v, full=%v", k, halfSpectrum[k], fullSpectrum[k])
		}
	}

	// Verify conjugate symmetry in full spectrum
	for k := 1; k < n/2; k++ {
		expected := cmplx.Conj(fullSpectrum[n-k])
		got := fullSpectrum[k]

		if !complexNear128(expected, got, 1e-11) {
			t.Errorf("Symmetry violation at k=%d: X[%d]=%v, conj(X[%d])=%v", k, k, got, n-k, expected)
		}
	}
}

// TestPlanReal64_DCandNyquist verifies DC and Nyquist are purely real.
func TestPlanReal64_DCandNyquist(t *testing.T) {
	t.Parallel()

	n := 128

	plan, err := NewPlanReal64(n)
	if err != nil {
		t.Fatalf("NewPlanReal64 failed: %v", err)
	}

	// Constant signal (DC only)
	input := make([]float64, n)
	for i := range input {
		input[i] = 2.5
	}

	output := make([]complex128, plan.SpectrumLen())

	err = plan.Forward(output, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// DC should be real
	if math.Abs(imag(output[0])) > 1e-12 {
		t.Errorf("DC has imaginary part: %v", output[0])
	}

	// Nyquist should be real
	if math.Abs(imag(output[n/2])) > 1e-12 {
		t.Errorf("Nyquist has imaginary part: %v", output[n/2])
	}
}

// TestPlanReal64_Normalized tests normalized forward transform.
func TestPlanReal64_Normalized(t *testing.T) {
	t.Parallel()

	n := 64

	plan, err := NewPlanReal64(n)
	if err != nil {
		t.Fatalf("NewPlanReal64 failed: %v", err)
	}

	input := make([]float64, n)
	for i := range input {
		input[i] = 1.0
	}

	outputNormal := make([]complex128, plan.SpectrumLen())

	err = plan.Forward(outputNormal, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	outputNormalized := make([]complex128, plan.SpectrumLen())

	err = plan.ForwardNormalized(outputNormalized, input)
	if err != nil {
		t.Fatalf("ForwardNormalized failed: %v", err)
	}

	// Check scaling factor
	scale := 1.0 / float64(n)
	for k := range outputNormalized {
		expected := complex(real(outputNormal[k])*scale, imag(outputNormal[k])*scale)
		if !complexNear128(outputNormalized[k], expected, 1e-12) {
			t.Errorf("bin[%d]: got %v, want %v", k, outputNormalized[k], expected)
		}
	}
}

// TestPlanReal64_ZeroAlloc verifies zero allocations during transform.
//
//nolint:paralleltest
func TestPlanReal64_ZeroAlloc(t *testing.T) {
	// Note: t.Parallel() cannot be used here because testing.AllocsPerRun
	// panics when called during a parallel test.
	n := 256

	plan, err := NewPlanReal64(n)
	if err != nil {
		t.Fatalf("NewPlanReal64 failed: %v", err)
	}

	input := make([]float64, n)
	output := make([]complex128, plan.SpectrumLen())

	// Warm up - run multiple times to ensure runtime is stable
	for range 10 {
		_ = plan.Forward(output, input)
	}

	// Test allocations - use 200 runs for better statistical accuracy
	allocs := testing.AllocsPerRun(200, func() {
		_ = plan.Forward(output, input)
	})

	// Allow tiny tolerance for GC/runtime noise (< 0.01 allocs per run)
	if allocs >= 0.01 {
		t.Errorf("Forward allocates %f times per run, want < 0.01", allocs)
	}

	recovered := make([]float64, n)

	// Warm up inverse
	for range 10 {
		_ = plan.Inverse(recovered, output)
	}

	allocs = testing.AllocsPerRun(200, func() {
		_ = plan.Inverse(recovered, output)
	})

	if allocs > 0 {
		t.Errorf("Inverse allocates %f times per run, want 0", allocs)
	}
}

// Helper functions

func complexify64(realData []float64) []complex128 {
	result := make([]complex128, len(realData))
	for i, v := range realData {
		result[i] = complex(v, 0)
	}

	return result
}

func complexNear128(a, b complex128, tol float64) bool {
	return cmplx.Abs(a-b) < tol
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}

	return x
}
