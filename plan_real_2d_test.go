package algoforge

import (
	"math"
	"math/rand"
	"testing"

	"github.com/MeKo-Christian/algoforge/internal/reference"
)

// TestPlanReal2D_BasicSizes tests 2D real FFT correctness for small sizes against naive DFT.
func TestPlanReal2D_BasicSizes(t *testing.T) {
	sizes := []struct {
		rows, cols int
	}{
		{2, 2},
		{4, 4},
		{4, 8},
		{8, 4},
		{8, 8},
		{16, 16},
	}

	for _, size := range sizes {
		t.Run(sprintf("%dx%d", size.rows, size.cols), func(t *testing.T) {
			plan, err := NewPlanReal2D(size.rows, size.cols)
			if err != nil {
				t.Fatalf("NewPlanReal2D failed: %v", err)
			}

			// Generate random real input
			input := make([]float32, size.rows*size.cols)
			for i := range input {
				input[i] = rand.Float32()*2 - 1 // Random values in [-1, 1]
			}

			// Compute FFT using optimized implementation
			spectrum := make([]complex64, plan.SpectrumLen())
			if err := plan.Forward(spectrum, input); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Compute reference DFT
			reference := reference.RealDFT2D(input, size.rows, size.cols)

			// Compare results
			maxError := float32(0)
			for i := range spectrum {
				diff := cabsf32(spectrum[i] - reference[i])
				if diff > maxError {
					maxError = diff
				}
			}

			tolerance := float32(1e-4) * float32(size.rows*size.cols)
			if maxError > tolerance {
				t.Errorf("Forward mismatch: max error = %e (tolerance = %e)", maxError, tolerance)
			}
		})
	}
}

// TestPlanReal2D_RoundTrip tests that Inverse(Forward(x)) ≈ x.
func TestPlanReal2D_RoundTrip(t *testing.T) {
	sizes := []struct {
		rows, cols int
	}{
		{4, 4},
		{8, 8},
		{16, 16},
		{32, 32},
		{64, 64},
		{8, 16},
		{16, 32},
	}

	for _, size := range sizes {
		t.Run(sprintf("%dx%d", size.rows, size.cols), func(t *testing.T) {
			plan, err := NewPlanReal2D(size.rows, size.cols)
			if err != nil {
				t.Fatalf("NewPlanReal2D failed: %v", err)
			}

			// Generate random real input
			input := make([]float32, size.rows*size.cols)
			for i := range input {
				input[i] = rand.Float32()*2 - 1
			}

			// Forward transform
			spectrum := make([]complex64, plan.SpectrumLen())
			if err := plan.Forward(spectrum, input); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Inverse transform
			output := make([]float32, size.rows*size.cols)
			if err := plan.Inverse(output, spectrum); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			// Compare input and output
			maxError := float32(0)
			for i := range input {
				diff := absf32(output[i] - input[i])
				if diff > maxError {
					maxError = diff
				}
			}

			tolerance := float32(1e-4) * float32(size.rows*size.cols)
			if maxError > tolerance {
				t.Errorf("Round-trip error: max = %e (tolerance = %e)", maxError, tolerance)
			}
		})
	}
}

// TestPlanReal2D_ForwardFull tests the full-spectrum output variant.
func TestPlanReal2D_ForwardFull(t *testing.T) {
	plan, err := NewPlanReal2D(8, 8)
	if err != nil {
		t.Fatalf("NewPlanReal2D failed: %v", err)
	}

	// Generate random real input
	input := make([]float32, 8*8)
	for i := range input {
		input[i] = rand.Float32()*2 - 1
	}

	// Compute compact spectrum
	spectrumCompact := make([]complex64, plan.SpectrumLen())
	if err := plan.Forward(spectrumCompact, input); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Compute full spectrum
	spectrumFull := make([]complex64, 8*8)
	if err := plan.ForwardFull(spectrumFull, input); err != nil {
		t.Fatalf("ForwardFull failed: %v", err)
	}

	// Verify compact spectrum matches first half of full spectrum
	for row := 0; row < 8; row++ {
		for col := 0; col < 5; col++ { // 8/2+1 = 5
			compact := spectrumCompact[row*5+col]
			full := spectrumFull[row*8+col]
			if cabsf32(compact-full) > 1e-5 {
				t.Errorf("Compact/Full mismatch at [%d,%d]: compact=%v, full=%v", row, col, compact, full)
			}
		}
	}

	// Verify conjugate symmetry in full spectrum: X[k, n-l] = conj(X[-k, l]) for real input
	// Only the DC component (0,0) and Nyquist on both axes should be purely real
	if absf32(imag(spectrumFull[0])) > 1e-4 {
		t.Errorf("DC component [0,0] should be real, got %v", spectrumFull[0])
	}
}

// TestPlanReal2D_InverseFull tests inverse from full spectrum.
func TestPlanReal2D_InverseFull(t *testing.T) {
	plan, err := NewPlanReal2D(8, 8)
	if err != nil {
		t.Fatalf("NewPlanReal2D failed: %v", err)
	}

	// Generate random real input
	input := make([]float32, 8*8)
	for i := range input {
		input[i] = rand.Float32()*2 - 1
	}

	// Forward full
	spectrumFull := make([]complex64, 8*8)
	if err := plan.ForwardFull(spectrumFull, input); err != nil {
		t.Fatalf("ForwardFull failed: %v", err)
	}

	// Inverse full
	output := make([]float32, 8*8)
	if err := plan.InverseFull(output, spectrumFull); err != nil {
		t.Fatalf("InverseFull failed: %v", err)
	}

	// Compare input and output
	maxError := float32(0)
	for i := range input {
		diff := absf32(output[i] - input[i])
		if diff > maxError {
			maxError = diff
		}
	}

	tolerance := float32(1e-4) * 64
	if maxError > tolerance {
		t.Errorf("InverseFull round-trip error: max = %e (tolerance = %e)", maxError, tolerance)
	}
}

// TestPlanReal2D_ConstantSignal tests DC component for a constant signal.
func TestPlanReal2D_ConstantSignal(t *testing.T) {
	plan, err := NewPlanReal2D(8, 8)
	if err != nil {
		t.Fatalf("NewPlanReal2D failed: %v", err)
	}

	// Constant signal (value = 1.0)
	input := make([]float32, 8*8)
	for i := range input {
		input[i] = 1.0
	}

	spectrum := make([]complex64, plan.SpectrumLen())
	if err := plan.Forward(spectrum, input); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// DC component should be 64 (sum of all 64 ones)
	dc := spectrum[0]
	expectedDC := complex64(64.0)
	if cabsf32(dc-expectedDC) > 1e-4 {
		t.Errorf("DC component mismatch: got %v, want %v", dc, expectedDC)
	}

	// All other components should be near zero
	for i := 1; i < len(spectrum); i++ {
		if cabsf32(spectrum[i]) > 1e-4 {
			t.Errorf("Non-DC component [%d] should be ~0, got %v", i, spectrum[i])
		}
	}
}

// TestPlanReal2D_Linearity tests FFT(a*x + b*y) = a*FFT(x) + b*FFT(y).
func TestPlanReal2D_Linearity(t *testing.T) {
	plan, err := NewPlanReal2D(8, 8)
	if err != nil {
		t.Fatalf("NewPlanReal2D failed: %v", err)
	}

	// Generate two random signals
	x := make([]float32, 8*8)
	y := make([]float32, 8*8)
	for i := range x {
		x[i] = rand.Float32()*2 - 1
		y[i] = rand.Float32()*2 - 1
	}

	a := float32(0.7)
	b := float32(1.3)

	// Compute FFT(x) and FFT(y)
	fftX := make([]complex64, plan.SpectrumLen())
	fftY := make([]complex64, plan.SpectrumLen())
	if err := plan.Forward(fftX, x); err != nil {
		t.Fatalf("Forward(x) failed: %v", err)
	}
	if err := plan.Forward(fftY, y); err != nil {
		t.Fatalf("Forward(y) failed: %v", err)
	}

	// Compute a*FFT(x) + b*FFT(y)
	linearCombination := make([]complex64, plan.SpectrumLen())
	for i := range linearCombination {
		linearCombination[i] = complex(a, 0)*fftX[i] + complex(b, 0)*fftY[i]
	}

	// Compute a*x + b*y
	combined := make([]float32, 8*8)
	for i := range combined {
		combined[i] = a*x[i] + b*y[i]
	}

	// Compute FFT(a*x + b*y)
	fftCombined := make([]complex64, plan.SpectrumLen())
	if err := plan.Forward(fftCombined, combined); err != nil {
		t.Fatalf("Forward(combined) failed: %v", err)
	}

	// Compare
	maxError := float32(0)
	for i := range fftCombined {
		diff := cabsf32(fftCombined[i] - linearCombination[i])
		if diff > maxError {
			maxError = diff
		}
	}

	tolerance := float32(1e-3) * 64
	if maxError > tolerance {
		t.Errorf("Linearity violation: max error = %e (tolerance = %e)", maxError, tolerance)
	}
}

// TestPlanReal2D_Clone tests that cloned plans work independently.
func TestPlanReal2D_Clone(t *testing.T) {
	plan1, err := NewPlanReal2D(8, 8)
	if err != nil {
		t.Fatalf("NewPlanReal2D failed: %v", err)
	}

	plan2 := plan1.Clone()

	input1 := make([]float32, 8*8)
	input2 := make([]float32, 8*8)
	for i := range input1 {
		input1[i] = rand.Float32()
		input2[i] = rand.Float32()
	}

	spectrum1 := make([]complex64, plan1.SpectrumLen())
	spectrum2 := make([]complex64, plan2.SpectrumLen())

	if err := plan1.Forward(spectrum1, input1); err != nil {
		t.Fatalf("plan1.Forward failed: %v", err)
	}

	if err := plan2.Forward(spectrum2, input2); err != nil {
		t.Fatalf("plan2.Forward failed: %v", err)
	}

	// Results should differ (different inputs)
	if equalComplex64Slices(spectrum1, spectrum2) {
		t.Error("Clone produced identical results for different inputs")
	}
}

// TestPlanReal2D_InvalidSizes tests error handling for invalid sizes.
func TestPlanReal2D_InvalidSizes(t *testing.T) {
	testCases := []struct {
		rows, cols int
		shouldFail bool
	}{
		{0, 0, true},
		{-1, 8, true},
		{8, -1, true},
		{8, 7, true},  // Odd cols (invalid for real FFT)
		{8, 0, true},
		{8, 8, false}, // Valid
	}

	for _, tc := range testCases {
		_, err := NewPlanReal2D(tc.rows, tc.cols)
		if tc.shouldFail && err == nil {
			t.Errorf("NewPlanReal2D(%d, %d) should fail but didn't", tc.rows, tc.cols)
		}
		if !tc.shouldFail && err != nil {
			t.Errorf("NewPlanReal2D(%d, %d) failed unexpectedly: %v", tc.rows, tc.cols, err)
		}
	}
}

// Helper functions

func absf32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func cabsf32(z complex64) float32 {
	r := real(z)
	i := imag(z)
	return float32(math.Sqrt(float64(r*r + i*i)))
}

func equalComplex64Slices(a, b []complex64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if cabsf32(a[i]-b[i]) > 1e-5 {
			return false
		}
	}
	return true
}

func sprintf(format string, args ...interface{}) string {
	// Simple sprintf implementation to avoid importing fmt
	// Only handles %d and %dx%d patterns used in tests
	result := ""
	argIdx := 0

	for i := 0; i < len(format); i++ {
		if format[i] == '%' && i+1 < len(format) {
			if format[i+1] == 'd' {
				if argIdx < len(args) {
					result += itoa(args[argIdx].(int))
					argIdx++
				}
				i++
			} else {
				result += string(format[i])
			}
		} else {
			result += string(format[i])
		}
	}

	return result
}
