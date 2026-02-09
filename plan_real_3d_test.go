package algofft

import (
	"math/rand"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestPlanReal3D_BasicSizes tests 3D real FFT correctness for small sizes against naive DFT.
func TestPlanReal3D_BasicSizes(t *testing.T) {
	t.Parallel()

	sizes := []struct {
		depth, height, width int
	}{
		{2, 2, 2},
		{4, 4, 4},
		{8, 8, 8},
		{4, 4, 8},
		{4, 8, 8},
		{8, 4, 4},
	}

	for _, size := range sizes {
		t.Run(sprintf3d(size.depth, size.height, size.width), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanReal3D(size.depth, size.height, size.width)
			if err != nil {
				t.Fatalf("NewPlanReal3D failed: %v", err)
			}

			// Generate random real input
			input := make([]float32, size.depth*size.height*size.width)
			for i := range input {
				input[i] = rand.Float32()*2 - 1 // Random values in [-1, 1]
			}

			// Compute FFT using optimized implementation
			spectrum := make([]complex64, plan.SpectrumLen())

			err = plan.Forward(spectrum, input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Compute reference DFT
			reference := reference.RealDFT3D(input, size.depth, size.height, size.width)

			// Compare results
			maxError := float32(0)

			for i := range spectrum {
				diff := cabsf32(spectrum[i] - reference[i])
				if diff > maxError {
					maxError = diff
				}
			}

			tolerance := float32(1e-4) * float32(size.depth*size.height*size.width)
			if maxError > tolerance {
				t.Errorf("Forward mismatch: max error = %e (tolerance = %e)", maxError, tolerance)
			}
		})
	}
}

// TestPlanReal3D_RoundTrip tests that Inverse(Forward(x)) ≈ x.
func TestPlanReal3D_RoundTrip(t *testing.T) {
	t.Parallel()

	sizes := []struct {
		depth, height, width int
	}{
		{4, 4, 4},
		{8, 8, 8},
		{16, 16, 16},
		{8, 8, 16},
		{8, 16, 16},
	}

	for _, size := range sizes {
		t.Run(sprintf3d(size.depth, size.height, size.width), func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlanReal3D(size.depth, size.height, size.width)
			if err != nil {
				t.Fatalf("NewPlanReal3D failed: %v", err)
			}

			// Generate random real input
			input := make([]float32, size.depth*size.height*size.width)
			for i := range input {
				input[i] = rand.Float32()*2 - 1
			}

			// Forward transform
			spectrum := make([]complex64, plan.SpectrumLen())

			err = plan.Forward(spectrum, input)
			if err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Inverse transform
			output := make([]float32, size.depth*size.height*size.width)

			err = plan.Inverse(output, spectrum)
			if err != nil {
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

			tolerance := float32(1e-4) * float32(size.depth*size.height*size.width)
			if maxError > tolerance {
				t.Errorf("Round-trip error: max = %e (tolerance = %e)", maxError, tolerance)
			}
		})
	}
}

// TestPlanReal3D_ForwardFull tests the full-spectrum output variant.
func TestPlanReal3D_ForwardFull(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanReal3D(4, 4, 4)
	if err != nil {
		t.Fatalf("NewPlanReal3D failed: %v", err)
	}

	// Generate random real input
	input := make([]float32, 4*4*4)
	for i := range input {
		input[i] = rand.Float32()*2 - 1
	}

	// Compute compact spectrum
	spectrumCompact := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(spectrumCompact, input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Compute full spectrum
	spectrumFull := make([]complex64, 4*4*4)

	err = plan.ForwardFull(spectrumFull, input)
	if err != nil {
		t.Fatalf("ForwardFull failed: %v", err)
	}

	// Verify compact spectrum matches first half of full spectrum
	halfWidth := 4/2 + 1

	for d := range 4 {
		for h := range 4 {
			for w := range halfWidth {
				compact := spectrumCompact[d*4*halfWidth+h*halfWidth+w]

				full := spectrumFull[d*4*4+h*4+w]
				if cabsf32(compact-full) > 1e-5 {
					t.Errorf("Compact/Full mismatch at [%d,%d,%d]: compact=%v, full=%v", d, h, w, compact, full)
				}
			}
		}
	}
}

// TestPlanReal3D_InverseFull tests inverse from full spectrum.
func TestPlanReal3D_InverseFull(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanReal3D(4, 4, 4)
	if err != nil {
		t.Fatalf("NewPlanReal3D failed: %v", err)
	}

	// Generate random real input
	input := make([]float32, 4*4*4)
	for i := range input {
		input[i] = rand.Float32()*2 - 1
	}

	// Forward full
	spectrumFull := make([]complex64, 4*4*4)

	err = plan.ForwardFull(spectrumFull, input)
	if err != nil {
		t.Fatalf("ForwardFull failed: %v", err)
	}

	// Inverse full
	output := make([]float32, 4*4*4)

	err = plan.InverseFull(output, spectrumFull)
	if err != nil {
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

// TestPlanReal3D_ConstantSignal tests DC component for a constant signal.
func TestPlanReal3D_ConstantSignal(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanReal3D(4, 4, 4)
	if err != nil {
		t.Fatalf("NewPlanReal3D failed: %v", err)
	}

	// Constant signal (value = 1.0)
	input := make([]float32, 4*4*4)
	for i := range input {
		input[i] = 1.0
	}

	spectrum := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(spectrum, input)
	if err != nil {
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

// TestPlanReal3D_Linearity tests FFT(a*x + b*y) = a*FFT(x) + b*FFT(y).
func TestPlanReal3D_Linearity(t *testing.T) {
	t.Parallel()

	plan, err := NewPlanReal3D(4, 4, 4)
	if err != nil {
		t.Fatalf("NewPlanReal3D failed: %v", err)
	}

	// Generate two random signals
	x := make([]float32, 4*4*4)
	y := make([]float32, 4*4*4)

	for i := range x {
		x[i] = rand.Float32()*2 - 1
		y[i] = rand.Float32()*2 - 1
	}

	a := float32(0.7)
	b := float32(1.3)

	// Compute FFT(x) and FFT(y)
	fftX := make([]complex64, plan.SpectrumLen())

	fftY := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(fftX, x)
	if err != nil {
		t.Fatalf("Forward(x) failed: %v", err)
	}

	err = plan.Forward(fftY, y)
	if err != nil {
		t.Fatalf("Forward(y) failed: %v", err)
	}

	// Compute a*FFT(x) + b*FFT(y)
	linearCombination := make([]complex64, plan.SpectrumLen())
	for i := range linearCombination {
		linearCombination[i] = complex(a, 0)*fftX[i] + complex(b, 0)*fftY[i]
	}

	// Compute a*x + b*y
	combined := make([]float32, 4*4*4)
	for i := range combined {
		combined[i] = a*x[i] + b*y[i]
	}

	// Compute FFT(a*x + b*y)
	fftCombined := make([]complex64, plan.SpectrumLen())

	err = plan.Forward(fftCombined, combined)
	if err != nil {
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

// TestPlanReal3D_Clone tests that cloned plans work independently.
func TestPlanReal3D_Clone(t *testing.T) {
	t.Parallel()

	plan1, err := NewPlanReal3D(4, 4, 4)
	if err != nil {
		t.Fatalf("NewPlanReal3D failed: %v", err)
	}

	plan2 := plan1.Clone()

	input1 := make([]float32, 4*4*4)
	input2 := make([]float32, 4*4*4)

	for i := range input1 {
		input1[i] = rand.Float32()
		input2[i] = rand.Float32()
	}

	spectrum1 := make([]complex64, plan1.SpectrumLen())
	spectrum2 := make([]complex64, plan2.SpectrumLen())

	err = plan1.Forward(spectrum1, input1)
	if err != nil {
		t.Fatalf("plan1.Forward failed: %v", err)
	}

	err = plan2.Forward(spectrum2, input2)
	if err != nil {
		t.Fatalf("plan2.Forward failed: %v", err)
	}

	// Results should differ (different inputs)
	if equalComplex64Slices(spectrum1, spectrum2) {
		t.Error("Clone produced identical results for different inputs")
	}
}

// TestPlanReal3D_InvalidSizes tests error handling for invalid sizes.
func TestPlanReal3D_InvalidSizes(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		depth, height, width int
		shouldFail           bool
	}{
		{0, 0, 0, true},
		{-1, 4, 4, true},
		{4, -1, 4, true},
		{4, 4, -1, true},
		{4, 4, 7, true}, // Odd width (invalid for real FFT)
		{4, 4, 0, true},
		{4, 4, 4, false}, // Valid
	}

	for _, testCase := range testCases {
		_, err := NewPlanReal3D(testCase.depth, testCase.height, testCase.width)
		if testCase.shouldFail && err == nil {
			t.Errorf("NewPlanReal3D(%d, %d, %d) should fail but didn't", testCase.depth, testCase.height, testCase.width)
		}

		if !testCase.shouldFail && err != nil {
			t.Errorf("NewPlanReal3D(%d, %d, %d) failed unexpectedly: %v", testCase.depth, testCase.height, testCase.width, err)
		}
	}
}

// Helper function for 3D size formatting.
func sprintf3d(d, h, w int) string {
	return itoa(d) + "x" + itoa(h) + "x" + itoa(w)
}
