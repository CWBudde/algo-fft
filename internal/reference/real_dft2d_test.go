package reference

import (
	"math"
	"testing"
)

const realTolerance = 1e-5

// TestRealDFT2D_SingleElement tests RealDFT2D with a 1×1 matrix.
func TestRealDFT2D_SingleElement(t *testing.T) {
	t.Parallel()

	input := []float32{5.0}
	result := RealDFT2D(input, 1, 1)

	if len(result) != 1 {
		t.Fatalf("RealDFT2D 1×1 returned %d elements, want 1", len(result))
	}

	// Single element should equal itself
	if math.Abs(float64(real(result[0]))-5.0) > realTolerance {
		t.Errorf("RealDFT2D([5.0]) = %v, want [5.0]", result[0])
	}
}

// TestRealDFT2D_Zeros tests that DFT of zeros is zero.
func TestRealDFT2D_Zeros(t *testing.T) {
	t.Parallel()

	input := make([]float32, 4*6)
	result := RealDFT2D(input, 4, 6)

	halfCols := 6/2 + 1
	expected := 4 * halfCols

	if len(result) != expected {
		t.Fatalf("RealDFT2D zeros: got %d elements, want %d", len(result), expected)
	}

	// All zeros in should give all zeros out
	for i, val := range result {
		if math.Abs(float64(real(val))) > realTolerance || math.Abs(float64(imag(val))) > realTolerance {
			t.Errorf("RealDFT2D(zeros)[%d] = %v, want [0+0i]", i, val)
		}
	}
}

// TestRealDFT2D_OutputShape tests output dimensions are correct.
func TestRealDFT2D_OutputShape(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		rows, cols int
	}{
		{"2×4", 2, 4},
		{"3×8", 3, 8},
		{"5×6", 5, 6},
		{"8×16", 8, 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			input := make([]float32, tt.rows*tt.cols)
			for i := range input {
				input[i] = float32(i)
			}

			result := RealDFT2D(input, tt.rows, tt.cols)

			halfCols := tt.cols/2 + 1
			expected := tt.rows * halfCols

			if len(result) != expected {
				t.Errorf("RealDFT2D(%d×%d) got %d elements, want %d", tt.rows, tt.cols, len(result), expected)
			}
		})
	}
}

// TestRealDFT2D_RoundTrip tests that IDFT(DFT(x)) ≈ x.
func TestRealDFT2D_RoundTrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		rows, cols int
	}{
		{"2×4", 2, 4},
		{"3×8", 3, 8},
		{"4×6", 4, 6},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			// Create test input
			input := make([]float32, tt.rows*tt.cols)
			for i := range input {
				input[i] = float32(i)*0.1 - 5.0
			}

			// Forward transform
			spectrum := RealDFT2D(input, tt.rows, tt.cols)

			// Inverse transform
			output := RealIDFT2D(spectrum, tt.rows, tt.cols)

			// Check output length
			if len(output) != len(input) {
				t.Fatalf("RealIDFT2D returned %d elements, want %d", len(output), len(input))
			}

			// Check each element is close
			for i := range input {
				if math.Abs(float64(output[i])-float64(input[i])) > 1e-4 {
					t.Errorf("RoundTrip[%d]: got %f, want %f (diff: %f)",
						i, output[i], input[i], math.Abs(float64(output[i])-float64(input[i])))
				}
			}
		})
	}
}

// TestRealDFT2D_DC tests DC component (all ones input).
func TestRealDFT2D_DC(t *testing.T) {
	t.Parallel()

	rows, cols := 3, 4

	input := make([]float32, rows*cols)
	for i := range input {
		input[i] = 1.0 // All ones
	}

	result := RealDFT2D(input, rows, cols)

	// DC component should be sum of all elements
	dcExpected := float32(rows * cols)
	if math.Abs(float64(real(result[0]))-float64(dcExpected)) > realTolerance {
		t.Errorf("DC component = %v, want %v", real(result[0]), dcExpected)
	}

	// Check imaginary part of DC is near zero
	if math.Abs(float64(imag(result[0]))) > realTolerance {
		t.Errorf("DC imaginary part = %v, want ≈0", imag(result[0]))
	}
}

// TestRealDFT2D_RealInputSymmetry tests hermitian symmetry property of real FFT.
func TestRealDFT2D_RealInputSymmetry(t *testing.T) {
	t.Parallel()

	rows, cols := 2, 4

	input := make([]float32, rows*cols)
	for i := range input {
		input[i] = float32(i) * 0.5
	}

	result := RealDFT2D(input, rows, cols)
	halfCols := cols/2 + 1

	// The output should have reduced size due to hermitian symmetry
	// 2 × 4 real input -> 2 × (4/2 + 1) = 2 × 3 complex output
	if len(result) != rows*halfCols {
		t.Errorf("RealDFT2D output size = %d, want %d", len(result), rows*halfCols)
	}

	// Verify DC component is reasonable (should be sum-like)
	if real(result[0]) <= 0 {
		t.Errorf("RealDFT2D DC component should be positive, got %v", result[0])
	}
}

// TestRealIDFT2D_PanicOnMismatch tests that IDFT panics on size mismatch.
func TestRealIDFT2D_PanicOnMismatch(t *testing.T) {
	t.Parallel()

	// Spectrum has wrong size
	spectrum := make([]complex64, 12) // Wrong size for 3×8 inverse

	defer func() {
		if r := recover(); r == nil {
			t.Error("RealIDFT2D should panic on spectrum length mismatch")
		}
	}()

	RealIDFT2D(spectrum, 3, 8)
}

// TestRealDFT2D_PanicOnMismatch tests that DFT panics on size mismatch.
func TestRealDFT2D_PanicOnMismatch(t *testing.T) {
	t.Parallel()

	// Input has wrong size
	input := make([]float32, 10) // Wrong size for 3×8

	defer func() {
		if r := recover(); r == nil {
			t.Error("RealDFT2D should panic on input length mismatch")
		}
	}()

	RealDFT2D(input, 3, 8)
}

// TestRealDFT2D_Linearity tests linearity: DFT(a*x + b*y) = a*DFT(x) + b*DFT(y).
func TestRealDFT2D_Linearity(t *testing.T) {
	t.Parallel()

	rows, cols := 2, 4

	// Create two input signals
	x := make([]float32, rows*cols)

	y := make([]float32, rows*cols)
	for i := range x {
		x[i] = float32(i) * 0.5
		y[i] = float32(i)*0.2 + 1.0
	}

	// Compute DFT separately
	dftX := RealDFT2D(x, rows, cols)
	dftY := RealDFT2D(y, rows, cols)

	// Compute combined signal
	a, b := float32(2.0), float32(3.0)

	combined := make([]float32, rows*cols)
	for i := range x {
		combined[i] = a*x[i] + b*y[i]
	}

	// DFT of combined should equal a*DFT(x) + b*DFT(y)
	dftCombined := RealDFT2D(combined, rows, cols)

	halfCols := cols/2 + 1
	for i := range rows * halfCols {
		aComplex := complex(float64(a), 0)
		bComplex := complex(float64(b), 0)
		expected := complex64(aComplex*complex128(dftX[i]) + bComplex*complex128(dftY[i]))

		realDiff := math.Abs(float64(real(dftCombined[i]) - real(expected)))
		imagDiff := math.Abs(float64(imag(dftCombined[i]) - imag(expected)))

		if realDiff > 1e-4 || imagDiff > 1e-4 {
			t.Errorf("Linearity check failed at [%d]: got %v, want %v", i, dftCombined[i], expected)
		}
	}
}

// TestRealDFT2D_EvenCols tests even and odd column sizes.
func TestRealDFT2D_EvenCols(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cols int
	}{
		{"Even cols", 8},
		{"Odd cols", 7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			rows := 3

			input := make([]float32, rows*tt.cols)
			for i := range input {
				input[i] = float32(i) * 0.5
			}

			result := RealDFT2D(input, rows, tt.cols)

			halfCols := tt.cols/2 + 1
			expected := rows * halfCols

			if len(result) != expected {
				t.Errorf("RealDFT2D cols=%d: got %d elements, want %d", tt.cols, len(result), expected)
			}
		})
	}
}

// TestRealIDFT2D_RoundTrip tests that IDFT(DFT(x)) ≈ x for various inputs.
func TestRealIDFT2D_RoundTrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		rows, cols int
		inputFunc  func(int) float32
	}{
		{"Zeros", 3, 4, func(i int) float32 { return 0 }},
		{"Ones", 2, 6, func(i int) float32 { return 1 }},
		{"Ramp", 4, 8, func(i int) float32 { return float32(i) * 0.1 }},
		{"Sine", 3, 5, func(i int) float32 { return float32(math.Sin(float64(i) * 0.5)) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			input := make([]float32, tt.rows*tt.cols)
			for i := range input {
				input[i] = tt.inputFunc(i)
			}

			spectrum := RealDFT2D(input, tt.rows, tt.cols)
			output := RealIDFT2D(spectrum, tt.rows, tt.cols)

			for i := range input {
				if math.Abs(float64(output[i])-float64(input[i])) > 1e-4 {
					t.Errorf("%s RoundTrip[%d]: got %f, want %f", tt.name, i, output[i], input[i])
				}
			}
		})
	}
}

// TestRealDFT2D_Parseval tests energy conservation (Parseval's theorem).
func TestRealDFT2D_Parseval(t *testing.T) {
	t.Parallel()

	rows, cols := 3, 4

	input := make([]float32, rows*cols)
	for i := range input {
		input[i] = float32(i) * 0.5
	}

	// Time domain energy
	energyTime := float64(0)
	for _, v := range input {
		energyTime += float64(v) * float64(v)
	}

	// Frequency domain energy
	spectrum := RealDFT2D(input, rows, cols)
	energyFreq := float64(0)
	halfCols := cols/2 + 1

	for k := range rows {
		for l := range halfCols {
			idx := k*halfCols + l
			mag := real(spectrum[idx])*real(spectrum[idx]) + imag(spectrum[idx])*imag(spectrum[idx])
			// Account for symmetry of real FFT (except DC and Nyquist)
			if l > 0 && l < cols/2 {
				mag *= 2
			}

			energyFreq += float64(mag)
		}
	}

	energyFreq /= float64(rows * cols)

	if math.Abs(energyTime-energyFreq) > 1e-2 {
		t.Errorf("Parseval: time energy %f, freq energy %f, diff %f", energyTime, energyFreq, math.Abs(energyTime-energyFreq))
	}
}
