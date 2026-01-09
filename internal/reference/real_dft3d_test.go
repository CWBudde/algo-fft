package reference

import (
	"math"
	"testing"
)

// TestRealDFT3D_SingleElement tests RealDFT3D with a 1×1×1 volume.
func TestRealDFT3D_SingleElement(t *testing.T) {
	t.Parallel()

	input := []float32{7.0}
	result := RealDFT3D(input, 1, 1, 1)

	if len(result) != 1 {
		t.Fatalf("RealDFT3D 1×1×1 returned %d elements, want 1", len(result))
	}

	// Single element should equal itself
	if math.Abs(float64(real(result[0]))-7.0) > realTolerance {
		t.Errorf("RealDFT3D([7.0]) = %v, want [7.0]", result[0])
	}
}

// TestRealDFT3D_Zeros tests that DFT of zeros is zero.
func TestRealDFT3D_Zeros(t *testing.T) {
	t.Parallel()

	depth, height, width := 2, 3, 4
	input := make([]float32, depth*height*width)
	result := RealDFT3D(input, depth, height, width)

	halfWidth := width/2 + 1
	expected := depth * height * halfWidth

	if len(result) != expected {
		t.Fatalf("RealDFT3D zeros: got %d elements, want %d", len(result), expected)
	}

	// All zeros in should give all zeros out
	for i, val := range result {
		if math.Abs(float64(real(val))) > realTolerance || math.Abs(float64(imag(val))) > realTolerance {
			t.Errorf("RealDFT3D(zeros)[%d] = %v, want [0+0i]", i, val)
		}
	}
}

// TestRealDFT3D_OutputShape tests output dimensions are correct.
func TestRealDFT3D_OutputShape(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                 string
		depth, height, width int
	}{
		{"2×3×4", 2, 3, 4},
		{"3×4×8", 3, 4, 8},
		{"4×3×6", 4, 3, 6},
		{"2×2×8", 2, 2, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			input := make([]float32, tt.depth*tt.height*tt.width)
			for i := range input {
				input[i] = float32(i) * 0.1
			}

			result := RealDFT3D(input, tt.depth, tt.height, tt.width)

			halfWidth := tt.width/2 + 1
			expected := tt.depth * tt.height * halfWidth

			if len(result) != expected {
				t.Errorf("RealDFT3D(%d×%d×%d) got %d elements, want %d",
					tt.depth, tt.height, tt.width, len(result), expected)
			}
		})
	}
}

// TestRealDFT3D_RoundTrip tests that IDFT(DFT(x)) ≈ x.
func TestRealDFT3D_RoundTrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                 string
		depth, height, width int
	}{
		{"2×3×4", 2, 3, 4},
		{"3×2×6", 3, 2, 6},
		{"2×2×8", 2, 2, 8},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			// Create test input
			input := make([]float32, tt.depth*tt.height*tt.width)
			for i := range input {
				input[i] = float32(i)*0.1 - 5.0
			}

			// Forward transform
			spectrum := RealDFT3D(input, tt.depth, tt.height, tt.width)

			// Inverse transform
			output := RealIDFT3D(spectrum, tt.depth, tt.height, tt.width)

			// Check output length
			if len(output) != len(input) {
				t.Fatalf("RealIDFT3D returned %d elements, want %d", len(output), len(input))
			}

			// Check each element is close
			for i := range input {
				if math.Abs(float64(output[i])-float64(input[i])) > 1e-4 {
					t.Errorf("RoundTrip[%d]: got %f, want %f",
						i, output[i], input[i])
				}
			}
		})
	}
}

// TestRealDFT3D_DC tests DC component (all ones input).
func TestRealDFT3D_DC(t *testing.T) {
	t.Parallel()

	depth, height, width := 2, 3, 4

	input := make([]float32, depth*height*width)
	for i := range input {
		input[i] = 1.0 // All ones
	}

	result := RealDFT3D(input, depth, height, width)

	// DC component should be sum of all elements
	dcExpected := float32(depth * height * width)
	if math.Abs(float64(real(result[0]))-float64(dcExpected)) > realTolerance {
		t.Errorf("DC component = %v, want %v", real(result[0]), dcExpected)
	}

	// Check imaginary part of DC is near zero
	if math.Abs(float64(imag(result[0]))) > realTolerance {
		t.Errorf("DC imaginary part = %v, want ≈0", imag(result[0]))
	}
}

// TestRealDFT3D_OutputProperties tests output properties of real FFT.
func TestRealDFT3D_OutputProperties(t *testing.T) {
	t.Parallel()

	depth, height, width := 2, 3, 8

	input := make([]float32, depth*height*width)
	for i := range input {
		input[i] = float32(i) * 0.5
	}

	result := RealDFT3D(input, depth, height, width)
	halfWidth := width/2 + 1

	// Check output has correct dimensions
	if len(result) != depth*height*halfWidth {
		t.Errorf("RealDFT3D output size = %d, want %d", len(result), depth*height*halfWidth)
	}

	// DC component should be non-zero and positive (sum of all elements)
	if real(result[0]) <= 0 {
		t.Errorf("DC component should be positive, got %v", result[0])
	}
}

// TestRealIDFT3D_PanicOnMismatch tests that IDFT panics on size mismatch.
func TestRealIDFT3D_PanicOnMismatch(t *testing.T) {
	t.Parallel()

	// Spectrum has wrong size
	spectrum := make([]complex64, 20) // Wrong size for 2×3×8 inverse

	defer func() {
		if r := recover(); r == nil {
			t.Error("RealIDFT3D should panic on spectrum length mismatch")
		}
	}()

	RealIDFT3D(spectrum, 2, 3, 8)
}

// TestRealDFT3D_PanicOnMismatch tests that DFT panics on size mismatch.
func TestRealDFT3D_PanicOnMismatch(t *testing.T) {
	t.Parallel()

	// Input has wrong size
	input := make([]float32, 20) // Wrong size for 2×3×8

	defer func() {
		if r := recover(); r == nil {
			t.Error("RealDFT3D should panic on input length mismatch")
		}
	}()

	RealDFT3D(input, 2, 3, 8)
}

// TestRealDFT3D_Linearity tests linearity: DFT(a*x + b*y) = a*DFT(x) + b*DFT(y).
func TestRealDFT3D_Linearity(t *testing.T) {
	t.Parallel()

	depth, height, width := 2, 2, 4

	// Create two input signals
	x := make([]float32, depth*height*width)

	y := make([]float32, depth*height*width)
	for i := range x {
		x[i] = float32(i) * 0.5
		y[i] = float32(i)*0.2 + 1.0
	}

	// Compute DFT separately
	dftX := RealDFT3D(x, depth, height, width)
	dftY := RealDFT3D(y, depth, height, width)

	// Compute combined signal
	a, b := float32(2.0), float32(3.0)

	combined := make([]float32, depth*height*width)
	for i := range x {
		combined[i] = a*x[i] + b*y[i]
	}

	// DFT of combined should equal a*DFT(x) + b*DFT(y)
	dftCombined := RealDFT3D(combined, depth, height, width)

	halfWidth := width/2 + 1
	for i := range depth * height * halfWidth {
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

// TestRealDFT3D_EvenOddWidths tests even and odd width sizes.
func TestRealDFT3D_EvenOddWidths(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		width int
	}{
		{"Even width", 8},
		{"Odd width", 7},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			depth, height := 2, 3

			input := make([]float32, depth*height*tt.width)
			for i := range input {
				input[i] = float32(i) * 0.5
			}

			result := RealDFT3D(input, depth, height, tt.width)

			halfWidth := tt.width/2 + 1
			expected := depth * height * halfWidth

			if len(result) != expected {
				t.Errorf("RealDFT3D width=%d: got %d elements, want %d", tt.width, len(result), expected)
			}
		})
	}
}

// TestRealIDFT3D_RoundTrip tests IDFT(DFT(x)) ≈ x for various inputs.
func TestRealIDFT3D_RoundTrip(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                 string
		depth, height, width int
		inputFunc            func(int) float32
	}{
		{"Zeros", 2, 2, 4, func(i int) float32 { return 0 }},
		{"Ones", 2, 3, 4, func(i int) float32 { return 1 }},
		{"Ramp", 3, 2, 6, func(i int) float32 { return float32(i) * 0.1 }},
		{"Sine", 2, 2, 8, func(i int) float32 { return float32(math.Sin(float64(i) * 0.5)) }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			input := make([]float32, tt.depth*tt.height*tt.width)
			for i := range input {
				input[i] = tt.inputFunc(i)
			}

			spectrum := RealDFT3D(input, tt.depth, tt.height, tt.width)
			output := RealIDFT3D(spectrum, tt.depth, tt.height, tt.width)

			for i := range input {
				if math.Abs(float64(output[i])-float64(input[i])) > 1e-4 {
					t.Errorf("%s RoundTrip[%d]: got %f, want %f", tt.name, i, output[i], input[i])
				}
			}
		})
	}
}

// TestRealDFT3D_Parseval tests energy conservation (Parseval's theorem).
func TestRealDFT3D_Parseval(t *testing.T) {
	t.Parallel()

	depth, height, width := 2, 3, 4

	input := make([]float32, depth*height*width)
	for i := range input {
		input[i] = float32(i) * 0.5
	}

	// Time domain energy
	energyTime := float64(0)
	for _, v := range input {
		energyTime += float64(v) * float64(v)
	}

	// Frequency domain energy
	spectrum := RealDFT3D(input, depth, height, width)
	energyFreq := float64(0)
	halfWidth := width/2 + 1

	for kd := range depth {
		for kh := range height {
			for kw := range halfWidth {
				idx := kd*height*halfWidth + kh*halfWidth + kw
				mag := real(spectrum[idx])*real(spectrum[idx]) + imag(spectrum[idx])*imag(spectrum[idx])
				// Account for symmetry of real FFT (except DC and Nyquist)
				if kw > 0 && kw < width/2 {
					mag *= 2
				}

				energyFreq += float64(mag)
			}
		}
	}

	energyFreq /= float64(depth * height * width)

	if math.Abs(energyTime-energyFreq) > 1e-2 {
		t.Errorf("Parseval: time energy %f, freq energy %f, diff %f", energyTime, energyFreq, math.Abs(energyTime-energyFreq))
	}
}

// TestRealDFT3D_Scaling tests that doubling input doubles output.
func TestRealDFT3D_Scaling(t *testing.T) {
	t.Parallel()

	depth, height, width := 2, 2, 4

	input := make([]float32, depth*height*width)
	for i := range input {
		input[i] = float32(i) * 0.5
	}

	result1 := RealDFT3D(input, depth, height, width)

	// Double input values
	for i := range input {
		input[i] *= 2
	}

	result2 := RealDFT3D(input, depth, height, width)

	// Output should also double
	halfWidth := width/2 + 1
	for i := range depth * height * halfWidth {
		realDiff := math.Abs(float64(real(result2[i]) - 2*real(result1[i])))
		imagDiff := math.Abs(float64(imag(result2[i]) - 2*imag(result1[i])))

		if realDiff > 1e-4 || imagDiff > 1e-4 {
			t.Errorf("Scaling check failed at [%d]: got 2*%v = %v, want %v",
				i, result1[i], result2[i], complex64(2*complex128(result1[i])))
		}
	}
}
