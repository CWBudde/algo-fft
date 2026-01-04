package reference

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Test helpers

func complex3DNearlyEqual(a, b []complex64, depth, height, width int, tol float64) bool {
	if len(a) != depth*height*width || len(b) != depth*height*width {
		return false
	}

	for i := range a {
		if absComplex64(a[i]-b[i]) > tol {
			return false
		}
	}

	return true
}

func complex3D128NearlyEqual(a, b []complex128, depth, height, width int, tol float64) bool {
	if len(a) != depth*height*width || len(b) != depth*height*width {
		return false
	}

	for i := range a {
		if absComplex128(a[i]-b[i]) > tol {
			return false
		}
	}

	return true
}

func generateRandom3DSignal(depth, height, width int, seed uint64) []complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF)) //nolint:gosec // Intentionally non-crypto for reproducible tests

	signal := make([]complex64, depth*height*width)
	for i := range signal {
		re := float32(rng.Float64()*20 - 10) // Range: [-10, 10]
		im := float32(rng.Float64()*20 - 10)
		signal[i] = complex(re, im)
	}

	return signal
}

func generateRandom3DSignal128(depth, height, width int, seed uint64) []complex128 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF)) //nolint:gosec // Intentionally non-crypto for reproducible tests

	signal := make([]complex128, depth*height*width)
	for i := range signal {
		re := rng.Float64()*20 - 10
		im := rng.Float64()*20 - 10
		signal[i] = complex(re, im)
	}

	return signal
}

// Test round-trip: IDFT(DFT(x)) ≈ x

func TestNaiveDFT3D_RoundTrip(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		depth, height, width int
		name                 string
	}{
		{2, 2, 2, "2x2x2"},
		{4, 4, 4, "4x4x4"},
		{8, 8, 8, "8x8x8"},
		{2, 4, 8, "2x4x8_nonsquare"},
		{4, 2, 8, "4x2x8_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			original := generateRandom3DSignal(testCase.depth, testCase.height, testCase.width, 12345)

			// Forward then inverse
			freq := NaiveDFT3D(original, testCase.depth, testCase.height, testCase.width)
			roundTrip := NaiveIDFT3D(freq, testCase.depth, testCase.height, testCase.width)

			// Verify round-trip recovers original
			if !complex3DNearlyEqual(roundTrip, original, testCase.depth, testCase.height, testCase.width, 1e-3) {
				t.Errorf("Round-trip failed for %dx%dx%d volume", testCase.depth, testCase.height, testCase.width)

				totalElements := testCase.depth * testCase.height * testCase.width
				for i := 0; i < totalElements && i < 10; i++ {
					if absComplex64(roundTrip[i]-original[i]) > 1e-3 {
						d := i / (testCase.height * testCase.width)
						h := (i / testCase.width) % testCase.height
						w := i % testCase.width
						t.Errorf("  [%d,%d,%d]: got %v, want %v", d, h, w, roundTrip[i], original[i])
					}
				}
			}
		})
	}
}

func TestNaiveDFT3D128_RoundTrip(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		depth, height, width int
		name                 string
	}{
		{2, 2, 2, "2x2x2"},
		{4, 4, 4, "4x4x4"},
		{8, 8, 8, "8x8x8"},
		{2, 4, 8, "2x4x8_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			original := generateRandom3DSignal128(testCase.depth, testCase.height, testCase.width, 12345)

			freq := NaiveDFT3D128(original, testCase.depth, testCase.height, testCase.width)
			roundTrip := NaiveIDFT3D128(freq, testCase.depth, testCase.height, testCase.width)

			if !complex3D128NearlyEqual(roundTrip, original, testCase.depth, testCase.height, testCase.width, 1e-10) {
				t.Errorf("Round-trip failed for %dx%dx%d volume", testCase.depth, testCase.height, testCase.width)

				totalElements := testCase.depth * testCase.height * testCase.width
				for i := 0; i < totalElements && i < 10; i++ {
					if absComplex128(roundTrip[i]-original[i]) > 1e-10 {
						d := i / (testCase.height * testCase.width)
						h := (i / testCase.width) % testCase.height
						w := i % testCase.width
						t.Errorf("  [%d,%d,%d]: got %v, want %v", d, h, w, roundTrip[i], original[i])
					}
				}
			}
		})
	}
}

// Test 3D linearity: DFT(aX + bY) = a·DFT(X) + b·DFT(Y)

func TestNaiveDFT3D_Linearity(t *testing.T) {
	t.Parallel()

	depth, height, width := 4, 4, 4
	signalX := generateRandom3DSignal(depth, height, width, 111)
	signalY := generateRandom3DSignal(depth, height, width, 222)

	coeffA := complex64(complex(2.5, 0.5))
	coeffB := complex64(complex(-1.3, 0.7))

	// Compute aX + bY
	combined := make([]complex64, depth*height*width)
	for i := range combined {
		combined[i] = coeffA*signalX[i] + coeffB*signalY[i]
	}

	// DFT(aX + bY)
	dftCombined := NaiveDFT3D(combined, depth, height, width)

	// a·DFT(X) + b·DFT(Y)
	dftX := NaiveDFT3D(signalX, depth, height, width)
	dftY := NaiveDFT3D(signalY, depth, height, width)

	expected := make([]complex64, depth*height*width)
	for i := range expected {
		expected[i] = coeffA*dftX[i] + coeffB*dftY[i]
	}

	// Verify linearity
	if !complex3DNearlyEqual(dftCombined, expected, depth, height, width, 1e-2) {
		t.Errorf("Linearity property failed")

		for i := 0; i < depth*height*width && i < 10; i++ {
			if absComplex64(dftCombined[i]-expected[i]) > 1e-2 {
				d := i / (height * width)
				h := (i / width) % height
				w := i % width
				t.Errorf("  [%d,%d,%d]: got %v, want %v", d, h, w, dftCombined[i], expected[i])
			}
		}
	}
}

// Test 3D Parseval's theorem: Σ|x[d,h,w]|² = (1/(D*H*W))·Σ|X[kd,kh,kw]|²

func TestNaiveDFT3D_Parseval(t *testing.T) {
	t.Parallel()

	depth, height, width := 8, 8, 8
	signal := generateRandom3DSignal(depth, height, width, 54321)

	// Compute time-domain energy
	var timeEnergy float64

	for _, v := range signal {
		mag := absComplex64(v)
		timeEnergy += mag * mag
	}

	// Compute frequency-domain energy
	freq := NaiveDFT3D(signal, depth, height, width)

	var freqEnergy float64

	for _, v := range freq {
		mag := absComplex64(v)
		freqEnergy += mag * mag
	}

	// Apply normalization: (1/(D*H*W))
	freqEnergy /= float64(depth * height * width)

	// Check Parseval's theorem
	tolerance := 1e-1
	if math.Abs(timeEnergy-freqEnergy) > tolerance {
		t.Errorf("Parseval's theorem failed: time energy = %f, freq energy = %f", timeEnergy, freqEnergy)
	}
}

// Test known signal: constant (DC only)

func TestNaiveDFT3D_ConstantSignal(t *testing.T) {
	t.Parallel()

	depth, height, width := 4, 4, 4

	signal := make([]complex64, depth*height*width)
	for i := range signal {
		signal[i] = complex(1.0, 0.0) // Constant signal
	}

	freq := NaiveDFT3D(signal, depth, height, width)

	// Expected: DC component = depth*height*width, all other bins = 0
	expectedDC := complex(float32(depth*height*width), 0)
	if absComplex64(freq[0]-expectedDC) > 1e-5 {
		t.Errorf("DC component: got %v, want %v", freq[0], expectedDC)
	}

	// All other frequency bins should be near zero
	for i := 1; i < depth*height*width; i++ {
		if absComplex64(freq[i]) > 1e-5 {
			d := i / (height * width)
			h := (i / width) % height
			w := i % width
			t.Errorf("Non-DC bin [%d,%d,%d]: got %v, want near zero", d, h, w, freq[i])
		}
	}
}

// Test 3D pure sinusoid: single frequency (kd, kh, kw)

func TestNaiveDFT3D_PureSinusoid(t *testing.T) {
	t.Parallel()

	depth, height, width := 8, 8, 8
	kd, kh, kw := 2, 3, 1 // Frequency indices

	signal := make([]complex64, depth*height*width)
	for d := range depth {
		for h := range height {
			for w := range width {
				// exp(2πi*(kd*d/depth + kh*h/height + kw*w/width))
				phaseDepth := 2.0 * math.Pi * float64(kd*d) / float64(depth)
				phaseHeight := 2.0 * math.Pi * float64(kh*h) / float64(height)
				phaseWidth := 2.0 * math.Pi * float64(kw*w) / float64(width)
				phase := phaseDepth + phaseHeight + phaseWidth
				signal[d*height*width+h*width+w] = complex64(complex(math.Cos(phase), math.Sin(phase)))
			}
		}
	}

	freq := NaiveDFT3D(signal, depth, height, width)

	// Expected: peak at freq[kd, kh, kw] = depth*height*width, zeros elsewhere
	expectedPeak := complex(float32(depth*height*width), 0)
	peakIdx := kd*height*width + kh*width + kw

	if absComplex64(freq[peakIdx]-expectedPeak) > 1e-1 {
		t.Errorf("Peak at [%d,%d,%d]: got %v, want %v", kd, kh, kw, freq[peakIdx], expectedPeak)
	}

	// All other bins should be near zero
	for i := range depth * height * width {
		if i != peakIdx && absComplex64(freq[i]) > 1e-1 {
			d := i / (height * width)
			h := (i / width) % height
			w := i % width
			t.Errorf("Non-peak bin [%d,%d,%d]: got %v, want near zero", d, h, w, freq[i])
		}
	}
}

// Test separability: 3D DFT = DFT_depth(DFT_height(DFT_width(X)))

func TestNaiveDFT3D_Separability(t *testing.T) {
	t.Parallel()

	depth, height, width := 4, 4, 4
	signal := generateRandom3DSignal(depth, height, width, 99999)

	// Direct 3D DFT
	direct := NaiveDFT3D(signal, depth, height, width)

	// Separable: width-wise, then height-wise, then depth-wise
	temp := make([]complex64, depth*height*width)
	copy(temp, signal)

	// Apply 1D DFT along width (last dimension)
	for d := range depth {
		for h := range height {
			rowData := temp[d*height*width+h*width : d*height*width+(h+1)*width]
			rowFreq := NaiveDFT(rowData)
			copy(rowData, rowFreq)
		}
	}

	// Apply 1D DFT along height (middle dimension)
	for d := range depth {
		for w := range width {
			// Extract column
			colData := make([]complex64, height)
			for h := range height {
				colData[h] = temp[d*height*width+h*width+w]
			}

			// Transform column
			colFreq := NaiveDFT(colData)

			// Write back
			for h := range height {
				temp[d*height*width+h*width+w] = colFreq[h]
			}
		}
	}

	// Apply 1D DFT along depth (first dimension)
	separable := make([]complex64, depth*height*width)
	for h := range height {
		for w := range width {
			// Extract depth slice
			depthData := make([]complex64, depth)
			for d := range depth {
				depthData[d] = temp[d*height*width+h*width+w]
			}

			// Transform depth slice
			depthFreq := NaiveDFT(depthData)

			// Write back
			for d := range depth {
				separable[d*height*width+h*width+w] = depthFreq[d]
			}
		}
	}

	// Verify separability
	if !complex3DNearlyEqual(direct, separable, depth, height, width, 1e-2) {
		t.Errorf("Separability failed")

		for i := 0; i < depth*height*width && i < 10; i++ {
			if absComplex64(direct[i]-separable[i]) > 1e-2 {
				d := i / (height * width)
				h := (i / width) % height
				w := i % width
				t.Errorf("  [%d,%d,%d]: direct=%v, separable=%v", d, h, w, direct[i], separable[i])
			}
		}
	}
}

// Benchmark to validate complexity

func BenchmarkNaiveDFT3D_4x4x4(b *testing.B) {
	signal := generateRandom3DSignal(4, 4, 4, 111)

	b.ResetTimer()

	for range b.N {
		_ = NaiveDFT3D(signal, 4, 4, 4)
	}
}

func BenchmarkNaiveDFT3D_8x8x8(b *testing.B) {
	signal := generateRandom3DSignal(8, 8, 8, 111)

	b.ResetTimer()

	for range b.N {
		_ = NaiveDFT3D(signal, 8, 8, 8)
	}
}

func BenchmarkNaiveDFT3D_16x16x16(b *testing.B) {
	signal := generateRandom3DSignal(16, 16, 16, 111)

	b.ResetTimer()

	for range b.N {
		_ = NaiveDFT3D(signal, 16, 16, 16)
	}
}
