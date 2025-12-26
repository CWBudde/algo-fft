package reference

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Test helpers

func complex2DNearlyEqual(a, b []complex64, rows, cols int, tol float64) bool {
	if len(a) != rows*cols || len(b) != rows*cols {
		return false
	}

	for i := range a {
		if absComplex64(a[i]-b[i]) > tol {
			return false
		}
	}

	return true
}

func complex2D128NearlyEqual(a, b []complex128, rows, cols int, tol float64) bool {
	if len(a) != rows*cols || len(b) != rows*cols {
		return false
	}

	for i := range a {
		if absComplex128(a[i]-b[i]) > tol {
			return false
		}
	}

	return true
}

func generateRandom2DSignal(rows, cols int, seed uint64) []complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF)) //nolint:gosec // Intentionally non-crypto for reproducible tests

	signal := make([]complex64, rows*cols)
	for i := range signal {
		re := float32(rng.Float64()*20 - 10) // Range: [-10, 10]
		im := float32(rng.Float64()*20 - 10)
		signal[i] = complex(re, im)
	}

	return signal
}

func generateRandom2DSignal128(rows, cols int, seed uint64) []complex128 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF)) //nolint:gosec // Intentionally non-crypto for reproducible tests

	signal := make([]complex128, rows*cols)
	for i := range signal {
		re := rng.Float64()*20 - 10
		im := rng.Float64()*20 - 10
		signal[i] = complex(re, im)
	}

	return signal
}

// Test round-trip: IDFT(DFT(x)) ≈ x

func TestNaiveDFT2D_RoundTrip(t *testing.T) {
	testCases := []struct {
		rows, cols int
		name       string
	}{
		{2, 2, "2x2"},
		{4, 4, "4x4"},
		{8, 8, "8x8"},
		{4, 8, "4x8_nonsquare"},
		{8, 4, "8x4_nonsquare"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			original := generateRandom2DSignal(tc.rows, tc.cols, 12345)

			// Forward then inverse
			freq := NaiveDFT2D(original, tc.rows, tc.cols)
			roundTrip := NaiveIDFT2D(freq, tc.rows, tc.cols)

			// Verify round-trip recovers original
			if !complex2DNearlyEqual(roundTrip, original, tc.rows, tc.cols, 1e-4) {
				t.Errorf("Round-trip failed for %dx%d matrix", tc.rows, tc.cols)

				for i := 0; i < tc.rows*tc.cols && i < 10; i++ {
					if absComplex64(roundTrip[i]-original[i]) > 1e-4 {
						t.Errorf("  [%d]: got %v, want %v", i, roundTrip[i], original[i])
					}
				}
			}
		})
	}
}

func TestNaiveDFT2D128_RoundTrip(t *testing.T) {
	testCases := []struct {
		rows, cols int
		name       string
	}{
		{2, 2, "2x2"},
		{4, 4, "4x4"},
		{8, 8, "8x8"},
		{4, 8, "4x8_nonsquare"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			original := generateRandom2DSignal128(tc.rows, tc.cols, 12345)

			freq := NaiveDFT2D128(original, tc.rows, tc.cols)
			roundTrip := NaiveIDFT2D128(freq, tc.rows, tc.cols)

			if !complex2D128NearlyEqual(roundTrip, original, tc.rows, tc.cols, 1e-12) {
				t.Errorf("Round-trip failed for %dx%d matrix", tc.rows, tc.cols)

				for i := 0; i < tc.rows*tc.cols && i < 10; i++ {
					if absComplex128(roundTrip[i]-original[i]) > 1e-12 {
						t.Errorf("  [%d]: got %v, want %v", i, roundTrip[i], original[i])
					}
				}
			}
		})
	}
}

// Test 2D linearity: DFT(aX + bY) = a·DFT(X) + b·DFT(Y)

func TestNaiveDFT2D_Linearity(t *testing.T) {
	rows, cols := 4, 4
	signalX := generateRandom2DSignal(rows, cols, 111)
	signalY := generateRandom2DSignal(rows, cols, 222)

	coeffA := complex64(complex(2.5, 0.5))
	coeffB := complex64(complex(-1.3, 0.7))

	// Compute aX + bY
	combined := make([]complex64, rows*cols)
	for i := range combined {
		combined[i] = coeffA*signalX[i] + coeffB*signalY[i]
	}

	// DFT(aX + bY)
	dftCombined := NaiveDFT2D(combined, rows, cols)

	// a·DFT(X) + b·DFT(Y)
	dftX := NaiveDFT2D(signalX, rows, cols)
	dftY := NaiveDFT2D(signalY, rows, cols)

	expected := make([]complex64, rows*cols)
	for i := range expected {
		expected[i] = coeffA*dftX[i] + coeffB*dftY[i]
	}

	// Verify linearity
	if !complex2DNearlyEqual(dftCombined, expected, rows, cols, 1e-3) {
		t.Errorf("Linearity property failed")

		for i := 0; i < rows*cols && i < 10; i++ {
			if absComplex64(dftCombined[i]-expected[i]) > 1e-3 {
				t.Errorf("  [%d]: got %v, want %v", i, dftCombined[i], expected[i])
			}
		}
	}
}

// Test 2D Parseval's theorem: Σ|x[m,n]|² = (1/MN)·Σ|X[k,l]|²

func TestNaiveDFT2D_Parseval(t *testing.T) {
	rows, cols := 8, 8
	signal := generateRandom2DSignal(rows, cols, 54321)

	// Compute time-domain energy
	var timeEnergy float64

	for _, v := range signal {
		mag := absComplex64(v)
		timeEnergy += mag * mag
	}

	// Compute frequency-domain energy
	freq := NaiveDFT2D(signal, rows, cols)

	var freqEnergy float64

	for _, v := range freq {
		mag := absComplex64(v)
		freqEnergy += mag * mag
	}

	// Apply normalization: (1/MN)
	freqEnergy /= float64(rows * cols)

	// Check Parseval's theorem
	tolerance := 1e-2
	if math.Abs(timeEnergy-freqEnergy) > tolerance {
		t.Errorf("Parseval's theorem failed: time energy = %f, freq energy = %f", timeEnergy, freqEnergy)
	}
}

// Test known signal: constant (DC only)

func TestNaiveDFT2D_ConstantSignal(t *testing.T) {
	rows, cols := 4, 4

	signal := make([]complex64, rows*cols)
	for i := range signal {
		signal[i] = complex(1.0, 0.0) // Constant signal
	}

	freq := NaiveDFT2D(signal, rows, cols)

	// Expected: DC component = rows*cols, all other bins = 0
	expectedDC := complex(float32(rows*cols), 0)
	if absComplex64(freq[0]-expectedDC) > 1e-5 {
		t.Errorf("DC component: got %v, want %v", freq[0], expectedDC)
	}

	// All other frequency bins should be near zero
	for i := 1; i < rows*cols; i++ {
		if absComplex64(freq[i]) > 1e-5 {
			row, col := i/cols, i%cols
			t.Errorf("Non-DC bin [%d,%d]: got %v, want near zero", row, col, freq[i])
		}
	}
}

// Test 2D pure sinusoid: single frequency (kx, ky)

func TestNaiveDFT2D_PureSinusoid(t *testing.T) {
	rows, cols := 8, 8
	kx, ky := 2, 3 // Frequency indices

	signal := make([]complex64, rows*cols)
	for m := range rows {
		for n := range cols {
			// exp(2πi*(kx*m/rows + ky*n/cols))
			phaseRow := 2.0 * math.Pi * float64(kx*m) / float64(rows)
			phaseCol := 2.0 * math.Pi * float64(ky*n) / float64(cols)
			phase := phaseRow + phaseCol
			signal[m*cols+n] = complex64(complex(math.Cos(phase), math.Sin(phase)))
		}
	}

	freq := NaiveDFT2D(signal, rows, cols)

	// Expected: peak at freq[kx, ky] = rows*cols, zeros elsewhere
	expectedPeak := complex(float32(rows*cols), 0)
	peakIdx := kx*cols + ky

	if absComplex64(freq[peakIdx]-expectedPeak) > 1e-2 {
		t.Errorf("Peak at [%d,%d]: got %v, want %v", kx, ky, freq[peakIdx], expectedPeak)
	}

	// All other bins should be near zero
	for i := range rows * cols {
		if i != peakIdx && absComplex64(freq[i]) > 1e-2 {
			row, col := i/cols, i%cols
			t.Errorf("Non-peak bin [%d,%d]: got %v, want near zero", row, col, freq[i])
		}
	}
}

// Test separability: 2D DFT = DFT_cols(DFT_rows(X))

func TestNaiveDFT2D_Separability(t *testing.T) {
	rows, cols := 4, 4
	signal := generateRandom2DSignal(rows, cols, 99999)

	// Direct 2D DFT
	direct := NaiveDFT2D(signal, rows, cols)

	// Separable: row-wise then column-wise
	temp := make([]complex64, rows*cols)
	copy(temp, signal)

	// Apply 1D DFT to each row
	for row := range rows {
		rowData := temp[row*cols : (row+1)*cols]
		rowFreq := NaiveDFT(rowData)
		copy(rowData, rowFreq)
	}

	// Apply 1D DFT to each column
	separable := make([]complex64, rows*cols)
	for col := range cols {
		// Extract column
		colData := make([]complex64, rows)
		for row := range rows {
			colData[row] = temp[row*cols+col]
		}

		// Transform column
		colFreq := NaiveDFT(colData)

		// Write back
		for row := range rows {
			separable[row*cols+col] = colFreq[row]
		}
	}

	// Verify separability
	if !complex2DNearlyEqual(direct, separable, rows, cols, 1e-3) {
		t.Errorf("Separability failed")

		for i := 0; i < rows*cols && i < 10; i++ {
			if absComplex64(direct[i]-separable[i]) > 1e-3 {
				row, col := i/cols, i%cols
				t.Errorf("  [%d,%d]: direct=%v, separable=%v", row, col, direct[i], separable[i])
			}
		}
	}
}

// Benchmark to validate complexity

func BenchmarkNaiveDFT2D_4x4(b *testing.B) {
	signal := generateRandom2DSignal(4, 4, 111)

	b.ResetTimer()

	for range b.N {
		_ = NaiveDFT2D(signal, 4, 4)
	}
}

func BenchmarkNaiveDFT2D_8x8(b *testing.B) {
	signal := generateRandom2DSignal(8, 8, 111)

	b.ResetTimer()

	for range b.N {
		_ = NaiveDFT2D(signal, 8, 8)
	}
}

func BenchmarkNaiveDFT2D_16x16(b *testing.B) {
	signal := generateRandom2DSignal(16, 16, 111)

	b.ResetTimer()

	for range b.N {
		_ = NaiveDFT2D(signal, 16, 16)
	}
}
