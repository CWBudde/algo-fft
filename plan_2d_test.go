package algofft

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// Test helpers

func generateRandom2DSignal(rows, cols int, seed uint64) []complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF)) //nolint:gosec

	signal := make([]complex64, rows*cols)
	for i := range signal {
		re := float32(rng.Float64()*20 - 10)
		im := float32(rng.Float64()*20 - 10)
		signal[i] = complex(re, im)
	}

	return signal
}

func generateRandom2DSignal128(rows, cols int, seed uint64) []complex128 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF)) //nolint:gosec

	signal := make([]complex128, rows*cols)
	for i := range signal {
		re := rng.Float64()*20 - 10
		im := rng.Float64()*20 - 10
		signal[i] = complex(re, im)
	}

	return signal
}

// Test Plan2D creation

func TestNewPlan2D_ValidDimensions(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		rows, cols int
		name       string
	}{
		{1, 1, "1x1"},
		{2, 2, "2x2_square"},
		{4, 4, "4x4_square"},
		{8, 8, "8x8_square"},
		{4, 8, "4x8_nonsquare"},
		{8, 4, "8x4_nonsquare"},
		{16, 32, "16x32_large_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan2D[complex64](testCase.rows, testCase.cols)
			if err != nil {
				t.Fatalf("NewPlan2D(%d, %d) failed: %v", testCase.rows, testCase.cols, err)
			}

			if plan.Rows() != testCase.rows {
				t.Errorf("Rows() = %d, want %d", plan.Rows(), testCase.rows)
			}

			if plan.Cols() != testCase.cols {
				t.Errorf("Cols() = %d, want %d", plan.Cols(), testCase.cols)
			}

			if plan.Len() != testCase.rows*testCase.cols {
				t.Errorf("Len() = %d, want %d", plan.Len(), testCase.rows*testCase.cols)
			}
		})
	}
}

func TestNewPlan2D_InvalidDimensions(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		rows, cols int
		name       string
	}{
		{0, 0, "zero_zero"},
		{0, 4, "zero_rows"},
		{4, 0, "zero_cols"},
		{-1, 4, "negative_rows"},
		{4, -1, "negative_cols"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			_, err := NewPlan2D[complex64](testCase.rows, testCase.cols)
			if !errors.Is(err, ErrInvalidLength) {
				t.Errorf("NewPlan2D(%d, %d) = %v, want ErrInvalidLength", testCase.rows, testCase.cols, err)
			}
		})
	}
}

func TestNewPlan2D32_64(t *testing.T) {
	t.Parallel()

	t.Run("NewPlan2D32", func(t *testing.T) {
		t.Parallel()

		plan, err := NewPlan2D32(4, 4)
		if err != nil {
			t.Fatalf("NewPlan2D32 failed: %v", err)
		}

		if plan.Rows() != 4 || plan.Cols() != 4 {
			t.Errorf("Dimensions mismatch")
		}
	})

	t.Run("NewPlan2D64", func(t *testing.T) {
		t.Parallel()

		plan, err := NewPlan2D64(4, 4)
		if err != nil {
			t.Fatalf("NewPlan2D64 failed: %v", err)
		}

		if plan.Rows() != 4 || plan.Cols() != 4 {
			t.Errorf("Dimensions mismatch")
		}
	})
}

// Test validation

func TestPlan2D_NilSlices(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan2D[complex64](4, 4)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	validData := make([]complex64, 16)

	t.Run("nil_dst", func(t *testing.T) {
		t.Parallel()

		err := plan.Forward(nil, validData)
		if !errors.Is(err, ErrNilSlice) {
			t.Errorf("Forward(nil, validData) = %v, want ErrNilSlice", err)
		}
	})

	t.Run("nil_src", func(t *testing.T) {
		t.Parallel()

		err := plan.Forward(validData, nil)
		if !errors.Is(err, ErrNilSlice) {
			t.Errorf("Forward(validData, nil) = %v, want ErrNilSlice", err)
		}
	})
}

func TestPlan2D_LengthMismatch(t *testing.T) {
	t.Parallel()

	plan, err := NewPlan2D[complex64](4, 4)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	validData := make([]complex64, 16)
	wrongData := make([]complex64, 10)

	t.Run("wrong_dst_length", func(t *testing.T) {
		t.Parallel()

		err := plan.Forward(wrongData, validData)
		if !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("Forward with wrong dst length = %v, want ErrLengthMismatch", err)
		}
	})

	t.Run("wrong_src_length", func(t *testing.T) {
		t.Parallel()

		err := plan.Forward(validData, wrongData)
		if !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("Forward with wrong src length = %v, want ErrLengthMismatch", err)
		}
	})
}

// Test correctness against reference implementation

func TestPlan2D_ForwardMatchesReference(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		rows, cols int
		name       string
	}{
		{2, 2, "2x2"},
		{4, 4, "4x4"},
		{8, 8, "8x8"},
		{16, 16, "16x16"},
		{4, 8, "4x8_nonsquare"},
		{8, 4, "8x4_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan2D[complex64](testCase.rows, testCase.cols)
			if err != nil {
				t.Fatalf("NewPlan2D failed: %v", err)
			}

			src := generateRandom2DSignal(testCase.rows, testCase.cols, 12345)
			dst := make([]complex64, testCase.rows*testCase.cols)

			// Compute with Plan2D
			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Compute with reference
			want := reference.NaiveDFT2D(src, testCase.rows, testCase.cols)

			// Compare results
			tol := 1e-3

			for i := range dst {
				row, col := i/testCase.cols, i%testCase.cols
				assertApproxComplex64Tolf(t, dst[i], want[i], tol, "[%d,%d]", row, col)
			}
		})
	}
}

func TestPlan2D_InverseMatchesReference(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		rows, cols int
		name       string
	}{
		{2, 2, "2x2"},
		{4, 4, "4x4"},
		{8, 8, "8x8"},
		{16, 16, "16x16"},
		{4, 8, "4x8_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan2D[complex64](testCase.rows, testCase.cols)
			if err != nil {
				t.Fatalf("NewPlan2D failed: %v", err)
			}

			src := generateRandom2DSignal(testCase.rows, testCase.cols, 54321)
			dst := make([]complex64, testCase.rows*testCase.cols)

			// Compute with Plan2D
			if err := plan.Inverse(dst, src); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			// Compute with reference
			want := reference.NaiveIDFT2D(src, testCase.rows, testCase.cols)

			// Compare results
			tol := 1e-3

			for i := range dst {
				row, col := i/testCase.cols, i%testCase.cols
				assertApproxComplex64Tolf(t, dst[i], want[i], tol, "[%d,%d]", row, col)
			}
		})
	}
}

func TestPlan2D_BatchStrideForward(t *testing.T) {
	t.Parallel()

	const (
		rows   = 4
		cols   = 4
		batch  = 2
		stride = rows*cols + 5
	)

	plan, err := NewPlan2DWithOptions[complex64](rows, cols, PlanOptions{
		Batch:  batch,
		Stride: stride,
	})
	if err != nil {
		t.Fatalf("NewPlan2DWithOptions failed: %v", err)
	}

	src := make([]complex64, batch*stride)
	dst := make([]complex64, batch*stride)

	signals := make([][]complex64, batch)
	for b := range batch {
		signal := generateRandom2DSignal(rows, cols, uint64(100+b))
		signals[b] = signal
		copy(src[b*stride:b*stride+rows*cols], signal)
	}

	if err := plan.Forward(dst, src); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	tol := 1e-3

	for b := range batch {
		want := reference.NaiveDFT2D(signals[b], rows, cols)

		got := dst[b*stride : b*stride+rows*cols]
		for i := range got {
			row, col := i/cols, i%cols
			assertApproxComplex64Tolf(t, got[i], want[i], tol, "[batch=%d %d,%d]", b, row, col)
		}
	}
}

func TestPlan2D_ForwardMatchesReference128(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		rows, cols int
		name       string
	}{
		{4, 4, "4x4"},
		{8, 8, "8x8"},
		{4, 8, "4x8_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan2D[complex128](testCase.rows, testCase.cols)
			if err != nil {
				t.Fatalf("NewPlan2D failed: %v", err)
			}

			src := generateRandom2DSignal128(testCase.rows, testCase.cols, 12345)
			dst := make([]complex128, testCase.rows*testCase.cols)

			if err := plan.Forward(dst, src); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			want := reference.NaiveDFT2D128(src, testCase.rows, testCase.cols)

			tol := 1e-10

			for i := range dst {
				row, col := i/testCase.cols, i%testCase.cols
				assertApproxComplex128Tolf(t, dst[i], want[i], tol, "[%d,%d]", row, col)
			}
		})
	}
}

// Test round-trip: Inverse(Forward(x)) ≈ x

func TestPlan2D_RoundTrip(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		rows, cols int
		name       string
	}{
		{8, 8, "8x8"},
		{16, 16, "16x16"},
		{32, 32, "32x32"},
		{64, 64, "64x64"},
		{8, 16, "8x16_nonsquare"},
		{16, 32, "16x32_nonsquare"},
		{32, 64, "32x64_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan2D[complex64](testCase.rows, testCase.cols)
			if err != nil {
				t.Fatalf("NewPlan2D failed: %v", err)
			}

			original := generateRandom2DSignal(testCase.rows, testCase.cols, 99999)
			freq := make([]complex64, testCase.rows*testCase.cols)
			roundTrip := make([]complex64, testCase.rows*testCase.cols)

			// Forward
			if err := plan.Forward(freq, original); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			// Inverse
			if err := plan.Inverse(roundTrip, freq); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			// Verify round-trip
			tol := 1e-3

			for i := range original {
				row, col := i/testCase.cols, i%testCase.cols
				assertApproxComplex64Tolf(t, roundTrip[i], original[i], tol, "[%d,%d]", row, col)
			}
		})
	}
}

func TestPlan2D_RoundTrip128(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		rows, cols int
		name       string
	}{
		{8, 8, "8x8"},
		{16, 16, "16x16"},
		{32, 32, "32x32"},
		{8, 16, "8x16_nonsquare"},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			t.Parallel()

			plan, err := NewPlan2D[complex128](testCase.rows, testCase.cols)
			if err != nil {
				t.Fatalf("NewPlan2D failed: %v", err)
			}

			original := generateRandom2DSignal128(testCase.rows, testCase.cols, 99999)
			freq := make([]complex128, testCase.rows*testCase.cols)
			roundTrip := make([]complex128, testCase.rows*testCase.cols)

			if err := plan.Forward(freq, original); err != nil {
				t.Fatalf("Forward failed: %v", err)
			}

			if err := plan.Inverse(roundTrip, freq); err != nil {
				t.Fatalf("Inverse failed: %v", err)
			}

			tol := 1e-10

			for i := range original {
				row, col := i/testCase.cols, i%testCase.cols
				assertApproxComplex128Tolf(t, roundTrip[i], original[i], tol, "[%d,%d]", row, col)
			}
		})
	}
}

func TestPlan2D_InPlaceMatchesOutOfPlace(t *testing.T) {
	t.Parallel()

	rows, cols := 8, 8

	plan, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	src := generateRandom2DSignal(rows, cols, 77777)

	// Out-of-place
	outOfPlace := make([]complex64, rows*cols)
	if err := plan.Forward(outOfPlace, src); err != nil {
		t.Fatalf("Out-of-place Forward failed: %v", err)
	}

	// In-place
	inPlace := append([]complex64(nil), src...)
	if err := plan.ForwardInPlace(inPlace); err != nil {
		t.Fatalf("In-place Forward failed: %v", err)
	}

	// Compare
	tol := 1e-6
	for i := range outOfPlace {
		assertApproxComplex64Tolf(t, inPlace[i], outOfPlace[i], tol, "[%d]", i)
	}
}

// Test mathematical properties

func TestPlan2D_Linearity(t *testing.T) {
	t.Parallel()

	rows, cols := 8, 8

	plan, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	signalX := generateRandom2DSignal(rows, cols, 111)
	signalY := generateRandom2DSignal(rows, cols, 222)

	coeffA := complex64(complex(2.5, 0.5))
	coeffB := complex64(complex(-1.3, 0.7))

	// Compute aX + bY
	combined := make([]complex64, rows*cols)
	for i := range combined {
		combined[i] = coeffA*signalX[i] + coeffB*signalY[i]
	}

	// FFT(aX + bY)
	fftCombined := make([]complex64, rows*cols)
	if err := plan.Forward(fftCombined, combined); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// a·FFT(X) + b·FFT(Y)
	fftX := make([]complex64, rows*cols)
	fftY := make([]complex64, rows*cols)

	if err := plan.Forward(fftX, signalX); err != nil {
		t.Fatalf("Forward X failed: %v", err)
	}

	if err := plan.Forward(fftY, signalY); err != nil {
		t.Fatalf("Forward Y failed: %v", err)
	}

	expected := make([]complex64, rows*cols)
	for i := range expected {
		expected[i] = coeffA*fftX[i] + coeffB*fftY[i]
	}

	// Verify linearity
	tol := 1e-2
	for i := range fftCombined {
		assertApproxComplex64Tolf(t, fftCombined[i], expected[i], tol, "[%d]", i)
	}
}

func TestPlan2D_Parseval(t *testing.T) {
	t.Parallel()

	rows, cols := 8, 8

	plan, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	signal := generateRandom2DSignal(rows, cols, 54321)

	// Time-domain energy
	var timeEnergy float64

	for _, v := range signal {
		mag := cmplx.Abs(complex128(v))
		timeEnergy += mag * mag
	}

	// Frequency-domain energy
	freq := make([]complex64, rows*cols)
	if err := plan.Forward(freq, signal); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	var freqEnergy float64

	for _, v := range freq {
		mag := cmplx.Abs(complex128(v))
		freqEnergy += mag * mag
	}

	// Normalization: (1/MN)
	freqEnergy /= float64(rows * cols)

	// Check Parseval's theorem
	tolerance := 1e-1
	if math.Abs(timeEnergy-freqEnergy) > tolerance {
		t.Errorf("Parseval's theorem failed: time=%f, freq=%f", timeEnergy, freqEnergy)
	}
}

func TestPlan2D_Separability(t *testing.T) {
	t.Parallel()

	rows, cols := 4, 4

	plan, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	signal := generateRandom2DSignal(rows, cols, 88888)

	// Direct 2D FFT
	direct := make([]complex64, rows*cols)
	if err := plan.Forward(direct, signal); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Separable: row-wise then column-wise
	rowPlan, _ := NewPlanT[complex64](cols)
	colPlan, _ := NewPlanT[complex64](rows)

	temp := append([]complex64(nil), signal...)

	// Transform rows
	for row := range rows {
		rowData := temp[row*cols : (row+1)*cols]

		err := rowPlan.InPlace(rowData)
		if err != nil {
			t.Fatalf("InPlace row %d failed: %v", row, err)
		}
	}

	// Transform columns
	separable := make([]complex64, rows*cols)
	copy(separable, temp)

	colData := make([]complex64, rows)
	for col := range cols {
		for row := range rows {
			colData[row] = temp[row*cols+col]
		}

		err := colPlan.InPlace(colData)
		if err != nil {
			t.Fatalf("InPlace col %d failed: %v", col, err)
		}

		for row := range rows {
			separable[row*cols+col] = colData[row]
		}
	}

	// Verify
	tol := 1e-3
	for i := range direct {
		assertApproxComplex64Tolf(t, separable[i], direct[i], tol, "[%d]", i)
	}
}

// Test signal properties

func TestPlan2D_ConstantSignal(t *testing.T) {
	t.Parallel()

	rows, cols := 8, 8

	plan, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	signal := make([]complex64, rows*cols)
	for i := range signal {
		signal[i] = complex(1.0, 0.0)
	}

	freq := make([]complex64, rows*cols)
	if err := plan.Forward(freq, signal); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// DC component should be rows*cols
	expectedDC := complex(float32(rows*cols), 0)
	tol := 1e-4
	assertApproxComplex64Tolf(t, freq[0], expectedDC, tol, "DC")

	// All other bins should be near zero
	for i := 1; i < rows*cols; i++ {
		mag := cmplx.Abs(complex128(freq[i]))
		if mag > 1e-4 {
			row, col := i/cols, i%cols
			t.Errorf("Non-DC bin [%d,%d]: mag=%v, want near zero", row, col, mag)
		}
	}
}

func TestPlan2D_PureSinusoid2D(t *testing.T) {
	t.Parallel()

	rows, cols := 8, 8
	kx, ky := 2, 3

	plan, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	signal := make([]complex64, rows*cols)
	for m := range rows {
		for n := range cols {
			phaseRow := 2.0 * math.Pi * float64(kx*m) / float64(rows)
			phaseCol := 2.0 * math.Pi * float64(ky*n) / float64(cols)
			phase := phaseRow + phaseCol
			signal[m*cols+n] = complex64(complex(math.Cos(phase), math.Sin(phase)))
		}
	}

	freq := make([]complex64, rows*cols)
	if err := plan.Forward(freq, signal); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// Peak at [kx, ky]
	peakIdx := kx*cols + ky
	expectedPeak := complex(float32(rows*cols), 0)
	tol := 1e-2
	assertApproxComplex64Tolf(t, freq[peakIdx], expectedPeak, tol, "[%d,%d]", kx, ky)

	// Other bins near zero
	for i := range freq {
		if i != peakIdx {
			mag := cmplx.Abs(complex128(freq[i]))
			if mag > 2e-2 {
				row, col := i/cols, i%cols
				t.Errorf("Non-peak bin [%d,%d]: mag=%v, want near zero", row, col, mag)
			}
		}
	}
}

// Test Clone

func TestPlan2D_Clone(t *testing.T) {
	t.Parallel()

	rows, cols := 8, 8

	original, err := NewPlan2D[complex64](rows, cols)
	if err != nil {
		t.Fatalf("NewPlan2D failed: %v", err)
	}

	clone := original.Clone()

	// Verify dimensions match
	if clone.Rows() != original.Rows() {
		t.Errorf("Clone Rows mismatch")
	}

	if clone.Cols() != original.Cols() {
		t.Errorf("Clone Cols mismatch")
	}

	// Verify independent operation
	signal := generateRandom2DSignal(rows, cols, 11111)
	freq1 := make([]complex64, rows*cols)
	freq2 := make([]complex64, rows*cols)

	if err := original.Forward(freq1, signal); err != nil {
		t.Fatalf("original.Forward failed: %v", err)
	}

	if err := clone.Forward(freq2, signal); err != nil {
		t.Fatalf("clone.Forward failed: %v", err)
	}

	// Results should match
	tol := 1e-6
	for i := range freq1 {
		assertApproxComplex64Tolf(t, freq2[i], freq1[i], tol, "[%d]", i)
	}
}
