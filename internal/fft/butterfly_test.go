package fft

import (
	"math/cmplx"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// TestButterfly2 verifies the radix-2 butterfly operation.
func TestButterfly2(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		testButterfly2_64(t)
	})
	t.Run("complex128", func(t *testing.T) {
		testButterfly2_128(t)
	})
}

func testButterfly2_64(t *testing.T) {
	t.Helper()

	testCases := []struct {
		a, b, w complex64
	}{
		{1, 2, 1},
		{1 + 2i, 3 + 4i, 0.707 + 0.707i},
		{complex(1, 0), complex(0, 1), complex(0, -1)},
	}

	for _, tc := range testCases {
		// Reference: radix-2 butterfly
		// out0 = a + w*b
		// out1 = a - w*b
		want0 := tc.a + tc.w*tc.b
		want1 := tc.a - tc.w*tc.b

		got0, got1 := butterfly2(tc.a, tc.b, tc.w)

		if cmplx.Abs(complex128(got0-want0)) > testTol64 {
			t.Errorf("butterfly2(%v, %v, %v) out0: got %v, want %v", tc.a, tc.b, tc.w, got0, want0)
		}

		if cmplx.Abs(complex128(got1-want1)) > testTol64 {
			t.Errorf("butterfly2(%v, %v, %v) out1: got %v, want %v", tc.a, tc.b, tc.w, got1, want1)
		}
	}
}

func testButterfly2_128(t *testing.T) {
	t.Helper()

	testCases := []struct {
		a, b, w complex128
	}{
		{1, 2, 1},
		{1 + 2i, 3 + 4i, 0.707 + 0.707i},
		{complex(1, 0), complex(0, 1), complex(0, -1)},
	}

	for _, tc := range testCases {
		want0 := tc.a + tc.w*tc.b
		want1 := tc.a - tc.w*tc.b

		got0, got1 := butterfly2(tc.a, tc.b, tc.w)

		if cmplx.Abs(got0-want0) > testTol128 {
			t.Errorf("butterfly2(%v, %v, %v) out0: got %v, want %v", tc.a, tc.b, tc.w, got0, want0)
		}

		if cmplx.Abs(got1-want1) > testTol128 {
			t.Errorf("butterfly2(%v, %v, %v) out1: got %v, want %v", tc.a, tc.b, tc.w, got1, want1)
		}
	}
}

// TestButterfly3 verifies the radix-3 butterfly against naive DFT.
func TestButterfly3(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		testButterfly3_64(t)
	})
	t.Run("complex128", func(t *testing.T) {
		testButterfly3_128(t)
	})
}

func testButterfly3_64(t *testing.T) {
	t.Helper()

	// Generate several test inputs
	testInputs := [][]complex64{
		{1, 2, 3},
		{1 + 1i, 2 + 2i, 3 + 3i},
		{0.5, -0.5, 0.25},
		randomComplex64(3, 11111),
		randomComplex64(3, 22222),
	}

	for _, input := range testInputs {
		// Compute reference using naive DFT
		want := reference.NaiveDFT(input)

		// Apply butterfly
		got0, got1, got2 := butterfly3Forward(input[0], input[1], input[2])
		got := []complex64{got0, got1, got2}

		// Verify results
		assertComplex64SliceClose(t, got, want, 3)
	}
}

func testButterfly3_128(t *testing.T) {
	t.Helper()

	testInputs := [][]complex128{
		{1, 2, 3},
		{1 + 1i, 2 + 2i, 3 + 3i},
		{0.5, -0.5, 0.25},
		randomComplex128(3, 11111),
		randomComplex128(3, 22222),
	}

	for _, input := range testInputs {
		want := reference.NaiveDFT128(input)

		got0, got1, got2 := butterfly3Forward(input[0], input[1], input[2])
		got := []complex128{got0, got1, got2}

		assertComplex128SliceClose(t, got, want, 3)
	}
}

// TestButterfly3Inverse verifies inverse radix-3 butterfly.
func TestButterfly3Inverse(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		testButterfly3Inverse_64(t)
	})
	t.Run("complex128", func(t *testing.T) {
		testButterfly3Inverse_128(t)
	})
}

func testButterfly3Inverse_64(t *testing.T) {
	t.Helper()

	testInputs := [][]complex64{
		{1, 2, 3},
		randomComplex64(3, 33333),
		randomComplex64(3, 44444),
	}

	for _, input := range testInputs {
		// Forward then inverse should recover input (scaled by n)
		fwd0, fwd1, fwd2 := butterfly3Forward(input[0], input[1], input[2])
		inv0, inv1, inv2 := butterfly3Inverse(fwd0, fwd1, fwd2)
		got := []complex64{inv0, inv1, inv2}

		// Scale by n=3 for comparison
		want := make([]complex64, 3)
		for i := range input {
			want[i] = input[i] * 3
		}

		assertComplex64SliceClose(t, got, want, 3)
	}
}

func testButterfly3Inverse_128(t *testing.T) {
	t.Helper()

	testInputs := [][]complex128{
		{1, 2, 3},
		randomComplex128(3, 33333),
		randomComplex128(3, 44444),
	}

	for _, input := range testInputs {
		fwd0, fwd1, fwd2 := butterfly3Forward(input[0], input[1], input[2])
		inv0, inv1, inv2 := butterfly3Inverse(fwd0, fwd1, fwd2)
		got := []complex128{inv0, inv1, inv2}

		want := make([]complex128, 3)
		for i := range input {
			want[i] = input[i] * 3
		}

		assertComplex128SliceClose(t, got, want, 3)
	}
}

// TestButterfly4 verifies the radix-4 butterfly against naive DFT.
func TestButterfly4(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		testButterfly4_64(t)
	})
	t.Run("complex128", func(t *testing.T) {
		testButterfly4_128(t)
	})
}

func testButterfly4_64(t *testing.T) {
	t.Helper()

	testInputs := [][]complex64{
		{1, 2, 3, 4},
		{1 + 1i, 2 + 2i, 3 + 3i, 4 + 4i},
		{0.25, -0.25, 0.5, -0.5},
		randomComplex64(4, 55555),
		randomComplex64(4, 66666),
	}

	for _, input := range testInputs {
		want := reference.NaiveDFT(input)

		got0, got1, got2, got3 := butterfly4Forward(input[0], input[1], input[2], input[3])
		got := []complex64{got0, got1, got2, got3}

		assertComplex64SliceClose(t, got, want, 4)
	}
}

func testButterfly4_128(t *testing.T) {
	t.Helper()

	testInputs := [][]complex128{
		{1, 2, 3, 4},
		{1 + 1i, 2 + 2i, 3 + 3i, 4 + 4i},
		{0.25, -0.25, 0.5, -0.5},
		randomComplex128(4, 55555),
		randomComplex128(4, 66666),
	}

	for _, input := range testInputs {
		want := reference.NaiveDFT128(input)

		got0, got1, got2, got3 := butterfly4Forward(input[0], input[1], input[2], input[3])
		got := []complex128{got0, got1, got2, got3}

		assertComplex128SliceClose(t, got, want, 4)
	}
}

// TestButterfly4Inverse verifies inverse radix-4 butterfly.
func TestButterfly4Inverse(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		testButterfly4Inverse_64(t)
	})
	t.Run("complex128", func(t *testing.T) {
		testButterfly4Inverse_128(t)
	})
}

func testButterfly4Inverse_64(t *testing.T) {
	t.Helper()

	testInputs := [][]complex64{
		{1, 2, 3, 4},
		randomComplex64(4, 77777),
		randomComplex64(4, 88888),
	}

	for _, input := range testInputs {
		fwd0, fwd1, fwd2, fwd3 := butterfly4Forward(input[0], input[1], input[2], input[3])
		inv0, inv1, inv2, inv3 := butterfly4Inverse(fwd0, fwd1, fwd2, fwd3)
		got := []complex64{inv0, inv1, inv2, inv3}

		want := make([]complex64, 4)
		for i := range input {
			want[i] = input[i] * 4
		}

		assertComplex64SliceClose(t, got, want, 4)
	}
}

func testButterfly4Inverse_128(t *testing.T) {
	t.Helper()

	testInputs := [][]complex128{
		{1, 2, 3, 4},
		randomComplex128(4, 77777),
		randomComplex128(4, 88888),
	}

	for _, input := range testInputs {
		fwd0, fwd1, fwd2, fwd3 := butterfly4Forward(input[0], input[1], input[2], input[3])
		inv0, inv1, inv2, inv3 := butterfly4Inverse(fwd0, fwd1, fwd2, fwd3)
		got := []complex128{inv0, inv1, inv2, inv3}

		want := make([]complex128, 4)
		for i := range input {
			want[i] = input[i] * 4
		}

		assertComplex128SliceClose(t, got, want, 4)
	}
}

// TestButterfly5 verifies the radix-5 butterfly against naive DFT.
func TestButterfly5(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		testButterfly5_64(t)
	})
	t.Run("complex128", func(t *testing.T) {
		testButterfly5_128(t)
	})
}

func testButterfly5_64(t *testing.T) {
	t.Helper()

	testInputs := [][]complex64{
		{1, 2, 3, 4, 5},
		{1 + 1i, 2 + 2i, 3 + 3i, 4 + 4i, 5 + 5i},
		{0.2, -0.2, 0.4, -0.4, 0.6},
		randomComplex64(5, 99999),
		randomComplex64(5, 10101),
	}

	for _, input := range testInputs {
		want := reference.NaiveDFT(input)

		got0, got1, got2, got3, got4 := butterfly5Forward(input[0], input[1], input[2], input[3], input[4])
		got := []complex64{got0, got1, got2, got3, got4}

		assertComplex64SliceClose(t, got, want, 5)
	}
}

func testButterfly5_128(t *testing.T) {
	t.Helper()

	testInputs := [][]complex128{
		{1, 2, 3, 4, 5},
		{1 + 1i, 2 + 2i, 3 + 3i, 4 + 4i, 5 + 5i},
		{0.2, -0.2, 0.4, -0.4, 0.6},
		randomComplex128(5, 99999),
		randomComplex128(5, 10101),
	}

	for _, input := range testInputs {
		want := reference.NaiveDFT128(input)

		got0, got1, got2, got3, got4 := butterfly5Forward(input[0], input[1], input[2], input[3], input[4])
		got := []complex128{got0, got1, got2, got3, got4}

		assertComplex128SliceClose(t, got, want, 5)
	}
}

// TestButterfly5Inverse verifies inverse radix-5 butterfly.
func TestButterfly5Inverse(t *testing.T) {
	t.Parallel()

	t.Run("complex64", func(t *testing.T) {
		testButterfly5Inverse_64(t)
	})
	t.Run("complex128", func(t *testing.T) {
		testButterfly5Inverse_128(t)
	})
}

func testButterfly5Inverse_64(t *testing.T) {
	t.Helper()

	testInputs := [][]complex64{
		{1, 2, 3, 4, 5},
		randomComplex64(5, 20202),
		randomComplex64(5, 30303),
	}

	for _, input := range testInputs {
		fwd0, fwd1, fwd2, fwd3, fwd4 := butterfly5Forward(input[0], input[1], input[2], input[3], input[4])
		inv0, inv1, inv2, inv3, inv4 := butterfly5Inverse(fwd0, fwd1, fwd2, fwd3, fwd4)
		got := []complex64{inv0, inv1, inv2, inv3, inv4}

		want := make([]complex64, 5)
		for i := range input {
			want[i] = input[i] * 5
		}

		assertComplex64SliceClose(t, got, want, 5)
	}
}

func testButterfly5Inverse_128(t *testing.T) {
	t.Helper()

	testInputs := [][]complex128{
		{1, 2, 3, 4, 5},
		randomComplex128(5, 20202),
		randomComplex128(5, 30303),
	}

	for _, input := range testInputs {
		fwd0, fwd1, fwd2, fwd3, fwd4 := butterfly5Forward(input[0], input[1], input[2], input[3], input[4])
		inv0, inv1, inv2, inv3, inv4 := butterfly5Inverse(fwd0, fwd1, fwd2, fwd3, fwd4)
		got := []complex128{inv0, inv1, inv2, inv3, inv4}

		want := make([]complex128, 5)
		for i := range input {
			want[i] = input[i] * 5
		}

		assertComplex128SliceClose(t, got, want, 5)
	}
}
