package algofft

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// TestPlanForwardBatch_Correctness verifies batch forward transform correctness.
//
//nolint:gocognit
func TestPlanForwardBatch_Correctness(t *testing.T) {
	t.Run("count=1_matches_single", func(t *testing.T) {
		t.Parallel()

		n := 16

		plan, err := NewPlan(n)
		if err != nil {
			t.Fatal(err)
		}

		src := make([]complex64, n)
		src[0] = 1 // impulse

		// Single FFT
		expectedDst := make([]complex64, n)
		err = plan.Forward(expectedDst, src)
		if err != nil {
			t.Fatal(err)
		}

		// Batch with count=1
		batchSrc := make([]complex64, n)
		copy(batchSrc, src)

		batchDst := make([]complex64, n)
		err = plan.ForwardBatch(batchDst, batchSrc, 1)
		if err != nil {
			t.Fatal(err)
		}

		// Compare
		for i := range n {
			if batchDst[i] != expectedDst[i] {
				t.Errorf("index %d: got %v, want %v", i, batchDst[i], expectedDst[i])
			}
		}
	})

	t.Run("count=4_size=16", func(t *testing.T) {
		t.Parallel()

		n := 16
		count := 4

		plan, err := NewPlan(n)
		if err != nil {
			t.Fatal(err)
		}

		// Create batch input with different signals
		batchSrc := make([]complex64, n*count)
		for batch := range count {
			// Each batch has a different impulse position
			batchSrc[batch*n+batch] = complex(float32(batch+1), 0)
		}

		// Compute batch FFT
		batchDst := make([]complex64, n*count)
		err = plan.ForwardBatch(batchDst, batchSrc, count)
		if err != nil {
			t.Fatal(err)
		}

		// Verify each FFT independently
		for batch := range count {
			srcSlice := batchSrc[batch*n : (batch+1)*n]
			dstSlice := batchDst[batch*n : (batch+1)*n]

			// Compute expected result
			expected := make([]complex64, n)

			err := plan.Forward(expected, srcSlice)
			if err != nil {
				t.Fatal(err)
			}

			// Compare
			for i := range n {
				if !complexEqual64(dstSlice[i], expected[i], 1e-5) {
					t.Errorf("batch %d, index %d: got %v, want %v", batch, i, dstSlice[i], expected[i])
				}
			}
		}
	})

	t.Run("complex128", func(t *testing.T) {
		t.Parallel()

		n := 32
		count := 2

		plan, err := NewPlan64(n)
		if err != nil {
			t.Fatal(err)
		}

		// Create batch input
		batchSrc := make([]complex128, n*count)
		batchSrc[0] = 1   // First batch: impulse at 0
		batchSrc[n+5] = 2 // Second batch: impulse at 5

		// Compute batch FFT
		batchDst := make([]complex128, n*count)
		err = plan.ForwardBatch(batchDst, batchSrc, count)
		if err != nil {
			t.Fatal(err)
		}

		// Verify each FFT independently
		for batch := range count {
			srcSlice := batchSrc[batch*n : (batch+1)*n]
			dstSlice := batchDst[batch*n : (batch+1)*n]

			expected := make([]complex128, n)

			err := plan.Forward(expected, srcSlice)
			if err != nil {
				t.Fatal(err)
			}

			for i := range n {
				if !complexEqual128(dstSlice[i], expected[i], 1e-12) {
					t.Errorf("batch %d, index %d: got %v, want %v", batch, i, dstSlice[i], expected[i])
				}
			}
		}
	})
}

// TestPlanInverseBatch_Correctness verifies batch inverse transform correctness.
func TestPlanInverseBatch_Correctness(t *testing.T) {
	t.Parallel()
	t.Run("roundtrip", func(t *testing.T) {
		t.Parallel()

		n := 64
		count := 3

		plan, err := NewPlan(n)
		if err != nil {
			t.Fatal(err)
		}

		// Create batch input with random data
		original := make([]complex64, n*count)
		for i := range original {
			original[i] = complex(float32(i%10+1), float32((i*7)%10+1))
		}

		// Forward then inverse
		freq := make([]complex64, n*count)
		err = plan.ForwardBatch(freq, original, count)
		if err != nil {
			t.Fatal(err)
		}

		roundtrip := make([]complex64, n*count)
		err = plan.InverseBatch(roundtrip, freq, count)
		if err != nil {
			t.Fatal(err)
		}

		// Compare
		for i := range original {
			if !complexEqual64(roundtrip[i], original[i], 1e-4) {
				t.Errorf("index %d: got %v, want %v", i, roundtrip[i], original[i])
			}
		}
	})
}

// TestPlanBatch_Errors verifies error handling.
//
//nolint:funlen
func TestPlanBatch_Errors(t *testing.T) {
	t.Parallel()

	n := 16

	plan, err := NewPlan(n)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("nil_dst", func(t *testing.T) {
		t.Parallel()

		src := make([]complex64, n*2)

		err := plan.ForwardBatch(nil, src, 2)
		if !errors.Is(err, ErrNilSlice) {
			t.Errorf("got %v, want ErrNilSlice", err)
		}
	})

	t.Run("nil_src", func(t *testing.T) {
		t.Parallel()

		dst := make([]complex64, n*2)

		err := plan.ForwardBatch(dst, nil, 2)
		if !errors.Is(err, ErrNilSlice) {
			t.Errorf("got %v, want ErrNilSlice", err)
		}
	})

	t.Run("count_zero", func(t *testing.T) {
		t.Parallel()

		src := make([]complex64, n)

		dst := make([]complex64, n)

		err := plan.ForwardBatch(dst, src, 0)
		if !errors.Is(err, ErrInvalidLength) {
			t.Errorf("got %v, want ErrInvalidLength", err)
		}
	})

	t.Run("count_negative", func(t *testing.T) {
		t.Parallel()

		src := make([]complex64, n)

		dst := make([]complex64, n)

		err := plan.ForwardBatch(dst, src, -1)
		if !errors.Is(err, ErrInvalidLength) {
			t.Errorf("got %v, want ErrInvalidLength", err)
		}
	})

	t.Run("dst_too_small", func(t *testing.T) {
		t.Parallel()

		src := make([]complex64, n*4)

		dst := make([]complex64, n*3) // Too small for count=4

		err := plan.ForwardBatch(dst, src, 4)
		if !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("got %v, want ErrLengthMismatch", err)
		}
	})

	t.Run("src_too_small", func(t *testing.T) {
		t.Parallel()

		src := make([]complex64, n*3) // Too small for count=4

		dst := make([]complex64, n*4)

		err := plan.ForwardBatch(dst, src, 4)
		if !errors.Is(err, ErrLengthMismatch) {
			t.Errorf("got %v, want ErrLengthMismatch", err)
		}
	})

	// Same tests for InverseBatch
	t.Run("inverse_nil_dst", func(t *testing.T) {
		t.Parallel()

		src := make([]complex64, n*2)

		err := plan.InverseBatch(nil, src, 2)
		if !errors.Is(err, ErrNilSlice) {
			t.Errorf("got %v, want ErrNilSlice", err)
		}
	})

	t.Run("inverse_count_zero", func(t *testing.T) {
		t.Parallel()

		src := make([]complex64, n)

		dst := make([]complex64, n)

		err := plan.InverseBatch(dst, src, 0)
		if !errors.Is(err, ErrInvalidLength) {
			t.Errorf("got %v, want ErrInvalidLength", err)
		}
	})
}

// TestPlanBatch_InPlace verifies in-place batch transforms.
func TestPlanBatch_InPlace(t *testing.T) {
	t.Parallel()

	n := 32
	count := 4

	plan, err := NewPlan(n)
	if err != nil {
		t.Fatal(err)
	}

	// Create input data
	original := make([]complex64, n*count)
	for i := range original {
		original[i] = complex(float32(i%10), float32((i*3)%10))
	}

	// Out-of-place
	outOfPlace := make([]complex64, n*count)
	copy(outOfPlace, original)

	outDst := make([]complex64, n*count)
	err = plan.ForwardBatch(outDst, outOfPlace, count)
	if err != nil {
		t.Fatal(err)
	}

	// In-place
	inPlace := make([]complex64, n*count)
	copy(inPlace, original)

	err = plan.ForwardBatch(inPlace, inPlace, count)
	if err != nil {
		t.Fatal(err)
	}

	// Compare
	for i := range inPlace {
		if !complexEqual64(inPlace[i], outDst[i], 1e-5) {
			t.Errorf("index %d: got %v, want %v", i, inPlace[i], outDst[i])
		}
	}
}

// TestPlanBatch_LargeBatch verifies correctness with large batch counts.
func TestPlanBatch_LargeBatch(t *testing.T) {
	t.Parallel()

	n := 256
	count := 100

	plan, err := NewPlan(n)
	if err != nil {
		t.Fatal(err)
	}

	// Create batch input
	batchSrc := make([]complex64, n*count)
	for batch := range count {
		// Each batch has a unique pattern
		for i := range n {
			batchSrc[batch*n+i] = complex(float32((batch*i)%10), float32(((batch+i)*7)%10))
		}
	}

	// Compute batch FFT
	batchDst := make([]complex64, n*count)
	err = plan.ForwardBatch(batchDst, batchSrc, count)
	if err != nil {
		t.Fatal(err)
	}

	// Verify a few random batches
	testBatches := []int{0, 17, 42, 99}
	for _, batch := range testBatches {
		srcSlice := batchSrc[batch*n : (batch+1)*n]
		dstSlice := batchDst[batch*n : (batch+1)*n]

		expected := make([]complex64, n)

		err := plan.Forward(expected, srcSlice)
		if err != nil {
			t.Fatal(err)
		}

		for i := range n {
			if !complexEqual64(dstSlice[i], expected[i], 1e-4) {
				t.Errorf("batch %d, index %d: got %v, want %v", batch, i, dstSlice[i], expected[i])
			}
		}
	}
}

// Helper functions for comparing complex numbers

func complexEqual64(a, b complex64, tol float32) bool {
	diff := cmplx.Abs(complex128(a - b))

	mag := math.Max(cmplx.Abs(complex128(a)), cmplx.Abs(complex128(b)))
	if mag < 1e-10 {
		return diff < float64(tol)
	}

	return diff/mag < float64(tol)
}

func complexEqual128(a, b complex128, tol float64) bool {
	diff := cmplx.Abs(a - b)

	mag := math.Max(cmplx.Abs(a), cmplx.Abs(b))
	if mag < 1e-10 {
		return diff < tol
	}

	return diff/mag < tol
}

// Benchmarks

// BenchmarkPlanForwardBatch benchmarks batch FFT processing.
func BenchmarkPlanForwardBatch(b *testing.B) {
	sizes := []int{64, 256, 1024}
	counts := []int{1, 4, 16, 64}

	for _, n := range sizes {
		for _, count := range counts {
			b.Run(formatBenchName(n, count), func(b *testing.B) {
				plan, err := NewPlan(n)
				if err != nil {
					b.Fatal(err)
				}

				src := make([]complex64, n*count)
				dst := make([]complex64, n*count)

				// Fill with some data
				for i := range src {
					src[i] = complex(float32(i%10), float32((i*7)%10))
				}

				b.ReportAllocs()
				b.SetBytes(int64(8 * n * count)) // 8 bytes per complex64
				b.ResetTimer()

				for range b.N {
					_ = plan.ForwardBatch(dst, src, count)
				}
			})
		}
	}
}

// BenchmarkPlanInverseBatch benchmarks batch inverse FFT.
func BenchmarkPlanInverseBatch(b *testing.B) {
	sizes := []int{64, 256, 1024}
	counts := []int{1, 4, 16, 64}

	for _, n := range sizes {
		for _, count := range counts {
			b.Run(formatBenchName(n, count), func(b *testing.B) {
				plan, err := NewPlan(n)
				if err != nil {
					b.Fatal(err)
				}

				src := make([]complex64, n*count)
				dst := make([]complex64, n*count)

				// Fill with some data
				for i := range src {
					src[i] = complex(float32(i%10), float32((i*7)%10))
				}

				b.ReportAllocs()
				b.SetBytes(int64(8 * n * count))
				b.ResetTimer()

				for range b.N {
					_ = plan.InverseBatch(dst, src, count)
				}
			})
		}
	}
}

// BenchmarkBatchVsIndividual compares batch API to individual calls.
func BenchmarkBatchVsIndividual(b *testing.B) {
	n := 1024
	count := 16

	b.Run("batch", func(b *testing.B) {
		plan, err := NewPlan(n)
		if err != nil {
			b.Fatal(err)
		}

		src := make([]complex64, n*count)
		dst := make([]complex64, n*count)

		b.ReportAllocs()
		b.SetBytes(int64(8 * n * count))
		b.ResetTimer()

		for range b.N {
			_ = plan.ForwardBatch(dst, src, count)
		}
	})

	b.Run("individual", func(b *testing.B) {
		plan, err := NewPlan(n)
		if err != nil {
			b.Fatal(err)
		}

		src := make([]complex64, n*count)
		dst := make([]complex64, n*count)

		b.ReportAllocs()
		b.SetBytes(int64(8 * n * count))
		b.ResetTimer()

		for range b.N {
			for j := range count {
				start := j * n
				end := start + n
				_ = plan.Forward(dst[start:end], src[start:end])
			}
		}
	})
}

// BenchmarkBatchInPlace benchmarks in-place batch transforms.
func BenchmarkBatchInPlace(b *testing.B) {
	sizes := []int{64, 256, 1024}
	counts := []int{4, 16}

	for _, n := range sizes {
		for _, count := range counts {
			b.Run(formatBenchName(n, count)+"_outofplace", func(b *testing.B) {
				plan, err := NewPlan(n)
				if err != nil {
					b.Fatal(err)
				}

				src := make([]complex64, n*count)
				dst := make([]complex64, n*count)

				b.ReportAllocs()
				b.SetBytes(int64(8 * n * count))
				b.ResetTimer()

				for range b.N {
					_ = plan.ForwardBatch(dst, src, count)
				}
			})

			b.Run(formatBenchName(n, count)+"_inplace", func(b *testing.B) {
				plan, err := NewPlan(n)
				if err != nil {
					b.Fatal(err)
				}

				data := make([]complex64, n*count)

				b.ReportAllocs()
				b.SetBytes(int64(8 * n * count))
				b.ResetTimer()

				for range b.N {
					_ = plan.ForwardBatch(data, data, count)
				}
			})
		}
	}
}

// BenchmarkBatchComplex128 benchmarks batch processing with complex128.
func BenchmarkBatchComplex128(b *testing.B) {
	sizes := []int{64, 256, 1024}
	counts := []int{4, 16}

	for _, n := range sizes {
		for _, count := range counts {
			b.Run(formatBenchName(n, count), func(b *testing.B) {
				plan, err := NewPlan64(n)
				if err != nil {
					b.Fatal(err)
				}

				src := make([]complex128, n*count)
				dst := make([]complex128, n*count)

				// Fill with some data
				for i := range src {
					src[i] = complex(float64(i%10), float64((i*7)%10))
				}

				b.ReportAllocs()
				b.SetBytes(int64(16 * n * count)) // 16 bytes per complex128
				b.ResetTimer()

				for range b.N {
					_ = plan.ForwardBatch(dst, src, count)
				}
			})
		}
	}
}

func formatBenchName(n, count int) string {
	return "n" + itoa(n) + "_count" + itoa(count)
}
