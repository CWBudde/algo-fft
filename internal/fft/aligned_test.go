package fft

import (
	"testing"
	"unsafe"
)

// TestAllocAlignedComplex64 tests SIMD-aligned complex64 allocation.
func TestAllocAlignedComplex64(t *testing.T) {
	t.Parallel()

	sizes := []int{1, 8, 16, 64, 256, 1024, 4096}

	for _, size := range sizes {
		data, backing := AllocAlignedComplex64(size)

		// Check length
		if len(data) != size {
			t.Errorf("size=%d: len(data)=%d, want %d", size, len(data), size)
		}

		// Check capacity
		if cap(data) < size {
			t.Errorf("size=%d: cap(data)=%d, want >=%d", size, cap(data), size)
		}

		// Check alignment (32-byte for AVX2)
		ptr := uintptr(unsafe.Pointer(&data[0]))
		if ptr%32 != 0 {
			t.Errorf("size=%d: data pointer 0x%x not 32-byte aligned", size, ptr)
		}

		// Check backing buffer exists
		if backing == nil {
			t.Errorf("size=%d: backing buffer is nil", size)
		}

		// Verify we can write to the buffer
		for i := range data {
			data[i] = complex(float32(i), float32(-i))
		}

		// Verify the values
		for i := range data {
			expected := complex(float32(i), float32(-i))
			if data[i] != expected {
				t.Errorf("size=%d: data[%d]=%v, want %v", size, i, data[i], expected)
			}
		}
	}
}

// TestAllocAlignedComplex128 tests SIMD-aligned complex128 allocation.
func TestAllocAlignedComplex128(t *testing.T) {
	t.Parallel()

	sizes := []int{1, 8, 16, 64, 256, 1024}

	for _, size := range sizes {
		data, backing := AllocAlignedComplex128(size)

		// Check length
		if len(data) != size {
			t.Errorf("size=%d: len(data)=%d, want %d", size, len(data), size)
		}

		// Check capacity
		if cap(data) < size {
			t.Errorf("size=%d: cap(data)=%d, want >=%d", size, cap(data), size)
		}

		// Check alignment (32-byte for AVX2)
		ptr := uintptr(unsafe.Pointer(&data[0]))
		if ptr%32 != 0 {
			t.Errorf("size=%d: data pointer 0x%x not 32-byte aligned", size, ptr)
		}

		// Check backing buffer exists
		if backing == nil {
			t.Errorf("size=%d: backing buffer is nil", size)
		}

		// Verify we can write to the buffer
		for i := range data {
			data[i] = complex(float64(i), float64(-i)*0.5)
		}

		// Verify the values
		for i := range data {
			expected := complex(float64(i), float64(-i)*0.5)
			if data[i] != expected {
				t.Errorf("size=%d: data[%d]=%v, want %v", size, i, data[i], expected)
			}
		}
	}
}

// TestAllocAligned_ZeroSize tests edge case of zero-size allocation.
func TestAllocAligned_ZeroSize(t *testing.T) {
	t.Parallel()

	data64, backing64 := AllocAlignedComplex64(0)
	// Zero-size allocation returns nil slices
	if data64 != nil {
		t.Errorf("AllocAlignedComplex64(0) returned non-nil data: %v", data64)
	}

	if backing64 != nil {
		t.Errorf("AllocAlignedComplex64(0) returned non-nil backing: %v", backing64)
	}

	data128, backing128 := AllocAlignedComplex128(0)
	// Zero-size allocation returns nil slices
	if data128 != nil {
		t.Errorf("AllocAlignedComplex128(0) returned non-nil data: %v", data128)
	}

	if backing128 != nil {
		t.Errorf("AllocAlignedComplex128(0) returned non-nil backing: %v", backing128)
	}
}

// TestAllocAligned_LargeSize tests large allocations.
func TestAllocAligned_LargeSize(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large allocation test in short mode")
	}

	t.Parallel()

	const largeSize = 1024 * 1024 // 1M elements

	data64, backing64 := AllocAlignedComplex64(largeSize)
	if len(data64) != largeSize {
		t.Errorf("AllocAlignedComplex64(%d) returned length %d", largeSize, len(data64))
	}

	if backing64 == nil {
		t.Error("Large allocation returned nil backing")
	}

	// Check alignment
	ptr := uintptr(unsafe.Pointer(&data64[0]))
	if ptr%32 != 0 {
		t.Errorf("Large allocation pointer 0x%x not 32-byte aligned", ptr)
	}

	// Spot check some values to ensure buffer is writable
	data64[0] = 1 + 2i
	data64[largeSize/2] = 3 + 4i
	data64[largeSize-1] = 5 + 6i

	if data64[0] != 1+2i || data64[largeSize/2] != 3+4i || data64[largeSize-1] != 5+6i {
		t.Error("Large allocation buffer not writable correctly")
	}
}

// TestAlignPtr tests the alignment helper function.
func TestAlignPtr(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		ptr      uintptr
		align    int
		expected uintptr
	}{
		{0, 32, 0},     // Already aligned
		{1, 32, 32},    // Round up to next 32-byte boundary
		{31, 32, 32},   // Just before boundary
		{32, 32, 32},   // Exactly on boundary
		{33, 32, 64},   // Just after boundary
		{64, 32, 64},   // Another boundary
		{100, 32, 128}, // Mid-range
		{0, 16, 0},     // 16-byte alignment
		{15, 16, 16},   // 16-byte alignment
		{16, 16, 16},   // 16-byte alignment
		{17, 16, 32},   // 16-byte alignment
	}

	for _, tc := range testCases {
		result := alignPtr(tc.ptr, tc.align)
		if result != tc.expected {
			t.Errorf("alignPtr(0x%x, %d) = 0x%x, want 0x%x", tc.ptr, tc.align, result, tc.expected)
		}

		// Verify result is aligned
		if result%uintptr(tc.align) != 0 {
			t.Errorf("alignPtr(0x%x, %d) = 0x%x is not aligned to %d", tc.ptr, tc.align, result, tc.align)
		}

		// Verify result >= original pointer
		if result < tc.ptr {
			t.Errorf("alignPtr(0x%x, %d) = 0x%x is less than original", tc.ptr, tc.align, result)
		}
	}
}

// BenchmarkAllocAlignedComplex64 benchmarks aligned allocation.
func BenchmarkAllocAlignedComplex64(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(itoa(size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 8)) // complex64 is 8 bytes

			for b.Loop() {
				_, _ = AllocAlignedComplex64(size)
			}
		})
	}
}

func BenchmarkAllocAlignedComplex128(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, size := range sizes {
		b.Run(itoa(size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(size * 16)) // complex128 is 16 bytes

			for b.Loop() {
				_, _ = AllocAlignedComplex128(size)
			}
		})
	}
}
