package memory

import (
	"testing"
	"unsafe"
)

// TestAllocAlignedGeneric verifies the generic AllocAligned wrapper matches the
// concrete helpers: correct length, 64-byte alignment, a live backing, and a
// writable buffer, for both complex64 and complex128.
func TestAllocAlignedGeneric(t *testing.T) {
	t.Parallel()

	sizes := []int{1, 8, 16, 64, 256, 1024}

	t.Run("complex64", func(t *testing.T) {
		t.Parallel()

		for _, size := range sizes {
			data, backing := AllocAligned[complex64](size)

			if len(data) != size {
				t.Errorf("size=%d: len(data)=%d, want %d", size, len(data), size)
			}

			if backing == nil {
				t.Errorf("size=%d: backing buffer is nil", size)
			}

			ptr := uintptr(unsafe.Pointer(&data[0]))
			if ptr%AlignmentBytes != 0 {
				t.Errorf("size=%d: pointer 0x%x not %d-byte aligned", size, ptr, AlignmentBytes)
			}

			for i := range data {
				data[i] = complex(float32(i), float32(-i))
			}

			for i := range data {
				if want := complex(float32(i), float32(-i)); data[i] != want {
					t.Errorf("size=%d: data[%d]=%v, want %v", size, i, data[i], want)
				}
			}
		}
	})

	t.Run("complex128", func(t *testing.T) {
		t.Parallel()

		for _, size := range sizes {
			data, backing := AllocAligned[complex128](size)

			if len(data) != size {
				t.Errorf("size=%d: len(data)=%d, want %d", size, len(data), size)
			}

			if backing == nil {
				t.Errorf("size=%d: backing buffer is nil", size)
			}

			ptr := uintptr(unsafe.Pointer(&data[0]))
			if ptr%AlignmentBytes != 0 {
				t.Errorf("size=%d: pointer 0x%x not %d-byte aligned", size, ptr, AlignmentBytes)
			}

			for i := range data {
				data[i] = complex(float64(i), float64(-i))
			}

			for i := range data {
				if want := complex(float64(i), float64(-i)); data[i] != want {
					t.Errorf("size=%d: data[%d]=%v, want %v", size, i, data[i], want)
				}
			}
		}
	})
}

// TestAllocAlignedGenericZero verifies n <= 0 returns a nil buffer and backing,
// matching the concrete helpers.
func TestAllocAlignedGenericZero(t *testing.T) {
	t.Parallel()

	data, backing := AllocAligned[complex64](0)
	if data != nil || backing != nil {
		t.Errorf("AllocAligned[complex64](0) = (%v, %v), want (nil, nil)", data, backing)
	}

	data128, backing128 := AllocAligned[complex128](-4)
	if data128 != nil || backing128 != nil {
		t.Errorf("AllocAligned[complex128](-4) = (%v, %v), want (nil, nil)", data128, backing128)
	}
}
