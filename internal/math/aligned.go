package math

import "unsafe"

// AlignmentBytes defines the byte alignment for SIMD-friendly buffers.
const AlignmentBytes = 64

// AllocAlignedComplex64 allocates a complex64 slice aligned to AlignmentBytes.
// The returned raw slice must be kept alive to avoid GC reclaiming the backing memory.
func AllocAlignedComplex64(n int) ([]complex64, []byte) {
	if n <= 0 {
		return nil, nil
	}

	size := n * int(unsafe.Sizeof(complex64(0)))
	raw := make([]byte, size+AlignmentBytes-1)
	base := uintptr(unsafe.Pointer(&raw[0]))
	aligned := alignPtr(base, AlignmentBytes)
	offset := int(aligned - base)
	data := unsafe.Slice((*complex64)(unsafe.Pointer(&raw[offset])), n)

	return data, raw
}

// AllocAlignedComplex128 allocates a complex128 slice aligned to AlignmentBytes.
// The returned raw slice must be kept alive to avoid GC reclaiming the backing memory.
func AllocAlignedComplex128(n int) ([]complex128, []byte) {
	if n <= 0 {
		return nil, nil
	}

	size := n * int(unsafe.Sizeof(complex128(0)))
	raw := make([]byte, size+AlignmentBytes-1)
	base := uintptr(unsafe.Pointer(&raw[0]))
	aligned := alignPtr(base, AlignmentBytes)
	offset := int(aligned - base)
	data := unsafe.Slice((*complex128)(unsafe.Pointer(&raw[offset])), n)

	return data, raw
}

// AlignPtr aligns ptr to the given alignment boundary.
func AlignPtr(ptr uintptr, alignment int) uintptr {
	mask := uintptr(alignment - 1)
	return (ptr + mask) & ^mask
}

// alignPtr is a private alias.
func alignPtr(ptr uintptr, alignment int) uintptr {
	return AlignPtr(ptr, alignment)
}
