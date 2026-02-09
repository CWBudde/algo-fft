package algofft

import (
	"fmt"

	"github.com/cwbudde/algo-fft/internal/cpu"
	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// PlanND is a pre-computed N-dimensional FFT plan for arbitrary dimensions.
// Plans are reusable and safe for concurrent use during transforms (but not during creation).
//
// The N-D FFT uses the dimension-by-dimension decomposition algorithm:
// transforms are applied sequentially along each axis from innermost to outermost.
//
// Data layout is row-major with the last dimension varying fastest:
// index = d[0]*stride[0] + d[1]*stride[1] + ... + d[N-1]*stride[N-1]
//
// The generic type parameter T must be either complex64 or complex128.
type PlanND[T Complex] struct {
	dims    []int      // Dimension sizes [d0, d1, ..., dN-1]
	plans   []*Plan[T] // 1D plans for each dimension
	scratch []T        // Working buffer (size = product of all dims)
	strides []int      // Pre-computed strides for each dimension
	options PlanOptions

	// backing keeps aligned scratch buffer alive for GC
	scratchBacking []byte
}

// NewPlanND creates a new N-dimensional FFT plan for the given dimension sizes.
//
// dims specifies the size of each dimension. For example:
//   - NewPlanND[complex64]([]int{8, 16, 32}) creates an 8×16×32 3D FFT
//   - NewPlanND[complex64]([]int{4, 4, 4, 4}) creates a 4D FFT
//
// All dimensions must be ≥ 1. The plan supports arbitrary sizes via Bluestein's algorithm,
// though power-of-2 and highly-composite sizes are most efficient.
//
// The plan pre-allocates all necessary buffers, enabling zero-allocation transforms.
//
// For concurrent use, create separate plans via Clone() for each goroutine.
func NewPlanND[T Complex](dims []int) (*PlanND[T], error) {
	return NewPlanNDWithOptions[T](dims, PlanOptions{})
}

// NewPlanNDWithOptions creates a new N-dimensional FFT plan with explicit planner options.
//
//nolint:funlen
func NewPlanNDWithOptions[T Complex](dims []int, opts PlanOptions) (*PlanND[T], error) {
	if len(dims) == 0 {
		return nil, ErrInvalidLength
	}

	opts = normalizePlanOptions(opts)
	features := cpu.DetectFeatures()

	// Validate all dimensions
	totalSize := 1

	for i, d := range dims {
		if d <= 0 {
			return nil, fmt.Errorf("dimension %d has invalid size %d: %w", i, d, ErrInvalidLength)
		}

		totalSize *= d
	}

	// Create a copy of dims to avoid external mutations
	dimsCopy := make([]int, len(dims))
	copy(dimsCopy, dims)

	childOpts := opts
	childOpts.Batch = 0
	childOpts.Stride = 0
	childOpts.InPlace = false

	// Create 1D plans for each dimension
	plans := make([]*Plan[T], len(dims))
	for i, size := range dimsCopy {
		plan, err := newPlanWithFeatures[T](size, features, childOpts)
		if err != nil {
			return nil, fmt.Errorf("failed to create plan for dimension %d (size %d): %w", i, size, err)
		}

		plans[i] = plan
	}

	// Allocate scratch buffer (aligned for SIMD)
	var (
		scratch        []T
		scratchBacking []byte
	)

	switch any(scratch).(type) {
	case []complex64:
		s, b := mem.AllocAlignedComplex64(totalSize)
		scratch = any(s).([]T)
		scratchBacking = b
	case []complex128:
		s, b := mem.AllocAlignedComplex128(totalSize)
		scratch = any(s).([]T)
		scratchBacking = b
	}

	// Pre-compute strides for efficient indexing
	strides := make([]int, len(dims))

	stride := 1
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= dimsCopy[i]
	}

	return &PlanND[T]{
		dims:           dimsCopy,
		plans:          plans,
		scratch:        scratch,
		strides:        strides,
		scratchBacking: scratchBacking,
		options:        opts,
	}, nil
}

// NewPlanND32 creates a new N-dimensional FFT plan using complex64 precision.
// This is a convenience wrapper for NewPlanND[complex64].
func NewPlanND32(dims []int) (*PlanND[complex64], error) {
	return NewPlanNDWithOptions[complex64](dims, PlanOptions{})
}

// NewPlanND64 creates a new N-dimensional FFT plan using complex128 precision.
// This is a convenience wrapper for NewPlanND[complex128].
func NewPlanND64(dims []int) (*PlanND[complex128], error) {
	return NewPlanNDWithOptions[complex128](dims, PlanOptions{})
}

// Dims returns a copy of the dimension sizes.
func (p *PlanND[T]) Dims() []int {
	result := make([]int, len(p.dims))
	copy(result, p.dims)

	return result
}

// NDims returns the number of dimensions.
func (p *PlanND[T]) NDims() int {
	return len(p.dims)
}

// Len returns the total number of elements (product of all dimensions).
func (p *PlanND[T]) Len() int {
	total := 1
	for _, d := range p.dims {
		total *= d
	}

	return total
}

// String returns a human-readable description of the PlanND for debugging.
func (p *PlanND[T]) String() string {
	var zero T

	typeName := "complex64"
	if _, ok := any(zero).(complex128); ok {
		typeName = "complex128"
	}

	dimsStr := ""

	for i, d := range p.dims {
		if i > 0 {
			dimsStr += "x"
		}

		dimsStr += itoa(d)
	}

	return fmt.Sprintf("PlanND[%s](%s)", typeName, dimsStr)
}

// Forward computes the N-D FFT: dst = FFT_ND(src).
//
// The input src and output dst must both have exactly Len() elements.
// Data is expected in row-major order with the last dimension varying fastest.
//
// Supports in-place operation (dst == src).
func (p *PlanND[T]) Forward(dst, src []T) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	batch, stride, err := resolveBatchStride(p.Len(), p.options)
	if err != nil {
		return err
	}

	for b := range batch {
		srcOff := b * stride

		dstOff := b * stride
		if srcOff+p.Len() > len(src) || dstOff+p.Len() > len(dst) {
			return ErrLengthMismatch
		}

		err = p.forwardSingle(dst[dstOff:dstOff+p.Len()], src[srcOff:srcOff+p.Len()])
		if err != nil {
			return err
		}
	}

	return nil
}

// Inverse computes the N-D IFFT: dst = IFFT_ND(src).
//
// The input src and output dst must both have exactly Len() elements.
// Data is expected in row-major order with the last dimension varying fastest.
//
// Supports in-place operation (dst == src).
func (p *PlanND[T]) Inverse(dst, src []T) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	batch, stride, err := resolveBatchStride(p.Len(), p.options)
	if err != nil {
		return err
	}

	for b := range batch {
		srcOff := b * stride

		dstOff := b * stride
		if srcOff+p.Len() > len(src) || dstOff+p.Len() > len(dst) {
			return ErrLengthMismatch
		}

		err = p.inverseSingle(dst[dstOff:dstOff+p.Len()], src[srcOff:srcOff+p.Len()])
		if err != nil {
			return err
		}
	}

	return nil
}

// ForwardInPlace computes the N-D FFT in-place: data = FFT_ND(data).
// This is equivalent to Forward(data, data).
func (p *PlanND[T]) ForwardInPlace(data []T) error {
	return p.Forward(data, data)
}

// InverseInPlace computes the N-D IFFT in-place: data = IFFT_ND(data).
// This is equivalent to Inverse(data, data).
func (p *PlanND[T]) InverseInPlace(data []T) error {
	return p.Inverse(data, data)
}

// Clone creates an independent copy of the PlanND for concurrent use.
//
// The clone shares immutable data but has its own:
// - Scratch buffer (for thread safety)
// - 1D plan instances (cloned from originals)
//
// This allows multiple goroutines to perform transforms concurrently.
func (p *PlanND[T]) Clone() *PlanND[T] {
	// Allocate new scratch buffer
	var (
		scratch        []T
		scratchBacking []byte
	)

	totalSize := p.Len()

	switch any(scratch).(type) {
	case []complex64:
		s, b := mem.AllocAlignedComplex64(totalSize)
		scratch = any(s).([]T)
		scratchBacking = b
	case []complex128:
		s, b := mem.AllocAlignedComplex128(totalSize)
		scratch = any(s).([]T)
		scratchBacking = b
	}

	// Clone all 1D plans
	plans := make([]*Plan[T], len(p.plans))
	for i, plan := range p.plans {
		plans[i] = plan.Clone()
	}

	// Copy dimensions and strides
	dims := make([]int, len(p.dims))
	copy(dims, p.dims)

	strides := make([]int, len(p.strides))
	copy(strides, p.strides)

	return &PlanND[T]{
		dims:           dims,
		plans:          plans,
		scratch:        scratch,
		strides:        strides,
		scratchBacking: scratchBacking,
		options:        p.options,
	}
}

// validate checks that dst and src have the correct length for this plan.
func (p *PlanND[T]) validate(dst, src []T) error {
	expectedLen := p.Len()

	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(dst) != expectedLen {
		return ErrLengthMismatch
	}

	if len(src) != expectedLen {
		return ErrLengthMismatch
	}

	return nil
}

// transformDimension applies 1D FFT along the specified dimension.
// This extracts slices along the dimension, transforms them, and writes back.
func (p *PlanND[T]) transformDimension(data []T, dim int, forward bool) error {
	dimSize := p.dims[dim]
	plan := p.plans[dim]

	// Allocate buffer for one slice along this dimension
	sliceData := make([]T, dimSize)

	// Total number of slices to process
	totalSlices := p.Len() / dimSize

	// Iterate through all slices along this dimension
	for sliceIdx := range totalSlices {
		// Extract slice
		p.extractSlice(data, sliceData, sliceIdx, dim)

		// Transform slice
		var err error
		if forward {
			err = plan.InPlace(sliceData)
		} else {
			err = plan.InverseInPlace(sliceData)
		}

		if err != nil {
			return err
		}

		// Write back
		p.writeSlice(data, sliceData, sliceIdx, dim)
	}

	return nil
}

// extractSlice extracts a 1D slice along the specified dimension.
// sliceIdx identifies which slice (0 to totalSlices-1).
func (p *PlanND[T]) extractSlice(data, dst []T, sliceIdx, dim int) {
	dimSize := p.dims[dim]
	dimStride := p.strides[dim]

	// Compute base offset for this slice
	baseOffset := p.sliceIndexToOffset(sliceIdx, dim)

	// Extract elements along the dimension
	for i := range dimSize {
		offset := baseOffset + i*dimStride
		dst[i] = data[offset]
	}
}

// writeSlice writes a 1D slice back along the specified dimension.
func (p *PlanND[T]) writeSlice(data, src []T, sliceIdx, dim int) {
	dimSize := p.dims[dim]
	dimStride := p.strides[dim]

	// Compute base offset for this slice
	baseOffset := p.sliceIndexToOffset(sliceIdx, dim)

	// Write elements along the dimension
	for i := range dimSize {
		offset := baseOffset + i*dimStride
		data[offset] = src[i]
	}
}

func (p *PlanND[T]) forwardSingle(dst, src []T) error {
	err := p.validate(dst, src)
	if err != nil {
		return err
	}

	work := p.scratch
	copy(work, src)

	for dim := len(p.dims) - 1; dim >= 0; dim-- {
		err = p.transformDimension(work, dim, true)
		if err != nil {
			return err
		}
	}

	copy(dst, work)

	return nil
}

func (p *PlanND[T]) inverseSingle(dst, src []T) error {
	err := p.validate(dst, src)
	if err != nil {
		return err
	}

	work := p.scratch
	copy(work, src)

	for dim := len(p.dims) - 1; dim >= 0; dim-- {
		err = p.transformDimension(work, dim, false)
		if err != nil {
			return err
		}
	}

	copy(dst, work)

	return nil
}

// sliceIndexToOffset converts a linear slice index to the base offset in scratch buffer.
// This computes the offset for the first element of the slice.
func (p *PlanND[T]) sliceIndexToOffset(sliceIdx, dim int) int {
	// Build array of "reduced" dimensions (all dims except the transform dimension)
	reducedDims := make([]int, 0, len(p.dims)-1)
	for d := range len(p.dims) {
		if d != dim {
			reducedDims = append(reducedDims, p.dims[d])
		}
	}

	// Convert linear sliceIdx to coordinates in reduced space
	coords := make([]int, len(reducedDims))

	remaining := sliceIdx
	for i := len(reducedDims) - 1; i >= 0; i-- {
		coords[i] = remaining % reducedDims[i]
		remaining /= reducedDims[i]
	}

	// Map reduced coordinates back to full coordinates and compute offset
	offset := 0
	reducedIdx := 0

	for d := range len(p.dims) {
		if d == dim {
			// This dimension is set to 0 (first element along transform axis)
			continue
		}

		offset += coords[reducedIdx] * p.strides[d]
		reducedIdx++
	}

	return offset
}
