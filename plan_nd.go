package algofft

import (
	"fmt"
	"strings"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// PlanND is a pre-computed N-dimensional FFT plan for arbitrary dimensions.
// Plans are reusable and safe for concurrent use during transforms (but not
// during creation): scratch buffers are borrowed per call from an internal
// cache, so multiple goroutines may share one instance.
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
	strides []int      // Pre-computed strides for each dimension
	options PlanOptions

	// scratch hands out per-call working buffers for thread-safety.
	scratch *residentCache[planNDScratch[T]]
}

// planNDScratch is one per-call scratch set for PlanND transforms.
type planNDScratch[T Complex] struct {
	work        []T    // Working buffer (size = product of all dims)
	workBacking []byte // Keeps the aligned working buffer alive for GC
	slice       []T    // One transform slice, sized max(dims)
	reducedDims []int  // Reduced dimension sizes (cap = len(dims))
	coords      []int  // Reduced-space coordinates (cap = len(dims))
}

func newPlanNDScratchCache[T Complex](dims []int) *residentCache[planNDScratch[T]] {
	totalSize := 1
	maxDim := 0

	for _, d := range dims {
		totalSize *= d
		if d > maxDim {
			maxDim = d
		}
	}

	numDims := len(dims)

	return newResidentCache(func() *planNDScratch[T] {
		work, backing := allocAlignedSlice[T](totalSize)

		return &planNDScratch[T]{
			work:        work,
			workBacking: backing,
			slice:       make([]T, maxDim),
			reducedDims: make([]int, 0, numDims),
			coords:      make([]int, numDims),
		}
	})
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
// A single plan instance may be shared by multiple goroutines.
func NewPlanND[T Complex](dims []int) (*PlanND[T], error) {
	return NewPlanNDWithOptions[T](dims, PlanOptions{})
}

// NewPlanNDWithOptions creates a new N-dimensional FFT plan with explicit planner options.
func NewPlanNDWithOptions[T Complex](dims []int, opts PlanOptions) (*PlanND[T], error) {
	if len(dims) == 0 {
		return nil, ErrInvalidLength
	}

	opts = normalizePlanOptions(opts)
	features := cpu.DetectFeatures()

	// Validate all dimensions
	for i, d := range dims {
		if d <= 0 {
			return nil, fmt.Errorf("dimension %d has invalid size %d: %w", i, d, ErrInvalidLength)
		}
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

	// Pre-compute strides for efficient indexing
	strides := make([]int, len(dims))

	stride := 1
	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= dimsCopy[i]
	}

	return &PlanND[T]{
		dims:    dimsCopy,
		plans:   plans,
		scratch: newPlanNDScratchCache[T](dimsCopy),
		strides: strides,
		options: opts,
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

	var dims strings.Builder

	for i, d := range p.dims {
		if i > 0 {
			dims.WriteString("x")
		}

		dims.WriteString(itoa(d))
	}

	return fmt.Sprintf("PlanND[%s](%s)", typeName, dims.String())
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

// Clone creates an independent copy of the PlanND.
//
// A single PlanND is already safe for concurrent transforms, so cloning is
// not required for concurrency; it remains available for callers that want
// isolated scratch caches. The clone shares the concurrency-safe 1D child
// plans but has its own scratch cache.
func (p *PlanND[T]) Clone() *PlanND[T] {
	// Copy dimensions and strides
	dims := make([]int, len(p.dims))
	copy(dims, p.dims)

	strides := make([]int, len(p.strides))
	copy(strides, p.strides)

	return &PlanND[T]{
		dims:    dims,
		plans:   p.plans,
		scratch: newPlanNDScratchCache[T](dims),
		strides: strides,
		options: p.options,
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
// All working buffers come from s, so the transform allocates nothing.
func (p *PlanND[T]) transformDimension(s *planNDScratch[T], dim int, forward bool) error {
	dimSize := p.dims[dim]
	plan := p.plans[dim]

	data := s.work
	sliceData := s.slice[:dimSize]

	// Reduced dimension sizes depend only on dim, so build them once per axis
	// rather than once per slice. The backing array is reused across calls.
	reduced := s.reducedDims[:0]

	for d := range len(p.dims) {
		if d != dim {
			reduced = append(reduced, p.dims[d])
		}
	}

	// Total number of slices to process
	totalSlices := p.Len() / dimSize

	// Iterate through all slices along this dimension
	for sliceIdx := range totalSlices {
		baseOffset := p.sliceIndexToOffset(reduced, s.coords, sliceIdx, dim)

		// Extract slice
		p.extractSlice(data, sliceData, baseOffset, dim)

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
		p.writeSlice(data, sliceData, baseOffset, dim)
	}

	return nil
}

// extractSlice extracts a 1D slice along the specified dimension, starting at
// baseOffset (the offset of the slice's first element).
func (p *PlanND[T]) extractSlice(data, dst []T, baseOffset, dim int) {
	dimSize := p.dims[dim]
	dimStride := p.strides[dim]

	// Extract elements along the dimension
	for i := range dimSize {
		dst[i] = data[baseOffset+i*dimStride]
	}
}

// writeSlice writes a 1D slice back along the specified dimension, starting at
// baseOffset (the offset of the slice's first element).
func (p *PlanND[T]) writeSlice(data, src []T, baseOffset, dim int) {
	dimSize := p.dims[dim]
	dimStride := p.strides[dim]

	// Write elements along the dimension
	for i := range dimSize {
		data[baseOffset+i*dimStride] = src[i]
	}
}

func (p *PlanND[T]) forwardSingle(dst, src []T) error {
	err := p.validate(dst, src)
	if err != nil {
		return err
	}

	s := p.scratch.get()
	defer p.scratch.put(s)

	work := s.work
	copy(work, src)

	for dim := len(p.dims) - 1; dim >= 0; dim-- {
		err = p.transformDimension(s, dim, true)
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

	s := p.scratch.get()
	defer p.scratch.put(s)

	work := s.work
	copy(work, src)

	for dim := len(p.dims) - 1; dim >= 0; dim-- {
		err = p.transformDimension(s, dim, false)
		if err != nil {
			return err
		}
	}

	copy(dst, work)

	return nil
}

// sliceIndexToOffset converts a linear slice index to the base offset (the
// offset of the slice's first element along dim).
//
// reducedDims holds the sizes of all dimensions except dim (as built by
// transformDimension); coords is scratch of length >= len(reducedDims) used to
// hold the reduced-space coordinates. Both are caller-owned so this allocates
// nothing.
func (p *PlanND[T]) sliceIndexToOffset(reducedDims, coords []int, sliceIdx, dim int) int {
	coords = coords[:len(reducedDims)]

	// Convert linear sliceIdx to coordinates in reduced space
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
