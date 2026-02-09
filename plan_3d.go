package algofft

import (
	"fmt"

	"github.com/cwbudde/algo-fft/internal/cpu"
	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// Plan3D is a pre-computed 3D FFT plan for a specific volume size and precision.
// Plans are reusable and safe for concurrent use during transforms (but not during creation).
//
// The 3D FFT uses the dimension-by-dimension decomposition algorithm:
// - Forward: FFT along width (innermost), then height, then depth (outermost)
// - Inverse: IFFT along width, then height, then depth
//
// Data layout is row-major: volume[d*height*width + h*width + w]
// where d is depth index, h is height index, w is width index.
//
// The generic type parameter T must be either complex64 or complex128.
type Plan3D[T Complex] struct {
	depth, height, width int      // Volume dimensions
	widthPlan            *Plan[T] // Plan for transforming along width (size=width)
	heightPlan           *Plan[T] // Plan for transforming along height (size=height)
	depthPlan            *Plan[T] // Plan for transforming along depth (size=depth)
	scratch              []T      // Working buffer (size=depth*height*width)
	dimScratch           []T      // Dimension scratch buffer for strided transforms (size=max(height,depth))
	options              PlanOptions

	// backing keeps aligned scratch buffer alive for GC
	scratchBacking []byte
}

// NewPlan3D creates a new 3D FFT plan for a depth×height×width volume.
//
// All dimensions must be ≥ 1. The plan supports arbitrary sizes via Bluestein's algorithm,
// though power-of-2 and highly-composite sizes (products of small primes) are most efficient.
//
// The plan pre-allocates all necessary buffers, enabling zero-allocation transforms.
//
// For concurrent use, create separate plans via Clone() for each goroutine.
func NewPlan3D[T Complex](depth, height, width int) (*Plan3D[T], error) {
	return NewPlan3DWithOptions[T](depth, height, width, PlanOptions{})
}

// NewPlan3DWithOptions creates a new 3D FFT plan with explicit planner options.
//
//nolint:funlen
func NewPlan3DWithOptions[T Complex](depth, height, width int, opts PlanOptions) (*Plan3D[T], error) {
	if depth <= 0 || height <= 0 || width <= 0 {
		return nil, ErrInvalidLength
	}

	opts = normalizePlanOptions(opts)
	features := cpu.DetectFeatures()

	childOpts := opts
	childOpts.Batch = 0
	childOpts.Stride = 0
	childOpts.InPlace = false

	// Create 1D plans for each dimension
	widthPlan, err := newPlanWithFeatures[T](width, features, childOpts)
	if err != nil {
		return nil, err
	}

	heightPlan, err := newPlanWithFeatures[T](height, features, childOpts)
	if err != nil {
		return nil, err
	}

	depthPlan, err := newPlanWithFeatures[T](depth, features, childOpts)
	if err != nil {
		return nil, err
	}

	// Allocate scratch buffer (aligned for SIMD)
	var (
		scratch        []T
		scratchBacking []byte
	)

	totalSize := depth * height * width

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

	// Allocate dimension scratch buffer for strided transforms
	dimScratchSize := height
	if depth > height {
		dimScratchSize = depth
	}

	dimScratch := make([]T, dimScratchSize)

	return &Plan3D[T]{
		depth:          depth,
		height:         height,
		width:          width,
		widthPlan:      widthPlan,
		heightPlan:     heightPlan,
		depthPlan:      depthPlan,
		scratch:        scratch,
		dimScratch:     dimScratch,
		scratchBacking: scratchBacking,
		options:        opts,
	}, nil
}

// NewPlan3D32 creates a new 3D FFT plan using complex64 precision.
// This is a convenience wrapper for NewPlan3D[complex64].
func NewPlan3D32(depth, height, width int) (*Plan3D[complex64], error) {
	return NewPlan3DWithOptions[complex64](depth, height, width, PlanOptions{})
}

// NewPlan3D64 creates a new 3D FFT plan using complex128 precision.
// This is a convenience wrapper for NewPlan3D[complex128].
func NewPlan3D64(depth, height, width int) (*Plan3D[complex128], error) {
	return NewPlan3DWithOptions[complex128](depth, height, width, PlanOptions{})
}

// Depth returns the depth dimension of the volume.
func (p *Plan3D[T]) Depth() int {
	return p.depth
}

// Height returns the height dimension of the volume.
func (p *Plan3D[T]) Height() int {
	return p.height
}

// Width returns the width dimension of the volume.
func (p *Plan3D[T]) Width() int {
	return p.width
}

// Len returns the total number of elements (depth × height × width).
func (p *Plan3D[T]) Len() int {
	return p.depth * p.height * p.width
}

// String returns a human-readable description of the Plan3D for debugging.
func (p *Plan3D[T]) String() string {
	var zero T

	typeName := "complex64"
	if _, ok := any(zero).(complex128); ok {
		typeName = "complex128"
	}

	return fmt.Sprintf("Plan3D[%s](%dx%dx%d)", typeName, p.depth, p.height, p.width)
}

// Forward computes the 3D FFT: dst = FFT3D(src).
//
// The input src and output dst must both be row-major volumes of size depth×height×width.
// Both slices must have exactly depth*height*width elements.
//
// Supports in-place operation (dst == src).
//
// Formula: X[kd,kh,kw] = Σ(d=0..depth-1) Σ(h=0..height-1) Σ(w=0..width-1)
//
//	x[d,h,w] * exp(-2πi*(kd*d/depth + kh*h/height + kw*w/width))
func (p *Plan3D[T]) Forward(dst, src []T) error {
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

// Inverse computes the 3D IFFT: dst = IFFT3D(src).
//
// The input src and output dst must both be row-major volumes of size depth×height×width.
// Both slices must have exactly depth*height*width elements.
//
// Supports in-place operation (dst == src).
//
// Formula: x[d,h,w] = (1/(depth*height*width)) * Σ(kd=0..depth-1) Σ(kh=0..height-1) Σ(kw=0..width-1)
//
//	X[kd,kh,kw] * exp(2πi*(kd*d/depth + kh*h/height + kw*w/width))
func (p *Plan3D[T]) Inverse(dst, src []T) error {
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

// ForwardInPlace computes the 3D FFT in-place: data = FFT3D(data).
// This is equivalent to Forward(data, data).
func (p *Plan3D[T]) ForwardInPlace(data []T) error {
	return p.Forward(data, data)
}

// InverseInPlace computes the 3D IFFT in-place: data = IFFT3D(data).
// This is equivalent to Inverse(data, data).
func (p *Plan3D[T]) InverseInPlace(data []T) error {
	return p.Inverse(data, data)
}

// Clone creates an independent copy of the Plan3D for concurrent use.
//
// The clone shares immutable data but has its own:
// - Scratch buffer (for thread safety)
// - 1D plan instances (cloned from originals)
//
// This allows multiple goroutines to perform transforms concurrently.
func (p *Plan3D[T]) Clone() *Plan3D[T] {
	// Allocate new scratch buffer
	var (
		scratch        []T
		scratchBacking []byte
	)

	totalSize := p.depth * p.height * p.width

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

	// Allocate dimension scratch buffer for strided transforms
	dimScratchSize := p.height
	if p.depth > p.height {
		dimScratchSize = p.depth
	}

	dimScratch := make([]T, dimScratchSize)

	return &Plan3D[T]{
		depth:          p.depth,
		height:         p.height,
		width:          p.width,
		widthPlan:      p.widthPlan.Clone(),
		heightPlan:     p.heightPlan.Clone(),
		depthPlan:      p.depthPlan.Clone(),
		scratch:        scratch,
		dimScratch:     dimScratch,
		scratchBacking: scratchBacking,
		options:        p.options,
	}
}

// validate checks that dst and src have the correct length for this plan.
func (p *Plan3D[T]) validate(dst, src []T) error {
	expectedLen := p.depth * p.height * p.width

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

// transformWidth transforms along the width dimension (innermost).
// Each row of width elements is transformed in-place.
func (p *Plan3D[T]) transformWidth(data []T, forward bool) {
	for d := range p.depth {
		for h := range p.height {
			offset := d*p.height*p.width + h*p.width

			rowData := data[offset : offset+p.width]
			if forward {
				_ = p.widthPlan.InPlace(rowData)
			} else {
				_ = p.widthPlan.InverseInPlace(rowData)
			}
		}
	}
}

// transformHeight transforms along the height dimension (middle).
// For each depth slice, columns along height are extracted, transformed, and written back.
func (p *Plan3D[T]) transformHeight(data []T, forward bool) {
	colData := p.dimScratch[:p.height]

	for d := range p.depth {
		for w := range p.width {
			// Extract column along height
			for h := range p.height {
				colData[h] = data[d*p.height*p.width+h*p.width+w]
			}

			// Transform column
			if forward {
				_ = p.heightPlan.InPlace(colData)
			} else {
				_ = p.heightPlan.InverseInPlace(colData)
			}

			// Write back
			for h := range p.height {
				data[d*p.height*p.width+h*p.width+w] = colData[h]
			}
		}
	}
}

// transformDepth transforms along the depth dimension (outermost).
// For each (height, width) position, a slice along depth is extracted, transformed, and written back.
func (p *Plan3D[T]) transformDepth(data []T, forward bool) {
	depthData := p.dimScratch[:p.depth]

	for h := range p.height {
		for w := range p.width {
			// Extract slice along depth
			for d := range p.depth {
				depthData[d] = data[d*p.height*p.width+h*p.width+w]
			}

			// Transform depth slice
			if forward {
				_ = p.depthPlan.InPlace(depthData)
			} else {
				_ = p.depthPlan.InverseInPlace(depthData)
			}

			// Write back
			for d := range p.depth {
				data[d*p.height*p.width+h*p.width+w] = depthData[d]
			}
		}
	}
}

func (p *Plan3D[T]) forwardSingle(dst, src []T) error {
	err := p.validate(dst, src)
	if err != nil {
		return err
	}

	work := p.scratch
	copy(work, src)

	p.transformWidth(work, true)
	p.transformHeight(work, true)
	p.transformDepth(work, true)

	copy(dst, work)

	return nil
}

func (p *Plan3D[T]) inverseSingle(dst, src []T) error {
	err := p.validate(dst, src)
	if err != nil {
		return err
	}

	work := p.scratch
	copy(work, src)

	p.transformWidth(work, false)
	p.transformHeight(work, false)
	p.transformDepth(work, false)

	copy(dst, work)

	return nil
}
