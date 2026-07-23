package algofft

import (
	"fmt"
)

// Plan3D is a pre-computed 3D FFT plan for a specific volume size and precision.
// Plans are reusable and safe for concurrent use during transforms (but not
// during creation): scratch buffers are borrowed per call from an internal
// cache, so multiple goroutines may share one instance.
//
// Plan3D is a typed wrapper over the N-dimensional engine (PlanND) using the
// dimension-by-dimension decomposition: FFT along width (innermost), then
// height, then depth (outermost).
//
// Data layout is row-major: volume[d*height*width + h*width + w]
// where d is depth index, h is height index, w is width index.
//
// The generic type parameter T must be either complex64 or complex128.
type Plan3D[T Complex] struct {
	depth, height, width int
	nd                   *PlanND[T]
}

// NewPlan3D creates a new 3D FFT plan for a depth×height×width volume.
//
// All dimensions must be ≥ 1. The plan supports arbitrary sizes via Bluestein's algorithm,
// though power-of-2 and highly-composite sizes (products of small primes) are most efficient.
//
// The plan pre-allocates all necessary buffers, enabling zero-allocation transforms.
//
// A single plan instance may be shared by multiple goroutines.
func NewPlan3D[T Complex](depth, height, width int) (*Plan3D[T], error) {
	return NewPlan3DWithOptions[T](depth, height, width, PlanOptions{})
}

// NewPlan3DWithOptions creates a new 3D FFT plan with explicit planner options.
func NewPlan3DWithOptions[T Complex](depth, height, width int, opts PlanOptions) (*Plan3D[T], error) {
	if depth <= 0 {
		return nil, fmt.Errorf("depth has invalid size %d: %w", depth, ErrInvalidLength)
	}

	if height <= 0 {
		return nil, fmt.Errorf("height has invalid size %d: %w", height, ErrInvalidLength)
	}

	if width <= 0 {
		return nil, fmt.Errorf("width has invalid size %d: %w", width, ErrInvalidLength)
	}

	nd, err := NewPlanNDWithOptions[T]([]int{depth, height, width}, opts)
	if err != nil {
		return nil, err
	}

	return &Plan3D[T]{depth: depth, height: height, width: width, nd: nd}, nil
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
	return fmt.Sprintf("Plan3D[%s](%dx%dx%d)", complexTypeName[T](), p.depth, p.height, p.width)
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
	return p.nd.Forward(dst, src)
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
	return p.nd.Inverse(dst, src)
}

// ForwardInPlace computes the 3D FFT in-place: data = FFT3D(data).
// This is equivalent to Forward(data, data).
func (p *Plan3D[T]) ForwardInPlace(data []T) error {
	return p.nd.ForwardInPlace(data)
}

// InverseInPlace computes the 3D IFFT in-place: data = IFFT3D(data).
// This is equivalent to Inverse(data, data).
func (p *Plan3D[T]) InverseInPlace(data []T) error {
	return p.nd.InverseInPlace(data)
}

// Clone creates an independent copy of the Plan3D.
//
// A single Plan3D is already safe for concurrent transforms, so cloning is
// not required for concurrency; it remains available for callers that want
// isolated scratch caches. The clone shares the concurrency-safe 1D child
// plans but has its own scratch cache.
func (p *Plan3D[T]) Clone() *Plan3D[T] {
	return &Plan3D[T]{depth: p.depth, height: p.height, width: p.width, nd: p.nd.Clone()}
}
