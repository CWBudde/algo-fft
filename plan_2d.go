package algofft

import (
	"fmt"
)

// Plan2D is a pre-computed 2D FFT plan for a specific matrix size and precision.
// Plans are reusable and safe for concurrent use during transforms (but not
// during creation): scratch buffers are borrowed per call from an internal
// cache, so multiple goroutines may share one instance.
//
// Plan2D is a typed wrapper over the N-dimensional engine (PlanND) using the
// row-column decomposition: FFT rows, then FFT columns. Square matrices
// transform columns via a cache-friendly transpose.
//
// Data layout is row-major: matrix[row*cols + col]
//
// The generic type parameter T must be either complex64 or complex128.
type Plan2D[T Complex] struct {
	rows, cols int
	nd         *PlanND[T]
}

// NewPlan2D creates a new 2D FFT plan for a rows×cols matrix.
//
// Both rows and cols must be ≥ 1. The plan supports arbitrary sizes via Bluestein's algorithm,
// though power-of-2 and highly-composite sizes (products of small primes) are most efficient.
//
// The plan pre-allocates all necessary buffers, enabling zero-allocation transforms.
//
// A single plan instance may be shared by multiple goroutines.
func NewPlan2D[T Complex](rows, cols int) (*Plan2D[T], error) {
	return NewPlan2DWithOptions[T](rows, cols, PlanOptions{})
}

// NewPlan2DWithOptions creates a new 2D FFT plan with explicit planner options.
func NewPlan2DWithOptions[T Complex](rows, cols int, opts PlanOptions) (*Plan2D[T], error) {
	if rows <= 0 {
		return nil, fmt.Errorf("rows has invalid size %d: %w", rows, ErrInvalidLength)
	}

	if cols <= 0 {
		return nil, fmt.Errorf("cols has invalid size %d: %w", cols, ErrInvalidLength)
	}

	nd, err := NewPlanNDWithOptions[T]([]int{rows, cols}, opts)
	if err != nil {
		return nil, err
	}

	return &Plan2D[T]{rows: rows, cols: cols, nd: nd}, nil
}

// NewPlan2D32 creates a new 2D FFT plan using complex64 precision.
// This is a convenience wrapper for NewPlan2D[complex64].
func NewPlan2D32(rows, cols int) (*Plan2D[complex64], error) {
	return NewPlan2DWithOptions[complex64](rows, cols, PlanOptions{})
}

// NewPlan2D64 creates a new 2D FFT plan using complex128 precision.
// This is a convenience wrapper for NewPlan2D[complex128].
func NewPlan2D64(rows, cols int) (*Plan2D[complex128], error) {
	return NewPlan2DWithOptions[complex128](rows, cols, PlanOptions{})
}

// Rows returns the number of rows in the matrix.
func (p *Plan2D[T]) Rows() int {
	return p.rows
}

// Cols returns the number of columns in the matrix.
func (p *Plan2D[T]) Cols() int {
	return p.cols
}

// Len returns the total number of elements (rows × cols).
func (p *Plan2D[T]) Len() int {
	return p.rows * p.cols
}

// String returns a human-readable description of the Plan2D for debugging.
func (p *Plan2D[T]) String() string {
	return fmt.Sprintf("Plan2D[%s](%dx%d)", complexTypeName[T](), p.rows, p.cols)
}

// Forward computes the 2D FFT: dst = FFT2D(src).
//
// The input src and output dst must both be row-major matrices of size rows×cols.
// Both slices must have exactly rows*cols elements.
//
// Supports in-place operation (dst == src).
//
// Formula: X[k,l] = Σ(m=0..rows-1) Σ(n=0..cols-1) x[m,n] * exp(-2πi*(km/rows + ln/cols)).
func (p *Plan2D[T]) Forward(dst, src []T) error {
	return p.nd.Forward(dst, src)
}

// Inverse computes the 2D IFFT: dst = IFFT2D(src).
//
// The input src and output dst must both be row-major matrices of size rows×cols.
// Both slices must have exactly rows*cols elements.
//
// Supports in-place operation (dst == src).
//
// Formula: x[m,n] = (1/(rows*cols)) * Σ(k=0..rows-1) Σ(l=0..cols-1) X[k,l] * exp(2πi*(km/rows + ln/cols)).
func (p *Plan2D[T]) Inverse(dst, src []T) error {
	return p.nd.Inverse(dst, src)
}

// ForwardInPlace computes the 2D FFT in-place: data = FFT2D(data).
// This is equivalent to Forward(data, data).
func (p *Plan2D[T]) ForwardInPlace(data []T) error {
	return p.nd.ForwardInPlace(data)
}

// InverseInPlace computes the 2D IFFT in-place: data = IFFT2D(data).
// This is equivalent to Inverse(data, data).
func (p *Plan2D[T]) InverseInPlace(data []T) error {
	return p.nd.InverseInPlace(data)
}

// Clone creates an independent copy of the Plan2D.
//
// A single Plan2D is already safe for concurrent transforms, so cloning is
// not required for concurrency; it remains available for callers that want
// isolated scratch caches. The clone shares the concurrency-safe 1D child
// plans, but has its own scratch cache.
func (p *Plan2D[T]) Clone() *Plan2D[T] {
	return &Plan2D[T]{rows: p.rows, cols: p.cols, nd: p.nd.Clone()}
}
