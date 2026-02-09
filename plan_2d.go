package algofft

import (
	"fmt"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// Plan2D is a pre-computed 2D FFT plan for a specific matrix size and precision.
// Plans are reusable and safe for concurrent use during transforms (but not during creation).
//
// The 2D FFT uses the row-column decomposition algorithm:
// - Forward: FFT rows, then FFT columns
// - Inverse: IFFT rows, then IFFT columns
//
// Data layout is row-major: matrix[row*cols + col]
//
// The generic type parameter T must be either complex64 or complex128.
type Plan2D[T Complex] struct {
	rows, cols int      // Matrix dimensions
	rowPlan    *Plan[T] // Plan for transforming rows (size=cols)
	colPlan    *Plan[T] // Plan for transforming columns (size=rows)
	scratch    []T      // Working buffer (size=rows*cols)
	colScratch []T      // Column scratch buffer for strided transforms (size=rows)
	options    PlanOptions

	// Transpose support for square matrices
	transposePairs []fft.TransposePair

	// backing keeps aligned scratch buffer alive for GC
	scratchBacking []byte
}

// NewPlan2D creates a new 2D FFT plan for a rows×cols matrix.
//
// Both rows and cols must be ≥ 1. The plan supports arbitrary sizes via Bluestein's algorithm,
// though power-of-2 and highly-composite sizes (products of small primes) are most efficient.
//
// The plan pre-allocates all necessary buffers, enabling zero-allocation transforms.
//
// For concurrent use, create separate plans via Clone() for each goroutine.
func NewPlan2D[T Complex](rows, cols int) (*Plan2D[T], error) {
	return NewPlan2DWithOptions[T](rows, cols, PlanOptions{})
}

// NewPlan2DWithOptions creates a new 2D FFT plan with explicit planner options.
func NewPlan2DWithOptions[T Complex](rows, cols int, opts PlanOptions) (*Plan2D[T], error) {
	if rows <= 0 || cols <= 0 {
		return nil, ErrInvalidLength
	}

	opts = normalizePlanOptions(opts)
	features := cpu.DetectFeatures()

	childOpts := opts
	childOpts.Batch = 0
	childOpts.Stride = 0
	childOpts.InPlace = false

	// Create 1D plans for rows and columns
	rowPlan, err := newPlanWithFeatures[T](cols, features, childOpts)
	if err != nil {
		return nil, err
	}

	colPlan, err := newPlanWithFeatures[T](rows, features, childOpts)
	if err != nil {
		return nil, err
	}

	// Allocate scratch buffer (aligned for SIMD)
	var (
		scratch        []T
		scratchBacking []byte
	)

	totalSize := rows * cols

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

	// Allocate column scratch buffer for strided transforms
	colScratch := make([]T, rows)

	p := &Plan2D[T]{
		rows:           rows,
		cols:           cols,
		rowPlan:        rowPlan,
		colPlan:        colPlan,
		scratch:        scratch,
		colScratch:     colScratch,
		scratchBacking: scratchBacking,
		options:        opts,
	}

	// Pre-compute transpose pairs for square matrices (optimization)
	if rows == cols {
		p.transposePairs = fft.ComputeSquareTransposePairs(rows)
	}

	return p, nil
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
	var zero T

	typeName := "complex64"
	if _, ok := any(zero).(complex128); ok {
		typeName = "complex128"
	}

	return fmt.Sprintf("Plan2D[%s](%dx%d)", typeName, p.rows, p.cols)
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

// Inverse computes the 2D IFFT: dst = IFFT2D(src).
//
// The input src and output dst must both be row-major matrices of size rows×cols.
// Both slices must have exactly rows*cols elements.
//
// Supports in-place operation (dst == src).
//
// Formula: x[m,n] = (1/(rows*cols)) * Σ(k=0..rows-1) Σ(l=0..cols-1) X[k,l] * exp(2πi*(km/rows + ln/cols)).
func (p *Plan2D[T]) Inverse(dst, src []T) error {
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

// ForwardInPlace computes the 2D FFT in-place: data = FFT2D(data).
// This is equivalent to Forward(data, data).
func (p *Plan2D[T]) ForwardInPlace(data []T) error {
	return p.Forward(data, data)
}

// InverseInPlace computes the 2D IFFT in-place: data = IFFT2D(data).
// This is equivalent to Inverse(data, data).
func (p *Plan2D[T]) InverseInPlace(data []T) error {
	return p.Inverse(data, data)
}

// Clone creates an independent copy of the Plan2D for concurrent use.
//
// The clone shares immutable data (transpose pairs) but has its own:
// - Scratch buffer (for thread safety)
// - 1D plan instances (cloned from originals)
//
// This allows multiple goroutines to perform transforms concurrently.
func (p *Plan2D[T]) Clone() *Plan2D[T] {
	// Allocate new scratch buffer
	var (
		scratch        []T
		scratchBacking []byte
	)

	totalSize := p.rows * p.cols

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

	// Allocate column scratch buffer for strided transforms
	colScratch := make([]T, p.rows)

	return &Plan2D[T]{
		rows:           p.rows,
		cols:           p.cols,
		rowPlan:        p.rowPlan.Clone(),
		colPlan:        p.colPlan.Clone(),
		scratch:        scratch,
		colScratch:     colScratch,
		scratchBacking: scratchBacking,
		transposePairs: p.transposePairs, // Shared (immutable)
		options:        p.options,
	}
}

// validate checks that dst and src have the correct length for this plan.
func (p *Plan2D[T]) validate(dst, src []T) error {
	expectedLen := p.rows * p.cols

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

// transformColumnsViaTranspose transforms columns using transpose for square matrices.
// This is more cache-friendly than strided access.
func (p *Plan2D[T]) transformColumnsViaTranspose(data []T, forward bool) {
	// Transpose: columns become rows
	fft.ApplyTransposePairs(data, p.transposePairs)

	// Transform each column (now a row)
	for row := range p.rows {
		rowData := data[row*p.cols : (row+1)*p.cols]
		if forward {
			_ = p.colPlan.InPlace(rowData)
		} else {
			_ = p.colPlan.InverseInPlace(rowData)
		}
	}

	// Transpose back
	fft.ApplyTransposePairs(data, p.transposePairs)
}

// transformColumnsStrided transforms columns using strided access for non-square matrices.
func (p *Plan2D[T]) transformColumnsStrided(data []T, forward bool) {
	colData := p.colScratch

	for col := range p.cols {
		// Extract column
		for row := range p.rows {
			colData[row] = data[row*p.cols+col]
		}

		// Transform column
		if forward {
			_ = p.colPlan.InPlace(colData)
		} else {
			_ = p.colPlan.InverseInPlace(colData)
		}

		// Write back
		for row := range p.rows {
			data[row*p.cols+col] = colData[row]
		}
	}
}

func (p *Plan2D[T]) forwardSingle(dst, src []T) error {
	err := p.validate(dst, src)
	if err != nil {
		return err
	}

	work := p.scratch
	copy(work, src)

	// Transform rows
	for row := range p.rows {
		rowData := work[row*p.cols : (row+1)*p.cols]

		err := p.rowPlan.InPlace(rowData)
		if err != nil {
			return err
		}
	}

	// Transform columns
	if p.rows == p.cols {
		p.transformColumnsViaTranspose(work, true)
	} else {
		p.transformColumnsStrided(work, true)
	}

	copy(dst, work)

	return nil
}

func (p *Plan2D[T]) inverseSingle(dst, src []T) error {
	err := p.validate(dst, src)
	if err != nil {
		return err
	}

	work := p.scratch
	copy(work, src)

	// Transform rows (inverse)
	for row := range p.rows {
		rowData := work[row*p.cols : (row+1)*p.cols]

		err := p.rowPlan.InverseInPlace(rowData)
		if err != nil {
			return err
		}
	}

	// Transform columns (inverse)
	if p.rows == p.cols {
		p.transformColumnsViaTranspose(work, false)
	} else {
		p.transformColumnsStrided(work, false)
	}

	copy(dst, work)

	return nil
}
