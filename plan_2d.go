package algofft

import (
	"fmt"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
)

// Plan2D is a pre-computed 2D FFT plan for a specific matrix size and precision.
// Plans are reusable and safe for concurrent use during transforms (but not
// during creation): scratch buffers are borrowed per call from an internal
// cache, so multiple goroutines may share one instance.
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
	options    PlanOptions

	// scratch hands out per-call working buffers for thread-safety.
	scratch *residentCache[plan2DScratch[T]]

	// Transpose support for square matrices
	transposePairs []fft.TransposePair
}

// plan2DScratch is one per-call scratch set for Plan2D transforms.
type plan2DScratch[T Complex] struct {
	work        []T    // Working buffer (size=rows*cols)
	workBacking []byte // Keeps the aligned working buffer alive for GC
	col         []T    // Column buffer for strided transforms (size=rows)
}

func newPlan2DScratchCache[T Complex](rows, cols int) *residentCache[plan2DScratch[T]] {
	return newResidentCache(func() *plan2DScratch[T] {
		work, backing := allocAlignedSlice[T](rows * cols)

		return &plan2DScratch[T]{
			work:        work,
			workBacking: backing,
			col:         make([]T, rows),
		}
	})
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

	p := &Plan2D[T]{
		rows:    rows,
		cols:    cols,
		rowPlan: rowPlan,
		colPlan: colPlan,
		scratch: newPlan2DScratchCache[T](rows, cols),
		options: opts,
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

	typeName := precisionNameComplex64
	if _, ok := any(zero).(complex128); ok {
		typeName = precisionNameComplex128
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

// Clone creates an independent copy of the Plan2D.
//
// A single Plan2D is already safe for concurrent transforms, so cloning is
// not required for concurrency; it remains available for callers that want
// isolated scratch caches. The clone shares immutable data (transpose pairs)
// and the concurrency-safe 1D child plans, but has its own scratch cache.
func (p *Plan2D[T]) Clone() *Plan2D[T] {
	return &Plan2D[T]{
		rows:           p.rows,
		cols:           p.cols,
		rowPlan:        p.rowPlan,
		colPlan:        p.colPlan,
		scratch:        newPlan2DScratchCache[T](p.rows, p.cols),
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
func (p *Plan2D[T]) transformColumnsStrided(data, colData []T, forward bool) {
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

	s := p.scratch.get()
	defer p.scratch.put(s)

	work := s.work
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
		p.transformColumnsStrided(work, s.col, true)
	}

	copy(dst, work)

	return nil
}

func (p *Plan2D[T]) inverseSingle(dst, src []T) error {
	err := p.validate(dst, src)
	if err != nil {
		return err
	}

	s := p.scratch.get()
	defer p.scratch.put(s)

	work := s.work
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
		p.transformColumnsStrided(work, s.col, false)
	}

	copy(dst, work)

	return nil
}
