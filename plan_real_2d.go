package algofft

import (
	"fmt"

	"github.com/cwbudde/algo-fft/internal/cpu"
	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// PlanReal2D is a pre-computed 2D real FFT plan for float32 input matrices.
// The forward transform exploits conjugate symmetry by computing only the
// non-redundant half of the spectrum along the last dimension.
//
// The 2D real FFT uses the row-column decomposition algorithm:
// - Forward: Real FFT on rows (produces M×(N/2+1) complex), then complex FFT on columns
// - Inverse: Complex IFFT on columns, then real IFFT on rows
//
// Data layout:
// - Input (real): row-major M×N float32 array
// - Compact output: row-major M×(N/2+1) complex64 array
// - Full output: row-major M×N complex64 array (with redundant conjugate pairs).
type PlanReal2D struct {
	rows, cols     int                // Input dimensions (M×N real values)
	halfCols       int                // N/2+1 (compact spectrum width)
	rowPlan        *PlanReal          // Real FFT for rows (size N → N/2+1)
	colPlans       []*Plan[complex64] // Complex FFT for each column (size M)
	scratchCompact []complex64        // Working buffer (M×(N/2+1))
	scratchFull    []complex64        // Full spectrum buffer (M×N) for ForwardFull
	options        PlanOptions

	// backing keeps aligned buffers alive for GC
	scratchCompactBacking []byte
	scratchFullBacking    []byte
}

// NewPlanReal2D creates a new 2D real FFT plan for an M×N real matrix.
//
// Both rows and cols must be ≥ 2, and cols must be even (required by the real FFT algorithm).
//
// The plan pre-allocates all necessary buffers, enabling zero-allocation transforms.
//
// For concurrent use, create separate plans via Clone() for each goroutine.
func NewPlanReal2D(rows, cols int) (*PlanReal2D, error) {
	return NewPlanReal2DWithOptions(rows, cols, PlanOptions{})
}

// NewPlanReal2DWithOptions creates a new 2D real FFT plan with explicit planner options.
func NewPlanReal2DWithOptions(rows, cols int, opts PlanOptions) (*PlanReal2D, error) {
	if rows <= 0 || cols <= 0 {
		return nil, ErrInvalidLength
	}

	if cols < 2 || cols%2 != 0 {
		return nil, ErrInvalidLength // Real FFT requires even N
	}

	opts = normalizePlanOptions(opts)
	features := cpu.DetectFeatures()

	childOpts := opts
	childOpts.Batch = 0
	childOpts.Stride = 0
	childOpts.InPlace = false

	// Create 1D real plan for rows
	rowPlan, err := newPlanRealWithFeatures(cols, features, childOpts)
	if err != nil {
		return nil, err
	}

	halfCols := cols/2 + 1

	// Create complex plans for columns (one for each column in compact spectrum)
	colPlans := make([]*Plan[complex64], halfCols)
	for i := range colPlans {
		plan, err := newPlanWithFeatures[complex64](rows, features, childOpts)
		if err != nil {
			return nil, err
		}

		colPlans[i] = plan
	}

	// Allocate scratch buffers (aligned for SIMD)
	compactSize := rows * halfCols
	fullSize := rows * cols

	scratchCompact, scratchCompactBacking := mem.AllocAlignedComplex64(compactSize)
	scratchFull, scratchFullBacking := mem.AllocAlignedComplex64(fullSize)

	return &PlanReal2D{
		rows:                  rows,
		cols:                  cols,
		halfCols:              halfCols,
		rowPlan:               rowPlan,
		colPlans:              colPlans,
		scratchCompact:        scratchCompact,
		scratchFull:           scratchFull,
		scratchCompactBacking: scratchCompactBacking,
		scratchFullBacking:    scratchFullBacking,
		options:               opts,
	}, nil
}

// Rows returns the number of rows in the input matrix.
func (p *PlanReal2D) Rows() int {
	return p.rows
}

// Cols returns the number of columns in the input matrix.
func (p *PlanReal2D) Cols() int {
	return p.cols
}

// Len returns the total number of real input elements (rows × cols).
func (p *PlanReal2D) Len() int {
	return p.rows * p.cols
}

// SpectrumLen returns the total number of complex values in compact output (rows × (cols/2+1)).
func (p *PlanReal2D) SpectrumLen() int {
	return p.rows * p.halfCols
}

// String returns a human-readable description of the PlanReal2D for debugging.
func (p *PlanReal2D) String() string {
	return fmt.Sprintf("PlanReal2D[float32→complex64](%dx%d → %dx%d)", p.rows, p.cols, p.rows, p.halfCols)
}

// Forward computes the 2D real FFT in compact format (memory-efficient).
//
// Input src: M×N row-major array of float32 (length M*N)
// Output dst: M×(N/2+1) row-major array of complex64 (length M*(N/2+1))
//
// The output exploits conjugate symmetry: only the non-redundant half-spectrum is stored.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal2D) Forward(dst []complex64, src []float32) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if p.options.Batch <= 1 && p.options.Stride <= 0 {
		return p.forwardSingle(dst, src)
	}

	batch, strideIn, strideOut, err := resolveBatchStrideReal(p.rows*p.cols, p.rows*p.halfCols, p.options)
	if err != nil {
		return err
	}

	for b := range batch {
		srcOff := b * strideIn

		dstOff := b * strideOut
		if srcOff+p.rows*p.cols > len(src) || dstOff+p.rows*p.halfCols > len(dst) {
			return ErrLengthMismatch
		}

		err = p.forwardSingle(dst[dstOff:dstOff+p.rows*p.halfCols], src[srcOff:srcOff+p.rows*p.cols])
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *PlanReal2D) forwardSingle(dst []complex64, src []float32) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(src) != p.rows*p.cols {
		return ErrLengthMismatch
	}

	if len(dst) != p.rows*p.halfCols {
		return ErrLengthMismatch
	}

	// Step 1: Real FFT on each row (float32 input → complex64 half-spectrum)
	for row := range p.rows {
		srcRow := src[row*p.cols : (row+1)*p.cols]
		dstRow := p.scratchCompact[row*p.halfCols : (row+1)*p.halfCols]

		err := p.rowPlan.Forward(dstRow, srcRow)
		if err != nil {
			return err
		}
	}

	// Step 2: Complex FFT on each column of the half-spectrum
	colData := make([]complex64, p.rows)

	for col := range p.halfCols {
		// Extract column
		for row := range p.rows {
			colData[row] = p.scratchCompact[row*p.halfCols+col]
		}

		// Transform column
		err := p.colPlans[col].InPlace(colData)
		if err != nil {
			return err
		}

		// Write back
		for row := range p.rows {
			p.scratchCompact[row*p.halfCols+col] = colData[row]
		}
	}

	// Copy result to dst
	copy(dst, p.scratchCompact)

	return nil
}

// ForwardFull computes the 2D real FFT with full spectrum output (includes redundant conjugates).
//
// Input src: M×N row-major array of float32 (length M*N)
// Output dst: M×N row-major array of complex64 (length M*N)
//
// The output is the complete spectrum with conjugate symmetry explicitly filled in.
// This is easier to work with but uses 2x memory compared to Forward().
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal2D) ForwardFull(dst []complex64, src []float32) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(src) != p.rows*p.cols {
		return ErrLengthMismatch
	}

	if len(dst) != p.rows*p.cols {
		return ErrLengthMismatch
	}

	// First compute compact spectrum
	err := p.Forward(p.scratchCompact, src)
	if err != nil {
		return err
	}

	// Expand to full spectrum using conjugate symmetry
	// For 2D real FFT: X[k, n-l] = conj(X[k, l]) for l = 1..n/2-1
	for row := range p.rows {
		// Copy half-spectrum to output
		for col := range p.halfCols {
			dst[row*p.cols+col] = p.scratchCompact[row*p.halfCols+col]
		}

		// Fill conjugate pairs for col > N/2
		for col := p.halfCols; col < p.cols; col++ {
			mirrorCol := p.cols - col
			// Need to conjugate and mirror row as well for 2D
			mirrorRow := (p.rows - row) % p.rows
			val := dst[mirrorRow*p.cols+mirrorCol]
			dst[row*p.cols+col] = complex(real(val), -imag(val))
		}
	}

	return nil
}

// Inverse computes the 2D real IFFT from compact half-spectrum.
//
// Input src: M×(N/2+1) row-major array of complex64
// Output dst: M×N row-major array of float32
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal2D) Inverse(dst []float32, src []complex64) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if p.options.Batch <= 1 && p.options.Stride <= 0 {
		return p.inverseSingle(dst, src)
	}

	batch, strideIn, strideOut, err := resolveBatchStrideReal(p.rows*p.cols, p.rows*p.halfCols, p.options)
	if err != nil {
		return err
	}

	for b := range batch {
		dstOff := b * strideIn

		srcOff := b * strideOut
		if dstOff+p.rows*p.cols > len(dst) || srcOff+p.rows*p.halfCols > len(src) {
			return ErrLengthMismatch
		}

		err = p.inverseSingle(dst[dstOff:dstOff+p.rows*p.cols], src[srcOff:srcOff+p.rows*p.halfCols])
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *PlanReal2D) inverseSingle(dst []float32, src []complex64) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(src) != p.rows*p.halfCols {
		return ErrLengthMismatch
	}

	if len(dst) != p.rows*p.cols {
		return ErrLengthMismatch
	}

	// Copy src to scratch
	copy(p.scratchCompact, src)

	// Step 1: Complex IFFT on each column
	colData := make([]complex64, p.rows)

	for col := range p.halfCols {
		// Extract column
		for row := range p.rows {
			colData[row] = p.scratchCompact[row*p.halfCols+col]
		}

		// Inverse transform column
		err := p.colPlans[col].InverseInPlace(colData)
		if err != nil {
			return err
		}

		// Write back
		for row := range p.rows {
			p.scratchCompact[row*p.halfCols+col] = colData[row]
		}
	}

	// Step 2: Real IFFT on each row (complex64 half-spectrum → float32)
	for row := range p.rows {
		srcRow := p.scratchCompact[row*p.halfCols : (row+1)*p.halfCols]
		dstRow := dst[row*p.cols : (row+1)*p.cols]

		err := p.rowPlan.Inverse(dstRow, srcRow)
		if err != nil {
			return err
		}
	}

	return nil
}

// InverseFull computes the 2D real IFFT from full spectrum.
//
// Input src: M×N row-major array of complex64
// Output dst: M×N row-major array of float32
//
// The input should have conjugate symmetry (as produced by ForwardFull).
// Only the non-redundant half is used; the rest is ignored.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal2D) InverseFull(dst []float32, src []complex64) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(src) != p.rows*p.cols {
		return ErrLengthMismatch
	}

	if len(dst) != p.rows*p.cols {
		return ErrLengthMismatch
	}

	// Extract compact half-spectrum from full spectrum
	for row := range p.rows {
		for col := range p.halfCols {
			p.scratchCompact[row*p.halfCols+col] = src[row*p.cols+col]
		}
	}

	// Use compact inverse
	return p.Inverse(dst, p.scratchCompact)
}

// Clone creates an independent copy of the PlanReal2D for concurrent use.
//
// The clone shares immutable data but has its own:
// - Scratch buffers (for thread safety)
// - 1D plan instances (cloned from originals)
//
// This allows multiple goroutines to perform transforms concurrently.
func (p *PlanReal2D) Clone() *PlanReal2D {
	// Allocate new scratch buffers
	compactSize := p.rows * p.halfCols
	fullSize := p.rows * p.cols

	scratchCompact, scratchCompactBacking := mem.AllocAlignedComplex64(compactSize)
	scratchFull, scratchFullBacking := mem.AllocAlignedComplex64(fullSize)

	// Clone column plans
	colPlans := make([]*Plan[complex64], p.halfCols)
	for i := range colPlans {
		colPlans[i] = p.colPlans[i].Clone()
	}

	return &PlanReal2D{
		rows:                  p.rows,
		cols:                  p.cols,
		halfCols:              p.halfCols,
		rowPlan:               p.rowPlan, // PlanReal doesn't have Clone yet, share for now
		colPlans:              colPlans,
		scratchCompact:        scratchCompact,
		scratchFull:           scratchFull,
		scratchCompactBacking: scratchCompactBacking,
		scratchFullBacking:    scratchFullBacking,
	}
}
