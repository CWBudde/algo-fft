package algofft

import (
	"fmt"

	"github.com/cwbudde/algo-fft/internal/cpu"
	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// PlanReal2D is a pre-computed 2D real FFT plan for float32 input matrices.
// The forward transform exploits conjugate symmetry by computing only the
// non-redundant half of the spectrum along the last dimension.
// Plans are reusable and safe for concurrent use during transforms: scratch
// buffers are borrowed per call from an internal cache, so multiple
// goroutines may share one instance.
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
	rows, cols int                // Input dimensions (M×N real values)
	halfCols   int                // N/2+1 (compact spectrum width)
	rowPlan    *PlanReal          // Real FFT for rows (size N → N/2+1)
	colPlans   []*Plan[complex64] // Complex FFT for each column (size M)
	options    PlanOptions

	// scratch hands out per-call working buffers for thread-safety.
	scratch *residentCache[planReal2DScratch]
}

// planReal2DScratch is one per-call scratch set for PlanReal2D transforms.
type planReal2DScratch struct {
	compact        []complex64 // Working buffer (M×(N/2+1))
	compactBacking []byte      // Keeps the aligned buffer alive for GC
	colData        []complex64 // Column working buffer (length rows)
}

func newPlanReal2DScratchCache(rows, halfCols int) *residentCache[planReal2DScratch] {
	return newResidentCache(func() *planReal2DScratch {
		compact, backing := mem.AllocAlignedComplex64(rows * halfCols)

		return &planReal2DScratch{
			compact:        compact,
			compactBacking: backing,
			colData:        make([]complex64, rows),
		}
	})
}

// NewPlanReal2D creates a new 2D real FFT plan for an M×N real matrix.
//
// Both rows and cols must be ≥ 2, and cols must be even (required by the real FFT algorithm).
//
// The plan pre-allocates all necessary buffers, enabling zero-allocation transforms.
//
// A single plan instance may be shared by multiple goroutines.
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

	return &PlanReal2D{
		rows:     rows,
		cols:     cols,
		halfCols: halfCols,
		rowPlan:  rowPlan,
		colPlans: colPlans,
		scratch:  newPlanReal2DScratchCache(rows, halfCols),
		options:  opts,
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

	s := p.scratch.get()
	defer p.scratch.put(s)

	compact := s.compact

	// Compute the compact spectrum directly into the borrowed buffer. Calling
	// the exported Forward here would nest a second scratch borrow and break
	// the zero-allocation guarantee once GC drains the overflow pool.
	err := p.forwardCompactInto(compact, s.colData, src)
	if err != nil {
		return err
	}

	// Expand to full spectrum using conjugate symmetry
	// For 2D real FFT: X[k, n-l] = conj(X[k, l]) for l = 1..n/2-1
	for row := range p.rows {
		// Copy half-spectrum to output
		for col := range p.halfCols {
			dst[row*p.cols+col] = compact[row*p.halfCols+col]
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

	s := p.scratch.get()
	defer p.scratch.put(s)

	compact := s.compact

	// Copy src to scratch
	copy(compact, src)

	// Step 1: Complex IFFT on each column
	colData := make([]complex64, p.rows)

	for col := range p.halfCols {
		// Extract column
		for row := range p.rows {
			colData[row] = compact[row*p.halfCols+col]
		}

		// Inverse transform column
		err := p.colPlans[col].InverseInPlace(colData)
		if err != nil {
			return err
		}

		// Write back
		for row := range p.rows {
			compact[row*p.halfCols+col] = colData[row]
		}
	}

	// Step 2: Real IFFT on each row (complex64 half-spectrum → float32)
	for row := range p.rows {
		srcRow := compact[row*p.halfCols : (row+1)*p.halfCols]
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

	s := p.scratch.get()
	defer p.scratch.put(s)

	compact := s.compact

	// Extract compact half-spectrum from full spectrum
	for row := range p.rows {
		for col := range p.halfCols {
			compact[row*p.halfCols+col] = src[row*p.cols+col]
		}
	}

	// Use compact inverse
	return p.Inverse(dst, compact)
}

// Clone creates an independent copy of the PlanReal2D.
//
// A single PlanReal2D is already safe for concurrent transforms, so cloning
// is not required for concurrency; it remains available for callers that want
// isolated scratch caches. The clone shares the concurrency-safe child plans
// but has its own scratch cache.
func (p *PlanReal2D) Clone() *PlanReal2D {
	return &PlanReal2D{
		rows:     p.rows,
		cols:     p.cols,
		halfCols: p.halfCols,
		rowPlan:  p.rowPlan,
		colPlans: p.colPlans,
		scratch:  newPlanReal2DScratchCache(p.rows, p.halfCols),
		options:  p.options,
	}
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

	s := p.scratch.get()
	defer p.scratch.put(s)

	err := p.forwardCompactInto(s.compact, s.colData, src)
	if err != nil {
		return err
	}

	// Copy result to dst
	copy(dst, s.compact)

	return nil
}

// forwardCompactInto computes the compact half-spectrum of src into compact
// (length rows*halfCols), using colData (length rows) as the column working
// buffer. It borrows no scratch of its own, so callers that already hold a
// scratch set reuse its buffers and stay allocation-free — this lets
// ForwardFull run without nesting a second scratch borrow through Forward.
func (p *PlanReal2D) forwardCompactInto(compact, colData []complex64, src []float32) error {
	// Step 1: Real FFT on each row (float32 input → complex64 half-spectrum)
	for row := range p.rows {
		srcRow := src[row*p.cols : (row+1)*p.cols]
		dstRow := compact[row*p.halfCols : (row+1)*p.halfCols]

		err := p.rowPlan.Forward(dstRow, srcRow)
		if err != nil {
			return err
		}
	}

	// Step 2: Complex FFT on each column of the half-spectrum
	for col := range p.halfCols {
		// Extract column
		for row := range p.rows {
			colData[row] = compact[row*p.halfCols+col]
		}

		// Transform column
		err := p.colPlans[col].InPlace(colData)
		if err != nil {
			return err
		}

		// Write back
		for row := range p.rows {
			compact[row*p.halfCols+col] = colData[row]
		}
	}

	return nil
}
