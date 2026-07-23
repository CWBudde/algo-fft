package algofft

import (
	"fmt"

	"github.com/cwbudde/algo-fft/internal/cpu"
	m "github.com/cwbudde/algo-fft/internal/math"
	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// PlanReal2D is a pre-computed 2D real FFT plan for real input matrices.
// The forward transform exploits conjugate symmetry by computing only the
// non-redundant half of the spectrum along the last dimension.
// Plans are reusable and safe for concurrent use during transforms: scratch
// buffers are borrowed per call from an internal cache, so multiple
// goroutines may share one instance.
//
// Type parameters:
//   - F: float type (float32 or float64)
//   - C: complex type (complex64 or complex128), must match F
//
// The 2D real FFT uses the row-column decomposition algorithm:
// - Forward: Real FFT on rows (produces M×(N/2+1) complex), then complex FFT on columns
// - Inverse: Complex IFFT on columns, then real IFFT on rows
//
// Data layout:
// - Input (real): row-major M×N float array
// - Compact output: row-major M×(N/2+1) complex array
// - Full output: row-major M×N complex array (with redundant conjugate pairs).
type PlanReal2D[F Float, C Complex] struct {
	rows, cols int             // Input dimensions (M×N real values)
	halfCols   int             // N/2+1 (compact spectrum width)
	rowPlan    *PlanReal[F, C] // Real FFT for rows (size N → N/2+1)
	colPlans   []*Plan[C]      // Complex FFT for each column (size M)

	// scratch hands out per-call working buffers for thread-safety.
	scratch *residentCache[planReal2DScratch[C]]
}

// planReal2DScratch is one per-call scratch set for PlanReal2D transforms.
type planReal2DScratch[C Complex] struct {
	compact        []C    // Working buffer (M×(N/2+1))
	compactBacking []byte // Keeps the aligned buffer alive for GC
	colData        []C    // Column working buffer (length rows)
}

func newPlanReal2DScratchCache[C Complex](rows, halfCols int) *residentCache[planReal2DScratch[C]] {
	return newResidentCache(func() *planReal2DScratch[C] {
		compact, backing := mem.AllocAligned[C](rows * halfCols)

		return &planReal2DScratch[C]{
			compact:        compact,
			compactBacking: backing,
			colData:        make([]C, rows),
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
//
// Example:
//
//	plan32, err := algofft.NewPlanReal2D[float32, complex64](480, 640)
//	plan64, err := algofft.NewPlanReal2D[float64, complex128](480, 640)
func NewPlanReal2D[F Float, C Complex](rows, cols int) (*PlanReal2D[F, C], error) {
	return NewPlanReal2DWithOptions[F, C](rows, cols, PlanOptions{})
}

// NewPlanReal2DWithOptions creates a new 2D real FFT plan with explicit planner options.
func NewPlanReal2DWithOptions[F Float, C Complex](rows, cols int, opts PlanOptions) (*PlanReal2D[F, C], error) {
	if rows <= 0 {
		return nil, fmt.Errorf("rows has invalid size %d: %w", rows, ErrInvalidLength)
	}

	if cols < 2 || cols%2 != 0 {
		// Real FFT requires even N
		return nil, fmt.Errorf("cols has invalid size %d (must be even and >= 2): %w", cols, ErrInvalidLength)
	}

	opts = normalizePlanOptions(opts)
	features := cpu.DetectFeatures()

	// Create 1D real plan for rows
	rowPlan, err := newPlanRealWithFeatures[F, C](cols, features, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to create row-transform plan (size %d): %w", cols, err)
	}

	halfCols := cols/2 + 1

	// Create complex plans for columns (one for each column in compact spectrum)
	colPlans := make([]*Plan[C], halfCols)
	for i := range colPlans {
		plan, err := newPlanWithFeatures[C](rows, features, opts)
		if err != nil {
			return nil, fmt.Errorf("failed to create column-transform plan (size %d): %w", rows, err)
		}

		colPlans[i] = plan
	}

	return &PlanReal2D[F, C]{
		rows:     rows,
		cols:     cols,
		halfCols: halfCols,
		rowPlan:  rowPlan,
		colPlans: colPlans,
		scratch:  newPlanReal2DScratchCache[C](rows, halfCols),
	}, nil
}

// NewPlanReal2D32 creates a new single-precision 2D real FFT plan.
// This is one-line sugar for NewPlanReal2D[float32, complex64](rows, cols).
func NewPlanReal2D32(rows, cols int) (*PlanReal2D[float32, complex64], error) {
	return NewPlanReal2D[float32, complex64](rows, cols)
}

// NewPlanReal2D64 creates a new double-precision 2D real FFT plan.
// This is one-line sugar for NewPlanReal2D[float64, complex128](rows, cols).
func NewPlanReal2D64(rows, cols int) (*PlanReal2D[float64, complex128], error) {
	return NewPlanReal2D[float64, complex128](rows, cols)
}

// Rows returns the number of rows in the input matrix.
func (p *PlanReal2D[F, C]) Rows() int {
	return p.rows
}

// Cols returns the number of columns in the input matrix.
func (p *PlanReal2D[F, C]) Cols() int {
	return p.cols
}

// Len returns the total number of real input elements (rows × cols).
func (p *PlanReal2D[F, C]) Len() int {
	return p.rows * p.cols
}

// SpectrumLen returns the total number of complex values in compact output (rows × (cols/2+1)).
func (p *PlanReal2D[F, C]) SpectrumLen() int {
	return p.rows * p.halfCols
}

// String returns a human-readable description of the PlanReal2D for debugging.
func (p *PlanReal2D[F, C]) String() string {
	return fmt.Sprintf("PlanReal2D[%s](%dx%d → %dx%d)", realPlanTypeNames[C](), p.rows, p.cols, p.rows, p.halfCols)
}

// Forward computes the 2D real FFT in compact format (memory-efficient).
//
// Input src: M×N row-major array of floats (length M*N)
// Output dst: M×(N/2+1) row-major array of complex values (length M*(N/2+1))
//
// The output exploits conjugate symmetry: only the non-redundant half-spectrum is stored.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal2D[F, C]) Forward(dst []C, src []F) error {
	err := validateDstSrc(dst, src, p.rows*p.halfCols, p.rows*p.cols)
	if err != nil {
		return err
	}

	s := p.scratch.get()
	defer p.scratch.put(s)

	err = p.forwardCompactInto(s.compact, s.colData, src)
	if err != nil {
		return err
	}

	// Copy result to dst
	copy(dst, s.compact)

	return nil
}

// ForwardFull computes the 2D real FFT with full spectrum output (includes redundant conjugates).
//
// Input src: M×N row-major array of floats (length M*N)
// Output dst: M×N row-major array of complex values (length M*N)
//
// The output is the complete spectrum with conjugate symmetry explicitly filled in.
// This is easier to work with but uses 2x memory compared to Forward().
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal2D[F, C]) ForwardFull(dst []C, src []F) error {
	err := validateDstSrc(dst, src, p.rows*p.cols, p.rows*p.cols)
	if err != nil {
		return err
	}

	s := p.scratch.get()
	defer p.scratch.put(s)

	compact := s.compact

	// Compute the compact spectrum directly into the borrowed buffer. Calling
	// the exported Forward here would nest a second scratch borrow and break
	// the zero-allocation guarantee once GC drains the overflow pool.
	err = p.forwardCompactInto(compact, s.colData, src)
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
			dst[row*p.cols+col] = m.ConjugateOf(val)
		}
	}

	return nil
}

// Inverse computes the 2D real IFFT from compact half-spectrum.
//
// Input src: M×(N/2+1) row-major array of complex values
// Output dst: M×N row-major array of floats
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal2D[F, C]) Inverse(dst []F, src []C) error {
	err := validateDstSrc(dst, src, p.rows*p.cols, p.rows*p.halfCols)
	if err != nil {
		return err
	}

	s := p.scratch.get()
	defer p.scratch.put(s)

	compact := s.compact

	// Copy src to scratch
	copy(compact, src)

	// Step 1: Complex IFFT on each column
	colData := s.colData

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

	// Step 2: Real IFFT on each row (complex half-spectrum → float)
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
// Input src: M×N row-major array of complex values
// Output dst: M×N row-major array of floats
//
// The input should have conjugate symmetry (as produced by ForwardFull).
// Only the non-redundant half is used; the rest is ignored.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal2D[F, C]) InverseFull(dst []F, src []C) error {
	err := validateDstSrc(dst, src, p.rows*p.cols, p.rows*p.cols)
	if err != nil {
		return err
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
func (p *PlanReal2D[F, C]) Clone() *PlanReal2D[F, C] {
	return &PlanReal2D[F, C]{
		rows:     p.rows,
		cols:     p.cols,
		halfCols: p.halfCols,
		rowPlan:  p.rowPlan,
		colPlans: p.colPlans,
		scratch:  newPlanReal2DScratchCache[C](p.rows, p.halfCols),
	}
}

// forwardCompactInto computes the compact half-spectrum of src into compact
// (length rows*halfCols), using colData (length rows) as the column working
// buffer. It borrows no scratch of its own, so callers that already hold a
// scratch set reuse its buffers and stay allocation-free — this lets
// ForwardFull run without nesting a second scratch borrow through Forward.
func (p *PlanReal2D[F, C]) forwardCompactInto(compact, colData []C, src []F) error {
	// Step 1: Real FFT on each row (float input → complex half-spectrum)
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
		err := p.colPlans[col].ForwardInPlace(colData)
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
