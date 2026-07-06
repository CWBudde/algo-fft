package algofft

import (
	"fmt"

	mem "github.com/cwbudde/algo-fft/internal/memory"
)

// PlanReal3D is a pre-computed 3D real FFT plan for float32 input volumes.
// The forward transform exploits conjugate symmetry by computing only the
// non-redundant half of the spectrum along the last dimension.
// Plans are reusable and safe for concurrent use during transforms: scratch
// buffers are borrowed per call from an internal cache, so multiple
// goroutines may share one instance.
//
// The 3D real FFT uses the dimension-by-dimension decomposition algorithm:
// - Forward: Real FFT along width (innermost), then complex FFT along height and depth
// - Inverse: Complex IFFT along depth and height, then real IFFT along width
//
// Data layout:
// - Input (real): row-major D×H×W float32 array
// - Compact output: row-major D×H×(W/2+1) complex64 array
// - Full output: row-major D×H×W complex64 array (with redundant conjugate pairs).
type PlanReal3D struct {
	depth, height, width int                // Input dimensions (D×H×W real values)
	halfWidth            int                // W/2+1 (compact spectrum width)
	widthPlan            *PlanReal          // Real FFT for width (size W → W/2+1)
	heightPlans          []*Plan[complex64] // Complex FFT for height (one per width column)
	depthPlans           []*Plan[complex64] // Complex FFT for depth (one per height×width position)

	// scratch hands out per-call working buffers for thread-safety.
	scratch *residentCache[planReal3DScratch]
}

// planReal3DScratch is one per-call scratch set for PlanReal3D transforms.
type planReal3DScratch struct {
	compact        []complex64 // Working buffer (D×H×(W/2+1))
	compactBacking []byte      // Keeps the aligned buffer alive for GC
}

func newPlanReal3DScratchCache(depth, height, halfWidth int) *residentCache[planReal3DScratch] {
	return newResidentCache(func() *planReal3DScratch {
		compact, backing := mem.AllocAlignedComplex64(depth * height * halfWidth)

		return &planReal3DScratch{compact: compact, compactBacking: backing}
	})
}

// NewPlanReal3D creates a new 3D real FFT plan for a D×H×W real volume.
//
// All dimensions must be ≥ 2, and width must be even (required by the real FFT algorithm).
//
// The plan pre-allocates all necessary buffers, enabling zero-allocation transforms.
//
// A single plan instance may be shared by multiple goroutines.
func NewPlanReal3D(depth, height, width int) (*PlanReal3D, error) {
	if depth <= 0 || height <= 0 || width <= 0 {
		return nil, ErrInvalidLength
	}

	if width < 2 || width%2 != 0 {
		return nil, ErrInvalidLength // Real FFT requires even W
	}

	// Create 1D real plan for width
	widthPlan, err := NewPlanReal(width)
	if err != nil {
		return nil, err
	}

	halfWidth := width/2 + 1

	// Create complex plans for height (one for each column in compact spectrum)
	heightPlans := make([]*Plan[complex64], halfWidth)
	for i := range heightPlans {
		plan, err := NewPlanT[complex64](height)
		if err != nil {
			return nil, err
		}

		heightPlans[i] = plan
	}

	// Create complex plans for depth (one for each height×width position)
	depthPlans := make([]*Plan[complex64], height*halfWidth)
	for i := range depthPlans {
		plan, err := NewPlanT[complex64](depth)
		if err != nil {
			return nil, err
		}

		depthPlans[i] = plan
	}

	return &PlanReal3D{
		depth:       depth,
		height:      height,
		width:       width,
		halfWidth:   halfWidth,
		widthPlan:   widthPlan,
		heightPlans: heightPlans,
		depthPlans:  depthPlans,
		scratch:     newPlanReal3DScratchCache(depth, height, halfWidth),
	}, nil
}

// Depth returns the depth dimension of the input volume.
func (p *PlanReal3D) Depth() int {
	return p.depth
}

// Height returns the height dimension of the input volume.
func (p *PlanReal3D) Height() int {
	return p.height
}

// Width returns the width dimension of the input volume.
func (p *PlanReal3D) Width() int {
	return p.width
}

// Len returns the total number of real input elements (depth × height × width).
func (p *PlanReal3D) Len() int {
	return p.depth * p.height * p.width
}

// SpectrumLen returns the total number of complex values in compact output.
func (p *PlanReal3D) SpectrumLen() int {
	return p.depth * p.height * p.halfWidth
}

// String returns a human-readable description of the PlanReal3D for debugging.
func (p *PlanReal3D) String() string {
	return fmt.Sprintf("PlanReal3D[float32→complex64](%dx%dx%d → %dx%dx%d)",
		p.depth, p.height, p.width, p.depth, p.height, p.halfWidth)
}

// Forward computes the 3D real FFT in compact format (memory-efficient).
//
// Input src: D×H×W row-major array of float32 (length D*H*W)
// Output dst: D×H×(W/2+1) row-major array of complex64 (length D*H*(W/2+1))
//
// The output exploits conjugate symmetry: only the non-redundant half-spectrum is stored.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
//
//nolint:gocognit
func (p *PlanReal3D) Forward(dst []complex64, src []float32) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	expectedSrcLen := p.depth * p.height * p.width
	expectedDstLen := p.depth * p.height * p.halfWidth

	if len(src) != expectedSrcLen {
		return ErrLengthMismatch
	}

	if len(dst) != expectedDstLen {
		return ErrLengthMismatch
	}

	s := p.scratch.get()
	defer p.scratch.put(s)

	compact := s.compact

	// Step 1: Real FFT along width (innermost dimension)
	for d := range p.depth {
		for h := range p.height {
			srcOffset := d*p.height*p.width + h*p.width
			dstOffset := d*p.height*p.halfWidth + h*p.halfWidth

			srcRow := src[srcOffset : srcOffset+p.width]
			dstRow := compact[dstOffset : dstOffset+p.halfWidth]

			err := p.widthPlan.Forward(dstRow, srcRow)
			if err != nil {
				return err
			}
		}
	}

	// Step 2: Complex FFT along height (middle dimension)
	heightData := make([]complex64, p.height)

	for d := range p.depth {
		for w := range p.halfWidth {
			// Extract column along height
			for h := range p.height {
				heightData[h] = compact[d*p.height*p.halfWidth+h*p.halfWidth+w]
			}

			// Transform column
			err := p.heightPlans[w].InPlace(heightData)
			if err != nil {
				return err
			}

			// Write back
			for h := range p.height {
				compact[d*p.height*p.halfWidth+h*p.halfWidth+w] = heightData[h]
			}
		}
	}

	// Step 3: Complex FFT along depth (outermost dimension)
	depthData := make([]complex64, p.depth)

	for h := range p.height {
		for w := range p.halfWidth {
			// Extract slice along depth
			for d := range p.depth {
				depthData[d] = compact[d*p.height*p.halfWidth+h*p.halfWidth+w]
			}

			// Transform depth slice
			planIdx := h*p.halfWidth + w

			err := p.depthPlans[planIdx].InPlace(depthData)
			if err != nil {
				return err
			}

			// Write back
			for d := range p.depth {
				compact[d*p.height*p.halfWidth+h*p.halfWidth+w] = depthData[d]
			}
		}
	}

	// Copy result to dst
	copy(dst, compact)

	return nil
}

// ForwardFull computes the 3D real FFT with full spectrum output (includes redundant conjugates).
//
// Input src: D×H×W row-major array of float32 (length D*H*W)
// Output dst: D×H×W row-major array of complex64 (length D*H*W)
//
// The output is the complete spectrum with conjugate symmetry explicitly filled in.
// This is easier to work with but uses 2x memory compared to Forward().
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal3D) ForwardFull(dst []complex64, src []float32) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	expectedSrcLen := p.depth * p.height * p.width
	expectedDstLen := p.depth * p.height * p.width

	if len(src) != expectedSrcLen {
		return ErrLengthMismatch
	}

	if len(dst) != expectedDstLen {
		return ErrLengthMismatch
	}

	s := p.scratch.get()
	defer p.scratch.put(s)

	compact := s.compact

	// First compute compact spectrum
	err := p.Forward(compact, src)
	if err != nil {
		return err
	}

	// Expand to full spectrum using conjugate symmetry
	// For 3D real FFT: X[kd, kh, w-kw] = conj(X[kd, kh, kw]) for kw = 1..w/2-1
	for d := range p.depth {
		for h := range p.height {
			// Copy half-spectrum to output
			for w := range p.halfWidth {
				dst[d*p.height*p.width+h*p.width+w] = compact[d*p.height*p.halfWidth+h*p.halfWidth+w]
			}

			// Fill conjugate pairs for w > W/2
			for w := p.halfWidth; w < p.width; w++ {
				mirrorW := p.width - w
				// For 3D, need to mirror all dimensions for conjugate symmetry
				mirrorD := (p.depth - d) % p.depth
				mirrorH := (p.height - h) % p.height
				val := dst[mirrorD*p.height*p.width+mirrorH*p.width+mirrorW]
				dst[d*p.height*p.width+h*p.width+w] = complex(real(val), -imag(val))
			}
		}
	}

	return nil
}

// Inverse computes the 3D real IFFT from compact half-spectrum.
//
// Input src: D×H×(W/2+1) row-major array of complex64
// Output dst: D×H×W row-major array of float32
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
//
//nolint:gocognit
func (p *PlanReal3D) Inverse(dst []float32, src []complex64) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	expectedSrcLen := p.depth * p.height * p.halfWidth
	expectedDstLen := p.depth * p.height * p.width

	if len(src) != expectedSrcLen {
		return ErrLengthMismatch
	}

	if len(dst) != expectedDstLen {
		return ErrLengthMismatch
	}

	s := p.scratch.get()
	defer p.scratch.put(s)

	compact := s.compact

	// Copy src to scratch
	copy(compact, src)

	// Step 1: Complex IFFT along depth (outermost dimension)
	depthData := make([]complex64, p.depth)

	for h := range p.height {
		for w := range p.halfWidth {
			// Extract slice along depth
			for d := range p.depth {
				depthData[d] = compact[d*p.height*p.halfWidth+h*p.halfWidth+w]
			}

			// Inverse transform depth slice
			planIdx := h*p.halfWidth + w

			err := p.depthPlans[planIdx].InverseInPlace(depthData)
			if err != nil {
				return err
			}

			// Write back
			for d := range p.depth {
				compact[d*p.height*p.halfWidth+h*p.halfWidth+w] = depthData[d]
			}
		}
	}

	// Step 2: Complex IFFT along height (middle dimension)
	heightData := make([]complex64, p.height)

	for d := range p.depth {
		for w := range p.halfWidth {
			// Extract column along height
			for h := range p.height {
				heightData[h] = compact[d*p.height*p.halfWidth+h*p.halfWidth+w]
			}

			// Inverse transform column
			err := p.heightPlans[w].InverseInPlace(heightData)
			if err != nil {
				return err
			}

			// Write back
			for h := range p.height {
				compact[d*p.height*p.halfWidth+h*p.halfWidth+w] = heightData[h]
			}
		}
	}

	// Step 3: Real IFFT along width (innermost dimension)
	for d := range p.depth {
		for h := range p.height {
			srcOffset := d*p.height*p.halfWidth + h*p.halfWidth
			dstOffset := d*p.height*p.width + h*p.width

			srcRow := compact[srcOffset : srcOffset+p.halfWidth]
			dstRow := dst[dstOffset : dstOffset+p.width]

			err := p.widthPlan.Inverse(dstRow, srcRow)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// InverseFull computes the 3D real IFFT from full spectrum.
//
// Input src: D×H×W row-major array of complex64
// Output dst: D×H×W row-major array of float32
//
// The input should have conjugate symmetry (as produced by ForwardFull).
// Only the non-redundant half is used; the rest is ignored.
//
// Returns ErrNilSlice if dst or src is nil.
// Returns ErrLengthMismatch if slice lengths don't match plan dimensions.
func (p *PlanReal3D) InverseFull(dst []float32, src []complex64) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	expectedSrcLen := p.depth * p.height * p.width
	expectedDstLen := p.depth * p.height * p.width

	if len(src) != expectedSrcLen {
		return ErrLengthMismatch
	}

	if len(dst) != expectedDstLen {
		return ErrLengthMismatch
	}

	s := p.scratch.get()
	defer p.scratch.put(s)

	compact := s.compact

	// Extract compact half-spectrum from full spectrum
	for d := range p.depth {
		for h := range p.height {
			for w := range p.halfWidth {
				compact[d*p.height*p.halfWidth+h*p.halfWidth+w] = src[d*p.height*p.width+h*p.width+w]
			}
		}
	}

	// Use compact inverse
	return p.Inverse(dst, compact)
}

// Clone creates an independent copy of the PlanReal3D.
//
// A single PlanReal3D is already safe for concurrent transforms, so cloning
// is not required for concurrency; it remains available for callers that want
// isolated scratch caches. The clone shares the concurrency-safe child plans
// but has its own scratch cache.
func (p *PlanReal3D) Clone() *PlanReal3D {
	return &PlanReal3D{
		depth:       p.depth,
		height:      p.height,
		width:       p.width,
		halfWidth:   p.halfWidth,
		widthPlan:   p.widthPlan,
		heightPlans: p.heightPlans,
		depthPlans:  p.depthPlans,
		scratch:     newPlanReal3DScratchCache(p.depth, p.height, p.halfWidth),
	}
}
