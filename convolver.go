package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
	m "github.com/cwbudde/algo-fft/internal/math"
)

// Convolver computes linear convolutions of fixed-length inputs, reusing one
// FFT plan and preallocated buffers across calls. Use it instead of the
// one-shot Convolve function when convolving in a loop: after creation, the
// Convolver.Convolve method performs no planning and no allocations.
//
// A Convolver is safe for concurrent use: scratch buffers are borrowed per
// call from an internal cache, so multiple goroutines may share one instance.
type Convolver[T Complex] struct {
	lenA, lenB int
	convLen    int
	fftLen     int
	plan       *Plan[T]
	scratch    *residentCache[convolverScratch[T]]
}

// convolverScratch is one per-call scratch set for Convolver runs.
type convolverScratch[T Complex] struct {
	aPadded []T
	bPadded []T
	aFreq   []T
	bFreq   []T
	time    []T
}

// NewConvolver creates a convolver for inputs of length lenA and lenB.
// The output of each Convolve call has length lenA+lenB-1.
//
// It produces the same results as the one-shot Convolve function, which it
// replaces for repeated use.
func NewConvolver[T Complex](lenA, lenB int) (*Convolver[T], error) {
	if lenA < 1 || lenB < 1 {
		return nil, ErrInvalidLength
	}

	convLen := lenA + lenB - 1
	fftLen := fastConvolutionLength(convLen)

	plan, err := NewPlan[T](fftLen)
	if err != nil {
		return nil, err
	}

	return &Convolver[T]{
		lenA:    lenA,
		lenB:    lenB,
		convLen: convLen,
		fftLen:  fftLen,
		plan:    plan,
		scratch: newConvolverScratchCache[T](fftLen),
	}, nil
}

func newConvolverScratchCache[T Complex](fftLen int) *residentCache[convolverScratch[T]] {
	return newResidentCache(func() *convolverScratch[T] {
		return &convolverScratch[T]{
			aPadded: make([]T, fftLen),
			bPadded: make([]T, fftLen),
			aFreq:   make([]T, fftLen),
			bFreq:   make([]T, fftLen),
			time:    make([]T, fftLen),
		}
	})
}

// Len returns the output length of each convolution (lenA+lenB-1).
func (c *Convolver[T]) Len() int {
	return c.convLen
}

// Convolve computes the linear convolution of a and b into dst.
// a must have length lenA, b length lenB, and dst length lenA+lenB-1
// (as passed to and returned by the constructor).
func (c *Convolver[T]) Convolve(dst, a, b []T) error {
	if dst == nil || a == nil || b == nil {
		return ErrNilSlice
	}

	if len(a) != c.lenA || len(b) != c.lenB || len(dst) != c.convLen {
		return ErrLengthMismatch
	}

	s := c.scratch.get()
	defer c.scratch.put(s)

	copy(s.aPadded, a)
	clear(s.aPadded[len(a):])
	copy(s.bPadded, b)
	clear(s.bPadded[len(b):])

	if err := c.plan.Forward(s.aFreq, s.aPadded); err != nil {
		return err
	}

	if err := c.plan.Forward(s.bFreq, s.bPadded); err != nil {
		return err
	}

	fft.ComplexMulArrayInPlace(s.aFreq, s.bFreq)

	if err := c.plan.Inverse(s.time, s.aFreq); err != nil {
		return err
	}

	copy(dst, s.time[:c.convLen])

	return nil
}

// Correlator computes cross-correlations of fixed-length inputs, reusing one
// FFT plan and preallocated buffers across calls. Use it instead of the
// one-shot CrossCorrelate function when correlating in a loop: after
// creation, the Correlator.CrossCorrelate method performs no planning and no
// allocations.
//
// A Correlator is safe for concurrent use: scratch buffers are borrowed per
// call from an internal cache, so multiple goroutines may share one instance.
type Correlator[T Complex] struct {
	conv    *Convolver[T]
	scratch *residentCache[[]T]
}

// NewCorrelator creates a correlator for inputs of length lenA and lenB.
// The output of each CrossCorrelate call has length lenA+lenB-1.
//
// It produces the same results as the one-shot CrossCorrelate function, which
// it replaces for repeated use.
func NewCorrelator[T Complex](lenA, lenB int) (*Correlator[T], error) {
	conv, err := NewConvolver[T](lenA, lenB)
	if err != nil {
		return nil, err
	}

	return &Correlator[T]{
		conv: conv,
		scratch: newResidentCache(func() *[]T {
			s := make([]T, lenB)

			return &s
		}),
	}, nil
}

// Len returns the output length of each correlation (lenA+lenB-1).
func (c *Correlator[T]) Len() int {
	return c.conv.convLen
}

// CrossCorrelate computes the full cross-correlation of a and b into dst.
// a must have length lenA, b length lenB, and dst length lenA+lenB-1
// (as passed to and returned by the constructor).
// Output index k corresponds to lag k-(lenB-1).
func (c *Correlator[T]) CrossCorrelate(dst, a, b []T) error {
	if dst == nil || a == nil || b == nil {
		return ErrNilSlice
	}

	if len(a) != c.conv.lenA || len(b) != c.conv.lenB || len(dst) != c.conv.convLen {
		return ErrLengthMismatch
	}

	bRevConj := c.scratch.get()
	defer c.scratch.put(bRevConj)

	rev := *bRevConj
	for i := range b {
		rev[i] = fft.ConjugateOf(b[len(b)-1-i])
	}

	return c.conv.Convolve(dst, a, rev)
}

// RealConvolver computes linear convolutions of fixed-length real inputs,
// reusing one real FFT plan and preallocated buffers across calls. Use it
// instead of the one-shot ConvolveReal function when convolving in a loop:
// after creation, the RealConvolver.Convolve method performs no planning and
// no allocations.
//
// A RealConvolver is safe for concurrent use: scratch buffers are borrowed
// per call from an internal cache, so multiple goroutines may share one
// instance.
type RealConvolver[F Float, C Complex] struct {
	lenA, lenB int
	convLen    int
	fftLen     int
	plan       *PlanReal[F, C]
	scratch    *residentCache[realConvolverScratch[F, C]]
}

// realConvolverScratch is one per-call scratch set for RealConvolver runs.
type realConvolverScratch[F Float, C Complex] struct {
	aPadded []F
	bPadded []F
	aFreq   []C
	bFreq   []C
	time    []F
}

// NewRealConvolver creates a real-input convolver for inputs of length lenA
// and lenB. The output of each Convolve call has length lenA+lenB-1.
//
// It produces the same results as the one-shot ConvolveReal function, which
// it replaces for repeated use.
func NewRealConvolver[F Float, C Complex](lenA, lenB int) (*RealConvolver[F, C], error) {
	if lenA < 1 || lenB < 1 {
		return nil, ErrInvalidLength
	}

	convLen := lenA + lenB - 1
	fftLen := max(m.NextPowerOfTwo(convLen), 2)

	plan, err := NewPlanReal[F, C](fftLen)
	if err != nil {
		return nil, err
	}

	specLen := plan.SpectrumLen()

	return &RealConvolver[F, C]{
		lenA:    lenA,
		lenB:    lenB,
		convLen: convLen,
		fftLen:  fftLen,
		plan:    plan,
		scratch: newResidentCache(func() *realConvolverScratch[F, C] {
			return &realConvolverScratch[F, C]{
				aPadded: make([]F, fftLen),
				bPadded: make([]F, fftLen),
				aFreq:   make([]C, specLen),
				bFreq:   make([]C, specLen),
				time:    make([]F, fftLen),
			}
		}),
	}, nil
}

// Len returns the output length of each convolution (lenA+lenB-1).
func (c *RealConvolver[F, C]) Len() int {
	return c.convLen
}

// Convolve computes the linear convolution of a and b into dst.
// a must have length lenA, b length lenB, and dst length lenA+lenB-1
// (as passed to and returned by the constructor).
func (c *RealConvolver[F, C]) Convolve(dst, a, b []F) error {
	if dst == nil || a == nil || b == nil {
		return ErrNilSlice
	}

	if len(a) != c.lenA || len(b) != c.lenB || len(dst) != c.convLen {
		return ErrLengthMismatch
	}

	s := c.scratch.get()
	defer c.scratch.put(s)

	copy(s.aPadded, a)
	clear(s.aPadded[len(a):])
	copy(s.bPadded, b)
	clear(s.bPadded[len(b):])

	if err := c.plan.Forward(s.aFreq, s.aPadded); err != nil {
		return err
	}

	if err := c.plan.Forward(s.bFreq, s.bPadded); err != nil {
		return err
	}

	fft.ComplexMulArrayInPlace(s.aFreq, s.bFreq)

	if err := c.plan.Inverse(s.time, s.aFreq); err != nil {
		return err
	}

	copy(dst, s.time[:c.convLen])

	return nil
}
