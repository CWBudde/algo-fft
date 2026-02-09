package algofft

import (
	"math"
	"unsafe"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
)

// PlanReal is a pre-computed real FFT plan for float32 input.
// The forward transform returns the non-redundant half-spectrum with length N/2+1.
//
// Output bins obey conjugate symmetry for real inputs:
//
//	X[k] = conj(X[N-k]) for k = 1..N/2-1
//
// Index 0 is DC and index N/2 is Nyquist (purely real for even N).
type PlanReal struct {
	n    int
	half int

	plan    *Plan[complex64]
	weight  []complex64
	buf     []complex64
	options PlanOptions
}

// NewPlanReal creates a new real FFT plan for length n.
// Currently, only even lengths are supported by the real FFT pack method.
func NewPlanReal(n int) (*PlanReal, error) {
	return NewPlanRealWithOptions(n, PlanOptions{})
}

// NewPlanRealWithOptions creates a new real FFT plan with explicit planner options.
func NewPlanRealWithOptions(n int, opts PlanOptions) (*PlanReal, error) {
	return newPlanRealWithFeatures(n, cpu.DetectFeatures(), normalizePlanOptions(opts))
}

func newPlanRealWithFeatures(n int, features cpu.Features, opts PlanOptions) (*PlanReal, error) {
	if n < 2 || n%2 != 0 {
		return nil, ErrInvalidLength
	}

	childOpts := opts
	childOpts.Batch = 0
	childOpts.Stride = 0
	// The real-FFT pack/unpack path uses the child complex plan in-place on p.buf.
	childOpts.InPlace = true

	plan, err := newPlanWithFeatures[complex64](n/2, features, childOpts)
	if err != nil {
		return nil, err
	}

	// Precompute U[k] weights for recombination:
	// U[k] = 0.5 * (1 + i*W_N^k) where W_N^k = exp(-2πik/N).
	weight := make([]complex64, n/2+1)
	for k := range weight {
		theta := 2 * math.Pi * float64(k) / float64(n)
		weight[k] = complex64(complex(0.5*(1+math.Sin(theta)), 0.5*math.Cos(theta)))
	}

	return &PlanReal{
		n:       n,
		half:    n / 2,
		plan:    plan,
		weight:  weight,
		buf:     make([]complex64, n/2),
		options: opts,
	}, nil
}

// Len returns the number of real samples for this plan.
func (p *PlanReal) Len() int {
	return p.n
}

// SpectrumLen returns the number of complex frequency bins (N/2+1).
func (p *PlanReal) SpectrumLen() int {
	return p.half + 1
}

// Forward computes the real-to-complex FFT.
// dst must have length N/2+1 and src must have length N.
func (p *PlanReal) Forward(dst []complex64, src []float32) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if p.options.Batch <= 1 && p.options.Stride <= 0 {
		return p.forwardSingle(dst, src)
	}

	batch, strideIn, strideOut, err := resolveBatchStrideReal(p.n, p.half+1, p.options)
	if err != nil {
		return err
	}

	for b := range batch {
		srcOff := b * strideIn

		dstOff := b * strideOut
		if srcOff+p.n > len(src) || dstOff+p.half+1 > len(dst) {
			return ErrLengthMismatch
		}

		err = p.forwardSingle(dst[dstOff:dstOff+p.half+1], src[srcOff:srcOff+p.n])
		if err != nil {
			return err
		}
	}

	return nil
}

// ForwardNormalized computes the real-to-complex FFT and scales the result by 1/N.
func (p *PlanReal) ForwardNormalized(dst []complex64, src []float32) error {
	err := p.Forward(dst, src)
	if err != nil {
		return err
	}

	scale := float32(1.0 / float64(p.n))
	scaleSpectrumComplex64(dst, scale)

	return nil
}

// ForwardUnitary computes the real-to-complex FFT and scales the result by 1/sqrt(N).
func (p *PlanReal) ForwardUnitary(dst []complex64, src []float32) error {
	err := p.Forward(dst, src)
	if err != nil {
		return err
	}

	scale := float32(1.0 / math.Sqrt(float64(p.n)))
	scaleSpectrumComplex64(dst, scale)

	return nil
}

// Inverse computes the complex-to-real inverse FFT.
// dst must have length N and src must have length N/2+1.
func (p *PlanReal) Inverse(dst []float32, src []complex64) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if p.options.Batch <= 1 && p.options.Stride <= 0 {
		return p.inverseSingle(dst, src)
	}

	batch, strideIn, strideOut, err := resolveBatchStrideReal(p.n, p.half+1, p.options)
	if err != nil {
		return err
	}

	for b := range batch {
		dstOff := b * strideIn

		srcOff := b * strideOut
		if dstOff+p.n > len(dst) || srcOff+p.half+1 > len(src) {
			return ErrLengthMismatch
		}

		err = p.inverseSingle(dst[dstOff:dstOff+p.n], src[srcOff:srcOff+p.half+1])
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *PlanReal) forwardSingle(dst []complex64, src []float32) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(src) != p.n || len(dst) != p.half+1 {
		return ErrLengthMismatch
	}

	srcAsComplex := unsafe.Slice((*complex64)(unsafe.Pointer(&src[0])), p.half)
	copy(p.buf, srcAsComplex)

	err := p.plan.Forward(p.buf, p.buf)
	if err != nil {
		return err
	}

	y0 := p.buf[0]
	y0r := real(y0)
	y0i := imag(y0)
	dst[0] = complex(y0r+y0i, 0)
	dst[p.half] = complex(y0r-y0i, 0)

	// Recombination step: extract X[k] from the N/2-point FFT of packed data.
	// Given z[m] = x[2m] + i*x[2m+1], we computed Y = FFT(z).
	// With A[k] = Y[k], B[k] = conj(Y[N/2-k]), and U[k] = 0.5 * (1 + i*W_N^k),
	// the spectrum is recovered via: X[k] = A[k] - U[k] * (A[k] - B[k]).
	for k := 1; k < p.half; k++ {
		a := p.buf[k]
		bSrc := p.buf[p.half-k]
		b := complex(real(bSrc), -imag(bSrc)) // conj(Y[N/2-k])

		c := p.weight[k] * (a - b)
		dst[k] = a - c
	}

	return nil
}

func (p *PlanReal) inverseSingle(dst []float32, src []complex64) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(dst) != p.n || len(src) != p.half+1 {
		return ErrLengthMismatch
	}

	const spectrumEps = 1e-4

	if math.Abs(float64(imag(src[0]))) > spectrumEps || math.Abs(float64(imag(src[p.half]))) > spectrumEps {
		return ErrInvalidSpectrum
	}

	fft.RepackInverseComplex64(p.buf, src, p.weight)

	err := p.plan.Inverse(p.buf, p.buf)
	if err != nil {
		return err
	}

	dstAsComplex := unsafe.Slice((*complex64)(unsafe.Pointer(&dst[0])), p.half)
	copy(dstAsComplex, p.buf)

	return nil
}

func scaleSpectrumComplex64(dst []complex64, scale float32) {
	if scale == 1 {
		return
	}

	fft.ScaleComplex64InPlace(dst, scale)
}
