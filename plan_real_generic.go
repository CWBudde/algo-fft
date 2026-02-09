package algofft

import (
	"math"
	"unsafe"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
)

// PlanRealT is a generic pre-computed real FFT plan supporting both float32 and float64 input.
// The forward transform returns the non-redundant half-spectrum with length N/2+1.
//
// Type parameters:
//   - F: float type (float32 or float64)
//   - C: complex type (complex64 or complex128), must match F
//
// Output bins obey conjugate symmetry for real inputs:
//
//	X[k] = conj(X[N-k]) for k = 1..N/2-1
//
// Index 0 is DC and index N/2 is Nyquist (purely real for even N).
type PlanRealT[F Float, C Complex] struct {
	n    int
	half int

	plan    *Plan[C]
	weight  []C
	buf     []C
	options PlanOptions
}

// NewPlanRealT creates a new generic real FFT plan for length n.
// The type parameter F determines the precision (float32 or float64).
// The complex type C must match F (float32→complex64, float64→complex128).
//
// Example:
//
//	// Float32 precision
//	plan32, err := algofft.NewPlanRealT[float32, complex64](4096)
//
//	// Float64 precision
//	plan64, err := algofft.NewPlanRealT[float64, complex128](4096)
func NewPlanRealT[F Float, C Complex](n int) (*PlanRealT[F, C], error) {
	return NewPlanRealTWithOptions[F, C](n, PlanOptions{})
}

// NewPlanRealTWithOptions creates a new generic real FFT plan with explicit planner options.
func NewPlanRealTWithOptions[F Float, C Complex](n int, opts PlanOptions) (*PlanRealT[F, C], error) {
	return newPlanRealTWithFeatures[F, C](n, cpu.DetectFeatures(), normalizePlanOptions(opts))
}

func newPlanRealTWithFeatures[F Float, C Complex](n int, features cpu.Features, opts PlanOptions) (*PlanRealT[F, C], error) {
	if n < 2 || n%2 != 0 {
		return nil, ErrInvalidLength
	}

	childOpts := opts
	childOpts.Batch = 0
	childOpts.Stride = 0
	// The real-FFT pack/unpack path uses the child complex plan in-place on p.buf.
	childOpts.InPlace = true

	plan, err := newPlanWithFeatures[C](n/2, features, childOpts)
	if err != nil {
		return nil, err
	}

	// Precompute U[k] weights for recombination:
	// U[k] = 0.5 * (1 + i*W_N^k) where W_N^k = exp(-2πik/N).
	weight := make([]C, n/2+1)
	for k := range weight {
		theta := 2 * math.Pi * float64(k) / float64(n)

		// Compute at full precision then cast to target type
		re := 0.5 * (1 + math.Sin(theta))
		im := 0.5 * math.Cos(theta)

		// Type switch to handle both precisions
		var zero C
		switch any(zero).(type) {
		case complex64:
			weight[k] = any(complex(float32(re), float32(im))).(C)
		case complex128:
			weight[k] = any(complex(re, im)).(C)
		}
	}

	return &PlanRealT[F, C]{
		n:       n,
		half:    n / 2,
		plan:    plan,
		weight:  weight,
		buf:     make([]C, n/2),
		options: opts,
	}, nil
}

// Len returns the number of real samples for this plan.
func (p *PlanRealT[F, C]) Len() int {
	return p.n
}

// SpectrumLen returns the number of complex frequency bins (N/2+1).
func (p *PlanRealT[F, C]) SpectrumLen() int {
	return p.half + 1
}

// Forward computes the real-to-complex FFT.
// dst must have length N/2+1 and src must have length N.
func (p *PlanRealT[F, C]) Forward(dst []C, src []F) error {
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

func (p *PlanRealT[F, C]) forwardSingle(dst []C, src []F) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(src) != p.n || len(dst) != p.half+1 {
		return ErrLengthMismatch
	}

	// Pack real samples into complex buffer: z[k] = src[2k] + i*src[2k+1]
	// Memory layout of []float32{r0,i0,r1,i1,...} is identical to []complex64,
	// so we can use unsafe.Slice to reinterpret and copy efficiently.
	var zero C
	switch any(zero).(type) {
	case complex64:
		srcF32 := any(src).([]float32)
		bufC64 := any(p.buf).([]complex64)
		srcAsComplex := unsafe.Slice((*complex64)(unsafe.Pointer(&srcF32[0])), p.half)
		copy(bufC64, srcAsComplex)
	case complex128:
		srcF64 := any(src).([]float64)
		bufC128 := any(p.buf).([]complex128)
		srcAsComplex := unsafe.Slice((*complex128)(unsafe.Pointer(&srcF64[0])), p.half)
		copy(bufC128, srcAsComplex)
	}

	// Perform N/2 complex FFT
	err := p.plan.Forward(p.buf, p.buf)
	if err != nil {
		return err
	}

	// Extract DC and Nyquist bins
	y0 := p.buf[0]

	switch any(zero).(type) {
	case complex64:
		y0C64 := any(y0).(complex64)
		dstC64 := any(dst).([]complex64)
		y0r := real(y0C64)
		y0i := imag(y0C64)
		dstC64[0] = complex(y0r+y0i, 0)
		dstC64[p.half] = complex(y0r-y0i, 0)
	case complex128:
		y0C128 := any(y0).(complex128)
		dstC128 := any(dst).([]complex128)
		y0r := real(y0C128)
		y0i := imag(y0C128)
		dstC128[0] = complex(y0r+y0i, 0)
		dstC128[p.half] = complex(y0r-y0i, 0)
	}

	// Recombination step: extract X[k] from the N/2-point FFT of packed data.
	// Given z[m] = x[2m] + i*x[2m+1], we computed Y = FFT(z).
	// With A[k] = Y[k], B[k] = conj(Y[N/2-k]), and U[k] = 0.5 * (1 + i*W_N^k),
	// the spectrum is recovered via: X[k] = A[k] - U[k] * (A[k] - B[k]).
	switch any(zero).(type) {
	case complex64:
		bufC64 := any(p.buf).([]complex64)
		dstC64 := any(dst).([]complex64)

		weightC64 := any(p.weight).([]complex64)
		for k := 1; k < p.half; k++ {
			a := bufC64[k]
			bSrc := bufC64[p.half-k]
			b := complex(real(bSrc), -imag(bSrc)) // conj(Y[N/2-k])

			c := weightC64[k] * (a - b)
			dstC64[k] = a - c
		}
	case complex128:
		bufC128 := any(p.buf).([]complex128)
		dstC128 := any(dst).([]complex128)

		weightC128 := any(p.weight).([]complex128)
		for k := 1; k < p.half; k++ {
			a := bufC128[k]
			bSrc := bufC128[p.half-k]
			b := complex(real(bSrc), -imag(bSrc)) // conj(Y[N/2-k])

			c := weightC128[k] * (a - b)
			dstC128[k] = a - c
		}
	}

	return nil
}

// ForwardNormalized computes the real-to-complex FFT and scales the result by 1/N.
func (p *PlanRealT[F, C]) ForwardNormalized(dst []C, src []F) error {
	err := p.Forward(dst, src)
	if err != nil {
		return err
	}

	scale := 1.0 / float64(p.n)
	scaleSpectrumGeneric(dst, scale)

	return nil
}

// ForwardUnitary computes the real-to-complex FFT and scales the result by 1/sqrt(N).
func (p *PlanRealT[F, C]) ForwardUnitary(dst []C, src []F) error {
	err := p.Forward(dst, src)
	if err != nil {
		return err
	}

	scale := 1.0 / math.Sqrt(float64(p.n))
	scaleSpectrumGeneric(dst, scale)

	return nil
}

// Inverse computes the complex-to-real inverse FFT.
// dst must have length N and src must have length N/2+1.
func (p *PlanRealT[F, C]) Inverse(dst []F, src []C) error {
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

func (p *PlanRealT[F, C]) inverseSingle(dst []F, src []C) error {
	if dst == nil || src == nil {
		return ErrNilSlice
	}

	if len(dst) != p.n || len(src) != p.half+1 {
		return ErrLengthMismatch
	}

	// Validate DC and Nyquist are real (imaginary parts near zero)
	var zero C

	spectrumEps := 1e-4

	switch any(zero).(type) {
	case complex64:
		srcC64 := any(src).([]complex64)
		if math.Abs(float64(imag(srcC64[0]))) > spectrumEps || math.Abs(float64(imag(srcC64[p.half]))) > spectrumEps {
			return ErrInvalidSpectrum
		}
	case complex128:
		srcC128 := any(src).([]complex128)

		spectrumEps = 1e-12 // Tighter tolerance for float64
		if math.Abs(imag(srcC128[0])) > spectrumEps || math.Abs(imag(srcC128[p.half])) > spectrumEps {
			return ErrInvalidSpectrum
		}
	}

	// Reconstruct packed buffer from half-spectrum
	switch any(zero).(type) {
	case complex64:
		srcC64 := any(src).([]complex64)
		bufC64 := any(p.buf).([]complex64)
		weightC64 := any(p.weight).([]complex64)
		fft.RepackInverseComplex64(bufC64, srcC64, weightC64)
	case complex128:
		srcC128 := any(src).([]complex128)
		bufC128 := any(p.buf).([]complex128)
		weightC128 := any(p.weight).([]complex128)
		fft.RepackInverseComplex128(bufC128, srcC128, weightC128)
	}

	// Inverse N/2 complex FFT
	err := p.plan.Inverse(p.buf, p.buf)
	if err != nil {
		return err
	}

	// Unpack complex buffer to real output
	// Memory layout of []complex64 is identical to []float32{r0,i0,r1,i1,...},
	// so we can use unsafe.Slice to reinterpret and copy efficiently.
	switch any(zero).(type) {
	case complex64:
		bufC64 := any(p.buf).([]complex64)
		dstF32 := any(dst).([]float32)
		dstAsComplex := unsafe.Slice((*complex64)(unsafe.Pointer(&dstF32[0])), p.half)
		copy(dstAsComplex, bufC64)
	case complex128:
		bufC128 := any(p.buf).([]complex128)
		dstF64 := any(dst).([]float64)
		dstAsComplex := unsafe.Slice((*complex128)(unsafe.Pointer(&dstF64[0])), p.half)
		copy(dstAsComplex, bufC128)
	}

	return nil
}

func scaleSpectrumGeneric[C Complex](dst []C, scale float64) {
	if scale == 1.0 {
		return
	}

	var zero C
	switch any(zero).(type) {
	case complex64:
		dstC64 := any(dst).([]complex64)

		fft.ScaleComplex64InPlace(dstC64, float32(scale))
	case complex128:
		dstC128 := any(dst).([]complex128)

		fft.ScaleComplex128InPlace(dstC128, scale)
	}
}
