package algofft

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
)

// Tolerances bounding the imaginary component allowed in the DC (and, for even
// lengths, Nyquist) bins of a spectrum passed to a real inverse transform.
// Real-input FFTs produce purely real DC/Nyquist bins; a spectrum whose
// imaginary part exceeds these bounds cannot come from real data and is
// rejected with ErrInvalidSpectrum. The float32 bound is looser to absorb the
// larger rounding error of complex64 arithmetic.
const (
	spectrumImagTol32 = 1e-4  // complex64 (float32) inputs
	spectrumImagTol64 = 1e-12 // complex128 (float64) inputs
)

// PlanReal is a generic pre-computed real FFT plan supporting both float32
// and float64 input. The forward transform returns the non-redundant
// half-spectrum with length N/2+1. Plans are reusable and safe for concurrent
// use during transforms: the pack/unpack buffer is borrowed per call from an
// internal cache, so multiple goroutines may share one instance.
//
// Type parameters:
//   - F: float type (float32 or float64)
//   - C: complex type (complex64 or complex128), must match F
//
// Output bins obey conjugate symmetry for real inputs:
//
//	X[k] = conj(X[N-k]) for k = 1..N/2-1
//
// Index 0 is DC and index N/2 is Nyquist (purely real for even N; odd
// lengths have no Nyquist bin).
//
// Even lengths use the packed half-size complex FFT method. Odd lengths are
// supported via an internal full-size complex FFT fallback — correct for any
// length the complex planner handles (including primes), at roughly 2× the
// memory and flops of the packed method.
type PlanReal[F Float, C Complex] struct {
	n    int
	half int

	plan   *Plan[C]
	weight []C
	buf    *residentCache[[]C]
}

func newPlanRealBufCache[C Complex](half int) *residentCache[[]C] {
	return newResidentCache(func() *[]C {
		b := make([]C, half)

		return &b
	})
}

// NewPlanReal creates a new generic real FFT plan for length n (n >= 2).
// The type parameter F determines the precision (float32 or float64).
// The complex type C must match F (float32→complex64, float64→complex128).
//
// Even lengths run the packed half-size method; odd lengths run a full-size
// complex FFT internally (see PlanReal).
//
// Example:
//
//	// Float32 precision
//	plan32, err := algofft.NewPlanReal[float32, complex64](4096)
//
//	// Float64 precision
//	plan64, err := algofft.NewPlanReal[float64, complex128](4096)
func NewPlanReal[F Float, C Complex](n int) (*PlanReal[F, C], error) {
	return NewPlanRealWithOptions[F, C](n, PlanOptions{})
}

// NewPlanRealWithOptions creates a new generic real FFT plan with explicit planner options.
func NewPlanRealWithOptions[F Float, C Complex](n int, opts PlanOptions) (*PlanReal[F, C], error) {
	return newPlanRealWithFeatures[F, C](n, cpu.DetectFeatures(), normalizePlanOptions(opts))
}

func newPlanRealWithFeatures[F Float, C Complex](
	n int, features cpu.Features, opts PlanOptions,
) (*PlanReal[F, C], error) {
	if n < 2 {
		return nil, ErrInvalidLength
	}

	if n%2 != 0 {
		return newPlanRealOddWithFeatures[F, C](n, features, opts)
	}

	plan, err := newPlanWithFeatures[C](n/2, features, opts)
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

	return &PlanReal[F, C]{
		n:      n,
		half:   n / 2,
		plan:   plan,
		weight: weight,
		buf:    newPlanRealBufCache[C](n / 2),
	}, nil
}

// Len returns the number of real samples for this plan.
func (p *PlanReal[F, C]) Len() int {
	return p.n
}

// SpectrumLen returns the number of complex frequency bins (N/2+1).
func (p *PlanReal[F, C]) SpectrumLen() int {
	return p.half + 1
}

// realPlanTypeNames names the F→C type pair of a real plan for String().
func realPlanTypeNames[C Complex]() string {
	if complexTypeName[C]() == precisionNameComplex128 {
		return "float64→complex128"
	}

	return "float32→complex64"
}

// String returns a human-readable description of the PlanReal for debugging.
func (p *PlanReal[F, C]) String() string {
	return fmt.Sprintf("PlanReal[%s](%d → %d)", realPlanTypeNames[C](), p.n, p.half+1)
}

// Clone creates an independent copy of the PlanReal.
//
// A single PlanReal is already safe for concurrent transforms, so cloning is
// not required for concurrency; it remains available for callers that want
// isolated scratch caches. The clone shares immutable data (the recombination
// weights) and the concurrency-safe child complex plan, but has its own
// pack/unpack buffer cache.
func (p *PlanReal[F, C]) Clone() *PlanReal[F, C] {
	return &PlanReal[F, C]{
		n:      p.n,
		half:   p.half,
		plan:   p.plan,
		weight: p.weight, // Shared (immutable)
		buf:    newPlanRealBufCache[C](p.plan.Len()),
	}
}

// Forward computes the real-to-complex FFT.
// dst must have length N/2+1 and src must have length N.
func (p *PlanReal[F, C]) Forward(dst []C, src []F) error {
	err := validateDstSrc(dst, src, p.half+1, p.n)
	if err != nil {
		return err
	}

	if p.n%2 != 0 {
		return p.forwardOdd(dst, src)
	}

	bufp := p.buf.get()
	defer p.buf.put(bufp)

	buf := *bufp

	// Pack real samples into complex buffer: z[k] = src[2k] + i*src[2k+1]
	// Memory layout of []float32{r0,i0,r1,i1,...} is identical to []complex64,
	// so we can use unsafe.Slice to reinterpret and copy efficiently.
	var zero C
	switch any(zero).(type) {
	case complex64:
		srcF32 := any(src).([]float32)
		bufC64 := any(buf).([]complex64)
		srcAsComplex := unsafe.Slice((*complex64)(unsafe.Pointer(&srcF32[0])), p.half)
		copy(bufC64, srcAsComplex)
	case complex128:
		srcF64 := any(src).([]float64)
		bufC128 := any(buf).([]complex128)
		srcAsComplex := unsafe.Slice((*complex128)(unsafe.Pointer(&srcF64[0])), p.half)
		copy(bufC128, srcAsComplex)
	}

	// Perform N/2 complex FFT
	err = p.plan.Forward(buf, buf)
	if err != nil {
		return err
	}

	// Extract DC and Nyquist bins
	y0 := buf[0]

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
		bufC64 := any(buf).([]complex64)
		dstC64 := any(dst).([]complex64)
		weightC64 := any(p.weight).([]complex64)
		fft.RecombineForwardComplex64(dstC64, bufC64, weightC64)
	case complex128:
		bufC128 := any(buf).([]complex128)
		dstC128 := any(dst).([]complex128)
		weightC128 := any(p.weight).([]complex128)
		fft.RecombineForwardComplex128(dstC128, bufC128, weightC128)
	}

	return nil
}

// ForwardNormalized computes the real-to-complex FFT and scales the result by 1/N.
func (p *PlanReal[F, C]) ForwardNormalized(dst []C, src []F) error {
	err := p.Forward(dst, src)
	if err != nil {
		return err
	}

	scale := 1.0 / float64(p.n)
	scaleSpectrumGeneric(dst, scale)

	return nil
}

// ForwardUnitary computes the real-to-complex FFT and scales the result by 1/sqrt(N).
func (p *PlanReal[F, C]) ForwardUnitary(dst []C, src []F) error {
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
func (p *PlanReal[F, C]) Inverse(dst []F, src []C) error {
	err := validateDstSrc(dst, src, p.n, p.half+1)
	if err != nil {
		return err
	}

	if p.n%2 != 0 {
		return p.inverseOdd(dst, src)
	}

	// Validate DC and Nyquist are real (imaginary parts near zero)
	var zero C

	switch any(zero).(type) {
	case complex64:
		srcC64 := any(src).([]complex64)
		if math.Abs(float64(imag(srcC64[0]))) > spectrumImagTol32 ||
			math.Abs(float64(imag(srcC64[p.half]))) > spectrumImagTol32 {
			return ErrInvalidSpectrum
		}
	case complex128:
		srcC128 := any(src).([]complex128)
		if math.Abs(imag(srcC128[0])) > spectrumImagTol64 || math.Abs(imag(srcC128[p.half])) > spectrumImagTol64 {
			return ErrInvalidSpectrum
		}
	}

	bufp := p.buf.get()
	defer p.buf.put(bufp)

	buf := *bufp

	// Reconstruct packed buffer from half-spectrum
	switch any(zero).(type) {
	case complex64:
		srcC64 := any(src).([]complex64)
		bufC64 := any(buf).([]complex64)
		weightC64 := any(p.weight).([]complex64)
		fft.RepackInverseComplex64(bufC64, srcC64, weightC64)
	case complex128:
		srcC128 := any(src).([]complex128)
		bufC128 := any(buf).([]complex128)
		weightC128 := any(p.weight).([]complex128)
		fft.RepackInverseComplex128(bufC128, srcC128, weightC128)
	}

	// Inverse N/2 complex FFT
	err = p.plan.Inverse(buf, buf)
	if err != nil {
		return err
	}

	// Unpack complex buffer to real output
	// Memory layout of []complex64 is identical to []float32{r0,i0,r1,i1,...},
	// so we can use unsafe.Slice to reinterpret and copy efficiently.
	switch any(zero).(type) {
	case complex64:
		bufC64 := any(buf).([]complex64)
		dstF32 := any(dst).([]float32)
		dstAsComplex := unsafe.Slice((*complex64)(unsafe.Pointer(&dstF32[0])), p.half)
		copy(dstAsComplex, bufC64)
	case complex128:
		bufC128 := any(buf).([]complex128)
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
