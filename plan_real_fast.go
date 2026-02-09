package algofft

import (
	"math"
	"unsafe"

	"github.com/cwbudde/algo-fft/internal/fft"
)

// FastPlanReal32 provides zero-overhead real FFT for float32/complex64.
// All operations are type-specialized with no runtime type switches.
//
// Unlike PlanRealT, FastPlanReal32:
//   - Has no runtime type switches
//   - Uses a FastPlan internally (direct codelet bindings)
//   - Performs no validation on Forward/Inverse calls
//
// Use NewFastPlanReal32 to create instances. Returns ErrNotImplemented if
// no codelet is available for the underlying complex FFT size.
type FastPlanReal32 struct {
	n      int
	half   int
	weight []complex64
	buf    []complex64
	inner  *FastPlan[complex64]
}

// NewFastPlanReal32 creates an optimized real FFT plan for float32.
// The size n must be a power of 2 and >= 2.
// Returns ErrNotImplemented if no codelet is available for size n/2.
func NewFastPlanReal32(n int) (*FastPlanReal32, error) {
	if n < 2 || n%2 != 0 {
		return nil, ErrInvalidLength
	}

	inner, err := NewFastPlan[complex64](n / 2)
	if err != nil {
		return nil, err
	}

	half := n / 2

	weight := make([]complex64, half+1)
	for k := range weight {
		theta := 2 * math.Pi * float64(k) / float64(n)
		weight[k] = complex(float32(0.5*(1+math.Sin(theta))), float32(0.5*math.Cos(theta)))
	}

	buf := make([]complex64, half)

	return &FastPlanReal32{
		n:      n,
		half:   half,
		weight: weight,
		buf:    buf,
		inner:  inner,
	}, nil
}

// Len returns the number of real samples.
func (fp *FastPlanReal32) Len() int {
	return fp.n
}

// SpectrumLen returns the number of complex frequency bins (N/2+1).
func (fp *FastPlanReal32) SpectrumLen() int {
	return fp.half + 1
}

// Forward computes real→complex FFT without validation.
// Caller guarantees: len(dst) >= n/2+1, len(src) >= n, slices non-nil.
func (fp *FastPlanReal32) Forward(dst []complex64, src []float32) {
	half := fp.half
	buf := fp.buf

	// Pack: z[k] = src[2k] + i*src[2k+1]
	srcAsComplex := unsafe.Slice((*complex64)(unsafe.Pointer(&src[0])), half)
	copy(buf, srcAsComplex)

	// N/2 complex FFT (direct call, no validation)
	fp.inner.Forward(buf, buf)

	// DC and Nyquist
	y0r := real(buf[0])
	y0i := imag(buf[0])
	dst[0] = complex(y0r+y0i, 0)
	dst[half] = complex(y0r-y0i, 0)

	// Recombination: X[k] = A[k] - U[k] * (A[k] - B[k])
	weight := fp.weight

	for k := 1; k < half; k++ {
		a := buf[k]
		bSrc := buf[half-k]
		b := complex(real(bSrc), -imag(bSrc)) // conj(Y[N/2-k])
		c := weight[k] * (a - b)
		dst[k] = a - c
	}
}

// ForwardNormalized computes real→complex FFT and scales the result by 1/N.
// Caller guarantees: len(dst) >= n/2+1, len(src) >= n, slices non-nil.
func (fp *FastPlanReal32) ForwardNormalized(dst []complex64, src []float32) {
	fp.Forward(dst, src)
	fft.ScaleComplex64InPlace(dst, float32(1.0/float64(fp.n)))
}

// ForwardUnitary computes real→complex FFT and scales the result by 1/sqrt(N).
// Caller guarantees: len(dst) >= n/2+1, len(src) >= n, slices non-nil.
func (fp *FastPlanReal32) ForwardUnitary(dst []complex64, src []float32) {
	fp.Forward(dst, src)
	fft.ScaleComplex64InPlace(dst, float32(1.0/math.Sqrt(float64(fp.n))))
}

// Inverse computes complex→real IFFT without validation.
// Caller guarantees: len(dst) >= n, len(src) >= n/2+1, slices non-nil.
//
// Note: Unlike the safe API, this does NOT validate that DC and Nyquist bins
// have zero imaginary parts. The caller must ensure the spectrum is valid.
func (fp *FastPlanReal32) Inverse(dst []float32, src []complex64) {
	half := fp.half
	buf := fp.buf

	// Reconstruct packed buffer from half-spectrum.
	fft.RepackInverseComplex64(buf, src, fp.weight)

	// Inverse N/2 complex FFT
	fp.inner.Inverse(buf, buf)

	// Unpack complex buffer to real output
	dstAsComplex := unsafe.Slice((*complex64)(unsafe.Pointer(&dst[0])), half)
	copy(dstAsComplex, buf)
}

// FastPlanReal64 provides zero-overhead real FFT for float64/complex128.
// All operations are type-specialized with no runtime type switches.
//
// Unlike PlanRealT, FastPlanReal64:
//   - Has no runtime type switches
//   - Uses a FastPlan internally (direct codelet bindings)
//   - Performs no validation on Forward/Inverse calls
//
// Use NewFastPlanReal64 to create instances. Returns ErrNotImplemented if
// no codelet is available for the underlying complex FFT size.
type FastPlanReal64 struct {
	n      int
	half   int
	weight []complex128
	buf    []complex128
	inner  *FastPlan[complex128]
}

// NewFastPlanReal64 creates an optimized real FFT plan for float64.
// The size n must be a power of 2 and >= 2.
// Returns ErrNotImplemented if no codelet is available for size n/2.
func NewFastPlanReal64(n int) (*FastPlanReal64, error) {
	if n < 2 || n%2 != 0 {
		return nil, ErrInvalidLength
	}

	inner, err := NewFastPlan[complex128](n / 2)
	if err != nil {
		return nil, err
	}

	half := n / 2

	weight := make([]complex128, half+1)
	for k := range weight {
		theta := 2 * math.Pi * float64(k) / float64(n)
		weight[k] = complex(0.5*(1+math.Sin(theta)), 0.5*math.Cos(theta))
	}

	buf := make([]complex128, half)

	return &FastPlanReal64{
		n:      n,
		half:   half,
		weight: weight,
		buf:    buf,
		inner:  inner,
	}, nil
}

// Len returns the number of real samples.
func (fp *FastPlanReal64) Len() int {
	return fp.n
}

// SpectrumLen returns the number of complex frequency bins (N/2+1).
func (fp *FastPlanReal64) SpectrumLen() int {
	return fp.half + 1
}

// Forward computes real→complex FFT without validation.
// Caller guarantees: len(dst) >= n/2+1, len(src) >= n, slices non-nil.
func (fp *FastPlanReal64) Forward(dst []complex128, src []float64) {
	half := fp.half
	buf := fp.buf

	// Pack: z[k] = src[2k] + i*src[2k+1]
	srcAsComplex := unsafe.Slice((*complex128)(unsafe.Pointer(&src[0])), half)
	copy(buf, srcAsComplex)

	// N/2 complex FFT (direct call, no validation)
	fp.inner.Forward(buf, buf)

	// DC and Nyquist
	y0r := real(buf[0])
	y0i := imag(buf[0])
	dst[0] = complex(y0r+y0i, 0)
	dst[half] = complex(y0r-y0i, 0)

	// Recombination: X[k] = A[k] - U[k] * (A[k] - B[k])
	weight := fp.weight

	for k := 1; k < half; k++ {
		a := buf[k]
		bSrc := buf[half-k]
		b := complex(real(bSrc), -imag(bSrc)) // conj(Y[N/2-k])
		c := weight[k] * (a - b)
		dst[k] = a - c
	}
}

// ForwardNormalized computes real→complex FFT and scales the result by 1/N.
// Caller guarantees: len(dst) >= n/2+1, len(src) >= n, slices non-nil.
func (fp *FastPlanReal64) ForwardNormalized(dst []complex128, src []float64) {
	fp.Forward(dst, src)
	fft.ScaleComplex128InPlace(dst, 1.0/float64(fp.n))
}

// ForwardUnitary computes real→complex FFT and scales the result by 1/sqrt(N).
// Caller guarantees: len(dst) >= n/2+1, len(src) >= n, slices non-nil.
func (fp *FastPlanReal64) ForwardUnitary(dst []complex128, src []float64) {
	fp.Forward(dst, src)
	fft.ScaleComplex128InPlace(dst, 1.0/math.Sqrt(float64(fp.n)))
}

// Inverse computes complex→real IFFT without validation.
// Caller guarantees: len(dst) >= n, len(src) >= n/2+1, slices non-nil.
//
// Note: Unlike the safe API, this does NOT validate that DC and Nyquist bins
// have zero imaginary parts. The caller must ensure the spectrum is valid.
func (fp *FastPlanReal64) Inverse(dst []float64, src []complex128) {
	half := fp.half
	buf := fp.buf

	// Reconstruct packed buffer from half-spectrum.
	fft.RepackInverseComplex128(buf, src, fp.weight)

	// Inverse N/2 complex FFT
	fp.inner.Inverse(buf, buf)

	// Unpack complex buffer to real output
	dstAsComplex := unsafe.Slice((*complex128)(unsafe.Pointer(&dst[0])), half)
	copy(dstAsComplex, buf)
}
