package algofft

import (
	"math"
	"unsafe"

	"github.com/cwbudde/algo-fft/internal/fft"
)

// FastPlanReal provides zero-overhead real FFT transforms.
// The recombination helpers are bound once at construction, so the transform
// hot path contains no runtime type switches.
//
// Type parameters:
//   - F: float type (float32 or float64)
//   - C: complex type (complex64 or complex128), must match F
//
// Unlike PlanReal, FastPlanReal:
//   - Uses a FastPlan internally (direct codelet bindings)
//   - Performs no validation on Forward/Inverse calls
//
// Use NewFastPlanReal to create instances. Returns ErrNotImplemented if no
// codelet is available for the underlying complex FFT size.
type FastPlanReal[F Float, C Complex] struct {
	n      int
	half   int
	weight []C
	buf    []C
	inner  *FastPlan[C]

	// Type-specialized helpers bound at construction (no per-call switches).
	recombineForward func(dst, src, weight []C)
	repackInverse    func(dst, src, weight []C)
	scaleInPlace     func(dst []C, scale float64)
}

// NewFastPlanReal creates an optimized real FFT plan.
// The size n must be a power of 2 and >= 2.
// Returns ErrNotImplemented if no codelet is available for size n/2.
//
// Example:
//
//	fp32, err := algofft.NewFastPlanReal[float32, complex64](1024)
//	fp64, err := algofft.NewFastPlanReal[float64, complex128](1024)
func NewFastPlanReal[F Float, C Complex](n int) (*FastPlanReal[F, C], error) {
	if n < 2 || n%2 != 0 {
		return nil, ErrInvalidLength
	}

	inner, err := NewFastPlan[C](n / 2)
	if err != nil {
		return nil, err
	}

	half := n / 2

	weight := make([]C, half+1)
	for k := range weight {
		theta := 2 * math.Pi * float64(k) / float64(n)
		re := 0.5 * (1 + math.Sin(theta))
		im := 0.5 * math.Cos(theta)

		var zero C
		switch any(zero).(type) {
		case complex64:
			weight[k] = any(complex(float32(re), float32(im))).(C)
		case complex128:
			weight[k] = any(complex(re, im)).(C)
		}
	}

	fp := &FastPlanReal[F, C]{
		n:      n,
		half:   half,
		weight: weight,
		buf:    make([]C, half),
		inner:  inner,
	}

	// Bind the type-specialized helpers once so transforms stay switch-free.
	var zero C
	switch any(zero).(type) {
	case complex64:
		fp.recombineForward = func(dst, src, weight []C) {
			fft.RecombineForwardComplex64(any(dst).([]complex64), any(src).([]complex64), any(weight).([]complex64))
		}
		fp.repackInverse = func(dst, src, weight []C) {
			fft.RepackInverseComplex64(any(dst).([]complex64), any(src).([]complex64), any(weight).([]complex64))
		}
		fp.scaleInPlace = func(dst []C, scale float64) {
			fft.ScaleComplex64InPlace(any(dst).([]complex64), float32(scale))
		}
	case complex128:
		fp.recombineForward = func(dst, src, weight []C) {
			fft.RecombineForwardComplex128(any(dst).([]complex128), any(src).([]complex128), any(weight).([]complex128))
		}
		fp.repackInverse = func(dst, src, weight []C) {
			fft.RepackInverseComplex128(any(dst).([]complex128), any(src).([]complex128), any(weight).([]complex128))
		}
		fp.scaleInPlace = func(dst []C, scale float64) {
			fft.ScaleComplex128InPlace(any(dst).([]complex128), scale)
		}
	}

	return fp, nil
}

// NewFastPlanReal32 creates an optimized real FFT plan for float32.
// This is one-line sugar for NewFastPlanReal[float32, complex64](n).
func NewFastPlanReal32(n int) (*FastPlanReal[float32, complex64], error) {
	return NewFastPlanReal[float32, complex64](n)
}

// NewFastPlanReal64 creates an optimized real FFT plan for float64.
// This is one-line sugar for NewFastPlanReal[float64, complex128](n).
func NewFastPlanReal64(n int) (*FastPlanReal[float64, complex128], error) {
	return NewFastPlanReal[float64, complex128](n)
}

// Len returns the number of real samples.
func (fp *FastPlanReal[F, C]) Len() int {
	return fp.n
}

// SpectrumLen returns the number of complex frequency bins (N/2+1).
func (fp *FastPlanReal[F, C]) SpectrumLen() int {
	return fp.half + 1
}

// String returns a human-readable description of the FastPlanReal for debugging.
func (fp *FastPlanReal[F, C]) String() string {
	return "FastPlanReal[" + realPlanTypeNames[C]() + "](" + itoa(fp.n) + " → " + itoa(fp.half+1) + ")"
}

// Clone creates an independent copy of the FastPlanReal with its own pack
// buffer and inner FastPlan scratch, so each goroutine can transform
// concurrently on its own clone. The immutable weight table is shared.
func (fp *FastPlanReal[F, C]) Clone() *FastPlanReal[F, C] {
	clone := *fp
	clone.buf = make([]C, fp.half)
	clone.inner = fp.inner.Clone()

	return &clone
}

// Forward computes real→complex FFT without validation.
// Caller guarantees: len(dst) >= n/2+1, len(src) >= n, slices non-nil.
func (fp *FastPlanReal[F, C]) Forward(dst []C, src []F) {
	half := fp.half
	buf := fp.buf

	// Pack: z[k] = src[2k] + i*src[2k+1]. The memory layout of the real
	// input pairs is identical to the complex layout, so reinterpret + copy.
	srcAsComplex := unsafe.Slice((*C)(unsafe.Pointer(&src[0])), half)
	copy(buf, srcAsComplex)

	// N/2 complex FFT (direct call, no validation)
	fp.inner.Forward(buf, buf)

	// DC and Nyquist
	y0 := buf[0]
	dst[0] = complexOf[C](realPart(y0)+imagPart(y0), 0)
	dst[half] = complexOf[C](realPart(y0)-imagPart(y0), 0)

	// Recombination: X[k] = A[k] - U[k] * (A[k] - B[k])
	fp.recombineForward(dst, buf, fp.weight)
}

// ForwardNormalized computes real→complex FFT and scales the result by 1/N.
// Caller guarantees: len(dst) >= n/2+1, len(src) >= n, slices non-nil.
func (fp *FastPlanReal[F, C]) ForwardNormalized(dst []C, src []F) {
	fp.Forward(dst, src)
	fp.scaleInPlace(dst, 1.0/float64(fp.n))
}

// ForwardUnitary computes real→complex FFT and scales the result by 1/sqrt(N).
// Caller guarantees: len(dst) >= n/2+1, len(src) >= n, slices non-nil.
func (fp *FastPlanReal[F, C]) ForwardUnitary(dst []C, src []F) {
	fp.Forward(dst, src)
	fp.scaleInPlace(dst, 1.0/math.Sqrt(float64(fp.n)))
}

// Inverse computes complex→real IFFT without validation.
// Caller guarantees: len(dst) >= n, len(src) >= n/2+1, slices non-nil.
//
// Note: Unlike the safe API, this does NOT validate that DC and Nyquist bins
// have zero imaginary parts. The caller must ensure the spectrum is valid.
func (fp *FastPlanReal[F, C]) Inverse(dst []F, src []C) {
	half := fp.half
	buf := fp.buf

	// Reconstruct packed buffer from half-spectrum.
	fp.repackInverse(buf, src, fp.weight)

	// Inverse N/2 complex FFT
	fp.inner.Inverse(buf, buf)

	// Unpack complex buffer to real output
	dstAsComplex := unsafe.Slice((*C)(unsafe.Pointer(&dst[0])), half)
	copy(dstAsComplex, buf)
}

// realPart and imagPart extract the components of either complex type at
// full float64 precision; complexOf rebuilds a value of the target type.
// These are used outside the per-bin hot loops only (DC/Nyquist handling).
func realPart[C Complex](v C) float64 {
	switch c := any(v).(type) {
	case complex64:
		return float64(real(c))
	case complex128:
		return real(c)
	default:
		return 0
	}
}

func imagPart[C Complex](v C) float64 {
	switch c := any(v).(type) {
	case complex64:
		return float64(imag(c))
	case complex128:
		return imag(c)
	default:
		return 0
	}
}

func complexOf[C Complex](re, im float64) C {
	var zero C
	if _, ok := any(zero).(complex64); ok {
		return any(complex(float32(re), float32(im))).(C)
	}

	return any(complex(re, im)).(C)
}
