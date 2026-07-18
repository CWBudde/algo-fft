package algofft

import (
	"math"

	"github.com/cwbudde/algo-fft/internal/cpu"
)

// This file implements the odd-length fallback for PlanRealT. The packed
// half-size method requires even n; odd lengths instead run a full-size
// complex FFT internally: the forward transform widens the real input into
// the complex buffer and keeps the non-redundant n/2+1 bins, the inverse
// rebuilds the full Hermitian spectrum from the half-spectrum before the
// complex inverse. This costs roughly 2× the packed method in memory and
// flops but keeps the real-FFT API available for every length the complex
// planner supports (including primes, via Bluestein/Rader).

func newPlanRealTOddWithFeatures[F Float, C Complex](
	n int, features cpu.Features, opts PlanOptions,
) (*PlanRealT[F, C], error) {
	childOpts := opts
	childOpts.Batch = 0
	childOpts.Stride = 0
	// The fallback runs the full-size complex plan in-place on the borrowed buffer.
	childOpts.InPlace = true

	plan, err := newPlanWithFeatures[C](n, features, childOpts)
	if err != nil {
		return nil, err
	}

	// No recombination weights: the fallback slices the full spectrum directly.
	return &PlanRealT[F, C]{
		n:       n,
		half:    n / 2,
		plan:    plan,
		buf:     newPlanRealTBufCache[C](n),
		options: opts,
	}, nil
}

// forwardOdd computes the real-to-complex FFT for odd n via the full-size
// complex plan. Lengths are validated by the caller (forwardSingle).
func (p *PlanRealT[F, C]) forwardOdd(dst []C, src []F) error {
	bufp := p.buf.get()
	defer p.buf.put(bufp)

	buf := *bufp

	// Widen real samples into the complex buffer (imaginary parts zero).
	var zero C
	switch any(zero).(type) {
	case complex64:
		srcF32 := any(src).([]float32)
		bufC64 := any(buf).([]complex64)

		for i, v := range srcF32 {
			bufC64[i] = complex(v, 0)
		}
	case complex128:
		srcF64 := any(src).([]float64)
		bufC128 := any(buf).([]complex128)

		for i, v := range srcF64 {
			bufC128[i] = complex(v, 0)
		}
	}

	err := p.plan.Forward(buf, buf)
	if err != nil {
		return err
	}

	// Keep the non-redundant half-spectrum: bins 0..n/2.
	copy(dst, buf[:p.half+1])

	return nil
}

// inverseOdd computes the complex-to-real inverse FFT for odd n via the
// full-size complex plan. Lengths are validated by the caller (inverseSingle).
func (p *PlanRealT[F, C]) inverseOdd(dst []F, src []C) error {
	// Odd lengths have no Nyquist bin; only DC must be (near-)real.
	var zero C
	switch any(zero).(type) {
	case complex64:
		srcC64 := any(src).([]complex64)
		if math.Abs(float64(imag(srcC64[0]))) > 1e-4 {
			return ErrInvalidSpectrum
		}
	case complex128:
		srcC128 := any(src).([]complex128)
		if math.Abs(imag(srcC128[0])) > 1e-12 {
			return ErrInvalidSpectrum
		}
	}

	bufp := p.buf.get()
	defer p.buf.put(bufp)

	buf := *bufp
	n := p.n

	// Rebuild the full Hermitian spectrum: X[n-k] = conj(X[k]) for k = 1..n/2.
	switch any(zero).(type) {
	case complex64:
		srcC64 := any(src).([]complex64)
		bufC64 := any(buf).([]complex64)
		copy(bufC64, srcC64)

		for k := 1; k <= p.half; k++ {
			v := srcC64[k]
			bufC64[n-k] = complex(real(v), -imag(v))
		}
	case complex128:
		srcC128 := any(src).([]complex128)
		bufC128 := any(buf).([]complex128)
		copy(bufC128, srcC128)

		for k := 1; k <= p.half; k++ {
			v := srcC128[k]
			bufC128[n-k] = complex(real(v), -imag(v))
		}
	}

	err := p.plan.Inverse(buf, buf)
	if err != nil {
		return err
	}

	// A Hermitian spectrum inverts to a real signal; drop the residual
	// imaginary parts (rounding noise only).
	switch any(zero).(type) {
	case complex64:
		bufC64 := any(buf).([]complex64)
		dstF32 := any(dst).([]float32)

		for i, v := range bufC64 {
			dstF32[i] = real(v)
		}
	case complex128:
		bufC128 := any(buf).([]complex128)
		dstF64 := any(dst).([]float64)

		for i, v := range bufC128 {
			dstF64[i] = real(v)
		}
	}

	return nil
}
