package algofft

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fft"
	"github.com/cwbudde/algo-fft/internal/transform"
)

// recursiveForward computes the forward FFT using recursive decomposition.
func (p *Plan[T]) recursiveForward(dst, src, scratch []T) error {
	if p.decompStrategy == nil {
		return ErrNotImplemented
	}

	features := cpu.DetectFeatures()

	// Determine registry based on precision
	var zero T
	switch any(zero).(type) {
	case complex64:
		registry := fft.Registry64
		src64 := any(src).([]complex64)
		dst64 := any(dst).([]complex64)
		twiddle64 := any(p.twiddle).([]complex64)
		scratch64 := any(scratch).([]complex64)

		transform.RecursiveForward(dst64, src64, p.decompStrategy, twiddle64, scratch64, registry, features)

		return nil
	case complex128:
		registry := fft.Registry128
		src128 := any(src).([]complex128)
		dst128 := any(dst).([]complex128)
		twiddle128 := any(p.twiddle).([]complex128)
		scratch128 := any(scratch).([]complex128)

		transform.RecursiveForward(dst128, src128, p.decompStrategy, twiddle128, scratch128, registry, features)

		return nil
	default:
		return ErrNotImplemented
	}
}

// recursiveInverse computes the inverse FFT using recursive decomposition.
func (p *Plan[T]) recursiveInverse(dst, src, scratch []T) error {
	if p.decompStrategy == nil {
		return ErrNotImplemented
	}

	features := cpu.DetectFeatures()

	// Determine registry based on precision
	var zero T
	switch any(zero).(type) {
	case complex64:
		registry := fft.Registry64
		src64 := any(src).([]complex64)
		dst64 := any(dst).([]complex64)
		twiddle64 := any(p.twiddle).([]complex64)
		scratch64 := any(scratch).([]complex64)

		transform.RecursiveInverse(dst64, src64, p.decompStrategy, twiddle64, scratch64, registry, features)

		return nil
	case complex128:
		registry := fft.Registry128
		src128 := any(src).([]complex128)
		dst128 := any(dst).([]complex128)
		twiddle128 := any(p.twiddle).([]complex128)
		scratch128 := any(scratch).([]complex128)

		transform.RecursiveInverse(dst128, src128, p.decompStrategy, twiddle128, scratch128, registry, features)

		return nil
	default:
		return ErrNotImplemented
	}
}
