package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
)

func (p *Plan[T]) bluesteinForward(dst, src, scratch, bluesteinScratch []T) error {
	for i := range p.n {
		scratch[i] = src[i] * p.bluesteinChirp[i]
	}

	var zero T
	for i := p.n; i < p.bluesteinM; i++ {
		scratch[i] = zero
	}

	fft.BluesteinConvolution(
		scratch, scratch, p.bluesteinFilter,
		p.bluesteinTwiddle, bluesteinScratch,
	)

	for i := range p.n {
		dst[i] = scratch[i] * p.bluesteinChirp[i]
	}

	return nil
}

func (p *Plan[T]) bluesteinInverse(dst, src, scratch, bluesteinScratch []T) error {
	for i := range p.n {
		scratch[i] = src[i] * p.bluesteinChirpInv[i]
	}

	var zero T
	for i := p.n; i < p.bluesteinM; i++ {
		scratch[i] = zero
	}

	fft.BluesteinConvolution(
		scratch, scratch, p.bluesteinFilterInv,
		p.bluesteinTwiddle, bluesteinScratch,
	)

	var scale T

	switch any(zero).(type) {
	case complex64:
		scale = any(complex(float32(1.0/float64(p.n)), 0)).(T)
	case complex128:
		scale = any(complex(1.0/float64(p.n), 0)).(T)
	}

	for i := range p.n {
		dst[i] = scratch[i] * p.bluesteinChirpInv[i] * scale
	}

	return nil
}
