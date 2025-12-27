package algofft

import (
	"github.com/MeKo-Christian/algofft/internal/fft"
)

func (p *Plan[T]) bluesteinForward(dst, src []T) error {
	for i := range p.n {
		p.scratch[i] = src[i] * p.bluesteinChirp[i]
	}

	var zero T
	for i := p.n; i < p.bluesteinM; i++ {
		p.scratch[i] = zero
	}

	fft.BluesteinConvolution(
		p.scratch, p.scratch, p.bluesteinFilter,
		p.bluesteinTwiddle, p.bluesteinScratch, p.bluesteinBitrev,
	)

	for i := range p.n {
		dst[i] = p.scratch[i] * p.bluesteinChirp[i]
	}

	return nil
}

func (p *Plan[T]) bluesteinInverse(dst, src []T) error {
	for i := range p.n {
		p.scratch[i] = src[i] * p.bluesteinChirpInv[i]
	}

	var zero T
	for i := p.n; i < p.bluesteinM; i++ {
		p.scratch[i] = zero
	}

	fft.BluesteinConvolution(
		p.scratch, p.scratch, p.bluesteinFilterInv,
		p.bluesteinTwiddle, p.bluesteinScratch, p.bluesteinBitrev,
	)

	var scale T

	switch any(zero).(type) {
	case complex64:
		scale = any(complex(float32(1.0/float64(p.n)), 0)).(T)
	case complex128:
		scale = any(complex(1.0/float64(p.n), 0)).(T)
	}

	for i := range p.n {
		dst[i] = p.scratch[i] * p.bluesteinChirpInv[i] * scale
	}

	return nil
}
