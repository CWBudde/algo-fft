package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
	m "github.com/cwbudde/algo-fft/internal/math"
)

// Rader runtime for prime-length plans (see internal/fft/rader.go for the
// table construction and the algorithm derivation). Both directions share the
// same structure:
//
//  1. Gather the nonzero-index inputs through the generator permutation while
//     accumulating the total sum (which is bin 0 of the result).
//  2. Run the length-(N-1) cyclic convolution against the precomputed
//     frequency-domain filter.
//  3. Scatter x[0] plus the convolution back through the inverse permutation.
//
// The convolution reuses the Bluestein plan fields: bluesteinM is N-1 here
// and the sub-FFT is exact, never padded.

func (p *Plan[T]) raderForward(dst, src, scratch, subScratch []T) error {
	l := p.n - 1
	x0 := src[0]
	sum := x0

	for q, j := range p.raderPermIn {
		v := src[j]
		scratch[q] = v
		sum += v
	}

	fft.BluesteinConvolution(
		scratch[:l], scratch[:l], p.bluesteinFilter,
		p.bluesteinTwiddle, subScratch[:l], p.bluesteinBitrev,
	)

	dst[0] = sum
	for i, k := range p.raderPermOut {
		dst[k] = x0 + scratch[i]
	}

	return nil
}

func (p *Plan[T]) raderInverse(dst, src, scratch, subScratch []T) error {
	l := p.n - 1
	x0 := src[0]
	sum := x0

	for q, j := range p.raderPermIn {
		v := src[j]
		scratch[q] = v
		sum += v
	}

	fft.BluesteinConvolution(
		scratch[:l], scratch[:l], p.bluesteinFilterInv,
		p.bluesteinTwiddle, subScratch[:l], p.bluesteinBitrev,
	)

	scale := m.ComplexFromFloat64[T](1.0/float64(p.n), 0)

	dst[0] = sum * scale
	for i, k := range p.raderPermOut {
		dst[k] = (x0 + scratch[i]) * scale
	}

	return nil
}
