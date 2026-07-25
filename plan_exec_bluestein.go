package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
	m "github.com/cwbudde/algo-fft/internal/math"
)

// bluesteinExecutor runs arbitrary-length transforms via Bluestein's
// algorithm (Chirp-Z transform): modulate by the chirp, run a padded cyclic
// convolution against the precomputed frequency-domain filter, and demodulate.
type bluesteinExecutor[T Complex] struct {
	n int // Logical transform length
	m int // Padded sub-FFT size M >= 2N-1 (power of two or mixed-radix; see bluesteinPadSize)

	chirp     []T // Size N
	chirpInv  []T // Size N
	filter    []T // Size M
	filterInv []T // Size M
	twiddle   []T // Size M (sub-FFT twiddles)

	// bitrev feeds only the power-of-two DIT sub-FFT path; nil for the
	// mixed-radix padded sizes, which run through the mixed-radix engine.
	bitrev []int
}

func (e *bluesteinExecutor[T]) forward(dst, src, scratch, sub []T) {
	for i := range e.n {
		scratch[i] = src[i] * e.chirp[i]
	}

	var zero T
	for i := e.n; i < e.m; i++ {
		scratch[i] = zero
	}

	fft.BluesteinConvolution(scratch, scratch, e.filter, e.twiddle, sub, e.bitrev)

	for i := range e.n {
		dst[i] = scratch[i] * e.chirp[i]
	}
}

func (e *bluesteinExecutor[T]) inverse(dst, src, scratch, sub []T) {
	for i := range e.n {
		scratch[i] = src[i] * e.chirpInv[i]
	}

	var zero T
	for i := e.n; i < e.m; i++ {
		scratch[i] = zero
	}

	fft.BluesteinConvolution(scratch, scratch, e.filterInv, e.twiddle, sub, e.bitrev)

	scale := m.ComplexFromFloat64[T](1.0/float64(e.n), 0)

	for i := range e.n {
		dst[i] = scratch[i] * e.chirpInv[i] * scale
	}
}
