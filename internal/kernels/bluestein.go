package kernels

import (
	"math"

	m "github.com/MeKo-Christian/algo-fft/internal/math"
)

// Helper wrappers from internal/math.
func complexFromFloat64[T Complex](re, im float64) T {
	return m.ComplexFromFloat64[T](re, im)
}

func conj[T Complex](val T) T {
	return m.Conj[T](val)
}

// ComputeChirpSequence computes the chirp sequence W_n^(k^2/2) = exp(-j * pi * k^2 / n)
// The sequence is of length n.
func ComputeChirpSequence[T Complex](n int) []T {
	chirp := make([]T, n)

	invN := 1.0 / float64(n)
	for k := range n {
		angle := -math.Pi * float64(k*k) * invN
		re := math.Cos(angle)
		im := math.Sin(angle)
		chirp[k] = complexFromFloat64[T](re, im)
	}

	return chirp
}

// ComputeBluesteinFilter computes the frequency-domain filter for Bluestein's algorithm.
// n is the original size, m is the padded size (power of 2 >= 2n-1).
// chirp is the sequence of length n computed by ComputeChirpSequence.
// twiddles are for size m.
// scratch is a pre-allocated buffer of size m for intermediate computations.
func ComputeBluesteinFilter[T Complex](n, m int, chirp []T, twiddles []T, scratch []T) []T {
	b := make([]T, m)
	// Construct b sequence
	// b_k = w_k^{-1} = conj(w_k)
	// b[k] = conj(chirp[k]) for 0 <= k < n
	// b[m-k] = conj(chirp[k]) for 1 <= k < n

	b[0] = conj(chirp[0])
	for k := 1; k < n; k++ {
		val := conj(chirp[k])
		b[k] = val
		b[m-k] = val
	}

	// Perform FFT using provided scratch buffer
	ditForward(b, b, twiddles, scratch)

	return b
}

// BluesteinConvolution performs the convolution y = x * b via FFT.
// dst is the output buffer of size m.
// x is the input sequence of size m (padded with zeros).
// filter is the frequency-domain filter (FFT of b) of size m.
// twiddles are for size m.
// scratch is a scratch buffer of size m.
func BluesteinConvolution[T Complex](dst, x, filter, twiddles, scratch []T) {
	// 1. FFT of x
	// We use dst as the work buffer. If dst != x, ditForward handles the copy/transform.
	ditForward(dst, x, twiddles, scratch)

	// 2. Multiply by filter
	for i := range dst {
		dst[i] *= filter[i]
	}

	// 3. IFFT
	ditInverse(dst, dst, twiddles, scratch)
}
