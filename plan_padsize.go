package algofft

import (
	m "github.com/cwbudde/algo-fft/internal/math"
)

// This file owns the padded-length model shared by Bluestein sub-FFT sizing
// (bluesteinPadSize) and one-shot convolution (fastConvolutionLength).
//
// Both callers may use any length at or above a required minimum, so the
// question is which of those lengths the engine executes fastest. The previous
// answer was "always the next power of two", enforced by a single scalar
// penalty applied to every mixed-radix candidate. One constant cannot be right
// for all of them: measured against the power-of-two endpoint of its own dyadic
// window, a mixed-radix sub-FFT's cost per m*log2(m) point-pass spans about 7x
// on shape alone (BenchmarkBluesteinPadShapes, i7-1255U/AVX2, complex64):
//
//	3072 = 2^10*3   0.83      2160 = 2^4*3^3*5   2.31
//	2560 = 2^9*5    0.96      3000 = 2^3*3*5^3   2.87
//	3584 = 2^9*7    1.39      2250 = 2*3^2*5^3   6.18
//
// A deep power-of-two part lands the mixed-radix schedule in a tuned codelet
// leaf and each surviving odd stage is overhead on top of it, so the cheap
// candidates are the ones whose odd part is a single small factor. Hence the
// model is a whitelist of candidate shapes rather than a continuous cost
// function: a shape is admitted only above the pad size where it was measured
// to win at *both* precisions.

// padShape is one candidate family. Within a window ending at the power of two
// P = 2^k, the family contributes the single candidate odd*2^j that is largest
// below P; every smaller multiple of odd falls below P/2 and therefore below
// the minimum any caller can ask for.
type padShape struct {
	odd int // odd part of the candidate

	// minPow2 is the smallest power-of-two pad this shape is measured to beat.
	// Below it the power of two is kept.
	minPow2 int
}

// padShapes lists the admitted candidate families in measured preference order.
//
// Calibrated with BenchmarkBluesteinPadFamilies (internal/fft) on an i7-1255U
// (AVX2), both precisions, ten windows from 2^7 to 2^16, as candidate ns/op
// over the window's power-of-two endpoint (c64/c128; < 1 is a win):
//
//	P      3*2^(k-2)     15*2^(k-4)    7*2^(k-3)
//	2^7    1.19 / 1.36   3.10 / 3.51   2.45 / 2.65
//	2^8    0.78 / 1.00   2.75 / 2.05   2.41 / 1.71
//	2^9    0.71 / 0.87   1.54 / 1.79   1.55 / 1.77
//	2^10   0.72 / 0.71   1.59 / 1.52   1.64 / 1.51
//	2^11   0.74 / 0.85   1.71 / 1.84   1.60 / 1.74
//	2^12   0.62 / 0.49   1.13 / 0.87   1.20 / 0.98
//	2^13   0.43 / 0.79   0.80 / 0.69   0.90 / 2.37
//	2^14   0.44 / 0.46   0.82 / 0.73   0.89 / 0.87
//	2^15   0.40 / 0.43   0.75 / 0.79   0.81 / 0.80
//	2^16   0.41 / 0.46   0.74 / 0.75   0.82 / 0.79
//
// 3*2^(k-2) turns over at 2^9 (2^8 is a wash for complex128) and 15*2^(k-4) at
// 2^13 (2^12 still loses 13% on complex64), which are the two thresholds below.
//
// The third measured family, 7*2^(k-3), is admitted by no threshold: it loses
// to 15*2^(k-4) in every window where either wins — a full-matrix radix-7
// butterfly costs more than the radix-3 plus radix-5 pair — and being the
// *smaller* of the two it is reachable only when 15*2^(k-4) is reachable as
// well. It is therefore dominated outright and left out.
//
// Shapes with several odd stages (2^a*3^3*5, 2^a*3*5^3, ...) were measured in
// BenchmarkBluesteinPadShapes and are all worse than 3*2^(k-2) wherever both
// are reachable, so they are not candidates either.
//
//nolint:gochecknoglobals // static lookup table
var padShapes = []padShape{
	{odd: 3, minPow2: 1 << 9},
	{odd: 15, minPow2: 1 << 13},
}

// cheapestPaddedLength returns the cheapest FFT length >= minM the engine can
// execute exactly. Candidates are the padShapes families, in preference order;
// the next power of two is the fallback and the incumbent every candidate has
// to beat. The choice is a pure function of minM.
func cheapestPaddedLength(minM int) int {
	pow2 := m.NextPowerOfTwo(minM)

	for _, shape := range padShapes {
		if pow2 < shape.minPow2 {
			continue
		}

		if cand := largestMultipleBelow(shape.odd, pow2); cand >= minM {
			return cand
		}
	}

	return pow2
}

// largestMultipleBelow returns odd*2^j, the largest power-of-two multiple of
// odd that stays strictly below limit (itself a power of two), or 0 when odd is
// not below limit. Doubling stops before the product can reach limit, so
// nothing overflows.
func largestMultipleBelow(odd, limit int) int {
	if odd >= limit {
		return 0
	}

	v := odd
	for v <= (limit-1)/2 {
		v *= 2
	}

	return v
}

// bluesteinPadSize returns the padded sub-FFT length for a Bluestein plan of
// logical length n: the cyclic convolution needs any length m >= 2n-1, costed
// via the shared pad model (see cheapestPaddedLength).
func bluesteinPadSize(n int) int {
	return cheapestPaddedLength(2*n - 1)
}
