package fft

import (
	stdmath "math"
	"strconv"

	"github.com/cwbudde/algo-fft/internal/kernels"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// Rader's algorithm computes a prime-length DFT as a cyclic convolution of
// length p-1. With g a primitive root mod p, mapping the input through
// j = g^(-q) and the output through k = g^m turns
//
//	X[g^m] = x[0] + Σ_q x[g^(-q)] · W^(g^(m-q)),  W = exp(-2πi/p)
//
// into the length-(p-1) cyclic convolution of a[q] = x[g^(-q)] with
// b[r] = W^(g^r), evaluated via FFT against the precomputed frequency-domain
// filter FFT(b). Unlike Bluestein, the sub-FFT is exact (no padding to
// >= 2p-1), so an eligible prime costs one length-(p-1) convolution instead
// of a length-~4p one.

// RaderEligible reports whether a length-n transform should use Rader's
// algorithm: n must be prime with n-1 executable by the mixed-radix engine
// (2/3/5/7/11-smooth, so the length-(n-1) cyclic convolution runs directly
// through the power-of-two DIT or mixed-radix engines) and the sub-FFT must
// be a measured win over Bluestein's padded power-of-two sub-FFT (see
// raderConvolutionWins). Primes below 7 are excluded because they are
// themselves 5-smooth and never reach the arbitrary-length path.
func RaderEligible(n int) bool {
	l := n - 1

	return n >= 7 && mathpkg.IsMixedRadixSmooth(l) && raderConvolutionWins(l) && mathpkg.IsPrime(n)
}

// raderConvolutionWins reports whether an exact length-l sub-FFT beats
// Bluestein's padded power-of-two sub-FFT of length >= 2l+1. Although l < the
// pad always, the mixed-radix engine costs more per point than the
// power-of-two DIT kernels, so smaller does not always mean faster. Measured
// on AVX2/AVX-512 amd64 for both precisions (BenchmarkRaderVsBluestein):
//
//   - power-of-two l wins 4-5x at every size (17, 257, 65537);
//   - other 5-smooth l wins 1.1-5.6x when its power-of-two part is >= 8 —
//     the mixed-radix schedule then ends in a tuned codelet leaf ([3, 32]
//     for 96, [3, 3, 128] for 1152) so the engine penalty stays small
//     (97, 401, 641, 769, 1153, 1601, 3001, 4001, 12289, 18433, 40961);
//   - shapes whose power-of-two part is <= 4 keep the odd combine stages
//     dominant and measured as losses (31, 61, 101, 151, 251), as did tiny
//     l (7..41) where fixed overheads dominate. Those stay on Bluestein.
//
// l bearing a factor 7 or 11 has its own shape rule, see rader7Or11Wins.
func raderConvolutionWins(l int) bool {
	if mathpkg.IsPowerOf2(l) {
		return true
	}

	if !mathpkg.IsHighlyComposite(l) {
		return rader7Or11Wins(l)
	}

	return l >= 96 && l&-l >= 8
}

// rader7Or11Wins is the win gate for sub-FFT lengths that need a radix-7 or
// radix-11 stage. Those stages are full-matrix DFT butterflies (see
// internal/kernels/radix{7,11}.go), so the schedule's odd tail weighs much
// more than it does for the 3/5 stages the 5-smooth rule above was fitted on.
// Measured with BenchmarkRader7And11VsBluestein (i7-1255U, AVX2, forward,
// both precisions; pow2 is the power-of-two part of l and o = l/pow2):
//
//   - l >= 2048: Bluestein's pad (>= 2l+1, measured 2.0-3.9x l here) lands on
//     a large power-of-two sub-FFT, so the exact sub-FFT wins at every shape
//     with pow2 >= 4 (2112, 2268, 2688, 2800, 4200, 4480, 6336, 7056, 7392,
//     9856, 9900, 12096, 12600, 14080, 15120, 30240: 1.1-3.4x). pow2 == 2
//     stays a wash or a loss (2310: 0.96/1.01x, 22050: 0.87/1.03x) — the
//     strided radix-2 tail cancels the smaller length, the same pattern the
//     5-smooth rule and planner.mixedRadix7And11Wins both show.
//   - l < 2048: the odd stages dominate the whole transform, and only a
//     single radix-7/11 stage (optionally with one radix-3) on top of a deep
//     power-of-two chain wins at both precisions: 112, 352, 448, 672, 1408
//     at 1.1-2.0x. Every shallower or odd-heavier shape measured 0.34-1.17x,
//     i.e. no win that holds across precisions (88 pow2 8; 196, 700 pow2 4;
//     126, 462 pow2 2; 280 pow2 8; 880 o=55; 1320 pow2 8; 2016 o=63), and
//     stays on Bluestein.
func rader7Or11Wins(l int) bool {
	pow2 := l & -l

	if l >= 2048 {
		return pow2 >= 4
	}

	return pow2 >= 16 && l/pow2 <= 33
}

// ComputeRaderTables precomputes the plan-time tables for a Rader transform
// of prime length p: the input gather permutation (permIn[q] = g^(-q) mod p),
// the output scatter permutation (permOut[m] = g^m mod p), the forward and
// inverse frequency-domain filters (FFT of W^(g^r) and its conjugate), the
// sub-FFT twiddles for length p-1, and the bit-reversal table when p-1 is a
// power of two. scratch must have length >= p-1.
//
//nolint:nonamedreturns // six related tables; names document the tuple
func ComputeRaderTables[T Complex](p int, scratch []T) (
	permIn, permOut []int, filter, filterInv, twiddle []T, bitrev []int,
) {
	l := p - 1
	g := mathpkg.PrimitiveRoot(p)

	permOut = make([]int, l)
	permOut[0] = 1

	for i := 1; i < l; i++ {
		permOut[i] = int(mathpkg.MulMod(uint64(permOut[i-1]), uint64(g), uint64(p)))
	}

	permIn = make([]int, l)
	permIn[0] = 1

	for q := 1; q < l; q++ {
		permIn[q] = permOut[l-q]
	}

	// b[r] = W^(g^r) with W = exp(-2πi/p); the inverse transform uses the
	// conjugate sequence (W^(-g^r)). Both are FFT'd in place below.
	filter = make([]T, l)
	filterInv = make([]T, l)

	invP := 1.0 / float64(p)
	for r := range l {
		angle := -2 * stdmath.Pi * float64(permOut[r]) * invP
		re, im := stdmath.Cos(angle), stdmath.Sin(angle)
		filter[r] = mathpkg.ComplexFromFloat64[T](re, im)
		filterInv[r] = mathpkg.ComplexFromFloat64[T](re, -im)
	}

	twiddle = mathpkg.ComputeTwiddleFactors[T](l)

	if mathpkg.IsPowerOf2(l) {
		bitrev = mathpkg.ComputeBitReversalIndices(l)
	}

	raderFilterFFT(filter, twiddle, scratch[:l])
	raderFilterFFT(filterInv, twiddle, scratch[:l])

	return permIn, permOut, filter, filterInv, twiddle, bitrev
}

// raderFilterFFT runs the plan-time forward FFT of a Rader filter sequence in
// place, dispatching on the sub-FFT length exactly like the runtime
// convolution (BluesteinConvolution): power-of-two lengths through the DIT
// driver, other smooth lengths through the mixed-radix engine.
func raderFilterFFT[T Complex](buf, twiddle, scratch []T) {
	if mathpkg.IsPowerOf2(len(buf)) {
		if !kernels.DITForward(buf, buf, twiddle, scratch) {
			panic("algofft: DIT driver rejected Rader filter FFT size " +
				strconv.Itoa(len(buf)) + " (planner/engine contract violation)")
		}

		return
	}

	mustMixedRadix(mixedRadixForward(buf, buf, twiddle, scratch), len(buf))
}
