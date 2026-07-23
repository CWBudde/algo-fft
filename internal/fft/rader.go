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
// algorithm: n must be prime with n-1 5-smooth (so the length-(n-1) cyclic
// convolution runs directly through the power-of-two DIT or mixed-radix
// engines) and the sub-FFT must be a measured win over Bluestein's padded
// power-of-two sub-FFT (see raderConvolutionWins). Primes below 7 are
// excluded because they are themselves 5-smooth and never reach the
// arbitrary-length path.
func RaderEligible(n int) bool {
	l := n - 1

	return n >= 7 && mathpkg.IsHighlyComposite(l) && raderConvolutionWins(l) && mathpkg.IsPrime(n)
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
func raderConvolutionWins(l int) bool {
	if mathpkg.IsPowerOf2(l) {
		return true
	}

	return l >= 96 && l&-l >= 8
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
// driver, other 5-smooth lengths through the mixed-radix engine.
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
