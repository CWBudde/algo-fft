package fft

import (
	"strconv"

	"github.com/cwbudde/algo-fft/internal/kernels"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// BluesteinSubFFT is a plan-time binding of the padded power-of-two sub-FFT.
//
// It exists because the unbound route (kernels.BluesteinConvolution ->
// bluesteinSubForward -> the hardcoded size switch in internal/kernels/dit.go)
// never consults the codelet registry, so the whole sub-FFT — which is ~96% of
// a Bluestein transform's work — ran in pure Go on every build. AVX2 codelets
// for exactly the padded sizes this path produces (2048, 4096, ...) were
// registered and unreachable, which is why n = 1009 and n = 2003 measured
// *slower* on the SIMD build than under -tags purego (PLAN.md P3).
//
// Binding at plan time is what makes the registry reachable: a codelet may
// require a prepared twiddle layout, and preparing one is plan-time work the
// kernel-level entry points have no way to do.
//
// A nil *BluesteinSubFFT selects the unbound route, which stays as the
// fallback for sizes no kernel claims.
type BluesteinSubFFT[T Complex] struct {
	Forward kernels.Kernel[T]
	Inverse kernels.Kernel[T]

	// TwiddleForward/TwiddleInverse are the (possibly codelet-specific)
	// twiddle tables prepared for the padded size. They alias the plain table
	// when the bound kernel declares no custom layout.
	TwiddleForward []T
	TwiddleInverse []T
}

// forward runs the bound forward sub-FFT. A bail is a contract violation
// rather than a fallback opportunity: the caller passes dst aliasing x, so a
// kernel that bailed part-way has already destroyed the input it would need to
// be re-run from. Plan construction verifies the binding end-to-end before
// installing it (see verifyBluesteinSub), so reaching this panic means the
// kernel's accept/reject decision is not a function of the sizes alone.
func (s *BluesteinSubFFT[T]) forward(dst, src, scratch []T) {
	if !s.Forward(dst, src, s.TwiddleForward, scratch) {
		panic("algofft: bound Bluestein sub-FFT kernel rejected size " +
			strconv.Itoa(len(src)) + " (planner/kernel contract violation)")
	}
}

// inverse is the inverse counterpart of forward; see there for why a bail panics.
func (s *BluesteinSubFFT[T]) inverse(dst, src, scratch []T) {
	if !s.Inverse(dst, src, s.TwiddleInverse, scratch) {
		panic("algofft: bound Bluestein sub-FFT kernel rejected size " +
			strconv.Itoa(len(src)) + " (planner/kernel contract violation)")
	}
}

// ComputeBluesteinFilter computes the frequency-domain Bluestein filter for
// padded size m. Power-of-two m runs the bound sub-FFT when one is supplied and
// the radix-2 DIT sub-FFT otherwise; any other m the pad chooser selects runs
// through the mixed-radix engine.
func ComputeBluesteinFilter[T Complex](n, m int, chirp, twiddles, scratch []T, sub *BluesteinSubFFT[T]) []T {
	if mathpkg.IsPowerOf2(m) {
		if sub == nil {
			return kernels.ComputeBluesteinFilter[T](n, m, chirp, twiddles, scratch)
		}

		b := kernels.BuildBluesteinSequence(n, m, chirp)
		sub.forward(b, b, scratch)

		return b
	}

	b := kernels.BuildBluesteinSequence(n, m, chirp)
	mustMixedRadix(mixedRadixForward(b, b, twiddles, scratch), m)

	return b
}

// BluesteinConvolution performs the cyclic convolution y = x * b via a padded
// sub-FFT of size m = len(filter). Power-of-two m uses the plan-bound kernels
// when sub is non-nil, else the radix-2 DIT kernels with the precomputed bitrev
// table; any other m dispatches to the mixed-radix engine (which ignores
// bitrev).
func BluesteinConvolution[T Complex](dst, x, filter, twiddles, scratch []T, bitrev []int, sub *BluesteinSubFFT[T]) {
	m := len(filter)
	if mathpkg.IsPowerOf2(m) && sub == nil {
		kernels.BluesteinConvolution[T](dst, x, filter, twiddles, scratch, bitrev)
		return
	}

	if sub != nil {
		sub.forward(dst, x, scratch)
	} else {
		mustMixedRadix(mixedRadixForward(dst, x, twiddles, scratch), m)
	}

	// The SIMD element-wise product, rather than the scalar loop the unbound
	// power-of-two route uses (internal/kernels cannot import internal/fft, so
	// the dispatch cannot go the other way there).
	ComplexMulArrayInPlace(dst[:m], filter)

	if sub != nil {
		sub.inverse(dst, dst, scratch)
		return
	}

	mustMixedRadix(mixedRadixInverse(dst, dst, twiddles, scratch), m)
}

// VerifyBluesteinSub reports whether a candidate binding actually runs at the
// padded size, in both directions. Plan construction calls it before installing
// the binding so that a kernel which declines the size (or its prepared twiddle
// layout) degrades to the unbound route at plan time instead of panicking mid
// transform. It writes only into the caller's scratch buffers.
func VerifyBluesteinSub[T Complex](sub *BluesteinSubFFT[T], m int, probe, scratch []T) bool {
	if sub == nil || sub.Forward == nil || sub.Inverse == nil {
		return false
	}

	if len(probe) < m || len(scratch) < m {
		return false
	}

	var zero T
	for i := range probe[:m] {
		probe[i] = zero
	}

	return sub.Forward(probe[:m], probe[:m], sub.TwiddleForward, scratch) &&
		sub.Inverse(probe[:m], probe[:m], sub.TwiddleInverse, scratch)
}

// mustMixedRadix panics when the mixed-radix engine rejects a Bluestein
// sub-FFT size. Plan construction only selects padded sizes the engine
// schedules unconditionally (see padShapes), so a failure here is a
// planner/engine contract violation — returning would leave dst partially
// written and surface as a silent wrong answer.
func mustMixedRadix(ok bool, m int) {
	if !ok {
		panic("algofft: mixed-radix engine rejected Bluestein sub-FFT size " + strconv.Itoa(m) +
			" (planner/engine contract violation)")
	}
}
