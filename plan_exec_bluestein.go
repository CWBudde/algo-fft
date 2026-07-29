package algofft

import (
	"github.com/cwbudde/algo-fft/internal/fft"
)

// bluesteinExecutor runs arbitrary-length transforms via Bluestein's
// algorithm (Chirp-Z transform): modulate by the chirp, run a padded cyclic
// convolution against the precomputed frequency-domain filter, and demodulate.
//
// Both modulation steps go through fft.ComplexMulArray rather than a scalar
// loop: it dispatches to the SIMD element-wise product, and its pure-Go
// fallback multiplies complex64 in single precision. A scalar
// `src[i] * e.chirp[i]` would instead widen each element to complex128 and
// round back (see math.MulComplex64) — one of the reasons complex64 measured
// slower than complex128 on this path (PLAN.md P5.0).
type bluesteinExecutor[T Complex] struct {
	n int // Logical transform length
	m int // Padded sub-FFT size M >= 2N-1 (power of two or mixed-radix; see bluesteinPadSize)

	chirp     []T // Size N
	chirpInv  []T // Size N
	filter    []T // Size M
	filterInv []T // Size M
	twiddle   []T // Size M (sub-FFT twiddles)

	// bitrev feeds only the *unbound* power-of-two DIT sub-FFT path; nil for
	// the mixed-radix padded sizes, which run through the mixed-radix engine.
	bitrev []int

	// sub is the plan-time-bound padded sub-FFT. It is what gives the
	// power-of-two padded sizes access to the codelet registry — the unbound
	// route enters a hardcoded size switch in internal/kernels that never
	// consults it, which is why those lengths got no SIMD at all and measured
	// slower on the default build than under -tags purego (PLAN.md P3).
	// nil selects the unbound route.
	sub *fft.BluesteinSubFFT[T]

	// subTwiddle*Backing retain the aligned backing arrays of the bound
	// kernel's prepared twiddle layouts, mirroring kernelExecutor.
	subTwiddleForwardBacking []byte
	subTwiddleInverseBacking []byte
}

func (e *bluesteinExecutor[T]) forward(dst, src, scratch, sub []T) {
	fft.ComplexMulArray(scratch[:e.n], src[:e.n], e.chirp)

	var zero T
	for i := e.n; i < e.m; i++ {
		scratch[i] = zero
	}

	fft.BluesteinConvolution(scratch, scratch, e.filter, e.twiddle, sub, e.bitrev, e.sub)

	fft.ComplexMulArray(dst[:e.n], scratch[:e.n], e.chirp)
}

func (e *bluesteinExecutor[T]) inverse(dst, src, scratch, sub []T) {
	fft.ComplexMulArray(scratch[:e.n], src[:e.n], e.chirpInv)

	var zero T
	for i := e.n; i < e.m; i++ {
		scratch[i] = zero
	}

	fft.BluesteinConvolution(scratch, scratch, e.filterInv, e.twiddle, sub, e.bitrev, e.sub)

	fft.ComplexMulArray(dst[:e.n], scratch[:e.n], e.chirpInv)
	fft.ScaleInPlace(dst[:e.n], 1.0/float64(e.n))
}
