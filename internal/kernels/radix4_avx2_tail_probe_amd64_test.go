//go:build amd64 && !purego && fftprobe

package kernels

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// TestRadix4AVX2NoTailProbeIsStagesOnly is what makes the no-tail probe usable
// as a measurement.
//
// The probe's claim is that passing r4End = n runs exactly the radix-4 stages
// the real kernel runs and omits only the radix-2 tail. Deriving that from the
// stage-loop bounds would be an argument; this checks it. Applying the missing
// combine to the probe's output in Go must reproduce the real kernel's output.
//
// The tolerance is not bit-exactness: the kernel fuses one of the four products
// of each complex multiply into a VFMADDSUB, which rounds once where the Go
// expression below rounds twice. It is tight enough that a missing, extra or
// differently ordered stage cannot pass -- those are O(1) wrong, not O(eps).
func TestRadix4AVX2NoTailProbeIsStagesOnly(t *testing.T) {
	for _, n := range radix4TailSizes {
		for _, inverse := range []bool{false, true} {
			name := "forward"
			if inverse {
				name = "inverse"
			}

			t.Run(name+"/"+itoa(n), func(t *testing.T) {
				checkNoTailProbe128(t, n, inverse)
				checkNoTailProbe64(t, n, inverse)
			})
		}
	}
}

// tailTwiddleOffset returns the index at which prepareTwiddleRadix4AVX2 starts
// writing the radix-2 tail's n/2 twiddles, derived the way the writer derives
// it rather than as a constant.
func tailTwiddleOffset(n int) int {
	limit, ok := radix4AVX2Limit(n)
	if !ok || limit == n {
		return -1
	}

	offset := 0
	for stage := 4; stage*4 <= limit; stage *= 4 {
		offset += 3 * stage
	}

	return offset
}

func checkNoTailProbe128(t *testing.T, n int, inverse bool) {
	t.Helper()

	rng := rand.New(rand.NewSource(7)) //nolint:gosec // deterministic test input
	src := make([]complex128, n)

	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	twiddle := make([]complex128, twiddleSizeRadix4AVX2Complex128(n))
	prepareTwiddleRadix4AVX2Complex128(n, inverse, twiddle)

	want := make([]complex128, n)
	got := make([]complex128, n)
	scratch := make([]complex128, n)

	real128, probe128 := forwardRadix4AVX2Complex128, forwardRadix4AVX2NoTailComplex128
	if inverse {
		real128, probe128 = inverseRadix4AVX2Complex128, inverseRadix4AVX2NoTailComplex128
	}

	if !real128(want, src, twiddle, scratch) || !probe128(got, src, twiddle, scratch) {
		t.Fatalf("kernel declined n=%d", n)
	}

	half := n / 2
	base := tailTwiddleOffset(n)

	for j := range half {
		a0, a1 := got[j], twiddle[base+j]*got[j+half]
		got[j], got[j+half] = a0+a1, a0-a1
	}

	var scale float64
	for _, v := range want {
		scale = math.Max(scale, cmplx.Abs(v))
	}

	const tol = 1e-12

	for i := range want {
		if d := cmplx.Abs(want[i] - got[i]); d > tol*scale {
			t.Fatalf("n=%d complex128: index %d differs by %g (scale %g)", n, i, d, scale)
		}
	}
}

func checkNoTailProbe64(t *testing.T, n int, inverse bool) {
	t.Helper()

	rng := rand.New(rand.NewSource(7)) //nolint:gosec // deterministic test input
	src := make([]complex64, n)

	for i := range src {
		src[i] = complex(float32(rng.Float64()*2-1), float32(rng.Float64()*2-1))
	}

	twiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
	prepareTwiddleRadix4AVX2(n, inverse, twiddle)

	want := make([]complex64, n)
	got := make([]complex64, n)
	scratch := make([]complex64, n)

	real64, probe64 := forwardRadix4AVX2Complex64, forwardRadix4AVX2NoTailComplex64
	if inverse {
		real64, probe64 = inverseRadix4AVX2Complex64, inverseRadix4AVX2NoTailComplex64
	}

	if !real64(want, src, twiddle, scratch) || !probe64(got, src, twiddle, scratch) {
		t.Fatalf("kernel declined n=%d", n)
	}

	half := n / 2
	base := tailTwiddleOffset(n)

	for j := range half {
		a0, a1 := got[j], twiddle[base+j]*got[j+half]
		got[j], got[j+half] = a0+a1, a0-a1
	}

	var scale float64
	for _, v := range want {
		scale = math.Max(scale, cmplx.Abs(complex128(v)))
	}

	const tol = 1e-4

	for i := range want {
		if d := cmplx.Abs(complex128(want[i]) - complex128(got[i])); d > tol*scale {
			t.Fatalf("n=%d complex64: index %d differs by %g (scale %g)", n, i, d, scale)
		}
	}
}
