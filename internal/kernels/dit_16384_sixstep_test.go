package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// The n = 16384 six-step codelet's direct tests.
//
// Its 4096 and 8192 siblings have had these since they were written; this size
// never did, and the gap only became visible on 2026-08-02 when the three rows
// moved behind `-tags fftprobe` (sixstep_codelet_probe.go). Without a spec row
// the registry-driven sweep no longer reaches the kernel in an ordinary build,
// and with no test either, `unused` reported both functions dead — correctly.
//
// That is the standing lesson in reverse: a registered fast path is not a
// reachable one, and a kernel reached only through the registry stops being
// tested the moment its row goes away. A probe-gated kernel needs a test that
// does not depend on the tag.
//
// Reference is the radix-4 codelet at the same size, which is what six-step
// competes with and loses to; the round-trip is checked separately so a
// consistent-but-wrong pair of transforms cannot pass.

func TestForwardDIT16384SixStep_Complex64(t *testing.T) {
	const n = 16384

	src := make([]complex64, n)
	dstSixStep := make([]complex64, n)
	dstRadix4 := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic test input
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	if !forwardDIT16384SixStepComplex64(dstSixStep, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384SixStepComplex64 returned false")
	}

	if !forwardDIT16384Radix4Complex64(dstRadix4, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384Radix4Complex64 returned false")
	}

	maxErr := float32(0)

	for i := range n {
		re := real(dstSixStep[i]) - real(dstRadix4[i])
		im := imag(dstSixStep[i]) - imag(dstRadix4[i])

		if err := float32(math.Sqrt(float64(re*re + im*im))); err > maxErr {
			maxErr = err
		}
	}

	t.Logf("Max error between six-step and radix-4: %e", maxErr)

	const tolerance = 1e-3
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func TestRoundTrip16384SixStep_Complex64(t *testing.T) {
	const n = 16384

	src := make([]complex64, n)
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	scratch := make([]complex64, n)

	rng := rand.New(rand.NewSource(7)) //nolint:gosec // deterministic test input
	for i := range src {
		src[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	if !forwardDIT16384SixStepComplex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384SixStepComplex64 returned false")
	}

	if !inverseDIT16384SixStepComplex64(inv, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16384SixStepComplex64 returned false")
	}

	maxErr := float32(0)

	for i := range n {
		re := real(inv[i]) - real(src[i])
		im := imag(inv[i]) - imag(src[i])

		if err := float32(math.Sqrt(float64(re*re + im*im))); err > maxErr {
			maxErr = err
		}
	}

	const tolerance = 1e-4
	if maxErr > tolerance {
		t.Errorf("Round-trip max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func TestForwardDIT16384SixStep_Complex128(t *testing.T) {
	const n = 16384

	src := make([]complex128, n)
	dstSixStep := make([]complex128, n)
	dstRadix4 := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)

	rng := rand.New(rand.NewSource(42)) //nolint:gosec // deterministic test input
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	if !forwardDIT16384SixStepComplex128(dstSixStep, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384SixStepComplex128 returned false")
	}

	if !forwardDIT16384Radix4Complex128(dstRadix4, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384Radix4Complex128 returned false")
	}

	maxErr := 0.0

	for i := range n {
		re := real(dstSixStep[i]) - real(dstRadix4[i])
		im := imag(dstSixStep[i]) - imag(dstRadix4[i])

		if err := math.Sqrt(re*re + im*im); err > maxErr {
			maxErr = err
		}
	}

	const tolerance = 1e-10
	if maxErr > tolerance {
		t.Errorf("Max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}

func TestRoundTrip16384SixStep_Complex128(t *testing.T) {
	const n = 16384

	src := make([]complex128, n)
	fwd := make([]complex128, n)
	inv := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)
	scratch := make([]complex128, n)

	rng := rand.New(rand.NewSource(7)) //nolint:gosec // deterministic test input
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	if !forwardDIT16384SixStepComplex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT16384SixStepComplex128 returned false")
	}

	if !inverseDIT16384SixStepComplex128(inv, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT16384SixStepComplex128 returned false")
	}

	maxErr := 0.0

	for i := range n {
		re := real(inv[i]) - real(src[i])
		im := imag(inv[i]) - imag(src[i])

		if err := math.Sqrt(re*re + im*im); err > maxErr {
			maxErr = err
		}
	}

	const tolerance = 1e-12
	if maxErr > tolerance {
		t.Errorf("Round-trip max error %e exceeds tolerance %e", maxErr, tolerance)
	}
}
