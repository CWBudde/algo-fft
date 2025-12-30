package fft

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestForwardDIT32MixedRadix24MatchesReference validates forward FFT against naive DFT.
func TestForwardDIT32MixedRadix24MatchesReference(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0xABCD01+uint64(n))
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT32MixedRadix24Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64SliceClose(t, dst, want, n)
}

// TestForwardDIT32MixedRadix24MatchesRadix2 validates against existing radix-2 implementation.
func TestForwardDIT32MixedRadix24MatchesRadix2(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0xBCDE02+uint64(n))
	dstRadix2 := make([]complex64, n)
	dstMixed := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	// Compute with radix-2 implementation
	if !forwardDIT32Complex64(dstRadix2, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32Complex64 (radix-2) failed")
	}

	// Compute with mixed radix implementation
	if !forwardDIT32MixedRadix24Complex64(dstMixed, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 failed")
	}

	assertComplex64SliceClose(t, dstMixed, dstRadix2, n)
}

// TestForwardDIT32MixedRadix24ImpulseResponse tests impulse response δ[0].
// Critical test: DC component must equal 1.0 (not 2.0 - was a bug in previous impl).
func TestForwardDIT32MixedRadix24ImpulseResponse(t *testing.T) {
	t.Parallel()

	const n = 32

	// Create impulse: δ[0] = 1, rest = 0
	src := make([]complex64, n)
	src[0] = complex(1.0, 0)

	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT32MixedRadix24Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 failed for impulse")
	}

	// All bins should equal 1.0 for impulse input
	want := make([]complex64, n)
	for i := 0; i < n; i++ {
		want[i] = complex(1.0, 0.0)
	}
	assertComplex64SliceClose(t, dst, want, n)
}

// TestForwardDIT32MixedRadix24Linearity tests that FFT is linear: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y).
func TestForwardDIT32MixedRadix24Linearity(t *testing.T) {
	t.Parallel()

	const n = 32
	a := complex(float32(2.5), float32(-0.7))
	b := complex(float32(-1.2), float32(1.8))

	x := randomComplex64(n, 0x12AB34+uint64(n))
	y := randomComplex64(n, 0x56CD78+uint64(n))

	// Compute FFT(a*x + b*y)
	input := make([]complex64, n)
	for i := 0; i < n; i++ {
		input[i] = a*x[i] + b*y[i]
	}

	dstCombined := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT32MixedRadix24Complex64(dstCombined, input, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 failed for combined input")
	}

	// Compute a*FFT(x) + b*FFT(y)
	dstX := make([]complex64, n)
	dstY := make([]complex64, n)

	if !forwardDIT32MixedRadix24Complex64(dstX, x, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 failed for x")
	}

	if !forwardDIT32MixedRadix24Complex64(dstY, y, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 failed for y")
	}

	want := make([]complex64, n)
	for i := 0; i < n; i++ {
		want[i] = a*dstX[i] + b*dstY[i]
	}

	assertComplex64SliceClose(t, dstCombined, want, n)
}

// TestInverseDIT32MixedRadix24MatchesReference validates inverse FFT against naive IDFT.
func TestInverseDIT32MixedRadix24MatchesReference(t *testing.T) {
	t.Parallel()

	const n = 32

	// First compute forward FFT to get frequency domain
	src := randomComplex64(n, 0x78ABCD+uint64(n))
	fwd := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT32MixedRadix24Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 failed")
	}

	// Now compute inverse and compare to reference IDFT
	dst := make([]complex64, n)
	if !inverseDIT32MixedRadix24Complex64(dst, fwd, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT32MixedRadix24Complex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64SliceClose(t, dst, want, n)
}

// TestRoundTripDIT32MixedRadix24 tests that inverse(forward(x)) ≈ x.
func TestRoundTripDIT32MixedRadix24(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0x9EF012+uint64(n))
	fwd := make([]complex64, n)
	inv := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	// Forward transform
	if !forwardDIT32MixedRadix24Complex64(fwd, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 failed")
	}

	// Inverse transform
	if !inverseDIT32MixedRadix24Complex64(inv, fwd, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT32MixedRadix24Complex64 failed")
	}

	assertComplex64SliceClose(t, inv, src, n)
}

// TestInverseDIT32MixedRadix24MatchesRadix2 validates inverse against radix-2 implementation.
func TestInverseDIT32MixedRadix24MatchesRadix2(t *testing.T) {
	t.Parallel()

	const n = 32

	// Get frequency-domain input
	src := randomComplex64(n, 0x345EF6+uint64(n))
	dstRadix2 := make([]complex64, n)
	dstMixed := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	// Compute with radix-2 inverse
	if !inverseDIT32Complex64(dstRadix2, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT32Complex64 (radix-2) failed")
	}

	// Compute with mixed radix inverse
	if !inverseDIT32MixedRadix24Complex64(dstMixed, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT32MixedRadix24Complex64 failed")
	}

	assertComplex64SliceClose(t, dstMixed, dstRadix2, n)
}

// TestParseval tests Parseval's theorem: sum(|x[n]|^2) = (1/N) * sum(|X[k]|^2).
func TestParsevalDIT32MixedRadix24(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0x789AB0+uint64(n))
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	if !forwardDIT32MixedRadix24Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 failed")
	}

	// Compute energy in time domain
	energyTime := float32(0)
	for i := 0; i < n; i++ {
		c := src[i]
		energyTime += real(c)*real(c) + imag(c)*imag(c)
	}

	// Compute energy in frequency domain
	energyFreq := float32(0)
	for i := 0; i < n; i++ {
		c := dst[i]
		energyFreq += real(c)*real(c) + imag(c)*imag(c)
	}

	// Parseval: energyTime = energyFreq / N
	wantEnergy := energyFreq / float32(n)
	diff := energyTime - wantEnergy
	if diff < 0 {
		diff = -diff
	}
	if diff > 1e-4 {
		t.Errorf("Parseval failed: time energy %f, freq energy %f, expected %f, diff %f",
			energyTime, energyFreq, wantEnergy, diff)
	}
}

// TestInPlaceTransformDIT32MixedRadix24 tests in-place forward transform.
func TestInPlaceForwardDIT32MixedRadix24(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0xCDEF12+uint64(n))
	data := make([]complex64, n)
	copy(data, src)

	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	// In-place transform
	if !forwardDIT32MixedRadix24Complex64(data, data, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 in-place failed")
	}

	// Compare with out-of-place
	dst := make([]complex64, n)
	if !forwardDIT32MixedRadix24Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("forwardDIT32MixedRadix24Complex64 out-of-place failed")
	}

	assertComplex64SliceClose(t, data, dst, n)
}

// TestInPlaceInverseDIT32MixedRadix24 tests in-place inverse transform.
func TestInPlaceInverseDIT32MixedRadix24(t *testing.T) {
	t.Parallel()

	const n = 32
	src := randomComplex64(n, 0x456789+uint64(n))
	data := make([]complex64, n)
	copy(data, src)

	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	// In-place transform
	if !inverseDIT32MixedRadix24Complex64(data, data, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT32MixedRadix24Complex64 in-place failed")
	}

	// Compare with out-of-place
	dst := make([]complex64, n)
	if !inverseDIT32MixedRadix24Complex64(dst, src, twiddle, scratch, bitrev) {
		t.Fatalf("inverseDIT32MixedRadix24Complex64 out-of-place failed")
	}

	assertComplex64SliceClose(t, data, dst, n)
}
