//go:build amd64 && asm && !purego

package kernels

import (
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

const (
	size384MixedTol64  = 1e-3  // Looser tolerance due to multiple FFT stages
	size384MixedTol128 = 1e-10 // Looser tolerance for multi-stage FFT (errors ~1e-12 to 5e-12)
)

// TestForwardDIT384MixedComplex64 tests the size-384 forward mixed-radix kernel.
func TestForwardDIT384MixedComplex64(t *testing.T) {
	t.Parallel()

	const n = 384

	src := randomComplex64(n, 0xDEADBEEF)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT384MixedComplex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT384MixedComplex64 failed")
	}

	want := reference.NaiveDFT(src)
	assertComplex64Close(t, dst, want, size384MixedTol64)
}

// TestInverseDIT384MixedComplex64 tests the size-384 inverse mixed-radix kernel.
func TestInverseDIT384MixedComplex64(t *testing.T) {
	t.Parallel()

	const n = 384

	src := randomComplex64(n, 0xCAFEBABE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT384MixedComplex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT384MixedComplex64 failed")
	}

	if !inverseDIT384MixedComplex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT384MixedComplex64 failed")
	}

	want := reference.NaiveIDFT(fwd)
	assertComplex64Close(t, dst, want, size384MixedTol64)
}

// TestRoundTripDIT384MixedComplex64 tests forward then inverse returns original.
func TestRoundTripDIT384MixedComplex64(t *testing.T) {
	t.Parallel()

	const n = 384

	src := randomComplex64(n, 0xFEEDFACE)
	fwd := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT384MixedComplex64(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT384MixedComplex64 failed")
	}

	if !inverseDIT384MixedComplex64(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT384MixedComplex64 failed")
	}

	assertComplex64Close(t, dst, src, size384MixedTol64)
}

// TestForwardDIT384MixedComplex64_AllZeros tests edge case with all zeros.
func TestForwardDIT384MixedComplex64_AllZeros(t *testing.T) {
	t.Parallel()

	const n = 384

	src := make([]complex64, n)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT384MixedComplex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT384MixedComplex64 failed")
	}

	// FFT of zeros should be zeros
	for i, v := range dst {
		if v != 0 {
			t.Errorf("dst[%d] = %v, want 0", i, v)
		}
	}
}

// TestForwardDIT384MixedComplex64_Impulse tests impulse response.
func TestForwardDIT384MixedComplex64_Impulse(t *testing.T) {
	t.Parallel()

	const n = 384

	src := make([]complex64, n)
	src[0] = 1 // Impulse at position 0
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if !forwardDIT384MixedComplex64(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT384MixedComplex64 failed")
	}

	// FFT of impulse should be all ones (DC component = 1)
	for i, v := range dst {
		if real(v) < 0.99 || real(v) > 1.01 || imag(v) < -0.01 || imag(v) > 0.01 {
			t.Errorf("dst[%d] = %v, want ~1+0i", i, v)
		}
	}
}

// TestForwardDIT384MixedComplex64_SliceTooSmall tests error handling.
func TestForwardDIT384MixedComplex64_SliceTooSmall(t *testing.T) {
	t.Parallel()

	const n = 384

	src := make([]complex64, n-1) // Too small
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	if forwardDIT384MixedComplex64(dst, src, twiddle, scratch) {
		t.Error("forwardDIT384MixedComplex64 should return false for too-small src")
	}
}

// TestForwardDIT384MixedComplex128 tests the size-384 forward mixed-radix kernel (complex128).
func TestForwardDIT384MixedComplex128(t *testing.T) {
	t.Parallel()

	const n = 384

	src := randomComplex128(n, 0xDEADBEEF)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT384MixedComplex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT384MixedComplex128 failed")
	}

	want := reference.NaiveDFT128(src)
	assertComplex128Close(t, dst, want, size384MixedTol128)
}

// TestInverseDIT384MixedComplex128 tests the size-384 inverse mixed-radix kernel (complex128).
func TestInverseDIT384MixedComplex128(t *testing.T) {
	t.Parallel()

	const n = 384

	src := randomComplex128(n, 0xCAFEBABE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT384MixedComplex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT384MixedComplex128 failed")
	}

	if !inverseDIT384MixedComplex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT384MixedComplex128 failed")
	}

	want := reference.NaiveIDFT128(fwd)
	assertComplex128Close(t, dst, want, size384MixedTol128)
}

// TestRoundTripDIT384MixedComplex128 tests forward then inverse returns original (complex128).
func TestRoundTripDIT384MixedComplex128(t *testing.T) {
	t.Parallel()

	const n = 384

	src := randomComplex128(n, 0xFEEDFACE)
	fwd := make([]complex128, n)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT384MixedComplex128(fwd, src, twiddle, scratch) {
		t.Fatal("forwardDIT384MixedComplex128 failed")
	}

	if !inverseDIT384MixedComplex128(dst, fwd, twiddle, scratch) {
		t.Fatal("inverseDIT384MixedComplex128 failed")
	}

	assertComplex128Close(t, dst, src, size384MixedTol128)
}

// TestForwardDIT384MixedComplex128_Impulse tests impulse response (complex128).
func TestForwardDIT384MixedComplex128_Impulse(t *testing.T) {
	t.Parallel()

	const n = 384

	src := make([]complex128, n)
	src[0] = 1 // Impulse at position 0
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardDIT384MixedComplex128(dst, src, twiddle, scratch) {
		t.Fatal("forwardDIT384MixedComplex128 failed")
	}

	// FFT of impulse should be all ones
	for i, v := range dst {
		if real(v) < 0.999999 || real(v) > 1.000001 || imag(v) < -0.000001 || imag(v) > 0.000001 {
			t.Errorf("dst[%d] = %v, want ~1+0i", i, v)
		}
	}
}

// BenchmarkForwardDIT384MixedComplex64 benchmarks the size-384 forward FFT (complex64).
func BenchmarkForwardDIT384MixedComplex64(b *testing.B) {
	const n = 384

	src := randomComplex64(n, 0xBEEF384)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8)) // 8 bytes per complex64
	b.ResetTimer()

	for b.Loop() {
		forwardDIT384MixedComplex64(dst, src, twiddle, scratch)
	}
}

// BenchmarkInverseDIT384MixedComplex64 benchmarks the size-384 inverse FFT (complex64).
func BenchmarkInverseDIT384MixedComplex64(b *testing.B) {
	const n = 384

	src := randomComplex64(n, 0xBEEF384)
	dst := make([]complex64, n)
	scratch := make([]complex64, n)
	twiddle := ComputeTwiddleFactors[complex64](n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 8))
	b.ResetTimer()

	for b.Loop() {
		inverseDIT384MixedComplex64(dst, src, twiddle, scratch)
	}
}

// BenchmarkForwardDIT384MixedComplex128 benchmarks the size-384 forward FFT (complex128).
func BenchmarkForwardDIT384MixedComplex128(b *testing.B) {
	const n = 384

	src := randomComplex128(n, 0xBEEF384)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16)) // 16 bytes per complex128
	b.ResetTimer()

	for b.Loop() {
		forwardDIT384MixedComplex128(dst, src, twiddle, scratch)
	}
}

// BenchmarkInverseDIT384MixedComplex128 benchmarks the size-384 inverse FFT (complex128).
func BenchmarkInverseDIT384MixedComplex128(b *testing.B) {
	const n = 384

	src := randomComplex128(n, 0xBEEF384)
	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	b.ReportAllocs()
	b.SetBytes(int64(n * 16))
	b.ResetTimer()

	for b.Loop() {
		inverseDIT384MixedComplex128(dst, src, twiddle, scratch)
	}
}
