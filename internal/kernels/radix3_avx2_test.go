//go:build amd64 && asm && !purego

package kernels

import (
	"testing"

	"github.com/cwbudde/algo-fft/internal/asm/amd64"
)

func TestButterfly3ForwardAVX2Complex64(t *testing.T) {
	t.Parallel()

	// Test 4 parallel butterflies with known inputs
	a0 := []complex64{1 + 0i, 2 + 0i, 3 + 0i, 4 + 0i}
	a1 := []complex64{0 + 1i, 1 + 1i, 2 + 1i, 3 + 1i}
	a2 := []complex64{0 - 1i, 1 - 1i, 2 - 1i, 3 - 1i}

	y0 := make([]complex64, 4)
	y1 := make([]complex64, 4)
	y2 := make([]complex64, 4)

	amd64.Butterfly3ForwardAVX2Complex64(y0, y1, y2, a0, a1, a2)

	// Compute expected results using scalar implementation
	want0 := make([]complex64, 4)
	want1 := make([]complex64, 4)
	want2 := make([]complex64, 4)

	for i := range 4 {
		want0[i], want1[i], want2[i] = butterfly3ForwardComplex64(a0[i], a1[i], a2[i])
	}

	// Compare results
	assertComplex64Close(t, y0, want0, 1e-5)
	assertComplex64Close(t, y1, want1, 1e-5)
	assertComplex64Close(t, y2, want2, 1e-5)
}

func TestButterfly3InverseAVX2Complex64(t *testing.T) {
	t.Parallel()

	// Test 4 parallel butterflies - use same value in all 4 lanes for easier debugging
	a0 := []complex64{1 + 2i, 1 + 2i, 1 + 2i, 1 + 2i}
	a1 := []complex64{-1 + 3i, -1 + 3i, -1 + 3i, -1 + 3i}
	a2 := []complex64{6 - 1i, 6 - 1i, 6 - 1i, 6 - 1i}

	y0 := make([]complex64, 4)
	y1 := make([]complex64, 4)
	y2 := make([]complex64, 4)

	amd64.Butterfly3InverseAVX2Complex64(y0, y1, y2, a0, a1, a2)

	// Compute expected results using scalar implementation
	want0 := make([]complex64, 4)
	want1 := make([]complex64, 4)
	want2 := make([]complex64, 4)

	for i := range 4 {
		want0[i], want1[i], want2[i] = butterfly3InverseComplex64(a0[i], a1[i], a2[i])
	}

	t.Logf("Input: a0=%v, a1=%v, a2=%v", a0[0], a1[0], a2[0])
	t.Logf("AVX2 result: y0=%v, y1=%v, y2=%v", y0[0], y1[0], y2[0])
	t.Logf("Scalar result: y0=%v, y1=%v, y2=%v", want0[0], want1[0], want2[0])

	// Compare results
	assertComplex64Close(t, y0, want0, 1e-5)
	assertComplex64Close(t, y1, want1, 1e-5)
	assertComplex64Close(t, y2, want2, 1e-5)
}

func TestButterfly3RoundTripAVX2(t *testing.T) {
	t.Parallel()

	// Test that inverse(forward(x)) * (1/3) ≈ x for 4 parallel butterflies
	// Note: Radix-3 DFT requires scaling by 1/3 on inverse to round-trip
	orig0 := []complex64{1 + 2i, 3 + 4i, 5 + 6i, 7 + 8i}
	orig1 := []complex64{-1 + 3i, 2 - 5i, 4 + 1i, -3 - 2i}
	orig2 := []complex64{6 - 1i, -2 + 7i, 3 - 4i, 8 + 5i}

	fwd0 := make([]complex64, 4)
	fwd1 := make([]complex64, 4)
	fwd2 := make([]complex64, 4)

	inv0 := make([]complex64, 4)
	inv1 := make([]complex64, 4)
	inv2 := make([]complex64, 4)

	// Forward transform
	amd64.Butterfly3ForwardAVX2Complex64(fwd0, fwd1, fwd2, orig0, orig1, orig2)

	// Inverse transform
	amd64.Butterfly3InverseAVX2Complex64(inv0, inv1, inv2, fwd0, fwd1, fwd2)

	// Apply radix-3 scaling factor
	for i := range inv0 {
		inv0[i] /= 3
		inv1[i] /= 3
		inv2[i] /= 3
	}

	// Results should match original inputs (after scaling)
	assertComplex64Close(t, inv0, orig0, 1e-5)
	assertComplex64Close(t, inv1, orig1, 1e-5)
	assertComplex64Close(t, inv2, orig2, 1e-5)
}

func BenchmarkButterfly3ForwardAVX2(b *testing.B) {
	a0 := make([]complex64, 4)
	a1 := make([]complex64, 4)
	a2 := make([]complex64, 4)
	y0 := make([]complex64, 4)
	y1 := make([]complex64, 4)
	y2 := make([]complex64, 4)

	// Initialize with random data
	for i := range 4 {
		a0[i] = complex(float32(i), float32(i+1))
		a1[i] = complex(float32(i+2), float32(i+3))
		a2[i] = complex(float32(i+4), float32(i+5))
	}

	b.SetBytes(4 * 3 * 8) // 4 butterflies * 3 complex64 inputs * 8 bytes
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		amd64.Butterfly3ForwardAVX2Complex64(y0, y1, y2, a0, a1, a2)
	}
}

func BenchmarkButterfly3ForwardScalar(b *testing.B) {
	a0 := make([]complex64, 4)
	a1 := make([]complex64, 4)
	a2 := make([]complex64, 4)
	y0 := make([]complex64, 4)
	y1 := make([]complex64, 4)
	y2 := make([]complex64, 4)

	for i := range 4 {
		a0[i] = complex(float32(i), float32(i+1))
		a1[i] = complex(float32(i+2), float32(i+3))
		a2[i] = complex(float32(i+4), float32(i+5))
	}

	b.SetBytes(4 * 3 * 8)
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		for j := range 4 {
			y0[j], y1[j], y2[j] = butterfly3ForwardComplex64(a0[j], a1[j], a2[j])
		}
	}
}
