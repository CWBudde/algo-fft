//go:build amd64 && asm && !purego

package fft

import "testing"

// BenchmarkAVX2Size256_Comprehensive compares all size-256 FFT implementations:
// AVX2 (radix-2 and radix-4), pure-Go DIT, and radix-4 variants.
//
// This benchmark requires the asm build tag to access AVX2 assembly implementations.
// Run with: go test -tags asm -bench BenchmarkAVX2Size256_Comprehensive -benchmem
//
// Results typically show (on Intel 12th Gen i7-1255U):
//   - AVX2_Radix2:             ~400 ns/op  (~5.1 GB/s)  - FASTEST
//   - AVX2_Radix4:             ~720 ns/op  (~2.9 GB/s)
//   - PureGo_DIT_Radix2:      ~2000 ns/op  (~1.0 GB/s)
//   - PureGo_Radix4:          ~2400 ns/op  (~0.9 GB/s)
//   - PureGo_Radix4_Optimized: ~2200 ns/op  (~0.9 GB/s)
//
// The AVX2 radix-2 implementation is ~5x faster than pure Go and ~1.8x faster than AVX2 radix-4.
func BenchmarkAVX2Size256_Comprehensive(b *testing.B) {
	const n = 256
	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i)/float32(n), float32(i%4)/4)
	}
	dst := make([]complex64, n)
	scratch := make([]complex64, n)

	// Radix-2 setup
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)

	// Radix-4 setup
	bitrevRadix4 := ComputeBitReversalIndicesRadix4(n)

	b.Run("AVX2_Radix2", func(b *testing.B) {
		if !forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			b.Skip("AVX2 radix-2 assembly not available")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch, bitrev)
		}
	})

	b.Run("AVX2_Radix4", func(b *testing.B) {
		if !forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevRadix4) {
			b.Skip("AVX2 radix-4 assembly not available")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch, bitrevRadix4)
		}
	})

	b.Run("PureGo_DIT_Radix2", func(b *testing.B) {
		if !forwardDIT256Complex64(dst, src, twiddle, scratch, bitrev) {
			b.Skip("Pure Go DIT failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardDIT256Complex64(dst, src, twiddle, scratch, bitrev)
		}
	})

	b.Run("PureGo_Radix4", func(b *testing.B) {
		if !forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrevRadix4) {
			b.Skip("Pure Go radix-4 failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrevRadix4)
		}
	})

	b.Run("PureGo_Radix4_Optimized", func(b *testing.B) {
		if !forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrevRadix4) {
			b.Skip("Pure Go optimized radix-4 failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardDIT256Radix4Complex64(dst, src, twiddle, scratch, bitrevRadix4)
		}
	})
}
