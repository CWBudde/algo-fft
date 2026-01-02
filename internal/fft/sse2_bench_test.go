//go:build amd64 && asm && !purego

package fft

import "testing"

func BenchmarkSSE2Forward(b *testing.B) {
	sse2Forward, _, sse2Available := getSSE2Kernels()
	if !sse2Available {
		b.Skip("SSE2 not available on this system")
	}

	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, bitrev, scratch := prepareFFTData(n)

			if !sse2Forward(dst, src, twiddle, scratch, bitrev) {
				b.Skip("SSE2 kernel failed")
			}

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))
			b.ResetTimer()

			for b.Loop() {
				sse2Forward(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

func BenchmarkSSE2Inverse(b *testing.B) {
	_, sse2Inverse, sse2Available := getSSE2Kernels()
	if !sse2Available {
		b.Skip("SSE2 not available on this system")
	}

	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, bitrev, scratch := prepareFFTData(n)

			if !sse2Inverse(dst, src, twiddle, scratch, bitrev) {
				b.Skip("SSE2 kernel failed")
			}

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))
			b.ResetTimer()

			for b.Loop() {
				sse2Inverse(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

// BenchmarkSSE2VsPureGo runs both SSE2 and pure-Go forward benchmarks for comparison.
func BenchmarkSSE2VsPureGo(b *testing.B) {
	sse2Forward, _, sse2Available := getSSE2Kernels()
	if !sse2Available {
		b.Skip("SSE2 not available on this system")
	}

	goForward, _ := getPureGoKernels()

	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, bitrev, scratch := prepareFFTData(n)

			b.Run("PureGo", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(n * 8))

				for b.Loop() {
					goForward(dst, src, twiddle, scratch, bitrev)
				}
			})

			if !sse2Forward(dst, src, twiddle, scratch, bitrev) {
				b.Run("SSE2", func(b *testing.B) {
					b.Skip("SSE2 kernel failed")
				})

				return
			}

			b.Run("SSE2", func(b *testing.B) {
				b.ReportAllocs()
				b.SetBytes(int64(n * 8))

				for b.Loop() {
					sse2Forward(dst, src, twiddle, scratch, bitrev)
				}
			})
		})
	}
}
