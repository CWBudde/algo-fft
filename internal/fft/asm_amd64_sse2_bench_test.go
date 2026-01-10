//go:build amd64 && asm && !purego

package fft

import "testing"

// BenchmarkSSE2Forward benchmarks SSE2 forward transform across various sizes.
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

// BenchmarkSSE2Inverse benchmarks SSE2 inverse transform across various sizes.
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

// BenchmarkSSE2Sizes benchmarks SSE2 across smaller sizes including size-32.
func BenchmarkSSE2Sizes(b *testing.B) {
	sse2Forward, _, sse2Available := getSSE2Kernels()
	if !sse2Available {
		b.Skip("SSE2 not available")
	}

	sizes := []int{8, 16, 32, 64, 128, 256}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, bitrev, scratch := prepareFFTData(n)

			if !sse2Forward(dst, src, twiddle, scratch, bitrev) {
				b.Fatal("SSE2 kernel failed")
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

// BenchmarkSSE2Size32Comparison compares different size-32 kernel implementations.
func BenchmarkSSE2Size32Comparison(b *testing.B) {
	const n = 32
	src := generateRandomComplex64(n, 0xBEEF32)
	dst := make([]complex64, n)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// SSE2 dispatch (which falls back to generic SSE2 for size 32)
	b.Run("SSE2_Dispatch", func(b *testing.B) {
		sse2Forward, _, sse2Available := getSSE2Kernels()
		if !sse2Available {
			b.Skip("SSE2 not available")
		}
		if !sse2Forward(dst, src, twiddle, scratch, bitrev) {
			b.Fatal("SSE2 dispatch failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for b.Loop() {
			sse2Forward(dst, src, twiddle, scratch, bitrev)
		}
	})

	// Generic SSE2 kernel directly
	b.Run("SSE2_Generic", func(b *testing.B) {
		if !forwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev) {
			b.Fatal("Generic SSE2 failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for b.Loop() {
			forwardSSE2Complex64Asm(dst, src, twiddle, scratch, bitrev)
		}
	})

	// SSE2 size-32 radix-32 kernel
	b.Run("SSE2_Radix32", func(b *testing.B) {
		if !forwardSSE2Size32Radix32Complex64Asm(dst, src, twiddle, scratch) {
			b.Fatal("SSE2 radix-32 failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for b.Loop() {
			forwardSSE2Size32Radix32Complex64Asm(dst, src, twiddle, scratch)
		}
	})

	// Pure Go for comparison
	b.Run("PureGo", func(b *testing.B) {
		goForward, _ := getPureGoKernels()
		if !goForward(dst, src, twiddle, scratch, bitrev) {
			b.Fatal("Pure Go failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for b.Loop() {
			goForward(dst, src, twiddle, scratch, bitrev)
		}
	})

	// AVX2 for comparison (this is the fastest)
	b.Run("AVX2_Generic", func(b *testing.B) {
		if !forwardAVX2Complex64Asm(dst, src, twiddle, scratch) {
			b.Fatal("AVX2 failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for b.Loop() {
			forwardAVX2Complex64Asm(dst, src, twiddle, scratch)
		}
	})

	// AVX2 size-32 radix-32 kernel (size-specific dispatch)
	b.Run("AVX2_Radix32", func(b *testing.B) {
		if !forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch) {
			b.Fatal("AVX2 radix-32 failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for b.Loop() {
			forwardAVX2Size32Complex64Asm(dst, src, twiddle, scratch)
		}
	})
}
