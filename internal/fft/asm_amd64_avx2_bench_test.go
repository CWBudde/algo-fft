//go:build amd64 && asm && !purego

package fft

import (
	"fmt"
	"testing"
)

// BenchmarkAVX2Forward benchmarks AVX2 forward transform across various sizes.
func BenchmarkAVX2Forward(b *testing.B) {
	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		b.Skip("AVX2 not available on this system")
	}

	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, bitrev, scratch := prepareFFTData(n)

			// Verify kernel works
			if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
				b.Skip("AVX2 kernel not yet implemented")
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 8)) // complex64 = 8 bytes

			for b.Loop() {
				avx2Forward(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

// BenchmarkAVX2Inverse benchmarks AVX2 inverse transform across various sizes.
func BenchmarkAVX2Inverse(b *testing.B) {
	_, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		b.Skip("AVX2 not available on this system")
	}

	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, bitrev, scratch := prepareFFTData(n)

			if !avx2Inverse(dst, src, twiddle, scratch, bitrev) {
				b.Skip("AVX2 kernel not yet implemented")
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 8))

			for b.Loop() {
				avx2Inverse(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

// BenchmarkAVX2StockhamForward benchmarks AVX2 Stockham forward transform.
func BenchmarkAVX2StockhamForward(b *testing.B) {
	avx2Forward, _, avx2Available := getAVX2StockhamKernels()
	if !avx2Available {
		b.Skip("AVX2 not available on this system")
	}

	sizes := []int{256, 512, 1024, 2048, 4096, 8192, 16384}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := generateRandomComplex64(n, 0xDEAD0000+uint64(n))
			twiddle, bitrev, scratch := prepareFFTData(n)
			dst := make([]complex64, n)

			if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
				b.Skip("AVX2 Stockham forward not implemented")
			}

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))
			b.ResetTimer()

			for range b.N {
				avx2Forward(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

// BenchmarkAVX2StockhamInverse benchmarks AVX2 Stockham inverse transform.
func BenchmarkAVX2StockhamInverse(b *testing.B) {
	_, avx2Inverse, avx2Available := getAVX2StockhamKernels()
	if !avx2Available {
		b.Skip("AVX2 not available on this system")
	}

	sizes := []int{256, 512, 1024, 2048, 4096, 8192, 16384}
	for _, n := range sizes {
		b.Run(fmt.Sprintf("N=%d", n), func(b *testing.B) {
			src := generateRandomComplex64(n, 0xBEEF0000+uint64(n))
			twiddle, bitrev, scratch := prepareFFTData(n)
			dst := make([]complex64, n)

			if !avx2Inverse(dst, src, twiddle, scratch, bitrev) {
				b.Skip("AVX2 Stockham inverse not implemented")
			}

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))
			b.ResetTimer()

			for range b.N {
				avx2Inverse(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

// BenchmarkPureGoForward benchmarks pure-Go forward transform for comparison.
func BenchmarkPureGoForward(b *testing.B) {
	goForward, _ := getPureGoKernels()

	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, bitrev, scratch := prepareFFTData(n)

			b.ResetTimer()
			b.SetBytes(int64(n * 8))

			for b.Loop() {
				goForward(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

// BenchmarkPureGoInverse benchmarks pure-Go inverse transform for comparison.
func BenchmarkPureGoInverse(b *testing.B) {
	_, goInverse := getPureGoKernels()

	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, bitrev, scratch := prepareFFTData(n)

			b.ResetTimer()
			b.SetBytes(int64(n * 8))

			for b.Loop() {
				goInverse(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

// BenchmarkAVX2VsPureGo runs both AVX2 and pure-Go benchmarks for comparison.
func BenchmarkAVX2VsPureGo(b *testing.B) {
	avx2Forward, _, avx2Available := getAVX2Kernels()
	goForward, _ := getPureGoKernels()

	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, bitrev, scratch := prepareFFTData(n)

			b.Run("PureGo", func(b *testing.B) {
				b.SetBytes(int64(n * 8))

				for b.Loop() {
					goForward(dst, src, twiddle, scratch, bitrev)
				}
			})

			if avx2Available {
				// Test if AVX2 is implemented
				if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
					b.Run("AVX2", func(b *testing.B) {
						b.Skip("AVX2 kernel not yet implemented")
					})

					return
				}

				b.Run("AVX2", func(b *testing.B) {
					b.SetBytes(int64(n * 8))

					for b.Loop() {
						avx2Forward(dst, src, twiddle, scratch, bitrev)
					}
				})
			}
		})
	}
}

// BenchmarkAVX2Forward128 benchmarks AVX2 complex128 forward transform.
func BenchmarkAVX2Forward128(b *testing.B) {
	avx2Forward, _, avx2Available := getAVX2Kernels128()
	if !avx2Available {
		b.Skip("AVX2 not available")
	}

	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := make([]complex128, n)
			dst := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex128, n)

			if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
				b.Skip("AVX2 complex128 not implemented")
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 16))

			for range b.N {
				avx2Forward(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

// BenchmarkAVX2Inverse128 benchmarks AVX2 complex128 inverse transform.
func BenchmarkAVX2Inverse128(b *testing.B) {
	_, avx2Inverse, avx2Available := getAVX2Kernels128()
	if !avx2Available {
		b.Skip("AVX2 not available")
	}

	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := make([]complex128, n)
			dst := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex128, n)

			if !avx2Inverse(dst, src, twiddle, scratch, bitrev) {
				b.Skip("AVX2 complex128 not implemented")
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 16))

			for range b.N {
				avx2Inverse(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}

// BenchmarkAVX2Size16_VsGeneric benchmarks the size-16 kernel vs generic AVX2.
func BenchmarkAVX2Size16_VsGeneric(b *testing.B) {
	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		b.Skip("AVX2 not available")
	}

	goForward, _ := getPureGoKernels()

	const n = 16

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i)/float32(n), float32(i%4)/4)
	}

	twiddle, bitrev, scratch := prepareFFTData(n)
	dst := make([]complex64, n)

	b.Run("AVX2", func(b *testing.B) {
		if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
			b.Skip("AVX2 kernel not implemented")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			avx2Forward(dst, src, twiddle, scratch, bitrev)
		}
	})

	b.Run("PureGo", func(b *testing.B) {
		if !goForward(dst, src, twiddle, scratch, bitrev) {
			b.Skip("Pure Go kernel failed")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			goForward(dst, src, twiddle, scratch, bitrev)
		}
	})
}

// BenchmarkAVX2Size32_VsGeneric benchmarks the size-32 kernel vs generic AVX2.
func BenchmarkAVX2Size32_VsGeneric(b *testing.B) {
	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		b.Skip("AVX2 not available")
	}

	goForward, _ := getPureGoKernels()

	const n = 32

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i)/float32(n), float32(i%4)/4)
	}

	twiddle, bitrev, scratch := prepareFFTData(n)
	dst := make([]complex64, n)

	b.Run("AVX2", func(b *testing.B) {
		if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
			b.Skip("AVX2 kernel not implemented")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			avx2Forward(dst, src, twiddle, scratch, bitrev)
		}
	})

	b.Run("PureGo", func(b *testing.B) {
		if !goForward(dst, src, twiddle, scratch, bitrev) {
			b.Skip("Pure Go kernel failed")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			goForward(dst, src, twiddle, scratch, bitrev)
		}
	})
}

// BenchmarkAVX2Size64_VsGeneric benchmarks the size-64 kernel vs generic AVX2.
func BenchmarkAVX2Size64_VsGeneric(b *testing.B) {
	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		b.Skip("AVX2 not available")
	}

	goForward, _ := getPureGoKernels()

	const n = 64

	src := make([]complex64, n)
	for i := range src {
		src[i] = complex(float32(i)/float32(n), float32(i%4)/4)
	}

	twiddle, bitrev, scratch := prepareFFTData(n)
	dst := make([]complex64, n)

	b.Run("AVX2", func(b *testing.B) {
		if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
			b.Skip("AVX2 kernel not implemented")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			avx2Forward(dst, src, twiddle, scratch, bitrev)
		}
	})

	b.Run("PureGo", func(b *testing.B) {
		if !goForward(dst, src, twiddle, scratch, bitrev) {
			b.Skip("Pure Go kernel failed")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			goForward(dst, src, twiddle, scratch, bitrev)
		}
	})
}

// BenchmarkAVX2Size256_Comprehensive compares all size-256 FFT implementations:
// AVX2 (radix-2 and radix-4), pure-Go DIT, and radix-4 variants.
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
		if !forwardDIT256Complex64(dst, src, twiddle, scratch) {
			b.Skip("Pure Go DIT failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardDIT256Complex64(dst, src, twiddle, scratch)
		}
	})

	b.Run("PureGo_Radix4", func(b *testing.B) {
		if !forwardDIT256Radix4Complex64(dst, src, twiddle, scratch) {
			b.Skip("Pure Go radix-4 failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardDIT256Radix4Complex64(dst, src, twiddle, scratch)
		}
	})

	b.Run("PureGo_Radix4_Optimized", func(b *testing.B) {
		if !forwardDIT256Radix4Complex64(dst, src, twiddle, scratch) {
			b.Skip("Pure Go optimized radix-4 failed")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardDIT256Radix4Complex64(dst, src, twiddle, scratch)
		}
	})
}
