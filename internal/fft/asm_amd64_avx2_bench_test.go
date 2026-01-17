//go:build amd64 && asm && !purego

// Package fft provides AVX2-optimized FFT benchmarks.
// Tests are in asm_amd64_avx2_test.go.
package fft

import (
	"fmt"
	"testing"
)

// =============================================================================
// Individual Kernel Benchmarks
// =============================================================================

// BenchmarkAVX2DITComplex64 benchmarks all AVX2 DIT kernels for complex64.
func BenchmarkAVX2DITComplex64(b *testing.B) {
	cases := []struct {
		name    string
		n       int
		forward func(dst, src, twiddle, scratch []complex64) bool
	}{
		{"Size4/Radix4", 4, forwardAVX2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, forwardAVX2Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, forwardAVX2Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, forwardAVX2Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, forwardAVX2Size16Radix2Complex64Asm},
		{"Size16/Radix4", 16, forwardAVX2Size16Radix4Complex64Asm},
		{"Size32/Radix2", 32, forwardAVX2Size32Radix2Complex64Asm},
		{"Size32/Radix32", 32, forwardAVX2Size32Complex64Asm},
		{"Size64/Radix2", 64, forwardAVX2Size64Radix2Complex64Asm},
		{"Size64/Radix4", 64, forwardAVX2Size64Radix4Complex64Asm},
		{"Size128", 128, forwardAVX2Size128Complex64Asm},
		{"Size256/Radix2", 256, forwardAVX2Size256Radix2Complex64Asm},
		{"Size256/Radix4", 256, forwardAVX2Size256Radix4Complex64Asm},
		{"Size512/Radix4Then2", 512, forwardAVX2Size512Radix4Then2Complex64Asm},
		{"Size512/Radix2", 512, forwardAVX2Size512Radix2Complex64Asm},
		{"Size512/Radix8", 512, forwardAVX2Size512Radix8Complex64Asm},
		{"Size512/Radix16x32", 512, forwardAVX2Size512Radix16x32Complex64Asm},
		{"Size1024/Radix4", 1024, forwardAVX2Size1024Radix4Complex64Asm},
		{"Size2048/Radix4Then2", 2048, forwardAVX2Size2048Radix4Then2Complex64Asm},
		{"Size4096/Radix4", 4096, forwardAVX2Size4096Radix4Complex64Asm},
		{"Size8192/Radix4Then2", 8192, forwardAVX2Size8192Radix4Then2Complex64Asm},
	}

	for _, testCase := range cases {
		b.Run(testCase.name, func(b *testing.B) {
			src := make([]complex64, testCase.n)
			dst := make([]complex64, testCase.n)
			scratch := make([]complex64, testCase.n)
			twiddle := ComputeTwiddleFactors[complex64](testCase.n)

			for i := range src {
				src[i] = complex(float32(i), float32(-i))
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(testCase.n * 8))

			for b.Loop() {
				testCase.forward(dst, src, twiddle, scratch)
			}
		})
	}
}

// BenchmarkAVX2DITComplex128 benchmarks all AVX2 DIT kernels for complex128.
func BenchmarkAVX2DITComplex128(b *testing.B) {
	cases := []struct {
		name    string
		n       int
		forward func(dst, src, twiddle, scratch []complex128) bool
	}{
		{"Size4/Radix4", 4, forwardAVX2Size4Radix4Complex128Asm},
		{"Size8/Radix2", 8, forwardAVX2Size8Radix2Complex128Asm},
		{"Size8/Radix4", 8, forwardAVX2Size8Radix4Complex128Asm},
		{"Size8/Radix8", 8, forwardAVX2Size8Radix8Complex128Asm},
		{"Size16/Radix2", 16, forwardAVX2Size16Radix2Complex128Asm},
		{"Size16/Radix4", 16, forwardAVX2Size16Radix4Complex128Asm},
		{"Size32/Radix2", 32, forwardAVX2Size32Complex128Asm},
		{"Size128/Radix2", 128, forwardAVX2Size128Radix2Complex128Asm},
		{"Size64/Radix2", 64, forwardAVX2Size64Radix2Complex128Asm},
		{"Size64/Radix4", 64, forwardAVX2Size64Radix4Complex128Asm},
		{"Size256/Radix2", 256, forwardAVX2Size256Radix2Complex128Asm},
		{"Size512/Radix2", 512, forwardAVX2Size512Radix2Complex128Asm},
		{"Size512/Radix4Then2", 512, forwardAVX2Size512Radix4Then2Complex128Asm},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			src := make([]complex128, tc.n)
			dst := make([]complex128, tc.n)
			scratch := make([]complex128, tc.n)
			twiddle := ComputeTwiddleFactors[complex128](tc.n)

			for i := range src {
				src[i] = complex(float64(i), float64(-i))
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(tc.n * 16))

			for b.Loop() {
				tc.forward(dst, src, twiddle, scratch)
			}
		})
	}
}

// =============================================================================
// Dispatcher Benchmarks
// =============================================================================

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
			twiddle, scratch := prepareFFTData[complex64](n)

			// Verify kernel works
			if !avx2Forward(dst, src, twiddle, scratch) {
				b.Skip("AVX2 kernel not yet implemented")
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 8))

			for b.Loop() {
				avx2Forward(dst, src, twiddle, scratch)
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
			twiddle, scratch := prepareFFTData[complex64](n)

			if !avx2Inverse(dst, src, twiddle, scratch) {
				b.Skip("AVX2 kernel not yet implemented")
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 8))

			for b.Loop() {
				avx2Inverse(dst, src, twiddle, scratch)
			}
		})
	}
}

// =============================================================================
// Stockham Benchmarks
// =============================================================================

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
			twiddle, scratch := prepareFFTData[complex64](n)
			dst := make([]complex64, n)

			if !avx2Forward(dst, src, twiddle, scratch) {
				b.Skip("AVX2 Stockham forward not implemented")
			}

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))
			b.ResetTimer()

			for range b.N {
				avx2Forward(dst, src, twiddle, scratch)
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
			twiddle, scratch := prepareFFTData[complex64](n)
			dst := make([]complex64, n)

			if !avx2Inverse(dst, src, twiddle, scratch) {
				b.Skip("AVX2 Stockham inverse not implemented")
			}

			b.ReportAllocs()
			b.SetBytes(int64(n * 8))
			b.ResetTimer()

			for range b.N {
				avx2Inverse(dst, src, twiddle, scratch)
			}
		})
	}
}

// =============================================================================
// Pure-Go Comparison Benchmarks
// =============================================================================

// BenchmarkPureGoForward benchmarks pure-Go forward transform for comparison.
func BenchmarkPureGoForward(b *testing.B) {
	goForward, _ := getPureGoKernels()

	sizes := []int{64, 256, 1024, 4096, 16384}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := generateRandomComplex64(n, uint64(n))
			dst := make([]complex64, n)
			twiddle, scratch := prepareFFTData[complex64](n)

			b.ResetTimer()
			b.SetBytes(int64(n * 8))

			for b.Loop() {
				goForward(dst, src, twiddle, scratch)
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
			twiddle, scratch := prepareFFTData[complex64](n)

			b.ResetTimer()
			b.SetBytes(int64(n * 8))

			for b.Loop() {
				goInverse(dst, src, twiddle, scratch)
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
			twiddle, scratch := prepareFFTData[complex64](n)

			b.Run("PureGo", func(b *testing.B) {
				b.SetBytes(int64(n * 8))

				for b.Loop() {
					goForward(dst, src, twiddle, scratch)
				}
			})

			if avx2Available {
				// Test if AVX2 is implemented
				if !avx2Forward(dst, src, twiddle, scratch) {
					b.Run("AVX2", func(b *testing.B) {
						b.Skip("AVX2 kernel not yet implemented")
					})

					return
				}

				b.Run("AVX2", func(b *testing.B) {
					b.SetBytes(int64(n * 8))

					for b.Loop() {
						avx2Forward(dst, src, twiddle, scratch)
					}
				})
			}
		})
	}
}

// =============================================================================
// Complex128 Benchmarks
// =============================================================================

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
			twiddle, scratch := prepareFFTData[complex128](n)

			if !avx2Forward(dst, src, twiddle, scratch) {
				b.Skip("AVX2 complex128 not implemented")
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 16))

			for range b.N {
				avx2Forward(dst, src, twiddle, scratch)
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
			twiddle, scratch := prepareFFTData[complex128](n)

			if !avx2Inverse(dst, src, twiddle, scratch) {
				b.Skip("AVX2 complex128 not implemented")
			}

			b.ResetTimer()
			b.SetBytes(int64(n * 16))

			for range b.N {
				avx2Inverse(dst, src, twiddle, scratch)
			}
		})
	}
}

// =============================================================================
// Size-Specific Comparison Benchmarks
// =============================================================================

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

	twiddle, scratch := prepareFFTData[complex64](n)
	dst := make([]complex64, n)

	b.Run("AVX2", func(b *testing.B) {
		if !avx2Forward(dst, src, twiddle, scratch) {
			b.Skip("AVX2 kernel not implemented")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			avx2Forward(dst, src, twiddle, scratch)
		}
	})

	b.Run("PureGo", func(b *testing.B) {
		if !goForward(dst, src, twiddle, scratch) {
			b.Skip("Pure Go kernel failed")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			goForward(dst, src, twiddle, scratch)
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

	twiddle, scratch := prepareFFTData[complex64](n)
	dst := make([]complex64, n)

	b.Run("AVX2", func(b *testing.B) {
		if !avx2Forward(dst, src, twiddle, scratch) {
			b.Skip("AVX2 kernel not implemented")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			avx2Forward(dst, src, twiddle, scratch)
		}
	})

	b.Run("PureGo", func(b *testing.B) {
		if !goForward(dst, src, twiddle, scratch) {
			b.Skip("Pure Go kernel failed")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			goForward(dst, src, twiddle, scratch)
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

	twiddle, scratch := prepareFFTData[complex64](n)
	dst := make([]complex64, n)

	b.Run("AVX2", func(b *testing.B) {
		if !avx2Forward(dst, src, twiddle, scratch) {
			b.Skip("AVX2 kernel not implemented")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			avx2Forward(dst, src, twiddle, scratch)
		}
	})

	b.Run("PureGo", func(b *testing.B) {
		if !goForward(dst, src, twiddle, scratch) {
			b.Skip("Pure Go kernel failed")
		}

		b.ResetTimer()
		b.SetBytes(int64(n * 8))

		for range b.N {
			goForward(dst, src, twiddle, scratch)
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

	twiddle := ComputeTwiddleFactors[complex64](n)

	b.Run("AVX2_Radix2", func(b *testing.B) {
		if !forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch) {
			b.Skip("AVX2 radix-2 assembly not available")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardAVX2Size256Radix2Complex64Asm(dst, src, twiddle, scratch)
		}
	})

	b.Run("AVX2_Radix4", func(b *testing.B) {
		if !forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch) {
			b.Skip("AVX2 radix-4 assembly not available")
		}
		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()
		for range b.N {
			forwardAVX2Size256Radix4Complex64Asm(dst, src, twiddle, scratch)
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
}

// BenchmarkAVX2GenericRadix4_VsRadix2 compares generic radix-4 vs radix-2 AVX2 DIT.
func BenchmarkAVX2GenericRadix4_VsRadix2(b *testing.B) {
	_, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		b.Skip("AVX2 not available")
	}

	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i)/float32(n), float32(i%4)/4)
			}

			dst := make([]complex64, n)
			twiddle, scratch := prepareFFTData[complex64](n)

			b.Run("Radix4", func(b *testing.B) {
				if !forwardAVX2Complex64Radix4Asm(dst, src, twiddle, scratch) {
					b.Skip("AVX2 radix-4 generic not available")
				}
				b.ReportAllocs()
				b.SetBytes(int64(n * 8))
				b.ResetTimer()
				for range b.N {
					forwardAVX2Complex64Radix4Asm(dst, src, twiddle, scratch)
				}
			})

			b.Run("Radix2", func(b *testing.B) {
				if !forwardAVX2Complex64GenericRadix2Asm(dst, src, twiddle, scratch) {
					b.Skip("AVX2 radix-2 generic not available")
				}
				b.ReportAllocs()
				b.SetBytes(int64(n * 8))
				b.ResetTimer()
				for range b.N {
					forwardAVX2Complex64GenericRadix2Asm(dst, src, twiddle, scratch)
				}
			})
		})
	}
}

// BenchmarkAVX2GenericRadix4Mixed_VsRadix2 compares mixed radix-4 vs radix-2 AVX2 DIT.
func BenchmarkAVX2GenericRadix4Mixed_VsRadix2(b *testing.B) {
	_, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		b.Skip("AVX2 not available")
	}

	sizes := []int{128, 512, 2048, 8192}

	for _, n := range sizes {
		b.Run(sizeString(n), func(b *testing.B) {
			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(i)/float32(n), float32(i%4)/4)
			}

			dst := make([]complex64, n)
			twiddle, scratch := prepareFFTData[complex64](n)

			b.Run("Radix4Mixed", func(b *testing.B) {
				if !forwardAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch) {
					b.Skip("AVX2 radix-4 mixed not available")
				}
				b.ReportAllocs()
				b.SetBytes(int64(n * 8))
				b.ResetTimer()
				for range b.N {
					forwardAVX2Complex64Radix4MixedAsm(dst, src, twiddle, scratch)
				}
			})

			b.Run("Radix2", func(b *testing.B) {
				if !forwardAVX2Complex64GenericRadix2Asm(dst, src, twiddle, scratch) {
					b.Skip("AVX2 radix-2 generic not available")
				}
				b.ReportAllocs()
				b.SetBytes(int64(n * 8))
				b.ResetTimer()
				for range b.N {
					forwardAVX2Complex64GenericRadix2Asm(dst, src, twiddle, scratch)
				}
			})
		})
	}
}

// =============================================================================
// Complex128 Radix-4 Generic Benchmarks
// =============================================================================

// BenchmarkAVX2GenericRadix4Complex128 benchmarks the generic radix-4 Complex128 kernel.
// Tests power-of-4 sizes (even log2): 64, 256, 1024, 4096
func BenchmarkAVX2GenericRadix4Complex128(b *testing.B) {
	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("Forward/%s", sizeString(n)), func(b *testing.B) {
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i)/float64(n), float64(i%4)/4)
			}

			dst := make([]complex128, n)
			twiddle, scratch := prepareFFTData[complex128](n)

			if !forwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch) {
				b.Skip("AVX2 radix-4 complex128 not available")
			}
			b.ReportAllocs()
			b.SetBytes(int64(n * 16))
			b.ResetTimer()
			for range b.N {
				forwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch)
			}
		})

		b.Run(fmt.Sprintf("Inverse/%s", sizeString(n)), func(b *testing.B) {
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i)/float64(n), float64(i%4)/4)
			}

			dst := make([]complex128, n)
			twiddle, scratch := prepareFFTData[complex128](n)

			if !inverseAVX2Complex128Radix4Asm(dst, src, twiddle, scratch) {
				b.Skip("AVX2 inverse radix-4 complex128 not available")
			}
			b.ReportAllocs()
			b.SetBytes(int64(n * 16))
			b.ResetTimer()
			for range b.N {
				inverseAVX2Complex128Radix4Asm(dst, src, twiddle, scratch)
			}
		})
	}
}

// BenchmarkAVX2GenericRadix4MixedComplex128 benchmarks the mixed radix-4 Complex128 kernel.
// Tests odd log2 sizes: 32, 128, 512, 2048
func BenchmarkAVX2GenericRadix4MixedComplex128(b *testing.B) {
	sizes := []int{32, 128, 512, 2048}

	for _, n := range sizes {
		b.Run(fmt.Sprintf("Forward/%s", sizeString(n)), func(b *testing.B) {
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i)/float64(n), float64(i%4)/4)
			}

			dst := make([]complex128, n)
			twiddle, scratch := prepareFFTData[complex128](n)

			if !forwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch) {
				b.Skip("AVX2 radix-4 mixed complex128 not available")
			}
			b.ReportAllocs()
			b.SetBytes(int64(n * 16))
			b.ResetTimer()
			for range b.N {
				forwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch)
			}
		})

		b.Run(fmt.Sprintf("Inverse/%s", sizeString(n)), func(b *testing.B) {
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(float64(i)/float64(n), float64(i%4)/4)
			}

			dst := make([]complex128, n)
			twiddle, scratch := prepareFFTData[complex128](n)

			if !inverseAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch) {
				b.Skip("AVX2 inverse radix-4 mixed complex128 not available")
			}
			b.ReportAllocs()
			b.SetBytes(int64(n * 16))
			b.ResetTimer()
			for range b.N {
				inverseAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch)
			}
		})
	}
}
