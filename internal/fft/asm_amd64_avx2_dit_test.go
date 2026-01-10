//go:build amd64 && asm && !purego

package fft

import (
	"testing"

	mathpkg "github.com/MeKo-Christian/algo-fft/internal/math"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// TestAVX2DITForwardComplex64 tests all AVX2 forward kernels for complex64
func TestAVX2DITForwardComplex64(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		n       int
		bitrev  func(int) []int
		forward func(dst, src, twiddle, scratch []complex64) bool
	}{
		{"Size4/Radix4", 4, ComputeBitReversalIndicesRadix4, forwardAVX2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, ComputeBitReversalIndices, forwardAVX2Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, ComputeBitReversalIndicesMixed24, forwardAVX2Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, mathpkg.ComputeIdentityIndices, forwardAVX2Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, ComputeBitReversalIndices, forwardAVX2Size16Complex64Asm},
		{"Size16/Radix4", 16, ComputeBitReversalIndicesRadix4, forwardAVX2Size16Radix4Complex64Asm},
		{"Size32/Radix2", 32, ComputeBitReversalIndices, forwardAVX2Size32Radix2Complex64Asm},
		{"Size32/Radix32", 32, mathpkg.ComputeIdentityIndices, forwardAVX2Size32Complex64Asm},
		{"Size64/Radix2", 64, ComputeBitReversalIndices, forwardAVX2Size64Complex64Asm},
		{"Size64/Radix4", 64, ComputeBitReversalIndicesRadix4, forwardAVX2Size64Radix4Complex64Asm},
		{"Size128", 128, ComputeBitReversalIndices, forwardAVX2Size128Complex64Asm},
		{"Size256/Radix2", 256, ComputeBitReversalIndices, forwardAVX2Size256Radix2Complex64Asm},
		{"Size256/Radix4", 256, ComputeBitReversalIndicesRadix4, forwardAVX2Size256Radix4Complex64Asm},
		{"Size512/Mixed24", 512, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size512/Radix2", 512, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size1024/Radix4", 1024, ComputeBitReversalIndicesRadix4, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size2048/Mixed24", 2048, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size4096/Radix4", 4096, ComputeBitReversalIndicesRadix4, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size8192/Mixed24", 8192, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(tc.n, 0xDEADBEEF+uint64(tc.n))
			dst := make([]complex64, tc.n)
			scratch := make([]complex64, tc.n)
			twiddle := ComputeTwiddleFactors[complex64](tc.n)

			if !tc.forward(dst, src, twiddle, scratch) {
				t.Fatalf("%s failed", tc.name)
			}

			want := reference.NaiveDFT(src)
			assertComplex64SliceClose(t, dst, want, tc.n)
		})
	}
}

// TestAVX2DITInverseComplex64 tests all AVX2 inverse kernels for complex64
func TestAVX2DITInverseComplex64(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		n       int
		bitrev  func(int) []int
		forward func(dst, src, twiddle, scratch []complex64) bool
		inverse func(dst, src, twiddle, scratch []complex64) bool
	}{
		{"Size4/Radix4", 4, ComputeBitReversalIndicesRadix4, forwardAVX2Size4Radix4Complex64Asm, inverseAVX2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, ComputeBitReversalIndices, forwardAVX2Size8Radix2Complex64Asm, inverseAVX2Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, ComputeBitReversalIndicesMixed24, forwardAVX2Size8Radix4Complex64Asm, inverseAVX2Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, mathpkg.ComputeIdentityIndices, forwardAVX2Size8Radix8Complex64Asm, inverseAVX2Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, ComputeBitReversalIndices, forwardAVX2Size16Complex64Asm, inverseAVX2Size16Complex64Asm},
		{"Size16/Radix4", 16, ComputeBitReversalIndicesRadix4, forwardAVX2Size16Radix4Complex64Asm, inverseAVX2Size16Radix4Complex64Asm},
		{"Size32/Radix2", 32, ComputeBitReversalIndices, forwardAVX2Size32Radix2Complex64Asm, inverseAVX2Size32Radix2Complex64Asm},
		{"Size32/Radix32", 32, mathpkg.ComputeIdentityIndices, forwardAVX2Size32Complex64Asm, inverseAVX2Size32Complex64Asm},
		{"Size64/Radix2", 64, ComputeBitReversalIndices, forwardAVX2Size64Complex64Asm, inverseAVX2Size64Complex64Asm},
		{"Size64/Radix4", 64, ComputeBitReversalIndicesRadix4, forwardAVX2Size64Radix4Complex64Asm, inverseAVX2Size64Radix4Complex64Asm},
		{"Size128", 128, ComputeBitReversalIndices, forwardAVX2Size128Complex64Asm, inverseAVX2Size128Complex64Asm},
		{"Size256/Radix2", 256, ComputeBitReversalIndices, forwardAVX2Size256Radix2Complex64Asm, inverseAVX2Size256Radix2Complex64Asm},
		{"Size256/Radix4", 256, ComputeBitReversalIndicesRadix4, forwardAVX2Size256Radix4Complex64Asm, inverseAVX2Size256Radix4Complex64Asm},
		{"Size512/Mixed24", 512, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}, func(dst, src, twiddle, scratch []complex64) bool {
			return inverseAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size512/Radix2", 512, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, nil)
		}, func(dst, src, twiddle, scratch []complex64) bool {
			return inverseAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size1024/Radix4", 1024, ComputeBitReversalIndicesRadix4, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch, nil)
		}, func(dst, src, twiddle, scratch []complex64) bool {
			return inverseAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size2048/Mixed24", 2048, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}, func(dst, src, twiddle, scratch []complex64) bool {
			return inverseAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size4096/Radix4", 4096, ComputeBitReversalIndicesRadix4, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch, nil)
		}, func(dst, src, twiddle, scratch []complex64) bool {
			return inverseAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size8192/Mixed24", 8192, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}, func(dst, src, twiddle, scratch []complex64) bool {
			return inverseAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(tc.n, 0xCAFE+uint64(tc.n))
			fwd := make([]complex64, tc.n)
			dst := make([]complex64, tc.n)
			scratch := make([]complex64, tc.n)
			twiddle := ComputeTwiddleFactors[complex64](tc.n)

			if !tc.forward(fwd, src, twiddle, scratch) {
				t.Fatalf("%s forward failed", tc.name)
			}

			if !tc.inverse(dst, fwd, twiddle, scratch) {
				t.Fatalf("%s inverse failed", tc.name)
			}

			want := reference.NaiveIDFT(fwd)
			assertComplex64SliceClose(t, dst, want, tc.n)
		})
	}
}

// TestAVX2DITForwardComplex128 tests all AVX2 forward kernels for complex128
func TestAVX2DITForwardComplex128(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		n       int
		bitrev  func(int) []int
		forward func(dst, src, twiddle, scratch []complex128, bitrev []int) bool
	}{
		{"Size4/Radix4", 4, ComputeBitReversalIndicesRadix4, forwardAVX2Size4Radix4Complex128Asm},
		{"Size8/Radix2", 8, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size8/Radix4", 8, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size8/Radix8", 8, mathpkg.ComputeIdentityIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size16", 16, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size32", 32, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size64/Radix2", 64, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size64/Radix4", 64, ComputeBitReversalIndicesRadix4, forwardAVX2Size64Radix4Complex128Asm},
		{"Size512/Radix2", 512, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size512/Mixed24", 512, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch)
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(tc.n, 0xBEEF+uint64(tc.n))
			dst := make([]complex128, tc.n)
			scratch := make([]complex128, tc.n)
			twiddle := ComputeTwiddleFactors[complex128](tc.n)

			var bitrev []int
			if tc.bitrev != nil {
				bitrev = tc.bitrev(tc.n)
			}

			if !tc.forward(dst, src, twiddle, scratch, bitrev) {
				t.Fatalf("%s failed", tc.name)
			}

			want := reference.NaiveDFT128(src)
			assertComplex128SliceClose(t, dst, want, tc.n)
		})
	}
}

// TestAVX2DITInverseComplex128 tests all AVX2 inverse kernels for complex128
func TestAVX2DITInverseComplex128(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name    string
		n       int
		bitrev  func(int) []int
		forward func(dst, src, twiddle, scratch []complex128, bitrev []int) bool
		inverse func(dst, src, twiddle, scratch []complex128, bitrev []int) bool
	}{
		{"Size4/Radix4", 4, ComputeBitReversalIndicesRadix4, forwardAVX2Size4Radix4Complex128Asm, inverseAVX2Size4Radix4Complex128Asm},
		{"Size8/Radix2", 8, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
		}, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return inverseAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size8/Radix4", 8, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
		}, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return inverseAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size8/Radix8", 8, mathpkg.ComputeIdentityIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
		}, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return inverseAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size16", 16, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch)
		}, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return inverseAVX2Size16Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size32", 32, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch)
		}, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return inverseAVX2Size32Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size64/Radix2", 64, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
		}, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return inverseAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size64/Radix4", 64, ComputeBitReversalIndicesRadix4, forwardAVX2Size64Radix4Complex128Asm, inverseAVX2Size64Radix4Complex128Asm},
		{"Size512/Radix2", 512, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
		}, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return inverseAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size512/Mixed24", 512, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch)
		}, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return inverseAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch)
		}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(tc.n, 0xFACE+uint64(tc.n))
			fwd := make([]complex128, tc.n)
			dst := make([]complex128, tc.n)
			scratch := make([]complex128, tc.n)
			twiddle := ComputeTwiddleFactors[complex128](tc.n)

			var bitrev []int
			if tc.bitrev != nil {
				bitrev = tc.bitrev(tc.n)
			}

			if !tc.forward(fwd, src, twiddle, scratch, bitrev) {
				t.Fatalf("%s forward failed", tc.name)
			}

			if !tc.inverse(dst, fwd, twiddle, scratch, bitrev) {
				t.Fatalf("%s inverse failed", tc.name)
			}

			want := reference.NaiveIDFT128(fwd)
			assertComplex128SliceClose(t, dst, want, tc.n)
		})
	}
}

// BenchmarkAVX2DITComplex64 benchmarks all AVX2 kernels for complex64
func BenchmarkAVX2DITComplex64(b *testing.B) {
	cases := []struct {
		name    string
		n       int
		bitrev  func(int) []int
		forward func(dst, src, twiddle, scratch []complex64) bool
	}{
		{"Size4/Radix4", 4, ComputeBitReversalIndicesRadix4, forwardAVX2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, ComputeBitReversalIndices, forwardAVX2Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, ComputeBitReversalIndicesMixed24, forwardAVX2Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, mathpkg.ComputeIdentityIndices, forwardAVX2Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, ComputeBitReversalIndices, forwardAVX2Size16Complex64Asm},
		{"Size16/Radix4", 16, ComputeBitReversalIndicesRadix4, forwardAVX2Size16Radix4Complex64Asm},
		{"Size32", 32, ComputeBitReversalIndices, forwardAVX2Size32Complex64Asm},
		{"Size64/Radix2", 64, ComputeBitReversalIndices, forwardAVX2Size64Complex64Asm},
		{"Size64/Radix4", 64, ComputeBitReversalIndicesRadix4, forwardAVX2Size64Radix4Complex64Asm},
		{"Size128", 128, ComputeBitReversalIndices, forwardAVX2Size128Complex64Asm},
		{"Size256/Radix2", 256, ComputeBitReversalIndices, forwardAVX2Size256Radix2Complex64Asm},
		{"Size256/Radix4", 256, ComputeBitReversalIndicesRadix4, forwardAVX2Size256Radix4Complex64Asm},
		{"Size512/Mixed24", 512, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size512Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size512/Radix2", 512, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size512Radix2Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size1024/Radix4", 1024, ComputeBitReversalIndicesRadix4, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size1024Radix4Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size2048/Mixed24", 2048, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size2048Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size4096/Radix4", 4096, ComputeBitReversalIndicesRadix4, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size4096Radix4Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
		{"Size8192/Mixed24", 8192, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex64) bool {
			return forwardAVX2Size8192Mixed24Complex64Asm(dst, src, twiddle, scratch, nil)
		}},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			src := make([]complex64, tc.n)
			dst := make([]complex64, tc.n)
			scratch := make([]complex64, tc.n)
			twiddle := ComputeTwiddleFactors[complex64](tc.n)

			for i := range src {
				src[i] = complex(float32(i), float32(-i))
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(tc.n * 8))

			for b.Loop() {
				tc.forward(dst, src, twiddle, scratch)
			}
		})
	}
}

// BenchmarkAVX2DITComplex128 benchmarks all AVX2 kernels for complex128
func BenchmarkAVX2DITComplex128(b *testing.B) {
	cases := []struct {
		name    string
		n       int
		bitrev  func(int) []int
		forward func(dst, src, twiddle, scratch []complex128, bitrev []int) bool
	}{
		{"Size4/Radix4", 4, ComputeBitReversalIndicesRadix4, forwardAVX2Size4Radix4Complex128Asm},
		{"Size8/Radix2", 8, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size8Radix2Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size8/Radix4", 8, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size8Radix4Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size8/Radix8", 8, mathpkg.ComputeIdentityIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size8Radix8Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size16", 16, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size16Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size32", 32, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size32Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size64/Radix2", 64, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size64Radix2Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size64/Radix4", 64, ComputeBitReversalIndicesRadix4, forwardAVX2Size64Radix4Complex128Asm},
		{"Size512/Radix2", 512, ComputeBitReversalIndices, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size512Radix2Complex128Asm(dst, src, twiddle, scratch)
		}},
		{"Size512/Mixed24", 512, ComputeBitReversalIndicesMixed24, func(dst, src, twiddle, scratch []complex128, _ []int) bool {
			return forwardAVX2Size512Mixed24Complex128Asm(dst, src, twiddle, scratch)
		}},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			src := make([]complex128, tc.n)
			dst := make([]complex128, tc.n)
			scratch := make([]complex128, tc.n)
			twiddle := ComputeTwiddleFactors[complex128](tc.n)

			var bitrev []int
			if tc.bitrev != nil {
				bitrev = tc.bitrev(tc.n)
			}

			for i := range src {
				src[i] = complex(float64(i), float64(-i))
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.SetBytes(int64(tc.n * 16))

			for b.Loop() {
				tc.forward(dst, src, twiddle, scratch, bitrev)
			}
		})
	}
}
