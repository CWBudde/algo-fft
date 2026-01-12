//go:build amd64 && asm && !purego

// Package fft provides comprehensive AVX2-optimized FFT tests.
// This file tests:
// - Individual kernel correctness (round-trip, DFT comparison)
// - Dispatcher layer (AVX2 vs Pure-Go)
// - Mathematical properties (Parseval, linearity)
// - Edge cases (zeros, impulse, DC, cosine)
//
// Benchmarks are in asm_amd64_avx2_bench_test.go.
package fft

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// =============================================================================
// Test Case Definitions
// =============================================================================

type avx2TestCase64 struct {
	name    string
	n       int
	forward func(dst, src, twiddle, scratch []complex64) bool
	inverse func(dst, src, twiddle, scratch []complex64) bool
}

type avx2TestCase128 struct {
	name    string
	n       int
	forward func(dst, src, twiddle, scratch []complex128) bool
	inverse func(dst, src, twiddle, scratch []complex128) bool
}

// allAVX2Cases64 returns all AVX2 complex64 kernel test cases.
func allAVX2Cases64() []avx2TestCase64 {
	return []avx2TestCase64{
		{"Size4/Radix4", 4, forwardAVX2Size4Radix4Complex64Asm, inverseAVX2Size4Radix4Complex64Asm},
		{"Size8/Radix2", 8, forwardAVX2Size8Radix2Complex64Asm, inverseAVX2Size8Radix2Complex64Asm},
		{"Size8/Radix4", 8, forwardAVX2Size8Radix4Complex64Asm, inverseAVX2Size8Radix4Complex64Asm},
		{"Size8/Radix8", 8, forwardAVX2Size8Radix8Complex64Asm, inverseAVX2Size8Radix8Complex64Asm},
		{"Size16/Radix2", 16, forwardAVX2Size16Radix2Complex64Asm, inverseAVX2Size16Radix2Complex64Asm},
		{"Size16/Radix4", 16, forwardAVX2Size16Radix4Complex64Asm, inverseAVX2Size16Radix4Complex64Asm},
		{"Size32/Radix2", 32, forwardAVX2Size32Radix2Complex64Asm, inverseAVX2Size32Radix2Complex64Asm},
		{"Size32/Radix32", 32, forwardAVX2Size32Complex64Asm, inverseAVX2Size32Complex64Asm},
		{"Size64/Radix2", 64, forwardAVX2Size64Radix2Complex64Asm, inverseAVX2Size64Radix2Complex64Asm},
		{"Size64/Radix4", 64, forwardAVX2Size64Radix4Complex64Asm, inverseAVX2Size64Radix4Complex64Asm},
		{"Size128/Mixed24", 128, forwardAVX2Size128Complex64Asm, inverseAVX2Size128Complex64Asm},
		{"Size256/Radix2", 256, forwardAVX2Size256Radix2Complex64Asm, inverseAVX2Size256Radix2Complex64Asm},
		{"Size256/Radix4", 256, forwardAVX2Size256Radix4Complex64Asm, inverseAVX2Size256Radix4Complex64Asm},
		{"Size512/Mixed24", 512, forwardAVX2Size512Mixed24Complex64Asm, inverseAVX2Size512Mixed24Complex64Asm},
		{"Size512/Radix2", 512, forwardAVX2Size512Radix2Complex64Asm, inverseAVX2Size512Radix2Complex64Asm},
		{"Size512/Radix8", 512, forwardAVX2Size512Radix8Complex64Asm, inverseAVX2Size512Radix8Complex64Asm},
		{"Size512/Radix16x32", 512, forwardAVX2Size512Radix16x32Complex64Asm, inverseAVX2Size512Radix16x32Complex64Asm},
		{"Size1024/Radix4", 1024, forwardAVX2Size1024Radix4Complex64Asm, inverseAVX2Size1024Radix4Complex64Asm},
		{"Size2048/Mixed24", 2048, forwardAVX2Size2048Mixed24Complex64Asm, inverseAVX2Size2048Mixed24Complex64Asm},
		{"Size4096/Radix4", 4096, forwardAVX2Size4096Radix4Complex64Asm, inverseAVX2Size4096Radix4Complex64Asm},
		{"Size8192/Mixed24", 8192, forwardAVX2Size8192Mixed24Complex64Asm, inverseAVX2Size8192Mixed24Complex64Asm},
	}
}

// allAVX2Cases128 returns all AVX2 complex128 kernel test cases.
func allAVX2Cases128() []avx2TestCase128 {
	return []avx2TestCase128{
		{"Size4/Radix4", 4, forwardAVX2Size4Radix4Complex128Asm, inverseAVX2Size4Radix4Complex128Asm},
		{"Size8/Radix2", 8, forwardAVX2Size8Radix2Complex128Asm, inverseAVX2Size8Radix2Complex128Asm},
		{"Size8/Radix4", 8, forwardAVX2Size8Radix4Complex128Asm, inverseAVX2Size8Radix4Complex128Asm},
		{"Size8/Radix8", 8, forwardAVX2Size8Radix8Complex128Asm, inverseAVX2Size8Radix8Complex128Asm},
		{"Size16/Radix2", 16, forwardAVX2Size16Radix2Complex128Asm, inverseAVX2Size16Radix2Complex128Asm},
		{"Size32/Radix2", 32, forwardAVX2Size32Complex128Asm, inverseAVX2Size32Complex128Asm},
		{"Size64/Radix2", 64, forwardAVX2Size64Radix2Complex128Asm, inverseAVX2Size64Radix2Complex128Asm},
		{"Size64/Radix4", 64, forwardAVX2Size64Radix4Complex128Asm, inverseAVX2Size64Radix4Complex128Asm},
		{"Size512/Radix2", 512, forwardAVX2Size512Radix2Complex128Asm, inverseAVX2Size512Radix2Complex128Asm},
		{"Size512/Mixed24", 512, forwardAVX2Size512Mixed24Complex128Asm, inverseAVX2Size512Mixed24Complex128Asm},
	}
}

// =============================================================================
// Test Helpers
// =============================================================================

func complexSliceEqual(a, b []complex64, relTol float32) bool {
	if len(a) != len(b) {
		return false
	}

	for i := range a {
		if !complexNearEqual(a[i], b[i], relTol) {
			return false
		}
	}

	return true
}

func complexNearEqual(a, b complex64, relTol float32) bool {
	diff := a - b
	diffMag := float32(real(diff)*real(diff) + imag(diff)*imag(diff))

	bMag := float32(real(b)*real(b) + imag(b)*imag(b))

	if bMag > 1e-10 {
		return diffMag <= relTol*relTol*bMag
	}

	return diffMag <= relTol*relTol
}

func getToleranceForSize(n int) float32 {
	switch {
	case n <= 16:
		return 1e-6
	case n <= 64:
		return 2e-6
	case n <= 256:
		return 5e-6
	case n <= 1024:
		return 1e-5
	case n <= 4096:
		return 2e-5
	case n <= 8192:
		return 3e-5
	case n <= 16384:
		return 5e-5
	default:
		return 1e-4
	}
}

func getToleranceForSize128(n int) float64 {
	switch {
	case n <= 16:
		return 1e-12
	case n <= 64:
		return 5e-12
	case n <= 256:
		return 1e-11
	case n <= 1024:
		return 5e-11
	case n <= 4096:
		return 1e-10
	case n <= 8192:
		return 5e-10
	case n <= 16384:
		return 1e-9
	default:
		return 5e-9
	}
}

func generateRandomComplex64(n int, seed uint64) []complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF))
	result := make([]complex64, n)

	for i := range result {
		re := rng.Float32()*2 - 1
		im := rng.Float32()*2 - 1
		result[i] = complex(re, im)
	}

	return result
}

func generateImpulse(n int) []complex64 {
	result := make([]complex64, n)
	result[0] = 1

	return result
}

func generateDC(n int, value complex64) []complex64 {
	result := make([]complex64, n)
	for i := range result {
		result[i] = value
	}

	return result
}

func generateCosine(n int, freqBin int) []complex64 {
	result := make([]complex64, n)
	for i := range result {
		angle := 2 * math.Pi * float64(freqBin) * float64(i) / float64(n)
		result[i] = complex(float32(math.Cos(angle)), 0)
	}

	return result
}

func computeEnergy(x []complex64) float64 {
	var energy float64
	for _, v := range x {
		re, im := float64(real(v)), float64(imag(v))
		energy += re*re + im*im
	}

	return energy
}

func prepareFFTData[T Complex](n int) ([]T, []T) {
	twiddle := ComputeTwiddleFactors[T](n)
	scratch := make([]T, n)

	return twiddle, scratch
}

// =============================================================================
// AVX2 Kernel Access Functions
// =============================================================================

//nolint:nonamedreturns
func getAVX2Kernels() (forward, inverse Kernel[complex64], available bool) {
	const archAMD64 = "amd64"
	if runtime.GOARCH != archAMD64 {
		return nil, nil, false
	}

	features := cpu.DetectFeatures()
	if !features.HasAVX2 {
		return nil, nil, false
	}

	return forwardAVX2Complex64, inverseAVX2Complex64, true
}

//nolint:nonamedreturns
func getAVX2StockhamKernels() (forward, inverse Kernel[complex64], available bool) {
	const archAMD64 = "amd64"
	if runtime.GOARCH != archAMD64 {
		return nil, nil, false
	}

	features := cpu.DetectFeatures()
	if !features.HasAVX2 {
		return nil, nil, false
	}

	return forwardAVX2StockhamComplex64, inverseAVX2StockhamComplex64, true
}

//nolint:nonamedreturns
func getPureGoKernels() (forward, inverse Kernel[complex64]) {
	return forwardDITComplex64, inverseDITComplex64
}

//nolint:nonamedreturns
func getAVX2Kernels128() (forward, inverse Kernel[complex128], available bool) {
	const archAMD64 = "amd64"
	if runtime.GOARCH != archAMD64 {
		return nil, nil, false
	}

	features := cpu.DetectFeatures()
	if !features.HasAVX2 {
		return nil, nil, false
	}

	return forwardAVX2Complex128, inverseAVX2Complex128, true
}

//nolint:nonamedreturns
func getPureGoKernels128() (forward, inverse Kernel[complex128]) {
	return forwardDITComplex128, inverseDITComplex128
}

// =============================================================================
// Kernel Round-Trip Tests
// =============================================================================

func TestAVX2RoundTripComplex64(t *testing.T) {
	t.Parallel()

	cases := allAVX2Cases64()

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(tc.n, 0xABCD01+uint64(tc.n))
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

			assertComplex64SliceClose(t, dst, src, tc.n)
		})
	}
}

func TestAVX2RoundTripComplex128(t *testing.T) {
	t.Parallel()

	cases := allAVX2Cases128()

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(tc.n, 0xA7128+uint64(tc.n))
			fwd := make([]complex128, tc.n)
			dst := make([]complex128, tc.n)
			scratch := make([]complex128, tc.n)
			twiddle := ComputeTwiddleFactors[complex128](tc.n)

			if !tc.forward(fwd, src, twiddle, scratch) {
				t.Fatalf("%s forward failed", tc.name)
			}

			if !tc.inverse(dst, fwd, twiddle, scratch) {
				t.Fatalf("%s inverse failed", tc.name)
			}

			assertComplex128SliceClose(t, dst, src, tc.n)
		})
	}
}

// =============================================================================
// Kernel vs DFT Tests
// =============================================================================

func TestAVX2ForwardVsDFTComplex64(t *testing.T) {
	t.Parallel()

	cases := allAVX2Cases64()

	for _, tc := range cases {
		if tc.n >= 8192 {
			continue
		}

		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(tc.n, 0xDF064+uint64(tc.n))
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

func TestAVX2InverseVsIDFTComplex64(t *testing.T) {
	t.Parallel()

	cases := allAVX2Cases64()

	for _, tc := range cases {
		if tc.n >= 8192 {
			continue
		}

		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(tc.n, 0x1DF064+uint64(tc.n))
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

func TestAVX2ForwardVsDFTComplex128(t *testing.T) {
	t.Parallel()

	cases := allAVX2Cases128()

	for _, tc := range cases {
		if tc.n >= 8192 {
			continue
		}

		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(tc.n, 0xDF128+uint64(tc.n))
			dst := make([]complex128, tc.n)
			scratch := make([]complex128, tc.n)
			twiddle := ComputeTwiddleFactors[complex128](tc.n)

			if !tc.forward(dst, src, twiddle, scratch) {
				t.Fatalf("%s failed", tc.name)
			}

			want := reference.NaiveDFT128(src)
			assertComplex128SliceClose(t, dst, want, tc.n)
		})
	}
}

func TestAVX2InverseVsIDFTComplex128(t *testing.T) {
	t.Parallel()

	cases := allAVX2Cases128()

	for _, tc := range cases {
		if tc.n >= 8192 {
			continue
		}

		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(tc.n, 0x1DF128+uint64(tc.n))
			fwd := make([]complex128, tc.n)
			dst := make([]complex128, tc.n)
			scratch := make([]complex128, tc.n)
			twiddle := ComputeTwiddleFactors[complex128](tc.n)

			if !tc.forward(fwd, src, twiddle, scratch) {
				t.Fatalf("%s forward failed", tc.name)
			}

			if !tc.inverse(dst, fwd, twiddle, scratch) {
				t.Fatalf("%s inverse failed", tc.name)
			}

			want := reference.NaiveIDFT128(fwd)
			assertComplex128SliceClose(t, dst, want, tc.n)
		})
	}
}

// =============================================================================
// AVX2 Dispatcher vs Pure-Go Tests
// =============================================================================

func TestAVX2Forward_VsPureGo(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	goForward, _ := getPureGoKernels()

	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64(n, uint64(uint(n)))
			twiddle, scratch := prepareFFTData[complex64](n)

			dstGo := make([]complex64, n)
			if !goForward(dstGo, src, twiddle, scratch) {
				t.Fatal("Pure-Go forward kernel failed")
			}

			dstAVX2 := make([]complex64, n)
			scratchAVX2 := make([]complex64, n)
			avx2Handled := avx2Forward(dstAVX2, src, twiddle, scratchAVX2)

			if !avx2Handled {
				t.Skip("AVX2 kernel returned false (not yet implemented)")
			}

			relTol := getToleranceForSize(n)
			if !complexSliceEqual(dstAVX2, dstGo, relTol) {
				t.Errorf("AVX2 forward result differs from pure-Go")
				reportDifferences(t, dstAVX2, dstGo, relTol)
			}
		})
	}
}

func TestAVX2Inverse_VsPureGo(t *testing.T) {
	t.Parallel()

	_, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	_, goInverse := getPureGoKernels()

	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64(n, uint64(uint(n))+1000)
			twiddle, scratch := prepareFFTData[complex64](n)

			dstGo := make([]complex64, n)
			if !goInverse(dstGo, src, twiddle, scratch) {
				t.Fatal("Pure-Go inverse kernel failed")
			}

			dstAVX2 := make([]complex64, n)
			scratchAVX2 := make([]complex64, n)
			avx2Handled := avx2Inverse(dstAVX2, src, twiddle, scratchAVX2)

			if !avx2Handled {
				t.Skip("AVX2 kernel returned false (not yet implemented)")
			}

			relTol := getToleranceForSize(n) * 2
			if !complexSliceEqual(dstAVX2, dstGo, relTol) {
				t.Errorf("AVX2 inverse result differs from pure-Go")
				reportDifferences(t, dstAVX2, dstGo, relTol)
			}
		})
	}
}

// =============================================================================
// AVX2 Stockham vs Pure-Go Stockham Tests
// =============================================================================

func TestAVX2StockhamForward_VsPureGo(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2StockhamKernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	sizes := []int{16, 32, 64, 128, 256, 1024, 2048, 8192}
	relTol := float32(1e-4)

	for _, n := range sizes {
		src := generateRandomComplex64(n, 0xABCDEF01+uint64(uint(n)))
		twiddle, scratch := prepareFFTData[complex64](n)

		dstAVX2 := make([]complex64, n)
		dstGo := make([]complex64, n)

		handled := avx2Forward(dstAVX2, src, twiddle, scratch)
		if !handled {
			t.Skip("AVX2 Stockham forward not implemented")
		}

		if !forwardStockhamComplex64(dstGo, src, twiddle, scratch) {
			t.Fatal("pure-Go Stockham forward failed")
		}

		if !complexSliceEqual(dstAVX2, dstGo, relTol) {
			t.Errorf("AVX2 Stockham forward differs from pure-Go (n=%d)", n)
		}
	}
}

func TestAVX2StockhamInverse_VsPureGo(t *testing.T) {
	t.Parallel()

	_, avx2Inverse, avx2Available := getAVX2StockhamKernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	sizes := []int{16, 32, 64, 128, 256, 1024, 2048, 8192}
	relTol := float32(1e-4)

	for _, n := range sizes {
		src := generateRandomComplex64(n, 0x12345678+uint64(n))
		twiddle, scratch := prepareFFTData[complex64](n)

		dstAVX2 := make([]complex64, n)
		dstGo := make([]complex64, n)

		handled := avx2Inverse(dstAVX2, src, twiddle, scratch)
		if !handled {
			t.Skip("AVX2 Stockham inverse not implemented")
		}

		if !inverseStockhamComplex64(dstGo, src, twiddle, scratch) {
			t.Fatal("pure-Go Stockham inverse failed")
		}

		if !complexSliceEqual(dstAVX2, dstGo, relTol) {
			t.Errorf("AVX2 Stockham inverse differs from pure-Go (n=%d)", n)
		}
	}
}

// =============================================================================
// AVX2 Dispatcher vs Reference DFT Tests
// =============================================================================

func TestAVX2VsReferenceDFT(t *testing.T) {
	t.Parallel()

	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	sizes := []int{16, 32, 64, 128, 256, 512, 1024}

	t.Run("Forward", func(t *testing.T) {
		t.Parallel()
		testAVX2VsReference(t, sizes, avx2Forward, reference.NaiveDFT, 2000, "forward")
	})

	t.Run("Inverse", func(t *testing.T) {
		t.Parallel()
		testAVX2VsReference(t, sizes, avx2Inverse, reference.NaiveIDFT, 3000, "inverse")
	})
}

func testAVX2VsReference(
	t *testing.T,
	sizes []int,
	avx2Kernel Kernel[complex64],
	refKernel func([]complex64) []complex64,
	seedOffset uint64,
	name string,
) {
	t.Helper()

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64(n, uint64(n)+seedOffset)
			twiddle, scratch := prepareFFTData[complex64](n)

			dstRef := refKernel(src)

			dstAVX2 := make([]complex64, n)
			avx2Handled := avx2Kernel(dstAVX2, src, twiddle, scratch)

			if !avx2Handled {
				t.Skip("AVX2 kernel returned false (not yet implemented)")
			}

			relTol := getToleranceForSize(n)
			if !complexSliceEqual(dstAVX2, dstRef, relTol) {
				t.Errorf("AVX2 %s result differs from reference", name)
				reportDifferences(t, dstAVX2, dstRef, relTol)
			}
		})
	}
}

func reportDifferences(t *testing.T, got, want []complex64, relTol float32) {
	t.Helper()

	count := 0

	for i := range got {
		if !complexNearEqual(got[i], want[i], relTol) {
			t.Errorf("  [%d]: got=%v, want=%v", i, got[i], want[i])

			count++
			if count >= 5 {
				t.Errorf("  ... (more differences)")
				break
			}
		}
	}
}

// =============================================================================
// AVX2 Dispatcher Round-Trip Tests
// =============================================================================

func TestAVX2RoundTrip(t *testing.T) {
	t.Parallel()

	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			original := generateRandomComplex64(n, uint64(n)+4000)
			twiddle, scratch := prepareFFTData[complex64](n)

			freq := make([]complex64, n)
			if !avx2Forward(freq, original, twiddle, scratch) {
				t.Skip("AVX2 forward kernel not yet implemented")
			}

			recovered := make([]complex64, n)
			scratch2 := make([]complex64, n)
			if !avx2Inverse(recovered, freq, twiddle, scratch2) {
				t.Skip("AVX2 inverse kernel not yet implemented")
			}

			relTol := getToleranceForSize(n)
			if !complexSliceEqual(recovered, original, relTol) {
				t.Errorf("Round-trip failed: IFFT(FFT(x)) != x")
				reportDifferences(t, recovered, original, relTol)
			}
		})
	}
}

// =============================================================================
// Property Tests (Mathematical Properties)
// =============================================================================

func TestAVX2Parseval(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	sizes := []int{16, 64, 256, 1024, 2048, 8192}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64(n, uint64(n)+5000)
			twiddle, scratch := prepareFFTData[complex64](n)

			energyTime := computeEnergy(src)

			freq := make([]complex64, n)
			if !avx2Forward(freq, src, twiddle, scratch) {
				t.Skip("AVX2 kernel not yet implemented")
			}

			energyFreq := computeEnergy(freq) / float64(n)

			relError := math.Abs(energyTime-energyFreq) / energyTime
			if relError > 1e-5 {
				t.Errorf("Parseval's theorem violated: time=%.10f, freq=%.10f, relError=%.2e",
					energyTime, energyFreq, relError)
			}
		})
	}
}

func TestAVX2Linearity(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	sizes := []int{16, 64, 256, 1024, 2048, 8192}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			x := generateRandomComplex64(n, uint64(n)+6000)
			y := generateRandomComplex64(n, uint64(n)+7000)
			a := complex(float32(0.7), float32(0.3))
			b := complex(float32(-0.4), float32(0.5))

			twiddle, scratch := prepareFFTData[complex64](n)

			combined := make([]complex64, n)
			for i := range combined {
				combined[i] = a*x[i] + b*y[i]
			}

			fftCombined := make([]complex64, n)
			if !avx2Forward(fftCombined, combined, twiddle, scratch) {
				t.Skip("AVX2 kernel not yet implemented")
			}

			fftX := make([]complex64, n)
			scratch2 := make([]complex64, n)
			if !avx2Forward(fftX, x, twiddle, scratch2) {
				t.Skip("AVX2 kernel not yet implemented")
			}

			fftY := make([]complex64, n)
			scratch3 := make([]complex64, n)
			if !avx2Forward(fftY, y, twiddle, scratch3) {
				t.Skip("AVX2 kernel not yet implemented")
			}

			linearCombination := make([]complex64, n)
			for i := range linearCombination {
				linearCombination[i] = a*fftX[i] + b*fftY[i]
			}

			const relTol = 1e-4
			if !complexSliceEqual(fftCombined, linearCombination, relTol) {
				t.Error("Linearity violated: FFT(a*x + b*y) != a*FFT(x) + b*FFT(y)")
			}
		})
	}
}

// =============================================================================
// Edge Case Tests
// =============================================================================

//nolint:gocognit
func TestAVX2EdgeCases(t *testing.T) {
	t.Parallel()

	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	t.Run("AllZeros", func(t *testing.T) {
		t.Parallel()

		n := 64
		src := make([]complex64, n)
		twiddle, scratch := prepareFFTData[complex64](n)

		dst := make([]complex64, n)
		if !avx2Forward(dst, src, twiddle, scratch) {
			t.Skip("AVX2 kernel not yet implemented")
		}

		for i, v := range dst {
			if real(v) != 0 || imag(v) != 0 {
				t.Errorf("FFT(zeros)[%d] = %v, expected 0", i, v)
			}
		}
	})

	t.Run("Impulse", func(t *testing.T) {
		t.Parallel()

		n := 64
		src := generateImpulse(n)
		twiddle, scratch := prepareFFTData[complex64](n)

		dst := make([]complex64, n)
		if !avx2Forward(dst, src, twiddle, scratch) {
			t.Skip("AVX2 kernel not yet implemented")
		}

		const tol float32 = 1e-6
		for i, v := range dst {
			if !complexNearEqual(v, 1, tol) {
				t.Errorf("FFT(impulse)[%d] = %v, expected 1", i, v)
			}
		}
	})

	t.Run("DC", func(t *testing.T) {
		t.Parallel()

		n := 64
		dcValue := complex(float32(3.5), float32(-2.1))
		src := generateDC(n, dcValue)
		twiddle, scratch := prepareFFTData[complex64](n)

		dst := make([]complex64, n)
		if !avx2Forward(dst, src, twiddle, scratch) {
			t.Skip("AVX2 kernel not yet implemented")
		}

		expectedDC := complex(float32(n), 0) * dcValue

		const tol float32 = 1e-5

		if !complexNearEqual(dst[0], expectedDC, tol) {
			t.Errorf("FFT(DC)[0] = %v, expected %v", dst[0], expectedDC)
		}

		for i := 1; i < n; i++ {
			if !complexNearEqual(dst[i], 0, tol) {
				t.Errorf("FFT(DC)[%d] = %v, expected 0", i, dst[i])
			}
		}
	})

	t.Run("Cosine", func(t *testing.T) {
		t.Parallel()

		n := 64
		freqBin := 5
		src := generateCosine(n, freqBin)
		twiddle, scratch := prepareFFTData[complex64](n)

		dst := make([]complex64, n)
		if !avx2Forward(dst, src, twiddle, scratch) {
			t.Skip("AVX2 kernel not yet implemented")
		}

		const tol float32 = 1e-4

		expectedMag := float32(n) / 2

		gotMag := float32(math.Sqrt(float64(real(dst[freqBin])*real(dst[freqBin]) +
			imag(dst[freqBin])*imag(dst[freqBin]))))
		if math.Abs(float64(gotMag-expectedMag)) > float64(tol*expectedMag) {
			t.Errorf("FFT(cos)[%d] magnitude = %v, expected ~%v", freqBin, gotMag, expectedMag)
		}

		negBin := n - freqBin

		gotMagNeg := float32(math.Sqrt(float64(real(dst[negBin])*real(dst[negBin]) +
			imag(dst[negBin])*imag(dst[negBin]))))
		if math.Abs(float64(gotMagNeg-expectedMag)) > float64(tol*expectedMag) {
			t.Errorf("FFT(cos)[%d] magnitude = %v, expected ~%v", negBin, gotMagNeg, expectedMag)
		}

		for i := range dst {
			if i == freqBin || i == negBin {
				continue
			}

			mag := float32(math.Sqrt(float64(real(dst[i])*real(dst[i]) +
				imag(dst[i])*imag(dst[i]))))
			if mag > 1e-3 {
				t.Errorf("FFT(cos)[%d] magnitude = %v, expected ~0", i, mag)
			}
		}
	})

	t.Run("InverseUndoesForward", func(t *testing.T) {
		t.Parallel()

		n := 128
		original := generateRandomComplex64(n, 99999)
		twiddle, scratch := prepareFFTData[complex64](n)

		freq := make([]complex64, n)
		if !avx2Forward(freq, original, twiddle, scratch) {
			t.Skip("AVX2 forward not yet implemented")
		}

		recovered := make([]complex64, n)
		scratch2 := make([]complex64, n)
		if !avx2Inverse(recovered, freq, twiddle, scratch2) {
			t.Skip("AVX2 inverse not yet implemented")
		}

		relTol := getToleranceForSize(n)
		if !complexSliceEqual(recovered, original, relTol) {
			t.Error("Inverse did not undo forward transform")
		}
	})
}

// =============================================================================
// Size Validation Tests
// =============================================================================

func TestAVX2ReturnsFailureForInvalidSizes(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	t.Run("TooSmall", func(t *testing.T) {
		t.Parallel()

		for _, n := range []int{1, 2, 4, 8} {
			src := make([]complex64, n)
			dst := make([]complex64, n)
			twiddle, scratch := prepareFFTData[complex64](n)

			if avx2Forward(dst, src, twiddle, scratch) {
				continue
			}
		}
	})

	t.Run("NonPowerOfTwo", func(t *testing.T) {
		t.Parallel()

		for _, n := range []int{17, 31, 100} {
			src := make([]complex64, n)
			dst := make([]complex64, n)
			twiddle := make([]complex64, n)
			scratch := make([]complex64, n)

			handled := avx2Forward(dst, src, twiddle, scratch)
			if handled {
				t.Errorf("AVX2 kernel should return false for non-power-of-2 size %d", n)
			}
		}
	})
}

// =============================================================================
// Allocation Tests
// =============================================================================

func TestAVX2ZeroAllocations(t *testing.T) { //nolint:paralleltest
	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	n := 1024
	src := generateRandomComplex64(n, 12345)
	dst := make([]complex64, n)
	twiddle, scratch := prepareFFTData[complex64](n)

	if !avx2Forward(dst, src, twiddle, scratch) {
		t.Skip("AVX2 forward kernel not yet implemented")
	}

	allocs := testing.AllocsPerRun(100, func() {
		avx2Forward(dst, src, twiddle, scratch)
	})

	if allocs != 0 {
		t.Errorf("AVX2 forward kernel allocated %v times, expected 0", allocs)
	}

	if !avx2Inverse(dst, src, twiddle, scratch) {
		t.Skip("AVX2 inverse kernel not yet implemented")
	}

	allocs = testing.AllocsPerRun(100, func() {
		avx2Inverse(dst, src, twiddle, scratch)
	})

	if allocs != 0 {
		t.Errorf("AVX2 inverse kernel allocated %v times, expected 0", allocs)
	}
}

// =============================================================================
// Complex128 Dispatcher Tests
// =============================================================================

func TestAVX2Forward128_VsPureGo(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels128()
	if !avx2Available {
		t.Skip("AVX2 not available")
	}

	goForward, _ := getPureGoKernels128()

	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := make([]complex128, n)

			rng := rand.New(rand.NewPCG(uint64(uint(n)), 1))
			for i := range src {
				src[i] = complex(rng.Float64(), rng.Float64())
			}

			twiddle, scratch := prepareFFTData[complex128](n)

			dstGo := make([]complex128, n)
			if !goForward(dstGo, src, twiddle, scratch) {
				t.Fatal("Pure-Go failed")
			}

			dstAVX2 := make([]complex128, n)
			scratchAVX2 := make([]complex128, n)
			if !avx2Forward(dstAVX2, src, twiddle, scratchAVX2) {
				t.Skip("AVX2 complex128 forward not implemented")
			}

			tol := getToleranceForSize128(n)
			for i := range dstAVX2 {
				if cmplx.Abs(dstAVX2[i]-dstGo[i]) > tol {
					t.Errorf("Mismatch at %d: AVX2=%v, Go=%v (tol=%v)", i, dstAVX2[i], dstGo[i], tol)
					break
				}
			}
		})
	}
}

func TestAVX2Inverse128_VsPureGo(t *testing.T) {
	t.Parallel()

	_, avx2Inverse, avx2Available := getAVX2Kernels128()
	if !avx2Available {
		t.Skip("AVX2 not available")
	}

	_, goInverse := getPureGoKernels128()

	sizes := []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := make([]complex128, n)

			rng := rand.New(rand.NewPCG(uint64(uint(n)), 2))
			for i := range src {
				src[i] = complex(rng.Float64(), rng.Float64())
			}

			twiddle, scratch := prepareFFTData[complex128](n)

			dstGo := make([]complex128, n)
			if !goInverse(dstGo, src, twiddle, scratch) {
				t.Fatal("Pure-Go failed")
			}

			dstAVX2 := make([]complex128, n)
			scratchAVX2 := make([]complex128, n)
			if !avx2Inverse(dstAVX2, src, twiddle, scratchAVX2) {
				t.Skip("AVX2 complex128 inverse not implemented")
			}

			tol := getToleranceForSize128(n)
			for i := range dstAVX2 {
				if cmplx.Abs(dstAVX2[i]-dstGo[i]) > tol {
					t.Errorf("Mismatch at %d: AVX2=%v, Go=%v (tol=%v)", i, dstAVX2[i], dstGo[i], tol)
					break
				}
			}
		})
	}
}

// =============================================================================
// Helpers
// =============================================================================

func sizeString(n int) string {
	switch {
	case n >= 1024*1024:
		return formatNumber(n/(1024*1024)) + "M"
	case n >= 1024:
		return formatNumber(n/1024) + "K"
	default:
		return formatNumber(n)
	}
}

func formatNumber(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}

	if n < 100 {
		return string(rune('0'+n/10)) + string(rune('0'+n%10))
	}

	if n < 1000 {
		return string(rune('0'+n/100)) + string(rune('0'+(n/10)%10)) + string(rune('0'+n%10))
	}

	return formatNumber(n/1000) + formatNumber(n%1000)
}
