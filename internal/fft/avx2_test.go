//go:build amd64

package fft

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/MeKo-Christian/algo-fft/internal/cpu"
	"github.com/MeKo-Christian/algo-fft/internal/reference"
)

// =============================================================================
// Test Helpers
// =============================================================================

// complexSliceEqual compares two complex64 slices within a relative tolerance.
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

// complexNearEqual checks if two complex64 values are approximately equal.
func complexNearEqual(a, b complex64, relTol float32) bool {
	diff := a - b
	diffMag := float32(real(diff)*real(diff) + imag(diff)*imag(diff))

	bMag := float32(real(b)*real(b) + imag(b)*imag(b))

	// Use relative tolerance for large values, absolute for small
	if bMag > 1e-10 {
		return diffMag <= relTol*relTol*bMag
	}

	return diffMag <= relTol*relTol
}

// generateRandomComplex64 creates a slice of random complex64 values.
func generateRandomComplex64(n int, seed uint64) []complex64 {
	rng := rand.New(rand.NewPCG(seed, seed^0xDEADBEEF))
	result := make([]complex64, n)

	for i := range result {
		re := rng.Float32()*2 - 1 // [-1, 1]
		im := rng.Float32()*2 - 1
		result[i] = complex(re, im)
	}

	return result
}

// generateImpulse creates a unit impulse signal.
func generateImpulse(n int) []complex64 {
	result := make([]complex64, n)
	result[0] = 1

	return result
}

// generateDC creates a constant (DC) signal.
func generateDC(n int, value complex64) []complex64 {
	result := make([]complex64, n)
	for i := range result {
		result[i] = value
	}

	return result
}

// generateCosine creates a cosine signal at a given frequency bin.
func generateCosine(n int, freqBin int) []complex64 {
	result := make([]complex64, n)
	for i := range result {
		angle := 2 * math.Pi * float64(freqBin) * float64(i) / float64(n)
		result[i] = complex(float32(math.Cos(angle)), 0)
	}

	return result
}

// computeEnergy returns the sum of squared magnitudes.
func computeEnergy(x []complex64) float64 {
	var energy float64
	for _, v := range x {
		re, im := float64(real(v)), float64(imag(v))
		energy += re*re + im*im
	}

	return energy
}

// prepareFFTData creates twiddle factors, bit-reversal indices, and scratch buffer.
func prepareFFTData(n int) ([]complex64, []int, []complex64) {
	twiddle := ComputeTwiddleFactors[complex64](n)
	bitrev := ComputeBitReversalIndices(n)
	scratch := make([]complex64, n)

	return twiddle, bitrev, scratch
}

// =============================================================================
// AVX2 Kernel Access Functions
// =============================================================================

// getAVX2Kernels returns the AVX2 kernels if available, or nil otherwise.
// This directly tests the AVX2 path, bypassing fallback mechanisms.
//
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

	// Return the raw AVX2 kernels (no fallback). This is important for tests that
	// validate the AVX2 kernels reject unsupported sizes.
	return forwardAVX2Complex64, inverseAVX2Complex64, true
}

// getAVX2StockhamKernels returns the AVX2 Stockham kernels if available.
//
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

// getPureGoKernels returns the pure-Go DIT kernels for comparison.
//
//nolint:nonamedreturns
func getPureGoKernels() (forward, inverse Kernel[complex64]) {
	return forwardDITComplex64, inverseDITComplex64
}

// =============================================================================
// 14.1.1: AVX2 vs Pure-Go DIT Tests
// =============================================================================

func TestAVX2Forward_VsPureGo(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	goForward, _ := getPureGoKernels()

	sizes := []int{16, 32, 64, 128, 256, 512, 1024, 2048}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64(n, uint64(uint(n)))
			twiddle, bitrev, scratch := prepareFFTData(n)

			// Compute with pure-Go (ground truth for this test)
			dstGo := make([]complex64, n)
			if !goForward(dstGo, src, twiddle, scratch, bitrev) {
				t.Fatal("Pure-Go forward kernel failed")
			}

			// Compute with AVX2
			dstAVX2 := make([]complex64, n)
			scratchAVX2 := make([]complex64, n)
			avx2Handled := avx2Forward(dstAVX2, src, twiddle, scratchAVX2, bitrev)

			if !avx2Handled {
				t.Skip("AVX2 kernel returned false (not yet implemented)")
			}

			// Compare results
			// Note: AVX2 and pure-Go may have small numerical differences due to
			// different instruction ordering and FMA. We use a slightly looser tolerance
			// since the reference DFT and round-trip tests validate correctness.
			const relTol = 1e-5
			if !complexSliceEqual(dstAVX2, dstGo, relTol) {
				t.Errorf("AVX2 forward result differs from pure-Go")

				for i := range dstAVX2 {
					if !complexNearEqual(dstAVX2[i], dstGo[i], relTol) {
						t.Errorf("  [%d]: AVX2=%v, Go=%v", i, dstAVX2[i], dstGo[i])

						if i >= 5 {
							t.Errorf("  ... (more differences)")
							break
						}
					}
				}
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

	sizes := []int{16, 32, 64, 128, 256, 512, 1024, 2048}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			// Use frequency-domain data as input
			src := generateRandomComplex64(n, uint64(uint(n))+1000)
			twiddle, bitrev, scratch := prepareFFTData(n)

			// Compute with pure-Go
			dstGo := make([]complex64, n)
			if !goInverse(dstGo, src, twiddle, scratch, bitrev) {
				t.Fatal("Pure-Go inverse kernel failed")
			}

			// Compute with AVX2
			dstAVX2 := make([]complex64, n)
			scratchAVX2 := make([]complex64, n)
			avx2Handled := avx2Inverse(dstAVX2, src, twiddle, scratchAVX2, bitrev)

			if !avx2Handled {
				t.Skip("AVX2 kernel returned false (not yet implemented)")
			}

			// Compare results
			// Note: AVX2 and pure-Go may have small numerical differences due to
			// different instruction ordering. We use a slightly looser tolerance
			// since the round-trip test validates overall correctness.
			const relTol = 2e-5
			if !complexSliceEqual(dstAVX2, dstGo, relTol) {
				t.Errorf("AVX2 inverse result differs from pure-Go")

				for i := range dstAVX2 {
					if !complexNearEqual(dstAVX2[i], dstGo[i], relTol) {
						t.Errorf("  [%d]: AVX2=%v, Go=%v", i, dstAVX2[i], dstGo[i])

						if i >= 5 {
							t.Errorf("  ... (more differences)")
							break
						}
					}
				}
			}
		})
	}
}

// 14.3: AVX2 Stockham vs Pure-Go Stockham Tests.
func TestAVX2StockhamForward_VsPureGo(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2StockhamKernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	sizes := []int{16, 32, 64, 128, 256, 1024}
	relTol := float32(1e-4)

	for _, n := range sizes {
		src := generateRandomComplex64(n, 0xABCDEF01+uint64(uint(n)))
		twiddle, bitrev, scratch := prepareFFTData(n)

		dstAVX2 := make([]complex64, n)
		dstGo := make([]complex64, n)

		handled := avx2Forward(dstAVX2, src, twiddle, scratch, bitrev)
		if !handled {
			t.Skip("AVX2 Stockham forward not implemented")
		}

		if !forwardStockhamComplex64(dstGo, src, twiddle, scratch, bitrev) {
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

	sizes := []int{16, 32, 64, 128, 256, 1024}
	relTol := float32(1e-4)

	for _, n := range sizes {
		src := generateRandomComplex64(n, 0x12345678+uint64(n))
		twiddle, bitrev, scratch := prepareFFTData(n)

		dstAVX2 := make([]complex64, n)
		dstGo := make([]complex64, n)

		handled := avx2Inverse(dstAVX2, src, twiddle, scratch, bitrev)
		if !handled {
			t.Skip("AVX2 Stockham inverse not implemented")
		}

		if !inverseStockhamComplex64(dstGo, src, twiddle, scratch, bitrev) {
			t.Fatal("pure-Go Stockham inverse failed")
		}

		if !complexSliceEqual(dstAVX2, dstGo, relTol) {
			t.Errorf("AVX2 Stockham inverse differs from pure-Go (n=%d)", n)
		}
	}
}

// =============================================================================
// 14.1.1: Correctness Tests Against Reference DFT
// =============================================================================

func TestAVX2VsReferenceDFT(t *testing.T) {
	t.Parallel()

	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	// Smaller sizes due to O(n²) complexity of reference DFT
	sizes := []int{16, 32, 64, 128, 256}

	t.Run("Forward", func(t *testing.T) {
		t.Parallel()
		testAVX2VsReference(t, sizes, avx2Forward, reference.NaiveDFT, 2000, "forward")
	})

	t.Run("Inverse", func(t *testing.T) {
		t.Parallel()
		testAVX2VsReference(t, sizes, avx2Inverse, reference.NaiveIDFT, 3000, "inverse")
	})
}

// testAVX2VsReference is a helper that compares AVX2 kernel output to reference DFT.
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
			twiddle, bitrev, scratch := prepareFFTData(n)

			// Compute with reference DFT (ground truth)
			dstRef := refKernel(src)

			// Compute with AVX2
			dstAVX2 := make([]complex64, n)
			avx2Handled := avx2Kernel(dstAVX2, src, twiddle, scratch, bitrev)

			if !avx2Handled {
				t.Skip("AVX2 kernel returned false (not yet implemented)")
			}

			// Compare results (looser tolerance due to algorithm differences)
			const relTol = 1e-5
			if !complexSliceEqual(dstAVX2, dstRef, relTol) {
				t.Errorf("AVX2 %s result differs from reference", name)
				reportDifferences(t, dstAVX2, dstRef, relTol)
			}
		})
	}
}

// reportDifferences logs up to 5 differences between two slices.
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
// 14.1.1: Round-Trip Tests
// =============================================================================

func TestAVX2RoundTrip(t *testing.T) {
	t.Parallel()

	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	sizes := []int{16, 32, 64, 128, 256, 512, 1024, 2048}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			original := generateRandomComplex64(n, uint64(n)+4000)
			twiddle, bitrev, scratch := prepareFFTData(n)

			// Forward transform
			freq := make([]complex64, n)
			if !avx2Forward(freq, original, twiddle, scratch, bitrev) {
				t.Skip("AVX2 forward kernel not yet implemented")
			}

			// Inverse transform
			recovered := make([]complex64, n)

			scratch2 := make([]complex64, n)
			if !avx2Inverse(recovered, freq, twiddle, scratch2, bitrev) {
				t.Skip("AVX2 inverse kernel not yet implemented")
			}

			// Compare original and recovered
			const relTol = 1e-5
			if !complexSliceEqual(recovered, original, relTol) {
				t.Errorf("Round-trip failed: IFFT(FFT(x)) != x")

				for i := range recovered {
					if !complexNearEqual(recovered[i], original[i], relTol) {
						t.Errorf("  [%d]: original=%v, recovered=%v", i, original[i], recovered[i])

						if i >= 5 {
							t.Errorf("  ... (more differences)")
							break
						}
					}
				}
			}
		})
	}
}

// =============================================================================
// 14.1.1: Property Tests
// =============================================================================

func TestAVX2Parseval(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	sizes := []int{16, 64, 256, 1024}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := generateRandomComplex64(n, uint64(n)+5000)
			twiddle, bitrev, scratch := prepareFFTData(n)

			// Compute time-domain energy
			energyTime := computeEnergy(src)

			// Compute frequency-domain result
			freq := make([]complex64, n)
			if !avx2Forward(freq, src, twiddle, scratch, bitrev) {
				t.Skip("AVX2 kernel not yet implemented")
			}

			// Compute frequency-domain energy (divide by N for Parseval)
			energyFreq := computeEnergy(freq) / float64(n)

			// Parseval's theorem: ||x||² = ||FFT(x)||² / N
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

	sizes := []int{16, 64, 256}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			x := generateRandomComplex64(n, uint64(n)+6000)
			y := generateRandomComplex64(n, uint64(n)+7000)
			a := complex(float32(0.7), float32(0.3))
			b := complex(float32(-0.4), float32(0.5))

			twiddle, bitrev, scratch := prepareFFTData(n)

			// Compute FFT(a*x + b*y)
			combined := make([]complex64, n)
			for i := range combined {
				combined[i] = a*x[i] + b*y[i]
			}

			fftCombined := make([]complex64, n)
			if !avx2Forward(fftCombined, combined, twiddle, scratch, bitrev) {
				t.Skip("AVX2 kernel not yet implemented")
			}

			// Compute a*FFT(x) + b*FFT(y)
			fftX := make([]complex64, n)

			scratch2 := make([]complex64, n)
			if !avx2Forward(fftX, x, twiddle, scratch2, bitrev) {
				t.Skip("AVX2 kernel not yet implemented")
			}

			fftY := make([]complex64, n)

			scratch3 := make([]complex64, n)
			if !avx2Forward(fftY, y, twiddle, scratch3, bitrev) {
				t.Skip("AVX2 kernel not yet implemented")
			}

			linearCombination := make([]complex64, n)
			for i := range linearCombination {
				linearCombination[i] = a*fftX[i] + b*fftY[i]
			}

			// Compare: FFT(a*x + b*y) should equal a*FFT(x) + b*FFT(y)
			// Note: Linearity test accumulates errors from 3 FFT operations plus
			// complex arithmetic, so we use a slightly looser tolerance.
			const relTol = 1e-4
			if !complexSliceEqual(fftCombined, linearCombination, relTol) {
				t.Error("Linearity violated: FFT(a*x + b*y) != a*FFT(x) + b*FFT(y)")
			}
		})
	}
}

// =============================================================================
// 14.1.1: Edge Case Tests
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
		src := make([]complex64, n) // all zeros
		twiddle, bitrev, scratch := prepareFFTData(n)

		dst := make([]complex64, n)
		if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
			t.Skip("AVX2 kernel not yet implemented")
		}

		// FFT of zeros should be zeros
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
		twiddle, bitrev, scratch := prepareFFTData(n)

		dst := make([]complex64, n)
		if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
			t.Skip("AVX2 kernel not yet implemented")
		}

		// FFT of impulse should be all ones
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
		twiddle, bitrev, scratch := prepareFFTData(n)

		dst := make([]complex64, n)
		if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
			t.Skip("AVX2 kernel not yet implemented")
		}

		// FFT of constant should have DC component = n * value, rest zero
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
		twiddle, bitrev, scratch := prepareFFTData(n)

		dst := make([]complex64, n)
		if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
			t.Skip("AVX2 kernel not yet implemented")
		}

		// FFT of cos(2πk*t/N) should have peaks at bins k and N-k
		const tol float32 = 1e-4

		expectedMag := float32(n) / 2

		// Check positive frequency bin
		gotMag := float32(math.Sqrt(float64(real(dst[freqBin])*real(dst[freqBin]) +
			imag(dst[freqBin])*imag(dst[freqBin]))))
		if math.Abs(float64(gotMag-expectedMag)) > float64(tol*expectedMag) {
			t.Errorf("FFT(cos)[%d] magnitude = %v, expected ~%v", freqBin, gotMag, expectedMag)
		}

		// Check negative frequency bin (conjugate symmetry)
		negBin := n - freqBin

		gotMagNeg := float32(math.Sqrt(float64(real(dst[negBin])*real(dst[negBin]) +
			imag(dst[negBin])*imag(dst[negBin]))))
		if math.Abs(float64(gotMagNeg-expectedMag)) > float64(tol*expectedMag) {
			t.Errorf("FFT(cos)[%d] magnitude = %v, expected ~%v", negBin, gotMagNeg, expectedMag)
		}

		// Other bins should be near zero
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

	t.Run("SmallSize_ReturnsTrue", func(t *testing.T) {
		t.Parallel()

		// AVX2 kernel should handle size 16 (minimum for vectorization)
		n := 16
		src := generateRandomComplex64(n, 12345)
		twiddle, bitrev, scratch := prepareFFTData(n)

		dst := make([]complex64, n)
		handled := avx2Forward(dst, src, twiddle, scratch, bitrev)

		// Even if not fully implemented, should either return true or skip
		if !handled {
			t.Skip("AVX2 kernel returned false for size 16 (not yet implemented)")
		}
	})

	t.Run("InverseUndoesForward", func(t *testing.T) {
		t.Parallel()

		n := 128
		original := generateRandomComplex64(n, 99999)
		twiddle, bitrev, scratch := prepareFFTData(n)

		// Forward
		freq := make([]complex64, n)
		if !avx2Forward(freq, original, twiddle, scratch, bitrev) {
			t.Skip("AVX2 forward not yet implemented")
		}

		// Inverse
		recovered := make([]complex64, n)

		scratch2 := make([]complex64, n)
		if !avx2Inverse(recovered, freq, twiddle, scratch2, bitrev) {
			t.Skip("AVX2 inverse not yet implemented")
		}

		const relTol = 1e-5
		if !complexSliceEqual(recovered, original, relTol) {
			t.Error("Inverse did not undo forward transform")
		}
	})
}

// =============================================================================
// 14.1.1: Size Validation Tests
// =============================================================================

func TestAVX2ReturnsFailureForInvalidSizes(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	t.Run("TooSmall", func(t *testing.T) {
		t.Parallel()

		// Sizes less than 16 should return false (delegate to scalar)
		for _, n := range []int{1, 2, 4, 8} {
			src := make([]complex64, n)
			dst := make([]complex64, n)
			twiddle := ComputeTwiddleFactors[complex64](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex64, n)

			handled := avx2Forward(dst, src, twiddle, scratch, bitrev)
			if handled {
				// If it handles it, that's fine too (may use scalar path internally)
				continue
			}
			// Returning false is expected - fallback will handle it
		}
	})

	t.Run("NonPowerOfTwo", func(t *testing.T) {
		t.Parallel()

		// Non-power-of-2 sizes should return false
		// Note: These require special handling (Bluestein or mixed-radix)
		for _, n := range []int{17, 31, 100} {
			src := make([]complex64, n)
			dst := make([]complex64, n)
			twiddle := make([]complex64, n)
			bitrev := make([]int, n)
			scratch := make([]complex64, n)

			handled := avx2Forward(dst, src, twiddle, scratch, bitrev)
			if handled {
				t.Errorf("AVX2 kernel should return false for non-power-of-2 size %d", n)
			}
		}
	})
}

// =============================================================================
// 14.1.1: Benchmarks
// =============================================================================

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

// 14.3: AVX2 Stockham Benchmarks (forward/inverse).
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

// =============================================================================
// Allocation Tests
// =============================================================================

func TestAVX2ZeroAllocations(t *testing.T) { //nolint:paralleltest // AllocsPerRun panics during parallel tests
	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	n := 1024
	src := generateRandomComplex64(n, 12345)
	dst := make([]complex64, n)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Verify kernel works
	if !avx2Forward(dst, src, twiddle, scratch, bitrev) {
		t.Skip("AVX2 forward kernel not yet implemented")
	}

	// Test forward allocations
	allocs := testing.AllocsPerRun(100, func() {
		avx2Forward(dst, src, twiddle, scratch, bitrev)
	})

	if allocs != 0 {
		t.Errorf("AVX2 forward kernel allocated %v times, expected 0", allocs)
	}

	// Test inverse allocations
	if !avx2Inverse(dst, src, twiddle, scratch, bitrev) {
		t.Skip("AVX2 inverse kernel not yet implemented")
	}

	allocs = testing.AllocsPerRun(100, func() {
		avx2Inverse(dst, src, twiddle, scratch, bitrev)
	})

	if allocs != 0 {
		t.Errorf("AVX2 inverse kernel allocated %v times, expected 0", allocs)
	}
}

// =============================================================================
// AVX2 Kernel Access Functions (complex128)
// =============================================================================

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

	// Return the raw AVX2 kernels (no fallback). This is important for tests that
	// validate the AVX2 kernels reject unsupported sizes.
	return forwardAVX2Complex128, inverseAVX2Complex128, true
}

//nolint:nonamedreturns
func getPureGoKernels128() (forward, inverse Kernel[complex128]) {
	return forwardDITComplex128, inverseDITComplex128
}

// =============================================================================
// 14.4: AVX2 complex128 Tests
// =============================================================================

func TestAVX2Forward128_VsPureGo(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels128()
	if !avx2Available {
		t.Skip("AVX2 not available")
	}

	goForward, _ := getPureGoKernels128()

	sizes := []int{16, 32, 64, 128, 256, 512, 1024}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := make([]complex128, n)

			rng := rand.New(rand.NewPCG(uint64(uint(n)), 1))
			for i := range src {
				src[i] = complex(rng.Float64(), rng.Float64())
			}

			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex128, n)

			dstGo := make([]complex128, n)
			if !goForward(dstGo, src, twiddle, scratch, bitrev) {
				t.Fatal("Pure-Go failed")
			}

			dstAVX2 := make([]complex128, n)

			scratchAVX2 := make([]complex128, n)
			if !avx2Forward(dstAVX2, src, twiddle, scratchAVX2, bitrev) {
				t.Skip("AVX2 complex128 forward not implemented")
			}

			for i := range dstAVX2 {
				if cmplx.Abs(dstAVX2[i]-dstGo[i]) > 1e-10 {
					t.Errorf("Mismatch at %d: AVX2=%v, Go=%v", i, dstAVX2[i], dstGo[i])
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

	sizes := []int{16, 32, 64, 128, 256, 512, 1024}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := make([]complex128, n)

			rng := rand.New(rand.NewPCG(uint64(uint(n)), 2))
			for i := range src {
				src[i] = complex(rng.Float64(), rng.Float64())
			}

			twiddle := ComputeTwiddleFactors[complex128](n)
			bitrev := ComputeBitReversalIndices(n)
			scratch := make([]complex128, n)

			dstGo := make([]complex128, n)
			if !goInverse(dstGo, src, twiddle, scratch, bitrev) {
				t.Fatal("Pure-Go failed")
			}

			dstAVX2 := make([]complex128, n)

			scratchAVX2 := make([]complex128, n)
			if !avx2Inverse(dstAVX2, src, twiddle, scratchAVX2, bitrev) {
				t.Skip("AVX2 complex128 inverse not implemented")
			}

			for i := range dstAVX2 {
				if cmplx.Abs(dstAVX2[i]-dstGo[i]) > 1e-10 {
					t.Errorf("Mismatch at %d: AVX2=%v, Go=%v", i, dstAVX2[i], dstGo[i])
					break
				}
			}
		})
	}
}

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

// =============================================================================
// 14.5.2: Size-Specific AVX2 Size-16 Kernel Tests
// =============================================================================

// TestAVX2Size16_VsReference tests the size-16 specific kernel against the
// reference naive DFT implementation.
func TestAVX2Size16_VsReference(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	const n = 16

	src := generateRandomComplex64(n, 12345)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Compute reference using naive DFT (complex64 version)
	dstRef := reference.NaiveDFT(src)

	// Compute with AVX2 (this will use the size-16 kernel via dispatch)
	dst := make([]complex64, n)
	ok := avx2Forward(dst, src, twiddle, scratch, bitrev)

	if !ok {
		t.Skip("AVX2 kernel not implemented (returned false)")
	}

	// Compare with reference
	const relTol float32 = 1e-5
	for i := range dst {
		if !complexNearEqual(dst[i], dstRef[i], relTol) {
			t.Errorf("[%d] AVX2 kernel = %v, reference = %v", i, dst[i], dstRef[i])
		}
	}
}

// TestAVX2Size16_VsGenericAVX2 tests the size-16 kernel against the generic AVX2 kernel.
func TestAVX2Size16_VsGenericAVX2(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	goForward, _ := getPureGoKernels()

	const n = 16

	src := generateRandomComplex64(n, 54321)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Compute with pure Go as ground truth
	dstGo := make([]complex64, n)

	scratchGo := make([]complex64, n)
	if !goForward(dstGo, src, twiddle, scratchGo, bitrev) {
		t.Fatal("Pure Go kernel failed")
	}

	// Compute with AVX2 (which dispatches to size-16 kernel)
	dst := make([]complex64, n)

	ok := avx2Forward(dst, src, twiddle, scratch, bitrev)
	if !ok {
		t.Skip("AVX2 kernel not implemented (returned false)")
	}

	// Compare results
	const relTol float32 = 1e-5
	if !complexSliceEqual(dst, dstGo, relTol) {
		for i := range dst {
			if !complexNearEqual(dst[i], dstGo[i], relTol) {
				t.Errorf("[%d] AVX2 = %v, Go = %v", i, dst[i], dstGo[i])
			}
		}
	}
}

// TestAVX2Size16_RoundTrip tests that Forward -> Inverse recovers the original signal.
func TestAVX2Size16_RoundTrip(t *testing.T) {
	t.Parallel()

	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	const n = 16

	src := generateRandomComplex64(n, 99999)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Forward with AVX2
	freq := make([]complex64, n)

	okFwd := avx2Forward(freq, src, twiddle, scratch, bitrev)
	if !okFwd {
		t.Skip("AVX2 forward kernel not implemented")
	}

	// Inverse with AVX2
	recovered := make([]complex64, n)
	scratchInv := make([]complex64, n)

	okInv := avx2Inverse(recovered, freq, twiddle, scratchInv, bitrev)
	if !okInv {
		t.Skip("AVX2 inverse not available")
	}

	// Compare recovered to original (note: inverse FFT scales by 1/n internally)
	const relTol float32 = 1e-5
	if !complexSliceEqual(recovered, src, relTol) {
		maxErr := float32(0)

		for i := range recovered {
			diff := recovered[i] - src[i]

			err := float32(real(diff)*real(diff) + imag(diff)*imag(diff))
			if err > maxErr {
				maxErr = err
			}
		}

		t.Errorf("Round-trip error: max squared diff = %e (tolerance = %e)", maxErr, relTol*relTol)
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

// 14.5.3: Size-Specific AVX2 Size-32 Kernel Tests
// =============================================================================

// TestAVX2Size32_VsReference tests the size-32 specific kernel against the
// reference naive DFT implementation.
func TestAVX2Size32_VsReference(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	const n = 32

	src := generateRandomComplex64(n, 32345)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Compute reference using naive DFT (complex64 version)
	dstRef := reference.NaiveDFT(src)

	// Compute with AVX2 (this will use the size-32 kernel via dispatch)
	dst := make([]complex64, n)
	ok := avx2Forward(dst, src, twiddle, scratch, bitrev)

	if !ok {
		t.Skip("AVX2 kernel not implemented (returned false)")
	}

	// Compare with reference
	const relTol float32 = 1e-5
	for i := range dst {
		if !complexNearEqual(dst[i], dstRef[i], relTol) {
			t.Errorf("[%d] AVX2 kernel = %v, reference = %v", i, dst[i], dstRef[i])
		}
	}
}

// TestAVX2Size32_VsGenericAVX2 tests the size-32 kernel against the generic AVX2 kernel.
func TestAVX2Size32_VsGenericAVX2(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	goForward, _ := getPureGoKernels()

	const n = 32

	src := generateRandomComplex64(n, 65432)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Compute with pure Go as ground truth
	dstGo := make([]complex64, n)

	scratchGo := make([]complex64, n)
	if !goForward(dstGo, src, twiddle, scratchGo, bitrev) {
		t.Fatal("Pure Go kernel failed")
	}

	// Compute with AVX2 (which dispatches to size-32 kernel)
	dst := make([]complex64, n)

	ok := avx2Forward(dst, src, twiddle, scratch, bitrev)
	if !ok {
		t.Skip("AVX2 kernel not implemented (returned false)")
	}

	// Compare results
	const relTol float32 = 1e-5
	if !complexSliceEqual(dst, dstGo, relTol) {
		for i := range dst {
			if !complexNearEqual(dst[i], dstGo[i], relTol) {
				t.Errorf("[%d] AVX2 = %v, Go = %v", i, dst[i], dstGo[i])
			}
		}
	}
}

// TestAVX2Size32_RoundTrip tests that Forward -> Inverse recovers the original signal.
func TestAVX2Size32_RoundTrip(t *testing.T) {
	t.Parallel()

	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	const n = 32

	src := generateRandomComplex64(n, 111222)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Forward with AVX2
	freq := make([]complex64, n)

	okFwd := avx2Forward(freq, src, twiddle, scratch, bitrev)
	if !okFwd {
		t.Skip("AVX2 forward kernel not implemented")
	}

	// Inverse with AVX2
	recovered := make([]complex64, n)
	scratchInv := make([]complex64, n)

	okInv := avx2Inverse(recovered, freq, twiddle, scratchInv, bitrev)
	if !okInv {
		t.Skip("AVX2 inverse not available")
	}

	// Compare recovered to original (note: inverse FFT scales by 1/n internally)
	const relTol float32 = 1e-5
	if !complexSliceEqual(recovered, src, relTol) {
		maxErr := float32(0)

		for i := range recovered {
			diff := recovered[i] - src[i]

			err := float32(real(diff)*real(diff) + imag(diff)*imag(diff))
			if err > maxErr {
				maxErr = err
			}
		}

		t.Errorf("Round-trip error: max squared diff = %e (tolerance = %e)", maxErr, relTol*relTol)
	}
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

// 14.5.4: Size-Specific AVX2 Size-64 Kernel Tests
// =============================================================================

// TestAVX2Size64_VsReference tests the size-64 specific kernel against the
// reference naive DFT implementation.
func TestAVX2Size64_VsReference(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	const n = 64

	src := generateRandomComplex64(n, 64345)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Compute reference using naive DFT (complex64 version)
	dstRef := reference.NaiveDFT(src)

	// Compute with AVX2 (this will use the size-64 kernel via dispatch)
	dst := make([]complex64, n)
	ok := avx2Forward(dst, src, twiddle, scratch, bitrev)

	if !ok {
		t.Skip("AVX2 kernel not implemented (returned false)")
	}

	// Compare with reference
	const relTol float32 = 1e-5
	for i := range dst {
		if !complexNearEqual(dst[i], dstRef[i], relTol) {
			t.Errorf("[%d] AVX2 kernel = %v, reference = %v", i, dst[i], dstRef[i])
		}
	}
}

// TestAVX2Size64_VsGenericAVX2 tests the size-64 kernel against the generic AVX2 kernel.
func TestAVX2Size64_VsGenericAVX2(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	goForward, _ := getPureGoKernels()

	const n = 64

	src := generateRandomComplex64(n, 76543)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Compute with pure Go as ground truth
	dstGo := make([]complex64, n)

	scratchGo := make([]complex64, n)
	if !goForward(dstGo, src, twiddle, scratchGo, bitrev) {
		t.Fatal("Pure Go kernel failed")
	}

	// Compute with AVX2 (which dispatches to size-64 kernel)
	dst := make([]complex64, n)

	ok := avx2Forward(dst, src, twiddle, scratch, bitrev)
	if !ok {
		t.Skip("AVX2 kernel not implemented (returned false)")
	}

	// Compare results
	const relTol float32 = 1e-5
	if !complexSliceEqual(dst, dstGo, relTol) {
		for i := range dst {
			if !complexNearEqual(dst[i], dstGo[i], relTol) {
				t.Errorf("[%d] AVX2 = %v, Go = %v", i, dst[i], dstGo[i])
			}
		}
	}
}

// TestAVX2Size64_RoundTrip tests that Forward -> Inverse recovers the original signal.
func TestAVX2Size64_RoundTrip(t *testing.T) {
	t.Parallel()

	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	const n = 64

	src := generateRandomComplex64(n, 222333)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Forward with AVX2
	freq := make([]complex64, n)

	okFwd := avx2Forward(freq, src, twiddle, scratch, bitrev)
	if !okFwd {
		t.Skip("AVX2 forward kernel not implemented")
	}

	// Inverse with AVX2
	recovered := make([]complex64, n)
	scratchInv := make([]complex64, n)

	okInv := avx2Inverse(recovered, freq, twiddle, scratchInv, bitrev)
	if !okInv {
		t.Skip("AVX2 inverse not available")
	}

	// Compare recovered to original (note: inverse FFT scales by 1/n internally)
	const relTol float32 = 1e-5
	if !complexSliceEqual(recovered, src, relTol) {
		maxErr := float32(0)

		for i := range recovered {
			diff := recovered[i] - src[i]

			err := float32(real(diff)*real(diff) + imag(diff)*imag(diff))
			if err > maxErr {
				maxErr = err
			}
		}

		t.Errorf("Round-trip error: max squared diff = %e (tolerance = %e)", maxErr, relTol*relTol)
	}
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

// =============================================================================
// 14.5.5: Size-Specific AVX2 Size-256 Kernel Tests
// =============================================================================
// Note: These tests require asm build tag to access assembly implementations

// TestAVX2Size256Radix2_VsReference tests the size-256 radix-2 kernel against
// the reference naive DFT implementation.
func TestAVX2Size256Radix2_VsReference(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	const n = 256

	src := generateRandomComplex64(n, 256345)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Compute reference using naive DFT
	dstRef := reference.NaiveDFT(src)

	// Compute with AVX2 (this will use the size-256 kernel via dispatch)
	dst := make([]complex64, n)
	ok := avx2Forward(dst, src, twiddle, scratch, bitrev)

	if !ok {
		t.Skip("AVX2 kernel not implemented (returned false)")
	}

	// Compare with reference
	const relTol float32 = 1e-5
	for i := range dst {
		if !complexNearEqual(dst[i], dstRef[i], relTol) {
			t.Errorf("[%d] AVX2 kernel = %v, reference = %v", i, dst[i], dstRef[i])
		}
	}
}

// TestAVX2Size256Radix2_VsGenericAVX2 tests the size-256 radix-2 kernel against
// the pure-Go implementation.
func TestAVX2Size256Radix2_VsGenericAVX2(t *testing.T) {
	t.Parallel()

	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	goForward, _ := getPureGoKernels()

	const n = 256

	src := generateRandomComplex64(n, 87654)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Compute with pure Go as ground truth
	dstGo := make([]complex64, n)

	scratchGo := make([]complex64, n)
	if !goForward(dstGo, src, twiddle, scratchGo, bitrev) {
		t.Fatal("Pure Go kernel failed")
	}

	// Compute with AVX2 (which dispatches to size-256 kernel)
	dst := make([]complex64, n)

	ok := avx2Forward(dst, src, twiddle, scratch, bitrev)
	if !ok {
		t.Skip("AVX2 kernel not implemented (returned false)")
	}

	// Compare results
	const relTol float32 = 1e-5
	if !complexSliceEqual(dst, dstGo, relTol) {
		for i := range dst {
			if !complexNearEqual(dst[i], dstGo[i], relTol) {
				t.Errorf("[%d] AVX2 = %v, Go = %v", i, dst[i], dstGo[i])
			}
		}
	}
}

func TestAVX2Size256Radix2Inverse_VsPureGo(t *testing.T) {
	t.Parallel()

	_, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	_, goInverse := getPureGoKernels()

	const n = 256

	src := generateRandomComplex64(n, 98765)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Compute with pure Go
	dstGo := make([]complex64, n)
	if !goInverse(dstGo, src, twiddle, scratch, bitrev) {
		t.Fatal("Pure-Go inverse kernel failed")
	}

	// Compute with AVX2
	dstAVX2 := make([]complex64, n)
	scratchAVX2 := make([]complex64, n)
	avx2Handled := avx2Inverse(dstAVX2, src, twiddle, scratchAVX2, bitrev)

	if !avx2Handled {
		t.Skip("AVX2 kernel returned false (not yet implemented)")
	}

	// Compare results
	const relTol = 2e-5
	if !complexSliceEqual(dstAVX2, dstGo, relTol) {
		t.Errorf("AVX2 inverse result differs from pure-Go")

		for i := range dstAVX2 {
			if !complexNearEqual(dstAVX2[i], dstGo[i], relTol) {
				t.Errorf("  [%d]: AVX2=%v, Go=%v", i, dstAVX2[i], dstGo[i])

				if i >= 5 {
					t.Errorf("  ... (more differences)")
					break
				}
			}
		}
	}
}

func TestAVX2Size256Radix2_RoundTrip(t *testing.T) {
	t.Parallel()

	avx2Forward, avx2Inverse, avx2Available := getAVX2Kernels()
	if !avx2Available {
		t.Skip("AVX2 not available on this system")
	}

	const n = 256

	src := generateRandomComplex64(n, 333444)
	twiddle, bitrev, scratch := prepareFFTData(n)

	// Forward with AVX2
	freq := make([]complex64, n)

	okFwd := avx2Forward(freq, src, twiddle, scratch, bitrev)
	if !okFwd {
		t.Skip("AVX2 forward kernel not implemented")
	}

	// Inverse with AVX2
	recovered := make([]complex64, n)
	scratchInv := make([]complex64, n)

	okInv := avx2Inverse(recovered, freq, twiddle, scratchInv, bitrev)
	if !okInv {
		t.Skip("AVX2 inverse not available")
	}

	// Compare recovered to original
	const relTol float32 = 1e-5
	if !complexSliceEqual(recovered, src, relTol) {
		maxErr := float32(0)

		for i := range recovered {
			diff := recovered[i] - src[i]

			err := float32(real(diff)*real(diff) + imag(diff)*imag(diff))
			if err > maxErr {
				maxErr = err
			}
		}

		t.Errorf("Round-trip error: max squared diff = %e (tolerance = %e)", maxErr, relTol*relTol)
	}
}

// BenchmarkAVX2Size256_VsGeneric benchmarks the size-256 kernel vs generic AVX2.
func BenchmarkAVX2Size256_VsGeneric(b *testing.B) {
	avx2Forward, _, avx2Available := getAVX2Kernels()
	if !avx2Available {
		b.Skip("AVX2 not available")
	}

	goForward, _ := getPureGoKernels()

	const n = 256

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

		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()

		for range b.N {
			avx2Forward(dst, src, twiddle, scratch, bitrev)
		}
	})

	b.Run("PureGo", func(b *testing.B) {
		if !goForward(dst, src, twiddle, scratch, bitrev) {
			b.Skip("Pure Go kernel failed")
		}

		b.ReportAllocs()
		b.SetBytes(int64(n * 8))
		b.ResetTimer()

		for range b.N {
			goForward(dst, src, twiddle, scratch, bitrev)
		}
	})
}
