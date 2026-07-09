//go:build amd64 && !purego

// Complex128 AVX2 kernel tests, split from asm_amd64_avx2_test.go to stay
// under the file-length limit. Shared helpers live in that file.
package fft

import (
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

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

func TestAVX2GenericRadix4Complex128Forward(t *testing.T) {
	t.Parallel()

	// Test power-of-4 sizes supported by the generic radix-4 kernel
	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			// Generate random input
			rng := rand.New(rand.NewPCG(uint64(uint(n))+0xC128, 0xABCD))
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
			}

			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			// Run the generic radix-4 kernel
			if !forwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch) {
				t.Fatalf("forwardAVX2Complex128Radix4Asm failed for size %d", n)
			}

			// Compare against reference DFT
			want := reference.NaiveDFT128(src)

			// Use slightly relaxed tolerance for larger sizes due to accumulated
			// floating-point errors in multi-stage FFT
			tol := getToleranceForSize128(n)
			if n >= 4096 {
				tol = 5e-10 // Slightly higher tolerance for very large transforms
			}
			for i := range dst {
				diff := cmplx.Abs(dst[i] - want[i])
				if diff > tol {
					t.Errorf("Mismatch at index %d: got %v, want %v (diff=%v, tol=%v)",
						i, dst[i], want[i], diff, tol)
					break
				}
			}
		})
	}
}

func TestAVX2GenericRadix4Complex128ForwardImpulse(t *testing.T) {
	// Impulse test - FFT of delta should be all 1s
	n := 64
	src := make([]complex128, n)
	src[0] = 1 // impulse at position 0

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardAVX2Complex128Radix4Asm(dst, src, twiddle, scratch) {
		t.Fatalf("Forward failed")
	}

	want := reference.NaiveDFT128(src)

	t.Logf("Impulse test results (first 10 elements):")
	for i := range 10 {
		diff := cmplx.Abs(dst[i] - want[i])
		t.Logf("  %d: got=%v, want=%v, diff=%v", i, dst[i], want[i], diff)
	}

	// Check all values are close to 1
	for i := range dst {
		diff := cmplx.Abs(dst[i] - want[i])
		if diff > 1e-10 {
			t.Errorf("Impulse: mismatch at %d: got %v, want %v, diff=%v", i, dst[i], want[i], diff)
		}
	}
}

func TestAVX2GenericRadix4Complex128Inverse(t *testing.T) {
	t.Parallel()

	// Test power-of-4 sizes supported by the generic radix-4 kernel
	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			// Generate random frequency-domain input
			rng := rand.New(rand.NewPCG(uint64(uint(n))+0xC128, 0xDCBA))
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
			}

			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			// Run the generic radix-4 inverse kernel
			if !inverseAVX2Complex128Radix4Asm(dst, src, twiddle, scratch) {
				t.Fatalf("inverseAVX2Complex128Radix4Asm failed for size %d", n)
			}

			// Scale result by 1/n (inverse FFT normalization)
			// Our assembly kernel does NOT include normalization
			scale := 1.0 / float64(n)
			for i := range dst {
				dst[i] = complex(real(dst[i])*scale, imag(dst[i])*scale)
			}

			// Compare against reference IDFT (which includes 1/n normalization)
			want := reference.NaiveIDFT128(src)

			// Use slightly relaxed tolerance for larger sizes
			tol := getToleranceForSize128(n)
			if n >= 4096 {
				tol = 5e-10
			}
			for i := range dst {
				diff := cmplx.Abs(dst[i] - want[i])
				if diff > tol {
					t.Errorf("Mismatch at index %d: got %v, want %v (diff=%v, tol=%v)",
						i, dst[i], want[i], diff, tol)
					break
				}
			}
		})
	}
}

func TestAVX2GenericRadix4Complex128RoundTrip(t *testing.T) {
	t.Parallel()

	// Test power-of-4 sizes supported by the generic radix-4 kernel
	sizes := []int{64, 256, 1024, 4096}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			// Generate random input
			rng := rand.New(rand.NewPCG(uint64(uint(n))+0xC128, 0xFEDC))
			original := make([]complex128, n)
			for i := range original {
				original[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
			}

			// Allocate buffers
			forward := make([]complex128, n)
			result := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			// Forward transform
			if !forwardAVX2Complex128Radix4Asm(forward, original, twiddle, scratch) {
				t.Fatalf("Forward failed for size %d", n)
			}

			// Inverse transform
			if !inverseAVX2Complex128Radix4Asm(result, forward, twiddle, scratch) {
				t.Fatalf("Inverse failed for size %d", n)
			}

			// Scale result by 1/n (IFFT normalization)
			scale := 1.0 / float64(n)
			for i := range result {
				result[i] = complex(real(result[i])*scale, imag(result[i])*scale)
			}

			// Compare against original
			tol := getToleranceForSize128(n)
			if n >= 4096 {
				tol = 5e-10
			}
			for i := range result {
				diff := cmplx.Abs(result[i] - original[i])
				if diff > tol {
					t.Errorf("Round-trip mismatch at index %d: got %v, want %v (diff=%v, tol=%v)",
						i, result[i], original[i], diff, tol)
					break
				}
			}
		})
	}
}

// =============================================================================
// Tests for Generic Radix-4 Mixed (odd log2) Complex128 Kernel
// =============================================================================

func TestAVX2GenericRadix4MixedComplex128Forward(t *testing.T) {
	t.Parallel()

	// Test odd log2 sizes supported by the mixed radix-4 kernel
	// n = 2^(2k+1) = 32, 128, 512, 2048, ...
	sizes := []int{32, 128, 512, 2048}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			// Generate random input
			rng := rand.New(rand.NewPCG(uint64(uint(n))+0xC128, 0xABCD))
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
			}

			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			// Run the generic radix-4 mixed kernel
			if !forwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch) {
				t.Fatalf("forwardAVX2Complex128Radix4MixedAsm failed for size %d", n)
			}

			// Compare against reference DFT
			want := reference.NaiveDFT128(src)

			// Use slightly relaxed tolerance for larger sizes
			tol := getToleranceForSize128(n)
			if n >= 2048 {
				tol = 5e-10
			}
			for i := range dst {
				diff := cmplx.Abs(dst[i] - want[i])
				if diff > tol {
					t.Errorf("Mismatch at index %d: got %v, want %v (diff=%v, tol=%v)",
						i, dst[i], want[i], diff, tol)
					break
				}
			}
		})
	}
}

func TestAVX2GenericRadix4MixedComplex128ForwardImpulse(t *testing.T) {
	// Impulse test - FFT of delta should be all 1s
	n := 128 // Use odd log2 size
	src := make([]complex128, n)
	src[0] = 1 // impulse at position 0

	dst := make([]complex128, n)
	scratch := make([]complex128, n)
	twiddle := ComputeTwiddleFactors[complex128](n)

	if !forwardAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch) {
		t.Fatalf("Forward failed")
	}

	want := reference.NaiveDFT128(src)

	t.Logf("Impulse test results (first 10 elements):")
	for i := range 10 {
		diff := cmplx.Abs(dst[i] - want[i])
		t.Logf("  %d: got=%v, want=%v, diff=%v", i, dst[i], want[i], diff)
	}

	// Check all values are close to 1
	for i := range dst {
		diff := cmplx.Abs(dst[i] - want[i])
		if diff > 1e-10 {
			t.Errorf("Impulse: mismatch at %d: got %v, want %v, diff=%v", i, dst[i], want[i], diff)
		}
	}
}

func TestAVX2GenericRadix4MixedComplex128Inverse(t *testing.T) {
	t.Parallel()

	// Test odd log2 sizes supported by the mixed radix-4 kernel
	sizes := []int{32, 128, 512, 2048}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			// Generate random frequency-domain input
			rng := rand.New(rand.NewPCG(uint64(uint(n))+0xC128, 0xDCBA))
			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
			}

			dst := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			// Run the generic radix-4 mixed inverse kernel
			if !inverseAVX2Complex128Radix4MixedAsm(dst, src, twiddle, scratch) {
				t.Fatalf("inverseAVX2Complex128Radix4MixedAsm failed for size %d", n)
			}

			// Scale result by 1/n (inverse FFT normalization)
			// Our assembly kernel does NOT include normalization
			scale := 1.0 / float64(n)
			for i := range dst {
				dst[i] = complex(real(dst[i])*scale, imag(dst[i])*scale)
			}

			// Compare against reference IDFT (which includes 1/n normalization)
			want := reference.NaiveIDFT128(src)

			// Use slightly relaxed tolerance for larger sizes
			tol := getToleranceForSize128(n)
			if n >= 2048 {
				tol = 5e-10
			}
			for i := range dst {
				diff := cmplx.Abs(dst[i] - want[i])
				if diff > tol {
					t.Errorf("Mismatch at index %d: got %v, want %v (diff=%v, tol=%v)",
						i, dst[i], want[i], diff, tol)
					break
				}
			}
		})
	}
}

func TestAVX2GenericRadix4MixedComplex128RoundTrip(t *testing.T) {
	t.Parallel()

	// Test odd log2 sizes supported by the mixed radix-4 kernel
	sizes := []int{32, 128, 512, 2048}

	for _, n := range sizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			// Generate random input
			rng := rand.New(rand.NewPCG(uint64(uint(n))+0xC128, 0x1234))
			original := make([]complex128, n)
			for i := range original {
				original[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
			}

			forward := make([]complex128, n)
			scratch := make([]complex128, n)
			twiddle := ComputeTwiddleFactors[complex128](n)

			// Forward transform
			if !forwardAVX2Complex128Radix4MixedAsm(forward, original, twiddle, scratch) {
				t.Fatalf("Forward transform failed for size %d", n)
			}

			// Inverse transform
			result := make([]complex128, n)
			if !inverseAVX2Complex128Radix4MixedAsm(result, forward, twiddle, scratch) {
				t.Fatalf("Inverse transform failed for size %d", n)
			}

			// Scale result by 1/n (IFFT normalization)
			scale := 1.0 / float64(n)
			for i := range result {
				result[i] = complex(real(result[i])*scale, imag(result[i])*scale)
			}

			// Compare against original
			tol := getToleranceForSize128(n)
			if n >= 2048 {
				tol = 5e-10
			}
			for i := range result {
				diff := cmplx.Abs(result[i] - original[i])
				if diff > tol {
					t.Errorf("Round-trip mismatch at index %d: got %v, want %v (diff=%v, tol=%v)",
						i, result[i], original[i], diff, tol)
					break
				}
			}
		})
	}
}
