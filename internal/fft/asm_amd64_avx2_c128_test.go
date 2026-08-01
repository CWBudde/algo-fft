//go:build amd64 && !purego

// Complex128 AVX2 kernel tests, split from asm_amd64_avx2_test.go to stay
// under the file-length limit. Shared helpers live in that file.
package fft

import (
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/cwbudde/algo-fft/internal/kernels"
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

// =============================================================================
// Tests for Generic Radix-4 Mixed (odd log2) Complex128 kernels.Kernel
// =============================================================================

// =============================================================================
// AVX2 Stockham complex128 vs Pure-Go Stockham Tests
// =============================================================================

// stockham128TestSizes covers the asm minimum (16), mid sizes, and large
// sizes with many stages where the Stockham path is actually selected.
var stockham128TestSizes = []int{16, 32, 64, 1024, 2048, 32768, 65536, 131072}

func TestAVX2StockhamForward128_VsPureGo(t *testing.T) {
	t.Parallel()

	if !cpuHasAVX2ForStockham128(t) {
		return
	}

	for _, n := range stockham128TestSizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := randomStockham128Input(n, 0x57C128)
			twiddle, scratch := prepareFFTData[complex128](n)

			dstGo := make([]complex128, n)
			if !kernels.ForwardStockhamComplex128(dstGo, src, twiddle, scratch) {
				t.Fatal("pure-Go Stockham forward failed")
			}

			dstAVX2 := make([]complex128, n)
			scratchAVX2 := make([]complex128, n)
			if !forwardAVX2StockhamComplex128(dstAVX2, src, twiddle, scratchAVX2) {
				t.Fatal("AVX2 Stockham complex128 forward failed")
			}

			assertStockham128Close(t, dstAVX2, dstGo, n)
		})
	}
}

func TestAVX2StockhamInverse128_VsPureGo(t *testing.T) {
	t.Parallel()

	if !cpuHasAVX2ForStockham128(t) {
		return
	}

	for _, n := range stockham128TestSizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := randomStockham128Input(n, 0x1C128)
			twiddle, scratch := prepareFFTData[complex128](n)

			dstGo := make([]complex128, n)
			if !kernels.InverseStockhamComplex128(dstGo, src, twiddle, scratch) {
				t.Fatal("pure-Go Stockham inverse failed")
			}

			dstAVX2 := make([]complex128, n)
			scratchAVX2 := make([]complex128, n)
			if !inverseAVX2StockhamComplex128(dstAVX2, src, twiddle, scratchAVX2) {
				t.Fatal("AVX2 Stockham complex128 inverse failed")
			}

			assertStockham128Close(t, dstAVX2, dstGo, n)
		})
	}
}

// TestAVX2StockhamComplex128InPlace verifies the dst==src aliased case takes
// the scratch-first ping-pong route and matches the out-of-place result.
func TestAVX2StockhamComplex128InPlace(t *testing.T) {
	t.Parallel()

	if !cpuHasAVX2ForStockham128(t) {
		return
	}

	for _, n := range []int{16, 2048, 65536} {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := randomStockham128Input(n, 0xA11A5)
			twiddle, scratch := prepareFFTData[complex128](n)

			want := make([]complex128, n)
			if !forwardAVX2StockhamComplex128(want, src, twiddle, scratch) {
				t.Fatal("out-of-place forward failed")
			}

			data := make([]complex128, n)
			copy(data, src)
			if !forwardAVX2StockhamComplex128(data, data, twiddle, scratch) {
				t.Fatal("in-place forward failed")
			}

			for i := range data {
				if data[i] != want[i] {
					t.Fatalf("in-place differs from out-of-place at %d: %v vs %v", i, data[i], want[i])
				}
			}

			wantInv := make([]complex128, n)
			if !inverseAVX2StockhamComplex128(wantInv, want, twiddle, scratch) {
				t.Fatal("out-of-place inverse failed")
			}

			copy(data, want)
			if !inverseAVX2StockhamComplex128(data, data, twiddle, scratch) {
				t.Fatal("in-place inverse failed")
			}

			for i := range data {
				if data[i] != wantInv[i] {
					t.Fatalf("in-place inverse differs at %d: %v vs %v", i, data[i], wantInv[i])
				}
			}
		})
	}
}

// TestAVX2StockhamComplex128RoundTrip checks Inverse(Forward(x)) ≈ x.
func TestAVX2StockhamComplex128RoundTrip(t *testing.T) {
	t.Parallel()

	if !cpuHasAVX2ForStockham128(t) {
		return
	}

	for _, n := range stockham128TestSizes {
		t.Run(sizeString(n), func(t *testing.T) {
			t.Parallel()

			src := randomStockham128Input(n, 0x57C129)
			twiddle, scratch := prepareFFTData[complex128](n)

			fwd := make([]complex128, n)
			if !forwardAVX2StockhamComplex128(fwd, src, twiddle, scratch) {
				t.Fatal("forward failed")
			}

			dst := make([]complex128, n)
			if !inverseAVX2StockhamComplex128(dst, fwd, twiddle, scratch) {
				t.Fatal("inverse failed")
			}

			tol := getToleranceForSize128(n)
			for i := range dst {
				if cmplx.Abs(dst[i]-src[i]) > tol {
					t.Errorf("round-trip mismatch at %d: got %v, want %v (tol=%v)", i, dst[i], src[i], tol)
					break
				}
			}
		})
	}
}

func cpuHasAVX2ForStockham128(t *testing.T) bool {
	t.Helper()

	_, _, avx2Available := getAVX2Kernels128()
	if !avx2Available {
		t.Skip("AVX2 not available")
	}

	return avx2Available
}

func randomStockham128Input(n int, seed uint64) []complex128 {
	src := make([]complex128, n)

	rng := rand.New(rand.NewPCG(seed, uint64(uint(n))))
	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	return src
}

// assertStockham128Close compares the asm and Go Stockham results. Both
// compute the same stage order, so differences come only from FMA contraction
// in the asm butterflies; the bound scales with accumulated stage error.
func assertStockham128Close(t *testing.T, got, want []complex128, n int) {
	t.Helper()

	tol := getToleranceForSize128(n)
	for i := range got {
		if cmplx.Abs(got[i]-want[i]) > tol {
			t.Errorf("mismatch at %d: asm=%v, go=%v (tol=%v)", i, got[i], want[i], tol)

			return
		}
	}
}
