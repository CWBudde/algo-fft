//go:build amd64 && !purego

// AVX-512 generic kernel tests: forward vs naive DFT, inverse vs naive IDFT,
// round-trip, and in-place (the copy-back bug class from P2.2). Benchmarks
// live in asm_amd64_avx512_bench_test.go.
package fft

import (
	"math/cmplx"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// sizeName formats a transform size as a subtest name (e.g. "Size1024").
func sizeName(n int) string {
	return "Size" + strconv.Itoa(n)
}

// assertAVX512Complex128Close compares against the naive-DFT reference with
// a size-scaled tolerance, mirroring the AVX2 complex128 tests. The floor for
// n >= 4096 is one tier higher (1e-9) than the AVX2 tests' 5e-10 because this
// kernel is radix-2 (log2(n) stages) while the AVX2 size-specific kernels are
// radix-4 (half the stages); the pure-Go radix-2 DIT shows the same deviation
// from the naive reference (5.3e-10 at n=8192) and this kernel matches that
// Go implementation to <1e-13.
func assertAVX512Complex128Close(t *testing.T, got, want []complex128, n int) {
	t.Helper()

	tol := getToleranceForSize128(n)
	if n >= 4096 && tol < 1e-9 {
		tol = 1e-9
	}

	for i := range got {
		if cmplx.Abs(got[i]-want[i]) > tol {
			t.Fatalf("n=%d index=%d got=%v want=%v (tol=%v)", n, i, got[i], want[i], tol)
		}
	}
}

// avx512TestSizes covers the supported range: every stage shape (scalar-only
// stages, strided vector stages, contiguous final stage) is exercised.
//
//nolint:gochecknoglobals // shared read-only test fixture
var avx512TestSizes = []int{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}

func TestAVX512ForwardComplex64VsReference(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	for _, n := range avx512TestSizes {
		t.Run(sizeName(n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x5120+uint64(n))
			dst := make([]complex64, n)
			twiddle, scratch := prepareFFTData[complex64](n)

			if !forwardAVX512Complex64(dst, src, twiddle, scratch) {
				t.Fatalf("forwardAVX512Complex64 returned false for n=%d", n)
			}

			want := reference.NaiveDFT(src)
			assertComplex64SliceClose(t, dst, want, n)
		})
	}
}

func TestAVX512InverseComplex64VsReference(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	for _, n := range avx512TestSizes {
		t.Run(sizeName(n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x5121+uint64(n))
			dst := make([]complex64, n)
			twiddle, scratch := prepareFFTData[complex64](n)

			if !inverseAVX512Complex64(dst, src, twiddle, scratch) {
				t.Fatalf("inverseAVX512Complex64 returned false for n=%d", n)
			}

			want := reference.NaiveIDFT(src)
			assertComplex64SliceClose(t, dst, want, n)
		})
	}
}

func TestAVX512RoundTripComplex64(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	for _, n := range avx512TestSizes {
		t.Run(sizeName(n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x5122+uint64(n))
			fwd := make([]complex64, n)
			dst := make([]complex64, n)
			twiddle, scratch := prepareFFTData[complex64](n)

			if !forwardAVX512Complex64(fwd, src, twiddle, scratch) {
				t.Fatalf("forward returned false for n=%d", n)
			}

			if !inverseAVX512Complex64(dst, fwd, twiddle, scratch) {
				t.Fatalf("inverse returned false for n=%d", n)
			}

			assertComplex64SliceClose(t, dst, src, n)
		})
	}
}

// TestAVX512InPlaceComplex64 exercises dst == src, which routes the transform
// through scratch and the copy-back path.
func TestAVX512InPlaceComplex64(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	for _, n := range avx512TestSizes {
		t.Run(sizeName(n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x5123+uint64(n))
			buf := make([]complex64, n)
			copy(buf, src)
			twiddle, scratch := prepareFFTData[complex64](n)

			if !forwardAVX512Complex64(buf, buf, twiddle, scratch) {
				t.Fatalf("in-place forward returned false for n=%d", n)
			}

			want := reference.NaiveDFT(src)
			assertComplex64SliceClose(t, buf, want, n)

			if !inverseAVX512Complex64(buf, buf, twiddle, scratch) {
				t.Fatalf("in-place inverse returned false for n=%d", n)
			}

			assertComplex64SliceClose(t, buf, src, n)
		})
	}
}

func TestAVX512ForwardComplex128VsReference(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	for _, n := range avx512TestSizes {
		t.Run(sizeName(n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x5124+uint64(n))
			dst := make([]complex128, n)
			twiddle, scratch := prepareFFTData[complex128](n)

			if !forwardAVX512Complex128(dst, src, twiddle, scratch) {
				t.Fatalf("forwardAVX512Complex128 returned false for n=%d", n)
			}

			want := reference.NaiveDFT128(src)
			assertAVX512Complex128Close(t, dst, want, n)
		})
	}
}

func TestAVX512InverseComplex128VsReference(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	for _, n := range avx512TestSizes {
		t.Run(sizeName(n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x5125+uint64(n))
			dst := make([]complex128, n)
			twiddle, scratch := prepareFFTData[complex128](n)

			if !inverseAVX512Complex128(dst, src, twiddle, scratch) {
				t.Fatalf("inverseAVX512Complex128 returned false for n=%d", n)
			}

			want := reference.NaiveIDFT128(src)
			assertAVX512Complex128Close(t, dst, want, n)
		})
	}
}

func TestAVX512RoundTripComplex128(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	for _, n := range avx512TestSizes {
		t.Run(sizeName(n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x5126+uint64(n))
			fwd := make([]complex128, n)
			dst := make([]complex128, n)
			twiddle, scratch := prepareFFTData[complex128](n)

			if !forwardAVX512Complex128(fwd, src, twiddle, scratch) {
				t.Fatalf("forward returned false for n=%d", n)
			}

			if !inverseAVX512Complex128(dst, fwd, twiddle, scratch) {
				t.Fatalf("inverse returned false for n=%d", n)
			}

			assertAVX512Complex128Close(t, dst, src, n)
		})
	}
}

func TestAVX512InPlaceComplex128(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	for _, n := range avx512TestSizes {
		t.Run(sizeName(n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x5127+uint64(n))
			buf := make([]complex128, n)
			copy(buf, src)
			twiddle, scratch := prepareFFTData[complex128](n)

			if !forwardAVX512Complex128(buf, buf, twiddle, scratch) {
				t.Fatalf("in-place forward returned false for n=%d", n)
			}

			want := reference.NaiveDFT128(src)
			assertAVX512Complex128Close(t, buf, want, n)

			if !inverseAVX512Complex128(buf, buf, twiddle, scratch) {
				t.Fatalf("in-place inverse returned false for n=%d", n)
			}

			assertAVX512Complex128Close(t, buf, src, n)
		})
	}
}

// TestAVX512FallbackConditions verifies the kernels decline (return false)
// inputs they do not support, so the dispatch chain can fall back safely.
func TestAVX512FallbackConditions(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	// Too small (n < 16).
	for _, n := range []int{2, 4, 8} {
		src := randomComplex64(n, 1)
		dst := make([]complex64, n)
		twiddle, scratch := prepareFFTData[complex64](n)

		if forwardAVX512Complex64(dst, src, twiddle, scratch) {
			t.Errorf("expected fallback for n=%d (complex64)", n)
		}

		src128 := randomComplex128(n, 1)
		dst128 := make([]complex128, n)
		twiddle128, scratch128 := prepareFFTData[complex128](n)

		if forwardAVX512Complex128(dst128, src128, twiddle128, scratch128) {
			t.Errorf("expected fallback for n=%d (complex128)", n)
		}
	}

	// Non-power-of-two.
	n := 24
	src := randomComplex64(n, 1)
	dst := make([]complex64, n)
	twiddle, scratch := prepareFFTData[complex64](n)

	if forwardAVX512Complex64(dst, src, twiddle, scratch) {
		t.Errorf("expected fallback for non-power-of-two n=%d", n)
	}

	// Short output/scratch slices must be rejected by the asm validation.
	n = 64
	src = randomComplex64(n, 2)
	twiddle, scratch = prepareFFTData[complex64](n)

	if kernelAccepted := forwardAVX512Complex64(make([]complex64, n-1), src, twiddle, scratch); kernelAccepted {
		t.Error("expected fallback for short dst")
	}

	if kernelAccepted := forwardAVX512Complex64(dst, src, twiddle, scratch[:n-1]); kernelAccepted {
		t.Error("expected fallback for short scratch")
	}
}

// TestAVX512DispatchSelectKernels exercises the HasAVX512 branch of the
// selectKernels* functions end-to-end for every strategy the AVX-512 chain
// touches, verifying results against the naive reference. The detected host
// features are passed directly, so no global feature state is mutated.
func TestAVX512DispatchSelectKernels(t *testing.T) {
	t.Parallel()
	requireAVX512(t)

	features := cpu.DetectFeatures()
	strategies := []fftypes.KernelStrategy{fftypes.KernelAuto, fftypes.KernelDIT, fftypes.KernelStockham}
	sizes := []int{16, 64, 1024, 4096}

	for _, st := range strategies {
		for _, n := range sizes {
			kern := selectKernelsComplex64WithStrategy(features, st)
			src := randomComplex64(n, 0x5130+uint64(n))
			dst := make([]complex64, n)
			rt := make([]complex64, n)
			twiddle, scratch := prepareFFTData[complex64](n)

			if !kern.Forward(dst, src, twiddle, scratch) {
				t.Fatalf("complex64 forward failed (strategy=%v n=%d)", st, n)
			}

			assertComplex64SliceClose(t, dst, reference.NaiveDFT(src), n)

			if !kern.Inverse(rt, dst, twiddle, scratch) {
				t.Fatalf("complex64 inverse failed (strategy=%v n=%d)", st, n)
			}

			assertComplex64SliceClose(t, rt, src, n)

			kernels128 := selectKernelsComplex128WithStrategy(features, st)
			src128 := randomComplex128(n, 0x5131+uint64(n))
			dst128 := make([]complex128, n)
			rt128 := make([]complex128, n)
			twiddle128, scratch128 := prepareFFTData[complex128](n)

			if !kernels128.Forward(dst128, src128, twiddle128, scratch128) {
				t.Fatalf("complex128 forward failed (strategy=%v n=%d)", st, n)
			}

			assertAVX512Complex128Close(t, dst128, reference.NaiveDFT128(src128), n)

			if !kernels128.Inverse(rt128, dst128, twiddle128, scratch128) {
				t.Fatalf("complex128 inverse failed (strategy=%v n=%d)", st, n)
			}

			assertAVX512Complex128Close(t, rt128, src128, n)
		}
	}
}

// TestPrewarmSizeCaches verifies that a prewarmed size serves its
// bit-reversal table without allocating (the wrappers stay allocation-free
// from the very first transform of a plan; see plan creation in the root
// package, which calls PrewarmSizeCaches).
//
//nolint:paralleltest // testing.AllocsPerRun panics inside parallel tests
func TestPrewarmSizeCaches(t *testing.T) {
	const n = 1 << 13

	PrewarmSizeCaches(n)

	if allocs := testing.AllocsPerRun(10, func() {
		if len(cachedBitReversalIndices(n)) != n {
			t.Fatal("unexpected table length")
		}
	}); allocs != 0 {
		t.Errorf("cachedBitReversalIndices allocated %v times per run after prewarm", allocs)
	}

	// Non-powers of two and non-positive sizes must be safe no-ops.
	PrewarmSizeCaches(0)
	PrewarmSizeCaches(-8)
	PrewarmSizeCaches(24)
}
