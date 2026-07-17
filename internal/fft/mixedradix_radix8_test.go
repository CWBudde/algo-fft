package fft

import (
	"math/cmplx"
	"math/rand"
	"slices"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// noCodelet forces the pure scheduling path (no codelet claiming), which is
// what purego and non-amd64 builds see.
func noCodelet(int) bool { return false }

func TestMixedRadixScheduleRadix8(t *testing.T) {
	t.Parallel()

	cases := []struct {
		n    int
		want []int
	}{
		{8, []int{8}},                 // 2^3: single pass
		{16, []int{4, 4}},             // 2^4: keep [4,4] over [8,2]
		{32, []int{8, 4}},             // 2^5: was [4,4,2]
		{64, []int{8, 8}},             // 2^6: was [4,4,4]
		{128, []int{8, 4, 4}},         // 2^7: was [4,4,4,2]
		{512, []int{8, 8, 8}},         // 2^9: was [4,4,4,4,2]
		{2048, []int{8, 8, 8, 4}},     // 2^11
		{24, []int{8, 3}},             // 8·3
		{40, []int{5, 8}},             // 5·8
		{96, []int{8, 4, 3}},          // 2^5·3: was [4,4,3,2]
		{480, []int{5, 8, 4, 3}},      // 2^5·3·5
		{1536, []int{8, 8, 8, 3}},     // 2^9·3
		{12288, []int{8, 8, 8, 8, 3}}, // 2^12·3
		{12, []int{4, 3}},             // no factor 8: unchanged
		{20, []int{5, 4}},             // no factor 8: unchanged
	}

	for _, tc := range cases {
		var radices [mixedRadixMaxStages]int

		count := mixedRadixSchedule(tc.n, &radices, noCodelet)
		if count == 0 {
			t.Fatalf("n=%d: schedule failed", tc.n)
		}

		got := radices[:count]
		if !slices.Equal(got, tc.want) {
			t.Errorf("n=%d: schedule %v, want %v", tc.n, got, tc.want)
		}

		product := 1
		for _, r := range got {
			product *= r
		}

		if product != tc.n {
			t.Errorf("n=%d: schedule %v product %d", tc.n, got, product)
		}
	}
}

// runPureMixedRadix64 runs the pure Go recursion driver (no codelet
// dispatch) on the codelet-free schedule, so the radix-8 stages are
// exercised on every build, including amd64 with AVX2 codelets registered.
func runPureMixedRadix64(dst, src, twiddle, scratch []complex64, inverse bool) bool {
	n := len(src)

	var radices [mixedRadixMaxStages]int

	count := mixedRadixSchedule(n, &radices, noCodelet)
	if count == 0 {
		return false
	}

	mixedRadixRecursivePingPongComplex64(dst, src, scratch, n, 1, 1, radices[:count], twiddle, inverse)

	if inverse {
		scale := complex(float32(1.0/float64(n)), 0)
		for i := range dst[:n] {
			dst[i] *= scale
		}
	}

	return true
}

func runPureMixedRadix128(dst, src, twiddle, scratch []complex128, inverse bool) bool {
	n := len(src)

	var radices [mixedRadixMaxStages]int

	count := mixedRadixSchedule(n, &radices, noCodelet)
	if count == 0 {
		return false
	}

	mixedRadixRecursivePingPongComplex128(dst, src, scratch, n, 1, 1, radices[:count], twiddle, inverse)

	if inverse {
		scale := complex(1.0/float64(n), 0)
		for i := range dst[:n] {
			dst[i] *= scale
		}
	}

	return true
}

// Sizes whose codelet-free schedule contains at least one radix-8 stage.
//
//nolint:gochecknoglobals
var radix8Sizes = []int{8, 24, 32, 40, 64, 96, 120, 128, 160, 480, 512, 640, 960, 1536}

func TestMixedRadixRadix8VsReferenceComplex64(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(7))

	for _, n := range radix8Sizes {
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(rng.Float64()*2-1), float32(rng.Float64()*2-1))
		}

		twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
		scratch := make([]complex64, n)
		dst := make([]complex64, n)

		if !runPureMixedRadix64(dst, src, twiddle, scratch, false) {
			t.Fatalf("n=%d: forward failed", n)
		}

		ref := reference.NaiveDFT(src)

		var maxRef float64
		for i := range ref {
			if m := cmplx.Abs(complex128(ref[i])); m > maxRef {
				maxRef = m
			}
		}

		if maxRef == 0 {
			maxRef = 1 // all-zero reference: fall back to absolute error
		}

		for i := range dst {
			if cmplx.Abs(complex128(dst[i]-ref[i]))/maxRef > 1e-5 {
				t.Errorf("n=%d: forward mismatch at %d: got %v want %v", n, i, dst[i], ref[i])
				break
			}
		}

		// Round-trip: inverse(forward(x)) ≈ x.
		back := make([]complex64, n)
		if !runPureMixedRadix64(back, dst, twiddle, scratch, true) {
			t.Fatalf("n=%d: inverse failed", n)
		}

		for i := range back {
			if cmplx.Abs(complex128(back[i]-src[i])) > 1e-4 {
				t.Errorf("n=%d: round-trip mismatch at %d: got %v want %v", n, i, back[i], src[i])
				break
			}
		}
	}
}

func TestMixedRadixRadix8VsReferenceComplex128(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(11))

	for _, n := range radix8Sizes {
		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
		}

		twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
		scratch := make([]complex128, n)
		dst := make([]complex128, n)

		if !runPureMixedRadix128(dst, src, twiddle, scratch, false) {
			t.Fatalf("n=%d: forward failed", n)
		}

		ref := reference.NaiveDFT128(src)

		var maxRef float64
		for i := range ref {
			if m := cmplx.Abs(ref[i]); m > maxRef {
				maxRef = m
			}
		}

		if maxRef == 0 {
			maxRef = 1 // all-zero reference: fall back to absolute error
		}

		for i := range dst {
			if cmplx.Abs(dst[i]-ref[i])/maxRef > 1e-11 {
				t.Errorf("n=%d: forward mismatch at %d: got %v want %v", n, i, dst[i], ref[i])
				break
			}
		}

		back := make([]complex128, n)
		if !runPureMixedRadix128(back, dst, twiddle, scratch, true) {
			t.Fatalf("n=%d: inverse failed", n)
		}

		for i := range back {
			if cmplx.Abs(back[i]-src[i]) > 1e-12 {
				t.Errorf("n=%d: round-trip mismatch at %d: got %v want %v", n, i, back[i], src[i])
				break
			}
		}
	}
}

// TestMixedRadixTransformRadix8InPlace covers the full transform entry point
// (schedule + driver + in-place scratch handling) for radix-8-bearing sizes.
func TestMixedRadixTransformRadix8InPlace(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(13))

	for _, n := range radix8Sizes {
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(rng.Float64()*2-1), float32(rng.Float64()*2-1))
		}

		twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
		scratch := make([]complex64, n)

		buf := make([]complex64, n)
		copy(buf, src)

		// In-place: dst == src.
		if !forwardMixedRadixComplex64(buf, buf, twiddle, scratch) {
			t.Fatalf("n=%d: in-place forward failed", n)
		}

		ref := reference.NaiveDFT(src)

		var maxRef float64
		for i := range ref {
			if m := cmplx.Abs(complex128(ref[i])); m > maxRef {
				maxRef = m
			}
		}

		if maxRef == 0 {
			maxRef = 1 // all-zero reference: fall back to absolute error
		}

		for i := range buf {
			if cmplx.Abs(complex128(buf[i]-ref[i]))/maxRef > 1e-5 {
				t.Errorf("n=%d: in-place forward mismatch at %d: got %v want %v", n, i, buf[i], ref[i])
				break
			}
		}

		if !inverseMixedRadixComplex64(buf, buf, twiddle, scratch) {
			t.Fatalf("n=%d: in-place inverse failed", n)
		}

		for i := range buf {
			if cmplx.Abs(complex128(buf[i]-src[i])) > 1e-4 {
				t.Errorf("n=%d: in-place round-trip mismatch at %d", n, i)
				break
			}
		}
	}
}

// legacySchedule replicates the pre-radix-8 scheduling (radix-4-major with a
// trailing 2) for benchmarking old vs new schedules through the same driver.
func legacySchedule(n int, radices *[mixedRadixMaxStages]int) int {
	count := 0

	for n > 1 {
		switch {
		case n%5 == 0:
			radices[count] = 5
			n /= 5
		case n%4 == 0:
			radices[count] = 4
			n /= 4
		case n%3 == 0:
			radices[count] = 3
			n /= 3
		case n%2 == 0:
			radices[count] = 2
			n /= 2
		default:
			return 0
		}

		count++
	}

	return count
}

func BenchmarkMixedRadixRadix8Schedule(b *testing.B) {
	for _, n := range []int{32, 64, 96, 128, 480, 512, 640, 960, 1536, 12288} {
		src := make([]complex64, n)
		for i := range src {
			src[i] = complex(float32(i%7)-3, float32(i%5)-2)
		}

		twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
		scratch := make([]complex64, n)
		dst := make([]complex64, n)

		var oldRadices, newRadices [mixedRadixMaxStages]int

		oldCount := legacySchedule(n, &oldRadices)
		newCount := mixedRadixSchedule(n, &newRadices, noCodelet)

		b.Run("radix42/"+itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(n * 8))

			for range b.N {
				mixedRadixRecursivePingPongComplex64(dst, src, scratch, n, 1, 1, oldRadices[:oldCount], twiddle, false)
			}
		})

		b.Run("radix8/"+itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(n * 8))

			for range b.N {
				mixedRadixRecursivePingPongComplex64(dst, src, scratch, n, 1, 1, newRadices[:newCount], twiddle, false)
			}
		})
	}
}

func BenchmarkMixedRadixRadix8ScheduleComplex128(b *testing.B) {
	for _, n := range []int{32, 64, 96, 128, 480, 512, 640, 960, 1536, 12288} {
		src := make([]complex128, n)
		for i := range src {
			src[i] = complex(float64(i%7)-3, float64(i%5)-2)
		}

		twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
		scratch := make([]complex128, n)
		dst := make([]complex128, n)

		var oldRadices, newRadices [mixedRadixMaxStages]int

		oldCount := legacySchedule(n, &oldRadices)
		newCount := mixedRadixSchedule(n, &newRadices, noCodelet)

		b.Run("radix42/"+itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(n * 16))

			for range b.N {
				mixedRadixRecursivePingPongComplex128(dst, src, scratch, n, 1, 1, oldRadices[:oldCount], twiddle, false)
			}
		})

		b.Run("radix8/"+itoa(n), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(n * 16))

			for range b.N {
				mixedRadixRecursivePingPongComplex128(dst, src, scratch, n, 1, 1, newRadices[:newCount], twiddle, false)
			}
		})
	}
}
