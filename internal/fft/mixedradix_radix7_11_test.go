package fft

import (
	"math/cmplx"
	"math/rand"
	"slices"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

func TestMixedRadixScheduleRadix7And11(t *testing.T) {
	t.Parallel()

	cases := []struct {
		n    int
		want []int
	}{
		{7, []int{7}},             // single radix-7 stage
		{11, []int{11}},           // single radix-11 stage
		{14, []int{7, 2}},         // 7 stripped before the pow2 part
		{21, []int{7, 3}},         // 7·3
		{35, []int{5, 7}},         // radix-5 case fires first
		{49, []int{7, 7}},         // prime power
		{55, []int{5, 11}},        // 5·11
		{63, []int{7, 3, 3}},      // 7·3^2
		{77, []int{7, 11}},        // both new radices
		{121, []int{11, 11}},      // prime power
		{231, []int{7, 11, 3}},    // 3·7·11
		{448, []int{7, 8, 8}},     // 2^6·7: pow2 part keeps its radix-8 passes
		{616, []int{7, 11, 8}},    // 2^3·7·11
		{704, []int{11, 8, 8}},    // 2^6·11
		{1344, []int{7, 8, 8, 3}}, // 2^6·3·7
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

// TestMixedRadixScheduleRejectsNonSmooth locks in that factors outside
// {2,3,5,7,11} still fail scheduling, so those sizes keep falling back to
// Bluestein at the plan layer.
func TestMixedRadixScheduleRejectsNonSmooth(t *testing.T) {
	t.Parallel()

	for _, n := range []int{13, 17, 26, 39, 91, 143, 1001} {
		var radices [mixedRadixMaxStages]int

		if count := mixedRadixSchedule(n, &radices, noCodelet); count != 0 {
			t.Errorf("n=%d: schedule succeeded with %v, want rejection", n, radices[:count])
		}
	}
}

// Sizes whose codelet-free schedule contains at least one radix-7 or
// radix-11 stage.
//
//nolint:gochecknoglobals
var radix7And11Sizes = []int{
	7, 11, 14, 21, 22, 33, 35, 49, 55, 63, 77, 99, 112, 121,
	154, 176, 231, 385, 448, 539, 616, 693, 704, 847, 1232, 1331, 1344, 2401,
}

func TestMixedRadixRadix7And11VsReferenceComplex64(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(17))

	for _, n := range radix7And11Sizes {
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

func TestMixedRadixRadix7And11VsReferenceComplex128(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(19))

	for _, n := range radix7And11Sizes {
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

// TestMixedRadixTransformRadix7And11InPlace covers the full transform entry
// point (schedule + driver + in-place scratch handling), which on AVX2 builds
// also exercises the codelet-dispatching recursion hook for the pow2 parts.
func TestMixedRadixTransformRadix7And11InPlace(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(23))

	for _, n := range radix7And11Sizes {
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
