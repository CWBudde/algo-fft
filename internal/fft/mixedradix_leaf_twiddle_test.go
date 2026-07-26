//go:build amd64 && !purego

package fft

import (
	"math/cmplx"
	"math/rand"
	"strconv"
	"testing"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/reference"
)

// leafTwiddleSizes are lengths whose mixed-radix schedule ends in a codelet
// leaf, i.e. exactly the shapes where the driver substitutes the cached
// size-n twiddle table for a stride-step gather:
//
//	96 = [3 32]   448 = [7 64]   480 = [5 3 32]
//	704 = [11 64] 768 = [3 256]  1000 = [5 5 5 8]  2205 = [5 7 7 3 3]
//
// 2205 is included as the counter-case: its schedule has no codelet leaf, so
// it must be unaffected.
var leafTwiddleSizes = []int{96, 448, 480, 704, 768, 1000, 2205} //nolint:gochecknoglobals

// TestMixedRadixLeafTwiddleMatchesReference compares the mixed-radix engine
// bin-by-bin against the naive DFT on a broadband random signal.
//
// The input matters as much as the sizes: an impulse has an all-ones spectrum,
// so it cannot detect a wrong twiddle table (every twiddle multiplies a zero)
// nor a wrong output ordering, and Parseval and linearity are blind to both.
// A wrong leaf twiddle table is precisely the failure this substitution could
// introduce, so the check must be element-wise against a reference.
func TestMixedRadixLeafTwiddleMatchesReference(t *testing.T) {
	t.Parallel()

	for _, n := range leafTwiddleSizes {
		t.Run("c64/"+strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			rng := rand.New(rand.NewSource(int64(n))) //nolint:gosec // deterministic test vector

			src := make([]complex64, n)
			for i := range src {
				src[i] = complex(float32(rng.NormFloat64()), float32(rng.NormFloat64()))
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)
			scratch := make([]complex64, n)
			dst := make([]complex64, n)

			if !forwardMixedRadixComplex64(dst, src, twiddle, scratch) {
				t.Fatal("forwardMixedRadixComplex64 failed")
			}

			ref := reference.NaiveDFT(src)

			// Tolerance scales with n: the spectrum of a unit-variance
			// broadband signal has magnitude ~sqrt(n), and float32 accumulates
			// over log(n) stages.
			tol := 2e-4 * float64(n)
			for i := range dst {
				if diff := cmplx.Abs(complex128(dst[i] - ref[i])); diff > tol {
					t.Fatalf("n=%d forward bin %d: got %v want %v (diff %g > %g)",
						n, i, dst[i], ref[i], diff, tol)
				}
			}

			// Round-trip on the same broadband signal.
			fwd := make([]complex64, n)
			copy(fwd, dst)

			if !inverseMixedRadixComplex64(dst, fwd, twiddle, scratch) {
				t.Fatal("inverseMixedRadixComplex64 failed")
			}

			for i := range dst {
				if diff := cmplx.Abs(complex128(dst[i] - src[i])); diff > 1e-3 {
					t.Fatalf("n=%d inverse sample %d: got %v want %v (diff %g)",
						n, i, dst[i], src[i], diff)
				}
			}
		})

		t.Run("c128/"+strconv.Itoa(n), func(t *testing.T) {
			t.Parallel()

			rng := rand.New(rand.NewSource(int64(n))) //nolint:gosec // deterministic test vector

			src := make([]complex128, n)
			for i := range src {
				src[i] = complex(rng.NormFloat64(), rng.NormFloat64())
			}

			twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)
			scratch := make([]complex128, n)
			dst := make([]complex128, n)

			if !forwardMixedRadixComplex128(dst, src, twiddle, scratch) {
				t.Fatal("forwardMixedRadixComplex128 failed")
			}

			ref := reference.NaiveDFT128(src)

			tol := 1e-10 * float64(n)
			for i := range dst {
				if diff := cmplx.Abs(dst[i] - ref[i]); diff > tol {
					t.Fatalf("n=%d forward bin %d: got %v want %v (diff %g > %g)",
						n, i, dst[i], ref[i], diff, tol)
				}
			}

			fwd := make([]complex128, n)
			copy(fwd, dst)

			if !inverseMixedRadixComplex128(dst, fwd, twiddle, scratch) {
				t.Fatal("inverseMixedRadixComplex128 failed")
			}

			for i := range dst {
				if diff := cmplx.Abs(dst[i] - src[i]); diff > 1e-10 {
					t.Fatalf("n=%d inverse sample %d: got %v want %v (diff %g)",
						n, i, dst[i], src[i], diff)
				}
			}
		})
	}
}

// leafTwiddleUsable itself is tested in mixedradix_stage_twiddle_test.go,
// alongside the portable code it now lives in.

// TestLeafTwiddleTableValues checks that the cached tables really are the
// standard size-n twiddle factors and that repeated lookups share one table.
func TestLeafTwiddleTableValues(t *testing.T) {
	t.Parallel()

	for _, n := range []int{8, 32, 64, 256} {
		got64 := leafTwiddle64(n)
		want64 := mathpkg.ComputeTwiddleFactors[complex64](n)

		if len(got64) != n {
			t.Fatalf("leafTwiddle64(%d) length = %d, want %d", n, len(got64), n)
		}

		for i := range want64 {
			if got64[i] != want64[i] {
				t.Fatalf("leafTwiddle64(%d)[%d] = %v, want %v", n, i, got64[i], want64[i])
			}
		}

		got128 := leafTwiddle128(n)
		want128 := mathpkg.ComputeTwiddleFactors[complex128](n)

		for i := range want128 {
			if got128[i] != want128[i] {
				t.Fatalf("leafTwiddle128(%d)[%d] = %v, want %v", n, i, got128[i], want128[i])
			}
		}

		// Second lookup must hand back the cached table, not a fresh one.
		if again := leafTwiddle64(n); &again[0] != &got64[0] {
			t.Errorf("leafTwiddle64(%d) returned a fresh table on the second call", n)
		}
	}
}
