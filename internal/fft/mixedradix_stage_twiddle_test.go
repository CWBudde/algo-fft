package fft

import (
	"math/cmplx"
	"math/rand"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/kernels"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// TestLeafTwiddleUsable pins the invariant that makes a standard size-n twiddle
// table interchangeable with a stride-step gather. A table longer than n*step
// encodes different roots of unity, so the driver must fall back to gathering.
func TestLeafTwiddleUsable(t *testing.T) {
	t.Parallel()

	cases := []struct {
		n, step, tableLen int
		want              bool
	}{
		{n: 32, step: 3, tableLen: 96, want: true},   // leaf of the n=96 schedule
		{n: 256, step: 3, tableLen: 768, want: true}, // leaf of the n=768 schedule
		{n: 96, step: 1, tableLen: 96, want: true},   // root node
		{n: 32, step: 3, tableLen: 192, want: false}, // oversized table
		{n: 32, step: 3, tableLen: 95, want: false},  // short table
		{n: 0, step: 1, tableLen: 0, want: false},
		{n: 32, step: 0, tableLen: 0, want: false},
	}

	for _, c := range cases {
		if got := leafTwiddleUsable(c.n, c.step, c.tableLen); got != c.want {
			t.Errorf("leafTwiddleUsable(%d, %d, %d) = %v, want %v",
				c.n, c.step, c.tableLen, got, c.want)
		}
	}
}

// TestStageTwiddleMatchesStridedGather is the correctness argument for the
// whole per-stage table: entry j*span+k must hold exactly the value the scalar
// stage reads as twiddle[j*k*step] from the root table. Both a root node
// (step == 1) and interior nodes (step > 1) are checked, since the point of
// the table is that the stride drops out.
func TestStageTwiddleMatchesStridedGather(t *testing.T) {
	t.Parallel()

	cases := []struct{ n, radix, step int }{
		{n: 96, radix: 3, step: 1},   // root of the n=96 schedule
		{n: 1000, radix: 5, step: 1}, // root of [5 5 5 8]
		{n: 200, radix: 5, step: 5},  // its second level
		{n: 80, radix: 5, step: 25},  // exactly at the size gate (64 multiplies)
		{n: 704, radix: 11, step: 1},
		{n: 128, radix: 8, step: 11},
		{n: 256, radix: 2, step: 3},
		{n: 128, radix: 4, step: 7},
	}

	for _, c := range cases {
		for _, inverse := range []bool{false, true} {
			name := strconv.Itoa(c.n) + "r" + strconv.Itoa(c.radix) + "s" + strconv.Itoa(c.step)
			if inverse {
				name += "/inv"
			}

			t.Run(name, func(t *testing.T) {
				t.Parallel()

				tableLen := c.n * c.step
				root := mathpkg.ComputeTwiddleFactors[complex64](tableLen)
				root128 := mathpkg.ComputeTwiddleFactors[complex128](tableLen)

				got := stageTwiddle64(c.n, c.radix, c.step, tableLen, inverse)
				if got == nil {
					t.Fatalf("stageTwiddle64(%d, %d, %d, %d) = nil", c.n, c.radix, c.step, tableLen)
				}

				got128 := stageTwiddle128(c.n, c.radix, c.step, tableLen, inverse)
				if got128 == nil {
					t.Fatal("stageTwiddle128 = nil")
				}

				span := c.n / c.radix
				for j := 1; j < c.radix; j++ {
					for k := range span {
						want := root[j*k*c.step]
						want128 := root128[j*k*c.step]

						if inverse {
							want = conj(want)
							want128 = conj(want128)
						}

						// The table is built from a size-n root table rather
						// than by subsampling the caller's, so allow a last-ulp
						// difference instead of demanding bit equality.
						if d := cmplx.Abs(complex128(got[j*span+k] - want)); d > 1e-6 {
							t.Fatalf("c64 table[%d*%d+%d] = %v, want %v (diff %g)",
								j, span, k, got[j*span+k], want, d)
						}

						if d := cmplx.Abs(got128[j*span+k] - want128); d > 1e-14 {
							t.Fatalf("c128 table[%d*%d+%d] = %v, want %v (diff %g)",
								j, span, k, got128[j*span+k], want128, d)
						}
					}
				}

				// Row 0 is the implicit unit row that lets the table and the
				// data share an index; the stage never multiplies through it.
				for k := range span {
					if got[k] != 1 || got128[k] != 1 {
						t.Fatalf("row 0 entry %d = %v/%v, want 1", k, got[k], got128[k])
					}
				}
			})
		}
	}
}

// TestStageTwiddleRejectsUnusableShapes checks the two conditions that must
// send a stage down the scalar path.
func TestStageTwiddleRejectsUnusableShapes(t *testing.T) {
	t.Parallel()

	// Invariant violated: the root table is longer than n*step, so its entries
	// are different roots of unity.
	if got := stageTwiddle64(32, 2, 3, 192, false); got != nil {
		t.Error("stageTwiddle64 accepted an oversized root table")
	}

	if got := stageTwiddle128(32, 2, 3, 192, false); got != nil {
		t.Error("stageTwiddle128 accepted an oversized root table")
	}

	// Radix the butterfly loop cannot execute. 13 is not in the schedule's
	// vocabulary; if it ever is, the stage must learn it before this changes.
	// n is chosen well above the size gate so only the radix can reject it.
	if got := stageTwiddle64(169, 13, 1, 169, false); got != nil {
		t.Error("stageTwiddle64 built a table for radix 13")
	}

	if mixedRadixStageVectorizable(169, 13) {
		t.Error("mixedRadixStageVectorizable(169, 13) = true")
	}

	// Radix 7 is admitted only where the fused kernel will execute it: the
	// two-pass form it would otherwise fall into measured slower than the
	// scalar stage at every size tried. Assert the coupling rather than a
	// fixed answer, since it is the machine that decides.
	if got, want := mixedRadixStageVectorizable(448, 7), mixedRadixStageFused(64, 7); got != want {
		t.Errorf("mixedRadixStageVectorizable(448, 7) = %v, want %v (fused kernel available)", got, want)
	}

	// The size gate still applies on top: n = 21 has 18 multiplies.
	if mixedRadixStageVectorizable(21, 7) {
		t.Error("mixedRadixStageVectorizable(21, 7) = true")
	}

	// Stage too small to be worth vectorising: n = 3, radix 3 is the span-1
	// tail of a deep schedule such as 2205 = [5 7 7 3 3].
	if mixedRadixStageVectorizable(3, 3) {
		t.Error("mixedRadixStageVectorizable(3, 3) = true")
	}

	if got := stageTwiddle64(3, 3, 735, 2205, false); got != nil {
		t.Error("stageTwiddle64 built a table for a span-1 stage")
	}
}

// TestMixedRadixStageMatchesScalar runs each supported radix through the
// vectorised stage and through an independent scalar re-derivation of the same
// butterfly, on random broadband data. This pins the fast path against the
// definition rather than against itself.
func TestMixedRadixStageMatchesScalar(t *testing.T) {
	t.Parallel()

	for _, radix := range []int{2, 3, 4, 5, 7, 8, 11} {
		for _, inverse := range []bool{false, true} {
			name := "r" + strconv.Itoa(radix)
			if inverse {
				name += "/inv"
			}

			t.Run(name, func(t *testing.T) {
				t.Parallel()

				const span = 64

				// Radix 7 only reaches this path where the fused kernel runs
				// it; elsewhere stageTwiddle64 correctly returns nil and the
				// stage stays scalar. TestMixedRadixStageGoRadix7 covers the
				// two-pass arm on every build.
				if radix == 7 && !mixedRadixStageFused(span, radix) {
					t.Skip("no fused radix-7 kernel on this build")
				}

				n := radix * span
				rng := rand.New(rand.NewSource(int64(n))) //nolint:gosec // deterministic test vector

				src := make([]complex64, n)
				for i := range src {
					src[i] = complex(float32(rng.NormFloat64()), float32(rng.NormFloat64()))
				}

				// Scalar reference: multiply by W_n^(j*k), then butterfly.
				root := mathpkg.ComputeTwiddleFactors[complex64](n)
				want := make([]complex64, n)

				for k := range span {
					a := make([]complex64, radix)
					a[0] = src[k]

					for j := 1; j < radix; j++ {
						w := root[j*k]
						if inverse {
							w = conj(w)
						}

						a[j] = mathpkg.MulComplex64(w, src[j*span+k])
					}

					out := scalarButterfly64(t, a, inverse)
					for j := range radix {
						want[j*span+k] = out[j]
					}
				}

				table := stageTwiddle64(n, radix, 1, n, inverse)
				if table == nil {
					t.Fatalf("stageTwiddle64(%d, %d, 1, %d) = nil", n, radix, n)
				}

				input := make([]complex64, n)
				copy(input, src)
				got := make([]complex64, n)

				mixedRadixStageComplex64(got, input, table, n, span, radix, inverse)

				for i := range got {
					if d := cmplx.Abs(complex128(got[i] - want[i])); d > 1e-4 {
						t.Fatalf("radix %d element %d: got %v want %v (diff %g)",
							radix, i, got[i], want[i], d)
					}
				}
			})
		}
	}
}

// scalarButterfly64 applies the radix-len(a) butterfly the mixed-radix driver
// uses, without any twiddle handling.
func scalarButterfly64(t *testing.T, a []complex64, inverse bool) []complex64 {
	t.Helper()

	switch len(a) {
	case 2:
		return []complex64{a[0] + a[1], a[0] - a[1]}
	case 3:
		if inverse {
			y0, y1, y2 := kernels.Butterfly3InverseComplex64(a[0], a[1], a[2])

			return []complex64{y0, y1, y2}
		}

		y0, y1, y2 := kernels.Butterfly3ForwardComplex64(a[0], a[1], a[2])

		return []complex64{y0, y1, y2}
	case 4:
		if inverse {
			y0, y1, y2, y3 := kernels.Butterfly4InverseComplex64(a[0], a[1], a[2], a[3])

			return []complex64{y0, y1, y2, y3}
		}

		y0, y1, y2, y3 := kernels.Butterfly4ForwardComplex64(a[0], a[1], a[2], a[3])

		return []complex64{y0, y1, y2, y3}
	case 5:
		if inverse {
			y0, y1, y2, y3, y4 := kernels.Butterfly5InverseComplex64(a[0], a[1], a[2], a[3], a[4])

			return []complex64{y0, y1, y2, y3, y4}
		}

		y0, y1, y2, y3, y4 := kernels.Butterfly5ForwardComplex64(a[0], a[1], a[2], a[3], a[4])

		return []complex64{y0, y1, y2, y3, y4}
	case 8:
		if inverse {
			y0, y1, y2, y3, y4, y5, y6, y7 := kernels.Butterfly8InverseComplex64(
				a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
			)

			return []complex64{y0, y1, y2, y3, y4, y5, y6, y7}
		}

		y0, y1, y2, y3, y4, y5, y6, y7 := kernels.Butterfly8ForwardComplex64(
			a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7],
		)

		return []complex64{y0, y1, y2, y3, y4, y5, y6, y7}
	case 7:
		var buf [7]complex64

		copy(buf[:], a)

		if inverse {
			kernels.Butterfly7InverseComplex64(&buf)
		} else {
			kernels.Butterfly7ForwardComplex64(&buf)
		}

		return buf[:]
	case 11:
		var buf [11]complex64

		copy(buf[:], a)

		if inverse {
			kernels.Butterfly11InverseComplex64(&buf)
		} else {
			kernels.Butterfly11ForwardComplex64(&buf)
		}

		return buf[:]
	default:
		t.Fatalf("scalarButterfly64: unsupported radix %d", len(a))

		return nil
	}
}
