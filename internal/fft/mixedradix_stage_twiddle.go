package fft

import (
	"sync"

	"github.com/cwbudde/algo-fft/internal/kernels"
	m "github.com/cwbudde/algo-fft/internal/math"
)

// Contiguous per-stage twiddle tables for the mixed-radix butterfly.
//
// The scalar stage applies its twiddles inside the k loop as
// twiddle[j*k*step] * input[j*span+k]. The data operand is contiguous in k,
// but the twiddle operand walks the root table with stride j*step, so the
// stage cannot be handed to a vectorised complex multiply as written.
//
// The stride is removable. The recursion maintains n*step == len(twiddle) at
// every node (the root has step == 1, and a child multiplies step by exactly
// the radix it divides n by), so with L = len(twiddle),
//
//	twiddle[j*k*step] = W_L^(j*k*step) = W_n^(j*k)
//
// and j*k < radix*span = n, so the whole stage is a permutation of the
// standard size-n twiddle table. Materialising it as a contiguous table laid
// out exactly like the data — entry j*span+k holds W_n^(j*k) — turns the
// stage into one in-place array multiply over input[span:n] followed by a
// twiddle-free butterfly loop. Rows j >= 1 are the only ones read; row 0 is
// all ones and exists only so the table and the data share an index.
//
// Tables are immutable once published, shared across plans, and keyed by the
// stage shape rather than by plan, which is what lets the recursion reach them
// without plan context. Compare mixedradix_leaf_twiddle.go, which does the
// same for whole sub-transforms dispatched to a codelet.

// stageTwiddleKey identifies one mixed-radix butterfly stage. The direction is
// part of the key because the inverse stage needs the conjugated table, and
// conjugating on the fly would put a per-element operation back in the loop
// this table exists to empty.
type stageTwiddleKey struct {
	n       int
	radix   int
	inverse bool
}

//nolint:gochecknoglobals
var (
	stageTwiddleCache64  sync.Map // map[stageTwiddleKey][]complex64
	stageTwiddleCache128 sync.Map // map[stageTwiddleKey][]complex128
)

// leafTwiddleUsable reports whether a table of standard size-n twiddle factors
// is interchangeable with a stride-step gather from a table of length tableLen,
// i.e. whether the recursion invariant n*step == len(twiddle) holds. Callers
// must check it rather than assume it: a longer table encodes different roots
// of unity, and substituting the size-n one would silently transform the wrong
// thing.
func leafTwiddleUsable(n, step, tableLen int) bool {
	return n > 0 && step > 0 && n*step == tableLen
}

// mixedRadixStageMinMuls is the smallest number of twiddle multiplies, i.e.
// n - span, for which the vectorised stage is worth taking. Below it the fixed
// costs — the table lookup, the extra pass over the data, and a call into the
// array multiply that cannot use its 4-wide body — exceed what the scalar
// stage spends on twiddles. The scalar stage is not slow in absolute terms:
// its twiddle operand is strided but stays inside an L1-resident table, so the
// vectorised form has to win on issue width alone.
//
// Both this threshold and the radix set below were measured on an i7-1255U
// (AVX2) as an interleaved sweep against the same binary with the path
// disabled, so they are tuning, not correctness:
//
//   - Ungated, deep schedules over small factors collapse: n = 2205 =
//     [5 7 7 3 3] ends in 245 span-3 and 735 span-1 stages and cost +80%
//     (complex64). Any threshold from 16 up removes that.
//   - Radix 7 loses at every threshold tried (16 … 256) *in the two-pass
//     form*: n = 448 = [7 64] has one radix-7 stage with 384 multiplies and
//     ran +6 to +8% slower with it vectorised. Excluding radix 7 and keeping
//     the rest moved the geomean over the mixed-radix benchmark set from
//     -2.3% to -3.2%. That measurement is about the two-pass form only — see
//     mixedRadixStageVectorizable, which admits radix 7 exactly when the
//     fused kernel will execute it. Radix 11 is the opposite case and is kept
//     unconditionally (n = 704 = [11 64], -7% complex64).
//
// The threshold was re-derived for the fused assembly stage (2026-07-27), on
// the expectation that a kernel paying none of those fixed costs would break
// even sooner. It does not: 64 is a local optimum for the fused path too, and
// one threshold serves both. Single-binary env-knob sweep, 6 canary-gated
// interleaved rounds, five lengths × both precisions × both directions,
// geomean against this value — 32: +8.9%, 48: +1.5%, 128: +2.7%. Each arm
// regressed exactly the lengths whose stage set it changed and left the others
// inside ±2.5%, which is what says the numbers are the threshold and not the
// machine: 48 admits only n = 2205's radix-7 span-9 stage and moves only 2205
// (+2.9…+7.4%); 32 also admits n = 1000's radix-5 span-8 stage (+20…+38%) and
// n = 3600's radix-3 span-16 stage (+7.5…+14%); 128 drops n = 3600's 96-multiply
// and n = 12000's 64-multiply stages and regresses those two alone (+6.4…+11.7%).
//
// The reason is that the fused kernel did not shed the fixed cost so much as
// replace it. Its prologue broadcasts up to six constants and derives up to six
// row offsets before the vector loop starts, and at span 8 that loop runs twice
// — so the break-even sits where the prologue amortises, not where a second
// pass over memory would have.
//
// The two-pass form's own value was re-derived on top of the dispatch hoist
// (2026-07-27), because the -3.2% and -7% above predate both the hoist and the
// fused kernels. It is gone. Three arms from one binary — fused where
// available, every stage forced two-pass, and no vectorised stage at all — over
// ten mixed-radix lengths in both precisions, 7 canary-gated interleaved rounds:
//
//   - The vectorised stage is worth -34% overall, and forcing the two-pass form
//     gives up +49.7% of that, so the fused kernels carry essentially all of it.
//   - The two-pass form alone is worth -1.5% on AVX2 and +1.3% on SSE2 (i.e. it
//     costs time there). n = 704 has no fused stage, so its two arms are the
//     same code and its cell measures the noise floor: +-2%. Neither figure
//     clears it, and per-cell they straddle zero (-7.5% at 448/c128, +9.9% at
//     768/c64).
//
// So the radix set below is no longer carrying a measurable win on amd64: on
// this hardware only n = 704's radix-11 stage still reached the two-pass path
// at all, and admitting it was worth +0.2%/-0.3%. It is kept because the same
// code is the only vectorised stage on the tiers that have no fused kernels
// (NEON, WASM, purego), which this machine cannot measure — not because it pays
// here.
//
// That last stage went fused too (2026-07-27, avx2_*_mixedradix_stage11.s), and
// it is the largest single win the fused kernels have produced: n = 704 dropped
// 10244 -> 2609 ns (complex64) and 11597 -> 4191 ns (complex128), i.e. -74% and
// -64%. Same protocol, 8 canary-gated rounds, all accepted; the eighteen cells
// with no radix-11 stage spanned -2.2%…+5.4%, so the effect is 30-60x the floor.
// Radix 11 needs no gate change for it: the size threshold already admitted it,
// and the fused kernel simply takes over where AVX2 is present.
//
// Note what that measurement is not. Roughly a 1.6x of the 3.9x is the
// conjugate-pair butterfly rather than the fusion — kernels.Butterfly11* is the
// full 11x11 matrix (100 complex multiplies against the pair form's 50 real-by-
// complex ones), and the same swap written in plain Go benchmarks 113.6 -> 72.1
// ns. That half is still on the table for every tier without a fused kernel;
// see PLAN.md P5.1.
const mixedRadixStageMinMuls = 64

// mixedRadixStageVectorizable reports whether the butterfly loop below can
// execute the given radix and whether the stage is large enough to be worth
// vectorising. It gates table construction so that an unsupported radix falls
// back to the scalar stage — which owns the panic that reports a
// scheduler/driver contract violation — instead of building a table no one
// reads.
//
// Radix 7 is admitted only where the fused kernel will run it: its two-pass
// form is a measured regression (see mixedRadixStageMinMuls), so the size
// threshold alone is not enough to justify leaving the scalar stage — the extra
// pass over memory is what it loses on, and only the fused kernel removes that
// pass.
func mixedRadixStageVectorizable(n, radix int) bool {
	span := n / radix

	if n-span < mixedRadixStageMinMuls {
		return false
	}

	switch radix {
	case 2, 3, 4, 5, 8, 11:
		return true
	case 7:
		return mixedRadixStageFused(span, radix)
	default:
		return false
	}
}

// stageTwiddle64 returns the contiguous stage table for a radix-r butterfly
// over a size-n sub-transform, or nil when the stage must use the scalar path:
// the recursion invariant does not hold, or the butterfly loop cannot execute
// this radix.
func stageTwiddle64(n, radix, step, tableLen int, inverse bool) []complex64 {
	if !leafTwiddleUsable(n, step, tableLen) || !mixedRadixStageVectorizable(n, radix) {
		return nil
	}

	key := stageTwiddleKey{n: n, radix: radix, inverse: inverse}
	if v, ok := stageTwiddleCache64.Load(key); ok {
		table, _ := v.([]complex64)

		return table
	}

	base := m.ComputeTwiddleFactors[complex64](n)
	span := n / radix
	table := make([]complex64, n)

	for k := range span {
		table[k] = 1
	}

	for j := 1; j < radix; j++ {
		row := table[j*span : (j+1)*span]
		for k := range row {
			w := base[j*k]
			if inverse {
				w = conj(w)
			}

			row[k] = w
		}
	}

	actual, _ := stageTwiddleCache64.LoadOrStore(key, table)
	stored, _ := actual.([]complex64)

	return stored
}

// stageTwiddle128 is the complex128 counterpart of stageTwiddle64.
func stageTwiddle128(n, radix, step, tableLen int, inverse bool) []complex128 {
	if !leafTwiddleUsable(n, step, tableLen) || !mixedRadixStageVectorizable(n, radix) {
		return nil
	}

	key := stageTwiddleKey{n: n, radix: radix, inverse: inverse}
	if v, ok := stageTwiddleCache128.Load(key); ok {
		table, _ := v.([]complex128)

		return table
	}

	base := m.ComputeTwiddleFactors[complex128](n)
	span := n / radix
	table := make([]complex128, n)

	for k := range span {
		table[k] = 1
	}

	for j := 1; j < radix; j++ {
		row := table[j*span : (j+1)*span]
		for k := range row {
			w := base[j*k]
			if inverse {
				w = conj(w)
			}

			row[k] = w
		}
	}

	actual, _ := stageTwiddleCache128.LoadOrStore(key, table)
	stored, _ := actual.([]complex128)

	return stored
}

// mixedRadixStageComplex64 applies one radix-r butterfly stage using the
// contiguous table from stageTwiddle64.
//
// The twiddles are applied first, as a single in-place array multiply over the
// r-1 non-trivial rows, which is contiguous on both operands and so takes the
// SIMD path. The butterfly loop that follows contains no twiddle arithmetic at
// all, and the radix switch sits outside it rather than inside.
//
// Multiplying input in place is safe: input is this node's dst or work buffer,
// never src, and each k touches only the r positions j*span+k, reading all of
// them before writing any — so dst may alias input, as it does at a leaf node.
func mixedRadixStageComplex64(dst, input, table []complex64, n, span, radix int, inverse bool) {
	// Preferred: one fused assembly pass that never writes the twiddled rows
	// back to memory. Available for a subset of the radices this function can
	// execute; see mixedradix_stage_asm_amd64.go.
	if mixedRadixStageAsm64(dst, input, table, n, span, radix, inverse) {
		return
	}

	ComplexMulArrayInPlaceComplex64(input[span:n], table[span:n])

	switch radix {
	case 2:
		for k := range span {
			a0, a1 := input[k], input[span+k]
			dst[k] = a0 + a1
			dst[span+k] = a0 - a1
		}
	case 3:
		for k := range span {
			var y0, y1, y2 complex64
			if inverse {
				y0, y1, y2 = kernels.Butterfly3InverseComplex64(input[k], input[span+k], input[2*span+k])
			} else {
				y0, y1, y2 = kernels.Butterfly3ForwardComplex64(input[k], input[span+k], input[2*span+k])
			}

			dst[k], dst[span+k], dst[2*span+k] = y0, y1, y2
		}
	case 4:
		for k := range span {
			var y0, y1, y2, y3 complex64
			if inverse {
				y0, y1, y2, y3 = kernels.Butterfly4InverseComplex64(
					input[k], input[span+k], input[2*span+k], input[3*span+k],
				)
			} else {
				y0, y1, y2, y3 = kernels.Butterfly4ForwardComplex64(
					input[k], input[span+k], input[2*span+k], input[3*span+k],
				)
			}

			dst[k], dst[span+k], dst[2*span+k], dst[3*span+k] = y0, y1, y2, y3
		}
	case 5:
		for k := range span {
			var y0, y1, y2, y3, y4 complex64
			if inverse {
				y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex64(
					input[k], input[span+k], input[2*span+k], input[3*span+k], input[4*span+k],
				)
			} else {
				y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex64(
					input[k], input[span+k], input[2*span+k], input[3*span+k], input[4*span+k],
				)
			}

			dst[k], dst[span+k], dst[2*span+k] = y0, y1, y2
			dst[3*span+k], dst[4*span+k] = y3, y4
		}
	case 8:
		for k := range span {
			var y0, y1, y2, y3, y4, y5, y6, y7 complex64
			if inverse {
				y0, y1, y2, y3, y4, y5, y6, y7 = kernels.Butterfly8InverseComplex64(
					input[k], input[span+k], input[2*span+k], input[3*span+k],
					input[4*span+k], input[5*span+k], input[6*span+k], input[7*span+k],
				)
			} else {
				y0, y1, y2, y3, y4, y5, y6, y7 = kernels.Butterfly8ForwardComplex64(
					input[k], input[span+k], input[2*span+k], input[3*span+k],
					input[4*span+k], input[5*span+k], input[6*span+k], input[7*span+k],
				)
			}

			dst[k], dst[span+k], dst[2*span+k], dst[3*span+k] = y0, y1, y2, y3
			dst[4*span+k], dst[5*span+k], dst[6*span+k], dst[7*span+k] = y4, y5, y6, y7
		}
	case 7:
		// Reachable only on the fallback side of the fused kernel: the gate in
		// mixedRadixStageVectorizable admits radix 7 only when
		// mixedRadixStageFused is true, but that predicate can go stale
		// relative to a given call (forced CPU features in a test, a span the
		// kernel's vector loop cannot cover). Executing the stage is the
		// correct response; panicking here would be a crash on a legal input.
		for k := range span {
			var a [7]complex64
			for j := range 7 {
				a[j] = input[j*span+k]
			}

			if inverse {
				kernels.Butterfly7InverseComplex64(&a)
			} else {
				kernels.Butterfly7ForwardComplex64(&a)
			}

			for j := range 7 {
				dst[j*span+k] = a[j]
			}
		}
	case 11:
		for k := range span {
			var a [11]complex64
			for j := range 11 {
				a[j] = input[j*span+k]
			}

			if inverse {
				kernels.Butterfly11InverseComplex64(&a)
			} else {
				kernels.Butterfly11ForwardComplex64(&a)
			}

			for j := range 11 {
				dst[j*span+k] = a[j]
			}
		}
	default:
		// Unreachable: stageTwiddle64 returns nil for any radix this switch
		// cannot execute — including radix 7, which the scalar stage keeps —
		// so the caller takes the scalar path instead.
		panic("algofft: mixed-radix stage reached an unvectorizable radix")
	}
}

// mixedRadixStageComplex128 is the complex128 counterpart of
// mixedRadixStageComplex64. See it for why the in-place multiply is safe.
func mixedRadixStageComplex128(dst, input, table []complex128, n, span, radix int, inverse bool) {
	// See the complex64 twin for why the fused kernel comes first.
	if mixedRadixStageAsm128(dst, input, table, n, span, radix, inverse) {
		return
	}

	ComplexMulArrayInPlaceComplex128(input[span:n], table[span:n])

	switch radix {
	case 2:
		for k := range span {
			a0, a1 := input[k], input[span+k]
			dst[k] = a0 + a1
			dst[span+k] = a0 - a1
		}
	case 3:
		for k := range span {
			var y0, y1, y2 complex128
			if inverse {
				y0, y1, y2 = kernels.Butterfly3InverseComplex128(input[k], input[span+k], input[2*span+k])
			} else {
				y0, y1, y2 = kernels.Butterfly3ForwardComplex128(input[k], input[span+k], input[2*span+k])
			}

			dst[k], dst[span+k], dst[2*span+k] = y0, y1, y2
		}
	case 4:
		for k := range span {
			var y0, y1, y2, y3 complex128
			if inverse {
				y0, y1, y2, y3 = kernels.Butterfly4InverseComplex128(
					input[k], input[span+k], input[2*span+k], input[3*span+k],
				)
			} else {
				y0, y1, y2, y3 = kernels.Butterfly4ForwardComplex128(
					input[k], input[span+k], input[2*span+k], input[3*span+k],
				)
			}

			dst[k], dst[span+k], dst[2*span+k], dst[3*span+k] = y0, y1, y2, y3
		}
	case 5:
		for k := range span {
			var y0, y1, y2, y3, y4 complex128
			if inverse {
				y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex128(
					input[k], input[span+k], input[2*span+k], input[3*span+k], input[4*span+k],
				)
			} else {
				y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex128(
					input[k], input[span+k], input[2*span+k], input[3*span+k], input[4*span+k],
				)
			}

			dst[k], dst[span+k], dst[2*span+k] = y0, y1, y2
			dst[3*span+k], dst[4*span+k] = y3, y4
		}
	case 8:
		for k := range span {
			var y0, y1, y2, y3, y4, y5, y6, y7 complex128
			if inverse {
				y0, y1, y2, y3, y4, y5, y6, y7 = kernels.Butterfly8InverseComplex128(
					input[k], input[span+k], input[2*span+k], input[3*span+k],
					input[4*span+k], input[5*span+k], input[6*span+k], input[7*span+k],
				)
			} else {
				y0, y1, y2, y3, y4, y5, y6, y7 = kernels.Butterfly8ForwardComplex128(
					input[k], input[span+k], input[2*span+k], input[3*span+k],
					input[4*span+k], input[5*span+k], input[6*span+k], input[7*span+k],
				)
			}

			dst[k], dst[span+k], dst[2*span+k], dst[3*span+k] = y0, y1, y2, y3
			dst[4*span+k], dst[5*span+k], dst[6*span+k], dst[7*span+k] = y4, y5, y6, y7
		}
	case 7:
		// Reachable only on the fallback side of the fused kernel: the gate in
		// mixedRadixStageVectorizable admits radix 7 only when
		// mixedRadixStageFused is true, but that predicate can go stale
		// relative to a given call (forced CPU features in a test, a span the
		// kernel's vector loop cannot cover). Executing the stage is the
		// correct response; panicking here would be a crash on a legal input.
		for k := range span {
			var a [7]complex128
			for j := range 7 {
				a[j] = input[j*span+k]
			}

			if inverse {
				kernels.Butterfly7InverseComplex128(&a)
			} else {
				kernels.Butterfly7ForwardComplex128(&a)
			}

			for j := range 7 {
				dst[j*span+k] = a[j]
			}
		}
	case 11:
		for k := range span {
			var a [11]complex128
			for j := range 11 {
				a[j] = input[j*span+k]
			}

			if inverse {
				kernels.Butterfly11InverseComplex128(&a)
			} else {
				kernels.Butterfly11ForwardComplex128(&a)
			}

			for j := range 11 {
				dst[j*span+k] = a[j]
			}
		}
	default:
		// Unreachable: see the complex64 twin.
		panic("algofft: mixed-radix stage reached an unvectorizable radix")
	}
}
