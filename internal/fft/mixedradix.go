package fft

import (
	"math/bits"
	"strconv"
	"sync"

	"github.com/cwbudde/algo-fft/internal/kernels"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/registry"
)

const mixedRadixMaxStages = 64

// radixSchedulePool recycles the per-call radix schedule buffer. The buffer
// escapes to the heap because its slice flows into the indirect recursion hook
// (recursiveStep64/128), which defeats stack allocation; pooling keeps the
// mixed-radix path allocation-free after warm-up, matching the zero-allocation
// guarantee the power-of-2 kernels already meet. The buffer is used only within
// a single synchronous transform, so it is safe to recycle.
//
//nolint:gochecknoglobals
var radixSchedulePool = sync.Pool{
	New: func() any { return new([mixedRadixMaxStages]int) },
}

func forwardMixedRadixComplex64(dst, src, twiddle, scratch []complex64) bool {
	return mixedRadixForward[complex64](dst, src, twiddle, scratch)
}

func inverseMixedRadixComplex64(dst, src, twiddle, scratch []complex64) bool {
	return mixedRadixInverse[complex64](dst, src, twiddle, scratch)
}

func forwardMixedRadixComplex128(dst, src, twiddle, scratch []complex128) bool {
	return mixedRadixForward[complex128](dst, src, twiddle, scratch)
}

func inverseMixedRadixComplex128(dst, src, twiddle, scratch []complex128) bool {
	return mixedRadixInverse[complex128](dst, src, twiddle, scratch)
}

func mixedRadixForward[T Complex](dst, src, twiddle, scratch []T) bool {
	return mixedRadixTransform(dst, src, twiddle, scratch, false)
}

func mixedRadixInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	return mixedRadixTransform(dst, src, twiddle, scratch, true)
}

// Recursion hooks for SIMD acceleration.
// By default, these point to the pure Go implementations.
// SIMD-optimized files (like mixedradix_avx2.go) can override these in init().
//
// The trailing leaf parameter carries the codelet the schedule's final stage
// resolves to, or nil when that stage has none; see leafCodelet64/128.
var (
	recursiveStep64  func(dst, src, work []complex64, n, stride, step int, radices []int, twiddle []complex64, inverse bool, leaf *registry.CodeletEntry[complex64])
	recursiveStep128 func(dst, src, work []complex128, n, stride, step int, radices []int, twiddle []complex128, inverse bool, leaf *registry.CodeletEntry[complex128])
)

// leafCodelet64/128 resolve the codelet a leaf sub-transform of the given radix
// will be dispatched to, or nil when the pure Go butterfly must handle it.
//
// This is the same lookup codeletSchedulable64/128 wraps, hoisted out of the
// recursion: the scheduler emits a composite radix only as the schedule's final
// stage, and checks the registry for the remaining size at every step before
// that, so a codelet can only ever match at a leaf — every leaf of one transform
// has the same size, and therefore the same entry. Resolving it once per
// transform instead of once per node removes a feature detection, a map lookup
// and a priority scan from each of the (up to n/radix) leaf dispatches.
var (
	leafCodelet64  func(radix int) *registry.CodeletEntry[complex64]
	leafCodelet128 func(radix int) *registry.CodeletEntry[complex128]
)

// codeletSchedulable64/128 report whether the installed recursion driver can
// execute a composite radix of the given size directly via a codelet. The
// pure Go driver only knows radices 2/3/4/5/7/8/11, so scheduling any larger radix
// would panic at transform time (its butterfly switch treats unknown radices
// as a scheduler/driver contract violation). SIMD builds (mixedradix_avx2.go)
// override these with a predicate matching exactly what their recursion hook
// dispatches.
var (
	codeletSchedulable64  func(int) bool
	codeletSchedulable128 func(int) bool
)

//nolint:gochecknoinits
func init() {
	recursiveStep64 = mixedRadixRecursivePingPongComplex64
	recursiveStep128 = mixedRadixRecursivePingPongComplex128
	codeletSchedulable64 = func(int) bool { return false }
	codeletSchedulable128 = func(int) bool { return false }
	leafCodelet64 = func(int) *registry.CodeletEntry[complex64] { return nil }
	leafCodelet128 = func(int) *registry.CodeletEntry[complex128] { return nil }
}

func mixedRadixTransform[T Complex](dst, src, twiddle, scratch []T, inverse bool) bool {
	n := len(src)
	if n == 0 {
		return true
	}

	if len(dst) < n || len(twiddle) < n || len(scratch) < n {
		return false
	}

	if n == 1 {
		dst[0] = src[0]
		return true
	}

	var (
		hasCodelet func(int) bool
		zero       T
	)

	// The schedule buffer is pooled rather than stack-allocated because it
	// escapes through the indirect recursion hook below (see radixSchedulePool).
	radices := radixSchedulePool.Get().(*[mixedRadixMaxStages]int) //nolint:forcetypeassert
	defer radixSchedulePool.Put(radices)

	// A composite radix may only be scheduled when the recursion driver can
	// dispatch it to a codelet (see codeletSchedulable64/128). Checking the
	// registry alone is not enough: generic (non-SIMD) codelets are registered
	// there too, but the drivers cannot invoke them for sub-transforms.

	switch any(zero).(type) {
	case complex64:
		hasCodelet = codeletSchedulable64
	case complex128:
		hasCodelet = codeletSchedulable128
	default:
		hasCodelet = func(int) bool { return false }
	}

	stageCount := mixedRadixSchedule(n, radices, hasCodelet)
	if stageCount == 0 {
		return false
	}

	work := dst
	workIsDst := true

	if kernels.SameSlice(dst, src) {
		work = scratch
		workIsDst = false
	}

	// Call through recursion hooks.
	// Resolve the leaf codelet once, from the schedule's final stage, and thread
	// it down the recursion (see leafCodelet64/128).
	leafRadix := radices[stageCount-1]

	switch any(zero).(type) {
	case complex64:
		var leaf *registry.CodeletEntry[complex64]
		if leafRadix > mixedRadixCodeletMinSize {
			leaf = leafCodelet64(leafRadix)
		}

		recursiveStep64(
			any(work).([]complex64),    //nolint:forcetypeassert
			any(src).([]complex64),     //nolint:forcetypeassert
			any(scratch).([]complex64), //nolint:forcetypeassert
			n, 1, 1, radices[:stageCount],
			any(twiddle).([]complex64), //nolint:forcetypeassert
			inverse, leaf,
		)
	case complex128:
		var leaf *registry.CodeletEntry[complex128]
		if leafRadix > mixedRadixCodeletMinSize {
			leaf = leafCodelet128(leafRadix)
		}

		recursiveStep128(
			any(work).([]complex128),    //nolint:forcetypeassert
			any(src).([]complex128),     //nolint:forcetypeassert
			any(scratch).([]complex128), //nolint:forcetypeassert
			n, 1, 1, radices[:stageCount],
			any(twiddle).([]complex128), //nolint:forcetypeassert
			inverse, leaf,
		)
	default:
		return false
	}

	if !workIsDst {
		copy(dst, work)
	}

	if inverse {
		scale := complexFromFloat64[T](1.0/float64(n), 0)
		for i := range dst {
			dst[i] *= scale
		}
	}

	return true
}

// mixedRadixCodeletMinSize is the smallest sub-transform the drivers will hand
// to a codelet. Radices at or below it (2, 3, 4, 5) have a pure-Go butterfly
// that the recursion executes inline; routing them through a codelet instead
// costs a call, a strided twiddle gather and two sync.Pool round-trips to do a
// handful of butterflies. The scheduler has always used this bound when
// deciding whether to emit a composite radix, so matching it in the dispatch
// hooks keeps dispatch a superset of what the schedule can emit — the
// invariant that stops a scheduled composite radix from reaching the pure-Go
// butterfly, which panics on radices it cannot execute.
//
// Measured at n = 4900 = [5,5,7,7,4] (complex64, AVX2), where the trailing
// radix-4 leaf ran 1225 codelet dispatches per transform.
const mixedRadixCodeletMinSize = 5

func mixedRadixSchedule(n int, radices *[mixedRadixMaxStages]int, hasCodelet func(int) bool) int {
	if n < 2 {
		return 0
	}

	count := 0

	// Registry-aware scheduling:
	// If we have a registered codelet for the current size 'n', use it directly!
	// This prevents breaking down large sizes (e.g., 256, 512) into small radices
	// when we have highly optimized AVX2 kernels for them.
	//
	// We verify if *any* codelet exists, not just AVX2, because even a generic
	// codelet for size N might be faster than recursive decomposition.
	//
	// Note: We skip this check for very small sizes (<= 5) as they are handled
	// by the switch statement anyway, and looking them up might be slower.
	if n > 5 && hasCodelet(n) {
		radices[count] = n
		return count + 1
	}

	// oddFirst: strip odd factors before the powers of two when the
	// power-of-two part of n (or a radix-4 suffix of it) can be dispatched to
	// a codelet — that keeps it intact for the per-step codelet check below.
	// Without a reachable codelet the reorder pays and buys nothing.
	oddFirst := false

	for pow2 := n & -n; pow2 >= 8; pow2 /= 4 {
		if hasCodelet(pow2) {
			oddFirst = true
			break
		}
	}

	for n > 1 {
		// Check again at each step: if the remaining size 'n' has a kernel, use it.
		// e.g., 768 = 3 * 256. First loop picks 3. Second loop sees 256.
		// Instead of 256 -> 4*4*4*4, we want 256 directly.
		if n > 5 && hasCodelet(n) {
			radices[count] = n
			count++

			return count
		}

		switch {
		case n%5 == 0:
			radices[count] = 5
			n /= 5
		case n%7 == 0:
			// Factors 7 and 11 are stripped unconditionally, like 5: they
			// must come off before the power-of-two cases below can claim the
			// remainder, and pulling them ahead of the pow2 part keeps that
			// part intact for the per-step codelet check above.
			radices[count] = 7
			n /= 7
		case n%11 == 0:
			radices[count] = 11
			n /= 11
		case oddFirst && n%3 == 0:
			// Strip the factors of 3 before the powers of two so the
			// power-of-two part stays intact until the codelet check above
			// can claim it whole: 768 schedules as [3, 256] with a tuned
			// size-256 codelet leaf instead of fragmenting into
			// [4, 4, 4, 4, 3]. Applied only when a codelet actually exists
			// for a power-of-two suffix (see oddFirst) — otherwise the
			// reorder measured slower than the radix-4-major order below.
			radices[count] = 3
			n /= 3
		case !oddFirst && radix8Profitable(n):
			// A radix-8 stage covers three powers of two per pass instead of
			// two, cutting the pass count (and per-point twiddle loads) for
			// power-of-two parts 2^e with e ≥ 3: 2^5 runs as [8,4] instead of
			// [4,4,2], 2^6 as [8,8] instead of [4,4,4]. Skipped when a
			// codelet is reachable on the radix-4 suffix chain (oddFirst) —
			// splitting by 8 would step over the codelet size and lose the
			// tuned leaf. e == 4 keeps [4,4] over [8,2]: same pass count,
			// and the radix-2 tail measured slower.
			radices[count] = 8
			n /= 8
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
		if count >= mixedRadixMaxStages {
			return 0
		}
	}

	return count
}

// radix8Profitable reports whether the scheduler should emit a radix-8 stage
// for the current remaining size n. The power-of-two part 2^e of n must hold
// at least three factors of two, and e == 4 is excluded so 2^4 stays [4,4]
// rather than [8,2] (equal pass count, slower radix-2 tail).
func radix8Profitable(n int) bool {
	e := bits.TrailingZeros(uint(n))
	return e >= 3 && e != 4
}

// mixedRadixRecursivePingPongComplex64 is a specialized complex64 version that calls
// type-specific butterfly functions to avoid generic overhead.
func mixedRadixRecursivePingPongComplex64(dst, src, work []complex64, n, stride, step int, radices []int, twiddle []complex64, inverse bool, leaf *registry.CodeletEntry[complex64]) {
	if n == 1 {
		dst[0] = src[0]
		return
	}

	radix := radices[0]
	span := n / radix
	nextRadices := radices[1:]

	// Recursively process sub-transforms
	for j := range radix {
		if len(nextRadices) == 0 {
			dst[j*span] = src[j*stride]
		} else {
			recursiveStep64(work[j*span:], src[j*stride:], dst[j*span:], span, stride*radix, step*radix, nextRadices, twiddle, inverse, leaf)
		}
	}

	// Determine where the recursive calls wrote their data
	var input []complex64
	if len(nextRadices) == 0 {
		input = dst
	} else {
		input = work
	}

	// Preferred path: apply the stage's twiddles as one contiguous array
	// multiply, then run a twiddle-free butterfly loop. Available whenever the
	// recursion invariant holds and the radix is one the stage can execute;
	// see mixedradix_stage_twiddle.go.
	if table := stageTwiddle64(n, radix, step, len(twiddle), inverse); table != nil {
		mixedRadixStageComplex64(dst, input, table, n, span, radix, inverse)

		return
	}

	// Fallback: the scalar stage, which reads each twiddle from the root
	// table with stride j*step. Radices 7, 8 and 11 live in their own function;
	// radices 2-5 stay inline because the deep schedules that miss the
	// vectorised path pay a call per stage otherwise -- n = 2205 = [5 7 7 3 3]
	// runs 980 span-1 and span-3 radix-3 stages, and extracting them cost it
	// about 5 percent.
	if radix == 7 || radix == 8 || radix == 11 {
		mixedRadixWideScalarStageComplex64(dst, input, twiddle, span, step, radix, inverse)

		return
	}

	for k := range span {
		switch radix {
		case 2:
			w1 := twiddle[k*step]
			if inverse {
				w1 = conj(w1)
			}

			a0 := input[k]
			a1 := m.MulComplex64(w1, input[span+k])

			dst[k] = a0 + a1
			dst[span+k] = a0 - a1
		case 3:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
			}

			a0 := input[k]
			a1 := m.MulComplex64(w1, input[span+k])
			a2 := m.MulComplex64(w2, input[2*span+k])

			var y0, y1, y2 complex64
			if inverse {
				y0, y1, y2 = kernels.Butterfly3InverseComplex64(a0, a1, a2)
			} else {
				y0, y1, y2 = kernels.Butterfly3ForwardComplex64(a0, a1, a2)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
		case 4:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
			}

			a0 := input[k]
			a1 := m.MulComplex64(w1, input[span+k])
			a2 := m.MulComplex64(w2, input[2*span+k])
			a3 := m.MulComplex64(w3, input[3*span+k])

			var y0, y1, y2, y3 complex64
			if inverse {
				y0, y1, y2, y3 = kernels.Butterfly4InverseComplex64(a0, a1, a2, a3)
			} else {
				y0, y1, y2, y3 = kernels.Butterfly4ForwardComplex64(a0, a1, a2, a3)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
		case 5:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]
			w4 := twiddle[4*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
				w4 = conj(w4)
			}

			a0 := input[k]
			a1 := m.MulComplex64(w1, input[span+k])
			a2 := m.MulComplex64(w2, input[2*span+k])
			a3 := m.MulComplex64(w3, input[3*span+k])
			a4 := m.MulComplex64(w4, input[4*span+k])

			var y0, y1, y2, y3, y4 complex64
			if inverse {
				y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex64(a0, a1, a2, a3, a4)
			} else {
				y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex64(a0, a1, a2, a3, a4)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
			dst[4*span+k] = y4
		default:
			// A radix the driver cannot execute means the scheduler and the
			// recursion hook disagree — a programming error, never a runtime
			// input error. Returning here would leave dst partially written
			// and surface as a wrong answer with a nil error.
			panic("algofft: mixed-radix driver cannot execute radix " + strconv.Itoa(radix) +
				" (scheduler/driver contract violation)")
		}
	}
}

// mixedRadixWideScalarStageComplex64 applies the scalar radix-7, -8 and -11
// butterfly stages. They are split out of the driver because they carry most of
// its cyclomatic complexity while running far less often than radices 2-5.
func mixedRadixWideScalarStageComplex64(dst, input, twiddle []complex64, span, step, radix int, inverse bool) {
	for k := range span {
		switch radix {
		case 7:
			var a [7]complex64

			a[0] = input[k]
			for j := 1; j < 7; j++ {
				w := twiddle[j*k*step]
				if inverse {
					w = conj(w)
				}

				a[j] = m.MulComplex64(w, input[j*span+k])
			}

			if inverse {
				kernels.Butterfly7InverseComplex64(&a)
			} else {
				kernels.Butterfly7ForwardComplex64(&a)
			}

			for j := range 7 {
				dst[j*span+k] = a[j]
			}
		case 11:
			var a [11]complex64

			a[0] = input[k]
			for j := 1; j < 11; j++ {
				w := twiddle[j*k*step]
				if inverse {
					w = conj(w)
				}

				a[j] = m.MulComplex64(w, input[j*span+k])
			}

			if inverse {
				kernels.Butterfly11InverseComplex64(&a)
			} else {
				kernels.Butterfly11ForwardComplex64(&a)
			}

			for j := range 11 {
				dst[j*span+k] = a[j]
			}
		case 8:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]
			w4 := twiddle[4*k*step]
			w5 := twiddle[5*k*step]
			w6 := twiddle[6*k*step]
			w7 := twiddle[7*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
				w4 = conj(w4)
				w5 = conj(w5)
				w6 = conj(w6)
				w7 = conj(w7)
			}

			a0 := input[k]
			a1 := m.MulComplex64(w1, input[span+k])
			a2 := m.MulComplex64(w2, input[2*span+k])
			a3 := m.MulComplex64(w3, input[3*span+k])
			a4 := m.MulComplex64(w4, input[4*span+k])
			a5 := m.MulComplex64(w5, input[5*span+k])
			a6 := m.MulComplex64(w6, input[6*span+k])
			a7 := m.MulComplex64(w7, input[7*span+k])

			var y0, y1, y2, y3, y4, y5, y6, y7 complex64
			if inverse {
				y0, y1, y2, y3, y4, y5, y6, y7 = kernels.Butterfly8InverseComplex64(a0, a1, a2, a3, a4, a5, a6, a7)
			} else {
				y0, y1, y2, y3, y4, y5, y6, y7 = kernels.Butterfly8ForwardComplex64(a0, a1, a2, a3, a4, a5, a6, a7)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
			dst[4*span+k] = y4
			dst[5*span+k] = y5
			dst[6*span+k] = y6
			dst[7*span+k] = y7
		default:
			// A radix the driver cannot execute means the scheduler and the
			// recursion hook disagree - a programming error, never a runtime
			// input error. Returning here would leave dst partially written
			// and surface as a wrong answer with a nil error.
			panic("algofft: mixed-radix driver cannot execute radix " + strconv.Itoa(radix) +
				" (scheduler/driver contract violation)")
		}
	}
}

// mixedRadixRecursivePingPongComplex128 is a specialized complex128 version that calls
// type-specific butterfly functions to avoid generic overhead.
func mixedRadixRecursivePingPongComplex128(dst, src, work []complex128, n, stride, step int, radices []int, twiddle []complex128, inverse bool, leaf *registry.CodeletEntry[complex128]) {
	if n == 1 {
		dst[0] = src[0]
		return
	}

	radix := radices[0]
	span := n / radix
	nextRadices := radices[1:]

	// Recursively process sub-transforms
	for j := range radix {
		if len(nextRadices) == 0 {
			dst[j*span] = src[j*stride]
		} else {
			recursiveStep128(work[j*span:], src[j*stride:], dst[j*span:], span, stride*radix, step*radix, nextRadices, twiddle, inverse, leaf)
		}
	}

	// Determine where the recursive calls wrote their data
	var input []complex128
	if len(nextRadices) == 0 {
		input = dst
	} else {
		input = work
	}

	// See the complex64 twin for why this path is preferred.
	if table := stageTwiddle128(n, radix, step, len(twiddle), inverse); table != nil {
		mixedRadixStageComplex128(dst, input, table, n, span, radix, inverse)

		return
	}

	// See the complex64 twin for why radices 2-5 stay inline here.
	if radix == 7 || radix == 8 || radix == 11 {
		mixedRadixWideScalarStageComplex128(dst, input, twiddle, span, step, radix, inverse)

		return
	}

	for k := range span {
		switch radix {
		case 2:
			w1 := twiddle[k*step]
			if inverse {
				w1 = conj(w1)
			}

			a0 := input[k]
			a1 := m.MulComplex128(w1, input[span+k])

			dst[k] = a0 + a1
			dst[span+k] = a0 - a1
		case 3:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
			}

			a0 := input[k]
			a1 := m.MulComplex128(w1, input[span+k])
			a2 := m.MulComplex128(w2, input[2*span+k])

			var y0, y1, y2 complex128
			if inverse {
				y0, y1, y2 = kernels.Butterfly3InverseComplex128(a0, a1, a2)
			} else {
				y0, y1, y2 = kernels.Butterfly3ForwardComplex128(a0, a1, a2)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
		case 4:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
			}

			a0 := input[k]
			a1 := m.MulComplex128(w1, input[span+k])
			a2 := m.MulComplex128(w2, input[2*span+k])
			a3 := m.MulComplex128(w3, input[3*span+k])

			var y0, y1, y2, y3 complex128
			if inverse {
				y0, y1, y2, y3 = kernels.Butterfly4InverseComplex128(a0, a1, a2, a3)
			} else {
				y0, y1, y2, y3 = kernels.Butterfly4ForwardComplex128(a0, a1, a2, a3)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
		case 5:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]
			w4 := twiddle[4*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
				w4 = conj(w4)
			}

			a0 := input[k]
			a1 := m.MulComplex128(w1, input[span+k])
			a2 := m.MulComplex128(w2, input[2*span+k])
			a3 := m.MulComplex128(w3, input[3*span+k])
			a4 := m.MulComplex128(w4, input[4*span+k])

			var y0, y1, y2, y3, y4 complex128
			if inverse {
				y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex128(a0, a1, a2, a3, a4)
			} else {
				y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex128(a0, a1, a2, a3, a4)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
			dst[4*span+k] = y4
		default:
			// A radix the driver cannot execute means the scheduler and the
			// recursion hook disagree — a programming error, never a runtime
			// input error. Returning here would leave dst partially written
			// and surface as a wrong answer with a nil error.
			panic("algofft: mixed-radix driver cannot execute radix " + strconv.Itoa(radix) +
				" (scheduler/driver contract violation)")
		}
	}
}

// mixedRadixWideScalarStageComplex128 applies the scalar radix-7, -8 and -11
// butterfly stages. They are split out of the driver because they carry most of
// its cyclomatic complexity while running far less often than radices 2-5.
func mixedRadixWideScalarStageComplex128(dst, input, twiddle []complex128, span, step, radix int, inverse bool) {
	for k := range span {
		switch radix {
		case 7:
			var a [7]complex128

			a[0] = input[k]
			for j := 1; j < 7; j++ {
				w := twiddle[j*k*step]
				if inverse {
					w = conj(w)
				}

				a[j] = m.MulComplex128(w, input[j*span+k])
			}

			if inverse {
				kernels.Butterfly7InverseComplex128(&a)
			} else {
				kernels.Butterfly7ForwardComplex128(&a)
			}

			for j := range 7 {
				dst[j*span+k] = a[j]
			}
		case 11:
			var a [11]complex128

			a[0] = input[k]
			for j := 1; j < 11; j++ {
				w := twiddle[j*k*step]
				if inverse {
					w = conj(w)
				}

				a[j] = m.MulComplex128(w, input[j*span+k])
			}

			if inverse {
				kernels.Butterfly11InverseComplex128(&a)
			} else {
				kernels.Butterfly11ForwardComplex128(&a)
			}

			for j := range 11 {
				dst[j*span+k] = a[j]
			}
		case 8:
			w1 := twiddle[k*step]
			w2 := twiddle[2*k*step]
			w3 := twiddle[3*k*step]
			w4 := twiddle[4*k*step]
			w5 := twiddle[5*k*step]
			w6 := twiddle[6*k*step]
			w7 := twiddle[7*k*step]

			if inverse {
				w1 = conj(w1)
				w2 = conj(w2)
				w3 = conj(w3)
				w4 = conj(w4)
				w5 = conj(w5)
				w6 = conj(w6)
				w7 = conj(w7)
			}

			a0 := input[k]
			a1 := m.MulComplex128(w1, input[span+k])
			a2 := m.MulComplex128(w2, input[2*span+k])
			a3 := m.MulComplex128(w3, input[3*span+k])
			a4 := m.MulComplex128(w4, input[4*span+k])
			a5 := m.MulComplex128(w5, input[5*span+k])
			a6 := m.MulComplex128(w6, input[6*span+k])
			a7 := m.MulComplex128(w7, input[7*span+k])

			var y0, y1, y2, y3, y4, y5, y6, y7 complex128
			if inverse {
				y0, y1, y2, y3, y4, y5, y6, y7 = kernels.Butterfly8InverseComplex128(a0, a1, a2, a3, a4, a5, a6, a7)
			} else {
				y0, y1, y2, y3, y4, y5, y6, y7 = kernels.Butterfly8ForwardComplex128(a0, a1, a2, a3, a4, a5, a6, a7)
			}

			dst[k] = y0
			dst[span+k] = y1
			dst[2*span+k] = y2
			dst[3*span+k] = y3
			dst[4*span+k] = y4
			dst[5*span+k] = y5
			dst[6*span+k] = y6
			dst[7*span+k] = y7
		default:
			// A radix the driver cannot execute means the scheduler and the
			// recursion hook disagree - a programming error, never a runtime
			// input error. Returning here would leave dst partially written
			// and surface as a wrong answer with a nil error.
			panic("algofft: mixed-radix driver cannot execute radix " + strconv.Itoa(radix) +
				" (scheduler/driver contract violation)")
		}
	}
}
