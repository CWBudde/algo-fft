package kernels

import (
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/math"
)

// ForwardFourStepComplex64 performs a forward four-step FFT on complex64 data.
func ForwardFourStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	return fourStepForward[complex64](dst, src, twiddle, scratch)
}

// InverseFourStepComplex64 performs an inverse four-step FFT on complex64 data.
func InverseFourStepComplex64(dst, src, twiddle, scratch []complex64) bool {
	return fourStepInverse[complex64](dst, src, twiddle, scratch)
}

// The four-step algorithm is the rectangular generalization of six-step: the
// length n is decomposed as n1×n2 (any power-of-two split, not just √n×√n),
// so it also covers the non-square powers of two six-step declines, and the
// split can be tilted so a row pass fits the detected cache sizes.
//
// Layout walk-through (row-major matrices):
//  1. transpose src (n1×n2) → dst (n2×n1)
//  2. FFT the n2 rows of length n1 (the column DFTs of the original matrix)
//  3. multiply dst[j][i] by W_n^(i·j) (conjugated for the inverse)
//  4. transpose dst (n2×n1) → scratch (n1×n2)
//  5. FFT the n1 rows of length n2
//  6. transpose scratch (n1×n2) → dst (n2×n1): natural-order output
//
// Each transpose is out-of-place into the buffer whose contents are stale,
// so no extra memory beyond the usual length-n scratch is needed; the
// per-pass row twiddle/scratch buffers are carved from the stale buffer too.

func fourStepForward[T Complex](dst, src, twiddle, scratch []T) bool {
	return fourStepDispatch(dst, src, twiddle, scratch, false)
}

func fourStepInverse[T Complex](dst, src, twiddle, scratch []T) bool {
	return fourStepDispatch(dst, src, twiddle, scratch, true)
}

func fourStepDispatch[T Complex](dst, src, twiddle, scratch []T, inverse bool) bool {
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

	if n < 4 || !math.IsPowerOf2(n) {
		return false
	}

	n1, _ := fourStepSplit(n, complexElemSize[T](), cpu.DetectCaches())

	return fourStepTransform(dst, src, twiddle, scratch, n1, inverse)
}

// fourStepTransform runs the four-step transform with an explicit n1×(n/n1)
// split. Exposed separately from the dispatch so tests and benchmarks can
// sweep split choices. n must be a power of two with 2 <= n1 <= n/2.
func fourStepTransform[T Complex](dst, src, twiddle, scratch []T, n1 int, inverse bool) bool {
	n := len(src)

	n2 := n / n1
	if n1 < 2 || n2 < 2 || n1*n2 != n {
		return false
	}

	// Step 1: transpose the n1×n2 input into dst as n2×n1. When dst aliases
	// src, stage the input through scratch first; the copy is consumed by the
	// transpose, so scratch is stale again afterwards.
	if sameSlice(dst, src) {
		copy(scratch, src[:n])
		math.TransposeRect(dst, scratch[:n], n1, n2)
	} else {
		math.TransposeRect(dst, src, n1, n2)
	}

	// Step 2: FFT the n2 rows of length n1. Row twiddle/scratch buffers live
	// in scratch, whose contents are stale during this pass.
	rowTwiddle := scratch[:n1]
	rowScratch := scratch[n1 : 2*n1]
	fillRowTwiddle(rowTwiddle, twiddle, n2)

	for r := range n2 {
		row := dst[r*n1 : (r+1)*n1]
		if !rowStockham(row, rowTwiddle, rowScratch, inverse) {
			return false
		}
	}

	// Step 3: twiddle multiply dst[j][i] by W_n^(i·j). The index walks in
	// steps of j with a subtract-wrap, avoiding a modulo per element.
	for j := range n2 {
		row := dst[j*n1 : (j+1)*n1]
		idx := 0

		for i := range row {
			w := twiddle[idx]
			if inverse {
				w = conj(w)
			}

			row[i] *= w

			idx += j
			if idx >= n {
				idx -= n
			}
		}
	}

	// Step 4: transpose dst (n2×n1) → scratch (n1×n2).
	math.TransposeRect(scratch, dst[:n], n2, n1)

	// Step 5: FFT the n1 rows of length n2, buffers carved from the now-stale
	// dst.
	rowTwiddle = dst[:n2]
	rowScratch = dst[n2 : 2*n2]
	fillRowTwiddle(rowTwiddle, twiddle, n1)

	for r := range n1 {
		row := scratch[r*n2 : (r+1)*n2]
		if !rowStockham(row, rowTwiddle, rowScratch, inverse) {
			return false
		}
	}

	// Step 6: transpose scratch (n1×n2) → dst (n2×n1) for natural order.
	math.TransposeRect(dst, scratch[:n], n1, n2)

	return true
}

// rowStockham runs one in-place row FFT in the requested direction.
func rowStockham[T Complex](row, rowTwiddle, rowScratch []T, inverse bool) bool {
	if inverse {
		return stockhamInverse(row, row, rowTwiddle, rowScratch)
	}

	return stockhamForward(row, row, rowTwiddle, rowScratch)
}

// Row-pass working-set penalties used by the fourStepSplit cost model. A row
// FFT of length m streams the row, a same-size scratch, and the row twiddle
// table (~3·m·elemSize bytes); passes that spill L1 or L2 pay progressively
// for the extra memory traffic. The exact weights only steer the split
// choice; measure-mode planning (Wisdom) remains the final arbiter between
// strategies.
const (
	fourStepRowBuffers = 3

	fourStepL2Penalty   = 1.4
	fourStepSpillFactor = 3.0
)

// fourStepSplit chooses the n = n1×n2 decomposition for the four-step
// transform from the detected cache sizes. It minimizes the modeled cost of
// the two row passes, n2·cost(n1) + n1·cost(n2), where a row pass whose
// working set exceeds L1d (or L2) is penalized. Ties prefer the balanced
// split (four-step then degenerates to six-step's √n×√n), then n1 <= n2.
func fourStepSplit(n, elemSize int, caches cpu.CacheInfo) (int, int) {
	bestN1 := 0
	bestCost := 0.0
	bestImbalance := 0

	for n1 := 2; n1 <= n/2; n1 *= 2 {
		n2 := n / n1

		cost := float64(n2)*fourStepRowCost(n1, elemSize, caches) +
			float64(n1)*fourStepRowCost(n2, elemSize, caches)

		imbalance := log2(n2) - log2(n1)
		if imbalance < 0 {
			imbalance = -imbalance
		}

		better := bestN1 == 0 || cost < bestCost ||
			(cost == bestCost && imbalance < bestImbalance)
		if better {
			bestN1, bestCost, bestImbalance = n1, cost, imbalance
		}
	}

	return bestN1, n / max(bestN1, 1)
}

// fourStepRowCost models the cost of one row FFT of length m: m·log2(m)
// butterfly work scaled by the cache-residency penalty of its working set.
func fourStepRowCost(m, elemSize int, caches cpu.CacheInfo) float64 {
	work := float64(m) * float64(log2(m))

	working := fourStepRowBuffers * m * elemSize
	switch {
	case working <= caches.L1DataBytes:
		return work
	case working <= caches.L2Bytes:
		return work * fourStepL2Penalty
	default:
		return work * fourStepSpillFactor
	}
}

// complexElemSize returns the byte size of the complex element type.
func complexElemSize[T Complex]() int {
	var zero T
	if _, ok := any(zero).(complex64); ok {
		return 8
	}

	return 16
}
