//go:build amd64 && !purego

package kernels

import (
	"math"
	"math/bits"
	"sync"

	"github.com/cwbudde/algo-fft/internal/asm/amd64"
	m "github.com/cwbudde/algo-fft/internal/math"
)

// Radix-4 DIT, 256-bit wide.
//
// The older per-size radix-4 codelets (avx2_f32_size*_radix4*.s) are VEX-encoded
// but XMM-width: every butterfly loads a single complex64 with VMOVSD and does
// its arithmetic in X registers, so they move 64 of a possible 256 bits per
// instruction. The kernel here processes four butterflies per instruction in Y
// registers instead, and is size-generic rather than one hand-rolled file per n.
//
// Layout, writing n = 4^k or n = 2*4^k:
//
//	Stage 1: n/4 groups x 1 butterfly, no twiddles.
//	Stage s: n/(4m) groups x m butterflies, m = 4^(s-1), while 4m <= r4End.
//	Tail:    for n = 2*4^k only, one radix-2 stage over n/2 butterflies.
//
// where r4End is n for a power of four and n/2 otherwise. In the latter case the
// radix-4 stages never span more than n/2, so they transform the even- and
// odd-indexed halves independently and the radix-2 tail combines them -- the
// classic radix-2 split with radix-4 doing the halves.
//
// Within a group the m butterflies are contiguous, so stages 2..k vectorise
// four butterflies at a time with plain VMOVUPS loads. Stage 1 has one
// butterfly per group and its four inputs are adjacent, so it instead loads
// four groups (16 complex64) and transposes them into a0..a3 vectors.
//
// The twiddle for butterfly j of a stage depends only on j, not on the group, so
// each radix-4 stage needs 3*m twiddles laid out as three contiguous planes
// (w1[0..m-1], w2[0..m-1], w3[0..m-1]) and the tail needs n/2 more. That makes
// every twiddle load a contiguous 256-bit read and removes the per-butterfly
// index arithmetic the old kernels did. Either way the total is n-4 elements:
// sum(3*4^(s-1)) for s = 2..k is 4^k-4.

const (
	// radix4AVX2MinSize is the smallest n the kernel handles. Stage 1 consumes
	// four groups (16 elements) per iteration, so n must be at least 16.
	radix4AVX2MinSize = 16
	// radix4AVX2MaxSize bounds the cached permutation tables.
	radix4AVX2MaxSize = 1 << 16
)

// radix4AVX2Limit reports the largest span the radix-4 stages may cover, and
// whether n has a shape the kernel supports at all. The limit is n when n is a
// power of four and n/2 when n is twice one; the latter then needs a radix-2
// tail.
func radix4AVX2Limit(n int) (limit int, ok bool) {
	if n < radix4AVX2MinSize || n > radix4AVX2MaxSize || n&(n-1) != 0 {
		return 0, false
	}

	// A power of two is a power of four iff its set bit sits at an even
	// position.
	if bits.TrailingZeros(uint(n))%2 == 0 {
		return n, true
	}

	return n / 2, true
}

// radix4AVX2SizeOK reports whether the kernel can handle length n.
func radix4AVX2SizeOK(n int) bool {
	_, ok := radix4AVX2Limit(n)

	return ok
}

// twiddleSizeRadix4AVX2 returns the element count of the packed twiddle table.
//
// Only the first n-4 elements carry data. The request is padded to n+4 so that
// the table is strictly longer than the standard length-n DIT twiddle table:
// callers that hand a prepared-twiddle codelet the plain table are expected to
// be caught by the kernel's own length check (see internal/fft/measure.go),
// and at n-4 the plain table would pass that check and be silently transformed
// against the wrong factors.
func twiddleSizeRadix4AVX2(n int) int {
	if !radix4AVX2SizeOK(n) {
		return 0
	}

	return n + 4
}

// prepareTwiddleRadix4AVX2 fills dst with the per-stage twiddle planes described
// above. For the inverse transform the imaginary parts are negated, which is the
// conjugate W^-k = conj(W^k).
func prepareTwiddleRadix4AVX2(n int, inverse bool, dst []complex64) {
	limit, ok := radix4AVX2Limit(n)
	if !ok || len(dst) < twiddleSizeRadix4AVX2(n) {
		return
	}

	clear(dst[:twiddleSizeRadix4AVX2(n)])

	sign := -1.0
	if inverse {
		sign = 1.0
	}

	// w(e) = exp(sign * 2*pi*i*e/n), matching math.ComputeTwiddleFactors but
	// evaluated directly so no n-element intermediate table is needed.
	w := func(e int) complex64 {
		sin, cos := math.Sincos(sign * 2 * math.Pi * float64(e%n) / float64(n))

		return complex(float32(cos), float32(sin))
	}

	offset := 0

	for stage := 4; stage*4 <= limit; stage *= 4 {
		step := n / (4 * stage)

		for mul := 1; mul <= 3; mul++ {
			for j := range stage {
				dst[offset+j] = w(mul * j * step)
			}

			offset += stage
		}
	}

	// Radix-2 tail: one twiddle per butterfly, W_n^j for j = 0..n/2-1.
	if limit != n {
		for j := range n / 2 {
			dst[offset+j] = w(j)
		}
	}
}

// radix4GroupIndexTable memoises one stage-1 group index table.
type radix4GroupIndexTable struct {
	once sync.Once
	idx  []int32
}

//nolint:gochecknoglobals // memoised permutation tables, one slot per log2(n)
var radix4GroupIndexTables [17]radix4GroupIndexTable

// radix4GroupIndices returns the memoised stage-1 group index table for n:
// entry g is the source index of the first input of group g.
//
// The full digit-reversal permutation p satisfies p[4g+d] = p[4g] + d*(n/4),
// because d is the least significant base-4 digit of the index and therefore
// becomes the most significant digit after reversal. Storing only p[4g] shrinks
// the table by 4x, and int32 halves it again versus the int64 DATA blobs the old
// kernels embedded in the binary: 16 KiB instead of 128 KiB at n = 16384.
//
// The permutation itself comes from internal/math rather than being rederived
// here, so it cannot drift from the one the rest of the library uses.
func radix4GroupIndices(n int) []int32 {
	slot := &radix4GroupIndexTables[bits.TrailingZeros(uint(n))]
	slot.once.Do(func() {
		limit, ok := radix4AVX2Limit(n)
		if !ok {
			return
		}

		var full []int
		if limit == n {
			full = m.ComputeBitReversalIndicesRadix4(n)
		} else {
			full = m.ComputeBitReversalIndicesRadix4Then2(n)
		}

		idx := make([]int32, n/4)
		for g := range idx {
			idx[g] = int32(full[4*g])
		}

		slot.idx = idx
	})

	return slot.idx
}

// forwardRadix4AVX2Complex64 is the CodeletFunc entry point for the forward
// transform. twiddle must be the table produced by prepareTwiddleRadix4AVX2.
func forwardRadix4AVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	return amd64.Radix4Complex64Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), limit, false, false, 1)
}

// inverseRadix4AVX2Complex64 is the CodeletFunc entry point for the inverse
// transform. The 1/n normalisation is exact here (n is a power of two) and is
// folded into stage 1 rather than costing a separate pass over the data.
func inverseRadix4AVX2Complex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	scale := float32(1) / float32(n)

	return amd64.Radix4Complex64Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), limit, true, false, scale)
}

// The fused-tail variants.
//
// For n = 2*4^k the kernel runs a separate radix-2 pass over the whole buffer
// after the radix-4 stages. Measured against a probe that skips that pass
// outright, it costs 9-15% of the kernel at every such size -- so folding it
// into the last radix-4 stage, where both of its operands are already in
// registers, looked like a uniform win. It is not, and the reason is worth
// keeping: fusing doubles the number of live streams from four to eight, and
// past a point that costs more than the pass it saves.
//
// Canary-gated sweep on the i7-1255U (AVX2), fused as a ratio to the separate
// tail, forward/inverse, 7-10 accepted groups per cell:
//
//	n      stride   complex64      complex128
//	128    128/256B 0.955 / 0.979  0.935 / 0.934
//	512    512B/1K  0.971 / 1.005  1.002 / 1.020
//	2048   2K/4K    0.943 / 0.974  1.110 / 1.104
//	8192   8K/16K   1.034 / 1.021  1.006 / 1.077
//	32768  32K/64K  1.004 / 1.013  1.020 / 1.000
//
// (stride is the last stage's m*sizeof(T), the distance between the eight
// streams.) It is therefore selected per size from this table rather than
// applied wherever the shape allows: see the Priority rows in
// cmd/gencodelets/specs.go, which is where every other per-size measured fact
// in this library lives. Sizes not listed there keep the separate tail.
//
// The L1-set reading of that table -- worst at complex128 n = 2048 because the
// stride is exactly 4 KiB and every stream lands on one set -- was tested on a
// second machine on 2026-08-01 and does not survive. A Xeon Gold 5218 has 8-way
// L1d against this laptop's 12-way with the same 4 KiB aliasing, so conflict
// misses should get worse there. complex64 does get worse (+0.07 to +0.10 at
// 2048/8192/32768); complex128, which has twice the byte stride and identical
// set-aliasing, gets *better* (-0.13 at 512, -0.08 at 2048). A way-count
// mechanism cannot produce that sign split.
//
// What survives is the register budget described in the r4_fused_loop header:
// Y0..Y7 are pinned across group 1's whole computation, leaving six scratch
// registers and forcing group 1 to re-load its twiddle broadcasts every
// iteration. Do not re-derive the cache story from this table.
//
// Fusion is closed as a way to recover the tail. Across ten cells on two hosts
// it never captures more than 2.4%, while the dit<N>_radix4_notail_avx2 probe
// puts the tail's cost at 6.7-13.3% here and 7.7-23.4% on the Xeon. See
// docs/CODELET_BENCHMARKS.md, "The radix-4 tail on the Xeon".
//
// Re-measure with `just bench-gated <sizes>` after building the candidate
// sweep with -tags fftprobe, which registers both variants side by side.
func forwardRadix4AVX2FusedComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	return amd64.Radix4Complex64Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), limit, false, true, 1)
}

// inverseRadix4AVX2FusedComplex64 is the inverse twin of
// forwardRadix4AVX2FusedComplex64.
func inverseRadix4AVX2FusedComplex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	limit, ok := radix4AVX2Limit(n)
	if !ok {
		return false
	}

	scale := float32(1) / float32(n)

	return amd64.Radix4Complex64Asm(dst, src, twiddle, scratch, radix4GroupIndices(n), limit, true, true, scale)
}
