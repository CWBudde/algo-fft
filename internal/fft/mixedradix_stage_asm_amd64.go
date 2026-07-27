//go:build amd64 && !purego

package fft

import (
	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/kernels"
	m "github.com/cwbudde/algo-fft/internal/math"
)

// Fused mixed-radix stage kernels.
//
// mixedRadixStageComplex64/128 run a stage as two passes: one array multiply
// that applies the stage's twiddles in place, then a twiddle-free butterfly
// loop. Both passes are contiguous, but the data crosses memory twice and the
// r values of one butterfly are re-loaded after the multiply wrote them.
//
// The assembly kernels below do the whole stage in one pass. Each YMM lane is
// a different k running the same butterfly, so the butterfly needs no
// cross-lane movement; the twiddled inputs stay in registers between the
// multiply and the butterfly, and rows 1..r-1 are never written back. The
// twiddle multiply and the butterfly are also independent chains for adjacent
// k, which is what lets one loop body issue both.
//
// Only radices 3 and 5 have kernels. They cover the stages that matter for the
// practical DSP lengths: 1000 = [5 5 5 8], 3600 = [5 5 3 3 16],
// 12000 = [5 5 5 3 32] and 44100 = [5 5 7 7 4 3 3] are radix-3/5 at every
// stage the vectorised path is allowed to take. Radix 2/4/8/11 stages fall
// through to the two-pass Go stage, which is unchanged.

// mixedRadixStageAsmMinSpan is the smallest span for which a fused kernel is
// dispatched. Below it the kernel's vector loop has no whole block to run and
// the Go tail would do the entire stage — strictly worse than the two-pass
// path, which at least vectorises its multiply. The gate in
// mixedRadixStageVectorizable already implies a larger span than this for
// every radix here, so it is a floor, not a tuning knob.
const mixedRadixStageAsmMinSpan = 4

// mixedRadixStageAsm64 runs one whole stage with a fused assembly kernel and
// reports whether it did. A false result means the caller must run the
// two-pass Go stage; nothing has been written to dst in that case.
//
// The length checks are not redundant with the caller's: the kernels index
// input, table and dst up to n-1 without bounds checks of their own.
func mixedRadixStageAsm64(dst, input, table []complex64, n, span, radix int, inverse bool) bool {
	if span < mixedRadixStageAsmMinSpan || len(dst) < n || len(input) < n || len(table) < n {
		return false
	}

	if !cpu.DetectFeatures().HasAVX2 {
		return false
	}

	switch radix {
	case 3:
		amd64.MixedRadixStage3Complex64AVX2Asm(dst, input, table, span, inverse)
	case 5:
		amd64.MixedRadixStage5Complex64AVX2Asm(dst, input, table, span, inverse)
	default:
		return false
	}

	// The kernels process whole 4-element blocks of k only.
	mixedRadixStageTail64(dst, input, table, span, radix, inverse, span&^3)

	return true
}

// mixedRadixStageTail64 runs the fused stage in Go for k in [from, span), the
// values the vector loop could not cover. It multiplies through
// m.MulComplex64 rather than the * operator because Go widens a scalar
// complex64 product to complex128 and back.
func mixedRadixStageTail64(dst, input, table []complex64, span, radix int, inverse bool, from int) {
	for k := from; k < span; k++ {
		switch radix {
		case 3:
			a0 := input[k]
			a1 := m.MulComplex64(input[span+k], table[span+k])
			a2 := m.MulComplex64(input[2*span+k], table[2*span+k])

			var y0, y1, y2 complex64
			if inverse {
				y0, y1, y2 = kernels.Butterfly3InverseComplex64(a0, a1, a2)
			} else {
				y0, y1, y2 = kernels.Butterfly3ForwardComplex64(a0, a1, a2)
			}

			dst[k], dst[span+k], dst[2*span+k] = y0, y1, y2
		case 5:
			a0 := input[k]
			a1 := m.MulComplex64(input[span+k], table[span+k])
			a2 := m.MulComplex64(input[2*span+k], table[2*span+k])
			a3 := m.MulComplex64(input[3*span+k], table[3*span+k])
			a4 := m.MulComplex64(input[4*span+k], table[4*span+k])

			var y0, y1, y2, y3, y4 complex64
			if inverse {
				y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex64(a0, a1, a2, a3, a4)
			} else {
				y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex64(a0, a1, a2, a3, a4)
			}

			dst[k], dst[span+k], dst[2*span+k] = y0, y1, y2
			dst[3*span+k], dst[4*span+k] = y3, y4
		}
	}
}

// mixedRadixStageAsm128 is the complex128 counterpart of mixedRadixStageAsm64.
func mixedRadixStageAsm128(dst, input, table []complex128, n, span, radix int, inverse bool) bool {
	if span < mixedRadixStageAsmMinSpan || len(dst) < n || len(input) < n || len(table) < n {
		return false
	}

	if !cpu.DetectFeatures().HasAVX2 {
		return false
	}

	switch radix {
	case 3:
		amd64.MixedRadixStage3Complex128AVX2Asm(dst, input, table, span, inverse)
	case 5:
		amd64.MixedRadixStage5Complex128AVX2Asm(dst, input, table, span, inverse)
	default:
		return false
	}

	// The complex128 kernels process whole 2-element blocks of k.
	mixedRadixStageTail128(dst, input, table, span, radix, inverse, span&^1)

	return true
}

// mixedRadixStageTail128 is the complex128 counterpart of
// mixedRadixStageTail64. complex128 products need no helper: Go computes them
// at their own width.
func mixedRadixStageTail128(dst, input, table []complex128, span, radix int, inverse bool, from int) {
	for k := from; k < span; k++ {
		switch radix {
		case 3:
			a0 := input[k]
			a1 := input[span+k] * table[span+k]
			a2 := input[2*span+k] * table[2*span+k]

			var y0, y1, y2 complex128
			if inverse {
				y0, y1, y2 = kernels.Butterfly3InverseComplex128(a0, a1, a2)
			} else {
				y0, y1, y2 = kernels.Butterfly3ForwardComplex128(a0, a1, a2)
			}

			dst[k], dst[span+k], dst[2*span+k] = y0, y1, y2
		case 5:
			a0 := input[k]
			a1 := input[span+k] * table[span+k]
			a2 := input[2*span+k] * table[2*span+k]
			a3 := input[3*span+k] * table[3*span+k]
			a4 := input[4*span+k] * table[4*span+k]

			var y0, y1, y2, y3, y4 complex128
			if inverse {
				y0, y1, y2, y3, y4 = kernels.Butterfly5InverseComplex128(a0, a1, a2, a3, a4)
			} else {
				y0, y1, y2, y3, y4 = kernels.Butterfly5ForwardComplex128(a0, a1, a2, a3, a4)
			}

			dst[k], dst[span+k], dst[2*span+k] = y0, y1, y2
			dst[3*span+k], dst[4*span+k] = y3, y4
		}
	}
}
