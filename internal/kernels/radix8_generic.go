package kernels

import (
	"math"
	"math/bits"
	"sync"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// Radix-8 DIT, size-generic, pure Go.
//
// Why radix-8 at all: a radix-8 ladder makes log2(n)/3 passes over the buffer
// where radix-4 makes log2(n)/2 -- 20-40% fewer, best at n = 512 (3 against 5)
// and n = 32768 (5 against 8). Real operation counts barely move (4.08 against
// 4.25 N*log2(N)), so the whole case rests on passes. That is the same case
// that made the 256-bit radix-4 kernels win, one radix up.
//
// Every earlier radix-8 kernel in this tree measured slower than radix-4 while
// making fewer passes, which is the signature of the butterfly rather than the
// ladder. The per-size ones threw the advantage away in two different places:
// the AVX2 size-512 kernel is XMM-width throughout despite a header promising
// Y registers, and the pure-Go size-512 codelet spends a full complex multiply
// on W_8^2 = -i (a free swap-and-negate) and on W_8^{1,3} (half that), and
// computes each of those products twice per butterfly. This kernel uses the
// specialised rotations and is size-generic rather than one file per n.
//
// Layout, writing n = 8^k, 2*8^k or 4*8^k:
//
//	Stage 1: n/8 groups x 1 butterfly, no twiddles, fused with the gather.
//	Stage s: n/(8m) groups x m butterflies, m = 8^(s-1), while 8m <= limit.
//	Tail:    for n != 8^k only, one radix-2 or radix-4 stage over the whole
//	         buffer.
//
// where limit is n for a power of eight, n/2 for twice one and n/4 for four
// times one. In the latter two cases the radix-8 stages never span more than
// limit, so they transform the tail-many interleaved sub-sequences
// independently and the tail stage combines them. Those three shapes cover
// every power of two, so no size is out of reach.
//
// The twiddle for butterfly j of a stage depends only on j, not on the group,
// so each radix-8 stage needs 7*m twiddles laid out as seven contiguous planes
// (w1[0..m-1] ... w7[0..m-1]); the radix-2 tail needs n/2 more and the radix-4
// tail 3*(n/4). Either way the total is n-8 elements, exactly as the radix-4
// kernel's is n-4: sum(7*8^(s-1)) for s = 2..k is 8^k-8.
//
// Every stage is in place. Only the stage-1 gather writes somewhere new, so the
// kernel touches one buffer rather than the one-array-per-stage style the
// per-size codelets use -- which is what makes the pass count honest.

const (
	// radix8MinSize is the smallest n the kernel handles: one stage-1 group.
	radix8MinSize = 8
	// radix8MaxSize bounds the cached permutation tables.
	radix8MaxSize = 1 << 16
)

// radix8Limit reports the largest span the radix-8 stages may cover, the radix
// of the single tail stage that finishes the transform (1 meaning none), and
// whether n has a shape the kernel supports at all.
func radix8Limit(n int) (limit, tail int, ok bool) {
	if n < radix8MinSize || n > radix8MaxSize || n&(n-1) != 0 {
		return 0, 0, false
	}

	// n = 2^t is 8^k, 2*8^k or 4*8^k according to t mod 3.
	switch bits.TrailingZeros(uint(n)) % 3 {
	case 0:
		return n, 1, true
	case 1:
		return n / 2, 2, true
	default:
		return n / 4, 4, true
	}
}

// radix8SizeOK reports whether the kernel can handle length n.
func radix8SizeOK(n int) bool {
	_, _, ok := radix8Limit(n)

	return ok
}

// twiddleSizeRadix8 returns the element count of the packed twiddle table.
//
// Only the first n-8 elements carry data. The request is padded to n+8 so that
// the table is strictly longer than the standard length-n DIT twiddle table:
// at n-8 a caller that handed this kernel the plain table would pass the
// length check and be silently transformed against the wrong factors.
func twiddleSizeRadix8(n int) int {
	if !radix8SizeOK(n) {
		return 0
	}

	return n + 8
}

// prepareTwiddleRadix8Complex64 fills dst with the per-stage twiddle planes
// described above. For the inverse transform the imaginary parts are negated,
// which is the conjugate W^-k = conj(W^k), so the stage code is direction-blind
// apart from its butterfly.
func prepareTwiddleRadix8Complex64(n int, inverse bool, dst []complex64) {
	limit, tail, ok := radix8Limit(n)
	if !ok || len(dst) < twiddleSizeRadix8(n) {
		return
	}

	clear(dst[:twiddleSizeRadix8(n)])

	sign := -1.0
	if inverse {
		sign = 1.0
	}

	// w(e) = exp(sign * 2*pi*i*e/n), matching math.ComputeTwiddleFactors but
	// evaluated directly so no n-element intermediate table is needed.
	w := func(e int) complex64 {
		sin, cos := math.Sincos(sign * 2 * math.Pi * float64(e%n) / float64(n))

		return mathpkg.ComplexFromFloat64[complex64](cos, sin)
	}

	offset := 0

	// Radix-8 stages: butterfly j of block d is scaled by W_(8m)^(jd).
	for m := 8; m*8 <= limit; m *= 8 {
		step := n / (8 * m)

		for d := 1; d <= 7; d++ {
			for j := range m {
				dst[offset+j] = w(d * j * step)
			}

			offset += m
		}
	}

	// Tail stage: span n, so its step is 1.
	for d := 1; d < tail; d++ {
		m := n / tail

		for j := range m {
			dst[offset+j] = w(d * j)
		}

		offset += m
	}
}

// radix8GroupIndexTable memoises one stage-1 group index table.
type radix8GroupIndexTable struct {
	once sync.Once
	idx  []int32
}

//nolint:gochecknoglobals // memoised permutation tables, one slot per log2(n)
var radix8GroupIndexTables [17]radix8GroupIndexTable

// radix8GroupIndices returns the memoised stage-1 group index table for n:
// entry g is the source index of the first input of group g.
//
// The digit-reversal permutation p satisfies p[8g+d] = p[8g] + d*(n/8) for all
// three supported shapes, because d is the least significant base-8 digit of
// the index and therefore becomes the most significant digit after reversal --
// and in the split shapes the leading radix-2/radix-4 block only shifts the
// whole run by a constant. Storing only p[8g] shrinks the table by 8x, and
// int32 halves it again against an int64 table.
//
// The permutation itself comes from internal/math rather than being rederived
// here, so it cannot drift from the one the rest of the library uses.
func radix8GroupIndices(n int) []int32 {
	if !radix8SizeOK(n) {
		return nil
	}

	slot := &radix8GroupIndexTables[bits.TrailingZeros(uint(n))]
	slot.once.Do(func() {
		_, tail, ok := radix8Limit(n)
		if !ok {
			return
		}

		var full []int

		switch tail {
		case 1:
			full = mathpkg.ComputeBitReversalIndicesRadix8(n)
		case 2:
			full = mathpkg.ComputeBitReversalIndicesRadix8Then2(n)
		default:
			full = mathpkg.ComputeBitReversalIndicesRadix8Then4(n)
		}

		if len(full) != n {
			return
		}

		idx := make([]int32, n/8)
		for g := range idx {
			idx[g] = int32(full[8*g])
		}

		slot.idx = idx
	})

	return slot.idx
}

// radix8PrologueComplex64 validates the arguments common to both directions and returns
// the working buffer, the stage-1 group table and the ladder shape.
func radix8PrologueComplex64(dst, src, twiddle, scratch []complex64) (
	work []complex64, groups []int32, limit, tail int, ok bool,
) {
	n := len(src)

	limit, tail, ok = radix8Limit(n)
	if !ok || len(dst) < n || len(scratch) < n || len(twiddle) < twiddleSizeRadix8(n) {
		return nil, nil, 0, 0, false
	}

	groups = radix8GroupIndices(n)
	if len(groups) != n/8 {
		return nil, nil, 0, 0, false
	}

	work = dst[:n]
	if &dst[0] == &src[0] {
		// Stage 1 gathers, so it cannot write over its own source.
		work = scratch[:n]
	}

	return work, groups, limit, tail, true
}

// The eight-point butterflies below are butterfly8ForwardComplex64 and
// butterfly8InverseComplex64 from radix8.go, unrolled into the stage loops.
// Those functions are the readable statement of the arithmetic and are what
// the mixed-radix engine calls, but they are well over the inliner's budget,
// and a call per butterfly with eight arguments and eight results costs more
// than the stage it serves. TestRadix8LadderMatchesButterfly pins the two
// against each other so the copy cannot drift.

// forwardRadix8Complex64 is the CodeletFunc entry point for the forward
// transform. twiddle must be the table produced by prepareTwiddleRadix8Complex64.
func forwardRadix8Complex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	work, groups, limit, tail, ok := radix8PrologueComplex64(dst, src, twiddle, scratch)
	if !ok {
		return false
	}

	s := src[:n]
	stride := n / 8

	// Stage 1: one twiddle-free butterfly per group, fused with the gather.
	for g, first := range groups {
		p := int(first)

		x0 := s[p]
		x1 := s[p+stride]
		x2 := s[p+2*stride]
		x3 := s[p+3*stride]
		x4 := s[p+4*stride]
		x5 := s[p+5*stride]
		x6 := s[p+6*stride]
		x7 := s[p+7*stride]

		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(imag(a3), -real(a3)) // a1 - i*a3
		e3 := a1 + complex(-imag(a3), real(a3)) // a1 + i*a3
		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(imag(a7), -real(a7)) // a5 - i*a7
		o3 := a5 + complex(-imag(a7), real(a7)) // a5 + i*a7

		t1 := complex(root2Over2*(real(o1)+imag(o1)), root2Over2*(imag(o1)-real(o1)))
		t2 := complex(imag(o2), -real(o2))
		t3 := complex(root2Over2*(imag(o3)-real(o3)), -root2Over2*(real(o3)+imag(o3)))

		out := work[8*g : 8*g+8 : 8*g+8]
		out[0] = e0 + o0
		out[1] = e1 + t1
		out[2] = e2 + t2
		out[3] = e3 + t3
		out[4] = e0 - o0
		out[5] = e1 - t1
		out[6] = e2 - t2
		out[7] = e3 - t3
	}

	offset := radix8StagesForwardComplex64(work, twiddle, n, limit)
	radix8TailForwardComplex64(work, twiddle, n, tail, offset)

	if &work[0] != &dst[0] {
		copy(dst[:n], work)
	}

	return true
}

// radix8StagesForwardComplex64 runs the twiddled radix-8 stages in place and
// returns the number of twiddle elements consumed.
func radix8StagesForwardComplex64(work, twiddle []complex64, n, limit int) int {
	offset := 0

	for m := 8; m*8 <= limit; m *= 8 {
		span := 8 * m

		w1 := twiddle[offset : offset+m]
		w2 := twiddle[offset+m : offset+2*m]
		w3 := twiddle[offset+2*m : offset+3*m]
		w4 := twiddle[offset+3*m : offset+4*m]
		w5 := twiddle[offset+4*m : offset+5*m]
		w6 := twiddle[offset+5*m : offset+6*m]
		w7 := twiddle[offset+6*m : offset+7*m]
		offset += 7 * m

		for base := 0; base < n; base += span {
			blk := work[base : base+span : base+span]

			for j := range m {
				x0 := blk[j]
				x1 := mathpkg.MulComplex64(w1[j], blk[j+m])
				x2 := mathpkg.MulComplex64(w2[j], blk[j+2*m])
				x3 := mathpkg.MulComplex64(w3[j], blk[j+3*m])
				x4 := mathpkg.MulComplex64(w4[j], blk[j+4*m])
				x5 := mathpkg.MulComplex64(w5[j], blk[j+5*m])
				x6 := mathpkg.MulComplex64(w6[j], blk[j+6*m])
				x7 := mathpkg.MulComplex64(w7[j], blk[j+7*m])

				a0 := x0 + x4
				a1 := x0 - x4
				a2 := x2 + x6
				a3 := x2 - x6
				a4 := x1 + x5
				a5 := x1 - x5
				a6 := x3 + x7
				a7 := x3 - x7

				e0 := a0 + a2
				e2 := a0 - a2
				e1 := a1 + complex(imag(a3), -real(a3))
				e3 := a1 + complex(-imag(a3), real(a3))
				o0 := a4 + a6
				o2 := a4 - a6
				o1 := a5 + complex(imag(a7), -real(a7))
				o3 := a5 + complex(-imag(a7), real(a7))

				t1 := complex(root2Over2*(real(o1)+imag(o1)), root2Over2*(imag(o1)-real(o1)))
				t2 := complex(imag(o2), -real(o2))
				t3 := complex(root2Over2*(imag(o3)-real(o3)), -root2Over2*(real(o3)+imag(o3)))

				blk[j] = e0 + o0
				blk[j+m] = e1 + t1
				blk[j+2*m] = e2 + t2
				blk[j+3*m] = e3 + t3
				blk[j+4*m] = e0 - o0
				blk[j+5*m] = e1 - t1
				blk[j+6*m] = e2 - t2
				blk[j+7*m] = e3 - t3
			}
		}
	}

	return offset
}

// radix8TailForwardComplex64 runs the single widest stage for the shapes that
// need one: radix-2 for n = 2*8^k, radix-4 for n = 4*8^k, nothing otherwise.
func radix8TailForwardComplex64(work, twiddle []complex64, n, tail, offset int) {
	switch tail {
	case 2:
		m := n / 2
		w1 := twiddle[offset : offset+m]

		for j := range m {
			a := work[j]
			b := mathpkg.MulComplex64(w1[j], work[j+m])
			work[j] = a + b
			work[j+m] = a - b
		}

	case 4:
		m := n / 4
		w1 := twiddle[offset : offset+m]
		w2 := twiddle[offset+m : offset+2*m]
		w3 := twiddle[offset+2*m : offset+3*m]

		for j := range m {
			x0 := work[j]
			x1 := mathpkg.MulComplex64(w1[j], work[j+m])
			x2 := mathpkg.MulComplex64(w2[j], work[j+2*m])
			x3 := mathpkg.MulComplex64(w3[j], work[j+3*m])

			t0 := x0 + x2
			t1 := x0 - x2
			t2 := x1 + x3
			t3 := x1 - x3

			work[j] = t0 + t2
			work[j+m] = t1 + complex(imag(t3), -real(t3)) // t1 - i*t3
			work[j+2*m] = t0 - t2
			work[j+3*m] = t1 + complex(-imag(t3), real(t3)) // t1 + i*t3
		}
	}
}

// inverseRadix8Complex64 is the CodeletFunc entry point for the inverse
// transform. The 1/n normalisation is exact here (n is a power of two) and is
// folded into the stage-1 gather rather than costing a separate pass over the
// data; by linearity the result is identical either way.
func inverseRadix8Complex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	work, groups, limit, tail, ok := radix8PrologueComplex64(dst, src, twiddle, scratch)
	if !ok {
		return false
	}

	s := src[:n]
	stride := n / 8
	scale := float32(1) / float32(n)

	for g, first := range groups {
		p := int(first)

		v0 := s[p]
		v1 := s[p+stride]
		v2 := s[p+2*stride]
		v3 := s[p+3*stride]
		v4 := s[p+4*stride]
		v5 := s[p+5*stride]
		v6 := s[p+6*stride]
		v7 := s[p+7*stride]

		x0 := complex(real(v0)*scale, imag(v0)*scale)
		x1 := complex(real(v1)*scale, imag(v1)*scale)
		x2 := complex(real(v2)*scale, imag(v2)*scale)
		x3 := complex(real(v3)*scale, imag(v3)*scale)
		x4 := complex(real(v4)*scale, imag(v4)*scale)
		x5 := complex(real(v5)*scale, imag(v5)*scale)
		x6 := complex(real(v6)*scale, imag(v6)*scale)
		x7 := complex(real(v7)*scale, imag(v7)*scale)

		a0 := x0 + x4
		a1 := x0 - x4
		a2 := x2 + x6
		a3 := x2 - x6
		a4 := x1 + x5
		a5 := x1 - x5
		a6 := x3 + x7
		a7 := x3 - x7

		e0 := a0 + a2
		e2 := a0 - a2
		e1 := a1 + complex(-imag(a3), real(a3)) // a1 + i*a3
		e3 := a1 + complex(imag(a3), -real(a3)) // a1 - i*a3
		o0 := a4 + a6
		o2 := a4 - a6
		o1 := a5 + complex(-imag(a7), real(a7))
		o3 := a5 + complex(imag(a7), -real(a7))

		t1 := complex(root2Over2*(real(o1)-imag(o1)), root2Over2*(imag(o1)+real(o1)))
		t2 := complex(-imag(o2), real(o2))
		t3 := complex(-root2Over2*(real(o3)+imag(o3)), root2Over2*(real(o3)-imag(o3)))

		out := work[8*g : 8*g+8 : 8*g+8]
		out[0] = e0 + o0
		out[1] = e1 + t1
		out[2] = e2 + t2
		out[3] = e3 + t3
		out[4] = e0 - o0
		out[5] = e1 - t1
		out[6] = e2 - t2
		out[7] = e3 - t3
	}

	offset := radix8StagesInverseComplex64(work, twiddle, n, limit)
	radix8TailInverseComplex64(work, twiddle, n, tail, offset)

	if &work[0] != &dst[0] {
		copy(dst[:n], work)
	}

	return true
}

// radix8StagesInverseComplex64 is the inverse twin of
// radix8StagesForwardComplex64. The twiddles are already conjugated by
// prepareTwiddleRadix8Complex64, so only the butterfly's internal rotations
// differ.
func radix8StagesInverseComplex64(work, twiddle []complex64, n, limit int) int {
	offset := 0

	for m := 8; m*8 <= limit; m *= 8 {
		span := 8 * m

		w1 := twiddle[offset : offset+m]
		w2 := twiddle[offset+m : offset+2*m]
		w3 := twiddle[offset+2*m : offset+3*m]
		w4 := twiddle[offset+3*m : offset+4*m]
		w5 := twiddle[offset+4*m : offset+5*m]
		w6 := twiddle[offset+5*m : offset+6*m]
		w7 := twiddle[offset+6*m : offset+7*m]
		offset += 7 * m

		for base := 0; base < n; base += span {
			blk := work[base : base+span : base+span]

			for j := range m {
				x0 := blk[j]
				x1 := mathpkg.MulComplex64(w1[j], blk[j+m])
				x2 := mathpkg.MulComplex64(w2[j], blk[j+2*m])
				x3 := mathpkg.MulComplex64(w3[j], blk[j+3*m])
				x4 := mathpkg.MulComplex64(w4[j], blk[j+4*m])
				x5 := mathpkg.MulComplex64(w5[j], blk[j+5*m])
				x6 := mathpkg.MulComplex64(w6[j], blk[j+6*m])
				x7 := mathpkg.MulComplex64(w7[j], blk[j+7*m])

				a0 := x0 + x4
				a1 := x0 - x4
				a2 := x2 + x6
				a3 := x2 - x6
				a4 := x1 + x5
				a5 := x1 - x5
				a6 := x3 + x7
				a7 := x3 - x7

				e0 := a0 + a2
				e2 := a0 - a2
				e1 := a1 + complex(-imag(a3), real(a3))
				e3 := a1 + complex(imag(a3), -real(a3))
				o0 := a4 + a6
				o2 := a4 - a6
				o1 := a5 + complex(-imag(a7), real(a7))
				o3 := a5 + complex(imag(a7), -real(a7))

				t1 := complex(root2Over2*(real(o1)-imag(o1)), root2Over2*(imag(o1)+real(o1)))
				t2 := complex(-imag(o2), real(o2))
				t3 := complex(-root2Over2*(real(o3)+imag(o3)), root2Over2*(real(o3)-imag(o3)))

				blk[j] = e0 + o0
				blk[j+m] = e1 + t1
				blk[j+2*m] = e2 + t2
				blk[j+3*m] = e3 + t3
				blk[j+4*m] = e0 - o0
				blk[j+5*m] = e1 - t1
				blk[j+6*m] = e2 - t2
				blk[j+7*m] = e3 - t3
			}
		}
	}

	return offset
}

// radix8TailInverseComplex64 is the inverse twin of
// radix8TailForwardComplex64.
func radix8TailInverseComplex64(work, twiddle []complex64, n, tail, offset int) {
	switch tail {
	case 2:
		m := n / 2
		w1 := twiddle[offset : offset+m]

		for j := range m {
			a := work[j]
			b := mathpkg.MulComplex64(w1[j], work[j+m])
			work[j] = a + b
			work[j+m] = a - b
		}

	case 4:
		m := n / 4
		w1 := twiddle[offset : offset+m]
		w2 := twiddle[offset+m : offset+2*m]
		w3 := twiddle[offset+2*m : offset+3*m]

		for j := range m {
			x0 := work[j]
			x1 := mathpkg.MulComplex64(w1[j], work[j+m])
			x2 := mathpkg.MulComplex64(w2[j], work[j+2*m])
			x3 := mathpkg.MulComplex64(w3[j], work[j+3*m])

			t0 := x0 + x2
			t1 := x0 - x2
			t2 := x1 + x3
			t3 := x1 - x3

			work[j] = t0 + t2
			work[j+m] = t1 + complex(-imag(t3), real(t3)) // t1 + i*t3
			work[j+2*m] = t0 - t2
			work[j+3*m] = t1 + complex(imag(t3), -real(t3)) // t1 - i*t3
		}
	}
}
