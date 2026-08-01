package kernels

import (
	"math"
	"math/bits"
	"sync"

	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// Radix-16 DIT, size-generic, pure Go.
//
// This exists to answer a go/no-go question before any assembly is written, by
// the same protocol that settled radix-8: pure Go has no register budget, so it
// isolates the only thing radix-16 has going for it -- passes over the buffer
// and butterfly operation count. If radix-16 cannot win here it will not win in
// assembly either, where it needs 16 live streams against AVX2's 16 YMM
// registers and AVX-512's 32 ZMM.
//
// The structural case is weaker than radix-8's was, and the numbers should be
// read with that in mind. Radix-16 makes log2(n)/4 passes against radix-8's
// log2(n)/3 -- a 25% reduction where radix-8 bought 33% over radix-4 -- while
// the twiddle table per stage grows from 7 planes to 15 and the stage-1 gather
// widens from an 8x8 digit-reversed transpose to a 16x16 one. Diminishing
// return against growing cost is exactly the shape that stops a radix ladder,
// so a loss here is a real answer and not a failed attempt.
//
// Layout, writing n = 16^k, 2*16^k, 4*16^k or 8*16^k:
//
//	Stage 1: n/16 groups x 1 butterfly, no twiddles, fused with the gather.
//	Stage s: n/(16m) groups x m butterflies, m = 16^(s-1), while 16m <= limit.
//	Tail:    for n != 16^k only, one radix-2, radix-4 or radix-8 stage over the
//	         whole buffer.
//
// where limit is n for a power of sixteen and n/2, n/4 or n/8 for the other
// three shapes. Those four shapes cover every power of two, so no size is out
// of reach. This mirrors radix8_generic.go exactly, one radix up, so the two
// can be compared without the comparison measuring a difference in scaffolding.
//
// Each radix-16 stage needs 15*m twiddles laid out as fifteen contiguous planes
// (w1[0..m-1] ... w15[0..m-1]); the tail needs (tail-1)*(n/tail) more. Either
// way the total is n-16 elements, exactly as the radix-8 kernel's is n-8.
//
// Every stage is in place. Only the stage-1 gather writes somewhere new, so the
// pass count is honest rather than one array per stage.

const (
	// radix16MinSize is the smallest n the kernel handles: one stage-1 group.
	radix16MinSize = 16
	// radix16MaxSize bounds the cached permutation tables.
	radix16MaxSize = 1 << 16
)

// radix16Limit reports the largest span the radix-16 stages may cover, the
// radix of the single tail stage that finishes the transform (1 meaning none),
// and whether n has a shape the kernel supports at all.
func radix16Limit(n int) (limit, tail int, ok bool) {
	if n < radix16MinSize || n > radix16MaxSize || n&(n-1) != 0 {
		return 0, 0, false
	}

	// n = 2^t is 16^k, 2*16^k, 4*16^k or 8*16^k according to t mod 4.
	switch bits.TrailingZeros(uint(n)) % 4 {
	case 0:
		return n, 1, true
	case 1:
		return n / 2, 2, true
	case 2:
		return n / 4, 4, true
	default:
		return n / 8, 8, true
	}
}

// radix16SizeOK reports whether the kernel can handle length n.
func radix16SizeOK(n int) bool {
	_, _, ok := radix16Limit(n)

	return ok
}

// twiddleSizeRadix16 returns the element count of the packed twiddle table.
//
// Only the first n-16 elements carry data. The request is padded to n+16 for
// the same reason radix-8's is padded to n+8: at n-16 a caller that handed this
// kernel a plain length-n DIT twiddle table would pass the length check and be
// silently transformed against the wrong factors.
func twiddleSizeRadix16(n int) int {
	if !radix16SizeOK(n) {
		return 0
	}

	return n + 16
}

// prepareTwiddleRadix16Complex64 fills dst with the per-stage twiddle planes
// described above. For the inverse transform the imaginary parts are negated,
// which is the conjugate W^-k = conj(W^k), so the stage code is direction-blind
// apart from its butterfly.
func prepareTwiddleRadix16Complex64(n int, inverse bool, dst []complex64) {
	limit, tail, ok := radix16Limit(n)
	if !ok || len(dst) < twiddleSizeRadix16(n) {
		return
	}

	clear(dst[:twiddleSizeRadix16(n)])

	sign := -1.0
	if inverse {
		sign = 1.0
	}

	w := func(e int) complex64 {
		sin, cos := math.Sincos(sign * 2 * math.Pi * float64(e%n) / float64(n))

		return mathpkg.ComplexFromFloat64[complex64](cos, sin)
	}

	offset := 0

	// Radix-16 stages: butterfly j of block d is scaled by W_(16m)^(jd).
	for m := 16; m*16 <= limit; m *= 16 {
		step := n / (16 * m)

		for d := 1; d <= 15; d++ {
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

// radix16GroupIndexTable memoises one stage-1 group index table.
type radix16GroupIndexTable struct {
	once sync.Once
	idx  []int32
}

//nolint:gochecknoglobals // memoised permutation tables, one slot per log2(n)
var radix16GroupIndexTables [17]radix16GroupIndexTable

// radix16GroupIndices returns the memoised stage-1 group index table for n:
// entry g is the source index of the first input of group g.
//
// The digit-reversal permutation p satisfies p[16g+d] = p[16g] + d*(n/16) for
// all four supported shapes, so storing only p[16g] shrinks the table by 16x
// and int32 halves it again. That stride property is asserted directly in
// internal/math's tests rather than being assumed here, because a table can be
// a valid bijection without it -- in which case a full-table kernel would still
// round-trip and only this compressed form would be wrong.
//
// The permutation itself comes from internal/math rather than being rederived
// here, so it cannot drift from the one the rest of the library uses.
func radix16GroupIndices(n int) []int32 {
	if !radix16SizeOK(n) {
		return nil
	}

	slot := &radix16GroupIndexTables[bits.TrailingZeros(uint(n))]
	slot.once.Do(func() {
		_, tail, ok := radix16Limit(n)
		if !ok {
			return
		}

		var full []int

		switch tail {
		case 1:
			full = mathpkg.ComputeBitReversalIndicesRadix16(n)
		case 2:
			full = mathpkg.ComputeBitReversalIndicesRadix16Then2(n)
		case 4:
			full = mathpkg.ComputeBitReversalIndicesRadix16Then4(n)
		default:
			full = mathpkg.ComputeBitReversalIndicesRadix16Then8(n)
		}

		if len(full) != n {
			return
		}

		idx := make([]int32, n/16)
		for g := range idx {
			idx[g] = int32(full[16*g])
		}

		slot.idx = idx
	})

	return slot.idx
}

// radix16PrologueComplex64 validates the arguments common to both directions
// and returns the working buffer, the stage-1 group table and the ladder shape.
func radix16PrologueComplex64(dst, src, twiddle, scratch []complex64) (
	work []complex64, groups []int32, limit, tail int, ok bool,
) {
	n := len(src)

	limit, tail, ok = radix16Limit(n)
	if !ok || len(dst) < n || len(scratch) < n || len(twiddle) < twiddleSizeRadix16(n) {
		return nil, nil, 0, 0, false
	}

	groups = radix16GroupIndices(n)
	if len(groups) != n/16 {
		return nil, nil, 0, 0, false
	}

	work = dst[:n]
	if &dst[0] == &src[0] {
		// Stage 1 gathers, so it cannot write over its own source.
		work = scratch[:n]
	}

	return work, groups, limit, tail, true
}

// The sixteen-point butterflies below are butterfly16ForwardComplex64 and
// butterfly16InverseComplex64 from radix16.go, unrolled into the stage loops
// with both of their radix-4 levels written out.
//
// The unrolling is not a style choice, and skipping it would have quietly
// biased the whole measurement. butterfly16ForwardComplex64 costs 1085 inline
// units against a budget of 80, and even the radix-4 level it is built from
// misses by one (81 against 80), so left as calls every butterfly would pay
// five of them -- against radix-8's zero, since radix8_generic.go unrolls for
// exactly this reason. Passing the sixteen values as arguments instead is no
// better: 32 float32s overflow the register ABI and spill to the stack. The
// probe would then be measuring Go's calling convention rather than the radix.
//
// TestRadix16LadderMatchesButterfly pins the unrolled copies against the
// readable original in radix16.go so the two cannot drift.

// forwardRadix16Complex64 is the CodeletFunc entry point for the forward
// transform. twiddle must be the table produced by
// prepareTwiddleRadix16Complex64.
func forwardRadix16Complex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	work, groups, limit, tail, ok := radix16PrologueComplex64(dst, src, twiddle, scratch)
	if !ok {
		return false
	}

	s := src[:n]
	st := n / 16

	// Stage 1: one twiddle-free butterfly per group, fused with the gather.
	for g, first := range groups {
		p := int(first)

		x0, x1, x2, x3 := s[p], s[p+st], s[p+2*st], s[p+3*st]
		x4, x5, x6, x7 := s[p+4*st], s[p+5*st], s[p+6*st], s[p+7*st]
		x8, x9, x10, x11 := s[p+8*st], s[p+9*st], s[p+10*st], s[p+11*st]
		x12, x13, x14, x15 := s[p+12*st], s[p+13*st], s[p+14*st], s[p+15*st]

		// Level 1: a radix-4 DFT down each stride-4 column. b<n0><k0>.
		u0, u1, u2, u3 := x0+x8, x0-x8, x4+x12, x4-x12
		b00, b02 := u0+u2, u0-u2
		b01 := u1 + complex(imag(u3), -real(u3))
		b03 := u1 + complex(-imag(u3), real(u3))

		v0, v1, v2, v3 := x1+x9, x1-x9, x5+x13, x5-x13
		b10, b12 := v0+v2, v0-v2
		b11 := v1 + complex(imag(v3), -real(v3))
		b13 := v1 + complex(-imag(v3), real(v3))

		q0, q1, q2, q3 := x2+x10, x2-x10, x6+x14, x6-x14
		b20, b22 := q0+q2, q0-q2
		b21 := q1 + complex(imag(q3), -real(q3))
		b23 := q1 + complex(-imag(q3), real(q3))

		z0, z1, z2, z3 := x3+x11, x3-x11, x7+x15, x7-x15
		b30, b32 := z0+z2, z0-z2
		b31 := z1 + complex(imag(z3), -real(z3))
		b33 := z1 + complex(-imag(z3), real(z3))

		// Twiddle layer: b<n0><k0> scaled by W16^(n0*k0). Row 0 and column 0
		// are W16^0 = 1 and drop out.
		c11 := mulW16Pow1Complex64(b11)
		c12 := mulW16Pow2Complex64(b12)
		c13 := mulW16Pow3Complex64(b13)
		c21 := mulW16Pow2Complex64(b21)
		c22 := mulNegIComplex64(b22)
		c23 := mulW16Pow6Complex64(b23)
		c31 := mulW16Pow3Complex64(b31)
		c32 := mulW16Pow6Complex64(b32)
		c33 := -mulW16Pow1Complex64(b33) // W16^9 = -W16^1

		// Level 2: a radix-4 DFT across the four columns, for each k0. Output
		// k1 of the k0-th column is X[k0+4*k1], hence the stride-4 stores.
		r0, r1, r2, r3 := b00+b20, b00-b20, b10+b30, b10-b30
		e0, e1, e2, e3 := b01+c21, b01-c21, c11+c31, c11-c31
		f0, f1, f2, f3 := b02+c22, b02-c22, c12+c32, c12-c32
		g0, g1, g2, g3 := b03+c23, b03-c23, c13+c33, c13-c33

		out := work[16*g : 16*g+16 : 16*g+16]
		out[0] = r0 + r2
		out[1] = e0 + e2
		out[2] = f0 + f2
		out[3] = g0 + g2
		out[4] = r1 + complex(imag(r3), -real(r3))
		out[5] = e1 + complex(imag(e3), -real(e3))
		out[6] = f1 + complex(imag(f3), -real(f3))
		out[7] = g1 + complex(imag(g3), -real(g3))
		out[8] = r0 - r2
		out[9] = e0 - e2
		out[10] = f0 - f2
		out[11] = g0 - g2
		out[12] = r1 + complex(-imag(r3), real(r3))
		out[13] = e1 + complex(-imag(e3), real(e3))
		out[14] = f1 + complex(-imag(f3), real(f3))
		out[15] = g1 + complex(-imag(g3), real(g3))
	}

	offset := radix16StagesForwardComplex64(work, twiddle, n, limit)
	radix16TailForwardComplex64(work, twiddle, n, tail, offset)

	if &work[0] != &dst[0] {
		copy(dst[:n], work)
	}

	return true
}

// radix16StagesForwardComplex64 runs the twiddled radix-16 stages in place and
// returns the number of twiddle elements consumed.
func radix16StagesForwardComplex64(work, twiddle []complex64, n, limit int) int {
	offset := 0

	for m := 16; m*16 <= limit; m *= 16 {
		span := 16 * m

		var w [16][]complex64
		for d := 1; d <= 15; d++ {
			w[d] = twiddle[offset+(d-1)*m : offset+d*m]
		}

		offset += 15 * m

		for base := 0; base < n; base += span {
			blk := work[base : base+span : base+span]

			for j := range m {
				x0 := blk[j]
				x1 := mathpkg.MulComplex64(w[1][j], blk[j+m])
				x2 := mathpkg.MulComplex64(w[2][j], blk[j+2*m])
				x3 := mathpkg.MulComplex64(w[3][j], blk[j+3*m])
				x4 := mathpkg.MulComplex64(w[4][j], blk[j+4*m])
				x5 := mathpkg.MulComplex64(w[5][j], blk[j+5*m])
				x6 := mathpkg.MulComplex64(w[6][j], blk[j+6*m])
				x7 := mathpkg.MulComplex64(w[7][j], blk[j+7*m])
				x8 := mathpkg.MulComplex64(w[8][j], blk[j+8*m])
				x9 := mathpkg.MulComplex64(w[9][j], blk[j+9*m])
				x10 := mathpkg.MulComplex64(w[10][j], blk[j+10*m])
				x11 := mathpkg.MulComplex64(w[11][j], blk[j+11*m])
				x12 := mathpkg.MulComplex64(w[12][j], blk[j+12*m])
				x13 := mathpkg.MulComplex64(w[13][j], blk[j+13*m])
				x14 := mathpkg.MulComplex64(w[14][j], blk[j+14*m])
				x15 := mathpkg.MulComplex64(w[15][j], blk[j+15*m])

				u0, u1, u2, u3 := x0+x8, x0-x8, x4+x12, x4-x12
				b00, b02 := u0+u2, u0-u2
				b01 := u1 + complex(imag(u3), -real(u3))
				b03 := u1 + complex(-imag(u3), real(u3))

				v0, v1, v2, v3 := x1+x9, x1-x9, x5+x13, x5-x13
				b10, b12 := v0+v2, v0-v2
				b11 := v1 + complex(imag(v3), -real(v3))
				b13 := v1 + complex(-imag(v3), real(v3))

				q0, q1, q2, q3 := x2+x10, x2-x10, x6+x14, x6-x14
				b20, b22 := q0+q2, q0-q2
				b21 := q1 + complex(imag(q3), -real(q3))
				b23 := q1 + complex(-imag(q3), real(q3))

				z0, z1, z2, z3 := x3+x11, x3-x11, x7+x15, x7-x15
				b30, b32 := z0+z2, z0-z2
				b31 := z1 + complex(imag(z3), -real(z3))
				b33 := z1 + complex(-imag(z3), real(z3))

				c11 := mulW16Pow1Complex64(b11)
				c12 := mulW16Pow2Complex64(b12)
				c13 := mulW16Pow3Complex64(b13)
				c21 := mulW16Pow2Complex64(b21)
				c22 := mulNegIComplex64(b22)
				c23 := mulW16Pow6Complex64(b23)
				c31 := mulW16Pow3Complex64(b31)
				c32 := mulW16Pow6Complex64(b32)
				c33 := -mulW16Pow1Complex64(b33)

				r0, r1, r2, r3 := b00+b20, b00-b20, b10+b30, b10-b30
				e0, e1, e2, e3 := b01+c21, b01-c21, c11+c31, c11-c31
				f0, f1, f2, f3 := b02+c22, b02-c22, c12+c32, c12-c32
				g0, g1, g2, g3 := b03+c23, b03-c23, c13+c33, c13-c33

				blk[j] = r0 + r2
				blk[j+m] = e0 + e2
				blk[j+2*m] = f0 + f2
				blk[j+3*m] = g0 + g2
				blk[j+4*m] = r1 + complex(imag(r3), -real(r3))
				blk[j+5*m] = e1 + complex(imag(e3), -real(e3))
				blk[j+6*m] = f1 + complex(imag(f3), -real(f3))
				blk[j+7*m] = g1 + complex(imag(g3), -real(g3))
				blk[j+8*m] = r0 - r2
				blk[j+9*m] = e0 - e2
				blk[j+10*m] = f0 - f2
				blk[j+11*m] = g0 - g2
				blk[j+12*m] = r1 + complex(-imag(r3), real(r3))
				blk[j+13*m] = e1 + complex(-imag(e3), real(e3))
				blk[j+14*m] = f1 + complex(-imag(f3), real(f3))
				blk[j+15*m] = g1 + complex(-imag(g3), real(g3))
			}
		}
	}

	return offset
}

// radix16TailForwardComplex64 runs the single widest stage for the shapes that
// need one: radix-2 for n = 2*16^k, radix-4 for 4*16^k, radix-8 for 8*16^k and
// nothing for a power of sixteen.
func radix16TailForwardComplex64(work, twiddle []complex64, n, tail, offset int) {
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

			t0, t1, t2, t3 := x0+x2, x0-x2, x1+x3, x1-x3

			work[j] = t0 + t2
			work[j+m] = t1 + complex(imag(t3), -real(t3)) // t1 - i*t3
			work[j+2*m] = t0 - t2
			work[j+3*m] = t1 + complex(-imag(t3), real(t3)) // t1 + i*t3
		}

	case 8:
		radix16Tail8ForwardComplex64(work, twiddle, n, offset)
	}
}

// radix16Tail8ForwardComplex64 is the radix-8 tail stage, the same butterfly
// radix8_generic.go uses in its own stages.
func radix16Tail8ForwardComplex64(work, twiddle []complex64, n, offset int) {
	m := n / 8

	var w [8][]complex64
	for d := 1; d <= 7; d++ {
		w[d] = twiddle[offset+(d-1)*m : offset+d*m]
	}

	for j := range m {
		x0 := work[j]
		x1 := mathpkg.MulComplex64(w[1][j], work[j+m])
		x2 := mathpkg.MulComplex64(w[2][j], work[j+2*m])
		x3 := mathpkg.MulComplex64(w[3][j], work[j+3*m])
		x4 := mathpkg.MulComplex64(w[4][j], work[j+4*m])
		x5 := mathpkg.MulComplex64(w[5][j], work[j+5*m])
		x6 := mathpkg.MulComplex64(w[6][j], work[j+6*m])
		x7 := mathpkg.MulComplex64(w[7][j], work[j+7*m])

		a0, a1, a2, a3 := x0+x4, x0-x4, x2+x6, x2-x6
		a4, a5, a6, a7 := x1+x5, x1-x5, x3+x7, x3-x7

		e0, e2 := a0+a2, a0-a2
		e1 := a1 + complex(imag(a3), -real(a3)) // a1 - i*a3
		e3 := a1 + complex(-imag(a3), real(a3)) // a1 + i*a3
		o0, o2 := a4+a6, a4-a6
		o1 := a5 + complex(imag(a7), -real(a7))
		o3 := a5 + complex(-imag(a7), real(a7))

		t1 := complex(root2Over2*(real(o1)+imag(o1)), root2Over2*(imag(o1)-real(o1)))
		t2 := complex(imag(o2), -real(o2))
		t3 := complex(root2Over2*(imag(o3)-real(o3)), -root2Over2*(real(o3)+imag(o3)))

		work[j] = e0 + o0
		work[j+m] = e1 + t1
		work[j+2*m] = e2 + t2
		work[j+3*m] = e3 + t3
		work[j+4*m] = e0 - o0
		work[j+5*m] = e1 - t1
		work[j+6*m] = e2 - t2
		work[j+7*m] = e3 - t3
	}
}

// inverseRadix16Complex64 is the CodeletFunc entry point for the inverse
// transform. The 1/n normalisation is exact here (n is a power of two) and is
// folded into the stage-1 gather rather than costing a separate pass over the
// data; by linearity the result is identical either way.
func inverseRadix16Complex64(dst, src, twiddle, scratch []complex64) bool {
	n := len(src)

	work, groups, limit, tail, ok := radix16PrologueComplex64(dst, src, twiddle, scratch)
	if !ok {
		return false
	}

	s := src[:n]
	st := n / 16
	k := float32(1) / float32(n)

	for g, first := range groups {
		p := int(first)

		// The 1/n scale rides along with the gather: two real multiplies per
		// element, and no extra pass over the buffer.
		d0, d1, d2, d3 := s[p], s[p+st], s[p+2*st], s[p+3*st]
		d4, d5, d6, d7 := s[p+4*st], s[p+5*st], s[p+6*st], s[p+7*st]
		d8, d9, d10, d11 := s[p+8*st], s[p+9*st], s[p+10*st], s[p+11*st]
		d12, d13, d14, d15 := s[p+12*st], s[p+13*st], s[p+14*st], s[p+15*st]

		x0 := complex(real(d0)*k, imag(d0)*k)
		x1 := complex(real(d1)*k, imag(d1)*k)
		x2 := complex(real(d2)*k, imag(d2)*k)
		x3 := complex(real(d3)*k, imag(d3)*k)
		x4 := complex(real(d4)*k, imag(d4)*k)
		x5 := complex(real(d5)*k, imag(d5)*k)
		x6 := complex(real(d6)*k, imag(d6)*k)
		x7 := complex(real(d7)*k, imag(d7)*k)
		x8 := complex(real(d8)*k, imag(d8)*k)
		x9 := complex(real(d9)*k, imag(d9)*k)
		x10 := complex(real(d10)*k, imag(d10)*k)
		x11 := complex(real(d11)*k, imag(d11)*k)
		x12 := complex(real(d12)*k, imag(d12)*k)
		x13 := complex(real(d13)*k, imag(d13)*k)
		x14 := complex(real(d14)*k, imag(d14)*k)
		x15 := complex(real(d15)*k, imag(d15)*k)

		u0, u1, u2, u3 := x0+x8, x0-x8, x4+x12, x4-x12
		b00, b02 := u0+u2, u0-u2
		b01 := u1 + complex(-imag(u3), real(u3))
		b03 := u1 + complex(imag(u3), -real(u3))

		v0, v1, v2, v3 := x1+x9, x1-x9, x5+x13, x5-x13
		b10, b12 := v0+v2, v0-v2
		b11 := v1 + complex(-imag(v3), real(v3))
		b13 := v1 + complex(imag(v3), -real(v3))

		q0, q1, q2, q3 := x2+x10, x2-x10, x6+x14, x6-x14
		b20, b22 := q0+q2, q0-q2
		b21 := q1 + complex(-imag(q3), real(q3))
		b23 := q1 + complex(imag(q3), -real(q3))

		z0, z1, z2, z3 := x3+x11, x3-x11, x7+x15, x7-x15
		b30, b32 := z0+z2, z0-z2
		b31 := z1 + complex(-imag(z3), real(z3))
		b33 := z1 + complex(imag(z3), -real(z3))

		c11 := mulW16PowNeg1Complex64(b11)
		c12 := mulW16PowNeg2Complex64(b12)
		c13 := mulW16PowNeg3Complex64(b13)
		c21 := mulW16PowNeg2Complex64(b21)
		c22 := mulPosIComplex64(b22)
		c23 := mulW16PowNeg6Complex64(b23)
		c31 := mulW16PowNeg3Complex64(b31)
		c32 := mulW16PowNeg6Complex64(b32)
		c33 := -mulW16PowNeg1Complex64(b33)

		r0, r1, r2, r3 := b00+b20, b00-b20, b10+b30, b10-b30
		e0, e1, e2, e3 := b01+c21, b01-c21, c11+c31, c11-c31
		f0, f1, f2, f3 := b02+c22, b02-c22, c12+c32, c12-c32
		g0, g1, g2, g3 := b03+c23, b03-c23, c13+c33, c13-c33

		out := work[16*g : 16*g+16 : 16*g+16]
		out[0] = r0 + r2
		out[1] = e0 + e2
		out[2] = f0 + f2
		out[3] = g0 + g2
		out[4] = r1 + complex(-imag(r3), real(r3))
		out[5] = e1 + complex(-imag(e3), real(e3))
		out[6] = f1 + complex(-imag(f3), real(f3))
		out[7] = g1 + complex(-imag(g3), real(g3))
		out[8] = r0 - r2
		out[9] = e0 - e2
		out[10] = f0 - f2
		out[11] = g0 - g2
		out[12] = r1 + complex(imag(r3), -real(r3))
		out[13] = e1 + complex(imag(e3), -real(e3))
		out[14] = f1 + complex(imag(f3), -real(f3))
		out[15] = g1 + complex(imag(g3), -real(g3))
	}

	offset := radix16StagesInverseComplex64(work, twiddle, n, limit)
	radix16TailInverseComplex64(work, twiddle, n, tail, offset)

	if &work[0] != &dst[0] {
		copy(dst[:n], work)
	}

	return true
}

// radix16StagesInverseComplex64 is the inverse twin of
// radix16StagesForwardComplex64. The twiddles are already conjugated by
// prepareTwiddleRadix16Complex64, so only the butterfly's internal rotations
// differ.
func radix16StagesInverseComplex64(work, twiddle []complex64, n, limit int) int {
	offset := 0

	for m := 16; m*16 <= limit; m *= 16 {
		span := 16 * m

		var w [16][]complex64
		for d := 1; d <= 15; d++ {
			w[d] = twiddle[offset+(d-1)*m : offset+d*m]
		}

		offset += 15 * m

		for base := 0; base < n; base += span {
			blk := work[base : base+span : base+span]

			for j := range m {
				x0 := blk[j]
				x1 := mathpkg.MulComplex64(w[1][j], blk[j+m])
				x2 := mathpkg.MulComplex64(w[2][j], blk[j+2*m])
				x3 := mathpkg.MulComplex64(w[3][j], blk[j+3*m])
				x4 := mathpkg.MulComplex64(w[4][j], blk[j+4*m])
				x5 := mathpkg.MulComplex64(w[5][j], blk[j+5*m])
				x6 := mathpkg.MulComplex64(w[6][j], blk[j+6*m])
				x7 := mathpkg.MulComplex64(w[7][j], blk[j+7*m])
				x8 := mathpkg.MulComplex64(w[8][j], blk[j+8*m])
				x9 := mathpkg.MulComplex64(w[9][j], blk[j+9*m])
				x10 := mathpkg.MulComplex64(w[10][j], blk[j+10*m])
				x11 := mathpkg.MulComplex64(w[11][j], blk[j+11*m])
				x12 := mathpkg.MulComplex64(w[12][j], blk[j+12*m])
				x13 := mathpkg.MulComplex64(w[13][j], blk[j+13*m])
				x14 := mathpkg.MulComplex64(w[14][j], blk[j+14*m])
				x15 := mathpkg.MulComplex64(w[15][j], blk[j+15*m])

				u0, u1, u2, u3 := x0+x8, x0-x8, x4+x12, x4-x12
				b00, b02 := u0+u2, u0-u2
				b01 := u1 + complex(-imag(u3), real(u3))
				b03 := u1 + complex(imag(u3), -real(u3))

				v0, v1, v2, v3 := x1+x9, x1-x9, x5+x13, x5-x13
				b10, b12 := v0+v2, v0-v2
				b11 := v1 + complex(-imag(v3), real(v3))
				b13 := v1 + complex(imag(v3), -real(v3))

				q0, q1, q2, q3 := x2+x10, x2-x10, x6+x14, x6-x14
				b20, b22 := q0+q2, q0-q2
				b21 := q1 + complex(-imag(q3), real(q3))
				b23 := q1 + complex(imag(q3), -real(q3))

				z0, z1, z2, z3 := x3+x11, x3-x11, x7+x15, x7-x15
				b30, b32 := z0+z2, z0-z2
				b31 := z1 + complex(-imag(z3), real(z3))
				b33 := z1 + complex(imag(z3), -real(z3))

				c11 := mulW16PowNeg1Complex64(b11)
				c12 := mulW16PowNeg2Complex64(b12)
				c13 := mulW16PowNeg3Complex64(b13)
				c21 := mulW16PowNeg2Complex64(b21)
				c22 := mulPosIComplex64(b22)
				c23 := mulW16PowNeg6Complex64(b23)
				c31 := mulW16PowNeg3Complex64(b31)
				c32 := mulW16PowNeg6Complex64(b32)
				c33 := -mulW16PowNeg1Complex64(b33)

				r0, r1, r2, r3 := b00+b20, b00-b20, b10+b30, b10-b30
				e0, e1, e2, e3 := b01+c21, b01-c21, c11+c31, c11-c31
				f0, f1, f2, f3 := b02+c22, b02-c22, c12+c32, c12-c32
				g0, g1, g2, g3 := b03+c23, b03-c23, c13+c33, c13-c33

				blk[j] = r0 + r2
				blk[j+m] = e0 + e2
				blk[j+2*m] = f0 + f2
				blk[j+3*m] = g0 + g2
				blk[j+4*m] = r1 + complex(-imag(r3), real(r3))
				blk[j+5*m] = e1 + complex(-imag(e3), real(e3))
				blk[j+6*m] = f1 + complex(-imag(f3), real(f3))
				blk[j+7*m] = g1 + complex(-imag(g3), real(g3))
				blk[j+8*m] = r0 - r2
				blk[j+9*m] = e0 - e2
				blk[j+10*m] = f0 - f2
				blk[j+11*m] = g0 - g2
				blk[j+12*m] = r1 + complex(imag(r3), -real(r3))
				blk[j+13*m] = e1 + complex(imag(e3), -real(e3))
				blk[j+14*m] = f1 + complex(imag(f3), -real(f3))
				blk[j+15*m] = g1 + complex(imag(g3), -real(g3))
			}
		}
	}

	return offset
}

// radix16TailInverseComplex64 is the inverse twin of
// radix16TailForwardComplex64.
func radix16TailInverseComplex64(work, twiddle []complex64, n, tail, offset int) {
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

			t0, t1, t2, t3 := x0+x2, x0-x2, x1+x3, x1-x3

			work[j] = t0 + t2
			work[j+m] = t1 + complex(-imag(t3), real(t3)) // t1 + i*t3
			work[j+2*m] = t0 - t2
			work[j+3*m] = t1 + complex(imag(t3), -real(t3)) // t1 - i*t3
		}

	case 8:
		radix16Tail8InverseComplex64(work, twiddle, n, offset)
	}
}

// radix16Tail8InverseComplex64 is the inverse twin of
// radix16Tail8ForwardComplex64.
func radix16Tail8InverseComplex64(work, twiddle []complex64, n, offset int) {
	m := n / 8

	var w [8][]complex64
	for d := 1; d <= 7; d++ {
		w[d] = twiddle[offset+(d-1)*m : offset+d*m]
	}

	for j := range m {
		x0 := work[j]
		x1 := mathpkg.MulComplex64(w[1][j], work[j+m])
		x2 := mathpkg.MulComplex64(w[2][j], work[j+2*m])
		x3 := mathpkg.MulComplex64(w[3][j], work[j+3*m])
		x4 := mathpkg.MulComplex64(w[4][j], work[j+4*m])
		x5 := mathpkg.MulComplex64(w[5][j], work[j+5*m])
		x6 := mathpkg.MulComplex64(w[6][j], work[j+6*m])
		x7 := mathpkg.MulComplex64(w[7][j], work[j+7*m])

		a0, a1, a2, a3 := x0+x4, x0-x4, x2+x6, x2-x6
		a4, a5, a6, a7 := x1+x5, x1-x5, x3+x7, x3-x7

		e0, e2 := a0+a2, a0-a2
		e1 := a1 + complex(-imag(a3), real(a3)) // a1 + i*a3
		e3 := a1 + complex(imag(a3), -real(a3)) // a1 - i*a3
		o0, o2 := a4+a6, a4-a6
		o1 := a5 + complex(-imag(a7), real(a7))
		o3 := a5 + complex(imag(a7), -real(a7))

		t1 := complex(root2Over2*(real(o1)-imag(o1)), root2Over2*(imag(o1)+real(o1)))
		t2 := complex(-imag(o2), real(o2))
		t3 := complex(-root2Over2*(real(o3)+imag(o3)), root2Over2*(real(o3)-imag(o3)))

		work[j] = e0 + o0
		work[j+m] = e1 + t1
		work[j+2*m] = e2 + t2
		work[j+3*m] = e3 + t3
		work[j+4*m] = e0 - o0
		work[j+5*m] = e1 - t1
		work[j+6*m] = e2 - t2
		work[j+7*m] = e3 - t3
	}
}
