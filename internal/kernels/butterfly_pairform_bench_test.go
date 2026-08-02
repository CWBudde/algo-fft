package kernels

// Benchmarks for the radix-5/7/11 conjugate-pair-form butterflies added in
// this change, plus a "Matrix" baseline that re-implements the ORIGINAL
// naive DFT-matrix form these butterflies used to be (kept only here, for
// comparison — the production code no longer has this path). See the task
// notes in internal/fft/mixedradix_stage_twiddle.go and the derivations in
// internal/asm/amd64/avx2_f32_mixedradix_stage{5,7,11}.s.
//
// Numbers from a run on this development machine are recorded in the PR/
// task report; they are CONTENDED (shared host, other agents building and
// testing concurrently) and indicative only. Re-measure on a quiet machine
// before trusting the magnitude of any improvement.

import (
	"math"
	"testing"

	m "github.com/cwbudde/algo-fft/internal/math"
)

//nolint:gochecknoglobals // benchmark-only baseline tables
var (
	oldRadix5MatrixFwd64 [4]complex64
	oldRadix5MatrixInv64 [4]complex64

	oldRadix7MatrixFwd64 [49]complex64
	oldRadix7MatrixInv64 [49]complex64

	oldRadix11MatrixFwd64 [121]complex64
	oldRadix11MatrixInv64 [121]complex64
)

//nolint:gochecknoinits
func init() {
	for k := 1; k <= 4; k++ {
		angle := -2 * math.Pi * float64(k) / 5
		oldRadix5MatrixFwd64[k-1] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
		oldRadix5MatrixInv64[k-1] = complex(float32(math.Cos(angle)), float32(-math.Sin(angle)))
	}

	for j := range 7 {
		for k := range 7 {
			angle := -2 * math.Pi * float64((j*k)%7) / 7
			oldRadix7MatrixFwd64[j*7+k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
			oldRadix7MatrixInv64[j*7+k] = complex(float32(math.Cos(angle)), float32(-math.Sin(angle)))
		}
	}

	for j := range 11 {
		for k := range 11 {
			angle := -2 * math.Pi * float64((j*k)%11) / 11
			oldRadix11MatrixFwd64[j*11+k] = complex(float32(math.Cos(angle)), float32(math.Sin(angle)))
			oldRadix11MatrixInv64[j*11+k] = complex(float32(math.Cos(angle)), float32(-math.Sin(angle)))
		}
	}
}

// oldButterfly5MatrixComplex64 is the pre-rewrite 5x5 DFT-matrix butterfly
// (20 complex multiplies), preserved here only as a benchmark baseline.
func oldButterfly5MatrixComplex64(
	a0, a1, a2, a3, a4 complex64, table *[4]complex64,
) (complex64, complex64, complex64, complex64, complex64) {
	w1, w2, w3, w4 := table[0], table[1], table[2], table[3]

	y0 := a0 + a1 + a2 + a3 + a4
	y1 := a0 + m.MulComplex64(a1, w1) + m.MulComplex64(a2, w2) + m.MulComplex64(a3, w3) + m.MulComplex64(a4, w4)
	y2 := a0 + m.MulComplex64(a1, w2) + m.MulComplex64(a2, w4) + m.MulComplex64(a3, w1) + m.MulComplex64(a4, w3)
	y3 := a0 + m.MulComplex64(a1, w3) + m.MulComplex64(a2, w1) + m.MulComplex64(a3, w4) + m.MulComplex64(a4, w2)
	y4 := a0 + m.MulComplex64(a1, w4) + m.MulComplex64(a2, w3) + m.MulComplex64(a3, w2) + m.MulComplex64(a4, w1)

	return y0, y1, y2, y3, y4
}

// oldButterfly7MatrixComplex64 is the pre-rewrite 7x7 DFT-matrix butterfly
// (42 complex multiplies), preserved here only as a benchmark baseline.
func oldButterfly7MatrixComplex64(a *[7]complex64, table *[49]complex64) [7]complex64 {
	var y [7]complex64

	sum := a[0]
	for k := 1; k < 7; k++ {
		sum += a[k]
	}

	y[0] = sum

	for j := 1; j < 7; j++ {
		acc := a[0]
		row := table[j*7 : j*7+7]

		for k := 1; k < 7; k++ {
			acc += m.MulComplex64(a[k], row[k])
		}

		y[j] = acc
	}

	return y
}

// oldButterfly11MatrixComplex64 is the pre-rewrite 11x11 DFT-matrix
// butterfly (100 complex multiplies), preserved here only as a benchmark
// baseline.
func oldButterfly11MatrixComplex64(a *[11]complex64, table *[121]complex64) [11]complex64 {
	var y [11]complex64

	sum := a[0]
	for k := 1; k < 11; k++ {
		sum += a[k]
	}

	y[0] = sum

	for j := 1; j < 11; j++ {
		acc := a[0]
		row := table[j*11 : j*11+11]

		for k := 1; k < 11; k++ {
			acc += m.MulComplex64(a[k], row[k])
		}

		y[j] = acc
	}

	return y
}

// --- radix 5 ---------------------------------------------------------------

func BenchmarkButterfly5ForwardComplex64Matrix(b *testing.B) {
	a0, a1, a2, a3, a4 := complex64(1+2i), complex64(-1+1i), complex64(0.5-1.5i), complex64(2+0i), complex64(4-2i)

	b.SetBytes(5 * 8)
	b.ReportAllocs()

	var y0, y1, y2, y3, y4 complex64

	for range b.N {
		y0, y1, y2, y3, y4 = oldButterfly5MatrixComplex64(a0, a1, a2, a3, a4, &oldRadix5MatrixFwd64)
	}

	sink5 = y0 + y1 + y2 + y3 + y4
}

func BenchmarkButterfly5ForwardComplex64PairForm(b *testing.B) {
	a0, a1, a2, a3, a4 := complex64(1+2i), complex64(-1+1i), complex64(0.5-1.5i), complex64(2+0i), complex64(4-2i)

	b.SetBytes(5 * 8)
	b.ReportAllocs()

	var y0, y1, y2, y3, y4 complex64

	for range b.N {
		y0, y1, y2, y3, y4 = Butterfly5ForwardComplex64(a0, a1, a2, a3, a4)
	}

	sink5 = y0 + y1 + y2 + y3 + y4
}

func BenchmarkButterfly5InverseComplex64Matrix(b *testing.B) {
	a0, a1, a2, a3, a4 := complex64(1+2i), complex64(-1+1i), complex64(0.5-1.5i), complex64(2+0i), complex64(4-2i)

	b.SetBytes(5 * 8)
	b.ReportAllocs()

	var y0, y1, y2, y3, y4 complex64

	for range b.N {
		y0, y1, y2, y3, y4 = oldButterfly5MatrixComplex64(a0, a1, a2, a3, a4, &oldRadix5MatrixInv64)
	}

	sink5 = y0 + y1 + y2 + y3 + y4
}

func BenchmarkButterfly5InverseComplex64PairForm(b *testing.B) {
	a0, a1, a2, a3, a4 := complex64(1+2i), complex64(-1+1i), complex64(0.5-1.5i), complex64(2+0i), complex64(4-2i)

	b.SetBytes(5 * 8)
	b.ReportAllocs()

	var y0, y1, y2, y3, y4 complex64

	for range b.N {
		y0, y1, y2, y3, y4 = Butterfly5InverseComplex64(a0, a1, a2, a3, a4)
	}

	sink5 = y0 + y1 + y2 + y3 + y4
}

func BenchmarkButterfly5ForwardComplex128PairForm(b *testing.B) {
	a0, a1, a2, a3, a4 := 1+2i, -1+1i, 0.5-1.5i, 2+0i, 4-2i

	b.SetBytes(5 * 16)
	b.ReportAllocs()

	var y0, y1, y2, y3, y4 complex128

	for range b.N {
		y0, y1, y2, y3, y4 = Butterfly5ForwardComplex128(a0, a1, a2, a3, a4)
	}

	sink5_128 = y0 + y1 + y2 + y3 + y4
}

func BenchmarkButterfly5InverseComplex128PairForm(b *testing.B) {
	a0, a1, a2, a3, a4 := 1+2i, -1+1i, 0.5-1.5i, 2+0i, 4-2i

	b.SetBytes(5 * 16)
	b.ReportAllocs()

	var y0, y1, y2, y3, y4 complex128

	for range b.N {
		y0, y1, y2, y3, y4 = Butterfly5InverseComplex128(a0, a1, a2, a3, a4)
	}

	sink5_128 = y0 + y1 + y2 + y3 + y4
}

// --- radix 7 ---------------------------------------------------------------

func benchRadix7Input() [7]complex64 {
	return [7]complex64{1 + 2i, -1 + 1i, 0.5 - 1.5i, 2 + 0i, 4 - 2i, -3 + 0.5i, 1 - 1i}
}

func benchRadix7Input128() [7]complex128 {
	return [7]complex128{1 + 2i, -1 + 1i, 0.5 - 1.5i, 2 + 0i, 4 - 2i, -3 + 0.5i, 1 - 1i}
}

func BenchmarkButterfly7ForwardComplex64Matrix(b *testing.B) {
	a := benchRadix7Input()

	b.SetBytes(7 * 8)
	b.ReportAllocs()

	var y [7]complex64

	for range b.N {
		y = oldButterfly7MatrixComplex64(&a, &oldRadix7MatrixFwd64)
	}

	sink7 = y[0]
}

func BenchmarkButterfly7ForwardComplex64PairForm(b *testing.B) {
	a := benchRadix7Input()

	b.SetBytes(7 * 8)
	b.ReportAllocs()

	for range b.N {
		aa := a
		Butterfly7ForwardComplex64(&aa)
		sink7 = aa[0]
	}
}

func BenchmarkButterfly7InverseComplex64Matrix(b *testing.B) {
	a := benchRadix7Input()

	b.SetBytes(7 * 8)
	b.ReportAllocs()

	var y [7]complex64

	for range b.N {
		y = oldButterfly7MatrixComplex64(&a, &oldRadix7MatrixInv64)
	}

	sink7 = y[0]
}

func BenchmarkButterfly7InverseComplex64PairForm(b *testing.B) {
	a := benchRadix7Input()

	b.SetBytes(7 * 8)
	b.ReportAllocs()

	for range b.N {
		aa := a
		Butterfly7InverseComplex64(&aa)
		sink7 = aa[0]
	}
}

func BenchmarkButterfly7ForwardComplex128PairForm(b *testing.B) {
	a := benchRadix7Input128()

	b.SetBytes(7 * 16)
	b.ReportAllocs()

	for range b.N {
		aa := a
		Butterfly7ForwardComplex128(&aa)
		sink7_128 = aa[0]
	}
}

func BenchmarkButterfly7InverseComplex128PairForm(b *testing.B) {
	a := benchRadix7Input128()

	b.SetBytes(7 * 16)
	b.ReportAllocs()

	for range b.N {
		aa := a
		Butterfly7InverseComplex128(&aa)
		sink7_128 = aa[0]
	}
}

// --- radix 11 ----------------------------------------------------------------

func benchRadix11Input() [11]complex64 {
	return [11]complex64{
		1 + 2i, -1 + 1i, 0.5 - 1.5i, 2 + 0i, 4 - 2i, -3 + 0.5i,
		1 - 1i, 0.25 + 0.75i, -2 + 2i, 3 - 3i, 0 + 1i,
	}
}

func benchRadix11Input128() [11]complex128 {
	return [11]complex128{
		1 + 2i, -1 + 1i, 0.5 - 1.5i, 2 + 0i, 4 - 2i, -3 + 0.5i,
		1 - 1i, 0.25 + 0.75i, -2 + 2i, 3 - 3i, 0 + 1i,
	}
}

func BenchmarkButterfly11ForwardComplex64Matrix(b *testing.B) {
	a := benchRadix11Input()

	b.SetBytes(11 * 8)
	b.ReportAllocs()

	var y [11]complex64

	for range b.N {
		y = oldButterfly11MatrixComplex64(&a, &oldRadix11MatrixFwd64)
	}

	sink11 = y[0]
}

func BenchmarkButterfly11ForwardComplex64PairForm(b *testing.B) {
	a := benchRadix11Input()

	b.SetBytes(11 * 8)
	b.ReportAllocs()

	for range b.N {
		aa := a
		Butterfly11ForwardComplex64(&aa)
		sink11 = aa[0]
	}
}

func BenchmarkButterfly11InverseComplex64Matrix(b *testing.B) {
	a := benchRadix11Input()

	b.SetBytes(11 * 8)
	b.ReportAllocs()

	var y [11]complex64

	for range b.N {
		y = oldButterfly11MatrixComplex64(&a, &oldRadix11MatrixInv64)
	}

	sink11 = y[0]
}

func BenchmarkButterfly11InverseComplex64PairForm(b *testing.B) {
	a := benchRadix11Input()

	b.SetBytes(11 * 8)
	b.ReportAllocs()

	for range b.N {
		aa := a
		Butterfly11InverseComplex64(&aa)
		sink11 = aa[0]
	}
}

func BenchmarkButterfly11ForwardComplex128PairForm(b *testing.B) {
	a := benchRadix11Input128()

	b.SetBytes(11 * 16)
	b.ReportAllocs()

	for range b.N {
		aa := a
		Butterfly11ForwardComplex128(&aa)
		sink11_128 = aa[0]
	}
}

func BenchmarkButterfly11InverseComplex128PairForm(b *testing.B) {
	a := benchRadix11Input128()

	b.SetBytes(11 * 16)
	b.ReportAllocs()

	for range b.N {
		aa := a
		Butterfly11InverseComplex128(&aa)
		sink11_128 = aa[0]
	}
}

//nolint:gochecknoglobals // benchmark result sinks, prevent dead-code elimination
var (
	sink5      complex64
	sink5_128  complex128
	sink7      complex64
	sink7_128  complex128
	sink11     complex64
	sink11_128 complex128
)
