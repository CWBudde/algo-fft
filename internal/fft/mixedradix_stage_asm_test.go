package fft

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

// The fused assembly stage kernels (mixedradix_stage_asm_amd64.go) replace
// both passes of mixedRadixStageComplex64/128, so nothing in the two-pass Go
// code can serve as their reference. These tests compute the stage from its
// definition instead — twiddle multiply, then the direct radix-r DFT
// butterfly — and compare element by element.
//
// Spans are chosen to straddle the vector/tail boundary in both precisions:
// the complex64 kernels cover whole 4-element blocks of k and the complex128
// kernels whole 2-element blocks, so a span of, say, 19 leaves three values
// of k for the Go tail in one precision and one in the other. A kernel that
// silently processed only its vector part would pass a span-16 test.

// stageReference64 computes one radix-r stage from its definition.
func stageReference64(input, table []complex64, span, radix int, inverse bool) []complex64 {
	n := span * radix
	out := make([]complex64, n)

	for k := range span {
		a := make([]complex128, radix)
		for j := range radix {
			v := complex128(input[j*span+k])
			if j > 0 {
				v *= complex128(table[j*span+k])
			}

			a[j] = v
		}

		for p := range radix {
			var acc complex128

			for j := range radix {
				angle := -2 * math.Pi * float64(p*j) / float64(radix)
				if inverse {
					angle = -angle
				}

				acc += a[j] * cmplx.Exp(complex(0, angle))
			}

			out[p*span+k] = complex64(acc)
		}
	}

	return out
}

// stageInputs64 builds a random stage input and a stage twiddle table of the
// shape the recursion produces (row 0 all ones, row j entry k = W_n^(j*k)).
func stageInputs64(span, radix int, inverse bool) (input, table []complex64) {
	n := span * radix
	rng := rand.New(rand.NewSource(int64(n*7 + radix))) //nolint:gosec

	input = make([]complex64, n)
	for i := range input {
		input[i] = complex(float32(rng.NormFloat64()), float32(rng.NormFloat64()))
	}

	table = make([]complex64, n)
	for k := range span {
		table[k] = 1
	}

	for j := 1; j < radix; j++ {
		for k := range span {
			angle := -2 * math.Pi * float64(j*k) / float64(n)
			if inverse {
				angle = -angle
			}

			table[j*span+k] = complex64(cmplx.Exp(complex(0, angle)))
		}
	}

	return input, table
}

func TestMixedRadixStageAsmComplex64(t *testing.T) {
	for _, radix := range []int{3, 5, 7} {
		for _, span := range []int{4, 7, 16, 19, 64, 65, 253} {
			for _, inverse := range []bool{false, true} {
				input, table := stageInputs64(span, radix, inverse)
				want := stageReference64(input, table, span, radix, inverse)

				n := span * radix
				got := make([]complex64, n)

				if !mixedRadixStageAsm64(got, input, table, n, span, radix, inverse) {
					t.Skipf("no fused kernel for radix %d span %d on this machine", radix, span)
				}

				for i := range want {
					if d := cmplx.Abs(complex128(got[i]) - complex128(want[i])); d > 1e-4 {
						t.Fatalf("radix=%d span=%d inverse=%v: index %d (k=%d row=%d) = %v, want %v (|d|=%g)",
							radix, span, inverse, i, i%span, i/span, got[i], want[i], d)
					}
				}
			}
		}
	}
}

// TestMixedRadixStageAsmAliasComplex64 covers the case the recursion actually
// hits most often: dst and input are the same slice. Every k must read all r
// of its rows before it writes any of them.
func TestMixedRadixStageAsmAliasComplex64(t *testing.T) {
	for _, radix := range []int{3, 5, 7} {
		for _, span := range []int{19, 64} {
			input, table := stageInputs64(span, radix, false)
			want := stageReference64(input, table, span, radix, false)

			n := span * radix
			inPlace := make([]complex64, n)
			copy(inPlace, input)

			if !mixedRadixStageAsm64(inPlace, inPlace, table, n, span, radix, false) {
				t.Skipf("no fused kernel for radix %d span %d on this machine", radix, span)
			}

			for i := range want {
				if d := cmplx.Abs(complex128(inPlace[i]) - complex128(want[i])); d > 1e-4 {
					t.Fatalf("radix=%d span=%d aliased: index %d = %v, want %v", radix, span, i, inPlace[i], want[i])
				}
			}
		}
	}
}

// stageReference128 is the complex128 counterpart of stageReference64.
func stageReference128(input, table []complex128, span, radix int, inverse bool) []complex128 {
	n := span * radix
	out := make([]complex128, n)

	for k := range span {
		a := make([]complex128, radix)
		for j := range radix {
			v := input[j*span+k]
			if j > 0 {
				v *= table[j*span+k]
			}

			a[j] = v
		}

		for p := range radix {
			var acc complex128

			for j := range radix {
				angle := -2 * math.Pi * float64(p*j) / float64(radix)
				if inverse {
					angle = -angle
				}

				acc += a[j] * cmplx.Exp(complex(0, angle))
			}

			out[p*span+k] = acc
		}
	}

	return out
}

func stageInputs128(span, radix int, inverse bool) (input, table []complex128) {
	n := span * radix
	rng := rand.New(rand.NewSource(int64(n*7 + radix))) //nolint:gosec

	input = make([]complex128, n)
	for i := range input {
		input[i] = complex(rng.NormFloat64(), rng.NormFloat64())
	}

	table = make([]complex128, n)
	for k := range span {
		table[k] = 1
	}

	for j := 1; j < radix; j++ {
		for k := range span {
			angle := -2 * math.Pi * float64(j*k) / float64(n)
			if inverse {
				angle = -angle
			}

			table[j*span+k] = cmplx.Exp(complex(0, angle))
		}
	}

	return input, table
}

func TestMixedRadixStageAsmComplex128(t *testing.T) {
	for _, radix := range []int{3, 5, 7} {
		for _, span := range []int{4, 7, 16, 19, 64, 65, 253} {
			for _, inverse := range []bool{false, true} {
				input, table := stageInputs128(span, radix, inverse)
				want := stageReference128(input, table, span, radix, inverse)

				n := span * radix
				got := make([]complex128, n)

				if !mixedRadixStageAsm128(got, input, table, n, span, radix, inverse) {
					t.Skipf("no fused kernel for radix %d span %d on this machine", radix, span)
				}

				for i := range want {
					if d := cmplx.Abs(got[i] - want[i]); d > 1e-9 {
						t.Fatalf("radix=%d span=%d inverse=%v: index %d (k=%d row=%d) = %v, want %v (|d|=%g)",
							radix, span, inverse, i, i%span, i/span, got[i], want[i], d)
					}
				}
			}
		}
	}
}

func TestMixedRadixStageAsmAliasComplex128(t *testing.T) {
	for _, radix := range []int{3, 5, 7} {
		for _, span := range []int{19, 64} {
			input, table := stageInputs128(span, radix, false)
			want := stageReference128(input, table, span, radix, false)

			n := span * radix
			inPlace := make([]complex128, n)
			copy(inPlace, input)

			if !mixedRadixStageAsm128(inPlace, inPlace, table, n, span, radix, false) {
				t.Skipf("no fused kernel for radix %d span %d on this machine", radix, span)
			}

			for i := range want {
				if d := cmplx.Abs(inPlace[i] - want[i]); d > 1e-9 {
					t.Fatalf("radix=%d span=%d aliased: index %d = %v, want %v", radix, span, i, inPlace[i], want[i])
				}
			}
		}
	}
}

// TestMixedRadixStageGoRadix7 covers the radix-7 arm of the two-pass Go stage.
// The gate in mixedRadixStageVectorizable admits radix 7 only where the fused
// kernel runs, so this arm is a fallback rather than a normal route — but it
// is reachable (a span the vector loop cannot cover, forced CPU features in a
// test) and used to be a panic, so it needs a reference check of its own.
func TestMixedRadixStageGoRadix7(t *testing.T) {
	const (
		radix = 7
		span  = 2 // below mixedRadixStageAsmMinSpan: the fused kernel declines
	)

	for _, inverse := range []bool{false, true} {
		if mixedRadixStageAsm64(nil, nil, nil, 0, span, radix, inverse) {
			t.Fatalf("span %d unexpectedly reached the fused kernel", span)
		}

		input, table := stageInputs64(span, radix, inverse)
		want := stageReference64(input, table, span, radix, inverse)

		n := span * radix
		got := make([]complex64, n)
		mixedRadixStageComplex64(got, input, table, n, span, radix, inverse)

		for i := range want {
			if d := cmplx.Abs(complex128(got[i]) - complex128(want[i])); d > 1e-4 {
				t.Fatalf("complex64 inverse=%v: index %d = %v, want %v", inverse, i, got[i], want[i])
			}
		}

		input128, table128 := stageInputs128(span, radix, inverse)
		want128 := stageReference128(input128, table128, span, radix, inverse)
		got128 := make([]complex128, n)
		mixedRadixStageComplex128(got128, input128, table128, n, span, radix, inverse)

		for i := range want128 {
			if d := cmplx.Abs(got128[i] - want128[i]); d > 1e-9 {
				t.Fatalf("complex128 inverse=%v: index %d = %v, want %v", inverse, i, got128[i], want128[i])
			}
		}
	}
}
