//go:build amd64 && !purego

package kernels

import (
	"math"
	"math/cmplx"
	"testing"

	amd64 "github.com/cwbudde/algo-fft/internal/asm/amd64"
	mathpkg "github.com/cwbudde/algo-fft/internal/math"
)

// The three assembly helpers below are the pieces the size-384 codelet delegates
// to: the radix-3 column butterflies and the twiddle multiply that sits between
// them and the 128-point sub-FFTs. They are tested here against their scalar Go
// equivalents rather than only through the whole 384-point transform.
//
// That matters because these functions were declared and assembled for a long
// time without ever being called on the complex64 side, and the strided
// twiddle[2k] gather in ApplyTwiddle384Complex64Asm was wrong the whole time:
// it used VINSERTPS (which moves one float32) where it needed VMOVLHPS (which
// moves a whole complex64), so it clobbered the imaginary part of every even
// element and left two lanes undefined. A kernel-level round-trip test would
// have caught it, but only once the function was wired in — and until then no
// test in the tree touched it at all.

// The size-384 codelet's fixed geometry: 384 = 128 x 3.
const (
	dit384AsmHelperSize   = 384
	dit384AsmHelperStride = 128
)

// dit384AsmHelperInput builds a signal whose every element is distinct in both
// components, so a lane that is dropped, duplicated or left undefined shows up
// rather than coinciding with its neighbour.
func dit384AsmHelperInput() []complex64 {
	data := make([]complex64, dit384AsmHelperSize)
	for i := range data {
		f := float64(i)
		data[i] = complex(
			float32(math.Cos(0.37*f)+0.11*f),
			float32(math.Sin(0.53*f)-0.07*f),
		)
	}

	return data
}

// TestApplyTwiddle384Complex64AsmMatchesScalar pins the twiddle multiply the
// size-384 codelet applies between the radix-3 stage and the sub-FFTs:
//
//	data[128+k] *= twiddle[k]    for k = 0..127
//	data[256+k] *= twiddle[2*k]  for k = 0..127
//
// The second loop is the strided one, and it is the one that was broken.
func TestApplyTwiddle384Complex64AsmMatchesScalar(t *testing.T) {
	t.Parallel()
	requireAVX2(t)

	const (
		n      = dit384AsmHelperSize
		stride = dit384AsmHelperStride
	)

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)

	got := dit384AsmHelperInput()
	want := dit384AsmHelperInput()

	amd64.ApplyTwiddle384Complex64Asm(got, twiddle)

	for k := range stride {
		want[stride+k] = mathpkg.MulComplex64(want[stride+k], twiddle[k])
		want[2*stride+k] = mathpkg.MulComplex64(want[2*stride+k], twiddle[2*k])
	}

	// The asm fuses the complex multiply with VFMADDSUB231PS, so it rounds once
	// where the scalar helper rounds twice; the bound is relative and loose
	// enough for that, and far tighter than any lane-selection error could hide
	// under. Both halves are reported, since only the strided one was suspect.
	var bad int

	for i := range n {
		tol := 1e-6 * (1 + cmplx.Abs(complex128(want[i])))
		if diff := cmplx.Abs(complex128(got[i] - want[i])); diff > tol {
			bad++
			if bad <= 4 {
				t.Errorf("element %d: got %v, want %v (diff %.3e > %.3e)",
					i, got[i], want[i], diff, tol)
			}
		}
	}

	if bad > 0 {
		t.Fatalf("%d of %d elements wrong", bad, n)
	}
}

// TestApplyConjTwiddle384Complex64AsmMatchesScalar is the inverse-direction
// twin: same strided access pattern, conjugated twiddles.
func TestApplyConjTwiddle384Complex64AsmMatchesScalar(t *testing.T) {
	t.Parallel()
	requireAVX2(t)

	const (
		n      = dit384AsmHelperSize
		stride = dit384AsmHelperStride
	)

	twiddle := mathpkg.ComputeTwiddleFactors[complex64](n)

	got := dit384AsmHelperInput()
	want := dit384AsmHelperInput()

	amd64.ApplyConjTwiddle384Complex64Asm(got, twiddle)

	for k := range stride {
		want[stride+k] = mathpkg.MulComplex64(want[stride+k], mathpkg.Conj(twiddle[k]))
		want[2*stride+k] = mathpkg.MulComplex64(want[2*stride+k], mathpkg.Conj(twiddle[2*k]))
	}

	var bad int

	for i := range n {
		tol := 1e-6 * (1 + cmplx.Abs(complex128(want[i])))
		if diff := cmplx.Abs(complex128(got[i] - want[i])); diff > tol {
			bad++
			if bad <= 4 {
				t.Errorf("element %d: got %v, want %v (diff %.3e > %.3e)",
					i, got[i], want[i], diff, tol)
			}
		}
	}

	if bad > 0 {
		t.Fatalf("%d of %d elements wrong", bad, n)
	}
}

// TestApplyConjTwiddle384Complex128AsmMatchesScalar is the complex128 twin. The
// two precisions use different shuffle idioms (VPERMPD over doubles against
// VMOVSLDUP/VMOVSHDUP over floats) and different sign masks, so neither
// inherits the other's coverage.
func TestApplyConjTwiddle384Complex128AsmMatchesScalar(t *testing.T) {
	t.Parallel()
	requireAVX2(t)

	const (
		n      = dit384AsmHelperSize
		stride = dit384AsmHelperStride
	)

	twiddle := mathpkg.ComputeTwiddleFactors[complex128](n)

	got := make([]complex128, n)
	want := make([]complex128, n)

	for i, v := range dit384AsmHelperInput() {
		got[i] = complex128(v)
		want[i] = complex128(v)
	}

	amd64.ApplyConjTwiddle384Complex128Asm(got, twiddle)

	for k := range stride {
		want[stride+k] *= mathpkg.Conj(twiddle[k])
		want[2*stride+k] *= mathpkg.Conj(twiddle[2*k])
	}

	var bad int

	for i := range n {
		tol := 1e-14 * (1 + cmplx.Abs(want[i]))
		if diff := cmplx.Abs(got[i] - want[i]); diff > tol {
			bad++
			if bad <= 4 {
				t.Errorf("element %d: got %v, want %v (diff %.3e > %.3e)",
					i, got[i], want[i], diff, tol)
			}
		}
	}

	if bad > 0 {
		t.Fatalf("%d of %d elements wrong", bad, n)
	}
}

// TestRadix3Butterflies384Complex64AsmMatchesScalar covers both directions of
// the radix-3 column stage against the scalar butterflies the codelet used
// before the assembly was wired in.
func TestRadix3Butterflies384Complex64AsmMatchesScalar(t *testing.T) {
	t.Parallel()
	requireAVX2(t)

	const (
		n      = dit384AsmHelperSize
		stride = dit384AsmHelperStride
	)

	for _, tc := range []struct {
		name   string
		asm    func([]complex64)
		scalar func(a0, a1, a2 complex64) (complex64, complex64, complex64)
	}{
		{"forward", amd64.Radix3Butterflies384ForwardComplex64Asm, butterfly3ForwardComplex64},
		{"inverse", amd64.Radix3Butterflies384InverseComplex64Asm, butterfly3InverseComplex64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := dit384AsmHelperInput()
			want := dit384AsmHelperInput()

			tc.asm(got)

			for k := range stride {
				y0, y1, y2 := tc.scalar(want[k], want[stride+k], want[2*stride+k])
				want[k], want[stride+k], want[2*stride+k] = y0, y1, y2
			}

			for i := range n {
				tol := 1e-6 * (1 + cmplx.Abs(complex128(want[i])))
				if diff := cmplx.Abs(complex128(got[i] - want[i])); diff > tol {
					t.Fatalf("element %d: got %v, want %v (diff %.3e > %.3e)",
						i, got[i], want[i], diff, tol)
				}
			}
		})
	}
}
