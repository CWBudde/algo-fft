package kernels

import (
	"math"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// radix16TestSizes covers all four shapes the ladder supports: 16^k (16, 256,
// 4096, 65536), 2*16^k (32, 512, 8192), 4*16^k (64, 1024, 16384) and 8*16^k
// (128, 2048, 32768).
//
//nolint:gochecknoglobals // shared table for the radix-16 test ladder
var radix16TestSizes = []int{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}

// radix16ReferenceSizes are the sizes small enough to check against the O(n^2)
// naive DFT. Larger sizes are cross-checked against the independent radix-8
// ladder instead (see TestRadix16GenericMatchesRadix8Peer).
//
//nolint:gochecknoglobals // shared table for the radix-16 test ladder
var radix16ReferenceSizes = []int{16, 32, 64, 128, 256, 512, 1024, 2048, 4096}

// radix16Tol64 scales the absolute float32 tolerance with the spectrum
// magnitude, which grows like sqrt(n) for unit-variance input.
func radix16Tol64(n int) float64 {
	return 1e-4 * math.Sqrt(float64(n)/512)
}

// radix16Tol128 does the same for float64.
func radix16Tol128(n int) float64 {
	return 1e-10 * math.Sqrt(float64(n)/512)
}

// radix16Buffers allocates the slices a complex64 codelet call needs, with the
// prepared twiddle table for the requested direction.
func radix16Buffers(n int, inverse bool) (dst, twiddle, scratch []complex64) {
	dst = make([]complex64, n)
	scratch = make([]complex64, n)
	twiddle = make([]complex64, twiddleSizeRadix16(n))
	prepareTwiddleRadix16Complex64(n, inverse, twiddle)

	return dst, twiddle, scratch
}

// radix16Buffers128 is the complex128 twin of radix16Buffers.
func radix16Buffers128(n int, inverse bool) (dst, twiddle, scratch []complex128) {
	dst = make([]complex128, n)
	scratch = make([]complex128, n)
	twiddle = make([]complex128, twiddleSizeRadix16(n))
	prepareTwiddleRadix16Complex128(n, inverse, twiddle)

	return dst, twiddle, scratch
}

// TestRadix16LimitShapes pins the shape classification, which decides both the
// permutation table and the tail stage. Getting it wrong selects a valid
// permutation for the wrong shape -- a plausible-but-wrong spectrum rather than
// an obvious crash.
func TestRadix16LimitShapes(t *testing.T) {
	t.Parallel()

	cases := []struct {
		n     int
		limit int
		tail  int
		ok    bool
	}{
		{16, 16, 1, true},
		{32, 16, 2, true},
		{64, 16, 4, true},
		{128, 16, 8, true},
		{256, 256, 1, true},
		{512, 256, 2, true},
		{1024, 256, 4, true},
		{2048, 256, 8, true},
		{4096, 4096, 1, true},
		{65536, 65536, 1, true},
		// Out of range or not a power of two.
		{8, 0, 0, false},
		{0, 0, 0, false},
		{24, 0, 0, false},
		{1 << 17, 0, 0, false},
	}

	for _, tc := range cases {
		limit, tail, ok := radix16Limit(tc.n)
		if limit != tc.limit || tail != tc.tail || ok != tc.ok {
			t.Errorf("radix16Limit(%d) = (%d, %d, %v), want (%d, %d, %v)",
				tc.n, limit, tail, ok, tc.limit, tc.tail, tc.ok)
		}
	}
}

// TestRadix16TwiddlePlanesAreFullyUsed checks the n-16 accounting: the prepared
// table must carry data in exactly its first n-16 slots, no more and no fewer.
// A short table leaves a stage multiplying by zero, which a round-trip test can
// mask.
func TestRadix16TwiddlePlanesAreFullyUsed(t *testing.T) {
	t.Parallel()

	for _, n := range radix16TestSizes {
		size := twiddleSizeRadix16(n)
		if size != n+16 {
			t.Errorf("twiddleSizeRadix16(%d) = %d, want %d", n, size, n+16)

			continue
		}

		tw := make([]complex64, size)
		prepareTwiddleRadix16Complex64(n, false, tw)

		// Every used slot is a unit-magnitude root; every unused slot is zero.
		for i := range n - 16 {
			if mag := float64(real(tw[i]))*float64(real(tw[i])) +
				float64(imag(tw[i]))*float64(imag(tw[i])); math.Abs(mag-1) > 1e-6 {
				t.Errorf("n=%d: twiddle[%d] = %v, |w|^2 = %g, want a unit root", n, i, tw[i], mag)

				break
			}
		}

		for i := n - 16; i < size; i++ {
			if tw[i] != 0 {
				t.Errorf("n=%d: twiddle[%d] = %v, want 0 (past the n-16 planes)", n, i, tw[i])

				break
			}
		}
	}
}

// TestRadix16RejectsWrongTwiddleLength is the reason the table is padded to
// n+16: a caller handing the ladder a plain length-n DIT table must be refused
// rather than silently transformed against the wrong factors.
func TestRadix16RejectsWrongTwiddleLength(t *testing.T) {
	t.Parallel()

	const n = 256

	dst := make([]complex64, n)
	src := make([]complex64, n)
	scratch := make([]complex64, n)

	for _, bad := range []int{0, n - 16, n, n + 15} {
		if forwardRadix16Complex64(dst, src, make([]complex64, bad), scratch) {
			t.Errorf("forwardRadix16Complex64 accepted a twiddle table of length %d", bad)
		}

		if inverseRadix16Complex64(dst, src, make([]complex64, bad), scratch) {
			t.Errorf("inverseRadix16Complex64 accepted a twiddle table of length %d", bad)
		}
	}
}

// TestRadix16GenericForwardMatchesReference is the primary correctness gate:
// every bin against the O(n^2) definition, per direction. Impulse, Parseval and
// linearity all pass over a wrong spectrum (docs/TESTING.md records four that
// did), so this has to be bin-for-bin against random input.
func TestRadix16GenericForwardMatchesReference(t *testing.T) {
	t.Parallel()
	skipNaiveReferenceIfSlow(t)

	for _, n := range radix16ReferenceSizes {
		t.Run(testName("fwd64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x1234+uint64(n))
			dst, twiddle, scratch := radix16Buffers(n, false)

			if !forwardRadix16Complex64(dst, src, twiddle, scratch) {
				t.Fatal("forwardRadix16Complex64 refused the call")
			}

			assertComplex64Close(t, dst, reference.NaiveDFT(src), radix16Tol64(n))
		})

		t.Run(testName("fwd128", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x1234+uint64(n))
			dst, twiddle, scratch := radix16Buffers128(n, false)

			if !forwardRadix16Complex128(dst, src, twiddle, scratch) {
				t.Fatal("forwardRadix16Complex128 refused the call")
			}

			assertComplex128Close(t, dst, reference.NaiveDFT128(src), radix16Tol128(n))
		})
	}
}

// TestRadix16GenericInverseMatchesReference checks the inverse independently of
// the forward, so a matched pair of sign errors cannot hide behind a round-trip.
func TestRadix16GenericInverseMatchesReference(t *testing.T) {
	t.Parallel()
	skipNaiveReferenceIfSlow(t)

	for _, n := range radix16ReferenceSizes {
		t.Run(testName("inv64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x9876+uint64(n))
			dst, twiddle, scratch := radix16Buffers(n, true)

			if !inverseRadix16Complex64(dst, src, twiddle, scratch) {
				t.Fatal("inverseRadix16Complex64 refused the call")
			}

			assertComplex64Close(t, dst, reference.NaiveIDFT(src), radix16Tol64(n))
		})

		t.Run(testName("inv128", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x9876+uint64(n))
			dst, twiddle, scratch := radix16Buffers128(n, true)

			if !inverseRadix16Complex128(dst, src, twiddle, scratch) {
				t.Fatal("inverseRadix16Complex128 refused the call")
			}

			assertComplex128Close(t, dst, reference.NaiveIDFT128(src), radix16Tol128(n))
		})
	}
}

// TestRadix16GenericMatchesRadix8Peer covers the sizes too large for the naive
// reference by cross-checking against the radix-8 ladder, which is an
// independent implementation with a different permutation, a different stage
// count and a different butterfly. Agreement there is strong evidence for both.
func TestRadix16GenericMatchesRadix8Peer(t *testing.T) {
	t.Parallel()

	for _, n := range []int{8192, 16384, 32768, 65536} {
		t.Run(testName("peer64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x5150+uint64(n))

			got, tw16, sc16 := radix16Buffers(n, false)
			if !forwardRadix16Complex64(got, src, tw16, sc16) {
				t.Fatal("forwardRadix16Complex64 refused the call")
			}

			want, tw8, sc8 := radix8Buffers(n, false)
			if !forwardRadix8Complex64(want, src, tw8, sc8) {
				t.Fatal("forwardRadix8Complex64 refused the call")
			}

			// Two different stage orders over float32, so the tolerance is
			// looser than against a wide-accumulating reference: this catches
			// structural errors, not last-bit rounding.
			assertComplex64Close(t, got, want, 4*radix16Tol64(n))
		})
	}
}

// TestRadix16GenericRoundTrip covers every supported size, including the ones
// no reference or peer check reaches.
func TestRadix16GenericRoundTrip(t *testing.T) {
	t.Parallel()

	for _, n := range radix16TestSizes {
		t.Run(testName("rt64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0xABCD+uint64(n))

			fwd, fwdTw, scratch := radix16Buffers(n, false)
			if !forwardRadix16Complex64(fwd, src, fwdTw, scratch) {
				t.Fatal("forwardRadix16Complex64 refused the call")
			}

			back, invTw, _ := radix16Buffers(n, true)
			if !inverseRadix16Complex64(back, fwd, invTw, scratch) {
				t.Fatal("inverseRadix16Complex64 refused the call")
			}

			assertComplex64Close(t, back, src, radix16Tol64(n))
		})

		t.Run(testName("rt128", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0xABCD+uint64(n))

			fwd, fwdTw, scratch := radix16Buffers128(n, false)
			if !forwardRadix16Complex128(fwd, src, fwdTw, scratch) {
				t.Fatal("forwardRadix16Complex128 refused the call")
			}

			back, invTw, _ := radix16Buffers128(n, true)
			if !inverseRadix16Complex128(back, fwd, invTw, scratch) {
				t.Fatal("inverseRadix16Complex128 refused the call")
			}

			assertComplex128Close(t, back, src, radix16Tol128(n))
		})
	}
}

// TestRadix16GenericInPlace covers the dst == src path, where the prologue has
// to redirect stage 1 into scratch because the gather cannot write over its own
// source. Aliasing bugs corrupt only some shapes, so every size is run.
func TestRadix16GenericInPlace(t *testing.T) {
	t.Parallel()

	for _, n := range radix16TestSizes {
		t.Run(testName("inplace64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0xFEED+uint64(n))

			want, tw, sc := radix16Buffers(n, false)
			if !forwardRadix16Complex64(want, src, tw, sc) {
				t.Fatal("out-of-place forward refused the call")
			}

			inPlace := make([]complex64, n)
			copy(inPlace, src)

			if !forwardRadix16Complex64(inPlace, inPlace, tw, make([]complex64, n)) {
				t.Fatal("in-place forward refused the call")
			}

			// Same arithmetic in the same order, so this is exact.
			for i := range inPlace {
				if inPlace[i] != want[i] {
					t.Fatalf("bin %d: in-place %v vs out-of-place %v", i, inPlace[i], want[i])
				}
			}
		})
	}
}

// TestRadix16LadderMatchesButterfly pins the unrolled butterfly copies in
// radix16_generic.go against the readable original in radix16.go.
//
// The ladder cannot call butterfly16ForwardComplex64: it costs 1085 inline
// units against a budget of 80, and paying calls per butterfly would have made
// the go/no-go measurement a measurement of Go's calling convention instead of
// the radix. That leaves four hand-unrolled copies, and this is what stops them
// drifting from the definition the reference tests validated.
//
// At n = 16 the ladder is exactly one twiddle-free stage-1 butterfly with no
// tail, so the forward comparison is exact rather than approximate.
func TestRadix16LadderMatchesButterfly(t *testing.T) {
	t.Parallel()

	const n = 16

	for seed := range uint64(16) {
		block := randomBlock16(seed + 4000)

		src := make([]complex64, n)
		copy(src, block[:])

		fwd := block
		butterfly16ForwardComplex64(&fwd)

		got, tw, sc := radix16Buffers(n, false)
		if !forwardRadix16Complex64(got, src, tw, sc) {
			t.Fatal("forwardRadix16Complex64 refused the call")
		}

		for i := range got {
			if got[i] != fwd[i] {
				t.Fatalf("seed %d, forward bin %d: ladder %v vs butterfly %v", seed, i, got[i], fwd[i])
			}
		}

		// The ladder folds 1/n into the inverse; the butterfly does not, so
		// this side carries the rounding of one extra multiply.
		inv := block
		butterfly16InverseComplex64(&inv)

		wantInv := make([]complex64, n)
		for i := range inv {
			wantInv[i] = complex(real(inv[i])/n, imag(inv[i])/n)
		}

		gotInv, twI, scI := radix16Buffers(n, true)
		if !inverseRadix16Complex64(gotInv, src, twI, scI) {
			t.Fatal("inverseRadix16Complex64 refused the call")
		}

		assertComplex64Close(t, gotInv, wantInv, 1e-6)
	}
}
