package kernels

import (
	"math"
	"testing"

	"github.com/cwbudde/algo-fft/internal/reference"
)

// radix8TestSizes covers all three shapes the ladder supports: 8^k (8, 64,
// 512, 4096, 32768), 2*8^k (16, 128, 1024, 8192, 65536) and 4*8^k (32, 256,
// 2048, 16384).
//
//nolint:gochecknoglobals // shared table for the radix-8 test ladder
var radix8TestSizes = []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536}

// radix8ReferenceSizes are the sizes small enough to check against the O(n^2)
// naive DFT. Larger sizes are cross-checked against an independent radix-4
// kernel instead (see TestRadix8GenericMatchesRadix4Peer).
//
//nolint:gochecknoglobals // shared table for the radix-8 test ladder
var radix8ReferenceSizes = []int{8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}

// radix8Tol64 scales the absolute float32 tolerance with the spectrum
// magnitude, which grows like sqrt(n) for unit-variance input.
func radix8Tol64(n int) float64 {
	return 1e-4 * math.Sqrt(float64(n)/512)
}

// radix8Tol128 does the same for float64.
func radix8Tol128(n int) float64 {
	return 1e-10 * math.Sqrt(float64(n)/512)
}

// radix8Buffers allocates the four slices a complex64 codelet call needs, with
// the prepared twiddle table for the requested direction.
func radix8Buffers(n int, inverse bool) (dst, twiddle, scratch []complex64) {
	dst = make([]complex64, n)
	scratch = make([]complex64, n)
	twiddle = make([]complex64, twiddleSizeRadix8(n))
	prepareTwiddleRadix8Complex64(n, inverse, twiddle)

	return dst, twiddle, scratch
}

// radix8Buffers128 is the complex128 twin of radix8Buffers.
func radix8Buffers128(n int, inverse bool) (dst, twiddle, scratch []complex128) {
	dst = make([]complex128, n)
	scratch = make([]complex128, n)
	twiddle = make([]complex128, twiddleSizeRadix8(n))
	prepareTwiddleRadix8Complex128(n, inverse, twiddle)

	return dst, twiddle, scratch
}

// TestRadix8LimitShapes pins the shape classification, which decides both the
// permutation table and the tail stage.
func TestRadix8LimitShapes(t *testing.T) {
	t.Parallel()

	cases := []struct {
		n     int
		limit int
		tail  int
		ok    bool
	}{
		{8, 8, 1, true},
		{16, 8, 2, true},
		{32, 8, 4, true},
		{64, 64, 1, true},
		{128, 64, 2, true},
		{256, 64, 4, true},
		{512, 512, 1, true},
		{32768, 32768, 1, true},
		{65536, 32768, 2, true},
		// Out of range or not a power of two.
		{4, 0, 0, false},
		{0, 0, 0, false},
		{12, 0, 0, false},
		{1 << 17, 0, 0, false},
	}

	for _, tc := range cases {
		limit, tail, ok := radix8Limit(tc.n)
		if limit != tc.limit || tail != tc.tail || ok != tc.ok {
			t.Errorf("radix8Limit(%d) = (%d, %d, %v), want (%d, %d, %v)",
				tc.n, limit, tail, ok, tc.limit, tc.tail, tc.ok)
		}
	}
}

// TestRadix8TwiddlePlanesAreFullyUsed checks the n-8 accounting: the prepared
// table must carry data in exactly its first n-8 slots, no more and no fewer.
// A short table would leave a stage multiplying by zero, which round-trip
// tests can mask.
func TestRadix8TwiddlePlanesAreFullyUsed(t *testing.T) {
	t.Parallel()

	for _, n := range radix8TestSizes {
		size := twiddleSizeRadix8(n)
		if size != n+8 {
			t.Errorf("twiddleSizeRadix8(%d) = %d, want %d", n, size, n+8)

			continue
		}

		tw := make([]complex64, size)
		prepareTwiddleRadix8Complex64(n, false, tw)

		// Every used slot is a unit-magnitude root; every unused slot is zero.
		for i := range n - 8 {
			if mag := float64(real(tw[i]))*float64(real(tw[i])) +
				float64(imag(tw[i]))*float64(imag(tw[i])); math.Abs(mag-1) > 1e-6 {
				t.Errorf("n=%d: twiddle[%d] = %v, |w|^2 = %g, want a unit root", n, i, tw[i], mag)

				break
			}
		}

		for i := n - 8; i < size; i++ {
			if tw[i] != 0 {
				t.Errorf("n=%d: twiddle[%d] = %v, want 0 (past the n-8 planes)", n, i, tw[i])

				break
			}
		}
	}
}

// TestRadix8GenericForwardMatchesReference checks the forward transform
// against the naive DFT at every shape.
func TestRadix8GenericForwardMatchesReference(t *testing.T) {
	t.Parallel()
	skipNaiveReferenceIfSlow(t)

	for _, n := range radix8ReferenceSizes {
		t.Run(testName("fwd64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x1234+uint64(n))
			dst, twiddle, scratch := radix8Buffers(n, false)

			if !forwardRadix8Complex64(dst, src, twiddle, scratch) {
				t.Fatal("forwardRadix8Complex64 refused the call")
			}

			assertComplex64Close(t, dst, reference.NaiveDFT(src), radix8Tol64(n))
		})

		t.Run(testName("fwd128", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x1234+uint64(n))
			dst, twiddle, scratch := radix8Buffers128(n, false)

			if !forwardRadix8Complex128(dst, src, twiddle, scratch) {
				t.Fatal("forwardRadix8Complex128 refused the call")
			}

			assertComplex128Close(t, dst, reference.NaiveDFT128(src), radix8Tol128(n))
		})
	}
}

// TestRadix8GenericInverseMatchesReference checks the inverse transform,
// including its folded 1/n, against the naive inverse DFT.
func TestRadix8GenericInverseMatchesReference(t *testing.T) {
	t.Parallel()
	skipNaiveReferenceIfSlow(t)

	for _, n := range radix8ReferenceSizes {
		t.Run(testName("inv64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x9876+uint64(n))
			dst, twiddle, scratch := radix8Buffers(n, true)

			if !inverseRadix8Complex64(dst, src, twiddle, scratch) {
				t.Fatal("inverseRadix8Complex64 refused the call")
			}

			// The inverse output has magnitude ~1/sqrt(n), so the forward
			// tolerance is generous here rather than tight.
			assertComplex64Close(t, dst, reference.NaiveIDFT(src), radix8Tol64(n))
		})

		t.Run(testName("inv128", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x9876+uint64(n))
			dst, twiddle, scratch := radix8Buffers128(n, true)

			if !inverseRadix8Complex128(dst, src, twiddle, scratch) {
				t.Fatal("inverseRadix8Complex128 refused the call")
			}

			assertComplex128Close(t, dst, reference.NaiveIDFT128(src), radix8Tol128(n))
		})
	}
}

// TestRadix8GenericMatchesRadix4Peer cross-checks the sizes too large for the
// naive DFT against an independent radix-4 kernel that is itself
// reference-validated at that size.
func TestRadix8GenericMatchesRadix4Peer(t *testing.T) {
	t.Parallel()

	peers := map[int]func(dst, src, twiddle, scratch []complex64) bool{
		8192:  forwardDIT8192Radix4Then2Complex64,
		16384: forwardDIT16384Radix4Complex64,
		32768: forwardDIT32768Radix4Then2Complex64,
	}

	for n, peer := range peers {
		t.Run(testName("peer64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x5150+uint64(n))

			got, twiddle, scratch := radix8Buffers(n, false)
			if !forwardRadix8Complex64(got, src, twiddle, scratch) {
				t.Fatal("forwardRadix8Complex64 refused the call")
			}

			want := make([]complex64, n)
			plain := ComputeTwiddleFactors[complex64](n)

			if !peer(want, src, plain, make([]complex64, n)) {
				t.Fatal("the radix-4 peer refused the call")
			}

			// Two different stage orders over float32, so the tolerance is
			// looser than against a wide-accumulating reference.
			assertComplex64Close(t, got, want, 4*radix8Tol64(n))
		})
	}
}

// TestRadix8GenericRoundTrip covers every supported size, including the ones
// with no reference or peer check.
func TestRadix8GenericRoundTrip(t *testing.T) {
	t.Parallel()

	for _, n := range radix8TestSizes {
		t.Run(testName("rt64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0xABCD+uint64(n))

			fwd, fwdTw, scratch := radix8Buffers(n, false)
			if !forwardRadix8Complex64(fwd, src, fwdTw, scratch) {
				t.Fatal("forwardRadix8Complex64 refused the call")
			}

			back, invTw, _ := radix8Buffers(n, true)
			if !inverseRadix8Complex64(back, fwd, invTw, scratch) {
				t.Fatal("inverseRadix8Complex64 refused the call")
			}

			assertComplex64Close(t, back, src, radix8Tol64(n))
		})

		t.Run(testName("rt128", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0xABCD+uint64(n))

			fwd, fwdTw, scratch := radix8Buffers128(n, false)
			if !forwardRadix8Complex128(fwd, src, fwdTw, scratch) {
				t.Fatal("forwardRadix8Complex128 refused the call")
			}

			back, invTw, _ := radix8Buffers128(n, true)
			if !inverseRadix8Complex128(back, fwd, invTw, scratch) {
				t.Fatal("inverseRadix8Complex128 refused the call")
			}

			assertComplex128Close(t, back, src, radix8Tol128(n))
		})
	}
}

// TestRadix8GenericInPlace checks that dst == src produces the same answer as
// the out-of-place call. Stage 1 gathers, so this is the case that needs the
// scratch buffer.
func TestRadix8GenericInPlace(t *testing.T) {
	t.Parallel()

	for _, n := range radix8TestSizes {
		t.Run(testName("inplace64", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex64(n, 0x0F0F+uint64(n))

			want, twiddle, scratch := radix8Buffers(n, false)
			if !forwardRadix8Complex64(want, src, twiddle, scratch) {
				t.Fatal("forwardRadix8Complex64 refused the out-of-place call")
			}

			inPlace := make([]complex64, n)
			copy(inPlace, src)

			if !forwardRadix8Complex64(inPlace, inPlace, twiddle, scratch) {
				t.Fatal("forwardRadix8Complex64 refused the in-place call")
			}

			assertComplex64Close(t, inPlace, want, 0)
		})

		t.Run(testName("inplaceInv128", n), func(t *testing.T) {
			t.Parallel()

			src := randomComplex128(n, 0x0F0F+uint64(n))

			want, twiddle, scratch := radix8Buffers128(n, true)
			if !inverseRadix8Complex128(want, src, twiddle, scratch) {
				t.Fatal("inverseRadix8Complex128 refused the out-of-place call")
			}

			inPlace := make([]complex128, n)
			copy(inPlace, src)

			if !inverseRadix8Complex128(inPlace, inPlace, twiddle, scratch) {
				t.Fatal("inverseRadix8Complex128 refused the in-place call")
			}

			assertComplex128Close(t, inPlace, want, 0)
		})
	}
}

// TestRadix8LadderMatchesButterfly pins the unrolled butterflies in the ladder
// against butterfly8{Forward,Inverse}Complex64 in radix8.go, which is the
// readable statement of the same arithmetic and what the mixed-radix engine
// calls. At n = 8 the ladder is exactly one twiddle-free butterfly over the
// identity permutation, so the two must agree bit for bit.
func TestRadix8LadderMatchesButterfly(t *testing.T) {
	t.Parallel()

	const n = 8

	src := randomComplex64(n, 0xB0B0)

	got, twiddle, scratch := radix8Buffers(n, false)
	if !forwardRadix8Complex64(got, src, twiddle, scratch) {
		t.Fatal("forwardRadix8Complex64 refused the call")
	}

	var want [n]complex64

	want[0], want[1], want[2], want[3], want[4], want[5], want[6], want[7] =
		butterfly8ForwardComplex64(src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7])

	for i := range n {
		if got[i] != want[i] {
			t.Errorf("forward index %d: ladder %v, butterfly8ForwardComplex64 %v", i, got[i], want[i])
		}
	}

	// The inverse ladder folds 1/n into stage 1, so compare against the
	// butterfly applied to the pre-scaled input.
	invGot, invTwiddle, invScratch := radix8Buffers(n, true)
	if !inverseRadix8Complex64(invGot, src, invTwiddle, invScratch) {
		t.Fatal("inverseRadix8Complex64 refused the call")
	}

	var scaled [n]complex64

	const scale = float32(1) / float32(n)

	for i := range n {
		scaled[i] = complex(real(src[i])*scale, imag(src[i])*scale)
	}

	var invWant [n]complex64

	invWant[0], invWant[1], invWant[2], invWant[3], invWant[4], invWant[5], invWant[6], invWant[7] =
		butterfly8InverseComplex64(scaled[0], scaled[1], scaled[2], scaled[3],
			scaled[4], scaled[5], scaled[6], scaled[7])

	for i := range n {
		if invGot[i] != invWant[i] {
			t.Errorf("inverse index %d: ladder %v, butterfly8InverseComplex64 %v", i, invGot[i], invWant[i])
		}
	}
}

// TestRadix8GenericRejects checks that the kernel bails rather than
// transforming against the wrong factors. The twiddle case matters most: a
// plain length-n DIT table is n-8 elements longer than the planes this kernel
// reads, so without the n+8 guard it would pass a naive length check.
func TestRadix8GenericRejects(t *testing.T) {
	t.Parallel()

	const n = 512

	full := make([]complex64, twiddleSizeRadix8(n))
	prepareTwiddleRadix8Complex64(n, false, full)

	cases := []struct {
		name                     string
		dst, src, twiddle, scrat []complex64
	}{
		{"plain twiddle table", make([]complex64, n), make([]complex64, n), make([]complex64, n), make([]complex64, n)},
		{"short dst", make([]complex64, n-1), make([]complex64, n), full, make([]complex64, n)},
		{"short scratch", make([]complex64, n), make([]complex64, n), full, make([]complex64, n-1)},
		{"unsupported length", make([]complex64, 4), make([]complex64, 4), full, make([]complex64, 4)},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			if forwardRadix8Complex64(tc.dst, tc.src, tc.twiddle, tc.scrat) {
				t.Error("forwardRadix8Complex64 accepted a call it should have refused")
			}

			if inverseRadix8Complex64(tc.dst, tc.src, tc.twiddle, tc.scrat) {
				t.Error("inverseRadix8Complex64 accepted a call it should have refused")
			}
		})
	}
}

// TestRadix8GenericZeroAlloc locks the transform-time allocation count at zero
// once the memoised permutation table exists.
func TestRadix8GenericZeroAlloc(t *testing.T) {
	// Not parallel: AllocsPerRun panics if another test is running.
	for _, n := range []int{512, 1024, 2048} {
		src := randomComplex64(n, 0x2222+uint64(n))
		dst, twiddle, scratch := radix8Buffers(n, false)

		// Warm the memoised group-index table.
		if !forwardRadix8Complex64(dst, src, twiddle, scratch) {
			t.Fatalf("n=%d: forwardRadix8Complex64 refused the call", n)
		}

		allocs := testing.AllocsPerRun(20, func() {
			forwardRadix8Complex64(dst, src, twiddle, scratch)
		})
		if allocs != 0 {
			t.Errorf("n=%d: forward allocated %.1f times per run, want 0", n, allocs)
		}

		invDst, invTwiddle, invScratch := radix8Buffers(n, true)

		if !inverseRadix8Complex64(invDst, src, invTwiddle, invScratch) {
			t.Fatalf("n=%d: inverseRadix8Complex64 refused the call", n)
		}

		allocs = testing.AllocsPerRun(20, func() {
			inverseRadix8Complex64(invDst, src, invTwiddle, invScratch)
		})
		if allocs != 0 {
			t.Errorf("n=%d: inverse allocated %.1f times per run, want 0", n, allocs)
		}
	}
}
