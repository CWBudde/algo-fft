package algofft

import (
	"math/cmplx"
	"math/rand"
	"testing"

	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/planner"
)

func TestFastConvolutionLength(t *testing.T) {
	t.Parallel()

	tests := []struct {
		convLen int
		want    int
	}{
		{1, 1},       // power of two, kept
		{2, 2},       // power of two, kept
		{128, 128},   // power of two, kept
		{120, 120},   // 5-smooth, kept
		{1536, 1536}, // 5-smooth, kept
		{448, 448},   // 7-smooth, pow2 part 64: passes the mixed-radix gate, kept
		{14, 16},     // 7-smooth but gated out (pow2 part 2): padded
		{31, 32},     // prime: padded
		{131, 256},   // prime: padded (pad 2^8 is below the 3*2^(k-2) threshold)
		{257, 384},   // Rader-eligible prime: padded to 3*2^7
		{511, 512},   // 7*73: padded
		{1009, 1024}, // prime: padded
		{2047, 2048}, // 23*89: padded
		{4001, 4096}, // Rader-eligible prime: padded
	}

	for _, tt := range tests {
		if got := fastConvolutionLength(tt.convLen); got != tt.want {
			t.Errorf("fastConvolutionLength(%d) = %d, want %d", tt.convLen, got, tt.want)
		}
	}
}

// TestFastConvolutionLength_Invariants sweeps small lengths and pins the
// contract: the result is never below convLen, exactly-executable lengths are
// kept unchanged, and padded results always land on an exactly-executable
// (power-of-two or 5-smooth) size.
func TestFastConvolutionLength_Invariants(t *testing.T) {
	t.Parallel()

	for n := 1; n <= 5000; n++ {
		got := fastConvolutionLength(n)
		if got < n {
			t.Fatalf("fastConvolutionLength(%d) = %d < n", n, got)
		}

		exact := m.IsPowerOf2(n) || planner.MixedRadixEligible(n)
		if exact && got != n {
			t.Fatalf("fastConvolutionLength(%d) = %d, want unchanged for exact length", n, got)
		}

		if !exact && !m.IsPowerOf2(got) && !m.IsHighlyComposite(got) {
			t.Fatalf("fastConvolutionLength(%d) = %d is not a fast pad size", n, got)
		}
	}
}

// TestConvolveAwkwardLengthsMatchNaive covers output lengths that are not
// executable exactly (prime and Rader-eligible convLen), so the FFT runs at
// the padded fast size and the result is truncated back to convLen.
func TestConvolveAwkwardLengthsMatchNaive(t *testing.T) {
	t.Parallel()

	cases := []struct{ lenA, lenB int }{
		{16, 16},   // convLen 31, prime
		{66, 66},   // convLen 131, prime
		{129, 129}, // convLen 257, Rader-eligible prime
		{500, 510}, // convLen 1009, prime
	}

	for _, tc := range cases {
		rng := rand.New(rand.NewSource(int64(tc.lenA)))
		a := make([]complex64, tc.lenA)
		b := make([]complex64, tc.lenB)

		for i := range a {
			a[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
		}

		for i := range b {
			b[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
		}

		want := naiveConvolveComplex64(a, b)
		got := make([]complex64, len(want))

		if err := Convolve(got, a, b); err != nil {
			t.Fatalf("Convolve(%dx%d) returned error: %v", tc.lenA, tc.lenB, err)
		}

		for i := range want {
			assertApproxComplex64f(t, got[i], want[i], 2e-2, "convLen %d: got[%d]", tc.lenA+tc.lenB-1, i)
		}
	}
}

func TestConvolve128AwkwardLengthMatchesNaive(t *testing.T) {
	t.Parallel()

	const lenA, lenB = 500, 510 // convLen 1009, prime

	rng := rand.New(rand.NewSource(2))
	a := make([]complex128, lenA)
	b := make([]complex128, lenB)

	for i := range a {
		a[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	for i := range b {
		b[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	want := make([]complex128, lenA+lenB-1)
	for i := range a {
		for j := range b {
			want[i+j] += a[i] * b[j]
		}
	}

	got := make([]complex128, len(want))
	if err := Convolve128(got, a, b); err != nil {
		t.Fatalf("Convolve128() returned error: %v", err)
	}

	for i := range want {
		if cmplx.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("got[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestConvolverAwkwardLengthMatchesNaive(t *testing.T) {
	t.Parallel()

	const lenA, lenB = 129, 129 // convLen 257, Rader-eligible prime

	rng := rand.New(rand.NewSource(3))
	a := make([]complex64, lenA)
	b := make([]complex64, lenB)

	for i := range a {
		a[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	for i := range b {
		b[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	conv, err := NewConvolver[complex64](lenA, lenB)
	if err != nil {
		t.Fatalf("NewConvolver failed: %v", err)
	}

	want := naiveConvolveComplex64(a, b)
	got := make([]complex64, conv.Len())

	if err := conv.Convolve(got, a, b); err != nil {
		t.Fatalf("Convolve() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex64f(t, got[i], want[i], 2e-2, "got[%d]", i)
	}
}

func TestCrossCorrelateAwkwardLengthMatchesNaive(t *testing.T) {
	t.Parallel()

	const lenA, lenB = 66, 66 // convLen 131, prime

	rng := rand.New(rand.NewSource(4))
	a := make([]complex64, lenA)
	b := make([]complex64, lenB)

	for i := range a {
		a[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	for i := range b {
		b[i] = complex(rng.Float32()*2-1, rng.Float32()*2-1)
	}

	// Cross-correlation is convolution with the reversed conjugate of b.
	bRevConj := make([]complex64, lenB)
	for i := range b {
		bRevConj[i] = complex64(cmplx.Conj(complex128(b[lenB-1-i])))
	}

	want := naiveConvolveComplex64(a, bRevConj)
	got := make([]complex64, len(want))

	if err := CrossCorrelate(got, a, b); err != nil {
		t.Fatalf("CrossCorrelate() returned error: %v", err)
	}

	for i := range want {
		assertApproxComplex64f(t, got[i], want[i], 2e-2, "got[%d]", i)
	}
}

//nolint:paralleltest // AllocsPerRun panics during parallel tests
func TestConvolver_ZeroAllocSteadyStatePadded(t *testing.T) {
	const lenA, lenB = 500, 510 // convLen 1009 (prime), padded to 1024

	conv, err := NewConvolver[complex64](lenA, lenB)
	if err != nil {
		t.Fatalf("NewConvolver failed: %v", err)
	}

	a := make([]complex64, lenA)
	b := make([]complex64, lenB)
	dst := make([]complex64, conv.Len())

	a[0], b[0] = 1, 1

	// Warm up the scratch cache.
	if err := conv.Convolve(dst, a, b); err != nil {
		t.Fatalf("warm-up Convolve failed: %v", err)
	}

	allocs := testing.AllocsPerRun(10, func() {
		if err := conv.Convolve(dst, a, b); err != nil {
			t.Errorf("Convolve failed: %v", err)
		}
	})

	if allocs != 0 {
		t.Errorf("Convolver.Convolve allocates %.1f per call, want 0", allocs)
	}
}
