//go:build amd64 && !purego

package kernels

import (
	"math/rand"
	"strconv"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// TestRadix4AVX2FusedMatchesUnfused requires bit-identical output, not merely
// close output.
//
// Folding the radix-2 tail into the last radix-4 stage reorders nothing
// arithmetically: the same products, sums and differences happen in the same
// order, and only the store-then-reload between the stage and the tail
// disappears. A round trip through memory is exact for both float widths, so
// any difference at all means the fused loop is not computing the same thing --
// which an approximate comparison would wave through as rounding.
//
// The size list deliberately includes the 4^k shapes, where the fused path must
// not engage at all, and n = 32, where for complex64 the last radix-4 stage is
// the hoisted m = 4 one so the separate tail must still run.
func TestRadix4AVX2FusedMatchesUnfused(t *testing.T) {
	if !cpu.DetectFeatures().HasAVX2 {
		t.Skip("AVX2 not available")
	}

	sizes := []int{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768}

	for _, n := range sizes {
		for _, inverse := range []bool{false, true} {
			name := "forward"
			if inverse {
				name = "inverse"
			}

			t.Run(name+"/"+strconv.Itoa(n), func(t *testing.T) {
				checkFusedMatches128(t, n, inverse)
				checkFusedMatches64(t, n, inverse)
			})
		}
	}
}

func checkFusedMatches128(t *testing.T, n int, inverse bool) {
	t.Helper()

	rng := rand.New(rand.NewSource(11)) //nolint:gosec // deterministic test input
	src := make([]complex128, n)

	for i := range src {
		src[i] = complex(rng.Float64()*2-1, rng.Float64()*2-1)
	}

	twiddle := make([]complex128, twiddleSizeRadix4AVX2Complex128(n))
	prepareTwiddleRadix4AVX2Complex128(n, inverse, twiddle)

	plain := make([]complex128, n)
	fused := make([]complex128, n)
	scratch := make([]complex128, n)

	plainFn, fusedFn := forwardRadix4AVX2Complex128, forwardRadix4AVX2FusedComplex128
	if inverse {
		plainFn, fusedFn = inverseRadix4AVX2Complex128, inverseRadix4AVX2FusedComplex128
	}

	if !plainFn(plain, src, twiddle, scratch) || !fusedFn(fused, src, twiddle, scratch) {
		t.Fatalf("kernel declined n=%d", n)
	}

	for i := range plain {
		if plain[i] != fused[i] {
			t.Fatalf("n=%d complex128: index %d: separate tail %v, fused %v", n, i, plain[i], fused[i])
		}
	}
}

func checkFusedMatches64(t *testing.T, n int, inverse bool) {
	t.Helper()

	rng := rand.New(rand.NewSource(11)) //nolint:gosec // deterministic test input
	src := make([]complex64, n)

	for i := range src {
		src[i] = complex(float32(rng.Float64()*2-1), float32(rng.Float64()*2-1))
	}

	twiddle := make([]complex64, twiddleSizeRadix4AVX2(n))
	prepareTwiddleRadix4AVX2(n, inverse, twiddle)

	plain := make([]complex64, n)
	fused := make([]complex64, n)
	scratch := make([]complex64, n)

	plainFn, fusedFn := forwardRadix4AVX2Complex64, forwardRadix4AVX2FusedComplex64
	if inverse {
		plainFn, fusedFn = inverseRadix4AVX2Complex64, inverseRadix4AVX2FusedComplex64
	}

	if !plainFn(plain, src, twiddle, scratch) || !fusedFn(fused, src, twiddle, scratch) {
		t.Fatalf("kernel declined n=%d", n)
	}

	for i := range plain {
		if plain[i] != fused[i] {
			t.Fatalf("n=%d complex64: index %d: separate tail %v, fused %v", n, i, plain[i], fused[i])
		}
	}
}

// TestRadix4AVX2FusedSelection pins which sizes take the fused tail.
//
// The choice is empirical -- fusing holds eight live streams instead of four
// and only pays off where they stay small -- so nothing about the code implies
// it, and a regenerate that dropped or widened a row would otherwise be
// invisible. complex128 at n = 2048 is the row that matters most: fusing costs
// 11% exactly there, which is the size the fusion was written for.
func TestRadix4AVX2FusedSelection(t *testing.T) {
	features := cpu.DetectFeatures()
	if !features.HasAVX2 {
		t.Skip("AVX2 not available")
	}

	fused64 := map[int]bool{128: true, 2048: true}
	fused128 := map[int]bool{128: true}

	for _, n := range []int{128, 512, 2048, 8192, 32768} {
		entry := registry.Registry64.Lookup(n, features)
		if entry == nil {
			t.Fatalf("complex64 n=%d: no codelet", n)
		}

		if want := signatureFor(n, fused64[n]); entry.Signature != want {
			t.Errorf("complex64 n=%d: signature %q, want %q", n, entry.Signature, want)
		}

		entry128 := registry.Registry128.Lookup(n, features)
		if entry128 == nil {
			t.Fatalf("complex128 n=%d: no codelet", n)
		}

		if want := signatureFor(n, fused128[n]); entry128.Signature != want {
			t.Errorf("complex128 n=%d: signature %q, want %q", n, entry128.Signature, want)
		}
	}
}

func signatureFor(n int, fused bool) string {
	if fused {
		return "dit" + strconv.Itoa(n) + "_radix4fused_avx2"
	}

	return "dit" + strconv.Itoa(n) + "_radix4_avx2"
}
