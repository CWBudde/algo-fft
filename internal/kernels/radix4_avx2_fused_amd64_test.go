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

// TestRadix4AVX2FusedSelection pins which codelet actually wins each of the
// n = 2*4^k sizes, fused tail included.
//
// Every one of these choices is empirical and nothing about the code implies
// it, so a regenerate that dropped or widened a row would otherwise be
// invisible. Two independent decisions are pinned here:
//
//   - Fused against separate radix-4 tail. Fusing holds eight live streams
//     instead of four and only pays off where they stay small. complex128 at
//     n = 2048 was the row that mattered most -- fusing costs 11% exactly
//     there, the size the fusion was written for -- though radix-8 has since
//     taken that row outright.
//   - Radix-8 against radix-4. The size-generic radix-8 ladder took 512 (both
//     precisions), 2048 (both) and 32768 complex128 in the 2026-07-30 sweep,
//     and lost 8192 (both) and 32768 complex64. That split is not arbitrary:
//     radix-8 wins where its last stage strides 512 bytes or less between its
//     eight streams and loses at 4 KiB or more, where they collide on one L1
//     set. See docs/CODELET_BENCHMARKS.md.
//
// Expected signatures are spelled out rather than derived, so that changing
// the answer requires editing the answer.
func TestRadix4AVX2FusedSelection(t *testing.T) {
	features := cpu.DetectFeatures()
	if !features.HasAVX2 {
		t.Skip("AVX2 not available")
	}

	// Registry ordering is SIMD-level major, so on an AVX-512 host Lookup
	// returns AVX-512 rows and none of the AVX2 signatures below can be
	// selected at all -- the assertion would be testing the host's ISA rather
	// than the per-size AVX2 tuning it exists to pin. Verified on the Xeon Gold
	// 5218, idle: n=128 selects dit128_radix8_then2_avx512 (c64) and
	// dit128_radix4_then2_avx512 (c128), n=8192 selects dit8192_radix2_avx512.
	if features.HasAVX512 {
		t.Skip("AVX-512 host: the registry selects AVX-512 rows, not the AVX2 rows pinned here")
	}

	want64 := map[int]string{
		128:   "dit128_radix4fused_avx2",
		512:   "dit512_radix8ladder_avx2",
		2048:  "dit2048_radix8ladder_avx2",
		8192:  "dit8192_radix4_avx2",
		32768: "dit32768_radix4_avx2",
	}

	want128 := map[int]string{
		128:   "dit128_radix4fused_avx2",
		512:   "dit512_radix8ladder_avx2",
		2048:  "dit2048_radix8ladder_avx2",
		8192:  "dit8192_radix4_avx2",
		32768: "dit32768_radix8ladder_avx2",
	}

	for _, n := range []int{128, 512, 2048, 8192, 32768} {
		entry := registry.Registry64.Lookup(n, features)
		if entry == nil {
			t.Fatalf("complex64 n=%d: no codelet", n)
		}

		if entry.Signature != want64[n] {
			t.Errorf("complex64 n=%d: signature %q, want %q", n, entry.Signature, want64[n])
		}

		entry128 := registry.Registry128.Lookup(n, features)
		if entry128 == nil {
			t.Fatalf("complex128 n=%d: no codelet", n)
		}

		if entry128.Signature != want128[n] {
			t.Errorf("complex128 n=%d: signature %q, want %q", n, entry128.Signature, want128[n])
		}
	}
}

// TestRadix4AVX2WisdomAlternates pins the separate-tail rows retained after
// Zen 2 reversed the Intel fused-tail result. They deliberately remain below
// the compiled-in defaults but must be present for per-host Wisdom tuning.
func TestRadix4AVX2WisdomAlternates(t *testing.T) {
	tests64 := map[int]string{
		128:  "dit128_radix4_avx2",
		2048: "dit2048_radix4_avx2",
	}

	for n, signature := range tests64 {
		assertCodeletPriority64(t, n, signature, 80)
	}

	assertCodeletPriority128(t, 128, "dit128_radix4_avx2", 80)
}

func assertCodeletPriority64(t *testing.T, n int, signature string, priority int) {
	t.Helper()

	entry := registry.Registry64.LookupBySignature(n, signature)
	if entry == nil {
		t.Fatalf("complex64 n=%d: wisdom candidate %s is not registered", n, signature)
	}

	if entry.Priority != priority {
		t.Fatalf("complex64 n=%d: %s priority %d, want %d", n, signature, entry.Priority, priority)
	}
}

func assertCodeletPriority128(t *testing.T, n int, signature string, priority int) {
	t.Helper()

	entry := registry.Registry128.LookupBySignature(n, signature)
	if entry == nil {
		t.Fatalf("complex128 n=%d: wisdom candidate %s is not registered", n, signature)
	}

	if entry.Priority != priority {
		t.Fatalf("complex128 n=%d: %s priority %d, want %d", n, signature, entry.Priority, priority)
	}
}
