//go:build amd64 && !purego

package kernels

import (
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/cwbudde/algo-fft/internal/cpu"
	m "github.com/cwbudde/algo-fft/internal/math"
	"github.com/cwbudde/algo-fft/internal/registry"
)

// TestRadix4AVX2Complex128Ranking is the complex128 twin of
// TestRadix4AVX2Ranking: it times every codelet registered for each size the
// 256-bit radix-4 kernel claims, so the Priority given to it in
// cmd/gencodelets/specs.go rests on a measurement rather than an assumption.
//
// For trustworthy absolute numbers run it pinned and idle:
//
//	taskset -c 0 go test -run TestRadix4AVX2Complex128Ranking -v ./internal/kernels/
func TestRadix4AVX2Complex128Ranking(t *testing.T) {
	if testing.Short() {
		t.Skip("timing test")
	}

	features := cpu.DetectFeatures()

	for _, n := range []int{16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536} {
		want := fmt.Sprintf("dit%d_radix4_avx2", n)

		results := timeCodelets128(t, n, features)
		if len(results) == 0 {
			t.Errorf("n=%d: no codelet ran", n)
			continue
		}

		report := make([]string, 0, len(results))
		for i, r := range results {
			report = append(report, fmt.Sprintf("\n  %2d. %-34s %9.0f ns", i+1, r.sig, r.ns))
		}

		t.Logf("n=%6d%s", n, strings.Join(report, ""))

		ours := 0.0

		for _, r := range results {
			if r.sig == want {
				ours = r.ns
			}
		}

		if ours == 0 {
			t.Errorf("n=%d: %s is not registered", n, want)
			continue
		}

		// n = 16 is logged but not asserted. Its margin over the next codelet
		// is ~1.4x (14 vs 20 ns, pinned and idle, consistent over three runs),
		// which is inside the tolerance this test has to allow for a contended
		// machine -- so an assertion here would report load, not a regression.
		// It does invert when the whole suite runs in parallel; check it with
		// taskset before believing either direction.
		if n == 16 {
			continue
		}

		if best := results[0].ns; ours > best*rankingTolerance {
			t.Errorf("n=%d: %s took %.0f ns, more than %.1fx the fastest codelet %s at %.0f ns",
				n, want, ours, rankingTolerance, results[0].sig, best)
		}
	}
}

// timeCodelets128 returns every runnable complex128 codelet for size n, fastest
// first. Each entry is timed as the best of several short runs, the statistic
// least sensitive to interference from other work on the machine.
func timeCodelets128(t *testing.T, n int, features cpu.Features) []codeletTiming {
	t.Helper()

	entries := registry.Registry128.GetAllForSize(n)
	results := make([]codeletTiming, 0, len(entries))

	for i := range entries {
		entry := entries[i]
		if !registry.CPUSupports(features, entry.SIMDLevel) {
			continue
		}

		src := randomComplex128(n, uint64(n))
		dst := make([]complex128, n)
		scratch := make([]complex128, 2*n)

		twiddle := m.ComputeTwiddleFactors[complex128](n)
		if entry.TwiddleSize != nil && entry.PrepareTwiddle != nil {
			if size := entry.TwiddleSize(n); size > 0 {
				twiddle = make([]complex128, size)
				entry.PrepareTwiddle(n, false, twiddle)
			}
		}

		if !entry.Forward(dst, src, twiddle, scratch) {
			t.Logf("n=%d %-34s declined", n, entry.Signature)
			continue
		}

		best := 0.0
		iters := 1 + 2_000_000/n

		for range 5 {
			start := time.Now()

			for range iters {
				entry.Forward(dst, src, twiddle, scratch)
			}

			if ns := float64(time.Since(start).Nanoseconds()) / float64(iters); best == 0 || ns < best {
				best = ns
			}
		}

		results = append(results, codeletTiming{entry.Signature, best})
	}

	sort.Slice(results, func(a, b int) bool { return results[a].ns < results[b].ns })

	return results
}
