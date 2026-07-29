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

// rankingTolerance is how much slower than the fastest codelet the 256-bit
// radix-4 kernel may measure before the test calls it a regression.
//
// The margins this guards are 2.5-3.5x, so a generous factor still catches a
// real loss while tolerating a contended machine. Timing rankings do invert
// under load -- not merely inflate -- so a tight bound here would be flaky
// rather than informative.
const rankingTolerance = 1.5

// rankingAttempts is how many independent timing passes a size gets before the
// test calls a breach real.
//
// Each pass is already a best-of-5, but the whole pass is only a few
// milliseconds wide at the larger sizes, so a burst of interference can cover
// all five rounds of one candidate and none of the next -- which inflates a
// single codelet rather than the group, and so shows up as a ranking change
// rather than as uniformly slower numbers. A real regression reproduces on
// every pass; a contended window does not. See PLAN.md 2.2.
const rankingAttempts = 3

// TestRadix4AVX2Ranking times every codelet registered for each size the 256-bit
// radix-4 kernel claims, so the Priority given to it in cmd/gencodelets/specs.go
// rests on a measurement rather than an assumption.
//
// For trustworthy absolute numbers run it pinned and idle:
//
//	taskset -c 0 go test -run TestRadix4AVX2Ranking -v ./internal/kernels/
func TestRadix4AVX2Ranking(t *testing.T) {
	if testing.Short() {
		t.Skip("timing test")
	}

	features := cpu.DetectFeatures()

	for _, n := range []int{128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536} {
		want := fmt.Sprintf("dit%d_radix4_avx2", n)

		var breach string

		for attempt := range rankingAttempts {
			results := timeCodelets(t, n, features)
			if len(results) == 0 {
				t.Errorf("n=%d: no codelet ran", n)

				break
			}

			report := make([]string, 0, len(results))
			for i, r := range results {
				report = append(report, fmt.Sprintf("\n  %2d. %-34s %9.0f ns", i+1, r.sig, r.ns))
			}

			t.Logf("n=%6d (pass %d)%s", n, attempt+1, strings.Join(report, ""))

			ours := 0.0

			for _, r := range results {
				if r.sig == want {
					ours = r.ns
				}
			}

			if ours == 0 {
				t.Errorf("n=%d: %s is not registered", n, want)

				break
			}

			best := results[0].ns
			if ours <= best*rankingTolerance {
				breach = ""

				break
			}

			breach = fmt.Sprintf("n=%d: %s took %.0f ns, more than %.1fx the fastest codelet %s at %.0f ns",
				n, want, ours, rankingTolerance, results[0].sig, best)
		}

		if breach != "" {
			t.Errorf("%s (reproduced on all %d passes)", breach, rankingAttempts)
		}
	}
}

type codeletTiming struct {
	sig string
	ns  float64
}

// timeCodelets returns every runnable codelet for size n, fastest first. Each
// entry is timed as the best of several short runs, the statistic least
// sensitive to interference from other work on the machine.
func timeCodelets(t *testing.T, n int, features cpu.Features) []codeletTiming {
	t.Helper()

	entries := registry.Registry64.GetAllForSize(n)
	results := make([]codeletTiming, 0, len(entries))

	for i := range entries {
		entry := entries[i]
		if !registry.CPUSupports(features, entry.SIMDLevel) {
			continue
		}

		src := randomComplex64(n, uint64(n))
		dst := make([]complex64, n)
		scratch := make([]complex64, 2*n)

		twiddle := m.ComputeTwiddleFactors[complex64](n)
		if entry.TwiddleSize != nil && entry.PrepareTwiddle != nil {
			if size := entry.TwiddleSize(n); size > 0 {
				twiddle = make([]complex64, size)
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
