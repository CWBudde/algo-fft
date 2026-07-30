package kernels

import (
	"fmt"
	"testing"
)

// The naive O(n²) reference DFT is the ground truth these kernel tests are
// built on, but its cost grows quadratically while the kernels it validates
// grow as n·log n. Under the race detector every load and store in that
// double loop is instrumented, and the largest sizes come to dominate the
// entire package: a measured run of internal/kernels took 1499 s under -race
// against 185 s untagged, with n = 32768 alone accounting for roughly 78% of
// each affected sweep (n² = 1.07e9 against 1.42e9 for the whole size list).
//
// That matters because Go's default test timeout is 10 minutes and neither
// `just test` nor the CI workflows override it, so the package could not
// finish under -race at all.
//
// The resolution is to bound the *reference* work under -race rather than to
// raise the timeout, because the race detector exists to find data races, not
// to re-verify arithmetic. The full-size reference sweep still runs — it is
// covered by the untagged amd64 CI job (.github/workflows/test-arch.yaml),
// which was added alongside this cap precisely so that the coverage removed
// here is not removed from CI.
//
// See also skipNaiveReferenceIfSlow in test_helpers.go, which covers the
// -short and QEMU cases on the same reasoning.

// naiveReferenceRaceMaxSize bounds the O(n²) naive reference under -race.
// 4096 keeps every distinct kernel *shape* in the race run — both n = 4^k
// (1024, 4096) and n = 2·4^k (512, 2048), each with its permutation, twiddle
// and tail path — while dropping only the three largest sizes, which repeat
// those same shapes at a quadratically higher reference cost.
const naiveReferenceRaceMaxSize = 4096

// skipNaiveReferenceAtSize skips a fixed-size naive-reference check that is
// affordable untagged but not under the race detector. It subsumes
// skipNaiveReferenceIfSlow, so callers need only this one.
func skipNaiveReferenceAtSize(t *testing.T, n int) {
	t.Helper()

	skipNaiveReferenceIfSlow(t)

	if raceDetectorEnabled && n > naiveReferenceRaceMaxSize {
		t.Skipf("naive DFT at n=%d is too slow under the race detector (cap %d)", n, naiveReferenceRaceMaxSize)
	}
}

// naiveReferenceSizes filters a sweep's size list down to the sizes whose
// naive reference is affordable in the current build. Sweeps use this instead
// of skipNaiveReferenceAtSize so that the small sizes still run under -race
// rather than the whole test skipping on account of its largest entry.
func naiveReferenceSizes(t *testing.T, sizes []int) []int {
	t.Helper()

	skipNaiveReferenceIfSlow(t)

	if !raceDetectorEnabled {
		return sizes
	}

	kept := make([]int, 0, len(sizes))
	dropped := make([]int, 0, len(sizes))

	for _, n := range sizes {
		if n <= naiveReferenceRaceMaxSize {
			kept = append(kept, n)
		} else {
			dropped = append(dropped, n)
		}
	}

	if len(dropped) > 0 {
		// Log rather than skip silently: a size list that quietly shrank
		// under -race would otherwise read as full coverage.
		t.Logf("race detector: skipping naive reference at %s (cap %d)", fmt.Sprint(dropped), naiveReferenceRaceMaxSize)
	}

	return kept
}
