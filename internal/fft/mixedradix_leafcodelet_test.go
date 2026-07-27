package fft

import "testing"

// mixedRadixBenchSizes covers the schedule shapes the leaf-codelet invariant
// has to hold for: pure powers of two, the 3/5/7/11 factors the scheduler
// strips ahead of them, and the practical DSP lengths.
func mixedRadixScheduleSizes() []int {
	sizes := []int{
		96, 100, 128, 240, 256, 448, 480, 512, 640, 704, 768, 960, 1000, 1024,
		1536, 2205, 3600, 4900, 12000, 12288, 44100,
	}

	for n := 2; n <= 4096; n++ {
		sizes = append(sizes, n)
	}

	return sizes
}

// TestMixedRadixCompositeRadixOnlyFinal pins the invariant the hoisted leaf
// codelet rests on: the scheduler emits a composite radix (one the pure Go
// butterfly cannot execute) only as the schedule's last stage.
//
// If this ever stopped holding, an interior node would carry a radix the
// butterfly switch treats as a driver-contract violation, and the recursion
// would panic instead of dispatching — the hoist removed the per-node lookup
// that used to catch such a radix on the way down.
func TestMixedRadixCompositeRadixOnlyFinal(t *testing.T) {
	t.Parallel()

	var radices [mixedRadixMaxStages]int

	for _, n := range mixedRadixScheduleSizes() {
		count := mixedRadixSchedule(n, &radices, codeletSchedulable64)
		if count == 0 {
			continue
		}

		for i := range count - 1 {
			switch radices[i] {
			case 2, 3, 4, 5, 7, 8, 11:
			default:
				t.Fatalf("n=%d schedule %v: composite radix %d at stage %d, not the last",
					n, radices[:count], radices[i], i)
			}
		}
	}
}

// TestMixedRadixLeafCodeletMatchesPerNodeLookup checks that resolving the leaf
// codelet once from radices[count-1] selects exactly the nodes a per-node
// registry lookup would have: every interior node's remaining size must have no
// codelet, and the final stage must have one whenever its radix is composite.
func TestMixedRadixLeafCodeletMatchesPerNodeLookup(t *testing.T) {
	t.Parallel()

	var radices [mixedRadixMaxStages]int

	for _, n := range mixedRadixScheduleSizes() {
		count := mixedRadixSchedule(n, &radices, codeletSchedulable64)
		if count == 0 {
			continue
		}

		// Interior nodes: the sub-transform size at stage i is the product of
		// the radices from i onwards, which is what the driver used to look up.
		remaining := n
		for i := range count - 1 {
			if remaining > mixedRadixCodeletMinSize && leafCodelet64(remaining) != nil {
				t.Fatalf("n=%d schedule %v: stage %d size %d resolves a codelet the hoist would skip",
					n, radices[:count], i, remaining)
			}

			remaining /= radices[i]
		}

		if remaining != radices[count-1] {
			t.Fatalf("n=%d schedule %v: leaf size %d != final radix %d",
				n, radices[:count], remaining, radices[count-1])
		}

		// The leaf: a composite radix is only schedulable because a codelet
		// exists for it, so the hoisted lookup must find one.
		switch remaining {
		case 2, 3, 4, 5, 7, 8, 11:
		default:
			if leafCodelet64(remaining) == nil {
				t.Fatalf("n=%d schedule %v: composite leaf radix %d has no codelet",
					n, radices[:count], remaining)
			}
		}
	}
}
