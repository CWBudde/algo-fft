package main

// Per-cell verdicts for the codelet quick-reference grids.
//
// A tick that only means "a row exists" cannot answer the question Phase 1
// asks, which is whether an algorithm is *reachable* on some host: a codelet
// outranked by a peer in its own SIMD tier is never returned by
// registry.Lookup, but the wisdom tuner can still select it on a machine where
// it wins — and that is the entire reason such rows are kept (PLAN.md §2.2).
// The two cases look identical in the registry and used to look identical
// here.

const (
	cellSelectable = "✓" // top of its (precision, size, rank tier) — what Lookup returns
	cellCandidate  = "·" // registered but outranked in its own tier; wisdom-reachable only
	cellProbe      = "p" // registered only under -tags fftprobe
	cellAbsent     = "—" // no kernel
	cellDisabled   = "✗" // registered with a negative priority: kept for the record, never run
)

// rankLevelOf returns the SIMD level a spec is *ordered* by, which is its
// RankLevel when set and its SIMDLevel otherwise. The distinction is the point
// of RankLevel: an AVX2-encoded but XMM-width codelet is demoted into the SSE2
// tier so its priority is comparable with the codelets it actually competes
// with, while still requiring AVX2 to execute.
func rankLevelOf(s codeletSpec) string {
	if s.RankLevel != "" {
		return s.RankLevel
	}

	return s.SIMDLevel
}

// rankGroup identifies the set of specs that compete for one Lookup result.
type rankGroup struct {
	prec  int
	size  int
	level string
}

// cellVerdicts computes the verdict for every (size, variant) × SIMD column
// cell of one precision's grid, merging the registered spec rows with the
// probe-only registrations authored in probes.go.
func cellVerdicts(prec int) map[gridKey]map[string]string {
	best := map[rankGroup]int{}

	for _, s := range codeletSpecs {
		if s.Prec != prec || s.Priority < 0 {
			continue
		}

		g := rankGroup{prec: s.Prec, size: s.Size, level: rankLevelOf(s)}
		if p, ok := best[g]; !ok || s.Priority > p {
			best[g] = s.Priority
		}
	}

	out := map[gridKey]map[string]string{}

	for _, s := range codeletSpecs {
		if s.Prec != prec {
			continue
		}

		verdict := cellCandidate

		switch {
		case s.Priority < 0:
			verdict = cellDisabled
		case s.Priority == best[rankGroup{prec: s.Prec, size: s.Size, level: rankLevelOf(s)}]:
			verdict = cellSelectable
		}

		setCell(out, gridKey{size: s.Size, variant: variantOf(s.Signature)}, s.SIMDLevel, verdict)
	}

	for _, note := range probeNotes {
		for _, c := range note.Cells {
			if c.Prec != prec {
				continue
			}

			setCell(out, gridKey{size: c.Size, variant: variantOf(c.signature())}, c.Level, cellProbe)
		}
	}

	return out
}

// setCell records a verdict, keeping the strongest one when two rows land on
// the same cell. A probe never overwrites a registered row: the grid should
// report what a production build does.
func setCell(out map[gridKey]map[string]string, key gridKey, level, verdict string) {
	if out[key] == nil {
		out[key] = map[string]string{}
	}

	if rankVerdict(verdict) > rankVerdict(out[key][level]) {
		out[key][level] = verdict
	}
}

// rankVerdict orders verdicts by how much of a claim they make, so merging is
// deterministic regardless of table order.
func rankVerdict(v string) int {
	switch v {
	case cellSelectable:
		return 4
	case cellCandidate:
		return 3
	case cellDisabled:
		return 2
	case cellProbe:
		return 1
	default:
		return 0
	}
}
