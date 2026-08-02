package main

// The algorithm × precision × ISA matrix (PLAN.md §1.2).
//
// The grids elsewhere in the inventory are indexed by *size*, which answers
// "what runs at n = 2048" and not "has anyone ever measured split-radix on
// NEON". §1.2 asks the second question, one row per algorithm family, and its
// deliverable used to be a hand-collected table. That table would have had more
// cells than any artifact §1.1 had to repair, so the two halves are split here:
//
//   - **Enumeration is generated.** The family axis is the union of the codelet
//     families derived from each spec Signature and the Algo field of the tier
//     table — the latter supplying every family that has no codelet rows at
//     all. That union is load-bearing rather than tidy: split-radix, which
//     §1.2 calls its single largest untested cell, is invisible to a
//     specs-only scan *because* having no rows is what makes it the gap.
//   - **Judgment is written**, in familyVerdicts below, and gated both ways by
//     matrix_test.go: a family in the tree with no verdict fails, and a verdict
//     naming a family that is no longer in the tree fails.
//
// The skeleton decides nothing. A grid of mostly-open verdict cells is a truer
// picture of how much of Phase 1 is undecided than a hand-collected table that
// quietly omits the families nobody has measured.

import (
	"bytes"
	"fmt"
	"sort"
	"strings"
)

// cellTier marks a family reachable through a non-codelet kernel tier rather
// than through the registry. It is not a registry verdict — nothing ranks it —
// so it needs a glyph of its own rather than borrowing cellSelectable.
const cellTier = "t"

// Verdict statuses. These say what has been *decided* about a family, which is
// independent of how many rows it has: a family can be broadly registered and
// still undecided, which is most of the table today.
const (
	famTuned    = "tuned"    // measured, the incumbent, rows carry the evidence
	famClosed   = "closed"   // measured and lost everywhere; do not re-attempt
	famOpen     = "open"     // no sweep has answered it; a PLAN task owns the question
	famDeferred = "deferred" // decided outside Phase 1, in the phase named by Note
	// famUntested — nobody has measured it and no task claims it, on purpose.
	// This is the one status that admits an absence rather than asserting a
	// result, so it is deliberately narrow: it is legitimate only for a family
	// nothing routes to without a *forced* strategy, which is what stops it
	// from meaning "we ship this unmeasured". The reachable half is enforced
	// mechanically in matrix_test.go — a family with registered codelet rows is
	// reachable by default and may not use this status — and Note must say what
	// would change the answer.
	famUntested = "untested"
	// famInstrument — not a candidate at all. A probe that computes a wrong
	// result by design, kept to price something; it must never acquire a
	// verdict that reads as "this kernel lost".
	famInstrument = "instrument"
)

// familyVerdict is the authored half of one matrix row.
type familyVerdict struct {
	Family string
	Status string
	// Verdict is the one-line conclusion, or the open question.
	Verdict string
	// Evidence is a "doc, heading" pair for famTuned/famClosed: the heading
	// must occur in the doc, so a verdict cannot cite a section that has been
	// renamed away.
	Evidence string
	// Tracked is a verbatim fragment of the open PLAN.md task that owns an
	// famOpen family, checked exactly as the census ratchet checks its own.
	Tracked string
	// Note names the phase that owns an famDeferred family, or — for
	// famUntested — what would turn the absence back into a question.
	Note string
}

// familyVerdicts is the hand-written half of the matrix. Every family the
// generator finds in the tree needs a row; see the file comment for the gate.
//
//nolint:gochecknoglobals // static configuration, verified by matrix_test.go
var familyVerdicts = []familyVerdict{
	{
		Family: "Radix-4", Status: famTuned,
		Verdict:  "the incumbent at nearly every size and ISA; the 256-bit AVX2 rewrite made it 2–4× faster and moved every crossover above it",
		Evidence: "docs/CODELET_BENCHMARKS.md, AVX2 tier (i7-1255U) — incumbent audit",
	},
	{
		Family: "Radix-8 (ladder)", Status: famTuned,
		Verdict:  "wins the generic tier and, on Skylake-SP, 16 AVX-512 cells; loses to radix-4 at the small sizes and on the i7-1255U above 2048 for complex128",
		Evidence: "docs/CODELET_BENCHMARKS.md, The AVX-512 radix-8 ladder: prediction half right, 16 rows promoted",
	},

	{
		Family: "Radix-16 (ladder)", Status: famClosed,
		Verdict:  "swept and lost every cell; the pass advantage is real and entirely consumed by the butterfly, and AVX-512's 32 ZMM leave it where AVX2 radix-8 already lost",
		Evidence: "docs/CODELET_BENCHMARKS.md, Generic tier — the radix-16 ladder, and where the radix ladder stops",
	},
	{
		Family: "Radix-4 (no tail — wrong result by design)", Status: famInstrument,
		Verdict: "measures what the separate `-then-2` tail pass costs by omitting it; 0.867–0.933 across all six groups. It is the *bound* the fused-tail row is judged against — the gap between 0.87 and the fused form's 0.94 is what fusion still leaves on the table — and it can never be a candidate itself",
		Note:    "the Radix-4 (fused tail) family, as its measuring instrument",
	},

	// Families a §1.2 item already owns. These are open on purpose: the
	// skeleton records the question, the item answers it.
	{
		Family: "Radix-16", Status: famTuned,
		Verdict:  "not closed — the flat n = 16 leaf is the selected row in the pure-Go, SSE3 and AVX-512 complex64 tiers and in pure-Go complex128; it is outranked only on AVX2, where the 2026-07-30 audit ranked every n = 16 candidate and radix-2 took the row. What is closed is the size-generic radix-16 ladder, which is a different family",
		Evidence: "docs/CODELET_BENCHMARKS.md, AVX2 tier (i7-1255U) — incumbent audit",
	},
	{
		Family: "Radix-2", Status: famTuned,
		Verdict:  "splits by precision, and not the way §1.2 assumed. complex64: it is the *incumbent* at n = 16, 32 and 64 on AVX2 (54.6 vs 124.6 ns at 64 — the only genuinely 256-bit size-64 codelet) and at n = 64 on AVX-512 and NEON. complex128: never selected at any n ≥ 16 in any tier, so it is dominated there",
		Evidence: "docs/CODELET_BENCHMARKS.md, AVX2 tier (i7-1255U) — incumbent audit",
	},
	{
		Family: "Six-step", Status: famOpen,
		Verdict: "rows pulled as a stale crossover, not a bad kernel; the crossing point moved up when radix-4 got faster",
		Tracked: "Re-derive the six-step / eight-step / four-step crossovers.",
	},
	{
		Family: "Six-step 64×128", Status: famOpen,
		Verdict: "the rectangular split of the same family; shares the crossover question",
		Tracked: "Re-derive the six-step / eight-step / four-step crossovers.",
	},
	{
		Family: "Eight-step", Status: famOpen,
		Verdict: "pure-Go only, no codelet rows at any ISA; crossover never re-derived after the radix-4 rewrite",
		Tracked: "Re-derive the six-step / eight-step / four-step crossovers.",
	},
	{
		Family: "Four-step", Status: famOpen,
		Verdict: "pure-Go only; splits n1×n2 from detected cache sizes, which is exactly the parameter a second host would move",
		Tracked: "Re-derive the six-step / eight-step / four-step crossovers.",
	},
	{
		Family: "Split-radix", Status: famOpen,
		Verdict: "the largest untested cell in the matrix: beat the auto path at every power of two ≥ 256 on purego, has no codelet row and no SIMD kernel at any ISA, and is auto-selected nowhere",
		Tracked: "Give split-radix a fair measurement.",
	},
	{
		Family: "Radix-32×32", Status: famOpen,
		Verdict: "lost as implementation-limited — only one of two stages vectorised — which per AGENTS.md disqualifies the file, not the algorithm",
		Tracked: "Decide the 32×32 / 16×32 decomposition family on merit.",
	},
	{
		Family: "Radix-16×32", Status: famOpen,
		Verdict: "same unvectorised second stage as its 32×32 sibling; same open question",
		Tracked: "Decide the 32×32 / 16×32 decomposition family on merit.",
	},
	{
		Family: "Mixed-2/4", Status: famTuned,
		Verdict:  "the plain separate-tail form, and outside AVX2 it is the *only* form that exists: it is the selected row at every odd-exponent size (32, 128, 512, 2048, 8192, 32768) on generic, SSE2, SSE3 and NEON in both precisions, uncontested because no fused or radix-8-then-2 kernel is built for those tiers. Where it is outranked it is by the radix-8 ladder, not by a different tail (generic 512/2048/8192). On AVX2 the family has no rows at all — the tail is absorbed into `dit<N>_radix4_avx2` there",
		Evidence: "docs/CODELET_BENCHMARKS.md, AVX2 tier (i7-1255U) — incumbent audit",
	},
	{
		Family: "Radix-4 (fused tail)", Status: famTuned,
		Verdict:  "decided per cell, and every cell now has a number. It wins n = 128 in **both** precisions (0.955/0.979 complex64, 0.935/0.934 complex128 — the largest fusion win in either) and is the registered incumbent there. At 512 and 2048 the radix-8 ladder beats it directly (0.952/0.987 and 0.940/0.961 at 512), so neither plain nor fused is the answer at those sizes. At 2048 complex128 fusing *costs* 11% where the stride is exactly 4 KiB. Above that the fused rows are probe-only. The mechanism is why this must stay per-cell rather than becoming a default: the fused loop keeps eight live streams instead of four and loses that trade at larger strides, which is a cache-geometry property and therefore exactly what the wisdom tuner has to be able to flip per host",
		Evidence: "docs/CODELET_BENCHMARKS.md, What the audit changed",
	},
	{
		Family: "Mixed-8/2", Status: famTuned,
		Verdict:  "the radix-8 spelling of the tail, and it exists on **AVX-512 complex64 only** — two cells, no other tier builds it. It holds n = 128, where the radix-8 ladder measured parity (1.039/0.997) and stayed probe-only; it lost n = 256 to that ladder, which was registered at 50 specifically to clear this row's 30. Its shape is why it is tier-bound: a register-resident radix-2 DIT with a fused in-register radix-8 leaf, keeping all 16 ZMM live from load to store — there is no AVX2 register file to port it to",
		Evidence: "docs/CODELET_BENCHMARKS.md, The AVX-512 radix-8 ladder: prediction half right, 16 rows promoted",
	},

	// Families the skeleton surfaced with no owner at all. Answered 2026-08-01
	// from the registry and the recorded sweeps; four turned out to have an
	// owner already — a task elsewhere in the PLAN that genuinely covers the
	// remaining question — which is a better answer than a new item.
	{
		Family: "Radix-8", Status: famTuned,
		Verdict:  "the flat n = 8 leaf, and it *has* been ranked against radix-2 and radix-4 there: the 2026-07-30 AVX2 audit covers n = 8 in both precisions and moved the complex128 AVX2 row to it (0.970 forward / 0.859 inverse over `dit8_radix4_avx2`). It is the selected row in seven of its nine registered cells — pure-Go, SSE3, SSE2, AVX-512 and NEON complex64, plus AVX2 complex128 — and is outranked in exactly two: AVX2 complex64, where `dit8_radix2_avx2` holds 12 against 11 and the loss is under 1.5× (it is absent from the shadowed-candidates table, which lists everything above that bar), and NEON complex128, which has no radix-8 row at all and is held by radix-4. The unrelated `dit512_radix8_generic` rows sit under the radix-8 ladder at the same size",
		Evidence: "docs/CODELET_BENCHMARKS.md, What the audit changed",
	},
	{
		Family: "Radix-32", Status: famOpen,
		Verdict: "the registry half is decided: n = 32 was in the 2026-07-30 AVX2 audit and radix-32 took no cell — 25 against radix-2's 30 on AVX2 complex64, 5 against `radix4_then2`'s 10/15 in pure Go — losing by under 1.5× everywhere, which is §2.2's keep-at-low-priority case and is what the table already does. What is *not* decided is the SSE3 cell, and it is not merely uncovered: `sse3_f32_size32_radix32.s` is live and tried **first** at n = 32 by the `KernelStrategy` switch in `internal/fft/kernels_amd64_size_specific.go`, while the registry has no SSE3 radix-32 row and selects `dit32_radix4_then2_sse3` there. The two selection paths disagree at one cell, and the task below owns that whole switch",
		Tracked: "Measure the cheap alternative first: drop the size-specific cases outright.",
	},
	{
		Family: "Generic radix-2", Status: famOpen,
		Verdict: "not a codelet family so much as two rows (n = 32 and n = 512, complex128 only) that register the size-generic NEON kernel — a radix-2 DIT ladder, per its own header — as a codelet at priority 1, under `radix4_then2` at 24. The structural argument that closed radix-2 on AVX2 complex128 applies unchanged here (log2 n full passes against half as many, at a width that holds one complex128 per register), but AGENTS.md is explicit that such an argument is not a substitute for the second measurement, and **no arm64 sweep has covered complex128 at any size**. Priority 1 is the right hold meanwhile; the task below is the one that needs the hardware",
		Tracked: "NEON: priority tuning on real arm64 hardware.",
	},
	{
		Family: "Mixed 128×3", Status: famOpen,
		Verdict: "n = 384, the only non-power-of-two in the codelet table, and uncontested in all four cells it occupies — its `✓` means \"the only row at that size\", not \"the best\". Which makes the open question not how it ranks but whether the other four tiers want the same 128×3 decomposition or none, and §1.5 already has that as one verdict for all four",
		Tracked: "Decide n = 384 once, for all four tiers.",
	},
	{
		Family: "DIT", Status: famOpen,
		Verdict: "the engine the codelet leaves hang off, reached whenever the auto heuristic stays below `ditAutoThreshold` — a threshold calibrated against kernels now 2–4× faster, so where it should sit is the same question §1.4 asks",
		Tracked: "Retune the strategy thresholds around the new codelets.",
	},
	{
		Family: "Stockham", Status: famOpen,
		Verdict: "what auto selects above the DIT threshold, so its verdict is the same threshold question; the packed variant is a separate axis whose gate is filled for AVX2 only",
		Tracked: "Retune the strategy thresholds around the new codelets.",
	},
	{
		Family: "Recursive", Status: famUntested,
		Verdict: "recursive decomposition bottoming out in registered codelet leaves. Never measured against the flat ladders, and — unlike every other family here — nothing routes to it by default: `resolveKernelStrategy` returns `KernelRecursive` only when it is forced, so no plan runs it unmeasured. That is what makes untested an honest answer rather than a shrug",
		Note:    "nobody, deliberately. It becomes a real question only if Phase 3 wants a recursive route for large n, or if a codelet-leaf ladder beats the flat one somewhere",
	},

	// Non-power-of-two routes. In the tree, and deliberately not Phase 1's call.
	{
		Family: "Mixed-radix engine", Status: famDeferred,
		Verdict: "the route every smooth non-power-of-two takes, and the worst external cells (2205 at 0.16×, 96 at 0.20×)",
		Note:    "Phase 2",
	},
	{
		Family: "Rader", Status: famDeferred,
		Verdict: "healthy at 0.78–1.58× vs FFTW3 with outright wins at 641, 4001 and 12289",
		Note:    "Phase 2",
	},
	{
		Family: "Bluestein", Status: famDeferred,
		Verdict: "mediocre at 0.42–0.62× vs FFTW3; the pad model is the lever",
		Note:    "Phase 2",
	},
}

// familyCells is one row of the generated grid.
type familyCells struct {
	Family string
	// Cells[prec][simd level] is the strongest verdict glyph for that cell.
	Cells map[int]map[string]string
	// Rows counts the registered spec rows behind the family, per precision.
	Rows map[int]int
}

// familyMatrix derives the whole grid from the specs, the probe cells and the
// tier table. Nothing here is authored; the authored half is familyVerdicts.
func familyMatrix() []familyCells {
	byFamily := map[string]*familyCells{}

	get := func(name string) *familyCells {
		if _, ok := byFamily[name]; !ok {
			byFamily[name] = &familyCells{
				Family: name,
				Cells:  map[int]map[string]string{64: {}, 128: {}},
				Rows:   map[int]int{},
			}
		}

		return byFamily[name]
	}

	best := bestPriorityByGroup()

	for _, s := range codeletSpecs {
		row := get(variantOf(s.Signature))
		row.Rows[s.Prec]++

		verdict := cellCandidate

		switch {
		case s.Priority < 0:
			verdict = cellDisabled
		case s.RankBelowGeneric:
			verdict = cellBelow
		case s.Priority == best[rankGroup{prec: s.Prec, size: s.Size, level: rankLevelOf(s)}]:
			verdict = cellSelectable
		}

		setFamilyCell(row, s.Prec, s.SIMDLevel, verdict)
	}

	for _, note := range probeNotes {
		for _, c := range note.Cells {
			setFamilyCell(get(variantOf(c.signature())), c.Prec, c.Level, cellProbe)
		}
	}

	for _, t := range tierRows {
		if t.Algo == "" {
			continue
		}

		row := get(t.Algo)
		for _, prec := range tierPrecisions(t.Prec) {
			setFamilyCell(row, prec, "SIMDNone", cellTier)
		}
	}

	out := make([]familyCells, 0, len(byFamily))
	for _, row := range byFamily {
		out = append(out, *row)
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Family < out[j].Family })

	return out
}

// bestPriorityByGroup is the top priority in each (precision, size, rank tier),
// i.e. what registry.Lookup returns for that group.
func bestPriorityByGroup() map[rankGroup]int {
	best := map[rankGroup]int{}

	for _, s := range codeletSpecs {
		if s.Priority < 0 {
			continue
		}

		g := rankGroup{prec: s.Prec, size: s.Size, level: rankLevelOf(s)}
		if p, ok := best[g]; !ok || s.Priority > p {
			best[g] = s.Priority
		}
	}

	return best
}

// setFamilyCell keeps the strongest glyph for a cell, using the same ordering
// the size grids use so the two never disagree about a shared cell.
func setFamilyCell(row *familyCells, prec int, level, verdict string) {
	if rankFamilyVerdict(verdict) > rankFamilyVerdict(row.Cells[prec][level]) {
		row.Cells[prec][level] = verdict
	}
}

// rankFamilyVerdict extends rankVerdict with the tier glyph, which outranks
// nothing: a family reachable only through a kernel tier has no registry
// standing, and a family with both should show the registry answer.
func rankFamilyVerdict(v string) int {
	if v == cellTier {
		return 1
	}

	if r := rankVerdict(v); r > 0 {
		return r + 1
	}

	return 0
}

// tierPrecisions expands a tier row's Prec field to the grid's precisions.
func tierPrecisions(prec string) []int {
	switch prec {
	case "complex64":
		return []int{64}
	case "complex128":
		return []int{128}
	default:
		return []int{64, 128}
	}
}

// renderMatrix writes the algorithm × precision × ISA section.
func renderMatrix(b *bytes.Buffer) {
	verdicts := map[string]familyVerdict{}
	for _, v := range familyVerdicts {
		verdicts[v.Family] = v
	}

	b.WriteString(strings.TrimLeft(`
## Algorithm × Precision × ISA

The grids above are indexed by size, which answers "what runs at n = 2048" but
not "has anyone measured split-radix anywhere". This is the second question,
one row per algorithm family — PLAN.md §1.2.

The family axis is the union of two derived sets: the families encoded in each
codelet `+"`Signature`"+`, and the `+"`Algo`"+` column of the tier table below, which
supplies every family that has **no codelet rows at all**. Split-radix is in
this table only because of the second source, and it is §1.2's largest open
cell — a specs-only scan would have omitted exactly the thing it exists to find.

Cells use the size grids' glyphs, plus `+"`t`"+` for a family reached through a
kernel tier rather than the codelet registry (it has no registry ranking, so it
is not a `+"`✓`"+`).

`, "\n"))

	b.WriteString("| Family | complex | ")

	for _, level := range simdColumns {
		b.WriteString(simdColumnNames[level] + " | ")
	}

	b.WriteString("Rows | Status |\n|---|---:|")
	b.WriteString(strings.Repeat(":-:|", len(simdColumns)))
	b.WriteString("---:|---|\n")

	for _, row := range familyMatrix() {
		for _, prec := range []int{64, 128} {
			fmt.Fprintf(b, "| %s | %d |", row.Family, prec)

			for _, level := range simdColumns {
				cell := row.Cells[prec][level]
				if cell == "" {
					cell = cellAbsent
				}

				fmt.Fprintf(b, " %s |", cell)
			}

			fmt.Fprintf(b, " %d | %s |\n", row.Rows[prec], verdicts[row.Family].Status)
		}
	}

	b.WriteString("\n### Family verdicts\n\n")
	b.WriteString("Hand-written, and gated in both directions: a family in the tree with no\n")
	b.WriteString("verdict fails the generator's tests, and a verdict naming a family that is\n")
	b.WriteString("no longer there fails too. An `open` family must quote an **open** PLAN.md\n")
	b.WriteString("task, so the question has an owner rather than a shrug.\n\n")

	for _, row := range familyMatrix() {
		v, ok := verdicts[row.Family]
		if !ok {
			fmt.Fprintf(b, "- **%s** — **no verdict; see cmd/gencodelets/matrix.go**\n", row.Family)

			continue
		}

		fmt.Fprintf(b, "- **%s** (%s) — %s%s\n", v.Family, v.Status, v.Verdict, verdictSource(v))
	}

	b.WriteString("\n")
}

// verdictSource renders where a verdict's authority comes from.
func verdictSource(v familyVerdict) string {
	switch {
	case v.Evidence != "":
		return ". Evidence: " + v.Evidence
	case v.Tracked != "":
		return ". Tracked by PLAN.md: " + strings.TrimSuffix(v.Tracked, ".")
	case v.Note != "":
		return ". Owned by " + v.Note
	default:
		return ""
	}
}
