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
	famTuned  = "tuned"  // measured, the incumbent, rows carry the evidence
	famClosed = "closed" // measured and lost everywhere; do not re-attempt
	// famOpen — the family's disposition is not settled and a PLAN task owns
	// it. Usually that is because no sweep has answered it. It also covers the
	// case a sweep answers the *file* rather than the algorithm: §2.2 forbids
	// closing a family on an implementation defect, so a measured-and-losing
	// kernel whose loss is diagnosed as fixable stays open under the task that
	// would fix it (six-step and four-step, 2026-08-02).
	famOpen     = "open"
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
		Family: "Radix-2 (ladder)", Status: famClosed,
		Verdict:  "the corrected size-generic NEON radix-2 DIT ladder loses every size from 4 through 65536 on an Apple M5: 2.65–6.18× in complex64 and 1.45–5.42× in complex128. The one sub-1.5× direction is paired with a 1.69× inverse loss, so every row remains probe-only",
		Evidence: "docs/CODELET_BENCHMARKS.md, NEON radix-2 ladder on Apple M5",
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
		Verdict: "swept on two hosts 2026-08-02 and it loses everywhere. The codelet half is a fair pure-Go fight — its rows are the tuned `forwardDIT64Radix4…` leaves, the incumbent is the pure-Go radix-8 ladder — and it loses 1.43–2.20× (i7-1255U) and 1.59–2.34× (Xeon) at 4096/8192/16384, behind even the radix-4 row the ladder replaced. The strategy half loses 17–35× across 16384–131072, but that number is confounded and must not be cited: `ForwardSixStepComplex64` hardwires its rows to the generic `stockhamForward` (87% of its cost), so on a SIMD build it is a scalar kernel racing AVX2. §2.2 keeps the family open on the second point; on the first, the migration was carried out 2026-08-02 — the six rows left `specs.go` and now register only under `-tags fftprobe`, from `internal/kernels/sixstep_codelet_probe.go`. The deferral's reason -- binding the rows may move the number -- did not apply to them: these codelets call the tuned radix-4 leaves, so the Phase 3 row-binding work cannot change their result. Three codelets retired; the decomposition not",
		Tracked: "Give the six-step and four-step row passes the registry's kernels.",
	},
	{
		Family: "Six-step 64×128", Status: famOpen,
		Verdict: "the rectangular split of the same family, and the worst cell of the fair comparison: 1.71/1.91 (c64) and 2.06/2.09 (c128) against `dit8192_radix8ladder_generic` on the i7-1255U, 2.00/2.17 and 2.16/2.35 on the Xeon — behind the radix-4-then-2 row the ladder replaced on both. Its rows are 64- and 128-point and both have codelets at every ISA, so it is also the cheapest place to demonstrate whether SIMD row binding changes the answer. Its two codelet rows moved behind `-tags fftprobe` with the rest of the family's on 2026-08-02",
		Tracked: "Give the six-step and four-step row passes the registry's kernels.",
	},
	{
		Family: "Four-step", Status: famOpen,
		Verdict: "swept 2026-08-02 across 16384–131072 in both builds and both precisions; it is last or next-to-last in every cell, losing 3–35× to the bound codelet. Same scalar-row-pass defect as six-step, plus one of its own: `fourStepSplit` derives the balanced √n×√n split at every size measured, and that split was the slowest of eleven at 2^20 and 5.9% off the best at 2^18 — the cache model is steering four-step onto six-step's shape, discarding the rectangular split that is its only distinguishing feature",
		Tracked: "Give the six-step and four-step row passes the registry's kernels.",
	},
	{
		Family: "Split-radix", Status: famOpen,
		Verdict: "measured on two hosts 2026-08-02, and the answer splits at exactly the registry boundary. Below 65536 it loses **all sixty-four cells** — 1.06–1.35 forward, 1.10–1.44 inverse across 256…32768, both precisions, Xeon and i7-1255U agreeing cell by cell to within about 0.1. Exactly one cell reaches §2.2's 1.5× bar on the Xeon and the laptop puts that same cell at 1.367, so not even one survives as a two-host result; it also beats `dit16384_radix4_generic`, so it is mid-pack rather than dominated. That is the registered-low-priority case, and its sixteen new rows sit at priority 1. Above 32768 it beats everything on the Xeon (0.840 vs the Stockham auto picks, 0.686 vs the DIT route) — but the generic ladder *stops* at 32768, so above it the DIT arm falls onto `dit.go`'s size switch and gets worse per point: it loses to every tuned codelet that exists and wins wherever none does, which is a coverage gap, not an algorithmic win. The laptop then splits that win by direction (forward 0.926/0.941, inverse 1.113/1.238), so registering it at 65536 is explicitly *not* the free win the Xeon column suggests. It had no rows until this sweep because the gated harness can only rank registered candidates — having none was both the finding and the obstacle",
		Tracked: "Extend the generic codelet ladder past 32768.",
	},
	{
		Family: "Radix-32×32", Status: famOpen,
		Verdict: "measured 2026-08-02 and **both halves of the old premise were stale**. The \"only one of two stages vectorised\" defect is in files that no longer exist — `avx2_f{32,64}_size1024_radix32x32.s` were deleted in `08c8e7b` — and the surviving pure-Go row measures **1.255×** against `dit1024_radix4_generic`, not the 7.2×/5.2× the task recorded. What the sweep does show is a forward/inverse asymmetry, on both hosts: 1.264 fwd / 1.794 inv (complex64) and 1.522 / 1.979 (complex128) on the Xeon, 1.161 / 1.410 and 1.443 / 1.403 on the i7-1255U. A decomposition that loses 1.26× one way and 1.79× the other has an inverse-path defect, so §2.2 keeps the family open rather than closing it on the number. Nor is it probe-gated: every Xeon inverse cell clears the 1.5× bar but **no laptop cell of either row does**, and a one-machine number is not grounds for leaving the registry. Demoted 25 → 1 instead, which is the disposition both hosts do justify",
		Tracked: "Find the 32×32 / 16×32 inverse-path defect.",
	},
	{
		Family: "Radix-16×32", Status: famOpen,
		Verdict: "the same asymmetry as its 32×32 sibling, milder and more consistent across hosts: 1.230 fwd / 1.339 inv (complex64) and 1.470 / 1.527 (complex128) on the Xeon, 1.293 / 1.263 and 1.363 / 1.464 on the i7-1255U (purego, canary-gated, 2026-08-02) — no cell reaching 1.5 on either machine except one Xeon inverse. Its AVX2 file went the same way (`1f7977b`), so it too is now a pure-Go-only family, and it is demoted 35 → 1 on the same reasoning. It is the cheaper of the two to diagnose: one stage smaller, and the forward loss is under the 1.5× bar in both precisions",
		Tracked: "Find the 32×32 / 16×32 inverse-path defect.",
	},
	{
		Family: "Mixed-2/4", Status: famTuned,
		Verdict:  "the plain separate-tail form remains the selected odd-exponent row on generic, SSE2 and SSE3. NEON now has a directly measured fused peer: plain loses at 32/128/512/2048/8192 and stays low-priority, but wins at 32768 by 14-20% and is selected there in both precisions. On AVX2 the family has no rows at all — the tail is absorbed into `dit<N>_radix4_avx2` there",
		Evidence: "docs/CODELET_BENCHMARKS.md, NEON fused radix-4/radix-2 tail on Apple M5",
	},
	{
		Family: "Radix-4 (fused tail)", Status: famTuned,
		Verdict:  "decided per ISA, cell and host. AVX2 keeps the two-Intel-host fused radix-4 forms at complex64 128/2048 and complex128 128, but Zen 2 reverses them by 2-4%; their separate counterparts are production Wisdom candidates at priority 80. Larger AVX2 fusion losses reach 1.72-3.48x on Zen 2 and stay probe-only. NEON fusion removes a complete store/reload pass and wins both precisions at 32/128/512/2048/8192; at 32768 it loses 14-20%, remains registered at priority 20, and the separate tail is selected",
		Evidence: "docs/CODELET_BENCHMARKS.md, The radix-4 tail on Zen 2: the host split becomes actionable (2026-08-09)",
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
