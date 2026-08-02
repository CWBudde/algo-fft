package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// The probe inventory. Kernels behind `-tags fftprobe` leave every production
// build and every registry lookup, which is exactly what makes them easy to
// forget: an unlisted probe is a measurement nobody can find and a question
// that quietly turns back into folklore. So the set of probe files is
// discovered from the tree (scanProbeFiles) and the verdict for each is
// authored here — a new probe file with no entry fails TestProbeNotesCoverTree
// rather than landing undocumented.

// probeStatus values. These describe what the last sweep concluded, not
// whether the file compiles.
const (
	probeOpen      = "open"            // the question is live; no sweep has answered it
	probePartial   = "partly promoted" // some cells won and became spec rows; the rest stay here
	probeClosed    = "closed"          // swept and lost everywhere; kept only to stay re-measurable
	probeSupport   = "support"         // shared helpers, not a kernel
	probeUndocumen = "UNDOCUMENTED"    // scanned from the tree with no note here
)

// probeNote is the authored half of a probe entry: what the file measures,
// what the sweep returned, and how to take the number again.
type probeNote struct {
	Subject string // what algorithm the probe measures, one line
	Status  string // one of the probeStatus constants
	Verdict string // what the last sweep returned, and what is still registered
	Record  string // where the sweep is written down
	Rederiv string // the command that reproduces it, "" for support files
	// Cells are the registry cells this file registers under -tags fftprobe,
	// which the quick-reference grids mark `p`. Empty for a probe that
	// registers nothing (a benchmark harness over the dispatch path) and for
	// support files. The registrations are not machine-extracted — the tail
	// probe picks its variant per size through a conditional that only a Go
	// evaluator would resolve — so TestProbeCellsMatchTheirFile checks each
	// cell's size and signature fragment against the file text instead.
	Cells []probeCell
}

// probeCell is one (precision, size, ISA tier) the probe registers a codelet
// at. Variant is the signature's middle token, e.g. "radix8ladder".
type probeCell struct {
	Prec    int
	Size    int
	Level   string // a simdColumns entry: the grid column the cell lands in
	Variant string
}

// signatureISA maps a SIMD level to the suffix the codelet signatures use.
//
//nolint:gochecknoglobals // static configuration for the inventory renderer
var signatureISA = map[string]string{
	"SIMDNone":   "generic",
	"SIMDSSE2":   "sse2",
	"SIMDSSE3":   "sse3",
	"SIMDAVX2":   "avx2",
	"SIMDAVX512": "avx512",
	"SIMDNEON":   "neon",
}

// signature reconstructs the registered signature, so a probe cell is named
// exactly the way the registry and the sweep name it.
func (c probeCell) signature() string {
	return "dit" + strconv.Itoa(c.Size) + "_" + c.Variant + "_" + signatureISA[c.Level]
}

// probeCells expands one (precision, tier, variant) across a list of sizes.
func probeCells(prec int, level, variant string, sizes ...int) []probeCell {
	out := make([]probeCell, 0, len(sizes))
	for _, size := range sizes {
		out = append(out, probeCell{Prec: prec, Size: size, Level: level, Variant: variant})
	}

	return out
}

// concatCells joins the per-precision groups of a probe's registrations.
func concatCells(groups ...[]probeCell) []probeCell {
	var out []probeCell
	for _, g := range groups {
		out = append(out, g...)
	}

	return out
}

// probeNotes is keyed by module-relative path.
//
//nolint:gochecknoglobals // static configuration for the inventory renderer
var probeNotes = map[string]probeNote{
	"internal/kernels/radix8_generic_probe.go": {
		Subject: "size-generic pure-Go radix-8 ladder against the pure-Go radix-4 ladder",
		Status:  probePartial,
		Verdict: "Forward geomean **0.87** over n = 512..32768 (2026-07-30). Thirteen of " +
			"twenty cells won and are now real rows in `cmd/gencodelets/specs.go`; they are " +
			"deliberately not registered here, since the same signature twice in one sweep " +
			"group reports a kernel against itself. What stays registered is where it lost: " +
			"n = 64 and 128 (forward 1.05-1.11, both precisions), 1024 complex128 (1.097) and " +
			"32768 (1.06/1.08). n = 32768 has the ladder's best pass ratio (5:8) and still " +
			"loses — its last stage holds eight streams 4096 elements apart, so all eight land " +
			"on the same L1 sets.",
		Record: "docs/CODELET_BENCHMARKS.md, \"Generic tier (i7-1255U, `-tags purego`) — the radix-8 ladder\"",
		Rederiv: "GOFLAGS=-tags=fftprobe,purego GOOD=<canary floor> " +
			"taskset -c 0 ./scripts/bench_gated.sh 512 1024 2048 4096 8192 16384 32768",
		Cells: concatCells(
			probeCells(64, "SIMDNone", "radix8ladder", 64, 128, 32768),
			probeCells(128, "SIMDNone", "radix8ladder", 64, 128, 1024, 32768),
		),
	},
	"internal/kernels/radix16_generic_probe.go": {
		Subject: "size-generic pure-Go radix-16 ladder against the pure-Go radix-8 ladder",
		Status:  probeClosed,
		Verdict: "**Loses every cell** (2026-08-01, 18 groups x 16 passes, full accounting): " +
			"1.018-1.356 against the radix-8 ladder across 256..32768 in both precisions. " +
			"Radix-16 makes 25-33% fewer passes at every size except 512 and the butterfly " +
			"consumes all of it; n = 1024 (1.018 forward) is the ceiling, not a lead, and it " +
			"still loses the inverse. n = 65536 is uncompared — no radix-8 peer is registered " +
			"there, so the probe is its own incumbent and its 1.000 means nothing. This is the " +
			"answer for **every** ISA: it ran in pure Go precisely so that no register budget " +
			"could confound it, and AVX2 has no room for 16 live streams at all.",
		Record: "docs/CODELET_BENCHMARKS.md, \"Generic tier — the radix-16 ladder, and where the radix ladder stops\"",
		Rederiv: "GOFLAGS=-tags=fftprobe,purego GOOD=<canary floor> " +
			"taskset -c 0 ./scripts/bench_gated.sh 256 512 1024 2048 4096 8192 16384 32768 65536",
		Cells: concatCells(
			probeCells(64, "SIMDNone", "radix16ladder", 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536),
			probeCells(128, "SIMDNone", "radix16ladder", 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536),
		),
	},
	"internal/kernels/radix8_avx2_probe_amd64.go": {
		Subject: "size-generic AVX2 radix-8 ladder against the AVX2 radix-4 rows",
		Status:  probePartial,
		Verdict: "Won cells are spec rows; the losses and ties stay registered. complex64: " +
			"32 and 64 never measured, 128 measures 0.984/0.989 — a 1.1-1.6% margin in the one " +
			"group that lost a pass to drift, below the 11-22% this project has promoted on — " +
			"and 4096-32768 lose at 1.011-1.078 (eight streams 4 KiB or more apart). " +
			"complex128: 32/64 unmeasured, 128 loses at 1.026/1.037, 256 and 16384 tie, 1024 " +
			"and 4096 win forward but lose inverse, 8192 loses. The 2026-08-01 re-sweep " +
			"reproduced the 2026-07-30 complex64 numbers across five weeks.",
		Record: "docs/CODELET_BENCHMARKS.md, \"The size-generic AVX2 radix-8 ladder\" and " +
			"\"n = 128 closed, and the odd-exponent question settled (2026-08-01)\"",
		Rederiv: "GOFLAGS=-tags=fftprobe GOOD=<canary floor> " +
			"taskset -c 0 ./scripts/bench_gated.sh 512 1024 2048 4096 8192 16384 32768",
		Cells: concatCells(
			probeCells(64, "SIMDAVX2", "radix8ladder", 32, 64, 128, 4096, 8192, 16384, 32768),
			probeCells(128, "SIMDAVX2", "radix8ladder", 32, 64, 128, 256, 1024, 4096, 8192, 16384),
		),
	},
	"internal/kernels/radix8_avx512_probe_amd64.go": {
		Subject: "size-generic AVX-512 radix-8 ladder against the AVX2 radix-4 rows",
		Status:  probePartial,
		Verdict: "The register-budget diagnosis held: 32 ZMM leave 21 scratch where 16 YMM left " +
			"five, and on a Xeon Gold 5218 (2026-07-31, 160 accepted / 0 rejected) **every size " +
			"from 256 up won** and is now a row in `cmd/gencodelets/specs_avx512.go`. What stays " +
			"here is n = 64 (complex64 1.968/1.861, complex128 1.464/1.455), n = 128 (complex128 " +
			"1.256/1.283 loss; complex64 1.039/0.997 is parity, not a win) and n = 32 complex128, " +
			"never measured. Note the `RankLevel: SIMDSSE2` demotion: registry order is " +
			"SIMD-level major, so an AVX-512 probe would otherwise _be_ the incumbent and the " +
			"sweep would report it 1.000 against itself — which the first run did.",
		Record: "docs/CODELET_BENCHMARKS.md, \"AVX-512 tier (Xeon Gold 5218)\" and " +
			"\"The radix-8 ladder on Skylake-SP — and the stride rule failing to transfer\"",
		Rederiv: "GOFLAGS=-tags=fftprobe taskset -c 0 ./scripts/bench_gated.sh " +
			"64 128 256 512 1024 2048 4096 8192 16384 32768   # AVX-512 host",
		Cells: concatCells(
			probeCells(64, "SIMDAVX512", "radix8ladder", 64, 128),
			probeCells(128, "SIMDAVX512", "radix8ladder", 32, 64, 128),
		),
	},
	"internal/kernels/radix4_avx2_tail_probe_amd64.go": {
		Subject: "the n = 2·4^k radix-2 tail: fused into the last radix-4 stage, or a separate pass",
		Status:  probeOpen,
		Verdict: "Standing harness rather than a one-shot question — the fused/unfused choice in " +
			"`cmd/gencodelets/specs.go` is empirical, and an empirical constant with no way to " +
			"re-derive it rots. At every 2·4^k size it registers the variant production does " +
			"_not_ use, so the comparison is available in both directions, plus a no-tail probe " +
			"that **computes the wrong answer on purpose**: the gap to the incumbent is the whole " +
			"cost of the tail and therefore the most any fusion could ever recover — 9-15% " +
			"measured, against the 4-6% the fusion actually gets where it wins.",
		Record:  "docs/CODELET_BENCHMARKS.md, \"n = 128 closed, and the odd-exponent question settled (2026-08-01)\"",
		Rederiv: "GOFLAGS=-tags=fftprobe GOOD=<canary floor> taskset -c 0 ./scripts/bench_gated.sh 128 512 2048 8192 32768",
		// The alternate variant is the one production does NOT use at that
		// size, so these never collide with a spec row —
		// TestProbeCellsDoNotShadowSpecRows is what keeps that true.
		Cells: concatCells(
			probeCells(64, "SIMDAVX2", "radix4", 128, 2048),
			probeCells(64, "SIMDAVX2", "radix4fused", 512, 8192, 32768),
			probeCells(64, "SIMDAVX2", "radix4_notail", 128, 512, 2048, 8192, 32768),
			probeCells(128, "SIMDAVX2", "radix4", 128),
			probeCells(128, "SIMDAVX2", "radix4fused", 512, 2048, 8192, 32768),
			probeCells(128, "SIMDAVX2", "radix4_notail", 128, 512, 2048, 8192, 32768),
		),
	},
	"internal/fft/radix4_c128_probe_amd64.go": {
		Subject: "complex128 generic AVX2 radix-4 / radix-4-then-2 against the generic AVX2 radix-2 dispatch",
		Status:  probeOpen,
		Verdict: "Wired into the production dispatch on 2026-08-01, verified against " +
			"`reference.NaiveDFT128`, confirmed by instrumentation to actually fire — and then " +
			"lost every size on the i7-1255U (forward 1.08-1.56, inverse 0.90-2.76 over " +
			"64..8192). That is decisive **on one machine**, which is not the same as a dead " +
			"kernel: complex128 on AVX2 is exactly where microarchitecture has been observed to " +
			"dominate here, the Skylake-SP radix-8 result having already refuted the i7-1255U " +
			"byte-stride rule outright. The width argument against it (a YMM holds two " +
			"complex128, not four) is the same species as the pass-count and Y-operand-census " +
			"predictions that were both wrong about this very kernel. A Xeon sweep decides it; " +
			"a second loss closes it and the files can go.",
		Record: "docs/CODELET_BENCHMARKS.md, \"complex128 generic AVX2: radix-2 wins on the i7-1255U, loses on the Xeon\"",
		Rederiv: "taskset -c 0 go test -tags fftprobe -run '^$' -bench 'BenchmarkC128Radix[24]' " +
			"-benchtime=0.5s -count=5 ./internal/fft/",
	},
	"internal/math/transpose_amd64.go": {
		Subject: "the AVX2 64x64/128x128 transposes — plain, fused twiddle, fused conjugate " +
			"twiddle — against the pure-Go blocked transpose they would displace",
		Status: probeOpen,
		Verdict: "**Unmeasured, and gated for reachability rather than for a result** " +
			"(2026-08-02). The six asm symbols are correct — `transpose_oop_test.go` checks all " +
			"three variants against a naive reference and against the pure-Go path at n = 64, " +
			"96 and 128 — but nothing in the library called them: the six-step and four-step " +
			"routes that want a square transpose are Phase 3. Correct-but-uncalled assembly is " +
			"the standing failure mode here, where a green suite proves nothing because the " +
			"registry-driven tests never reach the symbol, so the dispatch moved behind the tag " +
			"instead of staying nominally live. n = 96 is in the benchmark as a control: the " +
			"dispatch does not handle it, so both columns run identical code there and any gap " +
			"is noise. Phase 3 closes this by deleting the tag, not by a sweep — the sweep only " +
			"decides whether the asm is worth calling once something can.",
		Record: "PLAN.md, \"1.3 Resolve the unreachable assembly\"",
		Rederiv: "taskset -c 0 go test -tags fftprobe -run '^$' -bench BenchmarkTransposeProbe " +
			"-benchtime=0.5s -count=5 ./internal/math/",
	},
	"internal/kernels/sixstep_codelet_probe.go": {
		Subject: "the three six-step codelets against the pure-Go radix-8 ladder that replaced them",
		Status:  probeClosed,
		Verdict: "**Loses all twenty-four cells on two hosts** (2026-08-02). Against the group " +
			"incumbent `dit<N>_radix8ladder_generic`: 1.428-2.202 forward / 1.489-2.111 inverse " +
			"on the i7-1255U, 1.594-2.191 / 1.719-2.348 on the Xeon Gold 5218 — behind even the " +
			"`radix4`/`radix4_then2` rows the ladder replaced. This is a **fair** comparison and " +
			"the reason the rows left the registry: these codelets call the tuned pure-Go " +
			"radix-4 leaves, so on a purego sweep both arms are scalar. It is a verdict on the " +
			"three codelets, **not** on six-step — the separate *strategy* kernel " +
			"`ForwardSixStepComplex64` hardwires its row passes to the generic `stockhamForward` " +
			"(87% of its cost at n = 65536), and its 17-35x plan-level loss is confounded by " +
			"that and must not be cited. The family stays open under the Phase 3 row-binding " +
			"item.",
		Record: "docs/CODELET_BENCHMARKS.md, \"The six-step / four-step crossovers\"",
		Rederiv: "GOFLAGS=-tags=fftprobe,purego GOOD=<canary floor> " +
			"taskset -c 0 ./scripts/bench_gated.sh 4096 8192 16384",
		Cells: concatCells(
			probeCells(64, "SIMDNone", "sixstep", 4096, 16384),
			probeCells(64, "SIMDNone", "sixstep64x128", 8192),
			probeCells(128, "SIMDNone", "sixstep", 4096, 16384),
			probeCells(128, "SIMDNone", "sixstep64x128", 8192),
		),
	},
	"internal/kernels/avx2_size_specific_probe_amd64.go": {
		Subject: "the sixteen size-specific AVX2 `.s` files against the generic AVX2 fallback " +
			"that would replace them",
		Status: probeOpen,
		Verdict: "**No sweep has been taken.** The file registers both arms and documents the " +
			"protocol; the number is what PLAN.md §1.3's first item exists to produce, and that " +
			"item gates the other two — if the fallback is within noise the whole question " +
			"dissolves and ~26,000 lines go with it. Read the ratio between the two `sizespec` " +
			"rows in each group, never either row against the group incumbent, which is a tuned " +
			"codelet neither arm is competing with. Both arms are RankLevel-demoted to SIMDSSE2 " +
			"so that neither becomes the incumbent it is supposed to be measured against.",
		Record: "PLAN.md, \"1.3 Resolve the unreachable assembly\"",
		Rederiv: "GOFLAGS=-tags=fftprobe GOOD=<canary floor> " +
			"taskset -c 0 ./scripts/bench_gated.sh 4 8 16 32 64 128 256 512 2048 8192",
		// Cells is deliberately empty even though this file registers 20 codelets.
		// Its signatures are "sizespec<N>_<name>_avx2", not "dit<N>_<variant>_<isa>",
		// so probeCell.signature() cannot name them and the quick-reference grids
		// have no cell to mark `p`. Listing them here would mean inventing
		// signatures the registry never uses.
	},
	"internal/kernels/probe_util.go": {
		Subject: "shared helpers for the harnesses in this section (no kernel of its own)",
		Status:  probeSupport,
		Verdict: "Not a measurement. `itoa` builds the per-size signature strings without " +
			"pulling `strconv` into a probe file.",
		Record: "—",
	},
	"internal/asm/arm64/decl_probe.go": {
		Subject: "12 retired scalar-NEON DIT codelets (sizes 8/16/32/64/128/256, both " +
			"precisions) against the vectorized generic NEON kernel that supersedes them",
		Status: probeClosed,
		Verdict: "**Loses every cell, decisively** (Apple M5, 2026-08): each kernel contains " +
			"zero vector instructions — plain FMOVD/FADDD/FMULD scalar arithmetic under a " +
			"\"NEON\" name — and measured 2.7x-5.6x slower than the pure-Go codelet, well past " +
			"the 1.5x bar for \"keep, unregistered\" rather than \"keep at low priority\". " +
			"Twelve rows across `neon_f32_size{8,16,32,128,256}_radix2.s`, " +
			"`neon_f32_size8_radix4.s`, and `neon_f64_size{8,16,32,64,128,256}_radix2.s` moved " +
			"behind this tag together; the generic NEON kernel (`ForwardNEONComplex64Asm` / " +
			"`ForwardNEONComplex128Asm`) already covers every one of these sizes in production " +
			"dispatch (`internal/fft/kernels_arm64_size_specific.go`), so nothing regresses. " +
			"Kept compiled and correctness-tested — see " +
			"`internal/asm/arm64/neon_retired_scalar_probe_test.go` — because a scalar loss on " +
			"one host is still not a structural one under AGENTS.md §2.2: nothing here rules " +
			"out a future ARM core where these particular instruction sequences pipeline " +
			"better than the vector path.",
		Record: "docs/CODELET_BENCHMARKS.md",
		Rederiv: "GOARCH=arm64 GOOS=linux go test -tags fftprobe -run '^$' " +
			"-bench BenchmarkRetiredScalarNEON -benchtime=1s ./internal/asm/arm64/",
	},
}

// probeEntry pairs a probe file found in the tree with its authored note.
type probeEntry struct {
	Path       string
	Constraint string // the //go:build line, verbatim
	Note       probeNote
	Documented bool
}

// scanProbeFiles finds every non-test Go file in the module whose build
// constraint mentions fftprobe. It reads the constraint textually rather than
// evaluating it: the point is to enumerate the probe surface across all
// architectures, not to model what this host would compile.
func scanProbeFiles(root string) ([]probeEntry, error) {
	var found []probeEntry

	err := walkModule(root, ".go", func(rel, abs string) error {
		if strings.HasSuffix(rel, "_test.go") {
			return nil
		}

		constraint, err := buildConstraint(abs)
		if err != nil {
			return err
		}

		if !constraintSelectsProbe(constraint) {
			return nil
		}

		note, ok := probeNotes[rel]
		if !ok {
			note = probeNote{Status: probeUndocumen, Subject: "no entry in cmd/gencodelets/probes.go"}
		}

		found = append(found, probeEntry{Path: rel, Constraint: constraint, Note: note, Documented: ok})

		return nil
	})
	if err != nil {
		return nil, err
	}

	sort.Slice(found, func(i, j int) bool { return found[i].Path < found[j].Path })

	return found, nil
}

// constraintSelectsProbe reports whether a //go:build line puts its file in
// probe builds. Mentioning the tag is not enough: a file gated on `!fftprobe`
// is the *fallback* that ordinary builds take, the exact opposite of a probe,
// and counting it as one demands a verdict for code that has no measurement to
// record. That is not hypothetical — internal/math/transpose_noasm.go took
// that constraint when the AVX2 transposes were probe-gated, and a substring
// test flagged it.
func constraintSelectsProbe(constraint string) bool {
	return strings.Contains(strings.ReplaceAll(constraint, "!fftprobe", ""), "fftprobe")
}

// buildConstraint returns the //go:build line of a Go file, or "" if it has
// none. Only the prologue is scanned; a //go:build comment after the package
// clause is not a constraint.
func buildConstraint(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		switch {
		case strings.HasPrefix(line, "//go:build"):
			return strings.TrimSpace(strings.TrimPrefix(line, "//go:build")), nil
		case strings.HasPrefix(line, "package "):
			return "", nil
		}
	}

	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("read %s: %w", path, err)
	}

	return "", nil
}
