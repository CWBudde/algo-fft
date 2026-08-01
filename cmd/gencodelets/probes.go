package main

import (
	"bufio"
	"fmt"
	"os"
	"sort"
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
	probeOpen      = "open"             // the question is live; no sweep has answered it
	probePartial   = "partly promoted"  // some cells won and became spec rows; the rest stay here
	probeClosed    = "closed — lost"    // swept and lost everywhere; kept only to stay re-measurable
	probeSupport   = "support"          // shared helpers, not a kernel
	probeUndocumen = "UNDOCUMENTED"     // scanned from the tree with no note here
)

// probeNote is the authored half of a probe entry: what the file measures,
// what the sweep returned, and how to take the number again.
type probeNote struct {
	Subject string // what algorithm the probe measures, one line
	Status  string // one of the probeStatus constants
	Verdict string // what the last sweep returned, and what is still registered
	Record  string // where the sweep is written down
	Rederiv string // the command that reproduces it, "" for support files
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
			"SIMD-level major, so an AVX-512 probe would otherwise *be* the incumbent and the " +
			"sweep would report it 1.000 against itself — which the first run did.",
		Record: "docs/CODELET_BENCHMARKS.md, \"AVX-512 tier (Xeon Gold 5218)\" and " +
			"\"The radix-8 ladder on Skylake-SP — and the stride rule failing to transfer\"",
		Rederiv: "GOFLAGS=-tags=fftprobe taskset -c 0 ./scripts/bench_gated.sh " +
			"64 128 256 512 1024 2048 4096 8192 16384 32768   # AVX-512 host",
	},
	"internal/kernels/radix4_avx2_tail_probe_amd64.go": {
		Subject: "the n = 2*4^k radix-2 tail: fused into the last radix-4 stage, or a separate pass",
		Status:  probeOpen,
		Verdict: "Standing harness rather than a one-shot question — the fused/unfused choice in " +
			"`cmd/gencodelets/specs.go` is empirical, and an empirical constant with no way to " +
			"re-derive it rots. At every 2*4^k size it registers the variant production does " +
			"*not* use, so the comparison is available in both directions, plus a no-tail probe " +
			"that **computes the wrong answer on purpose**: the gap to the incumbent is the whole " +
			"cost of the tail and therefore the most any fusion could ever recover — 9-15% " +
			"measured, against the 4-6% the fusion actually gets where it wins.",
		Record:  "docs/CODELET_BENCHMARKS.md, \"n = 128 closed, and the odd-exponent question settled (2026-08-01)\"",
		Rederiv: "GOFLAGS=-tags=fftprobe GOOD=<canary floor> taskset -c 0 ./scripts/bench_gated.sh 128 512 2048 8192 32768",
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
	"internal/kernels/probe_util.go": {
		Subject: "shared helpers for the harnesses above (no kernel of its own)",
		Status:  probeSupport,
		Verdict: "Not a measurement. `itoa` builds the per-size signature strings without " +
			"pulling `strconv` into a probe file.",
		Record: "—",
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

		if !strings.Contains(constraint, "fftprobe") {
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
