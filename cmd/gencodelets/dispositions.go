package main

// The census ratchet.
//
// census.go answers "which assembly symbols are dark". That is a snapshot, and
// a snapshot regrows: every audit round so far has rediscovered symbols that
// went dark between rounds, because nothing failed when they did. This file is
// the gate — every non-live symbol must be named here, and an entry is not free
// to write.
//
// An entry is admissible in exactly two ways:
//
//   - terminal (dispProbed, dispKeep) — the symbol is *meant* to be unreachable
//     and will stay that way, with the reason written down; or
//   - tracked (dispTracked) — an **open** PLAN.md checkbox will decide and apply
//     its §2.2 disposition, quoted verbatim so the link cannot rot.
//
// Both directions are checked by dispositions_test.go. A new dark symbol fails
// TestEveryDarkSymbolHasADisposition; an entry that outlives its symbol fails
// TestNoStaleDispositions; and a tracked entry whose PLAN task does not exist,
// or has since been checked off, fails TestTrackedDispositionsNameAnOpenPlanTask
// — so closing the PLAN item forces the symbol to actually be resolved rather
// than the entry to be re-parked.
//
// That last rule is what separates this from an allowlist. A tracked entry is
// debt with a creditor: it cannot be added without an open task, and it cannot
// survive that task being closed.

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Disposition kinds.
const (
	// dispProbed — reachable only under `-tags fftprobe`. The §2.2 middle
	// verdict, already applied; the Probe-Gated Kernels section carries the
	// measurement. Terminal.
	dispProbed = "probe-gated"
	// dispKeep — intentionally unreachable and staying that way. Terminal, and
	// deliberately hard to justify: infrastructure, not kernels.
	dispKeep = "keep"
	// dispTracked — an open PLAN.md task decides and applies the disposition.
	dispTracked = "tracked"
)

// disposition is one allowlist entry, keyed by (Symbol, File) because the same
// name can be defined once per architecture.
type disposition struct {
	Symbol  string // the ·Name of the TEXT directive
	File    string // module-relative path of the defining .s file
	Kind    string // one of the disp* constants
	Tracked string // dispTracked: verbatim fragment of the open PLAN.md task
	Reason  string // terminal kinds: why it is unreachable on purpose
}

// dispositions is the ratchet's allowlist. Adding a row here is a claim that
// the tests below verify; see the file comment for what makes a row admissible.
//
//nolint:gochecknoglobals // static configuration, verified by dispositions_test.go
var dispositions = []disposition{
	// The AVX2 radix-3/radix-5 butterflies. Dark because
	// {forward,inverse}Radix3Complex64 has no non-test caller at all, so the
	// whole chain below it is unreachable — a dispatch toggle would not fix it.
	{
		Symbol: "Butterfly3ForwardAVX2Complex64", File: "internal/asm/amd64/avx2_f32_radix3.s",
		Kind: dispTracked, Tracked: "Give the AVX2 radix-3/radix-5 butterflies a disposition.",
	},
	{
		Symbol: "Butterfly3InverseAVX2Complex64", File: "internal/asm/amd64/avx2_f32_radix3.s",
		Kind: dispTracked, Tracked: "Give the AVX2 radix-3/radix-5 butterflies a disposition.",
	},
	{
		Symbol: "Butterfly5ForwardAVX2Complex64", File: "internal/asm/amd64/avx2_f32_radix5.s",
		Kind: dispTracked, Tracked: "Give the AVX2 radix-3/radix-5 butterflies a disposition.",
	},
	{
		Symbol: "Butterfly5InverseAVX2Complex64", File: "internal/asm/amd64/avx2_f32_radix5.s",
		Kind: dispTracked, Tracked: "Give the AVX2 radix-3/radix-5 butterflies a disposition.",
	},

	// The AVX2 transposes: correct and tested, but their only route in —
	// math.TransposeSquareOutOfPlaceComplex64 and the two fused-twiddle
	// dispatchers beside it — has no non-test caller, because the six-step and
	// four-step work that would use them is Phase 3. Probe-gated on 2026-08-02
	// (PLAN.md §1.3): internal/math/transpose_amd64.go now carries `fftprobe`,
	// so an ordinary amd64 build takes the pure-Go fallbacks and never links a
	// path it cannot exercise. Phase 3 closes this by removing the tag.
	{
		Symbol: "Transpose128x128Complex64AVX2Asm", File: "internal/asm/amd64/avx2_f32_transpose128x128.s",
		Kind: dispProbed, Reason: "called only from internal/math/transpose_amd64.go (-tags fftprobe); Phase 3 wires it",
	},
	{
		Symbol: "TransposeTwiddle128x128Complex64AVX2Asm", File: "internal/asm/amd64/avx2_f32_transpose128x128.s",
		Kind: dispProbed, Reason: "called only from internal/math/transpose_amd64.go (-tags fftprobe); Phase 3 wires it",
	},
	{
		Symbol: "TransposeTwiddleConj128x128Complex64AVX2Asm", File: "internal/asm/amd64/avx2_f32_transpose128x128.s",
		Kind: dispProbed, Reason: "called only from internal/math/transpose_amd64.go (-tags fftprobe); Phase 3 wires it",
	},
	{
		Symbol: "Transpose64x64Complex64AVX2Asm", File: "internal/asm/amd64/avx2_f32_transpose64x64.s",
		Kind: dispProbed, Reason: "called only from internal/math/transpose_amd64.go (-tags fftprobe); Phase 3 wires it",
	},
	{
		Symbol: "TransposeTwiddle64x64Complex64AVX2Asm", File: "internal/asm/amd64/avx2_f32_transpose64x64.s",
		Kind: dispProbed, Reason: "called only from internal/math/transpose_amd64.go (-tags fftprobe); Phase 3 wires it",
	},
	{
		Symbol: "TransposeTwiddleConj64x64Complex64AVX2Asm", File: "internal/asm/amd64/avx2_f32_transpose64x64.s",
		Kind: dispProbed, Reason: "called only from internal/math/transpose_amd64.go (-tags fftprobe); Phase 3 wires it",
	},

	// The complex128 generic AVX2 radix-4 pair: already probe-gated. Their only
	// Go callers live in internal/fft/radix4_c128_probe_amd64.go, which is
	// `//go:build fftprobe` — the census ignores build tags, so it sees the
	// callers and reports the second-order verdict rather than "orphan".
	{
		Symbol: "ForwardAVX2Complex128Radix4Asm", File: "internal/asm/amd64/avx2_f64_generic_radix4_even.s",
		Kind: dispProbed, Reason: "called only from radix4_c128_probe_amd64.go (-tags fftprobe); awaiting the Xeon sweep",
	},
	{
		Symbol: "InverseAVX2Complex128Radix4Asm", File: "internal/asm/amd64/avx2_f64_generic_radix4_even.s",
		Kind: dispProbed, Reason: "called only from radix4_c128_probe_amd64.go (-tags fftprobe); awaiting the Xeon sweep",
	},
	{
		Symbol: "ForwardAVX2Complex128Radix4MixedAsm", File: "internal/asm/amd64/avx2_f64_generic_radix4_odd.s",
		Kind: dispProbed, Reason: "called only from radix4_c128_probe_amd64.go (-tags fftprobe); awaiting the Xeon sweep",
	},
	{
		Symbol: "InverseAVX2Complex128Radix4MixedAsm", File: "internal/asm/amd64/avx2_f64_generic_radix4_odd.s",
		Kind: dispProbed, Reason: "called only from radix4_c128_probe_amd64.go (-tags fftprobe); awaiting the Xeon sweep",
	},

	// The toolchain stub: an empty RET per architecture, wrapped by asm.Stub,
	// which nothing calls. It is the smoke test that the package's assembly
	// links at all, so it is unreachable by construction and stays.
	{
		Symbol: "stubAsm", File: "internal/asm/stub_amd64.s",
		Kind: dispKeep, Reason: "assembly-linkage smoke test for package asm; unreachable by construction",
	},
	{
		Symbol: "stubAsm", File: "internal/asm/stub_arm64.s",
		Kind: dispKeep, Reason: "assembly-linkage smoke test for package asm; unreachable by construction",
	},

	// The 386 size-8/16 SSE1 kernels were deleted on 2026-08-02 (PLAN.md §1.3):
	// the SSE1 dispatch in internal/fft/kernels_386_asm.go special-cases only
	// n = 2 and 4, so nothing could reach them whatever their 1:1 thunks said.
	// Their entries are gone rather than converted to dispKeep, because a
	// disposition for a symbol that no longer exists fails the census tests
	// exactly as a dark symbol without one does.

	// The 12 retired scalar-NEON DIT codelets (2026-08-02, PLAN.md §2.2): each
	// contains zero vector instructions and measured 2.7x-5.6x slower than the
	// pure-Go codelet on an Apple M5, so they moved behind `-tags fftprobe`
	// rather than staying at a selectable priority. See
	// cmd/gencodelets/probes.go's internal/asm/arm64/decl_probe.go entry for the
	// full verdict; internal/asm/arm64/neon_retired_scalar_probe_test.go is the
	// correctness test and comparison benchmark that keeps the question
	// re-measurable.
	{
		Symbol: "ForwardNEONSize8Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size8_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize8Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size8_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize8Radix2Complex128Asm", File: "internal/asm/arm64/neon_f64_size8_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize8Radix2Complex128Asm", File: "internal/asm/arm64/neon_f64_size8_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize8Radix4Complex64Asm", File: "internal/asm/arm64/neon_f32_size8_radix4.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize8Radix4Complex64Asm", File: "internal/asm/arm64/neon_f32_size8_radix4.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize16Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size16_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize16Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size16_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize16Complex128Asm", File: "internal/asm/arm64/neon_f64_size16_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize16Complex128Asm", File: "internal/asm/arm64/neon_f64_size16_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize32Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size32_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize32Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size32_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize32Complex128Asm", File: "internal/asm/arm64/neon_f64_size32_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize32Complex128Asm", File: "internal/asm/arm64/neon_f64_size32_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize64Radix2Complex128Asm", File: "internal/asm/arm64/neon_f64_size64_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize64Radix2Complex128Asm", File: "internal/asm/arm64/neon_f64_size64_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize128Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size128_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize128Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size128_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize128Radix2Complex128Asm", File: "internal/asm/arm64/neon_f64_size128_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize128Radix2Complex128Asm", File: "internal/asm/arm64/neon_f64_size128_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize256Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size256_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize256Radix2Complex64Asm", File: "internal/asm/arm64/neon_f32_size256_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "ForwardNEONSize256Radix2Complex128Asm", File: "internal/asm/arm64/neon_f64_size256_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
	{
		Symbol: "InverseNEONSize256Radix2Complex128Asm", File: "internal/asm/arm64/neon_f64_size256_radix2.s",
		Kind: dispProbed, Reason: "called only from neon_retired_scalar_probe_test.go (-tags fftprobe); scalar, lost to pure Go 2.7x-5.6x on Apple M5",
	},
}

// dispositionKey identifies an entry. The same symbol name is defined once per
// architecture (stubAsm twice), so the file is part of the key.
type dispositionKey struct{ Symbol, File string }

// dispositionIndex keys the allowlist for lookup.
func dispositionIndex() map[dispositionKey]disposition {
	out := make(map[dispositionKey]disposition, len(dispositions))
	for _, d := range dispositions {
		out[dispositionKey{d.Symbol, d.File}] = d
	}

	return out
}

// Label renders the entry for a table cell.
func (d disposition) Label() string {
	if d.Kind == dispTracked {
		return "tracked: " + strings.TrimSuffix(d.Tracked, ".")
	}

	return d.Kind + ": " + d.Reason
}

// planTask is one PLAN.md checkbox, flattened to a single line.
type planTask struct {
	Text string
	Open bool
	Line int
}

// planTasks reads every checkbox in PLAN.md. Continuation lines are folded in,
// and Markdown emphasis and code spans are stripped, so a tracked fragment can
// be quoted as plain prose and still match a bolded task title.
func planTasks(root string) ([]planTask, error) {
	path := filepath.Join(root, "PLAN.md")

	file, err := os.Open(path) //nolint:gosec // path is derived from the module root
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer file.Close()

	var (
		tasks   []planTask
		current *planTask
	)

	flush := func() {
		if current != nil {
			current.Text = strings.Join(strings.Fields(current.Text), " ")
			tasks = append(tasks, *current)
			current = nil
		}
	}

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for line := 1; scanner.Scan(); line++ {
		raw := scanner.Text()
		trimmed := strings.TrimLeft(raw, " ")

		switch {
		case strings.HasPrefix(trimmed, "- [ ] "), strings.HasPrefix(trimmed, "- [x] "):
			flush()

			current = &planTask{
				Text: stripMarkup(trimmed[len("- [ ] "):]),
				Open: strings.HasPrefix(trimmed, "- [ ] "),
				Line: line,
			}
		case current == nil:
			// Not inside a task.
		case trimmed == "" || raw == trimmed || strings.HasPrefix(trimmed, "- "):
			// A blank line, an unindented line, or a sibling bullet ends it.
			flush()
		default:
			current.Text += " " + stripMarkup(trimmed)
		}
	}

	flush()

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	return tasks, nil
}

// stripMarkup removes the bold and code-span markers that would otherwise force
// every tracked fragment to reproduce PLAN.md's formatting verbatim. Underscores
// are left alone: they occur inside file names far more often than as emphasis.
func stripMarkup(s string) string {
	return strings.NewReplacer("**", "", "`", "").Replace(s)
}
