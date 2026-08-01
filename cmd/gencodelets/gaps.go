package main

import (
	"bytes"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// Size × ISA coverage gaps, derived from the spec table.
//
// The grids above answer "what is registered". They do not answer the question
// Phase 1 actually asks, which is what a *host* gets: `registry.Lookup` orders
// by rank level and then priority, but gates on SIMDLevel, so it walks past
// every entry the CPU cannot execute and returns the first one it can. A size
// with no AVX2 row is therefore not uncovered — it silently falls back to an
// SSE or pure-Go codelet, and that fallback is invisible in a per-tier tick.
//
// Everything here is computed from codeletSpecs by replaying that lookup, so a
// gap is a generated fact with a name and a number rather than something
// re-noticed each round.

// hostProfile is a CPU as registry.CPUSupports sees it: the set of SIMD levels
// whose codelets it may execute. The levels are feature bits, not a ladder —
// but every real amd64 CPU with a wider level also has the narrower ones, so
// the amd64 profiles are cumulative. NEON shares no bit with any of them.
type hostProfile struct {
	Name   string
	Arch   string
	Top    string   // the widest level this host executes
	Levels []string // every level it executes, including Top
}

//nolint:gochecknoglobals // static configuration for the inventory renderer
var hostProfiles = []hostProfile{
	{
		Name: "pure Go", Arch: "any", Top: "SIMDNone",
		Levels: []string{"SIMDNone"},
	},
	{
		Name: "amd64 SSE2", Arch: "amd64", Top: "SIMDSSE2",
		Levels: []string{"SIMDNone", "SIMDSSE2"},
	},
	{
		Name: "amd64 SSE3", Arch: "amd64", Top: "SIMDSSE3",
		Levels: []string{"SIMDNone", "SIMDSSE2", "SIMDSSE3"},
	},
	{
		Name: "amd64 AVX2+FMA", Arch: "amd64", Top: "SIMDAVX2",
		Levels: []string{"SIMDNone", "SIMDSSE2", "SIMDSSE3", "SIMDAVX2"},
	},
	{
		Name: "amd64 AVX-512", Arch: "amd64", Top: "SIMDAVX512",
		Levels: []string{"SIMDNone", "SIMDSSE2", "SIMDSSE3", "SIMDAVX2", "SIMDAVX512"},
	},
	{
		Name: "arm64 NEON", Arch: "arm64", Top: "SIMDNEON",
		Levels: []string{"SIMDNone", "SIMDNEON"},
	},
}

// levelOrder is the fftypes.SIMDLevel enum order, which is what
// registry.Register sorts by. simdColumns is declared in that same order.
func levelOrder(level string) int {
	for i, l := range simdColumns {
		if l == level {
			return i
		}
	}

	return -1
}

// rankedSpecs returns the specs for one precision and size in the order
// registry.Register leaves them: rank level descending, then priority
// descending. Disabled rows are dropped, matching lookupIn.
func rankedSpecs(prec, size int) []codeletSpec {
	var out []codeletSpec

	for _, s := range codeletSpecs {
		if s.Prec != prec || s.Size != size || s.Priority < 0 {
			continue
		}

		out = append(out, s)
	}

	sort.SliceStable(out, func(i, j int) bool {
		li, lj := levelOrder(rankLevelOf(out[i])), levelOrder(rankLevelOf(out[j]))
		if li != lj {
			return li > lj
		}

		return out[i].Priority > out[j].Priority
	})

	return out
}

// resolve replays registry.Lookup for one host: the first ranked entry whose
// SIMDLevel the host executes.
func (h hostProfile) resolve(prec, size int) (codeletSpec, bool) {
	supported := map[string]bool{}
	for _, l := range h.Levels {
		supported[l] = true
	}

	for _, s := range rankedSpecs(prec, size) {
		if supported[s.SIMDLevel] {
			return s, true
		}
	}

	return codeletSpec{}, false
}

// registeredSizes returns every size with at least one enabled row at this
// precision, ascending.
func registeredSizes(prec int) []int {
	seen := map[int]bool{}

	for _, s := range codeletSpecs {
		if s.Prec == prec && s.Priority >= 0 {
			seen[s.Size] = true
		}
	}

	sizes := make([]int, 0, len(seen))
	for size := range seen {
		sizes = append(sizes, size)
	}

	sort.Ints(sizes)

	return sizes
}

// tierSizes returns the sizes with an enabled row that *executes* at this
// level (SIMDLevel, not RankLevel — a demoted codelet still needs its ISA).
func tierSizes(prec int, level string) []int {
	seen := map[int]bool{}

	for _, s := range codeletSpecs {
		if s.Prec == prec && s.Priority >= 0 && s.SIMDLevel == level {
			seen[s.Size] = true
		}
	}

	sizes := make([]int, 0, len(seen))
	for size := range seen {
		sizes = append(sizes, size)
	}

	sort.Ints(sizes)

	return sizes
}

// hostCoverage is what one host profile gets at one precision.
type hostCoverage struct {
	Host     hostProfile
	Prec     int
	Served   int   // sizes with any codelet
	AtTop    int   // sizes served by a codelet of the host's own top level
	Fallback []int // sizes served, but by a narrower level than the host has
	Ceiling  int   // largest size served at the host's top level; 0 if none
}

func coverageFor(h hostProfile, prec int) hostCoverage {
	cov := hostCoverage{Host: h, Prec: prec}

	for _, size := range registeredSizes(prec) {
		spec, ok := h.resolve(prec, size)
		if !ok {
			continue
		}

		cov.Served++

		if spec.SIMDLevel == h.Top {
			cov.AtTop++

			if size > cov.Ceiling {
				cov.Ceiling = size
			}

			continue
		}

		if h.Top != "SIMDNone" {
			cov.Fallback = append(cov.Fallback, size)
		}
	}

	return cov
}

// renderGaps writes the coverage-gap section: what each host tier actually
// resolves to, and where each tier stops.
func renderGaps(b *bytes.Buffer) {
	b.WriteString(strings.TrimLeft(`
## Size × ISA Coverage Gaps

Derived from the spec table by replaying `+"`registry.Lookup`"+`, which orders by rank
level and priority but gates on `+"`SIMDLevel`"+`. A size with no row at the host's own
ISA is not uncovered: the lookup walks down to the widest level the host can
execute, so the visible symptom of a gap is an AVX-512 machine running an SSE2
codelet, not a missing kernel.

_Fallback_ below counts exactly that — sizes where the host's top ISA has no
codelet and a narrower one answers instead. _Ceiling_ is the largest size with
a codelet at the host's own top ISA; above it every size falls back, whatever
the grids show.

`, "\n"))

	b.WriteString("| Host | complex | Sizes served | At top ISA | Fallback | Top-ISA ceiling |\n")
	b.WriteString("|---|---:|---:|---:|---:|---:|\n")

	for _, h := range hostProfiles {
		for _, prec := range []int{64, 128} {
			cov := coverageFor(h, prec)

			ceiling := "—"
			if cov.Ceiling > 0 {
				ceiling = strconv.Itoa(cov.Ceiling)
			}

			fmt.Fprintf(b, "| %s | %d | %d | %d | %d | %s |\n",
				h.Name, prec, cov.Served, cov.AtTop, len(cov.Fallback), ceiling)
		}
	}

	b.WriteString("\n")
	renderTierGaps(b)
	renderPrecisionGaps(b)
}

// renderTierGaps lists, per ISA tier and precision, the registered sizes that
// tier does not cover — split into holes below its ceiling and sizes above it.
func renderTierGaps(b *bytes.Buffer) {
	b.WriteString("### Not covered, by tier\n\n")
	b.WriteString("A size counts as a gap when some tier at the same precision has a codelet\n")
	b.WriteString("for it and this one does not. Sizes nothing covers are not gaps; they are\n")
	b.WriteString("served by the size-generic kernels listed under Beyond the Codelet Registry.\n\n")
	b.WriteString("| Tier | complex | Rows | Sizes | Range | Holes below ceiling | Above ceiling |\n")
	b.WriteString("|---|---:|---:|---:|---|---|---|\n")

	for _, level := range simdColumns {
		for _, prec := range []int{64, 128} {
			rows := 0

			for _, s := range codeletSpecs {
				if s.Prec == prec && s.SIMDLevel == level && s.Priority >= 0 {
					rows++
				}
			}

			sizes := tierSizes(prec, level)
			if len(sizes) == 0 {
				fmt.Fprintf(b, "| %s | %d | 0 | 0 | — | — | — |\n", simdColumnNames[level], prec)

				continue
			}

			have := map[int]bool{}
			for _, size := range sizes {
				have[size] = true
			}

			ceiling := sizes[len(sizes)-1]

			var holes, above []int

			for _, size := range registeredSizes(prec) {
				if have[size] || size < sizes[0] {
					continue
				}

				if size < ceiling {
					holes = append(holes, size)
				} else {
					above = append(above, size)
				}
			}

			fmt.Fprintf(b, "| %s | %d | %d | %d | %d–%d | %s | %s |\n",
				simdColumnNames[level], prec, rows, len(sizes),
				sizes[0], ceiling, sizeList(holes), sizeList(above))
		}
	}

	b.WriteString("\n")
}

// renderPrecisionGaps lists the sizes a tier covers at one precision only.
// These are the cheapest gaps to close — the twin kernel already exists.
func renderPrecisionGaps(b *bytes.Buffer) {
	b.WriteString("### Covered at one precision only\n\n")
	b.WriteString("Sizes where a tier has a codelet for one precision and not the other. The\n")
	b.WriteString("algorithm is already implemented and measured at that tier, so these are the\n")
	b.WriteString("gaps with a known answer rather than an open question.\n\n")

	found := false

	for _, level := range simdColumns {
		only64 := precOnly(level, 64, 128)
		only128 := precOnly(level, 128, 64)

		if len(only64) == 0 && len(only128) == 0 {
			continue
		}

		found = true

		fmt.Fprintf(b, "- **%s** — complex64 only: %s; complex128 only: %s\n",
			simdColumnNames[level], sizeList(only64), sizeList(only128))
	}

	if !found {
		b.WriteString("None: every tier covers the same sizes at both precisions.\n")
	}

	b.WriteString("\n")
}

// precOnly returns the sizes this tier covers at have but not at want.
func precOnly(level string, have, want int) []int {
	other := map[int]bool{}
	for _, size := range tierSizes(want, level) {
		other[size] = true
	}

	var out []int

	for _, size := range tierSizes(have, level) {
		if !other[size] {
			out = append(out, size)
		}
	}

	return out
}

// sizeList renders a size slice for a table cell.
func sizeList(sizes []int) string {
	if len(sizes) == 0 {
		return "—"
	}

	parts := make([]string, 0, len(sizes))
	for _, size := range sizes {
		parts = append(parts, strconv.Itoa(size))
	}

	return strings.Join(parts, ", ")
}
