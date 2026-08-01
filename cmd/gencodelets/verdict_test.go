package main

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// knownPriorityTies are the (precision, size, tier) groups where two rows
// share the top priority. `registry.Register` sorts with sort.Slice, which is
// NOT stable, so a tie is resolved arbitrarily — the selected kernel is
// whatever the sort happened to leave first. Each entry here is a decision
// nobody has made, waiting on a measurement (PLAN.md §1.5); the list must
// shrink, never grow.
//
//nolint:gochecknoglobals // test fixture
var knownPriorityTies = map[rankGroup]string{
	{prec: 64, size: 128, level: "SIMDSSE3"}: "dit128_radix4_then2_sse3 vs dit128_radix2_sse3, both at 17; " +
		"the complex128 SSE2 pair at the same size prefers radix4_then2 (18 over 17), which is a hint, not a number",
}

// TestSelectableIsOnePerRankTier is the property the ✓ column claims: within
// one (precision, size, rank tier) exactly one codelet is what Lookup returns.
// Two ticks in one tier would mean the grid is asserting something the
// registry cannot do.
func TestSelectableIsOnePerRankTier(t *testing.T) {
	for _, prec := range []int{64, 128} {
		selectable := map[rankGroup]int{}

		for _, s := range codeletSpecs {
			if s.Prec != prec || s.Priority < 0 {
				continue
			}

			g := rankGroup{prec: prec, size: s.Size, level: rankLevelOf(s)}

			best := 0
			for _, peer := range codeletSpecs {
				if peer.Prec == prec && peer.Size == s.Size && rankLevelOf(peer) == rankLevelOf(s) &&
					peer.Priority > best {
					best = peer.Priority
				}
			}

			if s.Priority == best {
				selectable[g]++
			}
		}

		for g, n := range selectable {
			if n > 1 && knownPriorityTies[g] == "" {
				t.Errorf("complex%d n=%d tier %s: %d rows share the top priority — "+
					"registry.Register sorts with an unstable sort.Slice, so the winner is "+
					"arbitrary; give the loser a lower priority or add a knownPriorityTies entry",
					prec, g.size, g.level, n)
			}
		}

		for g := range knownPriorityTies {
			if g.prec == prec && selectable[g] < 2 {
				t.Errorf("knownPriorityTies has an entry for complex%d n=%d %s, which is no longer a tie — "+
					"delete it", prec, g.size, g.level)
			}
		}
	}
}

// TestRankLevelDemotesOnly enforces the AGENTS.md rule the AVX-512 probe
// header records paying for: RankLevel exists to demote a wide-ISA codelet
// into a tier where its priority is comparable, never to promote a narrow one
// into a tier it cannot win.
func TestRankLevelDemotesOnly(t *testing.T) {
	order := map[string]int{}
	for i, level := range simdColumns {
		order[level] = i
	}

	for _, s := range codeletSpecs {
		if s.RankLevel == "" {
			continue
		}

		if order[s.RankLevel] > order[s.SIMDLevel] {
			t.Errorf("%s (complex%d n=%d) ranks at %s but executes at %s — that is a promotion",
				s.Signature, s.Prec, s.Size, s.RankLevel, s.SIMDLevel)
		}
	}
}

// TestProbeCellsDoNotShadowSpecRows encodes the mistake the AVX-512 probe
// header records costing a full sweep: the same signature registered twice in
// one group makes the sweep report a kernel against itself, at a ratio of
// exactly 1.000.
func TestProbeCellsDoNotShadowSpecRows(t *testing.T) {
	registered := map[string]bool{}
	for _, s := range codeletSpecs {
		registered[strconv.Itoa(s.Prec)+"/"+s.Signature] = true
	}

	for path, note := range probeNotes {
		for _, c := range note.Cells {
			if registered[strconv.Itoa(c.Prec)+"/"+c.signature()] {
				t.Errorf("%s registers %s at complex%d, which is already a spec row — "+
					"a sweep would report it against itself", path, c.signature(), c.Prec)
			}
		}
	}
}

// TestProbeCellsMatchTheirFile checks the authored cells against the probe
// source. It cannot evaluate the tail probe's per-size conditional, so it
// checks the two things that do drift: a size that is no longer registered,
// and a signature that was renamed.
func TestProbeCellsMatchTheirFile(t *testing.T) {
	root, err := findModuleRoot("../..")
	if err != nil {
		t.Fatal(err)
	}

	for path, note := range probeNotes {
		if len(note.Cells) == 0 {
			continue
		}

		src, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
		if err != nil {
			t.Fatal(err)
		}

		text := string(src)

		for _, c := range note.Cells {
			fragment := "_" + c.Variant + "_" + signatureISA[c.Level]
			if !strings.Contains(text, fragment) {
				t.Errorf("%s: no signature fragment %q — the cell for complex%d n=%d is stale",
					path, fragment, c.Prec, c.Size)
			}

			if !containsSizeLiteral(text, c.Size) {
				t.Errorf("%s: size %d appears in no size table — the %s cell is stale",
					path, c.Size, c.Variant)
			}
		}
	}
}

// TestGridMarksProbeCells guards the join: a probe cell with no spec row must
// reach the rendered grid as `p`, which is the whole point of the column.
func TestGridMarksProbeCells(t *testing.T) {
	cells := cellVerdicts(64)

	// The pure-Go radix-8 ladder at n = 64 lost and stayed a probe.
	key := gridKey{size: 64, variant: variantOf("dit64_radix8ladder_generic")}
	if got := cells[key]["SIMDNone"]; got != cellProbe {
		t.Errorf("dit64_radix8ladder_generic renders %q, want %q", got, cellProbe)
	}

	// A promoted ladder cell must outrank the probe marking, not be masked by it.
	promoted := gridKey{size: 256, variant: variantOf("dit256_radix8ladder_generic")}
	if got := cells[promoted]["SIMDNone"]; got != cellSelectable && got != cellCandidate {
		t.Errorf("promoted dit256_radix8ladder_generic renders %q, want a registered verdict", got)
	}
}

// containsSizeLiteral reports whether the source mentions n as a standalone
// integer literal, i.e. in a size table rather than inside a larger number.
func containsSizeLiteral(text string, size int) bool {
	lit := strconv.Itoa(size)

	for i := 0; ; {
		j := strings.Index(text[i:], lit)
		if j < 0 {
			return false
		}

		start := i + j
		end := start + len(lit)

		if !isDigitByte(text, start-1) && !isDigitByte(text, end) {
			return true
		}

		i = start + 1
	}
}

func isDigitByte(text string, i int) bool {
	if i < 0 || i >= len(text) {
		return false
	}

	return text[i] >= '0' && text[i] <= '9'
}
