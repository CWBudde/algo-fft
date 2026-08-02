package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestEveryFamilyHasAVerdict is one half of the matrix gate. A new algorithm
// family — a spec Signature with a new variant token, or a tier row with a new
// Algo — arrives with no verdict, and the §1.2 matrix is only worth generating
// if that is a failure rather than a blank cell nobody notices.
func TestEveryFamilyHasAVerdict(t *testing.T) {
	have := map[string]bool{}
	for _, v := range familyVerdicts {
		have[v.Family] = true
	}

	for _, row := range familyMatrix() {
		if !have[row.Family] {
			t.Errorf("family %q is in the tree with no verdict — add one to matrix.go", row.Family)
		}
	}
}

// TestNoVerdictsForVanishedFamilies is the other half. A verdict that outlives
// its family is how a matrix starts describing a tree that no longer exists.
func TestNoVerdictsForVanishedFamilies(t *testing.T) {
	inTree := map[string]bool{}
	for _, row := range familyMatrix() {
		inTree[row.Family] = true
	}

	for _, v := range familyVerdicts {
		if !inTree[v.Family] {
			t.Errorf("verdict for %q, which is no longer a family in the tree", v.Family)
		}
	}
}

// TestFamilyVerdictsAreWellFormed enforces what each status has to carry: a
// settled verdict cites evidence, an open one names the task that owns it, a
// deferred one names the phase. A status with none of those is an opinion.
func TestFamilyVerdictsAreWellFormed(t *testing.T) {
	seen := map[string]bool{}

	for _, v := range familyVerdicts {
		if seen[v.Family] {
			t.Errorf("%s: duplicate verdict", v.Family)
		}

		seen[v.Family] = true

		if v.Verdict == "" {
			t.Errorf("%s: empty verdict", v.Family)
		}

		switch v.Status {
		case famTuned, famClosed:
			if v.Evidence == "" {
				t.Errorf("%s: a %s verdict must cite evidence", v.Family, v.Status)
			}
		case famOpen:
			if v.Tracked == "" {
				t.Errorf("%s: an open family must name the PLAN.md task that owns it", v.Family)
			}
		case famDeferred, famInstrument, famUntested:
			if v.Note == "" {
				t.Errorf("%s: a %s family must name what owns it", v.Family, v.Status)
			}
		default:
			t.Errorf("%s: unknown status %q", v.Family, v.Status)
		}
	}
}

// TestFamilyEvidenceResolves checks that a cited heading is still in the
// document that is supposed to carry it. A verdict pointing at a renamed
// section is the same defect as a stale inventory, one indirection further out.
func TestFamilyEvidenceResolves(t *testing.T) {
	root := repoRoot(t)

	for _, v := range familyVerdicts {
		if v.Evidence == "" {
			continue
		}

		doc, heading, ok := strings.Cut(v.Evidence, ", ")
		if !ok {
			t.Errorf("%s: evidence %q is not a \"doc, heading\" pair", v.Family, v.Evidence)

			continue
		}

		data, err := os.ReadFile(filepath.Join(root, doc))
		if err != nil {
			t.Errorf("%s: %v", v.Family, err)

			continue
		}

		if !strings.Contains(string(data), strings.TrimSpace(heading)) {
			t.Errorf("%s: %s has no section %q", v.Family, doc, heading)
		}
	}
}

// TestOpenFamiliesNameAnOpenPlanTask reuses the census ratchet's rule: the
// quoted fragment must match exactly one PLAN.md checkbox and that checkbox
// must still be open. Closing the §1.2 item therefore forces the family's
// verdict to be written rather than left open forever.
func TestOpenFamiliesNameAnOpenPlanTask(t *testing.T) {
	tasks, err := planTasks(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}

	for _, v := range familyVerdicts {
		if v.Status != famOpen {
			continue
		}

		var matches []planTask

		for _, task := range tasks {
			if strings.Contains(task.Text, v.Tracked) {
				matches = append(matches, task)
			}
		}

		switch {
		case len(matches) == 0:
			t.Errorf("%s: no PLAN.md task contains %q", v.Family, v.Tracked)
		case len(matches) > 1:
			t.Errorf("%s: %q matches %d PLAN.md tasks — quote more of it",
				v.Family, v.Tracked, len(matches))
		case !matches[0].Open:
			t.Errorf("%s: PLAN.md:%d %q is checked off — write the verdict",
				v.Family, matches[0].Line, v.Tracked)
		}
	}
}

// TestUntestedFamiliesAreNotInTheRegistry is the gate that keeps `untested`
// from becoming the bucket everything undecided falls into. The status admits
// that nobody has measured a family and nobody is going to; that is only
// defensible when nothing can reach the family without being asked for it by
// name, and a registered codelet row is exactly the kind of reachability that
// makes a default plan run something unmeasured.
func TestUntestedFamiliesAreNotInTheRegistry(t *testing.T) {
	rows := map[string]int{}
	for _, s := range codeletSpecs {
		rows[variantOf(s.Signature)]++
	}

	for _, v := range familyVerdicts {
		if v.Status != famUntested {
			continue
		}

		if n := rows[v.Family]; n > 0 {
			t.Errorf("%s: %d registered codelet rows, so a default plan can reach it — "+
				"%q is not an available answer", v.Family, n, famUntested)
		}
	}
}

// TestMatrixUnionIncludesTierOnlyFamilies is the point of deriving the family
// axis from two sources. Split-radix, four-step, Rader, Bluestein
// and the recursive decomposition have no codelet rows at all, so a
// specs-only scan omits precisely the families whose emptiness is the finding.
func TestMatrixUnionIncludesTierOnlyFamilies(t *testing.T) {
	fromSpecs := map[string]bool{}
	for _, s := range codeletSpecs {
		fromSpecs[variantOf(s.Signature)] = true
	}

	inMatrix := map[string]bool{}
	for _, row := range familyMatrix() {
		inMatrix[row.Family] = true
	}

	for _, family := range []string{"Split-radix", "Four-step", "Rader", "Bluestein", "Recursive"} {
		if fromSpecs[family] {
			t.Errorf("%s now has codelet rows; this test's premise needs revisiting", family)
		}

		if !inMatrix[family] {
			t.Errorf("%s is missing from the matrix — the tier half of the union is not wired", family)
		}
	}
}

// TestMatrixCellsAgreeWithTheSizeGrids checks the derived half against the
// existing per-size grids: a family cell claims a glyph only if some size cell
// of that family and ISA claims it too. The two views are computed separately
// and must not tell different stories about the same registry.
func TestMatrixCellsAgreeWithTheSizeGrids(t *testing.T) {
	for _, prec := range []int{64, 128} {
		grid := cellVerdicts(prec)

		fromGrid := map[string]map[string]bool{}

		for key, levels := range grid {
			if fromGrid[key.variant] == nil {
				fromGrid[key.variant] = map[string]bool{}
			}

			for level, verdict := range levels {
				fromGrid[key.variant][level+verdict] = true
			}
		}

		for _, row := range familyMatrix() {
			for level, verdict := range row.Cells[prec] {
				if verdict == cellTier {
					continue // no size grid carries the tier glyph
				}

				if !fromGrid[row.Family][level+verdict] {
					t.Errorf("complex%d %s: the matrix claims %q at %s, the size grid does not",
						prec, row.Family, verdict, level)
				}
			}
		}
	}
}

// TestRenderMatrixIncludesEveryFamily guards the wiring into the document.
func TestRenderMatrixIncludesEveryFamily(t *testing.T) {
	var buf bytes.Buffer

	renderMatrix(&buf)

	doc := buf.String()
	for _, row := range familyMatrix() {
		if !strings.Contains(doc, "| "+row.Family+" |") {
			t.Errorf("the matrix omits %s", row.Family)
		}
	}

	if strings.Contains(doc, "no verdict; see") {
		t.Error("the rendered matrix contains a family with no verdict")
	}
}
