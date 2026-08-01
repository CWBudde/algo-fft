package main

import (
	"bytes"
	"strings"
	"testing"
)

// darkSymbols returns every censused symbol that is not reachable from a
// production build, keyed the same way the allowlist is.
func darkSymbols(t *testing.T) map[dispositionKey]asmSymbol {
	t.Helper()

	c, err := runCensus(repoRoot(t))
	if err != nil {
		t.Fatalf("census: %v", err)
	}

	out := map[dispositionKey]asmSymbol{}

	for _, f := range c.Files {
		for _, s := range f.Symbols {
			if s.Status != symLive {
				out[dispositionKey{s.Name, s.File}] = s
			}
		}
	}

	return out
}

// TestEveryDarkSymbolHasADisposition is the ratchet. Every audit round so far
// has rediscovered symbols that went dark since the last one, because nothing
// failed when they did. Now something does: a new orphan / test-only /
// unreachable symbol fails here until it is named in dispositions.go with
// either a terminal reason or an open PLAN task.
func TestEveryDarkSymbolHasADisposition(t *testing.T) {
	index := dispositionIndex()

	for key, sym := range darkSymbols(t) {
		if _, ok := index[key]; !ok {
			t.Errorf("%s (%s) is %s and has no disposition — add one to dispositions.go",
				key.Symbol, key.File, sym.Status)
		}
	}
}

// TestNoStaleDispositions fails when an entry outlives the symbol it excuses:
// the symbol became live, was renamed, or its file was deleted. Without this
// the list only ever grows, which is how an allowlist forms.
func TestNoStaleDispositions(t *testing.T) {
	dark := darkSymbols(t)

	for _, d := range dispositions {
		if _, ok := dark[dispositionKey{d.Symbol, d.File}]; !ok {
			t.Errorf("%s (%s) has a disposition but is no longer dark — remove the entry",
				d.Symbol, d.File)
		}
	}
}

// TestTrackedDispositionsNameAnOpenPlanTask is what keeps a tracked entry from
// being a park. The quoted fragment must match exactly one PLAN.md checkbox,
// and that checkbox must still be open — so closing the PLAN item forces the
// symbol to be resolved rather than the entry to be rewritten.
func TestTrackedDispositionsNameAnOpenPlanTask(t *testing.T) {
	tasks, err := planTasks(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}

	if len(tasks) == 0 {
		t.Fatal("no checkboxes parsed out of PLAN.md — the scanner is broken, not the plan")
	}

	for _, d := range dispositions {
		if d.Kind != dispTracked {
			continue
		}

		var matches []planTask

		for _, task := range tasks {
			if strings.Contains(task.Text, d.Tracked) {
				matches = append(matches, task)
			}
		}

		switch {
		case len(matches) == 0:
			t.Errorf("%s: no PLAN.md task contains %q", d.Symbol, d.Tracked)
		case len(matches) > 1:
			t.Errorf("%s: %q matches %d PLAN.md tasks — quote more of it",
				d.Symbol, d.Tracked, len(matches))
		case !matches[0].Open:
			t.Errorf("%s: PLAN.md:%d %q is checked off — resolve the symbol and drop the entry",
				d.Symbol, matches[0].Line, d.Tracked)
		}
	}
}

// TestDispositionsAreWellFormed checks the shape of each entry: a known kind,
// a reason where the kind is terminal, a tracked task where it is not, and no
// duplicate keys (a second entry for one symbol would silently win or lose
// depending on map order).
func TestDispositionsAreWellFormed(t *testing.T) {
	seen := map[dispositionKey]bool{}

	for _, d := range dispositions {
		key := dispositionKey{d.Symbol, d.File}
		if seen[key] {
			t.Errorf("%s (%s): duplicate disposition", d.Symbol, d.File)
		}

		seen[key] = true

		switch d.Kind {
		case dispTracked:
			if d.Tracked == "" {
				t.Errorf("%s: a tracked disposition needs a PLAN.md task", d.Symbol)
			}

			if d.Reason != "" {
				t.Errorf("%s: a tracked disposition states its reason in PLAN.md, not here", d.Symbol)
			}
		case dispProbed, dispKeep:
			if d.Reason == "" {
				t.Errorf("%s: a terminal disposition needs a reason", d.Symbol)
			}

			if d.Tracked != "" {
				t.Errorf("%s: a terminal disposition has nothing left to track", d.Symbol)
			}
		default:
			t.Errorf("%s: unknown disposition kind %q", d.Symbol, d.Kind)
		}
	}
}

// TestPlanTaskScannerSeesBothStates guards the scanner itself: a parser that
// silently returned only open tasks, or folded a task's continuation lines
// away, would make TestTrackedDispositionsNameAnOpenPlanTask vacuous.
func TestPlanTaskScannerSeesBothStates(t *testing.T) {
	tasks, err := planTasks(repoRoot(t))
	if err != nil {
		t.Fatal(err)
	}

	var open, closed, multiline int

	for _, task := range tasks {
		if task.Open {
			open++
		} else {
			closed++
		}

		if len(task.Text) > 200 {
			multiline++
		}
	}

	if open == 0 || closed == 0 {
		t.Fatalf("scanned %d open and %d closed tasks; expected both", open, closed)
	}

	if multiline == 0 {
		t.Error("no task folded a continuation line — the scanner is dropping them")
	}
}

// TestCensusTableCarriesDispositions guards the wiring into the document: the
// non-live table is the list a reader acts on, so it must say what happens to
// each symbol rather than only that it is dark.
func TestCensusTableCarriesDispositions(t *testing.T) {
	c, err := runCensus(repoRoot(t))
	if err != nil {
		t.Fatalf("census: %v", err)
	}

	var buf bytes.Buffer

	renderCensusNonLive(&buf, c)

	doc := buf.String()
	for _, d := range dispositions {
		if !strings.Contains(doc, d.Label()) {
			t.Errorf("the non-live table omits the disposition of %s (%s)", d.Symbol, d.File)
		}
	}
}
