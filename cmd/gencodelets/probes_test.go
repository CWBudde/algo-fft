package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestProbeNotesCoverTree is the gate the section exists for: a new
// `-tags fftprobe` file lands with a verdict, or it does not land.
func TestProbeNotesCoverTree(t *testing.T) {
	probes := scanRepoProbes(t)

	if len(probes) == 0 {
		t.Fatal("no probe files found; the scan or the walk is broken")
	}

	for _, p := range probes {
		if !p.Documented {
			t.Errorf("%s carries the fftprobe constraint but has no entry in probeNotes "+
				"(cmd/gencodelets/probes.go) — add its verdict and the sweep that produced it", p.Path)
		}
	}
}

// TestProbeNotesHaveNoStaleEntries catches the other direction: a note for a
// file that was renamed or deleted describes a probe that no longer exists.
func TestProbeNotesHaveNoStaleEntries(t *testing.T) {
	found := map[string]bool{}
	for _, p := range scanRepoProbes(t) {
		found[p.Path] = true
	}

	for path := range probeNotes {
		if !found[path] {
			t.Errorf("probeNotes has an entry for %s, which is not an fftprobe-tagged file in the tree", path)
		}
	}
}

// TestProbeNotesAreComplete guards the fields a reader depends on. A verdict
// with no record is a claim with no evidence, and a measurement probe with no
// re-derive command cannot be re-run — both are how a probe becomes folklore.
func TestProbeNotesAreComplete(t *testing.T) {
	valid := map[string]bool{
		probeOpen: true, probePartial: true, probeClosed: true, probeSupport: true,
	}

	for path, note := range probeNotes {
		if !valid[note.Status] {
			t.Errorf("%s: status %q is not one of the probeStatus constants", path, note.Status)
		}

		if note.Subject == "" || note.Verdict == "" || note.Record == "" {
			t.Errorf("%s: Subject, Verdict and Record are all required", path)
		}

		if note.Status != probeSupport && note.Rederiv == "" {
			t.Errorf("%s: a measurement probe needs a Re-derive command", path)
		}
	}
}

// TestBuildConstraintReadsOnlyThePrologue: a //go:build comment below the
// package clause is documentation, not a constraint, and must not be read as
// one.
func TestBuildConstraintReadsOnlyThePrologue(t *testing.T) {
	dir := t.TempDir()

	cases := map[string]string{
		"tagged.go":  "//go:build amd64 && !purego && fftprobe\n\npackage p\n",
		"plain.go":   "package p\n\nfunc f() {}\n",
		"comment.go": "package p\n\n// A doc comment mentioning //go:build fftprobe\nfunc f() {}\n",
	}

	want := map[string]string{
		"tagged.go":  "amd64 && !purego && fftprobe",
		"plain.go":   "",
		"comment.go": "",
	}

	for name, src := range cases {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(src), 0o600); err != nil {
			t.Fatal(err)
		}

		got, err := buildConstraint(path)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}

		if got != want[name] {
			t.Errorf("buildConstraint(%s) = %q, want %q", name, got, want[name])
		}
	}
}

// TestRenderProbesIncludesEveryFile guards the wiring into the document.
func TestRenderProbesIncludesEveryFile(t *testing.T) {
	probes := scanRepoProbes(t)

	root, err := findModuleRoot("../..")
	if err != nil {
		t.Fatal(err)
	}

	c, err := runCensus(root)
	if err != nil {
		t.Fatal(err)
	}

	doc := string(renderInventory(c, probes))
	if !strings.Contains(doc, "## Probe-Gated Kernels") {
		t.Fatal("inventory is missing the probe section")
	}

	for _, p := range probes {
		if !strings.Contains(doc, p.Path) {
			t.Errorf("probe %s is not mentioned in the inventory", p.Path)
		}
	}
}

func scanRepoProbes(t *testing.T) []probeEntry {
	t.Helper()

	root, err := findModuleRoot("../..")
	if err != nil {
		t.Fatal(err)
	}

	probes, err := scanProbeFiles(root)
	if err != nil {
		t.Fatal(err)
	}

	return probes
}
