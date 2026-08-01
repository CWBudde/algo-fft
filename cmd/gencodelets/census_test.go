package main

import (
	"go/parser"
	"go/token"
	"strings"
	"testing"
)

// parseIndex folds a set of in-memory Go files into a fresh index. Keys are
// module-relative paths, so a key without a slash is the root package.
func parseIndex(t *testing.T, files map[string]string) *goIndex {
	t.Helper()

	index := newGoIndex()
	fset := token.NewFileSet()

	for rel, src := range files {
		file, err := parser.ParseFile(fset, rel, src, parser.SkipObjectResolution)
		if err != nil {
			t.Fatalf("parse %s: %v", rel, err)
		}

		index.indexFile(rel, strings.HasSuffix(rel, "_test.go"), file)
	}

	return index
}

func TestTextSymbolExtraction(t *testing.T) {
	src := `#include "textflag.h"

TEXT ·ForwardThingAsm(SB), NOSPLIT, $0-97
	VZEROUPPER
	RET

// A commented-out TEXT ·NotASymbol(SB) must not count.
TEXT ·InverseThingAsm(SB), NOSPLIT, $0-97
	RET
`

	got := textSymRe.FindAllStringSubmatch(src, -1)
	if len(got) != 2 {
		t.Fatalf("want 2 symbols, got %d: %v", len(got), got)
	}

	if got[0][1] != "ForwardThingAsm" || got[1][1] != "InverseThingAsm" {
		t.Errorf("unexpected symbols: %q, %q", got[0][1], got[1][1])
	}
}

func TestGloblScopeDistinguished(t *testing.T) {
	src := `GLOBL ·sharedTable(SB), RODATA, $64
GLOBL fileLocal<>(SB), RODATA, $64
`

	got := globlSymRe.FindAllStringSubmatch(src, -1)
	if len(got) != 2 {
		t.Fatalf("want 2 GLOBL matches, got %d", len(got))
	}

	if got[0][1] != "sharedTable" || got[0][2] != "" {
		t.Errorf("package-visible symbol misparsed: %v", got[0])
	}

	if got[1][2] != "<>" {
		t.Errorf("file-scoped symbol not marked: %v", got[1])
	}
}

// TestReachabilityIsTransitive is the property the whole census rests on: an
// assembly symbol behind a thunk that no live code calls must not count as
// reachable just because the thunk mentions it.
func TestReachabilityIsTransitive(t *testing.T) {
	index := parseIndex(t, map[string]string{
		"plan.go": `package algofft
func Forward() { liveHelper() }
`,
		"internal/fft/dispatch.go": `package fft
func liveHelper() { LiveAsm() }
func deadThunk() { DeadAsm() }
`,
		"internal/asm/amd64/decl.go": `package amd64
func LiveAsm()
func DeadAsm()
func OrphanAsm()
`,
	})

	live := index.reachable()

	if !live["LiveAsm"] {
		t.Error("LiveAsm should be reachable from the root package's exported API")
	}

	if live["DeadAsm"] {
		t.Error("DeadAsm is only referenced by an unreachable thunk; it must not be live")
	}

	for name, want := range map[string]string{
		"LiveAsm":   symLive,
		"DeadAsm":   symUnreachable,
		"OrphanAsm": symOrphan,
	} {
		if got := classify(name, live, index); got != want {
			t.Errorf("classify(%s) = %s, want %s", name, got, want)
		}
	}
}

func TestTestOnlyReferencesDoNotMakeSymbolsLive(t *testing.T) {
	index := parseIndex(t, map[string]string{
		"plan.go":                    "package algofft\nfunc Forward() {}\n",
		"internal/asm/amd64/decl.go": "package amd64\nfunc ProbeAsm()\n",
		"internal/fft/probe_test.go": "package fft\nfunc TestProbe() { ProbeAsm() }\n",
	})

	if got := classify("ProbeAsm", index.reachable(), index); got != symTestOnly {
		t.Errorf("classify(ProbeAsm) = %s, want %s", got, symTestOnly)
	}
}

// TestPackageInitializersAreRoots covers the registry pattern: codelets are
// wired up from init functions and package-level tables, never called from the
// root package by name.
func TestPackageInitializersAreRoots(t *testing.T) {
	index := parseIndex(t, map[string]string{
		"internal/kernels/reg.go": `package kernels
func init() { registerAll() }
func registerAll() { CodeletAsm() }
var table = []func(){ TableAsm }
`,
		"internal/asm/amd64/decl.go": "package amd64\nfunc CodeletAsm()\nfunc TableAsm()\n",
	})

	live := index.reachable()

	for _, name := range []string{"CodeletAsm", "TableAsm"} {
		if !live[name] {
			t.Errorf("%s should be reachable via init / package-level initializer", name)
		}
	}
}

func TestDeclFileRecordedForBodylessFuncsOnly(t *testing.T) {
	index := parseIndex(t, map[string]string{
		"internal/asm/amd64/decl.go": "package amd64\nfunc BoundAsm()\n",
		"internal/fft/dispatch.go":   "package fft\nfunc notAsm() {}\n",
	})

	if got := index.declFile["BoundAsm"]; got != "internal/asm/amd64/decl.go" {
		t.Errorf("declFile[BoundAsm] = %q", got)
	}

	if _, ok := index.declFile["notAsm"]; ok {
		t.Error("a func with a body must not be recorded as an assembly declaration")
	}
}

func TestCrossFileDataOnlyReportsForeignUsers(t *testing.T) {
	def := map[string]string{
		"sharedTable": "internal/asm/amd64/tables.s",
		"localTable":  "internal/asm/amd64/kernel.s",
	}
	refs := map[string][]string{
		"sharedTable": {"internal/asm/amd64/tables.s", "internal/asm/amd64/kernel.s"},
		"localTable":  {"internal/asm/amd64/kernel.s"},
	}

	got := crossFileData(def, refs)
	if len(got) != 1 {
		t.Fatalf("want 1 shared symbol, got %d: %+v", len(got), got)
	}

	if got[0].Name != "sharedTable" || len(got[0].UsedFrom) != 1 {
		t.Errorf("unexpected shared entry: %+v", got[0])
	}
}

// TestCensusOnRepo runs the real analysis over this module. It asserts the
// invariants a reader of the inventory relies on, not any particular verdict —
// verdicts move as kernels land.
func TestCensusOnRepo(t *testing.T) {
	root, err := findModuleRoot("../..")
	if err != nil {
		t.Fatal(err)
	}

	c, err := runCensus(root)
	if err != nil {
		t.Fatal(err)
	}

	if len(c.Files) == 0 || len(c.Groups) == 0 {
		t.Fatal("census found no assembly files")
	}

	statuses := map[string]bool{symLive: true, symUnreachable: true, symTestOnly: true, symOrphan: true}
	total, live := 0, 0
	seen := map[string]bool{}

	for _, f := range c.Files {
		if seen[f.Path] {
			t.Errorf("duplicate file in census: %s", f.Path)
		}

		seen[f.Path] = true

		for _, s := range f.Symbols {
			total++

			if !statuses[s.Status] {
				t.Errorf("%s: bad status %q", s.Name, s.Status)
			}

			if s.Status == symLive {
				live++
			}
		}
	}

	if total < 300 {
		t.Errorf("expected the tree's ~380 TEXT symbols, found %d", total)
	}

	// The dispatch entry points must never read as dead; if they do, the
	// root set is wrong and every "unreachable" verdict below is noise.
	for _, name := range []string{"ForwardAVX2Complex64Asm", "InverseAVX2Complex64Asm"} {
		if !symbolHasStatus(c, name, symLive) {
			t.Errorf("%s must be live; the reachability roots are broken", name)
		}
	}

	if live*4 < total {
		t.Errorf("only %d of %d symbols live — suspiciously low, check the roots", live, total)
	}
}

func symbolHasStatus(c *census, name, status string) bool {
	for _, f := range c.Files {
		for _, s := range f.Symbols {
			if s.Name == name {
				return s.Status == status
			}
		}
	}

	return false
}

// TestRenderInventoryIncludesCensus guards the wiring, since the document is
// what anyone actually reads.
func TestRenderInventoryIncludesCensus(t *testing.T) {
	root, err := findModuleRoot("../..")
	if err != nil {
		t.Fatal(err)
	}

	c, err := runCensus(root)
	if err != nil {
		t.Fatal(err)
	}

	doc := string(renderInventory(c, nil))
	for _, want := range []string{
		"## Assembly Symbol Census",
		"### Files with no live symbol",
		"### Symbols not reachable from a production build",
		"### Data symbols shared across files",
	} {
		if !strings.Contains(doc, want) {
			t.Errorf("inventory is missing section %q", want)
		}
	}
}
