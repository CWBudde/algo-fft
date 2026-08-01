package main

// Assembly symbol census.
//
// The codelet table says what is registered; it says nothing about the ~380
// assembly symbols in internal/asm, of which only some are reachable from a
// production build. Answering "can this .s file be deleted?" by hand is the
// pass that reclassified nine files in one round, and it has been re-derived
// (and got wrong) more than once — so it is computed here and rendered into
// the inventory as a build artifact.
//
// The analysis is deliberately over-approximating: references are matched by
// identifier name, ignoring build tags, packages and shadowing. That means a
// symbol reported *reachable* may in truth be dead, but a symbol reported
// unreachable is genuinely unreferenced by any live non-test code. Since the
// consequence of the report is deletion, the error is on the safe side.

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Symbol reachability verdicts, most to least reachable.
const (
	symLive        = "live"        // reachable from the public API, an init, or a package-level initializer
	symUnreachable = "unreachable" // referenced from non-test Go code that is itself unreachable
	symTestOnly    = "test-only"   // referenced only from _test.go files
	symOrphan      = "orphan"      // declared and referenced by nothing at all
)

// modulePath identifies the module root when walking up from the working
// directory (the generator runs from internal/kernels under go:generate).
const modulePath = "github.com/cwbudde/algo-fft"

var (
	textSymRe  = regexp.MustCompile(`(?m)^TEXT\s+·(\w+)\(SB\)`)
	globlSymRe = regexp.MustCompile(`(?m)^GLOBL\s+·?(\w+)(<>)?\(SB\)`)
	asmRefRe   = regexp.MustCompile(`·(\w+)(<>)?\(SB\)`)
)

// asmSymbol is one TEXT symbol and everything the census knows about it.
type asmSymbol struct {
	Name     string // Go-level name (the ·Name of the TEXT directive)
	File     string // module-relative path of the .s file defining it
	DeclFile string // module-relative path of the bodyless Go func declaration, "" if none
	Status   string // one of the sym* constants
}

// asmFile aggregates the symbols of one .s file.
type asmFile struct {
	Path      string // module-relative
	Group     string // "amd64", "arm64", "x86", …
	Symbols   []asmSymbol
	DeclFiles []string // distinct declaring Go files, module-relative
}

// Live reports how many of the file's symbols are reachable.
func (f asmFile) Live() int { return f.count(symLive) }

func (f asmFile) count(status string) int {
	n := 0

	for _, s := range f.Symbols {
		if s.Status == status {
			n++
		}
	}

	return n
}

// sharedData is a package-visible GLOBL data symbol referenced from a .s file
// other than the one defining it. Deleting the defining file breaks the build,
// and this has happened once already.
type sharedData struct {
	Name     string
	DefFile  string
	UsedFrom []string
}

// census is the whole result set.
type census struct {
	Files  []asmFile
	Groups []string // group names in walk order
	Shared []sharedData
}

// findModuleRoot walks up from dir until it finds the go.mod of this module.
func findModuleRoot(dir string) (string, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return "", fmt.Errorf("resolve %s: %w", dir, err)
	}

	for {
		data, err := os.ReadFile(filepath.Join(abs, "go.mod")) //nolint:gosec // path is derived from cwd
		if err == nil && strings.Contains(string(data), "module "+modulePath) {
			return abs, nil
		}

		parent := filepath.Dir(abs)
		if parent == abs {
			return "", fmt.Errorf("module %s not found above %s", modulePath, dir)
		}

		abs = parent
	}
}

// goIndex is the identifier-level reference graph of the module's Go sources.
type goIndex struct {
	// mentions maps a declaration name to every identifier its non-test
	// bodies mention. Same-named declarations are merged on purpose.
	mentions map[string]map[string]bool
	// declFile maps a bodyless (assembly-bound) func name to its declaring file.
	declFile map[string]string
	// refNonTest / refTest record where a name is mentioned at all.
	refNonTest map[string]bool
	refTest    map[string]bool
	// roots are the names the reachability search starts from.
	roots map[string]bool
}

func newGoIndex() *goIndex {
	return &goIndex{
		mentions:   map[string]map[string]bool{},
		declFile:   map[string]string{},
		refNonTest: map[string]bool{},
		refTest:    map[string]bool{},
		roots:      map[string]bool{},
	}
}

// rootPseudoDecl collects package-level initializer references, which have no
// enclosing function to name.
const rootPseudoDecl = "\x00pkginit"

// addMention records that decl mentions name.
func (g *goIndex) addMention(decl, name string) {
	if g.mentions[decl] == nil {
		g.mentions[decl] = map[string]bool{}
	}

	g.mentions[decl][name] = true
}

// indexFile folds one parsed Go file into the index.
//
// relPath is module-relative; isTest marks _test.go files, whose references
// keep a symbol alive for the test build but not for a production one.
func (g *goIndex) indexFile(relPath string, isTest bool, file *ast.File) {
	inRootPkg := !strings.Contains(relPath, "/")
	isMainCmd := strings.HasPrefix(relPath, "cmd/")

	for _, d := range file.Decls {
		switch decl := d.(type) {
		case *ast.FuncDecl:
			g.indexFunc(relPath, isTest, inRootPkg, isMainCmd, decl)
		case *ast.GenDecl:
			// Package-level var/const initializers run before main; treat
			// their references as roots.
			g.indexIdents(rootPseudoDecl, isTest, decl)

			if !isTest {
				g.roots[rootPseudoDecl] = true
			}
		}
	}
}

func (g *goIndex) indexFunc(relPath string, isTest, inRootPkg, isMainCmd bool, decl *ast.FuncDecl) {
	name := decl.Name.Name

	if decl.Body == nil {
		// A bodyless func is the Go side of an assembly symbol.
		if !isTest {
			g.declFile[name] = relPath
		}

		return
	}

	if !isTest {
		// The public API of the root package, every init, and every cmd
		// entry point are where a production build starts.
		switch {
		case name == "init", name == "main":
			g.roots[name] = true
		case inRootPkg && ast.IsExported(name):
			g.roots[name] = true
		case isMainCmd:
			g.roots[name] = true
		}
	}

	g.indexIdents(name, isTest, decl.Body)
}

// indexIdents records every identifier mentioned in node as a reference from decl.
func (g *goIndex) indexIdents(decl string, isTest bool, node ast.Node) {
	ast.Inspect(node, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}

		if isTest {
			g.refTest[id.Name] = true

			return true
		}

		g.refNonTest[id.Name] = true
		g.addMention(decl, id.Name)

		return true
	})
}

// reachable returns the set of names reachable from the roots.
func (g *goIndex) reachable() map[string]bool {
	live := map[string]bool{}
	queue := make([]string, 0, len(g.roots))

	for r := range g.roots {
		live[r] = true

		queue = append(queue, r)
	}

	for len(queue) > 0 {
		name := queue[len(queue)-1]
		queue = queue[:len(queue)-1]

		for m := range g.mentions[name] {
			if !live[m] {
				live[m] = true

				queue = append(queue, m)
			}
		}
	}

	return live
}

// buildGoIndex parses every Go file of the module, ignoring build constraints
// so that a symbol live only on arm64 is not reported dead on amd64.
func buildGoIndex(root string) (*goIndex, error) {
	index := newGoIndex()
	fset := token.NewFileSet()

	err := walkModule(root, ".go", func(rel, abs string) error {
		file, err := parser.ParseFile(fset, abs, nil, parser.SkipObjectResolution)
		if err != nil {
			return fmt.Errorf("parse %s: %w", rel, err)
		}

		isTest := strings.HasSuffix(rel, "_test.go")
		index.indexFile(rel, isTest, file)

		return nil
	})
	if err != nil {
		return nil, err
	}

	return index, nil
}

// walkModule visits every file with the given extension under root, skipping
// hidden directories, testdata, and any nested module.
func walkModule(root, ext string, visit func(rel, abs string) error) error {
	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return fmt.Errorf("relativize %s: %w", path, relErr)
		}

		rel = filepath.ToSlash(rel)

		if d.IsDir() {
			if rel == "." {
				return nil
			}

			base := d.Name()
			if strings.HasPrefix(base, ".") || base == "testdata" || base == "node_modules" {
				return fs.SkipDir
			}

			if _, statErr := os.Stat(filepath.Join(path, "go.mod")); statErr == nil {
				return fs.SkipDir // nested module, not our sources
			}

			return nil
		}

		if filepath.Ext(path) != ext {
			return nil
		}

		return visit(rel, path)
	})
	if walkErr != nil {
		return fmt.Errorf("walk %s: %w", root, walkErr)
	}

	return nil
}

// runCensus builds the assembly symbol census for the module at root.
func runCensus(root string) (*census, error) {
	index, err := buildGoIndex(root)
	if err != nil {
		return nil, err
	}

	live := index.reachable()

	result := &census{}
	seenGroup := map[string]bool{}
	globlDef := map[string]string{}  // package-visible GLOBL name -> defining file
	asmRefs := map[string][]string{} // symbol -> referencing .s files

	err = walkModule(filepath.Join(root, "internal", "asm"), ".s", func(rel, abs string) error {
		data, readErr := os.ReadFile(abs) //nolint:gosec // path comes from the walk
		if readErr != nil {
			return fmt.Errorf("read %s: %w", rel, readErr)
		}

		modRel := "internal/asm/" + rel
		group := "internal/asm (arch-neutral)"

		if i := strings.IndexByte(rel, '/'); i > 0 {
			group = rel[:i]
		}

		if !seenGroup[group] {
			seenGroup[group] = true

			result.Groups = append(result.Groups, group)
		}

		file := asmFile{Path: modRel, Group: group}
		declSeen := map[string]bool{}

		for _, m := range textSymRe.FindAllStringSubmatch(string(data), -1) {
			name := m[1]
			sym := asmSymbol{
				Name:     name,
				File:     modRel,
				DeclFile: index.declFile[name],
				Status:   classify(name, live, index),
			}
			file.Symbols = append(file.Symbols, sym)

			if sym.DeclFile != "" && !declSeen[sym.DeclFile] {
				declSeen[sym.DeclFile] = true

				file.DeclFiles = append(file.DeclFiles, sym.DeclFile)
			}
		}

		for _, m := range globlSymRe.FindAllStringSubmatch(string(data), -1) {
			if m[2] == "<>" {
				continue // file-scoped, cannot be shared
			}

			globlDef[m[1]] = modRel
		}

		for _, m := range asmRefRe.FindAllStringSubmatch(string(data), -1) {
			asmRefs[m[1]] = append(asmRefs[m[1]], modRel)
		}

		sort.Strings(file.DeclFiles)

		result.Files = append(result.Files, file)

		return nil
	})
	if err != nil {
		return nil, err
	}

	result.Shared = crossFileData(globlDef, asmRefs)

	sort.Slice(result.Files, func(i, j int) bool { return result.Files[i].Path < result.Files[j].Path })

	return result, nil
}

// classify assigns a reachability verdict to one assembly symbol.
func classify(name string, live map[string]bool, index *goIndex) string {
	switch {
	case live[name]:
		return symLive
	case index.refNonTest[name]:
		return symUnreachable
	case index.refTest[name]:
		return symTestOnly
	default:
		return symOrphan
	}
}

// crossFileData reduces the GLOBL definitions and assembly references to the
// data symbols used from a file other than the one defining them.
func crossFileData(globlDef map[string]string, asmRefs map[string][]string) []sharedData {
	var out []sharedData

	for name, def := range globlDef {
		users := map[string]bool{}

		for _, f := range asmRefs[name] {
			if f != def {
				users[f] = true
			}
		}

		if len(users) == 0 {
			continue
		}

		shared := sharedData{Name: name, DefFile: def}
		for f := range users {
			shared.UsedFrom = append(shared.UsedFrom, f)
		}

		sort.Strings(shared.UsedFrom)
		out = append(out, shared)
	}

	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })

	return out
}
