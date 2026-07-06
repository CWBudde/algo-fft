package asm

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// textSymbolRe matches Plan9 assembly TEXT directives like
// "TEXT ·ForwardAVX2Size64Complex64Asm(SB), NOSPLIT, $0-97".
var textSymbolRe = regexp.MustCompile(`(?m)^TEXT\s+·([A-Za-z0-9_]+)\(SB\)`)

// TestNoescapeDeclsHaveTextSymbols guards against decl↔assembly drift: every
// body-less Go function declaration in internal/asm (and its per-arch
// subdirectories) must have a matching TEXT symbol in a .s file of the same
// directory. A declaration without a TEXT body links only until someone calls
// it, then fails at link time — or worse, hides dead API surface.
//
// The check parses files irrespective of build tags, so amd64, arm64, and x86
// assembly are all verified from any platform.
func TestNoescapeDeclsHaveTextSymbols(t *testing.T) {
	t.Parallel()

	dirs, err := assemblyDirs(".")
	if err != nil {
		t.Fatalf("scanning directories: %v", err)
	}

	if len(dirs) == 0 {
		t.Fatal("no directories with .s files found under internal/asm")
	}

	for _, dir := range dirs {
		checkDeclTextParity(t, dir)
	}
}

// assemblyDirs returns root and any subdirectories containing .s files.
func assemblyDirs(root string) ([]string, error) {
	var dirs []string

	err := filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}

		if !d.IsDir() {
			return nil
		}

		entries, err := os.ReadDir(path)
		if err != nil {
			return fmt.Errorf("reading %s: %w", path, err)
		}

		for _, entry := range entries {
			if strings.HasSuffix(entry.Name(), ".s") {
				dirs = append(dirs, path)
				break
			}
		}

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walking %s: %w", root, err)
	}

	return dirs, nil
}

func checkDeclTextParity(t *testing.T, dir string) {
	t.Helper()

	decls, err := bodylessDecls(dir)
	if err != nil {
		t.Fatalf("%s: parsing Go declarations: %v", dir, err)
	}

	texts, err := textSymbols(dir)
	if err != nil {
		t.Fatalf("%s: scanning TEXT symbols: %v", dir, err)
	}

	for name, pos := range decls {
		if !texts[name] {
			t.Errorf("%s: declaration %s has no TEXT symbol in any .s file (declared at %s)", dir, name, pos)
		}
	}
}

// bodylessDecls parses every .go file in dir (ignoring build tags) and
// returns the names of function declarations without bodies.
func bodylessDecls(dir string) (map[string]string, error) {
	decls := make(map[string]string)

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", dir, err)
	}

	fset := token.NewFileSet()

	for _, entry := range entries {
		if !isGoSourceFile(entry) {
			continue
		}

		file, err := parser.ParseFile(fset, filepath.Join(dir, entry.Name()), nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, fmt.Errorf("parsing %s: %w", entry.Name(), err)
		}

		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Body != nil || fn.Recv != nil {
				continue
			}

			decls[fn.Name.Name] = fset.Position(fn.Pos()).String()
		}
	}

	return decls, nil
}

func isGoSourceFile(e os.DirEntry) bool {
	name := e.Name()

	return !e.IsDir() && strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go")
}

// textSymbols collects all TEXT directive symbol names from .s files in dir.
func textSymbols(dir string) (map[string]bool, error) {
	symbols := make(map[string]bool)

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", dir, err)
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".s") {
			continue
		}

		data, err := os.ReadFile(filepath.Join(dir, entry.Name()))
		if err != nil {
			return nil, fmt.Errorf("reading %s: %w", entry.Name(), err)
		}

		for _, match := range textSymbolRe.FindAllStringSubmatch(string(data), -1) {
			symbols[match[1]] = true
		}
	}

	return symbols, nil
}
