package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// The non-codelet kernel tiers, by name.
//
// These used to be three prose paragraphs, which is how internal/fft's own
// hand-written inventory tool rotted into claiming complex128 AVX2 was "NOT
// IMPLEMENTED" at every size. A tier here names its entry point, and the
// generator fails if that identifier is not declared where the row says it is;
// the size rule of a size-dispatching tier is read out of its `switch n`
// rather than typed, so it cannot disagree with the code it describes.

// tierRow is one dispatch tier outside the codelet registry.
type tierRow struct {
	Family string
	// Algo names the algorithm family this row implements, when that family
	// exists *only* here and has no codelet rows to derive it from — the pure-Go
	// families and the arbitrary-length engines. Empty for the per-ISA dispatch
	// rows, which are routes into the codelet registry rather than families of
	// their own. This is the second half of the family axis in matrix.go: a
	// specs-only scan cannot see split-radix precisely because having no
	// codelet row is what makes it a gap.
	Algo  string
	Pkg   string // module-relative package directory
	Entry string // the identifier a reader should start from
	Prec  string
	ISA   string
	// SizeFuncs are functions whose `switch n` case labels define the tier's
	// size-specific coverage. Empty means the tier is size-generic and Rule
	// describes it instead.
	SizeFuncs []string
	Rule      string
}

//nolint:gochecknoglobals // static configuration for the inventory renderer
var tierRows = []tierRow{
	{
		Family: "AVX-512 generic radix-2 DIT", Pkg: "internal/fft",
		Entry: "avx512FirstKernel", Prec: "both", ISA: "AVX-512",
		Rule: "any power of two n ≥ 16; declines below that. Yields to the AVX2 " +
			"size-specific sizes below and to an explicitly forced Stockham.",
	},
	{
		Family: "AVX2 size-specific DIT", Pkg: "internal/fft",
		Entry: "avx2SizeSpecificOrGenericDITComplex64", Prec: "complex64", ISA: "AVX2",
		SizeFuncs: []string{"avx2SizeSpecificOrGenericDITComplex64"},
	},
	{
		Family: "AVX2 size-specific DIT", Pkg: "internal/fft",
		Entry: "avx2SizeSpecificOrGenericDITComplex128", Prec: "complex128", ISA: "AVX2",
		SizeFuncs: []string{"avx2SizeSpecificOrGenericDITComplex128"},
	},
	{
		Family: "AVX2 generic DIT / Stockham", Pkg: "internal/fft",
		Entry: "avx2KernelComplex64", Prec: "both", ISA: "AVX2",
		Rule: "any power of two; DIT or Stockham per the resolved strategy.",
	},
	{
		Family: "SSE3 size-specific DIT", Pkg: "internal/fft",
		Entry: "sse3TrySizeSpecificForwardComplex64", Prec: "complex64", ISA: "SSE3",
		SizeFuncs: []string{
			"sse3TrySizeSpecificForwardComplex64",
			"sse3TrySizeSpecificInverseComplex64",
		},
	},
	{
		Family: "SSE2 generic radix-2 DIT", Pkg: "internal/fft",
		Entry: "forwardSSE2Complex64", Prec: "both", ISA: "SSE2",
		Rule: "any power of two. This is the whole complex128 SSE tier — amd64 " +
			"has no SSE3 complex128 dispatch at all.",
	},
	{
		Family: "NEON size-specific DIT", Pkg: "internal/fft",
		Entry: "neonSizeSpecificOrGenericDITComplex64", Prec: "complex64", ISA: "NEON",
		SizeFuncs: []string{"neonSizeSpecificOrGenericDITComplex64"},
	},
	{
		Family: "NEON size-specific DIT", Pkg: "internal/fft",
		Entry: "neonSizeSpecificOrGenericDITComplex128", Prec: "complex128", ISA: "NEON",
		SizeFuncs: []string{"neonSizeSpecificOrGenericDITComplex128"},
	},
	{
		Family: "NEON generic DIT", Pkg: "internal/fft",
		Entry: "forwardNEONComplex64", Prec: "both", ISA: "NEON",
		Rule: "any power of two; the complex128 side delegates to pure Go.",
	},
	{
		Family: "386 SSE3 dispatch", Pkg: "internal/fft",
		Entry: "forwardSSE3Complex64", Prec: "complex64", ISA: "SSE3 (386)",
		SizeFuncs: []string{"forwardSSE3Complex64"},
		Rule:      "larger powers of two fall through to the generic SSE2 kernel.",
	},
	{
		Family: "386 SSE2 dispatch", Pkg: "internal/fft",
		Entry: "forwardSSE2Complex128", Prec: "complex128", ISA: "SSE2 (386)",
		SizeFuncs: []string{"forwardSSE2Complex128"},
		Rule:      "no generic SSE2 complex128 kernel: every other size declines.",
	},
	{
		Family: "386 SSE1 dispatch", Pkg: "internal/fft",
		Entry: "forwardSSEComplex64", Prec: "complex64", ISA: "SSE (386)",
		SizeFuncs: []string{"forwardSSEComplex64"},
		Rule: "larger powers of two fall through to the generic SSE kernel. The " +
			"census reports the size-8 and size-16 386 SSE symbols as reachable " +
			"only through thunks nothing calls — this switch is why.",
	},
	{
		Family: "DIT (pure Go)", Algo: "DIT", Pkg: "internal/kernels",
		Entry: "ForwardDITComplex64", Prec: "both", ISA: "any",
		Rule: "any power of two; the auto heuristic picks it up to " +
			"`ditAutoThreshold` (internal/planner/selection.go).",
	},
	{
		Family: "Stockham (pure Go)", Algo: "Stockham", Pkg: "internal/kernels",
		Entry: "ForwardStockhamComplex64", Prec: "both", ISA: "any",
		Rule: "any power of two; the auto heuristic picks it above the DIT threshold.",
	},
	{
		Family: "Split-radix (pure Go)", Algo: "Split-radix", Pkg: "internal/kernels",
		Entry: "ForwardSplitRadixComplex64", Prec: "both", ISA: "any",
		Rule: "any power of two, reachable only through a forced " +
			"`KernelSplitRadix` — the auto heuristic never selects it.",
	},
	{
		Family: "Six-step (pure Go)", Algo: "Six-step", Pkg: "internal/kernels",
		Entry: "ForwardSixStepComplex64", Prec: "both", ISA: "any",
		Rule: "perfect squares only (even exponent); declines every other length.",
	},
	{
		Family: "Four-step (pure Go)", Algo: "Four-step", Pkg: "internal/kernels",
		Entry: "ForwardFourStepComplex64", Prec: "both", ISA: "any",
		Rule: "any power of two — the rectangular n1×n2 split covers the odd " +
			"exponents six-step declines, tilted by the detected L1d/L2 sizes.",
	},
	{
		Family: "Mixed-radix engine", Algo: "Mixed-radix engine", Pkg: "internal/fft",
		Entry: "forwardMixedRadixComplex64", Prec: "both", ISA: "any",
		Rule: "smooth lengths with factors 2/3/5/7/11 (`math.IsMixedRadixSmooth`); " +
			"the route every non-power-of-two outside Bluestein takes.",
	},
	{
		Family: "Rader", Algo: "Rader", Pkg: "internal/fft",
		Entry: "ComputeRaderTables", Prec: "both", ISA: "any",
		Rule: "prime lengths passing `RaderEligible`.",
	},
	{
		Family: "Bluestein", Algo: "Bluestein", Pkg: "internal/fft",
		Entry: "BluesteinConvolution", Prec: "both", ISA: "any",
		Rule: "arbitrary lengths; padded to a power of two the tiers above can serve.",
	},
	{
		Family: "Recursive decomposition", Algo: "Recursive", Pkg: "internal/transform",
		Entry: "PlanDecomposition", Prec: "both", ISA: "any",
		Rule: "powers of two, bottoming out in registered codelet leaves.",
	},
}

// renderTiers writes the non-codelet tier table.
func renderTiers(b *bytes.Buffer, root string) {
	b.WriteString(strings.TrimLeft(`
## Beyond the Codelet Registry

Codelets cover the tuned size-specific fast paths. Every other length is served
by the tiers below. _Entry point_ is the identifier to read first; the
generator's tests fail if it is not declared in the package named. _Sizes_ is
read out of the tier's `+"`switch n`"+` where it has one, so it cannot drift from the
dispatch it describes.

On amd64 the chain is tried widest-first (AVX-512 → AVX2 size-specific → AVX2
generic → SSE3 → SSE2 → pure Go), each tier declining to the next; a plan bound
to a registry codelet never enters it. Higher plan-level features (real FFT,
2D/3D/N-D, batch/strided, convolution) compose these 1D kernels.

`, "\n"))

	b.WriteString("| Family | Entry point | Precision | ISA | Sizes |\n")
	b.WriteString("|---|---|---|---|---|\n")

	for _, t := range tierRows {
		rule := t.Rule

		if len(t.SizeFuncs) > 0 {
			sizes, err := tierSwitchSizes(root, t)

			listed := "size-specific: " + sizeList(sizes)
			if err != nil {
				listed = "**scan failed: " + err.Error() + "**"
			}

			if rule != "" {
				rule = listed + "; " + rule
			} else {
				rule = listed
			}
		}

		fmt.Fprintf(b, "| %s | `%s.%s` | %s | %s | %s |\n",
			t.Family, t.Pkg, t.Entry, t.Prec, t.ISA, rule)
	}

	b.WriteString("\n")
}

// tierSwitchSizes reads the union of the tier's `switch n` case labels.
func tierSwitchSizes(root string, t tierRow) ([]int, error) {
	pkg, err := loadGoPkg(root, t.Pkg)
	if err != nil {
		return nil, err
	}

	seen := map[int]bool{}

	for _, fn := range t.SizeFuncs {
		sizes, from, ok := pkg.switchSizes(fn)
		if !ok {
			return nil, fmt.Errorf("%s: no func %s", t.Pkg, fn)
		}

		if len(from) > 1 {
			return nil, fmt.Errorf("%s.%s dispatches sizes in %s: the row would merge two tiers",
				t.Pkg, fn, strings.Join(from, " and "))
		}

		for _, size := range sizes {
			seen[size] = true
		}
	}

	out := make([]int, 0, len(seen))
	for size := range seen {
		out = append(out, size)
	}

	sort.Ints(out)

	return out, nil
}

// goPkg is the parsed non-test source of one package directory.
//
// A name can be declared more than once: the per-arch dispatch files are
// mutually exclusive by build tag, so internal/fft has several
// forwardSSE2Complex128, one real and the rest purego stubs. Loading across
// tags is what makes every architecture visible here at once, and the price is
// that a name maps to a list.
type goPkg struct {
	funcs map[string][]funcDecl
	decls map[string]bool // every top-level name, including vars and consts
}

type funcDecl struct {
	File string
	Decl *ast.FuncDecl
}

//nolint:gochecknoglobals // parse cache; the generator is single-threaded
var goPkgCache = map[string]*goPkg{}

// loadGoPkg parses every non-test .go file in a package directory, across all
// build tags — the tiers live in mutually exclusive per-arch files, so a
// tag-respecting load would see at most one architecture's dispatch.
func loadGoPkg(root, dir string) (*goPkg, error) {
	if p, ok := goPkgCache[dir]; ok {
		return p, nil
	}

	entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(dir)))
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", dir, err)
	}

	pkg := &goPkg{funcs: map[string][]funcDecl{}, decls: map[string]bool{}}
	fset := token.NewFileSet()

	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}

		file, err := parser.ParseFile(fset,
			filepath.Join(root, filepath.FromSlash(dir), name), nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, fmt.Errorf("parse %s/%s: %w", dir, name, err)
		}

		pkg.add(name, file)
	}

	goPkgCache[dir] = pkg

	return pkg, nil
}

func (p *goPkg) add(name string, file *ast.File) {
	for _, d := range file.Decls {
		switch decl := d.(type) {
		case *ast.FuncDecl:
			if decl.Recv == nil {
				p.funcs[decl.Name.Name] = append(p.funcs[decl.Name.Name],
					funcDecl{File: name, Decl: decl})
				p.decls[decl.Name.Name] = true
			}
		case *ast.GenDecl:
			for _, spec := range decl.Specs {
				switch s := spec.(type) {
				case *ast.ValueSpec:
					for _, n := range s.Names {
						p.decls[n.Name] = true
					}
				case *ast.TypeSpec:
					p.decls[s.Name.Name] = true
				}
			}
		}
	}
}

// switchSizes returns the integer case labels of every `switch n` in fn,
// including switches inside the closures it returns, plus the files that
// contributed them — more than one means the union spans two build tags and
// the row is describing two different dispatches at once.
func (p *goPkg) switchSizes(fn string) ([]int, []string, bool) {
	decls, ok := p.funcs[fn]
	if !ok {
		return nil, nil, false
	}

	var (
		sizes []int
		from  []string
	)

	for _, d := range decls {
		found := declSwitchSizes(d.Decl)
		if len(found) == 0 {
			continue
		}

		sizes = append(sizes, found...)
		from = append(from, d.File)
	}

	return sizes, from, true
}

// declSwitchSizes collects the integer case labels of every `switch n` in one
// declaration.
func declSwitchSizes(decl *ast.FuncDecl) []int {
	var sizes []int

	ast.Inspect(decl, func(n ast.Node) bool {
		sw, ok := n.(*ast.SwitchStmt)
		if !ok {
			return true
		}

		if tag, ok := sw.Tag.(*ast.Ident); !ok || tag.Name != "n" {
			return true
		}

		for _, stmt := range sw.Body.List {
			clause, ok := stmt.(*ast.CaseClause)
			if !ok {
				continue
			}

			for _, expr := range clause.List {
				lit, ok := expr.(*ast.BasicLit)
				if !ok || lit.Kind != token.INT {
					continue
				}

				if v, err := strconv.Atoi(lit.Value); err == nil {
					sizes = append(sizes, v)
				}
			}
		}

		return true
	})

	return sizes
}
