// Command genkernels generates the complex128 kernel twins in
// internal/kernels from their hand-written complex64 sources.
//
// For every non-test, non-generated, non-build-tagged file that contains
// functions whose name includes "Complex64", it emits a sibling
// <base>_c128.gen.go holding the Complex128 twins: same code with the element
// type, name suffixes, and float32 conversions rewritten. The complex64
// sources stay the single hand-maintained implementation; edit those and
// regenerate.
//
// A few Complex64 functions are excluded because their complex128
// counterparts are deliberately different (see excludedFuncs).
//
// Regenerate with: go generate ./internal/kernels/...
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// excludedFuncs are Complex64 functions whose Complex128 counterparts are
// deliberately not textual twins and stay hand-written:
//   - the radix-3/radix-4/radix-5 entry points: the complex128 side delegates
//     to the generic implementations instead of the monomorphized complex64
//     copies
//   - radix3/4/5TransformComplex64: the complex64-only monomorphizations
//     backing those entry points. They exist because Go compiles scalar
//     complex64 multiplication in double precision (see math.MulComplex64) and
//     a `[T Complex]` body cannot opt out; complex128 has nothing to gain
//   - the test helpers, whose float32 randomization/tolerances differ
//   - the 16384-point radix-4 kernel: its [16384]complexN stage arrays sit
//     exactly at the compiler's 128 KiB explicit-declaration stack limit as
//     complex64 but exceed it as complex128 (256 KiB each), so a textual twin
//     heap-allocates ~1.75 MiB per transform (measured, ~2× slower); the
//     hand-written complex128 version uses a different stage structure that
//     stays allocation-free (guarded by the codelet zero-alloc sweep)
//
//nolint:gochecknoglobals // static configuration table for the generator
var excludedFuncs = map[string]bool{
	"forwardRadix3Complex64":         true,
	"inverseRadix3Complex64":         true,
	"forwardRadix4Complex64":         true,
	"inverseRadix4Complex64":         true,
	"forwardRadix5Complex64":         true,
	"inverseRadix5Complex64":         true,
	"radix3TransformComplex64":       true,
	"radix4TransformComplex64":       true,
	"radix5TransformComplex64":       true,
	"randomComplex64":                true,
	"assertComplex64Close":           true,
	"forwardDIT16384Radix4Complex64": true,
	"inverseDIT16384Radix4Complex64": true,
}

func main() {
	dir := "internal/kernels"
	if len(os.Args) > 1 {
		dir = os.Args[1]
	}

	entries, err := filepath.Glob(filepath.Join(dir, "*.go"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "genkernels: %v\n", err)
		os.Exit(1)
	}

	sort.Strings(entries)

	var written []string

	for _, path := range entries {
		base := filepath.Base(path)
		if strings.HasSuffix(base, "_test.go") || strings.HasSuffix(base, ".gen.go") {
			continue
		}

		src, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "genkernels: %v\n", err)
			os.Exit(1)
		}

		// Build-tagged files (asm bridges, SIMD wrappers) pair with
		// architecture-specific code, not textual twins.
		if strings.Contains(string(src), "//go:build") {
			continue
		}

		out, err := generateFile(base, src, excludedFuncs)
		if err != nil {
			fmt.Fprintf(os.Stderr, "genkernels: %v\n", err)
			os.Exit(1)
		}

		if out == nil {
			continue
		}

		outPath := filepath.Join(dir, strings.TrimSuffix(base, ".go")+"_c128.gen.go")

		err = os.WriteFile(outPath, out, 0o644) //nolint:gosec
		if err != nil {
			fmt.Fprintf(os.Stderr, "genkernels: %v\n", err)
			os.Exit(1)
		}

		written = append(written, filepath.Base(outPath))
	}

	removeStaleOutputs(dir, written)

	fmt.Fprintf(os.Stderr, "genkernels: wrote %d files in %s\n", len(written), dir)
}

// removeStaleOutputs deletes previously generated twin files whose source no
// longer produces output (renamed, deleted, or newly excluded).
func removeStaleOutputs(dir string, written []string) {
	current := map[string]bool{}
	for _, name := range written {
		current[name] = true
	}

	stale, _ := filepath.Glob(filepath.Join(dir, "*_c128.gen.go"))

	for _, path := range stale {
		if current[filepath.Base(path)] {
			continue
		}

		if err := os.Remove(path); err != nil {
			fmt.Fprintf(os.Stderr, "genkernels: remove stale %s: %v\n", path, err)
			os.Exit(1)
		}

		fmt.Fprintf(os.Stderr, "genkernels: removed stale %s\n", filepath.Base(path))
	}
}
