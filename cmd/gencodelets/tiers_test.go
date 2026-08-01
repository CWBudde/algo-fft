package main

import (
	"bytes"
	"reflect"
	"sort"
	"strings"
	"testing"
)

// TestTierEntryPointsExist is what the old prose could not do: a tier row
// names an identifier, and this fails if that identifier is not declared in
// the package the row points at. internal/fft's own hand-written inventory
// tool (deleted with this change) drifted into claiming complex128 AVX2 was
// unimplemented at every size precisely because nothing checked it.
func TestTierEntryPointsExist(t *testing.T) {
	root := repoRoot(t)

	for _, row := range tierRows {
		pkg, err := loadGoPkg(root, row.Pkg)
		if err != nil {
			t.Fatalf("%s: %v", row.Pkg, err)
		}

		if !pkg.decls[row.Entry] {
			t.Errorf("%s: %s declares no %s", row.Family, row.Pkg, row.Entry)
		}

		for _, fn := range row.SizeFuncs {
			if _, ok := pkg.funcs[fn]; !ok {
				t.Errorf("%s: %s declares no func %s", row.Family, row.Pkg, fn)
			}
		}
	}
}

// TestTierSizesAreDerivable checks that every size-dispatching row actually
// yields sizes. A row whose switch stopped being a `switch n` would otherwise
// render as an empty cell, which reads as "covers nothing" rather than as a
// broken scan.
func TestTierSizesAreDerivable(t *testing.T) {
	root := repoRoot(t)

	for _, row := range tierRows {
		if len(row.SizeFuncs) == 0 {
			continue
		}

		sizes, err := tierSwitchSizes(root, row)
		if err != nil {
			t.Errorf("%s (%s): %v", row.Family, row.Entry, err)

			continue
		}

		if len(sizes) == 0 {
			t.Errorf("%s (%s): no sizes scanned", row.Family, row.Entry)
		}
	}
}

// TestAVX512CoversMatchesTheAVX2Switch enforces the "Keep in sync with that
// switch" comment in internal/fft/kernels_amd64_avx512.go. The Covers
// functions decide when the AVX-512 tier steps aside for a tuned AVX2 codelet;
// if they drift, an AVX-512 host either misses a codelet that beats it or
// hands a size to AVX2 that has nothing for it. Nothing in the test suite
// notices either.
func TestAVX512CoversMatchesTheAVX2Switch(t *testing.T) {
	root := repoRoot(t)

	pkg, err := loadGoPkg(root, "internal/fft")
	if err != nil {
		t.Fatal(err)
	}

	pairs := []struct{ covers, dispatch string }{
		{"avx2SizeSpecificDITComplex64Covers", "avx2SizeSpecificOrGenericDITComplex64"},
		{"avx2SizeSpecificDITComplex128Covers", "avx2SizeSpecificOrGenericDITComplex128"},
	}

	for _, p := range pairs {
		got := uniqueSizes(t, pkg, p.covers)
		want := uniqueSizes(t, pkg, p.dispatch)

		if !reflect.DeepEqual(got, want) {
			t.Errorf("%s covers %v but %s dispatches %v", p.covers, got, p.dispatch, want)
		}
	}
}

// TestDispatchDirectionsAgree checks that a size-specific tier covers the same
// sizes forward and inverse. A one-directional gap is invisible in a forward
// benchmark and turns an inverse transform at that size into a silently
// slower path.
func TestDispatchDirectionsAgree(t *testing.T) {
	root := repoRoot(t)

	pkg, err := loadGoPkg(root, "internal/fft")
	if err != nil {
		t.Fatal(err)
	}

	pairs := []struct{ forward, inverse string }{
		{"avx2SizeSpecificOrGenericDITComplex64", "avx2SizeSpecificOrGenericDITInverseComplex64"},
		{"avx2SizeSpecificOrGenericDITComplex128", "avx2SizeSpecificOrGenericDITInverseComplex128"},
		{"sse3TrySizeSpecificForwardComplex64", "sse3TrySizeSpecificInverseComplex64"},
		{"neonSizeSpecificOrGenericDITComplex64", "neonSizeSpecificOrGenericDITInverseComplex64"},
		{"neonSizeSpecificOrGenericDITComplex128", "neonSizeSpecificOrGenericDITInverseComplex128"},
	}

	for _, p := range pairs {
		fwd := uniqueSizes(t, pkg, p.forward)
		inv := uniqueSizes(t, pkg, p.inverse)

		if !reflect.DeepEqual(fwd, inv) {
			t.Errorf("%s dispatches %v but %s dispatches %v", p.forward, fwd, p.inverse, inv)
		}
	}
}

// TestRenderTiersIncludesEveryRow guards the wiring into the document.
func TestRenderTiersIncludesEveryRow(t *testing.T) {
	root := repoRoot(t)

	var buf bytes.Buffer

	renderTiers(&buf, root)

	doc := buf.String()
	for _, row := range tierRows {
		if !strings.Contains(doc, row.Entry) {
			t.Errorf("the tier table omits %s", row.Entry)
		}
	}
}

func uniqueSizes(t *testing.T, pkg *goPkg, fn string) []int {
	t.Helper()

	sizes, from, ok := pkg.switchSizes(fn)
	if !ok {
		t.Fatalf("internal/fft declares no func %s", fn)
	}

	if len(from) > 1 {
		t.Fatalf("%s has a size switch in %s — the comparison would merge tiers",
			fn, strings.Join(from, " and "))
	}

	seen := map[int]bool{}

	out := make([]int, 0, len(sizes))

	for _, size := range sizes {
		if !seen[size] {
			seen[size] = true

			out = append(out, size)
		}
	}

	sort.Ints(out)

	return out
}

func repoRoot(t *testing.T) string {
	t.Helper()

	root, err := findModuleRoot("../..")
	if err != nil {
		t.Fatal(err)
	}

	return root
}
