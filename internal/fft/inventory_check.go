//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"os"
	"sort"
	"text/tabwriter"
)

// This is a standalone tool to inventory all FFT implementations
// Run with: go run internal/fft/inventory_check.go

type Implementation struct {
	Size      int
	Type      string // "complex64" or "complex128"
	Algorithm string // "radix-2", "radix-4", "mixed-radix"
	SIMD      string // "none", "avx2", "sse2"
	Source    string // "go", "asm"
}

func main() {
	// Manually enumerate implementations based on codebase inspection
	implementations := []Implementation{
		// Size 4 - Pure Go
		{4, "complex64", "radix-4", "none", "go"},
		{4, "complex128", "radix-4", "none", "go"},

		// Size 4 - AVX2 Assembly
		{4, "complex64", "radix-4", "avx2", "asm"},

		// Size 8 - Pure Go (radix-2)
		{8, "complex64", "radix-2", "none", "go"},
		{8, "complex128", "radix-2", "none", "go"},

		// Size 8 - Pure Go (mixed-radix: radix-4 + radix-2)
		{8, "complex64", "mixed-radix", "none", "go"},
		{8, "complex128", "mixed-radix", "none", "go"},

		// Size 8 - AVX2 Assembly (radix-2)
		{8, "complex64", "radix-2", "avx2", "asm"},

		// Size 8 - AVX2 Assembly (mixed-radix: radix-4 + radix-2)
		{8, "complex64", "mixed-radix", "avx2", "asm"},

		// Size 16 - Pure Go (radix-2)
		{16, "complex64", "radix-2", "none", "go"},
		{16, "complex128", "radix-2", "none", "go"},

		// Size 16 - Pure Go (radix-4)
		{16, "complex64", "radix-4", "none", "go"},
		{16, "complex128", "radix-4", "none", "go"},

		// Size 16 - AVX2 Assembly (radix-2)
		{16, "complex64", "radix-2", "avx2", "asm"},

		// Size 32 - Pure Go (radix-2)
		{32, "complex64", "radix-2", "none", "go"},
		{32, "complex128", "radix-2", "none", "go"},

		// Size 32 - AVX2 Assembly (radix-2)
		{32, "complex64", "radix-2", "avx2", "asm"},

		// Size 64 - Pure Go (radix-2)
		{64, "complex64", "radix-2", "none", "go"},
		{64, "complex128", "radix-2", "none", "go"},

		// Size 64 - AVX2 Assembly (radix-2)
		{64, "complex64", "radix-2", "avx2", "asm"},

		// Size 128 - Pure Go (radix-2)
		{128, "complex64", "radix-2", "none", "go"},
		{128, "complex128", "radix-2", "none", "go"},

		// Size 128 - AVX2 Assembly (radix-2)
		{128, "complex64", "radix-2", "avx2", "asm"},

		// Size 256 - Pure Go (radix-2)
		{256, "complex64", "radix-2", "none", "go"},
		{256, "complex128", "radix-2", "none", "go"},

		// Size 256 - Pure Go (radix-4)
		{256, "complex64", "radix-4", "none", "go"},
		{256, "complex128", "radix-4", "none", "go"},

		// Size 256 - AVX2 Assembly (radix-2)
		{256, "complex64", "radix-2", "avx2", "asm"},

		// Size 256 - AVX2 Assembly (radix-4) - INCOMPLETE (forward only)
		{256, "complex64", "radix-4 (fwd only)", "avx2", "asm"},
	}

	// Sort by size, then type, then algorithm, then SIMD
	sort.Slice(implementations, func(i, j int) bool {
		if implementations[i].Size != implementations[j].Size {
			return implementations[i].Size < implementations[j].Size
		}
		if implementations[i].Type != implementations[j].Type {
			return implementations[i].Type < implementations[j].Type
		}
		if implementations[i].Algorithm != implementations[j].Algorithm {
			return implementations[i].Algorithm < implementations[j].Algorithm
		}
		if implementations[i].SIMD != implementations[j].SIMD {
			return implementations[i].SIMD < implementations[j].SIMD
		}
		return implementations[i].Source < implementations[j].Source
	})

	// Print summary table
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "SIZE\tTYPE\tALGORITHM\tSIMD\tSOURCE\tSTATUS")
	fmt.Fprintln(w, "----\t----\t---------\t----\t------\t------")

	for _, impl := range implementations {
		status := "✓"
		if impl.Algorithm == "radix-4 (fwd only)" {
			status = "⚠ incomplete"
		}
		fmt.Fprintf(w, "%d\t%s\t%s\t%s\t%s\t%s\n",
			impl.Size, impl.Type, impl.Algorithm, impl.SIMD, impl.Source, status)
	}
	w.Flush()

	// Print coverage summary
	fmt.Println("\n=== COVERAGE SUMMARY ===")

	sizes := []int{4, 8, 16, 32, 64, 128, 256}

	fmt.Println("\nGo Implementations:")
	fmt.Println("  Size | complex64 | complex128")
	fmt.Println("  -----|-----------|------------")
	for _, size := range sizes {
		c64Count := 0
		c128Count := 0
		for _, impl := range implementations {
			if impl.Size == size && impl.Source == "go" {
				if impl.Type == "complex64" {
					c64Count++
				} else if impl.Type == "complex128" {
					c128Count++
				}
			}
		}
		fmt.Printf("  %-4d | %-9d | %-10d\n", size, c64Count, c128Count)
	}

	fmt.Println("\nAVX2 Assembly Implementations:")
	fmt.Println("  Size | complex64 | complex128")
	fmt.Println("  -----|-----------|------------")
	for _, size := range sizes {
		c64Count := 0
		c128Count := 0
		for _, impl := range implementations {
			if impl.Size == size && impl.Source == "asm" && impl.SIMD == "avx2" {
				if impl.Type == "complex64" {
					c64Count++
				} else if impl.Type == "complex128" {
					c128Count++
				}
			}
		}
		status64 := fmt.Sprintf("%d", c64Count)
		status128 := fmt.Sprintf("%d", c128Count)
		if c64Count == 0 {
			status64 = "-"
		}
		if c128Count == 0 {
			status128 = "-"
		}
		fmt.Printf("  %-4d | %-9s | %-10s\n", size, status64, status128)
	}

	// Print missing implementations
	fmt.Println("\n=== MISSING IMPLEMENTATIONS ===")

	fmt.Println("\nAVX2 Assembly (complex128):")
	fmt.Println("  - All sizes (4, 8, 16, 32, 64, 128, 256) - NOT IMPLEMENTED")

	fmt.Println("\nAVX2 Assembly (complex64):")
	fmt.Println("  - Size 256 radix-4: Inverse function missing")

	fmt.Println("\nPotential Optimizations:")
	fmt.Println("  - Size 16 AVX2: Could add radix-4 variant (currently only radix-2)")
	fmt.Println("  - Size 32 AVX2: Could add radix-4 variant (currently only radix-2)")
	fmt.Println("  - Size 64 AVX2: Could add radix-4 variant (currently only radix-2)")
	fmt.Println("  - Size 128 AVX2: Could add radix-4 variant (currently only radix-2)")
}
