package planner

import (
	"sync"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
)

// dummyCodelet is a dummy function for testing.
func dummyCodelet[T Complex](dst, src, twiddle, scratch []T) {}

// TestCodeletRegistryRegisterAndLookup tests basic register and lookup operations.
func TestCodeletRegistryRegisterAndLookup(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	entry := CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit16_generic",
		Priority:  1,
	}

	registry.Register(entry)

	features := cpu.Features{
		Architecture: "amd64",
	}

	found := registry.Lookup(16, features)
	if found == nil {
		t.Fatal("expected to find registered codelet")
	}

	if found.Signature != "dit16_generic" {
		t.Errorf("expected signature \"dit16_generic\", got %q", found.Signature)
	}
}

// TestCodeletRegistryNotFound tests lookup of unregistered size.
func TestCodeletRegistryNotFound(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	entry := CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit16_generic",
		Priority:  1,
	}

	registry.Register(entry)

	features := cpu.Features{
		Architecture: "amd64",
	}

	found := registry.Lookup(32, features)
	if found != nil {
		t.Errorf("expected not to find unregistered size 32, got %v", found)
	}
}

// TestCodeletRegistryMultipleVariants tests registration of multiple variants for same size.
func TestCodeletRegistryMultipleVariants(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	// Register generic variant
	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit16_generic",
		Priority:  0,
	})

	// Register AVX2 variant (should be preferred)
	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDAVX2,
		Signature: "dit16_avx2",
		Priority:  0,
	})

	features := cpu.Features{
		Architecture: "amd64",
		HasAVX2:      true,
	}

	found := registry.Lookup(16, features)
	if found == nil {
		t.Fatal("expected to find codelet with AVX2")
	}

	if found.Signature != "dit16_avx2" {
		t.Errorf("expected AVX2 variant, got %q", found.Signature)
	}
}

// TestCodeletRegistryPreferHigherSIMD tests that higher SIMD levels are preferred.
func TestCodeletRegistryPreferHigherSIMD(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	// Register multiple SIMD variants
	variants := []struct {
		simd      SIMDLevel
		signature string
	}{
		{SIMDNone, "generic"},
		{SIMDSSE2, "sse2"},
		{SIMDAVX2, "avx2"},
		{SIMDAVX512, "avx512"},
	}

	for _, variant := range variants {
		registry.Register(CodeletEntry[complex64]{
			Size:      32,
			Forward:   dummyCodelet[complex64],
			Inverse:   dummyCodelet[complex64],
			Algorithm: KernelDIT,
			SIMDLevel: variant.simd,
			Signature: "dit32_" + variant.signature,
			Priority:  0,
		})
	}

	// With AVX512 CPU, should get avx512
	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
		HasAVX2:      true,
		HasAVX512:    true,
	}

	found := registry.Lookup(32, features)
	if found == nil {
		t.Fatal("expected to find codelet")
	}

	if found.Signature != "dit32_avx512" {
		t.Errorf("expected avx512 variant, got %q", found.Signature)
	}

	// With only AVX2 CPU, should get avx2
	features.HasAVX512 = false

	found = registry.Lookup(32, features)
	if found == nil {
		t.Fatal("expected to find codelet")
	}

	if found.Signature != "dit32_avx2" {
		t.Errorf("expected avx2 variant, got %q", found.Signature)
	}

	// With only SSE2 CPU, should get sse2
	features.HasAVX2 = false

	found = registry.Lookup(32, features)
	if found == nil {
		t.Fatal("expected to find codelet")
	}

	if found.Signature != "dit32_sse2" {
		t.Errorf("expected sse2 variant, got %q", found.Signature)
	}

	// With no SIMD, should get generic
	features.HasSSE2 = false

	found = registry.Lookup(32, features)
	if found == nil {
		t.Fatal("expected to find codelet")
	}

	if found.Signature != "dit32_generic" {
		t.Errorf("expected generic variant, got %q", found.Signature)
	}
}

// TestCodeletRegistryPriority tests priority ordering for same SIMD level.
func TestCodeletRegistryPriority(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	// Register two codelets with same SIMD level, different priority
	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "low_priority",
		Priority:  1,
	})

	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "high_priority",
		Priority:  10,
	})

	features := cpu.Features{
		Architecture: "amd64",
	}

	found := registry.Lookup(16, features)
	if found == nil {
		t.Fatal("expected to find codelet")
	}

	if found.Signature != "high_priority" {
		t.Errorf("expected high priority variant, got %q", found.Signature)
	}
}

// TestCodeletRegistryLookupBySignature tests lookup by signature string.
func TestCodeletRegistryLookupBySignature(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "dit16_generic",
		Priority:  1,
	})

	found := registry.LookupBySignature(16, "dit16_generic")
	if found == nil {
		t.Fatal("expected to find codelet by signature")
	}

	notFound := registry.LookupBySignature(16, "nonexistent")
	if notFound != nil {
		t.Errorf("expected not to find nonexistent signature, got %v", notFound)
	}
}

// TestCodeletRegistrySizes tests retrieval of registered sizes.
func TestCodeletRegistrySizes(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	sizes := []int{8, 16, 32, 64}
	for _, size := range sizes {
		registry.Register(CodeletEntry[complex64]{
			Size:      size,
			Forward:   dummyCodelet[complex64],
			Inverse:   dummyCodelet[complex64],
			Algorithm: KernelDIT,
			SIMDLevel: SIMDNone,
			Signature: "test",
			Priority:  0,
		})
	}

	got := registry.Sizes()
	if len(got) != len(sizes) {
		t.Errorf("expected %d sizes, got %d", len(sizes), len(got))
	}

	// Convert to map for easier comparison
	sizeMap := make(map[int]bool)
	for _, size := range got {
		sizeMap[size] = true
	}

	for _, size := range sizes {
		if !sizeMap[size] {
			t.Errorf("expected size %d in registry, not found", size)
		}
	}
}

// TestCodeletRegistryGetAvailableSizes tests filtering by CPU features.
func TestCodeletRegistryGetAvailableSizes(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	// Register codelets that require different SIMD levels
	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "generic",
		Priority:  0,
	})

	registry.Register(CodeletEntry[complex64]{
		Size:      32,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDAVX2,
		Signature: "avx2",
		Priority:  0,
	})

	registry.Register(CodeletEntry[complex64]{
		Size:      64,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: KernelDIT,
		SIMDLevel: SIMDNone,
		Signature: "generic",
		Priority:  0,
	})

	// Without AVX2, should get 16 and 64 only
	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	got := registry.GetAvailableSizes(features)
	if len(got) != 2 {
		t.Errorf("expected 2 available sizes without AVX2, got %d: %v", len(got), got)
	}

	// With AVX2, should get all three
	features.HasAVX2 = true

	got = registry.GetAvailableSizes(features)
	if len(got) != 3 {
		t.Errorf("expected 3 available sizes with AVX2, got %d: %v", len(got), got)
	}
}

// TestCodeletRegistrySorted tests that GetAvailableSizes returns sorted results.
func TestCodeletRegistrySorted(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	// Register in non-sorted order
	sizes := []int{64, 16, 32, 8, 128}
	for _, size := range sizes {
		registry.Register(CodeletEntry[complex64]{
			Size:      size,
			Forward:   dummyCodelet[complex64],
			Inverse:   dummyCodelet[complex64],
			Algorithm: KernelDIT,
			SIMDLevel: SIMDNone,
			Signature: "test",
			Priority:  0,
		})
	}

	features := cpu.Features{
		Architecture: "amd64",
	}

	got := registry.GetAvailableSizes(features)

	// Check sorted
	for i := 1; i < len(got); i++ {
		if got[i] < got[i-1] {
			t.Errorf("GetAvailableSizes not sorted: %v", got)
			break
		}
	}
}

// TestCodeletRegistryConcurrent tests concurrent registration and lookup.
func TestCodeletRegistryConcurrent(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	var waitGroup sync.WaitGroup

	const goroutines = 10

	// Concurrent registration
	for i := range goroutines {
		waitGroup.Add(1)

		go func(idx int) {
			defer waitGroup.Done()

			for j := range 10 {
				size := 16 + idx*100 + j
				registry.Register(CodeletEntry[complex64]{
					Size:      size,
					Forward:   dummyCodelet[complex64],
					Inverse:   dummyCodelet[complex64],
					Algorithm: KernelDIT,
					SIMDLevel: SIMDNone,
					Signature: "test",
					Priority:  0,
				})
			}
		}(i)
	}

	waitGroup.Wait()

	// Verify all registered
	sizes := registry.Sizes()
	if len(sizes) != goroutines*10 {
		t.Errorf("expected %d sizes, got %d", goroutines*10, len(sizes))
	}
}

// TestCPUSupports tests the cpuSupports helper function.
func TestCPUSupports(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		features cpu.Features
		level    SIMDLevel
		want     bool
	}{
		{"None level always supported", cpu.Features{}, SIMDNone, true},
		{"SSE2 with SSE2 support", cpu.Features{HasSSE2: true}, SIMDSSE2, true},
		{"SSE2 without SSE2 support", cpu.Features{HasSSE2: false}, SIMDSSE2, false},
		{"AVX2 with AVX2 support", cpu.Features{HasAVX2: true}, SIMDAVX2, true},
		{"AVX2 without AVX2 support", cpu.Features{HasAVX2: false}, SIMDAVX2, false},
		{"AVX512 with AVX512 support", cpu.Features{HasAVX512: true}, SIMDAVX512, true},
		{"NEON with NEON support", cpu.Features{HasNEON: true}, SIMDNEON, true},
		{"NEON without NEON support", cpu.Features{HasNEON: false}, SIMDNEON, false},
		{"Invalid level", cpu.Features{}, fftypes.SIMDLevel(99), false},
	}

	for _, tt := range tests {
		got := cpuSupports(tt.features, tt.level)
		if got != tt.want {
			t.Errorf("%s: cpuSupports() = %v, want %v", tt.name, got, tt.want)
		}
	}
}
