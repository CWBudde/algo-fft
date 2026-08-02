package registry

import (
	"sync"
	"testing"

	"github.com/cwbudde/algo-fft/internal/cpu"
	"github.com/cwbudde/algo-fft/internal/fftypes"
)

// dummyCodelet is a dummy function for testing.
func dummyCodelet[T fftypes.Complex](dst, src, twiddle, scratch []T) bool { return true }

// TestCodeletRegistryRegisterAndLookup tests basic register and lookup operations.
func TestCodeletRegistryRegisterAndLookup(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	entry := CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
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
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
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
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
		Signature: "dit16_generic",
		Priority:  0,
	})

	// Register AVX2 variant (should be preferred)
	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDAVX2,
		Signature: "dit16_avx2",
		Priority:  0,
	})

	features := cpu.Features{
		Architecture: "amd64",
		HasAVX2:      true,
		HasFMA:       true,
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
		simd      fftypes.SIMDLevel
		signature string
	}{
		{fftypes.SIMDNone, "generic"},
		{fftypes.SIMDSSE2, "sse2"},
		{fftypes.SIMDAVX2, "avx2"},
		{fftypes.SIMDAVX512, "avx512"},
	}

	for _, variant := range variants {
		registry.Register(CodeletEntry[complex64]{
			Size:      32,
			Forward:   dummyCodelet[complex64],
			Inverse:   dummyCodelet[complex64],
			Algorithm: fftypes.KernelDIT,
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
		HasFMA:       true,
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
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
		Signature: "low_priority",
		Priority:  1,
	})

	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
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
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
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

// TestCodeletRegistryLookupBySignatureSkipsDisabled verifies that a codelet
// disabled via negative priority is not reachable by signature. The wisdom
// binder in internal/planner resolves persisted algorithm names through
// LookupBySignature, so without this filter a stale or imported wisdom entry
// could resurrect a codelet that was deliberately measured to be slower.
func TestCodeletRegistryLookupBySignatureSkipsDisabled(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
		Signature: "dit16_disabled",
		Priority:  -1,
	})

	if got := registry.LookupBySignature(16, "dit16_disabled"); got != nil {
		t.Errorf("expected disabled codelet to be unreachable by signature, got %v", got)
	}

	// A disabled entry must not shadow an enabled one at the same size.
	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
		Signature: "dit16_enabled",
		Priority:  1,
	})

	if got := registry.LookupBySignature(16, "dit16_enabled"); got == nil {
		t.Error("expected enabled codelet to remain reachable by signature")
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
			Algorithm: fftypes.KernelDIT,
			SIMDLevel: fftypes.SIMDNone,
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
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
		Signature: "generic",
		Priority:  0,
	})

	registry.Register(CodeletEntry[complex64]{
		Size:      32,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDAVX2,
		Signature: "avx2",
		Priority:  0,
	})

	registry.Register(CodeletEntry[complex64]{
		Size:      64,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
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
	features.HasFMA = true

	got = registry.GetAvailableSizes(features)
	if len(got) != 3 {
		t.Errorf("expected 3 available sizes with AVX2, got %d: %v", len(got), got)
	}
}

// TestCodeletRegistryGetAvailableSizesDisabled verifies that GetAvailableSizes
// does not advertise a size whose only CPU-compatible codelet is disabled
// (Priority < 0), and that GetAvailableSizes stays consistent with Lookup.
func TestCodeletRegistryGetAvailableSizesDisabled(t *testing.T) {
	t.Parallel()

	registry := NewCodeletRegistry[complex64]()

	// Size 16: a normal, enabled codelet.
	registry.Register(CodeletEntry[complex64]{
		Size:      16,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
		Signature: "generic",
		Priority:  0,
	})

	// Size 32: served ONLY by a disabled codelet.
	registry.Register(CodeletEntry[complex64]{
		Size:      32,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
		Signature: "disabled",
		Priority:  -1,
	})

	features := cpu.Features{
		Architecture: "amd64",
		HasSSE2:      true,
	}

	// Lookup must reject the disabled-only size.
	if got := registry.Lookup(32, features); got != nil {
		t.Errorf("Lookup(32) = %v, want nil (only codelet is disabled)", got)
	}

	// GetAvailableSizes must not advertise the disabled-only size, and must
	// agree with Lookup for every size it does advertise.
	sizes := registry.GetAvailableSizes(features)
	for _, size := range sizes {
		if size == 32 {
			t.Errorf("GetAvailableSizes advertised size 32 served only by a disabled codelet: %v", sizes)
		}

		if registry.Lookup(size, features) == nil {
			t.Errorf("GetAvailableSizes advertised size %d but Lookup returns nil", size)
		}
	}

	if len(sizes) != 1 {
		t.Errorf("expected only size 16 available, got %v", sizes)
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
			Algorithm: fftypes.KernelDIT,
			SIMDLevel: fftypes.SIMDNone,
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
					Algorithm: fftypes.KernelDIT,
					SIMDLevel: fftypes.SIMDNone,
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
		level    fftypes.SIMDLevel
		want     bool
	}{
		{"None level always supported", cpu.Features{}, fftypes.SIMDNone, true},
		{"SSE2 with SSE2 support", cpu.Features{HasSSE2: true}, fftypes.SIMDSSE2, true},
		{"SSE2 without SSE2 support", cpu.Features{HasSSE2: false}, fftypes.SIMDSSE2, false},
		{"AVX2 with AVX2+FMA support", cpu.Features{HasAVX2: true, HasFMA: true}, fftypes.SIMDAVX2, true},
		{"AVX2 without FMA support", cpu.Features{HasAVX2: true, HasFMA: false}, fftypes.SIMDAVX2, false},
		{"AVX2 without AVX2 support", cpu.Features{HasAVX2: false}, fftypes.SIMDAVX2, false},
		{"AVX512 with AVX512 support", cpu.Features{HasAVX512: true}, fftypes.SIMDAVX512, true},
		{"NEON with NEON support", cpu.Features{HasNEON: true}, fftypes.SIMDNEON, true},
		{"NEON without NEON support", cpu.Features{HasNEON: false}, fftypes.SIMDNEON, false},
		{"Invalid level", cpu.Features{}, fftypes.SIMDLevel(99), false},
	}

	for _, tt := range tests {
		got := CPUSupports(tt.features, tt.level)
		if got != tt.want {
			t.Errorf("%s: CPUSupports() = %v, want %v", tt.name, got, tt.want)
		}
	}
}

// TestCodeletRegistryRankLevelDemotes verifies that RankLevel moves an entry
// into another tier for ordering while leaving eligibility on SIMDLevel. This
// is what lets a genuinely faster SSE2 codelet beat an AVX2-encoded but
// SSE-width sibling, which plain priorities cannot express.
func TestCodeletRegistryRankLevelDemotes(t *testing.T) {
	t.Parallel()

	reg := NewCodeletRegistry[complex64]()

	// AVX2-encoded but SSE-width: ranked into the SSE2 tier, below the SSE2
	// codelet that measures faster.
	reg.Register(CodeletEntry[complex64]{
		Size:      64,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDAVX2,
		RankLevel: fftypes.SIMDSSE2,
		Signature: "narrow_avx2",
		Priority:  15,
	})
	reg.Register(CodeletEntry[complex64]{
		Size:      64,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDSSE2,
		Signature: "fast_sse2",
		Priority:  19,
	})

	avx2 := cpu.Features{HasSSE2: true, HasAVX2: true, HasFMA: true}
	if got := reg.Lookup(64, avx2); got == nil || got.Signature != "fast_sse2" {
		t.Fatalf("AVX2 host: got %v, want fast_sse2", got)
	}

	// Eligibility is untouched: the demoted entry still requires AVX2, so an
	// SSE2-only host must never reach it — here it also picks fast_sse2, so
	// check the demoted entry is rejected on its own.
	sse2Only := cpu.Features{HasSSE2: true}
	if CPUSupports(sse2Only, fftypes.SIMDAVX2) {
		t.Fatal("SSE2-only host must not satisfy an AVX2 requirement")
	}

	if got := reg.Lookup(64, sse2Only); got == nil || got.Signature != "fast_sse2" {
		t.Fatalf("SSE2 host: got %v, want fast_sse2", got)
	}
}

// TestCodeletRegistryRankLevelUnsetKeepsSIMDOrder pins the default: with
// RankLevel unset, ordering is by SIMDLevel exactly as before.
func TestCodeletRegistryRankLevelUnsetKeepsSIMDOrder(t *testing.T) {
	t.Parallel()

	reg := NewCodeletRegistry[complex64]()

	reg.Register(CodeletEntry[complex64]{
		Size:      64,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDSSE2,
		Signature: "sse2",
		Priority:  100,
	})
	reg.Register(CodeletEntry[complex64]{
		Size:      64,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDAVX2,
		Signature: "avx2",
		Priority:  1,
	})

	avx2 := cpu.Features{HasSSE2: true, HasAVX2: true, HasFMA: true}
	if got := reg.Lookup(64, avx2); got == nil || got.Signature != "avx2" {
		t.Fatalf("got %v, want avx2 (SIMD level still dominates priority)", got)
	}
}

// TestCodeletRegistryRankBelowGeneric verifies the disposition for a codelet
// that measured slower than pure Go: it must lose to the generic sibling even
// though its SIMD level is higher and its priority is larger, because
// SIMD-level-major ordering otherwise makes a slow SIMD codelet unbeatable.
// Crucially it must remain wisdom-reachable and visible to GetAllForSize, which
// is what distinguishes this from Priority < 0.
func TestCodeletRegistryRankBelowGeneric(t *testing.T) {
	t.Parallel()

	reg := NewCodeletRegistry[complex64]()

	reg.Register(CodeletEntry[complex64]{
		Size:             64,
		Forward:          dummyCodelet[complex64],
		Inverse:          dummyCodelet[complex64],
		Algorithm:        fftypes.KernelDIT,
		SIMDLevel:        fftypes.SIMDNEON,
		RankBelowGeneric: true,
		Signature:        "slow_neon",
		Priority:         100,
	})
	reg.Register(CodeletEntry[complex64]{
		Size:      64,
		Forward:   dummyCodelet[complex64],
		Inverse:   dummyCodelet[complex64],
		Algorithm: fftypes.KernelDIT,
		SIMDLevel: fftypes.SIMDNone,
		Signature: "generic",
		Priority:  0,
	})

	neon := cpu.Features{HasNEON: true}
	if got := reg.Lookup(64, neon); got == nil || got.Signature != "generic" {
		t.Fatalf("NEON host: got %v, want generic", got)
	}

	// Still selectable by wisdom, which addresses it by signature and ignores
	// ordering entirely — the point of demoting rather than disabling.
	if got := reg.LookupBySignature(64, "slow_neon"); got == nil {
		t.Fatal("LookupBySignature must still resolve a below-generic codelet")
	}

	// Still enumerated, so the registry-driven reference tests keep covering it.
	found := false

	for _, e := range reg.GetAllForSize(64) {
		if e.Signature == "slow_neon" {
			found = true
		}
	}

	if !found {
		t.Fatal("GetAllForSize must still list a below-generic codelet")
	}
}

// TestCodeletRegistryRankBelowGenericStillRunsWhenAlone pins the fallback: the
// demotion is relative, so a size with no pure-Go codelet still gets the SIMD
// one rather than nothing.
func TestCodeletRegistryRankBelowGenericStillRunsWhenAlone(t *testing.T) {
	t.Parallel()

	reg := NewCodeletRegistry[complex64]()

	reg.Register(CodeletEntry[complex64]{
		Size:             64,
		Forward:          dummyCodelet[complex64],
		Inverse:          dummyCodelet[complex64],
		Algorithm:        fftypes.KernelDIT,
		SIMDLevel:        fftypes.SIMDNEON,
		RankBelowGeneric: true,
		Signature:        "slow_neon",
		Priority:         1,
	})

	if got := reg.Lookup(64, cpu.Features{HasNEON: true}); got == nil || got.Signature != "slow_neon" {
		t.Fatalf("got %v, want slow_neon (no generic sibling to lose to)", got)
	}
}
