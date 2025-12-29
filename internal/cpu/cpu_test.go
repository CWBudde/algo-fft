package cpu

import (
	"runtime"
	"sync"
	"testing"
)

// TestDetectFeatures tests that CPU feature detection returns valid results.
func TestDetectFeatures(t *testing.T) {
	t.Parallel()

	features := DetectFeatures()

	// Architecture should always be set
	if features.Architecture == "" {
		t.Error("Architecture should be set")
	}

	// Architecture should match runtime.GOARCH
	if features.Architecture != runtime.GOARCH {
		t.Errorf("Architecture mismatch: got %q, want %q", features.Architecture, runtime.GOARCH)
	}

	// On amd64, SSE2 should always be available (part of x86-64 baseline)
	if runtime.GOARCH == "amd64" && !features.HasSSE2 {
		t.Error("SSE2 should be available on amd64")
	}

	// On arm64, NEON should always be available (mandatory in ARMv8)
	if runtime.GOARCH == "arm64" && !features.HasNEON {
		t.Error("NEON should be available on arm64")
	}

	// Log detected features for informational purposes
	t.Logf("Detected features: %+v", features)
}

// TestQueryFunctions tests that query functions match struct fields.
//
//nolint:paralleltest // modifies global state via ResetDetection()
func TestQueryFunctions(t *testing.T) {
	// Note: Not parallel - modifies global state via ResetDetection()

	// Ensure we're using real detection, not forced features
	t.Cleanup(ResetDetection)

	ResetDetection()

	features := DetectFeatures()

	tests := []struct {
		name     string
		got      bool
		expected bool
	}{
		{"HasSSE2", HasSSE2(), features.HasSSE2},
		{"HasSSE3", HasSSE3(), features.HasSSE3},
		{"HasSSSE3", HasSSSE3(), features.HasSSSE3},
		{"HasSSE41", HasSSE41(), features.HasSSE41},
		{"HasAVX", HasAVX(), features.HasAVX},
		{"HasAVX2", HasAVX2(), features.HasAVX2},
		{"HasAVX512", HasAVX512(), features.HasAVX512},
		{"HasNEON", HasNEON(), features.HasNEON},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.got != tt.expected {
				t.Errorf("%s() = %v, want %v", tt.name, tt.got, tt.expected)
			}
		})
	}
}

// TestForcedFeatures tests that SetForcedFeatures overrides detection.
//
//nolint:paralleltest // subtests modify global state via SetForcedFeatures()
func TestForcedFeatures(t *testing.T) {
	// Note: Not parallel - subtests modify global state via SetForcedFeatures()

	// Test SSE2-only system
	//nolint:paralleltest // modifies global state
	t.Run("SSE2Only", func(t *testing.T) {
		defer ResetDetection()

		SetForcedFeatures(Features{
			HasSSE2:      true,
			Architecture: "amd64",
		})

		if !HasSSE2() {
			t.Error("Expected HasSSE2() to return true")
		}

		if HasSSE3() {
			t.Error("Expected HasSSE3() to return false")
		}

		if HasAVX() {
			t.Error("Expected HasAVX() to return false")
		}

		if HasAVX2() {
			t.Error("Expected HasAVX2() to return false")
		}

		if HasNEON() {
			t.Error("Expected HasNEON() to return false")
		}

		features := DetectFeatures()
		if features.Architecture != "amd64" {
			t.Errorf("Architecture = %q, want amd64", features.Architecture)
		}
	})

	// Test AVX2 system
	//nolint:paralleltest // modifies global state
	t.Run("AVX2System", func(t *testing.T) {
		defer ResetDetection()

		SetForcedFeatures(Features{
			HasSSE2:      true,
			HasSSE3:      true,
			HasSSSE3:     true,
			HasSSE41:     true,
			HasAVX:       true,
			HasAVX2:      true,
			Architecture: "amd64",
		})

		if !HasSSE2() {
			t.Error("Expected HasSSE2() to return true")
		}

		if !HasSSE3() {
			t.Error("Expected HasSSE3() to return true")
		}

		if !HasSSSE3() {
			t.Error("Expected HasSSSE3() to return true")
		}

		if !HasSSE41() {
			t.Error("Expected HasSSE41() to return true")
		}

		if !HasAVX() {
			t.Error("Expected HasAVX() to return true")
		}

		if !HasAVX2() {
			t.Error("Expected HasAVX2() to return true")
		}

		if HasAVX512() {
			t.Error("Expected HasAVX512() to return false")
		}
	})

	// Test ARM NEON system
	//nolint:paralleltest // modifies global state
	t.Run("NEONSystem", func(t *testing.T) {
		defer ResetDetection()

		SetForcedFeatures(Features{
			HasNEON:      true,
			Architecture: "arm64",
		})

		if !HasNEON() {
			t.Error("Expected HasNEON() to return true")
		}

		if HasAVX2() {
			t.Error("Expected HasAVX2() to return false on ARM")
		}

		features := DetectFeatures()
		if features.Architecture != "arm64" {
			t.Errorf("Architecture = %q, want arm64", features.Architecture)
		}
	})

	// Test ForceGeneric flag
	//nolint:paralleltest // modifies global state
	t.Run("ForceGeneric", func(t *testing.T) {
		defer ResetDetection()

		SetForcedFeatures(Features{
			HasAVX2:      true,
			ForceGeneric: true,
			Architecture: "amd64",
		})

		features := DetectFeatures()
		if !features.ForceGeneric {
			t.Error("Expected ForceGeneric to be true")
		}

		if !features.HasAVX2 {
			t.Error("Expected HasAVX2 to still be true (flags are independent)")
		}
	})
}

// TestResetDetection tests that ResetDetection clears forced features.
//
//nolint:paralleltest // modifies global state via SetForcedFeatures() and ResetDetection()
func TestResetDetection(t *testing.T) {
	// Note: Not parallel - modifies global state via SetForcedFeatures() and ResetDetection()

	// Set forced features
	SetForcedFeatures(Features{
		HasAVX2:      true,
		Architecture: "amd64",
	})

	// Verify forced features are active
	if !HasAVX2() {
		t.Fatal("Setup failed: forced features not applied")
	}

	// Reset to real detection
	ResetDetection()

	// After reset, should return to actual hardware detection
	actualFeatures := DetectFeatures()

	// Query functions should match actual hardware
	if HasAVX2() != actualFeatures.HasAVX2 {
		t.Error("Reset didn't restore real detection for AVX2")
	}

	if HasSSE2() != actualFeatures.HasSSE2 {
		t.Error("Reset didn't restore real detection for SSE2")
	}

	if HasNEON() != actualFeatures.HasNEON {
		t.Error("Reset didn't restore real detection for NEON")
	}

	// Architecture should match runtime
	if actualFeatures.Architecture != runtime.GOARCH {
		t.Errorf("After reset, Architecture = %q, want %q", actualFeatures.Architecture, runtime.GOARCH)
	}
}

// TestConcurrentDetection tests thread-safety of sync.Once caching.
//
//nolint:paralleltest // modifies global state via ResetDetection()
func TestConcurrentDetection(t *testing.T) {
	// Note: Not parallel - modifies global state via ResetDetection()

	// Reset to ensure we're testing the caching mechanism
	ResetDetection()
	defer ResetDetection()

	const goroutines = 100

	var waitGroup sync.WaitGroup

	results := make([]Features, goroutines)

	for i := range goroutines {
		waitGroup.Add(1)

		go func(index int) {
			defer waitGroup.Done()

			results[index] = DetectFeatures()
		}(i)
	}

	waitGroup.Wait()

	// All results should be identical (detection ran once, cached for all)
	first := results[0]
	for i := 1; i < goroutines; i++ {
		if results[i] != first {
			t.Errorf("Concurrent detection produced different results: got %+v, want %+v", results[i], first)
		}
	}
}

// TestDetectionCaching verifies that detection only runs once.
//
//nolint:paralleltest // modifies global state via ResetDetection()
func TestDetectionCaching(t *testing.T) {
	// Note: Not parallel - modifies global state via ResetDetection()

	// Reset to start fresh
	ResetDetection()
	defer ResetDetection()

	// First call should perform detection
	features1 := DetectFeatures()

	// Subsequent calls should return cached result
	features2 := DetectFeatures()
	features3 := DetectFeatures()

	// All results should be identical
	if features1 != features2 {
		t.Error("Second call returned different features")
	}

	if features1 != features3 {
		t.Error("Third call returned different features")
	}
}

// TestFeaturesStructFields verifies the Features struct has expected fields.
func TestFeaturesStructFields(t *testing.T) {
	features := Features{
		HasSSE2:      true,
		HasSSE3:      true,
		HasSSSE3:     true,
		HasSSE41:     true,
		HasAVX:       true,
		HasAVX2:      true,
		HasAVX512:    true,
		HasNEON:      true,
		ForceGeneric: true,
		Architecture: "test",
	}

	// Verify all fields are accessible and have the values we set
	if !features.HasSSE2 {
		t.Error("HasSSE2 field not working")
	}

	if !features.HasSSE3 {
		t.Error("HasSSE3 field not working")
	}

	if !features.HasSSSE3 {
		t.Error("HasSSSE3 field not working")
	}

	if !features.HasSSE41 {
		t.Error("HasSSE41 field not working")
	}

	if !features.HasAVX {
		t.Error("HasAVX field not working")
	}

	if !features.HasAVX2 {
		t.Error("HasAVX2 field not working")
	}

	if !features.HasAVX512 {
		t.Error("HasAVX512 field not working")
	}

	if !features.HasNEON {
		t.Error("HasNEON field not working")
	}

	if !features.ForceGeneric {
		t.Error("ForceGeneric field not working")
	}

	if features.Architecture != "test" {
		t.Error("Architecture field not working")
	}
}

// BenchmarkDetectFeatures benchmarks the cost of calling DetectFeatures.
func BenchmarkDetectFeatures(b *testing.B) {
	// Reset to ensure detection happens
	ResetDetection()

	b.ResetTimer()

	for range b.N {
		_ = DetectFeatures()
	}
}

// BenchmarkQueryFunctions benchmarks the cost of calling individual query functions.
func BenchmarkQueryFunctions(b *testing.B) {
	ResetDetection()

	b.Run("HasAVX2", func(b *testing.B) {
		for range b.N {
			_ = HasAVX2()
		}
	})

	b.Run("HasSSE41", func(b *testing.B) {
		for range b.N {
			_ = HasSSE41()
		}
	})

	b.Run("HasNEON", func(b *testing.B) {
		for range b.N {
			_ = HasNEON()
		}
	})
}
