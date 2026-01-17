// Package cpu provides CPU feature detection for FFT kernel selection.
//
// This package detects SIMD instruction set extensions (SSE, AVX, NEON) available
// on the current processor and caches the results for efficient querying.
//
// Detection is performed lazily on the first call to DetectFeatures() and the
// results are cached for subsequent calls using sync.Once for thread-safety.
//
// For testing purposes, feature detection can be overridden using SetForcedFeatures()
// and reset to actual hardware detection using ResetDetection().
package cpu

import (
	"sync"
)

// Features describes CPU capabilities relevant to FFT kernel selection.
//
// The struct groups features by architecture (x86/amd64 vs ARM) and includes
// control flags for testing and debugging.
type Features struct {
	// x86/amd64 SIMD features (detected via CPUID)
	HasSSE    bool // Streaming SIMD Extensions (always true on amd64)
	HasSSE2   bool // Streaming SIMD Extensions 2 (always true on amd64)
	HasSSE3   bool // Streaming SIMD Extensions 3
	HasSSSE3  bool // Supplemental Streaming SIMD Extensions 3
	HasSSE41  bool // Streaming SIMD Extensions 4.1
	HasAVX    bool // Advanced Vector Extensions
	HasAVX2   bool // Advanced Vector Extensions 2
	HasAVX512 bool // Advanced Vector Extensions 512

	// ARM SIMD features
	HasNEON bool // ARM Advanced SIMD (NEON)

	// Control flags
	ForceGeneric bool // Disable all SIMD optimizations (for testing/debugging)

	// Runtime information
	Architecture string // runtime.GOARCH (e.g., "amd64", "arm64")
}

//nolint:gochecknoglobals
var (
	// detectedFeatures holds the cached CPU features detected on this system.
	detectedFeatures Features

	// detectOnce ensures feature detection runs exactly once, thread-safely.
	detectOnce sync.Once

	// detectMutex serializes access to detectOnce/detectedFeatures so ResetDetection
	// can safely clear the cache even when tests run in parallel.
	detectMutex sync.Mutex

	// forcedFeatures allows overriding actual hardware detection for testing.
	// When non-nil, DetectFeatures() returns this value instead of real detection.
	forcedFeatures *Features

	// forcedMutex protects forcedFeatures from concurrent access during testing.
	forcedMutex sync.RWMutex
)

// DetectFeatures returns the CPU features available on the current system.
//
// Detection is performed once on the first call and cached for subsequent calls.
// This function is thread-safe and can be called concurrently from multiple goroutines.
//
// For testing, use SetForcedFeatures() to override the detected features.
func DetectFeatures() Features {
	forcedMutex.RLock()

	forced := forcedFeatures

	forcedMutex.RUnlock()

	if forced != nil {
		return *forced
	}

	detectMutex.Lock()
	detectOnce.Do(func() {
		detectedFeatures = detectFeaturesImpl()
	})

	features := detectedFeatures

	detectMutex.Unlock()

	return features
}

// HasSSE returns true if the CPU supports SSE instructions.
// On amd64, this is always true as SSE is part of the architecture baseline.
func HasSSE() bool {
	return DetectFeatures().HasSSE
}

// HasSSE2 returns true if the CPU supports SSE2 instructions.
// On amd64, this is always true as SSE2 is part of the architecture baseline.
func HasSSE2() bool {
	return DetectFeatures().HasSSE2
}

// HasSSE3 returns true if the CPU supports SSE3 instructions.
func HasSSE3() bool {
	return DetectFeatures().HasSSE3
}

// HasSSSE3 returns true if the CPU supports SSSE3 (Supplemental SSE3) instructions.
func HasSSSE3() bool {
	return DetectFeatures().HasSSSE3
}

// HasSSE41 returns true if the CPU supports SSE4.1 instructions.
func HasSSE41() bool {
	return DetectFeatures().HasSSE41
}

// HasAVX returns true if the CPU supports AVX instructions.
func HasAVX() bool {
	return DetectFeatures().HasAVX
}

// HasAVX2 returns true if the CPU supports AVX2 instructions.
func HasAVX2() bool {
	return DetectFeatures().HasAVX2
}

// HasAVX512 returns true if the CPU supports AVX-512 instructions.
func HasAVX512() bool {
	return DetectFeatures().HasAVX512
}

// HasNEON returns true if the CPU supports ARM NEON (Advanced SIMD) instructions.
// On ARMv8 (arm64), NEON is mandatory and this always returns true.
func HasNEON() bool {
	return DetectFeatures().HasNEON
}

// ForceSSEOnlyForTests forces feature detection to expose only SSE (not SSE2+).
// This is intended to exercise SSE-only dispatch paths in tests.
// Call ResetDetection() to restore normal detection.
func ForceSSEOnlyForTests() {
	feature := DetectFeatures()
	feature.HasSSE = true
	feature.HasSSE2 = false
	feature.HasSSE3 = false
	feature.HasSSSE3 = false
	feature.HasSSE41 = false
	feature.HasAVX = false
	feature.HasAVX2 = false
	feature.HasAVX512 = false
	feature.ForceGeneric = false
	SetForcedFeatures(feature)
}

// SetForcedFeatures overrides CPU feature detection with the specified features.
//
// This function is intended for testing purposes only and should not be used in
// production code. It allows testing kernel selection logic for CPU configurations
// that may not be available on the test machine.
//
// Call ResetDetection() to restore actual hardware feature detection.
//
// This function is thread-safe but should not be called concurrently with
// ResetDetection() or other SetForcedFeatures() calls.
func SetForcedFeatures(f Features) {
	forcedMutex.Lock()
	defer forcedMutex.Unlock()

	forced := f
	forcedFeatures = &forced
}

// ResetDetection clears any forced features set by SetForcedFeatures() and
// clears the detection cache, forcing re-detection on the next call to DetectFeatures().
//
// This function is intended for testing purposes to restore actual hardware
// feature detection after using SetForcedFeatures().
//
// This function is thread-safe but should not be called concurrently with
// SetForcedFeatures() or other ResetDetection() calls.
func ResetDetection() {
	forcedMutex.Lock()

	forcedFeatures = nil

	forcedMutex.Unlock()

	detectMutex.Lock()

	detectOnce = sync.Once{}
	detectedFeatures = Features{}

	detectMutex.Unlock()
}
