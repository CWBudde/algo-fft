package cpu

import (
	"sync"
)

// CacheInfo describes the data-cache sizes relevant to cache-blocked FFT
// decompositions. Sizes are per-core bytes; a zero field means "unknown"
// inside detection, but DetectCaches always substitutes defaults so callers
// never see zeros.
type CacheInfo struct {
	L1DataBytes int // Level-1 data cache size
	L2Bytes     int // Level-2 (unified) cache size
}

// Conservative fallback sizes used when detection is unavailable (non-Linux,
// WASM) or incomplete. Underestimating cache sizes only makes blocking
// decisions more conservative, so these are at the small end of common cores.
const (
	defaultL1DataBytes = 32 * 1024
	defaultL2Bytes     = 256 * 1024
)

//nolint:gochecknoglobals
var (
	// detectedCaches holds the cached cache-size detection result.
	detectedCaches CacheInfo

	// cacheDetectOnce ensures cache detection runs exactly once, thread-safely.
	cacheDetectOnce sync.Once

	// cacheDetectMutex serializes access so ResetCacheDetection can safely
	// clear the cache even when tests run in parallel.
	cacheDetectMutex sync.Mutex

	// forcedCaches overrides detection for testing when non-nil.
	forcedCaches *CacheInfo

	// forcedCachesMutex protects forcedCaches.
	forcedCachesMutex sync.RWMutex
)

// DetectCaches returns the per-core L1 data and L2 cache sizes. Detection runs
// once and is cached; fields that cannot be detected fall back to conservative
// defaults, so the result always has positive sizes.
//
// For testing, use SetForcedCaches() to override the detected values.
func DetectCaches() CacheInfo {
	forcedCachesMutex.RLock()

	forced := forcedCaches

	forcedCachesMutex.RUnlock()

	if forced != nil {
		return *forced
	}

	cacheDetectMutex.Lock()
	cacheDetectOnce.Do(func() {
		detectedCaches = withCacheDefaults(detectCachesImpl())
	})

	caches := detectedCaches

	cacheDetectMutex.Unlock()

	return caches
}

// SetForcedCaches overrides cache-size detection with the specified values.
// Intended for testing only. Call ResetCacheDetection() to restore detection.
func SetForcedCaches(c CacheInfo) {
	forcedCachesMutex.Lock()
	defer forcedCachesMutex.Unlock()

	forced := c
	forcedCaches = &forced
}

// ResetCacheDetection clears any forced values set by SetForcedCaches() and
// clears the detection cache, forcing re-detection on the next DetectCaches().
func ResetCacheDetection() {
	forcedCachesMutex.Lock()

	forcedCaches = nil

	forcedCachesMutex.Unlock()

	cacheDetectMutex.Lock()

	cacheDetectOnce = sync.Once{}
	detectedCaches = CacheInfo{}

	cacheDetectMutex.Unlock()
}

// withCacheDefaults fills unknown (zero or negative) fields with the
// conservative defaults and enforces L2 >= L1d.
func withCacheDefaults(c CacheInfo) CacheInfo {
	if c.L1DataBytes <= 0 {
		c.L1DataBytes = defaultL1DataBytes
	}

	if c.L2Bytes <= 0 {
		c.L2Bytes = defaultL2Bytes
	}

	if c.L2Bytes < c.L1DataBytes {
		c.L2Bytes = c.L1DataBytes
	}

	return c
}
