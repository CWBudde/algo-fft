//go:build !linux

package cpu

// detectCachesImpl has no detection source on this platform; DetectCaches
// falls back to the conservative defaults.
func detectCachesImpl() CacheInfo {
	return CacheInfo{}
}
