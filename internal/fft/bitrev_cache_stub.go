//go:build !amd64 || purego

package fft

// PrewarmSizeCaches is a no-op on builds without the amd64 SIMD kernel
// wrappers; see bitrev_cache_amd64.go for the SIMD implementation.
func PrewarmSizeCaches(int) {}
